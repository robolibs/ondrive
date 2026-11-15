//! TEB — Timed Elastic Band (Rösmann et al. 2012/2017), solved here by
//! projected gradient descent instead of g2o.
//!
//! The band is `n_poses` poses plus `n_poses - 1` time intervals. Residuals:
//! total time, velocity / yaw-rate / acceleration limits (including the
//! start edge against the measured speed and the goal edge), the
//! non-holonomic kinematic constraint (consecutive poses on a common arc),
//! minimum turning radius for Ackermann, path deviation, time-indexed
//! obstacle clearance at poses and segment midpoints, and a target on the
//! final pose. Pose 0 is anchored to the robot; the band is warm-started
//! and advanced along the path as the robot passes its poses.

#![allow(clippy::too_many_arguments)]

use crate::controller::{Controller, ControllerBase};
use crate::core::kinematics::{
    actuation_delay, axle_offset, finalize, finalize_holonomic, is_ackermann, is_holonomic, max_curvature, speed_bounds,
    wheelbase,
    world_to_body,
};
use crate::core::math::{normalize_angle, yaw_of};
use crate::core::obstacles::{CollisionChecker, robot_radius};
use crate::core::path::{PathCursor, cumulative_lengths, project, sample};
use crate::pred::mppi::{Along, Choice, Swerve, current_lateral_speed, current_speed, plan_avoidance, prepare, signed_speed_at};
use crate::types::{Goal, Obstacle, Path, RobotConstraints, RobotState, VelocityCommand, WorldConstraints};
use datapod::{Euler, Point, Pose, Quaternion};

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct TebConfig {
    pub n_poses: usize,
    pub dt_nominal: f64,
    pub iterations: usize,
    pub step_size: f64,

    pub weight_time: f64,
    pub weight_velocity_limit: f64,
    pub weight_angular_limit: f64,
    pub weight_acceleration_limit: f64,
    pub weight_path_deviation: f64,
    pub weight_obstacle: f64,
    pub weight_kinematic: f64,
    pub weight_goal: f64,

    pub obstacle_margin: f64,
    pub dt_min: f64,
    pub dt_max: f64,
    /// Seconds of band the command is read over.
    pub control_lookahead: f64,
    /// Time constant, seconds, of the low-pass filter on the commanded
    /// turn rate; 0 passes it straight through.
    #[serde(default = "default_turn_filter")]
    pub turn_filter: f64,
}

impl Default for TebConfig {
    fn default() -> Self {
        Self {
            n_poses: 12,
            dt_nominal: 0.3,
            iterations: 30,
            step_size: 0.05,
            weight_time: 1.0,
            weight_velocity_limit: 50.0,
            weight_angular_limit: 50.0,
            weight_acceleration_limit: 10.0,
            weight_path_deviation: 3.0,
            weight_obstacle: 100.0,
            weight_kinematic: 50.0,
            weight_goal: 5.0,
            obstacle_margin: 0.8,
            dt_min: 0.05,
            dt_max: 1.0,
            control_lookahead: 0.4,
            turn_filter: default_turn_filter(),
        }
    }
}

fn default_turn_filter() -> f64 {
    0.15
}

/// Share of the distance covered during the steering delay added to the
/// obstacle margin.
const LAG_MARGIN_GAIN: f64 = 0.25;

/// Poses of the band checked for collision before it is driven: all of
/// it, so a band that would be in the way when the traffic comes is never
/// begun (teb_local_planner checks `feasibility_check_no_poses`, 5).

const FEASIBILITY_POSES: usize = usize::MAX;
/// Metres between the checks along each stretch of band checked.
const FEASIBILITY_STEP: f64 = 0.1;
/// Factor on the nominal time interval of the slower alternative band.
const SLOWER: f64 = 2.0;

#[derive(Clone, Debug, Default)]
pub struct TebFollower {
    pub base: ControllerBase,
    pub teb_config: TebConfig,
    cursor: PathCursor,
    band: Band,
    band_initialised: bool,
    last_v: f64,
    last_vy: f64,
    last_w: f64,
    /// The way past the obstacles last chosen, see `plan_avoidance`.
    avoid: Choice,
    auto_path_goal: Option<(f64, f64, f64)>,
    /// Distance, metres, the pose rides ahead of the band's points, the axle the
    /// body turns about.
    axle: f64,
}

#[derive(Clone, Debug, Default)]
struct Band {
    x: Vec<f64>,
    y: Vec<f64>,
    yaw: Vec<f64>,
    dt: Vec<f64>,
    /// Arc length on the reference path where the band tail was sampled.
    tail_s: f64,
}

struct Problem<'a> {
    cfg: &'a TebConfig,
    constraints: &'a RobotConstraints,
    waypoints: &'a [Pose],
    cum: &'a [f64],
    checker: &'a CollisionChecker<'a>,
    target: (Point, f64),
    kappa_max: f64,
    v_lo: f64,
    v_hi: f64,
    v_start: f64,
    /// Measured turn rate the band sets off from.
    w_start: f64,
    /// Steering rate over wheelbase, for an Ackermann machine's turn-rate
    /// change limit `|v| · steer_per_wheelbase`; infinite when unlimited.
    steer_per_wheelbase: f64,
    ackermann: bool,
    holonomic: bool,
    /// Distance, metres, the pose rides ahead of the band's points, the axle the
    /// body turns about.
    axle: f64,
    /// Path segment the robot projects onto, where projections start.
    start_segment: usize,
}

impl Problem<'_> {
    /// Room kept from obstacles: `obstacle_margin`, plus `LAG_MARGIN_GAIN`
    /// of the distance the band's top speed covers while the steering
    /// takes effect.
    fn margin(&self) -> f64 {
        self.cfg.obstacle_margin + LAG_MARGIN_GAIN * self.v_hi.abs() * actuation_delay(self.constraints)
    }

    /// The pose of a body whose band point, its turning axle, is `(x, y)`.
    fn pose_of(&self, x: f64, y: f64, yaw: f64) -> (f64, f64) {
        (x + self.axle * yaw.cos(), y + self.axle * yaw.sin())
    }

    /// Penalty for coming within `margin` of an obstacle as it is predicted
    /// `t` seconds from now.
    /// Whether the body driven along the first `poses` poses of `b`, and
    /// every `FEASIBILITY_STEP` metres between them, stays clear of every
    /// obstacle as predicted at that moment (teb_local_planner's
    /// `isTrajectoryFeasible`).
    fn feasible(&self, b: &Band, poses: usize) -> bool {
        if !self.checker.has_obstacles() {
            return true;
        }
        let mut t = 0.0;
        for i in 0..poses.min(b.x.len().saturating_sub(1)) {
            let dt = b.dt[i].max(1e-4);
            let gap = (b.x[i + 1] - b.x[i]).hypot(b.y[i + 1] - b.y[i]);
            let parts = ((gap / FEASIBILITY_STEP).ceil() as usize).max(1);
            let turn = normalize_angle(b.yaw[i + 1] - b.yaw[i]);
            for k in 0..parts {
                let f = k as f64 / parts as f64;
                let yaw = b.yaw[i] + f * turn;
                let (x, y) = self.pose_of(b.x[i] + f * (b.x[i + 1] - b.x[i]), b.y[i] + f * (b.y[i + 1] - b.y[i]), yaw);
                if self.checker.clearance_at(t + f * dt, x, y, yaw) < 0.0 {
                    return false;
                }
            }
            t += dt;
        }
        true
    }

    fn obstacle_cost(&self, t: f64, x: f64, y: f64, yaw: f64) -> f64 {
        let (x, y) = self.pose_of(x, y, yaw);
        let c = self.cfg;
        if !self.checker.has_obstacles() {
            return 0.0;
        }
        let margin = self.margin();
        let d = self.checker.clearance_at_within(t, x, y, yaw, margin);
        if d < margin {
            let s = margin - d;
            c.weight_obstacle * s * s
        } else {
            0.0
        }
    }

    fn cost(&self, b: &Band) -> f64 {
        let n = b.x.len();
        let c = self.cfg;
        let a_max = self.constraints.max_linear_acceleration.abs();
        let mut cost = 0.0;
        let mut prev_v: Option<(f64, f64)> = None;
        let mut prev_w: Option<(f64, f64)> = None;
        let mut last_v = 0.0;
        let mut times = vec![0.0; n];
        for i in 0..n - 1 {
            times[i + 1] = times[i] + b.dt[i].max(1e-4);
        }

        for i in 0..n - 1 {
            let dt = b.dt[i].max(1e-4);
            cost += c.weight_time * dt;

            let dx = b.x[i + 1] - b.x[i];
            let dy = b.y[i + 1] - b.y[i];
            let ds = dx.hypot(dy);
            let (s0, c0) = b.yaw[i].sin_cos();
            let (s1, c1) = b.yaw[i + 1].sin_cos();
            let forward = (dx * (c0 + c1) + dy * (s0 + s1)).signum();
            let v = forward * ds / dt;
            let v_over = (v - self.v_hi).max(0.0) + (self.v_lo - v).max(0.0);
            cost += c.weight_velocity_limit * v_over * v_over;

            let dyaw = normalize_angle(b.yaw[i + 1] - b.yaw[i]);
            let w = dyaw / dt;
            let w_over = (w.abs() - self.constraints.max_angular_velocity.abs()).max(0.0);
            cost += c.weight_angular_limit * w_over * w_over;

            let (pw, pwdt) = prev_w.unwrap_or((self.w_start, dt));
            let alpha = (w - pw) / (0.5 * (pwdt + dt));
            let speed = v.abs().max(TURN_CHANGE_SPEED_FLOOR);
            let alpha_max = self.constraints.max_angular_acceleration.abs();
            let steering = if self.steer_per_wheelbase.is_finite() {
                speed * self.steer_per_wheelbase + (w.abs() / speed) * a_max
            } else if alpha_max > 0.0 {
                alpha_max
            } else {
                f64::INFINITY
            };
            let alpha_over = (alpha.abs() - steering).max(0.0);
            cost += c.weight_acceleration_limit * alpha_over * alpha_over;
            prev_w = Some((w, dt));

            let (pv, pdt) = prev_v.unwrap_or((self.v_start, dt));
            let a = (v - pv) / (0.5 * (pdt + dt));
            let a_over = (a.abs() - a_max).max(0.0);
            cost += c.weight_acceleration_limit * a_over * a_over;
            prev_v = Some((v, dt));
            last_v = v;

            // A holonomic platform can translate off its heading axis, so
            // the common-arc residual below does not apply to it.
            if !self.holonomic {
                let kin = (c0 + c1) * dy - (s0 + s1) * dx;
                cost += c.weight_kinematic * kin * kin;
            }

            if self.ackermann && self.kappa_max.is_finite() {
                let r = (dyaw.abs() - ds * self.kappa_max).max(0.0);
                cost += c.weight_kinematic * r * r;
            }

            cost += self.obstacle_cost(times[i] + 0.5 * dt, 0.5 * (b.x[i] + b.x[i + 1]), 0.5 * (b.y[i] + b.y[i + 1]), b.yaw[i]);
        }
        if let Some((_, pdt)) = prev_v {
            let a_goal = last_v / pdt.max(1e-4);
            let a_over = (a_goal.abs() - a_max).max(0.0);
            cost += 0.25 * c.weight_acceleration_limit * a_over * a_over;
        }

        if !self.waypoints.is_empty() {
            let mut hint = self.start_segment;
            for i in 1..n {
                let (px, py) = self.pose_of(b.x[i], b.y[i], b.yaw[i]);
                if let Some(p) = project(self.waypoints, self.cum, Point::new(px, py, 0.0), hint, 2, 64) {
                    cost += c.weight_path_deviation * p.distance * p.distance;
                    hint = p.segment;
                }
            }
        }

        for i in 0..n {
            cost += self.obstacle_cost(times[i], b.x[i], b.y[i], b.yaw[i]);
        }

        let (tp, th) = self.target;
        let (ex, ey) = self.pose_of(b.x[n - 1], b.y[n - 1], b.yaw[n - 1]);
        let dx = ex - tp.x;
        let dy = ey - tp.y;
        let dh = normalize_angle(b.yaw[n - 1] - th);
        cost += c.weight_goal * (dx * dx + dy * dy + 0.5 * dh * dh);
        cost
    }
}

/// Speed, m/s, below which the turn-rate change limit is taken at this speed.
const TURN_CHANGE_SPEED_FLOOR: f64 = 0.2;

/// Move a sampled band pose sideways off any obstacle it would sit on, so
/// the optimiser sees a lateral gradient instead of a symmetric saddle.
fn push_off_obstacles(
    p: Point,
    heading: f64,
    t: f64,
    checker: &CollisionChecker,
    margin: f64,
) -> Point {
    if !checker.has_obstacles() {
        return p;
    }
    let c = checker.clearance_at(t, p.x, p.y, heading);
    if c >= margin {
        return p;
    }
    let (s, co) = heading.sin_cos();
    let shift = margin - c;
    let left = Point::new(p.x - shift * s, p.y + shift * co, 0.0);
    let right = Point::new(p.x + shift * s, p.y - shift * co, 0.0);
    let cl = checker.clearance_at(t, left.x, left.y, heading);
    let cr = checker.clearance_at(t, right.x, right.y, heading);
    if cr > cl { right } else { left }
}

impl TebFollower {
    pub fn new() -> Self {
        Self::default()
    }

    /// The way past the obstacles ahead of arc length `from` that the
    /// avoidance planner shared with MPPI picks over space and time
    /// (`plan_avoidance`, committed across ticks): as poses for the band to
    /// follow, `None` when it keeps to the path, and the share of the speed
    /// it drives at.
    fn detour_guide(&mut self, state: &RobotState, from: f64, constraints: &RobotConstraints, obstacles: &[Obstacle]) -> (Option<Vec<Pose>>, f64) {
        if obstacles.is_empty() {
            self.avoid = Choice::default();
            return (None, 1.0);
        }
        let step = self.teb_config.dt_nominal.max(0.05);
        let n = 2 * self.teb_config.n_poses.max(3);
        let speed = (0.8 * constraints.max_linear_velocity).max(0.1);
        let along = Along { path: &self.base.path, cum: &self.cursor.cum, trajectory: None, clock: 0.0, from };
        let robot = (state.pose.point.x, state.pose.point.y, yaw_of(&state.pose));
        let plan = plan_avoidance(&along, obstacles, robot_radius(constraints), Swerve::of(constraints, speed), n, step, speed, robot, &mut self.avoid);
        let poses = plan.guide.map(|g| {
            let k = g.x.len();
            (0..k)
                .map(|i| {
                    let (a, b) = (i.saturating_sub(1), (i + 1).min(k - 1));
                    let yaw = (g.y[b] - g.y[a]).atan2(g.x[b] - g.x[a]);
                    Pose { point: Point::new(g.x[i], g.y[i], 0.0), rotation: Quaternion::from_euler(Euler::new(0.0, 0.0, yaw)) }
                })
                .collect()
        });
        (poses, plan.pace)
    }

    pub fn with_teb_config(cfg: TebConfig) -> Self {
        Self {
            teb_config: cfg,
            ..Default::default()
        }
    }

    pub fn set_teb_config(&mut self, cfg: TebConfig) {
        self.band_initialised = false;
        self.teb_config = cfg;
    }

    pub fn band_poses(&self) -> Vec<Point> {
        (0..self.band.x.len())
            .map(|i| {
                let yaw = self.band.yaw[i];
                Point::new(self.band.x[i] + self.axle * yaw.cos(), self.band.y[i] + self.axle * yaw.sin(), 0.0)
            })
            .collect()
    }

    /// The turning axle of a body whose pose is `p`, heading `yaw`: the point
    /// the band follows.
    fn axle_of(&self, p: Point, yaw: f64) -> Point {
        Point::new(p.x - self.axle * yaw.cos(), p.y - self.axle * yaw.sin(), 0.0)
    }

    fn nominal_step(&self, constraints: &RobotConstraints) -> f64 {
        (constraints.max_linear_velocity.abs() * self.teb_config.dt_nominal).max(0.05)
    }

    fn sample_band_pose(&self, s: f64, step: usize, checker: &CollisionChecker) -> (Point, f64) {
        let (p, h) = self.path_pose(s);
        let t = step as f64 * self.teb_config.dt_nominal;
        let p = push_off_obstacles(p, h, t, checker, self.teb_config.obstacle_margin);
        (p, h)
    }

    /// The path at arc length `s` and the body heading there: the direction
    /// of travel, turned round on stretches the path drives in reverse.
    fn path_pose(&self, s: f64) -> (Point, f64) {
        let (p, h) = sample(&self.base.path.waypoints, &self.cursor.cum, s);
        let reverse = signed_speed_at(&self.base.path.speeds, &self.cursor.cum, s).is_some_and(|v| v < 0.0);
        (p, if reverse { normalize_angle(h + std::f64::consts::PI) } else { h })
    }

    fn initialise_band(
        &mut self,
        state: &RobotState,
        s0: f64,
        constraints: &RobotConstraints,
        checker: &CollisionChecker,
        cusp: f64,
    ) {
        let n = self.teb_config.n_poses.max(3);
        let step = self.nominal_step(constraints);
        let mut b = Band {
            x: vec![0.0; n],
            y: vec![0.0; n],
            yaw: vec![0.0; n],
            dt: vec![self.teb_config.dt_nominal; n - 1],
            tail_s: s0,
        };
        let yaw0 = state.pose.rotation.to_euler().yaw;
        let start = self.axle_of(state.pose.point, yaw0);
        b.x[0] = start.x;
        b.y[0] = start.y;
        b.yaw[0] = yaw0;
        for i in 1..n {
            let s = (s0 + i as f64 * step).min(cusp);
            let (p, h) = self.sample_band_pose(s, i, checker);
            let p = self.axle_of(p, h);
            b.x[i] = p.x;
            b.y[i] = p.y;
            b.yaw[i] = h;
            b.tail_s = s;
        }
        self.band = b;
        self.band_initialised = true;
    }

    /// Re-anchor pose 0 to the robot, drop every pose the robot has passed,
    /// add new ones no further than arc length `cusp`, and keep the first
    /// interval consistent with the current speed.
    fn advance_band(
        &mut self,
        state: &RobotState,
        constraints: &RobotConstraints,
        checker: &CollisionChecker,
        v_now: f64,
        cusp: f64,
    ) {
        let n = self.band.x.len();
        let step = self.nominal_step(constraints);
        let yaw0 = state.pose.rotation.to_euler().yaw;
        let start = self.axle_of(state.pose.point, yaw0);
        let (px, py) = (start.x, start.y);
        for _ in 0..n {
            let dx = self.band.x[1] - px;
            let dy = self.band.y[1] - py;
            let tx = self.band.x[2] - self.band.x[1];
            let ty = self.band.y[2] - self.band.y[1];
            let passed = dx * tx + dy * ty < 0.0 || dx.hypot(dy) < 0.25 * step;
            if !passed {
                break;
            }
            self.band.x.remove(1);
            self.band.y.remove(1);
            self.band.yaw.remove(1);
            self.band.dt.remove(0);
            self.band.tail_s = (self.band.tail_s + step).min(cusp);
            let (p, h) = self.sample_band_pose(self.band.tail_s, n - 1, checker);
            let p = self.axle_of(p, h);
            self.band.x.push(p.x);
            self.band.y.push(p.y);
            self.band.yaw.push(h);
            self.band.dt.push(self.teb_config.dt_nominal);
        }
        self.band.x[0] = px;
        self.band.y[0] = py;
        self.band.yaw[0] = yaw0;
        let ds = (self.band.x[1] - px).hypot(self.band.y[1] - py);
        if v_now.abs() > 1e-3 {
            self.band.dt[0] =
                (ds / v_now.abs()).clamp(self.teb_config.dt_min, self.teb_config.dt_max);
        }
    }

    fn optimise(&mut self, problem: &Problem) {
        let cfg = self.teb_config.clone();
        let n = self.band.x.len();
        let n_pose_vars = (n - 1) * 3;
        let n_vars = n_pose_vars + (n - 1);
        let eps = 1e-4;
        let mut grad = vec![0.0; n_vars];

        let get = |b: &Band, k: usize| -> f64 {
            if k < n_pose_vars {
                let i = k / 3 + 1;
                match k % 3 {
                    0 => b.x[i],
                    1 => b.y[i],
                    _ => b.yaw[i],
                }
            } else {
                b.dt[k - n_pose_vars]
            }
        };
        let set = |b: &mut Band, k: usize, v: f64| {
            if k < n_pose_vars {
                let i = k / 3 + 1;
                match k % 3 {
                    0 => b.x[i] = v,
                    1 => b.y[i] = v,
                    _ => b.yaw[i] = normalize_angle(v),
                }
            } else {
                b.dt[k - n_pose_vars] = v.clamp(cfg.dt_min, cfg.dt_max);
            }
        };
        let block = |k: usize| -> usize {
            if k < n_pose_vars {
                if k % 3 == 2 { 1 } else { 0 }
            } else {
                2
            }
        };

        let mut current = problem.cost(&self.band);
        for _ in 0..cfg.iterations {
            let mut work = self.band.clone();
            for (k, g) in grad.iter_mut().enumerate() {
                let o = get(&work, k);
                set(&mut work, k, o + eps);
                let cp = problem.cost(&work);
                set(&mut work, k, o - eps);
                let cm = problem.cost(&work);
                set(&mut work, k, o);
                *g = (cp - cm) / (2.0 * eps);
            }
            let mut g_inf = [0.0_f64; 3];
            for (k, g) in grad.iter().enumerate() {
                let b = block(k);
                g_inf[b] = g_inf[b].max(g.abs());
            }
            if g_inf.iter().all(|g| *g < 1e-9) {
                break;
            }
            let mut scale = 1.0;
            let mut improved = false;
            for _ in 0..6 {
                let mut cand = self.band.clone();
                for (k, g) in grad.iter().enumerate() {
                    let step = scale * cfg.step_size / g_inf[block(k)].max(1.0);
                    let o = get(&self.band, k);
                    set(&mut cand, k, o - step * g);
                }
                let c = problem.cost(&cand);
                if c < current {
                    let rel = (current - c) / current.abs().max(1e-12);
                    current = c;
                    self.band = cand;
                    improved = rel > 1e-4;
                    break;
                }
                scale *= 0.5;
            }
            if !improved {
                break;
            }
        }
    }
}

impl Controller for TebFollower {
    fn params(&self) -> serde_json::Value {
        serde_json::to_value(&self.teb_config).unwrap_or_default()
    }

    fn set_params(&mut self, params: &serde_json::Value) -> Result<(), String> {
        let cfg = crate::controller::merged(&self.teb_config, params)?;
        self.set_teb_config(cfg);
        Ok(())
    }

    fn compute_control(
        &mut self,
        state: &RobotState,
        goal: &Goal,
        constraints: &RobotConstraints,
        dt: f64,
        world: Option<&WorldConstraints>,
    ) -> VelocityCommand {
        let goal_key = (
            goal.target_pose.point.x,
            goal.target_pose.point.y,
            yaw_of(&goal.target_pose),
        );
        let stale_auto = self.auto_path_goal.is_some_and(|g| g != goal_key);
        if self.base.path.waypoints.is_empty() || stale_auto {
            let mut p = Path::default();
            p.waypoints.push(state.pose);
            p.waypoints.push(goal.target_pose);
            self.cursor.set_path(&p.waypoints);
            self.base.path = p;
            self.base.path_index = 0;
            self.band_initialised = false;
            self.auto_path_goal = Some(goal_key);
        }
        let prep = match prepare(&mut self.base, &mut self.cursor, state, goal, constraints, dt) {
            Ok(p) => p,
            Err(cmd) => {
                self.last_v = 0.0;
                return cmd;
            }
        };
        let checker = CollisionChecker::new(world, constraints, 0.0).with_spread(1.0);
        self.axle = axle_offset(constraints);
        let holonomic = is_holonomic(constraints.steering_type);
        let v_now = current_speed(state, self.last_v);
        let vy_now = if holonomic { current_lateral_speed(state, self.last_vy) } else { 0.0 };
        let n = self.teb_config.n_poses.max(3);
        let release = (0.5 * prep.pos_tol).max(0.05);
        let cusp = crate::core::path::cusp_after(&self.base.path.speeds, &self.cursor.cum, prep.proj.arc_length + release)
            .unwrap_or(f64::INFINITY);
        if !self.band_initialised || self.band.x.len() != n {
            self.initialise_band(state, prep.proj.arc_length, constraints, &checker, cusp);
        } else {
            self.advance_band(state, constraints, &checker, v_now, cusp);
        }

        let (v_lo, v_hi) = speed_bounds(constraints, prep.allow_reverse);
        let total = self.cursor.total_length();
        let mut target = self.path_pose(self.band.tail_s.min(total));
        let mut waypoints = self.base.path.waypoints.clone();
        let mut cum = self.cursor.cum.clone();
        let mut start_segment = prep.proj.segment;
        let obstacles: &[Obstacle] = world.map_or(&[], |w| &w.obstacles);
        let (guide, pace) = self.detour_guide(state, prep.proj.arc_length, constraints, obstacles);
        if let Some(guide) = guide {
            cum = cumulative_lengths(&guide);
            target = guide.last().map(|p| (p.point, yaw_of(p))).unwrap_or(target);
            waypoints = guide;
            start_segment = 0;
        }
        let v_hi_paced = v_hi * pace.clamp(0.0, 1.0);
        let cfg = self.teb_config.clone();
        let problem = Problem {
            cfg: &cfg,
            constraints,
            waypoints: &waypoints,
            cum: &cum,
            checker: &checker,
            target,
            kappa_max: max_curvature(constraints),
            v_lo,
            v_hi: v_hi_paced,
            v_start: v_now,
            w_start: if state.velocity_valid { state.velocity.angular } else { self.last_w },
            steer_per_wheelbase: if is_ackermann(constraints.steering_type) && constraints.max_steering_rate > 0.0 {
                constraints.max_steering_rate / wheelbase(constraints)
            } else {
                f64::INFINITY
            },
            ackermann: is_ackermann(constraints.steering_type),
            holonomic,
            start_segment,
            axle: self.axle,
        };
        // The band as warm-started and the same band at `SLOWER` times its
        // nominal timing, which meets the obstacles later (the time
        // alternative of homotopy-class planning), are both optimised against
        // every obstacle; the cheaper one that stays clear is driven.
        let warm = self.band.clone();
        self.optimise(&problem);
        if checker.has_obstacles() {
            let first = std::mem::replace(&mut self.band, warm);
            for dt in self.band.dt.iter_mut() {
                *dt = (cfg.dt_nominal * SLOWER).clamp(cfg.dt_min, cfg.dt_max);
            }
            self.optimise(&problem);
            let slower = std::mem::replace(&mut self.band, first);
            let score = |b: &Band| if problem.feasible(b, FEASIBILITY_POSES) { problem.cost(b) } else { f64::INFINITY };
            if score(&slower) < score(&self.band) {
                self.band = slower;
            }
        }
        // A band running into an obstacle is dropped, to be laid afresh next
        // tick, and the robot brakes meanwhile.
        if !problem.feasible(&self.band, FEASIBILITY_POSES) {
            self.band_initialised = false;
            let a_max = constraints.max_linear_acceleration.abs().max(1e-6);
            let v = v_now.signum() * (v_now.abs() - a_max * dt).max(0.0);
            self.base.status.mode = "teb_infeasible".into();
            let cmd = finalize(v, 0.0, constraints, &self.base.config, prep.allow_reverse, "TEB infeasible band");
            self.last_v = if cmd.valid { v } else { 0.0 };
            self.last_w = 0.0;
            return cmd;
        }

        let dt0 = self.band.dt[0].max(1e-3);
        let dx = self.band.x[1] - self.band.x[0];
        let dy = self.band.y[1] - self.band.y[0];
        let yaw0 = self.band.yaw[0];
        // The turn and speed are read over the first poses spanning
        // `control_lookahead` seconds (teb_local_planner's control look-ahead poses),
        // not one segment, which the last iterations of the solve jostle.
        let (mut k, mut span, mut length) = (1, dt0, dx.hypot(dy));
        while k + 1 < self.band.x.len() && span < self.teb_config.control_lookahead {
            span += self.band.dt[k].max(1e-3);
            length += (self.band.x[k + 1] - self.band.x[k]).hypot(self.band.y[k + 1] - self.band.y[k]);
            k += 1;
        }
        let w_band = normalize_angle(self.band.yaw[k] - self.band.yaw[0]) / span;
        let w = self.last_w + (w_band - self.last_w) * (dt / (self.teb_config.turn_filter + dt)).min(1.0);
        let a_max = constraints.max_linear_acceleration.abs().max(1e-6);

        if holonomic {
            let (bx, by) = world_to_body(dx, dy, yaw0);
            let vx = (bx / dt0).clamp(v_now - a_max * dt, v_now + a_max * dt);
            let vy = (by / dt0).clamp(vy_now - a_max * dt, vy_now + a_max * dt);
            self.base.status.mode = "teb_holonomic".into();
            let cmd = finalize_holonomic(vx, vy, w, constraints, &self.base.config, "TEB tracking");
            self.last_v = if cmd.valid { vx } else { 0.0 };
            self.last_vy = if cmd.valid { vy } else { 0.0 };
            return cmd;
        }

        let forward = (dx * yaw0.cos() + dy * yaw0.sin()).signum();
        // Capped to stop by the cusp, braking after the actuation delay.
        let lead = v_now.abs() * actuation_delay(constraints);
        // The stop holds for the cusp ahead of the pose, the band runs on past it.
        let stop_at = crate::core::path::cusp_after(&self.base.path.speeds, &self.cursor.cum, prep.proj.arc_length).unwrap_or(f64::INFINITY);
        let stop = (2.0 * a_max * (stop_at - prep.proj.arc_length - lead).max(0.0)).sqrt();
        // Braking to turn round, the ramp runs from the last command, not the
        // lagging measurement.
        let base = if forward * v_now < 0.0 { self.last_v } else { v_now };
        let v = (forward * length / span).clamp(base - a_max * dt, base + a_max * dt).clamp(-stop, stop);
        self.base.status.mode = "teb".into();
        let cmd = finalize(v, w, constraints, &self.base.config, prep.allow_reverse, "TEB tracking");
        self.last_v = if cmd.valid { v.clamp(v_lo, v_hi) } else { 0.0 };
        self.last_w = if cmd.valid { w } else { 0.0 };
        cmd
    }

    fn set_path(&mut self, path: Path) {
        self.cursor.set_path(&path.waypoints);
        self.base.path = path;
        self.base.path_index = 0;
        self.base.status = Default::default();
        self.band_initialised = false;
        self.auto_path_goal = None;
    }

    fn reset(&mut self) {
        self.set_path(Path::default());
        self.last_v = 0.0;
        self.last_w = 0.0;
        self.last_vy = 0.0;
    }

    fn get_type(&self) -> &'static str {
        "teb_follower"
    }

    fn predicted_trajectory(&self) -> Vec<Point> {
        self.band_poses()
    }

    fn base(&self) -> &ControllerBase {
        &self.base
    }

    fn base_mut(&mut self) -> &mut ControllerBase {
        &mut self.base
    }
}
