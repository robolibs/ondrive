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
    finalize, finalize_holonomic, is_ackermann, is_holonomic, max_curvature, speed_bounds,
    world_to_body,
};
use crate::core::math::{normalize_angle, yaw_of};
use crate::core::obstacles::CollisionChecker;
use crate::core::path::{PathCursor, project, sample};
use crate::pred::mppi::{current_lateral_speed, current_speed, prepare};
use crate::types::{Goal, Path, RobotConstraints, RobotState, VelocityCommand, WorldConstraints};
use datapod::{Point, Pose};

#[derive(Clone, Debug)]
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
}

impl Default for TebConfig {
    fn default() -> Self {
        Self {
            n_poses: 6,
            dt_nominal: 0.3,
            iterations: 30,
            step_size: 0.05,
            weight_time: 1.0,
            weight_velocity_limit: 50.0,
            weight_angular_limit: 50.0,
            weight_acceleration_limit: 10.0,
            weight_path_deviation: 1.0,
            weight_obstacle: 100.0,
            weight_kinematic: 50.0,
            weight_goal: 5.0,
            obstacle_margin: 0.3,
            dt_min: 0.05,
            dt_max: 1.0,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct TebFollower {
    pub base: ControllerBase,
    pub teb_config: TebConfig,
    cursor: PathCursor,
    band: Band,
    band_initialised: bool,
    last_v: f64,
    last_vy: f64,
    auto_path_goal: Option<(f64, f64, f64)>,
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
    ackermann: bool,
    holonomic: bool,
}

impl Problem<'_> {
    fn obstacle_cost(&self, step: usize, x: f64, y: f64, yaw: f64) -> f64 {
        let c = self.cfg;
        if !self.checker.has_obstacles() {
            return 0.0;
        }
        let d = self.checker.clearance(step, x, y, yaw);
        if d < c.obstacle_margin {
            let s = c.obstacle_margin - d;
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
        let mut last_v = 0.0;

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

            cost += self.obstacle_cost(i, 0.5 * (b.x[i] + b.x[i + 1]), 0.5 * (b.y[i] + b.y[i + 1]), b.yaw[i]);
        }
        if let Some((_, pdt)) = prev_v {
            let a_goal = last_v / pdt.max(1e-4);
            let a_over = (a_goal.abs() - a_max).max(0.0);
            cost += 0.25 * c.weight_acceleration_limit * a_over * a_over;
        }

        if !self.waypoints.is_empty() {
            let mut hint = 0;
            for i in 1..n {
                if let Some(p) = project(
                    self.waypoints,
                    self.cum,
                    Point::new(b.x[i], b.y[i], 0.0),
                    hint,
                    2,
                    usize::MAX,
                ) {
                    cost += c.weight_path_deviation * p.distance * p.distance;
                    hint = p.segment;
                }
            }
        }

        for i in 0..n {
            cost += self.obstacle_cost(i, b.x[i], b.y[i], b.yaw[i]);
        }

        let (tp, th) = self.target;
        let dx = b.x[n - 1] - tp.x;
        let dy = b.y[n - 1] - tp.y;
        let dh = normalize_angle(b.yaw[n - 1] - th);
        cost += c.weight_goal * (dx * dx + dy * dy + 0.5 * dh * dh);
        cost
    }
}

/// Move a sampled band pose sideways off any obstacle it would sit on, so
/// the optimiser sees a lateral gradient instead of a symmetric saddle.
fn push_off_obstacles(
    p: Point,
    heading: f64,
    step: usize,
    checker: &CollisionChecker,
    margin: f64,
) -> Point {
    if !checker.has_obstacles() {
        return p;
    }
    let c = checker.clearance(step, p.x, p.y, heading);
    if c >= margin {
        return p;
    }
    let (s, co) = heading.sin_cos();
    let shift = margin - c;
    let left = Point::new(p.x - shift * s, p.y + shift * co, 0.0);
    let right = Point::new(p.x + shift * s, p.y - shift * co, 0.0);
    let cl = checker.clearance(step, left.x, left.y, heading);
    let cr = checker.clearance(step, right.x, right.y, heading);
    if cr > cl { right } else { left }
}

impl TebFollower {
    pub fn new() -> Self {
        Self::default()
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
        self.band
            .x
            .iter()
            .zip(self.band.y.iter())
            .map(|(&x, &y)| Point::new(x, y, 0.0))
            .collect()
    }

    fn nominal_step(&self, constraints: &RobotConstraints) -> f64 {
        (constraints.max_linear_velocity.abs() * self.teb_config.dt_nominal).max(0.05)
    }

    fn sample_band_pose(&self, s: f64, step: usize, checker: &CollisionChecker) -> (Point, f64) {
        let (p, h) = sample(&self.base.path.waypoints, &self.cursor.cum, s);
        let p = push_off_obstacles(p, h, step, checker, self.teb_config.obstacle_margin);
        (p, h)
    }

    fn initialise_band(
        &mut self,
        state: &RobotState,
        s0: f64,
        constraints: &RobotConstraints,
        checker: &CollisionChecker,
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
        b.x[0] = state.pose.point.x;
        b.y[0] = state.pose.point.y;
        b.yaw[0] = state.pose.rotation.to_euler().yaw;
        for i in 1..n {
            let s = s0 + i as f64 * step;
            let (p, h) = self.sample_band_pose(s, i, checker);
            b.x[i] = p.x;
            b.y[i] = p.y;
            b.yaw[i] = h;
            b.tail_s = s;
        }
        self.band = b;
        self.band_initialised = true;
    }

    /// Re-anchor pose 0 to the robot, drop every pose the robot has passed
    /// and keep the first interval consistent with the current speed.
    fn advance_band(
        &mut self,
        state: &RobotState,
        constraints: &RobotConstraints,
        checker: &CollisionChecker,
        v_now: f64,
    ) {
        let n = self.band.x.len();
        let step = self.nominal_step(constraints);
        let px = state.pose.point.x;
        let py = state.pose.point.y;
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
            self.band.tail_s += step;
            let (p, h) = self.sample_band_pose(self.band.tail_s, n - 1, checker);
            self.band.x.push(p.x);
            self.band.y.push(p.y);
            self.band.yaw.push(h);
            self.band.dt.push(self.teb_config.dt_nominal);
        }
        self.band.x[0] = px;
        self.band.y[0] = py;
        self.band.yaw[0] = state.pose.rotation.to_euler().yaw;
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
        let checker = CollisionChecker::new(world, constraints, 0.0);
        let holonomic = is_holonomic(constraints.steering_type);
        let v_now = current_speed(state, self.last_v);
        let vy_now = if holonomic { current_lateral_speed(state, self.last_vy) } else { 0.0 };
        let n = self.teb_config.n_poses.max(3);
        if !self.band_initialised || self.band.x.len() != n {
            self.initialise_band(state, prep.proj.arc_length, constraints, &checker);
        } else {
            self.advance_band(state, constraints, &checker, v_now);
        }

        let (v_lo, v_hi) = speed_bounds(constraints, prep.allow_reverse);
        let total = self.cursor.total_length();
        let target = sample(&self.base.path.waypoints, &self.cursor.cum, self.band.tail_s.min(total));
        let waypoints = self.base.path.waypoints.clone();
        let cum = self.cursor.cum.clone();
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
            v_hi,
            v_start: v_now,
            ackermann: is_ackermann(constraints.steering_type),
            holonomic,
        };
        self.optimise(&problem);

        let dt0 = self.band.dt[0].max(1e-3);
        let dx = self.band.x[1] - self.band.x[0];
        let dy = self.band.y[1] - self.band.y[0];
        let yaw0 = self.band.yaw[0];
        let w = normalize_angle(self.band.yaw[1] - self.band.yaw[0]) / dt0;
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

        let ds = dx.hypot(dy);
        let forward = (dx * yaw0.cos() + dy * yaw0.sin()).signum();
        let v = (forward * ds / dt0).clamp(v_now - a_max * dt, v_now + a_max * dt);

        self.base.status.mode = "teb".into();
        let cmd = finalize(v, w, constraints, &self.base.config, prep.allow_reverse, "TEB tracking");
        self.last_v = if cmd.valid { v.clamp(v_lo, v_hi) } else { 0.0 };
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
