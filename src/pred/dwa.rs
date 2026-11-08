//! DWA (Fox, Burgard & Thrun 1997). Samples `(v, omega)` inside the dynamic
//! window reachable within one control period, rolls each pair out, drops
//! pairs that cannot stop before an obstacle, and maximises a weighted sum
//! of heading, clearance, velocity and goal-distance scores.

use crate::controller::{Controller, ControllerBase, effective_tolerances};
use crate::core::kinematics::{
    can_turn_in_place, finalize, finalize_holonomic, is_holonomic, max_curvature, reverse_allowed,
    speed_bounds,
};
use crate::core::math::{heading_error, normalize_angle};
use crate::core::obstacles::CollisionChecker;
use crate::types::{DebugFrame, DebugLevel, Goal, RobotConstraints, RobotState, VelocityCommand, WorldConstraints};
use datapod::Point;
use std::f64::consts::PI;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DwaConfig {
    pub predict_time: f64,
    pub dt: f64,
    pub v_samples: usize,
    pub w_samples: usize,
    /// Lateral-velocity samples for a holonomic platform; ignored (fixed at
    /// zero) for any other steering type.
    pub vy_samples: usize,
    pub weight_heading: f64,
    pub weight_distance: f64,
    pub weight_velocity: f64,
    pub weight_clearance: f64,
    pub target_velocity: f64,
    /// Clearance below which a rollout counts as a collision.
    pub obstacle_margin: f64,
    /// Clearance beyond which extra clearance is not rewarded.
    pub clearance_cap: f64,
}

impl Default for DwaConfig {
    fn default() -> Self {
        Self {
            predict_time: 1.5,
            dt: 0.1,
            v_samples: 10,
            w_samples: 21,
            vy_samples: 7,
            weight_heading: 1.0,
            weight_distance: 1.0,
            weight_velocity: 0.5,
            weight_clearance: 2.0,
            target_velocity: 1.0,
            obstacle_margin: 0.2,
            clearance_cap: 1.5,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct DwaFollower {
    pub base: ControllerBase,
    pub dwa_config: DwaConfig,
    last_v: f64,
    last_w: f64,
    last_vy: f64,
    debug_level: DebugLevel,
    debug: DebugFrame,
}

struct Sample {
    v: f64,
    vy: f64,
    w: f64,
    heading: f64,
    distance: f64,
    velocity: f64,
    clearance: f64,
}


impl DwaFollower {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_dwa_config(cfg: DwaConfig) -> Self {
        Self {
            base: ControllerBase::default(),
            dwa_config: cfg,
            last_v: 0.0,
            last_w: 0.0,
            last_vy: 0.0,
            debug_level: DebugLevel::Off,
            debug: DebugFrame::default(),
        }
    }

    pub fn set_dwa_config(&mut self, cfg: DwaConfig) {
        self.dwa_config = cfg;
    }
}

impl Controller for DwaFollower {
    fn params(&self) -> serde_json::Value {
        serde_json::to_value(&self.dwa_config).unwrap_or_default()
    }

    fn set_params(&mut self, params: &serde_json::Value) -> Result<(), String> {
        let cfg = crate::controller::merged(&self.dwa_config, params)?;
        self.set_dwa_config(cfg);
        Ok(())
    }

    fn set_debug(&mut self, level: DebugLevel) {
        self.debug_level = level;
        self.debug = DebugFrame::default();
    }

    fn debug_frame(&self) -> DebugFrame {
        self.debug.clone()
    }

    fn compute_control(
        &mut self,
        state: &RobotState,
        goal: &Goal,
        constraints: &RobotConstraints,
        dt: f64,
        world: Option<&WorldConstraints>,
    ) -> VelocityCommand {
        if !(dt.is_finite() && dt > 0.0) {
            return VelocityCommand::invalid("dt must be positive and finite");
        }
        let cfg = self.base.config.clone();
        let dwa = self.dwa_config.clone();
        let allow_reverse = reverse_allowed(&cfg, state);

        if let Some(cmd) = self.base.arrival(state, goal, constraints, false) {
            self.last_v = 0.0;
            self.last_w = cmd.angular_velocity;
            self.last_vy = 0.0;
            return cmd;
        }
        let holonomic = is_holonomic(constraints.steering_type);
        let dist_to_goal = self.base.status.distance_to_goal;
        let bearing = heading_error(&state.pose, goal.target_pose.point);
        self.base.status.heading_error = bearing;
        self.base.status.cross_track_error = 0.0;
        self.base.status.goal_reached = false;

        let (_, ang_tol) = effective_tolerances(goal, &cfg);
        if state.turn_first && can_turn_in_place(constraints.steering_type) && bearing.abs() > ang_tol
        {
            self.base.status.mode = "turning".into();
            let omega = cfg.kp_angular.max(0.1) * bearing;
            let cmd = finalize(0.0, omega, constraints, &cfg, allow_reverse, "Turning to align");
            self.last_v = 0.0;
            self.last_w = omega;
            self.last_vy = 0.0;
            return cmd;
        }

        let (v_lo, v_hi) = speed_bounds(constraints, allow_reverse);
        let w_lim = constraints.max_angular_velocity.abs();
        let v_now = if state.velocity_valid {
            state.velocity.linear
        } else {
            self.last_v
        }
        .clamp(v_lo, v_hi);
        let w_now = if state.velocity_valid {
            state.velocity.angular
        } else {
            self.last_w
        }
        .clamp(-w_lim, w_lim);
        let vy_lim = constraints.max_linear_velocity.abs();
        let vy_now = if holonomic {
            if state.velocity_valid {
                state.velocity.lateral
            } else {
                self.last_vy
            }
            .clamp(-vy_lim, vy_lim)
        } else {
            0.0
        };

        let a_lin = constraints.max_linear_acceleration.abs().max(1e-6);
        let a_ang = constraints.max_angular_acceleration.abs().max(1e-6);
        let v_min = (v_now - a_lin * dt).max(v_lo);
        let v_max = (v_now + a_lin * dt).min(v_hi);
        let w_min = (w_now - a_ang * dt).max(-w_lim);
        let w_max = (w_now + a_ang * dt).min(w_lim);
        let (vy_min, vy_max) = if holonomic {
            ((vy_now - a_lin * dt).max(-vy_lim), (vy_now + a_lin * dt).min(vy_lim))
        } else {
            (0.0, 0.0)
        };

        let v_samples = dwa.v_samples.max(2);
        let w_samples = dwa.w_samples.max(3) | 1;
        let vy_samples = if holonomic { dwa.vy_samples.max(3) | 1 } else { 1 };
        let sim_dt = dwa.dt.max(1e-3);
        let steps = (dwa.predict_time / sim_dt).ceil().max(1.0) as usize;
        let kappa_max = max_curvature(constraints);
        let checker = CollisionChecker::new(world, constraints, 0.0).with_step(sim_dt).with_spread(2.0);
        let goal_p = goal.target_pose.point;
        let target_v = dwa
            .target_velocity
            .min(v_hi)
            .min((cfg.kp_linear.max(0.1) * dist_to_goal).max(0.15 * v_hi));

        let x0 = state.pose.point.x;
        let y0 = state.pose.point.y;
        let yaw0 = state.pose.rotation.to_euler().yaw;

        // Against obstacles standing or crossing the course to the goal, while
        // braking from here keeps clear, arcs that would leave the robot unable
        // to stop clear of them are refused, so the robot gives way.
        let course = normalize_angle(yaw0 + bearing);
        let yield_to: Option<WorldConstraints> = world.map(|w| WorldConstraints {
            obstacles: w.obstacles.iter().filter(|o| !along_heading(o, course)).cloned().collect(),
            ..Default::default()
        });
        let tail = yield_to
            .as_ref()
            .filter(|w| !w.obstacles.is_empty())
            .map(|w| CollisionChecker::new(Some(w), constraints, 0.0).with_step(sim_dt).with_spread(2.0));
        let yielding = tail.as_ref().is_some_and(|c| !hit_while_braking(c, (x0, y0, yaw0), v_now.abs(), a_lin, 0, sim_dt));
        let mut samples: Vec<Sample> = Vec::with_capacity(v_samples * w_samples * vy_samples);
        for vi in 0..v_samples {
            let v = v_min + (v_max - v_min) * vi as f64 / (v_samples - 1) as f64;
            for vyi in 0..vy_samples {
                let vy = if vy_samples > 1 {
                    vy_min + (vy_max - vy_min) * vyi as f64 / (vy_samples - 1) as f64
                } else {
                    0.0
                };
                let speed_mag = v.hypot(vy);
                for wi in 0..w_samples {
                    let w = w_min + (w_max - w_min) * wi as f64 / (w_samples - 1) as f64;
                    if kappa_max.is_finite() && w.abs() > v.abs() * kappa_max + 1e-9 {
                        continue;
                    }
                    let (mut x, mut y, mut yaw) = (x0, y0, yaw0);
                    let step_len = speed_mag * sim_dt;
                    let mut travelled = 0.0;
                    let mut horizon = 0.0;
                    let mut prev_clearance = checker.clearance(0, x, y, yaw);
                    let start_clearance = prev_clearance;
                    let mut min_clearance = prev_clearance;
                    let mut free_distance: Option<f64> = None;
                    let mut stop_at = ((x, y, yaw), 0);
                    for step in 1..=steps {
                        let before = ((x, y, yaw), step - 1);
                        if holonomic {
                            let (s, c) = yaw.sin_cos();
                            x += (v * c - vy * s) * sim_dt;
                            y += (v * s + vy * c) * sim_dt;
                        } else {
                            x += v * yaw.cos() * sim_dt;
                            y += v * yaw.sin() * sim_dt;
                        }
                        yaw = normalize_angle(yaw + w * sim_dt);
                        let c = checker.clearance(step, x, y, yaw);
                        min_clearance = min_clearance.min(c);
                        let entering = c < dwa.obstacle_margin
                            && c < prev_clearance - 1e-9
                            && c < start_clearance;
                        let deeper = c < 0.0 && c < prev_clearance - 1e-9;
                        if deeper || entering {
                            let boundary = if c < 0.0 { 0.0 } else { dwa.obstacle_margin };
                            let remaining = (prev_clearance - boundary).clamp(0.0, step_len);
                            free_distance = Some(travelled + remaining);
                            stop_at = before;
                            break;
                        }
                        travelled += step_len;
                        prev_clearance = c;
                        horizon += step_len;
                        if horizon >= dist_to_goal {
                            break;
                        }
                    }
                    let ((end, end_step), end_speed) = match free_distance {
                        Some(_) => (stop_at, 0.0),
                        None => (((x, y, yaw), steps), speed_mag),
                    };
                    if yielding && tail.as_ref().is_some_and(|c| hit_while_braking(c, end, end_speed, a_lin, end_step, sim_dt)) {
                        continue;
                    }
                    let clearance = match free_distance {
                        Some(s_free) => {
                            if speed_mag * speed_mag > 2.0 * a_lin * s_free + 1e-9 {
                                continue;
                            }
                            s_free
                        }
                        None => {
                            speed_mag * dwa.predict_time
                                + (min_clearance - dwa.obstacle_margin).clamp(0.0, dwa.clearance_cap)
                        }
                    };
                    let dx = goal_p.x - x;
                    let dy = goal_p.y - y;
                    let mut heading_err = normalize_angle(dy.atan2(dx) - yaw).abs();
                    if v < 0.0 && vy.abs() < 1e-9 {
                        heading_err = PI - heading_err;
                    }
                    samples.push(Sample {
                        v,
                        vy,
                        w,
                        heading: 1.0 - heading_err / PI,
                        distance: dx.hypot(dy),
                        velocity: 1.0 - ((speed_mag - target_v).abs() / target_v.max(1e-6)).min(1.0),
                        clearance,
                    });
                }
            }
        }

        if samples.is_empty()
            && let Some((v, w)) = evade(&checker, (x0, y0, yaw0), (v_min, v_max, v_samples), (w_min, w_max, w_samples), steps, sim_dt, kappa_max)
        {
            self.base.status.mode = "dwa_evade".into();
            let cmd = finalize(v, w, constraints, &cfg, allow_reverse, "DWA: evading");
            self.last_v = v;
            self.last_w = w;
            self.last_vy = 0.0;
            return cmd;
        }
        if samples.is_empty() {
            self.base.status.mode = "dwa_blocked".into();
            let cmd = finalize(0.0, 0.0, constraints, &cfg, allow_reverse, "DWA: no admissible velocity");
            self.last_v = 0.0;
            self.last_w = 0.0;
            self.last_vy = 0.0;
            return cmd;
        }

        let d_max = samples.iter().map(|s| s.distance).fold(0.0, f64::max).max(1e-9);
        let d_min = samples.iter().map(|s| s.distance).fold(f64::INFINITY, f64::min);
        let c_max = samples.iter().map(|s| s.clearance).fold(0.0, f64::max).max(1e-9);
        let mut best: Option<(f64, f64, f64, f64)> = None;
        let mut scores = Vec::with_capacity(samples.len());
        for s in &samples {
            let distance_score = if d_max - d_min > 1e-9 {
                (d_max - s.distance) / (d_max - d_min)
            } else {
                1.0
            };
            let score = dwa.weight_heading * s.heading
                + dwa.weight_distance * distance_score
                + dwa.weight_velocity * s.velocity
                + dwa.weight_clearance * (s.clearance / c_max);
            if best.is_none_or(|b| score > b.0) {
                best = Some((score, s.v, s.vy, s.w));
            }
            scores.push(score);
        }
        if self.debug_level == DebugLevel::Full {
            let stride = (samples.len() / SHOWN_ARCS).max(1);
            (self.debug.rollouts, self.debug.costs) = samples
                .iter()
                .zip(&scores)
                .step_by(stride)
                .map(|(s, score)| (arc((x0, y0, yaw0), (s.v, s.vy, s.w), steps, sim_dt, holonomic), -score))
                .unzip();
        }
        let (_, mut v, mut vy, mut w) = best.unwrap();

        let mut mode = "dwa";
        if v.hypot(vy).abs() < 1e-9
            && dist_to_goal > cfg.goal_tolerance
            && let Some(turn) = recovery_turn(&checker, x0, y0, yaw0, dwa.obstacle_margin)
        {
            if can_turn_in_place(constraints.steering_type) {
                v = 0.0;
                vy = 0.0;
                w = turn * (w_now.abs() + a_ang * dt).min(w_lim).max(0.2 * w_lim);
                mode = "dwa_recovery";
            } else if allow_reverse
                && let Some((rv, rw)) = reverse_arc(&checker, x0, y0, yaw0, turn, constraints, dwa.obstacle_margin)
            {
                v = rv;
                vy = 0.0;
                w = rw;
                mode = "dwa_recovery_reverse";
            }
        }

        self.base.status.mode = mode.into();
        let cmd = if holonomic {
            finalize_holonomic(v, vy, w, constraints, &cfg, "DWA tracking")
        } else {
            finalize(v, w, constraints, &cfg, allow_reverse, "DWA tracking")
        };
        self.last_v = v;
        self.last_w = w;
        self.last_vy = vy;
        cmd
    }

    fn reset(&mut self) {
        self.base.path = Default::default();
        self.base.path_index = 0;
        self.base.status = Default::default();
        self.last_v = 0.0;
        self.last_w = 0.0;
        self.last_vy = 0.0;
    }

    fn get_type(&self) -> &'static str {
        "dwa_follower"
    }

    fn base(&self) -> &ControllerBase {
        &self.base
    }

    fn base_mut(&mut self) -> &mut ControllerBase {
        &mut self.base
    }
}

/// Whether the robot, braking straight from `(x, y, yaw)` at `speed` with
/// `decel` after `steps` of `dt`, is hit by an obstacle moving in on it
/// before it stops or within `BRAKING_TAIL_S` of standing after: an arc
/// ending there leads to a collision
/// it can no longer avoid (an inevitable collision state, Fraichard 2004).
fn hit_while_braking(
    checker: &CollisionChecker,
    (mut x, mut y, yaw): (f64, f64, f64),
    mut speed: f64,
    decel: f64,
    steps: usize,
    dt: f64,
) -> bool {
    if !checker.has_obstacles() {
        return false;
    }
    let tail = ((speed / decel.max(1e-3) + BRAKING_TAIL_S) / dt).ceil() as usize;
    let (s, c) = yaw.sin_cos();
    for k in 1..=tail {
        speed = (speed - decel * dt).max(0.0);
        x += speed * c * dt;
        y += speed * s * dt;
        if checker.clearance(steps + k, x, y, yaw) < 0.0 {
            return true;
        }
    }
    false
}

/// Whether `o` moves along the line of a robot heading `yaw` (more along
/// than across, faster than a crawl), towards it or ahead of it, rather
/// than stands or crosses: traffic stopping does not get clear of.
fn along_heading(o: &crate::types::Obstacle, yaw: f64) -> bool {
    use crate::core::obstacles::modes_at;
    let (Some(now), Some(soon)) = (modes_at(o, 0.0, 0.1).next(), modes_at(o, 1.0, 0.1).next()) else {
        return false;
    };
    let (vx, vy) = (soon.x - now.x, soon.y - now.y);
    let (s, c) = yaw.sin_cos();
    vx.hypot(vy) > 0.1 && (vx * c + vy * s).abs() >= (vx * -s + vy * c).abs()
}

/// Seconds of standing, after braking to a stop at the end of an arc, that
/// must stay clear.
const BRAKING_TAIL_S: f64 = 2.0;

/// Arcs shown at most when the controller records everything.
const SHOWN_ARCS: usize = 48;

/// The arc `(v, vy, w)` held from `(x, y, yaw)` for `steps` of `dt`.
fn arc((x0, y0, yaw0): (f64, f64, f64), (v, vy, w): (f64, f64, f64), steps: usize, dt: f64, holonomic: bool) -> Vec<Point> {
    let (mut x, mut y, mut yaw) = (x0, y0, yaw0);
    let mut out = vec![Point::new(x, y, 0.0)];
    for _ in 0..steps {
        let (s, c) = yaw.sin_cos();
        if holonomic {
            x += (v * c - vy * s) * dt;
            y += (v * s + vy * c) * dt;
        } else {
            x += v * c * dt;
            y += v * s * dt;
        }
        yaw = normalize_angle(yaw + w * dt);
        out.push(Point::new(x, y, 0.0));
    }
    out
}

/// The `(v, w)` of the window whose arc keeps the largest clearance from
/// the obstacles as they move over `steps` of `dt`: the way out when no
/// velocity is admissible and an obstacle closes on the robot where it
/// stands, as one coming head on does. `None` when standing still stays
/// clear over the steps and `BRAKING_TAIL_S` more, so stopping is safe.
fn evade(
    checker: &CollisionChecker,
    (x0, y0, yaw0): (f64, f64, f64),
    (v_min, v_max, v_n): (f64, f64, usize),
    (w_min, w_max, w_n): (f64, f64, usize),
    steps: usize,
    dt: f64,
    kappa_max: f64,
) -> Option<(f64, f64)> {
    if !checker.has_obstacles() {
        return None;
    }
    let hold = steps + (BRAKING_TAIL_S / dt).ceil() as usize;
    if (1..=hold).all(|k| checker.clearance(k, x0, y0, yaw0) > 0.0) {
        return None;
    }
    let staying = (1..=steps).map(|k| checker.clearance(k, x0, y0, yaw0)).fold(f64::INFINITY, f64::min);
    let (v_n, w_n) = (v_n.max(2), w_n.max(2));
    let mut best: Option<(f64, f64, f64)> = Some((staying, 0.0, 0.0));
    for vi in 0..v_n {
        let v = v_min + (v_max - v_min) * vi as f64 / (v_n - 1) as f64;
        for wi in 0..w_n {
            let w = w_min + (w_max - w_min) * wi as f64 / (w_n - 1) as f64;
            if kappa_max.is_finite() && w.abs() > v.abs() * kappa_max + 1e-9 {
                continue;
            }
            let (mut x, mut y, mut yaw) = (x0, y0, yaw0);
            let mut worst = f64::INFINITY;
            for step in 1..=steps {
                x += v * yaw.cos() * dt;
                y += v * yaw.sin() * dt;
                yaw = normalize_angle(yaw + w * dt);
                worst = worst.min(checker.clearance(step, x, y, yaw));
            }
            if best.is_none_or(|b| worst > b.0) {
                best = Some((worst, v, w));
            }
        }
    }
    best.map(|(_, v, w)| (v, w))
}

/// Free straight-line distance from `(x, y)` along `heading` before contact
/// or a drop below the current clearance inside the margin.
fn probe_free_distance(
    checker: &CollisionChecker,
    x: f64,
    y: f64,
    heading: f64,
    margin: f64,
    length: f64,
) -> f64 {
    let step = 0.05;
    let start = checker.clearance(0, x, y, heading);
    let (s, c) = heading.sin_cos();
    let mut travelled = 0.0;
    while travelled < length {
        travelled += step;
        let cl = checker.clearance(0, x + travelled * c, y + travelled * s, heading);
        if cl < 0.0 || (cl < margin && cl < start) {
            return travelled - step;
        }
    }
    length
}

/// Direction (+1 left, -1 right) to rotate toward the heading with the most
/// free space when no forward sample is admissible, or `None` when no other
/// heading is better than the current one.
fn recovery_turn(
    checker: &CollisionChecker,
    x: f64,
    y: f64,
    yaw: f64,
    margin: f64,
) -> Option<f64> {
    const PROBE: f64 = 1.5;
    let current = probe_free_distance(checker, x, y, yaw, margin, PROBE);
    let mut best: Option<(f64, f64)> = None;
    for k in 1..=6 {
        for sign in [1.0, -1.0] {
            let angle = sign * k as f64 * PI / 6.0;
            let free = probe_free_distance(checker, x, y, yaw + angle, margin, PROBE);
            let score = free - 0.05 * k as f64;
            if free > current + 0.1 && best.is_none_or(|b| score > b.0) {
                best = Some((score, sign));
            }
        }
    }
    best.map(|b| b.1)
}

/// A car's way out when nothing ahead is admissible: back up along an arc
/// that swings the nose toward `turn` (+1 left, -1 right), at full lock,
/// half lock or straight, whichever stays clear for `REVERSE_PROBE_S`.
/// Returns the `(v, omega)` to command.
fn reverse_arc(
    checker: &CollisionChecker,
    x: f64,
    y: f64,
    yaw: f64,
    turn: f64,
    constraints: &RobotConstraints,
    margin: f64,
) -> Option<(f64, f64)> {
    const REVERSE_PROBE_S: f64 = 1.5;
    const STEP: f64 = 0.1;
    let (lo, _) = speed_bounds(constraints, true);
    let v = lo.max(-0.5 * constraints.max_linear_velocity.abs()).min(-0.05);
    let kmax = max_curvature(constraints).min(10.0);
    for k in [-turn * kmax, -turn * 0.5 * kmax, 0.0] {
        let (mut px, mut py, mut th) = (x, y, yaw);
        let mut clear = true;
        let mut t = 0.0;
        while t < REVERSE_PROBE_S {
            px += v * th.cos() * STEP;
            py += v * th.sin() * STEP;
            th += v * k * STEP;
            t += STEP;
            if checker.clearance_at(t, px, py, th) < margin {
                clear = false;
                break;
            }
        }
        if clear {
            return Some((v, v * k));
        }
    }
    None
}
