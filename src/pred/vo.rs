//! Velocity-obstacle path following. Each tick the command the path asks
//! for, a pursuit of the point one lookahead ahead of the robot, is kept
//! unless, held for the next `horizon_s`, it brings the robot into an
//! obstacle's predicted track sooner than the robot could stop. Then the
//! reachable command nearest to it that keeps every obstacle beyond that
//! time is taken; when none does, the one that meets them latest.

use crate::controller::{Controller, ControllerBase};
use crate::core::kinematics::{
    body_heading_on_path, braked_for_bends, finalize, is_ackermann, max_curvature, path_speed, wheelbase,
};
use crate::core::math::normalize_angle;
use crate::core::obstacles::{CollisionChecker, arc_course, robot_radius};
use crate::core::path::{PathCursor, curvature_at_arc_length, cusp_after, direction_at, sample, sharpest_ahead, speed_cap};
use crate::pred::mppi::{Swerve, build_reference, pick_detour, prepare};
use crate::types::{DebugFrame, DebugLevel, Goal, Path, RobotConstraints, RobotState, VelocityCommand, WorldConstraints};
use datapod::Point;
use std::f64::consts::PI;

/// Step, seconds, the detour round what blocks the path is laid out at,
/// and the share of top speed it is laid out for.
const DETOUR_DT: f64 = 0.2;
const DETOUR_SPEED_SHARE: f64 = 0.8;
/// Steps the detour runs on past the lookahead point.
const DETOUR_TAIL: usize = 6;
/// Share of the body's width kept as room from obstacles, when that is more
/// than `margin`.
const WIDTH_ROOM_SHARE: f64 = 0.25;
/// Lookahead shrink per unit curvature of the sharpest followable bend
/// within reach.
const LOOKAHEAD_BEND_GAIN: f64 = 1.0;
/// Metres a robot inside its margin may drift closer before a command counts
/// as meeting the obstacle.
const MARGIN_SLACK: f64 = 0.02;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct VelocityObstacleConfig {
    /// Seconds a command is held and checked against the obstacles.
    pub horizon_s: f64,
    /// Step, seconds, of that check.
    pub dt: f64,
    /// Speeds and curvatures sampled across the reachable window.
    pub speed_samples: usize,
    pub curvature_samples: usize,
    /// Seconds of reaction added to the stopping time every obstacle must
    /// stay beyond.
    pub reaction_s: f64,
    /// Room, metres, kept beyond the footprint.
    pub margin: f64,
    /// Weight of a change of curvature against one of speed, both taken
    /// relative to their range.
    pub weight_turn: f64,
    /// Seconds over which the window of reachable commands is taken.
    pub window_s: f64,
}

impl Default for VelocityObstacleConfig {
    fn default() -> Self {
        Self {
            horizon_s: 5.0,
            dt: 0.1,
            speed_samples: 9,
            curvature_samples: 25,
            reaction_s: 1.0,
            margin: 0.3,
            weight_turn: 1.0,
            window_s: 1.0,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct VelocityObstacleFollower {
    pub base: ControllerBase,
    pub vo_config: VelocityObstacleConfig,
    cursor: PathCursor,
    plan: Vec<Point>,
    side: f64,
    debug_level: DebugLevel,
}

impl VelocityObstacleFollower {
    pub fn new() -> Self {
        Self::default()
    }
}

/// One command tried: speed along the direction of travel, curvature of
/// that travel, and how soon held it meets an obstacle.
struct Candidate {
    speed: f64,
    curvature: f64,
    meets: f64,
}

impl Controller for VelocityObstacleFollower {
    fn compute_control(
        &mut self,
        state: &RobotState,
        goal: &Goal,
        constraints: &RobotConstraints,
        dt: f64,
        world: Option<&WorldConstraints>,
    ) -> VelocityCommand {
        let prep = match prepare(&mut self.base, &mut self.cursor, state, goal, constraints, dt) {
            Ok(p) => p,
            Err(cmd) => return cmd,
        };
        let cfg = self.base.config.clone();
        let vo = self.vo_config.clone();
        let (waypoints, speeds, cum) = (&self.base.path.waypoints, &self.base.path.speeds, &self.cursor.cum);
        let (x, y, yaw) = (state.pose.point.x, state.pose.point.y, state.pose.rotation.to_euler().yaw);
        let s = prep.proj.arc_length;
        let length = self.cursor.total_length();
        let cap = cusp_after(speeds, cum, s).unwrap_or(length);
        let back = prep.allow_reverse && direction_at(speeds, &prep.proj) == Some(-1);
        let body_turn = if back { PI } else { 0.0 };
        let travel = yaw + body_turn;
        // An Ackermann machine steers by its rear axle, which moves along its
        // heading; the pose rides `axle` ahead of it on the body.
        let axle = if is_ackermann(constraints.steering_type) { constraints.rear_wheelbase.max(0.0) } else { 0.0 };
        let (rx, ry) = (x - axle * yaw.cos(), y - axle * yaw.sin());

        // The command the path asks for: pursuit of the point a lookahead on,
        // taken on the cheaper detour round what blocks the path ahead.
        let half_width = crate::core::obstacles::footprint_extent(constraints).map_or(0.0, |(_, _, w)| w);
        let margin = vo.margin.max(WIDTH_ROOM_SHARE * 2.0 * half_width);
        let checker = CollisionChecker::new(world, constraints, margin).with_step(vo.dt).with_spread(1.0);
        let base_lookahead = cfg.lookahead_distance.max(0.5);
        let bend = sharpest_ahead(waypoints, cum, s, base_lookahead, max_curvature(constraints));
        let lookahead = (base_lookahead / (1.0 + LOOKAHEAD_BEND_GAIN * bend * base_lookahead)).clamp(0.5 * base_lookahead, base_lookahead);
        let (mut target, mut target_heading) = sample(waypoints, cum, (s + lookahead).min(cap));
        let obstacles = world.map_or(&[][..], |w| &w.obstacles[..]);
        if obstacles.is_empty() {
            self.side = 0.0;
        } else {
            let speed = (DETOUR_SPEED_SHARE * constraints.max_linear_velocity).max(0.1);
            let ahead = (lookahead.min(cap - s).max(0.0) / (speed * DETOUR_DT)).round() as usize;
            let reference = build_reference(&self.base.path, cum, s, ahead + DETOUR_TAIL, DETOUR_DT, speed, 0.0);
            let room = robot_radius(constraints) + margin;
            if let Some(g) = pick_detour(&reference, obstacles, room, DETOUR_DT, &mut self.side, Swerve::of(constraints, speed)) {
                let k = ahead.min(g.x.len() - 1);
                target = Point::new(g.x[k], g.y[k], 0.0);
                target_heading = g.yaw[k];
            }
        }
        // The rear axle aims where it stands when the pose is at the target,
        // facing along the way there.
        let bend = curvature_at_arc_length(waypoints, cum, (s + lookahead).min(cap));
        let body = body_heading_on_path(target_heading, bend, axle, back);
        let (dx, dy) = (target.x - axle * body.cos() - rx, target.y - axle * body.sin() - ry);
        let reach = dx.hypot(dy).max(1e-6);
        let alpha = normalize_angle(dy.atan2(dx) - travel);
        let k_lim = if max_curvature(constraints).is_finite() {
            max_curvature(constraints)
        } else {
            constraints.max_angular_velocity.abs() / 0.3
        };
        let k_pref = (2.0 * alpha.sin() / reach).clamp(-k_lim, k_lim);
        let nominal = speed_cap(speeds, &prep.proj).map_or(constraints.max_linear_velocity, |c| c.min(constraints.max_linear_velocity));
        let left = self.base.distance_left(&prep.proj, length).min(cap - s);
        let v_pref = path_speed(nominal, k_pref, left, prep.pos_tol, cfg.kp_linear, constraints);
        let v_measured = if state.velocity_valid { state.velocity.linear } else { v_pref };
        let v_pref = braked_for_bends(v_pref, v_measured, nominal, left, |d| curvature_at_arc_length(waypoints, cum, s + d), constraints);

        // Commands reachable within the window.
        let v_max = constraints.max_linear_velocity.abs();
        let accel = constraints.max_linear_acceleration.abs().max(0.1);
        let v_now = if state.velocity_valid { state.velocity.linear.abs() } else { v_pref };
        let (v_lo, v_hi) = ((v_now - accel * vo.window_s).max(0.0), (v_now + accel * vo.window_s).min(v_max));
        let k_now = if state.velocity_valid && state.velocity.linear.abs() > 0.1 {
            state.velocity.angular / state.velocity.linear.abs()
        } else {
            k_pref
        };
        let k_step = if is_ackermann(constraints.steering_type) && constraints.max_steering_rate > 0.0 {
            constraints.max_steering_rate * vo.window_s / wheelbase(constraints).max(0.1)
        } else {
            2.0 * k_lim
        };
        let (k_lo, k_hi) = ((k_now - k_step).max(-k_lim), (k_now + k_step).min(k_lim));
        let mut speeds_tried: Vec<f64> = (0..vo.speed_samples.max(2))
            .map(|i| v_lo + (v_hi - v_lo) * i as f64 / (vo.speed_samples.max(2) - 1) as f64)
            .collect();
        speeds_tried.extend([0.0, v_pref.clamp(v_lo, v_hi)]);
        let mut curvatures: Vec<f64> = (0..vo.curvature_samples.max(3))
            .map(|i| k_lo + (k_hi - k_lo) * i as f64 / (vo.curvature_samples.max(3) - 1) as f64)
            .collect();
        curvatures.push(k_pref.clamp(k_lo, k_hi));

        // How soon each command meets an obstacle, held for `window_s` and
        // driven straight on after: a turn away, not a circle back.
        let steps = (vo.horizon_s / vo.dt).ceil() as usize;
        let course_of = |speed: f64, curvature: f64| {
            let turning = arc_course((rx, ry, travel), curvature);
            let turned = speed * vo.window_s;
            let (tx, ty, th) = turning(turned);
            move |d: f64| {
                let (cx, cy, ch) =
                    if d <= turned { turning(d) } else { (tx + (d - turned) * th.cos(), ty + (d - turned) * th.sin(), th) };
                let body = ch - body_turn;
                (cx + axle * body.cos(), cy + axle * body.sin(), ch)
            }
        };
        // Already inside the margin, only coming closer still counts.
        let floor = if checker.has_obstacles() { checker.clearance(0, x, y, yaw).min(0.0) - MARGIN_SLACK } else { 0.0 };
        let meets = |speed: f64, curvature: f64| -> f64 {
            if !checker.has_obstacles() {
                return f64::INFINITY;
            }
            let course = course_of(speed, curvature);
            (1..=steps)
                .find(|&k| {
                    let (px, py, heading) = course(speed * k as f64 * vo.dt);
                    checker.clearance(k, px, py, heading - if back { PI } else { 0.0 }) < floor
                })
                .map_or(f64::INFINITY, |k| k as f64 * vo.dt)
        };
        let candidates: Vec<Candidate> = speeds_tried
            .iter()
            .flat_map(|&speed| curvatures.iter().map(move |&curvature| (speed, curvature)))
            .map(|(speed, curvature)| Candidate { speed, curvature, meets: meets(speed, curvature) })
            .collect();
        let needed = |c: &Candidate| (c.speed / accel + vo.reaction_s).min(vo.horizon_s);
        let cost = |c: &Candidate| {
            let dv = (c.speed - v_pref) / v_max.max(0.1);
            let dk = (c.curvature - k_pref) / (2.0 * k_lim).max(1e-6);
            dv * dv + vo.weight_turn * dk * dk
        };
        let safe = candidates.iter().filter(|c| c.meets >= needed(c)).min_by(|a, b| cost(a).total_cmp(&cost(b)));
        let (chosen, mode) = match safe {
            Some(c) => (c, if c.meets.is_finite() || (c.speed - v_pref).abs() > 1e-3 || (c.curvature - k_pref).abs() > 1e-3 { "vo_avoiding" } else { "vo_tracking" }),
            None => (
                candidates
                    .iter()
                    .max_by(|a, b| (a.meets - needed(a)).total_cmp(&(b.meets - needed(b))).then(b.speed.total_cmp(&a.speed)))
                    .expect("at least one command is tried"),
                "vo_evading",
            ),
        };

        if self.debug_level != DebugLevel::Off {
            let course = course_of(chosen.speed, chosen.curvature);
            self.plan = (0..=steps).map(|k| {
                let (px, py, _) = course(chosen.speed * k as f64 * vo.dt);
                Point::new(px, py, 0.0)
            }).collect();
        }
        self.base.status.mode = mode.into();
        let direction = if back { -1.0 } else { 1.0 };
        finalize(direction * chosen.speed, chosen.speed * chosen.curvature, constraints, &cfg, prep.allow_reverse, "VO tracking")
    }

    fn params(&self) -> serde_json::Value {
        serde_json::to_value(&self.vo_config).unwrap_or_default()
    }

    fn set_params(&mut self, params: &serde_json::Value) -> Result<(), String> {
        self.vo_config = crate::controller::merged(&self.vo_config, params)?;
        Ok(())
    }

    fn set_debug(&mut self, level: DebugLevel) {
        self.debug_level = level;
    }

    fn predicted_trajectory(&self) -> Vec<Point> {
        self.plan.clone()
    }

    fn debug_frame(&self) -> DebugFrame {
        DebugFrame::default()
    }

    fn set_path(&mut self, path: Path) {
        self.cursor.set_path(&path.waypoints);
        self.base.path = path;
        self.base.path_index = 0;
        self.base.status = Default::default();
        self.plan.clear();
        self.side = 0.0;
    }

    fn reset(&mut self) {
        self.set_path(Path::default());
    }

    fn get_type(&self) -> &'static str {
        "velocity_obstacle_follower"
    }

    fn base(&self) -> &ControllerBase {
        &self.base
    }

    fn base_mut(&mut self) -> &mut ControllerBase {
        &mut self.base
    }
}
