//! Regulated Pure Pursuit (Macenski et al. 2023, the Nav2 variant).
//! Pure Pursuit geometry with a velocity-scaled lookahead (moved onto a
//! detour round obstacles blocking the path ahead), speed regulated
//! by path curvature and obstacle proximity, a collision check along the
//! commanded arc, and optional rotate-to-heading for platforms that can
//! turn in place.

use crate::controller::{Controller, ControllerBase, effective_tolerances};
use crate::core::kinematics::{
    can_turn_in_place, finalize, finalize_holonomic, heading_speed_scale, holonomic_point_command,
    axle_point_offset, braked_for_bends, followable, is_ackermann, is_holonomic, max_curvature, path_speed,
    reverse_allowed,
    wheelbase, world_to_body,
};
use crate::core::math::{heading_error, normalize_angle};
use crate::core::obstacles::{CollisionChecker, robot_radius};
use crate::pred::mppi::{Swerve, build_reference, pick_detour};
use crate::core::path::{PathCursor, curvature_at_arc_length, cusp_after, direction_at, sample_past, speed_cap};
use crate::types::{Goal, Path, RobotConstraints, RobotState, VelocityCommand, WorldConstraints};
use datapod::Point;
use std::f64::consts::PI;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct RegulatedPursuitConfig {
    pub min_lookahead: f64,
    pub max_lookahead: f64,
    /// Lookahead = `lookahead_time * |v|`, clamped to the bounds above.
    pub lookahead_time: f64,
    /// Turning radius below which speed is scaled down proportionally.
    pub regulated_min_radius: f64,
    /// Obstacle clearance below which speed is scaled down proportionally.
    pub proximity_distance: f64,
    /// Lowest speed (m/s) the two regulations may produce.
    pub min_speed: f64,
    /// Remaining path (to its end or the next cusp) over which the speed
    /// scales down linearly; nearer than braking at
    /// `max_linear_acceleration` needs, it is capped to stop there.
    pub approach_distance: f64,
    /// Lowest speed (m/s) of that approach.
    pub min_approach_speed: f64,
    /// Invalidate the command when the commanded arc, simulated for
    /// `collision_time` but no further than the lookahead point, hits an
    /// obstacle.
    pub use_collision_check: bool,
    pub collision_time: f64,
    /// Rotate in place first when the heading error exceeds the threshold.
    pub use_rotate_to_heading: bool,
    pub rotate_heading_threshold: f64,
    /// Turn rate (rad/s) of that rotation, eased into the target heading.
    pub rotate_speed: f64,
}

impl Default for RegulatedPursuitConfig {
    fn default() -> Self {
        Self {
            min_lookahead: 0.6,
            max_lookahead: 2.0,
            lookahead_time: 1.5,
            regulated_min_radius: 0.9,
            proximity_distance: 0.8,
            min_speed: 0.25,
            approach_distance: 0.6,
            min_approach_speed: 0.05,
            use_collision_check: true,
            collision_time: 1.0,
            use_rotate_to_heading: true,
            rotate_heading_threshold: 0.785,
            rotate_speed: 1.8,
        }
    }
}

/// Seconds between the points of the detour the lookahead may move onto.
const DETOUR_DT: f64 = 0.2;
/// Share of the top speed that detour takes the robot at.
const DETOUR_SPEED_SHARE: f64 = 0.8;
/// Room, metres, the detour keeps beyond the robot's radius on top of the
/// shared detour margin.
const DETOUR_ROOM: f64 = 0.4;
/// Shares of the regulated speed tried, fastest first, until the arc is clear.
const YIELD_SHARES: [f64; 4] = [1.0, 0.75, 0.5, 0.25];
/// Seconds at least that the yield check watches the arc for traffic.
const YIELD_LOOKOUT_S: f64 = 2.0;
/// Room, metres, the collision checks keep beyond the footprint, the
/// obstacles grown by one predicted spread (the costmap inflation of Nav2).
const CLEARANCE_MARGIN: f64 = 0.1;

/// Lookahead floor and curvature-regulation radius for the machine at
/// `speed`: on an Ackermann machine the lookahead is at least a wheelbase
/// plus the distance covered while the steering swings half way to lock,
/// and speed regulates below twice its tightest turning radius; otherwise
/// the configured values.
fn machine_scale(rpp: &RegulatedPursuitConfig, constraints: &RobotConstraints, speed: f64) -> (f64, f64) {
    let floor = rpp.min_lookahead.max(1e-3);
    if !is_ackermann(constraints.steering_type) {
        return (floor, rpp.regulated_min_radius);
    }
    let swing = if constraints.max_steering_rate > 0.0 && constraints.max_steering_angle > 0.0 {
        0.5 * constraints.max_steering_angle / constraints.max_steering_rate
    } else {
        0.0
    };
    let kappa = max_curvature(constraints);
    let radius = if kappa.is_finite() && kappa > 0.0 { 2.0 / kappa } else { 0.0 };
    (floor.max(wheelbase(constraints) + speed * swing), rpp.regulated_min_radius.max(radius))
}

#[derive(Clone, Debug, Default)]
pub struct RegulatedPursuitFollower {
    pub base: ControllerBase,
    pub rpp_config: RegulatedPursuitConfig,
    cursor: PathCursor,
    lookahead_point: Option<Point>,
    /// Last commanded speed, standing in for a measurement.
    last_speed: f64,
    /// Side the lookahead last went round obstacles on: +1 left, −1 right, 0 none.
    side: f64,
}

impl RegulatedPursuitFollower {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_rpp_config(cfg: RegulatedPursuitConfig) -> Self {
        Self {
            rpp_config: cfg,
            ..Default::default()
        }
    }

    pub fn set_rpp_config(&mut self, cfg: RegulatedPursuitConfig) {
        self.rpp_config = cfg;
    }

    pub fn lookahead_point(&self) -> Option<Point> {
        self.lookahead_point
    }

    /// Smallest clearance along the arc `(v, omega)` from `origin` for
    /// `horizon` seconds, the robot held where it is once it has gone `reach`
    /// metres (or a second of the arc), against the obstacles as they move.
    fn arc_clearance(
        checker: &CollisionChecker,
        origin: Point,
        yaw: f64,
        v: f64,
        omega: f64,
        horizon: f64,
        reach: f64,
    ) -> f64 {
        if !checker.has_obstacles() {
            return f64::INFINITY;
        }
        let step = 0.05;
        let (mut x, mut y, mut th, mut t, mut travelled) = (origin.x, origin.y, yaw, 0.0, 0.0);
        let mut worst = f64::INFINITY;
        let reach = reach.max(v.abs() * horizon.min(1.0));
        loop {
            worst = worst.min(checker.clearance_at(t, x, y, th));
            if t >= horizon {
                return worst;
            }
            if travelled < reach {
                x += v * step * th.cos();
                y += v * step * th.sin();
                th = normalize_angle(th + omega * step);
                travelled += v.abs() * step;
            }
            t += step;
        }
    }

    /// True when the footprint, driven from `(origin, yaw)` at `(v, omega)`
    /// for `horizon` seconds and held where it is once it has gone `reach`
    /// metres, meets an obstacle as predicted at each moment.
    fn arc_collides(
        checker: &CollisionChecker,
        origin: Point,
        yaw: f64,
        v: f64,
        omega: f64,
        horizon: f64,
        reach: f64,
    ) -> bool {
        if !checker.has_obstacles() {
            return false;
        }
        let step = 0.05;
        let (mut x, mut y, mut th, mut t, mut travelled) = (origin.x, origin.y, yaw, 0.0, 0.0);
        loop {
            if checker.clearance_at(t, x, y, th) < 0.0 {
                return true;
            }
            if t >= horizon {
                return false;
            }
            if travelled < reach {
                x += v * step * th.cos();
                y += v * step * th.sin();
                th = normalize_angle(th + omega * step);
                travelled += v.abs() * step;
            }
            t += step;
        }
    }
}

impl Controller for RegulatedPursuitFollower {
    fn params(&self) -> serde_json::Value {
        serde_json::to_value(&self.rpp_config).unwrap_or_default()
    }

    fn set_params(&mut self, params: &serde_json::Value) -> Result<(), String> {
        let cfg = crate::controller::merged(&self.rpp_config, params)?;
        self.set_rpp_config(cfg);
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
        if !(dt.is_finite() && dt > 0.0) {
            return VelocityCommand::invalid("dt must be positive and finite");
        }
        let cfg = self.base.config.clone();
        let rpp = self.rpp_config.clone();
        let allow_reverse = reverse_allowed(&cfg, state);
        let yaw = state.pose.rotation.to_euler().yaw;

        if self.base.path.waypoints.is_empty() {
            if let Some(cmd) = self.base.arrival(state, goal, constraints, false) {
                return cmd;
            }
            if is_holonomic(constraints.steering_type) {
                self.base.status.mode = "rpp_goal_holonomic".into();
                return holonomic_point_command(
                    state.pose.point,
                    yaw,
                    goal.target_pose.point,
                    goal.target_pose.rotation.to_euler().yaw,
                    self.base.status.distance_to_goal,
                    &cfg,
                    constraints,
                    "Moving to goal",
                );
            }
            let mut bearing = heading_error(&state.pose, goal.target_pose.point);
            let mut direction = 1.0;
            if allow_reverse && bearing.abs() > PI / 2.0 {
                direction = -1.0;
                bearing = normalize_angle(bearing + PI);
            }
            let v = direction
                * path_speed(
                    constraints.max_linear_velocity,
                    0.0,
                    self.base.status.distance_to_goal,
                    cfg.goal_tolerance,
                    cfg.kp_linear,
                    constraints,
                )
                * heading_speed_scale(bearing, constraints);
            self.base.status.mode = "rpp_goal".into();
            return finalize(v, cfg.kp_angular.max(0.1) * bearing, constraints, &cfg, allow_reverse, "Moving to goal");
        }

        let axle_offset = if is_ackermann(constraints.steering_type) {
            constraints.rear_wheelbase.max(0.0)
        } else {
            0.0
        };
        let rear = Point::new(
            state.pose.point.x - axle_offset * yaw.cos(),
            state.pose.point.y - axle_offset * yaw.sin(),
            0.0,
        );
        // Progress, cusps and arrival go by the robot's pose; the rear axle
        // chases the lookahead point.
        let (pos_tol, ang_tol) = effective_tolerances(goal, &cfg);
        let Some(at_robot) = self.cursor.project(&self.base.path.waypoints, state.pose.point, self.base.path_index) else {
            return VelocityCommand::invalid("no path");
        };
        let at_robot =
            self.cursor.past_cusp(&self.base.path.waypoints, &self.base.path.speeds, state.pose.point, at_robot, crate::controller::cusp_reach(pos_tol, state, constraints));
        self.base.path_index = at_robot.segment;

        let length = self.cursor.total_length();
        if let Some(cmd) = self.base.arrival_on_path(state, goal, constraints, &at_robot, length) {
            self.lookahead_point = None;
            return cmd;
        }
        let proj = if axle_offset > 0.0 {
            self.cursor
                .project_on_leg(&self.base.path.waypoints, &self.base.path.speeds, rear, at_robot.segment, at_robot.arc_length)
                .unwrap_or(at_robot)
        } else {
            at_robot
        };

        let measured = if state.velocity_valid { state.velocity.linear.abs() } else { self.last_speed };
        let (floor, regulated_radius) = machine_scale(&rpp, constraints, measured);
        let lookahead = (rpp.lookahead_time * measured).clamp(floor, rpp.max_lookahead.max(floor));
        // The path is followed up to its next cusp at most, as if it ended there.
        let stop_at = cusp_after(&self.base.path.speeds, &self.cursor.cum, at_robot.arc_length).unwrap_or(length);
        // Reversing, the rear axle leads the pose and aims past the leg's end.
        let trail = if direction_at(&self.base.path.speeds, &at_robot) == Some(-1) { axle_offset } else { 0.0 };
        let s_target = (proj.arc_length + lookahead).min(stop_at + trail);
        let (on_path, target_heading) = sample_past(&self.base.path.waypoints, &self.cursor.cum, s_target, stop_at);
        // The rear axle runs inside the bend its pose follows.
        let bend = curvature_at_arc_length(&self.base.path.waypoints, &self.cursor.cum, s_target.min(stop_at));
        let inward = axle_point_offset(followable(bend, constraints), axle_offset, 0.0);
        let mut target =
            Point::new(on_path.x - inward * target_heading.sin(), on_path.y + inward * target_heading.cos(), on_path.z);
        let obstacles = world.map_or(&[][..], |w| &w.obstacles[..]);
        if obstacles.is_empty() {
            self.side = 0.0;
        } else {
            // The lookahead point is taken on the cheaper detour round what
            // blocks the path ahead, timed against the obstacles' predictions.
            let speed = (DETOUR_SPEED_SHARE * constraints.max_linear_velocity).max(0.1);
            let ahead = (lookahead.min(stop_at - proj.arc_length).max(0.0) / (speed * DETOUR_DT)).round() as usize;
            let reference = build_reference(&self.base.path, &self.cursor.cum, proj.arc_length, ahead + 6, DETOUR_DT, speed, 0.0);
            if let Some(g) = pick_detour(&reference, obstacles, robot_radius(constraints) + DETOUR_ROOM, DETOUR_DT, &mut self.side, Swerve::of(constraints, speed)) {
                let k = ahead.min(g.x.len() - 1);
                target = Point::new(g.x[k], g.y[k], 0.0);
            }
        }
        self.lookahead_point = Some(target);
        let remaining = (stop_at - at_robot.arc_length).max(0.0);

        let dx = target.x - rear.x;
        let dy = target.y - rear.y;
        let ld = dx.hypot(dy);
        let checker = CollisionChecker::new(world, constraints, CLEARANCE_MARGIN).with_spread(1.0);
        let nominal = speed_cap(&self.base.path.speeds, &proj)
            .map_or(constraints.max_linear_velocity, |s| s.min(constraints.max_linear_velocity));
        let proximity = |speed: f64| -> f64 {
            if rpp.proximity_distance <= 0.0 {
                return speed;
            }
            let c = checker.clearance(0, state.pose.point.x, state.pose.point.y, yaw);
            if c < rpp.proximity_distance { speed * c.max(0.0) / rpp.proximity_distance } else { speed }
        };
        let braking = constraints.max_linear_acceleration;
        let approach = |speed: f64| -> f64 {
            let floor = rpp.min_approach_speed.min(speed);
            let speed = if braking > 0.0 { speed.min((2.0 * braking * remaining).sqrt()) } else { speed };
            if rpp.approach_distance > 0.0 && remaining < rpp.approach_distance {
                (speed * remaining / rpp.approach_distance).max(floor)
            } else {
                speed.max(floor)
            }
        };
        self.base.status.cross_track_error = proj.lateral_error;
        self.base.status.heading_error = normalize_angle(proj.heading - yaw);
        self.base.status.goal_reached = false;

        if is_holonomic(constraints.steering_type) {
            let speed = approach(proximity(nominal).max(rpp.min_speed.min(nominal)).min(nominal));
            let travel_heading = if ld > 1e-9 { dy.atan2(dx) } else { yaw };
            if rpp.use_collision_check
                && Self::arc_collides(&checker, rear, travel_heading, speed, 0.0, rpp.collision_time, ld)
            {
                self.base.status.mode = "rpp_blocked_holonomic".into();
                return VelocityCommand::invalid("collision ahead on the commanded path");
            }
            let (ux, uy) = if ld > 1e-9 { (dx / ld, dy / ld) } else { (0.0, 0.0) };
            let (vx, vy) = world_to_body(speed * ux, speed * uy, yaw);
            let omega = cfg.kp_angular.max(0.1) * normalize_angle(proj.heading - yaw);
            self.base.status.mode = "regulated_pursuit_holonomic".into();
            self.last_speed = speed;
            return finalize_holonomic(vx, vy, omega, constraints, &cfg, "Following path");
        }

        // Body-frame lookahead point: its side sets the curvature, its sign
        // ahead or behind the driving direction.
        let (sn, cs) = yaw.sin_cos();
        let (bx, by) = (cs * dx + sn * dy, -sn * dx + cs * dy);
        let kappa = if ld > 1e-6 { 2.0 * by / (ld * ld) } else { 0.0 };
        let direction = match crate::core::path::direction_at(&self.base.path.speeds, &proj) {
            Some(d) if allow_reverse => d as f64,
            _ if allow_reverse && bx < 0.0 => -1.0,
            _ => 1.0,
        };
        let alpha = normalize_angle(by.atan2(bx) + if direction < 0.0 { PI } else { 0.0 });

        let turn_in_place = can_turn_in_place(constraints.steering_type)
            && ((rpp.use_rotate_to_heading && alpha.abs() > rpp.rotate_heading_threshold)
                || (state.turn_first && alpha.abs() > ang_tol));
        if turn_in_place {
            self.base.status.mode = "rpp_rotate".into();
            let ease = (2.0 * constraints.max_angular_acceleration.abs().max(1e-3) * alpha.abs()).sqrt();
            let omega = alpha.signum() * rpp.rotate_speed.abs().min(ease);
            if rpp.use_collision_check
                && Self::arc_collides(&checker, state.pose.point, yaw, 0.0, omega, (alpha / omega).abs().min(rpp.collision_time), f64::INFINITY)
            {
                self.base.status.mode = "rpp_blocked".into();
                return VelocityCommand::invalid("collision while rotating to heading");
            }
            self.last_speed = 0.0;
            return finalize(0.0, omega, constraints, &cfg, allow_reverse, "Rotating to heading");
        }

        let mut v = nominal;
        if kappa.abs() > 1e-9 && regulated_radius > 0.0 && 1.0 / kappa.abs() < regulated_radius {
            v *= (1.0 / kappa.abs()) / regulated_radius;
        }
        v = proximity(v).min(v);
        v = v.max(rpp.min_speed.min(nominal));
        let (waypoints, cum, s) = (&self.base.path.waypoints, &self.cursor.cum, at_robot.arc_length);
        v = braked_for_bends(v, measured, nominal, remaining, |d| followable(curvature_at_arc_length(waypoints, cum, s + d), constraints), constraints);
        let mut v = direction * approach(v);
        if rpp.use_collision_check {
            // The fastest share of the speed whose arc, over the check time
            // and the time to stop from it, stays clear of the obstacles as
            // they move: slowing to let crossing traffic pass.
            let a = constraints.max_linear_acceleration.abs().max(1e-3);
            let clear = |s: f64| {
                let u = v * s;
                !Self::arc_collides(&checker, rear, yaw, u, u * kappa, rpp.collision_time.max(YIELD_LOOKOUT_S) + u.abs() / a, ld)
            };
            let share = YIELD_SHARES.into_iter().find(|s| clear(*s)).unwrap_or_else(|| {
                // Nothing is clear: the share, standing still included, that
                // keeps the most room while the obstacles move.
                let room = |s: f64| {
                    let u = v * s;
                    Self::arc_clearance(&checker, rear, yaw, u, u * kappa, rpp.collision_time.max(YIELD_LOOKOUT_S), ld)
                };
                YIELD_SHARES.into_iter().chain([0.0]).max_by(|a, b| room(*a).total_cmp(&room(*b))).unwrap_or(0.0)
            });
            if share == 0.0 && Self::arc_clearance(&checker, rear, yaw, 0.0, 0.0, rpp.collision_time, ld) < 0.0 {
                self.base.status.mode = "rpp_blocked".into();
                return VelocityCommand::invalid("collision ahead on the commanded arc");
            }
            v *= share;
        }
        let omega = v * kappa;
        self.base.status.mode = if direction > 0.0 {
            "regulated_pursuit".into()
        } else {
            "regulated_pursuit_reverse".into()
        };
        self.last_speed = v.abs();
        finalize(v, omega, constraints, &cfg, allow_reverse, "Following path")
    }

    fn set_path(&mut self, path: Path) {
        self.cursor.set_path(&path.waypoints);
        self.base.path = path;
        self.base.path_index = 0;
        self.base.status = Default::default();
        self.lookahead_point = None;
    }

    fn reset(&mut self) {
        self.set_path(Path::default());
    }

    fn get_type(&self) -> &'static str {
        "regulated_pursuit_follower"
    }

    fn base(&self) -> &ControllerBase {
        &self.base
    }

    fn base_mut(&mut self) -> &mut ControllerBase {
        &mut self.base
    }
}
