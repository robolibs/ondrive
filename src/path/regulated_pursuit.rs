//! Regulated Pure Pursuit (Macenski et al. 2023, the Nav2 variant).
//! Pure Pursuit geometry with a velocity-scaled lookahead, speed regulated
//! by path curvature and obstacle proximity, a collision check along the
//! commanded arc, and optional rotate-to-heading for platforms that can
//! turn in place.

use crate::controller::{Controller, ControllerBase, effective_tolerances};
use crate::core::kinematics::{
    can_turn_in_place, finalize, heading_speed_scale, is_ackermann, max_curvature, path_speed,
    reverse_allowed,
};
use crate::core::math::{heading_error, normalize_angle};
use crate::core::obstacles::{robot_radius, world_clearance};
use crate::core::path::{PathCursor, sample, speed_cap};
use crate::types::{Goal, Path, RobotConstraints, RobotState, VelocityCommand, WorldConstraints};
use datapod::Point;
use std::f64::consts::PI;

#[derive(Clone, Debug)]
pub struct RegulatedPursuitConfig {
    pub min_lookahead: f64,
    pub max_lookahead: f64,
    /// Lookahead = `lookahead_time * |v|`, clamped to the bounds above.
    pub lookahead_time: f64,
    /// Turning radius below which speed is scaled down proportionally.
    pub regulated_min_radius: f64,
    /// Obstacle clearance below which speed is scaled down proportionally.
    pub proximity_distance: f64,
    /// Lowest speed fraction the two regulations may produce.
    pub min_speed_scale: f64,
    /// Invalidate the command when the commanded arc hits an obstacle
    /// within the stopping distance.
    pub use_collision_check: bool,
    /// Rotate in place first when the heading error exceeds the threshold.
    pub use_rotate_to_heading: bool,
    pub rotate_heading_threshold: f64,
}

impl Default for RegulatedPursuitConfig {
    fn default() -> Self {
        Self {
            min_lookahead: 0.6,
            max_lookahead: 2.0,
            lookahead_time: 1.2,
            regulated_min_radius: 0.9,
            proximity_distance: 0.8,
            min_speed_scale: 0.25,
            use_collision_check: true,
            use_rotate_to_heading: true,
            rotate_heading_threshold: 0.8,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct RegulatedPursuitFollower {
    pub base: ControllerBase,
    pub rpp_config: RegulatedPursuitConfig,
    cursor: PathCursor,
    lookahead_point: Option<Point>,
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

    /// True when the arc of curvature `kappa` from `rear` collides within
    /// `length` metres.
    fn arc_collides(
        world: Option<&WorldConstraints>,
        rear: Point,
        yaw: f64,
        kappa: f64,
        direction: f64,
        length: f64,
        radius: f64,
    ) -> bool {
        let Some(w) = world else {
            return false;
        };
        if w.obstacles.is_empty() {
            return false;
        }
        let step = 0.05;
        let n = (length / step).ceil().max(1.0) as usize;
        let (mut x, mut y, mut th) = (rear.x, rear.y, yaw);
        for i in 1..=n {
            x += direction * step * th.cos();
            y += direction * step * th.sin();
            th = normalize_angle(th + direction * step * kappa);
            if world_clearance(Some(w), i / 2, x, y, radius) < 0.0 {
                return true;
            }
        }
        false
    }
}

impl Controller for RegulatedPursuitFollower {
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
        let Some(proj) = self.cursor.project(&self.base.path.waypoints, rear, self.base.path_index) else {
            return VelocityCommand::invalid("no path");
        };
        self.base.path_index = proj.segment;

        let (pos_tol, ang_tol) = effective_tolerances(goal, &cfg);
        let passed_end = proj.beyond_end && proj.distance < 2.0 * pos_tol;
        if let Some(cmd) = self.base.arrival(state, goal, constraints, passed_end) {
            self.lookahead_point = None;
            return cmd;
        }

        let lookahead = (rpp.lookahead_time * state.velocity.linear.abs())
            .clamp(rpp.min_lookahead.max(1e-3), rpp.max_lookahead.max(rpp.min_lookahead));
        let (target, _) = sample(&self.base.path.waypoints, &self.cursor.cum, proj.arc_length + lookahead);
        self.lookahead_point = Some(target);

        let dx = target.x - rear.x;
        let dy = target.y - rear.y;
        let ld = dx.hypot(dy);
        let alpha = normalize_angle(dy.atan2(dx) - yaw);
        let kappa = if ld > 1e-6 { 2.0 * alpha.sin() / ld } else { 0.0 };

        let mut direction = 1.0;
        let mut alpha_eff = alpha;
        if allow_reverse && alpha.abs() > PI / 2.0 {
            direction = -1.0;
            alpha_eff = normalize_angle(alpha + PI);
        }

        self.base.status.cross_track_error = proj.lateral_error;
        self.base.status.heading_error = normalize_angle(proj.heading - yaw);
        self.base.status.goal_reached = false;

        let turn_in_place = can_turn_in_place(constraints.steering_type)
            && ((rpp.use_rotate_to_heading && alpha_eff.abs() > rpp.rotate_heading_threshold)
                || (state.turn_first && alpha_eff.abs() > ang_tol));
        if turn_in_place {
            self.base.status.mode = "rpp_rotate".into();
            let omega = cfg.kp_angular.max(0.1) * alpha_eff;
            return finalize(0.0, omega, constraints, &cfg, allow_reverse, "Rotating to heading");
        }

        let nominal = speed_cap(&self.base.path.speeds, &proj)
            .map_or(constraints.max_linear_velocity, |s| s.min(constraints.max_linear_velocity));
        let mut v = path_speed(nominal, 0.0, self.base.status.distance_to_goal, pos_tol, cfg.kp_linear, constraints)
            * heading_speed_scale(alpha_eff, constraints);

        let radius = robot_radius(constraints);
        let mut scale: f64 = 1.0;
        if kappa.abs() > 1e-9 && rpp.regulated_min_radius > 0.0 {
            scale = scale.min((1.0 / kappa.abs()) / rpp.regulated_min_radius);
        }
        if rpp.proximity_distance > 0.0 {
            let c = world_clearance(world, 0, state.pose.point.x, state.pose.point.y, radius);
            if c < rpp.proximity_distance {
                scale = scale.min(c.max(0.0) / rpp.proximity_distance);
            }
        }
        v *= scale.max(rpp.min_speed_scale.clamp(0.0, 1.0));

        let kmax = max_curvature(constraints);
        let kappa_cmd = if kmax.is_finite() { kappa.clamp(-kmax, kmax) } else { kappa };
        if rpp.use_collision_check {
            let stopping = v * v / (2.0 * constraints.max_linear_acceleration.abs().max(1e-6));
            let check_len = (stopping + lookahead).max(0.3);
            if Self::arc_collides(world, rear, yaw, kappa_cmd, direction, check_len, radius) {
                self.base.status.mode = "rpp_blocked".into();
                return VelocityCommand::invalid("collision ahead on the commanded arc");
            }
        }

        let v = direction * v;
        let mut omega = v * kappa;
        if can_turn_in_place(constraints.steering_type) {
            let s = heading_speed_scale(alpha_eff, constraints);
            omega += (1.0 - s) * cfg.kp_angular.max(0.1) * alpha_eff;
        }
        self.base.status.mode = if direction > 0.0 {
            "regulated_pursuit".into()
        } else {
            "regulated_pursuit_reverse".into()
        };
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
