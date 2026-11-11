//! Pure Pursuit (Coulter 1992). The rear axle chases a point at the
//! lookahead arc length ahead of its projection on the path; the arc
//! through both points has curvature `2 sin(alpha) / L_d`.

use crate::controller::{Controller, ControllerBase, effective_tolerances};
use crate::core::kinematics::{
    can_turn_in_place, finalize, finalize_holonomic, heading_speed_scale, holonomic_point_command,
    body_heading_on_path, braked_for_bends, followable, is_ackermann, is_holonomic, max_curvature, path_speed,
    reverse_allowed,
    world_to_body,
};
use crate::core::math::{heading_error, normalize_angle};
use crate::core::path::{
    PathCursor, curvature_at_arc_length, cusp_after, direction_at, sample_past, sharpest_ahead, speed_cap,
};
use crate::types::{Goal, Path, RobotConstraints, RobotState, VelocityCommand, WorldConstraints};
use datapod::Point;
use std::f64::consts::PI;

/// Lookahead shrink per unit of path curvature (metres of lookahead per
/// unit curvature of the sharpest bend within reach), keeping bends from being cut.
const CURVATURE_LOOKAHEAD_GAIN: f64 = 1.0;

#[derive(Clone, Debug, Default)]
pub struct PurePursuitFollower {
    pub base: ControllerBase,
    cursor: PathCursor,
    lookahead_point: Option<Point>,
    last_speed: f64,
}

impl PurePursuitFollower {
    pub fn new() -> Self {
        Self::default()
    }

    /// The point currently being chased, for visualisation.
    pub fn lookahead_point(&self) -> Option<Point> {
        self.lookahead_point
    }

    fn chase_goal(
        &mut self,
        state: &RobotState,
        goal: &Goal,
        constraints: &RobotConstraints,
        allow_reverse: bool,
    ) -> VelocityCommand {
        let cfg = self.base.config.clone();
        if is_holonomic(constraints.steering_type) {
            self.base.status.mode = "pure_pursuit_goal_holonomic".into();
            let yaw = state.pose.rotation.to_euler().yaw;
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
        let dist = self.base.status.distance_to_goal;
        let omega = cfg.kp_angular.max(0.1) * bearing;
        let v = direction
            * path_speed(
                constraints.max_linear_velocity,
                0.0,
                dist,
                cfg.goal_tolerance,
                cfg.kp_linear,
                constraints,
            )
            * heading_speed_scale(bearing, constraints);
        self.base.status.mode = "pure_pursuit_goal".into();
        finalize(v, omega, constraints, &cfg, allow_reverse, "Moving to goal")
    }
}

impl Controller for PurePursuitFollower {
    fn compute_control(
        &mut self,
        state: &RobotState,
        goal: &Goal,
        constraints: &RobotConstraints,
        dt: f64,
        _world: Option<&WorldConstraints>,
    ) -> VelocityCommand {
        if !(dt.is_finite() && dt > 0.0) {
            return VelocityCommand::invalid("dt must be positive and finite");
        }
        if self.cursor.cum.len() != self.base.path.waypoints.len() {
            self.cursor.set_path(&self.base.path.waypoints);
            self.base.path_index = 0;
        }
        let cfg = self.base.config.clone();
        let allow_reverse = reverse_allowed(&cfg, state);
        let yaw = state.pose.rotation.to_euler().yaw;

        if self.base.path.waypoints.is_empty() {
            if let Some(cmd) = self.base.arrival(state, goal, constraints, false) {
                return cmd;
            }
            return self.chase_goal(state, goal, constraints, allow_reverse);
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

        let length = self.cursor.cum.last().copied().unwrap_or(0.0);
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

        let base_lookahead = cfg.lookahead_distance.max(1e-3);
        let kappa_path = sharpest_ahead(&self.base.path.waypoints, &self.cursor.cum, proj.arc_length, base_lookahead, max_curvature(constraints));
        let lookahead = ((base_lookahead + cfg.lookahead_time.max(0.0) * state.velocity.linear.abs())
            / (1.0 + CURVATURE_LOOKAHEAD_GAIN * kappa_path * base_lookahead))
            .clamp(0.5 * base_lookahead, 3.0 * base_lookahead);
        let mut stop_distance = self.base.distance_left(&at_robot, length);
        let mut cap = length;
        if let Some(cusp) = cusp_after(&self.base.path.speeds, &self.cursor.cum, at_robot.arc_length) {
            cap = cusp;
            stop_distance = stop_distance.min((cusp - at_robot.arc_length).max(0.0));
        }
        // The rear axle aims where it stands when the pose is on the path at
        // the lookahead point, facing along it; past an open end, the path runs
        // straight on along its last heading.
        let back = direction_at(&self.base.path.speeds, &at_robot) == Some(-1);
        let open_end = cap >= length && !self.base.path.is_closed;
        let s_target = if open_end { at_robot.arc_length + lookahead } else { (at_robot.arc_length + lookahead).min(cap) };
        let (on_path, heading) = sample_past(&self.base.path.waypoints, &self.cursor.cum, s_target, cap);
        let bend = if s_target > cap { 0.0 } else { curvature_at_arc_length(&self.base.path.waypoints, &self.cursor.cum, s_target) };
        let body = body_heading_on_path(heading, followable(bend, constraints), axle_offset, back);
        let target = Point::new(on_path.x - axle_offset * body.cos(), on_path.y - axle_offset * body.sin(), on_path.z);
        self.lookahead_point = Some(target);

        let dx = target.x - rear.x;
        let dy = target.y - rear.y;
        let ld = dx.hypot(dy);

        if is_holonomic(constraints.steering_type) {
            self.base.status.cross_track_error = proj.lateral_error;
            self.base.status.heading_error = normalize_angle(proj.heading - yaw);
            self.base.status.goal_reached = false;
            self.base.status.mode = "pure_pursuit_holonomic".into();

            let nominal = speed_cap(&self.base.path.speeds, &proj)
                .map_or(constraints.max_linear_velocity, |s| {
                    s.min(constraints.max_linear_velocity)
                });
            let speed = path_speed(nominal, 0.0, stop_distance, pos_tol, cfg.kp_linear, constraints);
            let (ux, uy) = if ld > 1e-9 { (dx / ld, dy / ld) } else { (0.0, 0.0) };
            let (vx, vy) = world_to_body(speed * ux, speed * uy, yaw);
            let omega = cfg.kp_angular.max(0.1) * normalize_angle(proj.heading - yaw);
            return finalize_holonomic(vx, vy, omega, constraints, &cfg, "Following path");
        }

        let alpha = normalize_angle(dy.atan2(dx) - yaw);
        let kappa = if ld > 1e-6 {
            2.0 * alpha.sin() / ld
        } else {
            0.0
        };

        let mut direction = 1.0;
        let mut alpha_eff = alpha;
        if allow_reverse && alpha.abs() > PI / 2.0 {
            direction = -1.0;
            alpha_eff = normalize_angle(alpha + PI);
        }

        self.base.status.cross_track_error = proj.lateral_error;
        self.base.status.heading_error = normalize_angle(proj.heading - yaw);
        self.base.status.goal_reached = false;

        if state.turn_first
            && can_turn_in_place(constraints.steering_type)
            && alpha_eff.abs() > ang_tol
        {
            self.base.status.mode = "turning".into();
            let omega = cfg.kp_angular.max(0.1) * alpha_eff;
            return finalize(0.0, omega, constraints, &cfg, allow_reverse, "Turning to align");
        }

        let nominal = speed_cap(&self.base.path.speeds, &proj)
            .map_or(constraints.max_linear_velocity, |s| {
                s.min(constraints.max_linear_velocity)
            });
        let scale = heading_speed_scale(alpha_eff, constraints);
        let v = path_speed(nominal, kappa, stop_distance, pos_tol, cfg.kp_linear, constraints) * scale;
        let v_now = if state.velocity_valid { state.velocity.linear } else { self.last_speed };
        let (waypoints, cum, s) = (&self.base.path.waypoints, &self.cursor.cum, at_robot.arc_length);
        let v = braked_for_bends(v, v_now, nominal, stop_distance, |d| curvature_at_arc_length(waypoints, cum, s + d), constraints);
        self.last_speed = v;
        let v = direction * v;

        let mut omega = v * kappa;
        if can_turn_in_place(constraints.steering_type) {
            omega += (1.0 - scale) * cfg.kp_angular.max(0.1) * alpha_eff;
        }

        self.base.status.mode = if direction > 0.0 {
            "pure_pursuit".into()
        } else {
            "pure_pursuit_reverse".into()
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
        self.base.status = Default::default();
    }

    fn get_type(&self) -> &'static str {
        "pure_pursuit_follower"
    }

    fn base(&self) -> &ControllerBase {
        &self.base
    }

    fn base_mut(&mut self) -> &mut ControllerBase {
        &mut self.base
    }
}
