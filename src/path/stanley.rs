//! Stanley controller (Hoffmann et al. 2007):
//! `delta = k_h * theta_e + atan(k * e / (k_soft + v))` evaluated at the
//! front axle, with the mirrored form at the rear axle when reversing.

use crate::controller::{Controller, ControllerBase, effective_tolerances};
use crate::core::kinematics::{
    path_to_body,
    can_turn_in_place, finalize, finalize_holonomic, heading_speed_scale, holonomic_point_command,
    is_ackermann, is_holonomic, path_speed, reverse_allowed, steering_limit, steering_to_curvature,
    wheelbase,
};
use crate::core::math::{heading_error, normalize_angle};
use crate::core::path::{
    PathCursor, PathProjection, curvature_at_projection, project, speed_cap,
};
use crate::types::{Goal, Path, RobotConstraints, RobotState, VelocityCommand, WorldConstraints};
use datapod::Point;
use std::f64::consts::PI;


#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct StanleyConfig {
    /// Speed (m/s) added under the cross-track term so it stays bounded
    /// at low speed.
    pub k_soft: f64,
    /// Gain on the gap between the path's yaw rate and the measured one.
    pub k_yaw_rate: f64,
}

impl Default for StanleyConfig {
    fn default() -> Self {
        Self { k_soft: 0.5, k_yaw_rate: 0.0 }
    }
}

#[derive(Clone, Debug, Default)]
pub struct StanleyFollower {
    pub base: ControllerBase,
    pub stanley_config: StanleyConfig,
    cursor: PathCursor,
}

impl StanleyFollower {
    pub fn new() -> Self {
        Self::default()
    }

    /// Projection of `p` onto the leg from segment `from` up to waypoint
    /// `leg_end`: an axle ahead of the robot never lands past the cusp.
    fn project_on_leg(&self, p: Point, from: usize, leg_end: usize) -> Option<PathProjection> {
        project(&self.base.path.waypoints, &self.cursor.cum, p, from, 0, leg_end.saturating_sub(from).max(1))
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
            self.base.status.mode = "stanley_goal_holonomic".into();
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
        let omega = cfg.kp_angular.max(0.1) * bearing;
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
        self.base.status.mode = "stanley_goal".into();
        finalize(v, omega, constraints, &cfg, allow_reverse, "Moving to goal")
    }
}

impl Controller for StanleyFollower {
    fn params(&self) -> serde_json::Value {
        serde_json::to_value(&self.stanley_config).unwrap_or_default()
    }

    fn set_params(&mut self, params: &serde_json::Value) -> Result<(), String> {
        self.stanley_config = crate::controller::merged(&self.stanley_config, params)?;
        Ok(())
    }

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

        let ackermann = is_ackermann(constraints.steering_type);
        let front_offset = if ackermann {
            (wheelbase(constraints) - constraints.rear_wheelbase.max(0.0)).max(0.0)
        } else {
            0.0
        };
        let rear_offset = if ackermann {
            constraints.rear_wheelbase.max(0.0)
        } else {
            0.0
        };
        let (s, c) = yaw.sin_cos();
        let (pos_tol, ang_tol) = effective_tolerances(goal, &cfg);
        let Some(at_robot) = self.cursor.project(&self.base.path.waypoints, state.pose.point, self.base.path_index) else {
            return VelocityCommand::invalid("no path");
        };
        let at_robot =
            self.cursor.past_cusp(&self.base.path.waypoints, &self.base.path.speeds, state.pose.point, at_robot, pos_tol.max(0.1));
        self.base.path_index = at_robot.segment;
        let length = self.cursor.total_length();
        if let Some(cmd) = self.base.arrival_on_path(state, goal, constraints, &at_robot, length) {
            return cmd;
        }
        let cusp = self.cursor.cusp_ahead(&self.base.path.speeds, at_robot.arc_length);
        let leg_end = cusp.unwrap_or(self.base.path.waypoints.len() - 1);
        let stop_distance = cusp.map_or(f64::INFINITY, |k| self.cursor.cum[k] - at_robot.arc_length);
        let reverse = match crate::core::path::direction_at(&self.base.path.speeds, &at_robot) {
            Some(d) => allow_reverse && d < 0,
            None => allow_reverse && normalize_angle(at_robot.heading - yaw).abs() > PI / 2.0,
        };
        let axle = if reverse { -rear_offset } else { front_offset };
        let reference = Point::new(state.pose.point.x + axle * c, state.pose.point.y + axle * s, 0.0);
        let proj = self.project_on_leg(reference, at_robot.segment, leg_end).unwrap_or(at_robot);

        if is_holonomic(constraints.steering_type) {
            let theta_e = normalize_angle(proj.heading - yaw);
            self.base.status.cross_track_error = proj.lateral_error;
            self.base.status.heading_error = theta_e;
            self.base.status.goal_reached = false;
            self.base.status.mode = "stanley_holonomic".into();

            let kappa_path = curvature_at_projection(&self.base.path.waypoints, &self.cursor.cum, &proj);
            let nominal = speed_cap(&self.base.path.speeds, &proj)
                .map_or(constraints.max_linear_velocity, |v| {
                    v.min(constraints.max_linear_velocity)
                });
            let forward = path_speed(
                nominal,
                kappa_path,
                self.base.distance_left(&at_robot, length).min(stop_distance),
                pos_tol,
                cfg.kp_linear,
                constraints,
            );
            let (vx, vy) = path_to_body(forward, -cfg.k_cross_track * proj.lateral_error, proj.heading, yaw);
            let omega = cfg.kp_angular.max(0.1) * theta_e;
            return finalize_holonomic(vx, vy, omega, constraints, &cfg, "Following path");
        }

        let (direction, theta_e) =
            if reverse { (-1.0, normalize_angle(proj.heading - (yaw + PI))) } else { (1.0, normalize_angle(proj.heading - yaw)) };

        self.base.status.cross_track_error = proj.lateral_error;
        self.base.status.heading_error = theta_e;
        self.base.status.goal_reached = false;

        if state.turn_first && can_turn_in_place(constraints.steering_type) && theta_e.abs() > ang_tol
        {
            self.base.status.mode = "turning".into();
            let omega = cfg.kp_angular.max(0.1) * theta_e;
            return finalize(0.0, omega, constraints, &cfg, allow_reverse, "Turning to align");
        }

        let kappa_path = curvature_at_projection(&self.base.path.waypoints, &self.cursor.cum, &proj);
        let nominal = speed_cap(&self.base.path.speeds, &proj)
            .map_or(constraints.max_linear_velocity, |v| {
                v.min(constraints.max_linear_velocity)
            });
        let v = direction
            * path_speed(
                nominal,
                kappa_path,
                self.base.distance_left(&at_robot, length).min(stop_distance),
                pos_tol,
                cfg.kp_linear,
                constraints,
            )
            * heading_speed_scale(theta_e, constraints);

        let sc = &self.stanley_config;
        let speed = if state.velocity_valid { state.velocity.linear.abs() } else { v.abs() };
        let cross_term = (-cfg.k_cross_track * proj.lateral_error).atan2(sc.k_soft.max(1e-3) + speed);
        let mut raw = cfg.k_heading * theta_e + cross_term;
        if state.velocity_valid && sc.k_yaw_rate != 0.0 {
            raw += sc.k_yaw_rate * (v * kappa_path - state.velocity.angular);
        }

        let omega = if ackermann {
            let dmax = steering_limit(constraints);
            let delta = (direction * raw).clamp(-dmax, dmax);
            v * steering_to_curvature(delta, constraints)
        } else {
            cfg.kp_angular.max(0.1) * raw
        };

        self.base.status.mode = if direction > 0.0 {
            "stanley_forward".into()
        } else {
            "stanley_reverse".into()
        };
        let msg = if direction > 0.0 {
            "Following path"
        } else {
            "Reversing"
        };
        finalize(v, omega, constraints, &cfg, allow_reverse, msg)
    }

    fn set_path(&mut self, path: Path) {
        self.cursor.set_path(&path.waypoints);
        self.base.path = path;
        self.base.path_index = 0;
        self.base.status = Default::default();
    }

    fn reset(&mut self) {
        self.set_path(Path::default());
        self.base.status = Default::default();
    }

    fn get_type(&self) -> &'static str {
        "stanley_follower"
    }

    fn base(&self) -> &ControllerBase {
        &self.base
    }

    fn base_mut(&mut self) -> &mut ControllerBase {
        &mut self.base
    }
}
