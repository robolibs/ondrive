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
    PathProjection, cumulative_lengths, curvature_at_projection, project, speed_cap,
};
use crate::types::{Goal, Path, RobotConstraints, RobotState, VelocityCommand, WorldConstraints};
use datapod::Point;
use std::f64::consts::PI;

const SEARCH_WINDOW: usize = 64;

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
    cum: Vec<f64>,
    started: bool,
}

impl StanleyFollower {
    pub fn new() -> Self {
        Self::default()
    }

    fn project_at(&mut self, p: Point) -> Option<PathProjection> {
        let window = if self.started {
            SEARCH_WINDOW
        } else {
            usize::MAX
        };
        let proj = project(
            &self.base.path.waypoints,
            &self.cum,
            p,
            self.base.path_index,
            2,
            window,
        )?;
        self.started = true;
        self.base.path_index = proj.segment;
        Some(proj)
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
        if self.cum.len() != self.base.path.waypoints.len() {
            self.cum = cumulative_lengths(&self.base.path.waypoints);
            self.started = false;
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
        let front = Point::new(
            state.pose.point.x + front_offset * c,
            state.pose.point.y + front_offset * s,
            0.0,
        );

        let Some(mut proj) = self.project_at(front) else {
            return VelocityCommand::invalid("no path");
        };
        let (pos_tol, ang_tol) = effective_tolerances(goal, &cfg);
        let length = self.cum.last().copied().unwrap_or(0.0);
        let at_robot = if ackermann {
            project(
                &self.base.path.waypoints,
                &self.cum,
                state.pose.point,
                self.base.path_index,
                SEARCH_WINDOW,
                SEARCH_WINDOW,
            )
        } else {
            None
        };
        let arrival_proj = at_robot.as_ref().unwrap_or(&proj);
        if let Some(cmd) = self.base.arrival_on_path(state, goal, constraints, arrival_proj, length) {
            return cmd;
        }

        if is_holonomic(constraints.steering_type) {
            let theta_e = normalize_angle(proj.heading - yaw);
            self.base.status.cross_track_error = proj.lateral_error;
            self.base.status.heading_error = theta_e;
            self.base.status.goal_reached = false;
            self.base.status.mode = "stanley_holonomic".into();

            let kappa_path = curvature_at_projection(&self.base.path.waypoints, &self.cum, &proj);
            let nominal = speed_cap(&self.base.path.speeds, &proj)
                .map_or(constraints.max_linear_velocity, |v| {
                    v.min(constraints.max_linear_velocity)
                });
            let forward = path_speed(
                nominal,
                kappa_path,
                self.base.status.distance_to_goal,
                pos_tol,
                cfg.kp_linear,
                constraints,
            );
            let (vx, vy) = path_to_body(forward, -cfg.k_cross_track * proj.lateral_error, proj.heading, yaw);
            let omega = cfg.kp_angular.max(0.1) * theta_e;
            return finalize_holonomic(vx, vy, omega, constraints, &cfg, "Following path");
        }

        let mut theta_e = normalize_angle(proj.heading - yaw);
        let mut direction = 1.0;
        let reverse = match crate::core::path::direction_at(&self.base.path.speeds, &proj) {
            Some(d) => allow_reverse && d < 0,
            None => allow_reverse && theta_e.abs() > PI / 2.0,
        };
        if reverse {
            direction = -1.0;
            let rear = Point::new(
                state.pose.point.x - rear_offset * c,
                state.pose.point.y - rear_offset * s,
                0.0,
            );
            if let Some(p) = self.project_at(rear) {
                proj = p;
            }
            theta_e = normalize_angle(proj.heading - (yaw + PI));
        }

        self.base.status.cross_track_error = proj.lateral_error;
        self.base.status.heading_error = theta_e;
        self.base.status.goal_reached = false;

        if state.turn_first && can_turn_in_place(constraints.steering_type) && theta_e.abs() > ang_tol
        {
            self.base.status.mode = "turning".into();
            let omega = cfg.kp_angular.max(0.1) * theta_e;
            return finalize(0.0, omega, constraints, &cfg, allow_reverse, "Turning to align");
        }

        let kappa_path = curvature_at_projection(&self.base.path.waypoints, &self.cum, &proj);
        let nominal = speed_cap(&self.base.path.speeds, &proj)
            .map_or(constraints.max_linear_velocity, |v| {
                v.min(constraints.max_linear_velocity)
            });
        let v = direction
            * path_speed(
                nominal,
                kappa_path,
                self.base.status.distance_to_goal,
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
        self.cum = cumulative_lengths(&path.waypoints);
        self.base.path = path;
        self.base.path_index = 0;
        self.base.status = Default::default();
        self.started = false;
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
