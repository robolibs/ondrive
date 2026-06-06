use crate::controller::{Controller, ControllerBase, is_goal_reached};
use crate::core::math::heading_error;
use crate::types::{
    Goal, RobotConstraints, RobotState, SteeringType, VelocityCommand, WorldConstraints,
};
use std::f64::consts::PI;

const HEADING_DEADBAND: f64 = 0.1;

#[derive(Clone, Debug, Default)]
pub struct PidFollower {
    pub base: ControllerBase,
    linear_integral: f64,
    angular_integral: f64,
    last_distance_error: f64,
    last_heading_error: f64,
}

impl PidFollower {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Controller for PidFollower {
    fn compute_control(
        &mut self,
        state: &RobotState,
        goal: &Goal,
        constraints: &RobotConstraints,
        dt: f64,
        _world: Option<&WorldConstraints>,
    ) -> VelocityCommand {
        let cfg = self.base.config.clone();

        let (reached, dist, yaw_diff) = is_goal_reached(
            &state.pose,
            &goal.target_pose,
            cfg.goal_tolerance,
            cfg.angular_tolerance,
        );
        self.base.status.distance_to_goal = dist;
        self.base.status.heading_error = yaw_diff;

        if reached {
            self.base.status.goal_reached = true;
            self.base.status.mode = "stopped".into();
            return VelocityCommand {
                valid: true,
                status_message: "Goal reached".into(),
                ..VelocityCommand::default()
            };
        }

        let is_diff = matches!(
            constraints.steering_type,
            SteeringType::Differential | SteeringType::SkidSteer
        );

        let dx = goal.target_pose.point.x - state.pose.point.x;
        let dy = goal.target_pose.point.y - state.pose.point.y;
        let distance_error = (dx * dx + dy * dy).sqrt();
        let heading_err = heading_error(&state.pose, goal.target_pose.point);

        self.base.status.distance_to_goal = distance_error;
        self.base.status.heading_error = heading_err;
        self.base.status.goal_reached = false;

        let angular_control = if heading_err.abs() > HEADING_DEADBAND {
            self.angular_integral += heading_err * dt;
            let derivative = (heading_err - self.last_heading_error) / dt;
            self.last_heading_error = heading_err;
            (cfg.kp_angular * 3.0) * heading_err
                + cfg.ki_angular * self.angular_integral
                + cfg.kd_angular * derivative
        } else {
            self.angular_integral = 0.0;
            self.last_heading_error = 0.0;
            0.0
        };

        if is_diff && state.turn_first && heading_err.abs() > cfg.angular_tolerance {
            self.base.status.mode = "turning".into();
            return VelocityCommand {
                valid: true,
                status_message: "Turning to align".into(),
                linear_velocity: 0.0,
                angular_velocity: angular_control.clamp(
                    -constraints.max_angular_velocity,
                    constraints.max_angular_velocity,
                ),
                ..VelocityCommand::default()
            };
        }

        self.linear_integral += distance_error * dt;
        let linear_derivative = (distance_error - self.last_distance_error) / dt;
        self.last_distance_error = distance_error;

        let mut linear_control = cfg.kp_linear * distance_error
            + cfg.ki_linear * self.linear_integral
            + cfg.kd_linear * linear_derivative;

        let turn_reduction = (1.0 - (heading_err.abs() / (PI / 6.0)).min(0.95)).max(0.05);
        linear_control *= turn_reduction;

        if heading_err.abs() > PI / 3.0 {
            linear_control *= 0.1;
        }

        self.base.status.mode = "tracking".into();
        VelocityCommand {
            valid: true,
            status_message: "Tracking goal".into(),
            linear_velocity: linear_control.clamp(
                constraints.min_linear_velocity,
                constraints.max_linear_velocity,
            ),
            angular_velocity: angular_control.clamp(
                -constraints.max_angular_velocity,
                constraints.max_angular_velocity,
            ),
            ..VelocityCommand::default()
        }
    }

    fn reset(&mut self) {
        self.base.path.waypoints.clear();
        self.base.path_index = 0;
        self.base.status = Default::default();
        self.linear_integral = 0.0;
        self.angular_integral = 0.0;
        self.last_distance_error = 0.0;
        self.last_heading_error = 0.0;
    }

    fn get_type(&self) -> &'static str {
        "pid_follower"
    }

    fn base(&self) -> &ControllerBase {
        &self.base
    }

    fn base_mut(&mut self) -> &mut ControllerBase {
        &mut self.base
    }
}
