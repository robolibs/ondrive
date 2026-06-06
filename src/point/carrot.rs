use crate::controller::{Controller, ControllerBase, is_goal_reached};
use crate::core::math::heading_error;
use crate::types::{
    Goal, RobotConstraints, RobotState, SteeringType, VelocityCommand, WorldConstraints,
};
use std::f64::consts::PI;

#[derive(Clone, Debug, Default)]
pub struct CarrotFollower {
    pub base: ControllerBase,
}

impl CarrotFollower {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Controller for CarrotFollower {
    fn compute_control(
        &mut self,
        state: &RobotState,
        goal: &Goal,
        constraints: &RobotConstraints,
        _dt: f64,
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
        let distance = (dx * dx + dy * dy).sqrt();
        let heading_err = heading_error(&state.pose, goal.target_pose.point);

        self.base.status.distance_to_goal = distance;
        self.base.status.heading_error = heading_err;
        self.base.status.goal_reached = false;

        let angular_control = cfg.kp_angular * heading_err;

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

        let turn_reduction = 1.0 - (heading_err.abs() / PI).min(0.8);
        let distance_scale = (distance / (cfg.goal_tolerance * 5.0)).min(1.0);
        let linear_control = cfg.kp_linear * distance * turn_reduction * distance_scale;

        self.base.status.mode = "carrot".into();
        VelocityCommand {
            valid: true,
            status_message: "Chasing carrot".into(),
            linear_velocity: linear_control.clamp(0.0, constraints.max_linear_velocity),
            angular_velocity: angular_control.clamp(
                -constraints.max_angular_velocity,
                constraints.max_angular_velocity,
            ),
            ..VelocityCommand::default()
        }
    }

    fn get_type(&self) -> &'static str {
        "carrot_follower"
    }

    fn base(&self) -> &ControllerBase {
        &self.base
    }

    fn base_mut(&mut self) -> &mut ControllerBase {
        &mut self.base
    }
}
