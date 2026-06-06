#![allow(clippy::needless_range_loop)]

use crate::controller::{Controller, ControllerBase, is_goal_reached};
use crate::core::math::{heading_error as heading_err_to_point, normalize_angle};
use crate::types::{
    Goal, RobotConstraints, RobotState, SteeringType, VelocityCommand, WorldConstraints,
};
use datapod::{Point, Pose};
use std::f64::consts::PI;

#[derive(Clone, Debug, Default)]
pub struct StanleyFollower {
    pub base: ControllerBase,
}

impl StanleyFollower {
    pub fn new() -> Self {
        Self::default()
    }

    fn update_path_index(&mut self, current: &Pose) {
        let waypoints = &self.base.path.waypoints;
        if waypoints.is_empty() {
            return;
        }
        while self.base.path_index + 1 < waypoints.len() {
            let wp = waypoints[self.base.path_index].point;
            let dist = current.point.distance_to(wp);
            if dist < 1.0 {
                self.base.path_index += 1;
            } else {
                break;
            }
        }
    }

    fn find_closest_path_point(&self, current: &Pose) -> (Point, f64, f64) {
        let waypoints = &self.base.path.waypoints;
        if waypoints.is_empty() {
            return (Point::new(0.0, 0.0, 0.0), 0.0, 0.0);
        }

        let search_start = self.base.path_index.saturating_sub(5);
        let search_end = (self.base.path_index + 20).min(waypoints.len());

        let mut min_distance = f64::MAX;
        let mut closest_idx = self.base.path_index;

        for i in search_start..search_end {
            let d = current.point.distance_to(waypoints[i].point);
            if d < min_distance {
                min_distance = d;
                closest_idx = i;
            }
        }

        let closest = waypoints[closest_idx].point;

        let path_heading = if closest_idx + 1 < waypoints.len() {
            let next = waypoints[closest_idx + 1].point;
            (next.y - closest.y).atan2(next.x - closest.x)
        } else if closest_idx > 0 {
            let prev = waypoints[closest_idx - 1].point;
            (closest.y - prev.y).atan2(closest.x - prev.x)
        } else {
            0.0
        };

        let dx = current.point.x - closest.x;
        let dy = current.point.y - closest.y;
        let mut cte = (dx * dx + dy * dy).sqrt();

        let sign_check = dy * path_heading.cos() - dx * path_heading.sin();
        if sign_check > 0.0 {
            cte = -cte;
        }

        (closest, path_heading, cte)
    }

    fn compute_goal_control(
        &self,
        state: &RobotState,
        goal: &Goal,
        constraints: &RobotConstraints,
    ) -> VelocityCommand {
        let heading_err = heading_err_to_point(&state.pose, goal.target_pose.point);
        let angular = (self.base.config.kp_angular * heading_err).clamp(
            -constraints.max_angular_velocity,
            constraints.max_angular_velocity,
        );
        VelocityCommand {
            valid: true,
            status_message: "Moving to goal".into(),
            linear_velocity: constraints.max_linear_velocity,
            angular_velocity: angular,
            ..VelocityCommand::default()
        }
    }
}

impl Controller for StanleyFollower {
    fn compute_control(
        &mut self,
        state: &RobotState,
        goal: &Goal,
        constraints: &RobotConstraints,
        _dt: f64,
        _world: Option<&WorldConstraints>,
    ) -> VelocityCommand {
        let cfg = self.base.config.clone();

        let is_diff = matches!(
            constraints.steering_type,
            SteeringType::Differential | SteeringType::SkidSteer
        );

        if self.base.path.waypoints.is_empty() {
            return self.compute_goal_control(state, goal, constraints);
        }

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

        self.update_path_index(&state.pose);
        let (_closest, path_heading, cte) = self.find_closest_path_point(&state.pose);

        let mut heading_err = normalize_angle(path_heading - state.pose.rotation.to_euler().yaw);

        if is_diff && state.turn_first && heading_err.abs() > cfg.angular_tolerance {
            let kp_angular = 2.0;
            let angular = (kp_angular * heading_err).clamp(
                -constraints.max_angular_velocity,
                constraints.max_angular_velocity,
            );
            self.base.status.distance_to_goal =
                state.pose.point.distance_to(goal.target_pose.point);
            self.base.status.cross_track_error = cte;
            self.base.status.heading_error = heading_err;
            self.base.status.goal_reached = false;
            self.base.status.mode = "turning".into();
            return VelocityCommand {
                valid: true,
                status_message: "Turning to align".into(),
                linear_velocity: 0.0,
                angular_velocity: angular,
                ..VelocityCommand::default()
            };
        }

        let mut velocity_direction = 1.0;
        if cfg.allow_reverse && heading_err.abs() > PI / 2.0 {
            velocity_direction = -1.0;
            heading_err = normalize_angle(heading_err + PI);
        }

        let linear = velocity_direction * constraints.max_linear_velocity;
        let k_cte = 1.0;
        let velocity = linear.abs().max(0.1);

        let delta_e = (k_cte * cte).atan2(velocity);
        let delta = normalize_angle(heading_err + delta_e);

        let angular = if is_diff {
            let kp_angular = 2.0;
            kp_angular * delta
        } else {
            let wheelbase = if constraints.wheelbase > 0.0 {
                constraints.wheelbase
            } else {
                1.0
            };
            velocity * delta.tan() / wheelbase
        };
        let angular = angular.clamp(
            -constraints.max_angular_velocity,
            constraints.max_angular_velocity,
        );

        self.base.status.distance_to_goal = state.pose.point.distance_to(goal.target_pose.point);
        self.base.status.cross_track_error = cte;
        self.base.status.heading_error = heading_err;
        self.base.status.goal_reached = false;
        self.base.status.mode = if velocity_direction > 0.0 {
            "stanley_forward".into()
        } else {
            "stanley_reverse".into()
        };

        VelocityCommand {
            valid: true,
            status_message: if velocity_direction > 0.0 {
                "Following path".into()
            } else {
                "Reversing".into()
            },
            linear_velocity: linear,
            angular_velocity: angular,
            ..VelocityCommand::default()
        }
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
