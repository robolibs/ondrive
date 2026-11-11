//! Polar-coordinate pose regulator (Astolfi / Aicardi et al.) for platforms
//! that can turn in place. In the goal frame the error is `(rho, alpha,
//! beta)` and the law `v = k_rho rho`, `omega = k_alpha alpha + k_beta beta`
//! converges to position and orientation simultaneously for
//! `k_rho > 0`, `k_beta < 0`, `k_alpha > k_rho`. Goals behind the robot are
//! approached in reverse when allowed. A platform that cannot turn in
//! place is brought to the pose along planned curves (`PoseReachFollower`).

use crate::controller::{Controller, ControllerBase, check_goal, pass_through};
use crate::core::kinematics::{
    can_turn_in_place, finalize, finalize_holonomic, is_holonomic, reverse_allowed, stop,
    world_to_body,
};
use crate::core::math::{normalize_angle, yaw_of};
use crate::point::PoseReachFollower;
use crate::types::{ControllerConfig, Goal, RobotConstraints, RobotState, VelocityCommand, WorldConstraints};
use std::f64::consts::PI;

#[derive(Clone, Debug, Default)]
pub struct PoseRegulatorFollower {
    pub base: ControllerBase,
    /// Driving direction chosen for the goal it belongs to.
    direction: Option<((f64, f64, f64), f64)>,
    /// Curve planner and tracker for a platform that cannot turn in place.
    curves: PoseReachFollower,
}

impl PoseRegulatorFollower {
    pub fn new() -> Self {
        Self::default()
    }

    /// Gains derived from the generic config: `k_rho = kp_linear`,
    /// `k_alpha = max(kp_angular, k_rho + 0.5)`, `k_beta = -0.5 k_heading`.
    pub fn gains(&self) -> (f64, f64, f64) {
        let cfg = &self.base.config;
        let k_rho = cfg.kp_linear.max(0.1);
        let k_alpha = cfg.kp_angular.max(k_rho + 0.5);
        let k_beta = -0.5 * cfg.k_heading.max(0.1);
        (k_rho, k_alpha, k_beta)
    }
}

impl Controller for PoseRegulatorFollower {
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
        let allow_reverse = reverse_allowed(&cfg, state);
        let check = check_goal(&state.pose, goal, &cfg);
        self.base.status.distance_to_goal = check.distance;
        self.base.status.heading_error = check.yaw_error;
        self.base.status.cross_track_error = 0.0;
        if check.reached && pass_through(goal) == 0.0 {
            self.base.status.goal_reached = true;
            self.base.status.mode = "stopped".into();
            return stop("Goal reached");
        }
        self.base.status.goal_reached = false;

        if is_holonomic(constraints.steering_type) {
            self.base.status.mode = "regulating_holonomic".into();
            let (k_rho, k_alpha, _) = self.gains();
            let dx = goal.target_pose.point.x - state.pose.point.x;
            let dy = goal.target_pose.point.y - state.pose.point.y;
            let rho = dx.hypot(dy);
            let speed = (k_rho * rho).min(constraints.max_linear_velocity.abs());
            let yaw = yaw_of(&state.pose);
            let (ux, uy) = if rho > 1e-9 { (dx / rho, dy / rho) } else { (0.0, 0.0) };
            let (vx, vy) = world_to_body(speed * ux, speed * uy, yaw);
            let omega = k_alpha * check.yaw_error;
            return finalize_holonomic(vx, vy, omega, constraints, &cfg, "Regulating to pose");
        }
        if !can_turn_in_place(constraints.steering_type) {
            let cmd = self.curves.compute_control(state, goal, constraints, dt, world);
            let inner = self.curves.get_status();
            self.base.status.goal_reached = inner.goal_reached;
            self.base.status.cross_track_error = inner.cross_track_error;
            self.base.status.mode = format!("pose_regulator/{}", inner.mode);
            return cmd;
        }

        let (k_rho, k_alpha, k_beta) = self.gains();
        let yaw = yaw_of(&state.pose);
        let goal_yaw = yaw_of(&goal.target_pose);
        let dx = goal.target_pose.point.x - state.pose.point.x;
        let dy = goal.target_pose.point.y - state.pose.point.y;
        let rho = dx.hypot(dy);

        // Inside the position tolerance only the orientation remains.
        if check.position_ok {
            self.base.status.mode = "aligning".into();
            let omega = k_alpha * check.yaw_error;
            return finalize(0.0, omega, constraints, &cfg, allow_reverse, "Aligning to goal orientation");
        }

        let theta = normalize_angle(yaw - goal_yaw);
        let mut alpha = normalize_angle(dy.atan2(dx) - yaw);
        let key = (goal.target_pose.point.x, goal.target_pose.point.y, goal_yaw);
        let direction = match self.direction {
            Some((k, d)) if k == key && (d > 0.0 || allow_reverse) => d,
            _ => {
                let d = if allow_reverse && alpha.abs() > PI / 2.0 { -1.0 } else { 1.0 };
                self.direction = Some((key, d));
                d
            }
        };
        if direction < 0.0 {
            alpha = normalize_angle(alpha + PI);
        }
        let beta = normalize_angle(-theta - alpha);
        let wanted = k_rho * rho;
        let scale = if wanted > 1e-9 { (constraints.max_linear_velocity.abs() / wanted).min(1.0) } else { 1.0 };
        let v = direction * wanted * scale;
        let omega = (k_alpha * alpha + k_beta * beta) * scale;
        self.base.status.mode = if direction > 0.0 { "regulating".into() } else { "regulating_reverse".into() };
        finalize(v, omega, constraints, &cfg, allow_reverse, "Regulating to pose")
    }

    fn set_config(&mut self, config: ControllerConfig) {
        self.curves.set_config(config.clone());
        self.base.config = config;
    }

    fn reset(&mut self) {
        self.curves.reset();
        self.direction = None;
        self.base.status = Default::default();
    }

    fn get_type(&self) -> &'static str {
        "pose_regulator_follower"
    }

    fn base(&self) -> &ControllerBase {
        &self.base
    }

    fn base_mut(&mut self) -> &mut ControllerBase {
        &mut self.base
    }
}
