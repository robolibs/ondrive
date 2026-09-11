//! Carrot point-to-point controller: proportional heading control onto the
//! bearing to the goal with speed scaled by distance and heading error.

use crate::controller::{Controller, ControllerBase, effective_tolerances};
use crate::core::kinematics::{
    can_turn_in_place, finalize, heading_speed_scale, reverse_allowed, unreachable_arc_speed_floor,
};
use crate::core::math::{heading_error, normalize_angle};
use crate::types::{Goal, RobotConstraints, RobotState, VelocityCommand, WorldConstraints};
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
        dt: f64,
        _world: Option<&WorldConstraints>,
    ) -> VelocityCommand {
        if !(dt.is_finite() && dt > 0.0) {
            return VelocityCommand::invalid("dt must be positive and finite");
        }
        let cfg = self.base.config.clone();
        let allow_reverse = reverse_allowed(&cfg, state);

        if let Some(cmd) = self.base.arrival(state, goal, constraints, false) {
            return cmd;
        }

        let distance = state.pose.point.distance_to_2d(goal.target_pose.point);
        let mut bearing_err = heading_error(&state.pose, goal.target_pose.point);
        let mut direction = 1.0;
        if allow_reverse && bearing_err.abs() > PI / 2.0 {
            direction = -1.0;
            bearing_err = normalize_angle(bearing_err + PI);
        }

        self.base.status.distance_to_goal = distance;
        self.base.status.heading_error = bearing_err;
        self.base.status.cross_track_error = 0.0;
        self.base.status.goal_reached = false;

        let omega = cfg.kp_angular * bearing_err;

        let (_, ang_tol) = effective_tolerances(goal, &cfg);
        if state.turn_first
            && can_turn_in_place(constraints.steering_type)
            && bearing_err.abs() > ang_tol
        {
            self.base.status.mode = "turning".into();
            return finalize(0.0, omega, constraints, &cfg, allow_reverse, "Turning to align");
        }

        let carrot = cfg.lookahead_distance.max(1e-3);
        let speed = cfg.kp_linear * distance.min(carrot) * (distance / carrot).min(1.0).sqrt()
            + cfg.kp_linear * (distance - carrot).max(0.0);
        let mut v = speed.max(0.0) * heading_speed_scale(bearing_err, constraints);
        if let Some(floor) = unreachable_arc_speed_floor(bearing_err, distance, constraints) {
            v = v.max(floor);
        }
        let v = direction * v;

        self.base.status.mode = "carrot".into();
        finalize(v, omega, constraints, &cfg, allow_reverse, "Chasing carrot")
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
