//! Dual-loop PID point-to-point controller: a distance loop sets the speed
//! toward the goal, a heading loop steers onto the bearing. Both loops use
//! clamped integrals, wrapped derivatives and a `dt` guard.

use crate::controller::{Controller, ControllerBase};
use crate::core::kinematics::{
    can_turn_in_place, finalize, finalize_holonomic, heading_speed_scale, is_holonomic,
    reverse_allowed, unreachable_arc_speed_floor, world_to_body,
};
use crate::core::math::{heading_error, normalize_angle, yaw_of};
use crate::types::{Goal, RobotConstraints, RobotState, VelocityCommand, WorldConstraints};
use datapod::Point;
use std::f64::consts::PI;

/// Time constant of the distance integral's leak while the robot closes in.
const INTEGRAL_LEAK_S: f64 = 1.0;

#[derive(Clone, Debug, Default)]
pub struct PidFollower {
    pub base: ControllerBase,
    linear_integral: f64,
    angular_integral: f64,
    last_distance_error: f64,
    last_heading_error: f64,
    has_last: bool,
    last_direction: f64,
    last_goal: Option<Point>,
    linear_derivative: f64,
}

impl PidFollower {
    pub fn new() -> Self {
        Self::default()
    }

    fn clear_loops(&mut self) {
        self.linear_integral = 0.0;
        self.linear_derivative = 0.0;
        self.angular_integral = 0.0;
        self.last_distance_error = 0.0;
        self.last_heading_error = 0.0;
        self.has_last = false;
        self.last_direction = 0.0;
    }

    /// Distance loop on the signed error `e` (negative once past the goal):
    /// PID with a low-pass derivative. The integral builds only while the
    /// robot is not closing in and the output is not saturated, and leaks
    /// away while it is closing in. Returns a speed in `[0, limit]`.
    fn linear_loop(&mut self, e: f64, dt: f64, cfg: &crate::types::ControllerConfig, limit: f64) -> f64 {
        let raw_derivative = if self.has_last { (e - self.last_distance_error) / dt } else { 0.0 };
        self.last_distance_error = e;
        let filter = if cfg.kp_linear > 1e-9 { 0.1 * cfg.kd_linear.abs() / cfg.kp_linear } else { 0.0 };
        self.linear_derivative += dt / (filter + dt) * (raw_derivative - self.linear_derivative);
        let limit = limit.max(0.0);
        let output = |integral: f64| cfg.kp_linear * e + cfg.ki_linear * integral + cfg.kd_linear * self.linear_derivative;
        let raw = output(self.linear_integral);
        let saturated = (raw >= limit && e > 0.0) || (raw <= 0.0 && e < 0.0);
        let closing = e * raw_derivative < -0.05 * e.abs();
        if closing {
            self.linear_integral *= (-dt / INTEGRAL_LEAK_S).exp();
        } else if !saturated {
            self.linear_integral = clamped_integral(self.linear_integral, e, dt, cfg.ki_linear, limit);
        }
        output(self.linear_integral).clamp(0.0, limit)
    }

    fn note_goal(&mut self, goal: &Goal) {
        let p = goal.target_pose.point;
        let changed = match self.last_goal {
            Some(q) => q.distance_to_2d(p) > 1e-9,
            None => true,
        };
        if changed {
            self.clear_loops();
            self.last_goal = Some(p);
        }
    }

    /// Holonomic point control: translation (toward the goal position) and
    /// rotation (toward the goal orientation) are independent PID loops,
    /// each transformed into the body frame at the end.
    fn compute_holonomic(
        &mut self,
        state: &RobotState,
        goal: &Goal,
        constraints: &RobotConstraints,
        dt: f64,
        cfg: &crate::types::ControllerConfig,
    ) -> VelocityCommand {
        let dx = goal.target_pose.point.x - state.pose.point.x;
        let dy = goal.target_pose.point.y - state.pose.point.y;
        let distance = dx.hypot(dy);
        let orientation_err = normalize_angle(yaw_of(&goal.target_pose) - yaw_of(&state.pose));

        let speed = self.linear_loop(distance, dt, cfg, constraints.max_linear_velocity);

        self.angular_integral = clamped_integral(
            self.angular_integral,
            orientation_err,
            dt,
            cfg.ki_angular,
            constraints.max_angular_velocity,
        );
        let angular_derivative = if self.has_last {
            normalize_angle(orientation_err - self.last_heading_error) / dt
        } else {
            0.0
        };
        let omega = cfg.kp_angular * orientation_err
            + cfg.ki_angular * self.angular_integral
            + cfg.kd_angular * angular_derivative;

        self.last_heading_error = orientation_err;
        self.has_last = true;
        self.last_direction = 1.0;

        self.base.status.distance_to_goal = distance;
        self.base.status.heading_error = orientation_err;
        self.base.status.cross_track_error = 0.0;
        self.base.status.goal_reached = false;
        self.base.status.mode = "tracking_holonomic".into();

        let yaw = yaw_of(&state.pose);
        let (ux, uy) = if distance > 1e-9 {
            (dx / distance, dy / distance)
        } else {
            (0.0, 0.0)
        };
        let (vx, vy) = world_to_body(speed * ux, speed * uy, yaw);
        finalize_holonomic(vx, vy, omega, constraints, cfg, "Tracking goal")
    }
}

fn clamped_integral(acc: f64, err: f64, dt: f64, gain: f64, output_limit: f64) -> f64 {
    let next = acc + err * dt;
    if gain > 1e-12 && output_limit.is_finite() {
        let bound = (output_limit / gain).abs();
        next.clamp(-bound, bound)
    } else {
        next
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
        if !(dt.is_finite() && dt > 0.0) {
            return VelocityCommand::invalid("dt must be positive and finite");
        }
        let cfg = self.base.config.clone();
        let allow_reverse = reverse_allowed(&cfg, state);
        self.note_goal(goal);

        if let Some(cmd) = self.base.arrival(state, goal, constraints, false) {
            self.clear_loops();
            return cmd;
        }

        if is_holonomic(constraints.steering_type) {
            return self.compute_holonomic(state, goal, constraints, dt, &cfg);
        }

        let distance = state.pose.point.distance_to_2d(goal.target_pose.point);
        let mut bearing_err = heading_error(&state.pose, goal.target_pose.point);
        let mut direction = 1.0;
        if allow_reverse && bearing_err.abs() > PI / 2.0 {
            direction = -1.0;
            bearing_err = normalize_angle(bearing_err + PI);
        }
        if self.last_direction != direction {
            self.has_last = false;
            self.angular_integral = 0.0;
        }
        self.last_direction = direction;

        self.angular_integral = clamped_integral(
            self.angular_integral,
            bearing_err,
            dt,
            cfg.ki_angular,
            constraints.max_angular_velocity,
        );
        let angular_derivative = if self.has_last {
            normalize_angle(bearing_err - self.last_heading_error) / dt
        } else {
            0.0
        };
        let omega = cfg.kp_angular * bearing_err
            + cfg.ki_angular * self.angular_integral
            + cfg.kd_angular * angular_derivative;

        let along = distance * bearing_err.cos();
        let speed = self.linear_loop(along, dt, &cfg, constraints.max_linear_velocity);

        self.last_heading_error = bearing_err;
        self.has_last = true;

        self.base.status.distance_to_goal = distance;
        self.base.status.heading_error = bearing_err;
        self.base.status.cross_track_error = 0.0;
        self.base.status.goal_reached = false;

        let (_, ang_tol) = crate::controller::effective_tolerances(goal, &cfg);
        let turning = state.turn_first
            && can_turn_in_place(constraints.steering_type)
            && bearing_err.abs() > ang_tol;
        if turning {
            self.base.status.mode = "turning".into();
            return finalize(0.0, omega, constraints, &cfg, allow_reverse, "Turning to align");
        }

        let mut v = speed.max(0.0) * heading_speed_scale(bearing_err, constraints);
        if let Some(floor) = unreachable_arc_speed_floor(bearing_err, distance, constraints) {
            v = v.max(floor);
        }
        let v = direction * v;
        self.base.status.mode = "tracking".into();
        finalize(v, omega, constraints, &cfg, allow_reverse, "Tracking goal")
    }

    fn reset(&mut self) {
        self.base.path = Default::default();
        self.base.path_index = 0;
        self.base.status = Default::default();
        self.clear_loops();
        self.last_goal = None;
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
