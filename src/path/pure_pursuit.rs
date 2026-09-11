//! Pure Pursuit (Coulter 1992). The rear axle chases a point at the
//! lookahead arc length ahead of its projection on the path; the arc
//! through both points has curvature `2 sin(alpha) / L_d`.

use crate::controller::{Controller, ControllerBase, effective_tolerances};
use crate::core::kinematics::{
    can_turn_in_place, finalize, heading_speed_scale, is_ackermann, path_speed, reverse_allowed,
};
use crate::core::math::{heading_error, normalize_angle};
use crate::core::path::{cumulative_lengths, cusp_after, project, sample, speed_cap};
use crate::types::{Goal, Path, RobotConstraints, RobotState, VelocityCommand, WorldConstraints};
use datapod::Point;
use std::f64::consts::PI;

const LOOKAHEAD_VELOCITY_GAIN: f64 = 0.3;
const SEARCH_WINDOW: usize = 64;

#[derive(Clone, Debug, Default)]
pub struct PurePursuitFollower {
    pub base: ControllerBase,
    cum: Vec<f64>,
    started: bool,
    lookahead_point: Option<Point>,
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

        let window = if self.started {
            SEARCH_WINDOW
        } else {
            usize::MAX
        };
        let Some(proj) = project(
            &self.base.path.waypoints,
            &self.cum,
            rear,
            self.base.path_index,
            2,
            window,
        ) else {
            return VelocityCommand::invalid("no path");
        };
        self.started = true;
        self.base.path_index = proj.segment;

        let (pos_tol, ang_tol) = effective_tolerances(goal, &cfg);
        let passed_end = proj.beyond_end && proj.distance < 2.0 * pos_tol;
        if let Some(cmd) = self.base.arrival(state, goal, constraints, passed_end) {
            self.lookahead_point = None;
            return cmd;
        }

        let lookahead = (cfg.lookahead_distance
            + LOOKAHEAD_VELOCITY_GAIN * state.velocity.linear.abs())
        .clamp(cfg.lookahead_distance.max(1e-3), 3.0 * cfg.lookahead_distance.max(1e-3));
        let mut s_target = proj.arc_length + lookahead;
        let mut stop_distance = self.base.status.distance_to_goal;
        if let Some(cusp) = cusp_after(&self.base.path.speeds, &self.cum, proj.arc_length) {
            s_target = s_target.min(cusp);
            stop_distance = stop_distance.min((cusp - proj.arc_length).max(0.0));
        }
        let (target, _) = sample(&self.base.path.waypoints, &self.cum, s_target);
        self.lookahead_point = Some(target);

        let dx = target.x - rear.x;
        let dy = target.y - rear.y;
        let ld = dx.hypot(dy);
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
        let v = direction
            * path_speed(nominal, kappa, stop_distance, pos_tol, cfg.kp_linear, constraints)
            * scale;

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
        self.cum = cumulative_lengths(&path.waypoints);
        self.base.path = path;
        self.base.path_index = 0;
        self.base.status = Default::default();
        self.started = false;
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
