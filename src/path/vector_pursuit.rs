//! Vector Pursuit (Wit 2000): Pure Pursuit that also uses the orientation
//! of the lookahead point. The translation screw gives the Pure Pursuit arc
//! `kappa_t = 2 sin(alpha) / L_d`; the rotation screw about the pole of the
//! pose change gives `kappa_r = 2 sin(dtheta / 2) / L_d`. The two agree on
//! circular arcs and are blended with `k_orientation` elsewhere.

use crate::controller::{Controller, ControllerBase, effective_tolerances};
use crate::core::kinematics::{
    can_turn_in_place, finalize, heading_speed_scale, is_ackermann, path_speed, reverse_allowed,
};
use crate::core::math::{heading_error, normalize_angle};
use crate::core::path::{PathCursor, curvature_at_projection, cusp_after, sample, speed_cap};
use crate::types::{Goal, Path, RobotConstraints, RobotState, VelocityCommand, WorldConstraints};
use datapod::Point;
use std::f64::consts::PI;

/// Lookahead shrink per unit of path curvature (metres of lookahead per
/// unit curvature), keeping tight bends from being cut.
const CURVATURE_LOOKAHEAD_GAIN: f64 = 0.5;

#[derive(Clone, Debug)]
pub struct VectorPursuitConfig {
    /// Weight of the rotation screw in the blend (0 = plain Pure Pursuit,
    /// 1 = pole rotation only), applied in full from `lateral_scale` of
    /// lateral error and fading to zero on the path.
    pub k_orientation: f64,
    pub lateral_scale: f64,
}

impl Default for VectorPursuitConfig {
    fn default() -> Self {
        Self { k_orientation: 0.5, lateral_scale: 0.5 }
    }
}

#[derive(Clone, Debug, Default)]
pub struct VectorPursuitFollower {
    pub base: ControllerBase,
    pub vp_config: VectorPursuitConfig,
    cursor: PathCursor,
    lookahead_point: Option<Point>,
}

impl VectorPursuitFollower {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_vp_config(cfg: VectorPursuitConfig) -> Self {
        Self { vp_config: cfg, ..Default::default() }
    }

    pub fn lookahead_point(&self) -> Option<Point> {
        self.lookahead_point
    }
}

impl Controller for VectorPursuitFollower {
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
        let yaw = state.pose.rotation.to_euler().yaw;

        if self.base.path.waypoints.is_empty() {
            if let Some(cmd) = self.base.arrival(state, goal, constraints, false) {
                return cmd;
            }
            let mut bearing = heading_error(&state.pose, goal.target_pose.point);
            let mut direction = 1.0;
            if allow_reverse && bearing.abs() > PI / 2.0 {
                direction = -1.0;
                bearing = normalize_angle(bearing + PI);
            }
            let v = direction
                * path_speed(constraints.max_linear_velocity, 0.0, self.base.status.distance_to_goal, cfg.goal_tolerance, cfg.kp_linear, constraints)
                * heading_speed_scale(bearing, constraints);
            self.base.status.mode = "vector_pursuit_goal".into();
            return finalize(v, cfg.kp_angular.max(0.1) * bearing, constraints, &cfg, allow_reverse, "Moving to goal");
        }

        let axle_offset = if is_ackermann(constraints.steering_type) { constraints.rear_wheelbase.max(0.0) } else { 0.0 };
        let rear = Point::new(state.pose.point.x - axle_offset * yaw.cos(), state.pose.point.y - axle_offset * yaw.sin(), 0.0);
        let Some(proj) = self.cursor.project(&self.base.path.waypoints, rear, self.base.path_index) else {
            return VelocityCommand::invalid("no path");
        };
        self.base.path_index = proj.segment;

        let (pos_tol, ang_tol) = effective_tolerances(goal, &cfg);
        let passed_end = proj.beyond_end && proj.distance < 2.0 * pos_tol;
        if let Some(cmd) = self.base.arrival(state, goal, constraints, passed_end) {
            self.lookahead_point = None;
            return cmd;
        }

        let base_lookahead = cfg.lookahead_distance.max(1e-3);
        let kappa_path = curvature_at_projection(&self.base.path.waypoints, &self.cursor.cum, &proj).abs();
        let lookahead = ((base_lookahead + cfg.lookahead_time.max(0.0) * state.velocity.linear.abs())
            / (1.0 + CURVATURE_LOOKAHEAD_GAIN * kappa_path * base_lookahead))
            .clamp(0.5 * base_lookahead, 3.0 * base_lookahead);
        let mut s_target = proj.arc_length + lookahead;
        let mut stop_distance = self.base.status.distance_to_goal;
        if let Some(cusp) = cusp_after(&self.base.path.speeds, &self.cursor.cum, proj.arc_length) {
            s_target = s_target.min(cusp);
            stop_distance = stop_distance.min((cusp - proj.arc_length).max(0.0));
        }
        let (target, target_heading) = sample(&self.base.path.waypoints, &self.cursor.cum, s_target);
        self.lookahead_point = Some(target);

        let dx = target.x - rear.x;
        let dy = target.y - rear.y;
        let ld = dx.hypot(dy);
        let alpha = normalize_angle(dy.atan2(dx) - yaw);
        let mut direction = 1.0;
        let mut alpha_eff = alpha;
        if allow_reverse && alpha.abs() > PI / 2.0 {
            direction = -1.0;
            alpha_eff = normalize_angle(alpha + PI);
        }
        let kappa_t = if ld > 1e-6 { 2.0 * alpha.sin() / ld } else { 0.0 };
        // Rotation screw: the circle through the rear axle whose pole turns the
        // heading by dtheta over the chord L_d.
        let travel_heading = if direction > 0.0 { yaw } else { normalize_angle(yaw + PI) };
        let dtheta = normalize_angle(target_heading - travel_heading);
        let kappa_r = if ld > 1e-6 { direction * 2.0 * (0.5 * dtheta).sin() / ld } else { 0.0 };
        let k = self.vp_config.k_orientation.clamp(0.0, 1.0)
            * (proj.lateral_error.abs() / self.vp_config.lateral_scale.max(1e-6)).clamp(0.0, 1.0);
        let kappa = (1.0 - k) * kappa_t + k * kappa_r;

        self.base.status.cross_track_error = proj.lateral_error;
        self.base.status.heading_error = normalize_angle(proj.heading - yaw);
        self.base.status.goal_reached = false;

        if state.turn_first && can_turn_in_place(constraints.steering_type) && alpha_eff.abs() > ang_tol {
            self.base.status.mode = "turning".into();
            return finalize(0.0, cfg.kp_angular.max(0.1) * alpha_eff, constraints, &cfg, allow_reverse, "Turning to align");
        }

        let nominal = speed_cap(&self.base.path.speeds, &proj)
            .map_or(constraints.max_linear_velocity, |s| s.min(constraints.max_linear_velocity));
        let scale = heading_speed_scale(alpha_eff, constraints);
        let v = direction * path_speed(nominal, kappa, stop_distance, pos_tol, cfg.kp_linear, constraints) * scale;
        let mut omega = v * kappa;
        if can_turn_in_place(constraints.steering_type) {
            omega += (1.0 - scale) * cfg.kp_angular.max(0.1) * alpha_eff;
        }
        self.base.status.mode = if direction > 0.0 { "vector_pursuit".into() } else { "vector_pursuit_reverse".into() };
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
    }

    fn get_type(&self) -> &'static str {
        "vector_pursuit_follower"
    }

    fn base(&self) -> &ControllerBase {
        &self.base
    }

    fn base_mut(&mut self) -> &mut ControllerBase {
        &mut self.base
    }
}
