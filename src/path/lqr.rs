//! Discrete LQR on the kinematic lateral error model
//! `x = [e_lat, theta_e]`, `u = omega - v * kappa_path`:
//!   e(k+1)     = e - v dt theta_e
//!   theta(k+1) = theta_e - dt u
//! On a platform whose steering lags (`actuation_delay` tau > 0) the yaw
//! rate it turns at, `w = omega_actual - v * kappa_path`, is a third state:
//!   theta(k+1) = theta_e - dt w
//!   w(k+1)     = w + dt / tau (u - w)
//! The Riccati equation is iterated to convergence every tick at the
//! current linearisation speed; the commanded yaw rate is the path
//! curvature feedforward plus `-K x`.

use crate::controller::{Controller, ControllerBase, effective_tolerances};
use crate::core::kinematics::{
    path_to_body, axle_offset, bend_heading, braked_for_bends, followable,
    actuation_delay, can_turn_in_place, finalize, finalize_holonomic, heading_speed_scale, is_holonomic,
    path_speed, reverse_allowed,
};
use crate::core::math::normalize_angle;
use crate::core::path::{PathCursor, curvature_at_arc_length, curvature_at_projection, cusp_after, speed_cap};
use crate::types::{Goal, Path, RobotConstraints, RobotState, VelocityCommand, WorldConstraints};
use nalgebra::{Matrix2, Matrix3, SMatrix, SVector, Vector2, Vector3};

const DARE_MAX_ITERATIONS: usize = 20_000;
const DARE_TOLERANCE: f64 = 1e-10;
const MIN_LINEARISATION_SPEED: f64 = 0.1;

/// Iterate `P = A'PA - A'PB (R + B'PB)^-1 B'PA + Q` to a fixed point and
/// return the gain `K = (R + B'PB)^-1 B'PA`.
pub fn lqr_gain<const N: usize>(
    p0: Option<SMatrix<f64, N, N>>,
    a: &SMatrix<f64, N, N>,
    b: &SVector<f64, N>,
    q: &SMatrix<f64, N, N>,
    r: f64,
) -> (SVector<f64, N>, SMatrix<f64, N, N>) {
    let mut p = p0.unwrap_or(*q);
    let at = a.transpose();
    for _ in 0..DARE_MAX_ITERATIONS {
        let pb = p * b;
        let denom = r + b.dot(&pb);
        let denom = if denom.abs() < 1e-12 { 1e-12 } else { denom };
        let bt_p_a = (b.transpose() * p * a).transpose();
        let p_next = at * p * a - (at * pb) * bt_p_a.transpose() / denom + q;
        let delta = (p_next - p).norm();
        p = p_next;
        if delta < DARE_TOLERANCE * p.norm().max(1.0) {
            break;
        }
    }
    let pb = p * b;
    let denom = r + b.dot(&pb);
    let denom = if denom.abs() < 1e-12 { 1e-12 } else { denom };
    ((b.transpose() * p * a).transpose() / denom, p)
}

#[derive(Clone, Debug, Default)]
pub struct LqrFollower {
    pub base: ControllerBase,
    cursor: PathCursor,
    last_p: Option<Matrix2<f64>>,
    last_p_lagged: Option<Matrix3<f64>>,
    last_omega: f64,
    last_speed: f64,
}

impl LqrFollower {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Controller for LqrFollower {
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
        if self.base.path.waypoints.is_empty() {
            return VelocityCommand::invalid("no path");
        }
        if self.cursor.cum.len() != self.base.path.waypoints.len() {
            self.cursor.set_path(&self.base.path.waypoints);
            self.base.path_index = 0;
        }
        let cfg = self.base.config.clone();
        let allow_reverse = reverse_allowed(&cfg, state);
        let yaw = state.pose.rotation.to_euler().yaw;

        let (pos_tol, ang_tol) = effective_tolerances(goal, &cfg);
        let Some(proj) = self.cursor.project(&self.base.path.waypoints, state.pose.point, self.base.path_index) else {
            return VelocityCommand::invalid("no path");
        };
        let proj = self.cursor.past_cusp(&self.base.path.waypoints, &self.base.path.speeds, state.pose.point, proj, crate::controller::cusp_reach(pos_tol, state, constraints));
        self.base.path_index = proj.segment;
        let stop_distance = cusp_after(&self.base.path.speeds, &self.cursor.cum, proj.arc_length)
            .map_or(f64::INFINITY, |c| c - proj.arc_length);

        let length = self.cursor.cum.last().copied().unwrap_or(0.0);
        if let Some(cmd) = self.base.arrival_on_path(state, goal, constraints, &proj, length) {
            return cmd;
        }

        let theta_e = normalize_angle(proj.heading - yaw);
        let e_lat = proj.lateral_error;
        self.base.status.cross_track_error = e_lat;
        self.base.status.heading_error = theta_e;
        self.base.status.goal_reached = false;

        if is_holonomic(constraints.steering_type) {
            self.base.status.mode = "lqr_holonomic".into();
            let kappa_path = curvature_at_projection(&self.base.path.waypoints, &self.cursor.cum, &proj);
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
            let (vx, vy) = path_to_body(forward, -cfg.k_cross_track.max(1e-6) * e_lat, proj.heading, yaw);
            let omega = cfg.kp_angular.max(0.1) * theta_e;
            return finalize_holonomic(vx, vy, omega, constraints, &cfg, "LQR tracking");
        }

        // Reversing runs the same law on a virtual vehicle facing the other
        // way, driven forward.
        let reverse = match crate::core::path::direction_at(&self.base.path.speeds, &proj) {
            Some(d) => allow_reverse && d < 0,
            None => allow_reverse && theta_e.abs() > std::f64::consts::FRAC_PI_2,
        };
        let (theta_e, sign) = if reverse {
            (normalize_angle(proj.heading - yaw - std::f64::consts::PI), -1.0)
        } else {
            (theta_e, 1.0)
        };

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
        let trail = bend_heading(followable(kappa_path, constraints), sign * axle_offset(constraints));
        let theta_e = normalize_angle(theta_e - trail);
        let left = self.base.distance_left(&proj, length).min(stop_distance);
        let v = path_speed(nominal, kappa_path, left, pos_tol, cfg.kp_linear, constraints)
            * heading_speed_scale(theta_e, constraints);
        let v_now = if state.velocity_valid { state.velocity.linear } else { self.last_speed };
        let (waypoints, cum, s) = (&self.base.path.waypoints, &self.cursor.cum, proj.arc_length);
        let v = braked_for_bends(v, v_now, nominal, left, |d| curvature_at_arc_length(waypoints, cum, s + d), constraints);
        self.last_speed = v;
        let v_lin = v.abs().max(MIN_LINEARISATION_SPEED);
        let kappa_path = kappa_path / trail.cos();

        let (q_lat, q_head) = (cfg.k_cross_track.max(1e-6), cfg.k_heading.max(1e-6));
        let tau = 0.5 * actuation_delay(constraints);
        let u = if tau > 0.0 {
            let lag = (dt / tau).min(1.0);
            let a = Matrix3::new(1.0, -v_lin * dt, 0.0, 0.0, 1.0, -dt, 0.0, 0.0, 1.0 - lag);
            let b = Vector3::new(0.0, 0.0, lag);
            let q = Matrix3::new(q_lat, 0.0, 0.0, 0.0, q_head, 0.0, 0.0, 0.0, 0.0);
            let (k, p) = lqr_gain(self.last_p_lagged, &a, &b, &q, 1.0);
            self.last_p_lagged = Some(p);
            let turning = if state.velocity_valid { state.velocity.angular } else { self.last_omega };
            -k.dot(&Vector3::new(e_lat, theta_e, turning - v * kappa_path))
        } else {
            let a = Matrix2::new(1.0, -v_lin * dt, 0.0, 1.0);
            let b = Vector2::new(0.0, -dt);
            let q = Matrix2::new(q_lat, 0.0, 0.0, q_head);
            let (k, p) = lqr_gain(self.last_p, &a, &b, &q, 1.0);
            self.last_p = Some(p);
            -k.dot(&Vector2::new(e_lat, theta_e))
        };

        let omega = v * kappa_path + u;
        self.last_omega = omega;
        self.base.status.mode = if reverse { "lqr_reverse".into() } else { "lqr".into() };
        finalize(sign * v, omega, constraints, &cfg, allow_reverse, "LQR tracking")
    }

    fn set_path(&mut self, path: Path) {
        self.cursor.set_path(&path.waypoints);
        self.base.path = path;
        self.base.path_index = 0;
        self.base.status = Default::default();
        self.last_p = None;
        self.last_p_lagged = None;
        self.last_omega = 0.0;
        self.last_speed = 0.0;
    }

    fn reset(&mut self) {
        self.set_path(Path::default());
        self.base.status = Default::default();
    }

    fn get_type(&self) -> &'static str {
        "lqr_follower"
    }

    fn base(&self) -> &ControllerBase {
        &self.base
    }

    fn base_mut(&mut self) -> &mut ControllerBase {
        &mut self.base
    }
}
