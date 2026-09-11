//! Discrete LQR on the kinematic lateral error model
//! `x = [e_lat, theta_e]`, `u = omega - v * kappa_path`:
//!   e(k+1)     = e - v dt theta_e
//!   theta(k+1) = theta_e - dt u
//! The Riccati equation is iterated to convergence every tick at the
//! current linearisation speed; the commanded yaw rate is the path
//! curvature feedforward plus `-K x`.

use crate::controller::{Controller, ControllerBase, effective_tolerances};
use crate::core::kinematics::{
    can_turn_in_place, finalize, heading_speed_scale, path_speed, reverse_allowed,
};
use crate::core::math::normalize_angle;
use crate::core::path::{cumulative_lengths, curvature_at_projection, project, speed_cap};
use crate::types::{Goal, Path, RobotConstraints, RobotState, VelocityCommand, WorldConstraints};
use nalgebra::{Matrix2, Vector2};

const DARE_MAX_ITERATIONS: usize = 20_000;
const DARE_TOLERANCE: f64 = 1e-10;
const MIN_LINEARISATION_SPEED: f64 = 0.1;
const SEARCH_WINDOW: usize = 64;

/// Iterate `P = A'PA - A'PB (R + B'PB)^-1 B'PA + Q` to a fixed point and
/// return the gain `K = (R + B'PB)^-1 B'PA`.
pub fn lqr_gain(
    p0: Option<Matrix2<f64>>,
    a: &Matrix2<f64>,
    b: &Vector2<f64>,
    q: &Matrix2<f64>,
    r: f64,
) -> (Vector2<f64>, Matrix2<f64>) {
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
    cum: Vec<f64>,
    started: bool,
    last_p: Option<Matrix2<f64>>,
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
        if self.cum.len() != self.base.path.waypoints.len() {
            self.cum = cumulative_lengths(&self.base.path.waypoints);
            self.started = false;
            self.base.path_index = 0;
        }
        let cfg = self.base.config.clone();
        let allow_reverse = reverse_allowed(&cfg, state);
        let yaw = state.pose.rotation.to_euler().yaw;

        let window = if self.started {
            SEARCH_WINDOW
        } else {
            usize::MAX
        };
        let Some(proj) = project(
            &self.base.path.waypoints,
            &self.cum,
            state.pose.point,
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
            return cmd;
        }

        let theta_e = normalize_angle(proj.heading - yaw);
        let e_lat = proj.lateral_error;
        self.base.status.cross_track_error = e_lat;
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
        let v = path_speed(
            nominal,
            kappa_path,
            self.base.status.distance_to_goal,
            pos_tol,
            cfg.kp_linear,
            constraints,
        ) * heading_speed_scale(theta_e, constraints);
        let v_lin = v.abs().max(MIN_LINEARISATION_SPEED);

        let a = Matrix2::new(1.0, -v_lin * dt, 0.0, 1.0);
        let b = Vector2::new(0.0, -dt);
        let q = Matrix2::new(cfg.k_cross_track.max(1e-6), 0.0, 0.0, cfg.k_heading.max(1e-6));
        let (k, p) = lqr_gain(self.last_p, &a, &b, &q, 1.0);
        self.last_p = Some(p);
        let x = Vector2::new(e_lat, theta_e);
        let u = -k.dot(&x);

        let omega = v * kappa_path + u;
        self.base.status.mode = "lqr".into();
        finalize(v, omega, constraints, &cfg, allow_reverse, "LQR tracking")
    }

    fn set_path(&mut self, path: Path) {
        self.cum = cumulative_lengths(&path.waypoints);
        self.base.path = path;
        self.base.path_index = 0;
        self.base.status = Default::default();
        self.started = false;
        self.last_p = None;
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
