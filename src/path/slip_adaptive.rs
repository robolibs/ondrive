//! Slip-adaptive path tracker for field vehicles (Lenain, Thuilot, Cariou &
//! Martinet 2006, "High accuracy path tracking for vehicles in presence of
//! sliding"). The kinematic bicycle at its unsteered (rear) axle is extended
//! with two sideslip angles, β_R at the rear axle and β_F at the steered
//! one, estimated each tick by a kinematic observer: β_R from the measured
//! lateral speed of the rear axle against its forward speed, and β_F from the
//! gap between the measured yaw rate and the one the steering (followed
//! through its rate limit) gives without sliding. With θ₂ = θ̃ + β_R and
//! α = 1 − c·y, the chained-form law
//! `δ = atan(tan β_R + L/cos β_R · [c·cos θ₂/α + cos³θ₂/α² ·
//!  (−K_d·α·tan θ₂ − K_p·y + c·α·tan² θ₂)]) − β_F`
//! drives the lateral error y to `y'' + K_d·y' + K_p·y = 0` over arc length,
//! with sliding cancelled. A machine without steered wheels follows the same
//! law as a curvature of its motion. Reversing runs the law on the machine
//! turned round, where the steered axle trails and the steering flips sign.

use crate::controller::{Controller, ControllerBase};
use crate::core::kinematics::{actuation_delay, finalize, finalize_holonomic, is_ackermann, is_holonomic, max_curvature, path_speed, wheelbase};
use crate::core::math::{normalize_angle, yaw_of};
use crate::core::path::{PathCursor, curvature_at_arc_length, curvature_at_projection, sample, speed_cap};
use crate::pred::mppi::{prepare, signed_speed_at};
use crate::types::{Goal, Path, RobotConstraints, RobotState, VelocityCommand, WorldConstraints};
use std::f64::consts::PI;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SlipAdaptiveConfig {
    /// Distance, metres, over which the lateral error settles (critically
    /// damped); 0 picks 2.5 wheelbases, at least 1.5 m and at least the
    /// distance covered over 1.5 full steering swings.
    pub settle_distance: f64,
    /// Time constant, seconds, of the sideslip observer's low-pass filter.
    pub observer_time: f64,
    /// Largest sideslip angle, radians, the observer reports.
    pub max_sideslip: f64,
    /// Forward speed, m/s, below which the observer holds its estimates.
    pub observer_speed: f64,
}

impl Default for SlipAdaptiveConfig {
    fn default() -> Self {
        Self { settle_distance: 0.0, observer_time: 0.5, max_sideslip: 0.35, observer_speed: 0.2 }
    }
}

#[derive(Clone, Debug, Default)]
pub struct SlipAdaptiveFollower {
    pub base: ControllerBase,
    pub slip_config: SlipAdaptiveConfig,
    cursor: PathCursor,
    /// Estimated sideslip angles `(β_R, β_F)`, radians.
    sideslip: (f64, f64),
    /// Steering angle the actuator is taken to hold, following the commands
    /// at the steering rate.
    steering: f64,
}

impl SlipAdaptiveFollower {
    pub fn new() -> Self {
        Self::default()
    }

    /// The estimated sideslip angles `(β_R, β_F)`, radians.
    pub fn sideslip(&self) -> (f64, f64) {
        self.sideslip
    }

    /// One observer step from `state`, travelling in `direction` (+1 ahead,
    /// −1 in reverse), over `dt` seconds.
    fn observe(&mut self, state: &RobotState, constraints: &RobotConstraints, direction: f64, dt: f64) {
        let cfg = &self.slip_config;
        let forward = direction * state.velocity.linear;
        if !state.velocity_valid || forward < cfg.observer_speed {
            return;
        }
        let rear = if is_ackermann(constraints.steering_type) { constraints.rear_wheelbase.max(0.0) } else { 0.0 };
        let lateral = direction * (state.velocity.lateral - state.velocity.angular * rear);
        let beta_r = lateral.atan2(forward);
        let beta_f = if is_ackermann(constraints.steering_type) {
            let l = wheelbase(constraints);
            let steer = direction * self.steering;
            (state.velocity.angular * l / forward + beta_r.tan()).atan() - steer
        } else {
            0.0
        };
        let gain = (dt / cfg.observer_time.max(dt)).min(1.0);
        let limit = cfg.max_sideslip.abs();
        let (br, bf) = self.sideslip;
        self.sideslip = ((br + gain * (beta_r - br)).clamp(-limit, limit), (bf + gain * (beta_f - bf)).clamp(-limit, limit));
    }
}

/// Heading error, or bearing of the path one settling distance ahead, in
/// radians, past which the chained-form law hands over to
/// pure pursuit of that point, over a further
/// `CHAINED_BLEND`: the law holds only well inside a quarter turn.
const CHAINED_LIMIT: f64 = 0.6;
const CHAINED_BLEND: f64 = 0.4;
/// Share of the platform's tightest curvature beyond which the path ahead
/// hands over to pure pursuit, which cuts what it cannot follow, fully past
/// `TIGHT_BLEND` more.
const TIGHT_LIMIT: f64 = 0.8;
const TIGHT_BLEND: f64 = 0.4;
/// Metres between the path curvatures read ahead for the speed.
const CURVATURE_SCAN: f64 = 0.5;

/// Curvature of the motion that brings lateral error `y` and heading error
/// `theta2` (both measured against the slid direction of travel) onto a
/// path of curvature `c`, with gains `(k_p, k_d)` per metre.
fn chained_curvature(y: f64, theta2: f64, c: f64, (k_p, k_d): (f64, f64)) -> f64 {
    let alpha = (1.0 - c * y).max(0.1);
    let t = theta2.clamp(-1.4, 1.4).tan();
    let cos = theta2.clamp(-1.4, 1.4).cos();
    c * cos / alpha + cos.powi(3) / alpha.powi(2) * (-k_d * alpha * t - k_p * y + c * alpha * t * t)
}

impl Controller for SlipAdaptiveFollower {
    fn params(&self) -> serde_json::Value {
        serde_json::to_value(&self.slip_config).unwrap_or_default()
    }

    fn set_params(&mut self, params: &serde_json::Value) -> Result<(), String> {
        self.slip_config = crate::controller::merged(&self.slip_config, params)?;
        Ok(())
    }

    fn compute_control(
        &mut self,
        state: &RobotState,
        goal: &Goal,
        constraints: &RobotConstraints,
        dt: f64,
        _world: Option<&WorldConstraints>,
    ) -> VelocityCommand {
        let prep = match prepare(&mut self.base, &mut self.cursor, state, goal, constraints, dt) {
            Ok(p) => p,
            Err(cmd) => return cmd,
        };
        let cfg = self.base.config.clone();
        let ackermann = is_ackermann(constraints.steering_type);
        let l = wheelbase(constraints);
        let reverse = prep.allow_reverse
            && signed_speed_at(&self.base.path.speeds, &self.cursor.cum, prep.proj.arc_length).is_some_and(|v| v < 0.0);
        let direction = if reverse { -1.0 } else { 1.0 };

        let here = curvature_at_projection(&self.base.path.waypoints, &self.cursor.cum, &prep.proj);
        let nominal = speed_cap(&self.base.path.speeds, &prep.proj)
            .map_or(constraints.max_linear_velocity, |s| s.abs().min(constraints.max_linear_velocity));
        let a_max = constraints.max_linear_acceleration.abs().max(0.1);
        let braking = nominal * nominal / (2.0 * a_max) + nominal * actuation_delay(constraints);
        let sharpest = (0..=(braking / CURVATURE_SCAN).ceil() as usize)
            .map(|i| curvature_at_arc_length(&self.base.path.waypoints, &self.cursor.cum, prep.proj.arc_length + i as f64 * CURVATURE_SCAN))
            .fold(here, |a, k| if k.abs() > a.abs() { k } else { a });
        let speed = path_speed(nominal, sharpest, self.base.status.distance_to_goal, prep.pos_tol, cfg.kp_linear, constraints);
        let preview = speed * actuation_delay(constraints);
        let kappa = if preview > 1e-3 {
            curvature_at_arc_length(&self.base.path.waypoints, &self.cursor.cum, prep.proj.arc_length + preview)
        } else {
            here
        };

        if is_holonomic(constraints.steering_type) {
            let yaw = yaw_of(&state.pose);
            let theta = normalize_angle(prep.proj.heading - yaw);
            let (s, c) = theta.sin_cos();
            let lateral = -cfg.k_cross_track * prep.proj.lateral_error;
            let (vx, vy) = (speed * c - lateral * s, speed * s + lateral * c);
            self.base.status.mode = "slip_adaptive_holonomic".into();
            return finalize_holonomic(direction * vx, vy, cfg.kp_angular.max(0.1) * theta, constraints, &cfg, "Slip-adaptive tracking");
        }

        self.observe(state, constraints, direction, dt);
        let (beta_r, beta_f) = self.sideslip;
        let yaw = yaw_of(&state.pose) + if reverse { PI } else { 0.0 };
        let rear = if ackermann { constraints.rear_wheelbase.max(0.0) } else { 0.0 };
        let theta = normalize_angle(yaw - prep.proj.heading);
        let y = prep.proj.lateral_error - direction * rear * theta.sin();
        let theta2 = normalize_angle(theta + beta_r);

        let steer_time = if ackermann && constraints.max_steering_rate > 0.0 {
            constraints.max_steering_angle.abs() / constraints.max_steering_rate
        } else {
            0.0
        };
        let settle = if self.slip_config.settle_distance > 0.0 {
            self.slip_config.settle_distance
        } else {
            (2.5 * l).max(1.5).max(1.5 * speed * steer_time)
        };
        let omega_n = 2.0 / settle;
        let gains = (omega_n * omega_n, 2.0 * omega_n);
        let chained = chained_curvature(y, theta2, kappa, gains);
        let (target, _) = sample(&self.base.path.waypoints, &self.cursor.cum, prep.proj.arc_length + settle);
        let (s, c) = (yaw + beta_r).sin_cos();
        let (rx, ry) = (state.pose.point.x - direction * rear * yaw.cos(), state.pose.point.y - direction * rear * yaw.sin());
        let (dx, dy) = (target.x - rx, target.y - ry);
        let along = c * dx + s * dy;
        let across = -s * dx + c * dy;
        let pursuit = 2.0 * across / (along * along + across * across).max(1e-6);
        let bearing = across.atan2(along).abs().max(theta2.abs());
        let k_max = max_curvature(constraints);
        let tight = if k_max.is_finite() && k_max > 0.0 { ((kappa.abs() / k_max - TIGHT_LIMIT) / TIGHT_BLEND).clamp(0.0, 1.0) } else { 0.0 };
        let blend = ((bearing - CHAINED_LIMIT) / CHAINED_BLEND).clamp(0.0, 1.0).max(tight);
        let curvature = (1.0 - blend) * chained + blend * pursuit;
        self.base.status.cross_track_error = y;
        self.base.status.heading_error = theta;

        let v = direction * speed;
        let omega = if ackermann {
            let steer = ((beta_r.tan() + l / beta_r.cos() * curvature).atan() - beta_f).clamp(-PI / 2.0 + 0.05, PI / 2.0 - 0.05);
            let steer = direction * steer;
            let rate = constraints.max_steering_rate;
            self.steering = if rate > 0.0 { self.steering + (steer - self.steering).clamp(-rate * dt, rate * dt) } else { steer };
            v * steer.tan() / l
        } else {
            speed / beta_r.cos() * curvature
        };
        self.base.status.mode = if reverse { "slip_adaptive_reverse".into() } else { "slip_adaptive".into() };
        finalize(v, omega, constraints, &cfg, prep.allow_reverse, "Slip-adaptive tracking")
    }

    fn set_path(&mut self, path: Path) {
        self.cursor.set_path(&path.waypoints);
        self.base.path = path;
        self.base.path_index = 0;
        self.base.status = Default::default();
    }

    fn reset(&mut self) {
        self.set_path(Path::default());
        self.sideslip = (0.0, 0.0);
        self.steering = 0.0;
    }

    fn get_type(&self) -> &'static str {
        "slip_adaptive_follower"
    }

    fn base(&self) -> &ControllerBase {
        &self.base
    }

    fn base_mut(&mut self) -> &mut ControllerBase {
        &mut self.base
    }
}
