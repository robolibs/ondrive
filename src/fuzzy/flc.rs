//! FLC — Mamdani fuzzy path follower. Signed cross-track error and heading
//! error are fuzzified into seven triangular terms each, a 7×7 additive
//! rule base maps them onto a steering term, and the height (weighted
//! average of term centres) defuzzifier produces a steering correction.
//! The correction is added to the path-curvature feedforward.

use crate::controller::{Controller, ControllerBase};
use crate::core::kinematics::{
    path_to_body, axle_offset, axle_point_offset, bend_heading, braked_for_bends, followable,
    can_turn_in_place, finalize, finalize_holonomic, heading_speed_scale, is_ackermann,
    is_holonomic, path_speed, steering_to_curvature,
};
use crate::core::math::normalize_angle;
use crate::core::path::{PathCursor, curvature_at_arc_length, curvature_at_projection, cusp_after, direction_at, speed_cap};
use crate::pred::mppi::prepare;
use crate::types::{Goal, Path, RobotConstraints, RobotState, VelocityCommand, WorldConstraints};
use datapod::Point;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Term {
    NL,
    NM,
    NS,
    ZE,
    PS,
    PM,
    PL,
}

impl Term {
    pub fn as_idx(self) -> usize {
        match self {
            Term::NL => 0,
            Term::NM => 1,
            Term::NS => 2,
            Term::ZE => 3,
            Term::PS => 4,
            Term::PM => 5,
            Term::PL => 6,
        }
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct FlcConfig {
    /// Cross-track error mapped onto the full `[-1, 1]` input range.
    pub cte_range: f64,
    /// Heading error mapped onto the full `[-1, 1]` input range.
    pub heading_range: f64,
    /// Output span: steering angle (Ackermann) or yaw rate (others).
    pub max_steering: f64,
    pub base_velocity: f64,
    pub min_velocity: f64,
}

impl Default for FlcConfig {
    fn default() -> Self {
        Self {
            cte_range: 2.0,
            heading_range: 1.57,
            max_steering: 0.7,
            base_velocity: 0.6,
            min_velocity: 0.2,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct FlcFollower {
    pub base: ControllerBase,
    pub flc_config: FlcConfig,
    cursor: PathCursor,
    last_speed: f64,
}

const CENTERS: [f64; 7] = [-1.0, -2.0 / 3.0, -1.0 / 3.0, 0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0];
const HALF_WIDTH: f64 = 1.0 / 3.0;

fn triangular_mf(x: f64, center: f64, half_width: f64) -> f64 {
    (1.0 - (x - center).abs() / half_width).max(0.0)
}

/// Memberships of a normalised input in the seven terms.
pub fn fuzzify(normalized: f64) -> [f64; 7] {
    let clamped = normalized.clamp(-1.0, 1.0);
    let mut out = [0.0_f64; 7];
    for (i, o) in out.iter_mut().enumerate() {
        *o = triangular_mf(clamped, CENTERS[i], HALF_WIDTH);
    }
    out
}

/// Additive rule base: output term = clamp(cte term + heading term).
pub fn rule_output_idx(cte_idx: usize, heading_idx: usize) -> usize {
    let signed = (cte_idx as i64 - 3) + (heading_idx as i64 - 3);
    (signed.clamp(-3, 3) + 3) as usize
}

fn defuzzify(activations: [f64; 7]) -> f64 {
    let mut num = 0.0;
    let mut denom = 0.0;
    for i in 0..7 {
        num += activations[i] * CENTERS[i];
        denom += activations[i];
    }
    if denom.abs() < 1e-9 { 0.0 } else { num / denom }
}

/// Normalised steering correction in `[-1, 1]` from normalised inputs.
/// Positive output means turn left. `cte_norm` is positive when the robot
/// is right of the path, `heading_norm` positive when the path heading is
/// left of the robot heading.
pub fn infer(cte_norm: f64, heading_norm: f64) -> f64 {
    let mu_cte = fuzzify(cte_norm);
    let mu_heading = fuzzify(heading_norm);
    let mut activations = [0.0_f64; 7];
    for (i, &mc) in mu_cte.iter().enumerate() {
        if mc <= 0.0 {
            continue;
        }
        for (j, &mh) in mu_heading.iter().enumerate() {
            if mh <= 0.0 {
                continue;
            }
            let strength = mc.min(mh);
            let out = rule_output_idx(i, j);
            if strength > activations[out] {
                activations[out] = strength;
            }
        }
    }
    defuzzify(activations)
}

impl FlcFollower {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_flc_config(cfg: FlcConfig) -> Self {
        Self {
            flc_config: cfg,
            ..Default::default()
        }
    }

    pub fn set_flc_config(&mut self, cfg: FlcConfig) {
        self.flc_config = cfg;
    }
}

impl Controller for FlcFollower {
    fn params(&self) -> serde_json::Value {
        serde_json::to_value(&self.flc_config).unwrap_or_default()
    }

    fn set_params(&mut self, params: &serde_json::Value) -> Result<(), String> {
        let cfg = crate::controller::merged(&self.flc_config, params)?;
        self.set_flc_config(cfg);
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
        let flc = self.flc_config.clone();
        let mut e_lat = prep.proj.lateral_error;
        let mut kappa_path = curvature_at_projection(&self.base.path.waypoints, &self.cursor.cum, &prep.proj);
        // A reversing leg is driven as a vehicle facing the other way,
        // forward; the body trails the bend by its pose's lead either way.
        let reverse = match direction_at(&self.base.path.speeds, &prep.proj) {
            Some(d) => prep.allow_reverse && d < 0,
            None => prep.allow_reverse && prep.epsi.abs() > std::f64::consts::FRAC_PI_2,
        };
        let sign = if reverse { -1.0 } else { 1.0 };
        let mut epsi = prep.epsi;
        let trail = bend_heading(followable(kappa_path, constraints), sign * axle_offset(constraints));
        let mut heading_trail = trail;
        // Reversing, the pose trails the axle the body turns about, so the
        // errors are read at that leading axle, which runs inside the bend.
        let axle = axle_offset(constraints);
        if reverse && axle > 0.0 {
            let yaw = state.pose.rotation.to_euler().yaw;
            let rear = Point::new(state.pose.point.x - axle * yaw.cos(), state.pose.point.y - axle * yaw.sin(), 0.0);
            if let Some(at_rear) = self.cursor.project_on_leg(&self.base.path.waypoints, &self.base.path.speeds, rear, prep.proj.segment, prep.proj.arc_length) {
                kappa_path = curvature_at_projection(&self.base.path.waypoints, &self.cursor.cum, &at_rear);
                e_lat = at_rear.lateral_error - axle_point_offset(followable(kappa_path, constraints), axle, 0.0);
                epsi = normalize_angle(at_rear.heading - yaw);
                heading_trail = 0.0;
            }
        }
        let facing = if reverse { normalize_angle(epsi - std::f64::consts::PI) } else { epsi };
        let theta_e = normalize_angle(facing - heading_trail);

        if state.turn_first && can_turn_in_place(constraints.steering_type) && theta_e.abs() > prep.ang_tol
        {
            self.base.status.mode = "turning".into();
            let omega = cfg.kp_angular.max(0.1) * theta_e;
            return finalize(0.0, omega, constraints, &cfg, prep.allow_reverse, "Turning to align");
        }

        let u = infer(
            -e_lat / flc.cte_range.max(1e-6),
            theta_e / flc.heading_range.max(1e-6),
        );

        let steer_ratio = u.abs().min(1.0);
        let nominal = speed_cap(&self.base.path.speeds, &prep.proj)
            .map_or(flc.base_velocity, |s| s.min(flc.base_velocity))
            .min(constraints.max_linear_velocity);
        let s0 = prep.proj.arc_length;
        let cusp = cusp_after(&self.base.path.speeds, &self.cursor.cum, s0).map_or(f64::INFINITY, |c| c - s0);
        let left = self.base.distance_left(&prep.proj, self.cursor.total_length()).min(cusp);
        let v_path = path_speed(nominal * (1.0 - 0.5 * steer_ratio), kappa_path, left, prep.pos_tol, cfg.kp_linear, constraints)
            .max(flc.min_velocity.min(nominal).min(path_speed(nominal, 0.0, left, prep.pos_tol, cfg.kp_linear, constraints)));
        let v_now = if state.velocity_valid { state.velocity.linear } else { self.last_speed };
        let (waypoints, cum) = (&self.base.path.waypoints, &self.cursor.cum);
        let v_path = braked_for_bends(v_path, v_now, nominal, left, |d| curvature_at_arc_length(waypoints, cum, s0 + d), constraints);
        self.last_speed = v_path;
        let kappa_path = kappa_path / trail.cos();

        if is_holonomic(constraints.steering_type) {
            self.base.status.mode = "flc_holonomic".into();
            let yaw = state.pose.rotation.to_euler().yaw;
            let (vx, vy) = path_to_body(v_path, -cfg.k_cross_track.max(1e-6) * e_lat, prep.proj.heading, yaw);
            let omega = cfg.kp_angular.max(0.1) * theta_e;
            return finalize_holonomic(vx, vy, omega, constraints, &cfg, "FLC tracking");
        }

        let v = v_path * heading_speed_scale(theta_e, constraints);

        let omega = if is_ackermann(constraints.steering_type) {
            let delta_fb = (u * flc.max_steering).clamp(
                -constraints.max_steering_angle.abs(),
                constraints.max_steering_angle.abs(),
            );
            let kappa = steering_to_curvature(delta_fb, constraints) + kappa_path;
            v * kappa
        } else {
            u * flc.max_steering + v * kappa_path
        };

        self.base.status.mode = if reverse { "flc_reverse".into() } else { "flc".into() };
        finalize(sign * v, omega, constraints, &cfg, prep.allow_reverse, "FLC tracking")
    }

    fn set_path(&mut self, path: Path) {
        self.cursor.set_path(&path.waypoints);
        self.base.path = path;
        self.base.path_index = 0;
    }

    fn reset(&mut self) {
        self.set_path(Path::default());
        self.base.status = Default::default();
    }

    fn get_type(&self) -> &'static str {
        "fuzzy_follower"
    }

    fn base(&self) -> &ControllerBase {
        &self.base
    }

    fn base_mut(&mut self) -> &mut ControllerBase {
        &mut self.base
    }
}
