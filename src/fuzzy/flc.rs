//! FLC — Mamdani fuzzy path follower. Signed cross-track error and heading
//! error are fuzzified into seven triangular terms each, a 7×7 additive
//! rule base maps them onto a steering term, and the height (weighted
//! average of term centres) defuzzifier produces a steering correction.
//! The correction is added to the path-curvature feedforward.

use crate::controller::{Controller, ControllerBase};
use crate::core::kinematics::{
    can_turn_in_place, finalize, finalize_holonomic, heading_speed_scale, is_ackermann,
    is_holonomic, path_speed, steering_to_curvature,
};
use crate::core::path::{PathCursor, curvature_at_projection, speed_cap};
use crate::pred::mppi::prepare;
use crate::types::{Goal, Path, RobotConstraints, RobotState, VelocityCommand, WorldConstraints};

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

#[derive(Clone, Debug)]
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
        let e_lat = prep.proj.lateral_error;
        let theta_e = prep.epsi;

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
        let kappa_path = curvature_at_projection(&self.base.path.waypoints, &self.cursor.cum, &prep.proj);

        let steer_ratio = u.abs().min(1.0);
        let nominal = speed_cap(&self.base.path.speeds, &prep.proj)
            .map_or(flc.base_velocity, |s| s.min(flc.base_velocity))
            .min(constraints.max_linear_velocity);
        let v = path_speed(
            nominal * (1.0 - 0.5 * steer_ratio),
            kappa_path,
            self.base.status.distance_to_goal,
            prep.pos_tol,
            cfg.kp_linear,
            constraints,
        )
        .max(flc.min_velocity.min(nominal))
            * heading_speed_scale(theta_e, constraints);

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

        if is_holonomic(constraints.steering_type) {
            self.base.status.mode = "flc_holonomic".into();
            let lateral = -cfg.k_cross_track.max(1e-6) * e_lat;
            return finalize_holonomic(v, lateral, omega, constraints, &cfg, "FLC tracking");
        }

        self.base.status.mode = "flc".into();
        finalize(v, omega, constraints, &cfg, prep.allow_reverse, "FLC tracking")
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
