//! FLC — Mamdani-style fuzzy logic controller for path following.
//!
//! Two inputs (cross-track error, heading error) are fuzzified into seven
//! linguistic terms each (NL, NM, NS, ZE, PS, PM, PL) via triangular
//! membership functions. A 7×7 rule base maps inputs to a steering term;
//! centroid defuzzification produces the steering output. Linear in the
//! middle of each range, saturating near the edges.

#![allow(clippy::needless_range_loop)]

use crate::controller::{Controller, ControllerBase, is_goal_reached};
use crate::core::math::normalize_angle;
use crate::types::{
    Goal, RobotConstraints, RobotState, SteeringType, VelocityCommand, WorldConstraints,
};

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
    pub cte_range: f64,
    pub heading_range: f64,
    pub max_steering: f64,
    pub base_velocity: f64,
    pub min_velocity: f64,
    pub rule_weight: f64,
    pub use_cte_derivative: bool,
}

impl Default for FlcConfig {
    fn default() -> Self {
        Self {
            cte_range: 2.0,
            heading_range: 1.57,
            max_steering: 0.7,
            base_velocity: 0.6,
            min_velocity: 0.2,
            rule_weight: 1.0,
            use_cte_derivative: false,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct FlcFollower {
    pub base: ControllerBase,
    pub flc_config: FlcConfig,
    prev_cte: f64,
}

impl FlcFollower {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_flc_config(cfg: FlcConfig) -> Self {
        Self {
            base: ControllerBase::default(),
            flc_config: cfg,
            prev_cte: 0.0,
        }
    }

    pub fn set_flc_config(&mut self, cfg: FlcConfig) {
        self.flc_config = cfg;
    }
}

/// Triangular membership: peak at `center`, zero at `center ± half_width`.
fn triangular_mf(x: f64, center: f64, half_width: f64) -> f64 {
    (1.0 - (x - center).abs() / half_width).max(0.0)
}

/// Fuzzify a normalised value (clamped to [-1, 1]) into memberships for
/// seven overlapping triangular terms with centres at
/// {-1, -2/3, -1/3, 0, 1/3, 2/3, 1} and half-width 1/3.
fn fuzzify(normalized: f64) -> [f64; 7] {
    const CENTERS: [f64; 7] = [-1.0, -2.0 / 3.0, -1.0 / 3.0, 0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0];
    const HALF_WIDTH: f64 = 1.0 / 3.0;
    let clamped = normalized.clamp(-1.0, 1.0);
    let mut out = [0.0_f64; 7];
    for i in 0..7 {
        out[i] = triangular_mf(clamped, CENTERS[i], HALF_WIDTH);
    }
    out
}

/// Rule base: output-term index per (cte_idx, heading_idx) cell.
///
/// Additive rule: signed_steering = signed_cte + signed_heading, clamped.
/// Convention: positive CTE means the robot is right of the path; positive
/// heading error means the robot needs to turn left to align. Both correct
/// toward the path via positive (left) steering.
fn rule_output_idx(cte_idx: usize, heading_idx: usize) -> usize {
    let signed = (cte_idx as i64 - 3) + (heading_idx as i64 - 3);
    (signed.clamp(-3, 3) + 3) as usize
}

fn defuzzify(activations: [f64; 7], output_range: f64) -> f64 {
    // Output-term centers mirror the input ones on [-output_range, +output_range].
    const CENTERS_NORM: [f64; 7] = [-1.0, -2.0 / 3.0, -1.0 / 3.0, 0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0];
    let mut num = 0.0;
    let mut denom = 0.0;
    for i in 0..7 {
        num += activations[i] * CENTERS_NORM[i] * output_range;
        denom += activations[i];
    }
    if denom.abs() < 1e-9 { 0.0 } else { num / denom }
}

fn find_closest_path_point(
    waypoints: &[datapod::Pose],
    from_idx: usize,
    current_pose: &datapod::Pose,
) -> (usize, f64, f64) {
    // Returns (closest_idx, signed_cte, path_heading)
    let search_start = from_idx.saturating_sub(5);
    let search_end = (from_idx + 20).min(waypoints.len());

    let mut min_dist = f64::MAX;
    let mut closest_idx = from_idx;
    for i in search_start..search_end {
        let d = current_pose.point.distance_to(waypoints[i].point);
        if d < min_dist {
            min_dist = d;
            closest_idx = i;
        }
    }

    let closest = waypoints[closest_idx].point;
    let path_heading = if closest_idx + 1 < waypoints.len() {
        let next = waypoints[closest_idx + 1].point;
        (next.y - closest.y).atan2(next.x - closest.x)
    } else if closest_idx > 0 {
        let prev = waypoints[closest_idx - 1].point;
        (closest.y - prev.y).atan2(closest.x - prev.x)
    } else {
        0.0
    };

    let dx = current_pose.point.x - closest.x;
    let dy = current_pose.point.y - closest.y;
    let mut cte = (dx * dx + dy * dy).sqrt();
    let sign_check = dy * path_heading.cos() - dx * path_heading.sin();
    if sign_check > 0.0 {
        cte = -cte;
    }
    (closest_idx, cte, path_heading)
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
        let cfg = self.base.config.clone();
        let flc = self.flc_config.clone();

        let is_diff = matches!(
            constraints.steering_type,
            SteeringType::Differential | SteeringType::SkidSteer
        );

        if self.base.path.waypoints.is_empty() {
            return VelocityCommand::invalid("FLC: no path set");
        }

        let (reached, dist_to_goal, yaw_diff) = is_goal_reached(
            &state.pose,
            &goal.target_pose,
            cfg.goal_tolerance,
            cfg.angular_tolerance,
        );
        self.base.status.distance_to_goal = dist_to_goal;
        self.base.status.heading_error = yaw_diff;
        if reached {
            self.base.status.goal_reached = true;
            self.base.status.mode = "stopped".into();
            return VelocityCommand {
                valid: true,
                status_message: "Goal reached".into(),
                ..VelocityCommand::default()
            };
        }

        let (closest_idx, cte, path_heading) =
            find_closest_path_point(&self.base.path.waypoints, self.base.path_index, &state.pose);
        self.base.path_index = closest_idx;
        let heading_err = normalize_angle(path_heading - state.pose.rotation.to_euler().yaw);

        // Optional derivative-of-CTE input: skipped by default, but when
        // enabled it can be fuzzified alongside and used to shape rules.
        // Here we just keep the running state for the caller to inspect
        // via `self.prev_cte` — the rule table itself uses CTE and heading.
        let _d_cte = if flc.use_cte_derivative {
            (cte - self.prev_cte) / dt.max(1e-6)
        } else {
            0.0
        };
        self.prev_cte = cte;

        let cte_range = flc.cte_range.max(1e-6);
        let heading_range = flc.heading_range.max(1e-6);
        let mu_cte = fuzzify(cte / cte_range);
        let mu_heading = fuzzify(heading_err / heading_range);

        // For each rule (cte_idx, heading_idx), firing strength = min(μ_cte, μ_heading).
        // Aggregate per output term by taking the max firing strength.
        let mut output_activations = [0.0_f64; 7];
        for i in 0..7 {
            if mu_cte[i] <= 0.0 {
                continue;
            }
            for j in 0..7 {
                if mu_heading[j] <= 0.0 {
                    continue;
                }
                let strength = mu_cte[i].min(mu_heading[j]) * flc.rule_weight;
                let out_idx = rule_output_idx(i, j);
                if strength > output_activations[out_idx] {
                    output_activations[out_idx] = strength;
                }
            }
        }

        let steering_signed = defuzzify(output_activations, flc.max_steering);

        // Slow down when steering hard.
        let steer_ratio = (steering_signed.abs() / flc.max_steering.max(1e-6)).clamp(0.0, 1.0);
        let velocity = (flc.base_velocity * (1.0 - 0.5 * steer_ratio)).max(flc.min_velocity);

        let steering_output = if is_diff {
            steering_signed.clamp(
                -constraints.max_angular_velocity,
                constraints.max_angular_velocity,
            )
        } else {
            steering_signed.clamp(
                -constraints.max_steering_angle,
                constraints.max_steering_angle,
            )
        };

        let angular_velocity = if is_diff {
            steering_output
        } else {
            // Ackermann: convert steering angle to yaw-rate via v·tan(δ)/L.
            let lf = if constraints.wheelbase > 0.0 {
                constraints.wheelbase
            } else {
                1.0
            };
            (velocity * steering_output.tan() / lf).clamp(
                -constraints.max_angular_velocity,
                constraints.max_angular_velocity,
            )
        };

        self.base.status.distance_to_goal = dist_to_goal;
        self.base.status.cross_track_error = cte.abs();
        self.base.status.heading_error = heading_err;
        self.base.status.goal_reached = false;
        self.base.status.mode = "flc".into();

        VelocityCommand {
            valid: true,
            status_message: "FLC tracking".into(),
            linear_velocity: velocity.clamp(
                if cfg.allow_reverse {
                    constraints.min_linear_velocity
                } else {
                    0.0
                },
                constraints.max_linear_velocity,
            ),
            angular_velocity,
            ..VelocityCommand::default()
        }
    }

    fn reset(&mut self) {
        self.base.path.waypoints.clear();
        self.base.path_index = 0;
        self.base.status = Default::default();
        self.prev_cte = 0.0;
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
