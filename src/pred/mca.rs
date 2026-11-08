//! MCA — risk-aware MPPI (DRA-MPPI, Trevisan et al. 2025).
//!
//! Each rollout point's collision probability is the share of a fixed
//! quasi-random set of obstacle positions, drawn from the obstacle's
//! Gaussian-mixture prediction at that point's time, that falls within the
//! combined radius of an obstacle and one of the discs covering the robot.
//! Obstacles combine as independent events. The cost adds a soft term
//! proportional to that probability and, above `risk_threshold`, a hard term
//! that is larger the sooner the collision would come, so a batch whose every
//! rollout is at risk still prefers the one that brakes or turns away. The
//! reference speed inside the rollouts drops as the plan's imminent risk
//! rises.
//!
//! Without obstacles this is exactly MPPI.

use crate::controller::{Controller, ControllerBase};
use crate::core::obstacles::{CollisionChecker, ModeAt, footprint_discs, modes_at};

use crate::pred::mppi::{MppiConfig, MppiFollower, detour_relax};
use crate::types::{
    ControllerConfig, ControllerStatus, DebugFrame, DebugLevel, Goal, Obstacle, Path, RobotConstraints, RobotState,
    VelocityCommand, WorldConstraints,
};
use std::f64::consts::PI;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct McaConfig {
    pub horizon_steps: usize,
    pub dt: f64,

    pub num_samples: usize,
    /// Quasi-random obstacle positions per mode, the same set every tick.
    pub num_mc_samples: usize,
    pub temperature: f64,
    pub steering_noise: f64,
    pub acceleration_noise: f64,

    pub weight_cte: f64,
    pub weight_epsi: f64,
    pub weight_vel: f64,
    pub weight_steering: f64,
    pub weight_acceleration: f64,
    /// Cost per metre of path a rollout leaves undone within the horizon.
    pub weight_progress: f64,

    /// Cost per second of a collision probability of 1.
    pub weight_soft_risk: f64,
    /// Cost per second above `risk_threshold`, for a collision on the first
    /// step, falling linearly to zero at the horizon.
    pub weight_hard_risk: f64,
    pub risk_threshold: f64,

    pub min_velocity_scale: f64,
    pub risk_slowdown_gain: f64,
    pub robot_radius_margin: f64,

    pub ref_velocity: f64,
    pub turn_first_activation_deg: f64,
    pub turn_first_release_deg: f64,

    pub decel_distance: f64,
}

impl Default for McaConfig {
    fn default() -> Self {
        Self {
            horizon_steps: 20,
            dt: 0.2,
            num_samples: 400,
            num_mc_samples: 64,
            temperature: 1.0,
            steering_noise: 0.5,
            acceleration_noise: 0.4,
            weight_cte: 50.0,
            weight_epsi: 50.0,
            weight_vel: 50.0,
            weight_steering: 5.0,
            weight_acceleration: 5.0,
            weight_progress: 30.0,
            weight_soft_risk: 2000.0,
            weight_hard_risk: 2e4,
            risk_threshold: 0.05,
            min_velocity_scale: 0.4,
            risk_slowdown_gain: 2.0,
            robot_radius_margin: 0.1,
            ref_velocity: 1.0,
            turn_first_activation_deg: 60.0,
            turn_first_release_deg: 15.0,
            decel_distance: 2.0,
        }
    }
}

impl McaConfig {
    fn to_mppi(&self) -> MppiConfig {
        MppiConfig {
            horizon_steps: self.horizon_steps,
            dt: self.dt,
            num_samples: self.num_samples,
            temperature: self.temperature,
            steering_noise: self.steering_noise,
            acceleration_noise: self.acceleration_noise,
            weight_cte: self.weight_cte,
            weight_epsi: self.weight_epsi,
            weight_vel: self.weight_vel,
            weight_steering: self.weight_steering,
            weight_acceleration: self.weight_acceleration,
            weight_progress: self.weight_progress,
            ref_velocity: self.ref_velocity,
            turn_first_activation_deg: self.turn_first_activation_deg,
            turn_first_release_deg: self.turn_first_release_deg,
            decel_distance: self.decel_distance,
            ..MppiConfig::default()
        }
    }
}

/// Horizon steps whose collision probability drives the risk slowdown.
const IMMINENT_STEPS: usize = 5;

/// `m` standard-normal points from the Halton sequence (bases 2 and 3)
/// through the Box–Muller transform.
fn normal_set(m: usize) -> Vec<(f64, f64)> {
    let halton = |mut i: usize, base: usize| {
        let (mut f, mut r) = (1.0, 0.0);
        while i > 0 {
            f /= base as f64;
            r += f * (i % base) as f64;
            i /= base;
        }
        r
    };
    (1..=m.max(1))
        .map(|i| {
            let (u, v) = (halton(i, 2), halton(i, 3));
            let r = (-2.0 * u.ln()).sqrt();
            (r * (2.0 * PI * v).cos(), r * (2.0 * PI * v).sin())
        })
        .collect()
}

#[derive(Clone, Debug)]
pub struct McaFollower {
    pub mca_config: McaConfig,
    mppi: MppiFollower,
    normals: Vec<(f64, f64)>,
}

impl Default for McaFollower {
    fn default() -> Self {
        Self::with_mca_config(McaConfig::default())
    }
}

impl McaFollower {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_mca_config(cfg: McaConfig) -> Self {
        Self { mppi: MppiFollower::with_mppi_config(cfg.to_mppi()), normals: normal_set(cfg.num_mc_samples), mca_config: cfg }
    }

    pub fn with_seed(cfg: McaConfig, seed: u64) -> Self {
        Self {
            mppi: MppiFollower::with_seed(cfg.to_mppi(), seed.wrapping_add(1)),
            normals: normal_set(cfg.num_mc_samples),
            mca_config: cfg,
        }
    }

    pub fn set_mca_config(&mut self, cfg: McaConfig) {
        self.mppi.set_mppi_config(cfg.to_mppi());
        if cfg.num_mc_samples != self.normals.len() {
            self.normals = normal_set(cfg.num_mc_samples);
        }
        self.mca_config = cfg;
    }

    pub fn predicted_trajectory(&self) -> &[datapod::Point] {
        self.mppi.predicted_trajectory()
    }
}

/// Probability that a disc of `radius` at `(x, y)` meets an obstacle mode:
/// the share of `normals`, scaled by the mode's spread, inside the combined
/// radius.
fn hit_probability(m: &ModeAt, x: f64, y: f64, radius: f64, normals: &[(f64, f64)]) -> f64 {
    let reach = radius + m.radius;
    let d = (m.x - x).hypot(m.y - y);
    let spread = m.std_x.max(m.std_y);
    if d > reach + 4.0 * spread {
        return 0.0;
    }
    if spread < 1e-6 {
        return if d < reach { 1.0 } else { 0.0 };
    }
    let r2 = reach * reach;
    let hits = normals
        .iter()
        .filter(|(zx, zy)| (m.x + m.std_x * zx - x).powi(2) + (m.y + m.std_y * zy - y).powi(2) < r2)
        .count();
    hits as f64 / normals.len() as f64
}

impl Controller for McaFollower {
    fn params(&self) -> serde_json::Value {
        serde_json::to_value(&self.mca_config).unwrap_or_default()
    }

    fn set_params(&mut self, params: &serde_json::Value) -> Result<(), String> {
        let cfg = crate::controller::merged(&self.mca_config, params)?;
        self.set_mca_config(cfg);
        Ok(())
    }

    fn set_seed(&mut self, seed: u64) {
        self.mppi.set_seed(seed.wrapping_add(1));
    }

    fn set_debug(&mut self, level: DebugLevel) {
        self.mppi.set_debug(level);
    }

    fn debug_frame(&self) -> DebugFrame {
        self.mppi.debug_frame()
    }

    fn compute_control(
        &mut self,
        state: &RobotState,
        goal: &Goal,
        constraints: &RobotConstraints,
        dt: f64,
        world: Option<&WorldConstraints>,
    ) -> VelocityCommand {
        self.mppi.clear_guides();
        let obstacles: Vec<&Obstacle> = world
            .map(|w| {
                w.obstacles
                    .iter()
                    .filter(|o| o.modes.iter().any(|m| m.weight.is_finite() && m.weight > 0.0 && !m.mean_x.is_empty() && !m.mean_y.is_empty()))
                    .collect()
            })
            .unwrap_or_default();
        let has_grid = world.is_some_and(|w| w.grid.is_some());
        if obstacles.is_empty() && !has_grid {
            return self.mppi.step_with(state, goal, constraints, dt, &|_, _, _, _| 0.0, &|_, _| (1.0, 0.0), 1.0, "MCA tracking", "mca_tracking");
        }

        let (n, step) = self.mppi.plan_for(constraints);
        let (n, step) = (n.max(1), step.max(1e-3));
        let cfg = &self.mca_config;
        let margin = cfg.robot_radius_margin.max(0.0);
        let discs: Vec<(f64, f64)> = footprint_discs(constraints).into_iter().map(|(x, r)| (x, r + margin)).collect();
        let modes: Vec<Vec<Vec<ModeAt>>> = (0..n)
            .map(|i| obstacles.iter().map(|o| modes_at(o, (i + 1) as f64 * step, step).collect()).collect())
            .collect();
        let normals = &self.normals;
        let probability = |i: usize, x: f64, y: f64, yaw: f64| -> f64 {
            let (s, c) = yaw.sin_cos();
            let mut survive = 1.0;
            for obstacle in &modes[i.min(n - 1)] {
                let (mut p, mut total) = (0.0, 0.0);
                for m in obstacle {
                    let hit = discs
                        .iter()
                        .map(|(dx, r)| hit_probability(m, x + c * dx, y + s * dx, *r, normals))
                        .fold(0.0_f64, f64::max);
                    p += m.weight * hit;
                    total += m.weight;
                }
                if total > 0.0 {
                    survive *= 1.0 - p / total;
                }
            }
            1.0 - survive
        };
        let checker = CollisionChecker::new(world, constraints, margin).with_step(step);
        let (soft, hard, threshold) = (cfg.weight_soft_risk, cfg.weight_hard_risk, cfg.risk_threshold);
        let extra = |i: usize, x: f64, y: f64, yaw: f64| -> f64 {
            let mut p = probability(i, x, y, yaw);
            if has_grid && checker.grid_clearance(x, y, yaw) < 0.0 {
                p = 1.0;
            }
            let urgency = (n - i.min(n)) as f64 / n as f64;
            soft * p + if p > threshold { hard * urgency } else { 0.0 }
        };

        let plan = self.mppi.predicted_trajectory();
        let risk = plan
            .windows(2)
            .take(IMMINENT_STEPS)
            .enumerate()
            .map(|(i, w)| probability(i, w[1].x, w[1].y, (w[1].y - w[0].y).atan2(w[1].x - w[0].x)))
            .fold(0.0_f64, f64::max);
        let ref_scale = (1.0 - cfg.risk_slowdown_gain.max(0.0) * risk).clamp(cfg.min_velocity_scale.clamp(0.0, 1.0), 1.0);

        let reach = discs.iter().map(|d| d.1).fold(0.0_f64, f64::max);
        let speed = self.mppi.mppi_config.ref_velocity.min(constraints.max_linear_velocity);
        let all: &[Obstacle] = world.map_or(&[], |w| &w.obstacles);
        let detoured = self.mppi.detoured_points(state, all, reach, n, step, speed);
        let relax = detour_relax(&detoured, (speed * step).max(0.3));
        self.mppi.step_with(state, goal, constraints, dt, &extra, &relax, ref_scale, "MCA tracking", "mca_tracking")
    }

    fn set_path(&mut self, path: Path) {
        self.mppi.set_path(path);
    }

    fn set_trajectory(&mut self, trajectory: crate::types::Trajectory) {
        self.mppi.set_trajectory(trajectory);
    }

    fn set_time(&mut self, t: f64) {
        self.mppi.set_time(t);
    }

    fn reset(&mut self) {
        self.mppi.reset();
    }

    fn get_status(&self) -> ControllerStatus {
        self.mppi.get_status()
    }

    fn set_config(&mut self, config: ControllerConfig) {
        self.mppi.set_config(config);
    }

    fn get_config(&self) -> ControllerConfig {
        self.mppi.get_config()
    }

    fn get_type(&self) -> &'static str {
        "mca_follower"
    }

    fn predicted_trajectory(&self) -> Vec<datapod::Point> {
        self.mppi.predicted_trajectory().to_vec()
    }

    fn get_path_index(&self) -> usize {
        self.mppi.get_path_index()
    }

    fn base(&self) -> &ControllerBase {
        self.mppi.base()
    }

    fn base_mut(&mut self) -> &mut ControllerBase {
        self.mppi.base_mut()
    }
}
