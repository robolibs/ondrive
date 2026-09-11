//! MCA — risk-aware MPPI (DRA-MPPI, Trevisan & Alonso-Mora 2024).
//!
//! Obstacle positions are sampled from their Gaussian-mixture predictions
//! once per horizon step and shared by every rollout; the collision
//! probability of a rollout point is the fraction of samples inside the
//! combined radius. Soft cost is proportional to that probability, and a
//! hard penalty applies above `risk_threshold`.
//!
//! Without obstacles this is exactly MPPI.

use crate::controller::{Controller, ControllerBase};
use crate::pred::mppi::{MppiConfig, MppiFollower};
use crate::types::{
    ControllerConfig, ControllerStatus, Goal, Obstacle, Path, RobotConstraints, RobotState,
    VelocityCommand, WorldConstraints,
};
use rand::{Rng, SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, Normal};

#[derive(Clone, Debug)]
pub struct McaConfig {
    pub horizon_steps: usize,
    pub dt: f64,

    pub num_samples: usize,
    /// Obstacle position samples per obstacle per horizon step.
    pub num_mc_samples: usize,
    pub temperature: f64,
    pub steering_noise: f64,
    pub acceleration_noise: f64,

    pub weight_cte: f64,
    pub weight_epsi: f64,
    pub weight_vel: f64,
    pub weight_steering: f64,
    pub weight_acceleration: f64,

    pub weight_soft_risk: f64,
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
            dt: 0.1,
            num_samples: 400,
            num_mc_samples: 300,
            temperature: 1.0,
            steering_noise: 0.5,
            acceleration_noise: 0.3,
            weight_cte: 50.0,
            weight_epsi: 100.0,
            weight_vel: 100.0,
            weight_steering: 10.0,
            weight_acceleration: 5.0,
            weight_soft_risk: 50.0,
            weight_hard_risk: 1e4,
            risk_threshold: 0.05,
            min_velocity_scale: 0.1,
            risk_slowdown_gain: 5.0,
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
            ref_velocity: self.ref_velocity,
            turn_first_activation_deg: self.turn_first_activation_deg,
            turn_first_release_deg: self.turn_first_release_deg,
            decel_distance: self.decel_distance,
        }
    }
}

/// Horizon steps whose collision probability drives the risk slowdown.
const IMMINENT_STEPS: usize = 5;

/// Sampled obstacle positions for one horizon step: `(x, y, radius)`.
type StepSamples = Vec<Vec<(f64, f64, f64)>>;

#[derive(Clone, Debug)]
pub struct McaFollower {
    pub mca_config: McaConfig,
    mppi: MppiFollower,
    rng: StdRng,
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
        Self {
            mppi: MppiFollower::with_mppi_config(cfg.to_mppi()),
            mca_config: cfg,
            rng: StdRng::from_entropy(),
        }
    }

    pub fn with_seed(cfg: McaConfig, seed: u64) -> Self {
        Self {
            mppi: MppiFollower::with_seed(cfg.to_mppi(), seed.wrapping_add(1)),
            mca_config: cfg,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    pub fn set_mca_config(&mut self, cfg: McaConfig) {
        self.mppi.set_mppi_config(cfg.to_mppi());
        self.mca_config = cfg;
    }

    pub fn predicted_trajectory(&self) -> &[datapod::Point] {
        self.mppi.predicted_trajectory()
    }

    /// Draw `m` positions per obstacle for horizon steps `1..=n` from the
    /// Gaussian mixture, holding the last predicted step when the
    /// prediction is shorter than the horizon.
    fn sample_obstacles(&mut self, obstacles: &[Obstacle], n: usize, m: usize) -> StepSamples {
        let mut out: StepSamples = vec![Vec::new(); n];
        for obs in obstacles {
            let modes: Vec<_> = obs
                .modes
                .iter()
                .filter(|md| md.weight.is_finite() && md.weight > 0.0 && !md.mean_x.is_empty() && !md.mean_y.is_empty())
                .collect();
            if modes.is_empty() {
                continue;
            }
            let total_w: f64 = modes.iter().map(|md| md.weight).sum();
            for (step, bucket) in out.iter_mut().enumerate() {
                let t = step + 1;
                for _ in 0..m {
                    let mut pick = self.rng.r#gen::<f64>() * total_w;
                    let mut chosen = modes[modes.len() - 1];
                    for md in &modes {
                        if pick < md.weight {
                            chosen = md;
                            break;
                        }
                        pick -= md.weight;
                    }
                    let ti = t.min(chosen.mean_x.len() - 1).min(chosen.mean_y.len() - 1);
                    let sx = chosen.std_x.get(ti).copied().filter(|s| s.is_finite()).unwrap_or(0.0).max(0.0);
                    let sy = chosen.std_y.get(ti).copied().filter(|s| s.is_finite()).unwrap_or(0.0).max(0.0);
                    let x = match Normal::new(chosen.mean_x[ti], sx) {
                        Ok(d) if sx > 0.0 => d.sample(&mut self.rng),
                        _ => chosen.mean_x[ti],
                    };
                    let y = match Normal::new(chosen.mean_y[ti], sy) {
                        Ok(d) if sy > 0.0 => d.sample(&mut self.rng),
                        _ => chosen.mean_y[ti],
                    };
                    bucket.push((x, y, obs.radius.max(0.0)));
                }
            }
        }
        out
    }
}

fn robot_radius(constraints: &RobotConstraints, margin: f64) -> f64 {
    let r = 0.5 * constraints.robot_width.hypot(constraints.robot_length);
    (if r < 0.05 { 0.3 } else { r }) + margin.max(0.0)
}

impl Controller for McaFollower {
    fn compute_control(
        &mut self,
        state: &RobotState,
        goal: &Goal,
        constraints: &RobotConstraints,
        dt: f64,
        world: Option<&WorldConstraints>,
    ) -> VelocityCommand {
        let obstacles: Vec<Obstacle> = world
            .map(|w| {
                w.obstacles
                    .iter()
                    .filter(|o| {
                        o.modes.iter().any(|m| {
                            m.weight.is_finite()
                                && m.weight > 0.0
                                && !m.mean_x.is_empty()
                                && !m.mean_y.is_empty()
                        })
                    })
                    .cloned()
                    .collect()
            })
            .unwrap_or_default();
        if obstacles.is_empty() {
            return self.mppi.step_with(
                state,
                goal,
                constraints,
                dt,
                &|_, _, _| 0.0,
                1.0,
                "MCA tracking",
                "mca_tracking",
            );
        }

        let n = self.mca_config.horizon_steps.max(1);
        let m = self.mca_config.num_mc_samples.max(1);
        let samples = self.sample_obstacles(&obstacles, n, m);
        let per_obstacle = m as f64;
        let robot_r = robot_radius(constraints, self.mca_config.robot_radius_margin);
        let soft = self.mca_config.weight_soft_risk;
        let hard = self.mca_config.weight_hard_risk;
        let threshold = self.mca_config.risk_threshold;
        let obstacle_count = obstacles.len();

        let collision_probability = |step: usize, x: f64, y: f64| -> f64 {
            let bucket = &samples[step.min(n - 1)];
            if bucket.is_empty() {
                return 0.0;
            }
            let mut survive = 1.0;
            for o in 0..obstacle_count {
                let slice = &bucket[o * m..(o + 1) * m];
                let hits = slice
                    .iter()
                    .filter(|(ox, oy, r)| (x - ox).hypot(y - oy) < r + robot_r)
                    .count();
                survive *= 1.0 - hits as f64 / per_obstacle;
            }
            1.0 - survive
        };
        let extra = |step: usize, x: f64, y: f64| -> f64 {
            let p = collision_probability(step, x, y);
            soft * p + if p > threshold { hard } else { 0.0 }
        };

        let risk = self
            .mppi
            .predicted_trajectory()
            .iter()
            .enumerate()
            .skip(1)
            .take(IMMINENT_STEPS)
            .map(|(i, p)| collision_probability(i - 1, p.x, p.y))
            .fold(0.0_f64, f64::max);
        let ref_scale = (1.0 - self.mca_config.risk_slowdown_gain.max(0.0) * risk)
            .clamp(self.mca_config.min_velocity_scale.clamp(0.0, 1.0), 1.0);

        self.mppi.step_with(
            state,
            goal,
            constraints,
            dt,
            &extra,
            ref_scale,
            "MCA tracking",
            "mca_tracking",
        )
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
