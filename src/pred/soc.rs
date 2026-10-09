//! SOC — SVG-MPPI (Honda et al. 2024).
//!
//! 1. Guide particles are drawn around the warm-started mean control.
//! 2. Stein Variational Gradient Descent transports them toward the
//!    cost-based posterior with an RBF kernel (median-heuristic bandwidth).
//! 3. The best guide seeds the final MPPI sampling, with per-step variances
//!    adapted from the guide spread.
//! 4. The importance-weighted mean becomes the new control sequence.
//!
//! Controls are `(steer, accel, accel_lat)`; `accel_lat` (lateral
//! acceleration) only moves away from zero on a holonomic platform, where
//! it is optimised by exactly the same SVGD + MPPI process as the other
//! two channels.

#![allow(clippy::too_many_arguments)]

use crate::controller::{Controller, ControllerBase};
use crate::core::obstacles::CollisionChecker;
use crate::core::path::PathCursor;
use crate::pred::mppi::{
    CostWeights, Model, Reference, build_reference, build_timed_reference, command_from_controls,
    current_lateral_speed, current_speed, importance_weights, prepare, rollout,
    speed_at_arc_length, update_turn_in_place,
};
use crate::types::{
    Goal, Path, RobotConstraints, RobotState, Trajectory, VelocityCommand, WorldConstraints,
};
use datapod::Point;
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, Normal};

/// Clearance below which obstacles start contributing cost.
const OBSTACLE_INFLUENCE: f64 = 1.0;
/// Extra clearance treated as contact.
const OBSTACLE_SAFETY_MARGIN: f64 = 0.2;

#[derive(Clone, Debug)]
pub struct SocConfig {
    pub horizon_steps: usize,
    pub dt: f64,

    pub guide_samples: usize,
    pub num_samples: usize,

    pub temperature: f64,
    pub guide_temperature: f64,

    pub steering_noise: f64,
    pub acceleration_noise: f64,
    pub initial_steer_variance: f64,
    pub min_steer_variance: f64,
    pub max_steer_variance: f64,

    pub weight_cte: f64,
    pub weight_epsi: f64,
    pub weight_vel: f64,
    pub weight_steering: f64,
    pub weight_acceleration: f64,
    pub weight_obstacle: f64,

    pub ref_velocity: f64,

    pub svgd_iterations: usize,
    pub svgd_step_size: f64,
    pub kernel_bandwidth: f64,

    pub use_covariance_adaptation: bool,

    pub turn_first_activation_deg: f64,
    pub turn_first_release_deg: f64,

    pub decel_distance: f64,
}

impl Default for SocConfig {
    fn default() -> Self {
        Self {
            horizon_steps: 20,
            dt: 0.1,
            guide_samples: 64,
            num_samples: 512,
            temperature: 1.0,
            guide_temperature: 1.0,
            steering_noise: 0.5,
            acceleration_noise: 0.3,
            initial_steer_variance: 0.25,
            min_steer_variance: 1e-4,
            max_steer_variance: 1.0,
            weight_cte: 50.0,
            weight_epsi: 100.0,
            weight_vel: 100.0,
            weight_steering: 10.0,
            weight_acceleration: 5.0,
            weight_obstacle: 50.0,
            ref_velocity: 1.0,
            svgd_iterations: 3,
            svgd_step_size: 0.1,
            kernel_bandwidth: 1.0,
            use_covariance_adaptation: true,
            turn_first_activation_deg: 60.0,
            turn_first_release_deg: 15.0,
            decel_distance: 2.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SocFollower {
    pub base: ControllerBase,
    pub soc_config: SocConfig,
    cursor: PathCursor,
    mean_steering: Vec<f64>,
    mean_acceleration: Vec<f64>,
    mean_accel_lateral: Vec<f64>,
    predicted_trajectory: Vec<Point>,
    is_turning_in_place: bool,
    last_v: f64,
    last_vy: f64,
    shift_accum: f64,
    trajectory: Option<Trajectory>,
    clock: f64,
    rng: StdRng,
}

impl Default for SocFollower {
    fn default() -> Self {
        Self::with_soc_config(SocConfig::default())
    }
}

/// Central-difference gradient of the rollout cost with respect to the
/// flattened `[s0, a0, al0, s1, a1, al1, ...]` control vector.
fn cost_gradient(
    model: &Model,
    reference: &Reference,
    w: &CostWeights,
    extra: &dyn Fn(usize, f64, f64, f64) -> f64,
    steer: &mut [f64],
    accel: &mut [f64],
    accel_lat: &mut [f64],
) -> Vec<f64> {
    let n = steer.len();
    let eps = 1e-3;
    let mut grad = vec![0.0; 3 * n];
    for t in 0..n {
        let o = steer[t];
        steer[t] = o + eps;
        let (cp, _) = rollout(model, steer, accel, accel_lat, reference, w, extra, false);
        steer[t] = o - eps;
        let (cm, _) = rollout(model, steer, accel, accel_lat, reference, w, extra, false);
        steer[t] = o;
        grad[3 * t] = (cp - cm) / (2.0 * eps);

        let o = accel[t];
        accel[t] = o + eps;
        let (cp, _) = rollout(model, steer, accel, accel_lat, reference, w, extra, false);
        accel[t] = o - eps;
        let (cm, _) = rollout(model, steer, accel, accel_lat, reference, w, extra, false);
        accel[t] = o;
        grad[3 * t + 1] = (cp - cm) / (2.0 * eps);

        let o = accel_lat[t];
        accel_lat[t] = o + eps;
        let (cp, _) = rollout(model, steer, accel, accel_lat, reference, w, extra, false);
        accel_lat[t] = o - eps;
        let (cm, _) = rollout(model, steer, accel, accel_lat, reference, w, extra, false);
        accel_lat[t] = o;
        grad[3 * t + 2] = (cp - cm) / (2.0 * eps);
    }
    grad
}

impl SocFollower {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_soc_config(cfg: SocConfig) -> Self {
        Self::build(cfg, StdRng::from_entropy())
    }

    pub fn with_seed(cfg: SocConfig, seed: u64) -> Self {
        Self::build(cfg, StdRng::seed_from_u64(seed))
    }

    fn build(cfg: SocConfig, rng: StdRng) -> Self {
        let n = cfg.horizon_steps;
        Self {
            base: ControllerBase::default(),
            cursor: PathCursor::default(),
            mean_steering: vec![0.0; n],
            mean_acceleration: vec![0.0; n],
            mean_accel_lateral: vec![0.0; n],
            predicted_trajectory: Vec::new(),
            is_turning_in_place: false,
            last_v: 0.0,
            last_vy: 0.0,
            shift_accum: 0.0,
            trajectory: None,
            clock: 0.0,
            rng,
            soc_config: cfg,
        }
    }

    pub fn set_soc_config(&mut self, cfg: SocConfig) {
        let n = cfg.horizon_steps;
        self.mean_steering = vec![0.0; n];
        self.mean_acceleration = vec![0.0; n];
        self.mean_accel_lateral = vec![0.0; n];
        self.soc_config = cfg;
    }

    pub fn predicted_trajectory(&self) -> &[Point] {
        &self.predicted_trajectory
    }

    fn run_svgd(
        &self,
        model: &Model,
        reference: &Reference,
        w: &CostWeights,
        extra: &dyn Fn(usize, f64, f64, f64) -> f64,
        guides_s: &mut [Vec<f64>],
        guides_a: &mut [Vec<f64>],
        guides_al: &mut [Vec<f64>],
    ) {
        let n = self.soc_config.horizon_steps;
        let k = guides_s.len();
        let dims = 3 * n;
        let temperature = self.soc_config.guide_temperature.max(1e-6);
        let step = self.soc_config.svgd_step_size;

        for _ in 0..self.soc_config.svgd_iterations {
            let mut grad_log_p = vec![vec![0.0; dims]; k];
            let mut flat = vec![vec![0.0; dims]; k];
            for i in 0..k {
                let g = cost_gradient(
                    model,
                    reference,
                    w,
                    extra,
                    &mut guides_s[i],
                    &mut guides_a[i],
                    &mut guides_al[i],
                );
                for d in 0..dims {
                    grad_log_p[i][d] = -g[d] / temperature;
                }
                for t in 0..n {
                    flat[i][3 * t] = guides_s[i][t];
                    flat[i][3 * t + 1] = guides_a[i][t];
                    flat[i][3 * t + 2] = guides_al[i][t];
                }
            }

            let mut pair_sq = Vec::with_capacity(k * k / 2);
            for i in 0..k {
                for j in (i + 1)..k {
                    let s: f64 = (0..dims)
                        .map(|d| (flat[i][d] - flat[j][d]).powi(2))
                        .sum();
                    pair_sq.push(s);
                }
            }
            let bandwidth_sq = if pair_sq.is_empty() {
                self.soc_config.kernel_bandwidth.powi(2).max(1e-6)
            } else {
                pair_sq.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let median = pair_sq[pair_sq.len() / 2];
                (median / (2.0 * (k as f64).ln().max(1e-6)))
                    .max(self.soc_config.kernel_bandwidth.powi(2) * 1e-3)
            };

            let mut phi = vec![vec![0.0; dims]; k];
            for i in 0..k {
                for j in 0..k {
                    let dist_sq: f64 = (0..dims)
                        .map(|d| (flat[i][d] - flat[j][d]).powi(2))
                        .sum();
                    let kernel = (-dist_sq / (2.0 * bandwidth_sq)).exp();
                    for d in 0..dims {
                        phi[i][d] += kernel * grad_log_p[j][d]
                            - kernel * (flat[j][d] - flat[i][d]) / bandwidth_sq;
                    }
                }
                for p in phi[i].iter_mut() {
                    *p /= k as f64;
                }
            }

            let max_phi = phi
                .iter()
                .flat_map(|p| p.iter().map(|x| x.abs()))
                .fold(0.0_f64, f64::max)
                .max(1e-9);
            let scale = step / max_phi.max(1.0);
            for i in 0..k {
                for t in 0..n {
                    guides_s[i][t] += scale * phi[i][3 * t];
                    guides_a[i][t] += scale * phi[i][3 * t + 1];
                    guides_al[i][t] += scale * phi[i][3 * t + 2];
                }
                model.clamp_controls(&mut guides_s[i], &mut guides_a[i], &mut guides_al[i]);
            }
        }
    }
}

fn obstacle_penalty<'a>(
    checker: &'a CollisionChecker<'a>,
    weight: f64,
) -> impl Fn(usize, f64, f64, f64) -> f64 + 'a {
    move |step: usize, x: f64, y: f64, yaw: f64| -> f64 {
        if !checker.has_obstacles() {
            return 0.0;
        }
        let clearance = checker.clearance(step + 1, x, y, yaw);
        if clearance < 0.0 {
            weight * (100.0 + 10.0 * clearance.abs())
        } else if clearance < OBSTACLE_INFLUENCE {
            let r = (OBSTACLE_INFLUENCE - clearance) / OBSTACLE_INFLUENCE;
            weight * r * r
        } else {
            0.0
        }
    }
}

impl Controller for SocFollower {
    fn compute_control(
        &mut self,
        state: &RobotState,
        goal: &Goal,
        constraints: &RobotConstraints,
        dt: f64,
        world: Option<&WorldConstraints>,
    ) -> VelocityCommand {
        let prep = match prepare(&mut self.base, &mut self.cursor, state, goal, constraints, dt) {
            Ok(p) => p,
            Err(cmd) => {
                self.predicted_trajectory.clear();
                return cmd;
            }
        };
        let cfg = self.soc_config.clone();
        let turning = update_turn_in_place(
            &mut self.is_turning_in_place,
            state,
            constraints,
            prep.epsi,
            cfg.turn_first_activation_deg,
            cfg.turn_first_release_deg,
        );
        let ref_velocity = if turning { 0.2 } else { cfg.ref_velocity };

        let n = cfg.horizon_steps.max(1);
        if self.mean_steering.len() != n {
            self.mean_steering = vec![0.0; n];
            self.mean_acceleration = vec![0.0; n];
            self.mean_accel_lateral = vec![0.0; n];
        }
        let v_now = current_speed(state, self.last_v);
        let vy_now = current_lateral_speed(state, self.last_vy);
        let model = Model::new(state, constraints, v_now, vy_now, cfg.dt, prep.allow_reverse);
        let reference = match &self.trajectory {
            Some(traj) => build_timed_reference(traj, self.clock, n, cfg.dt),
            None => build_reference(
                &self.base.path,
                &self.cursor.cum,
                prep.proj.arc_length,
                n,
                cfg.dt,
                ref_velocity.min(constraints.max_linear_velocity),
                cfg.decel_distance,
            ),
        };
        let w = CostWeights {
            cte: cfg.weight_cte,
            epsi: cfg.weight_epsi,
            vel: if turning { 50.0 } else { cfg.weight_vel },
            steering: cfg.weight_steering,
            accel: cfg.weight_acceleration,
        };
        let checker = CollisionChecker::new(world, constraints, OBSTACLE_SAFETY_MARGIN);
        let extra = obstacle_penalty(&checker, cfg.weight_obstacle);
        model.clamp_controls(
            &mut self.mean_steering,
            &mut self.mean_acceleration,
            &mut self.mean_accel_lateral,
        );

        let k_guide = cfg.guide_samples.max(1);
        let guide_std_s = cfg.initial_steer_variance.max(1e-12).sqrt();
        let guide_std_a = cfg.acceleration_noise.max(1e-6);
        let ds = Normal::new(0.0, guide_std_s).unwrap();
        let da = Normal::new(0.0, guide_std_a).unwrap();
        let dal = Normal::new(0.0, guide_std_a).unwrap();
        let mut guides_s = vec![self.mean_steering.clone(); k_guide];
        let mut guides_a = vec![self.mean_acceleration.clone(); k_guide];
        let mut guides_al = vec![self.mean_accel_lateral.clone(); k_guide];
        for i in 1..k_guide {
            for t in 0..n {
                guides_s[i][t] += ds.sample(&mut self.rng);
                guides_a[i][t] += da.sample(&mut self.rng);
                guides_al[i][t] += dal.sample(&mut self.rng);
            }
            model.clamp_controls(&mut guides_s[i], &mut guides_a[i], &mut guides_al[i]);
        }

        if cfg.svgd_iterations > 0 {
            self.run_svgd(&model, &reference, &w, &extra, &mut guides_s, &mut guides_a, &mut guides_al);
        }

        let guide_costs: Vec<f64> = (0..k_guide)
            .map(|i| rollout(&model, &guides_s[i], &guides_a[i], &guides_al[i], &reference, &w, &extra, false).0)
            .collect();
        let best = guide_costs
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        let mut std_s = vec![cfg.steering_noise.max(1e-6); n];
        let mut std_a = vec![cfg.acceleration_noise.max(1e-6); n];
        let mut std_al = vec![cfg.acceleration_noise.max(1e-6); n];
        let var_lo = if cfg.min_steer_variance.is_finite() { cfg.min_steer_variance.max(0.0) } else { 1e-4 }
            .max((0.25 * cfg.steering_noise).powi(2));
        let acc_var_lo = (0.25 * cfg.acceleration_noise).powi(2);
        let var_hi = if cfg.max_steer_variance.is_finite() { cfg.max_steer_variance.max(var_lo) } else { 1.0 };
        let acc_var_hi = (cfg.acceleration_noise.max(1e-6) * 4.0).powi(2).max(var_lo);
        if cfg.use_covariance_adaptation && k_guide > 1 {
            for t in 0..n {
                let ms = guides_s.iter().map(|g| g[t]).sum::<f64>() / k_guide as f64;
                let ma = guides_a.iter().map(|g| g[t]).sum::<f64>() / k_guide as f64;
                let mal = guides_al.iter().map(|g| g[t]).sum::<f64>() / k_guide as f64;
                let vs = guides_s.iter().map(|g| (g[t] - ms).powi(2)).sum::<f64>() / k_guide as f64;
                let va = guides_a.iter().map(|g| (g[t] - ma).powi(2)).sum::<f64>() / k_guide as f64;
                let val = guides_al.iter().map(|g| (g[t] - mal).powi(2)).sum::<f64>() / k_guide as f64;
                std_s[t] = vs.clamp(var_lo, var_hi).max(1e-12).sqrt();
                std_a[t] = va.clamp(acc_var_lo, acc_var_hi.max(acc_var_lo)).max(1e-12).sqrt();
                std_al[t] = val.clamp(acc_var_lo, acc_var_hi.max(acc_var_lo)).max(1e-12).sqrt();
            }
        }

        let k = cfg.num_samples.max(1);
        let lambda = cfg.temperature.max(1e-6);
        let base_s = guides_s[best].clone();
        let base_a = guides_a[best].clone();
        let base_al = guides_al[best].clone();
        let mut costs = vec![0.0; k];
        let mut noise_s = vec![vec![0.0; n]; k];
        let mut noise_a = vec![vec![0.0; n]; k];
        let mut noise_al = vec![vec![0.0; n]; k];
        let mut steer = vec![0.0; n];
        let mut accel = vec![0.0; n];
        let mut accel_lat = vec![0.0; n];
        for j in 0..k {
            let mut control_cost = 0.0;
            for t in 0..n {
                let es = Normal::new(0.0, std_s[t]).unwrap().sample(&mut self.rng);
                let ea = Normal::new(0.0, std_a[t]).unwrap().sample(&mut self.rng);
                let eal = Normal::new(0.0, std_al[t]).unwrap().sample(&mut self.rng);
                noise_s[j][t] = es;
                noise_a[j][t] = ea;
                noise_al[j][t] = eal;
                steer[t] = base_s[t] + es;
                accel[t] = base_a[t] + ea;
                accel_lat[t] = base_al[t] + eal;
                control_cost += base_s[t] * es / (std_s[t] * std_s[t])
                    + base_a[t] * ea / (std_a[t] * std_a[t])
                    + base_al[t] * eal / (std_al[t] * std_al[t]);
            }
            let (c, _) = rollout(&model, &steer, &accel, &accel_lat, &reference, &w, &extra, false);
            costs[j] = c + lambda * control_cost;
        }
        let weights = importance_weights(&costs, lambda);
        for t in 0..n {
            let mut s = base_s[t];
            let mut a = base_a[t];
            let mut al = base_al[t];
            for j in 0..k {
                s += weights[j] * noise_s[j][t];
                a += weights[j] * noise_a[j][t];
                al += weights[j] * noise_al[j][t];
            }
            self.mean_steering[t] = s;
            self.mean_acceleration[t] = a;
            self.mean_accel_lateral[t] = al;
        }
        model.clamp_controls(
            &mut self.mean_steering,
            &mut self.mean_acceleration,
            &mut self.mean_accel_lateral,
        );

        let (_, traj) = rollout(
            &model,
            &self.mean_steering,
            &self.mean_acceleration,
            &self.mean_accel_lateral,
            &reference,
            &w,
            &extra,
            true,
        );
        self.predicted_trajectory = traj;

        let steer0 = self.mean_steering[0];
        let accel0 = self.mean_acceleration[0];
        let accel_lat0 = self.mean_accel_lateral[0];
        self.shift_accum += dt;
        if self.shift_accum >= cfg.dt {
            self.shift_accum -= cfg.dt;
            self.mean_steering.rotate_left(1);
            self.mean_acceleration.rotate_left(1);
            self.mean_accel_lateral.rotate_left(1);
            if n > 1 {
                self.mean_steering[n - 1] = self.mean_steering[n - 2];
                self.mean_acceleration[n - 1] = self.mean_acceleration[n - 2];
                self.mean_accel_lateral[n - 1] = self.mean_accel_lateral[n - 2];
            }
        }

        let dt_apply = dt.min(cfg.dt);
        let v_cap = speed_at_arc_length(&self.base.path.speeds, &self.cursor.cum, prep.proj.arc_length)
            .unwrap_or(f64::INFINITY);
        self.base.status.mode = "soc_tracking".into();
        let cmd = command_from_controls(
            v_now,
            vy_now,
            steer0,
            accel0,
            accel_lat0,
            dt_apply,
            v_cap,
            constraints,
            &self.base.config,
            prep.allow_reverse,
            turning,
            prep.epsi,
            "SOC tracking",
        );
        self.last_v = (v_now + accel0 * dt_apply).clamp(-v_cap, v_cap);
        self.last_vy = vy_now + accel_lat0 * dt_apply;
        cmd
    }

    fn set_trajectory(&mut self, trajectory: Trajectory) {
        self.set_path(trajectory.to_path());
        self.trajectory = Some(trajectory);
        self.clock = 0.0;
    }

    fn set_time(&mut self, t: f64) {
        self.clock = t;
    }

    fn set_path(&mut self, path: Path) {
        self.trajectory = None;
        self.cursor.set_path(&path.waypoints);
        self.base.path = path;
        self.base.path_index = 0;
        self.base.status = Default::default();
        let n = self.soc_config.horizon_steps;
        self.mean_steering = vec![0.0; n];
        self.mean_acceleration = vec![0.0; n];
        self.mean_accel_lateral = vec![0.0; n];
        self.predicted_trajectory.clear();
        self.is_turning_in_place = false;
        self.shift_accum = 0.0;
    }

    fn reset(&mut self) {
        self.set_path(Path::default());
        self.last_v = 0.0;
        self.last_vy = 0.0;
    }

    fn get_type(&self) -> &'static str {
        "soc_follower"
    }

    fn predicted_trajectory(&self) -> Vec<Point> {
        self.predicted_trajectory.clone()
    }

    fn base(&self) -> &ControllerBase {
        &self.base
    }

    fn base_mut(&mut self) -> &mut ControllerBase {
        &mut self.base
    }
}
