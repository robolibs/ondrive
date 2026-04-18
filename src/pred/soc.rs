//! SOC (Stochastic Optimal Control) — SVG-MPPI.
//!
//! Pipeline:
//!   1. Sample `guide_samples` guide particles from zero-mean Gaussians.
//!   2. Run `svgd_iterations` rounds of Stein Variational Gradient Descent
//!      on the guide-particle set. Each round uses finite-difference cost
//!      gradients, the RBF kernel with adaptive bandwidth from the median
//!      heuristic, and projected updates onto the feasible control range.
//!   3. Pick the lowest-cost guide particle, adapt per-step variances from
//!      the guide distribution, then draw `num_samples` final particles
//!      from Gaussians centred at the best guide with those variances.
//!   4. MPPI-weight the final particles to produce the output command.

use crate::controller::{Controller, ControllerBase, is_goal_reached};
use crate::core::math::normalize_angle;
use crate::types::{
    ControllerConfig, ControllerStatus, Goal, OutputUnits, Path, RobotConstraints, RobotState,
    SteeringType, VelocityCommand, WorldConstraints,
};
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, Normal};

#[derive(Clone, Debug)]
pub struct SocConfig {
    pub horizon_steps: usize,
    pub dt: f64,

    pub guide_samples: usize,
    pub num_samples: usize,
    pub grad_samples: usize,

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
    pub gaussian_fitting_lambda: f64,

    pub decel_distance: f64,
}

impl Default for SocConfig {
    fn default() -> Self {
        Self {
            horizon_steps: 20,
            dt: 0.1,
            guide_samples: 64,
            num_samples: 512,
            grad_samples: 32,
            temperature: 1.0,
            guide_temperature: 1.0,
            steering_noise: 0.5,
            acceleration_noise: 0.3,
            initial_steer_variance: 0.25,
            min_steer_variance: 1e-4,
            max_steer_variance: 1.0,
            weight_cte: 100.0,
            weight_epsi: 100.0,
            weight_vel: 1.0,
            weight_steering: 10.0,
            weight_acceleration: 5.0,
            weight_obstacle: 50.0,
            ref_velocity: 1.0,
            svgd_iterations: 3,
            svgd_step_size: 0.1,
            kernel_bandwidth: 1.0,
            use_covariance_adaptation: true,
            gaussian_fitting_lambda: 0.1,
            decel_distance: 2.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SocFollower {
    pub base: ControllerBase,
    pub soc_config: SocConfig,
    rng: StdRng,
}

impl Default for SocFollower {
    fn default() -> Self {
        Self::with_soc_config(SocConfig::default())
    }
}

impl SocFollower {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_soc_config(cfg: SocConfig) -> Self {
        Self {
            base: ControllerBase::default(),
            soc_config: cfg,
            rng: StdRng::from_entropy(),
        }
    }

    pub fn with_seed(cfg: SocConfig, seed: u64) -> Self {
        Self {
            base: ControllerBase::default(),
            soc_config: cfg,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    pub fn set_soc_config(&mut self, cfg: SocConfig) {
        self.soc_config = cfg;
    }

    /// Central-difference gradient of the rollout cost with respect to each
    /// element of the (steering, accel) control sequence. Returned layout
    /// matches the caller's flattened view: `grad[2*t]` is the steering
    /// component at step `t`, `grad[2*t+1]` the acceleration component.
    fn cost_gradient(
        soc_config: &SocConfig,
        path_waypoints: &[datapod::Pose],
        state: &RobotState,
        goal: &Goal,
        constraints: &RobotConstraints,
        is_diff: bool,
        steering: &[f64],
        accel: &[f64],
        obstacles_xy: &[(Vec<f64>, Vec<f64>, f64)],
    ) -> Vec<f64> {
        let n = soc_config.horizon_steps;
        let mut grad = vec![0.0_f64; 2 * n];
        let eps = 1e-3;

        let mut s_work = steering.to_vec();
        let mut a_work = accel.to_vec();
        // RNG unused (sigma None), but rollout_cost needs a reference; a
        // dummy deterministic RNG is fine.
        let mut dummy_rng = StdRng::seed_from_u64(0);

        for t in 0..n {
            // Steering component
            let orig = s_work[t];
            s_work[t] = orig + eps;
            let c_plus = Self::rollout_cost(
                &mut dummy_rng,
                soc_config,
                path_waypoints,
                state,
                goal,
                constraints,
                is_diff,
                &s_work,
                &a_work,
                None,
                None,
                None,
                None,
                obstacles_xy,
            );
            s_work[t] = orig - eps;
            let c_minus = Self::rollout_cost(
                &mut dummy_rng,
                soc_config,
                path_waypoints,
                state,
                goal,
                constraints,
                is_diff,
                &s_work,
                &a_work,
                None,
                None,
                None,
                None,
                obstacles_xy,
            );
            s_work[t] = orig;
            grad[2 * t] = (c_plus - c_minus) / (2.0 * eps);

            // Acceleration component
            let orig = a_work[t];
            a_work[t] = orig + eps;
            let c_plus = Self::rollout_cost(
                &mut dummy_rng,
                soc_config,
                path_waypoints,
                state,
                goal,
                constraints,
                is_diff,
                &s_work,
                &a_work,
                None,
                None,
                None,
                None,
                obstacles_xy,
            );
            a_work[t] = orig - eps;
            let c_minus = Self::rollout_cost(
                &mut dummy_rng,
                soc_config,
                path_waypoints,
                state,
                goal,
                constraints,
                is_diff,
                &s_work,
                &a_work,
                None,
                None,
                None,
                None,
                obstacles_xy,
            );
            a_work[t] = orig;
            grad[2 * t + 1] = (c_plus - c_minus) / (2.0 * eps);
        }

        grad
    }

    /// One or more SVGD iterations over the guide-particle set.
    fn run_svgd(
        &mut self,
        guide_steering: &mut [Vec<f64>],
        guide_acceleration: &mut [Vec<f64>],
        state: &RobotState,
        goal: &Goal,
        constraints: &RobotConstraints,
        is_diff: bool,
        obstacles_xy: &[(Vec<f64>, Vec<f64>, f64)],
    ) {
        let n_particles = guide_steering.len();
        let n = self.soc_config.horizon_steps;
        let dims = 2 * n;
        let temperature = self.soc_config.guide_temperature.max(1e-6);
        let step = self.soc_config.svgd_step_size;

        let flatten = |steering: &[f64], accel: &[f64]| -> Vec<f64> {
            let mut v = Vec::with_capacity(dims);
            for t in 0..n {
                v.push(steering[t]);
                v.push(accel[t]);
            }
            v
        };

        for _iter in 0..self.soc_config.svgd_iterations {
            // Per-particle gradients of log p(x) = -C(x) / T.
            let mut grad_log_p = vec![vec![0.0_f64; dims]; n_particles];
            let mut flats = Vec::with_capacity(n_particles);
            for i in 0..n_particles {
                let grad_c = Self::cost_gradient(
                    &self.soc_config,
                    &self.base.path.waypoints,
                    state,
                    goal,
                    constraints,
                    is_diff,
                    &guide_steering[i],
                    &guide_acceleration[i],
                    obstacles_xy,
                );
                for d in 0..dims {
                    grad_log_p[i][d] = -grad_c[d] / temperature;
                }
                flats.push(flatten(&guide_steering[i], &guide_acceleration[i]));
            }

            // Bandwidth via the median heuristic. Falls back to the config
            // value if the median is degenerate.
            let bandwidth_sq = {
                let mut pair_sq = Vec::with_capacity(n_particles * n_particles / 2);
                for i in 0..n_particles {
                    for j in (i + 1)..n_particles {
                        let mut s = 0.0;
                        for d in 0..dims {
                            let diff = flats[i][d] - flats[j][d];
                            s += diff * diff;
                        }
                        pair_sq.push(s);
                    }
                }
                if pair_sq.is_empty() {
                    self.soc_config.kernel_bandwidth.powi(2).max(1e-6)
                } else {
                    pair_sq.sort_by(|a, b| {
                        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    let median = pair_sq[pair_sq.len() / 2];
                    let heuristic =
                        median / (2.0 * (n_particles as f64).ln().max(1e-6));
                    heuristic.max(self.soc_config.kernel_bandwidth.powi(2) * 1e-3)
                }
            };

            // Compute φ for each particle.
            let mut phi = vec![vec![0.0_f64; dims]; n_particles];
            for i in 0..n_particles {
                for j in 0..n_particles {
                    let mut dist_sq = 0.0;
                    for d in 0..dims {
                        let diff = flats[i][d] - flats[j][d];
                        dist_sq += diff * diff;
                    }
                    let kernel = (-dist_sq / (2.0 * bandwidth_sq)).exp();
                    for d in 0..dims {
                        // k(x_j, x_i) · ∇ log p(x_j)
                        phi[i][d] += kernel * grad_log_p[j][d];
                        // + ∇_{x_j} k(x_j, x_i) = -k · (x_j - x_i) / h²
                        phi[i][d] +=
                            -kernel * (flats[j][d] - flats[i][d]) / bandwidth_sq;
                    }
                }
                for d in 0..dims {
                    phi[i][d] /= n_particles as f64;
                }
            }

            // Apply update and clip to feasible control range.
            for i in 0..n_particles {
                for t in 0..n {
                    guide_steering[i][t] += step * phi[i][2 * t];
                    guide_acceleration[i][t] += step * phi[i][2 * t + 1];

                    if is_diff {
                        guide_steering[i][t] = guide_steering[i][t].clamp(
                            -constraints.max_angular_velocity,
                            constraints.max_angular_velocity,
                        );
                    } else {
                        guide_steering[i][t] = guide_steering[i][t].clamp(
                            -constraints.max_steering_angle,
                            constraints.max_steering_angle,
                        );
                    }
                    guide_acceleration[i][t] = guide_acceleration[i][t].clamp(
                        -constraints.max_linear_acceleration,
                        constraints.max_linear_acceleration,
                    );
                }
            }
        }
    }

    fn rollout_cost<R: rand::Rng>(
        rng: &mut R,
        soc_config: &SocConfig,
        path_waypoints: &[datapod::Pose],
        state: &RobotState,
        goal: &Goal,
        constraints: &RobotConstraints,
        is_diff: bool,
        mean_steering: &[f64],
        mean_accel: &[f64],
        steering_sigma: Option<&[f64]>,
        accel_sigma: Option<&[f64]>,
        out_steering: Option<&mut [f64]>,
        out_accel: Option<&mut [f64]>,
        obstacles_xy: &[(Vec<f64>, Vec<f64>, f64)], // (mean_x, mean_y, radius)
    ) -> f64 {
        let n = soc_config.horizon_steps;
        let dt = soc_config.dt;
        let mut x = state.pose.point.x;
        let mut y = state.pose.point.y;
        let mut yaw = state.pose.rotation.to_euler().yaw;
        let mut v = state.velocity.linear;
        let mut cost = 0.0;

        let mut out_steering_opt = out_steering;
        let mut out_accel_opt = out_accel;

        for i in 0..n {
            let (delta_sample, a_sample) = match (steering_sigma, accel_sigma) {
                (Some(ss), Some(aa)) => {
                    let d = Normal::new(mean_steering[i], ss[i].sqrt().max(1e-9))
                        .unwrap()
                        .sample(rng);
                    let a = Normal::new(mean_accel[i], aa[i].sqrt().max(1e-9)).unwrap().sample(rng);
                    (d, a)
                }
                _ => (mean_steering[i], mean_accel[i]),
            };

            if let Some(out) = out_steering_opt.as_deref_mut() {
                out[i] = delta_sample;
            }
            if let Some(out) = out_accel_opt.as_deref_mut() {
                out[i] = a_sample;
            }

            let mut delta_or_omega = delta_sample;
            let mut a = a_sample;
            if is_diff {
                delta_or_omega = delta_or_omega.clamp(
                    -constraints.max_angular_velocity,
                    constraints.max_angular_velocity,
                );
            } else {
                delta_or_omega = delta_or_omega.clamp(
                    -constraints.max_steering_angle,
                    constraints.max_steering_angle,
                );
            }
            a = a.clamp(
                -constraints.max_linear_acceleration,
                constraints.max_linear_acceleration,
            );

            let lf = constraints.wheelbase;
            x += v * yaw.cos() * dt;
            y += v * yaw.sin() * dt;
            if is_diff {
                yaw += delta_or_omega * dt;
            } else {
                yaw += v * delta_or_omega / lf * dt;
            }
            yaw = normalize_angle(yaw);
            v += a * dt;
            v = v.clamp(constraints.min_linear_velocity, constraints.max_linear_velocity);

            // Cost: min distance from current point to any path waypoint
            let mut min_path_dist = f64::MAX;
            for wp in path_waypoints {
                let dx = x - wp.point.x;
                let dy = y - wp.point.y;
                let d = (dx * dx + dy * dy).sqrt();
                if d < min_path_dist {
                    min_path_dist = d;
                }
            }
            cost += soc_config.weight_cte * min_path_dist * min_path_dist;

            // Goal attraction (small)
            let dx_g = x - goal.target_pose.point.x;
            let dy_g = y - goal.target_pose.point.y;
            cost += 0.1 * (dx_g * dx_g + dy_g * dy_g).sqrt();

            // Obstacle clearance
            for (obs_mean_x, obs_mean_y, obs_radius) in obstacles_xy {
                if i < obs_mean_x.len() && i < obs_mean_y.len() {
                    let ox = obs_mean_x[i];
                    let oy = obs_mean_y[i];
                    let dist = ((x - ox).powi(2) + (y - oy).powi(2)).sqrt();
                    let clearance = dist - obs_radius - constraints.robot_width / 2.0;
                    if clearance < 0.0 {
                        cost += soc_config.weight_obstacle * 100.0;
                    } else if clearance < 0.3 {
                        cost += soc_config.weight_obstacle * (0.3 - clearance) / 0.3;
                    }
                }
            }

            cost += soc_config.weight_steering * delta_or_omega * delta_or_omega;
            cost += soc_config.weight_acceleration * a * a;
        }

        cost
    }
}

impl Controller for SocFollower {
    fn compute_control(
        &mut self,
        state: &RobotState,
        goal: &Goal,
        constraints: &RobotConstraints,
        _dt: f64,
        world: Option<&WorldConstraints>,
    ) -> VelocityCommand {
        if self.base.path.waypoints.is_empty() {
            return VelocityCommand::invalid("no path");
        }

        let is_diff = matches!(
            constraints.steering_type,
            SteeringType::Differential | SteeringType::SkidSteer
        );

        let cfg = self.base.config.clone();
        let (reached, dist, yaw_diff) = is_goal_reached(
            &state.pose,
            &goal.target_pose,
            cfg.goal_tolerance,
            cfg.angular_tolerance,
        );
        self.base.status.distance_to_goal = dist;
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

        let n = self.soc_config.horizon_steps;
        let k_guide = self.soc_config.guide_samples;
        let k_final = self.soc_config.num_samples;
        let dt_internal = self.soc_config.dt;

        // Extract obstacles (first mode only)
        let mut obstacles_xy: Vec<(Vec<f64>, Vec<f64>, f64)> = Vec::new();
        if let Some(w) = world {
            for obs in &w.obstacles {
                if let Some(m) = obs.modes.first() {
                    obstacles_xy.push((m.mean_x.clone(), m.mean_y.clone(), obs.radius));
                }
            }
        }

        // Guide particles: sample steering/accel from zero-mean Gaussians.
        let mut guide_steering: Vec<Vec<f64>> = vec![vec![0.0; n]; k_guide];
        let mut guide_acceleration: Vec<Vec<f64>> = vec![vec![0.0; n]; k_guide];
        let steering_dist =
            Normal::new(0.0, self.soc_config.initial_steer_variance.max(1e-9)).unwrap();
        let accel_dist = Normal::new(0.0, self.soc_config.acceleration_noise.max(1e-9)).unwrap();
        for k in 0..k_guide {
            for i in 0..n {
                guide_steering[k][i] = steering_dist.sample(&mut self.rng);
                guide_acceleration[k][i] = accel_dist.sample(&mut self.rng);
            }
        }

        // Stein Variational Gradient Descent on the guide-particle set.
        // Transport each particle toward the target posterior while a
        // repulsion term (from the kernel's gradient) keeps them diverse.
        if self.soc_config.svgd_iterations > 0 {
            self.run_svgd(
                &mut guide_steering,
                &mut guide_acceleration,
                state,
                goal,
                constraints,
                is_diff,
                &obstacles_xy,
            );
        }

        // Evaluate guide costs (after SVGD transport) by rollout.
        let mut guide_costs = vec![0.0_f64; k_guide];
        for k in 0..k_guide {
            guide_costs[k] = Self::rollout_cost(
                &mut self.rng, // unused when sigma is None
                &self.soc_config,
                &self.base.path.waypoints,
                state,
                goal,
                constraints,
                is_diff,
                &guide_steering[k],
                &guide_acceleration[k],
                None,
                None,
                None,
                None,
                &obstacles_xy,
            );
        }

        let best_guide_idx = guide_costs
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        // Adaptive per-step variances from guide distribution.
        let mut adapted_steer_cov = vec![0.0_f64; n];
        let mut adapted_accel_cov = vec![0.0_f64; n];
        for i in 0..n {
            let (mut sm, mut am) = (0.0, 0.0);
            for k in 0..k_guide {
                sm += guide_steering[k][i];
                am += guide_acceleration[k][i];
            }
            sm /= k_guide as f64;
            am /= k_guide as f64;
            let (mut sv, mut av) = (0.0, 0.0);
            for k in 0..k_guide {
                sv += (guide_steering[k][i] - sm).powi(2);
                av += (guide_acceleration[k][i] - am).powi(2);
            }
            sv /= k_guide as f64;
            av /= k_guide as f64;
            adapted_steer_cov[i] =
                sv.clamp(self.soc_config.min_steer_variance, self.soc_config.max_steer_variance);
            adapted_accel_cov[i] =
                av.clamp(self.soc_config.min_steer_variance, self.soc_config.max_steer_variance);
        }

        // Final MPPI sampling around best guide with adapted variance.
        let mut final_steering = vec![vec![0.0_f64; n]; k_final];
        let mut final_accel = vec![vec![0.0_f64; n]; k_final];
        let mut final_costs = vec![0.0_f64; k_final];
        let best_steer = guide_steering[best_guide_idx].clone();
        let best_accel = guide_acceleration[best_guide_idx].clone();

        for k in 0..k_final {
            let steer_slice: &mut [f64] = &mut final_steering[k];
            let accel_slice: &mut [f64] = &mut final_accel[k];
            final_costs[k] = Self::rollout_cost(
                &mut self.rng,
                &self.soc_config,
                &self.base.path.waypoints,
                state,
                goal,
                constraints,
                is_diff,
                &best_steer,
                &best_accel,
                Some(&adapted_steer_cov),
                Some(&adapted_accel_cov),
                Some(steer_slice),
                Some(accel_slice),
                &obstacles_xy,
            );
        }

        // MPPI weighting → expected control
        let temperature = self.soc_config.temperature.max(1e-6);
        let beta = 1.0 / temperature;
        let min_cost = final_costs.iter().cloned().fold(f64::INFINITY, f64::min);
        let mut weights = vec![0.0_f64; k_final];
        let mut weight_sum = 0.0_f64;
        for i in 0..k_final {
            let exp = (-beta * (final_costs[i] - min_cost)).max(-60.0);
            let w = exp.exp();
            weights[i] = w;
            weight_sum += w;
        }
        if weight_sum < 1e-12 {
            weight_sum = 1.0;
        }

        let mut steering_cmd = 0.0;
        let mut accel_cmd = 0.0;
        for k in 0..k_final {
            let w = weights[k] / weight_sum;
            steering_cmd += w * final_steering[k][0];
            accel_cmd += w * final_accel[k][0];
        }

        // Ref-velocity taper by remaining path distance
        let mut dist_to_end = 0.0;
        let start_idx = self.base.path_index;
        for j in start_idx..self.base.path.waypoints.len().saturating_sub(1) {
            dist_to_end += self.base.path.waypoints[j]
                .point
                .distance_to(self.base.path.waypoints[j + 1].point);
        }
        let effective_ref_velocity = if self.soc_config.decel_distance > 1e-6
            && dist_to_end < self.soc_config.decel_distance
        {
            (self.soc_config.ref_velocity * (dist_to_end / self.soc_config.decel_distance)).max(0.0)
        } else {
            self.soc_config.ref_velocity
        };

        // Use principled kinematic integration for output velocity:
        // reference-velocity bias from effective_ref_velocity keeps the
        // overall drive moving while still letting the sampler decelerate.
        let mut target_velocity = effective_ref_velocity + accel_cmd * dt_internal;
        let min_vel = if cfg.allow_reverse {
            constraints.min_linear_velocity
        } else {
            0.0
        };
        target_velocity = target_velocity.clamp(min_vel, constraints.max_linear_velocity);

        let angular_output = if is_diff {
            steering_cmd
                .clamp(-constraints.max_angular_velocity, constraints.max_angular_velocity)
        } else {
            steering_cmd
                .clamp(-constraints.max_steering_angle, constraints.max_steering_angle)
        };

        let (linear, angular) = match cfg.output_units {
            OutputUnits::Normalized => {
                let linear = if constraints.max_linear_velocity > 0.0 {
                    target_velocity / constraints.max_linear_velocity
                } else {
                    0.0
                };
                let angular = if is_diff {
                    if constraints.max_angular_velocity > 0.0 {
                        angular_output / constraints.max_angular_velocity
                    } else {
                        0.0
                    }
                } else if constraints.max_steering_angle > 0.0 {
                    angular_output / constraints.max_steering_angle
                } else {
                    0.0
                };
                (linear, angular)
            }
            OutputUnits::Physical => (target_velocity, angular_output),
        };

        self.base.status.distance_to_goal =
            state.pose.point.distance_to(goal.target_pose.point);
        self.base.status.goal_reached = false;
        self.base.status.mode = "soc_tracking".into();

        VelocityCommand {
            valid: true,
            status_message: "SOC tracking".into(),
            linear_velocity: linear,
            angular_velocity: angular,
            ..VelocityCommand::default()
        }
    }

    fn set_path(&mut self, path: Path) {
        self.base.path = path;
        self.base.path_index = 0;
    }

    fn reset(&mut self) {
        self.base.path.waypoints.clear();
        self.base.path_index = 0;
        self.base.status = ControllerStatus::default();
    }

    fn get_status(&self) -> ControllerStatus {
        self.base.status.clone()
    }

    fn set_config(&mut self, config: ControllerConfig) {
        self.base.config = config;
    }

    fn get_config(&self) -> ControllerConfig {
        self.base.config.clone()
    }

    fn get_type(&self) -> &'static str {
        "soc_follower"
    }

    fn base(&self) -> &ControllerBase {
        &self.base
    }

    fn base_mut(&mut self) -> &mut ControllerBase {
        &mut self.base
    }
}
