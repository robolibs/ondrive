//! MCA (Monte Carlo Approximation) — risk-aware MPPI with dynamic obstacle
//! collision probabilities evaluated by uniform-rejection Monte Carlo over
//! each horizon step's bounding box.
//!
//! When `WorldConstraints::obstacles` is empty, this behaves exactly like
//! MPPI (via a re-implemented rollout loop, not via inheritance).

use crate::controller::{Controller, ControllerBase, is_goal_reached};
use crate::core::math::normalize_angle;
use crate::pred::mppi::{MppiConfig, MppiFollower};
use crate::types::{
    Goal, Obstacle, OutputUnits, RobotConstraints, RobotState, SteeringType, VelocityCommand,
    WorldConstraints,
};
use datapod::Point;
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, Normal, Uniform};
use std::f64::consts::PI;

#[derive(Clone, Debug)]
pub struct McaConfig {
    pub horizon_steps: usize,
    pub dt: f64,

    pub num_samples: usize,
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
            num_mc_samples: 20000,
            temperature: 1.0,
            steering_noise: 0.5,
            acceleration_noise: 0.3,
            weight_cte: 100.0,
            weight_epsi: 100.0,
            weight_vel: 1.0,
            weight_steering: 10.0,
            weight_acceleration: 5.0,
            weight_soft_risk: 50.0,
            weight_hard_risk: 1e6,
            risk_threshold: 0.05,
            min_velocity_scale: 0.1,
            risk_slowdown_gain: 5.0,
            robot_radius_margin: 0.5,
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

#[derive(Clone, Debug)]
pub struct McaFollower {
    pub mca_config: McaConfig,
    mppi: MppiFollower,
    rng: StdRng,
    sample_collision_probs: Vec<Vec<f64>>,
    // Persistent mean controls (warm-start across ticks). MPPI shifts these
    // left by one after each tick and the first entry is the output.
    mean_steering: Vec<f64>,
    mean_acceleration: Vec<f64>,
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
        let n = cfg.horizon_steps;
        let mppi = MppiFollower::with_mppi_config(cfg.to_mppi());
        Self {
            mca_config: cfg,
            mppi,
            rng: StdRng::from_entropy(),
            sample_collision_probs: Vec::new(),
            mean_steering: vec![0.0; n],
            mean_acceleration: vec![0.0; n],
        }
    }

    pub fn with_seed(cfg: McaConfig, seed: u64) -> Self {
        let n = cfg.horizon_steps;
        let mppi = MppiFollower::with_seed(cfg.to_mppi(), seed.wrapping_add(1));
        Self {
            mca_config: cfg,
            mppi,
            rng: StdRng::seed_from_u64(seed),
            sample_collision_probs: Vec::new(),
            mean_steering: vec![0.0; n],
            mean_acceleration: vec![0.0; n],
        }
    }

    pub fn set_mca_config(&mut self, cfg: McaConfig) {
        let n = cfg.horizon_steps;
        self.mean_steering = vec![0.0; n];
        self.mean_acceleration = vec![0.0; n];
        self.mppi.set_mppi_config(cfg.to_mppi());
        self.mca_config = cfg;
    }

    pub fn sample_collision_probs(&self) -> &[Vec<f64>] {
        &self.sample_collision_probs
    }

    fn evaluate_obstacle_pdf(&self, obs: &Obstacle, timestep: usize, x: f64, y: f64) -> f64 {
        if obs.modes.is_empty() {
            return 0.0;
        }
        let mut total = 0.0;
        for mode in &obs.modes {
            if timestep >= mode.mean_x.len() || timestep >= mode.mean_y.len() {
                continue;
            }
            let mean_x = mode.mean_x[timestep];
            let mean_y = mode.mean_y[timestep];
            let std_x = mode.std_x[timestep].max(1e-6);
            let std_y = mode.std_y[timestep].max(1e-6);
            let dx = x - mean_x;
            let dy = y - mean_y;
            let exp_x = -(dx * dx) / (2.0 * std_x * std_x);
            let exp_y = -(dy * dy) / (2.0 * std_y * std_y);
            let norm_x = 1.0 / (std_x * (2.0 * PI).sqrt());
            let norm_y = 1.0 / (std_y * (2.0 * PI).sqrt());
            total += mode.weight * norm_x * norm_y * (exp_x + exp_y).exp();
        }
        total
    }

    fn compute_collision_probabilities(
        &mut self,
        trajectories: &[Vec<Point>],
        timestep: usize,
        robot_radius: f64,
        obstacles: &[Obstacle],
    ) -> Vec<f64> {
        let k = trajectories.len();
        let mut collision_probs = vec![0.0_f64; k];
        if obstacles.is_empty() {
            return collision_probs;
        }

        let mut x_min = f64::MAX;
        let mut x_max = f64::MIN;
        let mut y_min = f64::MAX;
        let mut y_max = f64::MIN;
        for t in trajectories {
            if timestep >= t.len() {
                continue;
            }
            let p = t[timestep];
            x_min = x_min.min(p.x);
            x_max = x_max.max(p.x);
            y_min = y_min.min(p.y);
            y_max = y_max.max(p.y);
        }
        if !x_min.is_finite() || !x_max.is_finite() {
            return collision_probs;
        }

        let max_obstacle_radius = obstacles
            .iter()
            .map(|o| o.radius)
            .fold(0.0_f64, f64::max);
        let total_radius = robot_radius + max_obstacle_radius;

        x_min -= total_radius;
        x_max += total_radius;
        y_min -= total_radius;
        y_max += total_radius;

        if x_max - x_min < 1e-9 || y_max - y_min < 1e-9 {
            return collision_probs;
        }

        let nmc = self.mca_config.num_mc_samples;
        let dist_x = Uniform::new(x_min, x_max);
        let dist_y = Uniform::new(y_min, y_max);

        let mut mc_x = Vec::with_capacity(nmc);
        let mut mc_y = Vec::with_capacity(nmc);
        let mut joint_prob = vec![0.0_f64; nmc];
        for _ in 0..nmc {
            mc_x.push(dist_x.sample(&mut self.rng));
            mc_y.push(dist_y.sample(&mut self.rng));
        }

        let area_element = (x_max - x_min) * (y_max - y_min) / nmc as f64;
        for j in 0..nmc {
            let mut prob_no_collision = 1.0;
            for obs in obstacles {
                let pdf = self.evaluate_obstacle_pdf(obs, timestep, mc_x[j], mc_y[j]);
                let marginal = (pdf * area_element).clamp(0.0, 1.0);
                prob_no_collision *= 1.0 - marginal;
            }
            joint_prob[j] = 1.0 - prob_no_collision;
        }

        let bbox_area = (x_max - x_min) * (y_max - y_min);

        for (k_idx, t) in trajectories.iter().enumerate() {
            if timestep >= t.len() {
                continue;
            }
            let robot_pos = t[timestep];
            let collision_radius = robot_radius + max_obstacle_radius;
            let collision_radius_sq = collision_radius * collision_radius;
            let collision_area = PI * collision_radius_sq;

            let mut sum_prob = 0.0;
            let mut count_in_region = 0;
            for j in 0..nmc {
                let dx = robot_pos.x - mc_x[j];
                let dy = robot_pos.y - mc_y[j];
                if dx * dx + dy * dy <= collision_radius_sq {
                    sum_prob += joint_prob[j];
                    count_in_region += 1;
                }
            }

            if count_in_region > 0 {
                let p = (collision_area / count_in_region as f64) * sum_prob / bbox_area
                    * nmc as f64;
                collision_probs[k_idx] = p.clamp(0.0, 1.0);
            }
        }

        collision_probs
    }
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
        // Without obstacles, behave exactly like MPPI.
        let has_obstacles = world.is_some_and(|w| !w.obstacles.is_empty());
        if !has_obstacles {
            let cmd = self.mppi.compute_control(state, goal, constraints, dt, world);
            // keep our own status mirrored so callers observe MCA state.
            return cmd;
        }
        let obstacles = world.unwrap().obstacles.clone();

        if self.mppi.base.path.waypoints.is_empty() {
            return VelocityCommand::invalid("no path");
        }

        let error = self.mppi.calculate_path_error(state);
        let working = self.mca_config.to_mppi();
        let is_diff = matches!(
            constraints.steering_type,
            SteeringType::Differential | SteeringType::SkidSteer
        );

        let cfg = self.mppi.base.config.clone();
        let (reached, dist, yaw_diff) = is_goal_reached(
            &state.pose,
            &goal.target_pose,
            cfg.goal_tolerance,
            cfg.angular_tolerance,
        );
        self.mppi.base.status.distance_to_goal = dist;
        self.mppi.base.status.heading_error = yaw_diff;
        if reached {
            self.mppi.base.status.goal_reached = true;
            self.mppi.base.status.mode = "stopped".into();
            return VelocityCommand {
                valid: true,
                status_message: "Goal reached".into(),
                ..VelocityCommand::default()
            };
        }

        let n = working.horizon_steps;
        let k = working.num_samples;
        let dt_internal = working.dt;

        let ref_traj = self.mppi.calculate_reference_trajectory(&error, &working);

        let steering_dist = Normal::new(0.0, working.steering_noise.max(1e-9)).unwrap();
        let accel_dist = Normal::new(0.0, working.acceleration_noise.max(1e-9)).unwrap();

        if self.mean_steering.len() != n {
            self.mean_steering = vec![0.0; n];
            self.mean_acceleration = vec![0.0; n];
        }

        let mut costs = vec![0.0_f64; k];
        let mut noise_steering = vec![vec![0.0_f64; n]; k];
        let mut noise_accel = vec![vec![0.0_f64; n]; k];
        let mut trajectories: Vec<Vec<Point>> = vec![Vec::with_capacity(n + 1); k];

        for sample_idx in 0..k {
            let mut x = state.pose.point.x;
            let mut y = state.pose.point.y;
            let mut yaw = state.pose.rotation.to_euler().yaw;
            let mut v = state.velocity.linear;
            let mut sample_cost = 0.0;
            trajectories[sample_idx].push(Point::new(x, y, 0.0));

            for i in 0..n {
                let eps_delta = steering_dist.sample(&mut self.rng);
                let eps_acc = accel_dist.sample(&mut self.rng);
                noise_steering[sample_idx][i] = eps_delta;
                noise_accel[sample_idx][i] = eps_acc;

                let mut delta_or_omega = self.mean_steering[i] + eps_delta;
                let mut a = self.mean_acceleration[i] + eps_acc;
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
                x += v * yaw.cos() * dt_internal;
                y += v * yaw.sin() * dt_internal;
                if is_diff {
                    yaw += delta_or_omega * dt_internal;
                } else {
                    yaw += v * delta_or_omega / lf * dt_internal;
                }
                yaw = normalize_angle(yaw);
                v += a * dt_internal;
                v = v.clamp(constraints.min_linear_velocity, constraints.max_linear_velocity);

                trajectories[sample_idx].push(Point::new(x, y, 0.0));

                let ref_idx = (i + 1).min(ref_traj.x.len() - 1);
                let r_x = ref_traj.x[ref_idx];
                let r_y = ref_traj.y[ref_idx];
                let r_yaw = ref_traj.yaw[ref_idx];
                let r_v = ref_traj.velocity[ref_idx];

                let dx = x - r_x;
                let dy = y - r_y;
                let cte = -dx * r_yaw.sin() + dy * r_yaw.cos();
                let epsi = normalize_angle(yaw - r_yaw);
                let vel_error = v - r_v;

                let mut cost = 0.0;
                cost += working.weight_cte * cte * cte;
                cost += working.weight_epsi * epsi * epsi;
                cost += working.weight_vel * vel_error * vel_error;
                cost += working.weight_steering * delta_or_omega * delta_or_omega;
                cost += working.weight_acceleration * a * a;
                sample_cost += cost * dt_internal;
            }
            costs[sample_idx] = sample_cost;
        }

        // Collision risk terms
        let mut robot_radius = constraints.robot_width.max(constraints.robot_length) / 2.0;
        if robot_radius < 0.1 {
            robot_radius = 0.5;
        }

        self.sample_collision_probs = vec![vec![0.0_f64; n]; k];
        for t in 0..n {
            let probs = self.compute_collision_probabilities(
                &trajectories,
                t + 1,
                robot_radius,
                &obstacles,
            );
            for idx in 0..k {
                self.sample_collision_probs[idx][t] = probs[idx];
                let soft = self.mca_config.weight_soft_risk * probs[idx];
                let hard = if probs[idx] > self.mca_config.risk_threshold {
                    self.mca_config.weight_hard_risk
                } else {
                    0.0
                };
                costs[idx] += (soft + hard) * dt_internal;
            }
        }

        // Importance weights
        let temperature = working.temperature.max(1e-6);
        let beta = 1.0 / temperature;
        let min_cost = costs.iter().cloned().fold(f64::INFINITY, f64::min);
        let mut weights = vec![0.0_f64; k];
        let mut weight_sum = 0.0_f64;
        for i in 0..k {
            let exp = (-beta * (costs[i] - min_cost)).max(-60.0);
            let w = exp.exp();
            weights[i] = w;
            weight_sum += w;
        }
        if weight_sum < 1e-12 {
            weight_sum = 1.0;
        }

        for i in 0..n {
            let mut d_delta = 0.0;
            let mut d_acc = 0.0;
            for j in 0..k {
                let w = weights[j] / weight_sum;
                d_delta += w * noise_steering[j][i];
                d_acc += w * noise_accel[j][i];
            }
            self.mean_steering[i] += d_delta;
            self.mean_acceleration[i] += d_acc;
            if is_diff {
                self.mean_steering[i] = self.mean_steering[i]
                    .clamp(-constraints.max_angular_velocity, constraints.max_angular_velocity);
            } else {
                self.mean_steering[i] = self.mean_steering[i]
                    .clamp(-constraints.max_steering_angle, constraints.max_steering_angle);
            }
            self.mean_acceleration[i] = self.mean_acceleration[i].clamp(
                -constraints.max_linear_acceleration,
                constraints.max_linear_acceleration,
            );
        }

        let steering_or_omega = *self.mean_steering.first().unwrap_or(&0.0);
        let acceleration = *self.mean_acceleration.first().unwrap_or(&0.0);

        // Shift the mean sequences left by one for next-tick warm-start.
        for i in 0..n.saturating_sub(1) {
            self.mean_steering[i] = self.mean_steering[i + 1];
            self.mean_acceleration[i] = self.mean_acceleration[i + 1];
        }
        if n > 0 {
            self.mean_steering[n - 1] = 0.0;
            self.mean_acceleration[n - 1] = 0.0;
        }

        self.mppi.base.status.distance_to_goal =
            state.pose.point.distance_to(goal.target_pose.point);
        self.mppi.base.status.cross_track_error = error.cte.abs();
        self.mppi.base.status.heading_error = error.epsi.abs();
        self.mppi.base.status.goal_reached = false;
        self.mppi.base.status.mode = "mca_tracking".into();

        let mut target_velocity = state.velocity.linear + acceleration * dt_internal;
        let min_vel = if cfg.allow_reverse {
            constraints.min_linear_velocity
        } else {
            0.0
        };
        target_velocity = target_velocity.clamp(min_vel, constraints.max_linear_velocity);

        let angular_output = if is_diff {
            steering_or_omega
                .clamp(-constraints.max_angular_velocity, constraints.max_angular_velocity)
        } else {
            steering_or_omega
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

        VelocityCommand {
            valid: true,
            status_message: "MCA tracking".into(),
            linear_velocity: linear,
            angular_velocity: angular,
            ..VelocityCommand::default()
        }
    }

    fn set_path(&mut self, path: crate::types::Path) {
        self.mppi.set_path(path);
    }

    fn reset(&mut self) {
        self.mppi.reset();
        self.sample_collision_probs.clear();
    }

    fn get_status(&self) -> crate::types::ControllerStatus {
        self.mppi.get_status()
    }

    fn set_config(&mut self, config: crate::types::ControllerConfig) {
        self.mppi.set_config(config);
    }

    fn get_config(&self) -> crate::types::ControllerConfig {
        self.mppi.get_config()
    }

    fn get_type(&self) -> &'static str {
        "mca_follower"
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
