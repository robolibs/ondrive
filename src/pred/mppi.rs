#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use crate::controller::{Controller, ControllerBase, is_goal_reached};
use crate::core::math::normalize_angle;
use crate::types::{
    Goal, OutputUnits, RobotConstraints, RobotState, SteeringType, VelocityCommand,
    WorldConstraints,
};
use datapod::Point;
use rand::{Rng, SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, Normal};
use std::f64::consts::PI;

#[derive(Clone, Debug)]
pub struct MppiConfig {
    pub horizon_steps: usize,
    pub dt: f64,

    pub num_samples: usize,
    pub temperature: f64,
    pub steering_noise: f64,
    pub acceleration_noise: f64,

    pub weight_cte: f64,
    pub weight_epsi: f64,
    pub weight_vel: f64,
    pub weight_steering: f64,
    pub weight_acceleration: f64,

    pub ref_velocity: f64,

    pub turn_first_activation_deg: f64,
    pub turn_first_release_deg: f64,

    /// Distance from the path end over which the reference velocity is
    /// tapered linearly toward zero.
    pub decel_distance: f64,
}

impl Default for MppiConfig {
    fn default() -> Self {
        Self {
            horizon_steps: 20,
            dt: 0.1,
            num_samples: 1000,
            temperature: 1.0,
            steering_noise: 0.5,
            acceleration_noise: 0.3,
            weight_cte: 100.0,
            weight_epsi: 100.0,
            weight_vel: 1.0,
            weight_steering: 10.0,
            weight_acceleration: 5.0,
            ref_velocity: 1.0,
            turn_first_activation_deg: 60.0,
            turn_first_release_deg: 15.0,
            decel_distance: 2.0,
        }
    }
}

pub(crate) struct PathError {
    pub nearest_index: usize,
    pub cte: f64,
    pub epsi: f64,
}

pub(crate) struct ReferenceTrajectory {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub yaw: Vec<f64>,
    pub velocity: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct MppiFollower {
    pub base: ControllerBase,
    pub mppi_config: MppiConfig,
    mean_steering: Vec<f64>,
    mean_acceleration: Vec<f64>,
    predicted_trajectory: Vec<Point>,
    is_turning_in_place: bool,
    rng: StdRng,
}

impl Default for MppiFollower {
    fn default() -> Self {
        Self::with_mppi_config(MppiConfig::default())
    }
}

impl MppiFollower {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_mppi_config(cfg: MppiConfig) -> Self {
        let n = cfg.horizon_steps;
        Self {
            base: ControllerBase::default(),
            mean_steering: vec![0.0; n],
            mean_acceleration: vec![0.0; n],
            predicted_trajectory: Vec::new(),
            is_turning_in_place: false,
            rng: StdRng::from_entropy(),
            mppi_config: cfg,
        }
    }

    pub fn with_seed(cfg: MppiConfig, seed: u64) -> Self {
        let n = cfg.horizon_steps;
        Self {
            base: ControllerBase::default(),
            mean_steering: vec![0.0; n],
            mean_acceleration: vec![0.0; n],
            predicted_trajectory: Vec::new(),
            is_turning_in_place: false,
            rng: StdRng::seed_from_u64(seed),
            mppi_config: cfg,
        }
    }

    pub fn set_mppi_config(&mut self, cfg: MppiConfig) {
        let n = cfg.horizon_steps;
        self.mean_steering = vec![0.0; n];
        self.mean_acceleration = vec![0.0; n];
        self.mppi_config = cfg;
    }

    pub fn predicted_trajectory(&self) -> &[Point] {
        &self.predicted_trajectory
    }

    pub(crate) fn calculate_path_error(&mut self, state: &RobotState) -> PathError {
        let waypoints = &self.base.path.waypoints;
        let mut min_distance = f64::MAX;
        let mut nearest_idx = self.base.path_index;

        for i in self.base.path_index..waypoints.len() {
            let dist = state.pose.point.distance_to(waypoints[i].point);
            if dist < min_distance {
                min_distance = dist;
                nearest_idx = i;
            }
            if i > self.base.path_index && dist > min_distance * 1.5 {
                break;
            }
        }

        let nearest_point = waypoints[nearest_idx].point;
        let path_heading = if nearest_idx + 1 < waypoints.len() {
            let next = waypoints[nearest_idx + 1].point;
            (next.y - nearest_point.y).atan2(next.x - nearest_point.x)
        } else {
            waypoints[nearest_idx].rotation.to_euler().yaw
        };

        let dx = state.pose.point.x - nearest_point.x;
        let dy = state.pose.point.y - nearest_point.y;
        let cte = -dx * path_heading.sin() + dy * path_heading.cos();
        let epsi = normalize_angle(state.pose.rotation.to_euler().yaw - path_heading);

        self.base.path_index = nearest_idx;
        PathError {
            nearest_index: nearest_idx,
            cte,
            epsi,
        }
    }

    pub(crate) fn calculate_reference_trajectory(
        &self,
        error: &PathError,
        working_cfg: &MppiConfig,
    ) -> ReferenceTrajectory {
        let waypoints = &self.base.path.waypoints;
        let start_idx = error.nearest_index;
        let horizon = working_cfg.horizon_steps;

        let mut remaining = vec![0.0_f64; waypoints.len()];
        for i in (0..waypoints.len().saturating_sub(1)).rev() {
            remaining[i] =
                remaining[i + 1] + waypoints[i].point.distance_to(waypoints[i + 1].point);
        }

        let mut ref_traj = ReferenceTrajectory {
            x: Vec::with_capacity(horizon + 1),
            y: Vec::with_capacity(horizon + 1),
            yaw: Vec::with_capacity(horizon + 1),
            velocity: Vec::with_capacity(horizon + 1),
        };

        for i in 0..=horizon {
            let distance_ahead = working_cfg.ref_velocity * working_cfg.dt * i as f64;

            let mut target_idx = start_idx;
            let mut accumulated = 0.0;
            while target_idx + 1 < waypoints.len() && accumulated < distance_ahead {
                accumulated += waypoints[target_idx]
                    .point
                    .distance_to(waypoints[target_idx + 1].point);
                if accumulated < distance_ahead {
                    target_idx += 1;
                }
            }
            let target_idx = target_idx.min(waypoints.len() - 1);

            ref_traj.x.push(waypoints[target_idx].point.x);
            ref_traj.y.push(waypoints[target_idx].point.y);

            let yaw = if target_idx + 1 < waypoints.len() {
                let next = waypoints[target_idx + 1].point;
                let curr = waypoints[target_idx].point;
                (next.y - curr.y).atan2(next.x - curr.x)
            } else {
                waypoints[target_idx].rotation.to_euler().yaw
            };
            ref_traj.yaw.push(yaw);

            let dist_to_end = remaining[target_idx];
            let ref_vel =
                if working_cfg.decel_distance > 1e-6 && dist_to_end < working_cfg.decel_distance {
                    (working_cfg.ref_velocity * (dist_to_end / working_cfg.decel_distance)).max(0.0)
                } else {
                    working_cfg.ref_velocity
                };
            ref_traj.velocity.push(ref_vel);
        }
        ref_traj
    }
}

/// Shared single-sample rollout used by MPPI and its variants (MCA, SOC).
/// Returns the sample cost and optionally the full trajectory.
pub(crate) struct RolloutResult {
    pub cost: f64,
    pub trajectory: Vec<Point>,
}

pub(crate) fn rollout_sample<R: Rng>(
    rng: &mut R,
    state: &RobotState,
    constraints: &RobotConstraints,
    working: &MppiConfig,
    mean_steering: &[f64],
    mean_acceleration: &[f64],
    ref_traj: &ReferenceTrajectory,
    is_diff: bool,
    steering_dist: &Normal<f64>,
    accel_dist: &Normal<f64>,
    out_noise_steering: &mut [f64],
    out_noise_accel: &mut [f64],
    collect_trajectory: bool,
) -> RolloutResult {
    let n = working.horizon_steps;
    let dt = working.dt;

    let mut x = state.pose.point.x;
    let mut y = state.pose.point.y;
    let mut yaw = state.pose.rotation.to_euler().yaw;
    let mut v = state.velocity.linear;

    let mut trajectory = Vec::new();
    if collect_trajectory {
        trajectory.reserve(n + 1);
        trajectory.push(Point::new(x, y, 0.0));
    }

    let mut sample_cost = 0.0;

    for i in 0..n {
        let eps_delta = steering_dist.sample(rng);
        let eps_acc = accel_dist.sample(rng);
        out_noise_steering[i] = eps_delta;
        out_noise_accel[i] = eps_acc;

        let mut delta_or_omega = mean_steering[i] + eps_delta;
        let mut a = mean_acceleration[i] + eps_acc;

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
        v = v.clamp(
            constraints.min_linear_velocity,
            constraints.max_linear_velocity,
        );

        if collect_trajectory {
            trajectory.push(Point::new(x, y, 0.0));
        }

        let ref_idx = (i + 1).min(ref_traj.x.len() - 1);
        let ref_x = ref_traj.x[ref_idx];
        let ref_y = ref_traj.y[ref_idx];
        let ref_yaw = ref_traj.yaw[ref_idx];
        let ref_v = ref_traj.velocity[ref_idx];

        let dx = x - ref_x;
        let dy = y - ref_y;
        let cte = -dx * ref_yaw.sin() + dy * ref_yaw.cos();
        let epsi = normalize_angle(yaw - ref_yaw);
        let vel_error = v - ref_v;

        let mut cost = 0.0;
        cost += working.weight_cte * cte * cte;
        cost += working.weight_epsi * epsi * epsi;
        cost += working.weight_vel * vel_error * vel_error;
        cost += working.weight_steering * delta_or_omega * delta_or_omega;
        cost += working.weight_acceleration * a * a;

        sample_cost += cost * dt;
    }

    RolloutResult {
        cost: sample_cost,
        trajectory,
    }
}

impl Controller for MppiFollower {
    fn compute_control(
        &mut self,
        state: &RobotState,
        goal: &Goal,
        constraints: &RobotConstraints,
        _dt: f64,
        _world: Option<&WorldConstraints>,
    ) -> VelocityCommand {
        if self.base.path.waypoints.is_empty() {
            return VelocityCommand::invalid("no path");
        }

        let error = self.calculate_path_error(state);

        let mut working = self.mppi_config.clone();
        let is_diff = matches!(
            constraints.steering_type,
            SteeringType::Differential | SteeringType::SkidSteer
        );

        if state.turn_first && is_diff {
            let activation = self.mppi_config.turn_first_activation_deg * PI / 180.0;
            let release = self.mppi_config.turn_first_release_deg * PI / 180.0;
            let heading_err_abs = error.epsi.abs();
            if !self.is_turning_in_place {
                if heading_err_abs > activation {
                    self.is_turning_in_place = true;
                }
            } else if heading_err_abs < release {
                self.is_turning_in_place = false;
            }
            if self.is_turning_in_place {
                working.ref_velocity = 0.2;
                working.weight_vel = 50.0;
            }
        } else {
            self.is_turning_in_place = false;
        }

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

        let n = working.horizon_steps;
        let k = working.num_samples;
        let dt = working.dt;

        if self.mean_steering.len() != n {
            self.mean_steering = vec![0.0; n];
            self.mean_acceleration = vec![0.0; n];
        }

        let ref_traj = self.calculate_reference_trajectory(&error, &working);

        let steering_dist = Normal::new(0.0, working.steering_noise.max(1e-9)).unwrap();
        let accel_dist = Normal::new(0.0, working.acceleration_noise.max(1e-9)).unwrap();

        let mut costs = vec![0.0_f64; k];
        let mut noise_steering = vec![vec![0.0_f64; n]; k];
        let mut noise_accel = vec![vec![0.0_f64; n]; k];
        let mut best_cost = f64::INFINITY;
        let mut best_trajectory: Vec<Point> = Vec::new();

        for idx in 0..k {
            let result = rollout_sample(
                &mut self.rng,
                state,
                constraints,
                &working,
                &self.mean_steering,
                &self.mean_acceleration,
                &ref_traj,
                is_diff,
                &steering_dist,
                &accel_dist,
                &mut noise_steering[idx],
                &mut noise_accel[idx],
                idx == 0, // only collect the first sample's trajectory initially
            );
            costs[idx] = result.cost;
            if result.cost < best_cost {
                best_cost = result.cost;
                if !result.trajectory.is_empty() {
                    best_trajectory = result.trajectory;
                }
            }
        }

        let temperature = working.temperature.max(1e-6);
        let beta = 1.0 / temperature;
        let min_cost = costs.iter().cloned().fold(f64::INFINITY, f64::min);

        let mut weights = vec![0.0_f64; k];
        let mut weight_sum = 0.0_f64;
        for i in 0..k {
            let exponent = (-beta * (costs[i] - min_cost)).max(-60.0);
            let w = exponent.exp();
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
                self.mean_steering[i] = self.mean_steering[i].clamp(
                    -constraints.max_angular_velocity,
                    constraints.max_angular_velocity,
                );
            } else {
                self.mean_steering[i] = self.mean_steering[i].clamp(
                    -constraints.max_steering_angle,
                    constraints.max_steering_angle,
                );
            }
            self.mean_acceleration[i] = self.mean_acceleration[i].clamp(
                -constraints.max_linear_acceleration,
                constraints.max_linear_acceleration,
            );
        }

        let steering_or_omega = *self.mean_steering.first().unwrap_or(&0.0);
        let acceleration = *self.mean_acceleration.first().unwrap_or(&0.0);

        // Shift means left for warm-start.
        for i in 0..n.saturating_sub(1) {
            self.mean_steering[i] = self.mean_steering[i + 1];
            self.mean_acceleration[i] = self.mean_acceleration[i + 1];
        }
        if n > 0 {
            self.mean_steering[n - 1] = 0.0;
            self.mean_acceleration[n - 1] = 0.0;
        }

        self.predicted_trajectory = best_trajectory;

        self.base.status.distance_to_goal = state.pose.point.distance_to(goal.target_pose.point);
        self.base.status.cross_track_error = error.cte.abs();
        self.base.status.heading_error = error.epsi.abs();
        self.base.status.goal_reached = false;
        self.base.status.mode = "mppi_tracking".into();

        // Target velocity from actual kinematic integration (same reasoning
        // as MPC): the sampler is already biasing toward deceleration via
        // the tapered reference velocity, so integrating current v with the
        // MPPI acceleration gives a clean output.
        let mut target_velocity = state.velocity.linear + acceleration * dt;
        let min_vel = if cfg.allow_reverse {
            constraints.min_linear_velocity
        } else {
            0.0
        };
        target_velocity = target_velocity.clamp(min_vel, constraints.max_linear_velocity);

        let angular_output = if is_diff {
            steering_or_omega.clamp(
                -constraints.max_angular_velocity,
                constraints.max_angular_velocity,
            )
        } else {
            steering_or_omega.clamp(
                -constraints.max_steering_angle,
                constraints.max_steering_angle,
            )
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

        let linear = if self.is_turning_in_place {
            0.0
        } else {
            linear
        };

        VelocityCommand {
            valid: true,
            status_message: "MPPI tracking".into(),
            linear_velocity: linear,
            angular_velocity: angular,
            ..VelocityCommand::default()
        }
    }

    fn reset(&mut self) {
        self.base.path.waypoints.clear();
        self.base.path_index = 0;
        self.base.status = Default::default();
        let n = self.mppi_config.horizon_steps;
        self.mean_steering = vec![0.0; n];
        self.mean_acceleration = vec![0.0; n];
        self.predicted_trajectory.clear();
        self.is_turning_in_place = false;
    }

    fn get_type(&self) -> &'static str {
        "mppi_follower"
    }

    fn base(&self) -> &ControllerBase {
        &self.base
    }

    fn base_mut(&mut self) -> &mut ControllerBase {
        &mut self.base
    }
}
