use crate::controller::{Controller, ControllerBase, is_goal_reached};
use crate::core::math::normalize_angle;
use crate::types::{
    Goal, OutputUnits, RobotConstraints, RobotState, SteeringType, VelocityCommand,
    WorldConstraints,
};
use datapod::Point;
use std::f64::consts::PI;

#[derive(Clone, Debug)]
pub struct MpcConfig {
    pub horizon_steps: usize,
    pub dt: f64,

    pub weight_cte: f64,
    pub weight_epsi: f64,
    pub weight_vel: f64,
    pub weight_steering: f64,
    pub weight_acceleration: f64,
    pub weight_steering_rate: f64,
    pub weight_acceleration_rate: f64,

    pub ref_velocity: f64,

    pub turn_first_activation_deg: f64,
    pub turn_first_release_deg: f64,

    /// Distance along the path from the end at which `ref_velocity` starts
    /// tapering linearly to zero. Prevents the MPC from overshooting the
    /// final waypoint when the path lacks a terminal-cost term. Set to 0.0
    /// to disable the taper (matches original C++ behaviour).
    pub approach_taper_distance: f64,
}

impl Default for MpcConfig {
    fn default() -> Self {
        Self {
            horizon_steps: 10,
            dt: 0.1,
            weight_cte: 100.0,
            weight_epsi: 100.0,
            weight_vel: 1.0,
            weight_steering: 10.0,
            weight_acceleration: 5.0,
            weight_steering_rate: 500.0,
            weight_acceleration_rate: 50.0,
            ref_velocity: 1.0,
            turn_first_activation_deg: 60.0,
            turn_first_release_deg: 15.0,
            approach_taper_distance: 2.0,
        }
    }
}

struct PathError {
    nearest_index: usize,
    cte: f64,
    epsi: f64,
}

struct ReferenceTrajectory {
    x: Vec<f64>,
    y: Vec<f64>,
    yaw: Vec<f64>,
    velocity: Vec<f64>,
}

struct MpcSolution {
    success: bool,
    steering: f64,
    acceleration: f64,
    predicted_x: Vec<f64>,
    predicted_y: Vec<f64>,
    steering_sequence: Vec<f64>,
    acceleration_sequence: Vec<f64>,
}

#[derive(Clone, Debug)]
pub struct MpcFollower {
    pub base: ControllerBase,
    pub mpc_config: MpcConfig,
    previous_steering: Vec<f64>,
    previous_acceleration: Vec<f64>,
    predicted_trajectory: Vec<Point>,
    is_turning_in_place: bool,
}

impl Default for MpcFollower {
    fn default() -> Self {
        Self::with_mpc_config(MpcConfig::default())
    }
}

impl MpcFollower {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_mpc_config(cfg: MpcConfig) -> Self {
        let n = cfg.horizon_steps;
        Self {
            base: ControllerBase::default(),
            previous_steering: vec![0.0; n],
            previous_acceleration: vec![0.0; n],
            predicted_trajectory: Vec::new(),
            is_turning_in_place: false,
            mpc_config: cfg,
        }
    }

    pub fn set_mpc_config(&mut self, cfg: MpcConfig) {
        let n = cfg.horizon_steps;
        self.previous_steering.resize(n, 0.0);
        self.previous_acceleration.resize(n, 0.0);
        self.mpc_config = cfg;
    }

    pub fn predicted_trajectory(&self) -> &[Point] {
        &self.predicted_trajectory
    }

    fn calculate_path_error(&mut self, state: &RobotState) -> PathError {
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

    fn calculate_reference_trajectory(
        &self,
        error: &PathError,
        base_ref_velocity: f64,
    ) -> ReferenceTrajectory {
        let waypoints = &self.base.path.waypoints;
        let start_idx = error.nearest_index;
        let horizon = self.mpc_config.horizon_steps;

        // Precompute remaining path length from each waypoint to the end so
        // we can taper the reference velocity near the path's final segment.
        // This gives the MPC a terminal deceleration profile without adding
        // a dedicated terminal cost term.
        let mut remaining = vec![0.0_f64; waypoints.len()];
        for i in (0..waypoints.len().saturating_sub(1)).rev() {
            remaining[i] =
                remaining[i + 1] + waypoints[i].point.distance_to(waypoints[i + 1].point);
        }

        let taper = self.mpc_config.approach_taper_distance;

        let mut ref_traj = ReferenceTrajectory {
            x: Vec::with_capacity(horizon + 1),
            y: Vec::with_capacity(horizon + 1),
            yaw: Vec::with_capacity(horizon + 1),
            velocity: Vec::with_capacity(horizon + 1),
        };

        for i in 0..=horizon {
            let distance_ahead = base_ref_velocity * self.mpc_config.dt * i as f64;

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

            let taper_scale = if taper > 1e-6 {
                (remaining[target_idx] / taper).clamp(0.0, 1.0)
            } else {
                1.0
            };
            ref_traj.velocity.push(base_ref_velocity * taper_scale);
        }
        ref_traj
    }

    fn solve_mpc(
        &self,
        state: &RobotState,
        ref_traj: &ReferenceTrajectory,
        constraints: &RobotConstraints,
        config: &MpcConfig,
    ) -> MpcSolution {
        let horizon = config.horizon_steps;
        if horizon == 0 {
            return MpcSolution {
                success: false,
                steering: 0.0,
                acceleration: 0.0,
                predicted_x: Vec::new(),
                predicted_y: Vec::new(),
                steering_sequence: Vec::new(),
                acceleration_sequence: Vec::new(),
            };
        }

        let num_vars = horizon * 2;
        let is_diff = matches!(
            constraints.steering_type,
            SteeringType::Differential | SteeringType::SkidSteer
        );

        let mut u = vec![0.0_f64; num_vars];
        for i in 0..horizon {
            u[2 * i] = self.previous_steering.get(i).copied().unwrap_or(0.0);
            u[2 * i + 1] = self.previous_acceleration.get(i).copied().unwrap_or(0.0);
        }

        let clamp_controls = |ctrl: &mut [f64]| {
            for i in 0..horizon {
                if is_diff {
                    ctrl[2 * i] = ctrl[2 * i]
                        .clamp(-constraints.max_angular_velocity, constraints.max_angular_velocity);
                } else {
                    ctrl[2 * i] = ctrl[2 * i]
                        .clamp(-constraints.max_steering_angle, constraints.max_steering_angle);
                }
                ctrl[2 * i + 1] = ctrl[2 * i + 1].clamp(
                    -constraints.max_linear_acceleration,
                    constraints.max_linear_acceleration,
                );
            }
        };

        let smooth_controls = |ctrl: &mut Vec<f64>| {
            if horizon < 3 {
                return;
            }
            let mut smoothed = vec![0.0_f64; ctrl.len()];
            for dim in 0..2 {
                for i in 0..horizon {
                    let idx_m2 = if i >= 2 { i - 2 } else { 0 };
                    let idx_m1 = if i >= 1 { i - 1 } else { 0 };
                    let idx_0 = i;
                    let idx_p1 = if i + 1 < horizon { i + 1 } else { horizon - 1 };
                    let idx_p2 = if i + 2 < horizon { i + 2 } else { horizon - 1 };

                    let y_m2 = ctrl[2 * idx_m2 + dim];
                    let y_m1 = ctrl[2 * idx_m1 + dim];
                    let y_0 = ctrl[2 * idx_0 + dim];
                    let y_p1 = ctrl[2 * idx_p1 + dim];
                    let y_p2 = ctrl[2 * idx_p2 + dim];

                    smoothed[2 * i + dim] = (1.0 / 35.0)
                        * (-3.0 * y_m2 + 12.0 * y_m1 + 17.0 * y_0 + 12.0 * y_p1 - 3.0 * y_p2);
                }
            }
            *ctrl = smoothed;
            clamp_controls(ctrl);
        };

        let simulate_and_cost =
            |ctrl: &[f64], collect: bool| -> (f64, Option<(Vec<f64>, Vec<f64>)>) {
                let dt = config.dt;
                let lf = constraints.wheelbase;

                let mut x = state.pose.point.x;
                let mut y = state.pose.point.y;
                let mut yaw = state.pose.rotation.to_euler().yaw;
                let mut v = state.velocity.linear;

                let mut traj_x: Vec<f64> = Vec::new();
                let mut traj_y: Vec<f64> = Vec::new();
                if collect {
                    traj_x.push(x);
                    traj_y.push(y);
                }

                let mut cost = 0.0;
                for i in 0..horizon {
                    let steering_or_omega = ctrl[2 * i];
                    let accel = ctrl[2 * i + 1];

                    x += v * yaw.cos() * dt;
                    y += v * yaw.sin() * dt;
                    if is_diff {
                        yaw += steering_or_omega * dt;
                    } else {
                        yaw += v * steering_or_omega / lf * dt;
                    }
                    while yaw > PI {
                        yaw -= 2.0 * PI;
                    }
                    while yaw < -PI {
                        yaw += 2.0 * PI;
                    }
                    v += accel * dt;
                    v = v.clamp(constraints.min_linear_velocity, constraints.max_linear_velocity);

                    if collect {
                        traj_x.push(x);
                        traj_y.push(y);
                    }

                    let ref_len = ref_traj.x.len();
                    let ref_idx = if ref_len == 0 { 0 } else { (i + 1).min(ref_len - 1) };

                    let ref_x = if ref_traj.x.is_empty() { x } else { ref_traj.x[ref_idx] };
                    let ref_y = if ref_traj.y.is_empty() { y } else { ref_traj.y[ref_idx] };
                    let ref_yaw = if ref_traj.yaw.is_empty() { yaw } else { ref_traj.yaw[ref_idx] };
                    let ref_v = if ref_traj.velocity.is_empty() {
                        config.ref_velocity
                    } else {
                        ref_traj.velocity[ref_idx]
                    };

                    let dx = x - ref_x;
                    let dy = y - ref_y;
                    let cte = -dx * ref_yaw.sin() + dy * ref_yaw.cos();
                    let mut epsi = yaw - ref_yaw;
                    while epsi > PI {
                        epsi -= 2.0 * PI;
                    }
                    while epsi < -PI {
                        epsi += 2.0 * PI;
                    }
                    let vel_error = v - ref_v;

                    cost += config.weight_cte * cte * cte;
                    cost += config.weight_epsi * epsi * epsi;
                    cost += config.weight_vel * vel_error * vel_error;
                    cost += config.weight_steering * steering_or_omega * steering_or_omega;
                    cost += config.weight_acceleration * accel * accel;

                    if i > 0 {
                        let prev_steering = ctrl[2 * (i - 1)];
                        let prev_accel = ctrl[2 * (i - 1) + 1];
                        let d_steer = steering_or_omega - prev_steering;
                        let d_acc = accel - prev_accel;
                        cost += config.weight_steering_rate * d_steer * d_steer;
                        cost += config.weight_acceleration_rate * d_acc * d_acc;
                    }
                }

                if collect {
                    (cost, Some((traj_x, traj_y)))
                } else {
                    (cost, None)
                }
            };

        clamp_controls(&mut u);
        smooth_controls(&mut u);

        let (mut best_cost, _) = simulate_and_cost(&u, false);
        if !best_cost.is_finite() {
            return MpcSolution {
                success: false,
                steering: 0.0,
                acceleration: 0.0,
                predicted_x: Vec::new(),
                predicted_y: Vec::new(),
                steering_sequence: Vec::new(),
                acceleration_sequence: Vec::new(),
            };
        }

        let mut best_u = u.clone();
        let mut grad = vec![0.0_f64; num_vars];

        const MAX_ITERATIONS: usize = 15;
        const EPS: f64 = 1e-3;
        const BASE_STEP: f64 = 0.1;

        for _iter in 0..MAX_ITERATIONS {
            for i in 0..num_vars {
                let orig = u[i];
                u[i] = orig + EPS;
                clamp_controls(&mut u);
                let (cost_plus, _) = simulate_and_cost(&u, false);
                u[i] = orig - EPS;
                clamp_controls(&mut u);
                let (cost_minus, _) = simulate_and_cost(&u, false);
                grad[i] = (cost_plus - cost_minus) / (2.0 * EPS);
                u[i] = orig;
            }

            let grad_norm_sq: f64 = grad.iter().map(|g| g * g).sum();
            if grad_norm_sq < 1e-8 {
                break;
            }

            let mut step = BASE_STEP;
            let mut improved = false;
            let mut candidate = vec![0.0_f64; num_vars];

            for _ls in 0..5 {
                for i in 0..num_vars {
                    candidate[i] = u[i] - step * grad[i];
                }
                clamp_controls(&mut candidate);
                smooth_controls(&mut candidate);

                let (cand_cost, _) = simulate_and_cost(&candidate, false);
                if cand_cost.is_finite() && cand_cost < best_cost {
                    best_cost = cand_cost;
                    best_u = candidate.clone();
                    u = candidate.clone();
                    improved = true;
                    break;
                }
                step *= 0.5;
            }

            if !improved {
                break;
            }
        }

        let (_final_cost, traj) = simulate_and_cost(&best_u, true);
        let (predicted_x, predicted_y) = traj.unwrap_or_default();

        let mut steering_sequence = vec![0.0; horizon];
        let mut acceleration_sequence = vec![0.0; horizon];
        for i in 0..horizon {
            steering_sequence[i] = best_u[2 * i];
            acceleration_sequence[i] = best_u[2 * i + 1];
        }

        MpcSolution {
            success: true,
            steering: best_u[0],
            acceleration: best_u[1],
            predicted_x,
            predicted_y,
            steering_sequence,
            acceleration_sequence,
        }
    }
}

impl Controller for MpcFollower {
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

        let mut working = self.mpc_config.clone();
        let is_diff = matches!(
            constraints.steering_type,
            SteeringType::Differential | SteeringType::SkidSteer
        );

        if state.turn_first && is_diff {
            let activation = self.mpc_config.turn_first_activation_deg * PI / 180.0;
            let release = self.mpc_config.turn_first_release_deg * PI / 180.0;
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

        let ref_traj = self.calculate_reference_trajectory(&error, working.ref_velocity);
        let solution = self.solve_mpc(state, &ref_traj, constraints, &working);
        if !solution.success {
            return VelocityCommand::invalid("MPC optimizer failed");
        }

        let horizon = self.mpc_config.horizon_steps;
        if self.previous_steering.len() != horizon {
            self.previous_steering = vec![0.0; horizon];
        }
        if self.previous_acceleration.len() != horizon {
            self.previous_acceleration = vec![0.0; horizon];
        }

        if !solution.steering_sequence.is_empty() && solution.steering_sequence.len() == horizon {
            for i in 0..horizon.saturating_sub(1) {
                self.previous_steering[i] = solution.steering_sequence[i + 1];
                self.previous_acceleration[i] = solution.acceleration_sequence[i + 1];
            }
            self.previous_steering[horizon - 1] = *solution.steering_sequence.last().unwrap();
            self.previous_acceleration[horizon - 1] =
                *solution.acceleration_sequence.last().unwrap();
        }

        self.predicted_trajectory.clear();
        for (x, y) in solution.predicted_x.iter().zip(solution.predicted_y.iter()) {
            self.predicted_trajectory.push(Point::new(*x, *y, 0.0));
        }

        self.base.status.distance_to_goal =
            state.pose.point.distance_to(goal.target_pose.point);
        self.base.status.cross_track_error = error.cte.abs();
        self.base.status.heading_error = error.epsi.abs();
        self.base.status.goal_reached = false;
        self.base.status.mode = "mpc_tracking".into();

        // Integrate current velocity with the MPC's chosen acceleration
        // rather than biasing off the (constant) base ref velocity. Combined
        // with the reference-trajectory velocity taper, this lets the MPC
        // cleanly decelerate into the final waypoint.
        let mut target_velocity = state.velocity.linear + solution.acceleration * working.dt;
        let min_vel = if cfg.allow_reverse {
            constraints.min_linear_velocity
        } else {
            0.0
        };
        target_velocity = target_velocity.clamp(min_vel, constraints.max_linear_velocity);

        let angular_output = if is_diff {
            solution
                .steering
                .clamp(-constraints.max_angular_velocity, constraints.max_angular_velocity)
        } else {
            solution
                .steering
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

        let linear = if self.is_turning_in_place { 0.0 } else { linear };

        VelocityCommand {
            valid: true,
            status_message: "MPC tracking".into(),
            linear_velocity: linear,
            angular_velocity: angular,
            ..VelocityCommand::default()
        }
    }

    fn reset(&mut self) {
        self.base.path.waypoints.clear();
        self.base.path_index = 0;
        self.base.status = Default::default();
        let n = self.mpc_config.horizon_steps;
        self.previous_steering = vec![0.0; n];
        self.previous_acceleration = vec![0.0; n];
        self.predicted_trajectory.clear();
        self.is_turning_in_place = false;
    }

    fn get_type(&self) -> &'static str {
        "mpc_follower"
    }

    fn base(&self) -> &ControllerBase {
        &self.base
    }

    fn base_mut(&mut self) -> &mut ControllerBase {
        &mut self.base
    }
}
