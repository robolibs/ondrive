//! TEB — Timed Elastic Band (simplified).
//!
//! Maintains a "band" of `n_poses` poses ahead of the robot plus `n_poses-1`
//! time-step variables Δt_i between them. The band is optimised on every
//! tick via projected gradient descent on a composite cost:
//!
//!   * total time       — sum(Δt_i)   (minimise traversal time)
//!   * velocity limits  — quadratic penalty on |v_i| − v_max above 0
//!   * turn-rate limits — quadratic penalty on |ω_i| − ω_max above 0
//!   * path deviation   — distance from each band pose to the reference path
//!   * obstacles        — quadratic penalty when clearance < margin
//!   * kinematic        — cost for velocity not aligned with band yaw
//!   * goal             — attractor on the final band pose
//!
//! The first band segment then yields the velocity command:
//!   v = dist(p_0, p_1) / Δt_0   (signed by yaw alignment)
//!   ω = normalize(yaw_1 − yaw_0) / Δt_0

use crate::controller::{Controller, ControllerBase, is_goal_reached};
use crate::core::math::normalize_angle;
use crate::types::{
    Goal, RobotConstraints, RobotState, SteeringType, VelocityCommand, WorldConstraints,
};
use datapod::Point;

#[derive(Clone, Debug)]
pub struct TebConfig {
    pub n_poses: usize,
    pub dt_nominal: f64,
    pub iterations: usize,
    pub step_size: f64,

    pub weight_time: f64,
    pub weight_velocity_limit: f64,
    pub weight_angular_limit: f64,
    pub weight_path_deviation: f64,
    pub weight_obstacle: f64,
    pub weight_kinematic: f64,
    pub weight_goal: f64,

    pub obstacle_margin: f64,
    pub dt_min: f64,
    pub dt_max: f64,
}

impl Default for TebConfig {
    fn default() -> Self {
        Self {
            n_poses: 6,
            dt_nominal: 0.2,
            iterations: 10,
            step_size: 0.02,
            weight_time: 1.0,
            weight_velocity_limit: 50.0,
            weight_angular_limit: 50.0,
            weight_path_deviation: 20.0,
            weight_obstacle: 100.0,
            weight_kinematic: 5.0,
            weight_goal: 5.0,
            obstacle_margin: 0.3,
            dt_min: 0.05,
            dt_max: 1.0,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct TebFollower {
    pub base: ControllerBase,
    pub teb_config: TebConfig,
    // Persistent band so we can warm-start across ticks.
    band_x: Vec<f64>,
    band_y: Vec<f64>,
    band_yaw: Vec<f64>,
    band_dt: Vec<f64>,
    band_initialised: bool,
}

impl TebFollower {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_teb_config(cfg: TebConfig) -> Self {
        Self {
            base: ControllerBase::default(),
            teb_config: cfg,
            ..Default::default()
        }
    }

    pub fn set_teb_config(&mut self, cfg: TebConfig) {
        self.band_initialised = false;
        self.teb_config = cfg;
    }

    pub fn band_poses(&self) -> Vec<Point> {
        self.band_x
            .iter()
            .zip(self.band_y.iter())
            .map(|(&x, &y)| Point::new(x, y, 0.0))
            .collect()
    }

    /// Initialise the band from the current state by stepping along the
    /// reference path at the nominal dt and velocity.
    fn initialise_band(&mut self, state: &RobotState, goal: &Goal) {
        let n = self.teb_config.n_poses;
        self.band_x = vec![0.0; n];
        self.band_y = vec![0.0; n];
        self.band_yaw = vec![0.0; n];
        self.band_dt = vec![self.teb_config.dt_nominal; n - 1];

        let waypoints = &self.base.path.waypoints;
        if waypoints.is_empty() {
            // Fall back to a straight line toward the goal.
            for i in 0..n {
                let t = i as f64 / (n - 1) as f64;
                self.band_x[i] =
                    state.pose.point.x + t * (goal.target_pose.point.x - state.pose.point.x);
                self.band_y[i] =
                    state.pose.point.y + t * (goal.target_pose.point.y - state.pose.point.y);
                self.band_yaw[i] = state.pose.rotation.to_euler().yaw;
            }
            self.band_initialised = true;
            return;
        }

        // Walk the reference path by ~1 m per step, starting at the nearest
        // waypoint.
        let (start_idx, _) = waypoints
            .iter()
            .enumerate()
            .map(|(i, p)| (i, state.pose.point.distance_to(p.point)))
            .min_by(|a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or((0, 0.0));
        self.base.path_index = start_idx;

        let step = self.teb_config.dt_nominal * 1.0; // nominal ~1 m/s
        self.band_x[0] = state.pose.point.x;
        self.band_y[0] = state.pose.point.y;
        self.band_yaw[0] = state.pose.rotation.to_euler().yaw;

        let mut target_idx = start_idx;
        let mut accumulated = 0.0;
        for i in 1..n {
            let want = i as f64 * step;
            while target_idx + 1 < waypoints.len() && accumulated < want {
                accumulated += waypoints[target_idx]
                    .point
                    .distance_to(waypoints[target_idx + 1].point);
                if accumulated < want {
                    target_idx += 1;
                }
            }
            let idx = target_idx.min(waypoints.len() - 1);
            self.band_x[i] = waypoints[idx].point.x;
            self.band_y[i] = waypoints[idx].point.y;
            self.band_yaw[i] = if idx + 1 < waypoints.len() {
                let next = waypoints[idx + 1].point;
                (next.y - waypoints[idx].point.y)
                    .atan2(next.x - waypoints[idx].point.x)
            } else {
                waypoints[idx].rotation.to_euler().yaw
            };
        }
        self.band_initialised = true;
    }

    /// Slide the band forward one segment each tick (drop p_0, append a
    /// fresh tail sample).
    fn shift_and_anchor(&mut self, state: &RobotState) {
        if !self.band_initialised {
            return;
        }
        let n = self.teb_config.n_poses;
        // Shift by one segment.
        for i in 0..n - 1 {
            self.band_x[i] = self.band_x[i + 1];
            self.band_y[i] = self.band_y[i + 1];
            self.band_yaw[i] = self.band_yaw[i + 1];
        }
        for i in 0..n - 2 {
            self.band_dt[i] = self.band_dt[i + 1];
        }
        // New tail extrapolates the last segment.
        let last = n - 1;
        let prev = n - 2;
        let dx = self.band_x[prev] - (if n >= 3 { self.band_x[n - 3] } else { self.band_x[0] });
        let dy = self.band_y[prev] - (if n >= 3 { self.band_y[n - 3] } else { self.band_y[0] });
        self.band_x[last] = self.band_x[prev] + dx;
        self.band_y[last] = self.band_y[prev] + dy;
        self.band_yaw[last] = self.band_yaw[prev];
        // Anchor first pose to the actual robot state.
        self.band_x[0] = state.pose.point.x;
        self.band_y[0] = state.pose.point.y;
        self.band_yaw[0] = state.pose.rotation.to_euler().yaw;
    }

    fn cost(
        cfg: &TebConfig,
        band_x: &[f64],
        band_y: &[f64],
        band_yaw: &[f64],
        band_dt: &[f64],
        path_waypoints: &[datapod::Pose],
        goal_point: Point,
        obstacles: &[(f64, f64, f64)],
        constraints: &RobotConstraints,
    ) -> f64 {
        let n = band_x.len();
        let mut cost = 0.0;

        for i in 0..n - 1 {
            let dt = band_dt[i].max(1e-4);
            cost += cfg.weight_time * dt;

            let dx = band_x[i + 1] - band_x[i];
            let dy = band_y[i + 1] - band_y[i];
            let seg_len = (dx * dx + dy * dy).sqrt();
            let v = seg_len / dt;
            let v_excess = (v - constraints.max_linear_velocity).max(0.0);
            cost += cfg.weight_velocity_limit * v_excess * v_excess;

            let dyaw = normalize_angle(band_yaw[i + 1] - band_yaw[i]);
            let w = dyaw / dt;
            let w_excess = (w.abs() - constraints.max_angular_velocity).max(0.0);
            cost += cfg.weight_angular_limit * w_excess * w_excess;

            // Kinematic: the segment direction should align with yaw[i].
            if seg_len > 1e-6 {
                let seg_heading = dy.atan2(dx);
                let mis = normalize_angle(seg_heading - band_yaw[i]).abs();
                cost += cfg.weight_kinematic * mis * mis;
            }
        }

        // Path deviation for each band pose (except pose 0, which is anchored).
        if !path_waypoints.is_empty() {
            for i in 1..n {
                let mut min_d2 = f64::MAX;
                for wp in path_waypoints {
                    let dx = band_x[i] - wp.point.x;
                    let dy = band_y[i] - wp.point.y;
                    let d2 = dx * dx + dy * dy;
                    if d2 < min_d2 {
                        min_d2 = d2;
                    }
                }
                cost += cfg.weight_path_deviation * min_d2;
            }
        }

        // Obstacle penalty.
        for i in 0..n {
            for (ox, oy, r) in obstacles {
                let dx = band_x[i] - ox;
                let dy = band_y[i] - oy;
                let d = (dx * dx + dy * dy).sqrt() - r;
                let margin = cfg.obstacle_margin;
                if d < margin {
                    let shortfall = margin - d;
                    cost += cfg.weight_obstacle * shortfall * shortfall;
                }
            }
        }

        // Goal attractor on the final band pose.
        let dx = band_x[n - 1] - goal_point.x;
        let dy = band_y[n - 1] - goal_point.y;
        cost += cfg.weight_goal * (dx * dx + dy * dy);

        cost
    }
}

impl Controller for TebFollower {
    fn compute_control(
        &mut self,
        state: &RobotState,
        goal: &Goal,
        constraints: &RobotConstraints,
        _dt: f64,
        world: Option<&WorldConstraints>,
    ) -> VelocityCommand {
        let cfg = self.base.config.clone();

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

        let is_diff = matches!(
            constraints.steering_type,
            SteeringType::Differential | SteeringType::SkidSteer
        );

        if !self.band_initialised || self.band_x.len() != self.teb_config.n_poses {
            self.initialise_band(state, goal);
        } else {
            self.shift_and_anchor(state);
        }

        let obstacles: Vec<(f64, f64, f64)> = world
            .map(|w| {
                w.obstacles
                    .iter()
                    .filter_map(|o| {
                        o.modes.first().and_then(|m| {
                            if m.mean_x.is_empty() {
                                None
                            } else {
                                Some((m.mean_x[0], m.mean_y[0], o.radius))
                            }
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();

        let n = self.teb_config.n_poses;
        let goal_point = goal.target_pose.point;
        let cfg_teb = self.teb_config.clone();
        let path_waypoints_snapshot = self.base.path.waypoints.clone();

        // Projected-gradient descent with finite-difference gradients. We
        // optimise poses 1..n and Δt 0..n-1 (pose 0 is anchored to state).
        let num_vars = (n - 1) * 3 + (n - 1);
        let eps = 1e-3;

        for _iter in 0..cfg_teb.iterations {
            let base_cost = Self::cost(
                &cfg_teb,
                &self.band_x,
                &self.band_y,
                &self.band_yaw,
                &self.band_dt,
                &path_waypoints_snapshot,
                goal_point,
                &obstacles,
                constraints,
            );

            let mut grad = vec![0.0_f64; num_vars];

            // Gradient via central differences.
            for d in 0..num_vars {
                let var_id = d;
                // Encoding: first (n-1)*3 are [x1,y1,yaw1, x2,y2,yaw2, ...],
                // next (n-1) are [dt0, dt1, ...].
                let pose_block = (n - 1) * 3;
                let (c_plus, c_minus) = if var_id < pose_block {
                    let i = var_id / 3 + 1;
                    let k = var_id % 3;
                    let orig = match k {
                        0 => self.band_x[i],
                        1 => self.band_y[i],
                        _ => self.band_yaw[i],
                    };
                    let set = |this: &mut Self, val: f64| match k {
                        0 => this.band_x[i] = val,
                        1 => this.band_y[i] = val,
                        _ => this.band_yaw[i] = val,
                    };
                    set(self, orig + eps);
                    let cp = Self::cost(
                        &cfg_teb,
                        &self.band_x,
                        &self.band_y,
                        &self.band_yaw,
                        &self.band_dt,
                        &path_waypoints_snapshot,
                        goal_point,
                        &obstacles,
                        constraints,
                    );
                    set(self, orig - eps);
                    let cm = Self::cost(
                        &cfg_teb,
                        &self.band_x,
                        &self.band_y,
                        &self.band_yaw,
                        &self.band_dt,
                        &path_waypoints_snapshot,
                        goal_point,
                        &obstacles,
                        constraints,
                    );
                    set(self, orig);
                    (cp, cm)
                } else {
                    let i = var_id - pose_block;
                    let orig = self.band_dt[i];
                    self.band_dt[i] = orig + eps;
                    let cp = Self::cost(
                        &cfg_teb,
                        &self.band_x,
                        &self.band_y,
                        &self.band_yaw,
                        &self.band_dt,
                        &path_waypoints_snapshot,
                        goal_point,
                        &obstacles,
                        constraints,
                    );
                    self.band_dt[i] = orig - eps;
                    let cm = Self::cost(
                        &cfg_teb,
                        &self.band_x,
                        &self.band_y,
                        &self.band_yaw,
                        &self.band_dt,
                        &path_waypoints_snapshot,
                        goal_point,
                        &obstacles,
                        constraints,
                    );
                    self.band_dt[i] = orig;
                    (cp, cm)
                };
                grad[d] = (c_plus - c_minus) / (2.0 * eps);
            }

            // Apply step, clamp dt to [dt_min, dt_max].
            let step = cfg_teb.step_size;
            for d in 0..num_vars {
                let pose_block = (n - 1) * 3;
                if d < pose_block {
                    let i = d / 3 + 1;
                    let k = d % 3;
                    let delta = step * grad[d];
                    match k {
                        0 => self.band_x[i] -= delta,
                        1 => self.band_y[i] -= delta,
                        _ => {
                            self.band_yaw[i] -= delta;
                            self.band_yaw[i] = normalize_angle(self.band_yaw[i]);
                        }
                    }
                } else {
                    let i = d - pose_block;
                    self.band_dt[i] -= step * grad[d];
                    self.band_dt[i] = self.band_dt[i].clamp(cfg_teb.dt_min, cfg_teb.dt_max);
                }
            }

            let new_cost = Self::cost(
                &cfg_teb,
                &self.band_x,
                &self.band_y,
                &self.band_yaw,
                &self.band_dt,
                &path_waypoints_snapshot,
                goal_point,
                &obstacles,
                constraints,
            );
            if new_cost >= base_cost {
                // No improvement — stop early to avoid divergence.
                break;
            }
        }

        // Extract first-segment command.
        let dt0 = self.band_dt[0].max(1e-3);
        let dx = self.band_x[1] - self.band_x[0];
        let dy = self.band_y[1] - self.band_y[0];
        let seg_len = (dx * dx + dy * dy).sqrt();
        let seg_heading = if seg_len > 1e-6 { dy.atan2(dx) } else { self.band_yaw[0] };
        let yaw0 = self.band_yaw[0];
        // Sign: if segment direction is opposite to current yaw, the
        // command is a reverse move.
        let alignment = (seg_heading - yaw0).cos();
        let v_sign = if alignment >= 0.0 { 1.0 } else { -1.0 };
        let mut v = v_sign * (seg_len / dt0);
        let min_vel = if cfg.allow_reverse {
            constraints.min_linear_velocity
        } else {
            0.0
        };
        v = v.clamp(min_vel, constraints.max_linear_velocity);

        let mut w = normalize_angle(self.band_yaw[1] - self.band_yaw[0]) / dt0;
        w = w.clamp(
            -constraints.max_angular_velocity,
            constraints.max_angular_velocity,
        );

        let angular_output = if is_diff {
            w
        } else {
            // For Ackermann, band-yaw change over one dt maps to a steering
            // angle via ω = v · tan(δ) / L  ⇒  δ = atan(ω L / v).
            let lf = if constraints.wheelbase > 0.0 {
                constraints.wheelbase
            } else {
                1.0
            };
            let delta = if v.abs() > 1e-3 {
                (w * lf / v).atan()
            } else {
                0.0
            };
            let delta = delta.clamp(
                -constraints.max_steering_angle,
                constraints.max_steering_angle,
            );
            let kp_steer = 2.0;
            (kp_steer * delta).clamp(
                -constraints.max_angular_velocity,
                constraints.max_angular_velocity,
            )
        };

        self.base.status.distance_to_goal = dist_to_goal;
        self.base.status.goal_reached = false;
        self.base.status.mode = "teb".into();

        VelocityCommand {
            valid: true,
            status_message: "TEB tracking".into(),
            linear_velocity: v,
            angular_velocity: angular_output,
            ..VelocityCommand::default()
        }
    }

    fn reset(&mut self) {
        self.base.path.waypoints.clear();
        self.base.path_index = 0;
        self.base.status = Default::default();
        self.band_initialised = false;
    }

    fn get_type(&self) -> &'static str {
        "teb_follower"
    }

    fn base(&self) -> &ControllerBase {
        &self.base
    }

    fn base_mut(&mut self) -> &mut ControllerBase {
        &mut self.base
    }
}
