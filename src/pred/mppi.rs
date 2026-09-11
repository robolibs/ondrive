//! MPPI (Williams et al. 2017) plus the rollout, reference-trajectory and
//! output helpers shared by the whole predictive family.
//!
//! Controls are `(steer, accel)` per horizon step: `steer` is a front-wheel
//! steering angle for Ackermann platforms and a yaw rate otherwise.

#![allow(clippy::too_many_arguments)]

use crate::controller::{Controller, ControllerBase, effective_tolerances};
use crate::core::kinematics::{
    can_turn_in_place, finalize, is_ackermann, reverse_allowed, steering_limit, wheelbase,
};
use crate::core::math::normalize_angle;
use crate::core::path::{PathCursor, PathProjection, sample};
use crate::types::{
    ControllerConfig, Goal, Path, RobotConstraints, RobotState, Trajectory, VelocityCommand,
    WorldConstraints,
};
use datapod::Point;
use rand::{SeedableRng, rngs::StdRng};
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
            weight_vel: 50.0,
            weight_steering: 10.0,
            weight_acceleration: 5.0,
            ref_velocity: 1.0,
            turn_first_activation_deg: 60.0,
            turn_first_release_deg: 15.0,
            decel_distance: 2.0,
        }
    }
}

/// Reference states along the path, one per horizon step plus the origin.
#[derive(Clone, Debug, Default)]
pub(crate) struct Reference {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub yaw: Vec<f64>,
    pub v: Vec<f64>,
}

pub(crate) fn build_reference(
    path: &Path,
    cum: &[f64],
    start_s: f64,
    horizon: usize,
    dt: f64,
    ref_velocity: f64,
    decel_distance: f64,
) -> Reference {
    let total = cum.last().copied().unwrap_or(0.0);
    let speeds = &path.speeds;
    let mut r = Reference {
        x: Vec::with_capacity(horizon + 1),
        y: Vec::with_capacity(horizon + 1),
        yaw: Vec::with_capacity(horizon + 1),
        v: Vec::with_capacity(horizon + 1),
    };
    let mut s = start_s;
    for _ in 0..=horizon {
        let (p, h) = sample(&path.waypoints, cum, s);
        let remaining = (total - s).max(0.0);
        let taper = if decel_distance > 1e-6 {
            (remaining / decel_distance).clamp(0.0, 1.0)
        } else {
            1.0
        };
        let mut v = ref_velocity * taper;
        if let Some(cap) = speed_at_arc_length(speeds, cum, s) {
            v = v.min(cap);
        }
        r.x.push(p.x);
        r.y.push(p.y);
        r.yaw.push(h);
        r.v.push(v);
        s += v.max(0.0) * dt;
    }
    r
}

/// Reference states sampled by time on a trajectory, from `t0` every `dt`.
pub(crate) fn build_timed_reference(traj: &Trajectory, t0: f64, horizon: usize, dt: f64) -> Reference {
    let mut r = Reference::default();
    for i in 0..=horizon {
        let s = traj.sample(t0 + i as f64 * dt);
        r.x.push(s.pose.point.x);
        r.y.push(s.pose.point.y);
        r.yaw.push(s.pose.rotation.to_euler().yaw);
        r.v.push(if s.finished { 0.0 } else { s.speed });
    }
    r
}

/// Per-waypoint speed cap at arc length `s`, if the path carries speeds.
pub(crate) fn speed_at_arc_length(speeds: &[f64], cum: &[f64], s: f64) -> Option<f64> {
    if speeds.is_empty() || speeds.len() != cum.len() {
        return None;
    }
    let idx = match cum.binary_search_by(|c| c.partial_cmp(&s).unwrap_or(std::cmp::Ordering::Equal)) {
        Ok(i) => i,
        Err(i) => i.saturating_sub(1),
    }
    .min(speeds.len() - 1);
    speeds.get(idx).copied().filter(|v| *v > 0.0)
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct CostWeights {
    pub cte: f64,
    pub epsi: f64,
    pub vel: f64,
    pub steering: f64,
    pub accel: f64,
}

/// Kinematic model shared by every rollout in one tick.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Model<'a> {
    pub constraints: &'a RobotConstraints,
    pub x0: f64,
    pub y0: f64,
    pub yaw0: f64,
    pub v0: f64,
    pub dt: f64,
    pub ackermann: bool,
    pub v_min: f64,
    pub v_max: f64,
    /// Distance from the rear axle forward to the pose origin.
    pub axle_offset: f64,
}

impl<'a> Model<'a> {
    pub fn new(
        state: &RobotState,
        constraints: &'a RobotConstraints,
        v0: f64,
        dt: f64,
        allow_reverse: bool,
    ) -> Self {
        let (v_min, v_max) = crate::core::kinematics::speed_bounds(constraints, allow_reverse);
        let ackermann = is_ackermann(constraints.steering_type);
        let axle_offset = if ackermann {
            constraints.rear_wheelbase.max(0.0)
        } else {
            0.0
        };
        let yaw0 = state.pose.rotation.to_euler().yaw;
        Self {
            constraints,
            x0: state.pose.point.x - axle_offset * yaw0.cos(),
            y0: state.pose.point.y - axle_offset * yaw0.sin(),
            yaw0,
            v0,
            dt,
            ackermann,
            v_min,
            v_max,
            axle_offset,
        }
    }

    pub fn steer_bound(&self) -> f64 {
        if self.ackermann {
            steering_limit(self.constraints)
        } else {
            self.constraints.max_angular_velocity.abs()
        }
    }

    pub fn accel_bound(&self) -> f64 {
        self.constraints.max_linear_acceleration.abs().max(1e-6)
    }

    pub fn clamp_controls(&self, steer: &mut [f64], accel: &mut [f64]) {
        let sb = self.steer_bound();
        let ab = self.accel_bound();
        for s in steer.iter_mut() {
            *s = s.clamp(-sb, sb);
        }
        for a in accel.iter_mut() {
            *a = a.clamp(-ab, ab);
        }
    }

    /// Pose origin for a rear-axle state.
    pub fn origin(&self, x: f64, y: f64, yaw: f64) -> (f64, f64) {
        (x + self.axle_offset * yaw.cos(), y + self.axle_offset * yaw.sin())
    }

    pub fn step(&self, x: &mut f64, y: &mut f64, yaw: &mut f64, v: &mut f64, steer: f64, accel: f64) {
        *x += *v * yaw.cos() * self.dt;
        *y += *v * yaw.sin() * self.dt;
        let w_max = self.constraints.max_angular_velocity.abs();
        if self.ackermann {
            let w = (*v * steer.tan() / wheelbase(self.constraints)).clamp(-w_max, w_max);
            *yaw += w * self.dt;
        } else {
            *yaw += steer * self.dt;
        }
        *yaw = normalize_angle(*yaw);
        *v = (*v + accel * self.dt).clamp(self.v_min, self.v_max);
    }
}

/// Roll out one control sequence and return `(cost, trajectory)`. `extra`
/// receives `(step index, x, y)` after every integration step and returns an
/// additional stage cost (obstacles, risk, ...).
pub(crate) fn rollout(
    model: &Model,
    steer: &[f64],
    accel: &[f64],
    reference: &Reference,
    w: &CostWeights,
    extra: &dyn Fn(usize, f64, f64) -> f64,
    collect: bool,
) -> (f64, Vec<Point>) {
    let n = steer.len().min(accel.len());
    let sb = model.steer_bound();
    let ab = model.accel_bound();
    let (mut x, mut y, mut yaw, mut v) = (model.x0, model.y0, model.yaw0, model.v0);
    let mut traj = Vec::new();
    if collect {
        traj.reserve(n + 1);
        let (ox, oy) = model.origin(x, y, yaw);
        traj.push(Point::new(ox, oy, 0.0));
    }
    let mut cost = 0.0;
    for i in 0..n {
        let s = steer[i].clamp(-sb, sb);
        let a = accel[i].clamp(-ab, ab);
        model.step(&mut x, &mut y, &mut yaw, &mut v, s, a);
        let (x, y) = model.origin(x, y, yaw);
        if collect {
            traj.push(Point::new(x, y, 0.0));
        }
        let ri = (i + 1).min(reference.x.len().saturating_sub(1));
        let (rx, ry, ryaw, rv) = if reference.x.is_empty() {
            (x, y, yaw, v)
        } else {
            (reference.x[ri], reference.y[ri], reference.yaw[ri], reference.v[ri])
        };
        let dx = x - rx;
        let dy = y - ry;
        let cte = -dx * ryaw.sin() + dy * ryaw.cos();
        let along = dx * ryaw.cos() + dy * ryaw.sin();
        let epsi = normalize_angle(yaw - ryaw);
        let ve = v - rv;
        let mut stage = w.cte * (cte * cte + 0.25 * along * along)
            + w.epsi * epsi * epsi
            + w.vel * ve * ve
            + w.steering * s * s
            + w.accel * a * a;
        stage += extra(i, x, y);
        cost += stage * model.dt;
    }
    (cost, traj)
}

/// Result of the shared pre-processing every predictive controller runs.
pub(crate) struct Prepared {
    pub proj: PathProjection,
    pub pos_tol: f64,
    pub ang_tol: f64,
    pub epsi: f64,
    pub allow_reverse: bool,
}

/// Projection, status bookkeeping and arrival handling. `Err(cmd)` carries
/// the command to return immediately (invalid input, or stop/align).
pub(crate) fn prepare(
    base: &mut ControllerBase,
    cursor: &mut PathCursor,
    state: &RobotState,
    goal: &Goal,
    constraints: &RobotConstraints,
    dt: f64,
) -> Result<Prepared, VelocityCommand> {
    if !(dt.is_finite() && dt > 0.0) {
        return Err(VelocityCommand::invalid("dt must be positive and finite"));
    }
    if base.path.waypoints.is_empty() {
        return Err(VelocityCommand::invalid("no path"));
    }
    let cfg = base.config.clone();
    let allow_reverse = reverse_allowed(&cfg, state);
    let Some(proj) = cursor.project(&base.path.waypoints, state.pose.point, base.path_index) else {
        return Err(VelocityCommand::invalid("no path"));
    };
    base.path_index = proj.segment;
    let (pos_tol, ang_tol) = effective_tolerances(goal, &cfg);
    let passed_end = proj.beyond_end && proj.distance < 2.0 * pos_tol;
    if let Some(cmd) = base.arrival(state, goal, constraints, passed_end) {
        return Err(cmd);
    }
    let yaw = state.pose.rotation.to_euler().yaw;
    let epsi = normalize_angle(proj.heading - yaw);
    base.status.cross_track_error = proj.lateral_error;
    base.status.heading_error = epsi;
    base.status.goal_reached = false;
    Ok(Prepared {
        proj,
        pos_tol,
        ang_tol,
        epsi,
        allow_reverse,
    })
}

/// Speed feedback with a fallback on the last commanded speed for callers
/// that do not report velocity.
pub(crate) fn current_speed(state: &RobotState, last_commanded: f64) -> f64 {
    if state.velocity.linear.abs() > 1e-9 {
        state.velocity.linear
    } else {
        last_commanded
    }
}

/// Turn-in-place hysteresis shared by the predictive controllers.
pub(crate) fn update_turn_in_place(
    flag: &mut bool,
    state: &RobotState,
    constraints: &RobotConstraints,
    epsi: f64,
    activation_deg: f64,
    release_deg: f64,
) -> bool {
    if state.turn_first && can_turn_in_place(constraints.steering_type) {
        let activation = activation_deg * PI / 180.0;
        let release = release_deg * PI / 180.0;
        if !*flag {
            if epsi.abs() > activation {
                *flag = true;
            }
        } else if epsi.abs() < release {
            *flag = false;
        }
    } else {
        *flag = false;
    }
    *flag
}

/// Convert the first `(steer, accel)` of a solution into a command.
pub(crate) fn command_from_controls(
    v_now: f64,
    steer: f64,
    accel: f64,
    dt: f64,
    v_cap: f64,
    constraints: &RobotConstraints,
    cfg: &ControllerConfig,
    allow_reverse: bool,
    turning_in_place: bool,
    epsi: f64,
    message: &str,
) -> VelocityCommand {
    if turning_in_place {
        let omega = cfg.kp_angular.max(0.1) * epsi;
        return finalize(0.0, omega, constraints, cfg, allow_reverse, "Turning to align");
    }
    let cap = if v_cap > 0.0 { v_cap } else { f64::INFINITY };
    let v = (v_now + accel * dt).clamp(-cap, cap);
    let omega = if is_ackermann(constraints.steering_type) {
        v * steer.tan() / wheelbase(constraints)
    } else {
        steer
    };
    finalize(v, omega, constraints, cfg, allow_reverse, message)
}

/// Importance weights `exp(-(S - min S) / lambda)`, normalised.
pub(crate) fn importance_weights(costs: &[f64], temperature: f64) -> Vec<f64> {
    let lambda = temperature.max(1e-6);
    let min = costs.iter().cloned().fold(f64::INFINITY, f64::min);
    let mut w: Vec<f64> = costs
        .iter()
        .map(|c| (-(c - min) / lambda).max(-700.0).exp())
        .collect();
    let sum: f64 = w.iter().sum();
    if sum > 1e-300 {
        for x in w.iter_mut() {
            *x /= sum;
        }
    } else {
        let n = w.len().max(1) as f64;
        for x in w.iter_mut() {
            *x = 1.0 / n;
        }
    }
    w
}

#[derive(Clone, Debug)]
pub struct MppiFollower {
    pub base: ControllerBase,
    pub mppi_config: MppiConfig,
    cursor: PathCursor,
    mean_steering: Vec<f64>,
    mean_acceleration: Vec<f64>,
    predicted_trajectory: Vec<Point>,
    is_turning_in_place: bool,
    last_v: f64,
    shift_accum: f64,
    trajectory: Option<Trajectory>,
    clock: f64,
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
        Self::build(cfg, StdRng::from_entropy())
    }

    pub fn with_seed(cfg: MppiConfig, seed: u64) -> Self {
        Self::build(cfg, StdRng::seed_from_u64(seed))
    }

    fn build(cfg: MppiConfig, rng: StdRng) -> Self {
        let n = cfg.horizon_steps;
        Self {
            base: ControllerBase::default(),
            mean_steering: vec![0.0; n],
            mean_acceleration: vec![0.0; n],
            predicted_trajectory: Vec::new(),
            is_turning_in_place: false,
            last_v: 0.0,
            shift_accum: 0.0,
            trajectory: None,
            clock: 0.0,
            rng,
            cursor: PathCursor::default(),
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

    pub(crate) fn weights(&self, cfg: &MppiConfig) -> CostWeights {
        CostWeights {
            cte: cfg.weight_cte,
            epsi: cfg.weight_epsi,
            vel: cfg.weight_vel,
            steering: cfg.weight_steering,
            accel: cfg.weight_acceleration,
        }
    }

    /// Full MPPI tick with an additional stage cost `extra(step, x, y)` and
    /// a scale on the executed speed (risk slowdown).
    pub(crate) fn step_with(
        &mut self,
        state: &RobotState,
        goal: &Goal,
        constraints: &RobotConstraints,
        dt: f64,
        extra: &dyn Fn(usize, f64, f64) -> f64,
        ref_scale: f64,
        message: &str,
        mode: &str,
    ) -> VelocityCommand {
        let prep = match prepare(&mut self.base, &mut self.cursor, state, goal, constraints, dt) {
            Ok(p) => p,
            Err(cmd) => {
                self.predicted_trajectory.clear();
                return cmd;
            }
        };
        let mut working = self.mppi_config.clone();
        let turning = update_turn_in_place(
            &mut self.is_turning_in_place,
            state,
            constraints,
            prep.epsi,
            working.turn_first_activation_deg,
            working.turn_first_release_deg,
        );
        if turning {
            working.ref_velocity = 0.2;
            working.weight_vel = 50.0;
        }
        let speed_scale = ref_scale.clamp(0.0, 1.0);

        let n = working.horizon_steps.max(1);
        let k = working.num_samples.max(1);
        if self.mean_steering.len() != n {
            self.mean_steering = vec![0.0; n];
            self.mean_acceleration = vec![0.0; n];
        }

        let v_now = current_speed(state, self.last_v);
        let model = Model::new(state, constraints, v_now, working.dt, prep.allow_reverse);
        let reference = match &self.trajectory {
            Some(traj) => build_timed_reference(traj, self.clock, n, working.dt),
            None => build_reference(
                &self.base.path,
                &self.cursor.cum,
                prep.proj.arc_length,
                n,
                working.dt,
                working.ref_velocity.min(constraints.max_linear_velocity),
                working.decel_distance,
            ),
        };
        let w = self.weights(&working);
        model.clamp_controls(&mut self.mean_steering, &mut self.mean_acceleration);

        let sigma_s = working.steering_noise.max(1e-6);
        let sigma_a = working.acceleration_noise.max(1e-6);
        let dist_s = Normal::new(0.0, sigma_s).unwrap();
        let dist_a = Normal::new(0.0, sigma_a).unwrap();
        let lambda = working.temperature.max(1e-6);

        let mut costs = vec![0.0; k];
        let mut noise_s = vec![vec![0.0; n]; k];
        let mut noise_a = vec![vec![0.0; n]; k];
        let mut steer = vec![0.0; n];
        let mut accel = vec![0.0; n];
        for j in 0..k {
            let mut control_cost = 0.0;
            for t in 0..n {
                let es = dist_s.sample(&mut self.rng);
                let ea = dist_a.sample(&mut self.rng);
                noise_s[j][t] = es;
                noise_a[j][t] = ea;
                steer[t] = self.mean_steering[t] + es;
                accel[t] = self.mean_acceleration[t] + ea;
                control_cost += self.mean_steering[t] * es / (sigma_s * sigma_s)
                    + self.mean_acceleration[t] * ea / (sigma_a * sigma_a);
            }
            let (c, _) = rollout(&model, &steer, &accel, &reference, &w, extra, false);
            costs[j] = c + lambda * control_cost;
        }

        let weights = importance_weights(&costs, lambda);
        for t in 0..n {
            let mut ds = 0.0;
            let mut da = 0.0;
            for j in 0..k {
                ds += weights[j] * noise_s[j][t];
                da += weights[j] * noise_a[j][t];
            }
            self.mean_steering[t] += ds;
            self.mean_acceleration[t] += da;
        }
        model.clamp_controls(&mut self.mean_steering, &mut self.mean_acceleration);

        let (_, traj) = rollout(
            &model,
            &self.mean_steering,
            &self.mean_acceleration,
            &reference,
            &w,
            extra,
            true,
        );
        self.predicted_trajectory = traj;

        let steer0 = self.mean_steering[0];
        let accel0 = self.mean_acceleration[0];
        self.shift_accum += dt;
        if self.shift_accum >= working.dt {
            self.shift_accum -= working.dt;
            self.mean_steering.rotate_left(1);
            self.mean_acceleration.rotate_left(1);
            if n > 1 {
                self.mean_steering[n - 1] = self.mean_steering[n - 2];
                self.mean_acceleration[n - 1] = self.mean_acceleration[n - 2];
            }
        }

        let dt_apply = dt.min(working.dt);
        let v_cap = speed_at_arc_length(&self.base.path.speeds, &self.cursor.cum, prep.proj.arc_length)
            .unwrap_or(f64::INFINITY)
            .min(speed_scale * constraints.max_linear_velocity.max(0.0));
        self.base.status.mode = mode.into();
        let cmd = command_from_controls(
            v_now,
            steer0,
            accel0,
            dt_apply,
            v_cap,
            constraints,
            &self.base.config,
            prep.allow_reverse,
            turning,
            prep.epsi,
            message,
        );
        self.last_v = if cmd.valid && matches!(self.base.config.output_units, crate::types::OutputUnits::Physical) {
            cmd.linear_velocity
        } else {
            (v_now + accel0 * dt_apply).clamp(-v_cap, v_cap)
        };
        cmd
    }
}

impl Controller for MppiFollower {
    fn compute_control(
        &mut self,
        state: &RobotState,
        goal: &Goal,
        constraints: &RobotConstraints,
        dt: f64,
        _world: Option<&WorldConstraints>,
    ) -> VelocityCommand {
        self.step_with(
            state,
            goal,
            constraints,
            dt,
            &|_, _, _| 0.0,
            1.0,
            "MPPI tracking",
            "mppi_tracking",
        )
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
        let n = self.mppi_config.horizon_steps;
        self.mean_steering = vec![0.0; n];
        self.mean_acceleration = vec![0.0; n];
        self.predicted_trajectory.clear();
        self.is_turning_in_place = false;
        self.shift_accum = 0.0;
    }

    fn reset(&mut self) {
        self.set_path(Path::default());
        self.last_v = 0.0;
    }

    fn get_type(&self) -> &'static str {
        "mppi_follower"
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
