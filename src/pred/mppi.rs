//! MPPI (Williams et al. 2017) plus the rollout, reference-trajectory and
//! output helpers shared by the whole predictive family.
//!
//! Controls are `(steer, accel, accel_lat)` per horizon step: `steer` is a
//! front-wheel steering angle for Ackermann platforms and a yaw rate
//! otherwise; `accel_lat` (lateral acceleration) is only free for a
//! holonomic platform — its bound is exactly zero otherwise, so `vy` never
//! leaves zero and every formula below reduces to the non-holonomic one
//! bit-for-bit.

#![allow(clippy::too_many_arguments)]

use crate::controller::{Controller, ControllerBase, effective_tolerances};
use crate::core::kinematics::{
    can_turn_in_place, finalize, finalize_holonomic, is_ackermann, is_holonomic, reverse_allowed,
    planning, steering_limit, wheelbase,
};
use crate::core::math::normalize_angle;
use crate::core::obstacles::{CollisionChecker, modes_at, robot_radius};
use crate::core::path::{PathCursor, PathProjection, sample};
use crate::pred::mpc::{DETOUR_MARGIN, DETOUR_RAMP, DETOURED_WEIGHT};
use crate::types::{
    ControllerConfig, DebugFrame, DebugLevel, Goal, Obstacle, Path, RobotConstraints, RobotState, Trajectory, VelocityCommand,
    WorldConstraints,
};
use datapod::Point;
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::Distribution;
use std::f64::consts::PI;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
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
    /// Cost per metre of path a rollout leaves undone within the horizon.
    pub weight_progress: f64,

    pub ref_velocity: f64,

    pub turn_first_activation_deg: f64,
    pub turn_first_release_deg: f64,

    /// Distance before the path end and each cusp over which the target
    /// speed slows to zero.
    pub decel_distance: f64,

    /// Share of each noise step carried over from the previous one.
    pub noise_correlation: f64,
    /// Weight of the control-cost term relative to the temperature.
    pub control_cost_ratio: f64,
    /// Share of samples drawn around zero controls instead of the plan.
    pub zero_mean_fraction: f64,
    /// Effective sample size band, as shares of `num_samples`, that the
    /// adaptive temperature keeps the weights in.
    pub ess_low: f64,
    pub ess_high: f64,
    /// Savitzky–Golay smoothing of the plan after each update.
    pub smoothing: bool,
    /// Obstacle critics against each step's predicted obstacles and the
    /// grid: `weight_collision` for every step in
    /// contact, `weight_repulsion · (1 − d / repulsion_distance)²` closer
    /// than `repulsion_distance`; 0 ignores obstacles.
    pub weight_collision: f64,
    pub weight_repulsion: f64,
    pub repulsion_distance: f64,
}

impl Default for MppiConfig {
    fn default() -> Self {
        Self {
            horizon_steps: 30,
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
            weight_progress: 30.0,
            ref_velocity: 1.0,
            turn_first_activation_deg: 60.0,
            turn_first_release_deg: 15.0,
            decel_distance: 2.0,
            noise_correlation: 0.7,
            control_cost_ratio: 0.05,
            zero_mean_fraction: 0.1,
            ess_low: 0.01,
            ess_high: 0.05,
            smoothing: true,
            weight_collision: 2000.0,
            weight_repulsion: 400.0,
            repulsion_distance: 1.0,
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
    /// Scale of the lateral-error weight at each point; empty is 1 throughout.
    pub lateral: Vec<f64>,
    /// Lateral error, metres, past which the full weight returns at each
    /// point (see [`eased_lateral`]); empty is 0 throughout.
    pub allow: Vec<f64>,
}

/// Cost of lateral error `e` whose weight is eased to `scale` up to `allow`
/// metres and full beyond: `scale e² + (1 − scale) max(0, |e| − allow)²`,
/// with its first and second derivatives in `e`.
pub(crate) fn eased_lateral(e: f64, scale: f64, allow: f64) -> (f64, f64, f64) {
    let excess = (e.abs() - allow).max(0.0);
    let rest = 1.0 - scale;
    (
        scale * e * e + rest * excess * excess,
        2.0 * scale * e + 2.0 * rest * excess * e.signum(),
        2.0 * scale + if excess > 0.0 { 2.0 * rest } else { 0.0 },
    )
}

/// Lateral room past a detour's own width before the full weight returns.
const ALLOW_MARGIN: f64 = 0.3;

impl Reference {
    pub fn lateral_at(&self, i: usize) -> f64 {
        self.lateral.get(i).copied().unwrap_or(1.0)
    }

    pub fn allow_at(&self, i: usize) -> f64 {
        self.allow.get(i).copied().unwrap_or(0.0)
    }

    /// This reference moved sideways round every obstacle mode it meets at
    /// its own time, passing them on `side` (+1 left, −1 right) with
    /// `reach` plus two spreads of room, ramped in and out over `ramp`
    /// points: the guidance that keeps a solve in one homotopy class.
    /// `None` when no obstacle comes near it.
    pub fn detour(&self, obstacles: &[Obstacle], reach: f64, dt: f64, side: f64, ramp: usize) -> Option<Reference> {
        let n = self.x.len();
        let mut need = vec![0.0_f64; n];
        for (i, need) in need.iter_mut().enumerate() {
            let (s, c) = self.yaw[i].sin_cos();
            for m in obstacles.iter().flat_map(|o| modes_at(o, i as f64 * dt, dt)) {
                let room = m.radius + reach + 2.0 * m.std_x.max(m.std_y);
                let (dx, dy) = (m.x - self.x[i], m.y - self.y[i]);
                let along = dx * c + dy * s;
                let lateral = -dx * s + dy * c;
                if along.abs() < room {
                    let edge = lateral + side * (room * room - along * along).sqrt();
                    *need = need.max(side * edge);
                }
            }
        }
        if need.iter().all(|v| *v <= 0.0) {
            return None;
        }
        let ramp = ramp.max(1) as f64;
        let mut out = self.clone();
        for i in 0..n {
            let shift = (0..n)
                .map(|j| need[j] * (1.0 - (i as f64 - j as f64).abs() / ramp).max(0.0))
                .fold(0.0_f64, f64::max);
            let (s, c) = self.yaw[i].sin_cos();
            out.x[i] -= side * shift * s;
            out.y[i] += side * shift * c;
        }
        out.lateral.clear();
        Some(out)
    }

    /// Eases the lateral and heading weights to `scale` wherever one of the
    /// `guides` leaves this reference, up to the guide's offset there plus a
    /// margin, so going round is not charged as tracking error but going
    /// further round than that is.
    pub fn relax_where_detoured<'a>(&mut self, guides: impl IntoIterator<Item = &'a Reference>, scale: f64) {
        let mut lateral = vec![1.0; self.x.len()];
        let mut allow = vec![0.0; self.x.len()];
        for g in guides {
            for i in 0..self.x.len() {
                let Some(offset) = g.x.get(i).zip(g.y.get(i)).map(|(x, y)| (x - self.x[i]).hypot(y - self.y[i])) else {
                    continue;
                };
                if offset > 1e-3 {
                    lateral[i] = scale;
                    allow[i] = f64::max(allow[i], offset + ALLOW_MARGIN);
                }
            }
        }
        if lateral.iter().any(|l| *l < 1.0) {
            self.lateral = lateral;
            self.allow = allow;
        }
    }
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
        lateral: Vec::new(),
        allow: Vec::new(),
    };
    let mut s = start_s;
    for _ in 0..=horizon {
        let (p, h) = sample(&path.waypoints, cum, s);
        let signed = signed_speed_at(speeds, cum, s);
        let reverse = signed.is_some_and(|v| v < 0.0);
        let stop = crate::core::path::cusp_after(speeds, cum, s).unwrap_or(total).min(total) - s;
        let taper = if decel_distance > 1e-6 {
            (stop.max(0.0) / decel_distance).clamp(0.0, 1.0)
        } else {
            1.0
        };
        let at_end = total - s <= 1e-6;
        let mut v = ref_velocity * if at_end { 0.0 } else { taper.max(0.1) };
        if let Some(cap) = signed.map(f64::abs).filter(|c| *c > 0.0) {
            v = v.min(cap);
        }
        r.x.push(p.x);
        r.y.push(p.y);
        r.yaw.push(if reverse { normalize_angle(h + PI) } else { h });
        r.v.push(if reverse { -v } else { v });
        s += v * dt;
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
    /// Cost per metre of corridor left ahead of a rollout's end.
    pub progress: f64,
}

/// Kinematic model shared by every rollout in one tick. State is
/// `(x, y, yaw, v, vy)`; `vy` (lateral speed) and its control `accel_lat`
/// only move away from zero on a holonomic platform.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Model<'a> {
    pub constraints: &'a RobotConstraints,
    pub x0: f64,
    pub y0: f64,
    pub yaw0: f64,
    pub v0: f64,
    pub vy0: f64,
    pub dt: f64,
    pub ackermann: bool,
    pub holonomic: bool,
    pub v_min: f64,
    pub v_max: f64,
    /// Distance from the rear axle forward to the pose origin.
    pub axle_offset: f64,
    /// Fastest change (per second) of the steering channel: the steering
    /// angle of a car, the yaw rate of anything else.
    pub steer_rate: f64,
    /// The steering channel as it stands now.
    pub steer0: f64,
}

impl<'a> Model<'a> {
    pub fn new(
        state: &RobotState,
        constraints: &'a RobotConstraints,
        v0: f64,
        vy0: f64,
        dt: f64,
        allow_reverse: bool,
    ) -> Self {
        let (v_min, v_max) = crate::core::kinematics::speed_bounds(constraints, allow_reverse);
        let ackermann = is_ackermann(constraints.steering_type);
        let holonomic = is_holonomic(constraints.steering_type);
        let axle_offset = if ackermann {
            constraints.rear_wheelbase.max(0.0)
        } else {
            0.0
        };
        let yaw0 = state.pose.rotation.to_euler().yaw;
        let rate = if ackermann { constraints.max_steering_rate } else { constraints.max_angular_acceleration };
        let steer_rate = if rate > 0.0 { rate } else { f64::INFINITY };
        let (v, w) = (state.velocity.linear, state.velocity.angular);
        let steer0 = match (state.velocity_valid, ackermann) {
            (true, true) if v.abs() > 0.1 => (wheelbase(constraints) * w / v).atan(),
            (true, false) => w,
            _ => 0.0,
        };
        Self {
            constraints,
            steer_rate,
            steer0,
            x0: state.pose.point.x - axle_offset * yaw0.cos(),
            y0: state.pose.point.y - axle_offset * yaw0.sin(),
            yaw0,
            v0,
            vy0: if holonomic { vy0 } else { 0.0 },
            dt,
            ackermann,
            holonomic,
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

    /// Lateral acceleration bound: zero except on a holonomic platform, so
    /// `vy` can never leave zero anywhere else.
    pub fn accel_lat_bound(&self) -> f64 {
        if self.holonomic {
            self.constraints.max_linear_acceleration.abs().max(1e-6)
        } else {
            0.0
        }
    }

    /// Largest lateral speed magnitude: zero except on a holonomic
    /// platform.
    pub fn vy_bound(&self) -> f64 {
        if self.holonomic {
            self.constraints.max_linear_velocity.abs()
        } else {
            0.0
        }
    }

    pub fn clamp_controls(&self, steer: &mut [f64], accel: &mut [f64], accel_lat: &mut [f64]) {
        let sb = self.steer_bound();
        let ab = self.accel_bound();
        let lb = self.accel_lat_bound();
        for s in steer.iter_mut() {
            *s = s.clamp(-sb, sb);
        }
        for a in accel.iter_mut() {
            *a = a.clamp(-ab, ab);
        }
        for a in accel_lat.iter_mut() {
            *a = a.clamp(-lb, lb);
        }
    }

    /// Pose origin for a rear-axle state.
    pub fn origin(&self, x: f64, y: f64, yaw: f64) -> (f64, f64) {
        (x + self.axle_offset * yaw.cos(), y + self.axle_offset * yaw.sin())
    }

    pub fn step(
        &self,
        x: &mut f64,
        y: &mut f64,
        yaw: &mut f64,
        v: &mut f64,
        vy: &mut f64,
        steer: f64,
        accel: f64,
        accel_lat: f64,
    ) {
        // vy is exactly 0.0 here unless holonomic (accel_lat is clamped to
        // a zero bound otherwise), so this reduces to the plain unicycle
        // position update for every other steering type.
        let (s, c) = yaw.sin_cos();
        *x += (*v * c - *vy * s) * self.dt;
        *y += (*v * s + *vy * c) * self.dt;
        let w_max = self.constraints.max_angular_velocity.abs();
        if self.ackermann {
            let w = (*v * steer.tan() / wheelbase(self.constraints)).clamp(-w_max, w_max);
            *yaw += w * self.dt;
        } else {
            *yaw += steer * self.dt;
        }
        *yaw = normalize_angle(*yaw);
        *v = (*v + accel * self.dt).clamp(self.v_min, self.v_max);
        let vyb = self.vy_bound();
        *vy = (*vy + accel_lat * self.dt).clamp(-vyb, vyb);
        let speed = v.hypot(*vy);
        if self.holonomic && speed > vyb && speed > 1e-12 {
            *v *= vyb / speed;
            *vy *= vyb / speed;
        }
    }
}

/// Roll out one control sequence and return `(cost, trajectory)`. `extra`
/// receives `(step index, x, y, yaw)` after every integration step and
/// returns an additional stage cost (obstacles, risk, ...).
pub(crate) fn rollout(
    model: &Model,
    steer: &[f64],
    accel: &[f64],
    accel_lat: &[f64],
    reference: &Reference,
    w: &CostWeights,
    extra: &dyn Fn(usize, f64, f64, f64) -> f64,
    collect: bool,
) -> (f64, Vec<Point>) {
    let n = steer.len().min(accel.len()).min(accel_lat.len());
    let sb = model.steer_bound();
    let ab = model.accel_bound();
    let lb = model.accel_lat_bound();
    let (mut x, mut y, mut yaw, mut v, mut vy) = (model.x0, model.y0, model.yaw0, model.v0, model.vy0);
    let mut actual = model.steer0.clamp(-sb, sb);
    let lag = model.steer_rate * model.dt;
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
        let al = accel_lat[i].clamp(-lb, lb);
        actual += (s - actual).clamp(-lag, lag);
        model.step(&mut x, &mut y, &mut yaw, &mut v, &mut vy, actual, a, al);
        let (x, y) = model.origin(x, y, yaw);
        let yaw_here = yaw;
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
        let lat = reference.lateral_at(ri);
        let mut stage = w.cte * (eased_lateral(cte, lat, reference.allow_at(ri)).0 + 0.25 * along * along)
            + lat * w.epsi * epsi * epsi
            + w.vel * ve * ve
            + w.steering * s * s
            + w.accel * (a * a + al * al);
        stage += extra(i, x, y, yaw_here);
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
    let Some(mut proj) = cursor.project(&base.path.waypoints, state.pose.point, base.path_index) else {
        return Err(VelocityCommand::invalid("no path"));
    };
    let (pos_tol, ang_tol) = effective_tolerances(goal, &cfg);
    // Within reach of a cusp, progress carries on along the leg leaving it.
    if let Some(c) = crate::core::path::cusp_after(&base.path.speeds, &cursor.cum, proj.arc_length) {
        let k = cursor.cum.partition_point(|s| *s < c - 1e-9).min(base.path.waypoints.len() - 1);
        if base.path.waypoints[k].point.distance_to_2d(state.pose.point) < pos_tol.max(0.1) {
            cursor.commit(k);
            if let Some(next) = cursor.project(&base.path.waypoints, state.pose.point, k) {
                proj = next;
            }
        }
    }
    base.path_index = proj.segment;
    if let Some(cmd) = base.arrival_on_path(state, goal, constraints, &proj, cursor.total_length()) {
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
    if state.velocity_valid {
        state.velocity.linear
    } else {
        last_commanded
    }
}

/// Lateral-speed feedback, analogous to `current_speed`. Only meaningful on
/// a holonomic platform; callers pass 0.0 as the fallback otherwise.
pub(crate) fn current_lateral_speed(state: &RobotState, last_commanded: f64) -> f64 {
    if state.velocity_valid {
        state.velocity.lateral
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

/// Convert the first `(steer, accel, accel_lat)` of a solution into a
/// command. On a holonomic platform `steer` is the yaw rate (holonomic is
/// never Ackermann) and `vy_now`/`accel_lat` drive the lateral axis
/// independently; otherwise `accel_lat` is always 0 and this is identical
/// to the old two-control conversion.
pub(crate) fn command_from_controls(
    v_now: f64,
    vy_now: f64,
    steer: f64,
    accel: f64,
    accel_lat: f64,
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
    if is_holonomic(constraints.steering_type) {
        let vy = vy_now + accel_lat * dt;
        return finalize_holonomic(v, vy, steer, constraints, cfg, message);
    }
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
    mean_accel_lateral: Vec<f64>,
    predicted_trajectory: Vec<Point>,
    is_turning_in_place: bool,
    last_v: f64,
    last_vy: f64,
    shift_accum: f64,
    trajectory: Option<Trajectory>,
    clock: f64,
    rng: StdRng,
    /// Adaptive temperature.
    lambda: f64,
    /// Steering channel of the last command.
    last_steer: f64,
    /// Side the detour last passed obstacles on: +1 left, −1 right, 0 none.
    side: f64,
    debug_level: DebugLevel,
    debug: DebugFrame,
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

    /// The steps and step this controller plans with on the machine of
    /// `constraints`.
    pub(crate) fn plan_for(&self, constraints: &RobotConstraints) -> (usize, f64) {
        let c = &self.mppi_config;
        planning(constraints, c.ref_velocity.min(constraints.max_linear_velocity), c.horizon_steps, c.dt)
    }

    /// Forgets the guides recorded for display.
    pub(crate) fn clear_guides(&mut self) {
        self.debug.guides.clear();
    }

    /// Points of the reference ahead, `n` steps of `step` at `speed`, that a
    /// detour round the obstacles on either side leaves: where tracking the
    /// path gives way to going round.
    pub(crate) fn detoured_points(
        &mut self,
        state: &RobotState,
        obstacles: &[Obstacle],
        reach: f64,
        n: usize,
        step: f64,
        speed: f64,
    ) -> Vec<(f64, f64, f64)> {
        if obstacles.is_empty() {
            return Vec::new();
        }
        let reference = match &self.trajectory {
            Some(traj) => build_timed_reference(traj, self.clock, n, step),
            None => {
                let Some(p) = self.cursor.project(&self.base.path.waypoints, state.pose.point, self.base.path_index) else {
                    return Vec::new();
                };
                build_reference(&self.base.path, &self.cursor.cum, p.arc_length, n, step, speed, 0.0)
            }
        };
        if self.debug_level >= DebugLevel::Basic {
            self.debug.guides = guide_paths(&reference, obstacles, reach, step);
        }
        detoured(&reference, obstacles, reach, step, &mut self.side)
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
            mean_accel_lateral: vec![0.0; n],
            predicted_trajectory: Vec::new(),
            is_turning_in_place: false,
            last_v: 0.0,
            last_vy: 0.0,
            shift_accum: 0.0,
            trajectory: None,
            clock: 0.0,
            rng,
            cursor: PathCursor::default(),
            lambda: cfg.temperature.max(1e-6),
            last_steer: 0.0,
            side: 0.0,
            debug_level: DebugLevel::Off,
            debug: DebugFrame::default(),
            mppi_config: cfg,
        }
    }

    pub fn set_mppi_config(&mut self, cfg: MppiConfig) {
        let n = cfg.horizon_steps;
        self.mean_steering = vec![0.0; n];
        self.mean_acceleration = vec![0.0; n];
        self.mean_accel_lateral = vec![0.0; n];
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
            progress: cfg.weight_progress,
        }
    }

    /// Full MPPI tick with an additional stage cost `extra(step, x, y, yaw)`
    /// and a scale on the executed speed (risk slowdown).
    pub(crate) fn step_with(
        &mut self,
        state: &RobotState,
        goal: &Goal,
        constraints: &RobotConstraints,
        dt: f64,
        extra: &dyn Fn(usize, f64, f64, f64) -> f64,
        relax: &dyn Fn(f64, f64) -> (f64, f64),
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
        (working.horizon_steps, working.dt) = self.plan_for(constraints);
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
            self.mean_accel_lateral = vec![0.0; n];
        }

        let v_now = current_speed(state, self.last_v);
        let vy_now = current_lateral_speed(state, self.last_vy);
        let mut model = Model::new(state, constraints, v_now, vy_now, working.dt, prep.allow_reverse);
        if !state.velocity_valid {
            model.steer0 = self.last_steer;
        }
        let reference = self.trajectory.as_ref().map(|traj| build_timed_reference(traj, self.clock, n, working.dt));
        let v_ref = working.ref_velocity.min(constraints.max_linear_velocity) * speed_scale;
        let reach = v_ref.max(0.2);
        let corridor = build_corridor(
            &self.base.path,
            &self.cursor.cum,
            prep.proj.arc_length,
            1.5 * reach * n as f64 * working.dt + 1.0,
            (0.5 * reach * working.dt).clamp(0.02, 0.2),
            v_ref,
            working.decel_distance,
        );
        let mut corridor = corridor;
        corridor.release = (0.5 * prep.pos_tol).max(0.05);
        for i in 0..corridor.x.len() {
            let (scale, allow) = relax(corridor.x[i], corridor.y[i]);
            corridor.lateral[i] = scale.clamp(0.0, 1.0);
            corridor.allow[i] = allow.max(0.0);
        }
        let w = self.weights(&working);
        let cost_of = |steer: &[f64], accel: &[f64], accel_lat: &[f64], collect: bool| match &reference {
            Some(r) => rollout(&model, steer, accel, accel_lat, r, &w, extra, collect),
            None => rollout_corridor(&model, steer, accel, accel_lat, &corridor, &w, extra, collect),
        };
        model.clamp_controls(
            &mut self.mean_steering,
            &mut self.mean_acceleration,
            &mut self.mean_accel_lateral,
        );

        let sigma_s = working.steering_noise.max(1e-6);
        let sigma_a = working.acceleration_noise.max(1e-6);
        let sigma_al = working.acceleration_noise.max(1e-6);
        let (sb, ab, lb) = (model.steer_bound(), model.accel_bound(), model.accel_lat_bound());
        let lambda = self.lambda.max(1e-6);
        let gamma = working.control_cost_ratio.max(0.0) * lambda;
        let zero_centred = 3 + (k as f64 * working.zero_mean_fraction.clamp(0.0, 1.0)) as usize;

        let mut costs = vec![0.0; k];
        let mut eps_s = vec![vec![0.0; n]; k];
        let mut eps_a = vec![vec![0.0; n]; k];
        let mut eps_al = vec![vec![0.0; n]; k];
        let mut steer = vec![0.0; n];
        let mut accel = vec![0.0; n];
        let mut accel_lat = vec![0.0; n];
        let mut kept: Vec<(Vec<Point>, f64)> = Vec::new();
        for j in 0..k {
            match j {
                // The previous plan, unperturbed.
                0 => {
                    steer.copy_from_slice(&self.mean_steering);
                    accel.copy_from_slice(&self.mean_acceleration);
                    accel_lat.copy_from_slice(&self.mean_accel_lateral);
                }
                // Braking to a stand on the previous steering.
                1 => {
                    steer.copy_from_slice(&self.mean_steering);
                    let (mut v, mut vy) = (model.v0, model.vy0);
                    for t in 0..n {
                        accel[t] = (-v / model.dt).clamp(-ab, ab);
                        accel_lat[t] = (-vy / model.dt).clamp(-lb, lb);
                        v += accel[t] * model.dt;
                        vy += accel_lat[t] * model.dt;
                    }
                }
                // Straight on at the present speed.
                2 => {
                    steer.fill(0.0);
                    accel.fill(0.0);
                    accel_lat.fill(0.0);
                }
                _ => {
                    correlated_noise(&mut steer, sigma_s, working.noise_correlation, &mut self.rng);
                    correlated_noise(&mut accel, sigma_a, working.noise_correlation, &mut self.rng);
                    correlated_noise(&mut accel_lat, sigma_al, working.noise_correlation, &mut self.rng);
                    if j >= zero_centred {
                        for t in 0..n {
                            steer[t] += self.mean_steering[t];
                            accel[t] += self.mean_acceleration[t];
                            accel_lat[t] += self.mean_accel_lateral[t];
                        }
                    }
                }
            }
            let mut control_cost = 0.0;
            for t in 0..n {
                steer[t] = steer[t].clamp(-sb, sb);
                accel[t] = accel[t].clamp(-ab, ab);
                accel_lat[t] = accel_lat[t].clamp(-lb, lb);
                eps_s[j][t] = steer[t] - self.mean_steering[t];
                eps_a[j][t] = accel[t] - self.mean_acceleration[t];
                eps_al[j][t] = accel_lat[t] - self.mean_accel_lateral[t];
                control_cost += self.mean_steering[t] * eps_s[j][t] / (sigma_s * sigma_s)
                    + self.mean_acceleration[t] * eps_a[j][t] / (sigma_a * sigma_a)
                    + self.mean_accel_lateral[t] * eps_al[j][t] / (sigma_al * sigma_al);
            }
            let keep = self.debug_level == DebugLevel::Full && j % ROLLOUT_STRIDE == 0;
            let (c, traj) = cost_of(&steer, &accel, &accel_lat, keep);
            costs[j] = c + gamma * control_cost * model.dt;
            if keep {
                kept.push((traj, costs[j]));
            }
        }

        if self.debug_level == DebugLevel::Full {
            (self.debug.rollouts, self.debug.costs) = kept.into_iter().unzip();
        }
        let weights = importance_weights(&costs, lambda);
        let ess = effective_sample_size(&weights);
        let (low, high) = (working.ess_low * k as f64, working.ess_high * k as f64);
        if ess < low.max(2.0) {
            self.lambda = (lambda * 1.2).min(1e6);
        } else if ess > high.max(3.0) {
            self.lambda = (lambda * 0.9).max(1e-6);
        }
        for t in 0..n {
            let (mut ds, mut da, mut dal) = (0.0, 0.0, 0.0);
            for j in 0..k {
                ds += weights[j] * eps_s[j][t];
                da += weights[j] * eps_a[j][t];
                dal += weights[j] * eps_al[j][t];
            }
            self.mean_steering[t] += ds;
            self.mean_acceleration[t] += da;
            self.mean_accel_lateral[t] += dal;
        }
        if working.smoothing {
            savitzky_golay(&mut self.mean_steering);
            savitzky_golay(&mut self.mean_acceleration);
            savitzky_golay(&mut self.mean_accel_lateral);
        }
        model.clamp_controls(
            &mut self.mean_steering,
            &mut self.mean_acceleration,
            &mut self.mean_accel_lateral,
        );

        let (_, traj) = cost_of(&self.mean_steering, &self.mean_acceleration, &self.mean_accel_lateral, true);
        self.predicted_trajectory = traj;

        let steer0 = self.mean_steering[0];
        let accel0 = self.mean_acceleration[0];
        let accel_lat0 = self.mean_accel_lateral[0];
        self.last_steer = steer0;
        crate::pred::mppi::shift_warm_start(
            &mut self.shift_accum,
            dt,
            working.dt,
            [&mut self.mean_steering, &mut self.mean_acceleration, &mut self.mean_accel_lateral],
        );

        let dt_apply = dt.min(working.dt);
        let v_cap = speed_at_arc_length(&self.base.path.speeds, &self.cursor.cum, prep.proj.arc_length)
            .unwrap_or(f64::INFINITY)
            .min(speed_scale * constraints.max_linear_velocity.max(0.0));
        self.base.status.mode = mode.into();
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
            message,
        );
        let physical = matches!(self.base.config.output_units, crate::types::OutputUnits::Physical);
        self.last_v = if cmd.valid && physical {
            cmd.linear_velocity
        } else {
            (v_now + accel0 * dt_apply).clamp(-v_cap, v_cap)
        };
        self.last_vy = if cmd.valid && physical {
            cmd.lateral_velocity
        } else {
            vy_now + accel_lat0 * dt_apply
        };
        cmd
    }
}

impl Controller for MppiFollower {
    fn params(&self) -> serde_json::Value {
        serde_json::to_value(&self.mppi_config).unwrap_or_default()
    }

    fn set_params(&mut self, params: &serde_json::Value) -> Result<(), String> {
        let cfg = crate::controller::merged(&self.mppi_config, params)?;
        self.set_mppi_config(cfg);
        Ok(())
    }

    fn set_seed(&mut self, seed: u64) {
        self.rng = StdRng::seed_from_u64(seed);
    }

    fn set_debug(&mut self, level: DebugLevel) {
        self.debug_level = level;
        self.debug = DebugFrame::default();
    }

    fn debug_frame(&self) -> DebugFrame {
        self.debug.clone()
    }

    fn compute_control(
        &mut self,
        state: &RobotState,
        goal: &Goal,
        constraints: &RobotConstraints,
        dt: f64,
        world: Option<&WorldConstraints>,
    ) -> VelocityCommand {
        self.debug.guides.clear();
        let cfg = &self.mppi_config;
        let obstacles: &[Obstacle] = world.map_or(&[], |w| &w.obstacles);
        let has_grid = world.is_some_and(|w| w.grid.is_some());
        if (obstacles.is_empty() && !has_grid) || (cfg.weight_collision <= 0.0 && cfg.weight_repulsion <= 0.0) {
            return self.step_with(state, goal, constraints, dt, &|_, _, _, _| 0.0, &|_, _| (1.0, 0.0), 1.0, "MPPI tracking", "mppi_tracking");
        }
        let (n, step) = self.plan_for(constraints);
        let (n, step) = (n.max(1), step.max(1e-3));
        let (collision, repulsion, reach) = (cfg.weight_collision, cfg.weight_repulsion, cfg.repulsion_distance.max(1e-3));
        let checker = CollisionChecker::new(world, constraints, 0.0).with_step(step).with_spread(1.0);
        let critic = |i: usize, x: f64, y: f64, yaw: f64| -> f64 {
            let d = checker.clearance_at((i + 1) as f64 * step, x, y, yaw);
            if d < 0.0 {
                collision
            } else if d < reach {
                repulsion * (1.0 - d / reach).powi(2)
            } else {
                0.0
            }
        };
        let speed = cfg.ref_velocity.min(constraints.max_linear_velocity);
        let detoured = self.detoured_points(state, obstacles, robot_radius(constraints), n, step, speed);
        let relax = detour_relax(&detoured, (speed * step).max(0.3));
        self.step_with(state, goal, constraints, dt, &critic, &relax, 1.0, "MPPI tracking", "mppi_tracking")
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
        self.mean_accel_lateral = vec![0.0; n];
        self.predicted_trajectory.clear();
        self.is_turning_in_place = false;
        self.shift_accum = 0.0;
    }

    fn reset(&mut self) {
        self.set_path(Path::default());
        self.last_v = 0.0;
        self.last_vy = 0.0;
        self.lambda = self.mppi_config.temperature.max(1e-6);
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

/// Advance warm-start control sequences by every whole planner step of
/// `step` seconds in `elapsed` plus the carried remainder, holding each
/// sequence's last control at the tail.
pub(crate) fn shift_warm_start<const N: usize>(carry: &mut f64, elapsed: f64, step: f64, sequences: [&mut Vec<f64>; N]) {
    *carry += elapsed;
    if step.is_nan() || step <= 0.0 || *carry < step {
        return;
    }
    let steps = (*carry / step).floor();
    *carry -= steps * step;
    for seq in sequences {
        let n = seq.len();
        if n == 0 {
            continue;
        }
        let k = (steps as usize).min(n);
        let tail = seq[n - 1];
        seq.rotate_left(k);
        for v in &mut seq[n - k..] {
            *v = tail;
        }
    }
}

/// The path ahead of the robot, densely sampled for a contouring cost:
/// point, the body yaw expected there (turned round on reverse stretches)
/// and the signed target speed, every `spacing` metres.
#[derive(Clone, Debug, Default)]
pub(crate) struct Corridor {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub yaw: Vec<f64>,
    pub v: Vec<f64>,
    /// Scale of the lateral and heading weights at each point, below 1
    /// where the path is left to go round an obstacle.
    pub lateral: Vec<f64>,
    /// Lateral error past which the full weight returns at each point.
    pub allow: Vec<f64>,
    pub spacing: f64,
    /// Indices of the cusps: the last point before the direction changes.
    pub cusps: Vec<usize>,
    /// How close a rollout must come to a cusp before it counts as
    /// progressing onto the next leg.
    pub release: f64,
}

/// Signed speed of the path at arc length `s`: that of the waypoint the
/// segment there leads to, if the path carries speeds.
fn signed_speed_at(speeds: &[f64], cum: &[f64], s: f64) -> Option<f64> {
    if speeds.is_empty() || speeds.len() != cum.len() {
        return None;
    }
    let i = cum.partition_point(|c| *c <= s).min(speeds.len() - 1);
    Some(speeds[i])
}

/// `length` metres of path from arc length `s0`. The target speed is
/// `ref_velocity`, capped by the path speeds and slowing as √(distance)
/// over `decel` metres before the path end and every cusp.
pub(crate) fn build_corridor(
    path: &Path,
    cum: &[f64],
    s0: f64,
    length: f64,
    spacing: f64,
    ref_velocity: f64,
    decel: f64,
) -> Corridor {
    let total = cum.last().copied().unwrap_or(0.0);
    let spacing = spacing.max(0.01);
    let mut c = Corridor { spacing, ..Corridor::default() };
    let mut last_reverse: Option<bool> = None;
    let mut s = s0.clamp(0.0, total);
    loop {
        let (p, heading) = sample(&path.waypoints, cum, s);
        let signed = signed_speed_at(&path.speeds, cum, s);
        let reverse = signed.is_some_and(|v| v < 0.0);
        let stop = crate::core::path::cusp_after(&path.speeds, cum, s).unwrap_or(total).min(total) - s;
        let taper = if decel > 1e-6 { (stop.max(0.0) / decel).min(1.0).sqrt() } else { 1.0 };
        let mut v = ref_velocity.max(0.0) * taper;
        if let Some(cap) = signed.map(f64::abs).filter(|c| *c > 0.0) {
            v = v.min(cap);
        }
        if last_reverse.is_some_and(|r| r != reverse) {
            c.cusps.push(c.x.len() - 1);
        }
        last_reverse = Some(reverse);
        c.x.push(p.x);
        c.y.push(p.y);
        c.yaw.push(if reverse { normalize_angle(heading + PI) } else { heading });
        c.v.push(if reverse { -v } else { v });
        c.lateral.push(1.0);
        c.allow.push(0.0);
        if s >= total || s >= s0 + length {
            break;
        }
        s = (s + spacing).min(total);
    }
    c
}

/// Cost and (optionally) points of one control sequence against a corridor:
/// lateral and heading error to the nearest corridor point, speed against
/// its target, control effort and `extra(step, x, y, yaw)`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn rollout_corridor(
    model: &Model,
    steer: &[f64],
    accel: &[f64],
    accel_lat: &[f64],
    corridor: &Corridor,
    w: &CostWeights,
    extra: &dyn Fn(usize, f64, f64, f64) -> f64,
    collect: bool,
) -> (f64, Vec<Point>) {
    let n = steer.len().min(accel.len()).min(accel_lat.len());
    let (sb, ab, lb) = (model.steer_bound(), model.accel_bound(), model.accel_lat_bound());
    let (mut x, mut y, mut yaw, mut v, mut vy) = (model.x0, model.y0, model.yaw0, model.v0, model.vy0);
    let mut actual = model.steer0.clamp(-sb, sb);
    let lag = model.steer_rate * model.dt;
    let mut traj = Vec::new();
    if collect {
        traj.reserve(n + 1);
        let (ox, oy) = model.origin(x, y, yaw);
        traj.push(Point::new(ox, oy, 0.0));
    }
    let m = corridor.x.len();
    let d2 = |k: usize, px: f64, py: f64| (corridor.x[k] - px).powi(2) + (corridor.y[k] - py).powi(2);
    let reach = (model.v_max.abs().max(model.v_min.abs()) * model.dt / corridor.spacing.max(1e-3)).ceil() as usize * 2 + 3;
    let mut j = 0usize;
    let release = corridor.release.max(corridor.spacing);
    let mut gates = corridor.cusps.iter().copied().peekable();
    let mut cost = 0.0;
    for i in 0..n {
        let s = steer[i].clamp(-sb, sb);
        let a = accel[i].clamp(-ab, ab);
        let al = accel_lat[i].clamp(-lb, lb);
        actual += (s - actual).clamp(-lag, lag);
        model.step(&mut x, &mut y, &mut yaw, &mut v, &mut vy, actual, a, al);
        let (px, py) = model.origin(x, y, yaw);
        if collect {
            traj.push(Point::new(px, py, 0.0));
        }
        let mut stage = w.steering * s * s + w.accel * (a * a + al * al);
        if m > 0 {
            // Nearest corridor point within one step's reach; ties go to the
            // later point, so a fold at a cusp is followed onto the next leg.
            while let Some(&g) = gates.peek() {
                if g < j || d2(g, px, py) < release * release {
                    gates.next();
                } else {
                    break;
                }
            }
            let limit = gates.peek().copied().unwrap_or(m - 1);
            let (mut best, mut best_d) = (j, d2(j, px, py));
            for k in j.saturating_sub(1)..=(j + reach).min(limit) {
                let d = d2(k, px, py);
                if d <= best_d {
                    (best, best_d) = (k, d);
                }
            }
            j = best;
            let (dx, dy) = (px - corridor.x[j], py - corridor.y[j]);
            let (sn, cs) = corridor.yaw[j].sin_cos();
            let lateral = -dx * sn + dy * cs;
            let heading = normalize_angle(yaw - corridor.yaw[j]);
            let speed = v - corridor.v[j];
            let relaxed = corridor.lateral[j];
            stage += w.cte * eased_lateral(lateral, relaxed, corridor.allow[j]).0
                + relaxed * w.epsi * heading * heading
                + w.vel * speed * speed;
        }
        stage += extra(i, px, py, yaw);
        cost += stage * model.dt;
    }
    if m > 0 {
        cost += w.progress * (m - 1 - j) as f64 * corridor.spacing;
    }
    (cost, traj)
}

/// 9-point Savitzky–Golay smoothing of a control sequence, ends held.
pub(crate) fn savitzky_golay(seq: &mut [f64]) {
    const C: [f64; 9] = [-21.0, 14.0, 39.0, 54.0, 59.0, 54.0, 39.0, 14.0, -21.0];
    let n = seq.len();
    if n < 3 {
        return;
    }
    let src = seq.to_vec();
    for (i, out) in seq.iter_mut().enumerate() {
        let mut acc = 0.0;
        for (k, c) in C.iter().enumerate() {
            let idx = (i as isize + k as isize - 4).clamp(0, n as isize - 1) as usize;
            acc += c * src[idx];
        }
        *out = acc / 231.0;
    }
}

/// Fill `out` with temporally correlated Gaussian noise of spread `sigma`:
/// each step keeps `rho` of the previous one.
pub(crate) fn correlated_noise(out: &mut [f64], sigma: f64, rho: f64, rng: &mut StdRng) {
    let rho = rho.clamp(0.0, 0.99);
    let fresh = (1.0 - rho * rho).sqrt();
    let mut prev = 0.0;
    for (t, e) in out.iter_mut().enumerate() {
        let w: f64 = rand_distr::StandardNormal.sample(rng);
        prev = if t == 0 { w } else { rho * prev + fresh * w };
        *e = sigma * prev;
    }
}

/// Whether `o` moves across the path (more across than along it, at the
/// point of `reference` nearest it) rather than stands on it or moves along
/// it: traffic to give way to, not to go round. Its velocity is read over
/// the first second of its prediction, in steps of `step`.
pub(crate) fn crosses(o: &Obstacle, reference: &Reference, step: f64) -> bool {
    let (Some(now), Some(soon)) = (modes_at(o, 0.0, step).next(), modes_at(o, 1.0, step).next()) else {
        return false;
    };
    let (vx, vy) = (soon.x - now.x, soon.y - now.y);
    if vx.hypot(vy) < CROSSING_MIN_SPEED {
        return false;
    }
    let Some(k) = (0..reference.x.len()).min_by(|a, b| {
        let d = |i: usize| (reference.x[i] - now.x).hypot(reference.y[i] - now.y);
        d(*a).total_cmp(&d(*b))
    }) else {
        return false;
    };
    let (s, c) = reference.yaw[k].sin_cos();
    (vx * -s + vy * c).abs() > (vx * c + vy * s).abs()
}

/// Speed, m/s, below which an obstacle counts as standing.
const CROSSING_MIN_SPEED: f64 = 0.1;

/// The obstacles of `obstacles` that are worth going round on `reference`:
/// all but the ones crossing it.
pub(crate) fn to_go_round(obstacles: &[Obstacle], reference: &Reference, step: f64) -> Vec<Obstacle> {
    obstacles.iter().filter(|o| !crosses(o, reference, step)).cloned().collect()
}

/// Points of `reference` (one per `step`) that the detour round `obstacles`
/// on the side `pick_detour` chooses (with `side` its memory) leaves, each
/// with its offset there. Obstacles crossing the path get no detour: the
/// critics have the robot give way to them.
pub(crate) fn detoured(
    reference: &Reference,
    obstacles: &[Obstacle],
    reach: f64,
    step: f64,
    side: &mut f64,
) -> Vec<(f64, f64, f64)> {
    let Some(g) = pick_detour(reference, obstacles, reach, step, side) else {
        return Vec::new();
    };
    (0..reference.x.len())
        .filter_map(|i| {
            let offset = (g.x[i] - reference.x[i]).hypot(g.y[i] - reference.y[i]);
            (offset > 1e-3).then_some((reference.x[i], reference.y[i], offset))
        })
        .collect()
}

/// The detours of `reference` round `obstacles` (not those crossing it) on
/// each side, as paths: what a controller steering by them shows.
pub(crate) fn guide_paths(reference: &Reference, obstacles: &[Obstacle], reach: f64, step: f64) -> Vec<Vec<Point>> {
    let obstacles = to_go_round(obstacles, reference, step);
    [1.0, -1.0]
        .into_iter()
        .filter_map(|side| reference.detour(&obstacles, reach + DETOUR_MARGIN, step, side, DETOUR_RAMP))
        .map(|g| (0..g.x.len()).map(|i| Point::new(g.x[i], g.y[i], 0.0)).collect())
        .collect()
}

/// The one detour of `reference` round `obstacles` (not those crossing it)
/// to follow: the side whose offsets add up least, `side` (last chosen, +1
/// left, −1 right, 0 none) kept unless the other is cheaper by
/// `SIDE_SWITCH_COST`; `side` is updated, and `None` when no detour is needed.
pub(crate) fn pick_detour(
    reference: &Reference,
    obstacles: &[Obstacle],
    reach: f64,
    step: f64,
    side: &mut f64,
) -> Option<Reference> {
    use crate::pred::mpc::SIDE_SWITCH_COST;
    let obstacles = to_go_round(obstacles, reference, step);
    let mut best: Option<(f64, f64, Reference)> = None;
    for s in [1.0, -1.0] {
        let Some(g) = reference.detour(&obstacles, reach + DETOUR_MARGIN, step, s, DETOUR_RAMP) else {
            continue;
        };
        let total: f64 = (0..g.x.len()).map(|i| (g.x[i] - reference.x[i]).hypot(g.y[i] - reference.y[i])).sum();
        let cost = total * if *side != 0.0 && s != *side { SIDE_SWITCH_COST } else { 1.0 };
        if best.as_ref().is_none_or(|b| cost < b.0) {
            best = Some((cost, s, g));
        }
    }
    *side = best.as_ref().map_or(0.0, |b| b.1);
    best.map(|b| b.2)
}

/// Every how many samples one rollout is kept for display.
const ROLLOUT_STRIDE: usize = 16;

/// `(DETOURED_WEIGHT, offset + margin)` of the nearest `detoured` point
/// within `near`, else `(1, 0)`: the easing of the lateral weight there.
pub(crate) fn detour_relax(detoured: &[(f64, f64, f64)], near: f64) -> impl Fn(f64, f64) -> (f64, f64) + '_ {
    move |x, y| {
        detoured
            .iter()
            .filter(|(px, py, _)| (px - x).hypot(py - y) < near)
            .map(|(_, _, offset)| offset + ALLOW_MARGIN)
            .reduce(f64::max)
            .map_or((1.0, 0.0), |allow| (DETOURED_WEIGHT, allow))
    }
}

/// Effective sample size `1 / Σ w²` of normalised weights.
pub(crate) fn effective_sample_size(weights: &[f64]) -> f64 {
    let s: f64 = weights.iter().map(|w| w * w).sum();
    if s > 0.0 { 1.0 / s } else { 0.0 }
}

/// What a sampling controller scores its rollouts against: a timed
/// reference, or the path ahead as a corridor.
pub(crate) enum Track {
    Timed(Reference),
    Path(Corridor),
}

impl Track {
    /// The track for one tick: the installed trajectory sampled from
    /// `clock`, else `n · dt` seconds of path from arc length `s0` at up to
    /// `v_ref`, its cusps released within half of `pos_tol`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        trajectory: Option<&Trajectory>,
        clock: f64,
        path: &Path,
        cum: &[f64],
        s0: f64,
        n: usize,
        dt: f64,
        v_ref: f64,
        decel: f64,
        pos_tol: f64,
    ) -> Self {
        if let Some(traj) = trajectory {
            return Track::Timed(build_timed_reference(traj, clock, n, dt));
        }
        let reach = v_ref.max(0.2);
        let mut corridor = build_corridor(path, cum, s0, 1.5 * reach * n as f64 * dt + 1.0, (0.5 * reach * dt).clamp(0.02, 0.2), v_ref, decel);
        corridor.release = (0.5 * pos_tol).max(0.05);
        Track::Path(corridor)
    }

    /// Eases the lateral and heading weights at each point by `relax`:
    /// `(scale, allow)`, see [`eased_lateral`].
    pub(crate) fn relax_with(&mut self, relax: &dyn Fn(f64, f64) -> (f64, f64)) {
        match self {
            Track::Timed(r) => {
                let eased: Vec<(f64, f64)> = (0..r.x.len()).map(|i| relax(r.x[i], r.y[i])).collect();
                r.lateral = eased.iter().map(|e| e.0.clamp(0.0, 1.0)).collect();
                r.allow = eased.iter().map(|e| e.1.max(0.0)).collect();
            }
            Track::Path(c) => {
                for i in 0..c.x.len() {
                    let (scale, allow) = relax(c.x[i], c.y[i]);
                    c.lateral[i] = scale.clamp(0.0, 1.0);
                    c.allow[i] = allow.max(0.0);
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn rollout(
        &self,
        model: &Model,
        steer: &[f64],
        accel: &[f64],
        accel_lat: &[f64],
        w: &CostWeights,
        extra: &dyn Fn(usize, f64, f64, f64) -> f64,
        collect: bool,
    ) -> (f64, Vec<Point>) {
        match self {
            Track::Timed(r) => rollout(model, steer, accel, accel_lat, r, w, extra, collect),
            Track::Path(c) => rollout_corridor(model, steer, accel, accel_lat, c, w, extra, collect),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::eased_lateral;

    #[test]
    fn eased_lateral_is_plain_outside_detours_and_full_past_the_allowance() {
        for e in [-1.3, -0.2, 0.0, 0.4, 2.0] {
            assert!((eased_lateral(e, 1.0, 0.0).0 - e * e).abs() < 1e-12);
        }
        let (inside, _, _) = eased_lateral(0.8, 0.1, 1.0);
        assert!((inside - 0.1 * 0.64).abs() < 1e-12);
        let (far, _, _) = eased_lateral(3.0, 0.1, 1.0);
        assert!((far - (0.1 * 9.0 + 0.9 * 4.0)).abs() < 1e-12);
        let h = 1e-6;
        for e in [-2.5, -0.7, 0.3, 1.6] {
            let (_, d, dd) = eased_lateral(e, 0.1, 1.0);
            let fd = (eased_lateral(e + h, 0.1, 1.0).0 - eased_lateral(e - h, 0.1, 1.0).0) / (2.0 * h);
            let fdd = (eased_lateral(e + h, 0.1, 1.0).1 - eased_lateral(e - h, 0.1, 1.0).1) / (2.0 * h);
            assert!((fd - d).abs() < 1e-5, "{e}: {fd} vs {d}");
            assert!((fdd - dd).abs() < 1e-4, "{e}: {fdd} vs {dd}");
        }
    }
}
