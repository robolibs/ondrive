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
    actuation_delay, can_turn_in_place, finalize, finalize_holonomic, is_ackermann, is_holonomic, reverse_allowed,
    max_curvature, planning, steering_limit, wheelbase,
};
use crate::core::math::normalize_angle;
use crate::core::obstacles::{CollisionChecker, footprint_extent, modes_at, robot_radius, yield_share};
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
    /// Sample the noise as integrated control rates (Smooth MPPI) rather
    /// than correlated controls.
    #[serde(default = "default_smooth_sampling")]
    pub smooth_sampling: bool,
    /// Obstacle critics against each step's predicted obstacles and the
    /// grid: `weight_collision` for every step in
    /// contact, `weight_repulsion · (1 − d / repulsion_distance)²` closer
    /// than `repulsion_distance`; 0 ignores obstacles.
    pub weight_collision: f64,
    pub weight_repulsion: f64,
    pub repulsion_distance: f64,
    /// Room, metres, beyond the footprint that the obstacle critics count
    /// as contact.
    #[serde(default = "default_safety_margin")]
    pub safety_margin: f64,
}

fn default_safety_margin() -> f64 {
    0.6
}

pub(crate) fn default_smooth_sampling() -> bool {
    true
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
            weight_progress: 80.0,
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
            smooth_sampling: true,
            weight_collision: 2000.0,
            weight_repulsion: 400.0,
            repulsion_distance: 1.0,
            safety_margin: default_safety_margin(),
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

/// How a machine swerves round an obstacle: its tightest turning radius and
/// the distance its steering takes to swing over, metres (0 for one that
/// turns in place), and its body as a capsule along the heading, `(back,
/// front, half width)` metres of the pose, replacing the disc of radius
/// `disc` round the pose (`None` keeps the disc).
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Swerve {
    pub radius: f64,
    pub lag: f64,
    pub body: Option<(f64, f64, f64)>,
    pub disc: f64,
}

impl Swerve {
    /// The swerve of the machine of `constraints` at `speed`.
    pub fn of(constraints: &RobotConstraints, speed: f64) -> Self {
        let k = max_curvature(constraints);
        Swerve {
            radius: if k.is_finite() && k > 0.0 { 1.0 / k } else { 0.0 },
            lag: 4.0 * speed.abs() * actuation_delay(constraints),
            body: footprint_extent(constraints),
            disc: robot_radius(constraints),
        }
    }

    /// Distance along the way over which a sideways shift of `d` metres can
    /// be driven: two opposite arcs of the tightest turn, and the steering
    /// swinging over between them.
    pub fn length(&self, d: f64) -> f64 {
        let d = d.abs().min(2.0 * self.radius);
        (4.0 * self.radius * d - d * d).max(0.0).sqrt() + self.lag
    }

    /// Distance ahead past a planning horizon a detour looks, to start
    /// swerving round what it meets at the horizon's end: the length of a
    /// shift by the body's width plus `SWERVE_ROOM`.
    pub fn lookahead(&self) -> f64 {
        let width = self.body.map_or(2.0 * self.disc, |b| 2.0 * b.2);
        self.length(width + SWERVE_ROOM)
    }

    /// `(back, front, half width, room)` of the body a detour keeps `reach`
    /// from (the disc's radius plus the room kept): the capsule, keeping
    /// what `reach` adds to the disc's radius but at least
    /// `DETOUR_BODY_MARGIN` and `BODY_ROOM_SHARE` of its width, else the
    /// pose with all of `reach`.
    fn keep(&self, reach: f64) -> (f64, f64, f64, f64) {
        match self.body {
            Some((back, front, half)) => {
                let room = (reach - self.disc).max(DETOUR_BODY_MARGIN).max(BODY_ROOM_SHARE * 2.0 * half);
                (back, front, half, room)
            }
            None => (0.0, 0.0, 0.0, reach),
        }
    }
}

/// Room, metres, beside the body that `Swerve::lookahead` swerves for: an
/// obstacle's width and the margin past it.
const SWERVE_ROOM: f64 = 1.5;
/// Least room, metres, a detour keeps beside the body: past the margin the
/// obstacle critics count as contact.
const DETOUR_BODY_MARGIN: f64 = 0.45;
/// Share of the body's width a detour keeps beside it at least: a wide,
/// long machine tracks its detour less closely than a small one.
const BODY_ROOM_SHARE: f64 = 0.4;

/// Room, metres, a wide body keeps from obstacles past what a small one
/// keeps: `BODY_ROOM_SHARE` of its width beyond `DETOUR_BODY_MARGIN`, 0 for
/// a disc or a body narrow enough.
pub(crate) fn body_room(constraints: &RobotConstraints) -> f64 {
    footprint_extent(constraints).map_or(0.0, |(_, _, half)| (BODY_ROOM_SHARE * 2.0 * half - DETOUR_BODY_MARGIN).max(0.0))
}

/// `obstacles` with the radius of each standing one grown by the
/// `body_room` of the machine of `constraints`.
pub(crate) fn with_body_room(obstacles: &[Obstacle], constraints: &RobotConstraints) -> Vec<Obstacle> {
    let room = body_room(constraints);
    obstacles
        .iter()
        .map(|o| {
            let mut o = o.clone();
            if room > 0.0 && standing(&o) {
                o.radius += room;
            }
            o
        })
        .collect()
}

/// Whether the prediction of `o` moves less than `CROSSING_MIN_SPEED` over
/// its first second.
pub(crate) fn standing(o: &Obstacle) -> bool {
    match (modes_at(o, 0.0, 1.0).next(), modes_at(o, 1.0, 1.0).next()) {
        (Some(now), Some(soon)) => (soon.x - now.x).hypot(soon.y - now.y) < CROSSING_MIN_SPEED,
        _ => true,
    }
}

impl Reference {
    pub fn lateral_at(&self, i: usize) -> f64 {
        self.lateral.get(i).copied().unwrap_or(1.0)
    }

    /// This reference driven at `share` of its pace: point `i` is the one it
    /// reached at `share · i`, interpolated, and its speeds scaled.
    pub fn paced(&self, share: f64) -> Reference {
        let n = self.x.len();
        let at = |v: &[f64], u: f64| -> f64 {
            let k = (u.floor() as usize).min(n.saturating_sub(2));
            let f = (u - k as f64).clamp(0.0, 1.0);
            if n < 2 { v.first().copied().unwrap_or(0.0) } else { v[k] + f * (v[k + 1] - v[k]) }
        };
        let mut out = self.clone();
        for i in 0..n {
            let u = share * i as f64;
            out.x[i] = at(&self.x, u);
            out.y[i] = at(&self.y, u);
            let k = (u.round() as usize).min(n - 1);
            out.yaw[i] = self.yaw[k];
            out.v[i] = share * self.v[k];
        }
        out
    }

    pub fn allow_at(&self, i: usize) -> f64 {
        self.allow.get(i).copied().unwrap_or(0.0)
    }

    /// This reference moved sideways round every obstacle mode it meets at
    /// its own time (passes within the mode's radius plus its spread, as the
    /// obstacle critics grow it, plus the room `swerve` keeps for `reach` of
    /// its body), passing them on
    /// `side` (+1 left, −1 right), ramped in and out over `ramp` points or
    /// the length `swerve` needs for the largest shift, if longer: the
    /// guidance that keeps a solve in one homotopy class. `None` when it
    /// meets no obstacle.
    pub fn detour(&self, obstacles: &[Obstacle], reach: f64, dt: f64, side: f64, ramp: usize, swerve: Swerve) -> Option<Reference> {
        let n = self.x.len();
        let (back, front, half, keep) = swerve.keep(reach);
        let mut need = vec![0.0_f64; n];
        for (i, need) in need.iter_mut().enumerate() {
            let (s, c) = self.yaw[i].sin_cos();
            for m in obstacles.iter().flat_map(|o| modes_at(o, i as f64 * dt, dt)) {
                let room = m.radius + half + keep + m.std_x.max(m.std_y);
                let (dx, dy) = (m.x - self.x[i], m.y - self.y[i]);
                let along = dx * c + dy * s;
                let past = along - along.clamp(back, front);
                let lateral = -dx * s + dy * c;
                if past * past + lateral * lateral < room * room {
                    let edge = lateral + side * (room * room - past * past).sqrt();
                    *need = need.max(side * edge);
                }
            }
        }
        if need.iter().all(|v| *v <= 0.0) {
            return None;
        }
        let span: f64 = (1..n).map(|i| (self.x[i] - self.x[i - 1]).hypot(self.y[i] - self.y[i - 1])).sum();
        let spacing = span / (n.max(2) - 1) as f64;
        let widest = need.iter().copied().fold(0.0_f64, f64::max);
        let turns = if spacing > 1e-6 { (swerve.length(widest) / spacing).ceil() as usize } else { 0 };
        let ramp = ramp.max(turns).max(1) as f64;
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
    proj = cursor.past_cusp(&base.path.waypoints, &base.path.speeds, state.pose.point, proj, pos_tol.max(0.1));
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

/// Speed a command integrates from: `last_commanded` while the measured
/// `v_now` stays within `slack` of it, else `v_now`.
pub(crate) fn held_speed(v_now: f64, last_commanded: f64, slack: f64) -> f64 {
    if (v_now - last_commanded).abs() <= slack { last_commanded } else { v_now }
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
    let cap = if v_cap >= 0.0 { v_cap } else { f64::INFINITY };
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
    /// Side (+1 left, −1 right, 0 none) and pace of the last way past the
    /// obstacles ahead: see `plan_avoidance`.
    avoid: (f64, f64),
    /// The detour taken round the obstacles ahead, pursued by one sample.
    guide: Vec<(f64, f64)>,
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

    /// The way past the obstacles ahead (`n` steps of `step` at `speed`, or
    /// the robot's speed if faster): the points of the reference its detour
    /// leaves, where tracking the path gives way to going round, and the
    /// share of the speed to drive, `plan_avoidance`'s pace held lower by
    /// `give_way_share` at that pace along its detour, or the path, and
    /// along the last plan driven.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn avoidance(
        &mut self,
        state: &RobotState,
        constraints: &RobotConstraints,
        obstacles: &[Obstacle],
        reach: f64,
        n: usize,
        step: f64,
        speed: f64,
    ) -> (Vec<(f64, f64, f64)>, f64) {
        if obstacles.is_empty() {
            self.avoid = (0.0, 1.0);
            return (Vec::new(), 1.0);
        }
        let from = match &self.trajectory {
            Some(_) => 0.0,
            None => match self.cursor.project(&self.base.path.waypoints, state.pose.point, self.base.path_index) {
                Some(p) => p.arc_length,
                None => return (Vec::new(), 1.0),
            },
        };
        let v = if state.velocity_valid { state.velocity.linear } else { self.last_v };
        let along = Along { path: &self.base.path, cum: &self.cursor.cum, trajectory: self.trajectory.as_ref(), clock: self.clock, from };
        let speed = speed.max(v.abs());
        let swerve = Swerve::of(constraints, speed);
        let robot = (state.pose.point.x, state.pose.point.y);
        let plan = plan_avoidance(&along, obstacles, reach, swerve, n, step, speed, robot, &mut self.avoid);
        if self.debug_level >= DebugLevel::Basic {
            self.debug.guides = guide_paths(&plan.reference, obstacles, reach, step, swerve);
        }
        let paced = speed * plan.pace;
        let share = match &plan.guide {
            Some(g) => {
                self.guide = (0..g.x.len()).map(|i| (g.x[i], g.y[i])).collect();
                give_way_share(&along.reference(n, step, paced), obstacles, constraints, &course_through(&self.guide), paced, v, step, GIVE_WAY_MARGIN)
            }
            None => give_way_ahead(&along, obstacles, constraints, n, step, paced, v, GIVE_WAY_MARGIN),
        };
        let planned: Vec<(f64, f64)> = self.predicted_trajectory.iter().map(|p| (p.x, p.y)).collect();
        let share = if planned.len() >= 2 {
            share.min(give_way_share(&along.reference(n, step, paced), obstacles, constraints, &course_through(&planned), paced, v, step, GIVE_WAY_MARGIN))
        } else {
            share
        };
        let points = plan.guide.as_ref().map_or(Vec::new(), |g| left_by(&plan.reference, g));
        (points, plan.pace * share)
    }

    /// Forgets the guides recorded for display.
    pub(crate) fn clear_guides(&mut self) {
        self.debug.guides.clear();
        self.guide.clear();
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
            avoid: (0.0, 1.0),
            guide: Vec::new(),
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
        relax: &dyn Fn(f64, f64) -> (f64, f64, f64),
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
        corridor.relax_with(relax);
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
                // Pure pursuit of the detour round the obstacles ahead.
                3 if self.guide.len() > 1 => pursue(&model, &self.guide, v_ref, &mut steer, &mut accel, &mut accel_lat),
                // The previous steering, closing on the reference speed at
                // full acceleration.
                4 => {
                    steer.copy_from_slice(&self.mean_steering);
                    let mut v = model.v0;
                    for t in 0..n {
                        accel[t] = ((v_ref - v) / model.dt).clamp(-ab, ab);
                        accel_lat[t] = 0.0;
                        v += accel[t] * model.dt;
                    }
                }
                _ if working.smooth_sampling => {
                    integrated_noise(&mut steer, sigma_s, &mut self.rng);
                    integrated_noise(&mut accel, sigma_a, &mut self.rng);
                    integrated_noise(&mut accel_lat, sigma_al, &mut self.rng);
                    if j >= zero_centred {
                        for t in 0..n {
                            steer[t] += self.mean_steering[t];
                            accel[t] += self.mean_acceleration[t];
                            accel_lat[t] += self.mean_accel_lateral[t];
                        }
                    }
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
        let top = constraints.max_linear_velocity;
        let v_cap = speed_at_arc_length(&self.base.path.speeds, &self.cursor.cum, prep.proj.arc_length)
            .unwrap_or(f64::INFINITY)
            .min(if top > 0.0 { top } else { f64::INFINITY })
            .min(crate::pred::mpc::REF_SPEED_HEADROOM * working.ref_velocity.max(0.1))
            * speed_scale;
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
        self.clear_guides();
        let cfg = &self.mppi_config;
        let obstacles: &[Obstacle] = world.map_or(&[], |w| &w.obstacles);
        let has_grid = world.is_some_and(|w| w.grid.is_some());
        if (obstacles.is_empty() && !has_grid) || (cfg.weight_collision <= 0.0 && cfg.weight_repulsion <= 0.0) {
            return self.step_with(state, goal, constraints, dt, &|_, _, _, _| 0.0, &|_, _| (1.0, 0.0, 0.0), 1.0, "MPPI tracking", "mppi_tracking");
        }
        let (n, step) = self.plan_for(constraints);
        let (n, step) = (n.max(1), step.max(1e-3));
        let (collision, repulsion, reach) = (cfg.weight_collision, cfg.weight_repulsion, cfg.repulsion_distance.max(1e-3));
        let checker = CollisionChecker::new(world, constraints, cfg.safety_margin.max(0.0)).with_step(step).with_spread(1.0);
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
        let (detoured, share) = self.avoidance(state, constraints, obstacles, robot_radius(constraints), n, step, speed);
        let relax = detour_relax(&detoured, (speed * step).max(0.3));
        self.step_with(state, goal, constraints, dt, &critic, &relax, share, "MPPI tracking", "mppi_tracking")
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
    /// Sideways offset, + to the left, the eased weight pulls towards at
    /// each point: the detour's.
    pub toward: Vec<f64>,
    pub spacing: f64,
    /// Indices of the cusps: the last point before the direction changes.
    pub cusps: Vec<usize>,
    /// How close a rollout must come to a cusp before it counts as
    /// progressing onto the next leg.
    pub release: f64,
}

impl Corridor {
    /// Eases the lateral and heading weights at each point by `relax`:
    /// `(scale, allow, toward)`.
    pub(crate) fn relax_with(&mut self, relax: &dyn Fn(f64, f64) -> (f64, f64, f64)) {
        for i in 0..self.x.len() {
            let (scale, allow, toward) = relax(self.x[i], self.y[i]);
            self.lateral[i] = scale.clamp(0.0, 1.0);
            self.allow[i] = allow.max(0.0);
            self.toward[i] = toward;
        }
    }
}

/// Signed speed of the path at arc length `s`: that of the waypoint the
/// segment there leads to, if the path carries speeds.
pub(crate) fn signed_speed_at(speeds: &[f64], cum: &[f64], s: f64) -> Option<f64> {
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
        c.toward.push(0.0);
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
            let excess = (lateral.abs() - corridor.allow[j]).max(0.0);
            let pulled = lateral - corridor.toward[j];
            stage += w.cte * (relaxed * pulled * pulled + (1.0 - relaxed) * excess * excess)
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

/// Fill `out` with Gaussian noise sampled in the space of control rates and
/// integrated (input lifting of Smooth MPPI, Kim et al. 2022): a random walk
/// whose spread grows to `sigma` at the last step, small at the first.
pub(crate) fn integrated_noise(out: &mut [f64], sigma: f64, rng: &mut StdRng) {
    let step = sigma / (out.len().max(1) as f64).sqrt();
    let mut walk = 0.0;
    for e in out.iter_mut() {
        let w: f64 = rand_distr::StandardNormal.sample(rng);
        walk += step * w;
        *e = walk;
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

/// Share of `speed` along `course`, setting off at `v_now`, that gives way
/// to the obstacles crossing `reference` (one point per `step`):
/// `yield_share` against those alone, keeping `margin` beyond the footprint.
#[allow(clippy::too_many_arguments)]
pub(crate) fn give_way_share(
    reference: &Reference,
    obstacles: &[Obstacle],
    constraints: &RobotConstraints,
    course: &dyn Fn(f64) -> (f64, f64, f64),
    speed: f64,
    v_now: f64,
    step: f64,
    margin: f64,
) -> f64 {
    let crossing = WorldConstraints {
        obstacles: obstacles.iter().filter(|o| crosses(o, reference, step, robot_radius(constraints))).cloned().collect(),
        ..Default::default()
    };
    if crossing.obstacles.is_empty() {
        return 1.0;
    }
    let checker = CollisionChecker::new(Some(&crossing), constraints, margin).with_spread(1.0);
    let decel = constraints.max_linear_acceleration.abs();
    yield_share(&checker, course, speed, v_now, GIVE_WAY_LOOKOUT_S, decel, &GIVE_WAY_SHARES)
}

/// Where a controller drives on from: the timed `trajectory` from `clock`,
/// or else `path` (cumulative lengths `cum`) from arc length `from`.
pub(crate) struct Along<'a> {
    pub path: &'a Path,
    pub cum: &'a [f64],
    pub trajectory: Option<&'a Trajectory>,
    pub clock: f64,
    pub from: f64,
}

impl Along<'_> {
    /// `n` steps of `step` of it at `speed`.
    pub fn reference(&self, n: usize, step: f64, speed: f64) -> Reference {
        match self.trajectory {
            Some(traj) => build_timed_reference(traj, self.clock, n, step),
            None => build_reference(self.path, self.cum, self.from, n, step, speed, 0.0),
        }
    }
}

/// Share of `speed`, setting off at `v_now`, that gives way to traffic
/// crossing the next `n` steps of `step` of `along`: `give_way_share` for
/// driving on along it, keeping `margin` beyond the footprint.
#[allow(clippy::too_many_arguments)]
pub(crate) fn give_way_ahead(
    along: &Along,
    obstacles: &[Obstacle],
    constraints: &RobotConstraints,
    n: usize,
    step: f64,
    speed: f64,
    v_now: f64,
    margin: f64,
) -> f64 {
    if obstacles.is_empty() {
        return 1.0;
    }
    let ahead = along.reference(((GIVE_WAY_COURSE_S / step).ceil() as usize).max(n), step, speed);
    let course = polyline_course((0..ahead.x.len()).map(|i| (ahead.x[i], ahead.y[i], ahead.yaw[i])).collect());
    give_way_share(&along.reference(n, step, speed), obstacles, constraints, &course, speed, v_now, step, margin)
}

/// How far `(x, y)` sits sideways of where `guide` starts, across
/// `reference`'s first point: the jump onto a detour begun too late.
pub(crate) fn guide_gap(reference: &Reference, guide: &[(f64, f64)], (x, y): (f64, f64)) -> f64 {
    let (Some(&(gx, gy)), Some(&rx), Some(&ry), Some(&yaw)) = (guide.first(), reference.x.first(), reference.y.first(), reference.yaw.first()) else {
        return f64::INFINITY;
    };
    let (s, c) = yaw.sin_cos();
    let lateral = |px: f64, py: f64| -(px - rx) * s + (py - ry) * c;
    (lateral(x, y) - lateral(gx, gy)).abs()
}

/// Sideways distance, metres, within which the robot counts as on a detour.
const SWERVE_SLACK: f64 = 0.4;

/// `polyline_course` through `points`, each heading towards its neighbours.
pub(crate) fn course_through(points: &[(f64, f64)]) -> impl Fn(f64) -> (f64, f64, f64) {
    let k = points.len();
    polyline_course(
        (0..k)
            .map(|i| {
                let (a, b) = (points[i.saturating_sub(1)], points[(i + 1).min(k - 1)]);
                (points[i].0, points[i].1, (b.1 - a.1).atan2(b.0 - a.0))
            })
            .collect(),
    )
}

/// The course along the polyline `points` (x, y, yaw): the pose after a
/// distance along it, held on the last heading past either end.
pub(crate) fn polyline_course(points: Vec<(f64, f64, f64)>) -> impl Fn(f64) -> (f64, f64, f64) {
    let mut cum = vec![0.0];
    for w in points.windows(2) {
        cum.push(cum.last().unwrap() + (w[1].0 - w[0].0).hypot(w[1].1 - w[0].1));
    }
    move |d| {
        let Some(&first) = points.first() else { return (0.0, 0.0, 0.0) };
        if d <= 0.0 || points.len() == 1 {
            let (x, y, yaw) = first;
            return (x + d * yaw.cos(), y + d * yaw.sin(), yaw);
        }
        let last = cum.len() - 1;
        if d >= cum[last] {
            let (x, y, yaw) = points[last];
            let over = d - cum[last];
            return (x + over * yaw.cos(), y + over * yaw.sin(), yaw);
        }
        let i = cum.partition_point(|c| *c <= d).saturating_sub(1).min(last - 1);
        let u = ((d - cum[i]) / (cum[i + 1] - cum[i]).max(1e-9)).clamp(0.0, 1.0);
        let (a, b) = (points[i], points[i + 1]);
        (a.0 + u * (b.0 - a.0), a.1 + u * (b.1 - a.1), a.2 + u * normalize_angle(b.2 - a.2))
    }
}

/// Shares of the speed tried, fastest first, until its course keeps clear
/// of crossing traffic.
pub(crate) const GIVE_WAY_SHARES: [f64; 4] = [1.0, 0.75, 0.5, 0.25];
/// Seconds the give-way check watches the course for crossing traffic.
pub(crate) const GIVE_WAY_LOOKOUT_S: f64 = 2.0;
/// Seconds of path at the reference speed the give-way course spans.
const GIVE_WAY_COURSE_S: f64 = 12.0;
/// Room, metres, the give-way check keeps beyond the footprint.
pub(crate) const GIVE_WAY_MARGIN: f64 = 0.1;

/// Whether `o` is about to cross the path: it moves across it (more across
/// than along, at the point of `reference` nearest it) rather than stands
/// on it or moves along it, and its prediction comes within `CROSSING_REACH`
/// of the stretch of path `reference` covers, or within the room a detour
/// with `reach` keeps from it: traffic to pass behind. Its velocity is read
/// over the first second of its prediction, in steps of `step`.
pub(crate) fn crosses(o: &Obstacle, reference: &Reference, step: f64, reach: f64) -> bool {
    moves_across(o, reference, step) && comes_near(o, reference, step, reach)
}

/// Whether any point of `o`'s prediction over the reference's span comes
/// within its radius plus `CROSSING_REACH`, or plus the detour room of
/// `reach`, of any point of `reference`.
fn comes_near(o: &Obstacle, reference: &Reference, step: f64, reach: f64) -> bool {
    (0..reference.x.len()).any(|k| {
        modes_at(o, k as f64 * step, step).any(|m| {
            let near = m.radius + CROSSING_REACH.max(reach + DETOUR_MARGIN + 2.0 * m.std_x.max(m.std_y));
            (0..reference.x.len()).any(|i| (m.x - reference.x[i]).hypot(m.y - reference.y[i]) < near)
        })
    })
}

/// Distance, metres beyond an obstacle's radius, within which its predicted
/// track counts as reaching the path.
const CROSSING_REACH: f64 = 2.0;

fn moves_across(o: &Obstacle, reference: &Reference, step: f64) -> bool {
    velocity_on(o, reference, step).is_some_and(|(across, along)| across.abs() > along.abs())
}

/// Velocity of `o` over the first second of its prediction, split across
/// (+ to the left) and along the point of `reference` nearest it; `None`
/// when it stands.
fn velocity_on(o: &Obstacle, reference: &Reference, step: f64) -> Option<(f64, f64)> {
    let (now, soon) = (modes_at(o, 0.0, step).next()?, modes_at(o, 1.0, step).next()?);
    let (vx, vy) = (soon.x - now.x, soon.y - now.y);
    if vx.hypot(vy) < CROSSING_MIN_SPEED {
        return None;
    }
    let k = (0..reference.x.len()).min_by(|a, b| {
        let d = |i: usize| (reference.x[i] - now.x).hypot(reference.y[i] - now.y);
        d(*a).total_cmp(&d(*b))
    })?;
    let (s, c) = reference.yaw[k].sin_cos();
    Some((vx * -s + vy * c, vx * c + vy * s))
}

/// The side (+1 left, −1 right) that passes behind `o` crossing
/// `reference`: the one it comes from.
fn behind(o: &Obstacle, reference: &Reference, step: f64) -> Option<f64> {
    velocity_on(o, reference, step).map(|(across, _)| -across.signum())
}

/// Whether the detour to `side` of `along` driven at `speed` for twice the
/// span of `reference` comes back within `settled` of it somewhere past
/// that span: a detour the horizon cut short rather than one racing traffic.
#[allow(clippy::too_many_arguments)]
fn settles_later(along: &Along, reference: &Reference, obstacles: &[Obstacle], reach: f64, step: f64, speed: f64, side: f64, swerve: Swerve, settled: f64) -> bool {
    let n = reference.x.len();
    let longer = along.reference(2 * n, step, speed);
    let Some(g) = longer.detour(obstacles, reach + DETOUR_MARGIN, step, side, DETOUR_RAMP, swerve) else {
        return true;
    };
    (n..g.x.len().min(longer.x.len())).any(|i| (g.x[i] - longer.x[i]).hypot(g.y[i] - longer.y[i]) <= settled)
}

/// Whether `o` stands now behind `rear`, metres along the heading of
/// `reference`'s first point (negative behind it), by more than its radius:
/// traffic the robot has already passed.
fn left_behind(o: &Obstacle, reference: &Reference, rear: f64) -> bool {
    let (Some(m), Some(&x), Some(&y), Some(&yaw)) = (modes_at(o, 0.0, 1.0).next(), reference.x.first(), reference.y.first(), reference.yaw.first()) else {
        return false;
    };
    let (s, c) = yaw.sin_cos();
    (m.x - x) * c + (m.y - y) * s < rear - m.radius
}

/// The side passing behind each obstacle of `obstacles` crossing `reference`.
fn behind_sides(obstacles: &[Obstacle], reference: &Reference, step: f64, reach: f64) -> Vec<f64> {
    obstacles.iter().filter(|o| crosses(o, reference, step, reach)).filter_map(|o| behind(o, reference, step)).collect()
}

/// The side passing behind all the traffic crossing `reference`, when it
/// all comes from the same side.
pub(crate) fn pass_behind(obstacles: &[Obstacle], reference: &Reference, step: f64, reach: f64) -> Option<f64> {
    let behinds = behind_sides(obstacles, reference, step, reach);
    behinds.first().copied().filter(|b| behinds.iter().all(|x| x == b))
}

/// Cost factor of a detour passing in front of traffic crossing the path
/// rather than behind it.
const AHEAD_COST: f64 = 2.0;

/// Speed, m/s, below which an obstacle counts as standing.
const CROSSING_MIN_SPEED: f64 = 0.1;

/// Points of `reference` that the detour `g` leaves, each with its offset
/// sideways, + to the left.
pub(crate) fn left_by(reference: &Reference, g: &Reference) -> Vec<(f64, f64, f64)> {
    (0..reference.x.len().min(g.x.len()))
        .filter_map(|i| {
            let (s, c) = reference.yaw[i].sin_cos();
            let offset = -(g.x[i] - reference.x[i]) * s + (g.y[i] - reference.y[i]) * c;
            (offset.abs() > 1e-3).then_some((reference.x[i], reference.y[i], offset))
        })
        .collect()
}

/// The detours of `reference` round `obstacles` on each side, as paths:
/// what a controller steering by them shows.
pub(crate) fn guide_paths(reference: &Reference, obstacles: &[Obstacle], reach: f64, step: f64, swerve: Swerve) -> Vec<Vec<Point>> {
    [1.0, -1.0]
        .into_iter()
        .filter_map(|side| reference.detour(obstacles, reach + DETOUR_MARGIN, step, side, DETOUR_RAMP, swerve))
        .map(|g| (0..g.x.len()).map(|i| Point::new(g.x[i], g.y[i], 0.0)).collect())
        .collect()
}

/// The one detour of `reference` round `obstacles`, standing, moving along
/// or crossing it, each met where it is at the reference's own time: behind
/// the crossing traffic when it all comes from one side, else the side
/// whose offsets add up least, `AHEAD_COST` times more for passing in front
/// of crossing traffic, `side` (last chosen, +1 left, −1 right, 0 none)
/// kept unless the other is cheaper by `SIDE_SWITCH_COST`; `side` is
/// updated, and `None` when no detour is needed.
pub(crate) fn pick_detour(
    reference: &Reference,
    obstacles: &[Obstacle],
    reach: f64,
    step: f64,
    side: &mut f64,
    swerve: Swerve,
) -> Option<Reference> {
    use crate::pred::mpc::SIDE_SWITCH_COST;
    let behinds = behind_sides(obstacles, reference, step, reach);
    let only = pass_behind(obstacles, reference, step, reach);
    let mut best: Option<(f64, f64, Reference)> = None;
    for s in [1.0, -1.0] {
        if only.is_some_and(|b| b != s) {
            continue;
        }
        let Some(g) = reference.detour(obstacles, reach + DETOUR_MARGIN, step, s, DETOUR_RAMP, swerve) else {
            continue;
        };
        let total: f64 = (0..g.x.len()).map(|i| (g.x[i] - reference.x[i]).hypot(g.y[i] - reference.y[i])).sum();
        let ahead = if behinds.iter().any(|b| *b != s) { AHEAD_COST } else { 1.0 };
        let cost = ahead * total * if *side != 0.0 && s != *side { SIDE_SWITCH_COST } else { 1.0 };
        if best.as_ref().is_none_or(|b| cost < b.0) {
            best = Some((cost, s, g));
        }
    }
    *side = best.as_ref().map_or(0.0, |b| b.1);
    best.map(|b| b.2)
}

/// How to get past the obstacles ahead: the share of the reference speed
/// to drive at, the reference ahead at that pace, and the detour of it to
/// steer by, if any.
pub(crate) struct Avoidance {
    pub pace: f64,
    pub reference: Reference,
    pub guide: Option<Reference>,
}

/// The way past `obstacles` ahead of `along` for the robot at `robot`:
/// of each pace of `AVOID_PACES` (only the first on a timed trajectory),
/// driven as is or detoured to either side over `n` steps of `step` at
/// that share of `speed` and on for `swerve`'s lookahead (at least
/// `CROSSING_LOOKOUT_S` while traffic crosses), the cheapest, never a
/// detour in front of crossing traffic still shifting at its end
/// (`AHEAD_TAIL`): a detour costs its widest shift, metres, plus the square
/// of its excess over the robot's width per width, `AHEAD_COST` times more
/// in front of crossing traffic, plus `LATE_COST` per metre the
/// robot sits off where it starts (`guide_gap`), and a pace `PACE_COST` per
/// share of the speed given up. `last` (side, pace) is kept unless another
/// is cheaper by `SIDE_SWITCH_COST`, and updated.
#[allow(clippy::too_many_arguments)]
pub(crate) fn plan_avoidance(
    along: &Along,
    obstacles: &[Obstacle],
    reach: f64,
    swerve: Swerve,
    n: usize,
    step: f64,
    speed: f64,
    robot: (f64, f64),
    last: &mut (f64, f64),
) -> Avoidance {
    use crate::pred::mpc::SIDE_SWITCH_COST;
    let (back, _, half, keep) = swerve.keep(reach);
    let width = 2.0 * if half > 0.0 { half } else { reach.max(0.1) };
    let start = along.reference(1, step, speed);
    let ahead: Vec<Obstacle> = obstacles.iter().filter(|o| !left_behind(o, &start, back - keep)).cloned().collect();
    let mut far = n + (swerve.lookahead() / (speed.abs() * step).max(1e-3)).ceil() as usize;
    if ahead.iter().any(|o| crosses(o, &along.reference(far, step, speed), step, reach)) {
        far = far.max((CROSSING_LOOKOUT_S / step.max(1e-3)).ceil() as usize);
    }
    let paces: &[f64] = if along.trajectory.is_some() { &AVOID_PACES[..1] } else { &AVOID_PACES };
    let mut best: Option<(f64, f64, Avoidance)> = None;
    for &pace in paces {
        let reference = along.reference((far as f64 / pace).ceil() as usize, step, speed * pace);
        let behinds = behind_sides(&ahead, &reference, step, reach);
        let mut options: Vec<(f64, f64, Option<Reference>)> = Vec::new();
        let mut met = false;
        for s in [1.0, -1.0] {
            let Some(g) = reference.detour(obstacles, reach + DETOUR_MARGIN, step, s, DETOUR_RAMP, swerve) else {
                continue;
            };
            met = true;
            let points: Vec<(f64, f64)> = (0..g.x.len()).map(|i| (g.x[i], g.y[i])).collect();
            let shift: Vec<f64> = (0..g.x.len()).map(|i| (g.x[i] - reference.x[i]).hypot(g.y[i] - reference.y[i])).collect();
            let widest = shift.iter().copied().fold(0.0_f64, f64::max);
            let in_front = behinds.iter().any(|b| *b != s);
            let racing = in_front
                && shift.last().is_some_and(|t| *t > AHEAD_TAIL * widest)
                && !settles_later(along, &reference, obstacles, reach, step, speed * pace, s, swerve, AHEAD_TAIL * widest);
            if racing {
                continue;
            }
            let late = LATE_COST * (guide_gap(&reference, &points, robot) - SWERVE_SLACK).max(0.0);
            let wide = widest + (widest - width).max(0.0).powi(2) / width;
            options.push((s, if in_front { AHEAD_COST } else { 1.0 } * wide + late, Some(g)));
        }
        if !met {
            options.push((0.0, 0.0, None));
        }
        for (s, shift, guide) in options {
            let kept = (s == last.0 || last.0 == 0.0) && (pace - last.1).abs() < 1e-9;
            let cost = (shift + PACE_COST * (1.0 - pace)) * if kept { 1.0 } else { SIDE_SWITCH_COST };
            if best.as_ref().is_none_or(|b| cost < b.0) {
                best = Some((cost, s, Avoidance { pace, reference: reference.clone(), guide }));
            }
        }
    }
    match best {
        Some((_, s, a)) => {
            *last = (s, a.pace);
            a
        }
        None => {
            let pace = *paces.last().unwrap_or(&1.0);
            *last = (0.0, pace);
            Avoidance { pace, reference: along.reference(far, step, speed), guide: None }
        }
    }
}

/// Shares of the reference speed an avoidance plan weighs, fastest first.
pub(crate) const AVOID_PACES: [f64; 6] = [1.0, 0.85, 0.7, 0.55, 0.4, 0.25];
/// Cost, in metres of detour, per metre the robot sits sideways of where a
/// detour starts beyond `SWERVE_SLACK`: a swerve begun too late.
const LATE_COST: f64 = 10.0;
/// Seconds ahead an avoidance plan looks at least while traffic crosses
/// the path, to see the crossing through.
const CROSSING_LOOKOUT_S: f64 = 15.0;
/// Share of its widest shift above which a detour in front of crossing
/// traffic still shifts at its end: racing the traffic past the reference.
const AHEAD_TAIL: f64 = 0.5;
/// Cost, in metres of detour, of giving up the whole reference speed.
const PACE_COST: f64 = 20.0;

/// Controls that drive `model` along `guide` by pure pursuit at `speed`:
/// the plan of an ancillary controller, sampled beside the noise (Biased-MPPI,
/// Trevisan & Alonso-Mora 2024), so a detour is among the candidates even
/// where noise round the previous plan would not find it.
pub(crate) fn pursue(
    model: &Model,
    guide: &[(f64, f64)],
    speed: f64,
    steer: &mut [f64],
    accel: &mut [f64],
    accel_lat: &mut [f64],
) {
    let (sb, ab) = (model.steer_bound(), model.accel_bound());
    let l = wheelbase(model.constraints);
    let near = if model.ackermann { 2.0 * l } else { 0.5 };
    let (mut x, mut y, mut yaw, mut v, mut vy) = (model.x0, model.y0, model.yaw0, model.v0, model.vy0);
    let mut from = 0;
    for t in 0..steer.len().min(accel.len()).min(accel_lat.len()) {
        let d = |p: &(f64, f64)| (p.0 - x).hypot(p.1 - y);
        from = (from..guide.len()).min_by(|a, b| d(&guide[*a]).total_cmp(&d(&guide[*b]))).unwrap_or(from);
        let lookahead = (v.abs() * PURSUE_TIME_S).max(near);
        let target = guide[from..].iter().find(|p| d(p) >= lookahead).unwrap_or(&guide[guide.len() - 1]);
        let dist = d(target).max(1e-3);
        let alpha = normalize_angle((target.1 - y).atan2(target.0 - x) - yaw);
        let kappa = 2.0 * alpha.sin() / dist;
        steer[t] = if model.ackermann { (kappa * l).atan() } else { speed.max(v.abs()) * kappa }.clamp(-sb, sb);
        accel[t] = ((speed - v) / model.dt).clamp(-ab, ab);
        accel_lat[t] = 0.0;
        model.step(&mut x, &mut y, &mut yaw, &mut v, &mut vy, steer[t], accel[t], 0.0);
    }
}

/// Seconds of speed the pursued detour's lookahead spans.
const PURSUE_TIME_S: f64 = 1.0;

/// Every how many samples one rollout is kept for display.
const ROLLOUT_STRIDE: usize = 16;

/// `(DETOURED_WEIGHT, |offset| + margin, offset)` of the widest `detoured`
/// point within `near`, else `(1, 0, 0)`: the easing of the lateral weight
/// there and the sideways offset it pulls towards.
pub(crate) fn detour_relax(detoured: &[(f64, f64, f64)], near: f64) -> impl Fn(f64, f64) -> (f64, f64, f64) + '_ {
    move |x, y| {
        detoured
            .iter()
            .filter(|(px, py, _)| (px - x).hypot(py - y) < near)
            .map(|(_, _, offset)| *offset)
            .reduce(|a, b| if b.abs() > a.abs() { b } else { a })
            .map_or((1.0, 0.0, 0.0), |offset| (DETOURED_WEIGHT, offset.abs() + ALLOW_MARGIN, offset))
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
    /// `(scale, allow, toward)`, see [`eased_lateral`]; a path's eased
    /// weight pulls towards `toward`.
    pub(crate) fn relax_with(&mut self, relax: &dyn Fn(f64, f64) -> (f64, f64, f64)) {
        match self {
            Track::Timed(r) => {
                let eased: Vec<(f64, f64, f64)> = (0..r.x.len()).map(|i| relax(r.x[i], r.y[i])).collect();
                r.lateral = eased.iter().map(|e| e.0.clamp(0.0, 1.0)).collect();
                r.allow = eased.iter().map(|e| e.1.max(0.0)).collect();
            }
            Track::Path(c) => c.relax_with(relax),
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
    use super::{Along, Reference, Swerve, detour_relax, eased_lateral, left_behind, left_by, pick_detour, plan_avoidance};
    use crate::types::{Footprint, GaussianMode, Obstacle, Path, RobotConstraints, SteeringType};
    use datapod::{Euler, Point, Pose, Quaternion};

    #[test]
    fn steers_past_traffic_crossing_into_its_way_rather_than_slowing() {
        let waypoints: Vec<Pose> = (0..=40)
            .map(|i| Pose { point: Point::new(i as f64, 0.0, 0.0), rotation: Quaternion::from_euler(Euler::new(0.0, 0.0, 0.0)) })
            .collect();
        let cum = crate::core::path::cumulative_lengths(&waypoints);
        let path = Path { waypoints, speeds: Vec::new(), is_closed: false };
        let along = Along { path: &path, cum: &cum, trajectory: None, clock: 0.0, from: 0.0 };
        let mut c = RobotConstraints::default();
        c.steering_type = SteeringType::Differential;
        c.max_linear_velocity = 1.0;
        c.footprint = Footprint::Polygon { points: vec![(0.5, 0.35), (-0.5, 0.35), (-0.5, -0.35), (0.5, -0.35)] };
        let (steps, dt) = (40, 0.2);
        let mode = GaussianMode {
            weight: 1.0,
            mean_x: vec![6.0; steps + 1],
            mean_y: (0..=steps).map(|k| 4.0 - 0.6 * k as f64 * dt).collect(),
            std_x: vec![0.05; steps + 1],
            std_y: vec![0.05; steps + 1],
        };
        let mover = Obstacle { id: 0, radius: 0.5, modes: vec![mode], dt };
        let reach = crate::core::obstacles::robot_radius(&c);
        let mut last = (0.0, 1.0);
        let plan = plan_avoidance(&along, &[mover], reach, Swerve::of(&c, 1.0), 30, 0.2, 1.0, (0.0, 0.0), &mut last);
        assert!(plan.guide.is_some(), "no detour, pace {}", plan.pace);
        assert!(plan.pace >= 0.75, "pace {}", plan.pace);
    }

    /// A straight reference along x from `robot_x` at 1 m/s and a mover
    /// crossing it at x = 15 from `mover_y`, 0.7 m/s to the left.
    fn crossing_ahead(robot_x: f64, mover_y: f64) -> (Reference, Obstacle) {
        let (n, step) = (60, 0.15);
        let reference = Reference {
            x: (0..=n).map(|i| robot_x + step * i as f64).collect(),
            y: vec![0.0; n + 1],
            yaw: vec![0.0; n + 1],
            v: vec![1.0; n + 1],
            ..Default::default()
        };
        let (steps, dt) = (40, 0.25);
        let mode = GaussianMode {
            weight: 1.0,
            mean_x: vec![15.0; steps + 1],
            mean_y: (0..=steps).map(|k| mover_y + 0.7 * k as f64 * dt).collect(),
            std_x: vec![0.05; steps + 1],
            std_y: vec![0.05; steps + 1],
        };
        (reference, Obstacle { id: 0, radius: 0.6, modes: vec![mode], dt })
    }

    #[test]
    fn ramps_a_detour_over_the_length_the_machine_can_swerve_in() {
        let (n, step) = (60, 0.25);
        let reference = Reference {
            x: (0..=n).map(|i| 0.3 * i as f64).collect(),
            y: vec![0.0; n + 1],
            yaw: vec![0.0; n + 1],
            v: vec![1.2; n + 1],
            ..Default::default()
        };
        let mode = GaussianMode { weight: 1.0, mean_x: vec![12.0; 2], mean_y: vec![0.0; 2], std_x: vec![0.0; 2], std_y: vec![0.0; 2] };
        let rock = Obstacle { id: 0, radius: 0.6, modes: vec![mode], dt: 1.0 };
        let shift_at = |swerve: Swerve, x: f64| {
            let g = reference.detour(std::slice::from_ref(&rock), 1.0, step, 1.0, 6, swerve).unwrap();
            let i = (x / 0.3).round() as usize;
            g.y[i]
        };
        let tractor = Swerve { radius: 5.3, lag: 3.75, ..Default::default() };
        assert_eq!(shift_at(Swerve::default(), 6.0), 0.0);
        assert!(shift_at(tractor, 6.0) > 0.2, "{}", shift_at(tractor, 6.0));
        assert!(shift_at(tractor, 12.0) >= 1.6);
    }

    #[test]
    fn passes_behind_traffic_crossing_from_the_right() {
        for (x, y) in [(5.9, -6.8), (8.8, -4.7), (11.0, -3.2)] {
            let (reference, mover) = crossing_ahead(x, y);
            let mut side = 0.0;
            pick_detour(&reference, &[mover], 0.6, 0.15, &mut side, Swerve::default());
            assert_eq!(side, -1.0, "robot at {x}, mover at {y}");
        }
    }

    #[test]
    fn detour_relax_pulls_towards_the_signed_offset_of_the_widest_detour() {
        let detoured = [(0.0, 0.0, 0.5), (1.0, 0.0, -1.2)];
        let relax = detour_relax(&detoured, 2.0);
        let (scale, allow, toward) = relax(0.5, 0.0);
        assert_eq!(scale, crate::pred::mpc::DETOURED_WEIGHT);
        assert!((allow - (1.2 + super::ALLOW_MARGIN)).abs() < 1e-12);
        assert_eq!(toward, -1.2);
        assert_eq!(relax(10.0, 0.0), (1.0, 0.0, 0.0));
    }

    #[test]
    fn left_by_signs_the_offset_to_the_left() {
        let reference = Reference { x: vec![0.0, 1.0, 2.0], y: vec![0.0; 3], yaw: vec![0.0; 3], v: vec![1.0; 3], ..Default::default() };
        let mut guide = reference.clone();
        guide.y = vec![0.0, 1.0, -0.5];
        let left = left_by(&reference, &guide);
        assert_eq!(left.len(), 2);
        assert!((left[0].2 - 1.0).abs() < 1e-12 && (left[1].2 + 0.5).abs() < 1e-12);
    }

    #[test]
    fn held_speed_keeps_the_last_command_unless_the_measurement_strays() {
        assert_eq!(super::held_speed(1.2, 1.0, 0.3), 1.0);
        assert_eq!(super::held_speed(0.8, 1.0, 0.3), 1.0);
        assert_eq!(super::held_speed(0.2, 1.0, 0.3), 0.2);
        assert_eq!(super::held_speed(-0.5, 0.0, 0.3), -0.5);
    }

    #[test]
    fn traffic_is_left_behind_once_past_the_rear() {
        let start = Reference { x: vec![10.0, 10.2], y: vec![0.0; 2], yaw: vec![0.0; 2], v: vec![1.0; 2], ..Default::default() };
        let at = |x: f64| {
            let mode = GaussianMode { weight: 1.0, mean_x: vec![x; 2], mean_y: vec![-2.0; 2], std_x: vec![0.0; 2], std_y: vec![0.0; 2] };
            Obstacle { id: 0, radius: 0.6, modes: vec![mode], dt: 0.2 }
        };
        assert!(left_behind(&at(7.0), &start, -1.5));
        assert!(!left_behind(&at(8.5), &start, -1.5));
        assert!(!left_behind(&at(14.0), &start, -1.5));
    }

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
