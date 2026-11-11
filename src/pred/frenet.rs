//! Frenet-frame optimal trajectories (Werling et al. 2010) for following a
//! path among moving obstacles. Each tick a lattice of candidates is laid in
//! the path's frame: a quintic sideways shift from the robot's offset,
//! heading and curvature to a sampled offset over a sampled length, driven at
//! a profile ramping to a sampled speed and braking for the path's end or
//! next cusp. Each is checked against the obstacles' predicted tracks; the
//! cheapest one that stays clear for the horizon (by time off the path,
//! steering, lost speed, closeness and change from the last choice) is
//! followed by pursuit of its point one lookahead on. With none clear, the
//! one that meets the obstacles latest is taken.

use crate::controller::{Controller, ControllerBase};
use crate::core::kinematics::{actuation_delay, finalize, is_ackermann, max_curvature, path_speed};
use crate::core::math::normalize_angle;
use crate::core::obstacles::{CollisionChecker, footprint_extent, robot_radius};
use crate::core::path::{
    PathCursor, curvature_at_arc_length, curvature_at_projection, cusp_after, direction_at, sample_past, sharpest_ahead,
    speed_cap,
};
use crate::pred::mppi::{Swerve, build_reference, pick_detour, prepare};
use crate::types::{DebugFrame, DebugLevel, Goal, Path, RobotConstraints, RobotState, VelocityCommand, WorldConstraints};
use datapod::Point;
use std::f64::consts::PI;

/// Share of the body's width kept as room from obstacles, when that is more
/// than `margin`.
const WIDTH_ROOM_SHARE: f64 = 0.25;
/// Lookahead shrink per unit curvature of the sharpest followable bend
/// within reach.
const LOOKAHEAD_BEND_GAIN: f64 = 1.0;
/// Metres a robot inside its margin may drift closer before a step counts
/// as meeting an obstacle.
const MARGIN_SLACK: f64 = 0.02;
/// Lengths of the sideways shift tried, as multiples of the shortest the
/// machine can drive it in.
const LENGTH_STRETCH: [f64; 3] = [1.0, 1.6, 2.5];
/// Heading error, radians, the shift's starting slope is taken at most.
const MAX_HEADING_ERROR: f64 = 1.2;
/// Weight of curvature beyond the machine's tightest.
const WEIGHT_STRAIN: f64 = 10.0;
/// Points along the path each shift's shape is judged at.
const SHAPE_SAMPLES: usize = 50;
/// Time constant, seconds, of the sideslip estimate, the largest it
/// reports, radians, and the speed, m/s, below which it holds.
const SLIP_TIME: f64 = 0.5;
const MAX_SLIP: f64 = 0.35;
const SLIP_SPEED: f64 = 0.2;
/// Time constant, seconds, over which the shortfall of curvature is
/// averaged, and the largest it may be as a share of the tightest.
const BIAS_TIME: f64 = 3.0;
const MAX_BIAS: f64 = 0.5;
/// Share of the tightest curvature past which a command counts as at the
/// limit, its shortfall the machine's and not a disturbance.
const SATURATED: f64 = 0.98;
/// Distance, metres, the chosen candidate's curvature is read ahead at
/// the least when reversing.
const MIN_PREVIEW: f64 = 0.2;
/// Points within braking distance the chosen shift's bends are read at.
const BRAKING_SAMPLES: usize = 10;
/// Step, seconds, the detour round what blocks the path is laid out at, the
/// share of top speed it is laid out for, and the steps it runs past the
/// horizon.
const DETOUR_DT: f64 = 0.2;
const DETOUR_SPEED_SHARE: f64 = 0.8;
const DETOUR_TAIL: usize = 6;
/// Share of the furthest offset a candidate may lie across the path from
/// the detour's side.
const SIDE_SLACK: f64 = 0.1;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct FrenetConfig {
    /// Seconds each candidate is driven and checked against the obstacles.
    pub horizon_s: f64,
    /// Step, seconds, of that check.
    pub dt: f64,
    /// Offsets tried across `[-max_offset, max_offset]` (made odd so the
    /// path itself is one).
    pub offset_samples: usize,
    /// Furthest offset, metres, either side of the path; 0 takes two body
    /// widths and margins.
    pub max_offset: f64,
    /// Speeds tried between standing and the path's speed, standing apart.
    pub speed_samples: usize,
    /// Room, metres, kept beyond the footprint.
    pub margin: f64,
    /// Clearance, metres beyond the margin, under which a candidate's
    /// closest approach costs.
    pub comfort: f64,
    pub weight_offset: f64,
    pub weight_turn: f64,
    pub weight_speed: f64,
    pub weight_clearance: f64,
    /// Weight of moving the target offset from the last tick's.
    pub weight_switch: f64,
    /// Seconds of travel on at which the chosen candidate's curvature is
    /// commanded; 0 pursues its point one lookahead on instead.
    pub preview_s: f64,
}

impl Default for FrenetConfig {
    fn default() -> Self {
        Self {
            horizon_s: 5.0,
            dt: 0.1,
            offset_samples: 13,
            max_offset: 0.0,
            speed_samples: 4,
            margin: 0.3,
            comfort: 1.0,
            weight_offset: 1.0,
            weight_turn: 1.0,
            weight_speed: 12.0,
            weight_clearance: 8.0,
            weight_switch: 0.5,
            preview_s: 0.1,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct FrenetFollower {
    pub base: ControllerBase,
    pub frenet_config: FrenetConfig,
    cursor: PathCursor,
    plan: Vec<Point>,
    offset: f64,
    side: f64,
    last_speed: f64,
    sideslip: f64,
    /// Whether the leg reverses and the curvature of travel commanded last
    /// tick, and how far the curvature falls short of the commands.
    last_command: Option<(bool, f64)>,
    bias: f64,
    debug_level: DebugLevel,
}

impl FrenetFollower {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Offset from the path as a quintic in arc length: from `d0` with slope
/// `d1` and second derivative `d2` to `to`, level, over `length` metres,
/// and held at `to` after.
#[derive(Clone, Copy, Debug)]
struct Shift {
    c: [f64; 6],
    length: f64,
    to: f64,
}

impl Shift {
    fn new(d0: f64, d1: f64, d2: f64, to: f64, length: f64) -> Self {
        let l = length.max(1e-3);
        let h = to - d0;
        let c3 = (20.0 * h - 12.0 * d1 * l - 3.0 * d2 * l * l) / (2.0 * l.powi(3));
        let c4 = (-30.0 * h + 16.0 * d1 * l + 3.0 * d2 * l * l) / (2.0 * l.powi(4));
        let c5 = (12.0 * h - 6.0 * d1 * l - d2 * l * l) / (2.0 * l.powi(5));
        Shift { c: [d0, d1, 0.5 * d2, c3, c4, c5], length: l, to }
    }

    /// Offset, slope and second derivative `sigma` metres on.
    fn at(&self, sigma: f64) -> (f64, f64, f64) {
        if sigma >= self.length {
            return (self.to, 0.0, 0.0);
        }
        let x = sigma.max(0.0);
        let c = &self.c;
        let d = c[0] + x * (c[1] + x * (c[2] + x * (c[3] + x * (c[4] + x * c[5]))));
        let d1 = c[1] + x * (2.0 * c[2] + x * (3.0 * c[3] + x * (4.0 * c[4] + x * 5.0 * c[5])));
        let d2 = 2.0 * c[2] + x * (6.0 * c[3] + x * (12.0 * c[4] + x * 20.0 * c[5]));
        (d, d1, d2)
    }
}

/// Curvature of the curve `d` off a path of curvature `kappa`, with slope
/// `d1` and second derivative `d2` in arc length (the path's change of
/// curvature neglected).
fn offset_curvature(kappa: f64, d: f64, d1: f64, d2: f64) -> f64 {
    let squeeze = 1.0 - kappa * d;
    (squeeze * (squeeze * kappa + d2) + 2.0 * kappa * d1 * d1) / (squeeze * squeeze + d1 * d1).max(1e-6).powf(1.5)
}

/// One candidate: its shift, target speed, cost and the time it first
/// meets an obstacle (infinite when it stays clear).
struct Candidate {
    shift: Shift,
    speed: f64,
    cost: f64,
    hit: f64,
}

impl Controller for FrenetFollower {
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
            Err(cmd) => return cmd,
        };
        let cfg = self.base.config.clone();
        let fc = self.frenet_config.clone();
        let (waypoints, speeds, cum) = (&self.base.path.waypoints, &self.base.path.speeds, &self.cursor.cum);
        let (x0, y0, yaw) = (state.pose.point.x, state.pose.point.y, state.pose.rotation.to_euler().yaw);
        let length = self.cursor.total_length();
        let cap = cusp_after(speeds, cum, prep.proj.arc_length).unwrap_or(length);
        let left0 = self.base.distance_left(&prep.proj, length).min(cap - prep.proj.arc_length).max(0.0);
        let back = prep.allow_reverse && direction_at(speeds, &prep.proj) == Some(-1);
        let travel = yaw + if back { PI } else { 0.0 };
        let body_turn = if back { PI } else { 0.0 };
        // An Ackermann machine is planned for by its rear axle, which moves
        // along its heading as the plan does; reversing, the plan runs on
        // past the leg's end for the pose trailing the axle to reach there.
        let axle = if is_ackermann(constraints.steering_type) { constraints.rear_wheelbase.max(0.0) } else { 0.0 };
        let lead = if back { axle } else { 0.0 };
        let (rx, ry) = (x0 - axle * yaw.cos(), y0 - axle * yaw.sin());
        let at = if axle > 0.0 {
            self.cursor
                .project_on_leg(waypoints, speeds, Point::new(rx, ry, 0.0), prep.proj.segment, prep.proj.arc_length)
                .unwrap_or(prep.proj)
        } else {
            prep.proj
        };
        let s0 = at.arc_length;
        let end = cap + lead;
        let bend_at = |s: f64| if s < cap { curvature_at_arc_length(waypoints, cum, s) } else { 0.0 };

        let width = (2.0 * footprint_extent(constraints).map_or(robot_radius(constraints), |(_, _, w)| w)).max(0.3);
        let margin = fc.margin.max(WIDTH_ROOM_SHARE * width);
        let checker = CollisionChecker::new(world, constraints, margin).with_step(fc.dt).with_spread(1.0);
        let k_lim = if max_curvature(constraints).is_finite() {
            max_curvature(constraints)
        } else {
            constraints.max_angular_velocity.abs() / 0.3
        };
        let v_max = constraints.max_linear_velocity.abs().max(0.1);
        let accel = constraints.max_linear_acceleration.abs().max(0.1);
        let nominal = speed_cap(speeds, &prep.proj).map_or(v_max, |c| c.min(v_max));
        let v_now = if state.velocity_valid { state.velocity.linear.abs() } else { self.last_speed };

        // The robot in the path's frame: offset, slope and second derivative.
        let kappa0 = curvature_at_projection(waypoints, cum, &at);
        let d0 = at.lateral_error;
        // The rear axle's sideslip, low-passed: the angle its travel makes
        // with the heading; a lateral speed of exactly zero is not measured.
        let forward = if back { -state.velocity.linear } else { state.velocity.linear };
        if state.velocity_valid && forward > SLIP_SPEED && state.velocity.lateral != 0.0 {
            let side = if back { -1.0 } else { 1.0 } * (state.velocity.lateral - state.velocity.angular * axle);
            let gain = (dt / SLIP_TIME).min(1.0);
            self.sideslip = (self.sideslip + gain * (side.atan2(forward) - self.sideslip)).clamp(-MAX_SLIP, MAX_SLIP);
        }
        let err = normalize_angle(travel + self.sideslip - at.heading).clamp(-MAX_HEADING_ERROR, MAX_HEADING_ERROR);
        let squeeze = (1.0 - kappa0 * d0).max(0.1);
        let d0p = squeeze * err.tan();
        let k_now = if state.velocity_valid && v_now > 0.1 { state.velocity.angular / v_now } else { kappa0 };
        // How far the curvature falls short of the one commanded, low-passed
        // over the leg: what a slope or slip takes off each command.
        match self.last_command {
            Some((was_back, commanded)) if was_back == back => {
                if state.velocity_valid && v_now > SLIP_SPEED && commanded.abs() < SATURATED * k_lim {
                    let gain = (dt / BIAS_TIME).min(1.0);
                    self.bias = (self.bias + gain * (commanded - k_now - self.bias)).clamp(-MAX_BIAS * k_lim, MAX_BIAS * k_lim);
                }
            }
            _ => self.bias = 0.0,
        }
        let d0pp = (squeeze / err.cos().powi(2) * (k_now * squeeze / err.cos() - kappa0)).clamp(-2.0 * k_lim, 2.0 * k_lim);

        // The lattice: offsets either side, the one held and the last chosen,
        // each over a few lengths, at a few speeds.
        let reach = if fc.max_offset > 0.0 { fc.max_offset } else { 2.0 * (width + margin) };
        let n_off = fc.offset_samples.max(1) | 1;
        let mut offsets: Vec<f64> = if n_off == 1 {
            vec![0.0]
        } else {
            (0..n_off).map(|i| -reach + 2.0 * reach * i as f64 / (n_off - 1) as f64).collect()
        };
        offsets.extend([d0.clamp(-reach, reach), self.offset.clamp(-reach, reach)]);
        // While what blocks the path ahead calls for a detour, the side it
        // takes, kept from tick to tick, is the only one tried.
        let obstacles = world.map_or(&[][..], |w| &w.obstacles[..]);
        if obstacles.is_empty() {
            self.side = 0.0;
        } else {
            let speed = (DETOUR_SPEED_SHARE * v_max).max(0.1);
            let steps = (nominal * fc.horizon_s / (speed * DETOUR_DT)).ceil() as usize + DETOUR_TAIL;
            let reference = build_reference(&self.base.path, cum, prep.proj.arc_length, steps, DETOUR_DT, speed, 0.0);
            let room = robot_radius(constraints) + margin;
            pick_detour(&reference, obstacles, room, DETOUR_DT, &mut self.side, Swerve::of(constraints, speed));
        }
        if self.side != 0.0 {
            let side = self.side;
            offsets.retain(|o| o * side >= -SIDE_SLACK * reach);
        }
        let n_speed = fc.speed_samples.max(1);
        let targets: Vec<f64> = (0..=n_speed).map(|i| nominal * i as f64 / n_speed as f64).collect();
        let swerve = Swerve::of(constraints, v_now.max(0.3 * nominal));
        let base_lookahead = cfg.lookahead_distance.max(0.5);
        let shifts: Vec<Shift> = offsets
            .iter()
            .flat_map(|&to| {
                let shortest = swerve.length(to - d0).max(base_lookahead);
                LENGTH_STRETCH.iter().map(move |m| Shift::new(d0, d0p, d0pp, to, m * shortest))
            })
            .collect();

        // Each shift's shape, judged over one stretch of path as long as the
        // longest of them: offset, steering and curvature past the tightest.
        let previous = self.offset;
        let stretch_len = shifts.iter().map(|s| s.length).fold(nominal * fc.horizon_s, f64::max);
        let shape = |shift: &Shift| -> f64 {
            let (mut off, mut turn, mut strain) = (0.0, 0.0, 0.0);
            for i in 1..=SHAPE_SAMPLES {
                let sigma = stretch_len * i as f64 / SHAPE_SAMPLES as f64;
                let (d, d1, d2) = shift.at(sigma);
                let kappa = bend_at(s0 + sigma);
                let k_x = offset_curvature(kappa, d, d1, d2);
                off += d.abs() / reach + (d / reach).powi(2);
                turn += (d2 / k_lim).powi(2);
                strain += (k_x.abs() / k_lim - 1.0).max(0.0).powi(2);
            }
            (fc.weight_offset * off + fc.weight_turn * turn + WEIGHT_STRAIN * strain) / SHAPE_SAMPLES as f64
                + fc.weight_switch * ((shift.to - previous) / reach).powi(2)
        };

        // Each candidate driven for the horizon: lost speed and closeness,
        // and the first step it comes nearer an obstacle than the margin allows.
        let steps = (fc.horizon_s / fc.dt).ceil().max(1.0) as usize;
        let floor = if checker.has_obstacles() { checker.clearance(0, x0, y0, yaw).min(0.0) - MARGIN_SLACK } else { 0.0 };
        let drive = |shift: &Shift, target: f64, mut trace: Option<&mut Vec<Point>>| -> (f64, f64) {
            let (mut s, mut v, mut px, mut py, mut heading) = (s0, v_now, rx, ry, travel);
            let mut v_free = v_now;
            let (mut slow, mut close) = (0.0, 0.0);
            let mut hit = f64::INFINITY;
            for k in 1..=steps {
                let (d, d1, d2) = shift.at(s - s0);
                let kappa = bend_at(s);
                let k_x = offset_curvature(kappa, d, d1, d2);
                let left = (left0 - (s - s0)).max(0.0);
                let feasible = path_speed(nominal, k_x, left, prep.pos_tol, cfg.kp_linear, constraints);
                let want = target.min(feasible);
                v += (want - v).clamp(-accel * fc.dt, accel * fc.dt);
                v_free += (feasible - v_free).clamp(-accel * fc.dt, accel * fc.dt);
                let stretch = ((1.0 - kappa * d).powi(2) + d1 * d1).sqrt().max(0.1);
                s = (s + v * fc.dt / stretch).min(end);
                let (d, _, _) = shift.at(s - s0);
                let (p, h) = sample_past(waypoints, cum, s, cap);
                let (x, y) = (p.x - d * h.sin(), p.y + d * h.cos());
                if (x - px).hypot(y - py) > 1e-4 {
                    heading = (y - py).atan2(x - px);
                }
                (px, py) = (x, y);
                let (bx, by) = (x + axle * (heading - body_turn).cos(), y + axle * (heading - body_turn).sin());
                if let Some(points) = trace.as_deref_mut() {
                    points.push(Point::new(bx, by, 0.0));
                }
                if checker.has_obstacles() {
                    let c = checker.clearance(k, bx, by, heading - body_turn);
                    if c < floor && hit.is_infinite() {
                        hit = k as f64 * fc.dt;
                    }
                    close = f64::max(close, (1.0 - c / fc.comfort.max(1e-3)).max(0.0).powi(2));
                }
                slow += (v_free - v).max(0.0) / v_max;
            }
            (fc.weight_speed * slow / steps as f64 + fc.weight_clearance * close, hit)
        };
        let candidates: Vec<Candidate> = shifts
            .iter()
            .flat_map(|shift| {
                let form = shape(shift);
                targets.iter().map(move |&speed| (*shift, speed, form))
            })
            .map(|(shift, speed, form)| {
                let (cost, hit) = drive(&shift, speed, None);
                Candidate { shift, speed, cost: form + cost, hit }
            })
            .collect();
        let clear = candidates.iter().filter(|c| c.hit.is_infinite()).min_by(|a, b| a.cost.total_cmp(&b.cost));
        let (chosen, mode) = match clear {
            Some(c) => (c, if checker.has_obstacles() { "frenet_avoiding" } else { "frenet_tracking" }),
            None => (
                candidates
                    .iter()
                    .max_by(|a, b| a.hit.total_cmp(&b.hit).then(b.cost.total_cmp(&a.cost)))
                    .expect("at least one candidate is tried"),
                "frenet_evading",
            ),
        };
        if self.debug_level != DebugLevel::Off {
            let mut points = vec![Point::new(x0, y0, 0.0)];
            drive(&chosen.shift, chosen.speed, Some(&mut points));
            self.plan = points;
        }
        self.offset = chosen.shift.to;

        // The chosen shift's own curvature `preview_s` of travel on, or with
        // no preview the pursuit of its point one lookahead on.
        let curvature = if fc.preview_s > 0.0 {
            let sigma = (v_now.max(0.1) * fc.preview_s).max(if back { MIN_PREVIEW } else { 0.0 }).min((end - s0).max(0.0));
            let (d, d1, d2) = chosen.shift.at(sigma);
            offset_curvature(bend_at(s0 + sigma), d, d1, d2)
        } else {
            let bend = sharpest_ahead(waypoints, cum, s0, base_lookahead, k_lim);
            let lookahead =
                (base_lookahead / (1.0 + LOOKAHEAD_BEND_GAIN * bend * base_lookahead)).clamp(0.5 * base_lookahead, base_lookahead);
            let s_aim = (s0 + lookahead).min(end);
            let (d_aim, _, _) = chosen.shift.at(s_aim - s0);
            let (p, h) = sample_past(waypoints, cum, s_aim, cap);
            let (dx, dy) = (p.x - d_aim * h.sin() - rx, p.y + d_aim * h.cos() - ry);
            let alpha = normalize_angle(dy.atan2(dx) - travel);
            2.0 * alpha.sin() / dx.hypot(dy).max(1e-6)
        }
        .clamp(-k_lim, k_lim);
        let curvature = (curvature + self.bias).clamp(-k_lim, k_lim);
        // No faster than braking at `accel`, after the actuation delay, slows
        // the robot to each bend of the chosen shift within reach.
        let delay = actuation_delay(constraints);
        let ahead = (v_now * v_now / (2.0 * accel) + v_now * delay).min((end - s0).max(0.0));
        let speed = (1..=BRAKING_SAMPLES).fold(
            chosen.speed.min(path_speed(nominal, curvature, left0, prep.pos_tol, cfg.kp_linear, constraints)),
            |speed, i| {
                let sigma = ahead * i as f64 / BRAKING_SAMPLES as f64;
                let (d, d1, d2) = chosen.shift.at(sigma);
                let bend = offset_curvature(bend_at(s0 + sigma), d, d1, d2);
                let at_bend = path_speed(nominal, bend, f64::INFINITY, prep.pos_tol, cfg.kp_linear, constraints);
                speed.min((at_bend * at_bend + 2.0 * accel * (sigma - v_now * delay).max(0.0)).sqrt())
            },
        );
        self.last_speed = speed;
        self.last_command = Some((back, curvature));
        self.base.status.mode = mode.into();
        let direction = if back { -1.0 } else { 1.0 };
        finalize(direction * speed, speed * curvature, constraints, &cfg, prep.allow_reverse, "Frenet tracking")
    }

    fn params(&self) -> serde_json::Value {
        serde_json::to_value(&self.frenet_config).unwrap_or_default()
    }

    fn set_params(&mut self, params: &serde_json::Value) -> Result<(), String> {
        self.frenet_config = crate::controller::merged(&self.frenet_config, params)?;
        Ok(())
    }

    fn set_debug(&mut self, level: DebugLevel) {
        self.debug_level = level;
    }

    fn predicted_trajectory(&self) -> Vec<Point> {
        self.plan.clone()
    }

    fn debug_frame(&self) -> DebugFrame {
        DebugFrame::default()
    }

    fn set_path(&mut self, path: Path) {
        self.cursor.set_path(&path.waypoints);
        self.cursor.hold_cusps = true;
        self.base.path = path;
        self.base.path_index = 0;
        self.base.status = Default::default();
        self.plan.clear();
        self.offset = 0.0;
        self.side = 0.0;
        self.last_speed = 0.0;
        self.sideslip = 0.0;
        self.last_command = None;
        self.bias = 0.0;
    }

    fn reset(&mut self) {
        self.set_path(Path::default());
    }

    fn get_type(&self) -> &'static str {
        "frenet_follower"
    }

    fn base(&self) -> &ControllerBase {
        &self.base
    }

    fn base_mut(&mut self) -> &mut ControllerBase {
        &mut self.base
    }
}

#[cfg(test)]
mod tests {
    use super::Shift;

    #[test]
    fn a_shift_meets_its_ends() {
        let shift = Shift::new(0.4, 0.2, -0.1, 2.0, 5.0);
        let (d, d1, d2) = shift.at(0.0);
        assert!((d - 0.4).abs() < 1e-9 && (d1 - 0.2).abs() < 1e-9 && (d2 + 0.1).abs() < 1e-9);
        let (d, d1, d2) = shift.at(5.0 - 1e-9);
        assert!((d - 2.0).abs() < 1e-6 && d1.abs() < 1e-6 && d2.abs() < 1e-6, "{d} {d1} {d2}");
        assert_eq!(shift.at(7.0), (2.0, 0.0, 0.0));
    }
}
