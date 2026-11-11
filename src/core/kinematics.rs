//! Steering-model helpers and the single place where a physical
//! `(v, omega)` pair becomes a `VelocityCommand`: curvature feasibility for
//! Ackermann, reverse policy, saturation and output-unit conversion.

use crate::core::math::normalize_angle;
use crate::types::{
    ControllerConfig, OutputUnits, RobotConstraints, RobotState, SteeringType, VelocityCommand,
};
use datapod::Point;

pub fn can_turn_in_place(steering: SteeringType) -> bool {
    !matches!(steering, SteeringType::Ackermann)
}

pub fn is_ackermann(steering: SteeringType) -> bool {
    matches!(steering, SteeringType::Ackermann)
}

/// True for platforms that can translate in any body-frame direction
/// independently of heading (omni/mecanum wheels), as opposed to a platform
/// that can only drive along its own forward axis.
pub fn is_holonomic(steering: SteeringType) -> bool {
    matches!(steering, SteeringType::Holonomic)
}

pub fn reverse_allowed(cfg: &ControllerConfig, state: &RobotState) -> bool {
    cfg.allow_reverse || state.allow_reverse
}

pub fn wheelbase(constraints: &RobotConstraints) -> f64 {
    if constraints.wheelbase > 1e-6 {
        constraints.wheelbase
    } else {
        1.0
    }
}

/// Distance, metres, the pose sits ahead of the axle the platform turns
/// about: the rear axle of an Ackermann platform, none for others.
pub fn axle_offset(constraints: &RobotConstraints) -> f64 {
    if is_ackermann(constraints.steering_type) { constraints.rear_wheelbase.max(0.0) } else { 0.0 }
}

/// Angle, radians, by which the heading of a pose `ahead` metres in front
/// of its turning axle trails the path in a bend of curvature `kappa` that
/// the pose follows; the yaw rate there is `v * kappa / cos(angle)`.
pub fn bend_heading(kappa: f64, ahead: f64) -> f64 {
    (ahead * kappa).clamp(-0.95, 0.95).asin()
}

/// `kappa` no sharper than the platform's tightest turn: the bend it drives
/// round a corner it cannot follow.
pub fn followable(kappa: f64, constraints: &RobotConstraints) -> f64 {
    let k = max_curvature(constraints);
    if k.is_finite() { kappa.clamp(-k, k) } else { kappa }
}

/// Heading of a body whose pose, `ahead` metres in front of its turning
/// axle, follows a path of travel `heading` and curvature `kappa`, driving
/// backwards when `back`.
pub fn body_heading_on_path(heading: f64, kappa: f64, ahead: f64, back: bool) -> f64 {
    if back {
        normalize_angle(heading + std::f64::consts::PI - bend_heading(kappa, -ahead))
    } else {
        normalize_angle(heading - bend_heading(kappa, ahead))
    }
}

/// Signed offset, metres (positive = left), from a bend of curvature
/// `kappa` that the pose follows to the bend traced by a point `lead`
/// metres ahead of the turning axle, for a pose `ahead` metres ahead of it.
pub fn axle_point_offset(kappa: f64, ahead: f64, lead: f64) -> f64 {
    let spread = lead * lead - ahead * ahead;
    -spread * kappa / ((1.0 + spread * kappa * kappa).max(0.0).sqrt() + 1.0)
}

/// Seconds a command takes to act: the time a rate-limited Ackermann
/// steering needs to swing half way to lock, at most 1.5 s; 0 for other
/// platforms.
pub fn actuation_delay(constraints: &RobotConstraints) -> f64 {
    if !is_ackermann(constraints.steering_type) {
        return 0.0;
    }
    let (lock, rate) = (constraints.max_steering_angle, constraints.max_steering_rate);
    if lock > 0.0 && rate > 0.0 { (0.5 * lock / rate).min(1.5) } else { 0.0 }
}

/// Steps and step of a predictive controller configured for `n` steps of
/// `dt` at `speed`: as configured, or a horizon long enough to stop twice
/// over with a second to react (to see crossing traffic in time to give way),
/// and on an Ackermann machine to span two of its tightest turning radii and
/// two half swings of its steering (to get round an obstacle), at most 12 s,
/// made of steps no longer than `MAX_PLANNING_STEP` (or `dt`, if longer).
pub fn planning(constraints: &RobotConstraints, speed: f64, n: usize, dt: f64) -> (usize, f64) {
    let n = n.max(1);
    let accel = constraints.max_linear_acceleration.abs();
    let stopping = if accel > 1e-6 { 2.0 * speed.abs() / accel + 1.0 } else { 0.0 };
    let k = max_curvature(constraints);
    let turning = if is_ackermann(constraints.steering_type) && k.is_finite() && k > 0.0 {
        2.0 / (k * speed.abs().max(0.1)) + 2.0 * actuation_delay(constraints)
    } else {
        0.0
    };
    let needed = stopping.max(turning).min(12.0);
    if needed <= n as f64 * dt {
        return (n, dt);
    }
    let step = (needed / n as f64).min(MAX_PLANNING_STEP.max(dt));
    ((needed / step).ceil() as usize, step)
}

/// Longest step `planning` stretches a horizon to, seconds.
pub const MAX_PLANNING_STEP: f64 = 0.25;

/// Largest path curvature the platform can follow. Unbounded for platforms
/// that can turn in place.
pub fn max_curvature(constraints: &RobotConstraints) -> f64 {
    if !is_ackermann(constraints.steering_type) {
        return f64::INFINITY;
    }
    let mut k = f64::INFINITY;
    if constraints.max_steering_angle > 0.0 {
        k = k.min(constraints.max_steering_angle.tan().abs() / wheelbase(constraints));
    }
    if constraints.min_turning_radius > 0.0 {
        k = k.min(1.0 / constraints.min_turning_radius);
    }
    k
}

pub fn steering_to_curvature(delta: f64, constraints: &RobotConstraints) -> f64 {
    delta.tan() / wheelbase(constraints)
}

pub fn curvature_to_steering(kappa: f64, constraints: &RobotConstraints) -> f64 {
    (kappa * wheelbase(constraints)).atan()
}

/// Longitudinal speed limits for this tick.
pub fn speed_bounds(constraints: &RobotConstraints, allow_reverse: bool) -> (f64, f64) {
    let hi = constraints.max_linear_velocity.max(0.0);
    let lo = if allow_reverse {
        constraints.min_linear_velocity.min(0.0)
    } else {
        0.0
    };
    (lo, hi)
}

/// Scale factor in `[floor, 1]` that slows the platform as the heading error
/// grows: `cos(err)` for platforms that can turn in place, with a floor for
/// Ackermann so it keeps rolling while it steers around.
pub fn heading_speed_scale(heading_error: f64, constraints: &RobotConstraints) -> f64 {
    let c = heading_error.cos();
    if is_ackermann(constraints.steering_type) {
        c.max(0.3)
    } else {
        c.max(0.0)
    }
}

/// Speed profile for path following: nominal speed reduced for path
/// curvature and tapered toward the goal so the platform stops inside the
/// tolerance instead of overshooting: no faster than `kp_linear` per metre
/// left, nor than braking at `max_linear_acceleration` stops it in time.
pub fn path_speed(
    nominal: f64,
    kappa: f64,
    dist_to_goal: f64,
    goal_tolerance: f64,
    kp_linear: f64,
    constraints: &RobotConstraints,
) -> f64 {
    let curvature_scale = 1.0 / (1.0 + 2.0 * kappa.abs() * wheelbase(constraints));
    let left = (dist_to_goal - 0.5 * goal_tolerance).max(0.0);
    let accel = constraints.max_linear_acceleration;
    let braking = if accel > 0.0 { (2.0 * accel * left).sqrt() } else { f64::INFINITY };
    let taper = (kp_linear.max(0.1) * left).min(braking).max(0.15 * constraints.max_linear_velocity.max(0.0));
    (nominal * curvature_scale.max(0.25)).min(taper).max(0.0)
}

/// Points within braking reach `braked_for_bends` reads the bends at.
const BRAKING_SAMPLES: usize = 10;

/// `speed`, capped so that braking at `max_linear_acceleration`, once the
/// actuation delay has passed, slows the robot from `v_now` to the
/// `path_speed` of `nominal` at the curvature `bend(sigma)` at each distance
/// `sigma` on within its braking distance, and no further than `reach`.
pub fn braked_for_bends(
    speed: f64,
    v_now: f64,
    nominal: f64,
    reach: f64,
    bend: impl Fn(f64) -> f64,
    constraints: &RobotConstraints,
) -> f64 {
    let accel = constraints.max_linear_acceleration.abs().max(0.1);
    let v_now = v_now.abs();
    let delay = actuation_delay(constraints);
    let ahead = (v_now * v_now / (2.0 * accel) + v_now * delay).min(reach.max(0.0));
    (1..=BRAKING_SAMPLES).fold(speed, |speed, i| {
        let sigma = ahead * i as f64 / BRAKING_SAMPLES as f64;
        let at_bend = path_speed(nominal, bend(sigma), f64::INFINITY, 0.0, 1.0, constraints);
        speed.min((at_bend * at_bend + 2.0 * accel * (sigma - v_now * delay).max(0.0)).sqrt())
    })
}

/// `(v, omega)` of a differential or skid-steer platform, scaled together
/// (keeping the curvature) so neither wheel of its `track_width` axle runs
/// faster than `max_linear_velocity`. Other platforms, and a zero track
/// width, pass through.
pub fn wheel_limited(v: f64, omega: f64, constraints: &RobotConstraints) -> (f64, f64) {
    let wheeled = matches!(constraints.steering_type, SteeringType::Differential | SteeringType::SkidSteer);
    let fastest = v.abs() + (0.5 * constraints.track_width * omega).abs();
    let cap = constraints.max_linear_velocity.max(0.0);
    if !wheeled || constraints.track_width <= 0.0 || fastest <= cap {
        return (v, omega);
    }
    let scale = cap / fastest;
    (v * scale, omega * scale)
}

/// Build the final command from a physical `(v, omega)` pair.
pub fn finalize(
    v: f64,
    omega: f64,
    constraints: &RobotConstraints,
    cfg: &ControllerConfig,
    allow_reverse: bool,
    message: &str,
) -> VelocityCommand {
    if !v.is_finite() || !omega.is_finite() {
        return VelocityCommand::invalid("non-finite command");
    }
    let (lo, hi) = speed_bounds(constraints, allow_reverse);
    let v = v.clamp(lo.min(hi), hi);

    let mut omega = omega;
    let mut steering_angle = 0.0;
    if is_ackermann(constraints.steering_type) {
        let kmax = max_curvature(constraints);
        let omega_max = if kmax.is_finite() {
            v.abs() * kmax
        } else {
            f64::INFINITY
        };
        omega = omega.clamp(-omega_max, omega_max);
        if v.abs() > 1e-6 {
            steering_angle = curvature_to_steering(omega / v, constraints);
        } else {
            omega = 0.0;
        }
    }
    let w_max = constraints.max_angular_velocity.max(0.0);
    let mut v = v;
    if omega.abs() > w_max {
        if v.abs() > 1e-6 {
            v *= w_max / omega.abs();
        }
        omega = omega.clamp(-w_max, w_max);
    }
    let (v, omega) = wheel_limited(v, omega, constraints);
    if is_ackermann(constraints.steering_type) && v.abs() > 1e-6 {
        steering_angle = curvature_to_steering(omega / v, constraints);
    }

    let (linear, angular) = match cfg.output_units {
        OutputUnits::Physical => (v, omega),
        OutputUnits::Normalized => (
            if hi > 0.0 { v / hi } else { 0.0 },
            if w_max > 0.0 { omega / w_max } else { 0.0 },
        ),
    };

    VelocityCommand {
        valid: true,
        status_message: message.into(),
        linear_velocity: linear,
        angular_velocity: angular,
        lateral_velocity: 0.0,
        steering_angle,
    }
}

/// Build the final command for a holonomic platform from a body-frame
/// `(vx, vy, omega)` triple: `vx` forward, `vy` left. Speed is clamped as a
/// vector (direction preserved) to `max_linear_velocity`; `omega` is
/// clamped independently to `max_angular_velocity`. Translation and
/// rotation are otherwise uncoupled, unlike `finalize`.
pub fn finalize_holonomic(
    vx: f64,
    vy: f64,
    omega: f64,
    constraints: &RobotConstraints,
    cfg: &ControllerConfig,
    message: &str,
) -> VelocityCommand {
    if !vx.is_finite() || !vy.is_finite() || !omega.is_finite() {
        return VelocityCommand::invalid("non-finite command");
    }
    let max_speed = constraints.max_linear_velocity.abs();
    let mag = vx.hypot(vy);
    let scale = if mag > max_speed && mag > 1e-12 {
        max_speed / mag
    } else {
        1.0
    };
    let vx = vx * scale;
    let vy = vy * scale;
    let w_max = constraints.max_angular_velocity.max(0.0);
    let omega = omega.clamp(-w_max, w_max);

    let (linear, lateral, angular) = match cfg.output_units {
        OutputUnits::Physical => (vx, vy, omega),
        OutputUnits::Normalized => (
            if max_speed > 0.0 { vx / max_speed } else { 0.0 },
            if max_speed > 0.0 { vy / max_speed } else { 0.0 },
            if w_max > 0.0 { omega / w_max } else { 0.0 },
        ),
    };

    VelocityCommand {
        valid: true,
        status_message: message.into(),
        linear_velocity: linear,
        angular_velocity: angular,
        lateral_velocity: lateral,
        steering_angle: 0.0,
    }
}

/// Rotate a world-frame vector into the body frame at `yaw`.
pub fn world_to_body(dx: f64, dy: f64, yaw: f64) -> (f64, f64) {
    let (s, c) = yaw.sin_cos();
    (c * dx + s * dy, -s * dx + c * dy)
}

/// Straight-line holonomic point control: translate directly toward
/// `target` at a speed shaped by `path_speed`, independently rotate toward
/// `target_yaw`. The point-goal fallback shared by the path followers on a
/// holonomic platform, where curvature-based geometry does not apply.
#[allow(clippy::too_many_arguments)]
pub fn holonomic_point_command(
    from: Point,
    yaw: f64,
    target: Point,
    target_yaw: f64,
    dist_to_goal: f64,
    cfg: &ControllerConfig,
    constraints: &RobotConstraints,
    message: &str,
) -> VelocityCommand {
    let dx = target.x - from.x;
    let dy = target.y - from.y;
    let d = dx.hypot(dy);
    let speed = path_speed(
        constraints.max_linear_velocity,
        0.0,
        dist_to_goal,
        cfg.goal_tolerance,
        cfg.kp_linear,
        constraints,
    );
    let (ux, uy) = if d > 1e-9 { (dx / d, dy / d) } else { (0.0, 0.0) };
    let (vx, vy) = world_to_body(speed * ux, speed * uy, yaw);
    let omega = cfg.kp_angular.max(0.1) * normalize_angle(target_yaw - yaw);
    finalize_holonomic(vx, vy, omega, constraints, cfg, message)
}

pub fn stop(message: &str) -> VelocityCommand {
    VelocityCommand {
        valid: true,
        status_message: message.into(),
        ..VelocityCommand::default()
    }
}

/// Steering-angle bound of `max_curvature`, which holds both
/// `max_steering_angle` and `min_turning_radius` (just under 90 degrees when
/// neither is set).
pub fn steering_limit(constraints: &RobotConstraints) -> f64 {
    let k = max_curvature(constraints);
    if k.is_finite() {
        curvature_to_steering(k, constraints).abs()
    } else {
        std::f64::consts::FRAC_PI_2 - 1e-6
    }
}

/// Speed floor for Ackermann platforms whose point goal cannot be reached
/// on any feasible arc from the current pose (the goal lies inside the
/// minimum turning circle). Slowing down cannot help there, so the platform
/// keeps rolling to complete a loop; `None` when the goal is reachable or
/// the platform can turn in place.
pub fn unreachable_arc_speed_floor(
    bearing_error: f64,
    distance: f64,
    constraints: &RobotConstraints,
) -> Option<f64> {
    if !is_ackermann(constraints.steering_type) || distance < 1e-6 {
        return None;
    }
    let kmax = max_curvature(constraints);
    if !kmax.is_finite() {
        return None;
    }
    let required = 2.0 * bearing_error.sin().abs() / distance;
    if required > kmax {
        Some(0.3 * constraints.max_linear_velocity.max(0.0))
    } else {
        None
    }
}

/// Body-frame `(vx, vy)` of a velocity given in the path frame: `along` the
/// path heading `path_heading` and `across` it to the left, for a robot
/// facing `yaw`.
pub fn path_to_body(along: f64, across: f64, path_heading: f64, yaw: f64) -> (f64, f64) {
    let (s, c) = path_heading.sin_cos();
    world_to_body(along * c - across * s, along * s + across * c, yaw)
}
