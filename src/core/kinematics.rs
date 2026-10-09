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
/// tolerance instead of overshooting.
pub fn path_speed(
    nominal: f64,
    kappa: f64,
    dist_to_goal: f64,
    goal_tolerance: f64,
    kp_linear: f64,
    constraints: &RobotConstraints,
) -> f64 {
    let curvature_scale = 1.0 / (1.0 + 2.0 * kappa.abs() * wheelbase(constraints));
    let taper = (kp_linear.max(0.1) * (dist_to_goal - 0.5 * goal_tolerance).max(0.0))
        .max(0.15 * constraints.max_linear_velocity.max(0.0));
    (nominal * curvature_scale.max(0.25)).min(taper).max(0.0)
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
    omega = omega.clamp(-w_max, w_max);
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
        ..VelocityCommand::default()
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
        ..VelocityCommand::default()
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

/// Steering-angle bound consistent with `max_curvature`: the configured
/// steering limit when set, otherwise the angle equivalent of the curvature
/// limit (or just under 90 degrees when unlimited).
pub fn steering_limit(constraints: &RobotConstraints) -> f64 {
    if constraints.max_steering_angle > 0.0 {
        return constraints.max_steering_angle.abs();
    }
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
