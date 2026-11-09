//! Control barrier function safety filter (Ames et al. 2019) on the final
//! `(v, ω)` command against moving obstacles. Every footprint disc and
//! obstacle mode gives a barrier `h`, the gap between them less the mode's
//! spread and a margin, held to `ḣ ≥ −γ·h` with each mode moving at its
//! predicted velocity (the time-varying CBF of moving obstacles). A disc
//! offset along the body makes `ω` enter `ḣ`, so the filter steers as well
//! as brakes. The command closest to the nominal one, by speed and a
//! weighted turn rate, is found exactly: with two inputs, the optimum of the
//! quadratic programme is the nominal command, its projection on one
//! constraint, or the meeting point of two.

use crate::core::kinematics::{is_ackermann, max_curvature};
use crate::core::obstacles::{footprint_discs, modes_at};
use crate::types::{RobotConstraints, WorldConstraints};
use datapod::Pose;

/// Tuning of the safety filter.
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct SafetyConfig {
    pub enabled: bool,
    /// Barrier decay rate, 1/s: the share of the remaining gap the robot may
    /// close per second.
    pub gamma: f64,
    /// Room, metres, kept beyond the footprint and each mode's spread.
    pub margin: f64,
    /// Standard deviations of a mode's predicted spread added to its radius.
    pub spread: f64,
    /// Gap, metres, beyond which an obstacle is left out.
    pub range: f64,
    /// Weight of a turn-rate change against a speed change, m².
    pub turn_weight: f64,
}

impl Default for SafetyConfig {
    fn default() -> Self {
        Self { enabled: false, gamma: 1.0, margin: 0.1, spread: 1.0, range: 4.0, turn_weight: 0.25 }
    }
}

/// `a_v · v + a_w · ω ≥ b`.
#[derive(Clone, Copy, Debug)]
struct Halfplane {
    a_v: f64,
    a_w: f64,
    b: f64,
}

impl Halfplane {
    fn slack(&self, v: f64, w: f64) -> f64 {
        self.a_v * v + self.a_w * w - self.b
    }
}

/// The command `(v, ω)` closest to `nominal` that keeps every barrier of the
/// robot at `pose` against `world`'s obstacles, with lateral speed `vy` held
/// as given; `None` when the nominal command already keeps them all.
pub fn filter(
    cfg: &SafetyConfig,
    constraints: &RobotConstraints,
    pose: &Pose,
    nominal: (f64, f64),
    vy: f64,
    world: Option<&WorldConstraints>,
) -> Option<(f64, f64)> {
    let world = world.filter(|w| cfg.enabled && !w.obstacles.is_empty())?;
    let (x, y) = (pose.point.x, pose.point.y);
    let yaw = pose.rotation.to_euler().yaw;
    let (s, c) = yaw.sin_cos();
    let mut planes = Vec::new();
    for (d, r) in footprint_discs(constraints) {
        let (px, py) = (x + d * c, y + d * s);
        for obstacle in &world.obstacles {
            let (Some(now), Some(soon)) = (modes_at(obstacle, 0.0, VELOCITY_DT).next(), modes_at(obstacle, VELOCITY_DT, VELOCITY_DT).next()) else {
                continue;
            };
            let (ox, oy) = (now.x, now.y);
            let (ovx, ovy) = ((soon.x - now.x) / VELOCITY_DT, (soon.y - now.y) / VELOCITY_DT);
            let (dx, dy) = (px - ox, py - oy);
            let dist = dx.hypot(dy).max(1e-6);
            let h = dist - r - now.radius - cfg.spread * now.std_x.max(now.std_y) - cfg.margin;
            if h > cfg.range {
                continue;
            }
            let (nx, ny) = (dx / dist, dy / dist);
            let along = nx * c + ny * s;
            let across = -nx * s + ny * c;
            planes.push(Halfplane { a_v: along, a_w: d * across, b: nx * ovx + ny * ovy - cfg.gamma * h - vy * across });
        }
    }
    if planes.iter().all(|p| p.slack(nominal.0, nominal.1) >= -1e-9) {
        return None;
    }
    planes.extend(bounds(constraints, nominal.0));
    Some(closest(&planes, nominal, cfg.turn_weight.max(1e-6)))
}

/// Speed and turn-rate limits of the platform, in the direction of travel
/// of `v_nominal`, as half-planes.
fn bounds(constraints: &RobotConstraints, v_nominal: f64) -> Vec<Halfplane> {
    let v_max = constraints.max_linear_velocity.abs();
    let w_max = constraints.max_angular_velocity.abs();
    let mut out = vec![Halfplane { a_v: 0.0, a_w: 1.0, b: -w_max }, Halfplane { a_v: 0.0, a_w: -1.0, b: -w_max }];
    let dir = if v_nominal < 0.0 { -1.0 } else { 1.0 };
    out.push(Halfplane { a_v: dir, a_w: 0.0, b: 0.0 });
    out.push(Halfplane { a_v: -dir, a_w: 0.0, b: -v_max });
    let kappa = max_curvature(constraints);
    if is_ackermann(constraints.steering_type) && kappa.is_finite() {
        out.push(Halfplane { a_v: dir * kappa, a_w: -1.0, b: 0.0 });
        out.push(Halfplane { a_v: dir * kappa, a_w: 1.0, b: 0.0 });
    }
    out
}

/// The point of the half-planes' intersection closest to `nominal` under the
/// metric `dv² + turn_weight · dω²`; when they do not meet, the candidate
/// that violates them least.
fn closest(planes: &[Halfplane], nominal: (f64, f64), turn_weight: f64) -> (f64, f64) {
    let (vn, wn) = nominal;
    let mut candidates = vec![nominal];
    for p in planes {
        let norm = p.a_v * p.a_v + p.a_w * p.a_w / turn_weight;
        if norm > 1e-12 {
            let t = -p.slack(vn, wn) / norm;
            candidates.push((vn + t * p.a_v, wn + t * p.a_w / turn_weight));
        }
    }
    for (i, p) in planes.iter().enumerate() {
        for q in &planes[i + 1..] {
            let det = p.a_v * q.a_w - p.a_w * q.a_v;
            if det.abs() > 1e-12 {
                candidates.push(((p.b * q.a_w - p.a_w * q.b) / det, (p.a_v * q.b - p.b * q.a_v) / det));
            }
        }
    }
    let cost = |(v, w): (f64, f64)| (v - vn).powi(2) + turn_weight * (w - wn).powi(2);
    let violation = |(v, w): (f64, f64)| planes.iter().map(|p| (-p.slack(v, w)).max(0.0)).fold(0.0_f64, f64::max);
    candidates
        .into_iter()
        .min_by(|a, b| {
            let (va, vb) = (violation(*a), violation(*b));
            if (va - vb).abs() > 1e-9 { va.total_cmp(&vb) } else { cost(*a).total_cmp(&cost(*b)) }
        })
        .unwrap_or(nominal)
}

/// Seconds over which an obstacle's velocity is read from its prediction.
const VELOCITY_DT: f64 = 0.2;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Footprint, GaussianMode, Obstacle, SteeringType};
    use datapod::{Euler, Point, Quaternion};

    fn constraints() -> RobotConstraints {
        let mut c = RobotConstraints::default();
        c.steering_type = SteeringType::Differential;
        c.max_linear_velocity = 1.5;
        c.max_angular_velocity = 1.0;
        c.footprint = Footprint::Disc { radius: 0.5 };
        c
    }

    fn pose(x: f64, y: f64, yaw: f64) -> Pose {
        Pose { point: Point::new(x, y, 0.0), rotation: Quaternion::from_euler(Euler::new(0.0, 0.0, yaw)) }
    }

    fn world(x: f64, y: f64, vx: f64, vy: f64) -> WorldConstraints {
        let steps = 10;
        let mode = GaussianMode {
            weight: 1.0,
            mean_x: (0..=steps).map(|k| x + vx * k as f64 * 0.2).collect(),
            mean_y: (0..=steps).map(|k| y + vy * k as f64 * 0.2).collect(),
            std_x: vec![0.0; steps + 1],
            std_y: vec![0.0; steps + 1],
        };
        WorldConstraints { obstacles: vec![Obstacle { id: 0, radius: 0.5, modes: vec![mode], dt: 0.2 }], ..Default::default() }
    }

    #[test]
    fn leaves_a_safe_command_alone() {
        let cfg = SafetyConfig { enabled: true, ..SafetyConfig::default() };
        assert!(filter(&cfg, &constraints(), &pose(0.0, 0.0, 0.0), (1.0, 0.0), 0.0, Some(&world(10.0, 3.0, 0.0, 0.0))).is_none());
    }

    #[test]
    fn brakes_for_an_obstacle_dead_ahead() {
        let cfg = SafetyConfig { enabled: true, ..SafetyConfig::default() };
        let (v, _) = filter(&cfg, &constraints(), &pose(0.0, 0.0, 0.0), (1.0, 0.0), 0.0, Some(&world(1.6, 0.0, 0.0, 0.0))).unwrap();
        assert!(v <= 0.5 + 1e-6, "v {v}");
    }

    #[test]
    fn backs_off_an_obstacle_closing_head_on() {
        let cfg = SafetyConfig { enabled: true, ..SafetyConfig::default() };
        let (v, _) = filter(&cfg, &constraints(), &pose(0.0, 0.0, 0.0), (0.5, 0.0), 0.0, Some(&world(1.3, 0.0, -1.0, 0.0))).unwrap();
        assert!(v <= 1e-9, "v {v}");
    }
}
