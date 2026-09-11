//! Obstacle queries shared by the obstacle-aware controllers: clearance of
//! a robot disc against the Gaussian-mode obstacles in `WorldConstraints`
//! at a given prediction step.

use crate::types::{Obstacle, RobotConstraints, WorldConstraints};

/// Disc radius covering the robot footprint (0.3 m when unset).
pub fn robot_radius(constraints: &RobotConstraints) -> f64 {
    let r = 0.5 * constraints.robot_width.hypot(constraints.robot_length);
    if r < 0.05 { 0.3 } else { r }
}

/// Obstacle mode positions at prediction `step`, holding the last entry
/// when the prediction is shorter: `(x, y, radius, weight)`.
pub fn obstacle_at(obs: &Obstacle, step: usize) -> impl Iterator<Item = (f64, f64, f64, f64)> + '_ {
    obs.modes.iter().filter_map(move |md| {
        if !md.weight.is_finite() || md.weight <= 0.0 || md.mean_x.is_empty() || md.mean_y.is_empty() {
            return None;
        }
        let i = step.min(md.mean_x.len() - 1).min(md.mean_y.len() - 1);
        Some((md.mean_x[i], md.mean_y[i], obs.radius.max(0.0), md.weight))
    })
}

/// Smallest clearance between a disc of `radius` at `(x, y)` and every
/// obstacle mode at `step`. Infinite without obstacles.
pub fn clearance(obstacles: &[Obstacle], step: usize, x: f64, y: f64, radius: f64) -> f64 {
    let mut best = f64::INFINITY;
    for obs in obstacles {
        for (ox, oy, r, _) in obstacle_at(obs, step) {
            best = best.min((x - ox).hypot(y - oy) - r - radius);
        }
    }
    best
}

pub fn world_clearance(world: Option<&WorldConstraints>, step: usize, x: f64, y: f64, radius: f64) -> f64 {
    world.map_or(f64::INFINITY, |w| clearance(&w.obstacles, step, x, y, radius))
}
