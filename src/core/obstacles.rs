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

/// Footprint-aware clearance against Gaussian obstacles and the occupancy
/// grid of a `WorldConstraints`. Built once per tick.
pub struct CollisionChecker<'a> {
    world: Option<&'a WorldConstraints>,
    /// Body-frame sample points of the footprint boundary.
    samples: Vec<(f64, f64)>,
    /// Disc radius used with the Gaussian obstacles.
    disc_radius: f64,
    /// Extra clearance treated as contact.
    pub margin: f64,
}

impl<'a> CollisionChecker<'a> {
    pub fn new(world: Option<&'a WorldConstraints>, constraints: &RobotConstraints, margin: f64) -> Self {
        let (samples, disc_radius) = match &constraints.footprint {
            crate::types::Footprint::Polygon { points } if points.len() >= 3 => {
                let mut s = Vec::new();
                let n = points.len();
                for i in 0..n {
                    let a = points[i];
                    let b = points[(i + 1) % n];
                    let len = (b.0 - a.0).hypot(b.1 - a.1);
                    let steps = (len / 0.1).ceil().max(1.0) as usize;
                    for k in 0..steps {
                        let t = k as f64 / steps as f64;
                        s.push((a.0 + t * (b.0 - a.0), a.1 + t * (b.1 - a.1)));
                    }
                }
                let r = points.iter().map(|p| p.0.hypot(p.1)).fold(0.0_f64, f64::max);
                (s, r)
            }
            crate::types::Footprint::Disc { radius } if *radius > 0.0 => (Vec::new(), *radius),
            _ => (Vec::new(), robot_radius(constraints)),
        };
        Self { world, samples, disc_radius, margin: margin.max(0.0) }
    }

    pub fn has_obstacles(&self) -> bool {
        self.world.is_some_and(|w| !w.obstacles.is_empty() || w.grid.is_some())
    }

    /// Clearance of the footprint at `(x, y, yaw)` for prediction `step`,
    /// with `margin` already subtracted. Infinite without obstacles.
    pub fn clearance(&self, step: usize, x: f64, y: f64, yaw: f64) -> f64 {
        let Some(world) = self.world else {
            return f64::INFINITY;
        };
        let mut best = f64::INFINITY;
        if !world.obstacles.is_empty() {
            if self.samples.is_empty() {
                best = best.min(clearance(&world.obstacles, step, x, y, self.disc_radius));
            } else {
                let (s, c) = yaw.sin_cos();
                for obs in &world.obstacles {
                    for (ox, oy, r, _) in obstacle_at(obs, step) {
                        let mut d = f64::INFINITY;
                        for (px, py) in &self.samples {
                            let wx = x + px * c - py * s;
                            let wy = y + px * s + py * c;
                            d = d.min((wx - ox).hypot(wy - oy));
                        }
                        best = best.min(d - r);
                    }
                }
            }
        }
        if let Some(grid) = &world.grid {
            if self.samples.is_empty() {
                best = best.min(grid.distance_to_occupied(x, y) - self.disc_radius);
            } else {
                let (s, c) = yaw.sin_cos();
                for (px, py) in &self.samples {
                    let wx = x + px * c - py * s;
                    let wy = y + px * s + py * c;
                    best = best.min(grid.distance_to_occupied(wx, wy));
                }
                if grid.is_occupied(x, y) {
                    best = best.min(-0.5 * grid.resolution);
                }
            }
        }
        best - self.margin
    }

    pub fn collides(&self, step: usize, x: f64, y: f64, yaw: f64) -> bool {
        self.clearance(step, x, y, yaw) < 0.0
    }

    /// Bounding disc radius of the footprint.
    pub fn radius(&self) -> f64 {
        self.disc_radius
    }
}
