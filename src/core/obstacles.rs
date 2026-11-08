//! Obstacle queries shared by the obstacle-aware controllers: where each
//! predicted obstacle is at a given time, and the clearance of the robot
//! footprint against those obstacles and the occupancy grid.

use crate::types::{Obstacle, RobotConstraints, WorldConstraints};

/// Disc radius covering the robot footprint (0.3 m when unset).
pub fn robot_radius(constraints: &RobotConstraints) -> f64 {
    match &constraints.footprint {
        crate::types::Footprint::Disc { radius } if *radius > 0.0 => *radius,
        crate::types::Footprint::Polygon { points } if points.len() >= 3 => {
            points.iter().map(|p| p.0.hypot(p.1)).fold(0.0_f64, f64::max)
        }
        _ => {
            let r = 0.5 * constraints.robot_width.hypot(constraints.robot_length);
            if r < 0.05 { 0.3 } else { r }
        }
    }
}

/// One mode of an obstacle at a moment of its prediction.
#[derive(Clone, Copy, Debug)]
pub struct ModeAt {
    pub x: f64,
    pub y: f64,
    pub std_x: f64,
    pub std_y: f64,
    pub weight: f64,
    pub radius: f64,
}

/// Value of a per-step series at fractional step `u`: linear between steps,
/// continued along the last step past the end.
fn series_at(v: &[f64], u: f64) -> f64 {
    match v.len() {
        0 => 0.0,
        1 => v[0],
        n => {
            let i = (u.max(0.0).floor() as usize).min(n - 2);
            v[i] + (u - i as f64) * (v[i + 1] - v[i])
        }
    }
}

/// Spread of a per-step series at fractional step `u`: as `series_at`, never
/// negative, and never below the last entry past the end.
fn spread_at(v: &[f64], u: f64) -> f64 {
    let s = series_at(v, u);
    let s = if u > v.len().saturating_sub(1) as f64 { s.max(v.last().copied().unwrap_or(0.0)) } else { s };
    if s.is_finite() { s.max(0.0) } else { 0.0 }
}

/// The modes of `obs` at `t` seconds from now. Its prediction steps are
/// `obs.dt` apart, or `step` apart when `obs.dt` is 0.
pub fn modes_at(obs: &Obstacle, t: f64, step: f64) -> impl Iterator<Item = ModeAt> + '_ {
    let h = if obs.dt > 0.0 { obs.dt } else { step.max(1e-9) };
    let u = (t / h).max(0.0);
    obs.modes.iter().filter_map(move |md| {
        let n = md.mean_x.len().min(md.mean_y.len());
        if !(md.weight.is_finite() && md.weight > 0.0) || n == 0 {
            return None;
        }
        Some(ModeAt {
            x: series_at(&md.mean_x[..n], u),
            y: series_at(&md.mean_y[..n], u),
            std_x: spread_at(&md.std_x, u),
            std_y: spread_at(&md.std_y, u),
            weight: md.weight,
            radius: obs.radius.max(0.0),
        })
    })
}

/// Smallest clearance between a disc of `radius` at `(x, y)` and every
/// obstacle mode at `t` seconds, for a controller planning in `step`s.
pub fn clearance_at(obstacles: &[Obstacle], t: f64, step: f64, x: f64, y: f64, radius: f64) -> f64 {
    let mut best = f64::INFINITY;
    for obs in obstacles {
        for m in modes_at(obs, t, step) {
            best = best.min((x - m.x).hypot(y - m.y) - m.radius - radius);
        }
    }
    best
}

/// Signed distance from body-frame point `(x, y)` to the polygon `poly`:
/// negative inside.
fn polygon_distance(poly: &[(f64, f64)], x: f64, y: f64) -> f64 {
    let n = poly.len();
    let mut d = f64::INFINITY;
    let mut inside = false;
    for i in 0..n {
        let (ax, ay) = poly[i];
        let (bx, by) = poly[(i + 1) % n];
        let (ex, ey) = (bx - ax, by - ay);
        let ll = ex * ex + ey * ey;
        let t = if ll > 0.0 { (((x - ax) * ex + (y - ay) * ey) / ll).clamp(0.0, 1.0) } else { 0.0 };
        d = d.min((ax + t * ex - x).hypot(ay + t * ey - y));
        if (ay > y) != (by > y) && x < ax + (y - ay) * ex / ey {
            inside = !inside;
        }
    }
    if inside { -d } else { d }
}

/// Footprint-aware clearance against predicted obstacles and the occupancy
/// grid of a `WorldConstraints`. Built once per tick.
pub struct CollisionChecker<'a> {
    world: Option<&'a WorldConstraints>,
    /// Body-frame footprint polygon; empty for a disc.
    polygon: Vec<(f64, f64)>,
    /// Body-frame points along the polygon boundary, for the grid.
    samples: Vec<(f64, f64)>,
    /// Disc radius, or the polygon's bounding radius.
    disc_radius: f64,
    /// Seconds per controller step, for `clearance(step, ..)`.
    step: f64,
    /// Obstacles grow by this many standard deviations of their predicted
    /// spread.
    spread: f64,
    /// Extra clearance treated as contact.
    pub margin: f64,
}

impl<'a> CollisionChecker<'a> {
    pub fn new(world: Option<&'a WorldConstraints>, constraints: &RobotConstraints, margin: f64) -> Self {
        let (polygon, samples) = match &constraints.footprint {
            crate::types::Footprint::Polygon { points } if points.len() >= 3 => {
                let mut s = Vec::new();
                let n = points.len();
                for i in 0..n {
                    let a = points[i];
                    let b = points[(i + 1) % n];
                    let steps = ((b.0 - a.0).hypot(b.1 - a.1) / 0.1).ceil().max(1.0) as usize;
                    for k in 0..steps {
                        let t = k as f64 / steps as f64;
                        s.push((a.0 + t * (b.0 - a.0), a.1 + t * (b.1 - a.1)));
                    }
                }
                (points.clone(), s)
            }
            _ => (Vec::new(), Vec::new()),
        };
        Self { world, polygon, samples, disc_radius: robot_radius(constraints), step: 0.1, spread: 0.0, margin: margin.max(0.0) }
    }

    /// Seconds per controller step, so `clearance(step, ..)` can read
    /// predictions timed with `Obstacle::dt`.
    pub fn with_step(mut self, seconds: f64) -> Self {
        if seconds > 0.0 {
            self.step = seconds;
        }
        self
    }

    /// Grow every predicted obstacle by `k` standard deviations of its
    /// spread at the moment checked.
    pub fn with_spread(mut self, k: f64) -> Self {
        self.spread = k.max(0.0);
        self
    }

    pub fn has_obstacles(&self) -> bool {
        self.world.is_some_and(|w| !w.obstacles.is_empty() || w.grid.is_some())
    }

    /// Clearance of the footprint at `(x, y, yaw)` at controller step `step`,
    /// with `margin` already subtracted. Infinite without obstacles.
    pub fn clearance(&self, step: usize, x: f64, y: f64, yaw: f64) -> f64 {
        self.clearance_at(step as f64 * self.step, x, y, yaw)
    }

    /// Clearance of the footprint at `(x, y, yaw)` against the obstacles as
    /// predicted `t` seconds from now, with `margin` already subtracted.
    pub fn clearance_at(&self, t: f64, x: f64, y: f64, yaw: f64) -> f64 {
        let Some(world) = self.world else {
            return f64::INFINITY;
        };
        let mut best = f64::INFINITY;
        if !world.obstacles.is_empty() {
            let (s, c) = yaw.sin_cos();
            for obs in &world.obstacles {
                for m in modes_at(obs, t, self.step) {
                    let grow = m.radius + self.spread * m.std_x.max(m.std_y);
                    let (dx, dy) = (m.x - x, m.y - y);
                    let d = if self.polygon.is_empty() {
                        dx.hypot(dy) - self.disc_radius
                    } else {
                        polygon_distance(&self.polygon, c * dx + s * dy, -s * dx + c * dy)
                    };
                    best = best.min(d - grow);
                }
            }
        }
        if let Some(grid) = &world.grid {
            if self.samples.is_empty() {
                best = best.min(grid.distance_to_occupied(x, y) - self.disc_radius);
            } else {
                let (s, c) = yaw.sin_cos();
                for (px, py) in &self.samples {
                    best = best.min(grid.distance_to_occupied(x + px * c - py * s, y + px * s + py * c));
                }
                if grid.is_occupied(x, y) {
                    best = best.min(-0.5 * grid.resolution);
                }
            }
        }
        best - self.margin
    }

    /// Clearance of the footprint at `(x, y, yaw)` against the occupancy grid
    /// alone, with `margin` subtracted. Infinite without a grid.
    pub fn grid_clearance(&self, x: f64, y: f64, yaw: f64) -> f64 {
        let Some(grid) = self.world.and_then(|w| w.grid.as_ref()) else {
            return f64::INFINITY;
        };
        let mut best = if self.samples.is_empty() {
            grid.distance_to_occupied(x, y) - self.disc_radius
        } else {
            let (s, c) = yaw.sin_cos();
            self.samples
                .iter()
                .map(|(px, py)| grid.distance_to_occupied(x + px * c - py * s, y + px * s + py * c))
                .fold(f64::INFINITY, f64::min)
        };
        if grid.is_occupied(x, y) {
            best = best.min(-0.5 * grid.resolution);
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

/// Discs covering the footprint along the body x axis: `(x offset, radius)`.
/// A box `L × W` gets `⌈L / W⌉` discs, which hug a long vehicle far closer
/// than one disc around it.
/// An exponential obstacle barrier at one pose, with its derivatives in the
/// pose's `(x, y, yaw)`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Barrier {
    pub cost: f64,
    pub grad: [f64; 3],
    /// Gauss–Newton Hessian: positive semi-definite.
    pub hess: [[f64; 3]; 3],
}

/// `weight · exp(−d / scale)` summed over every footprint disc and obstacle
/// mode at `t` seconds, `d` the clearance between disc and mode with the
/// mode's radius grown by `spread` standard deviations. Discs are
/// `(offset ahead of (x, y), radius)`.
#[allow(clippy::too_many_arguments)]
pub fn obstacle_barrier(
    obstacles: &[Obstacle],
    discs: &[(f64, f64)],
    t: f64,
    step: f64,
    spread: f64,
    (x, y, yaw): (f64, f64, f64),
    weight: f64,
    scale: f64,
) -> Barrier {
    let mut b = Barrier::default();
    if weight <= 0.0 || scale <= 0.0 {
        return b;
    }
    let (s, c) = yaw.sin_cos();
    for &(off, r) in discs {
        let (px, py) = (x + off * c, y + off * s);
        for obs in obstacles {
            for m in modes_at(obs, t, step) {
                let (dx, dy) = (px - m.x, py - m.y);
                let dist = dx.hypot(dy).max(1e-6);
                let d = dist - m.radius - r - spread * m.std_x.max(m.std_y);
                if d > 8.0 * scale {
                    continue;
                }
                let q = weight * (-d / scale).min(30.0).exp();
                let (nx, ny) = (dx / dist, dy / dist);
                let g = [nx, ny, off * (ny * c - nx * s)];
                b.cost += q;
                for k in 0..3 {
                    b.grad[k] -= q / scale * g[k];
                    for l in 0..3 {
                        b.hess[k][l] += q / (scale * scale) * g[k] * g[l];
                    }
                }
            }
        }
    }
    b
}

pub fn footprint_discs(constraints: &RobotConstraints) -> Vec<(f64, f64)> {
    let (front, back, half_width) = match &constraints.footprint {
        crate::types::Footprint::Polygon { points } if points.len() >= 3 => {
            let front = points.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
            let back = points.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
            let half = points.iter().map(|p| p.1.abs()).fold(0.0_f64, f64::max);
            (front, back, half)
        }
        crate::types::Footprint::Disc { radius } if *radius > 0.0 => return vec![(0.0, *radius)],
        _ if constraints.robot_length > 0.0 && constraints.robot_width > 0.0 => {
            (0.5 * constraints.robot_length, -0.5 * constraints.robot_length, 0.5 * constraints.robot_width)
        }
        _ => return vec![(0.0, robot_radius(constraints))],
    };
    let length = (front - back).max(1e-3);
    let width = (2.0 * half_width).max(1e-3);
    let n = (length / width).ceil().max(1.0) as usize;
    let half_step = 0.5 * length / n as f64;
    let radius = half_width.hypot(half_step);
    (0..n).map(|i| (back + half_step * (2 * i + 1) as f64, radius)).collect()
}

/// The fastest of `shares` of the command `(v, omega)` whose arc from
/// `(x, y, yaw)`, held for `lookout` seconds plus the time to stop from it at
/// `decel`, stays clear of `checker`'s obstacles as they move. When none
/// does, the share (standing still included) keeping the most room.
pub fn yield_share(
    checker: &CollisionChecker,
    (x, y, yaw): (f64, f64, f64),
    (v, omega): (f64, f64),
    lookout: f64,
    decel: f64,
    shares: &[f64],
) -> f64 {
    if !checker.has_obstacles() {
        return 1.0;
    }
    let room = |s: f64| {
        let (u, w) = (v * s, omega * s);
        let horizon = lookout + u.abs() / decel.max(1e-3);
        let step = 0.05;
        let (mut px, mut py, mut th, mut t) = (x, y, yaw, 0.0);
        let mut worst = f64::INFINITY;
        while t <= horizon {
            worst = worst.min(checker.clearance_at(t, px, py, th));
            px += u * step * th.cos();
            py += u * step * th.sin();
            th += w * step;
            t += step;
        }
        worst
    };
    shares
        .iter()
        .copied()
        .find(|s| room(*s) >= 0.0)
        .unwrap_or_else(|| shares.iter().copied().chain([0.0]).max_by(|a, b| room(*a).total_cmp(&room(*b))).unwrap_or(0.0))
}
