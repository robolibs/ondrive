//! Obstacle queries shared by the obstacle-aware controllers: where each
//! predicted obstacle is at a given time, and the clearance of the robot
//! footprint against those obstacles and the occupancy grid.

use crate::core::buckets::Buckets;
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
            let i = (u.max(0.0) as usize).min(n - 2);
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

/// For each mode of `obs` `modes_at` yields, in order, the most metres per
/// second its mean moves and the most its spread grows per second; infinite
/// where its prediction is not all finite.
pub fn mode_rates(obs: &Obstacle, step: f64) -> Vec<(f64, f64)> {
    let h = if obs.dt > 0.0 { obs.dt } else { step.max(1e-9) };
    let finite = |v: &[f64]| v.iter().all(|a| a.is_finite());
    let slope = |v: &[f64]| v.windows(2).map(|w| (w[1] - w[0]).abs()).fold(0.0_f64, f64::max);
    obs.modes
        .iter()
        .filter(|md| md.weight.is_finite() && md.weight > 0.0 && md.mean_x.len().min(md.mean_y.len()) > 0)
        .map(|md| {
            let n = md.mean_x.len().min(md.mean_y.len());
            let (x, y) = (&md.mean_x[..n], &md.mean_y[..n]);
            if !(finite(x) && finite(y) && finite(&md.std_x) && finite(&md.std_y)) {
                return (f64::INFINITY, f64::INFINITY);
            }
            let v = x.windows(2).zip(y.windows(2)).map(|(a, b)| (a[1] - a[0]).hypot(b[1] - b[0])).fold(0.0_f64, f64::max);
            (v / h, slope(&md.std_x).max(slope(&md.std_y)) / h)
        })
        .collect()
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
    let mut d2 = f64::INFINITY;
    let mut inside = false;
    for i in 0..n {
        let (ax, ay) = poly[i];
        let (bx, by) = poly[(i + 1) % n];
        let (ex, ey) = (bx - ax, by - ay);
        let ll = ex * ex + ey * ey;
        let t = if ll > 0.0 { (((x - ax) * ex + (y - ay) * ey) / ll).clamp(0.0, 1.0) } else { 0.0 };
        let (qx, qy) = (ax + t * ex - x, ay + t * ey - y);
        d2 = d2.min(qx * qx + qy * qy);
        if (ay > y) != (by > y) && x < ax + (y - ay) * ex / ey {
            inside = !inside;
        }
    }
    let d = d2.sqrt();
    if inside { -d } else { d }
}

/// Footprint-aware clearance against predicted obstacles and the occupancy
/// grid of a `WorldConstraints`. Built once per tick.
#[derive(Clone)]
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
    /// Centre and grown radius of every obstacle mode at each controller
    /// step, filled in as the steps are first asked for.
    table: Vec<std::sync::OnceLock<StepModes>>,
    times: TimeCache,
    split: std::sync::OnceLock<Split>,
}

/// The modes of the obstacles predicted to stay put, sorted into cells, and
/// the indices of the obstacles predicted to move.
#[derive(Clone, Debug)]
struct Split {
    still: StepModes,
    moving: Vec<usize>,
}

/// Whether every mode of `obs` is predicted at the same place and spread
/// over its whole horizon, so where it is does not depend on the time.
fn stays_put(obs: &Obstacle) -> bool {
    let flat = |v: &[f64]| v.iter().all(|a| a.is_finite()) && v.windows(2).all(|w| w[0] == w[1]);
    obs.modes.iter().all(|m| flat(&m.mean_x) && flat(&m.mean_y) && flat(&m.std_x) && flat(&m.std_y))
}

/// The obstacle modes at one controller step: centre and grown radius,
/// sorted into cells, and the widest grown radius.
#[derive(Clone, Debug)]
struct StepModes {
    modes: Vec<(f64, f64, f64)>,
    cells: Buckets,
    widest: f64,
}

/// Controller steps a checker keeps the obstacles' places for.
const TABLE_STEPS: usize = 256;

fn fresh_table() -> Vec<std::sync::OnceLock<StepModes>> {
    (0..TABLE_STEPS).map(|_| std::sync::OnceLock::new()).collect()
}

/// Slots `clearance_at_within` keeps the obstacle modes of recent times in.
const TIME_SLOTS: usize = 512;

/// Modes few enough to look at every one rather than through their cells.
const FEW_MODES: usize = 8;

/// The obstacle modes at times asked for by `clearance_at_within`, each in
/// a slot picked by its time; a clone starts empty.
#[derive(Default)]
struct TimeCache {
    slots: std::sync::Mutex<Vec<Option<(u64, std::sync::Arc<StepModes>)>>>,
}

impl Clone for TimeCache {
    fn clone(&self) -> Self {
        Self::default()
    }
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
        Self { world, polygon, samples, disc_radius: robot_radius(constraints), step: 0.1, spread: 0.0, margin: margin.max(0.0), table: fresh_table(), times: TimeCache::default(), split: std::sync::OnceLock::new() }
    }

    /// Seconds per controller step, so `clearance(step, ..)` can read
    /// predictions timed with `Obstacle::dt`.
    pub fn with_step(mut self, seconds: f64) -> Self {
        if seconds > 0.0 {
            self.step = seconds;
            self.table = fresh_table();
            self.times = TimeCache::default();
            self.split = std::sync::OnceLock::new();
        }
        self
    }

    /// Grow every predicted obstacle by `k` standard deviations of its
    /// spread at the moment checked.
    pub fn with_spread(mut self, k: f64) -> Self {
        self.spread = k.max(0.0);
        self.table = fresh_table();
        self.times = TimeCache::default();
        self.split = std::sync::OnceLock::new();
        self
    }

    pub fn has_obstacles(&self) -> bool {
        self.world.is_some_and(|w| !w.obstacles.is_empty() || w.grid.is_some())
    }

    /// Clearance of the footprint at `(x, y, yaw)` at controller step `step`,
    /// with `margin` already subtracted. Infinite without obstacles. Where
    /// the obstacles are at each step is worked out once per checker.
    pub fn clearance(&self, step: usize, x: f64, y: f64, yaw: f64) -> f64 {
        match self.table.get(step) {
            Some(cell) => {
                let at = cell.get_or_init(|| self.step_modes(step as f64 * self.step));
                self.clearance_among(at.modes.iter().copied(), x, y, yaw)
            }
            None => self.clearance_at(step as f64 * self.step, x, y, yaw),
        }
    }

    /// `clearance` where it is below `cap`, infinite where it is not.
    pub fn clearance_within(&self, step: usize, x: f64, y: f64, yaw: f64, cap: f64) -> f64 {
        match self.table.get(step) {
            Some(cell) => self.capped(cell.get_or_init(|| self.step_modes(step as f64 * self.step)), x, y, yaw, cap),
            None => self.clearance_at_within(step as f64 * self.step, x, y, yaw, cap),
        }
    }

    /// Clearance of the footprint at `(x, y, yaw)` against the obstacles as
    /// predicted `t` seconds from now, with `margin` already subtracted.
    pub fn clearance_at(&self, t: f64, x: f64, y: f64, yaw: f64) -> f64 {
        self.clearance_among(self.grown_modes(t), x, y, yaw)
    }

    /// `clearance_at` where it is below `cap`, infinite where it is not:
    /// the nearer of the obstacles that stay put, sorted into cells once,
    /// and those that move, whose modes at the last `TIME_SLOTS` or so times
    /// asked for are kept.
    pub fn clearance_at_within(&self, t: f64, x: f64, y: f64, yaw: f64, cap: f64) -> f64 {
        let split = self.split.get_or_init(|| self.split_modes());
        let still = self.capped(&split.still, x, y, yaw, cap);
        if split.moving.is_empty() {
            return still;
        }
        let slot = (t.to_bits().wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 55) as usize % TIME_SLOTS;
        let kept = self.times.slots.lock().ok().and_then(|s| s.get(slot).and_then(|k| k.clone()).filter(|k| k.0 == t.to_bits()));
        let at = match kept {
            Some((_, at)) => at,
            None => {
                let at = std::sync::Arc::new(self.modes_among(&split.moving, t));
                if let Ok(mut slots) = self.times.slots.lock() {
                    slots.resize(TIME_SLOTS, None);
                    slots[slot] = Some((t.to_bits(), at.clone()));
                }
                at
            }
        };
        still.min(self.capped(&at, x, y, yaw, cap))
    }

    /// `clearance_among` the modes `at`, where below `cap`, looking only at
    /// the modes that can come that near; infinite where it is not below.
    fn capped(&self, at: &StepModes, x: f64, y: f64, yaw: f64, cap: f64) -> f64 {
        let Some(world) = self.world else {
            return f64::INFINITY;
        };
        let (s, c) = yaw.sin_cos();
        let room = cap + self.margin;
        let seed = room + 1e-9 * (1.0 + room.abs());
        let mut best = seed;
        if at.modes.len() <= FEW_MODES {
            best = at.modes.iter().fold(best, |best, &m| self.nearer(m, x, y, s, c, best));
        } else {
            at.cells.within(x, y, seed + self.disc_radius + at.widest, |k| best = self.nearer(at.modes[k], x, y, s, c, best));
        }
        let best = self.grid_nearest(world, best, x, y, s, c);
        if best < seed { best - self.margin } else { f64::INFINITY }
    }

    /// Centre and grown radius of every obstacle mode `t` seconds from now.
    fn grown_modes(&self, t: f64) -> impl Iterator<Item = (f64, f64, f64)> + '_ {
        self.world.into_iter().flat_map(|w| w.obstacles.iter()).flat_map(move |obs| self.grown(obs, t))
    }

    /// `clearance_at` against the obstacle modes `modes` (centre and grown
    /// radius) and the occupancy grid.
    fn clearance_among(&self, modes: impl Iterator<Item = (f64, f64, f64)>, x: f64, y: f64, yaw: f64) -> f64 {
        let Some(world) = self.world else {
            return f64::INFINITY;
        };
        let (s, c) = yaw.sin_cos();
        let best = modes.fold(f64::INFINITY, |best, m| self.nearer(m, x, y, s, c, best));
        self.grid_nearest(world, best, x, y, s, c) - self.margin
    }

    /// The smaller of `best` and the clearance of the footprint at `(x, y)`
    /// (heading sine `s`, cosine `c`) to the mode `(mx, my, grow)`.
    fn nearer(&self, (mx, my, grow): (f64, f64, f64), x: f64, y: f64, s: f64, c: f64, best: f64) -> f64 {
        let (dx, dy) = (mx - x, my - y);
        let centre = (dx * dx + dy * dy).sqrt();
        // The footprint lies within the bounding disc, so a mode past it by
        // more than the nearest so far cannot be nearer.
        if centre - self.disc_radius - grow >= best {
            return best;
        }
        let d = if self.polygon.is_empty() {
            centre - self.disc_radius
        } else {
            polygon_distance(&self.polygon, c * dx + s * dy, -s * dx + c * dy)
        };
        best.min(d - grow)
    }

    /// `best` against the occupancy grid as well.
    fn grid_nearest(&self, world: &WorldConstraints, mut best: f64, x: f64, y: f64, s: f64, c: f64) -> f64 {
        if let Some(grid) = &world.grid {
            if self.samples.is_empty() {
                best = best.min(grid.distance_to_occupied(x, y) - self.disc_radius);
            } else {
                for (px, py) in &self.samples {
                    best = best.min(grid.distance_to_occupied(x + px * c - py * s, y + px * s + py * c));
                }
                if grid.is_occupied(x, y) {
                    best = best.min(-0.5 * grid.resolution);
                }
            }
        }
        best
    }

    /// The obstacle modes `t` seconds from now, sorted into cells.
    fn step_modes(&self, t: f64) -> StepModes {
        self.sorted(self.grown_modes(t).collect())
    }

    /// The modes of the obstacles at `indices`, `t` seconds from now, sorted
    /// into cells.
    fn modes_among(&self, indices: &[usize], t: f64) -> StepModes {
        let obstacles = self.world.map_or(&[][..], |w| &w.obstacles);
        self.sorted(indices.iter().flat_map(|&k| self.grown(&obstacles[k], t)).collect())
    }

    /// The obstacles that stay put, with their modes, and those that move.
    fn split_modes(&self) -> Split {
        let obstacles = self.world.map_or(&[][..], |w| &w.obstacles);
        let (still, moving): (Vec<usize>, Vec<usize>) = (0..obstacles.len()).partition(|&k| stays_put(&obstacles[k]));
        Split { still: self.modes_among(&still, 0.0), moving }
    }

    /// Centre and grown radius of each mode of `obs` `t` seconds from now.
    fn grown<'b>(&'b self, obs: &'b Obstacle, t: f64) -> impl Iterator<Item = (f64, f64, f64)> + 'b {
        modes_at(obs, t, self.step).map(move |m| (m.x, m.y, m.radius + self.spread * m.std_x.max(m.std_y)))
    }

    /// `modes`, centre and grown radius, sorted into cells.
    fn sorted(&self, modes: Vec<(f64, f64, f64)>) -> StepModes {
        let widest = modes.iter().map(|m| m.2).fold(0.0_f64, f64::max);
        let points: Vec<(f64, f64)> = modes.iter().map(|m| (m.0, m.1)).collect();
        let cells = Buckets::new(&points, (self.disc_radius + widest).max(1.0));
        StepModes { modes, cells, widest }
    }

    /// Distance from the footprint at `(x, y, yaw)` to the point `(px, py)`,
    /// negative inside it.
    pub fn body_distance(&self, px: f64, py: f64, x: f64, y: f64, yaw: f64) -> f64 {
        let (dx, dy) = (px - x, py - y);
        if self.polygon.is_empty() {
            return dx.hypot(dy) - self.disc_radius;
        }
        let (s, c) = yaw.sin_cos();
        polygon_distance(&self.polygon, c * dx + s * dy, -s * dx + c * dy)
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
    for obs in obstacles {
        for m in modes_at(obs, t, step) {
            for &(off, r) in discs {
                let (px, py) = (x + off * c, y + off * s);
                let (dx, dy) = (px - m.x, py - m.y);
                let dist = (dx * dx + dy * dy).sqrt().max(1e-6);
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

/// `obstacle_barrier` against fixed obstacles, discs, spread, weight and
/// scale at `t = j · step` for step `j`, each step's obstacle modes worked
/// out once and sorted into cells so a pose looks only at those in reach.
pub struct BarrierField<'a> {
    obstacles: &'a [Obstacle],
    discs: Vec<(f64, f64)>,
    step: f64,
    spread: f64,
    weight: f64,
    scale: f64,
    table: Vec<std::sync::OnceLock<BarrierStep>>,
}

/// The obstacle modes of one step: centre, radius and spread, sorted into
/// cells, and the distance from the pose past which none adds cost.
struct BarrierStep {
    modes: Vec<(f64, f64, f64, f64)>,
    cells: Buckets,
    reach: f64,
}

impl<'a> BarrierField<'a> {
    pub fn new(obstacles: &'a [Obstacle], discs: &[(f64, f64)], step: f64, spread: f64, weight: f64, scale: f64) -> Self {
        let steps = if obstacles.is_empty() { 0 } else { TABLE_STEPS };
        let table = (0..steps).map(|_| std::sync::OnceLock::new()).collect();
        Self { obstacles, discs: discs.to_vec(), step, spread, weight, scale, table }
    }

    pub fn is_empty(&self) -> bool {
        self.obstacles.is_empty()
    }

    fn step_at(&self, t: f64) -> BarrierStep {
        let modes: Vec<(f64, f64, f64, f64)> =
            self.obstacles.iter().flat_map(|o| modes_at(o, t, self.step)).map(|m| (m.x, m.y, m.radius, m.std_x.max(m.std_y))).collect();
        let body = self.discs.iter().map(|&(off, r)| off.abs() + r).fold(0.0_f64, f64::max);
        let widest = modes.iter().map(|m| m.2 + self.spread * m.3).fold(0.0_f64, f64::max);
        let reach = body + widest + 8.0 * self.scale + 1e-6;
        let points: Vec<(f64, f64)> = modes.iter().map(|m| (m.0, m.1)).collect();
        BarrierStep { cells: Buckets::new(&points, (0.5 * reach).max(0.5)), modes, reach }
    }

    /// The barrier at the pose `(x, y, yaw)` at step `j`.
    pub fn at(&self, j: usize, (x, y, yaw): (f64, f64, f64)) -> Barrier {
        let t = j as f64 * self.step;
        let Some(cell) = self.table.get(j) else {
            return obstacle_barrier(self.obstacles, &self.discs, t, self.step, self.spread, (x, y, yaw), self.weight, self.scale);
        };
        let mut b = Barrier::default();
        if self.weight <= 0.0 || self.scale <= 0.0 {
            return b;
        }
        let at = cell.get_or_init(|| self.step_at(t));
        let (s, c) = yaw.sin_cos();
        let (weight, scale) = (self.weight, self.scale);
        at.cells.within_sorted(x, y, at.reach, |k| k as u32, |found| {
            for &k in found {
                let (mx, my, radius, std) = at.modes[k as usize];
                for &(off, r) in &self.discs {
                    let (px, py) = (x + off * c, y + off * s);
                    let (dx, dy) = (px - mx, py - my);
                    let dist = (dx * dx + dy * dy).sqrt().max(1e-6);
                    let d = dist - radius - r - self.spread * std;
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
        });
        b
    }
}

pub fn footprint_discs(constraints: &RobotConstraints) -> Vec<(f64, f64)> {
    let Some((back, front, half_width)) = footprint_extent(constraints) else {
        return vec![(0.0, robot_radius(constraints))];
    };
    let length = (front - back).max(1e-3);
    let width = (2.0 * half_width).max(1e-3);
    let n = (length / width).ceil().max(1.0) as usize;
    let half_step = 0.5 * length / n as f64;
    let radius = half_width.hypot(half_step);
    (0..n).map(|i| (back + half_step * (2 * i + 1) as f64, radius)).collect()
}

/// The footprint's extent along the heading, `(back, front)` metres from
/// the pose, and its half width; `None` for a disc footprint or none set.
pub fn footprint_extent(constraints: &RobotConstraints) -> Option<(f64, f64, f64)> {
    match &constraints.footprint {
        crate::types::Footprint::Polygon { points } if points.len() >= 3 => {
            let front = points.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
            let back = points.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
            let half = points.iter().map(|p| p.1.abs()).fold(0.0_f64, f64::max);
            Some((back, front, half))
        }
        crate::types::Footprint::Disc { radius } if *radius > 0.0 => None,
        _ if constraints.robot_length > 0.0 && constraints.robot_width > 0.0 => {
            Some((-0.5 * constraints.robot_length, 0.5 * constraints.robot_length, 0.5 * constraints.robot_width))
        }
        _ => None,
    }
}

/// The fastest of `shares` of `speed` that keeps `checker`'s obstacles, as
/// they move, clear of the robot driving `course` (its pose after a signed
/// distance), setting off at `v_now` and changing speed at `decel`, over
/// `lookout` seconds plus the time to stop plus `STANDING_TAIL_S`: either
/// held throughout, passing ahead of them, or held for `lookout` and then
/// braked to a stand, letting them pass. When none does: standing still if
/// braking now keeps clear of their predicted tracks, else the share
/// (standing still included) keeping the most room.
pub fn yield_share(
    checker: &CollisionChecker,
    course: &dyn Fn(f64) -> (f64, f64, f64),
    speed: f64,
    v_now: f64,
    lookout: f64,
    decel: f64,
    shares: &[f64],
) -> f64 {
    if !checker.has_obstacles() {
        return 1.0;
    }
    let decel = decel.max(1e-3);
    let nominal = checker.clone().with_spread(0.0);
    let arc_on = |checker: &CollisionChecker, s: f64, brake: bool| {
        let u = speed * s;
        let horizon = lookout + u.abs().max(v_now.abs()) / decel + STANDING_TAIL_S;
        let step = 0.05;
        let (mut along, mut t, mut v) = (0.0, 0.0, v_now);
        let mut worst = f64::INFINITY;
        while t <= horizon {
            let (px, py, th) = course(along);
            worst = worst.min(checker.clearance_at(t, px, py, th));
            let aim = if brake && t >= lookout { 0.0 } else { u };
            v += (aim - v).clamp(-decel * step, decel * step);
            along += v * step;
            t += step;
        }
        worst
    };
    let room = |s: f64| arc_on(checker, s, false).max(arc_on(checker, s, true));
    shares.iter().copied().find(|s| room(*s) >= 0.0).unwrap_or_else(|| {
        if arc_on(&nominal, 0.0, true) >= 0.0 {
            return 0.0;
        }
        let (x0, y0, th0) = course(0.0);
        let contact = nominal.clearance_at(0.0, x0, y0, th0);
        if contact < 0.0 {
            return shares.iter().copied().find(|s| arc_on(&nominal, *s, false) >= contact - 1e-6).unwrap_or(0.0);
        }
        shares.iter().copied().chain([0.0]).max_by(|a, b| room(*a).total_cmp(&room(*b))).unwrap_or(0.0)
    })
}

/// Seconds the give-way check stands the robot after braking, watching for
/// traffic still crossing where it stopped.
const STANDING_TAIL_S: f64 = 4.0;

/// The course of an arc of curvature `kappa` from `(x, y, yaw)`: the pose
/// after a signed distance along it.
pub fn arc_course((x, y, yaw): (f64, f64, f64), kappa: f64) -> impl Fn(f64) -> (f64, f64, f64) {
    move |d| {
        if kappa.abs() < 1e-9 {
            (x + d * yaw.cos(), y + d * yaw.sin(), yaw)
        } else {
            let th = yaw + kappa * d;
            (x + (th.sin() - yaw.sin()) / kappa, y - (th.cos() - yaw.cos()) / kappa, th)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Footprint, GaussianMode};

    fn crossing(x: f64, y: f64, vy: f64) -> WorldConstraints {
        let steps = 40;
        let mode = GaussianMode {
            weight: 1.0,
            mean_x: vec![x; steps + 1],
            mean_y: (0..=steps).map(|k| y + vy * k as f64 * 0.2).collect(),
            std_x: vec![0.0; steps + 1],
            std_y: vec![0.0; steps + 1],
        };
        WorldConstraints { obstacles: vec![Obstacle { id: 0, radius: 0.6, modes: vec![mode], dt: 0.2 }], ..Default::default() }
    }

    fn constraints() -> RobotConstraints {
        let mut c = RobotConstraints::default();
        c.max_linear_velocity = 1.5;
        c.max_linear_acceleration = 0.5;
        c.footprint = Footprint::Disc { radius: 0.8 };
        c
    }

    fn share_with(world: &WorldConstraints, v_now: f64) -> f64 {
        let c = constraints();
        let checker = CollisionChecker::new(Some(world), &c, 0.1).with_spread(1.0);
        yield_share(&checker, &arc_course((0.0, 0.0, 0.0), 0.0), 1.2, v_now, 1.5, 0.5, &[1.0, 0.6, 0.3])
    }

    /// Many walkers with two modes each, spread over 40 × 30 m.
    fn crowd() -> Vec<Obstacle> {
        let mut s = 0x2545_f491_4f6c_dd1d_u64;
        let mut draw = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        (0..120)
            .map(|id| {
                let (x, y, vx, vy) = (40.0 * draw(), 30.0 * draw() - 15.0, draw() - 0.5, draw() - 0.5);
                let mode = |w: f64, turn: f64| GaussianMode {
                    weight: w,
                    mean_x: (0..=30).map(|k| x + vx * k as f64 * 0.2).collect(),
                    mean_y: (0..=30).map(|k| y + (vy + turn) * k as f64 * 0.2).collect(),
                    std_x: (0..=30).map(|k| 0.05 + 0.02 * k as f64).collect(),
                    std_y: (0..=30).map(|k| 0.05 + 0.03 * k as f64).collect(),
                };
                Obstacle { id, radius: 0.3 + 0.4 * draw(), modes: vec![mode(0.7, 0.0), mode(0.3, 0.4)], dt: 0.2 }
            })
            .collect()
    }

    #[test]
    fn capped_clearance_matches_every_obstacle_looked_at() {
        // The crowd moving, and with every other obstacle standing still.
        let mut mixed = crowd();
        for o in mixed.iter_mut().step_by(2) {
            for m in o.modes.iter_mut() {
                for v in [&mut m.mean_x, &mut m.mean_y, &mut m.std_x, &mut m.std_y] {
                    let first = v[0];
                    v.fill(first);
                }
            }
        }
        for obstacles in [crowd(), mixed] {
            let world = WorldConstraints { obstacles, ..Default::default() };
            let mut c = constraints();
            c.footprint = Footprint::Polygon { points: vec![(2.5, 1.1), (-2.5, 1.1), (-2.5, -1.1), (2.5, -1.1)] };
            let checker = CollisionChecker::new(Some(&world), &c, 0.2).with_step(0.15).with_spread(1.0);
            for cap in [0.0, 0.8, 3.0] {
                for step in [0, 3, 17, 40, 300] {
                    for k in 0..200 {
                        let (x, y, yaw) = (-5.0 + 0.27 * k as f64, 8.0 * (0.1 * k as f64).sin(), 0.05 * k as f64);
                        let t = step as f64 * 0.15;
                        let all = checker.clearance_at(t, x, y, yaw);
                        let want = if all < cap { all } else { f64::INFINITY };
                        assert_eq!(checker.clearance(step, x, y, yaw), all, "step {step} pose {k}");
                        assert_eq!(checker.clearance_within(step, x, y, yaw, cap), want, "step {step} pose {k} cap {cap}");
                        assert_eq!(checker.clearance_at_within(t, x, y, yaw, cap), want, "t {t} pose {k} cap {cap}");
                    }
                }
            }
        }
    }

    #[test]
    fn modes_move_and_spread_no_faster_than_their_rates() {
        for o in crowd().iter().take(20) {
            let rates = mode_rates(o, 0.15);
            for k in 0..120 {
                let (t0, t1) = (0.13 * k as f64, 0.13 * k as f64 + 0.37 * (k % 7) as f64);
                for ((a, b), (v, s)) in modes_at(o, t0, 0.15).zip(modes_at(o, t1, 0.15)).zip(&rates) {
                    let slack = 1e-9 + (t1 - t0) * 1e-9;
                    assert!((b.x - a.x).hypot(b.y - a.y) <= v * (t1 - t0) + slack, "obstacle {} from {t0} to {t1}", o.id);
                    assert!((b.std_x.max(b.std_y) - a.std_x.max(a.std_y)).abs() <= s * (t1 - t0) + slack);
                }
            }
        }
    }

    #[test]
    fn bucketed_barrier_matches_every_obstacle_looked_at() {
        let obstacles = crowd();
        let discs = [(-1.7, 1.43), (0.0, 1.43), (1.7, 1.43)];
        let field = BarrierField::new(&obstacles, &discs, 0.2, 1.0, 50.0, 0.2);
        for j in [1, 5, 22, 29] {
            for k in 0..200 {
                let pose = (-5.0 + 0.27 * k as f64, 8.0 * (0.1 * k as f64).sin(), 0.05 * k as f64);
                let a = field.at(j, pose);
                let b = obstacle_barrier(&obstacles, &discs, j as f64 * 0.2, 0.2, 1.0, pose, 50.0, 0.2);
                assert_eq!((a.cost, a.grad, a.hess), (b.cost, b.grad, b.hess), "step {j} pose {k}");
            }
        }
    }

    #[test]
    fn keeps_going_past_traffic_that_clears_first() {
        assert_eq!(share_with(&crossing(8.0, -4.0, 1.0), 1.0), 1.0);
    }

    #[test]
    fn slows_for_traffic_meeting_it_at_the_crossing() {
        assert!(share_with(&crossing(3.0, -2.5, 1.0), 1.0) < 1.0);
    }

    #[test]
    fn stands_while_in_contact_ahead() {
        assert_eq!(share_with(&crossing(1.0, -0.5, 0.4), 0.3), 0.0);
    }

    #[test]
    fn drives_clear_of_contact_behind() {
        assert_eq!(share_with(&crossing(-0.9, -0.9, 0.0), 0.3), 1.0);
    }
}
