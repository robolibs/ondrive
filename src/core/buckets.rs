//! Points sorted into the cells of a square grid, to find the ones near a
//! point without looking at every one.

/// Most cells along either side of the grid.
const MAX_SIDE: usize = 128;

thread_local! {
    /// Scratch list for `within_sorted`, kept between calls.
    static NEAR: std::cell::Cell<Vec<u32>> = const { std::cell::Cell::new(Vec::new()) };
}

#[derive(Clone, Debug, Default)]
pub struct Buckets {
    cell: f64,
    x0: f64,
    y0: f64,
    nx: usize,
    ny: usize,
    /// Where each cell's points start in `items`, the last entry their end.
    start: Vec<u32>,
    items: Vec<u32>,
}

impl Buckets {
    /// `points` sorted into square cells of at least `cell` metres, wider
    /// where the points spread over more than `MAX_SIDE` of them.
    pub fn new(points: &[(f64, f64)], cell: f64) -> Self {
        let finite = |p: &&(f64, f64)| p.0.is_finite() && p.1.is_finite();
        if !points.iter().any(|p| finite(&p)) {
            return Self::default();
        }
        let (mut x0, mut y0, mut x1, mut y1) = (f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
        for &(x, y) in points.iter().filter(finite) {
            (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x), y1.max(y));
        }
        let cell = cell.max((x1 - x0).max(y1 - y0) / MAX_SIDE as f64).max(1e-6);
        let nx = ((x1 - x0) / cell) as usize + 1;
        let ny = ((y1 - y0) / cell) as usize + 1;
        let mut b = Buckets { cell, x0, y0, nx, ny, start: vec![0; nx * ny + 1], items: Vec::new() };
        let slots: Vec<Option<usize>> = points.iter().map(|&(x, y)| b.slot(x, y)).collect();
        for s in slots.iter().flatten() {
            b.start[s + 1] += 1;
        }
        for c in 0..nx * ny {
            b.start[c + 1] += b.start[c];
        }
        let mut next = b.start.clone();
        b.items = vec![0; b.start[nx * ny] as usize];
        for (k, s) in slots.iter().enumerate() {
            if let Some(s) = *s {
                b.items[next[s] as usize] = k as u32;
                next[s] += 1;
            }
        }
        b
    }

    /// Cell column and row of `(x, y)`, which may lie off the grid.
    fn cell_at(&self, x: f64, y: f64) -> (i64, i64) {
        (floor((x - self.x0) / self.cell), floor((y - self.y0) / self.cell))
    }

    /// Cell index of a finite point on the grid.
    fn slot(&self, x: f64, y: f64) -> Option<usize> {
        if !(x.is_finite() && y.is_finite()) {
            return None;
        }
        let (i, j) = self.cell_at(x, y);
        Some((i.clamp(0, self.nx as i64 - 1) + j.clamp(0, self.ny as i64 - 1) * self.nx as i64) as usize)
    }

    fn visit_cell(&self, i: i64, j: i64, visit: &mut impl FnMut(usize)) {
        if i < 0 || j < 0 || i >= self.nx as i64 || j >= self.ny as i64 {
            return;
        }
        let c = (i + j * self.nx as i64) as usize;
        for &k in &self.items[self.start[c] as usize..self.start[c + 1] as usize] {
            visit(k as usize);
        }
    }

    /// `visit` with every point within `r` of `(x, y)`, and some others.
    pub fn within(&self, x: f64, y: f64, r: f64, mut visit: impl FnMut(usize)) {
        if self.items.is_empty() {
            return;
        }
        let (i0, j0) = self.cell_at(x - r, y - r);
        let (i1, j1) = self.cell_at(x + r, y + r);
        for j in j0.max(0)..=j1.min(self.ny as i64 - 1) {
            for i in i0.max(0)..=i1.min(self.nx as i64 - 1) {
                self.visit_cell(i, j, &mut visit);
            }
        }
    }

    /// `then` with `owner` of every point `within` reach, each owner once
    /// and in increasing order.
    pub fn within_sorted<R>(&self, x: f64, y: f64, r: f64, owner: impl Fn(usize) -> u32, then: impl FnOnce(&[u32]) -> R) -> R {
        let mut found = NEAR.with(|n| n.take());
        found.clear();
        self.within(x, y, r, |k| found.push(owner(k)));
        found.sort_unstable();
        found.dedup();
        let out = then(&found);
        NEAR.with(|n| n.set(found));
        out
    }

}

/// `v.floor() as i64` without the library call.
fn floor(v: f64) -> i64 {
    let i = v as i64;
    if (i as f64) > v { i.saturating_sub(1) } else { i }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scatter(n: usize) -> Vec<(f64, f64)> {
        let mut s = 0x9e37_79b9_7f4a_7c15_u64;
        let mut draw = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        (0..n).map(|_| (60.0 * draw() - 20.0, 40.0 * draw() - 10.0)).collect()
    }

    #[test]
    fn within_finds_every_point_in_reach() {
        let pts = scatter(300);
        let b = Buckets::new(&pts, 2.5);
        for &(qx, qy, r) in &[(0.0, 0.0, 3.0), (35.0, 25.0, 7.5), (-40.0, 5.0, 30.0), (12.3, -4.4, 0.5)] {
            let mut seen = vec![false; pts.len()];
            b.within(qx, qy, r, |k| seen[k] = true);
            for (k, &(x, y)) in pts.iter().enumerate() {
                if (x - qx).hypot(y - qy) <= r {
                    assert!(seen[k], "point {k} at ({x:.1}, {y:.1}) missed from ({qx}, {qy}) within {r}");
                }
            }
        }
    }

    #[test]
    fn within_sorted_lists_each_owner_once_in_order() {
        let pts = scatter(300);
        let b = Buckets::new(&pts, 2.5);
        let owner = |k: usize| (k / 3) as u32;
        let found = b.within_sorted(5.0, 5.0, 6.0, owner, |f| f.to_vec());
        assert!(found.windows(2).all(|w| w[0] < w[1]));
        for (k, &(x, y)) in pts.iter().enumerate() {
            if (x - 5.0).hypot(y - 5.0) <= 6.0 {
                assert!(found.contains(&owner(k)));
            }
        }
    }

    #[test]
    fn floor_matches_the_library_floor() {
        for v in [0.0, -0.0, 0.5, -0.5, 1.0, -1.0, 2.999, -2.999, 1e300, -1e300, f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1e-300] {
            assert_eq!(floor(v), v.floor() as i64, "{v}");
        }
    }
}
