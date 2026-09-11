//! Path geometry helpers shared by every path-following controller:
//! segment projection, arc-length sampling, curvature and end-of-path
//! detection. All lateral errors are signed with the convention
//! *positive = robot is to the left of the path direction*.

use crate::core::math::normalize_angle;
use datapod::{Point, Pose};

#[derive(Clone, Copy, Debug, Default)]
pub struct PathProjection {
    pub segment: usize,
    pub t: f64,
    pub point: Point,
    pub heading: f64,
    pub lateral_error: f64,
    pub distance: f64,
    pub arc_length: f64,
    pub beyond_end: bool,
}

pub fn cumulative_lengths(waypoints: &[Pose]) -> Vec<f64> {
    let mut out = Vec::with_capacity(waypoints.len());
    let mut acc = 0.0;
    out.push(0.0);
    for w in waypoints.windows(2) {
        acc += w[0].point.distance_to_2d(w[1].point);
        out.push(acc);
    }
    out
}

pub fn segment_heading(waypoints: &[Pose], segment: usize) -> f64 {
    let n = waypoints.len();
    if n < 2 {
        return waypoints
            .first()
            .map(|p| p.rotation.to_euler().yaw)
            .unwrap_or(0.0);
    }
    let seg = segment.min(n - 2);
    let mut i = seg;
    loop {
        let a = waypoints[i].point;
        let b = waypoints[i + 1].point;
        let dx = b.x - a.x;
        let dy = b.y - a.y;
        if dx.hypot(dy) > 1e-9 {
            return dy.atan2(dx);
        }
        if i + 2 < n {
            i += 1;
        } else {
            break;
        }
    }
    let mut i = seg;
    while i > 0 {
        let a = waypoints[i - 1].point;
        let b = waypoints[i].point;
        let dx = b.x - a.x;
        let dy = b.y - a.y;
        if dx.hypot(dy) > 1e-9 {
            return dy.atan2(dx);
        }
        i -= 1;
    }
    waypoints[seg].rotation.to_euler().yaw
}

pub fn end_heading(waypoints: &[Pose]) -> f64 {
    match waypoints.len() {
        0 => 0.0,
        1 => waypoints[0].rotation.to_euler().yaw,
        n => segment_heading(waypoints, n - 2),
    }
}

/// Project `p` onto the polyline, searching segments in
/// `[start.saturating_sub(lookback), start + window)`. Returns `None` for an
/// empty path. A single-waypoint path projects onto that point.
pub fn project(
    waypoints: &[Pose],
    cum: &[f64],
    p: Point,
    start: usize,
    lookback: usize,
    window: usize,
) -> Option<PathProjection> {
    let n = waypoints.len();
    if n == 0 {
        return None;
    }
    if n == 1 {
        let q = waypoints[0].point;
        let heading = waypoints[0].rotation.to_euler().yaw;
        let (lat, along) = frame_offsets(q, heading, p);
        return Some(PathProjection {
            segment: 0,
            t: 0.0,
            point: q,
            heading,
            lateral_error: lat,
            distance: p.distance_to_2d(q),
            arc_length: 0.0,
            beyond_end: along > 0.0,
        });
    }

    let seg_count = n - 1;
    let end = waypoints[n - 1].point;
    let first = start.min(seg_count - 1).saturating_sub(lookback);
    let last = start.saturating_add(window).min(seg_count);

    let mut best: Option<PathProjection> = None;
    for s in first..last {
        let a = waypoints[s].point;
        let b = waypoints[s + 1].point;
        let dx = b.x - a.x;
        let dy = b.y - a.y;
        let len2 = dx * dx + dy * dy;
        let t = if len2 < 1e-18 {
            1.0
        } else {
            (((p.x - a.x) * dx + (p.y - a.y) * dy) / len2).clamp(0.0, 1.0)
        };
        let q = Point::new(a.x + t * dx, a.y + t * dy, 0.0);
        let d = p.distance_to_2d(q);
        let better = match &best {
            None => true,
            Some(b) => d <= b.distance + 1e-9,
        };
        if better {
            let heading = segment_heading(waypoints, s);
            let (lat, along) = frame_offsets(q, heading, p);
            let seg_len = cum[s + 1] - cum[s];
            best = Some(PathProjection {
                segment: s,
                t,
                point: q,
                heading,
                lateral_error: lat,
                distance: d,
                arc_length: cum[s] + t * seg_len,
                beyond_end: (s + 1 == seg_count || q.distance_to_2d(end) < 1e-9)
                    && t >= 1.0 - 1e-9
                    && along > 0.0,
            });
        }
    }
    best
}

/// Signed (lateral, along-track) offsets of `p` relative to the frame at
/// `origin` with direction `heading`. Lateral is positive to the left.
pub fn frame_offsets(origin: Point, heading: f64, p: Point) -> (f64, f64) {
    let dx = p.x - origin.x;
    let dy = p.y - origin.y;
    let (s, c) = heading.sin_cos();
    let along = dx * c + dy * s;
    let lat = -dx * s + dy * c;
    (lat, along)
}

/// Pose (point, heading) at arc length `s`, clamped to the path extent.
pub fn sample(waypoints: &[Pose], cum: &[f64], s: f64) -> (Point, f64) {
    let n = waypoints.len();
    if n == 0 {
        return (Point::new(0.0, 0.0, 0.0), 0.0);
    }
    if n == 1 || s <= 0.0 {
        return (waypoints[0].point, segment_heading(waypoints, 0));
    }
    let total = cum[n - 1];
    if s >= total {
        return (waypoints[n - 1].point, end_heading(waypoints));
    }
    let seg = match cum.binary_search_by(|c| c.partial_cmp(&s).unwrap()) {
        Ok(i) => i.min(n - 2),
        Err(i) => i.saturating_sub(1).min(n - 2),
    };
    let seg_len = cum[seg + 1] - cum[seg];
    let t = if seg_len > 1e-12 {
        ((s - cum[seg]) / seg_len).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let a = waypoints[seg].point;
    let b = waypoints[seg + 1].point;
    (
        Point::new(a.x + t * (b.x - a.x), a.y + t * (b.y - a.y), 0.0),
        segment_heading(waypoints, seg),
    )
}

/// Signed curvature of the polyline around waypoint `idx` from the heading
/// change between its adjacent segments (positive = turning left).
pub fn curvature_at(waypoints: &[Pose], cum: &[f64], idx: usize) -> f64 {
    let n = waypoints.len();
    if n < 3 || idx == 0 || idx + 1 >= n {
        return 0.0;
    }
    let h1 = segment_heading(waypoints, idx - 1);
    let h2 = segment_heading(waypoints, idx);
    let ds = 0.5 * (cum[idx + 1] - cum[idx - 1]);
    if ds < 1e-9 {
        return 0.0;
    }
    normalize_angle(h2 - h1) / ds
}

/// Curvature at the projection: interpolates between the curvature at the
/// segment's two endpoints.
pub fn curvature_at_projection(waypoints: &[Pose], cum: &[f64], proj: &PathProjection) -> f64 {
    let k0 = curvature_at(waypoints, cum, proj.segment);
    let k1 = curvature_at(waypoints, cum, proj.segment + 1);
    k0 + (k1 - k0) * proj.t
}

/// Speed cap from `Path::speeds` at the projection, if the path carries
/// per-waypoint speeds.
pub fn speed_cap(speeds: &[f64], proj: &PathProjection) -> Option<f64> {
    if speeds.is_empty() {
        return None;
    }
    let i = if proj.t < 0.5 {
        proj.segment
    } else {
        proj.segment + 1
    };
    speeds.get(i).copied().filter(|v| *v > 0.0)
}

/// Per-controller projection state: cumulative lengths plus a flag that
/// widens the first search to the whole path.
#[derive(Clone, Debug, Default)]
pub struct PathCursor {
    pub cum: Vec<f64>,
    pub started: bool,
}

const CURSOR_WINDOW: usize = 64;

impl PathCursor {

    pub fn set_path(&mut self, waypoints: &[Pose]) {
        self.cum = cumulative_lengths(waypoints);
        self.started = false;
    }

    pub fn total_length(&self) -> f64 {
        self.cum.last().copied().unwrap_or(0.0)
    }

    /// Project `p`, searching the whole path on the first call and a window
    /// around `hint` afterwards.
    pub fn project(&mut self, waypoints: &[Pose], p: Point, hint: usize) -> Option<PathProjection> {
        if self.cum.len() != waypoints.len() {
            self.set_path(waypoints);
        }
        let window = if self.started {
            CURSOR_WINDOW
        } else {
            usize::MAX
        };
        let proj = project(waypoints, &self.cum, p, hint, 2, window)?;
        self.started = true;
        Some(proj)
    }
}
