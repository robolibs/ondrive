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

/// Furthest a projection with a finite search window may land past the end
/// of its hint segment, in metres. Progress that falls further behind the
/// robot catches up over the following calls.
const MAX_ADVANCE: f64 = 2.0;

/// Distances closer than this are a tie.
const TIE: f64 = 1e-9;

/// Project `p` onto the polyline, searching segments in
/// `[start.saturating_sub(lookback), start + window)` and, for a finite
/// `window`, starting no more than `MAX_ADVANCE` past the end of segment
/// `start`. Among equally near points the one closest in arc length to the
/// start of segment `start` wins, then the later one. Returns `None` for an
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
    let s_ref = cum[start.min(seg_count - 1)];
    let reach = if window == usize::MAX { f64::INFINITY } else { cum[start.min(seg_count - 1) + 1] + MAX_ADVANCE };

    let mut best: Option<(PathProjection, f64)> = None;
    for s in first..last {
        if cum[s] > reach {
            break;
        }
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
        let arc_length = cum[s] + t * (cum[s + 1] - cum[s]);
        let jump = (arc_length - s_ref).abs();
        let better = match &best {
            None => true,
            Some((b, b_jump)) => d < b.distance - TIE || (d <= b.distance + TIE && jump <= b_jump + TIE),
        };
        if better {
            let heading = segment_heading(waypoints, s);
            let (lat, along) = frame_offsets(q, heading, p);
            let proj = PathProjection {
                segment: s,
                t,
                point: q,
                heading,
                lateral_error: lat,
                distance: d,
                arc_length,
                beyond_end: (s + 1 == seg_count || q.distance_to_2d(end) < 1e-9)
                    && t >= 1.0 - 1e-9
                    && along > 0.0,
            };
            best = Some((proj, jump));
        }
    }
    best.map(|(proj, _)| proj)
}

/// First waypoint after segment `segment` where the path turns back on
/// itself (consecutive segments more than 90° apart), if any.
pub fn next_cusp(waypoints: &[Pose], segment: usize) -> Option<usize> {
    let mut prev: Option<(f64, f64)> = None;
    for s in segment..waypoints.len().saturating_sub(1) {
        let (a, b) = (waypoints[s].point, waypoints[s + 1].point);
        let (dx, dy) = (b.x - a.x, b.y - a.y);
        if dx.hypot(dy) < 1e-9 {
            continue;
        }
        if prev.is_some_and(|(px, py)| px * dx + py * dy < 0.0) {
            return Some(s);
        }
        prev = Some((dx, dy));
    }
    None
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
    let seg = cum.partition_point(|c| *c <= s).saturating_sub(1).min(n - 2);
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

/// `sample` at `s`, carried straight on past `cap` along the heading the
/// path arrives there with: a point a leading axle aims at for the pose
/// behind it to reach `cap`.
pub fn sample_past(waypoints: &[Pose], cum: &[f64], s: f64, cap: f64) -> (Point, f64) {
    if s <= cap {
        return sample(waypoints, cum, s);
    }
    let (p, _) = sample(waypoints, cum, cap);
    let (_, h) = sample(waypoints, cum, (cap - 1e-6).max(0.0));
    let over = s - cap;
    (Point::new(p.x + over * h.cos(), p.y + over * h.sin(), p.z), h)
}

/// Heading change, radians, past which a waypoint counts as a cusp for curvature.
const CUSP_TURN: f64 = 0.75 * std::f64::consts::PI;

/// Signed curvature of the polyline around waypoint `idx` from the heading
/// change between its adjacent segments (positive = turning left). At a
/// cusp, where the path turns back by more than `CUSP_TURN`, the change is
/// taken against the reversed heading: the body keeps its heading through it.
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
    let turn = normalize_angle(h2 - h1);
    let turn = if turn.abs() > CUSP_TURN { normalize_angle(turn - std::f64::consts::PI) } else { turn };
    turn / ds
}

/// Curvature at the projection: interpolates between the curvature at the
/// segment's two endpoints.
pub fn curvature_at_projection(waypoints: &[Pose], cum: &[f64], proj: &PathProjection) -> f64 {
    let k0 = curvature_at(waypoints, cum, proj.segment);
    let k1 = curvature_at(waypoints, cum, proj.segment + 1);
    k0 + (k1 - k0) * proj.t
}

/// Curvature at arc length `s` (clamped to the path), interpolated as
/// `curvature_at_projection` does.
pub fn curvature_at_arc_length(waypoints: &[Pose], cum: &[f64], s: f64) -> f64 {
    let n = cum.len();
    if n < 2 {
        return 0.0;
    }
    let s = s.clamp(0.0, cum[n - 1]);
    let segment = cum.partition_point(|c| *c <= s).saturating_sub(1).min(n - 2);
    let t = ((s - cum[segment]) / (cum[segment + 1] - cum[segment]).max(1e-9)).clamp(0.0, 1.0);
    let (k0, k1) = (curvature_at(waypoints, cum, segment), curvature_at(waypoints, cum, segment + 1));
    k0 + (k1 - k0) * t
}

/// Points sampled by `sharpest_ahead`.
const SHARPEST_SAMPLES: usize = 16;

/// Sharpest curvature, unsigned, of the path over the `reach` metres on
/// from arc length `s` among the bends no sharper than `followable`;
/// sharper ones are corners a robot cuts whatever it aims at.
pub fn sharpest_ahead(waypoints: &[Pose], cum: &[f64], s: f64, reach: f64, followable: f64) -> f64 {
    (0..=SHARPEST_SAMPLES)
        .map(|i| curvature_at_arc_length(waypoints, cum, s + reach * i as f64 / SHARPEST_SAMPLES as f64).abs())
        .filter(|k| *k <= followable)
        .fold(0.0, f64::max)
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
    speeds.get(i).copied().filter(|v| v.abs() > 0.0).map(f64::abs)
}

/// Driving direction (+1 / -1) along the projection's segment: the sign of
/// `Path::speeds` at the waypoint it leads to; `None` without speeds.
pub fn direction_at(speeds: &[f64], proj: &PathProjection) -> Option<i8> {
    if speeds.is_empty() {
        return None;
    }
    let i = (proj.segment + 1).min(speeds.len() - 1);
    speeds.get(i).map(|v| if *v < 0.0 { -1 } else { 1 })
}

/// Arc length of the first cusp after `s`: the last waypoint before the
/// sign of `Path::speeds` flips.
pub fn cusp_after(speeds: &[f64], cum: &[f64], s: f64) -> Option<f64> {
    if speeds.len() != cum.len() || speeds.len() < 2 {
        return None;
    }
    let start = cum.iter().position(|c| *c > s + 1e-9).unwrap_or(cum.len());
    let mut i = start.saturating_sub(1);
    while i + 1 < speeds.len() {
        if (speeds[i] < 0.0) != (speeds[i + 1] < 0.0) && cum[i] > s + 1e-9 {
            return Some(cum[i]);
        }
        i += 1;
    }
    None
}

/// Per-controller projection state: cumulative lengths plus a flag that
/// widens the first search to the whole path.
#[derive(Clone, Debug, Default)]
pub struct PathCursor {
    pub cum: Vec<f64>,
    pub started: bool,
    /// First segment projections may land on: progress committed past a
    /// cusp does not slide back over it.
    pub floor: usize,
}

const CURSOR_WINDOW: usize = 64;

impl PathCursor {
    pub fn set_path(&mut self, waypoints: &[Pose]) {
        self.cum = cumulative_lengths(waypoints);
        self.started = false;
        self.floor = 0;
    }

    /// Commit progress to segment `segment` and beyond.
    pub fn commit(&mut self, segment: usize) {
        self.floor = self.floor.max(segment);
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
        let hint = hint.max(self.floor);
        let lookback = 2.min(hint - self.floor);
        let proj = project(waypoints, &self.cum, p, hint, lookback, window)?;
        self.started = true;
        Some(proj)
    }

    /// `proj` of `p`, carried onto the leg leaving the next cusp once `p`
    /// comes within `reach` of it, with progress committed past the cusp.
    pub fn past_cusp(&mut self, waypoints: &[Pose], speeds: &[f64], p: Point, proj: PathProjection, reach: f64) -> PathProjection {
        let Some(k) = self.cusp_ahead(speeds, proj.arc_length) else {
            return proj;
        };
        if waypoints[k].point.distance_to_2d(p) >= reach {
            return proj;
        }
        self.commit(k);
        self.project(waypoints, p, k).unwrap_or(proj)
    }

    /// Waypoint index of the first cusp after arc length `s`.
    pub fn cusp_ahead(&self, speeds: &[f64], s: f64) -> Option<usize> {
        let c = cusp_after(speeds, &self.cum, s)?;
        Some(self.cum.partition_point(|x| *x < c - 1e-9).min(self.cum.len().saturating_sub(1)))
    }

    /// Projection of `p`, a point of the robot near its pose projected at
    /// `around` (arc length `s`), onto the leg holding it: between the
    /// progress committed past a cusp and the next cusp, searched both ways.
    pub fn project_on_leg(&self, waypoints: &[Pose], speeds: &[f64], p: Point, around: usize, s: f64) -> Option<PathProjection> {
        let leg_end = self.cusp_ahead(speeds, s).unwrap_or(waypoints.len().saturating_sub(1));
        let lookback = around.saturating_sub(self.floor).min(CURSOR_WINDOW);
        project(waypoints, &self.cum, p, around, lookback, leg_end.saturating_sub(around).max(1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datapod::{Euler, Quaternion};

    fn at(x: f64, y: f64) -> Pose {
        Pose { point: Point::new(x, y, 0.0), rotation: Quaternion::from_euler(Euler::new(0.0, 0.0, 0.0)) }
    }

    #[test]
    fn the_sharpest_bend_ahead_skips_corners_too_sharp_to_follow() {
        let mut waypoints: Vec<Pose> = (0..=20).map(|i| at(0.5 * i as f64, 0.0)).collect();
        waypoints.extend((1..=20).map(|i| at(10.0, 0.5 * i as f64)));
        let cum = cumulative_lengths(&waypoints);
        let corner = sharpest_ahead(&waypoints, &cum, 5.0, 8.0, f64::INFINITY);
        assert!(corner > 2.0, "the corner reads {corner:.2} 1/m");
        assert_eq!(sharpest_ahead(&waypoints, &cum, 5.0, 8.0, 0.5), 0.0);
        assert_eq!(sharpest_ahead(&waypoints, &cum, 0.0, 4.0, f64::INFINITY), 0.0);
    }
}
