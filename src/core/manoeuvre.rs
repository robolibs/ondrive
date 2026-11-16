//! Ways a machine that may reverse can drive where its path turns tighter
//! than it can steer, or onto a goal pose: Reeds–Shepp curves between the
//! poses either side, at a few radii from its tightest, the shortest whose
//! swept body keeps `room` from every obstacle that stays put, each cusp
//! charged as `CUSP_COST_M` metres.

use crate::core::curves::{CurvePath, reeds_shepp_all};
use crate::core::math::{normalize_angle, yaw_of};
use crate::core::obstacles::{CollisionChecker, stays_put};
use crate::core::path::{cumulative_lengths, sample};
use crate::turners::{Dubins, DubinsPath, DubinsSegmentType, Pose2D, ReedsShepp, ReedsSheppPath, ReedsSheppSegmentType, Sharper};
use crate::types::{Path, RobotConstraints, WorldConstraints};
use datapod::{Euler, Point, Pose, Quaternion};

/// Shares of the tightest turning radius the curves are tried at.
const RADII: [f64; 4] = [1.05, 1.2, 1.4, 1.7];
/// Metres of driving a cusp costs.
const CUSP_COST_M: f64 = 2.0;
/// Share over the tightest a path's curvature may run before it is driven
/// in more than one go.
const CURVATURE_SLACK: f64 = 1.05;
/// Metres between the poses of a planned way.
const STEP: f64 = 0.1;

/// The world's obstacles that stay put, the ones a way is planned round.
pub fn standing(world: Option<&WorldConstraints>) -> WorldConstraints {
    WorldConstraints { obstacles: world.map_or(Vec::new(), |w| w.obstacles.iter().filter(|o| stays_put(o)).cloned().collect()), ..Default::default() }
}

/// The cheapest way from `start` to `end` turning no tighter than `radius`
/// times each of `RADII`, with every pose of it keeping `room` from the
/// obstacles of `checker`.
pub fn way_between(start: &Pose, end: &Pose, radius: f64, checker: &CollisionChecker, room: f64) -> Option<CurvePath> {
    let clear = |c: &CurvePath| c.poses.iter().all(|p| checker.clearance(0, p.point.x, p.point.y, yaw_of(p)) >= room);
    let cost = |c: &CurvePath| c.length + CUSP_COST_M * c.directions.windows(2).filter(|w| w[0] != w[1]).count() as f64;
    RADII.iter().flat_map(|k| reeds_shepp_all(start, end, k * radius, STEP)).filter(|c| clear(c)).min_by(|a, b| cost(a).total_cmp(&cost(b)))
}

/// `path` with every stretch that turns tighter than `radius` allows, and
/// `radius` of it either side, driven as `way_between` the poses at its
/// ends instead, forward and reversing runs signed in its speeds at
/// `speed`, the rest keeping theirs. `None` where it has no such stretch,
/// already reverses, or one cannot be driven.
pub fn repaired(path: &Path, radius: f64, speed: f64, checker: &CollisionChecker, room: f64) -> Option<Path> {
    let w = &path.waypoints;
    if w.len() < 3 || path.is_closed || path.speeds.iter().any(|v| *v < 0.0) {
        return None;
    }
    let kept = |i: usize| path.speeds.get(i).copied().unwrap_or(speed);
    let cum = cumulative_lengths(w);
    let heading = |i: usize| (w[i + 1].point.y - w[i].point.y).atan2(w[i + 1].point.x - w[i].point.x);
    let tight: Vec<usize> = (1..w.len() - 1)
        .filter(|&i| {
            let ds = 0.5 * (cum[i + 1] - cum[i - 1]);
            ds > 1e-6 && normalize_angle(heading(i) - heading(i - 1)).abs() / ds > CURVATURE_SLACK / radius
        })
        .collect();
    let (&first, &last) = (tight.first()?, tight.last()?);
    let (from, to) = ((cum[first] - radius).max(0.0), (cum[last] + radius).min(*cum.last()?));
    let pose = |s: f64| {
        let (p, h) = sample(w, &cum, s);
        Pose { point: p, rotation: Quaternion::from_euler(Euler::new(0.0, 0.0, h)) }
    };
    let way = way_between(&pose(from), &pose(to), radius, checker, room)?;
    let before: Vec<usize> = (0..w.len()).filter(|&i| cum[i] < from).collect();
    let after: Vec<usize> = (0..w.len()).filter(|&i| cum[i] > to).collect();
    let mut waypoints: Vec<Pose> = before.iter().map(|&i| w[i]).collect();
    let mut speeds: Vec<f64> = before.iter().map(|&i| kept(i)).collect();
    waypoints.extend(way.poses.iter().map(|p| Pose { point: Point::new(p.point.x, p.point.y, w[first].point.z), rotation: p.rotation }));
    speeds.extend(way.directions.iter().map(|d| speed * *d as f64));
    waypoints.extend(after.iter().map(|&i| w[i]));
    speeds.extend(after.iter().map(|&i| kept(i)));
    Some(Path { waypoints, speeds, is_closed: false })
}

/// `path` split round its first stretch that turns tighter than `radius`
/// allows, `radius` of it either side: the path up to that, the pose `radius`
/// past it, and the path on from there. `None` where it has no such stretch,
/// is closed, or already reverses.
pub fn split_at_tight(path: &Path, radius: f64) -> Option<(Path, Pose, Path)> {
    let w = &path.waypoints;
    if w.len() < 3 || path.is_closed || path.speeds.iter().any(|v| *v < 0.0) {
        return None;
    }
    let cum = cumulative_lengths(w);
    let heading = |i: usize| (w[i + 1].point.y - w[i].point.y).atan2(w[i + 1].point.x - w[i].point.x);
    let tight: Vec<usize> = (1..w.len() - 1)
        .filter(|&i| {
            let ds = 0.5 * (cum[i + 1] - cum[i - 1]);
            ds > 1e-6 && normalize_angle(heading(i) - heading(i - 1)).abs() / ds > CURVATURE_SLACK / radius
        })
        .collect();
    let first = *tight.first()?;
    let mut last = first;
    for &i in &tight[1..] {
        if cum[i] - cum[last] > 2.0 * radius {
            break;
        }
        last = i;
    }
    let (from, to) = ((cum[first] - radius).max(0.0), (cum[last] + radius).min(*cum.last()?));
    let pose = |s: f64| {
        let (p, h) = sample(w, &cum, s);
        Pose { point: Point::new(p.x, p.y, w[first].point.z), rotation: Quaternion::from_euler(Euler::new(0.0, 0.0, h)) }
    };
    let speed = |i: usize| path.speeds.get(i).copied().unwrap_or(0.0);
    let part = |keep: &dyn Fn(f64) -> bool, ends: Pose, at_end: bool| {
        let kept: Vec<usize> = (0..w.len()).filter(|&i| keep(cum[i])).collect();
        let mut waypoints: Vec<Pose> = kept.iter().map(|&i| w[i]).collect();
        let mut speeds: Vec<f64> = kept.iter().map(|&i| speed(i)).collect();
        let edge = if at_end { speeds.last().copied() } else { speeds.first().copied() }.unwrap_or(0.0);
        if at_end {
            waypoints.push(ends);
            speeds.push(edge);
        } else {
            waypoints.insert(0, ends);
            speeds.insert(0, edge);
        }
        Path { waypoints, speeds, is_closed: false }
    };
    let before = part(&|s| s < from - 1e-6, pose(from), true);
    let after = part(&|s| s > to + 1e-6, pose(to), false);
    Some((before, pose(to), after))
}

/// One stretch of a planned turn: `length` metres at `curvature`, forward
/// (`dir` 1) or reversing (`dir` −1).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Piece {
    pub curvature: f64,
    pub dir: f64,
    pub length: f64,
}

/// A turn planned with the turners: which one, its pieces, the pose it
/// ends at, and how far along the path past the turn that lies.
#[derive(Clone, Debug, PartialEq)]
pub struct PlannedTurn {
    pub name: String,
    pub pieces: Vec<Piece>,
    pub end: Pose,
    pub along: f64,
}

/// Shares of the tightest turning radius turns are planned at.
const TURN_RADII: [f64; 4] = [1.0, 1.05, 1.25, 1.5];
/// Sharp-turn patterns tried, their corners joined by Reeds–Shepp curves.
const PATTERNS: [&str; 3] = ["three_point", "bulb", "fishtail"];
/// Metres between the poses a planned turn is checked for room at.
const CHECK_STEP: f64 = 0.05;


/// The pieces of a Reeds–Shepp curve of `radius`.
fn rs_pieces(path: &ReedsSheppPath, radius: f64) -> Vec<Piece> {
    path.segments
        .iter()
        .map(|s| {
            let turn = match s.r#type {
                ReedsSheppSegmentType::LeftForward | ReedsSheppSegmentType::LeftBackward => 1.0,
                ReedsSheppSegmentType::RightForward | ReedsSheppSegmentType::RightBackward => -1.0,
                _ => 0.0,
            };
            Piece { curvature: turn / radius, dir: if s.forward { 1.0 } else { -1.0 }, length: s.length * radius }
        })
        .collect()
}

/// The pieces of a Dubins curve of `radius`, all forward.
fn dubins_pieces(path: &DubinsPath, radius: f64) -> Vec<Piece> {
    path.segments
        .iter()
        .map(|s| {
            let turn = match s.r#type {
                DubinsSegmentType::Left => 1.0,
                DubinsSegmentType::Right => -1.0,
                DubinsSegmentType::Straight => 0.0,
            };
            Piece { curvature: turn / radius, dir: 1.0, length: s.length * radius }
        })
        .collect()
}

/// `pieces` with the empty ones dropped and runs at one curvature and
/// direction joined.
fn tidied(pieces: Vec<Piece>) -> Vec<Piece> {
    let mut out: Vec<Piece> = Vec::with_capacity(pieces.len());
    for p in pieces.into_iter().filter(|p| p.length > 1e-3) {
        match out.last_mut() {
            Some(q) if q.dir == p.dir && (q.curvature - p.curvature).abs() < 1e-9 => q.length += p.length,
            _ => out.push(p),
        }
    }
    out
}

/// `pose` moved `ahead` metres along its heading.
fn shifted(pose: (f64, f64, f64), ahead: f64) -> (f64, f64, f64) {
    (pose.0 + ahead * pose.2.cos(), pose.1 + ahead * pose.2.sin(), pose.2)
}

/// The pose `(x, y, heading)`, `ahead` metres in front of the axle the
/// machine turns about, that `pieces` driven by that axle end at from
/// `from`, or `None` where the body on the way comes nearer than `room` to
/// an obstacle of `checker`.
pub fn swept_pieces(checker: &CollisionChecker, from: (f64, f64, f64), pieces: &[Piece], room: f64, ahead: f64) -> Option<(f64, f64, f64)> {
    let (mut x, mut y, mut heading) = shifted(from, -ahead);
    for p in pieces {
        let steps = (p.length / CHECK_STEP).ceil().max(1.0) as usize;
        let ds = p.dir * p.length / steps as f64;
        for _ in 0..steps {
            let mid = heading + 0.5 * p.curvature * ds;
            x += ds * mid.cos();
            y += ds * mid.sin();
            heading += p.curvature * ds;
            let (px, py, _) = shifted((x, y, heading), ahead);
            if checker.clearance(0, px, py, heading) < room {
                return None;
            }
        }
    }
    Some(shifted((x, y, heading), ahead))
}

/// Every turn the turners give from `start` to one of `ends`, each a pose,
/// how far along the path past the turn it lies, and the seconds reaching
/// it saves, all poses `ahead`
/// metres in front of the axle the machine turns about, that axle turning
/// no tighter than `radius` times each of `TURN_RADII`: the Dubins and
/// Reeds–Shepp curves, and the sharp-turn patterns of a `body` (length,
/// width) with their corners joined by the cheapest clear Reeds–Shepp
/// curve; of those whose body keeps `room` from the obstacles of
/// `checker`, the cheapest by `cost` less the credit of its end. Where it
/// may not `reverse`, only the Dubins curves, and the patterns joined by
/// them.
#[allow(clippy::too_many_arguments)]
pub fn plan_turn(
    start: &Pose,
    ends: &[(Pose, f64, f64)],
    ahead: f64,
    radius: f64,
    body: (f64, f64),
    checker: &CollisionChecker,
    room: f64,
    cost: &dyn Fn(&[Piece]) -> f64,
    reverse: bool,
) -> Option<PlannedTurn> {
    let axle = |p: &Pose| {
        let (x, y, h) = shifted((p.point.x, p.point.y, yaw_of(p)), -ahead);
        Pose2D::new(x, y, h)
    };
    let a = axle(start);
    let origin = |p: &Pose2D| shifted((p.point.x, p.point.y, p.yaw), ahead);
    let clear = |from: &Pose2D, pieces: &[Piece]| swept_pieces(checker, origin(from), pieces, room, ahead).is_some();
    let curves = |from: &Pose2D, to: &Pose2D, r: f64| -> Vec<(String, Vec<Piece>)> {
        let forward = Dubins::new(r).get_all_paths(*from, *to, f64::MAX).into_iter().map(|p| (format!("dubins {}", p.name), tidied(dubins_pieces(&p, r))));
        let mut all: Vec<(String, Vec<Piece>)> = forward.collect();
        if reverse {
            all.extend(ReedsShepp::new(r).get_all_paths(*from, *to, f64::MAX).into_iter().map(|p| (format!("reeds-shepp {}", p.name), tidied(rs_pieces(&p, r)))));
        }
        all
    };
    let joined = |from: &Pose2D, to: &Pose2D, r: f64| {
        curves(from, to, r).into_iter().map(|(_, p)| p).filter(|p| clear(from, p)).min_by(|x, y| cost(x).total_cmp(&cost(y)))
    };
    let mut candidates: Vec<(PlannedTurn, f64)> = Vec::new();
    for (end, along, credit) in ends {
        let b = axle(end);
        let mut add = |name: String, pieces: Vec<Piece>| candidates.push((PlannedTurn { name, pieces, end: *end, along: *along }, *credit));
        for k in TURN_RADII {
            let r = k * radius;
            for (name, pieces) in curves(&a, &b, r) {
                add(name, pieces);
            }
            for pattern in PATTERNS {
                let corners = Sharper::new(r, body.0, body.1).plan_sharp_turn(a, b, pattern).waypoints;
                let legs: Option<Vec<Vec<Piece>>> = corners.windows(2).map(|w| joined(&w[0], &w[1], r)).collect();
                if let Some(legs) = legs {
                    add(pattern.to_string(), tidied(legs.concat()));
                }
            }
        }
    }
    candidates
        .into_iter()
        .filter(|(t, _)| !t.pieces.is_empty() && clear(&a, &t.pieces))
        .map(|(t, credit)| {
            let net = cost(&t.pieces) - credit;
            (t, net)
        })
        .min_by(|x, y| x.1.total_cmp(&y.1))
        .map(|(t, _)| t)
}

/// The poses of `path` at each of `at` metres along it that it reaches,
/// with how far along each lies.
pub fn poses_along(path: &Path, at: &[f64]) -> Vec<(Pose, f64)> {
    let w = &path.waypoints;
    if w.len() < 2 {
        return Vec::new();
    }
    let cum = cumulative_lengths(w);
    let total = cum.last().copied().unwrap_or(0.0);
    at.iter()
        .filter(|s| **s <= total)
        .map(|&s| {
            let (p, h) = sample(w, &cum, s);
            (Pose { point: Point::new(p.x, p.y, w[0].point.z), rotation: Quaternion::from_euler(Euler::new(0.0, 0.0, h)) }, s)
        })
        .collect()
}

/// `path` from `s` metres along it on, starting at its pose there.
pub fn path_from(path: &Path, s: f64) -> Path {
    let w = &path.waypoints;
    if w.len() < 2 || s <= 0.0 {
        return path.clone();
    }
    let cum = cumulative_lengths(w);
    let kept: Vec<usize> = (0..w.len()).filter(|&i| cum[i] > s + 1e-6).collect();
    let Some((start, _)) = poses_along(path, &[s]).into_iter().next() else {
        return path.clone();
    };
    let speed = |i: usize| path.speeds.get(i).copied().unwrap_or(0.0);
    let mut waypoints = vec![start];
    let mut speeds = vec![kept.first().map_or(0.0, |&i| speed(i))];
    waypoints.extend(kept.iter().map(|&i| w[i]));
    speeds.extend(kept.iter().map(|&i| speed(i)));
    Path { waypoints, speeds, is_closed: false }
}

/// The tightest radius an Ackermann `constraints` turns at, if it has one.
pub fn tightest(constraints: &RobotConstraints) -> Option<f64> {
    let k = crate::core::kinematics::max_curvature(constraints);
    (k.is_finite() && k > 0.0).then(|| 1.0 / k)
}

#[cfg(test)]
mod tests {
    use super::{repaired, standing, tightest};
    use crate::core::obstacles::CollisionChecker;
    use crate::types::{Footprint, GaussianMode, Obstacle, Path, RobotConstraints, SteeringType, WorldConstraints};
    use datapod::{Euler, Point, Pose, Quaternion};

    fn pose(x: f64, y: f64, yaw: f64) -> Pose {
        Pose { point: Point::new(x, y, 0.0), rotation: Quaternion::from_euler(Euler::new(0.0, 0.0, yaw)) }
    }

    fn hunter() -> RobotConstraints {
        let mut c = RobotConstraints::default();
        c.steering_type = SteeringType::Ackermann;
        c.wheelbase = 0.65;
        c.max_steering_angle = 0.46;
        c.max_linear_velocity = 1.5;
        c.footprint = Footprint::Polygon { points: vec![(0.575, 0.375), (-0.485, 0.375), (-0.485, -0.375), (0.575, -0.375)] };
        c
    }

    fn post(id: u64, x: f64, y: f64) -> Obstacle {
        let mode = GaussianMode { weight: 1.0, mean_x: vec![x; 31], mean_y: vec![y; 31], std_x: vec![0.03; 31], std_y: vec![0.03; 31] };
        Obstacle { id, radius: 0.22, modes: vec![mode], dt: 0.2 }
    }

    /// A U-turn of 0.45 m radius, three times tighter than the Hunter steers,
    /// in a street of posts too narrow to turn round in: driven in more than
    /// one go, with a reversing run, clear of the posts.
    #[test]
    fn a_turn_too_tight_is_driven_in_more_than_one_go() {
        let c = hunter();
        let r = 0.45;
        let mut waypoints: Vec<Pose> = (0..=24).map(|i| pose(0.5 * i as f64, 0.0, 0.0)).collect();
        waypoints.extend((1..24).map(|k| {
            let a = -std::f64::consts::FRAC_PI_2 + std::f64::consts::PI * k as f64 / 24.0;
            pose(12.0 + r * a.cos(), r + r * a.sin(), a + std::f64::consts::FRAC_PI_2)
        }));
        waypoints.extend((0..=24).map(|i| pose(12.0 - 0.5 * i as f64, 2.0 * r, std::f64::consts::PI)));
        let speeds = vec![0.0; waypoints.len()];
        let path = Path { waypoints, speeds, is_closed: false };
        let walls: Vec<Obstacle> = (0..40).flat_map(|i| [post(2 * i, 0.4 * i as f64, -1.4), post(2 * i + 1, 0.4 * i as f64, 2.6)]).collect();
        let world = WorldConstraints { obstacles: walls, ..Default::default() };
        let standing = standing(Some(&world));
        let checker = CollisionChecker::new(Some(&standing), &c, 0.0);
        let radius = tightest(&c).expect("an Ackermann machine turns no tighter than its lock");
        let way = repaired(&path, radius, 1.0, &checker, 0.3).expect("no way round the hairpin");
        assert!(way.speeds.iter().any(|v| *v < 0.0), "no reversing run");
        let worst = way.waypoints.iter().map(|p| checker.clearance(0, p.point.x, p.point.y, crate::core::math::yaw_of(p))).fold(f64::INFINITY, f64::min);
        assert!(worst >= 0.3, "comes {worst:.2} m from a post");
    }
}
