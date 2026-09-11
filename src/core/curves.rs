//! Dubins and Reeds-Shepp curves between poses for a car with a minimum
//! turning radius. Words are searched exhaustively (6 Dubins words, the 48
//! Reeds-Shepp words via reflection and time-flip of the base families) and
//! the shortest is sampled into poses with a driving direction per sample.

use crate::core::math::normalize_angle;
use datapod::{Euler, Point, Pose, Quaternion};
use std::f64::consts::PI;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Segment {
    Left,
    Straight,
    Right,
    None,
}

/// A curve as up to five signed segment lengths (in units of the turning
/// radius); a negative length is driven in reverse.
#[derive(Clone, Copy, Debug)]
pub struct Word {
    pub lengths: [f64; 5],
    pub segments: [Segment; 5],
}

impl Word {
    fn new(lengths: [f64; 5], segments: [Segment; 5]) -> Self {
        Self { lengths, segments }
    }

    pub fn length(&self) -> f64 {
        self.lengths.iter().map(|l| l.abs()).sum()
    }
}

/// Sampled curve: poses and the direction (+1 forward, -1 reverse) of the
/// motion that reaches each pose.
#[derive(Clone, Debug, Default)]
pub struct CurvePath {
    pub poses: Vec<Pose>,
    pub directions: Vec<i8>,
    pub length: f64,
}

fn yaw_of(p: &Pose) -> f64 {
    p.rotation.to_euler().yaw
}

fn pose_at(x: f64, y: f64, yaw: f64) -> Pose {
    Pose {
        point: Point::new(x, y, 0.0),
        rotation: Quaternion::from_euler(Euler::new(0.0, 0.0, yaw)),
    }
}

fn mod2pi(a: f64) -> f64 {
    let mut v = a % (2.0 * PI);
    if v < 0.0 {
        v += 2.0 * PI;
    }
    v
}

fn polar(x: f64, y: f64) -> (f64, f64) {
    (x.hypot(y), y.atan2(x))
}

const ZERO: f64 = 1e-10;

/// Goal expressed in the start frame and scaled by `1 / radius`.
fn relative(start: &Pose, goal: &Pose, radius: f64) -> (f64, f64, f64) {
    let dx = goal.point.x - start.point.x;
    let dy = goal.point.y - start.point.y;
    let th = yaw_of(start);
    let (s, c) = th.sin_cos();
    (
        (dx * c + dy * s) / radius,
        (-dx * s + dy * c) / radius,
        normalize_angle(yaw_of(goal) - th),
    )
}

// ---------------------------------------------------------------------------
// Dubins
// ---------------------------------------------------------------------------

fn dubins_words(x: f64, y: f64, phi: f64) -> Vec<Word> {
    use Segment::*;
    let d = x.hypot(y);
    let theta = mod2pi(y.atan2(x));
    let alpha = mod2pi(-theta);
    let beta = mod2pi(phi - theta);
    let (sa, ca) = alpha.sin_cos();
    let (sb, cb) = beta.sin_cos();
    let cab = (alpha - beta).cos();
    let mut out = Vec::new();

    let tmp = 2.0 + d * d - 2.0 * cab + 2.0 * d * (sa - sb);
    if tmp >= 0.0 {
        let t = mod2pi(-alpha + (cb - ca).atan2(d + sa - sb));
        let p = tmp.sqrt();
        let q = mod2pi(beta - (cb - ca).atan2(d + sa - sb));
        out.push(Word::new([t, p, q, 0.0, 0.0], [Left, Straight, Left, None, None]));
    }
    let tmp = 2.0 + d * d - 2.0 * cab + 2.0 * d * (sb - sa);
    if tmp >= 0.0 {
        let t = mod2pi(alpha - (ca - cb).atan2(d - sa + sb));
        let p = tmp.sqrt();
        let q = mod2pi(-beta + (ca - cb).atan2(d - sa + sb));
        out.push(Word::new([t, p, q, 0.0, 0.0], [Right, Straight, Right, None, None]));
    }
    let tmp = -2.0 + d * d + 2.0 * cab + 2.0 * d * (sa + sb);
    if tmp >= 0.0 {
        let p = tmp.sqrt();
        let t = mod2pi(-alpha + (-ca - cb).atan2(d + sa + sb) - (-2.0f64).atan2(p));
        let q = mod2pi(-mod2pi(beta) + (-ca - cb).atan2(d + sa + sb) - (-2.0f64).atan2(p));
        out.push(Word::new([t, p, q, 0.0, 0.0], [Left, Straight, Right, None, None]));
    }
    let tmp = d * d - 2.0 + 2.0 * cab - 2.0 * d * (sa + sb);
    if tmp >= 0.0 {
        let p = tmp.sqrt();
        let t = mod2pi(alpha - (ca + cb).atan2(d - sa - sb) + (2.0f64).atan2(p));
        let q = mod2pi(beta - (ca + cb).atan2(d - sa - sb) + (2.0f64).atan2(p));
        out.push(Word::new([t, p, q, 0.0, 0.0], [Right, Straight, Left, None, None]));
    }
    let tmp = (6.0 - d * d + 2.0 * cab + 2.0 * d * (sa - sb)) / 8.0;
    if tmp.abs() <= 1.0 {
        let p = mod2pi(2.0 * PI - tmp.acos());
        let t = mod2pi(alpha - (ca - cb).atan2(d - sa + sb) + mod2pi(p / 2.0));
        let q = mod2pi(alpha - beta - t + mod2pi(p));
        out.push(Word::new([t, p, q, 0.0, 0.0], [Right, Left, Right, None, None]));
    }
    let tmp = (6.0 - d * d + 2.0 * cab + 2.0 * d * (sb - sa)) / 8.0;
    if tmp.abs() <= 1.0 {
        let p = mod2pi(2.0 * PI - tmp.acos());
        let t = mod2pi(-alpha - (ca - cb).atan2(d + sa - sb) + p / 2.0);
        let q = mod2pi(mod2pi(beta) - alpha - t + mod2pi(p));
        out.push(Word::new([t, p, q, 0.0, 0.0], [Left, Right, Left, None, None]));
    }
    out
}

// ---------------------------------------------------------------------------
// Reeds-Shepp (after Reeds & Shepp 1990 / the OMPL formulation)
// ---------------------------------------------------------------------------

type Triple = Option<(f64, f64, f64)>;

fn lp_sp_lp(x: f64, y: f64, phi: f64) -> Triple {
    let (u, t) = polar(x - phi.sin(), y - 1.0 + phi.cos());
    if t >= -ZERO {
        let v = mod2pi(phi - t);
        if v >= -ZERO {
            return Some((t, u, v));
        }
    }
    None
}

fn lp_sp_rp(x: f64, y: f64, phi: f64) -> Triple {
    let (u1, t1) = polar(x + phi.sin(), y - 1.0 - phi.cos());
    let u1 = u1 * u1;
    if u1 >= 4.0 {
        let u = (u1 - 4.0).sqrt();
        let theta = (2.0f64).atan2(u);
        let t = mod2pi(t1 + theta);
        let v = mod2pi(t - phi);
        if t >= -ZERO && v >= -ZERO {
            return Some((t, u, v));
        }
    }
    None
}

fn lp_rm_l(x: f64, y: f64, phi: f64) -> Triple {
    let xi = x - phi.sin();
    let eta = y - 1.0 + phi.cos();
    let (u1, theta) = polar(xi, eta);
    if u1 <= 4.0 {
        let u = -2.0 * (0.25 * u1).asin();
        let t = mod2pi(theta + 0.5 * u + PI);
        let v = mod2pi(phi - t + u);
        if t >= -ZERO && u <= ZERO {
            return Some((t, u, v));
        }
    }
    None
}

fn tau_omega(u: f64, v: f64, xi: f64, eta: f64, phi: f64) -> (f64, f64) {
    let delta = mod2pi(u - v);
    let a = u.sin() - delta.sin();
    let b = u.cos() - delta.cos() - 1.0;
    let t1 = (eta * a - xi * b).atan2(xi * a + eta * b);
    let t2 = 2.0 * (delta.cos() - v.cos() - u.cos()) + 3.0;
    let tau = if t2 < 0.0 { mod2pi(t1 + PI) } else { mod2pi(t1) };
    let omega = mod2pi(tau - u + v - phi);
    (tau, omega)
}

fn lp_rup_lum_rm(x: f64, y: f64, phi: f64) -> Triple {
    let xi = x + phi.sin();
    let eta = y - 1.0 - phi.cos();
    let rho = 0.25 * (2.0 + (xi * xi + eta * eta).sqrt());
    if rho <= 1.0 {
        let u = rho.acos();
        let (t, v) = tau_omega(u, -u, xi, eta, phi);
        if t >= -ZERO && v <= ZERO {
            return Some((t, u, v));
        }
    }
    None
}

fn lp_rum_lum_rp(x: f64, y: f64, phi: f64) -> Triple {
    let xi = x + phi.sin();
    let eta = y - 1.0 - phi.cos();
    let rho = (20.0 - xi * xi - eta * eta) / 16.0;
    if (0.0..=1.0).contains(&rho) {
        let u = -rho.acos();
        if u >= -0.5 * PI {
            let (t, v) = tau_omega(u, u, xi, eta, phi);
            if t >= -ZERO && v >= -ZERO {
                return Some((t, u, v));
            }
        }
    }
    None
}

fn lp_rm_sm_lm(x: f64, y: f64, phi: f64) -> Triple {
    let xi = x - phi.sin();
    let eta = y - 1.0 + phi.cos();
    let (rho, theta) = polar(xi, eta);
    if rho >= 2.0 {
        let r = (rho * rho - 4.0).sqrt();
        let u = 2.0 - r;
        let t = mod2pi(theta + r.atan2(-2.0));
        let v = mod2pi(phi - 0.5 * PI - t);
        if t >= -ZERO && u <= ZERO && v <= ZERO {
            return Some((t, u, v));
        }
    }
    None
}

fn lp_rm_sm_rm(x: f64, y: f64, phi: f64) -> Triple {
    let xi = x + phi.sin();
    let eta = y - 1.0 - phi.cos();
    let (rho, theta) = polar(-eta, xi);
    if rho >= 2.0 {
        let t = theta;
        let u = 2.0 - rho;
        let v = mod2pi(t + 0.5 * PI - phi);
        if t >= -ZERO && u <= ZERO && v <= ZERO {
            return Some((t, u, v));
        }
    }
    None
}

fn lp_rm_s_lm_rp(x: f64, y: f64, phi: f64) -> Triple {
    let xi = x + phi.sin();
    let eta = y - 1.0 - phi.cos();
    let (rho, _) = polar(xi, eta);
    if rho >= 2.0 {
        let u = 4.0 - (rho * rho - 4.0).sqrt();
        if u <= ZERO {
            let t = mod2pi(((4.0 - u) * xi - 2.0 * eta).atan2(-2.0 * xi + (u - 4.0) * eta));
            let v = mod2pi(t - phi);
            if t >= -ZERO && v >= -ZERO {
                return Some((t, u, v));
            }
        }
    }
    None
}

fn flip(seg: Segment) -> Segment {
    match seg {
        Segment::Left => Segment::Right,
        Segment::Right => Segment::Left,
        s => s,
    }
}

/// Try one base family under the four symmetries: identity, time-flip
/// (reverse driving), reflection (mirror steering), and both.
fn variants(
    out: &mut Vec<Word>,
    x: f64,
    y: f64,
    phi: f64,
    f: fn(f64, f64, f64) -> Triple,
    segs: [Segment; 5],
    build: fn((f64, f64, f64)) -> [f64; 5],
) {
    let mirrored = [flip(segs[0]), flip(segs[1]), flip(segs[2]), flip(segs[3]), flip(segs[4])];
    if let Some(r) = f(x, y, phi) {
        out.push(Word::new(build(r), segs));
    }
    if let Some(r) = f(-x, y, -phi) {
        let l = build(r);
        out.push(Word::new([-l[0], -l[1], -l[2], -l[3], -l[4]], segs));
    }
    if let Some(r) = f(x, -y, -phi) {
        out.push(Word::new(build(r), mirrored));
    }
    if let Some(r) = f(-x, -y, phi) {
        let l = build(r);
        out.push(Word::new([-l[0], -l[1], -l[2], -l[3], -l[4]], mirrored));
    }
}

fn reeds_shepp_words(x: f64, y: f64, phi: f64) -> Vec<Word> {
    use Segment::*;
    let mut out = Vec::new();
    let tuv = |(t, u, v): (f64, f64, f64)| [t, u, v, 0.0, 0.0];
    variants(&mut out, x, y, phi, lp_sp_lp, [Left, Straight, Left, None, None], tuv);
    variants(&mut out, x, y, phi, lp_sp_rp, [Left, Straight, Right, None, None], tuv);
    variants(&mut out, x, y, phi, lp_rm_l, [Left, Right, Left, None, None], tuv);
    // Backward CCC: swap start and goal.
    let xb = x * phi.cos() + y * phi.sin();
    let yb = x * phi.sin() - y * phi.cos();
    variants(&mut out, xb, yb, phi, lp_rm_l, [Left, Right, Left, None, None], |(t, u, v)| {
        [v, u, t, 0.0, 0.0]
    });
    variants(&mut out, x, y, phi, lp_rup_lum_rm, [Left, Right, Left, Right, None], |(t, u, v)| {
        [t, u, -u, v, 0.0]
    });
    variants(&mut out, x, y, phi, lp_rum_lum_rp, [Left, Right, Left, Right, None], |(t, u, v)| {
        [t, u, u, v, 0.0]
    });
    variants(&mut out, x, y, phi, lp_rm_sm_lm, [Left, Right, Straight, Left, None], |(t, u, v)| {
        [t, -0.5 * PI, u, v, 0.0]
    });
    variants(&mut out, x, y, phi, lp_rm_sm_rm, [Left, Right, Straight, Right, None], |(t, u, v)| {
        [t, -0.5 * PI, u, v, 0.0]
    });
    variants(&mut out, xb, yb, phi, lp_rm_sm_lm, [Left, Straight, Right, Left, None], |(t, u, v)| {
        [v, u, -0.5 * PI, t, 0.0]
    });
    variants(&mut out, xb, yb, phi, lp_rm_sm_rm, [Right, Straight, Right, Left, None], |(t, u, v)| {
        [v, u, -0.5 * PI, t, 0.0]
    });
    variants(&mut out, x, y, phi, lp_rm_s_lm_rp, [Left, Right, Straight, Left, Right], |(t, u, v)| {
        [t, -0.5 * PI, u, -0.5 * PI, v]
    });
    out
}

/// Integrate a word from `start` with the given turning radius and sample
/// it every `spacing` metres.
fn sample_word(word: &Word, start: &Pose, radius: f64, spacing: f64) -> CurvePath {
    let mut out = CurvePath::default();
    let (mut x, mut y, mut th) = (start.point.x, start.point.y, yaw_of(start));
    out.poses.push(pose_at(x, y, th));
    out.directions.push(if word.lengths.iter().find(|l| l.abs() > ZERO).copied().unwrap_or(1.0) >= 0.0 { 1 } else { -1 });
    let ds = (spacing / radius).max(1e-4);
    for (len, seg) in word.lengths.iter().zip(word.segments.iter()) {
        if *seg == Segment::None || len.abs() <= ZERO {
            continue;
        }
        let dir = if *len >= 0.0 { 1 } else { -1 };
        let total = len.abs();
        let n = (total / ds).ceil().max(1.0) as usize;
        let step = total / n as f64 * dir as f64;
        for _ in 0..n {
            match seg {
                Segment::Left => {
                    let nth = th + step;
                    x += radius * (nth.sin() - th.sin());
                    y -= radius * (nth.cos() - th.cos());
                    th = nth;
                }
                Segment::Right => {
                    let nth = th - step;
                    x -= radius * (nth.sin() - th.sin());
                    y += radius * (nth.cos() - th.cos());
                    th = nth;
                }
                Segment::Straight => {
                    x += radius * step * th.cos();
                    y += radius * step * th.sin();
                }
                Segment::None => {}
            }
            out.poses.push(pose_at(x, y, normalize_angle(th)));
            out.directions.push(dir);
        }
        out.length += total * radius;
    }
    out
}

fn shortest(words: Vec<Word>) -> Option<Word> {
    words
        .into_iter()
        .filter(|w| w.lengths.iter().all(|l| l.is_finite()))
        .min_by(|a, b| a.length().partial_cmp(&b.length()).unwrap_or(std::cmp::Ordering::Equal))
}

/// Shortest forward-only curve (Dubins) from `start` to `goal`.
pub fn dubins(start: &Pose, goal: &Pose, radius: f64, spacing: f64) -> Option<CurvePath> {
    let radius = radius.max(1e-3);
    let (x, y, phi) = relative(start, goal, radius);
    let word = shortest(dubins_words(x, y, phi))?;
    Some(sample_word(&word, start, radius, spacing))
}

/// Shortest curve allowing reverse driving (Reeds-Shepp).
pub fn reeds_shepp(start: &Pose, goal: &Pose, radius: f64, spacing: f64) -> Option<CurvePath> {
    let radius = radius.max(1e-3);
    let (x, y, phi) = relative(start, goal, radius);
    let word = shortest(reeds_shepp_words(x, y, phi))?;
    Some(sample_word(&word, start, radius, spacing))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn end_error(c: &CurvePath, goal: &Pose) -> (f64, f64) {
        let last = c.poses.last().unwrap();
        (
            last.point.distance_to_2d(goal.point),
            normalize_angle(yaw_of(last) - yaw_of(goal)).abs(),
        )
    }

    #[test]
    fn dubins_reaches_goal_poses() {
        let start = pose_at(0.0, 0.0, 0.0);
        for (gx, gy, gth) in [(4.0, 0.0, 0.0), (3.0, 3.0, PI / 2.0), (-2.0, 1.0, PI), (0.5, 2.0, -PI / 2.0), (1.0, -1.0, 0.3)] {
            let goal = pose_at(gx, gy, gth);
            let c = dubins(&start, &goal, 1.0, 0.05).expect("dubins word");
            let (d, a) = end_error(&c, &goal);
            assert!(d < 0.05 && a < 0.05, "goal ({gx},{gy},{gth}): d={d:.3} a={a:.3}");
            assert!(c.directions.iter().all(|s| *s == 1));
        }
    }

    #[test]
    fn reeds_shepp_reaches_goal_poses_and_is_never_longer_than_dubins() {
        let start = pose_at(0.0, 0.0, 0.0);
        let goals = [
            (4.0, 0.0, 0.0),
            (3.0, 3.0, PI / 2.0),
            (-2.0, 1.0, PI),
            (0.5, 2.0, -PI / 2.0),
            (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.3, 0.3, 0.2),
            (-3.0, -2.0, 2.5),
        ];
        for (gx, gy, gth) in goals {
            let goal = pose_at(gx, gy, gth);
            let rs = reeds_shepp(&start, &goal, 1.0, 0.05).expect("rs word");
            let (d, a) = end_error(&rs, &goal);
            assert!(d < 0.05 && a < 0.05, "goal ({gx},{gy},{gth}): d={d:.3} a={a:.3}");
            let du = dubins(&start, &goal, 1.0, 0.05).unwrap();
            assert!(rs.length <= du.length + 1e-6, "goal ({gx},{gy},{gth}): rs {:.3} > dubins {:.3}", rs.length, du.length);
        }
    }

    #[test]
    fn reeds_shepp_uses_reverse_for_a_goal_behind() {
        let start = pose_at(0.0, 0.0, 0.0);
        let goal = pose_at(-2.0, 0.0, 0.0);
        let rs = reeds_shepp(&start, &goal, 1.0, 0.05).unwrap();
        assert!(rs.directions.contains(&-1));
        assert!((rs.length - 2.0).abs() < 0.05);
    }
}
