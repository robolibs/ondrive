//! Shortest forward path of bounded curvature between two poses (Dubins):
//! the best of the six words of left (L), right (R) and straight (S)
//! segments, sampled as poses.

use std::f64::consts::TAU;

/// A Dubins word: its three segment kinds and lengths in turning radii.
#[derive(Clone, Copy, Debug)]
struct Word {
    kinds: [Kind; 3],
    lengths: [f64; 3],
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Kind {
    Left,
    Straight,
    Right,
}

fn mod2pi(a: f64) -> f64 {
    a.rem_euclid(TAU)
}

/// Every word that joins the normalised start (`alpha`) and end (`beta`)
/// headings `d` turning radii apart.
fn words(alpha: f64, beta: f64, d: f64) -> Vec<Word> {
    use Kind::{Left as L, Right as R, Straight as S};
    let (sa, ca, sb, cb) = (alpha.sin(), alpha.cos(), beta.sin(), beta.cos());
    let cab = (alpha - beta).cos();
    let mut out = Vec::new();
    let p2 = 2.0 + d * d - 2.0 * cab + 2.0 * d * (sa - sb);
    if p2 >= 0.0 {
        let tmp = (cb - ca).atan2(d + sa - sb);
        out.push(Word { kinds: [L, S, L], lengths: [mod2pi(-alpha + tmp), p2.sqrt(), mod2pi(beta - tmp)] });
    }
    let p2 = 2.0 + d * d - 2.0 * cab + 2.0 * d * (sb - sa);
    if p2 >= 0.0 {
        let tmp = (ca - cb).atan2(d - sa + sb);
        out.push(Word { kinds: [R, S, R], lengths: [mod2pi(alpha - tmp), p2.sqrt(), mod2pi(-beta + tmp)] });
    }
    let p2 = -2.0 + d * d + 2.0 * cab + 2.0 * d * (sa + sb);
    if p2 >= 0.0 {
        let p = p2.sqrt();
        let tmp = (-ca - cb).atan2(d + sa + sb) - (-2.0_f64).atan2(p);
        out.push(Word { kinds: [L, S, R], lengths: [mod2pi(-alpha + tmp), p, mod2pi(-mod2pi(beta) + tmp)] });
    }
    let p2 = -2.0 + d * d + 2.0 * cab - 2.0 * d * (sa + sb);
    if p2 >= 0.0 {
        let p = p2.sqrt();
        let tmp = (ca + cb).atan2(d - sa - sb) - 2.0_f64.atan2(p);
        out.push(Word { kinds: [R, S, L], lengths: [mod2pi(alpha - tmp), p, mod2pi(beta - tmp)] });
    }
    let tmp = (6.0 - d * d + 2.0 * cab + 2.0 * d * (sa - sb)) / 8.0;
    if tmp.abs() <= 1.0 {
        let p = mod2pi(TAU - tmp.acos());
        let t = mod2pi(alpha - (ca - cb).atan2(d - sa + sb) + 0.5 * p);
        out.push(Word { kinds: [R, L, R], lengths: [t, p, mod2pi(alpha - beta - t + p)] });
    }
    let tmp = (6.0 - d * d + 2.0 * cab + 2.0 * d * (sb - sa)) / 8.0;
    if tmp.abs() <= 1.0 {
        let p = mod2pi(TAU - tmp.acos());
        let t = mod2pi(-alpha - (ca - cb).atan2(d + sa - sb) + 0.5 * p);
        out.push(Word { kinds: [L, R, L], lengths: [t, p, mod2pi(mod2pi(beta) - alpha - t + p)] });
    }
    out
}

/// Poses every `step` metres along the shortest Dubins path from `start` to
/// `goal` (each `(x, y, yaw)`) turning no tighter than `radius`, ending on
/// `goal`; with `radius` 0, the straight line between them.
pub fn dubins_path(start: (f64, f64, f64), goal: (f64, f64, f64), radius: f64, step: f64) -> Vec<(f64, f64, f64)> {
    let (dx, dy) = (goal.0 - start.0, goal.1 - start.1);
    let step = step.max(1e-3);
    if radius <= 1e-6 {
        let n = ((dx.hypot(dy) / step).ceil() as usize).max(1);
        let yaw = dy.atan2(dx);
        let mut out: Vec<(f64, f64, f64)> = (0..n).map(|i| {
            let u = i as f64 / n as f64;
            (start.0 + u * dx, start.1 + u * dy, yaw)
        }).collect();
        out.push(goal);
        return out;
    }
    let theta = dy.atan2(dx);
    let d = dx.hypot(dy) / radius;
    let (alpha, beta) = (mod2pi(start.2 - theta), mod2pi(goal.2 - theta));
    let Some(best) = words(alpha, beta, d).into_iter().min_by(|a, b| a.lengths.iter().sum::<f64>().total_cmp(&b.lengths.iter().sum::<f64>())) else {
        return vec![start, goal];
    };
    let (mut x, mut y, mut yaw) = start;
    let mut out = vec![start];
    let ds = step / radius;
    for (kind, len) in best.kinds.iter().zip(best.lengths) {
        let n = ((len / ds).ceil() as usize).max(1);
        let h = len / n as f64;
        for _ in 0..n {
            match kind {
                Kind::Straight => {
                    x += radius * h * yaw.cos();
                    y += radius * h * yaw.sin();
                }
                Kind::Left => {
                    x += radius * ((yaw + h).sin() - yaw.sin());
                    y -= radius * ((yaw + h).cos() - yaw.cos());
                    yaw += h;
                }
                Kind::Right => {
                    x -= radius * ((yaw - h).sin() - yaw.sin());
                    y += radius * ((yaw - h).cos() - yaw.cos());
                    yaw -= h;
                }
            }
            out.push((x, y, yaw));
        }
    }
    if let Some(last) = out.last_mut() {
        *last = goal;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ends_on_the_goal_for_any_pair_of_poses() {
        let cases = [
            ((0.0, 0.0, 0.0), (10.0, 5.0, 1.57)),
            ((0.0, 0.0, 0.0), (-4.0, 3.0, 3.1)),
            ((1.0, 2.0, 2.0), (1.5, 2.5, -2.0)),
            ((0.0, 0.0, 0.0), (0.0, 0.0, 3.14)),
            ((5.0, -3.0, -1.0), (-6.0, 8.0, 0.5)),
        ];
        for (s, g) in cases {
            let path = dubins_path(s, g, 2.0, 0.05);
            let gap = path.windows(2).map(|w| (w[1].0 - w[0].0).hypot(w[1].1 - w[0].1)).fold(0.0_f64, f64::max);
            let second_last = path[path.len() - 2];
            let end_jump = (second_last.0 - g.0).hypot(second_last.1 - g.1);
            assert!(end_jump < 0.06, "{s:?} -> {g:?}: last step {end_jump}");
            assert!(gap < 0.06, "{s:?} -> {g:?}: gap {gap}");
        }
    }

    #[test]
    fn turns_no_tighter_than_the_radius() {
        let path = dubins_path((0.0, 0.0, 0.0), (3.0, 4.0, 3.0), 2.5, 0.05);
        for w in path.windows(2).take(path.len() - 2) {
            let ds = (w[1].0 - w[0].0).hypot(w[1].1 - w[0].1);
            let dyaw = (w[1].2 - w[0].2).abs();
            assert!(dyaw <= ds / 2.5 + 1e-6, "dyaw {dyaw} over {ds}");
        }
    }
}
