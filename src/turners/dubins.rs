use std::f64::consts::PI;

use super::geometry::{Point2Ext, points_equal};

use super::Pose2D;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DubinsSegmentType {
    Left,
    Straight,
    Right,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DubinsSegment {
    pub r#type: DubinsSegmentType,
    pub length: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DubinsPath {
    pub segments: Vec<DubinsSegment>,
    pub waypoints: Vec<Pose2D>,
    pub total_length: f64,
    pub name: String,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dubins {
    turning_radius: f64,
}

#[derive(Clone, Copy)]
struct NormalizedDubinsPath {
    types: [DubinsSegmentType; 3],
    lengths: [f64; 3],
}

impl NormalizedDubinsPath {
    fn invalid() -> Self {
        Self {
            types: [
                DubinsSegmentType::Left,
                DubinsSegmentType::Straight,
                DubinsSegmentType::Left,
            ],
            lengths: [0.0, f64::MAX, 0.0],
        }
    }

    fn total_length(self) -> f64 {
        self.lengths[0] + self.lengths[1] + self.lengths[2]
    }

    fn is_valid(self) -> bool {
        self.lengths[1].is_finite() && self.lengths[1] < f64::MAX * 0.5
    }
}

impl Dubins {
    pub const fn new(turning_radius: f64) -> Self {
        Self { turning_radius }
    }

    pub fn plan_path(&self, start: Pose2D, goal: Pose2D, step_size: f64) -> DubinsPath {
        self.get_all_paths(start, goal, step_size.max(1e-3))
            .into_iter()
            .min_by(|a, b| {
                a.total_length
                    .partial_cmp(&b.total_length)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or_else(|| DubinsPath {
                segments: Vec::new(),
                waypoints: vec![start, goal],
                total_length: 0.0,
                name: String::new(),
            })
    }

    pub fn get_all_paths(&self, start: Pose2D, goal: Pose2D, step_size: f64) -> Vec<DubinsPath> {
        let radius = self.turning_radius.max(1e-9);
        let q0 = [start.point.x(), start.point.y(), start.yaw];
        let q1 = [goal.point.x(), goal.point.y(), goal.yaw];

        let dx = q1[0] - q0[0];
        let dy = q1[1] - q0[1];
        let d = (dx * dx + dy * dy).sqrt() / radius;
        let theta = dy.atan2(dx);
        let alpha = mod2pi(q0[2] - theta);
        let beta = mod2pi(q1[2] - theta);

        let candidates = [
            ("LSL", self.dubins_lsl(d, alpha, beta)),
            ("RSR", self.dubins_rsr(d, alpha, beta)),
            ("RSL", self.dubins_rsl(d, alpha, beta)),
            ("LSR", self.dubins_lsr(d, alpha, beta)),
            ("RLR", self.dubins_rlr(d, alpha, beta)),
            ("LRL", self.dubins_lrl(d, alpha, beta)),
        ];

        let mut paths = Vec::new();
        for (name, candidate) in candidates {
            if !candidate.is_valid() {
                continue;
            }

            let total_length = candidate.total_length() * radius;
            let mut waypoints = Vec::new();
            let mut seg = 0.0;
            while seg <= total_length {
                let qnew = self.interpolate(q0, candidate, seg / radius);
                waypoints.push(Pose2D::new(qnew[0], qnew[1], qnew[2]));
                seg += step_size;
            }
            if waypoints
                .last()
                .map(|pose| {
                    !points_equal(pose.point, goal.point, 1e-6)
                        || (pose.yaw - goal.yaw).abs() > 1e-6
                })
                .unwrap_or(true)
            {
                waypoints.push(goal);
            }

            let segments = candidate
                .lengths
                .into_iter()
                .zip(candidate.types)
                .map(|(length, r#type)| DubinsSegment { r#type, length })
                .collect();

            paths.push(DubinsPath {
                segments,
                waypoints,
                total_length,
                name: name.to_string(),
            });
        }
        paths
    }

    fn interpolate(&self, q0: [f64; 3], path: NormalizedDubinsPath, mut seg: f64) -> [f64; 3] {
        seg = seg.clamp(0.0, path.total_length());

        let mut s = [0.0, 0.0, q0[2]];
        for i in 0..3 {
            if seg <= 0.0 {
                break;
            }
            let v = seg.min(path.lengths[i]);
            seg -= v;
            let phi = s[2];
            match path.types[i] {
                DubinsSegmentType::Left => {
                    s[0] += (phi + v).sin() - phi.sin();
                    s[1] += -(phi + v).cos() + phi.cos();
                    s[2] = phi + v;
                }
                DubinsSegmentType::Right => {
                    s[0] += -(phi - v).sin() + phi.sin();
                    s[1] += (phi - v).cos() - phi.cos();
                    s[2] = phi - v;
                }
                DubinsSegmentType::Straight => {
                    s[0] += v * phi.cos();
                    s[1] += v * phi.sin();
                }
            }
        }

        s[0] = s[0] * self.turning_radius.max(1e-9) + q0[0];
        s[1] = s[1] * self.turning_radius.max(1e-9) + q0[1];
        s
    }

    fn dubins_lsl(&self, d: f64, alpha: f64, beta: f64) -> NormalizedDubinsPath {
        let ca = alpha.cos();
        let sa = alpha.sin();
        let cb = beta.cos();
        let sb = beta.sin();
        let tmp = 2.0 + d * d - 2.0 * (ca * cb + sa * sb - d * (sa - sb));
        if tmp >= DUBINS_ZERO {
            let theta = (cb - ca).atan2(d + sa - sb);
            let t = mod2pi(-alpha + theta);
            let p = tmp.max(0.0).sqrt();
            let q = mod2pi(beta - theta);
            return NormalizedDubinsPath {
                types: [
                    DubinsSegmentType::Left,
                    DubinsSegmentType::Straight,
                    DubinsSegmentType::Left,
                ],
                lengths: [t, p, q],
            };
        }
        NormalizedDubinsPath::invalid()
    }

    fn dubins_rsr(&self, d: f64, alpha: f64, beta: f64) -> NormalizedDubinsPath {
        let ca = alpha.cos();
        let sa = alpha.sin();
        let cb = beta.cos();
        let sb = beta.sin();
        let tmp = 2.0 + d * d - 2.0 * (ca * cb + sa * sb - d * (sb - sa));
        if tmp >= DUBINS_ZERO {
            let theta = (ca - cb).atan2(d - sa + sb);
            let t = mod2pi(alpha - theta);
            let p = tmp.max(0.0).sqrt();
            let q = mod2pi(-beta + theta);
            return NormalizedDubinsPath {
                types: [
                    DubinsSegmentType::Right,
                    DubinsSegmentType::Straight,
                    DubinsSegmentType::Right,
                ],
                lengths: [t, p, q],
            };
        }
        NormalizedDubinsPath::invalid()
    }

    fn dubins_rsl(&self, d: f64, alpha: f64, beta: f64) -> NormalizedDubinsPath {
        let ca = alpha.cos();
        let sa = alpha.sin();
        let cb = beta.cos();
        let sb = beta.sin();
        let tmp = d * d - 2.0 + 2.0 * (ca * cb + sa * sb - d * (sa + sb));
        if tmp >= DUBINS_ZERO {
            let p = tmp.max(0.0).sqrt();
            let theta = (ca + cb).atan2(d - sa - sb) - 2.0_f64.atan2(p);
            let t = mod2pi(alpha - theta);
            let q = mod2pi(beta - theta);
            return NormalizedDubinsPath {
                types: [
                    DubinsSegmentType::Right,
                    DubinsSegmentType::Straight,
                    DubinsSegmentType::Left,
                ],
                lengths: [t, p, q],
            };
        }
        NormalizedDubinsPath::invalid()
    }

    fn dubins_lsr(&self, d: f64, alpha: f64, beta: f64) -> NormalizedDubinsPath {
        let ca = alpha.cos();
        let sa = alpha.sin();
        let cb = beta.cos();
        let sb = beta.sin();
        let tmp = -2.0 + d * d + 2.0 * (ca * cb + sa * sb + d * (sa + sb));
        if tmp >= DUBINS_ZERO {
            let p = tmp.max(0.0).sqrt();
            let theta = (-ca - cb).atan2(d + sa + sb) - (-2.0_f64).atan2(p);
            let t = mod2pi(-alpha + theta);
            let q = mod2pi(-beta + theta);
            return NormalizedDubinsPath {
                types: [
                    DubinsSegmentType::Left,
                    DubinsSegmentType::Straight,
                    DubinsSegmentType::Right,
                ],
                lengths: [t, p, q],
            };
        }
        NormalizedDubinsPath::invalid()
    }

    fn dubins_rlr(&self, d: f64, alpha: f64, beta: f64) -> NormalizedDubinsPath {
        let ca = alpha.cos();
        let sa = alpha.sin();
        let cb = beta.cos();
        let sb = beta.sin();
        let tmp = 0.125 * (6.0 - d * d + 2.0 * (ca * cb + sa * sb + d * (sa - sb)));
        if tmp.abs() < 1.0 {
            let p = TWO_PI - tmp.acos();
            let theta = (ca - cb).atan2(d - sa + sb);
            let t = mod2pi(alpha - theta + 0.5 * p);
            let q = mod2pi(alpha - beta - t + p);
            return NormalizedDubinsPath {
                types: [
                    DubinsSegmentType::Right,
                    DubinsSegmentType::Left,
                    DubinsSegmentType::Right,
                ],
                lengths: [t, p, q],
            };
        }
        NormalizedDubinsPath::invalid()
    }

    fn dubins_lrl(&self, d: f64, alpha: f64, beta: f64) -> NormalizedDubinsPath {
        let ca = alpha.cos();
        let sa = alpha.sin();
        let cb = beta.cos();
        let sb = beta.sin();
        let tmp = 0.125 * (6.0 - d * d + 2.0 * (ca * cb + sa * sb - d * (sa - sb)));
        if tmp.abs() < 1.0 {
            let p = TWO_PI - tmp.acos();
            let theta = (-ca + cb).atan2(d + sa - sb);
            let t = mod2pi(-alpha + theta + 0.5 * p);
            let q = mod2pi(beta - alpha - t + p);
            return NormalizedDubinsPath {
                types: [
                    DubinsSegmentType::Left,
                    DubinsSegmentType::Right,
                    DubinsSegmentType::Left,
                ],
                lengths: [t, p, q],
            };
        }
        NormalizedDubinsPath::invalid()
    }
}

const TWO_PI: f64 = 2.0 * PI;
const DUBINS_EPS: f64 = 1e-6;
const DUBINS_ZERO: f64 = -1e-7;

fn mod2pi(x: f64) -> f64 {
    if x < 0.0 && x > DUBINS_ZERO {
        return 0.0;
    }
    let mut xm = x - TWO_PI * (x / TWO_PI).floor();
    if TWO_PI - xm < 0.5 * DUBINS_EPS {
        xm = 0.0;
    }
    xm
}
