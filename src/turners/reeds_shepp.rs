use std::f64::consts::PI;

use super::geometry::{Point2Ext, points_equal};

use super::Pose2D;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReedsSheppSegmentType {
    LeftForward,
    StraightForward,
    RightForward,
    LeftBackward,
    StraightBackward,
    RightBackward,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReedsSheppSegment {
    pub r#type: ReedsSheppSegmentType,
    pub length: f64,
    pub forward: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReedsSheppPath {
    pub segments: Vec<ReedsSheppSegment>,
    pub waypoints: Vec<Pose2D>,
    /// Parallel to `waypoints`. `true` at index i means the machine is
    /// moving IN REVERSE at that waypoint (velocity opposite to heading).
    /// This is ground truth from the RS segment table, not inferred.
    pub waypoint_reverse: Vec<bool>,
    pub total_length: f64,
    pub name: String,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReedsShepp {
    turning_radius: f64,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum RsPathSegmentType {
    Nop,
    Left,
    Straight,
    Right,
}

#[derive(Clone, Copy)]
struct NormalizedReedsSheppPath {
    types: [RsPathSegmentType; 5],
    lengths: [f64; 5],
    total_length: f64,
}

impl NormalizedReedsSheppPath {
    fn new(types: [RsPathSegmentType; 5], lengths: [f64; 5]) -> Self {
        let total_length = lengths.iter().map(|l| l.abs()).sum();
        Self {
            types,
            lengths,
            total_length,
        }
    }

    fn is_valid(self) -> bool {
        self.total_length.is_finite() && self.total_length < f64::MAX * 0.5
    }
}

const REEDS_SHEPP_PATH_TYPES: [[RsPathSegmentType; 5]; 18] = [
    [
        RsPathSegmentType::Left,
        RsPathSegmentType::Right,
        RsPathSegmentType::Left,
        RsPathSegmentType::Nop,
        RsPathSegmentType::Nop,
    ],
    [
        RsPathSegmentType::Right,
        RsPathSegmentType::Left,
        RsPathSegmentType::Right,
        RsPathSegmentType::Nop,
        RsPathSegmentType::Nop,
    ],
    [
        RsPathSegmentType::Left,
        RsPathSegmentType::Right,
        RsPathSegmentType::Left,
        RsPathSegmentType::Right,
        RsPathSegmentType::Nop,
    ],
    [
        RsPathSegmentType::Right,
        RsPathSegmentType::Left,
        RsPathSegmentType::Right,
        RsPathSegmentType::Left,
        RsPathSegmentType::Nop,
    ],
    [
        RsPathSegmentType::Left,
        RsPathSegmentType::Right,
        RsPathSegmentType::Straight,
        RsPathSegmentType::Left,
        RsPathSegmentType::Nop,
    ],
    [
        RsPathSegmentType::Right,
        RsPathSegmentType::Left,
        RsPathSegmentType::Straight,
        RsPathSegmentType::Right,
        RsPathSegmentType::Nop,
    ],
    [
        RsPathSegmentType::Left,
        RsPathSegmentType::Straight,
        RsPathSegmentType::Right,
        RsPathSegmentType::Left,
        RsPathSegmentType::Nop,
    ],
    [
        RsPathSegmentType::Right,
        RsPathSegmentType::Straight,
        RsPathSegmentType::Left,
        RsPathSegmentType::Right,
        RsPathSegmentType::Nop,
    ],
    [
        RsPathSegmentType::Left,
        RsPathSegmentType::Right,
        RsPathSegmentType::Straight,
        RsPathSegmentType::Right,
        RsPathSegmentType::Nop,
    ],
    [
        RsPathSegmentType::Right,
        RsPathSegmentType::Left,
        RsPathSegmentType::Straight,
        RsPathSegmentType::Left,
        RsPathSegmentType::Nop,
    ],
    [
        RsPathSegmentType::Right,
        RsPathSegmentType::Straight,
        RsPathSegmentType::Right,
        RsPathSegmentType::Left,
        RsPathSegmentType::Nop,
    ],
    [
        RsPathSegmentType::Left,
        RsPathSegmentType::Straight,
        RsPathSegmentType::Left,
        RsPathSegmentType::Right,
        RsPathSegmentType::Nop,
    ],
    [
        RsPathSegmentType::Left,
        RsPathSegmentType::Straight,
        RsPathSegmentType::Right,
        RsPathSegmentType::Nop,
        RsPathSegmentType::Nop,
    ],
    [
        RsPathSegmentType::Right,
        RsPathSegmentType::Straight,
        RsPathSegmentType::Left,
        RsPathSegmentType::Nop,
        RsPathSegmentType::Nop,
    ],
    [
        RsPathSegmentType::Left,
        RsPathSegmentType::Straight,
        RsPathSegmentType::Left,
        RsPathSegmentType::Nop,
        RsPathSegmentType::Nop,
    ],
    [
        RsPathSegmentType::Right,
        RsPathSegmentType::Straight,
        RsPathSegmentType::Right,
        RsPathSegmentType::Nop,
        RsPathSegmentType::Nop,
    ],
    [
        RsPathSegmentType::Left,
        RsPathSegmentType::Right,
        RsPathSegmentType::Straight,
        RsPathSegmentType::Left,
        RsPathSegmentType::Right,
    ],
    [
        RsPathSegmentType::Right,
        RsPathSegmentType::Left,
        RsPathSegmentType::Straight,
        RsPathSegmentType::Right,
        RsPathSegmentType::Left,
    ],
];

impl ReedsShepp {
    pub const fn new(turning_radius: f64) -> Self {
        Self { turning_radius }
    }

    pub fn plan_path(&self, start: Pose2D, goal: Pose2D, step_size: f64) -> ReedsSheppPath {
        self.get_all_paths(start, goal, step_size.max(1e-3))
            .into_iter()
            .min_by(|a, b| {
                a.total_length
                    .partial_cmp(&b.total_length)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or_else(|| ReedsSheppPath {
                segments: Vec::new(),
                waypoints: vec![start, goal],
                waypoint_reverse: vec![false, false],
                total_length: 0.0,
                name: String::new(),
            })
    }

    pub fn get_all_paths(
        &self,
        start: Pose2D,
        goal: Pose2D,
        step_size: f64,
    ) -> Vec<ReedsSheppPath> {
        let q0 = [start.point.x(), start.point.y(), start.yaw];
        let q1 = [goal.point.x(), goal.point.y(), goal.yaw];
        let rs_paths = self.get_all_reeds_shepp_paths(q0, q1);
        let mut paths = Vec::new();

        for rs_path in rs_paths {
            if !rs_path.is_valid() {
                continue;
            }

            let mut segments = Vec::new();
            for i in 0..5 {
                if rs_path.types[i] == RsPathSegmentType::Nop {
                    break;
                }
                let forward = rs_path.lengths[i] >= 0.0;
                let r#type = match (rs_path.types[i], forward) {
                    (RsPathSegmentType::Left, true) => ReedsSheppSegmentType::LeftForward,
                    (RsPathSegmentType::Straight, true) => ReedsSheppSegmentType::StraightForward,
                    (RsPathSegmentType::Right, true) => ReedsSheppSegmentType::RightForward,
                    (RsPathSegmentType::Left, false) => ReedsSheppSegmentType::LeftBackward,
                    (RsPathSegmentType::Straight, false) => ReedsSheppSegmentType::StraightBackward,
                    (RsPathSegmentType::Right, false) => ReedsSheppSegmentType::RightBackward,
                    (RsPathSegmentType::Nop, _) => continue,
                };
                segments.push(ReedsSheppSegment {
                    r#type,
                    length: rs_path.lengths[i].abs(),
                    forward,
                });
            }

            let radius = self.turning_radius.max(1e-9);
            let total_length = rs_path.total_length * radius;

            // Cumulative arc-length boundaries of each segment + its
            // forward flag. Used to classify every sampled waypoint.
            let mut segment_bounds: Vec<(f64, bool)> = Vec::with_capacity(segments.len());
            let mut cumulative = 0.0;
            for segment in &segments {
                cumulative += segment.length * radius;
                segment_bounds.push((cumulative, segment.forward));
            }

            let find_forward = |seg_pos: f64| -> bool {
                segment_bounds
                    .iter()
                    .find(|(boundary, _)| seg_pos <= *boundary + 1e-9)
                    .map(|(_, forward)| *forward)
                    .unwrap_or_else(|| {
                        segment_bounds
                            .last()
                            .map(|(_, forward)| *forward)
                            .unwrap_or(true)
                    })
            };

            let mut waypoints = Vec::new();
            let mut waypoint_reverse = Vec::new();
            let mut seg = 0.0;
            while seg <= total_length {
                let qnew = self.interpolate(q0, rs_path, seg / radius);
                waypoints.push(Pose2D::new(qnew[0], qnew[1], qnew[2]));
                waypoint_reverse.push(!find_forward(seg));
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
                waypoint_reverse.push(!find_forward(total_length));
            }

            paths.push(ReedsSheppPath {
                segments,
                waypoints,
                waypoint_reverse,
                total_length,
                name: self.path_name(rs_path),
            });
        }

        paths
    }

    fn get_all_reeds_shepp_paths(
        &self,
        q0: [f64; 3],
        q1: [f64; 3],
    ) -> Vec<NormalizedReedsSheppPath> {
        let dx = q1[0] - q0[0];
        let dy = q1[1] - q0[1];
        let dth = q1[2] - q0[2];
        let c = q0[2].cos();
        let s = q0[2].sin();
        let mut x = c * dx + s * dy;
        let mut y = -s * dx + c * dy;
        let phi = dth;

        x /= self.turning_radius.max(1e-9);
        y /= self.turning_radius.max(1e-9);

        let mut paths = Vec::new();
        let mut t = 0.0;
        let mut u = 0.0;
        let mut v = 0.0;

        if lp_sp_lp(x, y, phi, &mut t, &mut u, &mut v) {
            paths.push(NormalizedReedsSheppPath::new(
                REEDS_SHEPP_PATH_TYPES[14],
                [t, u, v, 0.0, 0.0],
            ));
        }
        if lp_sp_lp(-x, y, -phi, &mut t, &mut u, &mut v) {
            paths.push(NormalizedReedsSheppPath::new(
                REEDS_SHEPP_PATH_TYPES[14],
                [-t, -u, -v, 0.0, 0.0],
            ));
        }
        if lp_sp_lp(x, -y, -phi, &mut t, &mut u, &mut v) {
            paths.push(NormalizedReedsSheppPath::new(
                REEDS_SHEPP_PATH_TYPES[15],
                [t, u, v, 0.0, 0.0],
            ));
        }
        if lp_sp_lp(-x, -y, phi, &mut t, &mut u, &mut v) {
            paths.push(NormalizedReedsSheppPath::new(
                REEDS_SHEPP_PATH_TYPES[15],
                [-t, -u, -v, 0.0, 0.0],
            ));
        }

        if lp_sp_rp(x, y, phi, &mut t, &mut u, &mut v) {
            paths.push(NormalizedReedsSheppPath::new(
                REEDS_SHEPP_PATH_TYPES[12],
                [t, u, v, 0.0, 0.0],
            ));
        }
        if lp_sp_rp(-x, y, -phi, &mut t, &mut u, &mut v) {
            paths.push(NormalizedReedsSheppPath::new(
                REEDS_SHEPP_PATH_TYPES[12],
                [-t, -u, -v, 0.0, 0.0],
            ));
        }
        if lp_sp_rp(x, -y, -phi, &mut t, &mut u, &mut v) {
            paths.push(NormalizedReedsSheppPath::new(
                REEDS_SHEPP_PATH_TYPES[13],
                [t, u, v, 0.0, 0.0],
            ));
        }
        if lp_sp_rp(-x, -y, phi, &mut t, &mut u, &mut v) {
            paths.push(NormalizedReedsSheppPath::new(
                REEDS_SHEPP_PATH_TYPES[13],
                [-t, -u, -v, 0.0, 0.0],
            ));
        }

        if lp_rm_l(x, y, phi, &mut t, &mut u, &mut v) {
            paths.push(NormalizedReedsSheppPath::new(
                REEDS_SHEPP_PATH_TYPES[0],
                [t, u, v, 0.0, 0.0],
            ));
        }
        if lp_rm_l(-x, y, -phi, &mut t, &mut u, &mut v) {
            paths.push(NormalizedReedsSheppPath::new(
                REEDS_SHEPP_PATH_TYPES[0],
                [-t, -u, -v, 0.0, 0.0],
            ));
        }
        if lp_rm_l(x, -y, -phi, &mut t, &mut u, &mut v) {
            paths.push(NormalizedReedsSheppPath::new(
                REEDS_SHEPP_PATH_TYPES[1],
                [t, u, v, 0.0, 0.0],
            ));
        }
        if lp_rm_l(-x, -y, phi, &mut t, &mut u, &mut v) {
            paths.push(NormalizedReedsSheppPath::new(
                REEDS_SHEPP_PATH_TYPES[1],
                [-t, -u, -v, 0.0, 0.0],
            ));
        }

        let xb = x * phi.cos() + y * phi.sin();
        let yb = x * phi.sin() - y * phi.cos();
        if lp_rm_l(xb, yb, phi, &mut t, &mut u, &mut v) {
            paths.push(NormalizedReedsSheppPath::new(
                REEDS_SHEPP_PATH_TYPES[0],
                [v, u, t, 0.0, 0.0],
            ));
        }
        if lp_rm_l(-xb, yb, -phi, &mut t, &mut u, &mut v) {
            paths.push(NormalizedReedsSheppPath::new(
                REEDS_SHEPP_PATH_TYPES[0],
                [-v, -u, -t, 0.0, 0.0],
            ));
        }
        if lp_rm_l(xb, -yb, -phi, &mut t, &mut u, &mut v) {
            paths.push(NormalizedReedsSheppPath::new(
                REEDS_SHEPP_PATH_TYPES[1],
                [v, u, t, 0.0, 0.0],
            ));
        }
        if lp_rm_l(-xb, -yb, phi, &mut t, &mut u, &mut v) {
            paths.push(NormalizedReedsSheppPath::new(
                REEDS_SHEPP_PATH_TYPES[1],
                [-v, -u, -t, 0.0, 0.0],
            ));
        }

        if lp_rup_lum_rm(x, y, phi, &mut t, &mut u, &mut v) {
            paths.push(NormalizedReedsSheppPath::new(
                REEDS_SHEPP_PATH_TYPES[2],
                [t, u, -u, v, 0.0],
            ));
        }
        if lp_rup_lum_rm(-x, y, -phi, &mut t, &mut u, &mut v) {
            paths.push(NormalizedReedsSheppPath::new(
                REEDS_SHEPP_PATH_TYPES[2],
                [-t, -u, u, -v, 0.0],
            ));
        }
        if lp_rup_lum_rm(x, -y, -phi, &mut t, &mut u, &mut v) {
            paths.push(NormalizedReedsSheppPath::new(
                REEDS_SHEPP_PATH_TYPES[3],
                [t, u, -u, v, 0.0],
            ));
        }
        if lp_rup_lum_rm(-x, -y, phi, &mut t, &mut u, &mut v) {
            paths.push(NormalizedReedsSheppPath::new(
                REEDS_SHEPP_PATH_TYPES[3],
                [-t, -u, u, -v, 0.0],
            ));
        }

        if lp_rum_lum_rp(x, y, phi, &mut t, &mut u, &mut v) {
            paths.push(NormalizedReedsSheppPath::new(
                REEDS_SHEPP_PATH_TYPES[2],
                [t, u, u, v, 0.0],
            ));
        }
        if lp_rum_lum_rp(-x, y, -phi, &mut t, &mut u, &mut v) {
            paths.push(NormalizedReedsSheppPath::new(
                REEDS_SHEPP_PATH_TYPES[2],
                [-t, -u, -u, -v, 0.0],
            ));
        }
        if lp_rum_lum_rp(x, -y, -phi, &mut t, &mut u, &mut v) {
            paths.push(NormalizedReedsSheppPath::new(
                REEDS_SHEPP_PATH_TYPES[3],
                [t, u, u, v, 0.0],
            ));
        }
        if lp_rum_lum_rp(-x, -y, phi, &mut t, &mut u, &mut v) {
            paths.push(NormalizedReedsSheppPath::new(
                REEDS_SHEPP_PATH_TYPES[3],
                [-t, -u, -u, -v, 0.0],
            ));
        }

        if lp_rm_sm_lm(x, y, phi, &mut t, &mut u, &mut v) {
            paths.push(NormalizedReedsSheppPath::new(
                REEDS_SHEPP_PATH_TYPES[4],
                [t, -0.5 * PI, u, v, 0.0],
            ));
        }
        if lp_rm_sm_lm(-x, y, -phi, &mut t, &mut u, &mut v) {
            paths.push(NormalizedReedsSheppPath::new(
                REEDS_SHEPP_PATH_TYPES[4],
                [-t, 0.5 * PI, -u, -v, 0.0],
            ));
        }
        if lp_rm_sm_lm(x, -y, -phi, &mut t, &mut u, &mut v) {
            paths.push(NormalizedReedsSheppPath::new(
                REEDS_SHEPP_PATH_TYPES[5],
                [t, -0.5 * PI, u, v, 0.0],
            ));
        }
        if lp_rm_sm_lm(-x, -y, phi, &mut t, &mut u, &mut v) {
            paths.push(NormalizedReedsSheppPath::new(
                REEDS_SHEPP_PATH_TYPES[5],
                [-t, 0.5 * PI, -u, -v, 0.0],
            ));
        }

        if lp_rm_sm_rm(x, y, phi, &mut t, &mut u, &mut v) {
            paths.push(NormalizedReedsSheppPath::new(
                REEDS_SHEPP_PATH_TYPES[8],
                [t, -0.5 * PI, u, v, 0.0],
            ));
        }
        if lp_rm_sm_rm(-x, y, -phi, &mut t, &mut u, &mut v) {
            paths.push(NormalizedReedsSheppPath::new(
                REEDS_SHEPP_PATH_TYPES[8],
                [-t, 0.5 * PI, -u, -v, 0.0],
            ));
        }
        if lp_rm_sm_rm(x, -y, -phi, &mut t, &mut u, &mut v) {
            paths.push(NormalizedReedsSheppPath::new(
                REEDS_SHEPP_PATH_TYPES[9],
                [t, -0.5 * PI, u, v, 0.0],
            ));
        }
        if lp_rm_sm_rm(-x, -y, phi, &mut t, &mut u, &mut v) {
            paths.push(NormalizedReedsSheppPath::new(
                REEDS_SHEPP_PATH_TYPES[9],
                [-t, 0.5 * PI, -u, -v, 0.0],
            ));
        }

        if lp_rm_sm_lm(xb, yb, phi, &mut t, &mut u, &mut v) {
            paths.push(NormalizedReedsSheppPath::new(
                REEDS_SHEPP_PATH_TYPES[6],
                [v, u, -0.5 * PI, t, 0.0],
            ));
        }
        if lp_rm_sm_lm(-xb, yb, -phi, &mut t, &mut u, &mut v) {
            paths.push(NormalizedReedsSheppPath::new(
                REEDS_SHEPP_PATH_TYPES[6],
                [-v, -u, 0.5 * PI, -t, 0.0],
            ));
        }
        if lp_rm_sm_lm(xb, -yb, -phi, &mut t, &mut u, &mut v) {
            paths.push(NormalizedReedsSheppPath::new(
                REEDS_SHEPP_PATH_TYPES[7],
                [v, u, -0.5 * PI, t, 0.0],
            ));
        }
        if lp_rm_sm_lm(-xb, -yb, phi, &mut t, &mut u, &mut v) {
            paths.push(NormalizedReedsSheppPath::new(
                REEDS_SHEPP_PATH_TYPES[7],
                [-v, -u, 0.5 * PI, -t, 0.0],
            ));
        }

        if lp_rm_sm_rm(xb, yb, phi, &mut t, &mut u, &mut v) {
            paths.push(NormalizedReedsSheppPath::new(
                REEDS_SHEPP_PATH_TYPES[10],
                [v, u, -0.5 * PI, t, 0.0],
            ));
        }
        if lp_rm_sm_rm(-xb, yb, -phi, &mut t, &mut u, &mut v) {
            paths.push(NormalizedReedsSheppPath::new(
                REEDS_SHEPP_PATH_TYPES[10],
                [-v, -u, 0.5 * PI, -t, 0.0],
            ));
        }
        if lp_rm_sm_rm(xb, -yb, -phi, &mut t, &mut u, &mut v) {
            paths.push(NormalizedReedsSheppPath::new(
                REEDS_SHEPP_PATH_TYPES[11],
                [v, u, -0.5 * PI, t, 0.0],
            ));
        }
        if lp_rm_sm_rm(-xb, -yb, phi, &mut t, &mut u, &mut v) {
            paths.push(NormalizedReedsSheppPath::new(
                REEDS_SHEPP_PATH_TYPES[11],
                [-v, -u, 0.5 * PI, -t, 0.0],
            ));
        }

        if lp_rm_s_lm_rp(x, y, phi, &mut t, &mut u, &mut v) {
            paths.push(NormalizedReedsSheppPath::new(
                REEDS_SHEPP_PATH_TYPES[16],
                [t, -0.5 * PI, u, -0.5 * PI, v],
            ));
        }
        if lp_rm_s_lm_rp(-x, y, -phi, &mut t, &mut u, &mut v) {
            paths.push(NormalizedReedsSheppPath::new(
                REEDS_SHEPP_PATH_TYPES[16],
                [-t, 0.5 * PI, -u, 0.5 * PI, -v],
            ));
        }
        if lp_rm_s_lm_rp(x, -y, -phi, &mut t, &mut u, &mut v) {
            paths.push(NormalizedReedsSheppPath::new(
                REEDS_SHEPP_PATH_TYPES[17],
                [t, -0.5 * PI, u, -0.5 * PI, v],
            ));
        }
        if lp_rm_s_lm_rp(-x, -y, phi, &mut t, &mut u, &mut v) {
            paths.push(NormalizedReedsSheppPath::new(
                REEDS_SHEPP_PATH_TYPES[17],
                [-t, 0.5 * PI, -u, 0.5 * PI, -v],
            ));
        }

        paths
    }

    fn interpolate(&self, q0: [f64; 3], path: NormalizedReedsSheppPath, mut seg: f64) -> [f64; 3] {
        seg = seg.clamp(0.0, path.total_length);

        let mut s = [0.0, 0.0, q0[2]];
        for i in 0..5 {
            if seg <= 0.0 {
                break;
            }
            let v = if path.lengths[i] < 0.0 {
                let step = (-seg).max(path.lengths[i]);
                seg += step;
                step
            } else {
                let step = seg.min(path.lengths[i]);
                seg -= step;
                step
            };

            let phi = s[2];
            match path.types[i] {
                RsPathSegmentType::Left => {
                    s[0] += (phi + v).sin() - phi.sin();
                    s[1] += -(phi + v).cos() + phi.cos();
                    s[2] = phi + v;
                }
                RsPathSegmentType::Right => {
                    s[0] += -(phi - v).sin() + phi.sin();
                    s[1] += (phi - v).cos() - phi.cos();
                    s[2] = phi - v;
                }
                RsPathSegmentType::Straight => {
                    s[0] += v * phi.cos();
                    s[1] += v * phi.sin();
                }
                RsPathSegmentType::Nop => {}
            }
        }

        s[0] = s[0] * self.turning_radius.max(1e-9) + q0[0];
        s[1] = s[1] * self.turning_radius.max(1e-9) + q0[1];
        s
    }

    fn path_name(&self, path: NormalizedReedsSheppPath) -> String {
        let mut name = String::new();
        for i in 0..5 {
            if path.types[i] == RsPathSegmentType::Nop {
                break;
            }
            let ch = match path.types[i] {
                RsPathSegmentType::Left => {
                    if path.lengths[i] >= 0.0 {
                        'L'
                    } else {
                        'l'
                    }
                }
                RsPathSegmentType::Straight => {
                    if path.lengths[i] >= 0.0 {
                        'S'
                    } else {
                        's'
                    }
                }
                RsPathSegmentType::Right => {
                    if path.lengths[i] >= 0.0 {
                        'R'
                    } else {
                        'r'
                    }
                }
                RsPathSegmentType::Nop => break,
            };
            name.push(ch);
        }
        name
    }
}

const TWO_PI: f64 = 2.0 * PI;
const ZERO: f64 = 10.0 * f64::EPSILON;

fn mod2pi(x: f64) -> f64 {
    let mut v = x % TWO_PI;
    if v < -PI {
        v += TWO_PI;
    } else if v > PI {
        v -= TWO_PI;
    }
    v
}

fn polar(x: f64, y: f64) -> (f64, f64) {
    ((x * x + y * y).sqrt(), y.atan2(x))
}

fn tau_omega(u: f64, v: f64, xi: f64, eta: f64, phi: f64) -> (f64, f64) {
    let delta = mod2pi(u - v);
    let a = u.sin() - delta.sin();
    let b = u.cos() - delta.cos() - 1.0;
    let t1 = (eta * a - xi * b).atan2(xi * a + eta * b);
    let t2 = 2.0 * (delta.cos() - v.cos() - u.cos()) + 3.0;
    let tau = if t2 < 0.0 {
        mod2pi(t1 + PI)
    } else {
        mod2pi(t1)
    };
    let omega = mod2pi(tau - u + v - phi);
    (tau, omega)
}

fn lp_sp_lp(x: f64, y: f64, phi: f64, t: &mut f64, u: &mut f64, v: &mut f64) -> bool {
    let (radius, theta) = polar(x - phi.sin(), y - 1.0 + phi.cos());
    *u = radius;
    *t = theta;
    if *t >= -ZERO {
        *v = mod2pi(phi - *t);
        return *v >= -ZERO;
    }
    false
}

fn lp_sp_rp(x: f64, y: f64, phi: f64, t: &mut f64, u: &mut f64, v: &mut f64) -> bool {
    let (mut radius, theta) = polar(x + phi.sin(), y - 1.0 - phi.cos());
    radius *= radius;
    if radius >= 4.0 {
        *u = (radius - 4.0).sqrt();
        let theta2 = 2.0_f64.atan2(*u);
        *t = mod2pi(theta + theta2);
        *v = mod2pi(*t - phi);
        return *t >= -ZERO && *v >= -ZERO;
    }
    false
}

fn lp_rm_l(x: f64, y: f64, phi: f64, t: &mut f64, u: &mut f64, v: &mut f64) -> bool {
    let xi = x - phi.sin();
    let eta = y - 1.0 + phi.cos();
    let (radius, theta) = polar(xi, eta);
    if radius <= 4.0 {
        *u = -2.0 * (0.25 * radius).asin();
        *t = mod2pi(theta + 0.5 * *u + PI);
        *v = mod2pi(phi - *t + *u);
        return *t >= -ZERO && *u <= ZERO;
    }
    false
}

fn lp_rup_lum_rm(x: f64, y: f64, phi: f64, t: &mut f64, u: &mut f64, v: &mut f64) -> bool {
    let xi = x + phi.sin();
    let eta = y - 1.0 - phi.cos();
    let rho = 0.25 * (2.0 + (xi * xi + eta * eta).sqrt());
    if rho <= 1.0 {
        *u = rho.acos();
        (*t, *v) = tau_omega(*u, -*u, xi, eta, phi);
        return *t >= -ZERO && *v <= ZERO;
    }
    false
}

fn lp_rum_lum_rp(x: f64, y: f64, phi: f64, t: &mut f64, u: &mut f64, v: &mut f64) -> bool {
    let xi = x + phi.sin();
    let eta = y - 1.0 - phi.cos();
    let rho = (20.0 - xi * xi - eta * eta) / 16.0;
    if (0.0..=1.0).contains(&rho) {
        *u = -rho.acos();
        if *u >= -0.5 * PI {
            (*t, *v) = tau_omega(*u, *u, xi, eta, phi);
            return *t >= -ZERO && *v >= -ZERO;
        }
    }
    false
}

fn lp_rm_sm_lm(x: f64, y: f64, phi: f64, t: &mut f64, u: &mut f64, v: &mut f64) -> bool {
    let xi = x - phi.sin();
    let eta = y - 1.0 + phi.cos();
    let (rho, theta) = polar(xi, eta);
    if rho >= 2.0 {
        let r = (rho * rho - 4.0).sqrt();
        *u = 2.0 - r;
        *t = mod2pi(theta + r.atan2(-2.0));
        *v = mod2pi(phi - 0.5 * PI - *t);
        return *t >= -ZERO && *u <= ZERO && *v <= ZERO;
    }
    false
}

fn lp_rm_sm_rm(x: f64, y: f64, phi: f64, t: &mut f64, u: &mut f64, v: &mut f64) -> bool {
    let xi = x + phi.sin();
    let eta = y - 1.0 - phi.cos();
    let (rho, theta) = polar(-eta, xi);
    if rho >= 2.0 {
        *t = theta;
        *u = 2.0 - rho;
        *v = mod2pi(*t + 0.5 * PI - phi);
        return *t >= -ZERO && *u <= ZERO && *v <= ZERO;
    }
    false
}

fn lp_rm_s_lm_rp(x: f64, y: f64, phi: f64, t: &mut f64, u: &mut f64, v: &mut f64) -> bool {
    let xi = x + phi.sin();
    let eta = y - 1.0 - phi.cos();
    let (rho, _) = polar(xi, eta);
    if rho >= 2.0 {
        *u = 4.0 - (rho * rho - 4.0).sqrt();
        if *u <= ZERO {
            *t = mod2pi((((4.0 - *u) * xi) - 2.0 * eta).atan2((-2.0 * xi) + ((*u - 4.0) * eta)));
            *v = mod2pi(*t - phi);
            return *t >= -ZERO && *v >= -ZERO;
        }
    }
    false
}
