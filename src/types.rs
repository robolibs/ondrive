#![allow(clippy::derivable_impls)]

use datapod::{Point, Pose};
use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SteeringType {
    Differential,
    Ackermann,
    Holonomic,
    SkidSteer,
}

impl Default for SteeringType {
    fn default() -> Self {
        SteeringType::Ackermann
    }
}

#[derive(Copy, Clone, Debug, Default, Serialize, Deserialize)]
pub struct Velocity {
    pub linear: f64,
    pub angular: f64,
    pub lateral: f64,
}

#[derive(Clone, Debug, Default)]
pub struct RobotState {
    pub pose: Pose,
    pub velocity: Velocity,
    pub timestamp: f64,
    pub allow_reverse: bool,
    pub turn_first: bool,
    pub allow_move: bool,
    pub has_trailer: bool,
    pub trailer_pose: Pose,
}

#[derive(Clone, Debug, Default)]
pub struct Path {
    pub waypoints: Vec<Pose>,
    pub speeds: Vec<f64>,
    pub is_closed: bool,
}

#[derive(Clone, Debug)]
pub struct RobotConstraints {
    pub steering_type: SteeringType,
    pub wheelbase: f64,
    pub track_width: f64,
    pub wheel_radius: f64,

    pub max_linear_velocity: f64,
    pub min_linear_velocity: f64,
    pub max_angular_velocity: f64,
    pub max_linear_acceleration: f64,
    pub max_angular_acceleration: f64,

    pub max_steering_angle: f64,
    pub max_steering_rate: f64,
    pub min_turning_radius: f64,

    pub rear_wheelbase: f64,
    pub max_rear_steering_angle: f64,

    pub robot_width: f64,
    pub robot_length: f64,
}

impl Default for RobotConstraints {
    fn default() -> Self {
        Self {
            steering_type: SteeringType::Ackermann,
            wheelbase: 1.0,
            track_width: 0.5,
            wheel_radius: 0.1,
            max_linear_velocity: 1.0,
            min_linear_velocity: -0.5,
            max_angular_velocity: 1.0,
            max_linear_acceleration: 1.0,
            max_angular_acceleration: 1.0,
            max_steering_angle: 0.5,
            max_steering_rate: 1.0,
            min_turning_radius: 1.0,
            rear_wheelbase: 0.0,
            max_rear_steering_angle: 0.0,
            robot_width: 0.0,
            robot_length: 0.0,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Zone {
    pub boundary: Vec<Point>,
    pub max_speed: f64,
}

#[derive(Clone, Debug, Default)]
pub struct GaussianMode {
    pub weight: f64,
    pub mean_x: Vec<f64>,
    pub mean_y: Vec<f64>,
    pub std_x: Vec<f64>,
    pub std_y: Vec<f64>,
}

#[derive(Clone, Debug, Default)]
pub struct Obstacle {
    pub id: u64,
    pub radius: f64,
    pub modes: Vec<GaussianMode>,
}

#[derive(Clone, Debug, Default)]
pub struct WorldConstraints {
    pub zones: Vec<Zone>,
    pub obstacles: Vec<Obstacle>,
}

#[derive(Clone, Debug, Default)]
pub struct Goal {
    pub target_pose: Pose,
    pub target_velocity: Option<Velocity>,
    pub tolerance_position: f64,
    pub tolerance_orientation: f64,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum OutputType {
    VelocityCommand,
}

impl Default for OutputType {
    fn default() -> Self {
        OutputType::VelocityCommand
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum OutputUnits {
    Normalized,
    Physical,
}

impl Default for OutputUnits {
    fn default() -> Self {
        OutputUnits::Physical
    }
}

#[derive(Clone, Debug, Default)]
pub struct VelocityCommand {
    pub valid: bool,
    pub status_message: String,
    pub output_type: OutputType,

    pub linear_velocity: f64,
    pub angular_velocity: f64,
    pub lateral_velocity: f64,
    /// Ackermann front-wheel steering angle (rad) consistent with
    /// `angular_velocity`; zero for other steering types.
    pub steering_angle: f64,
}

impl VelocityCommand {
    pub fn zero() -> Self {
        Self::default()
    }

    pub fn invalid(msg: impl Into<String>) -> Self {
        Self {
            valid: false,
            status_message: msg.into(),
            ..Self::default()
        }
    }
}

#[derive(Clone, Debug)]
pub struct ControllerConfig {
    pub output_units: OutputUnits,

    pub kp_linear: f64,
    pub ki_linear: f64,
    pub kd_linear: f64,

    pub kp_angular: f64,
    pub ki_angular: f64,
    pub kd_angular: f64,

    pub lookahead_distance: f64,
    pub k_cross_track: f64,
    pub k_heading: f64,

    pub allow_reverse: bool,
    pub goal_tolerance: f64,
    pub angular_tolerance: f64,
}

impl Default for ControllerConfig {
    fn default() -> Self {
        Self {
            output_units: OutputUnits::Physical,
            kp_linear: 1.0,
            ki_linear: 0.0,
            kd_linear: 0.0,
            kp_angular: 1.0,
            ki_angular: 0.0,
            kd_angular: 0.0,
            lookahead_distance: 1.0,
            k_cross_track: 1.0,
            k_heading: 1.0,
            allow_reverse: false,
            goal_tolerance: 0.1,
            angular_tolerance: 0.1,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ControllerStatus {
    pub goal_reached: bool,
    pub distance_to_goal: f64,
    pub cross_track_error: f64,
    pub heading_error: f64,
    pub mode: String,
}

/// Time-stamped trajectory: a pose, time and speed per sample. Times are
/// seconds from the trajectory start and must be increasing.
#[derive(Clone, Debug, Default)]
pub struct Trajectory {
    pub poses: Vec<Pose>,
    pub times: Vec<f64>,
    pub speeds: Vec<f64>,
}

/// Reference state interpolated on a trajectory at one instant.
#[derive(Clone, Copy, Debug, Default)]
pub struct TrajectorySample {
    pub pose: Pose,
    pub speed: f64,
    pub yaw_rate: f64,
    /// True when `t` lies past the last sample.
    pub finished: bool,
}

impl Trajectory {
    /// Build a trajectory from a path driven at constant `speed`.
    pub fn from_path(path: &Path, speed: f64) -> Self {
        let speed = speed.abs().max(1e-6);
        let mut times = Vec::with_capacity(path.waypoints.len());
        let mut t = 0.0;
        for (i, w) in path.waypoints.iter().enumerate() {
            if i > 0 {
                t += path.waypoints[i - 1].point.distance_to_2d(w.point) / speed;
            }
            times.push(t);
        }
        Self {
            poses: path.waypoints.clone(),
            times,
            speeds: vec![speed; path.waypoints.len()],
        }
    }

    /// Spatial view of the trajectory for geometric followers.
    pub fn to_path(&self) -> Path {
        Path {
            waypoints: self.poses.clone(),
            speeds: self.speeds.clone(),
            is_closed: false,
        }
    }

    pub fn duration(&self) -> f64 {
        self.times.last().copied().unwrap_or(0.0)
    }

    /// Interpolated reference at time `t` (clamped to the trajectory span).
    pub fn sample(&self, t: f64) -> TrajectorySample {
        let n = self.poses.len();
        if n == 0 {
            return TrajectorySample::default();
        }
        let yaw = |p: &Pose| p.rotation.to_euler().yaw;
        if n == 1 || t <= self.times[0] {
            return TrajectorySample {
                pose: self.poses[0],
                speed: self.speeds.first().copied().unwrap_or(0.0),
                yaw_rate: 0.0,
                finished: n == 1,
            };
        }
        let last = n - 1;
        if t >= self.times[last] {
            return TrajectorySample {
                pose: self.poses[last],
                speed: 0.0,
                yaw_rate: 0.0,
                finished: true,
            };
        }
        let i = match self.times.binary_search_by(|x| x.partial_cmp(&t).unwrap_or(std::cmp::Ordering::Equal)) {
            Ok(i) => i.min(last - 1),
            Err(i) => i.saturating_sub(1).min(last - 1),
        };
        let (t0, t1) = (self.times[i], self.times[i + 1]);
        let span = (t1 - t0).max(1e-9);
        let a = ((t - t0) / span).clamp(0.0, 1.0);
        let (p0, p1) = (&self.poses[i], &self.poses[i + 1]);
        let y0 = yaw(p0);
        let dyaw = {
            let mut d = yaw(p1) - y0;
            while d > std::f64::consts::PI {
                d -= 2.0 * std::f64::consts::PI;
            }
            while d < -std::f64::consts::PI {
                d += 2.0 * std::f64::consts::PI;
            }
            d
        };
        let pose = Pose {
            point: Point::new(
                p0.point.x + a * (p1.point.x - p0.point.x),
                p0.point.y + a * (p1.point.y - p0.point.y),
                p0.point.z + a * (p1.point.z - p0.point.z),
            ),
            rotation: datapod::Quaternion::from_euler(datapod::Euler::new(0.0, 0.0, y0 + a * dyaw)),
        };
        let speed = match (self.speeds.get(i), self.speeds.get(i + 1)) {
            (Some(s0), Some(s1)) => s0 + a * (s1 - s0),
            _ => p0.point.distance_to_2d(p1.point) / span,
        };
        TrajectorySample {
            pose,
            speed,
            yaw_rate: dyaw / span,
            finished: false,
        }
    }
}
