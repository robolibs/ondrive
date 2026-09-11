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
