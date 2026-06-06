//! Python bindings via pyo3.
//!
//! Exposes Tracker + supporting types as ergonomic Python classes. Plain
//! data types are mirrored as `#[pyclass]`es with attribute-style accessors
//! and keyword-argument constructors so they feel native on the Python
//! side.

#![allow(clippy::too_many_arguments)]

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3::wrap_pyfunction;

use datapod::{Euler, Pose as DPose, Quaternion};

use crate::types::{
    ControllerConfig, ControllerStatus, GaussianMode as RsGaussianMode, Goal as RsGoal,
    Obstacle as RsObstacle, OutputUnits, Path as RsPath, RobotConstraints as RsConstraints,
    RobotState as RsRobotState, SteeringType, Velocity as RsVelocity,
    VelocityCommand as RsVelocityCommand, WorldConstraints as RsWorld,
};
use crate::{Tracker as RsTracker, TrackerKind};

// ---------------------------------------------------------------------------
// Enum wrappers.
// ---------------------------------------------------------------------------

fn kind_from_str(s: &str) -> PyResult<TrackerKind> {
    let lower = s.to_ascii_lowercase();
    Ok(match lower.as_str() {
        "pid" => TrackerKind::Pid,
        "carrot" => TrackerKind::Carrot,
        "pure_pursuit" | "pure-pursuit" | "purepursuit" => TrackerKind::PurePursuit,
        "stanley" => TrackerKind::Stanley,
        "lqr" => TrackerKind::Lqr,
        "mpc" => TrackerKind::Mpc,
        "mppi" => TrackerKind::Mppi,
        "mca" => TrackerKind::Mca,
        "soc" => TrackerKind::Soc,
        "dwa" => TrackerKind::Dwa,
        "teb" => TrackerKind::Teb,
        "flc" => TrackerKind::Flc,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown tracker kind: {other}"
            )));
        }
    })
}

fn kind_to_str(k: TrackerKind) -> &'static str {
    match k {
        TrackerKind::Pid => "pid",
        TrackerKind::Carrot => "carrot",
        TrackerKind::PurePursuit => "pure_pursuit",
        TrackerKind::Stanley => "stanley",
        TrackerKind::Lqr => "lqr",
        TrackerKind::Mpc => "mpc",
        TrackerKind::Mppi => "mppi",
        TrackerKind::Mca => "mca",
        TrackerKind::Soc => "soc",
        TrackerKind::Dwa => "dwa",
        TrackerKind::Teb => "teb",
        TrackerKind::Flc => "flc",
    }
}

fn steering_from_str(s: &str) -> PyResult<SteeringType> {
    match s.to_ascii_lowercase().as_str() {
        "differential" => Ok(SteeringType::Differential),
        "ackermann" => Ok(SteeringType::Ackermann),
        "holonomic" => Ok(SteeringType::Holonomic),
        "skid_steer" | "skid-steer" | "skidsteer" => Ok(SteeringType::SkidSteer),
        other => Err(PyValueError::new_err(format!(
            "unknown steering type: {other}"
        ))),
    }
}

fn steering_to_str(s: SteeringType) -> &'static str {
    match s {
        SteeringType::Differential => "differential",
        SteeringType::Ackermann => "ackermann",
        SteeringType::Holonomic => "holonomic",
        SteeringType::SkidSteer => "skid_steer",
    }
}

fn units_from_str(s: &str) -> PyResult<OutputUnits> {
    match s.to_ascii_lowercase().as_str() {
        "normalized" | "normalised" => Ok(OutputUnits::Normalized),
        "physical" => Ok(OutputUnits::Physical),
        other => Err(PyValueError::new_err(format!(
            "unknown output units: {other}"
        ))),
    }
}

fn units_to_str(u: OutputUnits) -> &'static str {
    match u {
        OutputUnits::Normalized => "normalized",
        OutputUnits::Physical => "physical",
    }
}

// ---------------------------------------------------------------------------
// Primitive helpers: Pose / Velocity tuples.
// ---------------------------------------------------------------------------

type PoseTuple = ((f64, f64, f64), f64); // ((x, y, z), yaw)
type VelocityTuple = (f64, f64, f64); // (linear, angular, lateral)

fn pose_from_tuple(t: PoseTuple) -> DPose {
    let ((x, y, z), yaw) = t;
    DPose {
        point: datapod::Point::new(x, y, z),
        rotation: Quaternion::from_euler(Euler::new(0.0, 0.0, yaw)),
    }
}

fn pose_to_tuple(p: DPose) -> PoseTuple {
    ((p.point.x, p.point.y, p.point.z), p.rotation.to_euler().yaw)
}

fn velocity_from_tuple(t: VelocityTuple) -> RsVelocity {
    RsVelocity {
        linear: t.0,
        angular: t.1,
        lateral: t.2,
    }
}

fn velocity_to_tuple(v: RsVelocity) -> VelocityTuple {
    (v.linear, v.angular, v.lateral)
}

// ---------------------------------------------------------------------------
// Config / state pyclasses.
// ---------------------------------------------------------------------------

#[pyclass(name = "ControllerConfig")]
#[derive(Clone)]
pub struct PyControllerConfig {
    inner: ControllerConfig,
}

#[pymethods]
impl PyControllerConfig {
    #[new]
    #[pyo3(signature = (
        output_units = "normalized",
        kp_linear = 1.0, ki_linear = 0.0, kd_linear = 0.0,
        kp_angular = 1.0, ki_angular = 0.0, kd_angular = 0.0,
        lookahead_distance = 1.0,
        k_cross_track = 1.0, k_heading = 1.0,
        allow_reverse = false,
        goal_tolerance = 0.1, angular_tolerance = 0.1,
    ))]
    fn new(
        output_units: &str,
        kp_linear: f64,
        ki_linear: f64,
        kd_linear: f64,
        kp_angular: f64,
        ki_angular: f64,
        kd_angular: f64,
        lookahead_distance: f64,
        k_cross_track: f64,
        k_heading: f64,
        allow_reverse: bool,
        goal_tolerance: f64,
        angular_tolerance: f64,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: ControllerConfig {
                output_units: units_from_str(output_units)?,
                kp_linear,
                ki_linear,
                kd_linear,
                kp_angular,
                ki_angular,
                kd_angular,
                lookahead_distance,
                k_cross_track,
                k_heading,
                allow_reverse,
                goal_tolerance,
                angular_tolerance,
            },
        })
    }

    #[staticmethod]
    fn default_() -> Self {
        Self {
            inner: ControllerConfig::default(),
        }
    }

    #[getter]
    fn output_units(&self) -> &'static str {
        units_to_str(self.inner.output_units)
    }
    #[setter]
    fn set_output_units(&mut self, v: &str) -> PyResult<()> {
        self.inner.output_units = units_from_str(v)?;
        Ok(())
    }

    #[getter]
    fn kp_linear(&self) -> f64 {
        self.inner.kp_linear
    }
    #[setter]
    fn set_kp_linear(&mut self, v: f64) {
        self.inner.kp_linear = v;
    }
    #[getter]
    fn ki_linear(&self) -> f64 {
        self.inner.ki_linear
    }
    #[setter]
    fn set_ki_linear(&mut self, v: f64) {
        self.inner.ki_linear = v;
    }
    #[getter]
    fn kd_linear(&self) -> f64 {
        self.inner.kd_linear
    }
    #[setter]
    fn set_kd_linear(&mut self, v: f64) {
        self.inner.kd_linear = v;
    }

    #[getter]
    fn kp_angular(&self) -> f64 {
        self.inner.kp_angular
    }
    #[setter]
    fn set_kp_angular(&mut self, v: f64) {
        self.inner.kp_angular = v;
    }
    #[getter]
    fn ki_angular(&self) -> f64 {
        self.inner.ki_angular
    }
    #[setter]
    fn set_ki_angular(&mut self, v: f64) {
        self.inner.ki_angular = v;
    }
    #[getter]
    fn kd_angular(&self) -> f64 {
        self.inner.kd_angular
    }
    #[setter]
    fn set_kd_angular(&mut self, v: f64) {
        self.inner.kd_angular = v;
    }

    #[getter]
    fn lookahead_distance(&self) -> f64 {
        self.inner.lookahead_distance
    }
    #[setter]
    fn set_lookahead_distance(&mut self, v: f64) {
        self.inner.lookahead_distance = v;
    }
    #[getter]
    fn k_cross_track(&self) -> f64 {
        self.inner.k_cross_track
    }
    #[setter]
    fn set_k_cross_track(&mut self, v: f64) {
        self.inner.k_cross_track = v;
    }
    #[getter]
    fn k_heading(&self) -> f64 {
        self.inner.k_heading
    }
    #[setter]
    fn set_k_heading(&mut self, v: f64) {
        self.inner.k_heading = v;
    }

    #[getter]
    fn allow_reverse(&self) -> bool {
        self.inner.allow_reverse
    }
    #[setter]
    fn set_allow_reverse(&mut self, v: bool) {
        self.inner.allow_reverse = v;
    }
    #[getter]
    fn goal_tolerance(&self) -> f64 {
        self.inner.goal_tolerance
    }
    #[setter]
    fn set_goal_tolerance(&mut self, v: f64) {
        self.inner.goal_tolerance = v;
    }
    #[getter]
    fn angular_tolerance(&self) -> f64 {
        self.inner.angular_tolerance
    }
    #[setter]
    fn set_angular_tolerance(&mut self, v: f64) {
        self.inner.angular_tolerance = v;
    }

    fn __repr__(&self) -> String {
        format!(
            "ControllerConfig(kp_linear={}, kp_angular={}, lookahead={}, goal_tol={}, units={})",
            self.inner.kp_linear,
            self.inner.kp_angular,
            self.inner.lookahead_distance,
            self.inner.goal_tolerance,
            units_to_str(self.inner.output_units),
        )
    }
}

#[pyclass(name = "RobotConstraints")]
#[derive(Clone)]
pub struct PyRobotConstraints {
    inner: RsConstraints,
}

#[pymethods]
impl PyRobotConstraints {
    #[new]
    #[pyo3(signature = (
        steering_type = "ackermann",
        wheelbase = 1.0, track_width = 0.5, wheel_radius = 0.1,
        max_linear_velocity = 1.0, min_linear_velocity = -0.5,
        max_angular_velocity = 1.0,
        max_linear_acceleration = 1.0, max_angular_acceleration = 1.0,
        max_steering_angle = 0.5, max_steering_rate = 1.0,
        min_turning_radius = 1.0,
        rear_wheelbase = 0.0, max_rear_steering_angle = 0.0,
        robot_width = 0.0, robot_length = 0.0,
    ))]
    fn new(
        steering_type: &str,
        wheelbase: f64,
        track_width: f64,
        wheel_radius: f64,
        max_linear_velocity: f64,
        min_linear_velocity: f64,
        max_angular_velocity: f64,
        max_linear_acceleration: f64,
        max_angular_acceleration: f64,
        max_steering_angle: f64,
        max_steering_rate: f64,
        min_turning_radius: f64,
        rear_wheelbase: f64,
        max_rear_steering_angle: f64,
        robot_width: f64,
        robot_length: f64,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: RsConstraints {
                steering_type: steering_from_str(steering_type)?,
                wheelbase,
                track_width,
                wheel_radius,
                max_linear_velocity,
                min_linear_velocity,
                max_angular_velocity,
                max_linear_acceleration,
                max_angular_acceleration,
                max_steering_angle,
                max_steering_rate,
                min_turning_radius,
                rear_wheelbase,
                max_rear_steering_angle,
                robot_width,
                robot_length,
            },
        })
    }

    #[staticmethod]
    fn default_() -> Self {
        Self {
            inner: RsConstraints::default(),
        }
    }

    #[getter]
    fn steering_type(&self) -> &'static str {
        steering_to_str(self.inner.steering_type)
    }
    #[setter]
    fn set_steering_type(&mut self, v: &str) -> PyResult<()> {
        self.inner.steering_type = steering_from_str(v)?;
        Ok(())
    }

    #[getter]
    fn wheelbase(&self) -> f64 {
        self.inner.wheelbase
    }
    #[setter]
    fn set_wheelbase(&mut self, v: f64) {
        self.inner.wheelbase = v;
    }
    #[getter]
    fn track_width(&self) -> f64 {
        self.inner.track_width
    }
    #[setter]
    fn set_track_width(&mut self, v: f64) {
        self.inner.track_width = v;
    }
    #[getter]
    fn wheel_radius(&self) -> f64 {
        self.inner.wheel_radius
    }
    #[setter]
    fn set_wheel_radius(&mut self, v: f64) {
        self.inner.wheel_radius = v;
    }

    #[getter]
    fn max_linear_velocity(&self) -> f64 {
        self.inner.max_linear_velocity
    }
    #[setter]
    fn set_max_linear_velocity(&mut self, v: f64) {
        self.inner.max_linear_velocity = v;
    }
    #[getter]
    fn min_linear_velocity(&self) -> f64 {
        self.inner.min_linear_velocity
    }
    #[setter]
    fn set_min_linear_velocity(&mut self, v: f64) {
        self.inner.min_linear_velocity = v;
    }
    #[getter]
    fn max_angular_velocity(&self) -> f64 {
        self.inner.max_angular_velocity
    }
    #[setter]
    fn set_max_angular_velocity(&mut self, v: f64) {
        self.inner.max_angular_velocity = v;
    }
    #[getter]
    fn max_linear_acceleration(&self) -> f64 {
        self.inner.max_linear_acceleration
    }
    #[setter]
    fn set_max_linear_acceleration(&mut self, v: f64) {
        self.inner.max_linear_acceleration = v;
    }
    #[getter]
    fn max_angular_acceleration(&self) -> f64 {
        self.inner.max_angular_acceleration
    }
    #[setter]
    fn set_max_angular_acceleration(&mut self, v: f64) {
        self.inner.max_angular_acceleration = v;
    }

    #[getter]
    fn max_steering_angle(&self) -> f64 {
        self.inner.max_steering_angle
    }
    #[setter]
    fn set_max_steering_angle(&mut self, v: f64) {
        self.inner.max_steering_angle = v;
    }
    #[getter]
    fn max_steering_rate(&self) -> f64 {
        self.inner.max_steering_rate
    }
    #[setter]
    fn set_max_steering_rate(&mut self, v: f64) {
        self.inner.max_steering_rate = v;
    }
    #[getter]
    fn min_turning_radius(&self) -> f64 {
        self.inner.min_turning_radius
    }
    #[setter]
    fn set_min_turning_radius(&mut self, v: f64) {
        self.inner.min_turning_radius = v;
    }

    #[getter]
    fn rear_wheelbase(&self) -> f64 {
        self.inner.rear_wheelbase
    }
    #[setter]
    fn set_rear_wheelbase(&mut self, v: f64) {
        self.inner.rear_wheelbase = v;
    }
    #[getter]
    fn max_rear_steering_angle(&self) -> f64 {
        self.inner.max_rear_steering_angle
    }
    #[setter]
    fn set_max_rear_steering_angle(&mut self, v: f64) {
        self.inner.max_rear_steering_angle = v;
    }

    #[getter]
    fn robot_width(&self) -> f64 {
        self.inner.robot_width
    }
    #[setter]
    fn set_robot_width(&mut self, v: f64) {
        self.inner.robot_width = v;
    }
    #[getter]
    fn robot_length(&self) -> f64 {
        self.inner.robot_length
    }
    #[setter]
    fn set_robot_length(&mut self, v: f64) {
        self.inner.robot_length = v;
    }

    fn __repr__(&self) -> String {
        format!(
            "RobotConstraints(steering={}, wheelbase={}, v_max={}, w_max={})",
            steering_to_str(self.inner.steering_type),
            self.inner.wheelbase,
            self.inner.max_linear_velocity,
            self.inner.max_angular_velocity,
        )
    }
}

#[pyclass(name = "RobotState")]
#[derive(Clone)]
pub struct PyRobotState {
    inner: RsRobotState,
}

#[pymethods]
impl PyRobotState {
    #[new]
    #[pyo3(signature = (
        pose = ((0.0, 0.0, 0.0), 0.0),
        velocity = (0.0, 0.0, 0.0),
        timestamp = 0.0,
        allow_reverse = false,
        turn_first = false,
        allow_move = true,
        has_trailer = false,
        trailer_pose = ((0.0, 0.0, 0.0), 0.0),
    ))]
    fn new(
        pose: PoseTuple,
        velocity: VelocityTuple,
        timestamp: f64,
        allow_reverse: bool,
        turn_first: bool,
        allow_move: bool,
        has_trailer: bool,
        trailer_pose: PoseTuple,
    ) -> Self {
        Self {
            inner: RsRobotState {
                pose: pose_from_tuple(pose),
                velocity: velocity_from_tuple(velocity),
                timestamp,
                allow_reverse,
                turn_first,
                allow_move,
                has_trailer,
                trailer_pose: pose_from_tuple(trailer_pose),
            },
        }
    }

    #[getter]
    fn pose(&self) -> PoseTuple {
        pose_to_tuple(self.inner.pose)
    }
    #[setter]
    fn set_pose(&mut self, v: PoseTuple) {
        self.inner.pose = pose_from_tuple(v);
    }
    #[getter]
    fn velocity(&self) -> VelocityTuple {
        velocity_to_tuple(self.inner.velocity)
    }
    #[setter]
    fn set_velocity(&mut self, v: VelocityTuple) {
        self.inner.velocity = velocity_from_tuple(v);
    }
    #[getter]
    fn timestamp(&self) -> f64 {
        self.inner.timestamp
    }
    #[setter]
    fn set_timestamp(&mut self, v: f64) {
        self.inner.timestamp = v;
    }
    #[getter]
    fn allow_reverse(&self) -> bool {
        self.inner.allow_reverse
    }
    #[setter]
    fn set_allow_reverse(&mut self, v: bool) {
        self.inner.allow_reverse = v;
    }
    #[getter]
    fn turn_first(&self) -> bool {
        self.inner.turn_first
    }
    #[setter]
    fn set_turn_first(&mut self, v: bool) {
        self.inner.turn_first = v;
    }
    #[getter]
    fn allow_move(&self) -> bool {
        self.inner.allow_move
    }
    #[setter]
    fn set_allow_move(&mut self, v: bool) {
        self.inner.allow_move = v;
    }

    fn __repr__(&self) -> String {
        let ((x, y, _), yaw) = pose_to_tuple(self.inner.pose);
        format!(
            "RobotState(x={:.3}, y={:.3}, yaw={:.3}, v={:.3})",
            x, y, yaw, self.inner.velocity.linear
        )
    }
}

#[pyclass(name = "Goal")]
#[derive(Clone)]
pub struct PyGoal {
    inner: RsGoal,
}

#[pymethods]
impl PyGoal {
    #[new]
    #[pyo3(signature = (
        target_pose,
        target_velocity = None,
        tolerance_position = 0.1,
        tolerance_orientation = 0.1,
    ))]
    fn new(
        target_pose: PoseTuple,
        target_velocity: Option<VelocityTuple>,
        tolerance_position: f64,
        tolerance_orientation: f64,
    ) -> Self {
        Self {
            inner: RsGoal {
                target_pose: pose_from_tuple(target_pose),
                target_velocity: target_velocity.map(velocity_from_tuple),
                tolerance_position,
                tolerance_orientation,
            },
        }
    }

    #[getter]
    fn target_pose(&self) -> PoseTuple {
        pose_to_tuple(self.inner.target_pose)
    }
    #[setter]
    fn set_target_pose(&mut self, v: PoseTuple) {
        self.inner.target_pose = pose_from_tuple(v);
    }
    #[getter]
    fn tolerance_position(&self) -> f64 {
        self.inner.tolerance_position
    }
    #[setter]
    fn set_tolerance_position(&mut self, v: f64) {
        self.inner.tolerance_position = v;
    }
    #[getter]
    fn tolerance_orientation(&self) -> f64 {
        self.inner.tolerance_orientation
    }
    #[setter]
    fn set_tolerance_orientation(&mut self, v: f64) {
        self.inner.tolerance_orientation = v;
    }
}

// ---------------------------------------------------------------------------
// Output / status.
// ---------------------------------------------------------------------------

#[pyclass(name = "VelocityCommand")]
#[derive(Clone)]
pub struct PyVelocityCommand {
    inner: RsVelocityCommand,
}

#[pymethods]
impl PyVelocityCommand {
    #[getter]
    fn valid(&self) -> bool {
        self.inner.valid
    }
    #[getter]
    fn linear_velocity(&self) -> f64 {
        self.inner.linear_velocity
    }
    #[getter]
    fn angular_velocity(&self) -> f64 {
        self.inner.angular_velocity
    }
    #[getter]
    fn lateral_velocity(&self) -> f64 {
        self.inner.lateral_velocity
    }
    #[getter]
    fn status_message(&self) -> String {
        self.inner.status_message.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "VelocityCommand(valid={}, v={:.3}, w={:.3})",
            self.inner.valid, self.inner.linear_velocity, self.inner.angular_velocity
        )
    }
}

#[pyclass(name = "ControllerStatus")]
#[derive(Clone)]
pub struct PyControllerStatus {
    inner: ControllerStatus,
}

#[pymethods]
impl PyControllerStatus {
    #[getter]
    fn goal_reached(&self) -> bool {
        self.inner.goal_reached
    }
    #[getter]
    fn distance_to_goal(&self) -> f64 {
        self.inner.distance_to_goal
    }
    #[getter]
    fn cross_track_error(&self) -> f64 {
        self.inner.cross_track_error
    }
    #[getter]
    fn heading_error(&self) -> f64 {
        self.inner.heading_error
    }
    #[getter]
    fn mode(&self) -> String {
        self.inner.mode.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "ControllerStatus(mode={}, dist={:.3}, cte={:.3}, goal_reached={})",
            self.inner.mode,
            self.inner.distance_to_goal,
            self.inner.cross_track_error,
            self.inner.goal_reached
        )
    }
}

// ---------------------------------------------------------------------------
// Path.
// ---------------------------------------------------------------------------

#[pyclass(name = "Path")]
#[derive(Clone)]
pub struct PyPath {
    inner: RsPath,
}

#[pymethods]
impl PyPath {
    #[new]
    #[pyo3(signature = (waypoints = Vec::new(), speeds = Vec::new(), is_closed = false))]
    fn new(waypoints: Vec<PoseTuple>, speeds: Vec<f64>, is_closed: bool) -> Self {
        let wps: Vec<DPose> = waypoints.into_iter().map(pose_from_tuple).collect();
        Self {
            inner: RsPath {
                waypoints: wps,
                speeds,
                is_closed,
            },
        }
    }

    #[pyo3(signature = (pose, speed=None))]
    fn add_waypoint(&mut self, pose: PoseTuple, speed: Option<f64>) {
        self.inner.waypoints.push(pose_from_tuple(pose));
        self.inner.speeds.push(speed.unwrap_or(0.0));
    }

    #[pyo3(signature = (x, y, yaw=None, speed=None))]
    fn add_waypoint_xy(&mut self, x: f64, y: f64, yaw: Option<f64>, speed: Option<f64>) {
        self.inner
            .waypoints
            .push(pose_from_tuple(((x, y, 0.0), yaw.unwrap_or(0.0))));
        self.inner.speeds.push(speed.unwrap_or(0.0));
    }

    fn clear(&mut self) {
        self.inner.waypoints.clear();
        self.inner.speeds.clear();
    }

    #[getter]
    fn is_closed(&self) -> bool {
        self.inner.is_closed
    }
    #[setter]
    fn set_is_closed(&mut self, v: bool) {
        self.inner.is_closed = v;
    }

    fn __len__(&self) -> usize {
        self.inner.waypoints.len()
    }

    fn waypoint(&self, idx: usize) -> PyResult<PoseTuple> {
        self.inner
            .waypoints
            .get(idx)
            .map(|p| pose_to_tuple(*p))
            .ok_or_else(|| PyValueError::new_err(format!("waypoint index {idx} out of range")))
    }

    fn waypoints(&self) -> Vec<PoseTuple> {
        self.inner
            .waypoints
            .iter()
            .copied()
            .map(pose_to_tuple)
            .collect()
    }

    fn smoothen(&mut self, max_segment_m: f64) {
        crate::smoothen_path(&mut self.inner, max_segment_m);
    }
}

// ---------------------------------------------------------------------------
// World / obstacles.
// ---------------------------------------------------------------------------

#[pyclass(name = "World")]
#[derive(Clone)]
pub struct PyWorld {
    inner: RsWorld,
}

#[pymethods]
impl PyWorld {
    #[new]
    fn new() -> Self {
        Self {
            inner: RsWorld::default(),
        }
    }

    #[pyo3(signature = (
        id = 0,
        x = 0.0, y = 0.0,
        radius = 0.3,
        std_x = 0.1, std_y = 0.1,
        horizon_steps = 10,
    ))]
    fn add_static_gaussian_obstacle(
        &mut self,
        id: u64,
        x: f64,
        y: f64,
        radius: f64,
        std_x: f64,
        std_y: f64,
        horizon_steps: usize,
    ) {
        let h = horizon_steps.max(1);
        self.inner.obstacles.push(RsObstacle {
            id,
            radius,
            modes: vec![RsGaussianMode {
                weight: 1.0,
                mean_x: vec![x; h],
                mean_y: vec![y; h],
                std_x: vec![std_x; h],
                std_y: vec![std_y; h],
            }],
        });
    }

    #[pyo3(signature = (id, radius, mean_x, mean_y, std_x = 0.1, std_y = 0.1))]
    fn add_trajectory_obstacle(
        &mut self,
        id: u64,
        radius: f64,
        mean_x: Vec<f64>,
        mean_y: Vec<f64>,
        std_x: f64,
        std_y: f64,
    ) -> PyResult<()> {
        if mean_x.len() != mean_y.len() || mean_x.is_empty() {
            return Err(PyValueError::new_err(
                "mean_x and mean_y must be non-empty and the same length",
            ));
        }
        let h = mean_x.len();
        self.inner.obstacles.push(RsObstacle {
            id,
            radius,
            modes: vec![RsGaussianMode {
                weight: 1.0,
                mean_x,
                mean_y,
                std_x: vec![std_x; h],
                std_y: vec![std_y; h],
            }],
        });
        Ok(())
    }

    fn clear(&mut self) {
        self.inner.obstacles.clear();
        self.inner.zones.clear();
    }

    fn obstacle_count(&self) -> usize {
        self.inner.obstacles.len()
    }
}

// ---------------------------------------------------------------------------
// Tracker.
// ---------------------------------------------------------------------------

#[pyclass(name = "Tracker", unsendable)]
pub struct PyTracker {
    inner: RsTracker,
}

#[pymethods]
impl PyTracker {
    #[new]
    fn new(kind: &str) -> PyResult<Self> {
        Ok(Self {
            inner: RsTracker::new(kind_from_str(kind)?),
        })
    }

    #[getter]
    fn kind(&self) -> &'static str {
        kind_to_str(self.inner.kind())
    }

    fn init(&mut self, constraints: PyRobotConstraints) {
        self.inner.init(constraints.inner);
    }

    fn set_config(&mut self, config: PyControllerConfig) {
        self.inner.set_config(config.inner);
    }

    fn get_config(&self) -> PyControllerConfig {
        PyControllerConfig {
            inner: self.inner.get_config(),
        }
    }

    fn set_goal(&mut self, goal: PyGoal) {
        self.inner.set_goal(goal.inner);
    }

    fn clear_goal(&mut self) {
        self.inner.clear_goal();
    }

    fn set_path(&mut self, path: PyPath) {
        self.inner.set_path(path.inner);
    }

    fn clear_path(&mut self) {
        self.inner.clear_path();
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn smoothen(&mut self, max_segment_m: f64) {
        self.inner.smoothen(max_segment_m);
    }

    #[pyo3(signature = (state, dt, world = None))]
    fn tick(&mut self, state: PyRobotState, dt: f64, world: Option<PyWorld>) -> PyVelocityCommand {
        let w = world.map(|w| w.inner);
        let cmd = self.inner.tick(&state.inner, dt, w.as_ref());
        PyVelocityCommand { inner: cmd }
    }

    fn emergency_stop(&mut self) -> PyVelocityCommand {
        PyVelocityCommand {
            inner: self.inner.emergency_stop(),
        }
    }

    fn get_status(&self) -> PyControllerStatus {
        PyControllerStatus {
            inner: self.inner.get_status(),
        }
    }

    fn is_goal_reached(&self) -> bool {
        self.inner.is_goal_reached()
    }

    fn current_target(&self) -> Option<(f64, f64, f64)> {
        self.inner.current_target().map(|p| (p.x, p.y, p.z))
    }

    fn __repr__(&self) -> String {
        format!("Tracker(kind={})", kind_to_str(self.inner.kind()))
    }
}

// ---------------------------------------------------------------------------
// Module-level helpers.
// ---------------------------------------------------------------------------

#[pyfunction]
fn smoothen_path(mut path: PyPath, max_segment_m: f64) -> PyPath {
    crate::smoothen_path(&mut path.inner, max_segment_m);
    path
}

#[pyfunction]
fn normalize_angle(a: f64) -> f64 {
    crate::normalize_angle(a)
}

#[pyfunction]
fn available_tracker_kinds() -> Vec<&'static str> {
    vec![
        "pid",
        "carrot",
        "pure_pursuit",
        "stanley",
        "lqr",
        "mpc",
        "mppi",
        "mca",
        "soc",
        "dwa",
        "teb",
        "flc",
    ]
}

#[pyfunction]
fn available_steering_types() -> Vec<&'static str> {
    vec!["differential", "ackermann", "holonomic", "skid_steer"]
}

// ---------------------------------------------------------------------------
// Module registration.
// ---------------------------------------------------------------------------

pub fn register_python_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    m.add_class::<PyControllerConfig>()?;
    m.add_class::<PyRobotConstraints>()?;
    m.add_class::<PyRobotState>()?;
    m.add_class::<PyGoal>()?;
    m.add_class::<PyPath>()?;
    m.add_class::<PyWorld>()?;
    m.add_class::<PyVelocityCommand>()?;
    m.add_class::<PyControllerStatus>()?;
    m.add_class::<PyTracker>()?;

    m.add_function(wrap_pyfunction!(smoothen_path, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_angle, m)?)?;
    m.add_function(wrap_pyfunction!(available_tracker_kinds, m)?)?;
    m.add_function(wrap_pyfunction!(available_steering_types, m)?)?;

    Ok(())
}

#[pymodule]
fn ondrive(m: &Bound<'_, PyModule>) -> PyResult<()> {
    register_python_module(m)
}
