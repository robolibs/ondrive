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
    ControllerConfig, ControllerStatus, DebugLevel, GaussianMode as RsGaussianMode, Goal as RsGoal,
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
        "regulated_pursuit" | "rpp" => TrackerKind::RegulatedPursuit,
        "pose_reach" => TrackerKind::PoseReach,
        "ilqr" => TrackerKind::Ilqr,
        "pose_regulator" => TrackerKind::PoseRegulator,
        "vector_pursuit" => TrackerKind::VectorPursuit,
        "kanayama" => TrackerKind::Kanayama,
        "apf" => TrackerKind::Apf,
        "ilc" => TrackerKind::Ilc,
        "slip_adaptive" => TrackerKind::SlipAdaptive,
        "velocity_obstacle" => TrackerKind::VelocityObstacle,
        "frenet" => TrackerKind::Frenet,
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
        TrackerKind::RegulatedPursuit => "regulated_pursuit",
        TrackerKind::PoseReach => "pose_reach",
        TrackerKind::Ilqr => "ilqr",
        TrackerKind::PoseRegulator => "pose_regulator",
        TrackerKind::VectorPursuit => "vector_pursuit",
        TrackerKind::Kanayama => "kanayama",
        TrackerKind::Apf => "apf",
        TrackerKind::Ilc => "ilc",
        TrackerKind::SlipAdaptive => "slip_adaptive",
        TrackerKind::VelocityObstacle => "velocity_obstacle",
        TrackerKind::Frenet => "frenet",
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
        output_units = "physical",
        kp_linear = 1.0, ki_linear = 0.0, kd_linear = 0.0,
        kp_angular = 1.0, ki_angular = 0.0, kd_angular = 0.0,
        lookahead_distance = 1.0, lookahead_time = 0.3,
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
        lookahead_time: f64,
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
                lookahead_time,
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
    fn lookahead_time(&self) -> f64 {
        self.inner.lookahead_time
    }
    #[setter]
    fn set_lookahead_time(&mut self, v: f64) {
        self.inner.lookahead_time = v;
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
        wheelbase = 1.0, track_width = 0.0,
        max_linear_velocity = 1.0, min_linear_velocity = -0.5,
        max_angular_velocity = 1.0,
        max_linear_acceleration = 1.0, max_angular_acceleration = 1.0,
        max_steering_angle = 0.5, max_steering_rate = 0.0,
        min_turning_radius = 1.0,
        rear_wheelbase = 0.0,
        robot_width = 0.0, robot_length = 0.0,
    ))]
    fn new(
        steering_type: &str,
        wheelbase: f64,
        track_width: f64,
        max_linear_velocity: f64,
        min_linear_velocity: f64,
        max_angular_velocity: f64,
        max_linear_acceleration: f64,
        max_angular_acceleration: f64,
        max_steering_angle: f64,
        max_steering_rate: f64,
        min_turning_radius: f64,
        rear_wheelbase: f64,
        robot_width: f64,
        robot_length: f64,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: RsConstraints {
                footprint: Default::default(),
                steering_type: steering_from_str(steering_type)?,
                wheelbase,
                track_width,
                max_linear_velocity,
                min_linear_velocity,
                max_angular_velocity,
                max_linear_acceleration,
                max_angular_acceleration,
                max_steering_angle,
                max_steering_rate,
                min_turning_radius,
                rear_wheelbase,
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

    /// Robot outline: a disc of `radius`, or a polygon of `(x, y)` body-frame
    /// points (x forward, y left).
    #[pyo3(signature = (radius = None, points = None))]
    fn set_footprint(&mut self, radius: Option<f64>, points: Option<Vec<(f64, f64)>>) -> PyResult<()> {
        self.inner.footprint = match (radius, points) {
            (Some(radius), None) => crate::types::Footprint::Disc { radius },
            (None, Some(points)) => crate::types::Footprint::Polygon { points },
            _ => return Err(PyValueError::new_err("give either radius or points")),
        };
        Ok(())
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
        velocity = None,
        timestamp = 0.0,
        allow_reverse = false,
        turn_first = false,
        allow_move = true,
    ))]
    fn new(
        pose: PoseTuple,
        velocity: Option<VelocityTuple>,
        timestamp: f64,
        allow_reverse: bool,
        turn_first: bool,
        allow_move: bool,
    ) -> Self {
        Self {
            inner: RsRobotState {
                pose: pose_from_tuple(pose),
                velocity: velocity.map(velocity_from_tuple).unwrap_or_default(),
                velocity_valid: velocity.is_some(),
                timestamp,
                allow_reverse,
                turn_first,
                allow_move,
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
    fn velocity(&self) -> Option<VelocityTuple> {
        self.inner.velocity_valid.then(|| velocity_to_tuple(self.inner.velocity))
    }
    #[setter]
    fn set_velocity(&mut self, v: Option<VelocityTuple>) {
        self.inner.velocity = v.map(velocity_from_tuple).unwrap_or_default();
        self.inner.velocity_valid = v.is_some();
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
        tolerance_position = 0.0,
        tolerance_orientation = 0.0,
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
    fn steering_angle(&self) -> f64 {
        self.inner.steering_angle
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

fn ondrive_traj(poses: Vec<datapod::Pose>, times: Vec<f64>, speeds: Vec<f64>) -> crate::types::Trajectory {
    crate::types::Trajectory { poses, times, speeds }
}

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
            dt: 0.0,
            modes: vec![RsGaussianMode {
                weight: 1.0,
                mean_x: vec![x; h],
                mean_y: vec![y; h],
                std_x: vec![std_x; h],
                std_y: vec![std_y; h],
            }],
        });
    }

    /// Install an occupancy grid: row-major `width * height` booleans whose
    /// cell (0, 0) starts at `(origin_x, origin_y)`.
    fn set_grid(&mut self, origin_x: f64, origin_y: f64, resolution: f64, width: usize, height: usize, occupied: Vec<bool>) -> PyResult<()> {
        if resolution <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err("resolution must be positive"));
        }
        self.inner.grid = Some(crate::types::OccupancyGrid::new(origin_x, origin_y, resolution, width, height, occupied));
        Ok(())
    }

    fn clear_grid(&mut self) {
        self.inner.grid = None;
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
            dt: 0.0,
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

    /// Add an obstacle predicted as a Gaussian mixture. Each mode is
    /// `(weight, mean_x, mean_y, std_x, std_y)` with one entry per horizon
    /// step, step 0 being now; `std_x` / `std_y` may also be single numbers.
    /// `dt` is the time between steps (0: the controller's own step).
    #[pyo3(signature = (id, radius, modes, dt = 0.0))]
    fn add_obstacle(&mut self, id: u64, radius: f64, modes: Vec<(f64, Vec<f64>, Vec<f64>, Spread, Spread)>, dt: f64) -> PyResult<()> {
        let mut out = Vec::with_capacity(modes.len());
        for (weight, mean_x, mean_y, std_x, std_y) in modes {
            let h = mean_x.len();
            if h == 0 || mean_y.len() != h {
                return Err(PyValueError::new_err("each mode needs non-empty mean_x and mean_y of the same length"));
            }
            out.push(RsGaussianMode { weight, mean_x, mean_y, std_x: std_x.steps(h)?, std_y: std_y.steps(h)? });
        }
        self.inner.obstacles.push(RsObstacle { id, radius, dt, modes: out });
        Ok(())
    }

    /// Add a speed-limit zone: inside the polygon `boundary` the tracker
    /// caps the speed to `max_speed`.
    fn add_zone(&mut self, boundary: Vec<(f64, f64)>, max_speed: f64) {
        let boundary = boundary.into_iter().map(|(x, y)| datapod::Point::new(x, y, 0.0)).collect();
        self.inner.zones.push(crate::types::Zone { boundary, max_speed });
    }

    fn obstacle_count(&self) -> usize {
        self.inner.obstacles.len()
    }
}

// ---------------------------------------------------------------------------
// Tracker.
// ---------------------------------------------------------------------------

#[pyclass(name = "Tracker")]
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

    /// Install a timed trajectory from a path and one time per waypoint.
    fn set_trajectory(&mut self, path: PyPath, times: Vec<f64>) -> PyResult<()> {
        let n = path.inner.waypoints.len();
        if times.len() != n {
            return Err(pyo3::exceptions::PyValueError::new_err("times must have one entry per waypoint"));
        }
        let mut speeds = path.inner.speeds.clone();
        speeds.resize(n, 0.0);
        self.inner.set_trajectory(ondrive_traj(path.inner.waypoints, times, speeds));
        Ok(())
    }

    fn trajectory_time(&self) -> f64 {
        self.inner.trajectory_time()
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
    fn tick(&mut self, py: Python<'_>, state: PyRef<'_, PyRobotState>, dt: f64, world: Option<PyRef<'_, PyWorld>>) -> PyResult<PyVelocityCommand> {
        let w = world.as_ref().map(|w| &w.inner);
        let s = &state.inner;
        let tracker = &mut self.inner;
        py.allow_threads(|| caught(|| Ok(PyVelocityCommand { inner: tracker.tick(s, dt, w) })))
    }

    /// The controller's own parameters as a dict (empty when it has none).
    fn params(&self, py: Python<'_>) -> PyResult<PyObject> {
        let text = self.inner.params().to_string();
        Ok(py.import("json")?.call_method1("loads", (text,))?.unbind())
    }

    /// Update some of the controller's own parameters, e.g.
    /// `set_params({"horizon_steps": 30})`.
    fn set_params(&mut self, py: Python<'_>, params: &Bound<'_, PyAny>) -> PyResult<()> {
        let text: String = py.import("json")?.call_method1("dumps", (params,))?.extract()?;
        let value: serde_json::Value = serde_json::from_str(&text).map_err(|e| PyValueError::new_err(e.to_string()))?;
        self.inner.set_params(&value).map_err(PyValueError::new_err)
    }

    /// Seed the random sampling of MPPI, MCA and SOC.
    fn set_seed(&mut self, seed: u64) {
        self.inner.set_seed(seed);
    }

    /// The safety filter's settings as a dict.
    fn safety(&self, py: Python<'_>) -> PyResult<PyObject> {
        let text = self.inner.safety().to_string();
        Ok(py.import("json")?.call_method1("loads", (text,))?.unbind())
    }

    /// Update some of the safety filter's settings, e.g.
    /// `set_safety({"enabled": False})`.
    fn set_safety(&mut self, py: Python<'_>, settings: &Bound<'_, PyAny>) -> PyResult<()> {
        let text: String = py.import("json")?.call_method1("dumps", (settings,))?.extract()?;
        let value: serde_json::Value = serde_json::from_str(&text).map_err(|e| PyValueError::new_err(e.to_string()))?;
        self.inner.set_safety(&value).map_err(PyValueError::new_err)
    }

    /// The controller's planned trajectory from the last tick, as `(x, y, z)`.
    fn predicted_trajectory(&self) -> Vec<(f64, f64, f64)> {
        self.inner.predicted_trajectory().into_iter().map(|p| (p.x, p.y, p.z)).collect()
    }

    /// How much of its internals the controller records each tick: "off",
    /// "basic" (guides round obstacles) or "full" (also sampled or solved
    /// trajectories with their costs).
    fn set_debug(&mut self, level: &str) -> PyResult<()> {
        let level = DebugLevel::parse(level).ok_or_else(|| PyValueError::new_err(format!("unknown debug level `{level}`")))?;
        self.inner.set_debug(level);
        Ok(())
    }

    /// The controller's internals from the last tick as a dict:
    /// `rollouts` (lists of `(x, y)`), `costs` (one per rollout, lower
    /// better) and `guides` (lists of `(x, y)`).
    fn debug_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let frame = self.inner.debug_frame();
        let path = |p: &Vec<datapod::Point>| p.iter().map(|q| (q.x, q.y)).collect::<Vec<_>>();
        let out = pyo3::types::PyDict::new(py);
        out.set_item("rollouts", frame.rollouts.iter().map(path).collect::<Vec<_>>())?;
        out.set_item("costs", frame.costs)?;
        out.set_item("guides", frame.guides.iter().map(path).collect::<Vec<_>>())?;
        Ok(out)
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

    fn is_path_completed(&self) -> bool {
        self.inner.is_path_completed()
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
        "regulated_pursuit",
        "pose_reach",
        "ilqr",
        "pose_regulator",
        "vector_pursuit",
        "kanayama",
        "apf",
        "ilc",
        "slip_adaptive",
        "velocity_obstacle",
        "frenet",
    ]
}

#[pyfunction]
fn available_steering_types() -> Vec<&'static str> {
    vec!["differential", "ackermann", "holonomic", "skid_steer"]
}

// ---------------------------------------------------------------------------
// Obstacle tracking.
// ---------------------------------------------------------------------------

/// Obstacles tracked over frames from their detected positions alone, each
/// with a velocity and a spread a Kalman filter keeps up to date.
#[pyclass(name = "ObstacleTracker")]
pub struct PyObstacleTracker {
    inner: crate::core::tracking::ObstacleTracker,
}

#[pymethods]
impl PyObstacleTracker {
    /// A tracker with the default settings, some replaced by `config`, e.g.
    /// `ObstacleTracker({"position_noise": 0.1})`.
    #[new]
    #[pyo3(signature = (config = None))]
    fn new(py: Python<'_>, config: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let mut cfg = crate::core::tracking::TrackingConfig::default();
        if let Some(c) = config {
            let text: String = py.import("json")?.call_method1("dumps", (c,))?.extract()?;
            let given: serde_json::Value = serde_json::from_str(&text).map_err(|e| PyValueError::new_err(e.to_string()))?;
            cfg = crate::controller::merged(&cfg, &given).map_err(PyValueError::new_err)?;
        }
        Ok(Self { inner: crate::core::tracking::ObstacleTracker::new(cfg) })
    }

    /// Fold in the detections seen at `time` seconds, each
    /// `(id or None, x, y, radius)`.
    fn update(&mut self, time: f64, detections: Vec<(Option<u64>, f64, f64, f64)>) {
        let d: Vec<crate::core::tracking::Detection> =
            detections.into_iter().map(|(id, x, y, radius)| crate::core::tracking::Detection { id, x, y, radius }).collect();
        self.inner.update(time, &d);
    }

    /// Put the tracked obstacles, with their predictions, into `world`,
    /// replacing those it had.
    fn fill(&self, mut world: PyRefMut<'_, PyWorld>) {
        self.inner.fill(&mut world.inner);
    }

    /// Velocity `(vx, vy)` estimated for the obstacle `id`, or None.
    fn velocity(&self, id: u64) -> Option<(f64, f64)> {
        self.inner.velocity(id)
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }
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
    m.add_class::<PyObstacleTracker>()?;

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

/// A standard deviation given once for every step or per step.
#[derive(FromPyObject)]
enum Spread {
    One(f64),
    Steps(Vec<f64>),
}

impl Spread {
    fn steps(self, h: usize) -> PyResult<Vec<f64>> {
        match self {
            Spread::One(s) => Ok(vec![s; h]),
            Spread::Steps(v) if v.len() == h => Ok(v),
            Spread::Steps(_) => Err(PyValueError::new_err("a per-step spread needs one entry per mean")),
        }
    }
}

/// Runs `f`, turning a panic into a `RuntimeError`.
fn caught<T>(f: impl FnOnce() -> PyResult<T>) -> PyResult<T> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(result) => result,
        Err(payload) => {
            let msg = payload
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "unknown panic".into());
            Err(pyo3::exceptions::PyRuntimeError::new_err(format!("ondrive panicked: {msg}")))
        }
    }
}
