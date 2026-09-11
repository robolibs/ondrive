//! C ABI for ondrive.
//!
//! Conventions: opaque Box-backed handles (free with the matching
//! *_free); fallible calls return bool/int with the reason in the
//! thread-local ondrive_last_error_message(); borrowed views are valid
//! only for the lifetime documented by the handle they came from.
//!
//! `include/ondrive.h` is generated from this file by cbindgen.

// extern "C" fns take raw pointers from C and deref them by design.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::cell::RefCell;
use std::ffi::{CStr, CString, c_char};
use std::ptr;

use datapod::{Euler, Point, Pose, Quaternion};

use crate::types::{
    ControllerConfig, ControllerStatus, GaussianMode, Goal, Obstacle, OutputUnits, Path,
    RobotConstraints, RobotState, SteeringType, Velocity, VelocityCommand, WorldConstraints,
};
use crate::{Tracker, TrackerKind};

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
    static LAST_STATUS_MODE: RefCell<Option<CString>> = const { RefCell::new(None) };
    static LAST_CMD_MESSAGE: RefCell<Option<CString>> = const { RefCell::new(None) };
}

fn clear_last_error() {
    LAST_ERROR.with(|s| *s.borrow_mut() = None);
}

fn set_last_error(msg: impl Into<String>) {
    let msg = msg.into().replace('\0', " ");
    LAST_ERROR.with(|s| {
        *s.borrow_mut() =
            Some(CString::new(msg).unwrap_or_else(|_| CString::new("ondrive ffi error").unwrap()));
    });
}

fn ok() -> bool {
    clear_last_error();
    true
}

fn fail(msg: impl Into<String>) -> bool {
    set_last_error(msg);
    false
}

// ===========================================================================
// Integer constants for enums (kept explicit so the header can mirror them).
// ===========================================================================

pub const ONDRIVE_KIND_PID: u32 = 0;
pub const ONDRIVE_KIND_CARROT: u32 = 1;
pub const ONDRIVE_KIND_PURE_PURSUIT: u32 = 2;
pub const ONDRIVE_KIND_STANLEY: u32 = 3;
pub const ONDRIVE_KIND_LQR: u32 = 4;
pub const ONDRIVE_KIND_MPC: u32 = 5;
pub const ONDRIVE_KIND_MPPI: u32 = 6;
pub const ONDRIVE_KIND_MCA: u32 = 7;
pub const ONDRIVE_KIND_SOC: u32 = 8;
pub const ONDRIVE_KIND_DWA: u32 = 9;
pub const ONDRIVE_KIND_TEB: u32 = 10;
pub const ONDRIVE_KIND_FLC: u32 = 11;

pub const ONDRIVE_STEERING_DIFFERENTIAL: u32 = 0;
pub const ONDRIVE_STEERING_ACKERMANN: u32 = 1;
pub const ONDRIVE_STEERING_HOLONOMIC: u32 = 2;
pub const ONDRIVE_STEERING_SKID_STEER: u32 = 3;

pub const ONDRIVE_UNITS_NORMALIZED: u32 = 0;
pub const ONDRIVE_UNITS_PHYSICAL: u32 = 1;

fn kind_from_u32(v: u32) -> Option<TrackerKind> {
    Some(match v {
        ONDRIVE_KIND_PID => TrackerKind::Pid,
        ONDRIVE_KIND_CARROT => TrackerKind::Carrot,
        ONDRIVE_KIND_PURE_PURSUIT => TrackerKind::PurePursuit,
        ONDRIVE_KIND_STANLEY => TrackerKind::Stanley,
        ONDRIVE_KIND_LQR => TrackerKind::Lqr,
        ONDRIVE_KIND_MPC => TrackerKind::Mpc,
        ONDRIVE_KIND_MPPI => TrackerKind::Mppi,
        ONDRIVE_KIND_MCA => TrackerKind::Mca,
        ONDRIVE_KIND_SOC => TrackerKind::Soc,
        ONDRIVE_KIND_DWA => TrackerKind::Dwa,
        ONDRIVE_KIND_TEB => TrackerKind::Teb,
        ONDRIVE_KIND_FLC => TrackerKind::Flc,
        _ => return None,
    })
}

fn steering_from_u32(v: u32) -> Option<SteeringType> {
    Some(match v {
        ONDRIVE_STEERING_DIFFERENTIAL => SteeringType::Differential,
        ONDRIVE_STEERING_ACKERMANN => SteeringType::Ackermann,
        ONDRIVE_STEERING_HOLONOMIC => SteeringType::Holonomic,
        ONDRIVE_STEERING_SKID_STEER => SteeringType::SkidSteer,
        _ => return None,
    })
}

fn steering_to_u32(s: SteeringType) -> u32 {
    match s {
        SteeringType::Differential => ONDRIVE_STEERING_DIFFERENTIAL,
        SteeringType::Ackermann => ONDRIVE_STEERING_ACKERMANN,
        SteeringType::Holonomic => ONDRIVE_STEERING_HOLONOMIC,
        SteeringType::SkidSteer => ONDRIVE_STEERING_SKID_STEER,
    }
}

fn units_from_u32(v: u32) -> Option<OutputUnits> {
    Some(match v {
        ONDRIVE_UNITS_NORMALIZED => OutputUnits::Normalized,
        ONDRIVE_UNITS_PHYSICAL => OutputUnits::Physical,
        _ => return None,
    })
}

fn units_to_u32(u: OutputUnits) -> u32 {
    match u {
        OutputUnits::Normalized => ONDRIVE_UNITS_NORMALIZED,
        OutputUnits::Physical => ONDRIVE_UNITS_PHYSICAL,
    }
}

// ===========================================================================
// POD structs.
// ===========================================================================

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct OndriveVec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct OndriveQuat {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub w: f64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct OndrivePose {
    pub position: OndriveVec3,
    pub rotation: OndriveQuat,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct OndriveVelocity {
    pub linear: f64,
    pub angular: f64,
    pub lateral: f64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct OndriveRobotState {
    pub pose: OndrivePose,
    pub velocity: OndriveVelocity,
    pub timestamp: f64,
    pub allow_reverse: bool,
    pub turn_first: bool,
    pub allow_move: bool,
    pub has_trailer: bool,
    pub trailer_pose: OndrivePose,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct OndriveGoal {
    pub target_pose: OndrivePose,
    pub has_target_velocity: bool,
    pub target_velocity: OndriveVelocity,
    pub tolerance_position: f64,
    pub tolerance_orientation: f64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct OndriveRobotConstraints {
    pub steering_type: u32,
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

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct OndriveControllerConfig {
    pub output_units: u32,
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

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct OndriveControllerStatus {
    pub goal_reached: bool,
    pub distance_to_goal: f64,
    pub cross_track_error: f64,
    pub heading_error: f64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct OndriveVelocityCommand {
    pub valid: bool,
    pub linear_velocity: f64,
    pub angular_velocity: f64,
    pub lateral_velocity: f64,
    /// Ackermann steering angle (rad) consistent with `angular_velocity`.
    pub steering_angle: f64,
    pub output_type: u32, // always VelocityCommand for now
}

// ===========================================================================
// Conversions.
// ===========================================================================

fn vec3_to_point(v: OndriveVec3) -> Point {
    Point::new(v.x, v.y, v.z)
}
fn point_to_vec3(p: Point) -> OndriveVec3 {
    OndriveVec3 {
        x: p.x,
        y: p.y,
        z: p.z,
    }
}

fn quat_to_rs(q: OndriveQuat) -> Quaternion {
    Quaternion {
        x: q.x,
        y: q.y,
        z: q.z,
        w: q.w,
    }
}
fn quat_to_ffi(q: Quaternion) -> OndriveQuat {
    OndriveQuat {
        x: q.x,
        y: q.y,
        z: q.z,
        w: q.w,
    }
}

fn pose_to_rs(p: OndrivePose) -> Pose {
    Pose {
        point: vec3_to_point(p.position),
        rotation: quat_to_rs(p.rotation),
    }
}
fn pose_to_ffi(p: Pose) -> OndrivePose {
    OndrivePose {
        position: point_to_vec3(p.point),
        rotation: quat_to_ffi(p.rotation),
    }
}

fn velocity_to_rs(v: OndriveVelocity) -> Velocity {
    Velocity {
        linear: v.linear,
        angular: v.angular,
        lateral: v.lateral,
    }
}
#[allow(dead_code)]
fn velocity_to_ffi(v: Velocity) -> OndriveVelocity {
    OndriveVelocity {
        linear: v.linear,
        angular: v.angular,
        lateral: v.lateral,
    }
}

fn state_to_rs(s: OndriveRobotState) -> RobotState {
    RobotState {
        pose: pose_to_rs(s.pose),
        velocity: velocity_to_rs(s.velocity),
        timestamp: s.timestamp,
        allow_reverse: s.allow_reverse,
        turn_first: s.turn_first,
        allow_move: s.allow_move,
        has_trailer: s.has_trailer,
        trailer_pose: pose_to_rs(s.trailer_pose),
    }
}

fn goal_to_rs(g: OndriveGoal) -> Goal {
    Goal {
        target_pose: pose_to_rs(g.target_pose),
        target_velocity: if g.has_target_velocity {
            Some(velocity_to_rs(g.target_velocity))
        } else {
            None
        },
        tolerance_position: g.tolerance_position,
        tolerance_orientation: g.tolerance_orientation,
    }
}

fn constraints_to_rs(c: OndriveRobotConstraints) -> Option<RobotConstraints> {
    let steering = steering_from_u32(c.steering_type)?;
    Some(RobotConstraints {
        steering_type: steering,
        wheelbase: c.wheelbase,
        track_width: c.track_width,
        wheel_radius: c.wheel_radius,
        max_linear_velocity: c.max_linear_velocity,
        min_linear_velocity: c.min_linear_velocity,
        max_angular_velocity: c.max_angular_velocity,
        max_linear_acceleration: c.max_linear_acceleration,
        max_angular_acceleration: c.max_angular_acceleration,
        max_steering_angle: c.max_steering_angle,
        max_steering_rate: c.max_steering_rate,
        min_turning_radius: c.min_turning_radius,
        rear_wheelbase: c.rear_wheelbase,
        max_rear_steering_angle: c.max_rear_steering_angle,
        robot_width: c.robot_width,
        robot_length: c.robot_length,
    })
}

fn constraints_to_ffi(c: &RobotConstraints) -> OndriveRobotConstraints {
    OndriveRobotConstraints {
        steering_type: steering_to_u32(c.steering_type),
        wheelbase: c.wheelbase,
        track_width: c.track_width,
        wheel_radius: c.wheel_radius,
        max_linear_velocity: c.max_linear_velocity,
        min_linear_velocity: c.min_linear_velocity,
        max_angular_velocity: c.max_angular_velocity,
        max_linear_acceleration: c.max_linear_acceleration,
        max_angular_acceleration: c.max_angular_acceleration,
        max_steering_angle: c.max_steering_angle,
        max_steering_rate: c.max_steering_rate,
        min_turning_radius: c.min_turning_radius,
        rear_wheelbase: c.rear_wheelbase,
        max_rear_steering_angle: c.max_rear_steering_angle,
        robot_width: c.robot_width,
        robot_length: c.robot_length,
    }
}

fn config_to_rs(c: OndriveControllerConfig) -> Option<ControllerConfig> {
    let units = units_from_u32(c.output_units)?;
    Some(ControllerConfig {
        output_units: units,
        kp_linear: c.kp_linear,
        ki_linear: c.ki_linear,
        kd_linear: c.kd_linear,
        kp_angular: c.kp_angular,
        ki_angular: c.ki_angular,
        kd_angular: c.kd_angular,
        lookahead_distance: c.lookahead_distance,
        k_cross_track: c.k_cross_track,
        k_heading: c.k_heading,
        allow_reverse: c.allow_reverse,
        goal_tolerance: c.goal_tolerance,
        angular_tolerance: c.angular_tolerance,
    })
}

fn config_to_ffi(c: &ControllerConfig) -> OndriveControllerConfig {
    OndriveControllerConfig {
        output_units: units_to_u32(c.output_units),
        kp_linear: c.kp_linear,
        ki_linear: c.ki_linear,
        kd_linear: c.kd_linear,
        kp_angular: c.kp_angular,
        ki_angular: c.ki_angular,
        kd_angular: c.kd_angular,
        lookahead_distance: c.lookahead_distance,
        k_cross_track: c.k_cross_track,
        k_heading: c.k_heading,
        allow_reverse: c.allow_reverse,
        goal_tolerance: c.goal_tolerance,
        angular_tolerance: c.angular_tolerance,
    }
}

fn status_to_ffi(s: &ControllerStatus) -> OndriveControllerStatus {
    OndriveControllerStatus {
        goal_reached: s.goal_reached,
        distance_to_goal: s.distance_to_goal,
        cross_track_error: s.cross_track_error,
        heading_error: s.heading_error,
    }
}

fn cmd_to_ffi(c: &VelocityCommand) -> OndriveVelocityCommand {
    OndriveVelocityCommand {
        valid: c.valid,
        linear_velocity: c.linear_velocity,
        angular_velocity: c.angular_velocity,
        lateral_velocity: c.lateral_velocity,
        steering_angle: c.steering_angle,
        output_type: 0, // VelocityCommand
    }
}

fn store_cmd_message(c: &VelocityCommand) {
    let s = c.status_message.clone().replace('\0', " ");
    LAST_CMD_MESSAGE.with(|slot| {
        *slot.borrow_mut() = Some(CString::new(s).unwrap_or_else(|_| CString::new("").unwrap()));
    });
}

fn store_status_mode(s: &ControllerStatus) {
    let mode = s.mode.clone().replace('\0', " ");
    LAST_STATUS_MODE.with(|slot| {
        *slot.borrow_mut() = Some(CString::new(mode).unwrap_or_else(|_| CString::new("").unwrap()));
    });
}

fn write_out<T>(out: *mut T, value: T) -> bool {
    if out.is_null() {
        return fail("null output pointer");
    }
    unsafe { ptr::write(out, value) };
    true
}

// ===========================================================================
// Error + enum accessors.
// ===========================================================================

#[unsafe(no_mangle)]
pub extern "C" fn ondrive_last_error_message() -> *const c_char {
    LAST_ERROR.with(|s| s.borrow().as_ref().map_or(ptr::null(), |m| m.as_ptr()))
}

#[unsafe(no_mangle)]
pub extern "C" fn ondrive_default_config() -> OndriveControllerConfig {
    config_to_ffi(&ControllerConfig::default())
}

#[unsafe(no_mangle)]
pub extern "C" fn ondrive_default_constraints() -> OndriveRobotConstraints {
    constraints_to_ffi(&RobotConstraints::default())
}

#[unsafe(no_mangle)]
pub extern "C" fn ondrive_quaternion_from_yaw(yaw: f64) -> OndriveQuat {
    let q = Quaternion::from_euler(Euler::new(0.0, 0.0, yaw));
    quat_to_ffi(q)
}

#[unsafe(no_mangle)]
pub extern "C" fn ondrive_quaternion_yaw(q: OndriveQuat) -> f64 {
    quat_to_rs(q).to_euler().yaw
}

// ===========================================================================
// Path opaque handle.
// ===========================================================================

pub struct OndrivePath {
    path: Path,
}

#[unsafe(no_mangle)]
pub extern "C" fn ondrive_path_new() -> *mut OndrivePath {
    clear_last_error();
    Box::into_raw(Box::new(OndrivePath {
        path: Path::default(),
    }))
}

#[unsafe(no_mangle)]
pub extern "C" fn ondrive_path_free(h: *mut OndrivePath) {
    if h.is_null() {
        return;
    }
    unsafe { drop(Box::from_raw(h)) };
}

#[unsafe(no_mangle)]
pub extern "C" fn ondrive_path_add_waypoint(
    h: *mut OndrivePath,
    pose: OndrivePose,
    speed: f64,
) -> bool {
    if h.is_null() {
        return fail("null path handle");
    }
    let p = unsafe { &mut *h };
    p.path.waypoints.push(pose_to_rs(pose));
    p.path.speeds.push(speed);
    ok()
}

#[unsafe(no_mangle)]
pub extern "C" fn ondrive_path_add_waypoint_xy(
    h: *mut OndrivePath,
    x: f64,
    y: f64,
    yaw: f64,
    speed: f64,
) -> bool {
    let pose = OndrivePose {
        position: OndriveVec3 { x, y, z: 0.0 },
        rotation: ondrive_quaternion_from_yaw(yaw),
    };
    ondrive_path_add_waypoint(h, pose, speed)
}

#[unsafe(no_mangle)]
pub extern "C" fn ondrive_path_len(h: *const OndrivePath) -> usize {
    if h.is_null() {
        set_last_error("null path handle");
        return 0;
    }
    unsafe { (*h).path.waypoints.len() }
}

#[unsafe(no_mangle)]
pub extern "C" fn ondrive_path_waypoint(
    h: *const OndrivePath,
    idx: usize,
    out: *mut OndrivePose,
) -> bool {
    if h.is_null() {
        return fail("null path handle");
    }
    let p = unsafe { &*h };
    let Some(wp) = p.path.waypoints.get(idx) else {
        return fail("waypoint index out of range");
    };
    write_out(out, pose_to_ffi(*wp))
}

#[unsafe(no_mangle)]
pub extern "C" fn ondrive_path_clear(h: *mut OndrivePath) -> bool {
    if h.is_null() {
        return fail("null path handle");
    }
    let p = unsafe { &mut *h };
    p.path.waypoints.clear();
    p.path.speeds.clear();
    ok()
}

#[unsafe(no_mangle)]
pub extern "C" fn ondrive_path_set_closed(h: *mut OndrivePath, closed: bool) -> bool {
    if h.is_null() {
        return fail("null path handle");
    }
    unsafe { (*h).path.is_closed = closed };
    ok()
}

#[unsafe(no_mangle)]
pub extern "C" fn ondrive_path_smoothen(h: *mut OndrivePath, max_segment_m: f64) -> bool {
    if h.is_null() {
        return fail("null path handle");
    }
    let p = unsafe { &mut *h };
    crate::smoothen_path(&mut p.path, max_segment_m);
    ok()
}

// ===========================================================================
// World (obstacles) opaque handle.
// ===========================================================================

pub struct OndriveWorld {
    world: WorldConstraints,
}

#[unsafe(no_mangle)]
pub extern "C" fn ondrive_world_new() -> *mut OndriveWorld {
    clear_last_error();
    Box::into_raw(Box::new(OndriveWorld {
        world: WorldConstraints::default(),
    }))
}

#[unsafe(no_mangle)]
pub extern "C" fn ondrive_world_free(h: *mut OndriveWorld) {
    if h.is_null() {
        return;
    }
    unsafe { drop(Box::from_raw(h)) };
}

#[unsafe(no_mangle)]
pub extern "C" fn ondrive_world_clear(h: *mut OndriveWorld) -> bool {
    if h.is_null() {
        return fail("null world handle");
    }
    unsafe {
        (*h).world.obstacles.clear();
        (*h).world.zones.clear();
    }
    ok()
}

#[unsafe(no_mangle)]
pub extern "C" fn ondrive_world_obstacle_count(h: *const OndriveWorld) -> usize {
    if h.is_null() {
        set_last_error("null world handle");
        return 0;
    }
    unsafe { (*h).world.obstacles.len() }
}

/// Add an obstacle with a single Gaussian mode whose mean stays at (x, y)
/// for `horizon_steps` time steps (standard deviations `std_x`, `std_y`).
#[unsafe(no_mangle)]
pub extern "C" fn ondrive_world_add_static_gaussian_obstacle(
    h: *mut OndriveWorld,
    id: u64,
    x: f64,
    y: f64,
    radius: f64,
    std_x: f64,
    std_y: f64,
    horizon_steps: usize,
) -> bool {
    if h.is_null() {
        return fail("null world handle");
    }
    let horizon = horizon_steps.max(1);
    let obstacle = Obstacle {
        id,
        radius,
        modes: vec![GaussianMode {
            weight: 1.0,
            mean_x: vec![x; horizon],
            mean_y: vec![y; horizon],
            std_x: vec![std_x; horizon],
            std_y: vec![std_y; horizon],
        }],
    };
    unsafe { (*h).world.obstacles.push(obstacle) };
    ok()
}

/// Add an obstacle with a provided mean trajectory and fixed std across the
/// horizon. `mean_x` / `mean_y` must point to `horizon_steps` doubles.
#[unsafe(no_mangle)]
pub extern "C" fn ondrive_world_add_trajectory_obstacle(
    h: *mut OndriveWorld,
    id: u64,
    radius: f64,
    mean_x: *const f64,
    mean_y: *const f64,
    std_x: f64,
    std_y: f64,
    horizon_steps: usize,
) -> bool {
    if h.is_null() {
        return fail("null world handle");
    }
    if mean_x.is_null() || mean_y.is_null() {
        return fail("null mean_x or mean_y");
    }
    if horizon_steps == 0 {
        return fail("horizon_steps must be >= 1");
    }
    let xs = unsafe { std::slice::from_raw_parts(mean_x, horizon_steps) }.to_vec();
    let ys = unsafe { std::slice::from_raw_parts(mean_y, horizon_steps) }.to_vec();
    let obs = Obstacle {
        id,
        radius,
        modes: vec![GaussianMode {
            weight: 1.0,
            mean_x: xs,
            mean_y: ys,
            std_x: vec![std_x; horizon_steps],
            std_y: vec![std_y; horizon_steps],
        }],
    };
    unsafe { (*h).world.obstacles.push(obs) };
    ok()
}

// ===========================================================================
// Tracker opaque handle.
// ===========================================================================

pub struct OndriveTracker {
    tracker: Tracker,
}

#[unsafe(no_mangle)]
pub extern "C" fn ondrive_tracker_new(kind: u32) -> *mut OndriveTracker {
    let Some(k) = kind_from_u32(kind) else {
        set_last_error(format!("unknown tracker kind: {kind}"));
        return ptr::null_mut();
    };
    clear_last_error();
    Box::into_raw(Box::new(OndriveTracker {
        tracker: Tracker::new(k),
    }))
}

#[unsafe(no_mangle)]
pub extern "C" fn ondrive_tracker_free(h: *mut OndriveTracker) {
    if h.is_null() {
        return;
    }
    unsafe { drop(Box::from_raw(h)) };
}

#[unsafe(no_mangle)]
pub extern "C" fn ondrive_tracker_init(
    h: *mut OndriveTracker,
    constraints: OndriveRobotConstraints,
) -> bool {
    if h.is_null() {
        return fail("null tracker handle");
    }
    let Some(c) = constraints_to_rs(constraints) else {
        return fail("invalid steering_type in constraints");
    };
    unsafe { (*h).tracker.init(c) };
    ok()
}

#[unsafe(no_mangle)]
pub extern "C" fn ondrive_tracker_set_config(
    h: *mut OndriveTracker,
    config: OndriveControllerConfig,
) -> bool {
    if h.is_null() {
        return fail("null tracker handle");
    }
    let Some(c) = config_to_rs(config) else {
        return fail("invalid output_units in config");
    };
    unsafe { (*h).tracker.set_config(c) };
    ok()
}

#[unsafe(no_mangle)]
pub extern "C" fn ondrive_tracker_get_config(
    h: *const OndriveTracker,
    out: *mut OndriveControllerConfig,
) -> bool {
    if h.is_null() {
        return fail("null tracker handle");
    }
    let cfg = unsafe { (*h).tracker.get_config() };
    write_out(out, config_to_ffi(&cfg))
}

#[unsafe(no_mangle)]
pub extern "C" fn ondrive_tracker_set_goal(h: *mut OndriveTracker, goal: OndriveGoal) -> bool {
    if h.is_null() {
        return fail("null tracker handle");
    }
    unsafe { (*h).tracker.set_goal(goal_to_rs(goal)) };
    ok()
}

#[unsafe(no_mangle)]
pub extern "C" fn ondrive_tracker_clear_goal(h: *mut OndriveTracker) -> bool {
    if h.is_null() {
        return fail("null tracker handle");
    }
    unsafe { (*h).tracker.clear_goal() };
    ok()
}

#[unsafe(no_mangle)]
pub extern "C" fn ondrive_tracker_set_path(
    h: *mut OndriveTracker,
    path: *const OndrivePath,
) -> bool {
    if h.is_null() {
        return fail("null tracker handle");
    }
    if path.is_null() {
        return fail("null path handle");
    }
    let p = unsafe { (*path).path.clone() };
    unsafe { (*h).tracker.set_path(p) };
    ok()
}

#[unsafe(no_mangle)]
pub extern "C" fn ondrive_tracker_clear_path(h: *mut OndriveTracker) -> bool {
    if h.is_null() {
        return fail("null tracker handle");
    }
    unsafe { (*h).tracker.clear_path() };
    ok()
}

#[unsafe(no_mangle)]
pub extern "C" fn ondrive_tracker_reset(h: *mut OndriveTracker) -> bool {
    if h.is_null() {
        return fail("null tracker handle");
    }
    unsafe { (*h).tracker.reset() };
    ok()
}

#[unsafe(no_mangle)]
pub extern "C" fn ondrive_tracker_smoothen(h: *mut OndriveTracker, max_segment_m: f64) -> bool {
    if h.is_null() {
        return fail("null tracker handle");
    }
    unsafe { (*h).tracker.smoothen(max_segment_m) };
    ok()
}

#[unsafe(no_mangle)]
pub extern "C" fn ondrive_tracker_tick(
    h: *mut OndriveTracker,
    state: OndriveRobotState,
    dt: f64,
    world: *const OndriveWorld,
    out_cmd: *mut OndriveVelocityCommand,
) -> bool {
    if h.is_null() {
        return fail("null tracker handle");
    }
    let w_ref = if world.is_null() {
        None
    } else {
        Some(unsafe { &(*world).world })
    };
    let cmd = unsafe { (*h).tracker.tick(&state_to_rs(state), dt, w_ref) };
    store_cmd_message(&cmd);
    write_out(out_cmd, cmd_to_ffi(&cmd))
}

#[unsafe(no_mangle)]
pub extern "C" fn ondrive_tracker_emergency_stop(
    h: *mut OndriveTracker,
    out_cmd: *mut OndriveVelocityCommand,
) -> bool {
    if h.is_null() {
        return fail("null tracker handle");
    }
    let cmd = unsafe { (*h).tracker.emergency_stop() };
    store_cmd_message(&cmd);
    write_out(out_cmd, cmd_to_ffi(&cmd))
}

#[unsafe(no_mangle)]
pub extern "C" fn ondrive_tracker_status(
    h: *const OndriveTracker,
    out: *mut OndriveControllerStatus,
) -> bool {
    if h.is_null() {
        return fail("null tracker handle");
    }
    let s = unsafe { (*h).tracker.get_status() };
    store_status_mode(&s);
    write_out(out, status_to_ffi(&s))
}

#[unsafe(no_mangle)]
pub extern "C" fn ondrive_tracker_status_mode() -> *const c_char {
    LAST_STATUS_MODE.with(|s| s.borrow().as_ref().map_or(ptr::null(), |m| m.as_ptr()))
}

#[unsafe(no_mangle)]
pub extern "C" fn ondrive_tracker_last_command_message() -> *const c_char {
    LAST_CMD_MESSAGE.with(|s| s.borrow().as_ref().map_or(ptr::null(), |m| m.as_ptr()))
}

#[unsafe(no_mangle)]
pub extern "C" fn ondrive_tracker_is_goal_reached(h: *const OndriveTracker) -> bool {
    if h.is_null() {
        set_last_error("null tracker handle");
        return false;
    }
    unsafe { (*h).tracker.is_goal_reached() }
}

/// True once a path driven without an explicit goal has been consumed.
#[unsafe(no_mangle)]
pub extern "C" fn ondrive_tracker_is_path_completed(h: *const OndriveTracker) -> bool {
    if h.is_null() {
        set_last_error("null tracker handle");
        return false;
    }
    unsafe { (*h).tracker.is_path_completed() }
}

#[unsafe(no_mangle)]
pub extern "C" fn ondrive_tracker_current_target(
    h: *const OndriveTracker,
    out: *mut OndriveVec3,
) -> bool {
    if h.is_null() {
        return fail("null tracker handle");
    }
    match unsafe { (*h).tracker.current_target() } {
        Some(p) => write_out(out, point_to_vec3(p)),
        None => fail("no goal or path set"),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn ondrive_tracker_kind(h: *const OndriveTracker) -> u32 {
    if h.is_null() {
        set_last_error("null tracker handle");
        return u32::MAX;
    }
    let kind = unsafe { (*h).tracker.kind() };
    match kind {
        TrackerKind::Pid => ONDRIVE_KIND_PID,
        TrackerKind::Carrot => ONDRIVE_KIND_CARROT,
        TrackerKind::PurePursuit => ONDRIVE_KIND_PURE_PURSUIT,
        TrackerKind::Stanley => ONDRIVE_KIND_STANLEY,
        TrackerKind::Lqr => ONDRIVE_KIND_LQR,
        TrackerKind::Mpc => ONDRIVE_KIND_MPC,
        TrackerKind::Mppi => ONDRIVE_KIND_MPPI,
        TrackerKind::Mca => ONDRIVE_KIND_MCA,
        TrackerKind::Soc => ONDRIVE_KIND_SOC,
        TrackerKind::Dwa => ONDRIVE_KIND_DWA,
        TrackerKind::Teb => ONDRIVE_KIND_TEB,
        TrackerKind::Flc => ONDRIVE_KIND_FLC,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn ondrive_tracker_constraints(
    h: *const OndriveTracker,
    out: *mut OndriveRobotConstraints,
) -> bool {
    if h.is_null() {
        return fail("null tracker handle");
    }
    let c = unsafe { (*h).tracker.constraints().clone() };
    write_out(out, constraints_to_ffi(&c))
}

// ===========================================================================
// Version string.
// ===========================================================================

#[unsafe(no_mangle)]
pub extern "C" fn ondrive_version() -> *const c_char {
    thread_local! {
        static V: CString = CString::new(env!("CARGO_PKG_VERSION")).unwrap();
    }
    V.with(|v| v.as_ptr())
}

// A helper to silence the unused-import warning when CStr is only used via
// macro expansion in the future (retain for forwards compatibility).
fn _keep_cstr_used(_p: *const c_char) {
    let _ = unsafe { CStr::from_ptr(_p) };
}
