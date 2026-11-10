use crate::core::kinematics::{can_turn_in_place, finalize, stop};
use crate::core::math::{distance_2d, normalize_angle, yaw_of};
use crate::core::path::PathProjection;
use crate::types::{
    ControllerConfig, ControllerStatus, DebugFrame, DebugLevel, Goal, Path, RobotConstraints, RobotState, Trajectory,
    VelocityCommand, WorldConstraints,
};
use datapod::{Point, Pose};
use std::f64::consts::PI;

pub trait Controller: Send + Sync {
    fn compute_control(
        &mut self,
        state: &RobotState,
        goal: &Goal,
        constraints: &RobotConstraints,
        dt: f64,
        world: Option<&WorldConstraints>,
    ) -> VelocityCommand;

    fn set_path(&mut self, path: Path) {
        let base = self.base_mut();
        base.path = path;
        base.path_index = 0;
        base.status = ControllerStatus::default();
    }

    /// Install a timed trajectory. Controllers that track time keep it;
    /// the default follows its poses as a path.
    fn set_trajectory(&mut self, trajectory: Trajectory) {
        self.set_path(trajectory.to_path());
    }

    /// Current time on the installed trajectory, seconds from its start.
    fn set_time(&mut self, _t: f64) {}

    fn reset(&mut self) {
        let base = self.base_mut();
        base.path.waypoints.clear();
        base.path.speeds.clear();
        base.path_index = 0;
        base.status = ControllerStatus::default();
    }

    fn get_status(&self) -> ControllerStatus {
        self.base().status.clone()
    }

    fn set_config(&mut self, config: ControllerConfig) {
        self.base_mut().config = config;
    }

    fn get_config(&self) -> ControllerConfig {
        self.base().config.clone()
    }

    fn get_type(&self) -> &'static str;

    fn get_path_index(&self) -> usize {
        self.base().path_index
    }

    /// Planned trajectory from the last tick, for predictive controllers.
    fn predicted_trajectory(&self) -> Vec<Point> {
        Vec::new()
    }

    /// The controller's own parameters (its `*Config`) as a JSON object;
    /// empty for a controller that has none.
    fn params(&self) -> serde_json::Value {
        serde_json::Value::Object(serde_json::Map::new())
    }

    /// Update some of the controller's own parameters from a JSON object.
    /// An unknown name or a wrong type is an error and changes nothing.
    fn set_params(&mut self, params: &serde_json::Value) -> Result<(), String> {
        merged(&serde_json::Map::new(), params).map(|_| ())
    }

    /// Seed the random sampling of a sampling controller; others ignore it.
    fn set_seed(&mut self, _seed: u64) {}

    /// How much of its internals the controller records each tick.
    fn set_debug(&mut self, _level: DebugLevel) {}

    /// Internals recorded on the last tick; empty unless `set_debug` asked.
    fn debug_frame(&self) -> DebugFrame {
        DebugFrame::default()
    }

    fn base(&self) -> &ControllerBase;
    fn base_mut(&mut self) -> &mut ControllerBase;
}

/// `current` with the fields named in the JSON object `params` replaced.
pub fn merged<T: serde::Serialize + serde::de::DeserializeOwned>(
    current: &T,
    params: &serde_json::Value,
) -> Result<T, String> {
    let serde_json::Value::Object(updates) = params else {
        return Err("parameters must be a JSON object".into());
    };
    let mut value = serde_json::to_value(current).map_err(|e| e.to_string())?;
    let serde_json::Value::Object(fields) = &mut value else {
        return Err("parameters are not a JSON object".into());
    };
    for (name, v) in updates {
        if !fields.contains_key(name) {
            return Err(format!("unknown parameter `{name}`"));
        }
        fields.insert(name.clone(), v.clone());
    }
    serde_json::from_value(value).map_err(|e| e.to_string())
}

#[derive(Clone, Debug, Default)]
pub struct ControllerBase {
    pub config: ControllerConfig,
    pub status: ControllerStatus,
    pub path: Path,
    pub path_index: usize,
}

/// Result of testing the robot pose against a goal.
#[derive(Clone, Copy, Debug, Default)]
pub struct GoalCheck {
    pub reached: bool,
    pub position_ok: bool,
    pub orientation_ok: bool,
    pub orientation_required: bool,
    pub distance: f64,
    pub yaw_error: f64,
    pub position_tolerance: f64,
    pub orientation_tolerance: f64,
}

/// Share of top speed under which a robot counts as stopped at a cusp.
const STOPPED_AT_CUSP: f64 = 0.1;

/// Distance from a cusp within which the robot counts as there: half the
/// position tolerance while it moves, all of it once stopped or when its
/// speed is not reported.
pub fn cusp_reach(pos_tol: f64, state: &RobotState, constraints: &RobotConstraints) -> f64 {
    let stopped = STOPPED_AT_CUSP * constraints.max_linear_velocity.abs();
    let moving = state.velocity_valid && state.velocity.linear.abs() >= stopped.max(0.05);
    (if moving { 0.5 * pos_tol } else { pos_tol }).max(0.1)
}

/// Goal tolerances override config tolerances when set (> 0). An
/// orientation tolerance of `PI` or more disables the orientation check.
pub fn effective_tolerances(goal: &Goal, cfg: &ControllerConfig) -> (f64, f64) {
    let pos = if goal.tolerance_position > 0.0 {
        goal.tolerance_position
    } else {
        cfg.goal_tolerance
    };
    let ang = if goal.tolerance_orientation > 0.0 {
        goal.tolerance_orientation
    } else {
        cfg.angular_tolerance
    };
    (pos.max(1e-6), ang.max(1e-6))
}

pub fn check_goal(current: &Pose, goal: &Goal, cfg: &ControllerConfig) -> GoalCheck {
    let (pos_tol, ang_tol) = effective_tolerances(goal, cfg);
    let distance = distance_2d(current.point, goal.target_pose.point);
    let yaw_error = normalize_angle(yaw_of(&goal.target_pose) - yaw_of(current));
    let orientation_required = ang_tol < PI;
    let position_ok = distance < pos_tol;
    let orientation_ok = yaw_error.abs() < ang_tol;
    GoalCheck {
        reached: position_ok && (!orientation_required || orientation_ok),
        position_ok,
        orientation_ok,
        orientation_required,
        distance,
        yaw_error,
        position_tolerance: pos_tol,
        orientation_tolerance: ang_tol,
    }
}

/// Kept for API compatibility: `(reached, distance, |yaw error|)` using the
/// given tolerances directly.
pub fn is_goal_reached(
    current: &Pose,
    goal: &Pose,
    tolerance: f64,
    angular_tolerance: f64,
) -> (bool, f64, f64) {
    let dist = distance_2d(current.point, goal.point);
    let angle_diff = normalize_angle(yaw_of(goal) - yaw_of(current)).abs();
    let reached = dist < tolerance && angle_diff < angular_tolerance;
    (reached, dist, angle_diff)
}

impl ControllerBase {
    /// Shared arrival logic. Returns `Some(cmd)` when the controller should
    /// stop or rotate in place instead of running its tracking law:
    /// - inside position tolerance and orientation satisfied (or not
    ///   required): stop, `goal_reached = true`;
    /// - inside position tolerance, orientation required and the platform
    ///   can turn in place: rotate toward the goal yaw;
    /// - inside position tolerance but the platform cannot turn in place:
    ///   stop, `goal_reached = true`.
    ///
    /// `extra_position_ok` lets path followers report "passed the end".
    pub fn arrival(
        &mut self,
        state: &RobotState,
        goal: &Goal,
        constraints: &RobotConstraints,
        extra_position_ok: bool,
    ) -> Option<VelocityCommand> {
        self.gated_arrival(state, goal, constraints, extra_position_ok, true)
    }

    /// Distance, metres, left to slow down over for a path follower at
    /// `proj` on its path of `path_length` metres: the rest of the path, or
    /// the way straight to the goal when that is longer.
    pub fn distance_left(&self, proj: &PathProjection, path_length: f64) -> f64 {
        self.status.distance_to_goal.max(path_length - proj.arc_length)
    }

    /// `arrival` for a path follower projected at `proj` on its path of
    /// `path_length` metres. A goal at the path end counts as reached only
    /// once the rest of the path is about as short as the way to the goal.
    pub fn arrival_on_path(
        &mut self,
        state: &RobotState,
        goal: &Goal,
        constraints: &RobotConstraints,
        proj: &PathProjection,
        path_length: f64,
    ) -> Option<VelocityCommand> {
        let (pos_tol, _) = effective_tolerances(goal, &self.config);
        let passed_end = proj.beyond_end && proj.distance < 2.0 * pos_tol;
        let goal_at_end = self
            .path
            .waypoints
            .last()
            .is_some_and(|w| w.point.distance_to_2d(goal.target_pose.point) < 1e-6);
        let to_goal = distance_2d(state.pose.point, goal.target_pose.point);
        let to_path = distance_2d(state.pose.point, proj.point);
        let early =
            goal_at_end && path_length - proj.arc_length > to_goal + to_path + 2.0 * pos_tol;
        self.gated_arrival(state, goal, constraints, passed_end, !early)
    }

    fn gated_arrival(
        &mut self,
        state: &RobotState,
        goal: &Goal,
        constraints: &RobotConstraints,
        extra_position_ok: bool,
        position_allowed: bool,
    ) -> Option<VelocityCommand> {
        let check = check_goal(&state.pose, goal, &self.config);
        self.status.distance_to_goal = check.distance;
        self.status.heading_error = check.yaw_error;

        let position_ok = (check.position_ok && position_allowed) || extra_position_ok;
        if !position_ok {
            self.status.goal_reached = false;
            return None;
        }

        let needs_alignment = check.orientation_required && !check.orientation_ok;
        if needs_alignment && can_turn_in_place(constraints.steering_type) {
            self.status.goal_reached = false;
            self.status.mode = "aligning".into();
            let omega = self.config.kp_angular.max(0.1) * check.yaw_error;
            return Some(finalize(
                0.0,
                omega,
                constraints,
                &self.config,
                false,
                "Aligning to goal orientation",
            ));
        }

        self.status.goal_reached = true;
        self.status.mode = "stopped".into();
        Some(stop("Goal reached"))
    }
}
