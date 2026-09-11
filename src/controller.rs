use crate::core::kinematics::{can_turn_in_place, finalize, stop};
use crate::core::math::{distance_2d, normalize_angle, yaw_of};
use crate::types::{
    ControllerConfig, ControllerStatus, Goal, Path, RobotConstraints, RobotState, Trajectory,
    VelocityCommand, WorldConstraints,
};
use datapod::{Point, Pose};
use std::f64::consts::PI;

pub trait Controller {
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

    fn base(&self) -> &ControllerBase;
    fn base_mut(&mut self) -> &mut ControllerBase;
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
        let check = check_goal(&state.pose, goal, &self.config);
        self.status.distance_to_goal = check.distance;
        self.status.heading_error = check.yaw_error;

        let position_ok = check.position_ok || extra_position_ok;
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
