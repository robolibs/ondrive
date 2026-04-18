use crate::core::math::{distance_2d, normalize_angle, yaw_of};
use crate::types::{
    ControllerConfig, ControllerStatus, Goal, Path, RobotConstraints, RobotState, VelocityCommand,
    WorldConstraints,
};
use datapod::Pose;

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
        self.base_mut().path = path;
        self.base_mut().path_index = 0;
    }

    fn reset(&mut self) {
        let base = self.base_mut();
        base.path.waypoints.clear();
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
