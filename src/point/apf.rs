//! Artificial potential field (Khatib 1986) as a fallback / recovery
//! behaviour: an attractive force toward the goal and repulsive forces from
//! obstacles within `influence`, the robot steering along the resultant.
//! Local minima are inherent: when the resultant vanishes away from the
//! goal the controller stops and reports mode `apf_local_minimum`.

use crate::controller::{Controller, ControllerBase};
use crate::core::kinematics::{finalize, heading_speed_scale, reverse_allowed, stop};
use crate::core::math::normalize_angle;
use crate::core::obstacles::{CollisionChecker, obstacle_at};
use crate::types::{Goal, RobotConstraints, RobotState, VelocityCommand, WorldConstraints};
use std::f64::consts::PI;

#[derive(Clone, Debug)]
pub struct ApfConfig {
    pub k_attract: f64,
    pub k_repulse: f64,
    /// Clearance below which an obstacle repels.
    pub influence: f64,
    /// Resultant magnitude below which the field counts as a local minimum.
    pub minimum_force: f64,
}

impl Default for ApfConfig {
    fn default() -> Self {
        Self { k_attract: 1.0, k_repulse: 0.3, influence: 1.5, minimum_force: 0.05 }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ApfFollower {
    pub base: ControllerBase,
    pub apf_config: ApfConfig,
}

impl ApfFollower {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_apf_config(cfg: ApfConfig) -> Self {
        Self { apf_config: cfg, ..Default::default() }
    }

    /// Resultant field at `(x, y)` with the robot heading `yaw`.
    pub fn force(&self, x: f64, y: f64, yaw: f64, goal: &Goal, constraints: &RobotConstraints, world: Option<&WorldConstraints>) -> (f64, f64) {
        let cfg = &self.apf_config;
        let gx = goal.target_pose.point.x - x;
        let gy = goal.target_pose.point.y - y;
        let gd = gx.hypot(gy);
        let (mut fx, mut fy) = if gd > 1.0 {
            (cfg.k_attract * gx / gd, cfg.k_attract * gy / gd)
        } else {
            (cfg.k_attract * gx, cfg.k_attract * gy)
        };
        let checker = CollisionChecker::new(world, constraints, 0.0);
        if let Some(w) = world {
            let r = checker.radius();
            for obs in &w.obstacles {
                for (ox, oy, orad, weight) in obstacle_at(obs, 0) {
                    let dx = x - ox;
                    let dy = y - oy;
                    let dist = dx.hypot(dy);
                    let d = (dist - orad - r).max(1e-3);
                    if d < cfg.influence && dist > 1e-9 {
                        let mag = cfg.k_repulse * weight * (1.0 / d - 1.0 / cfg.influence) / (d * d);
                        fx += mag * dx / dist;
                        fy += mag * dy / dist;
                    }
                }
            }
            if w.grid.is_some() {
                let h = 0.05;
                let c0 = checker.clearance(0, x, y, yaw);
                if c0 < cfg.influence {
                    let cx = (checker.clearance(0, x + h, y, yaw) - checker.clearance(0, x - h, y, yaw)) / (2.0 * h);
                    let cy = (checker.clearance(0, x, y + h, yaw) - checker.clearance(0, x, y - h, yaw)) / (2.0 * h);
                    let d = c0.max(1e-3);
                    let mag = cfg.k_repulse * (1.0 / d - 1.0 / cfg.influence) / (d * d);
                    let n = cx.hypot(cy).max(1e-9);
                    fx += mag * cx / n;
                    fy += mag * cy / n;
                }
            }
        }
        (fx, fy)
    }
}

impl Controller for ApfFollower {
    fn compute_control(
        &mut self,
        state: &RobotState,
        goal: &Goal,
        constraints: &RobotConstraints,
        dt: f64,
        world: Option<&WorldConstraints>,
    ) -> VelocityCommand {
        if !(dt.is_finite() && dt > 0.0) {
            return VelocityCommand::invalid("dt must be positive and finite");
        }
        let cfg = self.base.config.clone();
        let allow_reverse = reverse_allowed(&cfg, state);
        if let Some(cmd) = self.base.arrival(state, goal, constraints, false) {
            return cmd;
        }
        let yaw = state.pose.rotation.to_euler().yaw;
        let (fx, fy) = self.force(state.pose.point.x, state.pose.point.y, yaw, goal, constraints, world);
        let magnitude = fx.hypot(fy);
        self.base.status.cross_track_error = 0.0;
        self.base.status.goal_reached = false;
        if magnitude < self.apf_config.minimum_force {
            self.base.status.mode = "apf_local_minimum".into();
            return stop("APF local minimum");
        }
        let mut err = normalize_angle(fy.atan2(fx) - yaw);
        let mut direction = 1.0;
        if allow_reverse && err.abs() > PI / 2.0 {
            direction = -1.0;
            err = normalize_angle(err + PI);
        }
        self.base.status.heading_error = err;
        let dist = self.base.status.distance_to_goal;
        let v = direction
            * (cfg.kp_linear.max(0.1) * dist).min(constraints.max_linear_velocity)
            * heading_speed_scale(err, constraints)
            * magnitude.min(1.0);
        let omega = cfg.kp_angular.max(0.1) * err;
        self.base.status.mode = "apf".into();
        finalize(v, omega, constraints, &cfg, allow_reverse, "Following potential field")
    }

    fn get_type(&self) -> &'static str {
        "apf_follower"
    }

    fn base(&self) -> &ControllerBase {
        &self.base
    }

    fn base_mut(&mut self) -> &mut ControllerBase {
        &mut self.base
    }
}
