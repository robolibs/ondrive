//! Artificial potential field (Khatib 1986) as a fallback / recovery
//! behaviour: an attractive force toward the goal and repulsive forces from
//! obstacles within `influence` of the footprint, each at the nearest of its
//! predicted positions over the next 1.5 s, the robot steering along the
//! resultant.
//! Each repulsion carries a circulating part along the obstacle (Singh,
//! Stephanou & Wen 1996), turning the way that heads for the goal, so an
//! obstacle straight ahead is gone round rather than stopped at. Where the
//! resultant still vanishes away from the goal the controller stops and
//! reports mode `apf_local_minimum`.

use crate::controller::{Controller, ControllerBase, pass_through};
use crate::core::kinematics::{
    finalize, finalize_holonomic, heading_speed_scale, is_holonomic, reverse_allowed, stop,
    world_to_body,
};
use crate::core::math::{normalize_angle, yaw_of};
use crate::core::obstacles::{CollisionChecker, modes_at};
use crate::types::{Goal, RobotConstraints, RobotState, VelocityCommand, WorldConstraints};
use std::f64::consts::PI;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ApfConfig {
    pub k_attract: f64,
    pub k_repulse: f64,
    /// Clearance below which an obstacle repels.
    pub influence: f64,
    /// Resultant magnitude below which the field counts as a local minimum.
    pub minimum_force: f64,
    /// Circulating force along an obstacle, as a share of its repulsion;
    /// 0 is the plain field.
    pub circulation: f64,
}

impl Default for ApfConfig {
    fn default() -> Self {
        Self { k_attract: 1.0, k_repulse: 0.3, influence: 1.5, minimum_force: 0.05, circulation: 1.0 }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ApfFollower {
    pub base: ControllerBase,
    pub apf_config: ApfConfig,
    /// Way the field last circulated round an obstacle: +1 anticlockwise,
    /// −1 clockwise, 0 not yet.
    side: f64,
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
        self.field(x, y, yaw, goal, constraints, world).0
    }

    /// Resultant field, and the way it circulates round the nearest
    /// obstacle (`self.side` where nothing repels).
    fn field(
        &self,
        x: f64,
        y: f64,
        yaw: f64,
        goal: &Goal,
        constraints: &RobotConstraints,
        world: Option<&WorldConstraints>,
    ) -> ((f64, f64), f64) {
        let cfg = &self.apf_config;
        let gx = goal.target_pose.point.x - x;
        let gy = goal.target_pose.point.y - y;
        let gd = gx.hypot(gy);
        let (mut fx, mut fy) = if gd > 1.0 || (pass_through(goal) > 0.0 && gd > 1e-9) {
            (cfg.k_attract * gx / gd, cfg.k_attract * gy / gd)
        } else {
            (cfg.k_attract * gx, cfg.k_attract * gy)
        };
        let (ux, uy) = if gd > 1e-9 { (gx / gd, gy / gd) } else { (0.0, 0.0) };
        let remembered = if self.side != 0.0 { self.side } else { 1.0 };
        let (mut side, mut strongest) = (remembered, 0.0);
        let mut repel = |fx: &mut f64, fy: &mut f64, mag: f64, nx: f64, ny: f64| {
            let toward = ux * -ny + uy * nx;
            let s = if toward.abs() > CIRCULATION_TIE { toward.signum() } else { remembered };
            *fx += mag * (nx - cfg.circulation * s * ny);
            *fy += mag * (ny + cfg.circulation * s * nx);
            if mag > strongest {
                (side, strongest) = (s, mag);
            }
        };
        let checker = CollisionChecker::new(world, constraints, 0.0);
        if let Some(w) = world {
            let h = 0.05;
            for obs in &w.obstacles {
                let mut nearest: Vec<Option<crate::core::obstacles::ModeAt>> = Vec::new();
                for t in PREVIEW_S {
                    for (i, m) in modes_at(obs, t, 0.5).enumerate() {
                        if nearest.len() <= i {
                            nearest.push(None);
                        }
                        let closer = |n: &crate::core::obstacles::ModeAt| (m.x - x).hypot(m.y - y) < (n.x - x).hypot(n.y - y);
                        if nearest[i].as_ref().is_none_or(closer) {
                            nearest[i] = Some(m);
                        }
                    }
                }
                // Each repels by the footprint's clearance, pushing the way
                // that clearance grows.
                for m in nearest.into_iter().flatten() {
                    let gap = |px: f64, py: f64| checker.body_distance(m.x, m.y, px, py, yaw) - m.radius;
                    let d = gap(x, y).max(1e-3);
                    if d < cfg.influence {
                        let (nx, ny) = (gap(x + h, y) - gap(x - h, y), gap(x, y + h) - gap(x, y - h));
                        let n = nx.hypot(ny);
                        if n > 1e-9 {
                            let mag = cfg.k_repulse * m.weight * (1.0 / d - 1.0 / cfg.influence) / (d * d);
                            repel(&mut fx, &mut fy, mag, nx / n, ny / n);
                        }
                    }
                }
            }
            if w.grid.is_some() {
                let c0 = checker.clearance(0, x, y, yaw);
                if c0 < cfg.influence {
                    let cx = (checker.clearance(0, x + h, y, yaw) - checker.clearance(0, x - h, y, yaw)) / (2.0 * h);
                    let cy = (checker.clearance(0, x, y + h, yaw) - checker.clearance(0, x, y - h, yaw)) / (2.0 * h);
                    let d = c0.max(1e-3);
                    let mag = cfg.k_repulse * (1.0 / d - 1.0 / cfg.influence) / (d * d);
                    let n = cx.hypot(cy).max(1e-9);
                    repel(&mut fx, &mut fy, mag, cx / n, cy / n);
                }
            }
        }
        ((fx, fy), side)
    }
}

/// Goal-ward share of the tangent below which the circulation keeps its
/// remembered way rather than flip on noise.
const CIRCULATION_TIE: f64 = 0.05;

/// Seconds ahead at which each obstacle's prediction is read; it repels
/// from the nearest of them.
const PREVIEW_S: [f64; 4] = [0.0, 0.5, 1.0, 1.5];

impl Controller for ApfFollower {
    fn params(&self) -> serde_json::Value {
        serde_json::to_value(&self.apf_config).unwrap_or_default()
    }

    fn set_params(&mut self, params: &serde_json::Value) -> Result<(), String> {
        let cfg = crate::controller::merged(&self.apf_config, params)?;
        self.apf_config = cfg;
        Ok(())
    }

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
        let ((fx, fy), side) = self.field(state.pose.point.x, state.pose.point.y, yaw, goal, constraints, world);
        self.side = side;
        let magnitude = fx.hypot(fy);
        self.base.status.cross_track_error = 0.0;
        self.base.status.goal_reached = false;
        if magnitude < self.apf_config.minimum_force {
            self.base.status.mode = "apf_local_minimum".into();
            return stop("APF local minimum");
        }
        let dist = self.base.status.distance_to_goal;
        if is_holonomic(constraints.steering_type) {
            self.base.status.mode = "apf_holonomic".into();
            let speed =
                (cfg.kp_linear.max(0.1) * dist).max(pass_through(goal)).min(constraints.max_linear_velocity) * magnitude.min(1.0);
            let (ux, uy) = (fx / magnitude, fy / magnitude);
            let (vx, vy) = world_to_body(speed * ux, speed * uy, yaw);
            let omega = cfg.kp_angular.max(0.1) * normalize_angle(yaw_of(&goal.target_pose) - yaw);
            return finalize_holonomic(vx, vy, omega, constraints, &cfg, "Following potential field");
        }
        let mut err = normalize_angle(fy.atan2(fx) - yaw);
        let mut direction = 1.0;
        if allow_reverse && err.abs() > PI / 2.0 {
            direction = -1.0;
            err = normalize_angle(err + PI);
        }
        self.base.status.heading_error = err;
        let v = direction
            * (cfg.kp_linear.max(0.1) * dist).max(pass_through(goal)).min(constraints.max_linear_velocity)
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
