//! Kanayama trajectory tracker (Kanayama et al. 1990):
//! `v = v_r cos(e_theta) + k_x e_x`,
//! `omega = omega_r + v_r (k_y e_y + k_theta sin(e_theta))`, with the pose
//! error expressed in the robot frame against a moving reference. The
//! reference comes from the installed timed trajectory, or, without one,
//! from a point a short preview ahead of the projection on the path.

use crate::controller::{Controller, ControllerBase};
use crate::core::kinematics::{finalize, path_speed};
use crate::core::math::{normalize_angle, yaw_of};
use crate::core::path::{PathCursor, curvature_at_projection, sample, speed_cap};
use crate::pred::mppi::prepare;
use crate::types::{
    Goal, Path, RobotConstraints, RobotState, Trajectory, VelocityCommand, WorldConstraints,
};

/// Preview distance of the spatial reference ahead of the projection.
const PREVIEW: f64 = 0.3;

#[derive(Clone, Debug, Default)]
pub struct KanayamaFollower {
    pub base: ControllerBase,
    cursor: PathCursor,
    trajectory: Option<Trajectory>,
    clock: f64,
}

impl KanayamaFollower {
    pub fn new() -> Self {
        Self::default()
    }

    /// Gains from the generic config: `k_x = kp_linear`,
    /// `k_y = 3 k_cross_track`, `k_theta = 2 sqrt(k_y)`.
    pub fn gains(&self) -> (f64, f64, f64) {
        let cfg = &self.base.config;
        let k_x = cfg.kp_linear.max(0.1);
        let k_y = 3.0 * cfg.k_cross_track.max(0.05);
        (k_x, k_y, 2.0 * k_y.sqrt())
    }
}

impl Controller for KanayamaFollower {
    fn compute_control(
        &mut self,
        state: &RobotState,
        goal: &Goal,
        constraints: &RobotConstraints,
        dt: f64,
        _world: Option<&WorldConstraints>,
    ) -> VelocityCommand {
        let prep = match prepare(&mut self.base, &mut self.cursor, state, goal, constraints, dt) {
            Ok(p) => p,
            Err(cmd) => return cmd,
        };
        let cfg = self.base.config.clone();
        let yaw = yaw_of(&state.pose);

        let (x_r, y_r, yaw_r, v_r, omega_r) = match &self.trajectory {
            Some(traj) => {
                let s = traj.sample(self.clock);
                let v_r = if s.finished { 0.0 } else { s.speed };
                (s.pose.point.x, s.pose.point.y, yaw_of(&s.pose), v_r, if s.finished { 0.0 } else { s.yaw_rate })
            }
            None => {
                let (p, h) = sample(&self.base.path.waypoints, &self.cursor.cum, prep.proj.arc_length + PREVIEW);
                let kappa = curvature_at_projection(&self.base.path.waypoints, &self.cursor.cum, &prep.proj);
                let nominal = speed_cap(&self.base.path.speeds, &prep.proj)
                    .map_or(constraints.max_linear_velocity, |s| s.min(constraints.max_linear_velocity));
                let v_r = path_speed(nominal, kappa, self.base.status.distance_to_goal, prep.pos_tol, cfg.kp_linear, constraints);
                (p.x, p.y, h, v_r, v_r * kappa)
            }
        };

        let (s, c) = yaw.sin_cos();
        let dx = x_r - state.pose.point.x;
        let dy = y_r - state.pose.point.y;
        let e_x = c * dx + s * dy;
        let e_y = -s * dx + c * dy;
        let e_theta = normalize_angle(yaw_r - yaw);
        let (k_x, k_y, k_theta) = self.gains();

        let v = v_r * e_theta.cos() + k_x * e_x;
        let omega = omega_r + v_r.abs() * (k_y * e_y + k_theta * e_theta.sin());
        self.base.status.mode = if self.trajectory.is_some() { "kanayama_timed".into() } else { "kanayama".into() };
        finalize(v, omega, constraints, &cfg, prep.allow_reverse, "Kanayama tracking")
    }

    fn set_trajectory(&mut self, trajectory: Trajectory) {
        self.set_path(trajectory.to_path());
        self.trajectory = Some(trajectory);
        self.clock = 0.0;
    }

    fn set_time(&mut self, t: f64) {
        self.clock = t;
    }

    fn set_path(&mut self, path: Path) {
        self.trajectory = None;
        self.cursor.set_path(&path.waypoints);
        self.base.path = path;
        self.base.path_index = 0;
        self.base.status = Default::default();
    }

    fn reset(&mut self) {
        self.set_path(Path::default());
    }

    fn get_type(&self) -> &'static str {
        "kanayama_follower"
    }

    fn base(&self) -> &ControllerBase {
        &self.base
    }

    fn base_mut(&mut self) -> &mut ControllerBase {
        &mut self.base
    }
}
