//! Kanayama trajectory tracker (Kanayama et al. 1990):
//! `v = v_r cos(e_theta) + k_x e_x`,
//! `omega = omega_r + v_r k_y e_y + |v_r| k_theta sin(e_theta)` (the lateral
//! term keeps the sign of `v_r` so reversing corrects the right way), with
//! the pose error expressed in the robot frame against a moving reference.
//! The reference comes from the installed timed trajectory, or, without one,
//! from a point a short preview ahead of the projection on the path. Once a
//! trajectory has ended short of its final pose a pose regulator finishes.

use crate::controller::{Controller, ControllerBase};
use crate::core::kinematics::{
    actuation_delay, axle_offset, bend_heading, braked_for_bends, finalize, finalize_holonomic, followable, is_holonomic,
    path_speed,
};
use crate::core::math::{normalize_angle, yaw_of};
use crate::core::path::{
    PathCursor, curvature_at_arc_length, curvature_at_projection, cusp_after, direction_at, sample, speed_cap,
};
use std::f64::consts::PI;
use crate::point::PoseRegulatorFollower;
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
    settle: PoseRegulatorFollower,
    last_speed: f64,
}

impl KanayamaFollower {
    pub fn new() -> Self {
        Self::default()
    }

    /// Gains from the generic config: `k_x = kp_linear`,
    /// `k_y = 3 k_cross_track`, `k_theta = 2 sqrt(k_y)` (critically damped).
    pub fn gains(&self) -> (f64, f64, f64) {
        let cfg = &self.base.config;
        let k_x = cfg.kp_linear.max(0.1);
        let k_y = 3.0 * cfg.k_cross_track.max(0.05);
        (k_x, k_y, 2.0 * k_y.sqrt())
    }

    /// `gains` with `k_y` held so the error dynamics at `speed`, whose
    /// natural frequency is `|speed| sqrt(k_y)`, stay below two thirds of
    /// the actuator's bandwidth `1 / delay`.
    pub fn gains_for(&self, constraints: &RobotConstraints, speed: f64) -> (f64, f64, f64) {
        let (k_x, k_y, _) = self.gains();
        let delay = actuation_delay(constraints);
        let k_y = if delay > 0.0 && speed.abs() > 1e-3 {
            k_y.min((1.0 / (1.5 * delay * speed.abs())).powi(2))
        } else {
            k_y
        };
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

        let (x_r, y_r, yaw_r, v_r, omega_r, v_cap) = match &self.trajectory {
            Some(traj) => {
                let s = traj.sample(self.clock);
                if s.finished && !is_holonomic(constraints.steering_type) {
                    self.settle.set_config(cfg.clone());
                    let end = Goal { target_pose: s.pose, ..goal.clone() };
                    let cmd = self.settle.compute_control(state, &end, constraints, dt, None);
                    if cmd.valid {
                        self.base.status.goal_reached = self.settle.get_status().goal_reached;
                        self.base.status.mode = "kanayama_settling".into();
                        return cmd;
                    }
                }
                let v_r = if s.finished { 0.0 } else { s.speed };
                (s.pose.point.x, s.pose.point.y, yaw_of(&s.pose), v_r, if s.finished { 0.0 } else { s.yaw_rate }, f64::INFINITY)
            }
            None => {
                let s = prep.proj.arc_length;
                let cusp = cusp_after(&self.base.path.speeds, &self.cursor.cum, s).unwrap_or(f64::INFINITY);
                let (p, h) = sample(&self.base.path.waypoints, &self.cursor.cum, (s + PREVIEW).min(cusp));
                let kappa = curvature_at_projection(&self.base.path.waypoints, &self.cursor.cum, &prep.proj);
                let nominal = speed_cap(&self.base.path.speeds, &prep.proj)
                    .map_or(constraints.max_linear_velocity, |s| s.min(constraints.max_linear_velocity));
                let stop = self.base.distance_left(&prep.proj, self.cursor.total_length()).min(cusp - s);
                let v_r = path_speed(nominal, kappa, stop, prep.pos_tol, cfg.kp_linear, constraints);
                let v_now = if state.velocity_valid { state.velocity.linear } else { self.last_speed };
                let (waypoints, cum) = (&self.base.path.waypoints, &self.cursor.cum);
                let v_r = braked_for_bends(v_r, v_now, nominal, stop, |d| curvature_at_arc_length(waypoints, cum, s + d), constraints);
                self.last_speed = v_r;
                let v_cap = path_speed(constraints.max_linear_velocity, 0.0, stop, prep.pos_tol, cfg.kp_linear, constraints);
                let back = prep.allow_reverse && direction_at(&self.base.path.speeds, &prep.proj).is_some_and(|d| d < 0);
                // A reversing leg is tracked by the body facing against its
                // direction of travel, at negative speed; the body trails the
                // bend by its pose's lead on the rear axle either way.
                let trail = bend_heading(followable(kappa, constraints), if back { -1.0 } else { 1.0 } * axle_offset(constraints));
                let omega_r = v_r * kappa / trail.cos();
                if back {
                    (p.x, p.y, normalize_angle(h + PI - trail), -v_r, omega_r, v_cap)
                } else {
                    (p.x, p.y, normalize_angle(h - trail), v_r, omega_r, v_cap)
                }
            }
        };

        let (s, c) = yaw.sin_cos();
        let dx = x_r - state.pose.point.x;
        let dy = y_r - state.pose.point.y;
        let e_x = c * dx + s * dy;
        let e_y = -s * dx + c * dy;
        let e_theta = normalize_angle(yaw_r - yaw);
        let (k_x, k_y, k_theta) = self.gains_for(constraints, v_r);

        if is_holonomic(constraints.steering_type) {
            // e_x, e_y, e_theta are already body-frame errors against the
            // moving reference, so translation and rotation correct
            // independently with no coupling term needed.
            let vx = (v_r * e_theta.cos() + k_x * e_x).clamp(-v_cap, v_cap);
            let vy = v_r * e_theta.sin() + k_y * e_y;
            let omega = omega_r + k_theta * e_theta.sin();
            self.base.status.mode = if self.trajectory.is_some() {
                "kanayama_timed_holonomic".into()
            } else {
                "kanayama_holonomic".into()
            };
            return finalize_holonomic(vx, vy, omega, constraints, &cfg, "Kanayama tracking");
        }

        let v = (v_r * e_theta.cos() + k_x * e_x).clamp(-v_cap, v_cap);
        let omega = omega_r + v_r * k_y * e_y + v_r.abs() * k_theta * e_theta.sin();
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
