use crate::controller::{Controller, check_goal};
use crate::core::kinematics::stop;
use crate::core::path::{PathCursor, end_heading, segment_heading};
use crate::fuzzy::FlcFollower;
use crate::path::{
    KanayamaFollower, LqrFollower, PurePursuitFollower, RegulatedPursuitFollower,
    StanleyFollower, VectorPursuitFollower,
};
use crate::point::{CarrotFollower, PidFollower, PoseReachFollower, PoseRegulatorFollower};
use crate::pred::{
    DwaFollower, IlqrFollower, McaFollower, MpcFollower, MppiFollower, SocFollower, TebFollower,
};
use crate::types::{
    ControllerConfig, ControllerStatus, Goal, Path, RobotConstraints, RobotState, Trajectory,
    VelocityCommand, WorldConstraints,
};
use datapod::{Euler, Point, Pose, Quaternion};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TrackerKind {
    Pid,
    Carrot,
    PurePursuit,
    Stanley,
    Lqr,
    Mpc,
    Mppi,
    Mca,
    Soc,
    Dwa,
    Teb,
    Flc,
    RegulatedPursuit,
    PoseReach,
    Ilqr,
    PoseRegulator,
    VectorPursuit,
    Kanayama,
}

impl TrackerKind {
    /// Point-to-point controllers are driven through a path waypoint by
    /// waypoint by the tracker; path followers consume the whole path.
    pub fn is_point_controller(self) -> bool {
        matches!(self, TrackerKind::Pid | TrackerKind::Carrot | TrackerKind::Dwa | TrackerKind::PoseReach | TrackerKind::PoseRegulator)
    }
}

pub struct Tracker {
    kind: TrackerKind,
    controller: Box<dyn Controller>,
    constraints: RobotConstraints,
    goal: Option<Goal>,
    path: Option<Path>,
    cursor: PathCursor,
    waypoint_index: usize,
    projection_segment: usize,
    path_completed: bool,
    trajectory: Option<Trajectory>,
    clock: f64,
    clock_origin: Option<f64>,
}

impl Tracker {
    pub fn new(kind: TrackerKind) -> Self {
        let controller = make_controller(kind);
        Self {
            kind,
            controller,
            constraints: RobotConstraints::default(),
            goal: None,
            path: None,
            cursor: PathCursor::default(),
            waypoint_index: 0,
            projection_segment: 0,
            path_completed: false,
            trajectory: None,
            clock: 0.0,
            clock_origin: None,
        }
    }

    pub fn with_config(kind: TrackerKind, config: ControllerConfig) -> Self {
        let mut t = Self::new(kind);
        t.controller.set_config(config);
        t
    }

    pub fn init(&mut self, constraints: RobotConstraints) {
        self.constraints = constraints;
    }

    pub fn kind(&self) -> TrackerKind {
        self.kind
    }

    pub fn constraints(&self) -> &RobotConstraints {
        &self.constraints
    }

    pub fn set_goal(&mut self, goal: Goal) {
        self.goal = Some(goal);
    }

    pub fn goal(&self) -> Option<&Goal> {
        self.goal.as_ref()
    }

    pub fn clear_goal(&mut self) {
        self.goal = None;
    }

    pub fn set_path(&mut self, path: Path) {
        self.trajectory = None;
        self.clock = 0.0;
        self.clock_origin = None;
        self.cursor.set_path(&path.waypoints);
        self.path = Some(path.clone());
        self.waypoint_index = 0;
        self.projection_segment = 0;
        self.path_completed = false;
        self.controller.set_path(path);
    }

    pub fn path(&self) -> Option<&Path> {
        self.path.as_ref()
    }

    /// Install a timed trajectory. Its poses become the path; the tracker
    /// clock restarts and advances by `dt` every tick (or follows
    /// `RobotState::timestamp` when that is provided and increasing).
    pub fn set_trajectory(&mut self, trajectory: Trajectory) {
        self.cursor.set_path(&trajectory.poses);
        self.path = Some(trajectory.to_path());
        self.waypoint_index = 0;
        self.projection_segment = 0;
        self.path_completed = false;
        self.clock = 0.0;
        self.clock_origin = None;
        self.controller.set_trajectory(trajectory.clone());
        self.trajectory = Some(trajectory);
    }

    pub fn trajectory(&self) -> Option<&Trajectory> {
        self.trajectory.as_ref()
    }

    /// Seconds elapsed on the installed trajectory.
    pub fn trajectory_time(&self) -> f64 {
        self.clock
    }

    pub fn clear_path(&mut self) {
        self.path = None;
        self.trajectory = None;
        self.clock = 0.0;
        self.clock_origin = None;
        self.waypoint_index = 0;
        self.projection_segment = 0;
        self.path_completed = false;
        self.controller.set_path(Path::default());
    }

    pub fn set_config(&mut self, config: ControllerConfig) {
        self.controller.set_config(config);
    }

    pub fn get_config(&self) -> ControllerConfig {
        self.controller.get_config()
    }

    pub fn get_status(&self) -> ControllerStatus {
        self.controller.get_status()
    }

    pub fn controller(&self) -> &dyn Controller {
        self.controller.as_ref()
    }

    pub fn controller_mut(&mut self) -> &mut dyn Controller {
        self.controller.as_mut()
    }

    pub fn reset(&mut self) {
        self.controller.reset();
        self.waypoint_index = 0;
        self.path_completed = false;
    }

    /// One control period. Order of precedence:
    /// 1. `dt` must be positive and finite;
    /// 2. `allow_move == false` yields a valid zero command;
    /// 3. an explicit goal is used as-is; otherwise a goal is derived from
    ///    the path (the current waypoint for point controllers, the path
    ///    end for path followers);
    /// 4. with neither goal nor path the command is invalid.
    pub fn tick(
        &mut self,
        state: &RobotState,
        dt: f64,
        world: Option<&WorldConstraints>,
    ) -> VelocityCommand {
        if !(dt.is_finite() && dt > 0.0) {
            return VelocityCommand::invalid("dt must be positive and finite");
        }
        if !state.allow_move {
            return stop("Movement disabled");
        }
        if self.trajectory.is_some() {
            if state.timestamp > 0.0 {
                let origin = *self.clock_origin.get_or_insert(state.timestamp);
                self.clock = (state.timestamp - origin).max(self.clock);
            } else {
                self.clock += dt;
            }
            self.controller.set_time(self.clock);
        }

        let goal = match self.resolve_goal(state, world) {
            Ok(Some(g)) => g,
            Ok(None) => return stop("Path completed"),
            Err(msg) => return VelocityCommand::invalid(msg),
        };

        let cmd = self
            .controller
            .compute_control(state, &goal, &self.constraints, dt, world);

        if self.goal.is_none() && self.path.is_some() && self.controller.get_status().goal_reached
        {
            self.path_completed = true;
        }
        cmd
    }

    fn resolve_goal(
        &mut self,
        state: &RobotState,
        world: Option<&WorldConstraints>,
    ) -> Result<Option<Goal>, &'static str> {
        if let Some(g) = &self.goal {
            return Ok(Some(g.clone()));
        }
        let Some(path) = self.path.as_ref() else {
            return Err("no goal set");
        };
        if path.waypoints.is_empty() {
            return Err("path is empty");
        }
        if self.path_completed {
            return Ok(None);
        }
        let cfg = self.controller.get_config();

        if self.kind.is_point_controller() {
            let n = path.waypoints.len();
            let hint = self.projection_segment.min(n.saturating_sub(2));
            let Some(proj) = self.cursor.project(&path.waypoints, state.pose.point, hint) else {
                return Err("path is empty");
            };
            self.projection_segment = proj.segment;
            let cum = &self.cursor.cum;
            let tol = cfg.goal_tolerance;
            let ahead = cfg.lookahead_distance.max(2.0 * tol);
            let s_target = proj.arc_length + ahead;
            let mut idx = cum
                .iter()
                .position(|s| *s >= s_target - 1e-9)
                .unwrap_or(n - 1)
                .max(proj.segment + 1)
                .min(n - 1);
            if let Some(w) = world {
                let robot_r = 0.5 * self.constraints.robot_width.hypot(self.constraints.robot_length);
                let robot_r = if robot_r < 0.05 { 0.3 } else { robot_r };
                while idx + 1 < n && blocked(&path.waypoints[idx].point, w, robot_r) {
                    idx += 1;
                }
            }
            let last = idx + 1 >= n;
            let wp = path.waypoints[idx];
            let yaw = if last {
                end_heading(&path.waypoints)
            } else {
                segment_heading(&path.waypoints, idx)
            };
            self.waypoint_index = idx;
            return Ok(Some(Goal {
                target_pose: Pose {
                    point: wp.point,
                    rotation: Quaternion::from_euler(Euler::new(0.0, 0.0, yaw)),
                },
                target_velocity: None,
                tolerance_position: tol,
                tolerance_orientation: if last {
                    cfg.angular_tolerance
                } else {
                    std::f64::consts::PI
                },
            }));
        }

        let last = *path.waypoints.last().unwrap();
        Ok(Some(Goal {
            target_pose: Pose {
                point: last.point,
                rotation: Quaternion::from_euler(Euler::new(0.0, 0.0, end_heading(&path.waypoints))),
            },
            target_velocity: None,
            tolerance_position: cfg.goal_tolerance,
            tolerance_orientation: cfg.angular_tolerance,
        }))
    }

    /// Planned trajectory of the underlying controller, if it predicts one.
    pub fn predicted_trajectory(&self) -> Vec<Point> {
        self.controller.predicted_trajectory()
    }

    /// True if the last tick put the controller in the goal-reached state.
    pub fn is_goal_reached(&self) -> bool {
        self.controller.get_status().goal_reached
    }

    /// True once a path driven without an explicit goal has been consumed.
    pub fn is_path_completed(&self) -> bool {
        self.path_completed
    }

    /// The point the tracker is currently aiming at: the active goal's
    /// target point if a goal is set, otherwise the current path waypoint,
    /// otherwise `None`.
    pub fn current_target(&self) -> Option<Point> {
        if let Some(g) = &self.goal {
            return Some(g.target_pose.point);
        }
        let path = self.path.as_ref()?;
        let idx = if self.kind.is_point_controller() {
            self.waypoint_index
        } else {
            self.controller.get_path_index()
        }
        .min(path.waypoints.len().saturating_sub(1));
        Some(path.waypoints.get(idx)?.point)
    }

    /// Distance from `pose` to the active goal, if any.
    pub fn distance_to_goal(&self, pose: &Pose) -> Option<f64> {
        let g = self.goal.as_ref()?;
        Some(check_goal(pose, g, &self.controller.get_config()).distance)
    }

    /// Clear the goal and path, reset the controller, and return a safe
    /// zero-velocity command.
    pub fn emergency_stop(&mut self) -> VelocityCommand {
        self.clear_goal();
        self.clear_path();
        self.controller.reset();
        stop("Emergency stop")
    }

    /// Densify the current path by inserting interpolated poses so that
    /// no segment is longer than `max_segment_m`. Linear interpolation for
    /// positions, slerp for orientations. The controller receives the
    /// updated path and its path index is reset to 0.
    pub fn smoothen(&mut self, max_segment_m: f64) {
        let Some(path) = self.path.as_mut() else {
            return;
        };
        if path.waypoints.len() < 2 || max_segment_m <= 0.0 {
            return;
        }
        smoothen_path(path, max_segment_m);
        self.cursor.set_path(&path.waypoints);
        self.waypoint_index = 0;
        self.projection_segment = 0;
        self.controller.set_path(path.clone());
    }
}

fn densify_path(waypoints: &[Pose], speeds: &[f64], max_segment_m: f64) -> (Vec<Pose>, Vec<f64>) {
    let mut out = Vec::with_capacity(waypoints.len() * 2);
    let mut out_speeds = Vec::new();
    let has_speeds = speeds.len() == waypoints.len();
    out.push(waypoints[0]);
    if has_speeds {
        out_speeds.push(speeds[0]);
    }
    for i in 0..waypoints.len() - 1 {
        let start = waypoints[i];
        let end = waypoints[i + 1];
        let dx = end.point.x - start.point.x;
        let dy = end.point.y - start.point.y;
        let dz = end.point.z - start.point.z;
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();

        let n = ((dist / max_segment_m).ceil() as usize).max(1);
        for j in 1..n {
            let t = j as f64 / n as f64;
            let p = Point::new(
                start.point.x + t * dx,
                start.point.y + t * dy,
                start.point.z + t * dz,
            );
            let rot = slerp(start.rotation, end.rotation, t);
            out.push(Pose {
                point: p,
                rotation: rot,
            });
            if has_speeds {
                out_speeds.push(speeds[i] + t * (speeds[i + 1] - speeds[i]));
            }
        }
        out.push(end);
        if has_speeds {
            out_speeds.push(speeds[i + 1]);
        }
    }
    (out, out_speeds)
}

fn slerp(a: Quaternion, b: Quaternion, t: f64) -> Quaternion {
    let mut cos_half = a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z;
    let (bw, bx, by, bz) = if cos_half < 0.0 {
        cos_half = -cos_half;
        (-b.w, -b.x, -b.y, -b.z)
    } else {
        (b.w, b.x, b.y, b.z)
    };

    if cos_half > 0.9995 {
        let w = a.w + t * (bw - a.w);
        let x = a.x + t * (bx - a.x);
        let y = a.y + t * (by - a.y);
        let z = a.z + t * (bz - a.z);
        let n = (w * w + x * x + y * y + z * z).sqrt().max(1e-12);
        return Quaternion {
            w: w / n,
            x: x / n,
            y: y / n,
            z: z / n,
        };
    }

    let half = cos_half.acos();
    let sin_half = half.sin();
    let r0 = ((1.0 - t) * half).sin() / sin_half;
    let r1 = (t * half).sin() / sin_half;
    Quaternion {
        w: r0 * a.w + r1 * bw,
        x: r0 * a.x + r1 * bx,
        y: r0 * a.y + r1 * by,
        z: r0 * a.z + r1 * bz,
    }
}

/// Free-function path densification for callers that don't use `Tracker`.
/// Per-waypoint speeds are interpolated when present.
pub fn smoothen_path(path: &mut Path, max_segment_m: f64) {
    if path.waypoints.len() < 2 || max_segment_m <= 0.0 {
        return;
    }
    let (wps, speeds) = densify_path(&path.waypoints, &path.speeds, max_segment_m);
    path.waypoints = wps;
    if !speeds.is_empty() {
        path.speeds = speeds;
    }
}

fn make_controller(kind: TrackerKind) -> Box<dyn Controller> {
    match kind {
        TrackerKind::Pid => Box::new(PidFollower::new()),
        TrackerKind::Carrot => Box::new(CarrotFollower::new()),
        TrackerKind::PurePursuit => Box::new(PurePursuitFollower::new()),
        TrackerKind::Stanley => Box::new(StanleyFollower::new()),
        TrackerKind::Lqr => Box::new(LqrFollower::new()),
        TrackerKind::Mpc => Box::new(MpcFollower::new()),
        TrackerKind::Mppi => Box::new(MppiFollower::new()),
        TrackerKind::Mca => Box::new(McaFollower::new()),
        TrackerKind::Soc => Box::new(SocFollower::new()),
        TrackerKind::Dwa => Box::new(DwaFollower::new()),
        TrackerKind::Teb => Box::new(TebFollower::new()),
        TrackerKind::Flc => Box::new(FlcFollower::new()),
        TrackerKind::RegulatedPursuit => Box::new(RegulatedPursuitFollower::new()),
        TrackerKind::PoseReach => Box::new(PoseReachFollower::new()),
        TrackerKind::Ilqr => Box::new(IlqrFollower::new()),
        TrackerKind::PoseRegulator => Box::new(PoseRegulatorFollower::new()),
        TrackerKind::VectorPursuit => Box::new(VectorPursuitFollower::new()),
        TrackerKind::Kanayama => Box::new(KanayamaFollower::new()),
    }
}

/// True when `p` lies inside any obstacle's current footprint inflated by
/// the robot radius and a safety margin.
fn blocked(p: &Point, world: &WorldConstraints, robot_radius: f64) -> bool {
    const MARGIN: f64 = 0.3;
    world.obstacles.iter().any(|o| {
        o.modes.iter().any(|m| {
            m.weight > 0.0
                && !m.mean_x.is_empty()
                && !m.mean_y.is_empty()
                && (p.x - m.mean_x[0]).hypot(p.y - m.mean_y[0]) < o.radius + robot_radius + MARGIN
        })
    })
}
