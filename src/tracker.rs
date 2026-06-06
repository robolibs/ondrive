use crate::controller::Controller;
use crate::fuzzy::FlcFollower;
use crate::path::{LqrFollower, PurePursuitFollower, StanleyFollower};
use crate::point::{CarrotFollower, PidFollower};
use crate::pred::{DwaFollower, McaFollower, MpcFollower, MppiFollower, SocFollower, TebFollower};
use crate::types::{
    ControllerConfig, ControllerStatus, Goal, Path, RobotConstraints, RobotState, VelocityCommand,
    WorldConstraints,
};
use datapod::{Point, Pose, Quaternion};

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
}

pub struct Tracker {
    kind: TrackerKind,
    controller: Box<dyn Controller>,
    constraints: RobotConstraints,
    goal: Option<Goal>,
    path: Option<Path>,
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
        self.path = Some(path.clone());
        self.controller.set_path(path);
    }

    pub fn path(&self) -> Option<&Path> {
        self.path.as_ref()
    }

    pub fn clear_path(&mut self) {
        self.path = None;
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

    pub fn reset(&mut self) {
        self.controller.reset();
    }

    pub fn tick(
        &mut self,
        state: &RobotState,
        dt: f64,
        world: Option<&WorldConstraints>,
    ) -> VelocityCommand {
        let Some(goal) = self.goal.as_ref() else {
            return VelocityCommand::invalid("no goal set");
        };
        self.controller
            .compute_control(state, goal, &self.constraints, dt, world)
    }

    /// True if the last tick put the controller in the goal-reached state.
    pub fn is_goal_reached(&self) -> bool {
        self.controller.get_status().goal_reached
    }

    /// The point the tracker is currently aiming at: the active goal's
    /// target point if a goal is set, otherwise the current path waypoint,
    /// otherwise `None`.
    pub fn current_target(&self) -> Option<Point> {
        if let Some(g) = &self.goal {
            return Some(g.target_pose.point);
        }
        let path = self.path.as_ref()?;
        let idx = self
            .controller
            .get_path_index()
            .min(path.waypoints.len().saturating_sub(1));
        Some(path.waypoints.get(idx)?.point)
    }

    /// Clear the goal and path, reset the controller, and return a safe
    /// zero-velocity command.
    pub fn emergency_stop(&mut self) -> VelocityCommand {
        self.clear_goal();
        self.clear_path();
        self.controller.reset();
        VelocityCommand {
            valid: true,
            status_message: "Emergency stop".into(),
            ..VelocityCommand::default()
        }
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

        let densified = densify_path(&path.waypoints, max_segment_m);
        path.waypoints = densified;
        self.controller.set_path(path.clone());
    }
}

fn densify_path(waypoints: &[Pose], max_segment_m: f64) -> Vec<Pose> {
    let mut out = Vec::with_capacity(waypoints.len() * 2);
    out.push(waypoints[0]);
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
        }
        out.push(end);
    }
    out
}

fn slerp(a: Quaternion, b: Quaternion, t: f64) -> Quaternion {
    // Spherical linear interpolation between unit quaternions. We hand-roll
    // it so we don't assume a particular helper on the datapod side.
    let mut cos_half = a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z;
    let (bw, bx, by, bz) = if cos_half < 0.0 {
        cos_half = -cos_half;
        (-b.w, -b.x, -b.y, -b.z)
    } else {
        (b.w, b.x, b.y, b.z)
    };

    if cos_half > 0.9995 {
        // Linear fall-back when quaternions are nearly identical.
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
pub fn smoothen_path(path: &mut Path, max_segment_m: f64) {
    if path.waypoints.len() < 2 || max_segment_m <= 0.0 {
        return;
    }
    path.waypoints = densify_path(&path.waypoints, max_segment_m);
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
    }
}
