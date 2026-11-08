use crate::controller::{Controller, check_goal};
use crate::core::kinematics::{actuation_delay, curvature_to_steering, is_ackermann, is_holonomic, stop, wheelbase};
use crate::core::obstacles::CollisionChecker;
use crate::core::path::{PathCursor, cumulative_lengths, cusp_after, end_heading, next_cusp, sample, segment_heading};
use crate::fuzzy::FlcFollower;
use crate::path::{
    IlcFollower, KanayamaFollower, LqrFollower, PurePursuitFollower, RegulatedPursuitFollower,
    StanleyFollower, VectorPursuitFollower,
};
use crate::point::{
    ApfFollower, CarrotFollower, PidFollower, PoseReachFollower, PoseRegulatorFollower,
};
use crate::pred::{
    DwaFollower, IlqrFollower, McaFollower, MpcFollower, MppiFollower, SocFollower, TebFollower,
};
use crate::types::{
    ControllerConfig, ControllerStatus, Goal, OutputUnits, Path, RobotConstraints, RobotState, Trajectory,
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
    Apf,
    Ilc,
}

impl TrackerKind {
    /// Point-to-point controllers are driven through a path waypoint by
    /// waypoint by the tracker; path followers consume the whole path.
    pub fn is_point_controller(self) -> bool {
        matches!(self, TrackerKind::Pid | TrackerKind::Carrot | TrackerKind::Dwa | TrackerKind::PoseReach | TrackerKind::PoseRegulator | TrackerKind::Apf)
    }

    /// Whether the controller models the machine's actuation lag itself;
    /// the tracker hands every other one the state predicted through it.
    pub fn models_actuation(self) -> bool {
        matches!(self, TrackerKind::Mpc | TrackerKind::Mppi | TrackerKind::Mca | TrackerKind::Soc | TrackerKind::Ilqr)
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
    /// Side a point controller last went round obstacles on: +1 left, −1
    /// right, 0 none.
    detour_side: f64,
    projection_segment: usize,
    path_completed: bool,
    /// Whether the goal last handed to the controller is the path's end.
    aiming_at_end: bool,
    /// The goal point last handed to the controller.
    aim: Option<Point>,
    trajectory: Option<Trajectory>,
    clock: f64,
    clock_origin: Option<f64>,
    path_error: Option<&'static str>,
    lap: Option<Lap>,
    last_steering: Option<f64>,
    /// Last command sent, physical `(v, omega, vy)`.
    last_command: Option<(f64, f64, f64)>,
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
            detour_side: 0.0,
            projection_segment: 0,
            path_completed: false,
            aiming_at_end: true,
            aim: None,
            trajectory: None,
            clock: 0.0,
            clock_origin: None,
            path_error: None,
            lap: None,
            last_steering: None,
            last_command: None,
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

    /// Install a path. A closed path (`Path::is_closed`) is driven lap after
    /// lap and never completes.
    pub fn set_path(&mut self, path: Path) {
        self.trajectory = None;
        self.clock = 0.0;
        self.clock_origin = None;
        self.path_error = path_error(&path.waypoints, &path.speeds, None);
        self.lap = Lap::of(&path);
        let driven = self.lap.as_ref().map_or(path, |lap| lap.unrolled.clone());
        self.drive(driven);
    }

    /// Hand `path` to the cursor and the controller, with progress at its start.
    fn drive(&mut self, path: Path) {
        self.cursor.set_path(&path.waypoints);
        self.path = Some(path.clone());
        self.waypoint_index = 0;
        self.projection_segment = 0;
        self.path_completed = false;
        self.aim = None;
        self.controller.set_path(path);
    }

    pub fn path(&self) -> Option<&Path> {
        self.lap.as_ref().map(|lap| &lap.source).or(self.path.as_ref())
    }

    /// Install a timed trajectory. Its poses become the path; the tracker
    /// clock restarts and advances by `dt` every tick (or follows
    /// `RobotState::timestamp` when that is provided and increasing).
    pub fn set_trajectory(&mut self, trajectory: Trajectory) {
        self.cursor.set_path(&trajectory.poses);
        self.path_error = path_error(&trajectory.poses, &trajectory.speeds, Some(&trajectory.times));
        self.lap = None;
        self.path = Some(trajectory.to_path());
        self.waypoint_index = 0;
        self.projection_segment = 0;
        self.path_completed = false;
        self.aim = None;
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
        self.path_error = None;
        self.lap = None;
        self.trajectory = None;
        self.clock = 0.0;
        self.clock_origin = None;
        self.waypoint_index = 0;
        self.projection_segment = 0;
        self.path_completed = false;
        self.aim = None;
        self.controller.set_path(Path::default());
    }

    pub fn set_config(&mut self, config: ControllerConfig) {
        self.controller.set_config(config);
    }

    pub fn get_config(&self) -> ControllerConfig {
        self.controller.get_config()
    }

    /// The controller's own parameters (`MpcConfig`, `McaConfig`, ...) as a
    /// JSON object; empty for a controller that has none.
    pub fn params(&self) -> serde_json::Value {
        self.controller.params()
    }

    /// Update some of the controller's own parameters from a JSON object,
    /// e.g. `{"horizon_steps": 30}`. An unknown name or a wrong type is an
    /// error and changes nothing.
    pub fn set_params(&mut self, params: &serde_json::Value) -> Result<(), String> {
        self.controller.set_params(params)
    }

    /// Seed the random sampling of MPPI, MCA and SOC; other controllers
    /// ignore it.
    pub fn set_seed(&mut self, seed: u64) {
        self.controller.set_seed(seed);
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

    /// Clear the controller's internal state and restart progress on the
    /// installed path or trajectory, which stays installed.
    pub fn reset(&mut self) {
        self.controller.reset();
        if let Some(trajectory) = self.trajectory.clone() {
            self.set_trajectory(trajectory);
        } else if let Some(path) = self.path().cloned() {
            self.set_path(path);
        }
        self.waypoint_index = 0;
        self.path_completed = false;
        self.aim = None;
        self.last_steering = None;
        self.last_command = None;
    }

    /// One control period. Order of precedence:
    /// 1. `dt` must be positive and finite;
    /// 2. the state, goal, path, constraints and config must be free of
    ///    NaN (and the state, goal and path of infinities);
    /// 3. `allow_move == false` yields a valid zero command;
    /// 4. an explicit goal is used as-is; otherwise a goal is derived from
    ///    the path (the current waypoint for point controllers, the path
    ///    end for path followers);
    /// 5. with neither goal nor path the command is invalid.
    pub fn tick(
        &mut self,
        state: &RobotState,
        dt: f64,
        world: Option<&WorldConstraints>,
    ) -> VelocityCommand {
        if !(dt.is_finite() && dt > 0.0) {
            return VelocityCommand::invalid("dt must be positive and finite");
        }
        let input_error = state_error(state)
            .or_else(|| self.goal.as_ref().and_then(goal_error))
            .or(self.path_error)
            .or_else(|| constraints_error(&self.constraints))
            .or_else(|| config_error(&self.controller.get_config()));
        if let Some(msg) = input_error {
            return VelocityCommand::invalid(msg);
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

        self.next_lap(state);
        let goal = match self.resolve_goal(state, world) {
            Ok(Some(g)) => g,
            Ok(None) => return stop("Path completed"),
            Err(msg) => return VelocityCommand::invalid(msg),
        };
        self.aim = Some(goal.target_pose.point);

        let ahead = (!self.kind.models_actuation())
            .then(|| predicted_through_actuation(state, &self.constraints, self.last_command))
            .flatten();
        let cmd = self
            .controller
            .compute_control(ahead.as_ref().unwrap_or(state), &goal, &self.constraints, dt, world);

        if self.goal.is_none() && self.path.is_some() && self.aiming_at_end && self.controller.get_status().goal_reached {
            self.path_completed = true;
        }
        self.limit(cmd, state, dt, world)
    }

    /// The command held to what the platform can do: `max_linear_acceleration`
    /// and `max_angular_acceleration` around the last command, keeping the
    /// commanded curvature where the window allows (dynamic-window pure
    /// pursuit); then the speed limit of the zones the robot is in; then the
    /// Ackermann steering angle to `max_steering_rate`, held while standing.
    /// A full stop passes unchanged.
    fn limit(
        &mut self,
        mut cmd: VelocityCommand,
        state: &RobotState,
        dt: f64,
        world: Option<&WorldConstraints>,
    ) -> VelocityCommand {
        if !cmd.valid {
            return cmd;
        }
        let rate_cap = if self.kind.models_actuation() { None } else { self.steering_rate_speed(state) };
        let c = &self.constraints;
        let (linear_unit, angular_unit) = match self.controller.get_config().output_units {
            OutputUnits::Physical => (1.0, 1.0),
            OutputUnits::Normalized => (c.max_linear_velocity.max(0.0), c.max_angular_velocity.max(0.0)),
        };
        let holonomic = is_holonomic(c.steering_type);
        let mut v = cmd.linear_velocity * linear_unit;
        let mut vy = cmd.lateral_velocity * linear_unit;
        let mut w = cmd.angular_velocity * angular_unit;
        let stopping = v == 0.0 && vy == 0.0 && w == 0.0;

        if let Some((pv, pw, pvy)) = self.last_command.filter(|_| !stopping) {
            let a = if c.max_linear_acceleration > 0.0 { c.max_linear_acceleration * dt } else { f64::INFINITY };
            let alpha = if c.max_angular_acceleration > 0.0 { c.max_angular_acceleration * dt } else { f64::INFINITY };
            let (v_lo, v_hi, w_lo, w_hi) = (pv - a, pv + a, pw - alpha, pw + alpha);
            if holonomic || v.abs() < 1e-6 {
                v = v.clamp(v_lo, v_hi);
                vy = vy.clamp(pvy - a, pvy + a);
                w = w.clamp(w_lo, w_hi);
            } else {
                let k = w / v;
                let (mut lo, mut hi) = (v_lo, v_hi);
                if k.abs() > 1e-9 {
                    let (a1, a2) = (w_lo / k, w_hi / k);
                    lo = lo.max(a1.min(a2));
                    hi = hi.min(a1.max(a2));
                }
                if lo <= hi {
                    v = v.clamp(lo, hi);
                    w = k * v;
                } else {
                    v = v.clamp(v_lo, v_hi);
                    w = (k * v).clamp(w_lo, w_hi);
                }
            }
        }

        let zone_cap = world.and_then(|w| w.speed_limit_at(state.pose.point));
        if let Some(cap) = rate_cap.into_iter().chain(zone_cap).reduce(f64::min) {
            let speed = v.hypot(vy);
            if speed > cap {
                let scale = cap / speed;
                v *= scale;
                vy *= scale;
                if !holonomic {
                    w *= scale;
                }
            }
        }

        if is_ackermann(c.steering_type) {
            if v.abs() > 1e-6 {
                let mut delta = curvature_to_steering(w / v, c);
                if c.max_steering_rate > 0.0
                    && let Some(prev) = self.last_steering
                {
                    let step = c.max_steering_rate * dt;
                    delta = delta.clamp(prev - step, prev + step);
                    let w_max = c.max_angular_velocity.max(0.0);
                    w = (v * delta.tan() / wheelbase(c)).clamp(-w_max, w_max);
                    delta = curvature_to_steering(w / v, c);
                }
                self.last_steering = Some(delta);
                cmd.steering_angle = delta;
            } else if let Some(prev) = self.last_steering {
                cmd.steering_angle = prev;
            }
        }

        self.last_command = Some((v, w, vy));
        cmd.linear_velocity = if linear_unit > 0.0 { v / linear_unit } else { 0.0 };
        cmd.lateral_velocity = if linear_unit > 0.0 { vy / linear_unit } else { 0.0 };
        cmd.angular_velocity = if angular_unit > 0.0 { w / angular_unit } else { 0.0 };
        cmd
    }

    /// Highest speed at which a rate-limited Ackermann steering keeps up with
    /// the path's change of curvature over the stretch ahead (up to the next
    /// cusp): `rate / (L · max |dκ/ds|)`, curvature measured over a
    /// wheelbase; never below a share of the top speed.
    fn steering_rate_speed(&mut self, state: &RobotState) -> Option<f64> {
        let c = &self.constraints;
        if !is_ackermann(c.steering_type) || c.max_steering_rate <= 0.0 {
            return None;
        }
        let path = self.path.as_ref()?;
        let proj = self.cursor.project(&path.waypoints, state.pose.point, self.projection_segment)?;
        let cum = &self.cursor.cum;
        let total = cum.last().copied().unwrap_or(0.0);
        let l = wheelbase(c);
        let span = l.max(0.5);
        let end = (proj.arc_length + (c.max_linear_velocity * RATE_PREVIEW_S).max(2.0 * l))
            .min(cusp_after(&path.speeds, cum, proj.arc_length).unwrap_or(total))
            .min(total);
        let heading = |s: f64| sample(&path.waypoints, cum, s.clamp(0.0, total)).1;
        let kappa = |s: f64| crate::core::math::normalize_angle(heading(s + 0.5 * span) - heading(s - 0.5 * span)) / span;
        let mut worst = 0.0_f64;
        let mut s = proj.arc_length;
        while s + span <= end {
            worst = worst.max(((kappa(s + span) - kappa(s)) / span).abs());
            s += 0.5 * span;
        }
        (worst > 1e-9).then(|| (c.max_steering_rate / (l * worst)).max(RATE_FLOOR * c.max_linear_velocity))
    }

    /// On a closed path, restart the unrolled laps once the robot is into
    /// the second.
    fn next_lap(&mut self, state: &RobotState) {
        let (Some(lap), Some(path)) = (&self.lap, &self.path) else {
            return;
        };
        let Some(proj) = self.cursor.project(&path.waypoints, state.pose.point, self.projection_segment) else {
            return;
        };
        self.projection_segment = proj.segment;
        if proj.arc_length >= lap.length {
            let unrolled = lap.unrolled.clone();
            self.drive(unrolled);
        }
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
        self.aiming_at_end = true;

        if self.kind.is_point_controller() {
            let n = path.waypoints.len();
            let hint = self.projection_segment.min(n.saturating_sub(2));
            let Some(proj) = self.cursor.project(&path.waypoints, state.pose.point, hint) else {
                return Err("path is empty");
            };
            let cum = &self.cursor.cum;
            let tol = cfg.goal_tolerance;
            if self.lap.is_none() && proj.beyond_end && proj.distance < 2.0 * tol {
                self.path_completed = true;
                return Ok(None);
            }
            let speed_now = if state.velocity_valid { state.velocity.linear } else { self.last_command.map_or(0.0, |c| c.0) };
            let ahead = cfg.lookahead_distance.max(2.0 * tol);
            let target = |segment: usize, s: f64| {
                let idx = cum.iter().position(|c| *c >= s - 1e-9).unwrap_or(n - 1).max(segment + 1).min(n - 1);
                let cap = next_cusp(&path.waypoints, segment).unwrap_or(n - 1);
                (idx.min(cap), cap)
            };
            // The robot drives to a cusp, then on along the leg that leaves it.
            let mut segment = proj.segment;
            let (mut idx, mut cap) = target(segment, proj.arc_length + ahead);
            if cap < n - 1 && idx == cap && path.waypoints[cap].point.distance_to_2d(state.pose.point) < tol {
                segment = cap;
                (idx, cap) = target(segment, cum[segment] + ahead);
            }
            self.projection_segment = segment;
            let detour = world
                .filter(|w| !w.obstacles.is_empty())
                .and_then(|w| {
                    detour_target(path, cum, proj.arc_length, cum[idx], state.pose.point, tol, &w.obstacles, &self.constraints, speed_now, &mut self.detour_side)
                });
            if let Some(point) = detour {
                self.aiming_at_end = false;
                let heading = segment_heading(&path.waypoints, idx.min(n.saturating_sub(2)));
                return Ok(Some(Goal {
                    target_pose: Pose { point, rotation: Quaternion::from_euler(Euler::new(0.0, 0.0, heading)) },
                    target_velocity: None,
                    tolerance_position: tol,
                    tolerance_orientation: std::f64::consts::PI,
                }));
            }
            if world.is_some() {
                let checker = CollisionChecker::new(world, &self.constraints, WAYPOINT_MARGIN);
                while idx < cap && blocked(&path.waypoints, idx, &checker) {
                    idx += 1;
                }
            }
            let last = idx + 1 >= n;
            self.aiming_at_end = last;
            let wp = path.waypoints[idx];
            let yaw = if last {
                end_heading(&path.waypoints)
            } else if idx == cap {
                segment_heading(&path.waypoints, idx - 1)
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

    /// How much of its internals the controller records each tick.
    pub fn set_debug(&mut self, level: crate::types::DebugLevel) {
        self.controller.set_debug(level);
    }

    /// The controller's internals from its last tick: sampled or solved
    /// trajectories with their costs, and the guides it steered by.
    pub fn debug_frame(&self) -> crate::types::DebugFrame {
        self.controller.debug_frame()
    }

    /// The point the tracker is currently aiming at: the active goal's
    /// target point if a goal is set, otherwise the point the last tick
    /// handed the controller (a waypoint or a detour round an obstacle),
    /// otherwise the current path waypoint, otherwise `None`.
    pub fn current_target(&self) -> Option<Point> {
        if let Some(g) = &self.goal {
            return Some(g.target_pose.point);
        }
        if let Some(p) = self.aim.filter(|_| !self.path_completed) {
            return Some(p);
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
        if let Some(lap) = &self.lap {
            let mut ring = lap.ring.clone();
            smoothen_path(&mut ring, max_segment_m);
            self.set_path(ring);
            return;
        }
        let Some(path) = self.path.as_mut() else {
            return;
        };
        if path.waypoints.len() < 2 || max_segment_m <= 0.0 {
            return;
        }
        smoothen_path(path, max_segment_m);
        self.cursor.set_path(&path.waypoints);
        self.path_error = path_error(&path.waypoints, &path.speeds, None);
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
                let magnitude = speeds[i].abs() + t * (speeds[i + 1].abs() - speeds[i].abs());
                out_speeds.push(if speeds[i + 1] < 0.0 { -magnitude } else { magnitude });
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
/// Per-waypoint speeds are interpolated when present: the magnitude
/// linearly, the sign that of the waypoint each new point leads to.
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
        TrackerKind::Apf => Box::new(ApfFollower::new()),
        TrackerKind::Ilc => Box::new(IlcFollower::default()),
    }
}

/// Safety margin around obstacles when choosing a waypoint target.
const WAYPOINT_MARGIN: f64 = 0.3;

/// True when the robot footprint placed at waypoint `idx` collides now.
fn blocked(waypoints: &[Pose], idx: usize, checker: &CollisionChecker) -> bool {
    let p = waypoints[idx].point;
    let heading = segment_heading(waypoints, idx.min(waypoints.len().saturating_sub(2)));
    checker.collides(0, p.x, p.y, heading)
}

/// Where a point controller heads while obstacles block the path between
/// arc lengths `from` (the robot) and `to` (its target): the target moved
/// onto the cheaper detour round the obstacles' predictions, the robot
/// taken to drive the path at `speed_now` (no slower than a share of its
/// top speed), or the detour's
/// widest point when that comes first and is still ahead of `robot` by
/// more than `tol`. `side` (+1 left, −1 right, 0 none) is the side last
/// taken, kept unless the other is cheaper by `SIDE_SWITCH_COST`; `None`
/// when no detour is needed.
#[allow(clippy::too_many_arguments)]
fn detour_target(
    path: &Path,
    cum: &[f64],
    from: f64,
    to: f64,
    robot: Point,
    tol: f64,
    obstacles: &[crate::types::Obstacle],
    constraints: &RobotConstraints,
    speed_now: f64,
    side: &mut f64,
) -> Option<Point> {
    use crate::pred::mpc::{DETOUR_MARGIN, DETOUR_RAMP, SIDE_SWITCH_COST};
    let top = constraints.max_linear_velocity.abs();
    let speed = speed_now.abs().clamp(DETOUR_SPEED_FLOOR * top, top).max(0.1);
    let target_step = ((to - from).max(0.0) / (speed * DETOUR_DT)).round() as usize;
    let n = (target_step + DETOUR_RAMP).clamp(2, 80);
    let reference = crate::pred::mppi::build_reference(path, cum, from, n, DETOUR_DT, speed, 0.0);
    let along: Vec<crate::types::Obstacle> = obstacles
        .iter()
        .filter(|o| !crate::pred::mppi::crosses(o, &reference, DETOUR_DT))
        .cloned()
        .collect();
    if along.is_empty() {
        *side = 0.0;
        return None;
    }
    let obstacles = along.as_slice();
    let reach = crate::core::obstacles::robot_radius(constraints);
    let mut best: Option<(f64, f64, Point)> = None;
    for s in [1.0, -1.0] {
        let Some(g) = reference.detour(obstacles, reach + DETOUR_MARGIN, DETOUR_DT, s, DETOUR_RAMP) else {
            continue;
        };
        let shift: Vec<f64> = (0..g.x.len()).map(|i| (g.x[i] - reference.x[i]).hypot(g.y[i] - reference.y[i])).collect();
        let kt = target_step.min(shift.len() - 1);
        let widest = (kt / 2..=kt).max_by(|a, b| shift[*a].total_cmp(&shift[*b])).unwrap_or(kt);
        let ahead = |k: usize| Point::new(g.x[k], g.y[k], 0.0).distance_to_2d(robot) > 2.0 * tol;
        let k = if shift[widest] > shift[kt] + 1e-3 && ahead(widest) { widest } else { kt };
        if shift[k] <= 1e-3 {
            continue;
        }
        let cost = shift.iter().sum::<f64>() * if *side != 0.0 && s != *side { SIDE_SWITCH_COST } else { 1.0 };
        if best.as_ref().is_none_or(|b| cost < b.0) {
            best = Some((cost, s, Point::new(g.x[k], g.y[k], 0.0)));
        }
    }
    *side = best.as_ref().map_or(0.0, |b| b.1);
    best.map(|b| b.2)
}

/// Seconds between the points of a point controller's detour.
const DETOUR_DT: f64 = 0.2;
/// Share of the top speed below which a point controller's detour does not
/// take the robot's speed to fall.
const DETOUR_SPEED_FLOOR: f64 = 0.3;

/// Seconds of path at top speed checked for the steering-rate speed cap.
const RATE_PREVIEW_S: f64 = 2.0;
/// Share of the top speed the steering-rate cap never goes below.
const RATE_FLOOR: f64 = 0.2;

/// `state` carried forward through the actuation delay on its current arc
/// (delay compensation), from the measured velocity or else the last
/// command; `None` when there is no delay.
fn predicted_through_actuation(
    state: &RobotState,
    constraints: &RobotConstraints,
    last: Option<(f64, f64, f64)>,
) -> Option<RobotState> {
    let delay = actuation_delay(constraints);
    if delay <= 0.0 {
        return None;
    }
    let (v, w) = if state.velocity_valid {
        (state.velocity.linear, state.velocity.angular)
    } else {
        last.map_or((0.0, 0.0), |(v, w, _)| (v, w))
    };
    let steps = 10;
    let h = delay / steps as f64;
    let mut yaw = crate::core::math::yaw_of(&state.pose);
    let (mut x, mut y) = (state.pose.point.x, state.pose.point.y);
    for _ in 0..steps {
        x += v * (yaw + 0.5 * w * h).cos() * h;
        y += v * (yaw + 0.5 * w * h).sin() * h;
        yaw += w * h;
    }
    let mut ahead = state.clone();
    ahead.pose.point.x = x;
    ahead.pose.point.y = y;
    ahead.pose.rotation = datapod::Quaternion::from_euler(datapod::Euler::new(0.0, 0.0, yaw));
    Some(ahead)
}

fn finite_pose(p: &Pose) -> bool {
    let (t, q) = (p.point, p.rotation);
    [t.x, t.y, t.z, q.w, q.x, q.y, q.z].iter().all(|v| v.is_finite())
}

fn state_error(s: &RobotState) -> Option<&'static str> {
    let v = s.velocity;
    let ok = finite_pose(&s.pose) && [v.linear, v.angular, v.lateral, s.timestamp].iter().all(|x| x.is_finite());
    (!ok).then_some("robot state must be finite")
}

fn goal_error(g: &Goal) -> Option<&'static str> {
    let velocity_ok = g
        .target_velocity
        .is_none_or(|v| [v.linear, v.angular, v.lateral].iter().all(|x| x.is_finite()));
    let ok = finite_pose(&g.target_pose)
        && velocity_ok
        && !g.tolerance_position.is_nan()
        && !g.tolerance_orientation.is_nan();
    (!ok).then_some("goal must be finite")
}

fn path_error(poses: &[Pose], speeds: &[f64], times: Option<&[f64]>) -> Option<&'static str> {
    if !poses.iter().all(finite_pose) || !speeds.iter().all(|v| v.is_finite()) {
        return Some("path must be finite");
    }
    let times = times?;
    let ordered = times.windows(2).all(|w| w[0] <= w[1]);
    if times.len() != poses.len() || !times.iter().all(|t| t.is_finite()) || !ordered {
        return Some("trajectory needs one finite time per pose, in increasing order");
    }
    None
}

fn constraints_error(c: &RobotConstraints) -> Option<&'static str> {
    let magnitudes = [
        c.wheelbase,
        c.track_width,
        c.max_linear_velocity,
        c.max_angular_velocity,
        c.max_linear_acceleration,
        c.max_angular_acceleration,
        c.max_steering_angle,
        c.max_steering_rate,
        c.min_turning_radius,
        c.rear_wheelbase,
        c.robot_width,
        c.robot_length,
    ];
    let ok = !c.min_linear_velocity.is_nan() && magnitudes.iter().all(|m| *m >= 0.0);
    (!ok).then_some("robot constraints must be non-negative and not NaN")
}

fn config_error(c: &ControllerConfig) -> Option<&'static str> {
    let values = [
        c.kp_linear,
        c.ki_linear,
        c.kd_linear,
        c.kp_angular,
        c.ki_angular,
        c.kd_angular,
        c.k_cross_track,
        c.k_heading,
    ];
    let magnitudes = [c.lookahead_distance, c.lookahead_time, c.goal_tolerance, c.angular_tolerance];
    let ok = values.iter().all(|v| !v.is_nan()) && magnitudes.iter().all(|m| *m >= 0.0);
    (!ok).then_some("controller config must be non-negative where it is a distance, time or tolerance, and not NaN")
}

/// A closed path driven lap after lap. Controllers get it unrolled to two
/// laps, and get them again each time the robot starts the second.
#[derive(Clone, Debug)]
struct Lap {
    source: Path,
    ring: Path,
    length: f64,
    unrolled: Path,
}

impl Lap {
    fn of(path: &Path) -> Option<Self> {
        if !path.is_closed || path.waypoints.len() < 2 {
            return None;
        }
        let with_speeds = path.speeds.len() == path.waypoints.len();
        let mut ring = Path {
            waypoints: path.waypoints.clone(),
            speeds: if with_speeds { path.speeds.clone() } else { Vec::new() },
            is_closed: true,
        };
        let (first, last) = (ring.waypoints[0], *ring.waypoints.last().unwrap());
        if first.point.distance_to_2d(last.point) > 1e-9 {
            ring.waypoints.push(first);
            if with_speeds {
                ring.speeds.push(ring.speeds[0]);
            }
        }
        let length = cumulative_lengths(&ring.waypoints).last().copied().unwrap_or(0.0);
        if length.is_nan() || length <= 1e-6 {
            return None;
        }
        let mut unrolled = ring.clone();
        unrolled.is_closed = false;
        unrolled.waypoints.extend_from_slice(&ring.waypoints[1..]);
        if with_speeds {
            unrolled.speeds.extend_from_slice(&ring.speeds[1..]);
        }
        Some(Self { source: path.clone(), ring, length, unrolled })
    }
}
