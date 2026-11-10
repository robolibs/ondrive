use crate::controller::{Controller, check_goal};
use crate::core::kinematics::{actuation_delay, can_turn_in_place, curvature_to_steering, is_ackermann, is_holonomic, stop, wheelbase};
use crate::core::obstacles::CollisionChecker;
use crate::core::path::{PathCursor, cumulative_lengths, cusp_after, end_heading, next_cusp, sample, segment_heading};
use crate::fuzzy::FlcFollower;
use crate::path::{
    IlcFollower, KanayamaFollower, LqrFollower, PurePursuitFollower, RegulatedPursuitFollower, SlipAdaptiveFollower,
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
    SlipAdaptive,
}

impl TrackerKind {
    /// Point-to-point controllers are driven through a path waypoint by
    /// waypoint by the tracker; path followers consume the whole path.
    pub fn is_point_controller(self) -> bool {
        matches!(self, TrackerKind::Pid | TrackerKind::Carrot | TrackerKind::Dwa | TrackerKind::PoseReach | TrackerKind::PoseRegulator | TrackerKind::Apf)
    }

    /// Whether the controller, or the tracker on its behalf, steers round
    /// obstacles; every other one runs the safety filter by default.
    pub fn avoids_obstacles(self) -> bool {
        !matches!(
            self,
            TrackerKind::PurePursuit
                | TrackerKind::Stanley
                | TrackerKind::Lqr
                | TrackerKind::Kanayama
                | TrackerKind::VectorPursuit
                | TrackerKind::Flc
                | TrackerKind::Ilc
                | TrackerKind::SlipAdaptive
        )
    }

    /// Whether the controller models the machine's actuation lag itself;
    /// the tracker hands every other one the state predicted through it.
    pub fn models_actuation(self) -> bool {
        matches!(
            self,
            TrackerKind::Mpc | TrackerKind::Mppi | TrackerKind::Mca | TrackerKind::Soc | TrackerKind::Ilqr | TrackerKind::SlipAdaptive
        )
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
    /// Seconds the machine has stood while commanded to move.
    stalled_for: f64,
    /// Goal pose `(x, y, yaw)` the installed path was planned to, when the
    /// tracker planned it for a goal on its own.
    planned_for: Option<(f64, f64, f64)>,
    /// Control barrier filter on the final command against moving obstacles.
    safety: crate::core::safety::SafetyConfig,
    /// The caller's path while the tracker drives it detoured round standing
    /// obstacles, for a kind that does not avoid them itself.
    original: Option<Path>,
    /// Side that detour passes obstacles on: +1 left, −1 right, 0 none.
    follow_side: f64,
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
            stalled_for: 0.0,
            planned_for: None,
            safety: crate::core::safety::SafetyConfig { enabled: !kind.avoids_obstacles(), ..Default::default() },
            original: None,
            follow_side: 0.0,
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
        if self.planned_for.take().is_some() {
            self.path = None;
        }
    }

    /// Install a path. A closed path (`Path::is_closed`) is driven lap after
    /// lap and never completes.
    pub fn set_path(&mut self, path: Path) {
        self.planned_for = None;
        self.original = None;
        self.follow_side = 0.0;
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
        self.original = None;
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

    /// The safety filter's settings as a JSON object.
    pub fn safety(&self) -> serde_json::Value {
        serde_json::to_value(self.safety).unwrap_or_default()
    }

    /// Update some of the safety filter's settings from a JSON object, e.g.
    /// `{"enabled": false}`. An unknown name or a wrong type is an error and
    /// changes nothing.
    pub fn set_safety(&mut self, settings: &serde_json::Value) -> Result<(), String> {
        let mut merged = self.safety();
        let (Some(into), Some(from)) = (merged.as_object_mut(), settings.as_object()) else {
            return Err("safety settings must be an object".into());
        };
        for (k, v) in from {
            if !into.contains_key(k) {
                return Err(format!("unknown safety setting `{k}`"));
            }
            into.insert(k.clone(), v.clone());
        }
        self.safety = serde_json::from_value(merged).map_err(|e| e.to_string())?;
        Ok(())
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
        } else if let Some(path) = self.original.clone().or_else(|| self.path().cloned()) {
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
        self.plan_to_goal(state);
        self.steer_round(state, world);
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

        // A machine standing while told to move is commanded more, at the
        // acceleration limit, until it moves or reaches `BREAKAWAY_SPEED`.
        let speed = v.hypot(vy);
        let standing = state.velocity_valid
            && state.velocity.linear.hypot(state.velocity.lateral) < STANDING_SPEED
            && state.velocity.angular.abs() < STANDING_TURN;
        self.stalled_for = if standing && speed > 1e-3 { self.stalled_for + dt } else { 0.0 };
        if self.stalled_for > STALL_S
            && let Some((pv, _, pvy)) = self.last_command
        {
            let a = c.max_linear_acceleration.abs() * dt;
            let ramp = (pv.hypot(pvy) + a).min(BREAKAWAY_SPEED);
            if ramp > speed {
                let s = ramp / speed;
                let w_max = c.max_angular_velocity.abs();
                v *= s;
                vy *= s;
                w = if w_max > 0.0 { (w * s).clamp(-w_max, w_max) } else { w * s };
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

        if let Some((fv, fw)) = crate::core::safety::filter(&self.safety, c, &state.pose, (v, w), vy, world) {
            (v, w) = (fv, fw);
            cmd.status_message = format!("{} (safety filter)", cmd.status_message);
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

    /// For a path-following kind given a goal and no path of its own, plan
    /// the shortest Dubins path from `state` to the goal, turning no
    /// tighter than the platform can (`PLANNED_TURN_RADIUS` for one that
    /// turns in place), and drive it; planned again when the goal moves.
    fn plan_to_goal(&mut self, state: &RobotState) {
        let Some(goal) = &self.goal else {
            return;
        };
        if self.kind.is_point_controller() || self.trajectory.is_some() || (self.path.is_some() && self.planned_for.is_none()) {
            return;
        }
        let target = &goal.target_pose;
        let key = (target.point.x, target.point.y, target.rotation.to_euler().yaw);
        if self.planned_for == Some(key) {
            return;
        }
        let k = crate::core::kinematics::max_curvature(&self.constraints);
        let radius = if k.is_finite() && k > 0.0 { 1.0 / k } else { PLANNED_TURN_RADIUS };
        let start = (state.pose.point.x, state.pose.point.y, state.pose.rotation.to_euler().yaw);
        let waypoints = crate::core::dubins::dubins_path(start, key, radius, PLANNED_STEP)
            .into_iter()
            .map(|(x, y, yaw)| Pose { point: Point::new(x, y, target.point.z), rotation: Quaternion::from_euler(Euler::new(0.0, 0.0, yaw)) })
            .collect();
        self.drive(Path { waypoints, speeds: Vec::new(), is_closed: false });
        self.planned_for = Some(key);
    }

    /// For a kind that does not avoid obstacles, following a path of the
    /// caller's, drive that path detoured round the standing obstacles that
    /// block the body within `follow_lookahead`, on the side `pick_detour`
    /// takes, when the detour itself is clear; nothing while the path driven
    /// is clear.
    fn steer_round(&mut self, state: &RobotState, world: Option<&WorldConstraints>) {
        if self.kind.avoids_obstacles() || self.goal.is_some() || self.trajectory.is_some() || self.lap.is_some() {
            return;
        }
        let (Some(world), Some(driven)) = (world, self.path.as_ref()) else {
            return;
        };
        let standing: Vec<crate::types::Obstacle> = world.obstacles.iter().filter(|o| crate::pred::mppi::standing(o)).cloned().collect();
        if standing.is_empty() {
            return;
        }
        let rocks = WorldConstraints { obstacles: standing, ..Default::default() };
        let checker = CollisionChecker::new(Some(&rocks), &self.constraints, FOLLOW_MARGIN);
        let ahead = follow_lookahead(&self.constraints);
        let from = state.pose.point;
        if !blocked_ahead(driven, from, ahead, &checker) {
            return;
        }
        let base = self.original.clone().unwrap_or_else(|| driven.clone());
        let cfg = self.controller.get_config();
        let lead = if matches!(self.kind, TrackerKind::PurePursuit | TrackerKind::VectorPursuit) {
            cfg.lookahead_distance.max(0.0) + cfg.lookahead_time.max(0.0) * self.constraints.max_linear_velocity.abs()
        } else {
            0.0
        };
        let Some(path) = detoured(&base, from, ahead, lead, &rocks.obstacles, &self.constraints, &mut self.follow_side) else {
            return;
        };
        if blocked_ahead(&path, from, ahead, &checker) {
            return;
        }
        self.original = Some(base);
        self.drive(path);
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
            let ahead = cfg.lookahead_distance.max(2.0 * tol);
            let overshoot = (proj.distance.powi(2) - proj.lateral_error.powi(2)).max(0.0).sqrt();
            let circle = (1.0 / crate::core::kinematics::max_curvature(&self.constraints)).max(2.0 * tol);
            let passed = proj.lateral_error.abs() < 2.0 * tol && overshoot < ahead || proj.distance < circle;
            if self.lap.is_none() && proj.beyond_end && passed {
                self.path_completed = true;
                return Ok(None);
            }
            let speed_now = if state.velocity_valid { state.velocity.linear } else { self.last_command.map_or(0.0, |c| c.0) };
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
            let round_crossing = can_turn_in_place(self.constraints.steering_type) && self.kind != TrackerKind::Apf;
            let detour = world
                .filter(|w| !w.obstacles.is_empty())
                .and_then(|w| {
                    detour_target(path, cum, proj.arc_length, cum[idx], state.pose.point, tol, &w.obstacles, &self.constraints, speed_now, round_crossing, &mut self.detour_side)
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
            // An Ackermann robot is aimed one lookahead past the path's end
            // along its last segment, driving through the end, not round it.
            let point = if last && self.lap.is_none() && n >= 2 && is_ackermann(self.constraints.steering_type) {
                let (s, c) = segment_heading(&path.waypoints, n - 2).sin_cos();
                Point::new(wp.point.x + c * ahead, wp.point.y + s * ahead, wp.point.z)
            } else {
                wp.point
            };
            return Ok(Some(Goal {
                target_pose: Pose {
                    point,
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

    /// True if the last tick put the controller in the goal-reached state,
    /// or a path driven without an explicit goal has been completed.
    pub fn is_goal_reached(&self) -> bool {
        self.controller.get_status().goal_reached || (self.goal.is_none() && self.path_completed)
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
        TrackerKind::SlipAdaptive => Box::new(SlipAdaptiveFollower::new()),
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
/// onto the cheaper detour round the obstacles' predictions (behind the
/// traffic crossing the path when `round_crossing`, else left to give way
/// to), the robot taken to drive the path at
/// `speed_now` (no slower than a share of its top speed), or the detour's
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
    round_crossing: bool,
    side: &mut f64,
) -> Option<Point> {
    use crate::pred::mpc::{DETOUR_MARGIN, DETOUR_RAMP, SIDE_SWITCH_COST};
    let top = constraints.max_linear_velocity.abs();
    let speed = speed_now.abs().clamp(DETOUR_SPEED_FLOOR * top, top).max(0.1);
    let target_step = ((to - from).max(0.0) / (speed * DETOUR_DT)).round() as usize;
    let reach = crate::core::obstacles::robot_radius(constraints);
    let mut n = (target_step + DETOUR_RAMP).clamp(2, 80);
    let moving = |o: &crate::types::Obstacle| {
        use crate::core::obstacles::modes_at;
        let (Some(now), Some(soon)) = (modes_at(o, 0.0, DETOUR_DT).next(), modes_at(o, 1.0, DETOUR_DT).next()) else {
            return false;
        };
        (soon.x - now.x).hypot(soon.y - now.y) > MOVING_SPEED
    };
    if obstacles.iter().any(moving) {
        n = n.max((CROSSING_LOOKOUT_S / DETOUR_DT).ceil() as usize).min(80);
    }
    let reference = crate::pred::mppi::build_reference(path, cum, from, n, DETOUR_DT, speed, 0.0);
    let crossing = |o: &crate::types::Obstacle| crate::pred::mppi::crosses(o, &reference, DETOUR_DT, reach);
    let kept: Vec<crate::types::Obstacle> = if round_crossing {
        obstacles.to_vec()
    } else {
        obstacles.iter().filter(|o| !crossing(o)).cloned().collect()
    };
    if kept.is_empty() {
        *side = 0.0;
        return None;
    }
    let obstacles = kept.as_slice();
    let behind = crate::pred::mppi::pass_behind(obstacles, &reference, DETOUR_DT, reach);
    let mut best: Option<(f64, f64, Point)> = None;
    for s in [1.0, -1.0] {
        if behind.is_some_and(|b| b != s) {
            continue;
        }
        let Some(g) = reference.detour(obstacles, reach + DETOUR_MARGIN, DETOUR_DT, s, DETOUR_RAMP, crate::pred::mppi::Swerve::of(constraints, speed)) else {
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
/// Metres between the points a path follower's detour is planned on and
/// checked at.
const FOLLOW_STEP: f64 = 0.25;
/// Clearance, metres, below which a standing obstacle blocks the body on
/// the path a follower drives.
const FOLLOW_MARGIN: f64 = 0.2;
/// Least metres ahead a follower's path is checked for standing obstacles.
const FOLLOW_AHEAD: f64 = 15.0;
/// Metres past the stretch checked over which a follower's detour rejoins
/// its path.
const FOLLOW_REJOIN: f64 = 10.0;
/// Times the stretch a follower's detour is planned on may double to let it settle.
const FOLLOW_SPANS: usize = 3;

/// Metres ahead a follower's path is checked: `FOLLOW_AHEAD`, or three body
/// lengths when longer.
fn follow_lookahead(constraints: &RobotConstraints) -> f64 {
    let length = crate::core::obstacles::footprint_extent(constraints).map_or(0.0, |(back, front, _)| front - back);
    FOLLOW_AHEAD.max(3.0 * length)
}

/// Whether the body, placed along `path` every `FOLLOW_STEP` from the
/// point nearest `from` for `ahead` metres, meets an obstacle of `checker`.
fn blocked_ahead(path: &Path, from: Point, ahead: f64, checker: &CollisionChecker) -> bool {
    let cum = cumulative_lengths(&path.waypoints);
    let mut cursor = PathCursor::default();
    cursor.set_path(&path.waypoints);
    let Some(proj) = cursor.project(&path.waypoints, from, 0) else {
        return false;
    };
    let end = (proj.arc_length + ahead).min(cum.last().copied().unwrap_or(0.0));
    let mut s = proj.arc_length;
    while s <= end {
        let (p, yaw) = sample(&path.waypoints, &cum, s);
        if checker.clearance_at(0.0, p.x, p.y, yaw) < 0.0 {
            return true;
        }
        s += FOLLOW_STEP;
    }
    false
}

/// `base` from the point nearest `from` on, moved round `obstacles` by
/// `pick_detour` (its side kept in `side`) with its full shift held `lead`
/// metres longer before and after each obstacle, up to where it is back
/// on `base`, then the rest of `base`. The stretch planned starts at `ahead`
/// and `FOLLOW_REJOIN` metres and doubles, up to `FOLLOW_SPANS` times, until
/// the detour settles within it. `None` when nothing is to go round, it does
/// not settle, or `base` carries speeds that reverse.
#[allow(clippy::too_many_arguments)]
fn detoured(base: &Path, from: Point, ahead: f64, lead: f64, obstacles: &[crate::types::Obstacle], constraints: &RobotConstraints, side: &mut f64) -> Option<Path> {
    if base.speeds.iter().any(|v| *v < 0.0) {
        return None;
    }
    let cum = cumulative_lengths(&base.waypoints);
    let mut cursor = PathCursor::default();
    cursor.set_path(&base.waypoints);
    let proj = cursor.project(&base.waypoints, from, 0)?;
    let along = crate::pred::mppi::Along { path: base, cum: &cum, trajectory: None, clock: 0.0, from: proj.arc_length };
    let mut swerve = crate::pred::mppi::Swerve::of(constraints, constraints.max_linear_velocity.abs());
    if let Some((back, front, half)) = swerve.body {
        swerve.body = Some((back - lead, front + lead, half));
    }
    let reach = crate::core::obstacles::robot_radius(constraints);
    let total = cum.last().copied().unwrap_or(0.0);
    let mut span = ahead + FOLLOW_REJOIN;
    for _ in 0..FOLLOW_SPANS {
        let n = (span / FOLLOW_STEP).ceil() as usize;
        let reference = along.reference(n, 1.0, FOLLOW_STEP);
        let g = crate::pred::mppi::pick_detour(&reference, obstacles, reach, 1.0, side, swerve)?;
        let k = g.x.len().min(reference.x.len());
        let shifted = |i: usize| (g.x[i] - reference.x[i]).hypot(g.y[i] - reference.y[i]) > 1e-3;
        let reaches_end = proj.arc_length + span >= total;
        if shifted(k - 1) && !reaches_end {
            span *= 2.0;
            continue;
        }
        let last = (0..k).rev().find(|i| shifted(*i)).map_or(0, |i| (i + 1).min(k - 1));
        let z = base.waypoints.first().map_or(0.0, |w| w.point.z);
        let mut waypoints: Vec<Pose> = (0..=last)
            .map(|i| {
                let (a, b) = (i.saturating_sub(1), (i + 1).min(k - 1));
                let yaw = (g.y[b] - g.y[a]).atan2(g.x[b] - g.x[a]);
                Pose { point: Point::new(g.x[i], g.y[i], z), rotation: Quaternion::from_euler(Euler::new(0.0, 0.0, yaw)) }
            })
            .collect();
        let end = proj.arc_length + last as f64 * FOLLOW_STEP;
        waypoints.extend(base.waypoints.iter().zip(&cum).filter(|(_, c)| **c > end + 1e-6).map(|(w, _)| *w));
        return Some(Path { waypoints, speeds: Vec::new(), is_closed: false });
    }
    None
}
/// Seconds of path a point controller's detour spans while traffic moves.
const CROSSING_LOOKOUT_S: f64 = 12.0;
/// Speed, m/s, above which an obstacle counts as moving.
const MOVING_SPEED: f64 = 0.1;
/// Turning radius, metres, of a path planned to a goal for a platform that
/// turns in place.
const PLANNED_TURN_RADIUS: f64 = 0.5;
/// Metres between the poses of a path planned to a goal.
const PLANNED_STEP: f64 = 0.1;
/// Share of the top speed below which a point controller's detour does not
/// take the robot's speed to fall.
const DETOUR_SPEED_FLOOR: f64 = 0.3;

/// Measured speed, m/s, below which a machine counts as standing.
const STANDING_SPEED: f64 = 0.05;
/// Measured turn rate, rad/s, below which a machine counts as standing.
const STANDING_TURN: f64 = 0.05;
/// Seconds a machine stands while told to move before its command is ramped.
const STALL_S: f64 = 1.0;
/// Speed, m/s, the stall ramp goes up to.
const BREAKAWAY_SPEED: f64 = 0.3;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_path_follower_given_only_a_goal_plans_a_path_to_it() {
        let mut t = Tracker::new(TrackerKind::PurePursuit);
        let mut c = RobotConstraints::default();
        c.wheelbase = 0.65;
        c.max_steering_angle = 0.46;
        c.max_linear_velocity = 1.0;
        t.init(c);
        let target = Pose { point: Point::new(6.0, 3.0, 0.0), rotation: Quaternion::from_euler(Euler::new(0.0, 0.0, 1.57)) };
        t.set_goal(Goal { target_pose: target, target_velocity: None, tolerance_position: 0.3, tolerance_orientation: 0.3 });
        let state = RobotState { velocity_valid: true, allow_move: true, ..Default::default() };
        let cmd = t.tick(&state, 0.05, None);
        assert!(cmd.valid, "{}", cmd.status_message);
        let path = t.path().expect("a planned path");
        let end = path.waypoints.last().unwrap();
        assert!(end.point.distance_to_2d(target.point) < 1e-6);
        assert!(cmd.linear_velocity > 0.0);
    }

    #[test]
    fn a_path_follower_drives_round_a_rock_on_its_path() {
        let mut t = Tracker::new(TrackerKind::PurePursuit);
        let mut c = RobotConstraints::default();
        c.wheelbase = 0.65;
        c.max_steering_angle = 0.46;
        c.max_linear_velocity = 1.0;
        c.footprint = crate::types::Footprint::Polygon { points: vec![(0.5, 0.35), (-0.5, 0.35), (-0.5, -0.35), (0.5, -0.35)] };
        t.init(c);
        let waypoints: Vec<Pose> = (0..=60)
            .map(|i| Pose { point: Point::new(0.5 * i as f64, 0.0, 0.0), rotation: Quaternion::from_euler(Euler::new(0.0, 0.0, 0.0)) })
            .collect();
        t.set_path(Path { waypoints, speeds: Vec::new(), is_closed: false });
        let mode = crate::types::GaussianMode { weight: 1.0, mean_x: vec![10.0; 2], mean_y: vec![0.1; 2], std_x: vec![0.05; 2], std_y: vec![0.05; 2] };
        let world = WorldConstraints { obstacles: vec![crate::types::Obstacle { id: 0, radius: 0.6, modes: vec![mode], dt: 0.2 }], ..Default::default() };
        let state = RobotState { velocity_valid: true, allow_move: true, ..Default::default() };
        assert!(t.tick(&state, 0.05, Some(&world)).valid);
        let path = t.path().expect("a path");
        let beside = path.waypoints.iter().filter(|w| (w.point.x - 10.0).abs() < 0.5).map(|w| w.point.y.abs()).fold(0.0_f64, f64::max);
        assert!(beside > 0.6 + 0.35, "the path passes the rock {beside} m off its centre");
        let end = path.waypoints.last().unwrap();
        assert!((end.point.x - 30.0).abs() < 1e-6 && end.point.y.abs() < 1e-6);
    }

    #[test]
    fn a_stall_ramp_never_spins_a_machine_past_its_turn_rate() {
        let mut t = Tracker::new(TrackerKind::Mpc);
        let mut c = RobotConstraints::default();
        c.steering_type = crate::types::SteeringType::SkidSteer;
        c.max_linear_velocity = 1.0;
        c.max_angular_velocity = 1.5;
        c.max_linear_acceleration = 1.0;
        c.max_angular_acceleration = 3.0;
        t.init(c);
        let state = RobotState { velocity_valid: true, ..Default::default() };
        let turn = VelocityCommand { valid: true, linear_velocity: 0.002, angular_velocity: 1.0, ..Default::default() };
        for _ in 0..100 {
            let cmd = t.limit(turn.clone(), &state, 0.05, None);
            assert!(cmd.angular_velocity.abs() <= 1.5 + 1e-9, "w {}", cmd.angular_velocity);
        }
    }
}
