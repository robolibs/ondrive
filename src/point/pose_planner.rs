//! Pose-to-pose controller for car-like platforms: plans the shortest
//! Reeds-Shepp curve (Dubins when reverse is not allowed) from the current
//! pose to the goal pose with the platform's turning radius, splits it into
//! same-direction runs and tracks one run at a time with Pure Pursuit,
//! switching runs at the cusps. Replans when the robot drifts off the
//! curve or the goal changes. Handed a path, it follows the path with Pure
//! Pursuit to the goals the path holds.

use crate::controller::{Controller, ControllerBase, check_goal, pass_through, settled};
use crate::core::curves::{CurvePath, dubins_all, reeds_shepp_all};
use crate::core::obstacles::CollisionChecker;
use crate::core::kinematics::{
    can_turn_in_place, finalize, holonomic_point_command, is_ackermann, is_holonomic,
    max_curvature, path_speed, reverse_allowed, stop,
};
use crate::core::math::{normalize_angle, yaw_of};
use crate::core::path::end_heading;
use crate::path::PurePursuitFollower;
use crate::types::{
    ControllerConfig, ControllerStatus, Goal, OutputUnits, Path, RobotConstraints, RobotState,
    VelocityCommand, WorldConstraints,
};
use datapod::{Euler, Point, Pose, Quaternion};
use std::f64::consts::PI;

/// Shortest time between two replans while tracking the same goal.
const REPLAN_MIN_INTERVAL: f64 = 2.0;
/// Extra plans allowed to refine the final orientation.
const MAX_REFINEMENTS: usize = 6;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PoseReachConfig {
    /// Turning radius used for planning; 0 derives it from the constraints
    /// (`radius_margin / max_curvature`, or `default_radius` when unlimited).
    pub turning_radius: f64,
    pub default_radius: f64,
    /// Planning radius is `radius_margin` times the platform minimum so the
    /// follower keeps steering authority on the arcs.
    pub radius_margin: f64,
    pub sample_spacing: f64,
    /// Replan when the lateral distance to the curve exceeds this.
    pub replan_lateral: f64,
    /// Replan when the heading error to the curve exceeds this.
    pub replan_heading: f64,
    /// Lookahead handed to the internal Pure Pursuit follower.
    pub lookahead: f64,
    /// Distance to a run's end at which the next run takes over.
    pub cusp_tolerance: f64,
    /// Heading feedback gain blended in over the final approach.
    pub k_final_heading: f64,
    /// Planning cost of a metre driven in reverse, against 1 forward.
    pub reverse_penalty: f64,
    /// Planning cost of each change of direction, in metres.
    pub cusp_penalty: f64,
    /// Runs shorter than this (metres) cost three cusps more each.
    pub min_run: f64,
}

impl Default for PoseReachConfig {
    fn default() -> Self {
        Self {
            turning_radius: 0.0,
            default_radius: 0.6,
            radius_margin: 1.25,
            sample_spacing: 0.1,
            replan_lateral: 0.25,
            replan_heading: 0.5,
            lookahead: 0.5,
            cusp_tolerance: 0.1,
            k_final_heading: 2.0,
            reverse_penalty: 1.5,
            cusp_penalty: 1.0,
            min_run: 0.3,
        }
    }
}

/// One same-direction stretch of the plan.
#[derive(Clone, Debug)]
struct Run {
    path: Path,
    direction: i8,
}

#[derive(Clone, Debug, Default)]
pub struct PoseReachFollower {
    pub base: ControllerBase,
    pub pose_config: PoseReachConfig,
    follower: PurePursuitFollower,
    /// Follower of the path handed in, for goals on it.
    along: PurePursuitFollower,
    /// Whether `along` still has to be given the path.
    along_stale: bool,
    runs: Vec<Run>,
    run_index: usize,
    goal_index: usize,
    extension_end: Option<Pose>,
    planned_for: Option<(f64, f64, f64, bool)>,
    /// Goal the refinement count belongs to.
    refined_for: Option<(f64, f64, f64, bool)>,
    since_replan: f64,
    replans: usize,
    refinements: usize,
}

fn split_runs(curve: &CurvePath, speed: f64) -> Vec<Run> {
    let mut runs: Vec<Run> = Vec::new();
    for (i, pose) in curve.poses.iter().enumerate() {
        let dir = curve.directions[i];
        match runs.last_mut() {
            Some(run) if run.direction == dir => {
                run.path.waypoints.push(*pose);
                run.path.speeds.push(speed * dir as f64);
            }
            Some(run) => {
                // Start the next run at the cusp so runs share the cusp pose.
                let cusp = *run.path.waypoints.last().unwrap();
                let mut path = Path::default();
                path.waypoints.push(cusp);
                path.speeds.push(speed * dir as f64);
                path.waypoints.push(*pose);
                path.speeds.push(speed * dir as f64);
                runs.push(Run { path, direction: dir });
            }
            None => {
                let mut path = Path::default();
                path.waypoints.push(*pose);
                path.speeds.push(speed * dir as f64);
                runs.push(Run { path, direction: dir });
            }
        }
    }
    runs
}

impl PoseReachFollower {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_pose_config(cfg: PoseReachConfig) -> Self {
        Self {
            pose_config: cfg,
            ..Default::default()
        }
    }

    pub fn set_pose_config(&mut self, cfg: PoseReachConfig) {
        self.pose_config = cfg;
        self.planned_for = None;
    }

    /// The whole planned curve currently being tracked.
    pub fn plan(&self) -> Path {
        let mut p = Path::default();
        for run in &self.runs {
            p.waypoints.extend(run.path.waypoints.iter().copied());
            p.speeds.extend(run.path.speeds.iter().copied());
        }
        p
    }

    /// Number of plans computed since the last reset.
    pub fn replan_count(&self) -> usize {
        self.replans
    }

    /// Refinement attempts spent on the current goal.
    pub fn refinement_count(&self) -> usize {
        self.refinements
    }

    /// Lookahead of the inner Pure Pursuit: the configured one, no shorter
    /// than the controller config's.
    fn lookahead(&self) -> f64 {
        self.pose_config.lookahead.max(self.base.config.lookahead_distance)
    }

    fn radius(&self, constraints: &RobotConstraints) -> f64 {
        if self.pose_config.turning_radius > 0.0 {
            return self.pose_config.turning_radius;
        }
        let k = max_curvature(constraints);
        if k.is_finite() && k > 1e-6 && is_ackermann(constraints.steering_type) {
            self.pose_config.radius_margin.max(1.0) / k
        } else {
            self.pose_config.default_radius.max(0.05)
        }
    }

    fn current_run(&self) -> Option<&Run> {
        self.runs.get(self.run_index)
    }

    fn deviation(&self, state: &RobotState) -> (f64, f64) {
        let Some(run) = self.current_run() else {
            return (f64::INFINITY, PI);
        };
        let mut best = (f64::INFINITY, PI);
        for w in &run.path.waypoints {
            let d = w.point.distance_to_2d(state.pose.point);
            if d < best.0 {
                best = (d, normalize_angle(yaw_of(w) - yaw_of(&state.pose)).abs());
            }
        }
        best
    }

    fn load_run(&mut self, goal: &Goal, constraints: &RobotConstraints) {
        let Some(run) = self.runs.get(self.run_index) else {
            return;
        };
        let last = self.run_index + 1 >= self.runs.len();
        let mut path = run.path.clone();
        self.goal_index = path.waypoints.len().saturating_sub(1);
        self.extension_end = None;
        if last {
            // Extend past the goal along its heading so the follower stays
            // aligned during the final approach.
            let spacing = self.pose_config.sample_spacing.max(0.02);
            let gyaw = yaw_of(&goal.target_pose);
            let dir = run.direction as f64;
            let v = constraints.max_linear_velocity.abs().max(0.1);
            let n_ext = (self.lookahead() / spacing).ceil().max(1.0) as usize;
            for i in 1..=n_ext {
                let s = dir * i as f64 * spacing;
                path.waypoints.push(Pose {
                    point: Point::new(
                        goal.target_pose.point.x + s * gyaw.cos(),
                        goal.target_pose.point.y + s * gyaw.sin(),
                        0.0,
                    ),
                    rotation: goal.target_pose.rotation,
                });
                path.speeds.push(v * dir);
            }
            self.extension_end = path.waypoints.last().copied();
        }
        let mut cfg = self.base.config.clone();
        cfg.output_units = OutputUnits::Physical;
        cfg.lookahead_distance = self.lookahead();
        cfg.allow_reverse = run.direction < 0;
        self.follower.set_config(cfg);
        self.follower.set_path(path);
    }

    fn replan(
        &mut self,
        state: &RobotState,
        goal: &Goal,
        constraints: &RobotConstraints,
        reverse: bool,
        world: Option<&WorldConstraints>,
    ) -> bool {
        let radius = self.radius(constraints);
        let spacing = self.pose_config.sample_spacing.max(0.02);
        let curves = |to: &Pose| if reverse { reeds_shepp_all(&state.pose, to, radius, spacing) } else { dubins_all(&state.pose, to, radius, spacing) };
        let cost = |c: &CurvePath| plan_cost(c, &self.pose_config);
        let mut candidates = curves(&goal.target_pose);
        candidates.sort_by(|a, b| cost(a).total_cmp(&cost(b)));
        // A platform that cannot turn in place comes at the goal along a
        // straight run-in, forward or reversing, unless that costs a loop.
        if !can_turn_in_place(constraints.steering_type) {
            let run_in = self.lookahead() + crate::controller::effective_tolerances(goal, &self.base.config).0;
            let mut led: Vec<CurvePath> = [1i8, -1]
                .into_iter()
                .filter(|d| *d > 0 || reverse)
                .flat_map(|d| with_run_in(&goal.target_pose, run_in, d, spacing, &curves))
                .collect();
            led.sort_by(|a, b| cost(a).total_cmp(&cost(b)));
            let direct = candidates.first().map_or(f64::INFINITY, |c| cost(c));
            let straight_in = candidates.first().is_some_and(|c| ends_straight(c, &goal.target_pose, run_in));
            if !straight_in && led.first().is_some_and(|c| cost(c) <= direct + run_in) {
                led.extend(candidates);
                candidates = led;
            }
        }
        let checker = CollisionChecker::new(world, constraints, 0.0);
        let curve = if checker.has_obstacles() {
            candidates.into_iter().find(|c| {
                c.poses.iter().all(|p| !checker.collides(0, p.point.x, p.point.y, yaw_of(p)))
            })
        } else {
            candidates.into_iter().next()
        };
        let Some(curve) = curve else {
            return false;
        };
        let v = constraints.max_linear_velocity.abs().max(0.1);
        let runs = split_runs(&curve, v);
        if runs.is_empty() {
            return false;
        }
        self.runs = runs;
        self.run_index = 0;
        self.load_run(goal, constraints);
        true
    }

    /// Pursuit along the path handed in, to `goal` at its end.
    fn follow_along(
        &mut self,
        state: &RobotState,
        goal: &Goal,
        constraints: &RobotConstraints,
        dt: f64,
        world: Option<&WorldConstraints>,
    ) -> VelocityCommand {
        if self.along_stale || self.along.base.path.waypoints.len() != self.base.path.waypoints.len() {
            let mut cfg = self.base.config.clone();
            cfg.output_units = OutputUnits::Physical;
            self.along.set_config(cfg);
            self.along.set_path(self.base.path.clone());
            self.along_stale = false;
        }
        self.planned_for = None;
        self.base.status.mode = "pose_reach/along".into();
        let cmd = self.along.compute_control(state, goal, constraints, dt, world);
        self.base.status.goal_reached = self.along.get_status().goal_reached;
        cmd
    }

    /// Pursuit of a goal to pass at speed: along the line through it on its
    /// heading, run on past it, with no curve round to that heading.
    fn pass(
        &mut self,
        state: &RobotState,
        goal: &Goal,
        constraints: &RobotConstraints,
        dt: f64,
        world: Option<&WorldConstraints>,
        reverse: bool,
    ) -> VelocityCommand {
        let heading = yaw_of(&goal.target_pose);
        let back = reverse && normalize_angle(heading - yaw_of(&state.pose)).abs() > PI / 2.0;
        let key = (goal.target_pose.point.x, goal.target_pose.point.y, heading, reverse);
        if self.planned_for != Some(key) {
            let direction: i8 = if back { -1 } else { 1 };
            let v = constraints.max_linear_velocity.abs().max(0.1);
            let accel = constraints.max_linear_acceleration.abs().max(0.1);
            let on = self.lookahead().max(1.0) + v * v / (2.0 * accel);
            let (s, c) = heading.sin_cos();
            let p = goal.target_pose.point;
            let body = Quaternion::from_euler(Euler::new(0.0, 0.0, if back { heading + PI } else { heading }));
            let mut path = Path::default();
            for point in [state.pose.point, p, Point::new(p.x + on * c, p.y + on * s, p.z)] {
                path.waypoints.push(Pose { point, rotation: body });
                path.speeds.push(v * direction as f64);
            }
            self.runs = vec![Run { path: path.clone(), direction }];
            self.run_index = 0;
            let mut cfg = self.base.config.clone();
            cfg.output_units = OutputUnits::Physical;
            cfg.lookahead_distance = self.lookahead();
            cfg.allow_reverse = back;
            self.follower.set_config(cfg);
            self.follower.set_path(path);
            self.planned_for = Some(key);
        }
        self.base.status.goal_reached = false;
        self.base.status.mode = "pose_reach/pass".into();
        self.follower.compute_control(state, goal, constraints, dt, world)
    }

    /// The goal handed to the follower: the current run's end, or the real
    /// goal on the last run.
    fn run_goal(&self, goal: &Goal) -> Goal {
        let last = self.run_index + 1 >= self.runs.len();
        let end = if last {
            self.extension_end.unwrap_or(goal.target_pose)
        } else {
            *self.runs[self.run_index].path.waypoints.last().unwrap()
        };
        Goal {
            target_pose: end,
            target_velocity: None,
            tolerance_position: self.pose_config.cusp_tolerance,
            tolerance_orientation: PI,
        }
    }
}

impl Controller for PoseReachFollower {
    fn params(&self) -> serde_json::Value {
        serde_json::to_value(&self.pose_config).unwrap_or_default()
    }

    fn set_params(&mut self, params: &serde_json::Value) -> Result<(), String> {
        let cfg = crate::controller::merged(&self.pose_config, params)?;
        self.set_pose_config(cfg);
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
        let check = check_goal(&state.pose, goal, &self.base.config);
        let previous = self.base.status.distance_to_goal;
        self.base.status.distance_to_goal = check.distance;
        self.base.status.heading_error = check.yaw_error;
        if check.reached && settled(&check, state, goal, previous) && pass_through(goal) == 0.0 {
            self.base.status.goal_reached = true;
            self.base.status.mode = "stopped".into();
            return stop("Goal reached");
        }
        if is_holonomic(constraints.steering_type) {
            // A holonomic platform needs no curve: it translates and
            // rotates independently, so the Reeds-Shepp/Dubins planner
            // below (built for a minimum turning radius) does not apply.
            self.base.status.mode = "pose_reach_holonomic".into();
            let yaw = yaw_of(&state.pose);
            return holonomic_point_command(
                state.pose.point,
                yaw,
                goal.target_pose.point,
                yaw_of(&goal.target_pose),
                check.distance,
                &self.base.config,
                constraints,
                "Moving to goal",
            );
        }
        let reverse = reverse_allowed(&self.base.config, state);
        if let Some(along) = along_goal(&self.base.path, goal, self.base.config.angular_tolerance) {
            return self.follow_along(state, &along, constraints, dt, world);
        }
        if pass_through(goal) > 0.0 {
            return self.pass(state, goal, constraints, dt, world, reverse);
        }
        let key = (
            goal.target_pose.point.x,
            goal.target_pose.point.y,
            yaw_of(&goal.target_pose),
            reverse,
        );
        self.since_replan += dt;
        let (lateral, heading) = self.deviation(state);
        let needs = self.planned_for != Some(key)
            || self.runs.is_empty()
            || (self.since_replan >= REPLAN_MIN_INTERVAL
                && (lateral > self.pose_config.replan_lateral || heading > self.pose_config.replan_heading));
        if self.refined_for != Some(key) {
            self.refined_for = Some(key);
            self.refinements = 0;
        }
        if needs {
            if !self.replan(state, goal, constraints, reverse, world) {
                return VelocityCommand::invalid("no feasible curve to the goal pose");
            }
            self.planned_for = Some(key);
            self.since_replan = 0.0;
            self.replans += 1;
        }

        // Advance to the next run once the current one is finished.
        loop {
            let last = self.run_index + 1 >= self.runs.len();
            if last {
                break;
            }
            let check = check_goal(&state.pose, &self.run_goal(goal), &self.base.config);
            if check.position_ok || self.follower.get_status().goal_reached {
                self.run_index += 1;
                self.load_run(goal, constraints);
            } else {
                break;
            }
        }

        let run_goal = self.run_goal(goal);
        let cmd = self.follower.compute_control(state, &run_goal, constraints, dt, world);
        let inner = self.follower.get_status();
        self.base.status.cross_track_error = inner.cross_track_error;
        self.base.status.heading_error = inner.heading_error;
        self.base.status.goal_reached = false;
        let last = self.run_index + 1 >= self.runs.len();
        let passed_goal = last && (inner.goal_reached || self.follower.get_path_index() >= self.goal_index);
        if passed_goal {
            let aligned = !check.orientation_required || check.orientation_ok;
            let close = check.distance < 1.5 * check.position_tolerance;
            if close && !aligned && can_turn_in_place(constraints.steering_type) {
                self.base.status.mode = "pose_reach/aligning".into();
                let omega = self.base.config.kp_angular.max(0.1) * check.yaw_error;
                return finalize(0.0, omega, constraints, &self.base.config, reverse, "Aligning to goal orientation");
            }
            if (aligned && close) || self.refinements >= MAX_REFINEMENTS {
                self.base.status.goal_reached = true;
                self.base.status.mode = "stopped".into();
                return stop("Goal reached");
            }
            self.refinements += 1;
            self.planned_for = None;
            self.base.status.mode = "pose_reach/refine".into();
            return stop("Refining final orientation");
        }
        self.base.status.mode = format!(
            "pose_reach/run{}/{}",
            self.run_index,
            if self.runs[self.run_index].direction < 0 { "reverse" } else { "forward" }
        );
        if !cmd.valid {
            return cmd;
        }
        let (mut v, mut omega) = (cmd.linear_velocity, cmd.angular_velocity);
        // On the last run the robot slows to stop at the goal, turning the same.
        if last && v.abs() > 1e-9 {
            let cfg = &self.base.config;
            let cap = path_speed(v.abs(), 0.0, check.distance, check.position_tolerance, cfg.kp_linear, constraints);
            let scale = (cap / v.abs()).min(1.0);
            v *= scale;
            omega *= scale;
        }
        let approach = self.lookahead();
        if last && check.distance < approach {
            let blend = 1.0 - check.distance / approach;
            omega += blend * self.pose_config.k_final_heading * check.yaw_error;
        }
        finalize(v, omega, constraints, &self.base.config, reverse, &cmd.status_message)
    }

    fn set_path(&mut self, path: Path) {
        self.base.path = path;
        self.along_stale = true;
        self.planned_for = None;
        self.refined_for = None;
        self.runs.clear();
        self.run_index = 0;
        self.refinements = 0;
    }

    fn reset(&mut self) {
        self.base.status = ControllerStatus::default();
        self.follower.reset();
        self.along.reset();
        self.along_stale = true;
        self.runs.clear();
        self.run_index = 0;
        self.planned_for = None;
        self.refined_for = None;
        self.since_replan = 0.0;
        self.replans = 0;
        self.refinements = 0;
    }

    fn set_config(&mut self, config: ControllerConfig) {
        self.base.config = config;
        self.along_stale = true;
        self.planned_for = None;
    }

    fn get_type(&self) -> &'static str {
        "pose_reach_follower"
    }

    fn predicted_trajectory(&self) -> Vec<Point> {
        self.plan().waypoints.iter().map(|p| p.point).collect()
    }

    fn base(&self) -> &ControllerBase {
        &self.base
    }

    fn base_mut(&mut self) -> &mut ControllerBase {
        &mut self.base
    }
}

/// Distance, metres, within which a goal counts as a point of the path.
const ON_PATH: f64 = 1e-3;

/// Whether `p` lies within `tol` of the polyline of `path`.
fn on_path(path: &Path, p: Point, tol: f64) -> bool {
    path.waypoints.windows(2).any(|w| {
        let (a, b) = (w[0].point, w[1].point);
        let (dx, dy) = (b.x - a.x, b.y - a.y);
        let len2 = dx * dx + dy * dy;
        let t = if len2 > 1e-12 { (((p.x - a.x) * dx + (p.y - a.y) * dy) / len2).clamp(0.0, 1.0) } else { 0.0 };
        (a.x + t * dx - p.x).hypot(a.y + t * dy - p.y) < tol
    })
}

/// The end of `path` the follower along it drives to for `goal`, when `goal`
/// is a point of the path: on it with any heading, or at or straight on past
/// its open end facing the way the path ends.
fn along_goal(path: &Path, goal: &Goal, angular_tolerance: f64) -> Option<Goal> {
    let n = path.waypoints.len();
    if n < 2 {
        return None;
    }
    let p = goal.target_pose.point;
    let end = path.waypoints[n - 1].point;
    let heading = end_heading(&path.waypoints);
    let backwards = path.speeds.last().is_some_and(|v| *v < 0.0);
    let facing = heading + if backwards { PI } else { 0.0 };
    let free = pass_through(goal) > 0.0 || goal.tolerance_orientation >= PI - 1e-9;
    let tolerance_orientation = if free && on_path(path, p, ON_PATH) {
        angular_tolerance
    } else {
        let (c, s) = (heading.cos(), heading.sin());
        let (dx, dy) = (p.x - end.x, p.y - end.y);
        let turned = normalize_angle(yaw_of(&goal.target_pose) - facing).abs();
        let past_end = !path.is_closed && dx * c + dy * s > -ON_PATH && (dy * c - dx * s).abs() < ON_PATH;
        if !past_end || turned > goal.tolerance_orientation.max(STRAIGHT_IN) {
            return None;
        }
        goal.tolerance_orientation
    };
    Some(Goal {
        target_pose: Pose { point: end, rotation: Quaternion::from_euler(Euler::new(0.0, 0.0, facing)) },
        target_velocity: None,
        tolerance_position: goal.tolerance_position,
        tolerance_orientation,
    })
}

/// Heading, radians, within which a curve's last stretch counts as straight
/// on the goal's heading.
const STRAIGHT_IN: f64 = 0.02;

/// Whether `curve` comes at `goal` along a straight on the goal's heading of
/// at least `run_in` metres, or is that straight all along.
fn ends_straight(curve: &CurvePath, goal: &Pose, run_in: f64) -> bool {
    let heading = yaw_of(goal);
    let mut covered = 0.0;
    for w in curve.poses.windows(2).rev() {
        if normalize_angle(yaw_of(&w[0]) - heading).abs() > STRAIGHT_IN {
            return false;
        }
        covered += w[0].point.distance_to_2d(w[1].point);
        if covered >= run_in {
            return true;
        }
    }
    true
}

/// Each of the `curves` to the pose `run_in` metres short of `goal` along its
/// heading, driven in `direction` (+1 forward, −1 reversing), carried on
/// straight to `goal`.
fn with_run_in(goal: &Pose, run_in: f64, direction: i8, spacing: f64, curves: &dyn Fn(&Pose) -> Vec<CurvePath>) -> Vec<CurvePath> {
    let (s, c) = yaw_of(goal).sin_cos();
    let d = direction as f64;
    let pre = Pose { point: Point::new(goal.point.x - d * run_in * c, goal.point.y - d * run_in * s, goal.point.z), rotation: goal.rotation };
    let steps = (run_in / spacing).ceil().max(1.0) as usize;
    curves(&pre)
        .into_iter()
        .map(|mut curve| {
            for k in 1..=steps {
                let u = d * run_in * k as f64 / steps as f64;
                curve.poses.push(Pose { point: Point::new(pre.point.x + u * c, pre.point.y + u * s, goal.point.z), rotation: goal.rotation });
                curve.directions.push(direction);
            }
            curve.length += run_in;
            curve
        })
        .collect()
}

/// Planning cost of a curve: its length, reverse driving weighted by
/// `reverse_penalty`, `cusp_penalty` per change of direction, and three
/// cusps more for every run shorter than `min_run`.
fn plan_cost(c: &CurvePath, cfg: &PoseReachConfig) -> f64 {
    let mut cost = 0.0;
    let mut run = 0.0;
    let mut cusps = 0.0;
    for i in 1..c.poses.len() {
        let step = c.poses[i - 1].point.distance_to_2d(c.poses[i].point);
        let reverse = c.directions[i] < 0;
        cost += if reverse { cfg.reverse_penalty.max(1.0) * step } else { step };
        if c.directions[i] != c.directions[i - 1] {
            cusps += 1.0;
            if run < cfg.min_run {
                cusps += 3.0;
            }
            run = 0.0;
        }
        run += step;
    }
    cost + cfg.cusp_penalty.max(0.0) * cusps
}

#[cfg(test)]
mod tests {
    use super::*;

    fn straight(backwards: bool) -> Path {
        let yaw = if backwards { PI } else { 0.0 };
        Path {
            waypoints: (0..=10)
                .map(|i| Pose { point: Point::new(i as f64, 0.0, 0.0), rotation: Quaternion::from_euler(Euler::new(0.0, 0.0, yaw)) })
                .collect(),
            speeds: vec![if backwards { -1.0 } else { 1.0 }; 11],
            is_closed: false,
        }
    }

    fn goal(x: f64, y: f64, yaw: f64, heading_tolerance: f64) -> Goal {
        Goal {
            target_pose: Pose { point: Point::new(x, y, 0.0), rotation: Quaternion::from_euler(Euler::new(0.0, 0.0, yaw)) },
            target_velocity: None,
            tolerance_position: 0.25,
            tolerance_orientation: heading_tolerance,
        }
    }

    #[test]
    fn a_point_of_the_path_is_driven_to_along_it() {
        let g = along_goal(&straight(false), &goal(4.0, 0.0, 0.0, PI), 0.3).expect("on the path");
        assert!((g.target_pose.point.x - 10.0).abs() < 1e-9);
        assert!((g.tolerance_orientation - 0.3).abs() < 1e-9);
    }

    #[test]
    fn a_detour_off_the_path_is_planned() {
        assert!(along_goal(&straight(false), &goal(4.0, 0.5, 0.0, PI), 0.3).is_none());
    }

    #[test]
    fn an_aim_past_the_end_is_moved_onto_it() {
        let g = along_goal(&straight(false), &goal(12.0, 0.0, 0.0, 0.25), PI).expect("past the end");
        assert!((g.target_pose.point.x - 10.0).abs() < 1e-9);
        assert!((g.tolerance_orientation - 0.25).abs() < 1e-9);
    }

    #[test]
    fn an_end_facing_another_way_is_planned() {
        assert!(along_goal(&straight(false), &goal(10.0, 0.0, PI / 2.0, 0.25), PI).is_none());
    }

    #[test]
    fn a_reversing_end_faces_back_along_the_path() {
        let g = along_goal(&straight(true), &goal(12.0, 0.0, PI, 0.25), PI).expect("reversing end");
        assert!(normalize_angle(yaw_of(&g.target_pose) - PI).abs() < 1e-9);
    }
}
