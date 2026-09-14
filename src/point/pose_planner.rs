//! Pose-to-pose controller for car-like platforms: plans the shortest
//! Reeds-Shepp curve (Dubins when reverse is not allowed) from the current
//! pose to the goal pose with the platform's turning radius, splits it into
//! same-direction runs and tracks one run at a time with Pure Pursuit,
//! switching runs at the cusps. Replans when the robot drifts off the
//! curve or the goal changes.

use crate::controller::{Controller, ControllerBase, check_goal};
use crate::core::curves::{CurvePath, dubins_all, reeds_shepp_all};
use crate::core::obstacles::CollisionChecker;
use crate::core::kinematics::{
    can_turn_in_place, finalize, holonomic_point_command, is_ackermann, is_holonomic,
    max_curvature, reverse_allowed, stop,
};
use crate::core::math::{normalize_angle, yaw_of};
use crate::path::PurePursuitFollower;
use crate::types::{
    ControllerConfig, ControllerStatus, Goal, OutputUnits, Path, RobotConstraints, RobotState,
    VelocityCommand, WorldConstraints,
};
use datapod::{Point, Pose};
use std::f64::consts::PI;

/// Shortest time between two replans while tracking the same goal.
const REPLAN_MIN_INTERVAL: f64 = 2.0;
/// Extra plans allowed to refine the final orientation.
const MAX_REFINEMENTS: usize = 6;

#[derive(Clone, Debug)]
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
    runs: Vec<Run>,
    run_index: usize,
    goal_index: usize,
    extension_end: Option<Pose>,
    planned_for: Option<(f64, f64, f64, bool)>,
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
            let n_ext = (self.pose_config.lookahead / spacing).ceil().max(1.0) as usize;
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
        cfg.lookahead_distance = self.pose_config.lookahead;
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
        let candidates = if reverse {
            reeds_shepp_all(&state.pose, &goal.target_pose, radius, spacing)
        } else {
            dubins_all(&state.pose, &goal.target_pose, radius, spacing)
        };
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
        self.base.status.distance_to_goal = check.distance;
        self.base.status.heading_error = check.yaw_error;
        if check.reached {
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
        if needs {
            if !self.replan(state, goal, constraints, reverse, world) {
                return VelocityCommand::invalid("no feasible curve to the goal pose");
            }
            self.planned_for = Some(key);
            self.since_replan = 0.0;
            self.replans += 1;
            if self.planned_for != Some(key) {
                self.refinements = 0;
            }
        }

        // Advance to the next run once the current one is finished.
        loop {
            let last = self.run_index + 1 >= self.runs.len();
            if last {
                break;
            }
            let end = self.runs[self.run_index].path.waypoints.last().unwrap();
            let check = check_goal(&state.pose, &self.run_goal(goal), &self.base.config);
            let _ = end;
            if check.position_ok {
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
        let mut omega = cmd.angular_velocity;
        let approach = self.pose_config.lookahead;
        if last && check.distance < approach {
            let blend = 1.0 - check.distance / approach;
            omega += blend * self.pose_config.k_final_heading * check.yaw_error;
        }
        finalize(cmd.linear_velocity, omega, constraints, &self.base.config, reverse, &cmd.status_message)
    }

    fn set_path(&mut self, _path: Path) {
        self.planned_for = None;
    }

    fn reset(&mut self) {
        self.base.status = ControllerStatus::default();
        self.follower.reset();
        self.runs.clear();
        self.run_index = 0;
        self.planned_for = None;
        self.since_replan = 0.0;
        self.replans = 0;
        self.refinements = 0;
    }

    fn set_config(&mut self, config: ControllerConfig) {
        self.base.config = config;
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
