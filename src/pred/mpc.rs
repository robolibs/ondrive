//! Kinematic MPC: finite-horizon tracking cost (lateral, heading, speed,
//! control effort, control rate) minimised by projected gradient descent
//! with a backtracking line search, warm-started from the previous
//! solution shifted by one step. On a holonomic platform the third control
//! channel (lateral acceleration) is optimised the same way; its bound is
//! zero otherwise, so the solve is identical to the two-control case.

use crate::controller::{Controller, ControllerBase};
use crate::core::kinematics::planning_step;
use crate::core::obstacles::{CollisionChecker, footprint_discs, obstacle_barrier};
use crate::core::path::PathCursor;
use crate::pred::mppi::{
    CostWeights, Model, Reference, build_reference, build_timed_reference, command_from_controls,
    current_lateral_speed, current_speed, prepare, rollout, speed_at_arc_length,
    update_turn_in_place,
};
use crate::types::{
    DebugFrame, DebugLevel, Goal, Obstacle, Path, RobotConstraints, RobotState, Trajectory, VelocityCommand,
    WorldConstraints,
};
use datapod::Point;

const MAX_ITERATIONS: usize = 20;
const LINE_SEARCH_STEPS: usize = 6;
const FD_EPS: f64 = 1e-4;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MpcConfig {
    pub horizon_steps: usize,
    pub dt: f64,

    pub weight_cte: f64,
    pub weight_epsi: f64,
    pub weight_vel: f64,
    pub weight_steering: f64,
    pub weight_acceleration: f64,
    pub weight_steering_rate: f64,
    pub weight_acceleration_rate: f64,

    pub ref_velocity: f64,

    pub turn_first_activation_deg: f64,
    pub turn_first_release_deg: f64,

    /// Distance along the path from the end at which `ref_velocity` starts
    /// tapering linearly to zero. Set to 0.0 to disable.
    pub approach_taper_distance: f64,

    /// Obstacle barrier `weight_obstacle · exp(−clearance / obstacle_scale)`
    /// at each step against the obstacles predicted for that step's time
    /// and the grid; 0 ignores obstacles.
    pub weight_obstacle: f64,
    pub obstacle_scale: f64,
}

impl Default for MpcConfig {
    fn default() -> Self {
        Self {
            horizon_steps: 20,
            dt: 0.15,
            weight_cte: 100.0,
            weight_epsi: 100.0,
            weight_vel: 50.0,
            weight_steering: 10.0,
            weight_acceleration: 5.0,
            weight_steering_rate: 500.0,
            weight_acceleration_rate: 50.0,
            ref_velocity: 1.0,
            turn_first_activation_deg: 60.0,
            turn_first_release_deg: 15.0,
            approach_taper_distance: 2.0,
            weight_obstacle: 300.0,
            obstacle_scale: 0.3,
        }
    }
}

#[derive(Clone, Debug)]
pub struct MpcFollower {
    pub base: ControllerBase,
    pub mpc_config: MpcConfig,
    cursor: PathCursor,
    previous_steering: Vec<f64>,
    previous_acceleration: Vec<f64>,
    previous_accel_lateral: Vec<f64>,
    predicted_trajectory: Vec<Point>,
    is_turning_in_place: bool,
    last_v: f64,
    last_vy: f64,
    /// Steering the machine has reached, from the commands sent through
    /// its rate limit; `None` until the first command.
    actuated: Option<f64>,
    debug_level: DebugLevel,
    debug: DebugFrame,
    shift_accum: f64,
    /// Side obstacles were last passed on: +1 left, −1 right, 0 none.
    side: f64,
    trajectory: Option<Trajectory>,
    clock: f64,
}

impl Default for MpcFollower {
    fn default() -> Self {
        Self::with_mpc_config(MpcConfig::default())
    }
}

type StageExtra<'a> = dyn Fn(usize, f64, f64, f64) -> f64 + 'a;

/// Steering, acceleration and lateral acceleration over the horizon.
type Controls = (Vec<f64>, Vec<f64>, Vec<f64>);

/// Extra room the detour guides keep from an obstacle, metres.
pub(crate) const DETOUR_MARGIN: f64 = 0.25;
/// Steps over which a detour guide moves aside and back.
pub(crate) const DETOUR_RAMP: usize = 6;
/// Cost factor of passing obstacles on the other side than last tick.
pub(crate) const SIDE_SWITCH_COST: f64 = 1.15;
/// The points of a reference, as a path.
pub(crate) fn reference_path(r: &Reference) -> Vec<Point> {
    (0..r.x.len()).map(|i| Point::new(r.x[i], r.y[i], 0.0)).collect()
}

/// Lateral and heading weight scale where a detour leaves the reference.
pub(crate) const DETOURED_WEIGHT: f64 = 0.1;

/// The steering reached `dt` after commanding `command` from `previous`
/// (the model's starting steering before any command), through the
/// model's steering rate and lock.
pub(crate) fn actuated_after(model: &Model, previous: Option<f64>, command: f64, dt: f64) -> f64 {
    let sb = model.steer_bound();
    let from = previous.unwrap_or(model.steer0).clamp(-sb, sb);
    let step = if model.steer_rate.is_finite() { model.steer_rate * dt } else { f64::INFINITY };
    (from + (command.clamp(-sb, sb) - from).clamp(-step, step)).clamp(-sb, sb)
}

/// The references to solve against, each with the side it passes
/// obstacles on (0 the reference itself): `reference` and its detours
/// round the obstacles on either side (not those crossing the path, which
/// the barrier has it give way to), `reference` relaxed where they
/// leave it (topology-guided MPC, de Groot et al. 2024).
pub(crate) fn guided(reference: &mut Reference, obstacles: &[Obstacle], reach: f64, dt: f64) -> Vec<(Reference, f64)> {
    let obstacles = crate::pred::mppi::to_go_round(obstacles, reference, dt);
    let detours: Vec<(Reference, f64)> = [1.0, -1.0]
        .into_iter()
        .filter_map(|side| reference.detour(&obstacles, reach + DETOUR_MARGIN, dt, side, DETOUR_RAMP).map(|r| (r, side)))
        .collect();
    reference.relax_where_detoured(detours.iter().map(|d| &d.0), DETOURED_WEIGHT);
    std::iter::once((reference.clone(), 0.0)).chain(detours).collect()
}

#[allow(clippy::too_many_arguments)]
fn total_cost(
    model: &Model,
    reference: &Reference,
    w: &CostWeights,
    cfg: &MpcConfig,
    extra: &StageExtra,
    steer: &[f64],
    accel: &[f64],
    accel_lat: &[f64],
) -> f64 {
    let (mut c, _) = rollout(model, steer, accel, accel_lat, reference, w, extra, false);
    for i in 1..steer.len() {
        let ds = steer[i] - steer[i - 1];
        let da = accel[i] - accel[i - 1];
        let dal = accel_lat[i] - accel_lat[i - 1];
        c += (cfg.weight_steering_rate * ds * ds
            + cfg.weight_acceleration_rate * (da * da + dal * dal))
            * model.dt;
    }
    c
}

impl MpcFollower {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_mpc_config(cfg: MpcConfig) -> Self {
        let n = cfg.horizon_steps;
        Self {
            base: ControllerBase::default(),
            cursor: PathCursor::default(),
            previous_steering: vec![0.0; n],
            previous_acceleration: vec![0.0; n],
            previous_accel_lateral: vec![0.0; n],
            predicted_trajectory: Vec::new(),
            is_turning_in_place: false,
            last_v: 0.0,
            last_vy: 0.0,
            actuated: None,
            debug_level: DebugLevel::Off,
            debug: DebugFrame::default(),
            shift_accum: 0.0,
            side: 0.0,
            trajectory: None,
            clock: 0.0,
            mpc_config: cfg,
        }
    }

    pub fn set_mpc_config(&mut self, cfg: MpcConfig) {
        let n = cfg.horizon_steps;
        self.previous_steering = vec![0.0; n];
        self.previous_acceleration = vec![0.0; n];
        self.previous_accel_lateral = vec![0.0; n];
        self.mpc_config = cfg;
    }

    pub fn predicted_trajectory(&self) -> &[Point] {
        &self.predicted_trajectory
    }

    #[allow(clippy::too_many_arguments)]
    fn solve(
        &self,
        model: &Model,
        reference: &Reference,
        w: &CostWeights,
        cfg: &MpcConfig,
        extra: &StageExtra,
        steer: &mut [f64],
        accel: &mut [f64],
        accel_lat: &mut [f64],
    ) -> bool {
        let n = steer.len();
        model.clamp_controls(steer, accel, accel_lat);
        let mut best = total_cost(model, reference, w, cfg, extra, steer, accel, accel_lat);
        if !best.is_finite() {
            return false;
        }
        let bound_s = model.steer_bound().max(1e-6);
        let bound_a = model.accel_bound();
        let bound_al = model.accel_lat_bound().max(1e-6);
        let mut grad_s = vec![0.0; n];
        let mut grad_a = vec![0.0; n];
        let mut grad_al = vec![0.0; n];
        let mut cand_s = steer.to_vec();
        let mut cand_a = accel.to_vec();
        let mut cand_al = accel_lat.to_vec();

        for _ in 0..MAX_ITERATIONS {
            for i in 0..n {
                let o = steer[i];
                steer[i] = o + FD_EPS;
                let cp = total_cost(model, reference, w, cfg, extra, steer, accel, accel_lat);
                steer[i] = o - FD_EPS;
                let cm = total_cost(model, reference, w, cfg, extra, steer, accel, accel_lat);
                steer[i] = o;
                grad_s[i] = (cp - cm) / (2.0 * FD_EPS);

                let o = accel[i];
                accel[i] = o + FD_EPS;
                let cp = total_cost(model, reference, w, cfg, extra, steer, accel, accel_lat);
                accel[i] = o - FD_EPS;
                let cm = total_cost(model, reference, w, cfg, extra, steer, accel, accel_lat);
                accel[i] = o;
                grad_a[i] = (cp - cm) / (2.0 * FD_EPS);

                if bound_al > 1e-6 {
                    let o = accel_lat[i];
                    accel_lat[i] = o + FD_EPS;
                    let cp = total_cost(model, reference, w, cfg, extra, steer, accel, accel_lat);
                    accel_lat[i] = o - FD_EPS;
                    let cm = total_cost(model, reference, w, cfg, extra, steer, accel, accel_lat);
                    accel_lat[i] = o;
                    grad_al[i] = (cp - cm) / (2.0 * FD_EPS);
                }
            }
            let g_inf = grad_s
                .iter()
                .map(|g| g.abs() / bound_s)
                .chain(grad_a.iter().map(|g| g.abs() / bound_a))
                .chain(grad_al.iter().map(|g| g.abs() / bound_al))
                .fold(0.0_f64, f64::max);
            if g_inf < 1e-9 {
                break;
            }
            let mut step = 0.2 / g_inf;
            let mut improved = false;
            for _ in 0..LINE_SEARCH_STEPS {
                for i in 0..n {
                    cand_s[i] = steer[i] - step * grad_s[i];
                    cand_a[i] = accel[i] - step * grad_a[i];
                    cand_al[i] = accel_lat[i] - step * grad_al[i];
                }
                model.clamp_controls(&mut cand_s, &mut cand_a, &mut cand_al);
                let c = total_cost(model, reference, w, cfg, extra, &cand_s, &cand_a, &cand_al);
                if c.is_finite() && c < best {
                    best = c;
                    steer.copy_from_slice(&cand_s);
                    accel.copy_from_slice(&cand_a);
                    accel_lat.copy_from_slice(&cand_al);
                    improved = true;
                    break;
                }
                step *= 0.5;
            }
            if !improved {
                break;
            }
        }
        true
    }
}

impl Controller for MpcFollower {
    fn params(&self) -> serde_json::Value {
        serde_json::to_value(&self.mpc_config).unwrap_or_default()
    }

    fn set_params(&mut self, params: &serde_json::Value) -> Result<(), String> {
        let cfg = crate::controller::merged(&self.mpc_config, params)?;
        self.set_mpc_config(cfg);
        Ok(())
    }

    fn set_debug(&mut self, level: DebugLevel) {
        self.debug_level = level;
        self.debug = DebugFrame::default();
    }

    fn debug_frame(&self) -> DebugFrame {
        self.debug.clone()
    }

    fn compute_control(
        &mut self,
        state: &RobotState,
        goal: &Goal,
        constraints: &RobotConstraints,
        dt: f64,
        world: Option<&WorldConstraints>,
    ) -> VelocityCommand {
        let prep = match prepare(&mut self.base, &mut self.cursor, state, goal, constraints, dt) {
            Ok(p) => p,
            Err(cmd) => {
                self.predicted_trajectory.clear();
                return cmd;
            }
        };
        let mut cfg = self.mpc_config.clone();
        cfg.dt = planning_step(constraints, cfg.ref_velocity.min(constraints.max_linear_velocity), cfg.horizon_steps, cfg.dt);
        let turning = update_turn_in_place(
            &mut self.is_turning_in_place,
            state,
            constraints,
            prep.epsi,
            cfg.turn_first_activation_deg,
            cfg.turn_first_release_deg,
        );
        if turning {
            cfg.ref_velocity = 0.2;
            cfg.weight_vel = 50.0;
        }
        let n = cfg.horizon_steps.max(1);
        if self.previous_steering.len() != n {
            self.previous_steering = vec![0.0; n];
            self.previous_acceleration = vec![0.0; n];
            self.previous_accel_lateral = vec![0.0; n];
        }

        let v_now = current_speed(state, self.last_v);
        let vy_now = current_lateral_speed(state, self.last_vy);
        let mut model = Model::new(state, constraints, v_now, vy_now, cfg.dt, prep.allow_reverse);
        if model.ackermann && model.steer_rate.is_finite() && let Some(a) = self.actuated {
            model.steer0 = a;
        }
        let mut reference = match &self.trajectory {
            Some(traj) => build_timed_reference(traj, self.clock, n, cfg.dt),
            None => build_reference(
                &self.base.path,
                &self.cursor.cum,
                prep.proj.arc_length,
                n,
                cfg.dt,
                cfg.ref_velocity.min(constraints.max_linear_velocity),
                cfg.approach_taper_distance,
            ),
        };
        let w = CostWeights {
            cte: cfg.weight_cte,
            epsi: cfg.weight_epsi,
            vel: cfg.weight_vel,
            steering: cfg.weight_steering,
            accel: cfg.weight_acceleration,
            progress: 0.0,
        };

        let discs = footprint_discs(constraints);
        let obstacles: &[Obstacle] = world.map_or(&[], |w| &w.obstacles);
        let reach = discs.iter().map(|d| d.1).fold(0.0_f64, f64::max);
        let checker = CollisionChecker::new(world, constraints, 0.0);
        let grid = world.is_some_and(|w| w.grid.is_some());
        let (step_dt, weight, scale) = (cfg.dt, cfg.weight_obstacle, cfg.obstacle_scale);
        let barrier = |i: usize, x: f64, y: f64, yaw: f64| -> f64 {
            let t = (i + 1) as f64 * step_dt;
            let mut c = obstacle_barrier(obstacles, &discs, t, step_dt, 1.0, (x, y, yaw), weight, scale).cost;
            if grid && weight > 0.0 && scale > 0.0 {
                c += weight * (-checker.grid_clearance(x, y, yaw) / scale).min(30.0).exp();
            }
            c
        };

        let guides = guided(&mut reference, obstacles, reach, cfg.dt);
        let mut best: Option<(f64, f64, Controls)> = None;
        let mut tried: Vec<(Vec<Point>, f64)> = Vec::new();
        if self.debug_level >= DebugLevel::Basic {
            self.debug.guides = guides.iter().skip(1).map(|(g, _)| reference_path(g)).collect();
        }
        for (guide, side) in &guides {
            let mut steer = self.previous_steering.clone();
            let mut accel = self.previous_acceleration.clone();
            let mut accel_lat = self.previous_accel_lateral.clone();
            if !self.solve(&model, guide, &w, &cfg, &barrier, &mut steer, &mut accel, &mut accel_lat) {
                continue;
            }
            let switch = *side != 0.0 && self.side != 0.0 && *side != self.side;
            let c = total_cost(&model, &reference, &w, &cfg, &barrier, &steer, &accel, &accel_lat)
                * if switch { SIDE_SWITCH_COST } else { 1.0 };
            if self.debug_level == DebugLevel::Full {
                let (_, path) = rollout(&model, &steer, &accel, &accel_lat, &reference, &w, &barrier, true);
                tried.push((path, c));
            }
            if best.as_ref().is_none_or(|b| c < b.0) {
                best = Some((c, *side, (steer, accel, accel_lat)));
            }
        }
        if self.debug_level == DebugLevel::Full {
            (self.debug.rollouts, self.debug.costs) = tried.into_iter().unzip();
        }
        let Some((_, side, (mut steer, mut accel, mut accel_lat))) = best else {
            return VelocityCommand::invalid("MPC optimizer failed");
        };
        if side != 0.0 || guides.len() == 1 {
            self.side = side;
        }

        let (_, traj) = rollout(&model, &steer, &accel, &accel_lat, &reference, &w, &barrier, true);
        self.predicted_trajectory = traj;

        let steer0 = steer[0];
        self.actuated = Some(actuated_after(&model, self.actuated, steer0, dt));
        let accel0 = accel[0];
        let accel_lat0 = accel_lat[0];
        crate::pred::mppi::shift_warm_start(&mut self.shift_accum, dt, cfg.dt, [&mut steer, &mut accel, &mut accel_lat]);
        self.previous_steering = steer;
        self.previous_acceleration = accel;
        self.previous_accel_lateral = accel_lat;

        let dt_apply = dt.min(cfg.dt);
        let v_cap = speed_at_arc_length(&self.base.path.speeds, &self.cursor.cum, prep.proj.arc_length)
            .unwrap_or(f64::INFINITY);
        self.base.status.mode = "mpc_tracking".into();
        let cmd = command_from_controls(
            v_now,
            vy_now,
            steer0,
            accel0,
            accel_lat0,
            dt_apply,
            v_cap,
            constraints,
            &self.base.config,
            prep.allow_reverse,
            turning,
            prep.epsi,
            "MPC tracking",
        );
        self.last_v = (v_now + accel0 * dt_apply).clamp(-v_cap, v_cap);
        self.last_vy = vy_now + accel_lat0 * dt_apply;
        cmd
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
        let n = self.mpc_config.horizon_steps;
        self.previous_steering = vec![0.0; n];
        self.previous_acceleration = vec![0.0; n];
        self.previous_accel_lateral = vec![0.0; n];
        self.predicted_trajectory.clear();
        self.is_turning_in_place = false;
        self.shift_accum = 0.0;
        self.side = 0.0;
    }

    fn reset(&mut self) {
        self.set_path(Path::default());
        self.last_v = 0.0;
        self.last_vy = 0.0;
        self.actuated = None;
    }

    fn get_type(&self) -> &'static str {
        "mpc_follower"
    }

    fn predicted_trajectory(&self) -> Vec<Point> {
        self.predicted_trajectory.clone()
    }

    fn base(&self) -> &ControllerBase {
        &self.base
    }

    fn base_mut(&mut self) -> &mut ControllerBase {
        &mut self.base
    }
}
