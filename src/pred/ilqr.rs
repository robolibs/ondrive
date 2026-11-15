//! iLQR trajectory tracker: iterative LQR / DDP on the kinematic model with
//! analytic Jacobians, a regularised backward pass, a line-searched forward
//! pass with control clamping, and a warm start from the shifted previous
//! solution. State `[x, y, yaw, v, vy]`, controls
//! `[steer_or_omega, accel, accel_lat]`. `vy` and `accel_lat` (lateral
//! speed/acceleration) are only free on a holonomic platform — their bound
//! is exactly zero otherwise (enforced in the forward-pass control clamp),
//! so `vy` never leaves zero and the solve is identical to the old 4-state
//! system for every other steering type. Obstacles enter as an exponential
//! barrier on the footprint's clearance to each obstacle's prediction at the
//! step's time, with a Gauss–Newton Hessian (constrained iLQR).

use crate::controller::{Controller, ControllerBase};
use crate::core::kinematics::{planning, wheelbase};
use crate::core::math::normalize_angle;
use crate::core::obstacles::{BarrierField, footprint_discs};
use crate::core::path::PathCursor;
use crate::pred::mpc::{SIDE_SWITCH_COST, guided};
use crate::pred::mppi::{
    Choice, Model, Reference, build_reference, build_timed_reference, command_from_controls, eased_lateral,
    current_lateral_speed, current_speed, prepare, speed_at_arc_length, update_turn_in_place,
};
use crate::types::{
    DebugFrame, DebugLevel, Goal, Obstacle, Path, RobotConstraints, RobotState, Trajectory, VelocityCommand,
    WorldConstraints,
};
use datapod::Point;
use nalgebra::{Matrix3, Matrix3x6, Matrix6, Matrix6x3, Vector3, Vector6};

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct IlqrConfig {
    pub horizon_steps: usize,
    pub dt: f64,
    pub iterations: usize,
    /// Stop when the relative cost decrease of an iteration is below this.
    pub convergence: f64,

    pub weight_cte: f64,
    pub weight_epsi: f64,
    pub weight_vel: f64,
    pub weight_steering: f64,
    pub weight_acceleration: f64,
    pub weight_terminal: f64,
    /// Cost per (rad/s)² of turn rate off the reference's curvature times the speed.
    pub weight_yaw_rate: f64,

    pub ref_velocity: f64,
    pub turn_first_activation_deg: f64,
    pub turn_first_release_deg: f64,
    pub approach_taper_distance: f64,

    /// Obstacle barrier `weight_obstacle · exp(−clearance / obstacle_scale)`
    /// at each step; 0 ignores obstacles.
    pub weight_obstacle: f64,
    pub obstacle_scale: f64,
}

impl Default for IlqrConfig {
    fn default() -> Self {
        Self {
            horizon_steps: 20,
            dt: 0.15,
            iterations: 15,
            convergence: 1e-4,
            weight_cte: 100.0,
            weight_epsi: 100.0,
            weight_vel: 50.0,
            weight_steering: 10.0,
            weight_acceleration: 5.0,
            weight_terminal: 3.0,
            weight_yaw_rate: 100.0,
            ref_velocity: 1.0,
            turn_first_activation_deg: 60.0,
            turn_first_release_deg: 15.0,
            approach_taper_distance: 2.0,
            weight_obstacle: 3000.0,
            obstacle_scale: 0.2,
        }
    }
}

#[derive(Clone, Debug)]
pub struct IlqrFollower {
    pub base: ControllerBase,
    pub ilqr_config: IlqrConfig,
    cursor: PathCursor,
    steer: Vec<f64>,
    accel: Vec<f64>,
    accel_lat: Vec<f64>,
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
    /// Side and pace of the last way past the obstacles ahead, see
    /// `plan_avoidance`.
    avoid: Choice,
    last_iterations: usize,
    last_cost: f64,
    trajectory: Option<Trajectory>,
    clock: f64,
}

impl Default for IlqrFollower {
    fn default() -> Self {
        Self::with_ilqr_config(IlqrConfig::default())
    }
}

type State = Vector6<f64>;
type Control = Vector3<f64>;

struct Stage {
    cost: f64,
    lx: State,
    lu: Control,
    lxx: Matrix6<f64>,
    luu: Matrix3<f64>,
}

struct Problem<'a> {
    model: &'a Model<'a>,
    reference: &'a Reference,
    cfg: &'a IlqrConfig,
    /// Obstacle barrier for footprint discs offset ahead of the state point.
    field: &'a BarrierField<'a>,
}

impl Problem<'_> {
    /// The steering after one step from `x` under command `u`, and its
    /// derivative in the command: the gap to the command closes at most
    /// `steer_rate · dt` a step, smoothly (`R · tanh(gap / R)`), so the
    /// rate limit stays differentiable; the command itself with no rate.
    fn steering(&self, x: &State, u: &Control) -> (f64, f64) {
        let sb = self.model.steer_bound();
        let target = u[0].clamp(-sb, sb);
        let reach = self.model.steer_rate * self.model.dt;
        if !(reach.is_finite() && reach > 0.0) {
            return (target, 1.0);
        }
        let t = ((target - x[5]) / reach).tanh();
        ((x[5] + reach * t).clamp(-sb, sb), 1.0 - t * t)
    }

    fn step(&self, x: &State, u: &Control) -> State {
        let (mut px, mut py, mut yaw, mut v, mut vy) = (x[0], x[1], x[2], x[3], x[4]);
        let ab = self.model.accel_bound();
        let lb = self.model.accel_lat_bound();
        let (steer, _) = self.steering(x, u);
        self.model.step(&mut px, &mut py, &mut yaw, &mut v, &mut vy, steer, u[1].clamp(-ab, ab), u[2].clamp(-lb, lb));
        State::new(px, py, yaw, v, vy, steer)
    }

    fn jacobians(&self, x: &State, u: &Control) -> (Matrix6<f64>, Matrix6x3<f64>) {
        let dt = self.model.dt;
        let (s, c) = x[2].sin_cos();
        let (v, vy) = (x[3], x[4]);
        let (delta, k) = self.steering(x, u);
        let mut a = Matrix6::identity();
        a[(0, 2)] = (-v * s - vy * c) * dt;
        a[(0, 3)] = c * dt;
        a[(0, 4)] = -s * dt;
        a[(1, 2)] = (v * c - vy * s) * dt;
        a[(1, 3)] = s * dt;
        a[(1, 4)] = c * dt;
        a[(5, 5)] = 1.0 - k;
        let mut b = Matrix6x3::zeros();
        b[(5, 0)] = k;
        let yaw_per_steer = if self.model.ackermann {
            let l = wheelbase(self.model.constraints);
            a[(2, 3)] = delta.tan() / l * dt;
            v / (l * delta.cos().powi(2)) * dt
        } else {
            dt
        };
        a[(2, 5)] = yaw_per_steer * (1.0 - k);
        b[(2, 0)] = yaw_per_steer * k;
        b[(3, 1)] = dt;
        if self.model.accel_lat_bound() > 0.0 {
            b[(4, 2)] = dt;
        }
        (a, b)
    }

    /// Stage cost and derivatives for the state after step `i` (reference
    /// index `i + 1`) and the control applied at step `i`.
    fn stage(&self, i: usize, x: &State, u: &Control, terminal: bool) -> Stage {
        let cfg = self.cfg;
        let r = self.reference;
        let ri = (i + 1).min(r.x.len().saturating_sub(1));
        let scale = if terminal { cfg.weight_terminal } else { 1.0 };
        let (ox, oy) = self.model.origin(x[0], x[1], x[2]);
        let (rs, rc) = r.yaw[ri].sin_cos();
        let dx = ox - r.x[ri];
        let dy = oy - r.y[ri];
        let cte = -dx * rs + dy * rc;
        let along = dx * rc + dy * rs;
        let epsi = normalize_angle(x[2] - r.yaw[ri]);
        let ve = x[3] - r.v[ri];
        let w_cte = scale * cfg.weight_cte;
        let w_epsi = scale * cfg.weight_epsi * r.lateral_at(ri);
        let w_vel = scale * cfg.weight_vel;
        let lat = r.lateral_at(ri);
        let (ease, ease_d, ease_dd) = eased_lateral(cte, lat, r.allow_at(ri));

        let mut cost = w_cte * (ease + 0.25 * along * along)
            + w_epsi * epsi * epsi
            + w_vel * ve * ve
            + cfg.weight_steering * u[0] * u[0]
            + cfg.weight_acceleration * (u[1] * u[1] + u[2] * u[2]);

        // Gradient with respect to the origin position, then chain through
        // the rear-axle offset for the yaw component. vy has no direct
        // tracking term (no reference for it) — its influence on cost is
        // carried entirely through the position it feeds via the state
        // Jacobian's a[(0,4)]/a[(1,4)] terms, propagated by the backward
        // pass exactly like v's forward-velocity coupling.
        let g_ox = w_cte * (-ease_d * rs + 0.5 * along * rc);
        let g_oy = w_cte * (ease_d * rc + 0.5 * along * rs);
        let a_off = self.model.axle_offset;
        let (ys, yc) = x[2].sin_cos();
        let mut lx = State::zeros();
        lx[0] = g_ox;
        lx[1] = g_oy;
        lx[2] = 2.0 * w_epsi * epsi + g_ox * (-a_off * ys) + g_oy * (a_off * yc);
        lx[3] = 2.0 * w_vel * ve;

        let mut lxx = Matrix6::zeros();
        lxx[(0, 0)] = w_cte * (ease_dd * rs * rs + 0.5 * rc * rc);
        lxx[(0, 1)] = w_cte * (-ease_dd * rs * rc + 0.5 * rc * rs);
        lxx[(1, 0)] = lxx[(0, 1)];
        lxx[(1, 1)] = w_cte * (ease_dd * rc * rc + 0.5 * rs * rs);
        lxx[(2, 2)] = 2.0 * w_epsi;
        lxx[(3, 3)] = 2.0 * w_vel;

        // Turn rate against the reference's curvature at the present speed.
        let (omega, d_v, d_s) = if self.model.ackermann {
            let l = wheelbase(self.model.constraints);
            (x[3] * x[5].tan() / l, x[5].tan() / l, x[3] / (l * x[5].cos().powi(2)))
        } else {
            (x[5], 0.0, 1.0)
        };
        let (lo, hi) = (ri.saturating_sub(1), (ri + 1).min(r.yaw.len().saturating_sub(1)));
        let ds = (r.x[hi] - r.x[lo]).hypot(r.y[hi] - r.y[lo]);
        let kappa = if hi > lo && ds > 1e-6 { normalize_angle(r.yaw[hi] - r.yaw[lo]) / ds } else { 0.0 };
        let d_v = d_v - kappa;
        let (e_w, w_w) = (omega - kappa * x[3], scale * cfg.weight_yaw_rate);
        cost += w_w * e_w * e_w;
        lx[3] += 2.0 * w_w * e_w * d_v;
        lx[5] += 2.0 * w_w * e_w * d_s;
        lxx[(3, 3)] += 2.0 * w_w * d_v * d_v;
        lxx[(3, 5)] += 2.0 * w_w * d_v * d_s;
        lxx[(5, 3)] += 2.0 * w_w * d_v * d_s;
        lxx[(5, 5)] += 2.0 * w_w * d_s * d_s;

        if !self.field.is_empty() {
            let b = self.field.at(i + 1, (x[0], x[1], x[2]));
            cost += b.cost;
            for k in 0..3 {
                lx[k] += b.grad[k];
                for l in 0..3 {
                    lxx[(k, l)] += b.hess[k][l];
                }
            }
        }

        let lu = Control::new(
            2.0 * cfg.weight_steering * u[0],
            2.0 * cfg.weight_acceleration * u[1],
            2.0 * cfg.weight_acceleration * u[2],
        );
        let luu = Matrix3::from_diagonal(&Vector3::new(
            2.0 * cfg.weight_steering,
            2.0 * cfg.weight_acceleration,
            2.0 * cfg.weight_acceleration,
        ));
        Stage { cost, lx, lu, lxx, luu }
    }

    fn rollout(&self, x0: &State, us: &[Control]) -> (Vec<State>, f64) {
        let n = us.len();
        let mut xs = Vec::with_capacity(n + 1);
        xs.push(*x0);
        let mut cost = 0.0;
        for i in 0..n {
            let x = self.step(&xs[i], &us[i]);
            cost += self.stage(i, &x, &us[i], i + 1 == n).cost * self.model.dt;
            xs.push(x);
        }
        (xs, cost)
    }

    /// One iLQR solve: returns the control sequence, the cost and the number
    /// of iterations used.
    fn solve(&self, x0: &State, mut us: Vec<Control>) -> (Vec<Control>, f64, usize) {
        let n = us.len();
        let sb = self.model.steer_bound();
        let ab = self.model.accel_bound();
        let lb = self.model.accel_lat_bound();
        let clamp = |u: Control| Control::new(u[0].clamp(-sb, sb), u[1].clamp(-ab, ab), u[2].clamp(-lb, lb));
        for u in us.iter_mut() {
            *u = clamp(*u);
        }
        let (mut xs, mut cost) = self.rollout(x0, &us);
        let mut mu = 1e-3;
        let mut iterations = 0;
        for _ in 0..self.cfg.iterations {
            iterations += 1;
            // Backward pass: stage i's state terms join the cost-to-go at
            // x_{i+1} and reach x_i and u_i through the dynamics.
            let mut vx = State::zeros();
            let mut vxx = Matrix6::zeros();
            let mut ks: Vec<Control> = vec![Control::zeros(); n];
            let mut kks: Vec<Matrix3x6<f64>> = vec![Matrix3x6::zeros(); n];
            let mut ok = true;
            for i in (0..n).rev() {
                let st = self.stage(i, &xs[i + 1], &us[i], i + 1 == n);
                let (a, b) = self.jacobians(&xs[i], &us[i]);
                let wx = st.lx * self.model.dt + vx;
                let wxx = st.lxx * self.model.dt + vxx;
                let lu = st.lu * self.model.dt;
                let luu = st.luu * self.model.dt;
                let qx = a.transpose() * wx;
                let qu = lu + b.transpose() * wx;
                let qxx = a.transpose() * wxx * a;
                let quu = luu + b.transpose() * wxx * b + Matrix3::identity() * mu;
                let qux = b.transpose() * wxx * a;
                let Some((k, kk)) = box_step(&quu, &qu, &qux, &us[i], [sb, ab, lb]) else {
                    ok = false;
                    break;
                };
                vx = qx + kk.transpose() * quu * k + kk.transpose() * qu + qux.transpose() * k;
                vxx = qxx + kk.transpose() * quu * kk + kk.transpose() * qux + qux.transpose() * kk;
                vxx = 0.5 * (vxx + vxx.transpose());
                ks[i] = k;
                kks[i] = kk;
            }
            if !ok {
                mu *= 10.0;
                continue;
            }
            // Forward pass with line search.
            let mut improved = false;
            let mut alpha = 1.0;
            for _ in 0..8 {
                let mut new_us = Vec::with_capacity(n);
                let mut x = *x0;
                let mut new_xs = Vec::with_capacity(n + 1);
                new_xs.push(x);
                let mut new_cost = 0.0;
                for i in 0..n {
                    let du = alpha * ks[i] + kks[i] * (x - xs[i]);
                    let u = clamp(us[i] + du);
                    let nx = self.step(&x, &u);
                    new_cost += self.stage(i, &nx, &u, i + 1 == n).cost * self.model.dt;
                    new_us.push(u);
                    new_xs.push(nx);
                    x = nx;
                }
                if new_cost < cost {
                    let rel = (cost - new_cost) / cost.abs().max(1e-12);
                    us = new_us;
                    xs = new_xs;
                    cost = new_cost;
                    improved = true;
                    mu = (mu * 0.5).max(1e-6);
                    if rel < self.cfg.convergence {
                        return (us, cost, iterations);
                    }
                    break;
                }
                alpha *= 0.5;
            }
            if !improved {
                mu *= 10.0;
                if mu > 1e6 {
                    break;
                }
            }
        }
        (us, cost, iterations)
    }
}

impl IlqrFollower {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_ilqr_config(cfg: IlqrConfig) -> Self {
        let n = cfg.horizon_steps.max(1);
        Self {
            base: ControllerBase::default(),
            cursor: PathCursor::default(),
            steer: vec![0.0; n],
            accel: vec![0.0; n],
            accel_lat: vec![0.0; n],
            predicted_trajectory: Vec::new(),
            is_turning_in_place: false,
            last_v: 0.0,
            last_vy: 0.0,
            actuated: None,
            debug_level: DebugLevel::Off,
            debug: DebugFrame::default(),
            shift_accum: 0.0,
            side: 0.0,
            avoid: Choice::default(),
            last_iterations: 0,
            last_cost: 0.0,
            trajectory: None,
            clock: 0.0,
            ilqr_config: cfg,
        }
    }

    pub fn set_ilqr_config(&mut self, cfg: IlqrConfig) {
        let n = cfg.horizon_steps.max(1);
        self.steer = vec![0.0; n];
        self.accel = vec![0.0; n];
        self.accel_lat = vec![0.0; n];
        self.ilqr_config = cfg;
    }

    pub fn predicted_trajectory(&self) -> &[Point] {
        &self.predicted_trajectory
    }

    /// Iterations used by the last solve.
    pub fn last_iterations(&self) -> usize {
        self.last_iterations
    }

    /// Cost of the last solution.
    pub fn last_cost(&self) -> f64 {
        self.last_cost
    }
}

impl Controller for IlqrFollower {
    fn params(&self) -> serde_json::Value {
        serde_json::to_value(&self.ilqr_config).unwrap_or_default()
    }

    fn set_params(&mut self, params: &serde_json::Value) -> Result<(), String> {
        let cfg = crate::controller::merged(&self.ilqr_config, params)?;
        self.set_ilqr_config(cfg);
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
        let mut cfg = self.ilqr_config.clone();
        (cfg.horizon_steps, cfg.dt) = planning(constraints, cfg.ref_velocity.min(constraints.max_linear_velocity), cfg.horizon_steps, cfg.dt);
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
        }
        let n = cfg.horizon_steps.max(1);
        if self.steer.len() != n {
            self.steer = vec![0.0; n];
            self.accel = vec![0.0; n];
            self.accel_lat = vec![0.0; n];
        }
        let v_now = current_speed(state, self.last_v);
        let vy_now = current_lateral_speed(state, self.last_vy);
        let mut model = Model::new(state, constraints, v_now, vy_now, cfg.dt, prep.allow_reverse);
        if model.ackermann && model.steer_rate.is_finite() {
            if crate::pred::mpc::measures_steering(state) {
                self.actuated = Some(model.steer0);
            } else if let Some(a) = self.actuated {
                model.steer0 = a;
            }
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
        let footprint = footprint_discs(constraints);
        let discs: Vec<(f64, f64)> = footprint.into_iter().map(|(off, r)| (off + model.axle_offset, r)).collect();
        let obstacles: &[Obstacle] = world.map_or(&[], |w| &w.obstacles);
        let along = crate::pred::mppi::Along {
            path: &self.base.path,
            cum: &self.cursor.cum,
            trajectory: self.trajectory.as_ref(),
            clock: self.clock,
            from: prep.proj.arc_length,
        };
        let robot = (state.pose.point.x, state.pose.point.y, crate::core::math::yaw_of(&state.pose));
        let speed = cfg.ref_velocity.min(constraints.max_linear_velocity);
        let guides = guided(&mut reference, &along, obstacles, cfg.dt, speed, constraints, robot, &mut self.avoid);
        let kept = crate::pred::mppi::with_body_room(obstacles, constraints);
        let obstacles: &[Obstacle] = &kept;
        let field = BarrierField::new(obstacles, &discs, model.dt, 1.0, cfg.weight_obstacle, cfg.obstacle_scale);
        let problem = Problem { model: &model, reference: &reference, cfg: &cfg, field: &field };
        let x0 = State::new(model.x0, model.y0, model.yaw0, model.v0, model.vy0, model.steer0.clamp(-model.steer_bound(), model.steer_bound()));
        let warm: Vec<Control> = (0..n).map(|i| Control::new(self.steer[i], self.accel[i], self.accel_lat[i])).collect();
        let mut best: Option<(f64, f64, Vec<Control>, usize)> = None;
        let mut tried: Vec<(Vec<Point>, f64)> = Vec::new();
        if self.debug_level >= DebugLevel::Basic {
            self.debug.guides = guides.iter().skip(1).map(|g| crate::pred::mpc::reference_path(&g.reference)).collect();
        }
        for guide in &guides {
            let side = &guide.side;
            let guided = Problem { model: &model, reference: &guide.reference, cfg: &cfg, field: &field };
            let (us, _, iterations) = guided.solve(&x0, warm.clone());
            let switch = *side != 0.0 && self.side != 0.0 && *side != self.side;
            // Every solve is scored on the same objective: the reference at
            // its own pace, against every obstacle.
            let scored = &problem;
            let (xs, rolled) = scored.rollout(&x0, &us);
            if self.debug_level == DebugLevel::Full {
                tried.push((xs.iter().map(|x| { let (ox, oy) = model.origin(x[0], x[1], x[2]); Point::new(ox, oy, 0.0) }).collect(), rolled));
            }
            let c = rolled * if switch { SIDE_SWITCH_COST } else { 1.0 };
            if c.is_finite() && best.as_ref().is_none_or(|b| c < b.0) {
                best = Some((c, *side, us, iterations));
            }
        }
        if self.debug_level == DebugLevel::Full {
            (self.debug.rollouts, self.debug.costs) = tried.into_iter().unzip();
        }
        let Some((_, side, us, iterations)) = best else {
            return VelocityCommand::invalid("iLQR solve diverged");
        };
        if side != 0.0 || guides.len() == 1 {
            self.side = side;
        }
        let cost = problem.rollout(&x0, &us).1;
        self.last_iterations = iterations;
        self.last_cost = cost;
        if !cost.is_finite() {
            return VelocityCommand::invalid("iLQR solve diverged");
        }
        let (xs, _) = problem.rollout(&x0, &us);
        self.predicted_trajectory = xs
            .iter()
            .map(|x| {
                let (ox, oy) = model.origin(x[0], x[1], x[2]);
                Point::new(ox, oy, 0.0)
            })
            .collect();

        for (i, u) in us.iter().enumerate() {
            self.steer[i] = u[0];
            self.accel[i] = u[1];
            self.accel_lat[i] = u[2];
        }
        let steer0 = self.steer[0];
        self.actuated = Some(crate::pred::mpc::actuated_after(&model, self.actuated, steer0, dt));
        let accel0 = self.accel[0];
        let accel_lat0 = self.accel_lat[0];
        crate::pred::mppi::shift_warm_start(
            &mut self.shift_accum,
            dt,
            cfg.dt,
            [&mut self.steer, &mut self.accel, &mut self.accel_lat],
        );
        let dt_apply = dt.min(cfg.dt);
        let v_cap = speed_at_arc_length(&self.base.path.speeds, &self.cursor.cum, prep.proj.arc_length)
            .unwrap_or(f64::INFINITY)
            .min(crate::pred::mpc::REF_SPEED_HEADROOM * cfg.ref_velocity.max(0.1));
        self.base.status.mode = "ilqr_tracking".into();
        let v_base = crate::pred::mppi::held_speed(v_now, self.last_v, crate::pred::mppi::HELD_SPEED_SLACK);
        let cmd = command_from_controls(
            v_base,
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
            "iLQR tracking",
        );
        self.last_v = (v_base + accel0 * dt_apply).clamp(-v_cap, v_cap);
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
        let n = self.ilqr_config.horizon_steps.max(1);
        self.steer = vec![0.0; n];
        self.accel = vec![0.0; n];
        self.accel_lat = vec![0.0; n];
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
        "ilqr_follower"
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

/// Feedforward `k` and feedback `K` of one backward-pass stage with the
/// controls kept inside `±bounds` (box-DDP, Tassa et al. 2014): a control the
/// unconstrained step would push past its bound is held at the bound, its
/// feedback row is zeroed, and the free controls are solved given it.
fn box_step(
    quu: &Matrix3<f64>,
    qu: &Control,
    qux: &Matrix3x6<f64>,
    u: &Control,
    bounds: [f64; 3],
) -> Option<(Control, Matrix3x6<f64>)> {
    let inv = quu.try_inverse()?;
    let k = -inv * qu;
    let clamped: Vec<usize> = (0..3).filter(|&j| (u[j] + k[j]).abs() > bounds[j] + 1e-12).collect();
    if clamped.is_empty() {
        return Some((k, -inv * qux));
    }
    let free: Vec<usize> = (0..3).filter(|j| !clamped.contains(j)).collect();
    let mut k_box = Control::zeros();
    for &j in &clamped {
        k_box[j] = (u[j] + k[j]).clamp(-bounds[j], bounds[j]) - u[j];
    }
    let mut kk = Matrix3x6::zeros();
    if !free.is_empty() {
        let m = free.len();
        let mut a = nalgebra::DMatrix::<f64>::zeros(m, m);
        let mut rhs = nalgebra::DVector::<f64>::zeros(m);
        let mut rhs_x = nalgebra::DMatrix::<f64>::zeros(m, 6);
        for (r, &fr) in free.iter().enumerate() {
            rhs[r] = -qu[fr] - clamped.iter().map(|&c| quu[(fr, c)] * k_box[c]).sum::<f64>();
            for col in 0..6 {
                rhs_x[(r, col)] = -qux[(fr, col)];
            }
            for (c, &fc) in free.iter().enumerate() {
                a[(r, c)] = quu[(fr, fc)];
            }
        }
        let inv_free = a.try_inverse()?;
        let k_free = &inv_free * rhs;
        let kk_free = &inv_free * rhs_x;
        for (r, &fr) in free.iter().enumerate() {
            k_box[fr] = k_free[r];
            for col in 0..6 {
                kk[(fr, col)] = kk_free[(r, col)];
            }
        }
    }
    Some((k_box, kk))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SteeringType;

    fn solved(steering: SteeringType, steer_rate: f64) -> (f64, f64) {
        let constraints = RobotConstraints {
            steering_type: steering,
            wheelbase: 0.5,
            max_steering_angle: 1.2,
            min_turning_radius: 0.0,
            max_steering_rate: steer_rate,
            max_angular_velocity: 5.0,
            max_linear_acceleration: 5.0,
            max_linear_velocity: 2.0,
            ..Default::default()
        };
        let cfg = IlqrConfig { iterations: 300, convergence: 1e-14, ..Default::default() };
        let n = cfg.horizon_steps;
        let x0 = State::new(0.0, 0.0, 0.0, 0.5, 0.0, 0.0);
        let model = Model::new(&RobotState::default(), &constraints, 0.5, 0.0, cfg.dt, false);
        let reference = Reference {
            x: (0..=n).map(|i| 0.05 * i as f64).collect(),
            y: vec![0.3; n + 1],
            yaw: vec![0.0; n + 1],
            v: vec![0.5; n + 1],
            lateral: Vec::new(),
            allow: Vec::new(),
        };
        let problem = Problem { model: &model, reference: &reference, cfg: &cfg, field: &BarrierField::new(&[], &[], model.dt, 1.0, 0.0, 1.0) };
        let (us, cost, _) = problem.solve(&x0, vec![Control::zeros(); n]);
        let h = 1e-6;
        let mut worst: f64 = 0.0;
        for i in 0..n {
            for k in 0..2 {
                let (mut up, mut down) = (us.clone(), us.clone());
                up[i][k] += h;
                down[i][k] -= h;
                let g = (problem.rollout(&x0, &up).1 - problem.rollout(&x0, &down).1) / (2.0 * h);
                worst = worst.max(g.abs());
            }
        }
        (worst, cost)
    }

    #[test]
    fn a_solve_with_saturated_steering_meets_the_box_optimality_conditions() {
        let constraints = RobotConstraints {
            steering_type: SteeringType::Differential,
            max_angular_velocity: 0.3,
            max_linear_acceleration: 5.0,
            max_linear_velocity: 2.0,
            ..Default::default()
        };
        let cfg = IlqrConfig { iterations: 300, convergence: 1e-14, ..Default::default() };
        let n = cfg.horizon_steps;
        let x0 = State::new(0.0, 0.0, 0.0, 0.5, 0.0, 0.0);
        let model = Model::new(&RobotState::default(), &constraints, 0.5, 0.0, cfg.dt, false);
        let reference = Reference {
            x: (0..=n).map(|i| 0.05 * i as f64).collect(),
            y: vec![1.5; n + 1],
            yaw: vec![0.0; n + 1],
            v: vec![0.5; n + 1],
            lateral: Vec::new(),
            allow: Vec::new(),
        };
        let problem = Problem { model: &model, reference: &reference, cfg: &cfg, field: &BarrierField::new(&[], &[], model.dt, 1.0, 0.0, 1.0) };
        let (us, cost, _) = problem.solve(&x0, vec![Control::zeros(); n]);
        let bound = model.steer_bound();
        assert!(us.iter().any(|u| (u[0].abs() - bound).abs() < 1e-9), "steering never saturated");
        let h = 1e-6;
        for (i, u) in us.iter().enumerate() {
            let (mut up, mut down) = (us.clone(), us.clone());
            up[i][0] += h;
            down[i][0] -= h;
            let g = (problem.rollout(&x0, &up).1 - problem.rollout(&x0, &down).1) / (2.0 * h);
            let tol = 1e-3 * cost.max(1.0);
            if u[0] >= bound - 1e-9 {
                assert!(g <= tol, "step {i}: at the upper bound but the cost falls inward ({g:.3e})");
            } else if u[0] <= -bound + 1e-9 {
                assert!(g >= -tol, "step {i}: at the lower bound but the cost falls inward ({g:.3e})");
            } else {
                assert!(g.abs() < tol, "step {i}: free steering with gradient {g:.3e}");
            }
        }
    }

    #[test]
    fn a_converged_solve_is_a_stationary_point_of_its_cost() {
        for (steering, rate) in [(SteeringType::Differential, 0.0), (SteeringType::Ackermann, 0.0), (SteeringType::Ackermann, 0.6)] {
            let (gradient, cost) = solved(steering, rate);
            assert!(gradient < 1e-3 * cost.max(1.0), "{steering:?} at {rate} rad/s: gradient {gradient:.3e} at cost {cost:.3}");
        }
    }
}
