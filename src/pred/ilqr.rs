//! iLQR trajectory tracker: iterative LQR / DDP on the kinematic model with
//! analytic Jacobians, a regularised backward pass, a line-searched forward
//! pass with control clamping, and a warm start from the shifted previous
//! solution. State `[x, y, yaw, v]`, controls `[steer_or_omega, accel]`.

use crate::controller::{Controller, ControllerBase};
use crate::core::kinematics::wheelbase;
use crate::core::math::normalize_angle;
use crate::core::path::PathCursor;
use crate::pred::mppi::{
    Model, Reference, build_reference, command_from_controls, current_speed, prepare,
    speed_at_arc_length, update_turn_in_place,
};
use crate::types::{Goal, Path, RobotConstraints, RobotState, VelocityCommand, WorldConstraints};
use datapod::Point;
use nalgebra::{Matrix2, Matrix2x4, Matrix4, Matrix4x2, Vector2, Vector4};

#[derive(Clone, Debug)]
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

    pub ref_velocity: f64,
    pub turn_first_activation_deg: f64,
    pub turn_first_release_deg: f64,
    pub approach_taper_distance: f64,
}

impl Default for IlqrConfig {
    fn default() -> Self {
        Self {
            horizon_steps: 20,
            dt: 0.1,
            iterations: 15,
            convergence: 1e-4,
            weight_cte: 100.0,
            weight_epsi: 100.0,
            weight_vel: 50.0,
            weight_steering: 10.0,
            weight_acceleration: 5.0,
            weight_terminal: 3.0,
            ref_velocity: 1.0,
            turn_first_activation_deg: 60.0,
            turn_first_release_deg: 15.0,
            approach_taper_distance: 2.0,
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
    predicted_trajectory: Vec<Point>,
    is_turning_in_place: bool,
    last_v: f64,
    shift_accum: f64,
    last_iterations: usize,
    last_cost: f64,
}

impl Default for IlqrFollower {
    fn default() -> Self {
        Self::with_ilqr_config(IlqrConfig::default())
    }
}

type State = Vector4<f64>;
type Control = Vector2<f64>;

struct Stage {
    cost: f64,
    lx: State,
    lu: Control,
    lxx: Matrix4<f64>,
    luu: Matrix2<f64>,
}

struct Problem<'a> {
    model: &'a Model<'a>,
    reference: &'a Reference,
    cfg: &'a IlqrConfig,
}

impl Problem<'_> {
    fn step(&self, x: &State, u: &Control) -> State {
        let (mut px, mut py, mut yaw, mut v) = (x[0], x[1], x[2], x[3]);
        let sb = self.model.steer_bound();
        let ab = self.model.accel_bound();
        self.model
            .step(&mut px, &mut py, &mut yaw, &mut v, u[0].clamp(-sb, sb), u[1].clamp(-ab, ab));
        State::new(px, py, yaw, v)
    }

    fn jacobians(&self, x: &State, u: &Control) -> (Matrix4<f64>, Matrix4x2<f64>) {
        let dt = self.model.dt;
        let (s, c) = x[2].sin_cos();
        let v = x[3];
        let mut a = Matrix4::identity();
        a[(0, 2)] = -v * s * dt;
        a[(0, 3)] = c * dt;
        a[(1, 2)] = v * c * dt;
        a[(1, 3)] = s * dt;
        let mut b = Matrix4x2::zeros();
        if self.model.ackermann {
            let l = wheelbase(self.model.constraints);
            let sb = self.model.steer_bound();
            let delta = u[0].clamp(-sb, sb);
            a[(2, 3)] = delta.tan() / l * dt;
            b[(2, 0)] = v / (l * delta.cos().powi(2)) * dt;
        } else {
            b[(2, 0)] = dt;
        }
        b[(3, 1)] = dt;
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
        let w_epsi = scale * cfg.weight_epsi;
        let w_vel = scale * cfg.weight_vel;

        let cost = w_cte * (cte * cte + 0.25 * along * along)
            + w_epsi * epsi * epsi
            + w_vel * ve * ve
            + cfg.weight_steering * u[0] * u[0]
            + cfg.weight_acceleration * u[1] * u[1];

        // Gradient with respect to the origin position, then chain through
        // the rear-axle offset for the yaw component.
        let g_ox = 2.0 * w_cte * (-cte * rs + 0.25 * along * rc);
        let g_oy = 2.0 * w_cte * (cte * rc + 0.25 * along * rs);
        let a_off = self.model.axle_offset;
        let (ys, yc) = x[2].sin_cos();
        let mut lx = State::zeros();
        lx[0] = g_ox;
        lx[1] = g_oy;
        lx[2] = 2.0 * w_epsi * epsi + g_ox * (-a_off * ys) + g_oy * (a_off * yc);
        lx[3] = 2.0 * w_vel * ve;

        let mut lxx = Matrix4::zeros();
        let h = 2.0 * w_cte;
        lxx[(0, 0)] = h * (rs * rs + 0.25 * rc * rc);
        lxx[(0, 1)] = h * (-rs * rc + 0.25 * rc * rs);
        lxx[(1, 0)] = lxx[(0, 1)];
        lxx[(1, 1)] = h * (rc * rc + 0.25 * rs * rs);
        lxx[(2, 2)] = 2.0 * w_epsi;
        lxx[(3, 3)] = 2.0 * w_vel;

        let lu = Control::new(2.0 * cfg.weight_steering * u[0], 2.0 * cfg.weight_acceleration * u[1]);
        let luu = Matrix2::new(2.0 * cfg.weight_steering, 0.0, 0.0, 2.0 * cfg.weight_acceleration);
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
        let clamp = |u: Control| Control::new(u[0].clamp(-sb, sb), u[1].clamp(-ab, ab));
        for u in us.iter_mut() {
            *u = clamp(*u);
        }
        let (mut xs, mut cost) = self.rollout(x0, &us);
        let mut mu = 1e-3;
        let mut iterations = 0;
        for _ in 0..self.cfg.iterations {
            iterations += 1;
            // Backward pass.
            let terminal = self.stage(n - 1, &xs[n], &us[n - 1], true);
            let mut vx = terminal.lx * self.model.dt;
            let mut vxx = terminal.lxx * self.model.dt;
            let mut ks: Vec<Control> = vec![Control::zeros(); n];
            let mut kks: Vec<Matrix2x4<f64>> = vec![Matrix2x4::zeros(); n];
            let mut ok = true;
            for i in (0..n).rev() {
                let st = self.stage(i, &xs[i + 1], &us[i], i + 1 == n);
                let (a, b) = self.jacobians(&xs[i], &us[i]);
                let lx = st.lx * self.model.dt;
                let lu = st.lu * self.model.dt;
                let lxx = st.lxx * self.model.dt;
                let luu = st.luu * self.model.dt;
                // Value derivatives propagate through the state after the step.
                let qx = if i + 1 == n { lx + a.transpose() * vx } else { lx + a.transpose() * vx };
                let qu = lu + b.transpose() * vx;
                let qxx = lxx + a.transpose() * vxx * a;
                let quu = luu + b.transpose() * vxx * b + Matrix2::identity() * mu;
                let qux = b.transpose() * vxx * a;
                let Some(quu_inv) = quu.try_inverse() else {
                    ok = false;
                    break;
                };
                let k = -quu_inv * qu;
                let kk = -quu_inv * qux;
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
            predicted_trajectory: Vec::new(),
            is_turning_in_place: false,
            last_v: 0.0,
            shift_accum: 0.0,
            last_iterations: 0,
            last_cost: 0.0,
            ilqr_config: cfg,
        }
    }

    pub fn set_ilqr_config(&mut self, cfg: IlqrConfig) {
        let n = cfg.horizon_steps.max(1);
        self.steer = vec![0.0; n];
        self.accel = vec![0.0; n];
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
            Err(cmd) => {
                self.predicted_trajectory.clear();
                return cmd;
            }
        };
        let mut cfg = self.ilqr_config.clone();
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
        }
        let v_now = current_speed(state, self.last_v);
        let model = Model::new(state, constraints, v_now, cfg.dt, prep.allow_reverse);
        let reference = build_reference(
            &self.base.path,
            &self.cursor.cum,
            prep.proj.arc_length,
            n,
            cfg.dt,
            cfg.ref_velocity.min(constraints.max_linear_velocity),
            cfg.approach_taper_distance,
        );
        let problem = Problem { model: &model, reference: &reference, cfg: &cfg };
        let x0 = State::new(model.x0, model.y0, model.yaw0, model.v0);
        let us: Vec<Control> = (0..n).map(|i| Control::new(self.steer[i], self.accel[i])).collect();
        let (us, cost, iterations) = problem.solve(&x0, us);
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

        for i in 0..n {
            self.steer[i] = us[i][0];
            self.accel[i] = us[i][1];
        }
        let steer0 = self.steer[0];
        let accel0 = self.accel[0];
        self.shift_accum += dt;
        if self.shift_accum >= cfg.dt {
            self.shift_accum -= cfg.dt;
            self.steer.rotate_left(1);
            self.accel.rotate_left(1);
            if n > 1 {
                self.steer[n - 1] = self.steer[n - 2];
                self.accel[n - 1] = self.accel[n - 2];
            }
        }
        let dt_apply = dt.min(cfg.dt);
        let v_cap = speed_at_arc_length(&self.base.path.speeds, &self.cursor.cum, prep.proj.arc_length)
            .unwrap_or(f64::INFINITY);
        self.base.status.mode = "ilqr_tracking".into();
        let cmd = command_from_controls(
            v_now,
            steer0,
            accel0,
            dt_apply,
            v_cap,
            constraints,
            &self.base.config,
            prep.allow_reverse,
            turning,
            prep.epsi,
            "iLQR tracking",
        );
        self.last_v = (v_now + accel0 * dt_apply).clamp(-v_cap, v_cap);
        cmd
    }

    fn set_path(&mut self, path: Path) {
        self.cursor.set_path(&path.waypoints);
        self.base.path = path;
        self.base.path_index = 0;
        self.base.status = Default::default();
        let n = self.ilqr_config.horizon_steps.max(1);
        self.steer = vec![0.0; n];
        self.accel = vec![0.0; n];
        self.predicted_trajectory.clear();
        self.is_turning_in_place = false;
        self.shift_accum = 0.0;
    }

    fn reset(&mut self) {
        self.set_path(Path::default());
        self.last_v = 0.0;
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
