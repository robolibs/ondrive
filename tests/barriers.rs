//! MPC and iLQR obstacle barriers: the barrier's derivatives, and both
//! controllers passing a static obstacle on the path and a robot coming
//! head-on.
//!
//!     cargo test --test barriers

#![allow(clippy::field_reassign_with_default)]

use datapod::{Euler, Point, Pose, Quaternion};
use ondrive::core::obstacles::{footprint_discs, obstacle_barrier};
use ondrive::{
    Controller, ControllerConfig, GaussianMode, Goal, IlqrFollower, MpcFollower, Obstacle, OutputUnits, Path,
    RobotConstraints, RobotState, Velocity, WorldConstraints,
};

fn pose(x: f64, y: f64, yaw: f64) -> Pose {
    Pose { point: Point::new(x, y, 0.0), rotation: Quaternion::from_euler(Euler::new(0.0, 0.0, yaw)) }
}

fn obstacle(x: f64, y: f64, vx: f64, steps: usize, dt: f64) -> Obstacle {
    Obstacle {
        id: 0,
        radius: 0.35,
        dt,
        modes: vec![GaussianMode {
            weight: 1.0,
            mean_x: (0..steps).map(|k| x + vx * k as f64 * dt).collect(),
            mean_y: vec![y; steps],
            std_x: (0..steps).map(|k| 0.05 + 0.05 * k as f64 * dt).collect(),
            std_y: (0..steps).map(|k| 0.05 + 0.05 * k as f64 * dt).collect(),
        }],
    }
}

fn constraints() -> RobotConstraints {
    let mut c = RobotConstraints::default();
    c.max_linear_velocity = 1.0;
    c.max_angular_velocity = 1.5;
    c.max_linear_acceleration = 1.5;
    c.robot_width = 0.6;
    c.robot_length = 0.9;
    c
}

/// Clearance between the footprint discs at `state` and the obstacle now.
fn clearance(c: &RobotConstraints, state: &RobotState, obs: &Obstacle) -> f64 {
    let yaw = state.pose.rotation.to_euler().yaw;
    let m = &obs.modes[0];
    footprint_discs(c)
        .iter()
        .map(|(off, r)| {
            let (px, py) = (state.pose.point.x + off * yaw.cos(), state.pose.point.y + off * yaw.sin());
            (px - m.mean_x[0]).hypot(py - m.mean_y[0]) - r - obs.radius
        })
        .fold(f64::INFINITY, f64::min)
}

/// Drives `ctl` along a 14 m line through the obstacle `at(t)`; returns the
/// smallest clearance and how far along the robot got.
fn drive(ctl: &mut dyn Controller, at: &dyn Fn(f64) -> Obstacle) -> (f64, f64) {
    let c = constraints();
    let mut path = Path::default();
    path.waypoints = (0..=56).map(|i| pose(i as f64 * 0.25, 0.0, 0.0)).collect();
    let mut cfg = ControllerConfig::default();
    cfg.goal_tolerance = 0.3;
    cfg.angular_tolerance = std::f64::consts::PI;
    cfg.output_units = OutputUnits::Physical;
    ctl.set_config(cfg);
    ctl.set_path(path);
    let goal = Goal { target_pose: pose(14.0, 0.0, 0.0), tolerance_position: 0.3, ..Default::default() };
    let mut state = RobotState { pose: pose(0.0, 0.0, 0.0), allow_move: true, ..Default::default() };
    let dt = 0.1;
    let mut worst = f64::INFINITY;
    for k in 0..300 {
        let obs = at(k as f64 * dt);
        worst = worst.min(clearance(&c, &state, &obs));
        let world = WorldConstraints { obstacles: vec![obs], ..Default::default() };
        let cmd = ctl.compute_control(&state, &goal, &c, dt, Some(&world));
        if ctl.get_status().goal_reached {
            break;
        }
        assert!(cmd.valid, "{}", cmd.status_message);
        if std::env::var_os("BARRIER_TRACE").is_some() && k % 5 == 0 {
            println!(
                "t={:4.1} x={:5.2} y={:+.2} v={:+.2} w={:+.2} clear={:.2} {}",
                k as f64 * dt,
                state.pose.point.x,
                state.pose.point.y,
                cmd.linear_velocity,
                cmd.angular_velocity,
                worst,
                cmd.status_message
            );
        }
        let yaw = state.pose.rotation.to_euler().yaw;
        state.pose.point.x += cmd.linear_velocity * yaw.cos() * dt;
        state.pose.point.y += cmd.linear_velocity * yaw.sin() * dt;
        state.pose.rotation = Quaternion::from_euler(Euler::new(0.0, 0.0, yaw + cmd.angular_velocity * dt));
        state.velocity = Velocity { linear: cmd.linear_velocity, angular: cmd.angular_velocity, ..Default::default() };
        state.velocity_valid = true;
    }
    (worst, state.pose.point.x)
}

#[test]
fn the_barrier_gradient_matches_finite_differences() {
    let obs = [obstacle(1.0, 0.4, 0.3, 20, 0.2)];
    let discs = [(-0.3, 0.35), (0.3, 0.35)];
    let at = |p: (f64, f64, f64)| obstacle_barrier(&obs, &discs, 0.5, 0.1, 1.0, p, 300.0, 0.3);
    let p = (0.2, 0.1, 0.4);
    let b = at(p);
    assert!(b.cost > 0.0);
    let h = 1e-6;
    for k in 0..3 {
        let mut hi = p;
        let mut lo = p;
        match k {
            0 => (hi.0 += h, lo.0 -= h),
            1 => (hi.1 += h, lo.1 -= h),
            _ => (hi.2 += h, lo.2 -= h),
        };
        let fd = (at(hi).cost - at(lo).cost) / (2.0 * h);
        assert!((fd - b.grad[k]).abs() < 1e-4 * fd.abs().max(1.0), "d/d{k}: {fd} vs {}", b.grad[k]);
        assert!(b.hess[k][k] >= 0.0);
    }
}

#[test]
fn mpc_and_ilqr_pass_a_static_obstacle_on_the_path() {
    let still = |_: f64| obstacle(6.0, 0.0, 0.0, 30, 0.15);
    for (name, ctl) in [
        ("mpc", &mut MpcFollower::default() as &mut dyn Controller),
        ("ilqr", &mut IlqrFollower::default() as &mut dyn Controller),
    ] {
        let (clear, reached) = drive(ctl, &still);
        assert!(clear > 0.0, "{name} hit the obstacle: clearance {clear:.2}");
        assert!(reached > 12.0, "{name} stopped at x = {reached:.2}");
    }
}

#[test]
fn mpc_and_ilqr_avoid_a_robot_coming_head_on() {
    let coming = |t: f64| obstacle(12.0 - 0.6 * t, 0.0, -0.6, 30, 0.15);
    for (name, ctl) in [
        ("mpc", &mut MpcFollower::default() as &mut dyn Controller),
        ("ilqr", &mut IlqrFollower::default() as &mut dyn Controller),
    ] {
        let (clear, _) = drive(ctl, &coming);
        assert!(clear > 0.0, "{name} hit the oncoming robot: clearance {clear:.2}");
    }
}

#[test]
fn debug_frames_record_rollouts_and_the_guides_round_an_obstacle() {
    use ondrive::{DebugLevel, Tracker, TrackerKind};
    let mut path = Path::default();
    path.waypoints = (0..=40).map(|i| pose(i as f64 * 0.25, 0.0, 0.0)).collect();
    let world = WorldConstraints { obstacles: vec![obstacle(3.0, 0.0, 0.0, 30, 0.15)], ..Default::default() };
    let state = RobotState { pose: pose(0.0, 0.0, 0.0), allow_move: true, ..Default::default() };
    for kind in [TrackerKind::Mppi, TrackerKind::Mca, TrackerKind::Soc, TrackerKind::Mpc, TrackerKind::Ilqr, TrackerKind::Dwa] {
        let mut t = Tracker::new(kind);
        t.init(constraints());
        t.set_path(path.clone());
        t.tick(&state, 0.05, Some(&world));
        assert!(t.debug_frame().rollouts.is_empty(), "{kind:?} recorded with debugging off");
        t.set_debug(DebugLevel::Full);
        t.tick(&state, 0.05, Some(&world));
        let frame = t.debug_frame();
        assert!(!frame.rollouts.is_empty(), "{kind:?} recorded no rollouts");
        assert_eq!(frame.rollouts.len(), frame.costs.len(), "{kind:?}");
        if kind != TrackerKind::Dwa {
            assert!(!frame.guides.is_empty(), "{kind:?} recorded no guides round the obstacle");
        }
    }
}
