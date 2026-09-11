//! iLQR tracker behaviour.

#![allow(clippy::field_reassign_with_default)]

use datapod::{Euler, Point, Pose, Quaternion};
use ondrive::{
    Controller, ControllerConfig, Goal, IlqrConfig, IlqrFollower, MpcConfig, MpcFollower,
    OutputUnits, Path, RobotConstraints, RobotState, SteeringType,
};
use std::f64::consts::PI;

fn pose(x: f64, y: f64, yaw: f64) -> Pose {
    Pose {
        point: Point::new(x, y, 0.0),
        rotation: Quaternion::from_euler(Euler::new(0.0, 0.0, yaw)),
    }
}

fn sine() -> Path {
    let mut p = Path::default();
    p.waypoints = (0..=60)
        .map(|i| {
            let x = i as f64 * 0.25;
            pose(x, (x * 0.4).sin() * 1.0, 0.0)
        })
        .collect();
    p
}

fn constraints(steering: SteeringType) -> RobotConstraints {
    let mut c = RobotConstraints::default();
    c.steering_type = steering;
    c.max_linear_velocity = 1.0;
    c.min_linear_velocity = -0.6;
    c.max_angular_velocity = 2.0;
    c.max_linear_acceleration = 1.0;
    c.max_steering_angle = 0.6;
    c.min_turning_radius = 0.0;
    c.wheelbase = 0.5;
    c
}

fn config() -> ControllerConfig {
    let mut cfg = ControllerConfig::default();
    cfg.output_units = OutputUnits::Physical;
    cfg.goal_tolerance = 0.3;
    cfg.angular_tolerance = PI;
    cfg
}

fn goal(path: &Path) -> Goal {
    Goal { target_pose: *path.waypoints.last().unwrap(), tolerance_position: 0.3, tolerance_orientation: PI, ..Default::default() }
}

fn track(ctrl: &mut dyn Controller, c: &RobotConstraints, dt: f64, mut hook: impl FnMut(usize, &ondrive::VelocityCommand)) -> (bool, f64) {
    let path = sine();
    let g = goal(&path);
    ctrl.set_config(config());
    ctrl.set_path(path);
    let mut state = RobotState { pose: pose(0.0, 0.3, 0.0), allow_move: true, ..Default::default() };
    let mut max_cte: f64 = 0.0;
    for i in 0..(80.0 / dt) as usize {
        let cmd = ctrl.compute_control(&state, &g, c, dt, None);
        assert!(cmd.valid, "{}", cmd.status_message);
        hook(i, &cmd);
        let yaw = state.pose.rotation.to_euler().yaw;
        state.pose.point.x += cmd.linear_velocity * yaw.cos() * dt;
        state.pose.point.y += cmd.linear_velocity * yaw.sin() * dt;
        state.pose.rotation = Quaternion::from_euler(Euler::new(0.0, 0.0, yaw + cmd.angular_velocity * dt));
        state.velocity.linear = cmd.linear_velocity;
        if i as f64 * dt > 3.0 {
            max_cte = max_cte.max(ctrl.get_status().cross_track_error.abs());
        }
        if ctrl.get_status().goal_reached {
            return (true, max_cte);
        }
    }
    (false, max_cte)
}

#[test]
fn ilqr_tracks_as_well_as_mpc() {
    for steering in [SteeringType::Differential, SteeringType::Ackermann] {
        let c = constraints(steering);
        let mut ilqr = IlqrFollower::new();
        let (reached, cte_ilqr) = track(&mut ilqr, &c, 0.1, |_, _| {});
        assert!(reached, "{steering:?}: iLQR did not reach the end");
        let mut mpc = MpcFollower::with_mpc_config(MpcConfig::default());
        let (reached_mpc, cte_mpc) = track(&mut mpc, &c, 0.1, |_, _| {});
        assert!(reached_mpc);
        assert!(cte_ilqr <= cte_mpc + 0.05, "{steering:?}: iLQR settled CTE {cte_ilqr:.3} vs MPC {cte_mpc:.3}");
        assert!(cte_ilqr < 0.3, "{steering:?}: iLQR settled CTE {cte_ilqr:.3}");
    }
}

#[test]
fn ilqr_converges_quickly_from_warm_start() {
    let c = constraints(SteeringType::Ackermann);
    let mut ilqr = IlqrFollower::with_ilqr_config(IlqrConfig::default());
    let path = sine();
    let g = goal(&path);
    ilqr.set_config(config());
    ilqr.set_path(path);
    let mut state = RobotState { pose: pose(0.0, 0.3, 0.0), allow_move: true, ..Default::default() };
    let mut warm_iterations = Vec::new();
    for i in 0..60 {
        let cmd = ilqr.compute_control(&state, &g, &c, 0.1, None);
        if i >= 10 {
            warm_iterations.push(ilqr.last_iterations());
        }
        let yaw = state.pose.rotation.to_euler().yaw;
        state.pose.point.x += cmd.linear_velocity * yaw.cos() * 0.1;
        state.pose.point.y += cmd.linear_velocity * yaw.sin() * 0.1;
        state.pose.rotation = Quaternion::from_euler(Euler::new(0.0, 0.0, yaw + cmd.angular_velocity * 0.1));
        state.velocity.linear = cmd.linear_velocity;
    }
    let mean = warm_iterations.iter().sum::<usize>() as f64 / warm_iterations.len() as f64;
    assert!(mean < 10.0, "mean warm-start iterations {mean:.1}");
}

#[test]
fn ilqr_respects_acceleration_at_50hz() {
    let c = constraints(SteeringType::Differential);
    let mut ilqr = IlqrFollower::new();
    let mut prev = 0.0;
    track(&mut ilqr, &c, 0.02, |i, cmd| {
        if cmd.status_message == "Goal reached" {
            return;
        }
        let dv = (cmd.linear_velocity - prev).abs();
        assert!(dv <= c.max_linear_acceleration * 0.02 + 1e-6, "tick {i}: dv {dv:.4}");
        prev = cmd.linear_velocity;
    });
}
