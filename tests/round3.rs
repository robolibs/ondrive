//! Adaptive lookahead, iterative learning control and the potential field.

#![allow(clippy::field_reassign_with_default)]

use datapod::{Euler, Point, Pose, Quaternion};
use ondrive::{
    ApfFollower, Controller, ControllerConfig, GaussianMode, Goal, IlcFollower, Obstacle,
    OutputUnits, Path, PurePursuitFollower, RobotConstraints, RobotState, SteeringType,
    WorldConstraints,
};
use std::f64::consts::PI;

fn pose(x: f64, y: f64, yaw: f64) -> Pose {
    Pose { point: Point::new(x, y, 0.0), rotation: Quaternion::from_euler(Euler::new(0.0, 0.0, yaw)) }
}

fn constraints(steering: SteeringType) -> RobotConstraints {
    let mut c = RobotConstraints::default();
    c.steering_type = steering;
    c.max_linear_velocity = 1.0;
    c.min_linear_velocity = -0.6;
    c.max_angular_velocity = 2.0;
    c.max_linear_acceleration = 1.5;
    c.max_steering_angle = 0.6;
    c.min_turning_radius = 0.0;
    c.wheelbase = 0.5;
    c.robot_width = 0.4;
    c.robot_length = 0.6;
    c
}

fn config() -> ControllerConfig {
    let mut cfg = ControllerConfig::default();
    cfg.output_units = OutputUnits::Physical;
    cfg.goal_tolerance = 0.3;
    cfg.angular_tolerance = PI;
    cfg.lookahead_distance = 1.0;
    cfg.kp_linear = 1.5;
    cfg.kp_angular = 2.0;
    cfg
}

fn integrate(state: &mut RobotState, cmd: &ondrive::VelocityCommand, dt: f64) {
    let yaw = state.pose.rotation.to_euler().yaw;
    state.pose.point.x += cmd.linear_velocity * yaw.cos() * dt;
    state.pose.point.y += cmd.linear_velocity * yaw.sin() * dt;
    state.pose.rotation = Quaternion::from_euler(Euler::new(0.0, 0.0, yaw + cmd.angular_velocity * dt));
    state.velocity.linear = cmd.linear_velocity;
}

fn circle(r: f64) -> Path {
    let mut p = Path::default();
    let n = (1.5 * PI * r / 0.25) as usize;
    p.waypoints = (0..=n)
        .map(|i| {
            let a = i as f64 * 0.25 / r;
            pose(r * a.sin(), r * (1.0 - a.cos()), a)
        })
        .collect();
    p
}

#[test]
fn lookahead_grows_with_speed_and_shrinks_on_curvature() {
    let c = constraints(SteeringType::Ackermann);
    let straight = {
        let mut p = Path::default();
        p.waypoints = (0..=40).map(|i| pose(i as f64 * 0.25, 0.0, 0.0)).collect();
        p
    };
    let goal = Goal { target_pose: *straight.waypoints.last().unwrap(), tolerance_position: 0.3, tolerance_orientation: PI, ..Default::default() };
    let lookahead_at = |speed: f64, path: &Path, lookahead_time: f64| -> f64 {
        let mut pp = PurePursuitFollower::new();
        let mut cfg = config();
        cfg.lookahead_time = lookahead_time;
        pp.set_config(cfg);
        pp.set_path(path.clone());
        let state = RobotState {
            pose: path.waypoints[0],
            velocity: ondrive::Velocity { linear: speed, ..Default::default() },
            allow_move: true,
            ..Default::default()
        };
        pp.compute_control(&state, &goal, &c, 0.05, None);
        let target = pp.lookahead_point().unwrap();
        target.distance_to_2d(path.waypoints[0].point)
    };
    let slow = lookahead_at(0.0, &straight, 0.5);
    let fast = lookahead_at(1.0, &straight, 0.5);
    let fixed = lookahead_at(1.0, &straight, 0.0);
    assert!((slow - 1.0).abs() < 0.05, "slow lookahead {slow:.3}");
    assert!((fast - 1.5).abs() < 0.05, "fast lookahead {fast:.3}");
    assert!((fixed - 1.0).abs() < 0.05, "fixed lookahead {fixed:.3}");

    let tight = circle(1.0);
    let goal_t = Goal { target_pose: *tight.waypoints.last().unwrap(), tolerance_position: 0.3, tolerance_orientation: PI, ..Default::default() };
    let mut pp = PurePursuitFollower::new();
    pp.set_config(config());
    pp.set_path(tight.clone());
    let start = tight.waypoints[10];
    let state = RobotState { pose: start, allow_move: true, ..Default::default() };
    pp.compute_control(&state, &goal_t, &c, 0.05, None);
    let on_curve = pp.lookahead_point().unwrap().distance_to_2d(start.point);
    assert!(on_curve < 0.85, "lookahead on a 1 m radius should shrink, got chord {on_curve:.3}");
}

fn run_pass(ctrl: &mut dyn Controller, path: &Path, c: &RobotConstraints) -> f64 {
    let goal = Goal { target_pose: *path.waypoints.last().unwrap(), tolerance_position: 0.3, tolerance_orientation: PI, ..Default::default() };
    ctrl.set_path(path.clone());
    let mut state = RobotState { pose: pose(0.0, 0.0, 0.0), allow_move: true, ..Default::default() };
    let mut sum = 0.0;
    let mut n = 0;
    for i in 0..3000 {
        let cmd = ctrl.compute_control(&state, &goal, c, 0.05, None);
        assert!(cmd.valid);
        integrate(&mut state, &cmd, 0.05);
        if i > 40 {
            sum += ctrl.get_status().cross_track_error.abs();
            n += 1;
        }
        if ctrl.get_status().goal_reached {
            break;
        }
    }
    assert!(ctrl.get_status().goal_reached, "pass did not finish");
    sum / n.max(1) as f64
}

#[test]
fn ilc_reduces_tracking_error_pass_after_pass() {
    let c = constraints(SteeringType::Ackermann);
    let path = circle(2.0);
    let mut ilc = IlcFollower::new(Box::new(PurePursuitFollower::new()));
    let mut cfg = config();
    cfg.lookahead_distance = 1.5;
    ilc.set_config(cfg);
    let first = run_pass(&mut ilc, &path, &c);
    let second = run_pass(&mut ilc, &path, &c);
    let third = run_pass(&mut ilc, &path, &c);
    assert_eq!(ilc.passes(), 2);
    assert!(second < first * 0.7, "pass 2 mean error {second:.3} vs pass 1 {first:.3}");
    assert!(third <= second * 1.05, "pass 3 mean error {third:.3} vs pass 2 {second:.3}");
    assert!(ilc.corrections().iter().all(|c| c.abs() <= 0.5));
}

#[test]
fn ilc_resets_on_a_different_path() {
    let c = constraints(SteeringType::Differential);
    let mut ilc = IlcFollower::default();
    ilc.set_config(config());
    run_pass(&mut ilc, &circle(2.0), &c);
    run_pass(&mut ilc, &circle(2.0), &c);
    assert_eq!(ilc.passes(), 1);
    run_pass(&mut ilc, &circle(3.0), &c);
    assert_eq!(ilc.passes(), 0);
}

fn obstacle(x: f64, y: f64) -> WorldConstraints {
    WorldConstraints {
        obstacles: vec![Obstacle {
            id: 0,
            radius: 0.3,
            modes: vec![GaussianMode { weight: 1.0, mean_x: vec![x], mean_y: vec![y], std_x: vec![0.05], std_y: vec![0.05] }],
        }],
        ..Default::default()
    }
}

#[test]
fn apf_reaches_a_goal_around_an_offset_obstacle() {
    let c = constraints(SteeringType::Differential);
    let world = obstacle(4.0, 0.3);
    let mut apf = ApfFollower::new();
    apf.set_config(config());
    let goal = Goal { target_pose: pose(8.0, 0.0, 0.0), tolerance_position: 0.3, tolerance_orientation: PI, ..Default::default() };
    let mut state = RobotState { pose: pose(0.0, 0.0, 0.0), allow_move: true, ..Default::default() };
    let mut min_d = f64::MAX;
    let mut reached = false;
    for _ in 0..1200 {
        let cmd = apf.compute_control(&state, &goal, &c, 0.05, Some(&world));
        assert!(cmd.valid);
        integrate(&mut state, &cmd, 0.05);
        min_d = min_d.min((state.pose.point.x - 4.0).hypot(state.pose.point.y - 0.3));
        if apf.get_status().goal_reached {
            reached = true;
            break;
        }
    }
    assert!(reached, "APF ended at ({:.2},{:.2})", state.pose.point.x, state.pose.point.y);
    assert!(min_d > 0.3 + 0.36, "APF grazed the obstacle: {min_d:.2}");
}

#[test]
fn apf_reports_a_local_minimum_when_forces_cancel() {
    let c = constraints(SteeringType::Differential);
    let mut cfg = ondrive::ApfConfig::default();
    cfg.k_repulse = 5.0;
    let mut apf = ApfFollower::with_apf_config(cfg);
    apf.set_config(config());
    let goal = Goal { target_pose: pose(6.0, 0.0, 0.0), tolerance_position: 0.3, tolerance_orientation: PI, ..Default::default() };
    let world = obstacle(3.0, 0.0);
    let mut state = RobotState { pose: pose(0.0, 0.0, 0.0), allow_move: true, ..Default::default() };
    let mut saw_minimum = false;
    for _ in 0..600 {
        let cmd = apf.compute_control(&state, &goal, &c, 0.05, Some(&world));
        if apf.get_status().mode == "apf_local_minimum" {
            saw_minimum = true;
            assert!(cmd.valid && cmd.linear_velocity == 0.0);
            break;
        }
        integrate(&mut state, &cmd, 0.05);
    }
    assert!(saw_minimum, "APF never reported the head-on local minimum");
}
