//! Pose-to-pose reaching with Reeds-Shepp / Dubins curves.

#![allow(clippy::field_reassign_with_default)]

use datapod::{Euler, Point, Pose, Quaternion};
use ondrive::{
    Controller, ControllerConfig, Goal, OutputUnits, PoseReachFollower, RobotConstraints,
    RobotState, SteeringType,
};
use std::f64::consts::PI;

fn pose(x: f64, y: f64, yaw: f64) -> Pose {
    Pose {
        point: Point::new(x, y, 0.0),
        rotation: Quaternion::from_euler(Euler::new(0.0, 0.0, yaw)),
    }
}

fn car() -> RobotConstraints {
    let mut c = RobotConstraints::default();
    c.steering_type = SteeringType::Ackermann;
    c.max_linear_velocity = 0.6;
    c.min_linear_velocity = -0.4;
    c.max_angular_velocity = 2.0;
    c.max_linear_acceleration = 1.5;
    c.max_steering_angle = 0.6;
    c.min_turning_radius = 0.0;
    c.wheelbase = 0.5;
    c
}

fn config(reverse: bool) -> ControllerConfig {
    let mut cfg = ControllerConfig::default();
    cfg.output_units = OutputUnits::Physical;
    cfg.goal_tolerance = 0.15;
    cfg.angular_tolerance = 0.15;
    cfg.allow_reverse = reverse;
    cfg.kp_linear = 1.5;
    cfg.kp_angular = 2.0;
    cfg
}

struct Run {
    reached: bool,
    pose: Pose,
    reversed: bool,
    t: f64,
}

fn drive(goal: Pose, reverse: bool, start: Pose) -> Run {
    let mut ctrl = PoseReachFollower::new();
    ctrl.set_config(config(reverse));
    let c = car();
    let g = Goal {
        target_pose: goal,
        tolerance_position: 0.15,
        tolerance_orientation: 0.15,
        ..Default::default()
    };
    let mut state = RobotState { pose: start, allow_move: true, ..Default::default() };
    let dt = 0.05;
    let mut run = Run { reached: false, pose: start, reversed: false, t: 0.0 };
    for i in 0..3000 {
        let cmd = ctrl.compute_control(&state, &g, &c, dt, None);
        assert!(cmd.valid, "tick {i}: {}", cmd.status_message);
        if cmd.linear_velocity < -1e-6 {
            run.reversed = true;
        }
        assert!(reverse || cmd.linear_velocity >= -1e-9, "reversed without permission");
        let yaw = state.pose.rotation.to_euler().yaw;
        state.pose.point.x += cmd.linear_velocity * yaw.cos() * dt;
        state.pose.point.y += cmd.linear_velocity * yaw.sin() * dt;
        state.pose.rotation = Quaternion::from_euler(Euler::new(0.0, 0.0, yaw + cmd.angular_velocity * dt));
        state.velocity.linear = cmd.linear_velocity;
        run.t = (i + 1) as f64 * dt;
        if ctrl.get_status().goal_reached {
            run.reached = true;
            break;
        }
    }
    run.pose = state.pose;
    run
}

fn check(run: &Run, goal: &Pose, label: &str) {
    let d = run.pose.point.distance_to_2d(goal.point);
    let a = (run.pose.rotation.to_euler().yaw - goal.rotation.to_euler().yaw + PI).rem_euclid(2.0 * PI) - PI;
    assert!(run.reached, "{label}: not reached; ended {d:.2} m / {:.2} rad off after {:.1} s", a.abs(), run.t);
    assert!(d < 0.2 && a.abs() < 0.2, "{label}: pose error d={d:.3} a={a:.3}");
}

#[test]
fn parallel_park_beside_the_start() {
    let goal = pose(1.0, -1.2, 0.0);
    let run = drive(goal, true, pose(0.0, 0.0, 0.0));
    check(&run, &goal, "parallel park");
    assert!(run.reversed, "parallel parking needs a reverse segment");
}

#[test]
fn goal_90_degrees_off_at_one_metre() {
    let goal = pose(1.0, 1.0, PI / 2.0);
    for reverse in [false, true] {
        let run = drive(goal, reverse, pose(0.0, 0.0, 0.0));
        check(&run, &goal, &format!("90deg reverse={reverse}"));
    }
}

#[test]
fn goal_behind_with_and_without_reverse() {
    let goal = pose(-3.0, 0.0, 0.0);
    let with = drive(goal, true, pose(0.0, 0.0, 0.0));
    check(&with, &goal, "behind, reverse allowed");
    assert!(with.reversed);
    let without = drive(goal, false, pose(0.0, 0.0, 0.0));
    check(&without, &goal, "behind, forward only");
    assert!(!without.reversed);
    assert!(with.t < without.t, "reverse path should be quicker: {:.1} vs {:.1}", with.t, without.t);
}

#[test]
fn replans_when_the_goal_moves() {
    let mut ctrl = PoseReachFollower::new();
    ctrl.set_config(config(true));
    let c = car();
    let mut state = RobotState { pose: pose(0.0, 0.0, 0.0), allow_move: true, ..Default::default() };
    let first = Goal { target_pose: pose(4.0, 0.0, 0.0), tolerance_position: 0.15, tolerance_orientation: 0.15, ..Default::default() };
    for _ in 0..20 {
        let cmd = ctrl.compute_control(&state, &first, &c, 0.05, None);
        let yaw = state.pose.rotation.to_euler().yaw;
        state.pose.point.x += cmd.linear_velocity * yaw.cos() * 0.05;
        state.pose.point.y += cmd.linear_velocity * yaw.sin() * 0.05;
        state.pose.rotation = Quaternion::from_euler(Euler::new(0.0, 0.0, yaw + cmd.angular_velocity * 0.05));
    }
    let second = Goal { target_pose: pose(2.0, 2.0, PI / 2.0), tolerance_position: 0.15, tolerance_orientation: 0.15, ..Default::default() };
    let mut reached = false;
    for _ in 0..3000 {
        let cmd = ctrl.compute_control(&state, &second, &c, 0.05, None);
        assert!(cmd.valid);
        let yaw = state.pose.rotation.to_euler().yaw;
        state.pose.point.x += cmd.linear_velocity * yaw.cos() * 0.05;
        state.pose.point.y += cmd.linear_velocity * yaw.sin() * 0.05;
        state.pose.rotation = Quaternion::from_euler(Euler::new(0.0, 0.0, yaw + cmd.angular_velocity * 0.05));
        state.velocity.linear = cmd.linear_velocity;
        if ctrl.get_status().goal_reached {
            reached = true;
            break;
        }
    }
    assert!(reached);
    assert!(state.pose.point.distance_to_2d(Point::new(2.0, 2.0, 0.0)) < 0.2);
}
