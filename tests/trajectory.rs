//! Timed trajectories: sampling, tracker clock, and time-referenced tracking.

#![allow(clippy::field_reassign_with_default)]

use datapod::{Euler, Point, Pose, Quaternion};
use ondrive::{
    ControllerConfig, OutputUnits, Path, RobotConstraints, RobotState, SteeringType, Tracker,
    TrackerKind, Trajectory,
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
    c
}

fn config() -> ControllerConfig {
    let mut cfg = ControllerConfig::default();
    cfg.output_units = OutputUnits::Physical;
    cfg.goal_tolerance = 0.3;
    cfg.angular_tolerance = PI;
    cfg
}

/// Circle of radius `r` driven at `speed`, sampled every `dt`.
fn circle_trajectory(r: f64, speed: f64, duration: f64) -> Trajectory {
    let omega = speed / r;
    let dt = 0.1;
    let n = (duration / dt) as usize;
    let mut t = Trajectory::default();
    for i in 0..=n {
        let time = i as f64 * dt;
        let a = omega * time;
        t.poses.push(pose(r * a.sin(), r * (1.0 - a.cos()), a));
        t.times.push(time);
        t.speeds.push(speed);
    }
    t
}

#[test]
fn trajectory_sampling_interpolates_pose_speed_and_yaw_rate() {
    let traj = circle_trajectory(2.0, 0.5, 4.0);
    let s = traj.sample(1.05);
    let a: f64 = 0.25 * 1.05;
    assert!((s.pose.point.x - 2.0 * a.sin()).abs() < 0.02);
    assert!((s.pose.point.y - 2.0 * (1.0 - a.cos())).abs() < 0.02);
    assert!((s.pose.rotation.to_euler().yaw - a).abs() < 1e-6);
    assert!((s.speed - 0.5).abs() < 1e-9);
    assert!((s.yaw_rate - 0.25).abs() < 1e-6);
    assert!(!s.finished);
    let end = traj.sample(10.0);
    assert!(end.finished && end.speed == 0.0);
    let path = Path { waypoints: vec![pose(0.0, 0.0, 0.0), pose(2.0, 0.0, 0.0)], ..Default::default() };
    let from_path = Trajectory::from_path(&path, 0.5);
    assert!((from_path.duration() - 4.0).abs() < 1e-9);
}

#[test]
fn tracker_clock_follows_dt_and_timestamps() {
    let traj = circle_trajectory(3.0, 0.5, 10.0);
    let mut t = Tracker::new(TrackerKind::Kanayama);
    t.set_config(config());
    t.init(constraints(SteeringType::Differential));
    t.set_trajectory(traj.clone());
    let state = RobotState { pose: pose(0.0, 0.0, 0.0), allow_move: true, ..Default::default() };
    for _ in 0..10 {
        t.tick(&state, 0.1, None);
    }
    assert!((t.trajectory_time() - 1.0).abs() < 1e-9);

    let mut t2 = Tracker::new(TrackerKind::Kanayama);
    t2.set_config(config());
    t2.init(constraints(SteeringType::Differential));
    t2.set_trajectory(traj);
    for i in 0..10 {
        let s = RobotState { pose: pose(0.0, 0.0, 0.0), timestamp: 100.0 + i as f64 * 0.2, allow_move: true, ..Default::default() };
        t2.tick(&s, 0.1, None);
    }
    assert!((t2.trajectory_time() - 1.8).abs() < 1e-9, "clock {}", t2.trajectory_time());
}

fn track_timed(kind: TrackerKind, steering: SteeringType) -> (f64, f64, bool) {
    let traj = circle_trajectory(3.0, 0.6, 25.0);
    let mut t = Tracker::new(kind);
    t.set_config(config());
    t.init(constraints(steering));
    t.set_trajectory(traj.clone());
    let mut state = RobotState { pose: pose(0.0, 0.1, 0.0), allow_move: true, ..Default::default() };
    let dt = 0.1;
    let mut max_err: f64 = 0.0;
    let mut reached = false;
    let mut time = 0.0;
    for i in 0..400 {
        let cmd = t.tick(&state, dt, None);
        assert!(cmd.valid, "{kind:?}: {}", cmd.status_message);
        let yaw = state.pose.rotation.to_euler().yaw;
        state.pose.point.x += cmd.linear_velocity * yaw.cos() * dt;
        state.pose.point.y += cmd.linear_velocity * yaw.sin() * dt;
        state.pose.rotation = Quaternion::from_euler(Euler::new(0.0, 0.0, yaw + cmd.angular_velocity * dt));
        state.velocity.linear = cmd.linear_velocity;
        state.velocity.angular = cmd.angular_velocity;
        time = (i + 1) as f64 * dt;
        if time > 4.0 && time < 24.0 {
            let r = traj.sample(t.trajectory_time());
            max_err = max_err.max(state.pose.point.distance_to_2d(r.pose.point));
        }
        if t.is_goal_reached() {
            reached = true;
            break;
        }
    }
    (max_err, time, reached)
}

#[test]
fn timed_trackers_keep_schedule_on_a_circle() {
    for kind in [TrackerKind::Kanayama, TrackerKind::Mpc, TrackerKind::Ilqr, TrackerKind::Mppi] {
        for steering in [SteeringType::Differential, SteeringType::Ackermann] {
            let (max_err, time, reached) = track_timed(kind, steering);
            assert!(max_err < 0.5, "{kind:?} {steering:?}: fell {max_err:.2} m off schedule");
            assert!(reached, "{kind:?} {steering:?}: did not finish (t={time:.1})");
            assert!(time > 22.0 && time < 30.0, "{kind:?} {steering:?}: finished at t={time:.1}, trajectory lasts 25 s");
        }
    }
}
