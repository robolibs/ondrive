#![allow(clippy::field_reassign_with_default)]

use datapod::{Euler, Point, Pose, Quaternion};
use ondrive::{
    ControllerConfig, Goal, OutputUnits, Path, RobotConstraints, RobotState, Tracker, TrackerKind,
};

fn pose(x: f64, y: f64, yaw: f64) -> Pose {
    Pose {
        point: Point::new(x, y, 0.0),
        rotation: Quaternion::from_euler(Euler::new(0.0, 0.0, yaw)),
    }
}

fn straight_path(n: usize) -> Path {
    let mut p = Path::default();
    p.waypoints = (0..n).map(|i| pose(i as f64 * 0.5, 0.0, 0.0)).collect();
    p
}

fn sine_path(n: usize) -> Path {
    let mut p = Path::default();
    p.waypoints = (0..n)
        .map(|i| {
            let x = i as f64 * 0.5;
            pose(x, (x * 0.3).sin() * 0.8, 0.0)
        })
        .collect();
    p
}

fn simulate(
    tracker: &mut Tracker,
    start: Pose,
    max_ticks: usize,
    dt: f64,
) -> (bool, Pose, f64, f64) {
    let mut state = RobotState {
        pose: start,
        allow_move: true,
        ..Default::default()
    };
    let mut max_cte: f64 = 0.0;
    let mut t = 0.0;

    for _ in 0..max_ticks {
        let cmd = tracker.tick(&state, dt, None);
        if !cmd.valid {
            break;
        }
        let yaw = state.pose.rotation.to_euler().yaw;
        state.pose.point.x += cmd.linear_velocity * yaw.cos() * dt;
        state.pose.point.y += cmd.linear_velocity * yaw.sin() * dt;
        let new_yaw = yaw + cmd.angular_velocity * dt;
        state.pose.rotation = Quaternion::from_euler(Euler::new(0.0, 0.0, new_yaw));
        t += dt;
        max_cte = max_cte.max(tracker.get_status().cross_track_error.abs());
        if tracker.get_status().goal_reached {
            return (true, state.pose, t, max_cte);
        }
    }
    (false, state.pose, t, max_cte)
}

#[test]
fn pure_pursuit_follows_straight_line() {
    let path = straight_path(30);
    let final_wp = *path.waypoints.last().unwrap();

    let mut tracker = Tracker::new(TrackerKind::PurePursuit);
    let mut cfg = ControllerConfig::default();
    cfg.lookahead_distance = 1.0;
    cfg.goal_tolerance = 0.3;
    cfg.angular_tolerance = 1.0;
    cfg.output_units = OutputUnits::Physical;
    tracker.set_config(cfg);

    let mut c = RobotConstraints::default();
    c.max_linear_velocity = 1.0;
    c.max_angular_velocity = 2.0;
    c.wheelbase = 0.5;
    tracker.init(c);

    tracker.set_path(path);
    tracker.set_goal(Goal {
        target_pose: final_wp,
        tolerance_position: 0.3,
        tolerance_orientation: 1.0,
        ..Default::default()
    });

    let (reached, _, t, max_cte) = simulate(&mut tracker, pose(0.0, 0.0, 0.0), 3000, 0.05);
    assert!(
        reached,
        "pure pursuit did not reach final waypoint (t={t:.2})"
    );
    assert!(max_cte < 0.5, "cross-track error too large: {max_cte:.3}");
}

#[test]
fn stanley_follows_sine_path() {
    let path = sine_path(40);
    let final_wp = *path.waypoints.last().unwrap();

    let mut tracker = Tracker::new(TrackerKind::Stanley);
    let mut cfg = ControllerConfig::default();
    cfg.k_cross_track = 1.2;
    cfg.goal_tolerance = 0.3;
    cfg.angular_tolerance = 1.0;
    tracker.set_config(cfg);

    let mut c = RobotConstraints::default();
    c.max_linear_velocity = 0.5;
    c.max_angular_velocity = 2.0;
    c.wheelbase = 0.5;
    tracker.init(c);

    tracker.set_path(path);
    tracker.set_goal(Goal {
        target_pose: final_wp,
        tolerance_position: 0.3,
        tolerance_orientation: 1.0,
        ..Default::default()
    });

    let (reached, _, t, max_cte) = simulate(&mut tracker, pose(0.0, 0.2, 0.0), 5000, 0.05);
    assert!(reached, "stanley did not reach final waypoint (t={t:.2})");
    assert!(max_cte < 1.2, "cross-track error too large: {max_cte:.3}");
}

#[test]
fn stanley_stops_with_the_robot_at_the_goal() {
    let path = straight_path(41);
    let final_wp = *path.waypoints.last().unwrap();

    let mut tracker = Tracker::new(TrackerKind::Stanley);
    let mut cfg = ControllerConfig::default();
    cfg.goal_tolerance = 0.75;
    cfg.angular_tolerance = 1.0;
    cfg.output_units = OutputUnits::Physical;
    tracker.set_config(cfg);

    let mut c = RobotConstraints::default();
    c.max_linear_velocity = 2.0;
    c.max_angular_velocity = 1.0;
    c.wheelbase = 2.0;
    tracker.init(c);

    tracker.set_path(path);
    tracker.set_goal(Goal {
        target_pose: final_wp,
        tolerance_position: 0.75,
        tolerance_orientation: 1.0,
        ..Default::default()
    });

    let (reached, end, t, _) = simulate(&mut tracker, pose(0.0, 0.0, 0.0), 4000, 0.05);
    assert!(reached, "stanley did not reach the goal (t={t:.2})");
    let gap = end.point.distance_to_2d(final_wp.point);
    assert!(gap < 0.75, "stanley stopped {gap:.2} m from the goal");
}

#[test]
fn every_path_follower_reverses_out_of_a_cusp() {
    let mut legs: Vec<(f64, f64, f64)> = (0..17).map(|i| (1.5 * i as f64, 0.0, 1.0)).collect();
    for i in 1..=16 {
        let a = i as f64 * std::f64::consts::FRAC_PI_2 / 16.0;
        legs.push((24.0 - 9.0 * a.sin(), 9.0 * (1.0 - a.cos()), -1.0));
    }
    for kind in [
        TrackerKind::PurePursuit,
        TrackerKind::Stanley,
        TrackerKind::Lqr,
        TrackerKind::RegulatedPursuit,
        TrackerKind::VectorPursuit,
        TrackerKind::Kanayama,
    ] {
        let mut path = Path::default();
        for i in 0..legs.len() {
            let (nx, ny, _) = legs[(i + 1).min(legs.len() - 1)];
            let (px, py, _) = legs[i.saturating_sub(1)];
            let back = if legs[i].2 < 0.0 { std::f64::consts::PI } else { 0.0 };
            path.waypoints.push(pose(legs[i].0, legs[i].1, (ny - py).atan2(nx - px) + back));
            path.speeds.push(legs[i].2);
        }
        let end = *path.waypoints.last().unwrap();
        let mut tracker = Tracker::new(kind);
        let mut cfg = ControllerConfig::default();
        cfg.goal_tolerance = 0.75;
        cfg.output_units = OutputUnits::Physical;
        cfg.allow_reverse = true;
        tracker.set_config(cfg);
        let mut c = RobotConstraints::default();
        c.wheelbase = 2.0;
        c.max_steering_angle = 0.45;
        c.max_linear_velocity = 2.0;
        c.max_angular_velocity = 1.0;
        tracker.init(c);
        tracker.set_path(path);

        let mut state = RobotState { pose: pose(0.0, 0.0, 0.0), allow_move: true, velocity_valid: true, ..Default::default() };
        let (dt, mut worst) = (0.05, 0.0_f64);
        for _ in 0..2400 {
            let cmd = tracker.tick(&state, dt, None);
            let yaw = state.pose.rotation.to_euler().yaw;
            state.pose.point.x += cmd.linear_velocity * yaw.cos() * dt;
            state.pose.point.y += cmd.linear_velocity * yaw.sin() * dt;
            state.pose.rotation = Quaternion::from_euler(Euler::new(0.0, 0.0, yaw + cmd.angular_velocity * dt));
            state.velocity.linear = cmd.linear_velocity;
            state.velocity.angular = cmd.angular_velocity;
            worst = worst.max(tracker.get_status().cross_track_error.abs());
            if tracker.is_path_completed() || !cmd.valid {
                break;
            }
        }
        assert!(tracker.is_path_completed(), "{kind:?} never finished the reversing leg");
        assert!(worst < 0.5, "{kind:?} strayed {worst:.2} m off the path");
        let gap = state.pose.point.distance_to_2d(end.point);
        assert!(gap < 1.0, "{kind:?} stopped {gap:.2} m from the end");
    }
}

#[test]
fn followers_keep_their_speed_round_the_end_of_a_lap() {
    let radius = 7.5;
    for kind in [TrackerKind::PurePursuit, TrackerKind::Stanley, TrackerKind::Lqr, TrackerKind::VectorPursuit, TrackerKind::Kanayama] {
        let mut path = Path::default();
        path.is_closed = true;
        for i in 0..48 {
            let a = i as f64 * std::f64::consts::TAU / 48.0;
            path.waypoints.push(pose(radius * a.sin(), radius * (1.0 - a.cos()), a));
        }
        let mut tracker = Tracker::new(kind);
        let mut cfg = ControllerConfig::default();
        cfg.goal_tolerance = 0.3;
        cfg.output_units = OutputUnits::Physical;
        tracker.set_config(cfg);
        let mut c = RobotConstraints::default();
        c.wheelbase = 0.65;
        c.max_steering_angle = 0.46;
        c.max_linear_velocity = 1.0;
        c.max_angular_velocity = 1.5;
        tracker.init(c);
        tracker.set_path(path);

        let mut state = RobotState { pose: pose(0.0, 0.0, 0.0), allow_move: true, velocity_valid: true, ..Default::default() };
        let (dt, lap) = (0.05, std::f64::consts::TAU * radius);
        let (mut driven, mut slowest) = (0.0, f64::INFINITY);
        while driven < 1.5 * lap {
            let cmd = tracker.tick(&state, dt, None);
            assert!(cmd.valid && !tracker.is_path_completed(), "{kind:?} stopped after {driven:.1} m");
            let yaw = state.pose.rotation.to_euler().yaw;
            state.pose.point.x += cmd.linear_velocity * yaw.cos() * dt;
            state.pose.point.y += cmd.linear_velocity * yaw.sin() * dt;
            state.pose.rotation = Quaternion::from_euler(Euler::new(0.0, 0.0, yaw + cmd.angular_velocity * dt));
            state.velocity.linear = cmd.linear_velocity;
            state.velocity.angular = cmd.angular_velocity;
            driven += cmd.linear_velocity.abs() * dt;
            if driven > 0.5 * lap {
                slowest = slowest.min(cmd.linear_velocity);
            }
        }
        assert!(slowest > 0.5, "{kind:?} slowed to {slowest:.2} m/s going round the end of the lap");
    }
}
