#![allow(clippy::field_reassign_with_default)]

use approx::assert_relative_eq;
use datapod::{Euler, Point, Pose, Quaternion};
use ondrive::{
    ControllerConfig, Goal, Path, RobotConstraints, RobotState, Tracker, TrackerKind, smoothen_path,
};

fn pose_at(x: f64, y: f64, yaw: f64) -> Pose {
    Pose {
        point: Point::new(x, y, 0.0),
        rotation: Quaternion::from_euler(Euler::new(0.0, 0.0, yaw)),
    }
}

#[test]
fn emergency_stop_returns_zero_and_clears_state() {
    let mut tracker = Tracker::new(TrackerKind::Pid);
    tracker.set_goal(Goal {
        target_pose: pose_at(5.0, 0.0, 0.0),
        ..Default::default()
    });
    let mut path = Path::default();
    path.waypoints.push(pose_at(0.0, 0.0, 0.0));
    path.waypoints.push(pose_at(5.0, 0.0, 0.0));
    tracker.set_path(path);

    assert!(tracker.goal().is_some());
    assert!(tracker.path().is_some());

    let cmd = tracker.emergency_stop();
    assert!(cmd.valid);
    assert_eq!(cmd.linear_velocity, 0.0);
    assert_eq!(cmd.angular_velocity, 0.0);
    assert!(tracker.goal().is_none());
    assert!(tracker.path().is_none());
}

#[test]
fn current_target_prefers_goal_then_path() {
    let mut tracker = Tracker::new(TrackerKind::PurePursuit);
    assert!(tracker.current_target().is_none());

    let mut path = Path::default();
    path.waypoints.push(pose_at(0.0, 0.0, 0.0));
    path.waypoints.push(pose_at(1.0, 0.0, 0.0));
    path.waypoints.push(pose_at(2.0, 0.0, 0.0));
    tracker.set_path(path);

    let p = tracker
        .current_target()
        .expect("path waypoint should be target");
    assert_relative_eq!(p.x, 0.0, epsilon = 1e-9);

    tracker.set_goal(Goal {
        target_pose: pose_at(42.0, 7.0, 0.0),
        ..Default::default()
    });
    let p = tracker.current_target().expect("goal should override path");
    assert_relative_eq!(p.x, 42.0, epsilon = 1e-9);
    assert_relative_eq!(p.y, 7.0, epsilon = 1e-9);
}

#[test]
fn smoothen_densifies_path() {
    let mut tracker = Tracker::new(TrackerKind::Pid);
    let mut path = Path::default();
    path.waypoints.push(pose_at(0.0, 0.0, 0.0));
    path.waypoints.push(pose_at(10.0, 0.0, 0.0));
    tracker.set_path(path);

    assert_eq!(tracker.path().unwrap().waypoints.len(), 2);
    tracker.smoothen(1.0);
    // 10 m segment at 1 m resolution → 10 sub-segments → 11 waypoints.
    assert_eq!(tracker.path().unwrap().waypoints.len(), 11);

    let wps = &tracker.path().unwrap().waypoints;
    for (i, wp) in wps.iter().enumerate() {
        assert_relative_eq!(wp.point.x, i as f64, epsilon = 1e-9);
        assert_relative_eq!(wp.point.y, 0.0, epsilon = 1e-9);
    }
}

#[test]
fn free_smoothen_matches_tracker() {
    let mut path = Path::default();
    path.waypoints.push(pose_at(0.0, 0.0, 0.0));
    path.waypoints.push(pose_at(4.0, 3.0, 0.0));
    smoothen_path(&mut path, 1.0);
    // Segment length 5 m → ceil(5) = 5 sub-segments → 6 waypoints.
    assert_eq!(path.waypoints.len(), 6);
    let last = path.waypoints.last().unwrap();
    assert_relative_eq!(last.point.x, 4.0, epsilon = 1e-9);
    assert_relative_eq!(last.point.y, 3.0, epsilon = 1e-9);
}

#[test]
fn is_goal_reached_reflects_controller_status() {
    let mut tracker = Tracker::new(TrackerKind::Pid);
    tracker.set_config(ControllerConfig {
        kp_linear: 2.0,
        kp_angular: 1.5,
        goal_tolerance: 0.1,
        angular_tolerance: 0.5,
        ..ControllerConfig::default()
    });
    let mut c = RobotConstraints::default();
    c.max_linear_velocity = 0.5;
    c.max_angular_velocity = 2.0;
    tracker.init(c);
    tracker.set_goal(Goal {
        target_pose: pose_at(2.0, 0.0, 0.0),
        ..Default::default()
    });

    let mut state = RobotState {
        pose: pose_at(0.0, 0.0, 0.0),
        allow_move: true,
        ..Default::default()
    };

    assert!(!tracker.is_goal_reached());

    let dt = 0.05;
    for _ in 0..1000 {
        let cmd = tracker.tick(&state, dt, None);
        if !cmd.valid {
            break;
        }
        let yaw = state.pose.rotation.to_euler().yaw;
        state.pose.point.x += cmd.linear_velocity * yaw.cos() * dt;
        state.pose.point.y += cmd.linear_velocity * yaw.sin() * dt;
        let new_yaw = yaw + cmd.angular_velocity * dt;
        state.pose.rotation = Quaternion::from_euler(Euler::new(0.0, 0.0, new_yaw));
        if tracker.is_goal_reached() {
            break;
        }
    }
    assert!(tracker.is_goal_reached());
}

#[test]
fn reset_keeps_the_path_and_restarts_progress_on_it() {
    let mut path = Path::default();
    for i in 0..=50 {
        path.waypoints.push(pose_at(i as f64 * 0.1, 0.0, 0.0));
    }
    for i in 1..=50 {
        path.waypoints
            .push(pose_at(5.0, i as f64 * 0.1, std::f64::consts::FRAC_PI_2));
    }
    let state = RobotState {
        pose: pose_at(0.5, 0.0, 0.0),
        allow_move: true,
        ..Default::default()
    };
    let kinds = [
        TrackerKind::Pid,
        TrackerKind::Carrot,
        TrackerKind::PurePursuit,
        TrackerKind::Stanley,
        TrackerKind::Lqr,
        TrackerKind::Mpc,
        TrackerKind::Dwa,
        TrackerKind::Teb,
        TrackerKind::Flc,
        TrackerKind::RegulatedPursuit,
        TrackerKind::PoseReach,
        TrackerKind::Ilqr,
        TrackerKind::PoseRegulator,
        TrackerKind::VectorPursuit,
        TrackerKind::Kanayama,
        TrackerKind::Apf,
        TrackerKind::Ilc,
        TrackerKind::SlipAdaptive,
    ];
    for kind in kinds {
        let mut t = Tracker::new(kind);
        t.init(RobotConstraints {
            steering_type: ondrive::SteeringType::Differential,
            ..Default::default()
        });
        t.set_path(path.clone());
        let before = t.tick(&state, 0.05, None);
        t.tick(&state, 0.05, None);
        t.reset();
        assert_eq!(
            t.path().map(|p| p.waypoints.len()),
            Some(path.waypoints.len()),
            "{kind:?}"
        );
        let after = t.tick(&state, 0.05, None);
        assert!(after.valid, "{kind:?}: {}", after.status_message);
        assert_eq!(before.status_message, after.status_message, "{kind:?}");
        assert_relative_eq!(
            before.linear_velocity,
            after.linear_velocity,
            epsilon = 1e-9
        );
        assert_relative_eq!(
            before.angular_velocity,
            after.angular_velocity,
            epsilon = 1e-9
        );
    }
}
