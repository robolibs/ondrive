//! Progress along a path moves forward continuously: through reversing
//! cusps, past self-crossings and near-misses of the path end, and lap
//! after lap around a closed path.

#![allow(clippy::field_reassign_with_default)]

use datapod::{Euler, Point, Pose, Quaternion};
use ondrive::core::path::{cumulative_lengths, next_cusp, project};
use ondrive::{
    ControllerConfig, Path, RobotConstraints, RobotState, SteeringType, Tracker, TrackerKind,
    VelocityCommand, smoothen_path,
};

fn pose(x: f64, y: f64, yaw: f64) -> Pose {
    Pose {
        point: Point::new(x, y, 0.0),
        rotation: Quaternion::from_euler(Euler::new(0.0, 0.0, yaw)),
    }
}

/// Forward along x to 5 m, then back to 2 m.
fn cusp_path() -> Path {
    let mut path = Path::default();
    for i in 0..=50 {
        path.waypoints.push(pose(i as f64 * 0.1, 0.0, 0.0));
    }
    for i in 1..=30 {
        path.waypoints.push(pose(5.0 - i as f64 * 0.1, 0.0, 0.0));
    }
    path
}

fn polyline(points: &[(f64, f64)], closed: bool) -> Path {
    let mut path = Path {
        waypoints: points.iter().map(|(x, y)| pose(*x, *y, 0.0)).collect(),
        is_closed: closed,
        ..Default::default()
    };
    smoothen_path(&mut path, 0.1);
    path
}

fn tracker(kind: TrackerKind, reverse: bool, tolerance: f64) -> Tracker {
    let mut t = Tracker::new(kind);
    t.set_config(ControllerConfig {
        allow_reverse: reverse,
        goal_tolerance: tolerance,
        angular_tolerance: std::f64::consts::PI,
        lookahead_distance: 0.6,
        ..Default::default()
    });
    t.init(RobotConstraints {
        steering_type: SteeringType::Differential,
        max_angular_velocity: 2.0,
        ..Default::default()
    });
    t
}

fn step(state: &mut RobotState, cmd: &VelocityCommand, dt: f64) {
    let yaw = state.pose.rotation.to_euler().yaw;
    state.pose.point.x += cmd.linear_velocity * yaw.cos() * dt;
    state.pose.point.y += cmd.linear_velocity * yaw.sin() * dt;
    state.pose.rotation =
        Quaternion::from_euler(Euler::new(0.0, 0.0, yaw + cmd.angular_velocity * dt));
    state.velocity.linear = cmd.linear_velocity;
    state.velocity.angular = cmd.angular_velocity;
}

/// Drives until the path completes; returns every pose visited.
fn drive(t: &mut Tracker, start: Pose, ticks: usize) -> Vec<Point> {
    let mut state = RobotState {
        pose: start,
        allow_move: true,
        allow_reverse: true,
        ..Default::default()
    };
    let mut visited = vec![state.pose.point];
    for _ in 0..ticks {
        let cmd = t.tick(&state, 0.05, None);
        assert!(cmd.valid, "{:?}: {}", t.kind(), cmd.status_message);
        if t.is_path_completed() {
            break;
        }
        step(&mut state, &cmd, 0.05);
        visited.push(state.pose.point);
    }
    visited
}

fn passes(visited: &[Point], x: f64, y: f64, within: f64) -> bool {
    visited
        .iter()
        .any(|p| p.distance_to_2d(Point::new(x, y, 0.0)) < within)
}

#[test]
fn equally_near_legs_resolve_to_the_one_progress_is_on() {
    let path = cusp_path();
    let cum = cumulative_lengths(&path.waypoints);
    let p = Point::new(4.5, 0.0, 0.0);
    let first = project(&path.waypoints, &cum, p, 0, 2, usize::MAX).unwrap();
    assert!(
        (first.arc_length - 4.5).abs() < 1e-9,
        "first projection at s={}",
        first.arc_length
    );
    let on = project(
        &path.waypoints,
        &cum,
        Point::new(4.6, 0.0, 0.0),
        first.segment,
        2,
        64,
    )
    .unwrap();
    assert!(
        (on.arc_length - 4.6).abs() < 1e-9,
        "outgoing leg expected, got s={}",
        on.arc_length
    );
    let back = project(&path.waypoints, &cum, Point::new(4.6, 0.0, 0.0), 52, 2, 64).unwrap();
    assert!(
        (back.arc_length - 5.4).abs() < 1e-9,
        "return leg expected, got s={}",
        back.arc_length
    );
    assert_eq!(next_cusp(&path.waypoints, 0), Some(50));
}

#[test]
fn point_controllers_drive_to_the_cusp_before_turning_back() {
    for kind in [TrackerKind::Pid, TrackerKind::Carrot] {
        let mut t = tracker(kind, true, 0.1);
        t.set_path(cusp_path());
        let state = RobotState {
            pose: pose(4.5, 0.0, 0.0),
            allow_move: true,
            allow_reverse: true,
            ..Default::default()
        };
        t.tick(&state, 0.05, None);
        let target = t.current_target().unwrap();
        assert!(
            (target.x - 5.0).abs() < 1e-9,
            "{kind:?}: first target {target:?}"
        );

        let mut t = tracker(kind, true, 0.1);
        t.set_path(cusp_path());
        let visited = drive(&mut t, pose(0.0, 0.0, 0.0), 4000);
        assert!(t.is_path_completed(), "{kind:?}: not completed");
        assert!(
            passes(&visited, 5.0, 0.0, 0.15),
            "{kind:?}: never reached the cusp"
        );
        let end = visited.last().unwrap();
        assert!(
            end.distance_to_2d(Point::new(2.0, 0.0, 0.0)) < 0.3,
            "{kind:?}: ended at {end:?}"
        );
    }
}

#[test]
fn followers_drive_the_loop_of_a_self_crossing_path() {
    let path = polyline(
        &[(0.0, 0.0), (4.0, 0.0), (4.0, 1.5), (3.0, 1.5), (3.0, -2.0)],
        false,
    );
    for kind in [
        TrackerKind::PurePursuit,
        TrackerKind::Stanley,
        TrackerKind::Pid,
    ] {
        let mut t = tracker(kind, false, 0.2);
        t.set_path(path.clone());
        let visited = drive(&mut t, pose(0.0, 0.0, 0.0), 4000);
        assert!(t.is_path_completed(), "{kind:?}: not completed");
        assert!(
            passes(&visited, 4.0, 1.5, 0.4),
            "{kind:?}: skipped the loop"
        );
        let end = visited.last().unwrap();
        assert!(
            end.distance_to_2d(Point::new(3.0, -2.0, 0.0)) < 0.5,
            "{kind:?}: ended at {end:?}"
        );
    }
}

#[test]
fn followers_do_not_stop_where_the_path_passes_near_its_end() {
    let path = polyline(
        &[(0.0, 0.0), (0.0, 4.0), (2.0, 4.0), (2.0, 2.0), (0.3, 2.0)],
        false,
    );
    for kind in [
        TrackerKind::PurePursuit,
        TrackerKind::Stanley,
        TrackerKind::Lqr,
        TrackerKind::Mpc,
    ] {
        let mut t = tracker(kind, false, 0.4);
        t.set_path(path.clone());
        let visited = drive(&mut t, pose(0.0, 0.0, std::f64::consts::FRAC_PI_2), 4000);
        assert!(t.is_path_completed(), "{kind:?}: not completed");
        assert!(
            passes(&visited, 2.0, 4.0, 0.5),
            "{kind:?}: stopped before the far corner"
        );
    }
}

#[test]
fn a_closed_path_is_driven_lap_after_lap() {
    let square = polyline(&[(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)], true);
    for kind in [TrackerKind::PurePursuit, TrackerKind::Pid] {
        let mut t = tracker(kind, false, 0.2);
        t.set_path(square.clone());
        assert_eq!(t.path().unwrap().waypoints.len(), square.waypoints.len());
        let visited = drive(&mut t, pose(0.0, 0.0, 0.0), 3000);
        assert!(!t.is_path_completed(), "{kind:?}: a closed path completed");
        let travelled: f64 = visited.windows(2).map(|w| w[0].distance_to_2d(w[1])).sum();
        assert!(
            travelled > 2.5 * 16.0,
            "{kind:?}: only {travelled:.1} m in 150 s"
        );
        let corner_visits = visited
            .windows(2)
            .filter(|w| {
                w[0].distance_to_2d(Point::new(4.0, 4.0, 0.0)) >= 0.5
                    && w[1].distance_to_2d(Point::new(4.0, 4.0, 0.0)) < 0.5
            })
            .count();
        assert!(
            corner_visits >= 2,
            "{kind:?}: far corner visited {corner_visits} times"
        );
    }
}
