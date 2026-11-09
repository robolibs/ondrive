//! Non-finite and malformed input: rejected with an invalid command, never
//! a panic or a hang.

#![allow(clippy::field_reassign_with_default)]

use datapod::{Euler, Point, Pose, Quaternion};
use ondrive::core::path::{cumulative_lengths, sample};
use ondrive::ffi::*;
use ondrive::{
    Goal, Path, RobotConstraints, RobotState, SteeringType, Tracker, TrackerKind, Trajectory,
    normalize_angle,
};
use std::f64::consts::PI;
use std::ptr;

const KINDS: [TrackerKind; 21] = [
    TrackerKind::Pid,
    TrackerKind::Carrot,
    TrackerKind::PurePursuit,
    TrackerKind::Stanley,
    TrackerKind::Lqr,
    TrackerKind::Mpc,
    TrackerKind::Mppi,
    TrackerKind::Mca,
    TrackerKind::Soc,
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

fn pose(x: f64, y: f64, yaw: f64) -> Pose {
    Pose {
        point: Point::new(x, y, 0.0),
        rotation: Quaternion::from_euler(Euler::new(0.0, 0.0, yaw)),
    }
}

fn state(x: f64, y: f64) -> RobotState {
    RobotState {
        pose: pose(x, y, 0.0),
        allow_move: true,
        ..Default::default()
    }
}

fn line() -> Path {
    Path {
        waypoints: (0..=20).map(|i| pose(i as f64 * 0.25, 0.0, 0.0)).collect(),
        ..Default::default()
    }
}

fn tracker(kind: TrackerKind) -> Tracker {
    let mut t = Tracker::new(kind);
    t.init(RobotConstraints {
        steering_type: SteeringType::Differential,
        ..Default::default()
    });
    t
}

#[test]
fn normalize_angle_returns_for_every_input() {
    assert!(normalize_angle(f64::INFINITY).is_nan());
    assert!(normalize_angle(f64::NEG_INFINITY).is_nan());
    assert!(normalize_angle(f64::NAN).is_nan());
    assert_eq!(normalize_angle(PI), PI);
    assert_eq!(normalize_angle(-PI), -PI);
    assert!((normalize_angle(7.0) - (7.0 - 2.0 * PI)).abs() < 1e-12);
    assert!((normalize_angle(-7.0) - (2.0 * PI - 7.0)).abs() < 1e-12);
    let big = normalize_angle(1e12);
    assert!((-PI..=PI).contains(&big));
}

#[test]
fn every_tracker_rejects_a_nan_pose() {
    for kind in KINDS {
        let mut t = tracker(kind);
        t.set_path(line());
        let cmd = t.tick(&state(f64::NAN, 0.0), 0.05, None);
        assert!(!cmd.valid, "{kind:?}");
        assert_eq!(cmd.linear_velocity, 0.0, "{kind:?}");
    }
}

#[test]
fn every_tracker_rejects_a_nan_waypoint_and_goal() {
    for kind in KINDS {
        let mut t = tracker(kind);
        let mut path = line();
        path.waypoints[7].point.y = f64::NAN;
        t.set_path(path);
        assert!(!t.tick(&state(0.0, 0.0), 0.05, None).valid, "{kind:?} path");

        let mut t = tracker(kind);
        t.set_goal(Goal {
            target_pose: pose(f64::INFINITY, 0.0, 0.0),
            ..Default::default()
        });
        assert!(!t.tick(&state(0.0, 0.0), 0.05, None).valid, "{kind:?} goal");
    }
}

#[test]
fn bad_constraints_are_rejected_before_they_reach_a_clamp() {
    for kind in KINDS {
        let mut t = tracker(kind);
        let mut c = t.constraints().clone();
        c.max_angular_velocity = f64::NAN;
        t.init(c);
        t.set_path(line());
        assert!(!t.tick(&state(0.0, 0.0), 0.05, None).valid, "{kind:?} NaN");

        let mut t = tracker(kind);
        let mut c = t.constraints().clone();
        c.max_linear_velocity = -1.0;
        t.init(c);
        t.set_path(line());
        assert!(
            !t.tick(&state(0.0, 0.0), 0.05, None).valid,
            "{kind:?} negative"
        );
    }
}

#[test]
fn a_trajectory_with_missing_or_unsorted_times_is_rejected() {
    let path = line();
    for times in [vec![0.0, 1.0], (0..=20).rev().map(|i| i as f64).collect()] {
        let mut t = tracker(TrackerKind::Kanayama);
        t.set_trajectory(Trajectory {
            poses: path.waypoints.clone(),
            times: times.clone(),
            speeds: vec![1.0; path.waypoints.len()],
        });
        assert!(!t.tick(&state(0.0, 0.0), 0.05, None).valid);
        let short = Trajectory {
            poses: path.waypoints.clone(),
            times,
            speeds: Vec::new(),
        };
        let _ = short.sample(3.0);
    }
}

#[test]
fn sampling_a_path_at_nan_does_not_panic() {
    let path = line();
    let cum = cumulative_lengths(&path.waypoints);
    let _ = sample(&path.waypoints, &cum, f64::NAN);
    let mut bad = path.waypoints.clone();
    bad[3].point.x = f64::NAN;
    let cum = cumulative_lengths(&bad);
    let _ = sample(&bad, &cum, 2.0);
}

#[test]
fn ffi_grid_handles_empty_null_and_overflowing_sizes() {
    let w = ondrive_world_new();
    assert!(ondrive_world_set_grid(w, 0.0, 0.0, 0.1, 0, 0, ptr::null()));
    assert!(!ondrive_world_set_grid(w, 0.0, 0.0, 0.1, 4, 4, ptr::null()));
    let cell = [1u8];
    assert!(!ondrive_world_set_grid(
        w,
        0.0,
        0.0,
        0.1,
        usize::MAX,
        2,
        cell.as_ptr()
    ));
    ondrive_world_free(w);
}

#[test]
fn ffi_tick_reports_a_nan_pose_as_an_invalid_command() {
    let t = ondrive_tracker_new(ONDRIVE_KIND_PURE_PURSUIT);
    let path = ondrive_path_new();
    for i in 0..10 {
        ondrive_path_add_waypoint_xy(path, i as f64, 0.0, 0.0, 1.0);
    }
    assert!(ondrive_tracker_set_path(t, path));
    let mut s: OndriveRobotState = unsafe { std::mem::zeroed() };
    s.pose.rotation.w = 1.0;
    s.pose.position.x = f64::NAN;
    s.allow_move = true;
    let mut cmd: OndriveVelocityCommand = unsafe { std::mem::zeroed() };
    assert!(ondrive_tracker_tick(t, s, 0.05, ptr::null(), &mut cmd));
    assert!(!cmd.valid);
    ondrive_path_free(path);
    ondrive_tracker_free(t);
}
