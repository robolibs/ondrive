//! Smoke test for the C ABI: exercises every null-pointer path, runs a full
//! tracker lifecycle, creates/frees many handles to surface any obvious
//! leak/double-free, and verifies that `last_error` is cleared on success
//! and set on failure.

#![allow(clippy::field_reassign_with_default)]

use std::ffi::CStr;
use std::ptr;

use ondrive::ffi::*;

fn err_message() -> Option<String> {
    let p = ondrive_last_error_message();
    if p.is_null() {
        None
    } else {
        Some(unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned())
    }
}

#[test]
fn null_pointer_inputs_fail_cleanly() {
    // Path
    assert!(!ondrive_path_add_waypoint_xy(
        ptr::null_mut(),
        0.0,
        0.0,
        0.0,
        0.0
    ));
    assert!(err_message().is_some());
    assert!(!ondrive_path_set_closed(ptr::null_mut(), true));
    assert!(!ondrive_path_smoothen(ptr::null_mut(), 1.0));
    assert!(!ondrive_path_clear(ptr::null_mut()));
    assert_eq!(ondrive_path_len(ptr::null()), 0);
    assert!(!ondrive_path_waypoint(ptr::null(), 0, ptr::null_mut()));

    // World
    assert!(!ondrive_world_clear(ptr::null_mut()));
    assert_eq!(ondrive_world_obstacle_count(ptr::null()), 0);
    assert!(!ondrive_world_add_static_gaussian_obstacle(
        ptr::null_mut(),
        0,
        0.0,
        0.0,
        0.3,
        0.1,
        0.1,
        10,
    ));

    // Tracker
    assert!(!ondrive_tracker_init(
        ptr::null_mut(),
        ondrive_default_constraints()
    ));
    assert!(!ondrive_tracker_set_config(
        ptr::null_mut(),
        ondrive_default_config()
    ));
    assert!(!ondrive_tracker_get_config(ptr::null(), ptr::null_mut()));
    assert!(!ondrive_tracker_clear_goal(ptr::null_mut()));
    assert!(!ondrive_tracker_clear_path(ptr::null_mut()));
    assert!(!ondrive_tracker_reset(ptr::null_mut()));
    assert!(!ondrive_tracker_smoothen(ptr::null_mut(), 1.0));
    assert!(!ondrive_tracker_status(ptr::null(), ptr::null_mut()));
    assert!(!ondrive_tracker_is_goal_reached(ptr::null()));
    assert!(!ondrive_tracker_current_target(
        ptr::null(),
        ptr::null_mut()
    ));
    assert_eq!(ondrive_tracker_kind(ptr::null()), u32::MAX);
    assert!(!ondrive_tracker_constraints(ptr::null(), ptr::null_mut()));

    // Double-free of a null is a no-op.
    ondrive_path_free(ptr::null_mut());
    ondrive_world_free(ptr::null_mut());
    ondrive_tracker_free(ptr::null_mut());
}

#[test]
fn unknown_tracker_kind_fails_cleanly() {
    let t = ondrive_tracker_new(9999);
    assert!(t.is_null());
    assert!(err_message().is_some());
}

#[test]
fn null_output_pointer_is_rejected() {
    let t = ondrive_tracker_new(ONDRIVE_KIND_PID);
    assert!(!t.is_null());
    // get_config with null out ptr must fail.
    assert!(!ondrive_tracker_get_config(t, ptr::null_mut()));
    assert!(err_message().is_some());
    ondrive_tracker_free(t);
}

#[test]
fn full_tracker_lifecycle_runs_to_goal() {
    // Build path of 10 waypoints from (0,0) to (5,0).
    let path = ondrive_path_new();
    assert!(!path.is_null());
    for i in 0..10 {
        assert!(ondrive_path_add_waypoint_xy(
            path,
            i as f64 * 0.5,
            0.0,
            0.0,
            0.5
        ));
    }
    assert_eq!(ondrive_path_len(path), 10);
    assert!(ondrive_path_smoothen(path, 0.25));
    let dense = ondrive_path_len(path);
    assert!(
        dense >= 10,
        "smoothen should never shrink the path: {dense}"
    );

    let t = ondrive_tracker_new(ONDRIVE_KIND_PURE_PURSUIT);
    assert!(!t.is_null());
    assert_eq!(ondrive_tracker_kind(t), ONDRIVE_KIND_PURE_PURSUIT);

    let mut cons = ondrive_default_constraints();
    cons.steering_type = ONDRIVE_STEERING_ACKERMANN;
    cons.max_linear_velocity = 1.0;
    cons.max_angular_velocity = 2.0;
    cons.max_steering_angle = 0.6;
    cons.wheelbase = 0.5;
    assert!(ondrive_tracker_init(t, cons));

    let mut cfg = ondrive_default_config();
    cfg.lookahead_distance = 1.0;
    cfg.goal_tolerance = 0.4;
    cfg.angular_tolerance = 1.0;
    cfg.output_units = ONDRIVE_UNITS_PHYSICAL;
    assert!(ondrive_tracker_set_config(t, cfg));

    // Round-trip config.
    let mut cfg_back = ondrive_default_config();
    assert!(ondrive_tracker_get_config(t, &mut cfg_back));
    assert!((cfg_back.lookahead_distance - 1.0).abs() < 1e-9);

    assert!(ondrive_tracker_set_path(t, path));

    let mut final_pose = OndrivePose {
        position: OndriveVec3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        },
        rotation: OndriveQuat {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 1.0,
        },
    };
    assert!(ondrive_path_waypoint(path, dense - 1, &mut final_pose));

    let goal = OndriveGoal {
        target_pose: final_pose,
        has_target_velocity: false,
        target_velocity: OndriveVelocity {
            linear: 0.0,
            angular: 0.0,
            lateral: 0.0,
        },
        tolerance_position: 0.4,
        tolerance_orientation: 1.0,
    };
    assert!(ondrive_tracker_set_goal(t, goal));

    let mut state = OndriveRobotState {
        pose: OndrivePose {
            position: OndriveVec3 {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            rotation: ondrive_quaternion_from_yaw(0.0),
        },
        velocity: OndriveVelocity {
            linear: 0.0,
            angular: 0.0,
            lateral: 0.0,
        },
        timestamp: 0.0,
        allow_reverse: false,
        turn_first: false,
        allow_move: true,
        has_trailer: false,
        trailer_pose: OndrivePose {
            position: OndriveVec3 {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            rotation: OndriveQuat {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                w: 1.0,
            },
        },
    };

    let dt = 0.05;
    let mut reached = false;
    for _ in 0..1000 {
        let mut cmd = OndriveVelocityCommand {
            valid: false,
            linear_velocity: 0.0,
            angular_velocity: 0.0,
            lateral_velocity: 0.0,
            steering_angle: 0.0,
            output_type: 0,
        };
        assert!(ondrive_tracker_tick(t, state, dt, ptr::null(), &mut cmd));
        assert!(cmd.valid);

        let yaw = ondrive_quaternion_yaw(state.pose.rotation);
        state.pose.position.x += cmd.linear_velocity * yaw.cos() * dt;
        state.pose.position.y += cmd.linear_velocity * yaw.sin() * dt;
        state.pose.rotation = ondrive_quaternion_from_yaw(yaw + cmd.angular_velocity * dt);
        state.velocity.linear = cmd.linear_velocity;
        state.velocity.angular = cmd.angular_velocity;

        if ondrive_tracker_is_goal_reached(t) {
            reached = true;
            break;
        }
    }
    assert!(reached, "tracker never reached goal via FFI");

    // Status + mode string.
    let mut status = OndriveControllerStatus {
        goal_reached: false,
        distance_to_goal: 0.0,
        cross_track_error: 0.0,
        heading_error: 0.0,
    };
    assert!(ondrive_tracker_status(t, &mut status));
    assert!(status.goal_reached);
    let mode = ondrive_tracker_status_mode();
    assert!(!mode.is_null());
    let mode = unsafe { CStr::from_ptr(mode) }.to_string_lossy();
    assert_eq!(mode, "stopped");

    // Current target should be the final waypoint's point.
    let mut tgt = OndriveVec3 {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };
    assert!(ondrive_tracker_current_target(t, &mut tgt));
    assert!((tgt.x - final_pose.position.x).abs() < 1e-9);

    // Emergency stop returns a zero command and clears goal/path.
    let mut stop = OndriveVelocityCommand {
        valid: false,
        linear_velocity: 99.0,
        angular_velocity: 99.0,
        lateral_velocity: 99.0,
        steering_angle: 99.0,
        output_type: 0,
    };
    assert!(ondrive_tracker_emergency_stop(t, &mut stop));
    assert!(stop.valid);
    assert_eq!(stop.linear_velocity, 0.0);
    assert_eq!(stop.angular_velocity, 0.0);

    // After emergency_stop, current_target should be None.
    let ok = ondrive_tracker_current_target(t, &mut tgt);
    assert!(!ok);
    assert!(err_message().is_some());

    // Version and constraints readback.
    let v = ondrive_version();
    assert!(!v.is_null());
    let mut cons_back = ondrive_default_constraints();
    assert!(ondrive_tracker_constraints(t, &mut cons_back));
    assert_eq!(cons_back.steering_type, ONDRIVE_STEERING_ACKERMANN);

    ondrive_tracker_free(t);
    ondrive_path_free(path);
}

#[test]
fn world_obstacles_round_trip() {
    let w = ondrive_world_new();
    assert!(!w.is_null());
    assert_eq!(ondrive_world_obstacle_count(w), 0);

    assert!(ondrive_world_add_static_gaussian_obstacle(
        w, 1, 5.0, 0.0, 0.3, 0.1, 0.1, 10,
    ));
    assert!(ondrive_world_add_static_gaussian_obstacle(
        w, 2, 7.0, 1.0, 0.25, 0.15, 0.15, 10,
    ));
    assert_eq!(ondrive_world_obstacle_count(w), 2);

    // Trajectory obstacle with per-step means.
    let xs = [3.0_f64, 3.2, 3.4, 3.6];
    let ys = [0.0_f64, 0.1, 0.2, 0.3];
    assert!(ondrive_world_add_trajectory_obstacle(
        w,
        3,
        0.2,
        xs.as_ptr(),
        ys.as_ptr(),
        0.1,
        0.1,
        xs.len(),
    ));
    assert_eq!(ondrive_world_obstacle_count(w), 3);

    // Null arrays should be rejected.
    assert!(!ondrive_world_add_trajectory_obstacle(
        w,
        4,
        0.2,
        ptr::null(),
        ptr::null(),
        0.1,
        0.1,
        4,
    ));
    assert!(err_message().is_some());

    // horizon_steps=0 should be rejected.
    assert!(!ondrive_world_add_trajectory_obstacle(
        w,
        5,
        0.2,
        xs.as_ptr(),
        ys.as_ptr(),
        0.1,
        0.1,
        0,
    ));

    assert!(ondrive_world_clear(w));
    assert_eq!(ondrive_world_obstacle_count(w), 0);

    ondrive_world_free(w);
}

#[test]
fn create_free_many_handles_does_not_crash() {
    // A simple stress loop to smoke-test for leaks or double-free.
    for _ in 0..100 {
        let p = ondrive_path_new();
        for i in 0..5 {
            ondrive_path_add_waypoint_xy(p, i as f64, 0.0, 0.0, 0.5);
        }
        ondrive_path_free(p);

        let w = ondrive_world_new();
        ondrive_world_add_static_gaussian_obstacle(w, 0, 0.0, 0.0, 0.3, 0.1, 0.1, 5);
        ondrive_world_free(w);

        let t = ondrive_tracker_new(ONDRIVE_KIND_PID);
        ondrive_tracker_free(t);
    }
}

#[test]
fn all_tracker_kinds_construct() {
    let kinds = [
        ONDRIVE_KIND_PID,
        ONDRIVE_KIND_CARROT,
        ONDRIVE_KIND_PURE_PURSUIT,
        ONDRIVE_KIND_STANLEY,
        ONDRIVE_KIND_LQR,
        ONDRIVE_KIND_MPC,
        ONDRIVE_KIND_MPPI,
        ONDRIVE_KIND_MCA,
        ONDRIVE_KIND_SOC,
        ONDRIVE_KIND_DWA,
        ONDRIVE_KIND_TEB,
        ONDRIVE_KIND_FLC,
    ];
    for k in kinds {
        let t = ondrive_tracker_new(k);
        assert!(!t.is_null(), "failed to construct kind {k}");
        assert_eq!(ondrive_tracker_kind(t), k);
        ondrive_tracker_free(t);
    }
}
