//! Holonomic (omni/mecanum) steering: translation and rotation are
//! independent. These tests exercise the property a nonholonomic platform
//! cannot have — reaching a goal or tracking a path via `lateral_velocity`
//! without first rotating to face the direction of travel.

#![allow(clippy::field_reassign_with_default)]

use datapod::{Euler, Point, Pose, Quaternion};
use ondrive::core::kinematics::{finalize_holonomic, world_to_body};
use ondrive::{
    ApfFollower, CarrotFollower, Controller, ControllerConfig, DwaConfig, DwaFollower, FlcConfig,
    FlcFollower, GaussianMode, Goal, KanayamaFollower, LqrFollower, Obstacle, OutputUnits, Path,
    PidFollower, PoseRegulatorFollower, PurePursuitFollower, RegulatedPursuitFollower,
    RobotConstraints, RobotState, SteeringType, StanleyFollower, VectorPursuitFollower,
    VelocityCommand, WorldConstraints,
};
use std::f64::consts::{FRAC_PI_2, PI};

fn pose(x: f64, y: f64, yaw: f64) -> Pose {
    Pose {
        point: Point::new(x, y, 0.0),
        rotation: Quaternion::from_euler(Euler::new(0.0, 0.0, yaw)),
    }
}

fn constraints() -> RobotConstraints {
    let mut c = RobotConstraints::default();
    c.steering_type = SteeringType::Holonomic;
    c.max_linear_velocity = 1.0;
    c.max_angular_velocity = 2.0;
    c.max_linear_acceleration = 2.0;
    c.max_angular_acceleration = 4.0;
    c.robot_width = 0.4;
    c.robot_length = 0.4;
    c
}

fn config() -> ControllerConfig {
    let mut cfg = ControllerConfig::default();
    cfg.output_units = OutputUnits::Physical;
    cfg.goal_tolerance = 0.15;
    cfg.angular_tolerance = 0.15;
    cfg.lookahead_distance = 0.8;
    cfg.kp_linear = 1.2;
    cfg.kp_angular = 2.0;
    cfg.k_cross_track = 1.5;
    cfg.k_heading = 1.0;
    cfg
}

/// Integrate a holonomic command: body-frame `(linear, lateral)` rotates
/// into the world frame, `angular` integrates yaw directly.
fn integrate(state: &mut RobotState, cmd: &VelocityCommand, dt: f64) {
    let yaw = state.pose.rotation.to_euler().yaw;
    let (s, c) = yaw.sin_cos();
    let wx = c * cmd.linear_velocity - s * cmd.lateral_velocity;
    let wy = s * cmd.linear_velocity + c * cmd.lateral_velocity;
    state.pose.point.x += wx * dt;
    state.pose.point.y += wy * dt;
    state.pose.rotation = Quaternion::from_euler(Euler::new(0.0, 0.0, yaw + cmd.angular_velocity * dt));
    state.velocity.linear = cmd.linear_velocity;
    state.velocity.lateral = cmd.lateral_velocity;
    state.velocity.angular = cmd.angular_velocity;
}

// ---------------------------------------------------------------------
// Core kinematics
// ---------------------------------------------------------------------

#[test]
fn finalize_holonomic_clamps_speed_as_a_vector_and_omega_independently() {
    let mut c = constraints();
    c.max_linear_velocity = 1.0;
    c.max_angular_velocity = 2.0;
    let cfg = config();
    let cmd = finalize_holonomic(3.0, 4.0, 10.0, &c, &cfg, "test");
    assert!(cmd.valid);
    // Direction preserved (3:4:5 triangle), magnitude clamped to 1.
    assert!((cmd.linear_velocity - 0.6).abs() < 1e-9, "vx={}", cmd.linear_velocity);
    assert!((cmd.lateral_velocity - 0.8).abs() < 1e-9, "vy={}", cmd.lateral_velocity);
    assert!((cmd.angular_velocity - 2.0).abs() < 1e-9, "omega clamped independently");
    assert_eq!(cmd.steering_angle, 0.0);
}

#[test]
fn finalize_holonomic_normalizes_all_three_axes() {
    let mut c = constraints();
    c.max_linear_velocity = 2.0;
    c.max_angular_velocity = 4.0;
    let mut cfg = config();
    cfg.output_units = OutputUnits::Normalized;
    let cmd = finalize_holonomic(1.0, -1.0, 2.0, &c, &cfg, "test");
    assert!((cmd.linear_velocity - 0.5).abs() < 1e-9);
    assert!((cmd.lateral_velocity - (-0.5)).abs() < 1e-9);
    assert!((cmd.angular_velocity - 0.5).abs() < 1e-9);
}

#[test]
fn world_to_body_matches_standard_rotation() {
    // A world-frame vector pointing straight left of a robot facing +x
    // (yaw=0) is body-frame left (positive y), not forward.
    let (bx, by) = world_to_body(0.0, 1.0, 0.0);
    assert!(bx.abs() < 1e-9 && (by - 1.0).abs() < 1e-9);
    // At yaw=90deg (facing +y), world +x is now body -y (right).
    let (bx, by) = world_to_body(1.0, 0.0, FRAC_PI_2);
    assert!(bx.abs() < 1e-6 && (by + 1.0).abs() < 1e-6);
}

// ---------------------------------------------------------------------
// Point controllers: strafe to a goal beside/behind without turning
// ---------------------------------------------------------------------

fn assert_strafes_without_turning(mut ctrl: impl Controller, goal_xy: (f64, f64)) {
    let c = constraints();
    let goal = Goal {
        target_pose: pose(goal_xy.0, goal_xy.1, 0.0),
        tolerance_position: 0.15,
        tolerance_orientation: 0.15,
        ..Default::default()
    };
    ctrl.set_config(config());
    let mut state = RobotState {
        pose: pose(0.0, 0.0, 0.0),
        allow_move: true,
        ..Default::default()
    };
    let mut max_yaw_drift = 0.0_f64;
    let mut saw_lateral = false;
    let mut reached = false;
    for _ in 0..2000 {
        let cmd = ctrl.compute_control(&state, &goal, &c, 0.05, None);
        assert!(cmd.valid, "{}", cmd.status_message);
        if cmd.lateral_velocity.abs() > 0.05 {
            saw_lateral = true;
        }
        integrate(&mut state, &cmd, 0.05);
        max_yaw_drift = max_yaw_drift.max(state.pose.rotation.to_euler().yaw.abs());
        if ctrl.get_status().goal_reached {
            reached = true;
            break;
        }
    }
    assert!(reached, "did not reach ({:.2},{:.2}); ended at ({:.2},{:.2})", goal_xy.0, goal_xy.1, state.pose.point.x, state.pose.point.y);
    assert!(
        max_yaw_drift < 0.3,
        "yaw drifted to {max_yaw_drift:.3} rad even though start and goal orientation match; \
         a holonomic platform should not need to rotate to reach a goal beside or behind it"
    );
    // A goal directly behind (on the body x-axis the whole time, since yaw
    // never needs to change) is legitimately reached with zero lateral
    // velocity; only require strafing when the goal actually has a lateral
    // offset from the start heading.
    if goal_xy.1.abs() > 0.5 {
        assert!(saw_lateral, "never commanded a nonzero lateral velocity");
    }
}

#[test]
fn pid_holonomic_strafes_to_goals_it_is_not_facing() {
    assert_strafes_without_turning(PidFollower::new(), (0.0, 3.0)); // directly left
    assert_strafes_without_turning(PidFollower::new(), (-3.0, 0.0)); // directly behind
}

#[test]
fn carrot_holonomic_strafes_to_a_goal_beside_it() {
    assert_strafes_without_turning(CarrotFollower::new(), (0.0, -2.5));
}

#[test]
fn pose_regulator_holonomic_strafes_to_a_goal_behind_it() {
    assert_strafes_without_turning(PoseRegulatorFollower::new(), (-2.0, 1.0));
}

#[test]
fn apf_holonomic_strafes_toward_a_goal_beside_it() {
    assert_strafes_without_turning(ApfFollower::new(), (0.5, 4.0));
}

// ---------------------------------------------------------------------
// Path followers: track a straight path while starting perpendicular to it
// ---------------------------------------------------------------------

fn straight_path(len: f64) -> Path {
    let mut p = Path::default();
    p.waypoints = (0..=(len / 0.25) as usize)
        .map(|i| pose(i as f64 * 0.25, 0.0, 0.0))
        .collect();
    p
}

fn assert_tracks_while_perpendicular(mut ctrl: impl Controller, name: &str) {
    let c = constraints();
    let path = straight_path(10.0);
    let goal = Goal {
        target_pose: *path.waypoints.last().unwrap(),
        tolerance_position: 0.3,
        tolerance_orientation: PI,
        ..Default::default()
    };
    ctrl.set_config(config());
    ctrl.set_path(path);
    // Start ON the path line but rotated 90 degrees from its tangent: a
    // nonholonomic tracker here must first rotate before making progress.
    let mut state = RobotState {
        pose: pose(0.0, 0.4, FRAC_PI_2),
        allow_move: true,
        ..Default::default()
    };
    let mut x_after_half_second = None;
    let mut max_cte_settled: f64 = 0.0;
    let mut reached = false;
    for i in 0..3000 {
        let cmd = ctrl.compute_control(&state, &goal, &c, 0.05, None);
        assert!(cmd.valid, "{name}: {}", cmd.status_message);
        integrate(&mut state, &cmd, 0.05);
        if i == 10 {
            x_after_half_second = Some(state.pose.point.x);
        }
        if i > 100 {
            max_cte_settled = max_cte_settled.max(ctrl.get_status().cross_track_error.abs());
        }
        if ctrl.get_status().goal_reached {
            reached = true;
            break;
        }
    }
    assert!(reached, "{name}: did not reach the end; final=({:.2},{:.2})", state.pose.point.x, state.pose.point.y);
    assert!(
        x_after_half_second.unwrap() > 0.1,
        "{name}: made no forward progress in the first 0.5s while perpendicular to the path (x={:.3}); \
         a holonomic tracker should not need to rotate first",
        x_after_half_second.unwrap()
    );
    let final_yaw = state.pose.rotation.to_euler().yaw.abs();
    assert!(final_yaw < 0.3, "{name}: final yaw {final_yaw:.3} did not converge to the path tangent");
    assert!(max_cte_settled < 0.6, "{name}: settled cross-track error {max_cte_settled:.3}");
}

#[test]
fn pure_pursuit_holonomic_tracks_while_perpendicular_to_the_path() {
    assert_tracks_while_perpendicular(PurePursuitFollower::new(), "pure_pursuit");
}

#[test]
fn stanley_holonomic_tracks_while_perpendicular_to_the_path() {
    assert_tracks_while_perpendicular(StanleyFollower::new(), "stanley");
}

#[test]
fn regulated_pursuit_holonomic_tracks_while_perpendicular_to_the_path() {
    assert_tracks_while_perpendicular(RegulatedPursuitFollower::new(), "regulated_pursuit");
}

#[test]
fn vector_pursuit_holonomic_tracks_while_perpendicular_to_the_path() {
    assert_tracks_while_perpendicular(VectorPursuitFollower::new(), "vector_pursuit");
}

#[test]
fn lqr_holonomic_tracks_while_perpendicular_to_the_path() {
    assert_tracks_while_perpendicular(LqrFollower::new(), "lqr");
}

#[test]
fn kanayama_holonomic_tracks_while_perpendicular_to_the_path() {
    assert_tracks_while_perpendicular(KanayamaFollower::new(), "kanayama");
}

#[test]
fn flc_holonomic_tracks_while_perpendicular_to_the_path() {
    let mut ctrl = FlcFollower::with_flc_config(FlcConfig::default());
    ctrl.set_config(config());
    assert_tracks_while_perpendicular(ctrl, "flc");
}

// ---------------------------------------------------------------------
// Sign check: cross-track correction pulls the robot the right way
// ---------------------------------------------------------------------

#[test]
fn stanley_holonomic_strafes_toward_the_path_not_away_from_it() {
    let c = constraints();
    let path = straight_path(10.0);
    let goal = Goal { target_pose: *path.waypoints.last().unwrap(), tolerance_position: 0.3, tolerance_orientation: PI, ..Default::default() };
    let mut left = StanleyFollower::new();
    left.set_config(config());
    left.set_path(path.clone());
    let state_left = RobotState { pose: pose(1.0, 0.8, 0.0), allow_move: true, ..Default::default() };
    let cmd = left.compute_control(&state_left, &goal, &c, 0.05, None);
    assert!(cmd.valid);
    // Robot is left of the path (positive lateral error): correction must
    // pull it right, i.e. negative body-frame lateral velocity.
    assert!(cmd.lateral_velocity < -1e-6, "vy={} should be negative (pull right)", cmd.lateral_velocity);

    let mut right = StanleyFollower::new();
    right.set_config(config());
    right.set_path(path);
    let state_right = RobotState { pose: pose(1.0, -0.8, 0.0), allow_move: true, ..Default::default() };
    let cmd = right.compute_control(&state_right, &goal, &c, 0.05, None);
    assert!(cmd.lateral_velocity > 1e-6, "vy={} should be positive (pull left)", cmd.lateral_velocity);
}

// ---------------------------------------------------------------------
// DWA: obstacle avoidance using the lateral sampling dimension
// ---------------------------------------------------------------------

#[test]
fn dwa_holonomic_sidesteps_an_obstacle_directly_ahead() {
    let mut c = constraints();
    c.robot_width = 0.4;
    c.robot_length = 0.4;
    let mut cfg = DwaConfig::default();
    cfg.v_samples = 8;
    cfg.w_samples = 9;
    cfg.vy_samples = 9;
    let mut ctrl = DwaFollower::with_dwa_config(cfg);
    ctrl.set_config(config());
    let goal = Goal { target_pose: pose(6.0, 0.0, 0.0), tolerance_position: 0.3, tolerance_orientation: PI, ..Default::default() };
    let world = WorldConstraints {
        obstacles: vec![Obstacle {
            id: 0,
            radius: 0.3,
            modes: vec![GaussianMode { weight: 1.0, mean_x: vec![3.0], mean_y: vec![0.0], std_x: vec![0.05], std_y: vec![0.05] }],
        }],
        ..Default::default()
    };
    let mut state = RobotState { pose: pose(0.0, 0.0, 0.0), allow_move: true, ..Default::default() };
    let mut reached = false;
    let mut min_clear = f64::MAX;
    let mut used_lateral = false;
    for _ in 0..600 {
        let cmd = ctrl.compute_control(&state, &goal, &c, 0.1, Some(&world));
        assert!(cmd.valid, "{}", cmd.status_message);
        if cmd.lateral_velocity.abs() > 0.05 {
            used_lateral = true;
        }
        integrate(&mut state, &cmd, 0.1);
        min_clear = min_clear.min(((state.pose.point.x - 3.0).powi(2) + state.pose.point.y.powi(2)).sqrt());
        if ctrl.get_status().goal_reached {
            reached = true;
            break;
        }
    }
    assert!(reached, "final=({:.2},{:.2})", state.pose.point.x, state.pose.point.y);
    assert!(min_clear > 0.3 + 0.5 * (0.4_f64).hypot(0.4), "min_clear={min_clear:.3}");
    assert!(used_lateral, "DWA never sampled a nonzero lateral velocity to sidestep the obstacle");
}
