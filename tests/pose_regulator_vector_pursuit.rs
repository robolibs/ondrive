//! Astolfi pose regulator and Vector Pursuit behaviour.

#![allow(clippy::field_reassign_with_default)]

use datapod::{Euler, Point, Pose, Quaternion};
use ondrive::{
    Controller, ControllerConfig, Goal, OutputUnits, Path, PoseRegulatorFollower,
    PurePursuitFollower, RobotConstraints, RobotState, SteeringType, VectorPursuitFollower,
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
    cfg.goal_tolerance = 0.1;
    cfg.angular_tolerance = 0.1;
    cfg.lookahead_distance = 1.0;
    cfg.kp_linear = 1.0;
    cfg.kp_angular = 2.5;
    cfg.k_heading = 1.0;
    cfg
}

fn integrate(state: &mut RobotState, cmd: &ondrive::VelocityCommand, dt: f64) {
    let yaw = state.pose.rotation.to_euler().yaw;
    state.pose.point.x += cmd.linear_velocity * yaw.cos() * dt;
    state.pose.point.y += cmd.linear_velocity * yaw.sin() * dt;
    state.pose.rotation = Quaternion::from_euler(Euler::new(0.0, 0.0, yaw + cmd.angular_velocity * dt));
    state.velocity.linear = cmd.linear_velocity;
}

#[test]
fn pose_regulator_converges_in_position_and_orientation() {
    let c = constraints(SteeringType::Differential);
    for (gx, gy, gyaw) in [(3.0, 1.0, PI / 2.0), (2.0, -2.0, PI), (0.0, 2.0, 0.0), (-2.0, 0.5, -PI / 2.0)] {
        let mut ctrl = PoseRegulatorFollower::new();
        ctrl.set_config(config());
        let goal = Goal { target_pose: pose(gx, gy, gyaw), tolerance_position: 0.1, tolerance_orientation: 0.1, ..Default::default() };
        let mut state = RobotState { pose: pose(0.0, 0.0, 0.0), allow_move: true, ..Default::default() };
        let mut reached = false;
        for _ in 0..3000 {
            let cmd = ctrl.compute_control(&state, &goal, &c, 0.05, None);
            assert!(cmd.valid, "{}", cmd.status_message);
            assert!(cmd.linear_velocity >= -1e-9, "reversed without permission");
            integrate(&mut state, &cmd, 0.05);
            if ctrl.get_status().goal_reached {
                reached = true;
                break;
            }
        }
        assert!(reached, "goal ({gx},{gy},{gyaw}) not reached; ended at ({:.2},{:.2})", state.pose.point.x, state.pose.point.y);
        let yaw_err = (state.pose.rotation.to_euler().yaw - gyaw + PI).rem_euclid(2.0 * PI) - PI;
        assert!(state.pose.point.distance_to_2d(Point::new(gx, gy, 0.0)) < 0.15 && yaw_err.abs() < 0.15);
    }
}

#[test]
fn pose_regulator_reverses_to_a_goal_behind_when_allowed() {
    let c = constraints(SteeringType::Differential);
    let mut ctrl = PoseRegulatorFollower::new();
    let mut cfg = config();
    cfg.allow_reverse = true;
    ctrl.set_config(cfg);
    let goal = Goal { target_pose: pose(-2.0, 0.3, 0.0), tolerance_position: 0.1, tolerance_orientation: 0.1, ..Default::default() };
    let mut state = RobotState { pose: pose(0.0, 0.0, 0.0), allow_move: true, ..Default::default() };
    let mut reversed = false;
    let mut reached = false;
    for _ in 0..3000 {
        let cmd = ctrl.compute_control(&state, &goal, &c, 0.05, None);
        if cmd.linear_velocity < -1e-3 {
            reversed = true;
        }
        integrate(&mut state, &cmd, 0.05);
        if ctrl.get_status().goal_reached {
            reached = true;
            break;
        }
    }
    assert!(reached && reversed);
}

#[test]
fn pose_regulator_rejects_ackermann() {
    let c = constraints(SteeringType::Ackermann);
    let mut ctrl = PoseRegulatorFollower::new();
    ctrl.set_config(config());
    let goal = Goal { target_pose: pose(3.0, 0.0, 0.0), ..Default::default() };
    let state = RobotState { pose: pose(0.0, 0.0, 0.0), allow_move: true, ..Default::default() };
    assert!(!ctrl.compute_control(&state, &goal, &c, 0.05, None).valid);
}

fn corner_path() -> Path {
    let mut pts: Vec<(f64, f64)> = (0..=24).map(|i| (i as f64 * 0.25, 0.0)).collect();
    pts.extend((1..=24).map(|i| (6.0, i as f64 * 0.25)));
    let mut p = Path::default();
    for w in pts.windows(2) {
        let h = (w[1].1 - w[0].1).atan2(w[1].0 - w[0].0);
        p.waypoints.push(pose(w[0].0, w[0].1, h));
    }
    p.waypoints.push(pose(6.0, 6.0, PI / 2.0));
    p
}

fn max_cte_on_corner(ctrl: &mut dyn Controller, c: &RobotConstraints) -> (bool, f64) {
    let path = corner_path();
    let goal = Goal { target_pose: *path.waypoints.last().unwrap(), tolerance_position: 0.3, tolerance_orientation: PI, ..Default::default() };
    let mut cfg = config();
    cfg.goal_tolerance = 0.3;
    cfg.angular_tolerance = PI;
    ctrl.set_config(cfg);
    ctrl.set_path(path);
    let mut state = RobotState { pose: pose(0.0, 0.0, 0.0), allow_move: true, ..Default::default() };
    let mut max_cte: f64 = 0.0;
    for _ in 0..2000 {
        let cmd = ctrl.compute_control(&state, &goal, c, 0.05, None);
        assert!(cmd.valid);
        integrate(&mut state, &cmd, 0.05);
        max_cte = max_cte.max(ctrl.get_status().cross_track_error.abs());
        if ctrl.get_status().goal_reached {
            return (true, max_cte);
        }
    }
    (false, max_cte)
}

fn sine_path() -> Path {
    let mut p = Path::default();
    p.waypoints = (0..=80)
        .map(|i| {
            let x = i as f64 * 0.25;
            let x2 = x + 0.25;
            let f = |s: f64| (s * 0.45).sin() * 1.5;
            pose(x, f(x), (f(x2) - f(x)).atan2(0.25))
        })
        .collect();
    p
}

fn settled_cte_on_sine(ctrl: &mut dyn Controller, c: &RobotConstraints) -> (bool, f64) {
    let path = sine_path();
    let goal = Goal { target_pose: *path.waypoints.last().unwrap(), tolerance_position: 0.3, tolerance_orientation: PI, ..Default::default() };
    let mut cfg = config();
    cfg.goal_tolerance = 0.3;
    cfg.angular_tolerance = PI;
    ctrl.set_config(cfg);
    ctrl.set_path(path);
    let mut state = RobotState { pose: pose(0.0, 0.0, 0.0), allow_move: true, ..Default::default() };
    let mut max_cte: f64 = 0.0;
    for i in 0..3000 {
        let cmd = ctrl.compute_control(&state, &goal, c, 0.05, None);
        assert!(cmd.valid);
        integrate(&mut state, &cmd, 0.05);
        if i > 60 {
            max_cte = max_cte.max(ctrl.get_status().cross_track_error.abs());
        }
        if ctrl.get_status().goal_reached {
            return (true, max_cte);
        }
    }
    (false, max_cte)
}

#[test]
fn vector_pursuit_matches_pure_pursuit_on_curves() {
    for steering in [SteeringType::Differential, SteeringType::Ackermann] {
        let c = constraints(steering);
        let mut vp = VectorPursuitFollower::new();
        let (r1, cte_vp) = settled_cte_on_sine(&mut vp, &c);
        let mut pp = PurePursuitFollower::new();
        let (r2, cte_pp) = settled_cte_on_sine(&mut pp, &c);
        assert!(r1 && r2, "{steering:?}: reached vp={r1} pp={r2}");
        assert!(cte_vp <= cte_pp + 0.05, "{steering:?}: vector pursuit cte {cte_vp:.3} vs pure pursuit {cte_pp:.3}");
    }
}

fn heading_overshoot_from_offset(ctrl: &mut dyn Controller, c: &RobotConstraints) -> (bool, f64) {
    let mut path = Path::default();
    path.waypoints = (0..=60).map(|i| pose(i as f64 * 0.25, 0.0, 0.0)).collect();
    let goal = Goal { target_pose: *path.waypoints.last().unwrap(), tolerance_position: 0.3, tolerance_orientation: PI, ..Default::default() };
    let mut cfg = config();
    cfg.goal_tolerance = 0.3;
    cfg.angular_tolerance = PI;
    ctrl.set_config(cfg);
    ctrl.set_path(path);
    let mut state = RobotState { pose: pose(0.0, 1.0, 0.0), allow_move: true, ..Default::default() };
    let mut max_heading: f64 = 0.0;
    for _ in 0..2000 {
        let cmd = ctrl.compute_control(&state, &goal, c, 0.05, None);
        assert!(cmd.valid);
        integrate(&mut state, &cmd, 0.05);
        max_heading = max_heading.max(state.pose.rotation.to_euler().yaw.abs());
        if ctrl.get_status().goal_reached {
            return (true, max_heading);
        }
    }
    (false, max_heading)
}

#[test]
fn vector_pursuit_converges_with_less_heading_swing_than_pure_pursuit() {
    for steering in [SteeringType::Differential, SteeringType::Ackermann] {
        let c = constraints(steering);
        let mut vp = VectorPursuitFollower::new();
        let (r1, swing_vp) = heading_overshoot_from_offset(&mut vp, &c);
        let mut pp = PurePursuitFollower::new();
        let (r2, swing_pp) = heading_overshoot_from_offset(&mut pp, &c);
        assert!(r1 && r2, "{steering:?}: reached vp={r1} pp={r2}");
        assert!(swing_vp <= swing_pp + 1e-6, "{steering:?}: heading swing vp {swing_vp:.3} > pp {swing_pp:.3}");
    }
}

#[test]
fn vector_pursuit_completes_a_sharp_corner() {
    for steering in [SteeringType::Differential, SteeringType::Ackermann] {
        let c = constraints(steering);
        let mut vp = VectorPursuitFollower::new();
        let (reached, cte) = max_cte_on_corner(&mut vp, &c);
        assert!(reached && cte < 0.9, "{steering:?}: reached={reached} cte={cte:.3}");
    }
}
