//! Regulated Pure Pursuit behaviour.

#![allow(clippy::field_reassign_with_default)]

use datapod::{Euler, Point, Pose, Quaternion};
use ondrive::{
    Controller, ControllerConfig, GaussianMode, Goal, Obstacle, OutputUnits, Path,
    PurePursuitFollower, RegulatedPursuitFollower, RobotConstraints, RobotState, SteeringType,
    WorldConstraints,
};
use std::f64::consts::PI;

fn pose(x: f64, y: f64, yaw: f64) -> Pose {
    Pose {
        point: Point::new(x, y, 0.0),
        rotation: Quaternion::from_euler(Euler::new(0.0, 0.0, yaw)),
    }
}

fn path_through(pts: &[(f64, f64)]) -> Path {
    let mut p = Path::default();
    for w in pts.windows(2) {
        let h = (w[1].1 - w[0].1).atan2(w[1].0 - w[0].0);
        p.waypoints.push(pose(w[0].0, w[0].1, h));
    }
    let n = pts.len();
    p.waypoints.push(pose(pts[n - 1].0, pts[n - 1].1, (pts[n - 1].1 - pts[n - 2].1).atan2(pts[n - 1].0 - pts[n - 2].0)));
    p
}

fn line(len: f64) -> Path {
    let pts: Vec<(f64, f64)> = (0..=(len / 0.25) as usize).map(|i| (i as f64 * 0.25, 0.0)).collect();
    path_through(&pts)
}

fn hairpin() -> Path {
    let mut v: Vec<(f64, f64)> = (0..=16).map(|i| (i as f64 * 0.25, 0.0)).collect();
    v.extend((1..=12).map(|i| {
        let a = i as f64 / 12.0 * PI;
        (4.0 + 1.0 * a.sin(), 1.0 - 1.0 * a.cos())
    }));
    v.extend((1..=16).map(|i| (4.0 - i as f64 * 0.25, 2.0)));
    path_through(&v)
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

fn goal_at_end(path: &Path) -> Goal {
    Goal {
        target_pose: *path.waypoints.last().unwrap(),
        tolerance_position: 0.3,
        tolerance_orientation: PI,
        ..Default::default()
    }
}

struct Run {
    reached: bool,
    min_speed_mid: f64,
    max_speed: f64,
    pose: Pose,
    invalid_at: Option<Pose>,
}

fn simulate(ctrl: &mut dyn Controller, goal: &Goal, c: &RobotConstraints, start: Pose, world: Option<&WorldConstraints>) -> Run {
    let mut state = RobotState { pose: start, allow_move: true, ..Default::default() };
    let mut run = Run { reached: false, min_speed_mid: f64::MAX, max_speed: 0.0, pose: start, invalid_at: None };
    let dt = 0.05;
    for i in 0..1600 {
        let cmd = ctrl.compute_control(&state, goal, c, dt, world);
        if !cmd.valid {
            run.invalid_at = Some(state.pose);
            break;
        }
        run.max_speed = run.max_speed.max(cmd.linear_velocity.abs());
        if (200..600).contains(&i) {
            run.min_speed_mid = run.min_speed_mid.min(cmd.linear_velocity.abs());
        }
        let yaw = state.pose.rotation.to_euler().yaw;
        state.pose.point.x += cmd.linear_velocity * yaw.cos() * dt;
        state.pose.point.y += cmd.linear_velocity * yaw.sin() * dt;
        state.pose.rotation = Quaternion::from_euler(Euler::new(0.0, 0.0, yaw + cmd.angular_velocity * dt));
        state.velocity.linear = cmd.linear_velocity;
        state.velocity.angular = cmd.angular_velocity;
        if ctrl.get_status().goal_reached {
            run.reached = true;
            break;
        }
    }
    run.pose = state.pose;
    run
}

#[test]
fn rpp_slows_on_the_hairpin_and_reaches_the_end() {
    for steering in [SteeringType::Differential, SteeringType::Ackermann] {
        let path = hairpin();
        let mut rpp = RegulatedPursuitFollower::new();
        rpp.set_config(config());
        rpp.set_path(path.clone());
        let c = constraints(steering);
        let run = simulate(&mut rpp, &goal_at_end(&path), &c, pose(0.0, 0.0, 0.0), None);
        assert!(run.reached, "{steering:?}: ended at ({:.2},{:.2})", run.pose.point.x, run.pose.point.y);
        assert!(run.min_speed_mid < 0.75, "{steering:?}: no curvature regulation, min speed {:.2}", run.min_speed_mid);
    }
}

#[test]
fn rpp_matches_pure_pursuit_on_a_straight_line() {
    let path = line(10.0);
    let c = constraints(SteeringType::Ackermann);
    let mut rpp = RegulatedPursuitFollower::new();
    rpp.set_config(config());
    rpp.set_path(path.clone());
    let mut pp = PurePursuitFollower::new();
    pp.set_config(config());
    pp.set_path(path.clone());
    let a = simulate(&mut rpp, &goal_at_end(&path), &c, pose(0.0, 0.3, 0.0), None);
    let b = simulate(&mut pp, &goal_at_end(&path), &c, pose(0.0, 0.3, 0.0), None);
    assert!(a.reached && b.reached);
    assert!((a.max_speed - b.max_speed).abs() < 0.05);
    assert!(a.pose.point.distance_to_2d(b.pose.point) < 0.3);
}

#[test]
fn rpp_stops_before_an_obstacle_on_the_path() {
    let path = line(10.0);
    let world = WorldConstraints {
        obstacles: vec![Obstacle {
            id: 0,
            radius: 0.3,
            modes: vec![GaussianMode { weight: 1.0, mean_x: vec![5.0], mean_y: vec![0.0], std_x: vec![0.05], std_y: vec![0.05] }],
        }],
        ..Default::default()
    };
    let c = constraints(SteeringType::Differential);
    let mut rpp = RegulatedPursuitFollower::new();
    rpp.set_config(config());
    rpp.set_path(path.clone());
    let run = simulate(&mut rpp, &goal_at_end(&path), &c, pose(0.0, 0.0, 0.0), Some(&world));
    let stop = run.invalid_at.expect("command should become invalid before the obstacle");
    let d = (stop.point.x - 5.0).hypot(stop.point.y);
    assert!(d > 0.66, "invalidated only {d:.2} m from the obstacle centre");
    assert!(d < 3.0, "invalidated too early, {d:.2} m away");
}
