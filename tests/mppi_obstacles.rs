#![allow(clippy::field_reassign_with_default)]

use datapod::{Euler, Point, Pose, Quaternion};
use ondrive::{
    Controller, ControllerConfig, Footprint, GaussianMode, Goal, MppiConfig, MppiFollower, Obstacle, OutputUnits, Path,
    RobotConstraints, RobotState, SteeringType, Velocity, WorldConstraints,
};

fn pose_at(x: f64, y: f64, yaw: f64) -> Pose {
    Pose { point: Point::new(x, y, 0.0), rotation: Quaternion::from_euler(Euler::new(0.0, 0.0, yaw)) }
}

fn straight_path(length: f64) -> Path {
    let mut p = Path::default();
    p.waypoints = (0..=(length / 0.5) as usize).map(|i| pose_at(i as f64 * 0.5, 0.0, 0.0)).collect();
    p
}

fn post(id: u64, x: f64, y: f64, radius: f64) -> Obstacle {
    let n = 31;
    let mode = GaussianMode { weight: 1.0, mean_x: vec![x; n], mean_y: vec![y; n], std_x: vec![0.02; n], std_y: vec![0.02; n] };
    Obstacle { id, radius, dt: 0.2, modes: vec![mode] }
}

/// An Agilex Hunter: Ackermann, 0.65 m wheelbase, 0.46 rad lock turned at
/// 0.3 rad/s, 1.06 × 0.75 m.
fn hunter() -> RobotConstraints {
    let mut c = RobotConstraints::default();
    c.steering_type = SteeringType::Ackermann;
    c.wheelbase = 0.65;
    c.max_linear_velocity = 1.5;
    c.min_linear_velocity = 0.0;
    c.max_angular_velocity = 1.5;
    c.max_linear_acceleration = 1.0;
    c.max_angular_acceleration = 3.0;
    c.max_steering_angle = 0.46;
    c.max_steering_rate = 0.3;
    c.rear_wheelbase = 0.28;
    c.robot_width = 0.75;
    c.robot_length = 1.06;
    c.footprint = Footprint::Polygon { points: vec![(0.575, 0.375), (-0.485, 0.375), (-0.485, -0.375), (0.575, -0.375)] };
    c
}

fn follower_on(path: Path) -> MppiFollower {
    let mut cfg = MppiConfig::default();
    cfg.num_samples = 400;
    let mut follower = MppiFollower::with_seed(cfg, 7);
    let mut base = ControllerConfig::default();
    base.output_units = OutputUnits::Physical;
    base.goal_tolerance = 0.4;
    base.angular_tolerance = std::f64::consts::PI;
    follower.set_config(base);
    follower.set_path(path);
    follower
}

fn goal_at(path: &Path) -> Goal {
    Goal { target_pose: *path.waypoints.last().unwrap(), tolerance_position: 0.4, tolerance_orientation: std::f64::consts::PI, ..Default::default() }
}

/// Started on the second leg of an L, heading along it: it tracks that leg,
/// not the first.
#[test]
fn mppi_started_mid_path_tracks_the_path_where_it_stands() {
    let mut path = Path::default();
    path.waypoints = (0..=80).map(|i| pose_at(i as f64 * 0.5, 0.0, 0.0)).collect();
    path.waypoints.extend((1..=120).map(|i| pose_at(40.0, i as f64 * 0.5, std::f64::consts::FRAC_PI_2)));
    let goal = goal_at(&path);
    let mut follower = follower_on(path);
    let world = WorldConstraints { obstacles: vec![post(0, 30.0, 50.0, 0.5)], ..Default::default() };
    let state = RobotState {
        pose: pose_at(40.0, 30.0, std::f64::consts::FRAC_PI_2),
        velocity: Velocity { linear: 1.0, ..Default::default() },
        velocity_valid: true,
        allow_move: true,
        ..Default::default()
    };
    let cmd = follower.compute_control(&state, &goal, &hunter(), 0.1, Some(&world));
    assert!(cmd.valid);
    let status = follower.get_status();
    assert!(status.cross_track_error.abs() < 0.5, "{:.2} m off the leg it stands on", status.cross_track_error);
    assert!(status.heading_error.abs() < 0.2, "{:.2} rad off the leg it heads along", status.heading_error);
}

/// The Hunter at a stand 2.7 m short of a post just right of its heading,
/// in a field of posts: it drives on round it rather than waiting.
#[test]
fn mppi_drives_on_from_a_stand_behind_a_post() {
    let path = straight_path(30.0);
    let goal = goal_at(&path);
    let mut follower = follower_on(path);
    let posts = [(12.67, -1.27, 0.53), (9.47, 0.99, 0.63), (16.9, -0.22, 0.675), (5.75, -3.49, 0.67), (13.05, -5.07, 0.53)];
    let world = WorldConstraints { obstacles: posts.iter().enumerate().map(|(k, p)| post(k as u64, p.0, p.1, p.2)).collect(), ..Default::default() };
    let c = hunter();
    let (mut x, mut y, mut yaw, mut v, mut steer) = (10.0_f64, -1.67_f64, 0.25_f64, 0.0_f64, 0.0_f64);
    let dt = 0.1;
    let mut nearest = f64::INFINITY;
    for _ in 0..200 {
        let state = RobotState {
            pose: pose_at(x, y, yaw),
            velocity: Velocity { linear: v, angular: v * steer.tan() / c.wheelbase, ..Default::default() },
            velocity_valid: true,
            allow_move: true,
            ..Default::default()
        };
        let cmd = follower.compute_control(&state, &goal, &c, dt, Some(&world));
        if !cmd.valid || follower.get_status().goal_reached {
            break;
        }
        // The plant turns its wheels towards the commanded curvature at the
        // steering rate, and only while the command moves it.
        v = cmd.linear_velocity;
        if v.abs() > 0.05 {
            let target = (cmd.angular_velocity * c.wheelbase / v).atan().clamp(-c.max_steering_angle, c.max_steering_angle);
            steer += (target - steer).clamp(-c.max_steering_rate * dt, c.max_steering_rate * dt);
        }
        yaw += v * steer.tan() / c.wheelbase * dt;
        x += v * yaw.cos() * dt;
        y += v * yaw.sin() * dt;
        for p in &posts {
            nearest = nearest.min((p.0 - x).hypot(p.1 - y) - p.2);
        }
    }
    assert!(x > 18.0, "stood at ({x:.2}, {y:.2}) behind the posts");
    assert!(nearest > 0.45, "centre came {nearest:.2} m from a post's edge");
}
