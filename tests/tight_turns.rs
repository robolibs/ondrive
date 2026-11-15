//! A path turning tighter than the machine steers, driven in legs at full
//! lock by the tracker itself, clear of the street it turns in.

use datapod::{Euler, Point, Pose, Quaternion};
use ondrive::{
    ControllerConfig, Footprint, GaussianMode, Obstacle, OutputUnits, Path, RobotConstraints, RobotState, SteeringType,
    Tracker, TrackerKind, VelocityCommand, WorldConstraints,
};

fn pose(x: f64, y: f64, yaw: f64) -> Pose {
    Pose { point: Point::new(x, y, 0.0), rotation: Quaternion::from_euler(Euler::new(0.0, 0.0, yaw)) }
}

fn post(id: u64, x: f64, y: f64) -> Obstacle {
    let mode = GaussianMode { weight: 1.0, mean_x: vec![x; 31], mean_y: vec![y; 31], std_x: vec![0.02; 31], std_y: vec![0.02; 31] };
    Obstacle { id, radius: 0.22, dt: 0.2, modes: vec![mode] }
}

/// An Agilex Hunter: Ackermann, 0.65 m wheelbase, 0.46 rad lock, 1.06 × 0.75 m.
fn hunter() -> RobotConstraints {
    RobotConstraints {
        steering_type: SteeringType::Ackermann,
        wheelbase: 0.65,
        max_linear_velocity: 1.5,
        min_linear_velocity: -1.0,
        max_angular_velocity: 1.5,
        max_linear_acceleration: 1.0,
        max_angular_acceleration: 3.0,
        max_steering_angle: 0.46,
        rear_wheelbase: 0.28,
        robot_width: 0.75,
        robot_length: 1.06,
        footprint: Footprint::Polygon { points: vec![(0.575, 0.375), (-0.485, 0.375), (-0.485, -0.375), (0.575, -0.375)] },
        ..Default::default()
    }
}

/// East 10 m, a U-turn of 0.45 m radius three times tighter than the Hunter
/// steers, and back west 8 m.
fn u_turn() -> Path {
    let r = 0.45;
    let mut waypoints: Vec<Pose> = (0..=40).map(|i| pose(0.25 * i as f64, 0.0, 0.0)).collect();
    waypoints.extend((1..24).map(|k| {
        let a = -std::f64::consts::FRAC_PI_2 + std::f64::consts::PI * k as f64 / 24.0;
        pose(10.0 + r * a.cos(), r + r * a.sin(), a + std::f64::consts::FRAC_PI_2)
    }));
    waypoints.extend((0..=32).map(|i| pose(10.0 - 0.25 * i as f64, 2.0 * r, std::f64::consts::PI)));
    let speeds = vec![1.0; waypoints.len()];
    Path { waypoints, speeds, is_closed: false }
}

fn step(state: &mut RobotState, cmd: &VelocityCommand, c: &RobotConstraints, dt: f64) {
    let yaw = state.pose.rotation.to_euler().yaw;
    let w = cmd.linear_velocity * cmd.steering_angle.tan() / c.wheelbase;
    state.pose.point.x += cmd.linear_velocity * yaw.cos() * dt;
    state.pose.point.y += cmd.linear_velocity * yaw.sin() * dt;
    state.pose.rotation = Quaternion::from_euler(Euler::new(0.0, 0.0, yaw + w * dt));
    state.velocity.linear = cmd.linear_velocity;
    state.velocity.angular = w;
    state.velocity_valid = true;
}

#[test]
fn a_u_turn_tighter_than_the_lock_is_driven_in_legs_clear_of_the_street() {
    let c = hunter();
    let walls: Vec<Obstacle> = (0..60).flat_map(|i| [post(2 * i, 0.4 * i as f64 - 6.0, -1.4), post(2 * i + 1, 0.4 * i as f64 - 6.0, 2.3)]).collect();
    let world = WorldConstraints { obstacles: walls.clone(), ..Default::default() };
    for kind in [TrackerKind::Mppi, TrackerKind::Mpc] {
        let mut t = Tracker::new(kind);
        t.set_config(ControllerConfig {
            allow_reverse: true,
            goal_tolerance: 0.4,
            angular_tolerance: std::f64::consts::PI,
            output_units: OutputUnits::Physical,
            ..Default::default()
        });
        t.init(c.clone());
        t.set_seed(7);
        t.set_path(u_turn());
        let mut state = RobotState { pose: pose(0.0, 0.0, 0.0), allow_move: true, allow_reverse: true, velocity_valid: true, ..Default::default() };
        let (mut legs, mut last_dir, mut closest) = (0, 0.0_f64, f64::INFINITY);
        for _ in 0..2400 {
            let cmd = t.tick(&state, 0.05, Some(&world));
            assert!(cmd.valid, "{kind:?}: {}", cmd.status_message);
            if t.is_path_completed() {
                break;
            }
            let dir = cmd.linear_velocity.signum();
            if cmd.linear_velocity.abs() > 0.05 && dir != last_dir {
                legs += 1;
                last_dir = dir;
            }
            step(&mut state, &cmd, &c, 0.05);
            let (s, co) = state.pose.rotation.to_euler().yaw.sin_cos();
            for o in &walls {
                let (dx, dy) = (o.modes[0].mean_x[0] - state.pose.point.x, o.modes[0].mean_y[0] - state.pose.point.y);
                let (fx, fy) = (dx * co + dy * s, -dx * s + dy * co);
                closest = closest.min((fx - fx.clamp(-0.485, 0.575)).hypot(fy - fy.clamp(-0.375, 0.375)) - o.radius);
            }
        }
        assert!(t.is_path_completed(), "{kind:?}: not round, at {:?}", state.pose.point);
        assert!((2..=6).contains(&legs), "{kind:?}: {legs} legs");
        assert!(closest > 0.1, "{kind:?}: came {closest:.2} m from a post");
    }
}
