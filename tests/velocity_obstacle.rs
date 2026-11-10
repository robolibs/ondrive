#![allow(clippy::field_reassign_with_default)]

use datapod::{Euler, Point, Pose, Quaternion};
use ondrive::{
    ControllerConfig, Footprint, GaussianMode, Obstacle, OutputUnits, Path, RobotConstraints, RobotState, Tracker,
    TrackerKind, WorldConstraints,
};

fn pose(x: f64, y: f64, yaw: f64) -> Pose {
    Pose { point: Point::new(x, y, 0.0), rotation: Quaternion::from_euler(Euler::new(0.0, 0.0, yaw)) }
}

/// A mover's position at a time, and its radius.
type Mover = (Box<dyn Fn(f64) -> (f64, f64)>, f64);

/// Drive a velocity-obstacle follower along the x axis to x = 30 past
/// `movers`; the time it finished (or `None`) and the closest gap between
/// its 1 x 0.7 m body and any mover.
fn drive(movers: &[Mover]) -> (Option<f64>, f64) {
    let mut t = Tracker::new(TrackerKind::VelocityObstacle);
    let mut cfg = ControllerConfig::default();
    cfg.goal_tolerance = 0.3;
    cfg.lookahead_distance = 1.5;
    cfg.output_units = OutputUnits::Physical;
    t.set_config(cfg);
    let mut c = RobotConstraints::default();
    c.wheelbase = 0.65;
    c.max_steering_angle = 0.46;
    c.max_linear_velocity = 1.0;
    c.max_angular_velocity = 1.5;
    c.max_linear_acceleration = 1.0;
    c.footprint = Footprint::Polygon { points: vec![(0.5, 0.35), (-0.5, 0.35), (-0.5, -0.35), (0.5, -0.35)] };
    t.init(c);
    let waypoints: Vec<Pose> = (0..=60).map(|i| pose(0.5 * i as f64, 0.0, 0.0)).collect();
    t.set_path(Path { waypoints, speeds: Vec::new(), is_closed: false });

    let (dt, step, steps) = (0.05, 0.2, 30);
    let mut state = RobotState { allow_move: true, velocity_valid: true, ..Default::default() };
    let (mut clock, mut closest) = (0.0, f64::INFINITY);
    while clock < 120.0 {
        let obstacles = movers
            .iter()
            .enumerate()
            .map(|(id, (at, radius))| {
                let track: Vec<(f64, f64)> = (0..=steps).map(|k| at(clock + k as f64 * step)).collect();
                let mode = GaussianMode {
                    weight: 1.0,
                    mean_x: track.iter().map(|p| p.0).collect(),
                    mean_y: track.iter().map(|p| p.1).collect(),
                    std_x: vec![0.05; steps + 1],
                    std_y: vec![0.05; steps + 1],
                };
                Obstacle { id: id as u64, radius: *radius, modes: vec![mode], dt: step }
            })
            .collect();
        let world = WorldConstraints { obstacles, ..Default::default() };
        let cmd = t.tick(&state, dt, Some(&world));
        assert!(cmd.valid, "{}", cmd.status_message);
        let yaw = state.pose.rotation.to_euler().yaw;
        state.pose.point.x += cmd.linear_velocity * yaw.cos() * dt;
        state.pose.point.y += cmd.linear_velocity * yaw.sin() * dt;
        state.pose.rotation = Quaternion::from_euler(Euler::new(0.0, 0.0, yaw + cmd.angular_velocity * dt));
        state.velocity.linear = cmd.linear_velocity;
        state.velocity.angular = cmd.angular_velocity;
        clock += dt;
        let (s, co) = state.pose.rotation.to_euler().yaw.sin_cos();
        for (at, radius) in movers {
            let (mx, my) = at(clock);
            let (dx, dy) = (mx - state.pose.point.x, my - state.pose.point.y);
            let (fx, fy) = (dx * co + dy * s, -dx * s + dy * co);
            closest = closest.min((fx - fx.clamp(-0.5, 0.5)).hypot(fy - fy.clamp(-0.35, 0.35)) - radius);
        }
        if t.is_path_completed() {
            return (Some(clock), closest);
        }
    }
    (None, closest)
}

#[test]
fn it_follows_an_empty_path_at_speed() {
    let (done, _) = drive(&[]);
    let done = done.expect("the follower never reached the end");
    assert!(done < 40.0, "30 m took {done:.1} s");
}

#[test]
fn it_keeps_clear_of_traffic_crossing_its_path() {
    let (done, closest) = drive(&[(Box::new(|t: f64| (12.0, -8.0 + 0.8 * t)), 0.5)]);
    assert!(done.is_some(), "the follower never reached the end");
    assert!(closest > 0.3, "it came within {closest:.2} m of the crossing traffic");
}

#[test]
fn it_steers_round_traffic_coming_down_its_path() {
    let (done, closest) = drive(&[(Box::new(|t: f64| (28.0 - 0.6 * t, 0.1)), 0.5)]);
    assert!(done.is_some(), "the follower never reached the end");
    assert!(closest > 0.3, "it came within {closest:.2} m of the oncoming traffic");
}

#[test]
fn it_steers_round_a_rock_on_its_path() {
    let (done, closest) = drive(&[(Box::new(|_: f64| (12.0, 0.1)), 0.6)]);
    assert!(done.is_some(), "the follower never reached the end");
    assert!(closest > 0.2, "it came within {closest:.2} m of the rock");
}
