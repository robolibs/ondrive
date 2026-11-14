//! Tick times of controllers driving a Kubota-sized tractor along a 90 m
//! straight past a standing obstacle, traffic crossing its path and traffic
//! coming the other way, each obstacle predicted as a Gaussian 6 s ahead.
//!
//! `cargo run --release --example tick_profile -- [ticks] [kind ...]`

#![allow(clippy::field_reassign_with_default)]

use datapod::{Euler, Point, Pose, Quaternion};
use ondrive::{
    ControllerConfig, Footprint, GaussianMode, Obstacle, OutputUnits, Path, RobotConstraints, RobotState, SteeringType,
    Tracker, TrackerKind, Velocity, WorldConstraints,
};
use std::time::Instant;

const DT: f64 = 0.05;
const PRED_DT: f64 = 0.2;
const PRED_STEPS: usize = 30;

fn pose(x: f64, y: f64, yaw: f64) -> Pose {
    Pose { point: Point::new(x, y, 0.0), rotation: Quaternion::from_euler(Euler::new(0.0, 0.0, yaw)) }
}

fn kubota() -> RobotConstraints {
    let mut c = RobotConstraints::default();
    c.steering_type = SteeringType::Ackermann;
    c.wheelbase = 2.7;
    c.max_linear_velocity = 2.0;
    c.max_angular_velocity = 0.7;
    c.max_linear_acceleration = 0.6;
    c.max_angular_acceleration = 1.4;
    c.max_steering_angle = 0.6;
    c.max_steering_rate = 0.4;
    c.rear_wheelbase = 1.365;
    c.robot_width = 2.3;
    c.robot_length = 5.1;
    let (front, back) = (-0.015 + 2.55, -0.015 - 2.55);
    c.footprint = Footprint::Polygon { points: vec![(front, 1.15), (back, 1.15), (back, -1.15), (front, -1.15)] };
    c
}

/// An obstacle at `(x, y)` moving at `(vx, vy)`, predicted at constant
/// velocity with a spread growing while it moves.
fn obstacle(id: u64, x: f64, y: f64, vx: f64, vy: f64, radius: f64) -> Obstacle {
    let moving = vx.hypot(vy) > 0.05;
    let k = |i: usize| i as f64 * PRED_DT;
    let sd: Vec<f64> = (0..=PRED_STEPS).map(|i| 0.05 + if moving { 0.1 * k(i) } else { 0.0 }).collect();
    let mode = GaussianMode {
        weight: 1.0,
        mean_x: (0..=PRED_STEPS).map(|i| x + vx * k(i)).collect(),
        mean_y: (0..=PRED_STEPS).map(|i| y + vy * k(i)).collect(),
        std_x: sd.clone(),
        std_y: sd,
    };
    Obstacle { id, radius, dt: PRED_DT, modes: vec![mode] }
}

fn kind_of(name: &str) -> TrackerKind {
    match name {
        "mpc" => TrackerKind::Mpc,
        "ilqr" => TrackerKind::Ilqr,
        "mppi" => TrackerKind::Mppi,
        "mca" => TrackerKind::Mca,
        "soc" => TrackerKind::Soc,
        "teb" => TrackerKind::Teb,
        "dwa" => TrackerKind::Dwa,
        "velocity_obstacle" => TrackerKind::VelocityObstacle,
        "frenet" => TrackerKind::Frenet,
        "pure_pursuit" => TrackerKind::PurePursuit,
        other => panic!("unknown kind {other}"),
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let ticks: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(600);
    let mut kinds: Vec<String> = args.collect();
    if kinds.is_empty() {
        kinds = ["mppi", "mca", "soc", "mpc", "teb", "ilqr"].iter().map(|s| s.to_string()).collect();
    }
    let mut path = Path::default();
    path.waypoints = (0..=180).map(|i| pose(0.5 * i as f64, 0.0, 0.0)).collect();
    for name in &kinds {
        let mut t = Tracker::new(kind_of(name));
        let mut cfg = ControllerConfig::default();
        cfg.output_units = OutputUnits::Physical;
        cfg.lookahead_distance = 2.55;
        cfg.goal_tolerance = 0.75;
        t.set_config(cfg);
        t.init(kubota());
        t.set_seed(7);
        t.set_path(path.clone());
        let (mut x, mut y, mut yaw, mut v, mut w) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
        let mut times: Vec<f64> = Vec::with_capacity(ticks);
        for k in 0..ticks {
            let now = k as f64 * DT;
            let world = WorldConstraints {
                obstacles: vec![
                    obstacle(0, 24.0, 1.2, 0.0, 0.0, 1.5),
                    obstacle(1, 45.0, -12.0 + 0.7 * now, 0.0, 0.7, 0.5),
                    obstacle(2, 90.0 - 0.6 * now, 0.6, -0.6, 0.0, 0.5),
                ],
                ..Default::default()
            };
            let state = RobotState {
                pose: pose(x, y, yaw),
                velocity: Velocity { linear: v, angular: w, lateral: 0.0 },
                velocity_valid: true,
                allow_move: true,
                ..Default::default()
            };
            let start = Instant::now();
            let cmd = t.tick(&state, DT, Some(&world));
            times.push(start.elapsed().as_secs_f64() * 1e3);
            if cmd.valid {
                (v, w) = (cmd.linear_velocity, cmd.angular_velocity);
            }
            yaw += w * DT;
            x += v * yaw.cos() * DT;
            y += v * yaw.sin() * DT;
        }
        let mean = times.iter().sum::<f64>() / times.len() as f64;
        times.sort_by(f64::total_cmp);
        let at = |q: f64| times[((times.len() as f64 * q) as usize).min(times.len() - 1)];
        println!(
            "{name:18} ticks={ticks} mean={mean:6.2} ms median={:6.2} ms p95={:6.2} ms max={:6.2} ms  end x={x:5.1}",
            at(0.5),
            at(0.95),
            times[times.len() - 1]
        );
    }
}
