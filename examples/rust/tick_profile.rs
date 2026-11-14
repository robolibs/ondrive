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
    // `ONDRIVE_CROWD` more walkers, crossing from either side of the path
    // over `ONDRIVE_CROWD_SPAN` metres of it.
    let walkers: usize = std::env::var("ONDRIVE_CROWD").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let span: f64 = std::env::var("ONDRIVE_CROWD_SPAN").ok().and_then(|v| v.parse().ok()).unwrap_or(110.0);
    let mut seed = 0x2545_f491_4f6c_dd1d_u64;
    let mut draw = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };
    let crowd: Vec<(f64, f64, f64, f64)> = (0..walkers)
        .map(|_| {
            let (x, side, speed) = (10.0 + span * draw(), if draw() < 0.5 { -1.0 } else { 1.0 }, 0.4 + 0.8 * draw());
            (x, side * (4.0 + 20.0 * draw()), 0.3 * (draw() - 0.5), -side * speed)
        })
        .collect();
    // `ONDRIVE_POSTS` standing posts over the 14.5 m either side of the path
    // from x = 15 m, at least 5.4 m apart edge to edge, four of them on it.
    let posts_wanted: usize = std::env::var("ONDRIVE_POSTS").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let mut posts: Vec<(f64, f64, f64)> = if posts_wanted > 0 { vec![(27.0, 0.45, 1.35), (57.0, -0.6, 1.35), (87.0, 0.3, 1.35)] } else { Vec::new() };
    for _ in 0..20000 {
        if posts.len() >= posts_wanted {
            break;
        }
        let (x, y, r) = (15.0 + 75.0 * draw(), 13.5 * (2.0 * draw() - 1.0), 0.75 + 0.75 * draw());
        if posts.iter().all(|&(px, py, pr)| (x - px).hypot(y - py) >= r + pr + 5.4) {
            posts.push((x, y, r));
        }
    }
    // Or the posts listed `x y radius` per line in `ONDRIVE_POSTS_FILE`.
    if let Ok(file) = std::env::var("ONDRIVE_POSTS_FILE") {
        let text = std::fs::read_to_string(&file).unwrap_or_else(|e| panic!("{file}: {e}"));
        posts = text
            .lines()
            .filter_map(|l| {
                let v: Vec<f64> = l.split_whitespace().filter_map(|t| t.parse().ok()).collect();
                (v.len() == 3).then(|| (v[0], v[1], v[2]))
            })
            .collect();
    }
    let length: f64 = std::env::var("ONDRIVE_PATH_M").ok().and_then(|v| v.parse().ok()).unwrap_or(90.0);
    let mut path = Path::default();
    path.waypoints = (0..=(2.0 * length) as usize).map(|i| pose(0.5 * i as f64, 0.0, 0.0)).collect();
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
            let mut obstacles = vec![
                obstacle(0, 24.0, 1.2, 0.0, 0.0, 1.5),
                obstacle(1, 45.0, -12.0 + 0.7 * now, 0.0, 0.7, 0.5),
                obstacle(2, 90.0 - 0.6 * now, 0.6, -0.6, 0.0, 0.5),
            ];
            obstacles.extend(crowd.iter().enumerate().map(|(i, &(x, y, vx, vy))| obstacle(3 + i as u64, x + vx * now, y + vy * now, vx, vy, 0.4)));
            obstacles.extend(posts.iter().enumerate().map(|(i, &(x, y, r))| obstacle(1000 + i as u64, x, y, 0.0, 0.0, r)));
            let world = WorldConstraints { obstacles, ..Default::default() };
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
            "{name:18} ticks={ticks} mean={mean:6.2} ms median={:6.2} ms p95={:6.2} ms max={:6.2} ms  end x={x:5.1} pose=({x:.9}, {y:.9}, {yaw:.9})",
            at(0.5),
            at(0.95),
            times[times.len() - 1]
        );
    }
}
