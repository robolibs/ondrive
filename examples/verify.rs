//! Headless 2D verification matrix: every controller on both steering
//! models across straight, offset, heading-offset, sine, corner, circle,
//! hairpin, reverse and obstacle scenarios. Prints one line per case and a
//! failure count. Run with `cargo run --release --example verify`
//! (optionally `DT=0.02` in the environment to change the control period).

#![allow(clippy::field_reassign_with_default, clippy::type_complexity)]

use datapod::{Euler, Point, Pose, Quaternion};
use ondrive::*;
use std::f64::consts::PI;

fn pose(x: f64, y: f64, yaw: f64) -> Pose { Pose { point: Point::new(x, y, 0.0), rotation: Quaternion::from_euler(Euler::new(0.0, 0.0, yaw)) } }

fn path_from(pts: Vec<(f64, f64)>) -> Path {
    let mut p = Path::default();
    for w in pts.windows(2) { let (a, b) = (w[0], w[1]); let h = (b.1 - a.1).atan2(b.0 - a.0); p.waypoints.push(pose(a.0, a.1, h)); }
    let l = pts.len(); let (a, b) = (pts[l-2], pts[l-1]); p.waypoints.push(pose(b.0, b.1, (b.1-a.1).atan2(b.0-a.0)));
    p
}
fn polyline(f: impl Fn(f64) -> (f64, f64), s_end: f64, ds: f64) -> Path {
    let n = (s_end / ds).ceil() as usize;
    path_from((0..=n).map(|i| f(i as f64 * ds)).collect())
}

struct Scenario { name: &'static str, path: Path, start: Pose, reverse: bool, obstacle: Option<(f64, f64)> }

fn scenarios() -> Vec<Scenario> {
    let straight = polyline(|s| (s, 0.0), 15.0, 0.25);
    let sine = polyline(|s| (s, (s * 0.35).sin() * 1.2), 20.0, 0.25);
    let corner = { let mut v: Vec<(f64,f64)> = (0..=24).map(|i| (i as f64 * 0.25, 0.0)).collect(); v.extend((1..=24).map(|i| (6.0, i as f64 * 0.25))); path_from(v) };
    let circle = polyline(|s| { let a = s / 3.0; (3.0 * a.sin(), 3.0 * (1.0 - a.cos())) }, 3.0 * 1.5 * PI, 0.25);
    let hairpin = { let mut v: Vec<(f64,f64)> = (0..=16).map(|i| (i as f64 * 0.25, 0.0)).collect();
        v.extend((1..=12).map(|i| { let a = i as f64 / 12.0 * PI; (4.0 + 1.5 * a.sin(), 1.5 - 1.5 * a.cos()) }));
        v.extend((1..=16).map(|i| (4.0 - i as f64 * 0.25, 3.0))); path_from(v) };
    let behind = polyline(|s| (-s, 0.0), 8.0, 0.25);
    vec![
        Scenario { name: "straight", path: straight.clone(), start: pose(0.0, 0.0, 0.0), reverse: false, obstacle: None },
        Scenario { name: "offset-left-0.8", path: straight.clone(), start: pose(0.0, 0.8, 0.0), reverse: false, obstacle: None },
        Scenario { name: "heading+45", path: straight.clone(), start: pose(0.0, 0.0, PI / 4.0), reverse: false, obstacle: None },
        Scenario { name: "sine", path: sine.clone(), start: pose(0.0, -0.3, 0.0), reverse: false, obstacle: None },
        Scenario { name: "corner90", path: corner, start: pose(0.0, 0.0, 0.0), reverse: false, obstacle: None },
        Scenario { name: "circle-r3", path: circle, start: pose(0.0, 0.0, 0.0), reverse: false, obstacle: None },
        Scenario { name: "hairpin-r1.5", path: hairpin, start: pose(0.0, 0.0, 0.0), reverse: false, obstacle: None },
        Scenario { name: "behind-reverse", path: behind, start: pose(0.0, 0.2, 0.0), reverse: true, obstacle: None },
        Scenario { name: "sine-obstacle", path: sine, start: pose(0.0, 0.0, 0.0), reverse: false, obstacle: Some((8.0, (8.0f64 * 0.35).sin() * 1.2 + 0.4)) },
    ]
}

fn constraints(st: SteeringType) -> RobotConstraints {
    let mut c = RobotConstraints::default();
    c.steering_type = st; c.max_linear_velocity = 1.0; c.min_linear_velocity = -0.6; c.max_angular_velocity = 2.0;
    c.max_linear_acceleration = 1.5; c.max_angular_acceleration = 4.0; c.max_steering_angle = 0.6; c.min_turning_radius = 0.0;
    c.wheelbase = 0.5; c.rear_wheelbase = 0.0; c.robot_width = 0.4; c.robot_length = 0.6; c
}

fn main() {
    let kinds = [TrackerKind::Pid, TrackerKind::Carrot, TrackerKind::PurePursuit, TrackerKind::Stanley, TrackerKind::Lqr, TrackerKind::Mpc, TrackerKind::Mppi, TrackerKind::Mca, TrackerKind::Soc, TrackerKind::Dwa, TrackerKind::Teb, TrackerKind::Flc, TrackerKind::RegulatedPursuit, TrackerKind::Ilqr];
    let dt: f64 = std::env::var("DT").ok().and_then(|s| s.parse().ok()).unwrap_or(0.05);
    println!("{:<12} {:<12} {:<16} {:>5} {:>7} {:>7} {:>6} {:>6} {:>5} {:>6}", "kind", "steering", "scenario", "ok", "t", "cte", "vmax", "kviol", "nan", "obs");
    let mut failures = 0;
    for st in [SteeringType::Differential, SteeringType::Ackermann] {
        for kind in kinds {
            for sc in scenarios() {
                let want = |var: &str, actual: String| std::env::var(var).map(|w| w == actual).unwrap_or(true);
                if !want("KIND", format!("{kind:?}")) || !want("STEER", format!("{st:?}")) || !want("SCEN", sc.name.to_string()) { continue; }
                let obstacle_aware = matches!(kind, TrackerKind::Mca | TrackerKind::Soc | TrackerKind::Dwa | TrackerKind::Teb);
                if sc.obstacle.is_some() && !obstacle_aware { continue; }
                if sc.reverse && matches!(kind, TrackerKind::Lqr | TrackerKind::Flc | TrackerKind::Dwa | TrackerKind::Mpc | TrackerKind::Ilqr | TrackerKind::Mppi | TrackerKind::Mca | TrackerKind::Soc | TrackerKind::Teb) { continue; }
                let c = constraints(st);
                let kmax = c.max_steering_angle.tan() / c.wheelbase;
                let mut t = Tracker::new(kind);
                let mut cfg = ControllerConfig::default();
                cfg.goal_tolerance = 0.3; cfg.angular_tolerance = PI; cfg.lookahead_distance = 1.0; cfg.allow_reverse = sc.reverse;
                t.set_config(cfg); t.init(c.clone()); t.set_path(sc.path.clone());
                let world = sc.obstacle.map(|(ox, oy)| WorldConstraints { obstacles: vec![Obstacle { id: 0, radius: 0.3, modes: vec![GaussianMode { weight: 1.0, mean_x: vec![ox; 40], mean_y: vec![oy; 40], std_x: vec![0.05; 40], std_y: vec![0.05; 40] }] }], ..Default::default() });
                let mut state = RobotState { pose: sc.start, allow_move: true, ..Default::default() };
                let (mut reached, mut sim_t, mut cte_max, mut vmax, mut kviol, mut nan, mut min_obs) = (false, 0.0, 0.0f64, 0.0f64, 0usize, false, f64::MAX);
                let end = sc.path.waypoints.last().unwrap().point;
                for i in 0..4000 {
                    let cmd = t.tick(&state, dt, world.as_ref());
                    if !cmd.valid || !cmd.linear_velocity.is_finite() || !cmd.angular_velocity.is_finite() { nan = true; break; }
                    if st == SteeringType::Ackermann && cmd.angular_velocity.abs() > cmd.linear_velocity.abs() * kmax + 1e-6 { kviol += 1; }
                    vmax = vmax.max(cmd.linear_velocity.abs());
                    let yaw = state.pose.rotation.to_euler().yaw;
                    state.pose.point.x += cmd.linear_velocity * yaw.cos() * dt;
                    state.pose.point.y += cmd.linear_velocity * yaw.sin() * dt;
                    state.pose.rotation = Quaternion::from_euler(Euler::new(0.0, 0.0, yaw + cmd.angular_velocity * dt));
                    state.velocity.linear = cmd.linear_velocity; state.velocity.angular = cmd.angular_velocity;
                    sim_t = (i + 1) as f64 * dt;
                    if sim_t > 4.0 && state.pose.point.distance_to_2d(end) > 1.0 { cte_max = cte_max.max(t.get_status().cross_track_error.abs()); }
                    if let Some((ox, oy)) = sc.obstacle { min_obs = min_obs.min((state.pose.point.x - ox).hypot(state.pose.point.y - oy)); }
                    if t.is_goal_reached() { reached = true; break; }
                }
                let point_kind = matches!(kind, TrackerKind::Pid | TrackerKind::Carrot | TrackerKind::Dwa);
                let cte_limit = if point_kind || sc.obstacle.is_some() { f64::INFINITY } else if sc.name.starts_with("hairpin") || sc.name.starts_with("corner") { 0.9 } else { 0.5 };
                let obs_ok = sc.obstacle.is_none() || min_obs > 0.3 + 0.36;
                let ok = reached && !nan && kviol == 0 && cte_max <= cte_limit && obs_ok && vmax <= c.max_linear_velocity + 1e-6;
                if !ok { failures += 1; }
                println!("{:<12} {:<12} {:<16} {:>5} {:>7.1} {:>7.3} {:>6.2} {:>6} {:>5} {:>6}", format!("{kind:?}"), format!("{st:?}"), sc.name, if ok {"ok"} else {"FAIL"}, sim_t, cte_max, vmax, kviol, nan, if sc.obstacle.is_some() { format!("{min_obs:.2}") } else { "-".into() });
            }
        }
    }
    println!("failures: {failures}");
}
