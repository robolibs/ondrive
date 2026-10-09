//! One control tick per controller at default settings on a sine path.

#![allow(clippy::field_reassign_with_default)]

use criterion::{Criterion, criterion_group, criterion_main};
use datapod::{Euler, Point, Pose, Quaternion};
use ondrive::{ControllerConfig, Path, RobotConstraints, RobotState, SteeringType, Tracker, TrackerKind};
use std::f64::consts::PI;

fn pose(x: f64, y: f64, yaw: f64) -> Pose {
    Pose { point: Point::new(x, y, 0.0), rotation: Quaternion::from_euler(Euler::new(0.0, 0.0, yaw)) }
}

fn tick_benchmarks(c: &mut Criterion) {
    let kinds = [
        TrackerKind::Pid,
        TrackerKind::Carrot,
        TrackerKind::PurePursuit,
        TrackerKind::RegulatedPursuit,
        TrackerKind::VectorPursuit,
        TrackerKind::Stanley,
        TrackerKind::Lqr,
        TrackerKind::Kanayama,
        TrackerKind::Flc,
        TrackerKind::Mpc,
        TrackerKind::Ilqr,
        TrackerKind::Mppi,
        TrackerKind::Mca,
        TrackerKind::Soc,
        TrackerKind::Dwa,
        TrackerKind::Teb,
        TrackerKind::Ilc,
    ];
    let mut path = Path::default();
    path.waypoints = (0..=80).map(|i| { let x = i as f64 * 0.25; pose(x, (x * 0.35).sin() * 1.2, 0.0) }).collect();
    let mut group = c.benchmark_group("tick");
    group.sample_size(20);
    for kind in kinds {
        let mut t = Tracker::new(kind);
        let mut cfg = ControllerConfig::default();
        cfg.goal_tolerance = 0.3;
        cfg.angular_tolerance = PI;
        t.set_config(cfg);
        let mut con = RobotConstraints::default();
        con.steering_type = SteeringType::Ackermann;
        con.max_linear_velocity = 1.0;
        con.max_angular_velocity = 2.0;
        con.max_linear_acceleration = 1.5;
        con.max_steering_angle = 0.6;
        con.min_turning_radius = 0.0;
        con.wheelbase = 0.5;
        t.init(con);
        t.set_path(path.clone());
        let state = RobotState { pose: pose(2.0, 0.6, 0.2), allow_move: true, ..Default::default() };
        group.bench_function(format!("{kind:?}"), |b| b.iter(|| t.tick(&state, 0.05, None)));
    }
    group.finish();
}

criterion_group!(benches, tick_benchmarks);
criterion_main!(benches);
