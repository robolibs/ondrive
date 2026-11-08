//! Zone speed limits, the wheel-speed limit of a differential axle and the
//! steering-rate limit of an Ackermann platform.

#![allow(clippy::field_reassign_with_default)]

use datapod::{Euler, Point, Pose, Quaternion};
use ondrive::core::kinematics::finalize;
use ondrive::{
    ControllerConfig, OutputUnits, Path, RobotConstraints, RobotState, SteeringType, Tracker,
    TrackerKind, WorldConstraints, Zone,
};

fn pose(x: f64, y: f64, yaw: f64) -> Pose {
    Pose {
        point: Point::new(x, y, 0.0),
        rotation: Quaternion::from_euler(Euler::new(0.0, 0.0, yaw)),
    }
}

fn line(to: f64) -> Path {
    Path {
        waypoints: (0..=(to * 10.0) as usize)
            .map(|i| pose(i as f64 * 0.1, 0.0, 0.0))
            .collect(),
        ..Default::default()
    }
}

fn square(x0: f64, x1: f64, y0: f64, y1: f64, max_speed: f64) -> Zone {
    Zone {
        boundary: vec![
            Point::new(x0, y0, 0.0),
            Point::new(x1, y0, 0.0),
            Point::new(x1, y1, 0.0),
            Point::new(x0, y1, 0.0),
        ],
        max_speed,
    }
}

fn state(x: f64, y: f64, yaw: f64) -> RobotState {
    RobotState {
        pose: pose(x, y, yaw),
        allow_move: true,
        ..Default::default()
    }
}

#[test]
fn zones_contain_their_interior_and_the_slowest_wins() {
    let world = WorldConstraints {
        zones: vec![
            square(0.0, 4.0, -1.0, 1.0, 0.5),
            square(2.0, 3.0, -1.0, 1.0, 0.2),
        ],
        ..Default::default()
    };
    assert_eq!(world.speed_limit_at(Point::new(1.0, 0.0, 0.0)), Some(0.5));
    assert_eq!(world.speed_limit_at(Point::new(2.5, 0.0, 0.0)), Some(0.2));
    assert_eq!(world.speed_limit_at(Point::new(5.0, 0.0, 0.0)), None);
    assert_eq!(world.speed_limit_at(Point::new(1.0, 1.5, 0.0)), None);
}

#[test]
fn the_tracker_holds_the_speed_to_the_zone_limit() {
    let world = WorldConstraints {
        zones: vec![square(3.0, 6.0, -2.0, 2.0, 0.3)],
        ..Default::default()
    };
    for units in [OutputUnits::Physical, OutputUnits::Normalized] {
        for kind in [
            TrackerKind::PurePursuit,
            TrackerKind::Stanley,
            TrackerKind::Mpc,
            TrackerKind::Pid,
        ] {
            let mut t = Tracker::new(kind);
            t.set_config(ControllerConfig {
                output_units: units,
                ..Default::default()
            });
            t.init(RobotConstraints {
                steering_type: SteeringType::Differential,
                max_linear_velocity: 2.0,
                ..Default::default()
            });
            t.set_path(line(10.0));
            let scale = if units == OutputUnits::Normalized {
                2.0
            } else {
                1.0
            };
            let mut inside = state(4.0, 0.05, 0.0);
            inside.velocity.linear = 1.0;
            let mut outside = state(1.0, 0.05, 0.0);
            outside.velocity.linear = 1.0;
            let fast = t.tick(&outside, 0.05, Some(&world));
            let slow = t.tick(&inside, 0.05, Some(&world));
            assert!(slow.valid && fast.valid, "{kind:?}");
            assert!(
                slow.linear_velocity.abs() * scale <= 0.3 + 1e-9,
                "{kind:?} {units:?}: {} in the zone",
                slow.linear_velocity * scale
            );
            assert!(
                fast.linear_velocity.abs() * scale > 0.3,
                "{kind:?} {units:?}: {} outside the zone",
                fast.linear_velocity * scale
            );
        }
    }
}

#[test]
fn neither_wheel_of_a_differential_axle_exceeds_the_top_speed() {
    let cfg = ControllerConfig::default();
    for steering in [SteeringType::Differential, SteeringType::SkidSteer] {
        let c = RobotConstraints {
            steering_type: steering,
            track_width: 0.5,
            max_linear_velocity: 1.0,
            max_angular_velocity: 4.0,
            ..Default::default()
        };
        let cmd = finalize(1.0, 2.0, &c, &cfg, false, "");
        let (left, right) = (
            cmd.linear_velocity - 0.25 * cmd.angular_velocity,
            cmd.linear_velocity + 0.25 * cmd.angular_velocity,
        );
        assert!(
            left.abs().max(right.abs()) <= 1.0 + 1e-12,
            "{steering:?}: wheels {left:.3} / {right:.3}"
        );
        assert!(
            (cmd.angular_velocity / cmd.linear_velocity - 2.0).abs() < 1e-12,
            "{steering:?}: curvature changed"
        );

        let spin = finalize(0.0, 4.0, &c, &cfg, false, "");
        assert!(
            (spin.angular_velocity - 4.0).abs() < 1e-12,
            "{steering:?}: spinning needs only 1 m/s per wheel"
        );

        let unlimited = RobotConstraints {
            track_width: 0.0,
            ..c.clone()
        };
        let cmd = finalize(1.0, 2.0, &unlimited, &cfg, false, "");
        assert_eq!((cmd.linear_velocity, cmd.angular_velocity), (1.0, 2.0));
    }
}

#[test]
fn the_tracker_limits_how_fast_the_steering_angle_changes() {
    let mut path = line(3.0);
    for i in 1..=30 {
        path.waypoints
            .push(pose(3.0, i as f64 * 0.1, std::f64::consts::FRAC_PI_2));
    }
    for (rate, kind) in [
        (0.4, TrackerKind::PurePursuit),
        (0.4, TrackerKind::Stanley),
        (0.0, TrackerKind::PurePursuit),
    ] {
        let mut t = Tracker::new(kind);
        t.init(RobotConstraints {
            steering_type: SteeringType::Ackermann,
            wheelbase: 0.5,
            max_steering_angle: 0.6,
            min_turning_radius: 0.0,
            max_angular_velocity: 3.0,
            max_steering_rate: rate,
            ..Default::default()
        });
        t.set_path(path.clone());
        let mut s = state(2.2, -0.3, 0.6);
        s.velocity.linear = 1.0;
        let dt = 0.05;
        let mut prev: Option<f64> = None;
        let mut largest: f64 = 0.0;
        for _ in 0..80 {
            let cmd = t.tick(&s, dt, None);
            assert!(cmd.valid, "{kind:?}: {}", cmd.status_message);
            if cmd.linear_velocity.abs() > 1e-6 {
                if let Some(p) = prev {
                    largest = largest.max((cmd.steering_angle - p).abs());
                }
                prev = Some(cmd.steering_angle);
            }
            let yaw = s.pose.rotation.to_euler().yaw;
            s.pose.point.x += cmd.linear_velocity * yaw.cos() * dt;
            s.pose.point.y += cmd.linear_velocity * yaw.sin() * dt;
            s.pose.rotation =
                Quaternion::from_euler(Euler::new(0.0, 0.0, yaw + cmd.angular_velocity * dt));
        }
        if rate > 0.0 {
            assert!(
                largest <= rate * dt + 1e-9,
                "{kind:?}: steering moved {largest:.4} rad in one tick"
            );
        } else {
            assert!(
                largest > 0.4 * dt,
                "{kind:?}: steering never moved faster than the limit it does not have"
            );
        }
    }
}
