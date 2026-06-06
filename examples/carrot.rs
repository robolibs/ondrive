//! Carrot point-to-point navigation example.

#![allow(clippy::field_reassign_with_default)]

use datapod::{Euler, Point, Pose, Quaternion};
use ondrive::{ControllerConfig, Goal, RobotConstraints, RobotState, Tracker, TrackerKind};

fn main() {
    let waypoints = [
        Point::new(2.0, 0.0, 0.0),
        Point::new(4.0, 2.0, 0.0),
        Point::new(6.0, 2.5, 0.0),
        Point::new(8.0, 4.0, 0.0),
    ];

    let mut tracker = Tracker::new(TrackerKind::Carrot);

    let mut cfg = ControllerConfig::default();
    cfg.kp_linear = 0.8;
    cfg.kp_angular = 2.0;
    cfg.goal_tolerance = 0.2;
    cfg.angular_tolerance = 0.35;
    tracker.set_config(cfg);

    let mut constraints = RobotConstraints::default();
    constraints.max_linear_velocity = 0.4;
    constraints.max_angular_velocity = 1.0;
    tracker.init(constraints);

    let mut state = RobotState::default();
    state.pose = Pose {
        point: Point::new(0.0, 0.0, 0.0),
        rotation: Quaternion::from_euler(Euler::new(0.0, 0.0, 0.0)),
    };
    state.allow_move = true;

    let dt = 0.05;
    let mut t = 0.0;

    for (i, &wp) in waypoints.iter().enumerate() {
        let prev = if i == 0 {
            state.pose.point
        } else {
            waypoints[i - 1]
        };
        let goal_yaw = (wp.y - prev.y).atan2(wp.x - prev.x);
        tracker.set_goal(Goal {
            target_pose: Pose {
                point: wp,
                rotation: Quaternion::from_euler(Euler::new(0.0, 0.0, goal_yaw)),
            },
            tolerance_position: 0.2,
            tolerance_orientation: 0.6,
            ..Default::default()
        });

        for _ in 0..2000 {
            let cmd = tracker.tick(&state, dt, None);
            if !cmd.valid {
                break;
            }
            let yaw = state.pose.rotation.to_euler().yaw;
            state.pose.point.x += cmd.linear_velocity * yaw.cos() * dt;
            state.pose.point.y += cmd.linear_velocity * yaw.sin() * dt;
            let new_yaw = yaw + cmd.angular_velocity * dt;
            state.pose.rotation = Quaternion::from_euler(Euler::new(0.0, 0.0, new_yaw));
            t += dt;
            if tracker.get_status().goal_reached {
                println!("reached ({:.2},{:.2}) at t={:.2}", wp.x, wp.y, t);
                break;
            }
        }
    }
    println!("done: t={:.2}", t);
}
