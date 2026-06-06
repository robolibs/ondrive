#![allow(clippy::field_reassign_with_default)]

use datapod::{Euler, Point, Pose, Quaternion};
use ondrive::{ControllerConfig, Goal, RobotConstraints, RobotState, Tracker, TrackerKind};

#[test]
fn carrot_reaches_forward_goal() {
    let mut tracker = Tracker::new(TrackerKind::Carrot);
    let mut cfg = ControllerConfig::default();
    cfg.kp_linear = 0.8;
    cfg.kp_angular = 2.0;
    cfg.goal_tolerance = 0.2;
    cfg.angular_tolerance = 0.35;
    tracker.set_config(cfg);

    let mut c = RobotConstraints::default();
    c.max_linear_velocity = 0.6;
    c.max_angular_velocity = 1.5;
    tracker.init(c);

    tracker.set_goal(Goal {
        target_pose: Pose {
            point: Point::new(4.0, 1.0, 0.0),
            rotation: Quaternion::default(),
        },
        tolerance_position: 0.2,
        tolerance_orientation: 0.35,
        ..Default::default()
    });

    let mut state = RobotState {
        pose: Pose {
            point: Point::new(0.0, 0.0, 0.0),
            rotation: Quaternion::from_euler(Euler::new(0.0, 0.0, 0.0)),
        },
        allow_move: true,
        ..Default::default()
    };

    let dt = 0.05;
    let mut reached = false;
    for _ in 0..3000 {
        let cmd = tracker.tick(&state, dt, None);
        if !cmd.valid {
            break;
        }
        let yaw = state.pose.rotation.to_euler().yaw;
        state.pose.point.x += cmd.linear_velocity * yaw.cos() * dt;
        state.pose.point.y += cmd.linear_velocity * yaw.sin() * dt;
        let new_yaw = yaw + cmd.angular_velocity * dt;
        state.pose.rotation = Quaternion::from_euler(Euler::new(0.0, 0.0, new_yaw));
        if tracker.get_status().goal_reached {
            reached = true;
            break;
        }
    }
    assert!(
        reached,
        "Carrot did not reach goal; ended at {:?}",
        state.pose.point
    );
}
