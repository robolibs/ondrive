//! Timed Elastic Band tests.

#![allow(clippy::field_reassign_with_default)]

use datapod::{Euler, Point, Pose, Quaternion};
use ondrive::{
    Controller, ControllerConfig, Goal, OutputUnits, Path, RobotConstraints, RobotState, TebConfig,
    TebFollower,
};

fn pose_at(x: f64, y: f64, yaw: f64) -> Pose {
    Pose {
        point: Point::new(x, y, 0.0),
        rotation: Quaternion::from_euler(Euler::new(0.0, 0.0, yaw)),
    }
}

fn constraints_default() -> RobotConstraints {
    let mut c = RobotConstraints::default();
    c.max_linear_velocity = 1.0;
    c.max_angular_velocity = 2.0;
    c.max_linear_acceleration = 2.0;
    c.max_steering_angle = 0.6;
    c.wheelbase = 0.5;
    c.robot_width = 0.4;
    c.robot_length = 0.6;
    c
}

#[test]
fn teb_tracks_path() {
    let mut path = Path::default();
    path.waypoints = (0..25).map(|i| pose_at(i as f64 * 0.5, 0.0, 0.0)).collect();
    let final_wp = *path.waypoints.last().unwrap();

    let mut teb_cfg = TebConfig::default();
    teb_cfg.iterations = 6;
    teb_cfg.n_poses = 5;

    let mut follower = TebFollower::with_teb_config(teb_cfg);
    let mut base_cfg = ControllerConfig::default();
    base_cfg.goal_tolerance = 0.6;
    base_cfg.angular_tolerance = 2.0;
    base_cfg.output_units = OutputUnits::Physical;
    follower.set_config(base_cfg);
    follower.set_path(path);

    let c = constraints_default();
    let goal = Goal {
        target_pose: final_wp,
        tolerance_position: 0.6,
        tolerance_orientation: 2.0,
        ..Default::default()
    };

    let mut state = RobotState {
        pose: pose_at(0.0, 0.2, 0.0),
        allow_move: true,
        ..Default::default()
    };

    let dt = 0.1;
    let mut max_x: f64 = 0.0;
    let mut reached = false;
    for _ in 0..600 {
        let cmd = follower.compute_control(&state, &goal, &c, dt, None);
        if !cmd.valid {
            break;
        }
        let yaw = state.pose.rotation.to_euler().yaw;
        state.pose.point.x += cmd.linear_velocity * yaw.cos() * dt;
        state.pose.point.y += cmd.linear_velocity * yaw.sin() * dt;
        let new_yaw = yaw + cmd.angular_velocity * dt;
        state.pose.rotation = Quaternion::from_euler(Euler::new(0.0, 0.0, new_yaw));
        state.velocity.linear = cmd.linear_velocity;
        state.velocity.angular = cmd.angular_velocity;
        max_x = max_x.max(state.pose.point.x);
        if follower.get_status().goal_reached {
            reached = true;
            break;
        }
    }
    assert!(
        reached || max_x > 8.0,
        "TEB made no meaningful progress (max_x={:.2})",
        max_x
    );
}
