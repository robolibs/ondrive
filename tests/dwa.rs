//! Dynamic Window Approach tests.

use datapod::{Euler, Point, Pose, Quaternion};
use ondrive::{
    Controller, ControllerConfig, DwaConfig, DwaFollower, GaussianMode, Goal, Obstacle,
    OutputUnits, RobotConstraints, RobotState, WorldConstraints,
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
fn dwa_moves_toward_goal() {
    let mut dwa_cfg = DwaConfig::default();
    dwa_cfg.predict_time = 1.0;
    dwa_cfg.dt = 0.1;
    dwa_cfg.v_samples = 6;
    dwa_cfg.w_samples = 11;

    let mut follower = DwaFollower::with_dwa_config(dwa_cfg);
    let mut base_cfg = ControllerConfig::default();
    base_cfg.goal_tolerance = 0.4;
    base_cfg.angular_tolerance = 1.0;
    base_cfg.output_units = OutputUnits::Physical;
    follower.set_config(base_cfg);

    let c = constraints_default();
    let goal = Goal {
        target_pose: pose_at(10.0, 0.0, 0.0),
        tolerance_position: 0.4,
        tolerance_orientation: 1.0,
        ..Default::default()
    };

    let mut state = RobotState {
        pose: pose_at(0.0, 0.0, 0.0),
        allow_move: true,
        ..Default::default()
    };

    let dt = 0.1;
    let mut reached = false;
    let mut t = 0.0;
    for _ in 0..400 {
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
        t += dt;
        if follower.get_status().goal_reached {
            reached = true;
            break;
        }
    }
    assert!(
        reached,
        "DWA did not reach goal (t={t:.2}, final=({:.2},{:.2}))",
        state.pose.point.x, state.pose.point.y
    );
}

#[test]
fn dwa_avoids_obstacle() {
    let mut follower = DwaFollower::with_dwa_config(DwaConfig::default());
    let mut base_cfg = ControllerConfig::default();
    base_cfg.goal_tolerance = 0.5;
    base_cfg.output_units = OutputUnits::Physical;
    follower.set_config(base_cfg);

    let c = constraints_default();
    let goal = Goal {
        target_pose: pose_at(10.0, 0.0, 0.0),
        tolerance_position: 0.5,
        tolerance_orientation: 2.0,
        ..Default::default()
    };

    let world = WorldConstraints {
        obstacles: vec![Obstacle {
            id: 0,
            radius: 0.3,
            modes: vec![GaussianMode {
                weight: 1.0,
                mean_x: vec![5.0],
                mean_y: vec![0.0],
                std_x: vec![0.1],
                std_y: vec![0.1],
            }],
        }],
        ..Default::default()
    };

    let mut state = RobotState {
        pose: pose_at(0.0, 0.0, 0.0),
        allow_move: true,
        ..Default::default()
    };

    let dt = 0.1;
    let mut min_clearance = f64::MAX;
    for _ in 0..300 {
        let cmd = follower.compute_control(&state, &goal, &c, dt, Some(&world));
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
        let d = ((state.pose.point.x - 5.0).powi(2) + state.pose.point.y.powi(2)).sqrt();
        min_clearance = min_clearance.min(d);
        if follower.get_status().goal_reached || state.pose.point.x > 9.0 {
            break;
        }
    }
    assert!(
        min_clearance > 0.35,
        "DWA got too close to obstacle: min_clearance={min_clearance:.3}"
    );
}
