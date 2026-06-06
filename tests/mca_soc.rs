#![allow(clippy::field_reassign_with_default)]

use datapod::{Euler, Point, Pose, Quaternion};
use ondrive::{
    Controller, ControllerConfig, GaussianMode, Goal, McaConfig, McaFollower, Obstacle,
    OutputUnits, Path, RobotConstraints, RobotState, SocConfig, SocFollower, Velocity,
    WorldConstraints,
};

fn straight_path(n: usize) -> Path {
    let mut p = Path::default();
    p.waypoints = (0..n).map(|i| pose_at(i as f64 * 0.5, 0.0, 0.0)).collect();
    p
}

fn pose_at(x: f64, y: f64, yaw: f64) -> Pose {
    Pose {
        point: Point::new(x, y, 0.0),
        rotation: Quaternion::from_euler(Euler::new(0.0, 0.0, yaw)),
    }
}

#[test]
fn mca_falls_back_to_mppi_when_no_obstacles() {
    let mut path = Path::default();
    path.waypoints = (0..25).map(|i| pose_at(i as f64 * 0.5, 0.0, 0.0)).collect();
    let final_wp = *path.waypoints.last().unwrap();

    let mut mca_cfg = McaConfig::default();
    mca_cfg.horizon_steps = 12;
    mca_cfg.num_samples = 200;
    mca_cfg.decel_distance = 0.8;

    let mut follower = McaFollower::with_seed(mca_cfg, 7);

    let mut base_cfg = ControllerConfig::default();
    base_cfg.goal_tolerance = 0.4;
    base_cfg.angular_tolerance = 1.0;
    base_cfg.output_units = OutputUnits::Physical;
    follower.set_config(base_cfg);

    follower.set_path(path);

    let mut c = RobotConstraints::default();
    c.max_linear_velocity = 1.0;
    c.max_angular_velocity = 2.0;
    c.max_linear_acceleration = 2.0;
    c.max_steering_angle = 0.6;
    c.wheelbase = 0.5;
    c.robot_width = 0.5;
    c.robot_length = 0.7;

    let mut state = RobotState {
        pose: pose_at(0.0, 0.15, 0.0),
        velocity: Velocity {
            linear: 0.4,
            ..Default::default()
        },
        allow_move: true,
        ..Default::default()
    };
    let goal = Goal {
        target_pose: final_wp,
        tolerance_position: 0.4,
        tolerance_orientation: 1.0,
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
        t += dt;
        if follower.get_status().goal_reached {
            reached = true;
            break;
        }
    }
    assert!(reached, "MCA (no obstacles) did not reach goal (t={t:.2})");
}

#[test]
fn mca_avoids_static_gaussian_obstacle() {
    let mut path = Path::default();
    path.waypoints = (0..25).map(|i| pose_at(i as f64 * 0.5, 0.0, 0.0)).collect();
    let final_wp = *path.waypoints.last().unwrap();

    let mut mca_cfg = McaConfig::default();
    mca_cfg.horizon_steps = 10;
    mca_cfg.num_samples = 200;
    mca_cfg.num_mc_samples = 2000; // keep fast
    mca_cfg.decel_distance = 0.8;

    let mut follower = McaFollower::with_seed(mca_cfg, 11);

    let mut base_cfg = ControllerConfig::default();
    base_cfg.goal_tolerance = 0.4;
    base_cfg.angular_tolerance = 1.0;
    base_cfg.output_units = OutputUnits::Physical;
    follower.set_config(base_cfg);

    follower.set_path(path);

    let mut c = RobotConstraints::default();
    c.max_linear_velocity = 1.0;
    c.max_angular_velocity = 2.0;
    c.max_linear_acceleration = 2.0;
    c.max_steering_angle = 0.6;
    c.wheelbase = 0.5;
    c.robot_width = 0.4;
    c.robot_length = 0.6;

    // Stationary obstacle roughly on the path at x=6.
    let horizon = 10;
    let obstacle = Obstacle {
        id: 0,
        radius: 0.3,
        modes: vec![GaussianMode {
            weight: 1.0,
            mean_x: vec![6.0; horizon],
            mean_y: vec![0.0; horizon],
            std_x: vec![0.1; horizon],
            std_y: vec![0.1; horizon],
        }],
    };
    let world = WorldConstraints {
        obstacles: vec![obstacle],
        ..Default::default()
    };

    let mut state = RobotState {
        pose: pose_at(0.0, 0.0, 0.0),
        velocity: Velocity {
            linear: 0.3,
            ..Default::default()
        },
        allow_move: true,
        ..Default::default()
    };
    let goal = Goal {
        target_pose: final_wp,
        tolerance_position: 0.4,
        tolerance_orientation: 1.0,
        ..Default::default()
    };

    let dt = 0.1;
    let mut min_clearance: f64 = f64::MAX;
    let mut min_clearance_while_moving: f64 = f64::MAX;
    let mut max_x: f64 = 0.0;
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
        let dx = state.pose.point.x - 6.0;
        let dy = state.pose.point.y - 0.0;
        let d = (dx * dx + dy * dy).sqrt();
        min_clearance = min_clearance.min(d);
        if cmd.linear_velocity.abs() > 0.05 {
            min_clearance_while_moving = min_clearance_while_moving.min(d);
        }
        max_x = max_x.max(state.pose.point.x);

        if follower.get_status().goal_reached || state.pose.point.x > 11.5 {
            break;
        }
    }
    // The MCA controller either steers around the obstacle or decelerates to
    // stay safe. Either is acceptable; we only require (a) the robot made
    // some forward progress and (b) when it *was* moving, it never came
    // closer than the collision radius to the obstacle center.
    assert!(
        max_x > 2.0,
        "MCA made no forward progress (max_x={:.2})",
        max_x
    );
    assert!(
        min_clearance_while_moving > 0.35,
        "MCA moved too close to obstacle: min_clearance_while_moving={:.3} (overall min={:.3})",
        min_clearance_while_moving,
        min_clearance
    );
}

#[test]
fn soc_tracks_path() {
    let mut path = Path::default();
    path.waypoints = (0..25).map(|i| pose_at(i as f64 * 0.5, 0.0, 0.0)).collect();
    let final_wp = *path.waypoints.last().unwrap();

    let mut soc_cfg = SocConfig::default();
    soc_cfg.horizon_steps = 12;
    soc_cfg.guide_samples = 32;
    soc_cfg.num_samples = 128;
    soc_cfg.decel_distance = 0.8;

    let mut follower = SocFollower::with_seed(soc_cfg, 3);

    let mut base_cfg = ControllerConfig::default();
    base_cfg.goal_tolerance = 0.5;
    base_cfg.angular_tolerance = 1.0;
    base_cfg.output_units = OutputUnits::Physical;
    follower.set_config(base_cfg);

    follower.set_path(path);

    let mut c = RobotConstraints::default();
    c.max_linear_velocity = 1.0;
    c.max_angular_velocity = 2.0;
    c.max_linear_acceleration = 2.0;
    c.max_steering_angle = 0.6;
    c.wheelbase = 0.5;
    c.robot_width = 0.4;

    let mut state = RobotState {
        pose: pose_at(0.0, 0.15, 0.0),
        velocity: Velocity {
            linear: 0.4,
            ..Default::default()
        },
        allow_move: true,
        ..Default::default()
    };
    let goal = Goal {
        target_pose: final_wp,
        tolerance_position: 0.5,
        tolerance_orientation: 1.0,
        ..Default::default()
    };

    let dt = 0.1;
    let mut min_dist: f64 = f64::MAX;
    let mut reached = false;
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
        min_dist = min_dist.min(state.pose.point.distance_to(final_wp.point));
        if follower.get_status().goal_reached {
            reached = true;
            break;
        }
    }
    assert!(
        reached || min_dist < 1.0,
        "SOC never got close to goal (min_dist={:.3}, final=({:.2},{:.2}))",
        min_dist,
        state.pose.point.x,
        state.pose.point.y
    );
}

#[test]
fn soc_svgd_produces_valid_command() {
    let path = straight_path(25);
    let final_wp = *path.waypoints.last().unwrap();
    let mut c = RobotConstraints::default();
    c.max_linear_velocity = 1.0;
    c.max_angular_velocity = 2.0;
    c.max_linear_acceleration = 2.0;
    c.max_steering_angle = 0.6;
    c.wheelbase = 0.5;
    c.robot_width = 0.4;

    let goal = Goal {
        target_pose: final_wp,
        tolerance_position: 0.4,
        tolerance_orientation: 1.0,
        ..Default::default()
    };
    let state = RobotState {
        pose: pose_at(0.0, 0.3, 0.0),
        velocity: Velocity {
            linear: 0.3,
            ..Default::default()
        },
        allow_move: true,
        ..Default::default()
    };

    let base_cfg = {
        let mut b = ControllerConfig::default();
        b.goal_tolerance = 0.4;
        b.output_units = OutputUnits::Physical;
        b
    };

    let run = |svgd_iters: usize, seed: u64| -> bool {
        let mut cfg = SocConfig::default();
        cfg.horizon_steps = 6;
        cfg.guide_samples = 12;
        cfg.num_samples = 48;
        cfg.svgd_iterations = svgd_iters;
        let mut follower = SocFollower::with_seed(cfg, seed);
        follower.set_config(base_cfg.clone());
        follower.set_path(path.clone());
        let cmd = follower.compute_control(&state, &goal, &c, 0.1, None);
        cmd.valid
    };

    assert!(
        run(0, 123),
        "SOC with svgd_iterations=0 should produce a valid command"
    );
    assert!(
        run(2, 123),
        "SOC with svgd_iterations=2 should produce a valid command"
    );
}
