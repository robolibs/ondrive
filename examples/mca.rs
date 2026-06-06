//! MCA (risk-aware MPPI) example with a single static Gaussian obstacle.
//!
//! Uses the `McaFollower` directly so we can tune the Monte Carlo sampling
//! budget for the demo (the default 20 000 MC samples per horizon step is
//! too heavy for an interactive run).

#![allow(clippy::field_reassign_with_default)]

use datapod::{Euler, Point, Pose, Quaternion};
use ondrive::{
    Controller, ControllerConfig, GaussianMode, Goal, McaConfig, McaFollower, Obstacle,
    OutputUnits, Path, RobotConstraints, RobotState, WorldConstraints,
};

fn pose_at(x: f64, y: f64) -> Pose {
    Pose {
        point: Point::new(x, y, 0.0),
        rotation: Quaternion::from_euler(Euler::new(0.0, 0.0, 0.0)),
    }
}

fn main() {
    let mut path = Path::default();
    path.waypoints = (0..25).map(|i| pose_at(i as f64 * 0.5, 0.0)).collect();
    let final_wp = *path.waypoints.last().unwrap();

    let mut mca_cfg = McaConfig::default();
    mca_cfg.horizon_steps = 10;
    mca_cfg.num_samples = 200;
    mca_cfg.num_mc_samples = 2000;
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

    // Single Gaussian obstacle at (6, 0) across the full horizon.
    let horizon = 10;
    let world = WorldConstraints {
        obstacles: vec![Obstacle {
            id: 0,
            radius: 0.3,
            modes: vec![GaussianMode {
                weight: 1.0,
                mean_x: vec![6.0; horizon],
                mean_y: vec![0.0; horizon],
                std_x: vec![0.1; horizon],
                std_y: vec![0.1; horizon],
            }],
        }],
        ..Default::default()
    };

    let mut state = RobotState {
        pose: pose_at(0.0, 0.0),
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
    let mut t = 0.0;
    let mut last_print = -1.0;
    let mut min_clearance: f64 = f64::MAX;

    for _ in 0..200 {
        let cmd = follower.compute_control(&state, &goal, &c, dt, Some(&world));
        if !cmd.valid {
            println!("invalid: {}", cmd.status_message);
            break;
        }
        let yaw = state.pose.rotation.to_euler().yaw;
        state.pose.point.x += cmd.linear_velocity * yaw.cos() * dt;
        state.pose.point.y += cmd.linear_velocity * yaw.sin() * dt;
        let new_yaw = yaw + cmd.angular_velocity * dt;
        state.pose.rotation = Quaternion::from_euler(Euler::new(0.0, 0.0, new_yaw));
        state.velocity.linear = cmd.linear_velocity;
        t += dt;

        let clearance = ((state.pose.point.x - 6.0).powi(2) + state.pose.point.y.powi(2)).sqrt();
        min_clearance = min_clearance.min(clearance);

        if t - last_print >= 0.5 {
            println!(
                "t={:5.2} pos=({:5.2},{:5.2}) v={:.2} w={:.2} clearance={:.3}",
                t,
                state.pose.point.x,
                state.pose.point.y,
                cmd.linear_velocity,
                cmd.angular_velocity,
                clearance
            );
            last_print = t;
        }
        if follower.get_status().goal_reached || state.pose.point.x > final_wp.point.x + 1.0 {
            break;
        }
    }
    println!(
        "done: t={:.2}, final=({:.2},{:.2}), min_clearance={:.3}",
        t, state.pose.point.x, state.pose.point.y, min_clearance
    );
}
