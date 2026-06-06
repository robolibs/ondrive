#![allow(clippy::field_reassign_with_default)]

use datapod::{Euler, Point, Pose, Quaternion};
use ondrive::{
    Controller, ControllerConfig, Goal, MppiConfig, MppiFollower, OutputUnits, Path,
    RobotConstraints, RobotState, Velocity,
};

fn pose_at(x: f64, y: f64, yaw: f64) -> Pose {
    Pose {
        point: Point::new(x, y, 0.0),
        rotation: Quaternion::from_euler(Euler::new(0.0, 0.0, yaw)),
    }
}

#[test]
fn mppi_tracks_straight_line_seeded() {
    let mut path = Path::default();
    path.waypoints = (0..30).map(|i| pose_at(i as f64 * 0.5, 0.0, 0.0)).collect();
    let final_wp = *path.waypoints.last().unwrap();

    let mut cfg = MppiConfig::default();
    cfg.num_samples = 300; // keep the test fast
    cfg.horizon_steps = 15;
    cfg.dt = 0.1;
    // Shorter taper than the default — the default 2.0 m would scale the
    // reference velocity below what's needed to cross the 0.4 m tolerance.
    cfg.decel_distance = 0.8;

    let mut follower = MppiFollower::with_seed(cfg, 42);

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

    let mut state = RobotState {
        pose: pose_at(0.0, 0.2, 0.0),
        velocity: Velocity {
            linear: 0.5,
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
    let mut settled_max_cte: f64 = 0.0;
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

        if t > 3.0 && state.pose.point.x < final_wp.point.x - 1.0 {
            settled_max_cte = settled_max_cte.max(follower.get_status().cross_track_error);
        }

        if follower.get_status().goal_reached {
            reached = true;
            break;
        }
    }

    assert!(
        reached,
        "MPPI did not reach goal (t={t:.2}, final=({:.2},{:.2}))",
        state.pose.point.x, state.pose.point.y
    );
    assert!(
        settled_max_cte < 0.8,
        "post-settle CTE too large: {settled_max_cte:.3}"
    );
}
