#![allow(clippy::field_reassign_with_default)]

use datapod::{Euler, Point, Pose, Quaternion};
use ondrive::{
    ControllerConfig, Goal, OutputUnits, Path, RobotConstraints, RobotState, Tracker, TrackerKind,
};

fn pose_at(x: f64, y: f64, yaw: f64) -> Pose {
    Pose {
        point: Point::new(x, y, 0.0),
        rotation: Quaternion::from_euler(Euler::new(0.0, 0.0, yaw)),
    }
}

#[test]
fn mpc_tracks_and_stops_at_end() {
    let mut path = Path::default();
    path.waypoints = (0..30).map(|i| pose_at(i as f64 * 0.5, 0.0, 0.0)).collect();
    let final_wp = *path.waypoints.last().unwrap();

    let mut tracker = Tracker::new(TrackerKind::Mpc);
    let mut cfg = ControllerConfig::default();
    cfg.goal_tolerance = 0.4;
    cfg.angular_tolerance = 1.0;
    cfg.output_units = OutputUnits::Physical;
    tracker.set_config(cfg);

    let mut c = RobotConstraints::default();
    c.max_linear_velocity = 1.0;
    c.max_angular_velocity = 2.0;
    c.max_linear_acceleration = 2.0;
    c.max_steering_angle = 0.6;
    c.wheelbase = 0.5;
    tracker.init(c);

    tracker.set_path(path);
    tracker.set_goal(Goal {
        target_pose: final_wp,
        tolerance_position: 0.4,
        tolerance_orientation: 1.0,
        ..Default::default()
    });

    let mut state = RobotState {
        pose: pose_at(0.0, 0.2, 0.0),
        velocity: ondrive::Velocity {
            linear: 0.5,
            ..Default::default()
        },
        allow_move: true,
        ..Default::default()
    };

    let dt = 0.1;
    let mut reached = false;
    let mut settled_max_cte: f64 = 0.0;
    let mut t = 0.0;

    for _ in 0..500 {
        let cmd = tracker.tick(&state, dt, None);
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
            settled_max_cte = settled_max_cte.max(tracker.get_status().cross_track_error);
        }

        if tracker.get_status().goal_reached {
            reached = true;
            break;
        }
    }

    assert!(
        reached,
        "MPC did not reach goal (t={t:.2}, final_pos=({:.2},{:.2}))",
        state.pose.point.x, state.pose.point.y
    );
    assert!(
        settled_max_cte < 0.6,
        "post-settle CTE too large: {settled_max_cte:.3}"
    );
    assert!(
        state.pose.point.x < final_wp.point.x + 0.5,
        "MPC overshot the final waypoint by more than 0.5 m: final_x={:.3}",
        state.pose.point.x
    );
}
