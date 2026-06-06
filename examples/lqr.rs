//! LQR path-following example.

#![allow(clippy::field_reassign_with_default)]

use datapod::{Euler, Point, Pose, Quaternion};
use ondrive::{
    ControllerConfig, Goal, OutputUnits, Path, RobotConstraints, RobotState, Tracker, TrackerKind,
};

fn pose_at(x: f64, y: f64) -> Pose {
    Pose {
        point: Point::new(x, y, 0.0),
        rotation: Quaternion::from_euler(Euler::new(0.0, 0.0, 0.0)),
    }
}

fn main() {
    let mut path = Path::default();
    path.waypoints = (0..50)
        .map(|i| {
            let x = i as f64 * 0.4;
            let y = (x * 0.4).sin() * 1.0;
            pose_at(x, y)
        })
        .collect();

    let final_wp = *path.waypoints.last().unwrap();

    let mut tracker = Tracker::new(TrackerKind::Lqr);
    let mut cfg = ControllerConfig::default();
    cfg.goal_tolerance = 0.3;
    cfg.angular_tolerance = 1.0;
    cfg.output_units = OutputUnits::Physical;
    tracker.set_config(cfg);

    let mut c = RobotConstraints::default();
    c.max_linear_velocity = 0.6;
    c.max_angular_velocity = 2.0;
    c.max_steering_angle = 0.6;
    c.wheelbase = 0.5;
    tracker.init(c);

    tracker.set_path(path);
    tracker.set_goal(Goal {
        target_pose: final_wp,
        tolerance_position: 0.3,
        tolerance_orientation: 1.0,
        ..Default::default()
    });

    let mut state = RobotState {
        pose: pose_at(0.0, 0.1),
        allow_move: true,
        ..Default::default()
    };

    let dt = 0.05;
    let mut t = 0.0;
    let mut last_print = -1.0;

    for _ in 0..4000 {
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

        if t - last_print >= 0.5 {
            let s = tracker.get_status();
            println!(
                "t={:5.2} pos=({:5.2},{:5.2}) yaw={:5.2} cte={:6.3} hdg_err={:5.2}",
                t,
                state.pose.point.x,
                state.pose.point.y,
                yaw,
                s.cross_track_error,
                s.heading_error
            );
            last_print = t;
        }
        if tracker.get_status().goal_reached {
            println!("goal reached at t={:.2}", t);
            return;
        }
    }
    println!("budget exhausted at t={:.2}", t);
}
