//! Pure Pursuit path-following example.

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
    path.waypoints = (0..40)
        .map(|i| {
            let x = i as f64 * 0.5;
            let y = (x * 0.35).sin() * 1.2;
            pose_at(x, y)
        })
        .collect();

    let final_wp = *path.waypoints.last().unwrap();

    let mut tracker = Tracker::new(TrackerKind::PurePursuit);
    let mut cfg = ControllerConfig::default();
    cfg.lookahead_distance = 1.2;
    cfg.goal_tolerance = 0.3;
    cfg.angular_tolerance = 1.0;
    cfg.output_units = OutputUnits::Physical;
    tracker.set_config(cfg);

    let mut constraints = RobotConstraints::default();
    constraints.max_linear_velocity = 1.0;
    constraints.max_angular_velocity = 1.5;
    constraints.wheelbase = 0.5;
    tracker.init(constraints);

    tracker.set_path(path);
    tracker.set_goal(Goal {
        target_pose: final_wp,
        tolerance_position: 0.3,
        tolerance_orientation: 1.0,
        ..Default::default()
    });

    let mut state = RobotState {
        pose: pose_at(0.0, 0.0),
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
        t += dt;

        if t - last_print >= 0.5 {
            println!(
                "t={:5.2}  pos=({:5.2},{:5.2})  yaw={:5.2}  v={:.2}  w={:.2}  idx={}",
                t,
                state.pose.point.x,
                state.pose.point.y,
                yaw,
                cmd.linear_velocity,
                cmd.angular_velocity,
                tracker.get_status().mode,
            );
            last_print = t;
        }

        if tracker.get_status().goal_reached {
            println!("goal reached at t={:.2}", t);
            break;
        }
    }
}
