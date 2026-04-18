//! PID point-to-point navigation example with rerun visualisation.
//!
//! Run a rerun viewer or `rerun --serve` and this example streams its
//! simulation to it. Without a viewer it still runs and prints progress.

#[path = "common/viz.rs"]
mod viz;

use datapod::{Euler, Point, Pose, Quaternion};
use ondrive::{
    ControllerConfig, Goal, Path, RobotConstraints, RobotState, Tracker, TrackerKind,
    smoothen_path,
};

fn make_pose(x: f64, y: f64, yaw: f64) -> Pose {
    Pose {
        point: Point::new(x, y, 0.0),
        rotation: Quaternion::from_euler(Euler::new(0.0, 0.0, yaw)),
    }
}

fn main() {
    let rec = rerun::RecordingStreamBuilder::new("ondrive_pid_demo")
        .spawn()
        .ok();

    // Waypoint sequence — PID drives to each in turn.
    let waypoints = [
        Point::new(1.5, 0.5, 0.0),
        Point::new(3.0, 1.5, 0.0),
        Point::new(4.5, 2.0, 0.0),
        Point::new(6.0, 3.2, 0.0),
        Point::new(7.0, 3.8, 0.0),
        Point::new(8.0, 4.0, 0.0),
    ];

    // Build a Path just for visualisation (after densifying).
    let mut path_viz = Path::default();
    for wp in &waypoints {
        path_viz.waypoints.push(make_pose(wp.x, wp.y, 0.0));
    }
    smoothen_path(&mut path_viz, 0.25);
    if let Some(r) = rec.as_ref() {
        viz::show_path(r, &path_viz, "pid/path", viz::blue());
    }

    let mut tracker = Tracker::new(TrackerKind::Pid);
    let mut cfg = ControllerConfig::default();
    cfg.kp_linear = 2.5;
    cfg.kp_angular = 1.8;
    cfg.kd_angular = 0.2;
    cfg.goal_tolerance = 0.2;
    cfg.angular_tolerance = 0.35;
    tracker.set_config(cfg);

    let mut constraints = RobotConstraints::default();
    constraints.max_linear_velocity = 0.35;
    constraints.max_angular_velocity = 1.0;
    constraints.wheelbase = 0.45;
    tracker.init(constraints);

    let mut state = RobotState::default();
    state.pose = make_pose(0.0, 0.0, 0.0);
    state.allow_move = true;

    let dt = 0.05;
    let mut t = 0.0;
    let mut last_print = -1.0;

    for (i, &wp) in waypoints.iter().enumerate() {
        // Orient each goal along the direction of travel from the previous
        // waypoint so `is_goal_reached` (which requires orientation match)
        // fires naturally on arrival without needing a stop-and-rotate.
        let prev = if i == 0 {
            state.pose.point
        } else {
            waypoints[i - 1]
        };
        let goal_yaw = (wp.y - prev.y).atan2(wp.x - prev.x);
        let goal = Goal {
            target_pose: Pose {
                point: wp,
                rotation: Quaternion::from_euler(Euler::new(0.0, 0.0, goal_yaw)),
            },
            tolerance_position: 0.2,
            tolerance_orientation: 0.6,
            ..Default::default()
        };
        if let Some(r) = rec.as_ref() {
            viz::show_goal(r, &goal, &format!("pid/goal/{i}"), viz::green());
        }
        tracker.set_goal(goal);

        for _ in 0..2000 {
            let cmd = tracker.tick(&state, dt, None);
            if !cmd.valid {
                eprintln!("invalid: {}", cmd.status_message);
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

            if let Some(r) = rec.as_ref() {
                r.set_duration_secs("sim", t);
                viz::show_robot_state(r, &state, "pid/robot", viz::yellow(), 1.0);
                viz::show_controller_status(r, &tracker.get_status(), "pid/status");
            }

            if t - last_print >= 0.5 {
                println!(
                    "t={:5.2} pos=({:5.2},{:5.2}) yaw={:5.2} v={:.3} w={:.3} [{}]",
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

            if tracker.is_goal_reached() {
                println!(
                    "reached waypoint ({:.2},{:.2}) at t={:.2}",
                    wp.x, wp.y, t
                );
                break;
            }
        }
    }

    println!(
        "done: final pos=({:.2},{:.2}), t={:.2}s",
        state.pose.point.x, state.pose.point.y, t
    );
}
