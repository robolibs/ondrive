//! Shared scaffold for the controller demos: builds a tracker, streams the
//! path, goal, obstacles, robot, plan and status to a rerun viewer at
//! real-time speed, and prints progress.

#![allow(dead_code, clippy::field_reassign_with_default)]

use datapod::{Euler, Point, Pose, Quaternion};
use ondrive::{
    ControllerConfig, Path, RobotConstraints, RobotState, SteeringType, Tracker, TrackerKind,
    WorldConstraints,
};

use crate::viz;

pub fn pose(x: f64, y: f64, yaw: f64) -> Pose {
    Pose {
        point: Point::new(x, y, 0.0),
        rotation: Quaternion::from_euler(Euler::new(0.0, 0.0, yaw)),
    }
}

/// Polyline through `pts` with each waypoint oriented along its segment.
pub fn path_through(pts: &[(f64, f64)]) -> Path {
    let mut p = Path::default();
    for w in pts.windows(2) {
        let h = (w[1].1 - w[0].1).atan2(w[1].0 - w[0].0);
        p.waypoints.push(pose(w[0].0, w[0].1, h));
    }
    if pts.len() >= 2 {
        let a = pts[pts.len() - 2];
        let b = pts[pts.len() - 1];
        p.waypoints.push(pose(b.0, b.1, (b.1 - a.1).atan2(b.0 - a.0)));
    }
    p
}

/// Sampled curve `f(s)` for `s` in `[0, length]`.
pub fn curve(f: impl Fn(f64) -> (f64, f64), length: f64, spacing: f64) -> Path {
    let n = (length / spacing).ceil() as usize;
    let pts: Vec<(f64, f64)> = (0..=n).map(|i| f(i as f64 * spacing)).collect();
    path_through(&pts)
}

pub fn sine(length: f64, amplitude: f64, frequency: f64) -> Path {
    curve(|s| (s, (s * frequency).sin() * amplitude), length, 0.25)
}

pub fn constraints(steering: SteeringType, max_speed: f64) -> RobotConstraints {
    let mut c = RobotConstraints::default();
    c.steering_type = steering;
    c.max_linear_velocity = max_speed;
    c.min_linear_velocity = -0.6 * max_speed;
    c.max_angular_velocity = 2.0;
    c.max_linear_acceleration = 1.5;
    c.max_angular_acceleration = 4.0;
    c.max_steering_angle = 0.6;
    c.min_turning_radius = 0.0;
    c.wheelbase = 0.5;
    c.robot_width = 0.4;
    c.robot_length = 0.6;
    c
}

pub fn config() -> ControllerConfig {
    let mut cfg = ControllerConfig::default();
    cfg.goal_tolerance = 0.3;
    cfg.angular_tolerance = std::f64::consts::PI;
    cfg.lookahead_distance = 1.0;
    cfg.kp_linear = 1.5;
    cfg.kp_angular = 2.0;
    cfg.k_cross_track = 1.5;
    cfg.k_heading = 1.0;
    cfg
}

pub struct Demo {
    pub name: &'static str,
    pub kind: TrackerKind,
    pub path: Path,
    pub constraints: RobotConstraints,
    pub config: ControllerConfig,
    pub start: Pose,
    pub world: Option<WorldConstraints>,
    pub dt: f64,
    pub max_time: f64,
}

pub fn run(demo: Demo) {
    let rec = if std::env::var_os("ONDRIVE_NO_VIZ").is_some() {
        None
    } else {
        rerun::RecordingStreamBuilder::new(format!("ondrive_{}_demo", demo.name))
            .spawn()
            .ok()
    };
    let prefix = demo.name;

    let mut tracker = Tracker::new(demo.kind);
    tracker.set_config(demo.config);
    tracker.init(demo.constraints);
    if let Some(r) = rec.as_ref() {
        viz::show_path(r, &demo.path, &format!("{prefix}/path"), viz::blue());
    }
    tracker.set_path(demo.path.clone());

    let mut state = RobotState {
        pose: demo.start,
        allow_move: true,
        ..Default::default()
    };
    let dt = demo.dt;
    let mut t = 0.0;
    let mut last_print = -1.0;
    let mut step = 0usize;

    while t < demo.max_time {
        let cmd = tracker.tick(&state, dt, demo.world.as_ref());
        if !cmd.valid {
            println!("invalid command: {}", cmd.status_message);
            break;
        }
        let yaw = state.pose.rotation.to_euler().yaw;
        state.pose.point.x += cmd.linear_velocity * yaw.cos() * dt;
        state.pose.point.y += cmd.linear_velocity * yaw.sin() * dt;
        state.pose.rotation =
            Quaternion::from_euler(Euler::new(0.0, 0.0, yaw + cmd.angular_velocity * dt));
        state.velocity.linear = cmd.linear_velocity;
        state.velocity.angular = cmd.angular_velocity;
        t += dt;
        step += 1;

        let status = tracker.get_status();
        if let Some(r) = rec.as_ref() {
            r.set_duration_secs("sim", t);
            viz::show_robot_state(r, &state, &format!("{prefix}/robot"), viz::yellow(), 1.0);
            if let Some(target) = tracker.current_target() {
                let _ = r.log(
                    format!("{prefix}/target"),
                    &rerun::Points3D::new([[target.x as f32, target.y as f32, 0.0]])
                        .with_colors([viz::green()])
                        .with_radii([0.12]),
                );
            }
            let plan = tracker.predicted_trajectory();
            if !plan.is_empty() {
                viz::show_predicted_trajectory(r, &plan, &format!("{prefix}/plan"), viz::magenta());
            }
            if let Some(w) = demo.world.as_ref() {
                viz::show_obstacles(r, w, step, &format!("{prefix}/obstacles"));
            }
            viz::show_controller_status(r, &status, &format!("{prefix}/status"));
            viz::pace(dt);
        }

        if t - last_print >= 0.5 {
            println!(
                "t={:6.2} pos=({:6.2},{:6.2}) yaw={:5.2} v={:.2} w={:.2} steer={:.2} cte={:.3} [{}]",
                t,
                state.pose.point.x,
                state.pose.point.y,
                yaw,
                cmd.linear_velocity,
                cmd.angular_velocity,
                cmd.steering_angle,
                status.cross_track_error,
                status.mode,
            );
            last_print = t;
        }

        if tracker.is_goal_reached() || tracker.is_path_completed() {
            println!("goal reached at t={t:.2}");
            return;
        }
    }
    println!("time budget exhausted at t={t:.2}");
}
