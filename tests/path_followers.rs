use datapod::{Euler, Point, Pose, Quaternion};
use ondrive::{
    ControllerConfig, Goal, OutputUnits, Path, RobotConstraints, RobotState, Tracker, TrackerKind,
};

fn pose(x: f64, y: f64, yaw: f64) -> Pose {
    Pose {
        point: Point::new(x, y, 0.0),
        rotation: Quaternion::from_euler(Euler::new(0.0, 0.0, yaw)),
    }
}

fn straight_path(n: usize) -> Path {
    let mut p = Path::default();
    p.waypoints = (0..n).map(|i| pose(i as f64 * 0.5, 0.0, 0.0)).collect();
    p
}

fn sine_path(n: usize) -> Path {
    let mut p = Path::default();
    p.waypoints = (0..n)
        .map(|i| {
            let x = i as f64 * 0.5;
            pose(x, (x * 0.3).sin() * 0.8, 0.0)
        })
        .collect();
    p
}

fn simulate(
    tracker: &mut Tracker,
    start: Pose,
    max_ticks: usize,
    dt: f64,
) -> (bool, Pose, f64, f64) {
    let mut state = RobotState {
        pose: start,
        allow_move: true,
        ..Default::default()
    };
    let mut max_cte: f64 = 0.0;
    let mut t = 0.0;

    for _ in 0..max_ticks {
        let cmd = tracker.tick(&state, dt, None);
        if !cmd.valid {
            break;
        }
        let yaw = state.pose.rotation.to_euler().yaw;
        state.pose.point.x += cmd.linear_velocity * yaw.cos() * dt;
        state.pose.point.y += cmd.linear_velocity * yaw.sin() * dt;
        let new_yaw = yaw + cmd.angular_velocity * dt;
        state.pose.rotation = Quaternion::from_euler(Euler::new(0.0, 0.0, new_yaw));
        t += dt;
        max_cte = max_cte.max(tracker.get_status().cross_track_error.abs());
        if tracker.get_status().goal_reached {
            return (true, state.pose, t, max_cte);
        }
    }
    (false, state.pose, t, max_cte)
}

#[test]
fn pure_pursuit_follows_straight_line() {
    let path = straight_path(30);
    let final_wp = *path.waypoints.last().unwrap();

    let mut tracker = Tracker::new(TrackerKind::PurePursuit);
    let mut cfg = ControllerConfig::default();
    cfg.lookahead_distance = 1.0;
    cfg.goal_tolerance = 0.3;
    cfg.angular_tolerance = 1.0;
    cfg.output_units = OutputUnits::Physical;
    tracker.set_config(cfg);

    let mut c = RobotConstraints::default();
    c.max_linear_velocity = 1.0;
    c.max_angular_velocity = 2.0;
    c.wheelbase = 0.5;
    tracker.init(c);

    tracker.set_path(path);
    tracker.set_goal(Goal {
        target_pose: final_wp,
        tolerance_position: 0.3,
        tolerance_orientation: 1.0,
        ..Default::default()
    });

    let (reached, _, t, max_cte) = simulate(&mut tracker, pose(0.0, 0.0, 0.0), 3000, 0.05);
    assert!(reached, "pure pursuit did not reach final waypoint (t={t:.2})");
    assert!(max_cte < 0.5, "cross-track error too large: {max_cte:.3}");
}

#[test]
fn stanley_follows_sine_path() {
    let path = sine_path(40);
    let final_wp = *path.waypoints.last().unwrap();

    let mut tracker = Tracker::new(TrackerKind::Stanley);
    let mut cfg = ControllerConfig::default();
    cfg.k_cross_track = 1.2;
    cfg.goal_tolerance = 0.3;
    cfg.angular_tolerance = 1.0;
    tracker.set_config(cfg);

    let mut c = RobotConstraints::default();
    c.max_linear_velocity = 0.5;
    c.max_angular_velocity = 2.0;
    c.wheelbase = 0.5;
    tracker.init(c);

    tracker.set_path(path);
    tracker.set_goal(Goal {
        target_pose: final_wp,
        tolerance_position: 0.3,
        tolerance_orientation: 1.0,
        ..Default::default()
    });

    let (reached, _, t, max_cte) = simulate(&mut tracker, pose(0.0, 0.2, 0.0), 5000, 0.05);
    assert!(reached, "stanley did not reach final waypoint (t={t:.2})");
    assert!(max_cte < 1.2, "cross-track error too large: {max_cte:.3}");
}
