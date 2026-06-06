#![allow(clippy::field_reassign_with_default)]

use datapod::{Euler, Point, Pose, Quaternion};
use ondrive::{ControllerConfig, Goal, RobotConstraints, RobotState, Tracker, TrackerKind};

fn simulate_until_goal(
    tracker: &mut Tracker,
    start: Pose,
    goal: Pose,
    max_ticks: usize,
    dt: f64,
) -> (bool, RobotState, f64) {
    let mut state = RobotState {
        pose: start,
        allow_move: true,
        ..Default::default()
    };

    tracker.set_goal(Goal {
        target_pose: goal,
        tolerance_position: 0.1,
        tolerance_orientation: 0.2,
        ..Default::default()
    });

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
        if tracker.get_status().goal_reached {
            return (true, state, t);
        }
    }
    (false, state, t)
}

#[test]
fn pid_reaches_forward_goal() {
    let mut tracker = Tracker::new(TrackerKind::Pid);
    let mut cfg = ControllerConfig::default();
    cfg.kp_linear = 2.0;
    cfg.kp_angular = 1.5;
    cfg.goal_tolerance = 0.1;
    cfg.angular_tolerance = 0.2;
    tracker.set_config(cfg);

    let mut c = RobotConstraints::default();
    c.max_linear_velocity = 0.5;
    c.max_angular_velocity = 1.5;
    tracker.init(c);

    let start = Pose {
        point: Point::new(0.0, 0.0, 0.0),
        rotation: Quaternion::from_euler(Euler::new(0.0, 0.0, 0.0)),
    };
    let goal = Pose {
        point: Point::new(5.0, 0.0, 0.0),
        rotation: Quaternion::from_euler(Euler::new(0.0, 0.0, 0.0)),
    };

    let (reached, end, t) = simulate_until_goal(&mut tracker, start, goal, 2000, 0.05);
    assert!(reached, "PID did not reach goal within budget (t={t:.2})");
    assert!(
        (end.pose.point.x - goal.point.x).hypot(end.pose.point.y - goal.point.y) < 0.2,
        "final pos too far from goal: {:?}",
        end.pose.point
    );
}

#[test]
fn pid_reaches_diagonal_goal() {
    let mut tracker = Tracker::new(TrackerKind::Pid);
    let mut cfg = ControllerConfig::default();
    cfg.kp_linear = 2.0;
    cfg.kp_angular = 2.0;
    cfg.goal_tolerance = 0.15;
    cfg.angular_tolerance = 0.3;
    tracker.set_config(cfg);

    let mut c = RobotConstraints::default();
    c.max_linear_velocity = 0.5;
    c.max_angular_velocity = 2.0;
    tracker.init(c);

    let start = Pose {
        point: Point::new(0.0, 0.0, 0.0),
        rotation: Quaternion::from_euler(Euler::new(0.0, 0.0, 0.0)),
    };
    // Goal points in the direction of travel so orientation tolerance is
    // satisfied on arrival (PID will drive straight at the target and end
    // up facing it).
    let goal = Pose {
        point: Point::new(3.0, 3.0, 0.0),
        rotation: Quaternion::from_euler(Euler::new(0.0, 0.0, std::f64::consts::FRAC_PI_4)),
    };

    let (reached, _, t) = simulate_until_goal(&mut tracker, start, goal, 3000, 0.05);
    assert!(
        reached,
        "PID did not reach diagonal goal within budget (t={t:.2})"
    );
}
