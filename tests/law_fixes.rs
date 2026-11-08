//! Control laws checked against their published forms: reverse driving,
//! holonomic frames and what happens when a reference stops.

#![allow(clippy::field_reassign_with_default)]

use datapod::{Euler, Point, Pose, Quaternion};
use ondrive::{
    ControllerConfig, OutputUnits, RobotConstraints, RobotState, SteeringType, Tracker,
    TrackerKind, Trajectory, VelocityCommand,
};
use std::f64::consts::PI;

fn pose(x: f64, y: f64, yaw: f64) -> Pose {
    Pose {
        point: Point::new(x, y, 0.0),
        rotation: Quaternion::from_euler(Euler::new(0.0, 0.0, yaw)),
    }
}

fn constraints(steering: SteeringType) -> RobotConstraints {
    let mut c = RobotConstraints::default();
    c.steering_type = steering;
    c.max_linear_velocity = 1.0;
    c.min_linear_velocity = -1.0;
    c.max_angular_velocity = 2.0;
    c.max_linear_acceleration = 2.0;
    c.max_steering_angle = 0.6;
    c.min_turning_radius = 0.0;
    c.wheelbase = 0.5;
    c
}

fn config(reverse: bool) -> ControllerConfig {
    let mut cfg = ControllerConfig::default();
    cfg.output_units = OutputUnits::Physical;
    cfg.goal_tolerance = 0.05;
    cfg.angular_tolerance = 0.1;
    cfg.allow_reverse = reverse;
    cfg
}

fn step(state: &mut RobotState, cmd: &VelocityCommand, dt: f64) {
    let yaw = state.pose.rotation.to_euler().yaw;
    let (s, c) = yaw.sin_cos();
    state.pose.point.x += (cmd.linear_velocity * c - cmd.lateral_velocity * s) * dt;
    state.pose.point.y += (cmd.linear_velocity * s + cmd.lateral_velocity * c) * dt;
    state.pose.rotation =
        Quaternion::from_euler(Euler::new(0.0, 0.0, yaw + cmd.angular_velocity * dt));
}

/// Straight trajectory along x from `x0` to `x1` at `speed` (negative:
/// driven backwards, the robot facing +x throughout).
fn line_trajectory(x0: f64, x1: f64, speed: f64) -> Trajectory {
    let mut t = Trajectory::default();
    let n = 40;
    let duration = (x1 - x0).abs() / speed.abs();
    for i in 0..=n {
        let a = i as f64 / n as f64;
        t.poses.push(pose(x0 + a * (x1 - x0), 0.0, 0.0));
        t.times.push(a * duration);
        t.speeds.push(speed);
    }
    t
}

/// Drives `ticks` ticks; returns the final state and the largest |y| seen.
fn run(t: &mut Tracker, start: Pose, ticks: usize) -> (RobotState, f64) {
    let mut state = RobotState {
        pose: start,
        allow_move: true,
        allow_reverse: true,
        ..Default::default()
    };
    let mut widest: f64 = 0.0;
    for _ in 0..ticks {
        let cmd = t.tick(&state, 0.05, None);
        assert!(cmd.valid, "{}", cmd.status_message);
        step(&mut state, &cmd, 0.05);
        widest = widest.max(state.pose.point.y.abs());
    }
    (state, widest)
}

#[test]
fn kanayama_corrects_a_lateral_offset_while_reversing() {
    let mut t = Tracker::new(TrackerKind::Kanayama);
    t.set_config(config(true));
    t.init(constraints(SteeringType::Differential));
    t.set_trajectory(line_trajectory(6.0, 0.0, -0.5));
    let (end, _) = run(&mut t, pose(6.0, 0.3, 0.0), 200);
    assert!(
        end.pose.point.y.abs() < 0.05,
        "lateral offset {:.3} after reversing",
        end.pose.point.y
    );
    assert!(
        end.pose.point.x < 2.0,
        "only reached x = {:.2}",
        end.pose.point.x
    );
}

#[test]
fn holonomic_kanayama_tracks_with_its_heading_off_the_path() {
    let mut t = Tracker::new(TrackerKind::Kanayama);
    t.set_config(config(false));
    let mut c = constraints(SteeringType::Holonomic);
    c.max_angular_velocity = 0.05;
    t.init(c);
    t.set_trajectory(line_trajectory(0.0, 8.0, 0.5));
    let (end, widest) = run(&mut t, pose(0.0, 0.0, 0.6), 160);
    assert!(widest < 0.05, "drifted {widest:.3} off the line");
    assert!(
        end.pose.point.x > 3.0,
        "only reached x = {:.2}",
        end.pose.point.x
    );
}

#[test]
fn kanayama_settles_on_the_final_pose_after_the_trajectory_ends() {
    let mut t = Tracker::new(TrackerKind::Kanayama);
    t.set_config(config(false));
    t.init(constraints(SteeringType::Differential));
    t.set_trajectory(line_trajectory(0.0, 0.25, 0.5));
    let (end, _) = run(&mut t, pose(0.0, 0.4, 0.0), 600);
    let d = end.pose.point.distance_to_2d(Point::new(0.25, 0.0, 0.0));
    let yaw = end.pose.rotation.to_euler().yaw;
    assert!(d < 0.1, "stopped {d:.3} from the final pose");
    assert!(
        yaw.abs() < 0.15 || (yaw.abs() - PI).abs() > 3.0,
        "final heading {yaw:.3}"
    );
}

#[test]
fn holonomic_path_followers_work_facing_backwards() {
    for kind in [TrackerKind::Stanley, TrackerKind::Lqr, TrackerKind::Flc] {
        let mut t = Tracker::new(kind);
        t.set_config(config(false));
        let mut c = constraints(SteeringType::Holonomic);
        c.max_angular_velocity = 0.05;
        t.init(c);
        let mut path = ondrive::Path::default();
        for i in 0..=100 {
            path.waypoints.push(pose(i as f64 * 0.1, 0.0, 0.0));
        }
        t.set_path(path);
        let (end, _) = run(&mut t, pose(0.5, 0.3, PI), 80);
        assert!(
            end.pose.point.x > 1.5,
            "{kind:?}: went to x = {:.2}",
            end.pose.point.x
        );
        assert!(
            end.pose.point.y.abs() < 0.1,
            "{kind:?}: lateral {:.3}",
            end.pose.point.y
        );
    }
}

#[test]
fn pid_with_an_integral_term_still_slows_into_the_goal() {
    let mut t = Tracker::new(TrackerKind::Pid);
    let mut cfg = config(false);
    cfg.ki_linear = 0.5;
    cfg.goal_tolerance = 0.1;
    cfg.angular_tolerance = PI;
    t.set_config(cfg);
    t.init(constraints(SteeringType::Differential));
    t.set_goal(ondrive::Goal {
        target_pose: pose(5.0, 0.0, 0.0),
        ..Default::default()
    });
    let mut state = RobotState {
        pose: pose(0.0, 0.0, 0.0),
        allow_move: true,
        ..Default::default()
    };
    let mut arrival_speed = None;
    for _ in 0..600 {
        let cmd = t.tick(&state, 0.05, None);
        let d = state.pose.point.distance_to_2d(Point::new(5.0, 0.0, 0.0));
        if d < 0.3 && arrival_speed.is_none() {
            arrival_speed = Some(cmd.linear_velocity);
        }
        if t.is_goal_reached() {
            break;
        }
        step(&mut state, &cmd, 0.05);
    }
    let v = arrival_speed.expect("never got within 0.3 m");
    assert!(v < 0.5, "still at {v:.2} m/s 0.3 m from the goal");
    assert!(t.is_goal_reached(), "goal not reached");
}

#[test]
fn pose_reach_starts_each_goal_with_a_fresh_refinement_budget() {
    use ondrive::{Controller, Goal, PoseReachFollower};
    let mut ctrl = PoseReachFollower::new();
    ctrl.set_config(config(true));
    let c = constraints(SteeringType::Ackermann);
    let goal = |x: f64, yaw: f64| Goal {
        target_pose: pose(x, 0.0, yaw),
        tolerance_position: 0.3,
        tolerance_orientation: 0.1,
        ..Default::default()
    };
    // Goal A with a heading tolerance too tight to meet on arrival.
    let mut state = RobotState {
        pose: pose(0.0, 0.3, 0.0),
        allow_move: true,
        allow_reverse: true,
        ..Default::default()
    };
    let a = Goal {
        tolerance_orientation: 0.005,
        ..goal(3.0, 0.0)
    };
    for _ in 0..2000 {
        let cmd = ctrl.compute_control(&state, &a, &c, 0.05, None);
        if ctrl.refinement_count() > 0 {
            break;
        }
        step(&mut state, &cmd, 0.05);
    }
    assert!(ctrl.refinement_count() > 0, "goal A never refined");
    let far = RobotState {
        pose: pose(0.0, 0.0, 0.0),
        ..state
    };
    ctrl.compute_control(&far, &goal(8.0, 0.0), &c, 0.05, None);
    assert_eq!(
        ctrl.refinement_count(),
        0,
        "goal B inherited goal A's refinements"
    );
}

#[test]
fn pose_regulator_keeps_its_driving_direction_for_a_goal() {
    for side in [1.0, -1.0] {
        let mut t = Tracker::new(TrackerKind::PoseRegulator);
        let mut cfg = config(true);
        cfg.goal_tolerance = 0.05;
        cfg.angular_tolerance = 0.1;
        t.set_config(cfg);
        t.init(constraints(SteeringType::Differential));
        t.set_goal(ondrive::Goal {
            target_pose: pose(0.01, side * 2.0, side * PI / 2.0),
            ..Default::default()
        });
        let mut state = RobotState {
            pose: pose(0.0, 0.0, 0.0),
            allow_move: true,
            allow_reverse: true,
            ..Default::default()
        };
        let (mut flips, mut last) = (0, 0.0_f64);
        for _ in 0..800 {
            let cmd = t.tick(&state, 0.05, None);
            if cmd.linear_velocity.abs() > 0.02 {
                if last * cmd.linear_velocity < 0.0 {
                    flips += 1;
                }
                last = cmd.linear_velocity;
            }
            if t.is_goal_reached() {
                break;
            }
            step(&mut state, &cmd, 0.05);
        }
        assert!(t.is_goal_reached(), "side {side}: goal not reached");
        assert!(flips <= 1, "side {side}: direction flipped {flips} times");
    }
}

#[test]
fn dwa_backs_a_blocked_car_out() {
    use ondrive::{GaussianMode, Obstacle, WorldConstraints};
    let wall: Vec<Obstacle> = (-3..=3)
        .map(|i| Obstacle {
            id: i as u64,
            radius: 0.25,
            dt: 0.0,
            modes: vec![GaussianMode {
                weight: 1.0,
                mean_x: vec![0.9],
                mean_y: vec![i as f64 * 0.3],
                std_x: vec![0.0],
                std_y: vec![0.0],
            }],
        })
        .collect();
    let world = WorldConstraints {
        obstacles: wall,
        ..Default::default()
    };
    let mut t = Tracker::new(TrackerKind::Dwa);
    t.set_config(config(true));
    let mut c = constraints(SteeringType::Ackermann);
    c.robot_width = 0.4;
    c.robot_length = 0.6;
    t.init(c);
    t.set_goal(ondrive::Goal {
        target_pose: pose(3.0, 2.0, 0.0),
        tolerance_position: 0.3,
        tolerance_orientation: PI,
        ..Default::default()
    });
    let mut state = RobotState {
        pose: pose(0.0, 0.0, 0.0),
        allow_move: true,
        allow_reverse: true,
        ..Default::default()
    };
    let mut reversed = false;
    for _ in 0..60 {
        let cmd = t.tick(&state, 0.05, Some(&world));
        assert!(cmd.valid, "{}", cmd.status_message);
        reversed |= cmd.linear_velocity < -0.01;
        step(&mut state, &cmd, 0.05);
    }
    assert!(reversed, "the car never backed away from the wall");
}

#[test]
fn geometric_followers_drive_to_a_cusp_and_back() {
    for kind in [
        TrackerKind::RegulatedPursuit,
        TrackerKind::PurePursuit,
        TrackerKind::Stanley,
        TrackerKind::Lqr,
        TrackerKind::VectorPursuit,
    ] {
        let mut path = ondrive::Path::default();
        for i in 0..=50 {
            path.waypoints.push(pose(i as f64 * 0.1, 0.0, 0.0));
            path.speeds.push(0.8);
        }
        for i in 1..=30 {
            path.waypoints.push(pose(5.0 - i as f64 * 0.1, 0.0, 0.0));
            path.speeds.push(-0.8);
        }
        let mut t = Tracker::new(kind);
        let mut cfg = config(true);
        cfg.goal_tolerance = 0.15;
        cfg.angular_tolerance = PI;
        t.set_config(cfg);
        t.init(constraints(SteeringType::Differential));
        t.set_path(path);
        let mut state = RobotState {
            pose: pose(0.0, 0.0, 0.0),
            allow_move: true,
            allow_reverse: true,
            ..Default::default()
        };
        let mut furthest: f64 = 0.0;
        for _ in 0..2000 {
            let cmd = t.tick(&state, 0.05, None);
            assert!(cmd.valid, "{}", cmd.status_message);
            if t.is_path_completed() {
                break;
            }
            step(&mut state, &cmd, 0.05);
            furthest = furthest.max(state.pose.point.x);
        }
        assert!(
            t.is_path_completed(),
            "{kind:?}: not completed, at {:?}",
            state.pose.point
        );
        assert!(
            furthest > 4.8,
            "{kind:?}: turned back at x = {furthest:.2} before the cusp"
        );
        assert!(
            (state.pose.point.x - 2.0).abs() < 0.3,
            "{kind:?}: ended at x = {:.2}",
            state.pose.point.x
        );
        assert!(
            state.pose.rotation.to_euler().yaw.abs() < 0.3,
            "{kind:?}: turned round instead of reversing"
        );
    }
}
