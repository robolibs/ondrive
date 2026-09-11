//! Cross-controller behavioural guarantees: input guards, output units,
//! Ackermann feasibility, steering direction, reverse driving, goal
//! orientation handling and tracker path semantics.

#![allow(clippy::field_reassign_with_default)]

use datapod::{Euler, Point, Pose, Quaternion};
use ondrive::core::path::{cumulative_lengths, project};
use ondrive::{
    Controller, ControllerConfig, DwaConfig, DwaFollower, FlcFollower, Goal, LqrFollower,
    McaConfig, McaFollower, MpcConfig, MpcFollower, MppiConfig, MppiFollower, OutputUnits, Path,
    PidFollower, PurePursuitFollower, RobotConstraints, RobotState, SocConfig, SocFollower,
    StanleyFollower, SteeringType, TebConfig, TebFollower, Tracker, TrackerKind, CarrotFollower,
    IlqrConfig, IlqrFollower, PoseReachFollower, RegulatedPursuitFollower,
};
use std::f64::consts::{FRAC_PI_2, PI};

fn pose(x: f64, y: f64, yaw: f64) -> Pose {
    Pose {
        point: Point::new(x, y, 0.0),
        rotation: Quaternion::from_euler(Euler::new(0.0, 0.0, yaw)),
    }
}

fn line_path(from: f64, to: f64, step: f64) -> Path {
    let mut p = Path::default();
    let n = ((to - from) / step).abs().round() as usize;
    let dir = (to - from).signum();
    p.waypoints = (0..=n)
        .map(|i| pose(from + dir * i as f64 * step, 0.0, 0.0))
        .collect();
    p
}

fn sine_path(n: usize) -> Path {
    let mut p = Path::default();
    p.waypoints = (0..n)
        .map(|i| {
            let x = i as f64 * 0.4;
            pose(x, (x * 0.4).sin() * 1.0, 0.0)
        })
        .collect();
    p
}

fn constraints(steering: SteeringType) -> RobotConstraints {
    let mut c = RobotConstraints::default();
    c.steering_type = steering;
    c.max_linear_velocity = 1.0;
    c.min_linear_velocity = -0.6;
    c.max_angular_velocity = 2.0;
    c.max_linear_acceleration = 2.0;
    c.max_angular_acceleration = 4.0;
    c.max_steering_angle = 0.6;
    c.min_turning_radius = 0.0;
    c.wheelbase = 0.5;
    c.robot_width = 0.4;
    c.robot_length = 0.6;
    c
}

fn config(units: OutputUnits) -> ControllerConfig {
    let mut cfg = ControllerConfig::default();
    cfg.output_units = units;
    cfg.goal_tolerance = 0.4;
    cfg.angular_tolerance = PI;
    cfg.lookahead_distance = 1.0;
    cfg.kp_linear = 1.5;
    cfg.kp_angular = 2.0;
    cfg.k_cross_track = 1.5;
    cfg.k_heading = 1.0;
    cfg
}

const PATH_KINDS: [TrackerKind; 12] = [
    TrackerKind::RegulatedPursuit,
    TrackerKind::Ilqr,
    TrackerKind::PurePursuit,
    TrackerKind::Stanley,
    TrackerKind::Lqr,
    TrackerKind::Mpc,
    TrackerKind::Mppi,
    TrackerKind::Mca,
    TrackerKind::Soc,
    TrackerKind::Teb,
    TrackerKind::Flc,
    TrackerKind::Dwa,
];

const ALL_KINDS: [TrackerKind; 15] = [
    TrackerKind::RegulatedPursuit,
    TrackerKind::Ilqr,
    TrackerKind::PoseReach,
    TrackerKind::Pid,
    TrackerKind::Carrot,
    TrackerKind::PurePursuit,
    TrackerKind::Stanley,
    TrackerKind::Lqr,
    TrackerKind::Mpc,
    TrackerKind::Mppi,
    TrackerKind::Mca,
    TrackerKind::Soc,
    TrackerKind::Teb,
    TrackerKind::Flc,
    TrackerKind::Dwa,
];

/// Followers with small predictive budgets so the suite stays fast.
fn make(kind: TrackerKind) -> Box<dyn Controller> {
    match kind {
        TrackerKind::Pid => Box::new(PidFollower::new()),
        TrackerKind::Carrot => Box::new(CarrotFollower::new()),
        TrackerKind::PurePursuit => Box::new(PurePursuitFollower::new()),
        TrackerKind::Stanley => Box::new(StanleyFollower::new()),
        TrackerKind::Lqr => Box::new(LqrFollower::new()),
        TrackerKind::Flc => Box::new(FlcFollower::new()),
        TrackerKind::RegulatedPursuit => Box::new(RegulatedPursuitFollower::new()),
        TrackerKind::PoseReach => Box::new(PoseReachFollower::new()),
        TrackerKind::Ilqr => {
            let mut c = IlqrConfig::default();
            c.horizon_steps = 12;
            Box::new(IlqrFollower::with_ilqr_config(c))
        }
        TrackerKind::Mpc => {
            let mut c = MpcConfig::default();
            c.horizon_steps = 8;
            Box::new(MpcFollower::with_mpc_config(c))
        }
        TrackerKind::Mppi => {
            let mut c = MppiConfig::default();
            c.horizon_steps = 10;
            c.num_samples = 150;
            c.decel_distance = 0.8;
            Box::new(MppiFollower::with_seed(c, 1))
        }
        TrackerKind::Mca => {
            let mut c = McaConfig::default();
            c.horizon_steps = 10;
            c.num_samples = 150;
            c.num_mc_samples = 50;
            c.decel_distance = 0.8;
            Box::new(McaFollower::with_seed(c, 2))
        }
        TrackerKind::Soc => {
            let mut c = SocConfig::default();
            c.horizon_steps = 8;
            c.guide_samples = 8;
            c.num_samples = 64;
            c.svgd_iterations = 1;
            c.decel_distance = 0.8;
            Box::new(SocFollower::with_seed(c, 3))
        }
        TrackerKind::Teb => {
            let mut c = TebConfig::default();
            c.n_poses = 5;
            c.iterations = 6;
            Box::new(TebFollower::with_teb_config(c))
        }
        TrackerKind::Dwa => {
            let mut c = DwaConfig::default();
            c.v_samples = 6;
            c.w_samples = 11;
            Box::new(DwaFollower::with_dwa_config(c))
        }
    }
}

struct Run {
    reached: bool,
    ticks: usize,
    pose: Pose,
    max_abs_v: f64,
    max_abs_w: f64,
    max_cte_after_settle: f64,
}

fn simulate(
    ctrl: &mut dyn Controller,
    goal: &Goal,
    c: &RobotConstraints,
    start: Pose,
    max_ticks: usize,
    dt: f64,
    mut on_cmd: impl FnMut(usize, &ondrive::VelocityCommand, &RobotState),
) -> Run {
    let mut state = RobotState {
        pose: start,
        allow_move: true,
        ..Default::default()
    };
    let mut run = Run {
        reached: false,
        ticks: 0,
        pose: start,
        max_abs_v: 0.0,
        max_abs_w: 0.0,
        max_cte_after_settle: 0.0,
    };
    for i in 0..max_ticks {
        let cmd = ctrl.compute_control(&state, goal, c, dt, None);
        assert!(cmd.valid, "tick {i}: invalid command: {}", cmd.status_message);
        assert!(cmd.linear_velocity.is_finite() && cmd.angular_velocity.is_finite());
        on_cmd(i, &cmd, &state);
        run.max_abs_v = run.max_abs_v.max(cmd.linear_velocity.abs());
        run.max_abs_w = run.max_abs_w.max(cmd.angular_velocity.abs());
        let yaw = state.pose.rotation.to_euler().yaw;
        state.pose.point.x += cmd.linear_velocity * yaw.cos() * dt;
        state.pose.point.y += cmd.linear_velocity * yaw.sin() * dt;
        state.pose.rotation =
            Quaternion::from_euler(Euler::new(0.0, 0.0, yaw + cmd.angular_velocity * dt));
        state.velocity.linear = cmd.linear_velocity;
        state.velocity.angular = cmd.angular_velocity;
        run.ticks = i + 1;
        if i as f64 * dt > 3.0 {
            run.max_cte_after_settle = run
                .max_cte_after_settle
                .max(ctrl.get_status().cross_track_error.abs());
        }
        if ctrl.get_status().goal_reached {
            run.reached = true;
            break;
        }
    }
    run.pose = state.pose;
    run
}

fn goal_at_end(path: &Path, tol: f64, ang_tol: f64) -> Goal {
    Goal {
        target_pose: *path.waypoints.last().unwrap(),
        tolerance_position: tol,
        tolerance_orientation: ang_tol,
        ..Default::default()
    }
}

#[test]
fn every_controller_rejects_bad_dt() {
    for kind in ALL_KINDS {
        let mut ctrl = make(kind);
        ctrl.set_config(config(OutputUnits::Physical));
        ctrl.set_path(line_path(0.0, 5.0, 0.5));
        let goal = goal_at_end(&line_path(0.0, 5.0, 0.5), 0.3, PI);
        let state = RobotState {
            pose: pose(0.0, 0.2, 0.0),
            allow_move: true,
            ..Default::default()
        };
        let c = constraints(SteeringType::Differential);
        for bad in [0.0, -0.1, f64::NAN, f64::INFINITY] {
            let cmd = ctrl.compute_control(&state, &goal, &c, bad, None);
            assert!(!cmd.valid, "{kind:?} accepted dt={bad}");
        }
    }
}

#[test]
fn tracker_gates_on_allow_move_and_dt() {
    for kind in ALL_KINDS {
        let mut t = Tracker::new(kind);
        t.set_config(config(OutputUnits::Physical));
        t.init(constraints(SteeringType::Differential));
        t.set_goal(Goal {
            target_pose: pose(5.0, 0.0, 0.0),
            ..Default::default()
        });
        let state = RobotState {
            pose: pose(0.0, 0.0, 0.0),
            allow_move: false,
            ..Default::default()
        };
        let cmd = t.tick(&state, 0.1, None);
        assert!(cmd.valid && cmd.linear_velocity == 0.0 && cmd.angular_velocity == 0.0);
        let state = RobotState {
            allow_move: true,
            ..state
        };
        assert!(!t.tick(&state, 0.0, None).valid);
    }
}

#[test]
fn outputs_respect_units_and_limits() {
    for kind in PATH_KINDS {
        for (units, v_lim, w_lim) in [
            (OutputUnits::Normalized, 1.0, 1.0),
            (OutputUnits::Physical, 1.0, 2.0),
        ] {
            let mut ctrl = make(kind);
            ctrl.set_config(config(units));
            let path = sine_path(30);
            ctrl.set_path(path.clone());
            let goal = goal_at_end(&path, 0.4, PI);
            let c = constraints(SteeringType::Differential);
            let run = simulate(ctrl.as_mut(), &goal, &c, pose(0.0, 0.3, 0.0), 120, 0.1, |_, _, _| {});
            assert!(
                run.max_abs_v <= v_lim + 1e-9 && run.max_abs_w <= w_lim + 1e-9,
                "{kind:?} {units:?}: |v|max={:.3} |w|max={:.3}",
                run.max_abs_v,
                run.max_abs_w
            );
        }
    }
}

#[test]
fn ackermann_commands_are_kinematically_feasible() {
    for kind in PATH_KINDS {
        let mut ctrl = make(kind);
        ctrl.set_config(config(OutputUnits::Physical));
        let path = sine_path(30);
        ctrl.set_path(path.clone());
        let goal = goal_at_end(&path, 0.4, PI);
        let c = constraints(SteeringType::Ackermann);
        let kappa_max = c.max_steering_angle.tan() / c.wheelbase;
        simulate(ctrl.as_mut(), &goal, &c, pose(0.0, 0.3, 0.0), 150, 0.1, |i, cmd, _| {
            let v = cmd.linear_velocity;
            let w = cmd.angular_velocity;
            assert!(
                w.abs() <= v.abs() * kappa_max + 1e-6,
                "{kind:?} tick {i}: |w|={:.3} exceeds |v|*kappa_max={:.3}",
                w.abs(),
                v.abs() * kappa_max
            );
            assert!(cmd.steering_angle.abs() <= c.max_steering_angle + 1e-6);
            if v.abs() > 1e-3 {
                let expected = (w * c.wheelbase / v).atan();
                assert!(
                    (expected - cmd.steering_angle).abs() < 1e-6,
                    "{kind:?} tick {i}: steering_angle {:.4} inconsistent with v,w ({:.4})",
                    cmd.steering_angle,
                    expected
                );
            } else {
                assert!(w.abs() < 1e-9, "{kind:?}: Ackermann yaw rate without speed");
            }
        });
    }
}

#[test]
fn every_path_follower_reaches_end_of_sine_path() {
    for steering in [SteeringType::Differential, SteeringType::Ackermann] {
        for kind in PATH_KINDS {
            if kind == TrackerKind::Dwa {
                continue;
            }
            let mut ctrl = make(kind);
            ctrl.set_config(config(OutputUnits::Physical));
            let path = sine_path(30);
            ctrl.set_path(path.clone());
            let goal = goal_at_end(&path, 0.4, PI);
            let c = constraints(steering);
            let run = simulate(ctrl.as_mut(), &goal, &c, pose(0.0, 0.3, 0.0), 800, 0.1, |_, _, _| {});
            assert!(
                run.reached,
                "{kind:?} {steering:?} did not reach the path end; ended at ({:.2},{:.2}) after {} ticks",
                run.pose.point.x,
                run.pose.point.y,
                run.ticks
            );
            assert!(
                run.max_cte_after_settle < 0.6,
                "{kind:?} {steering:?}: settled CTE {:.3} too large",
                run.max_cte_after_settle
            );
        }
    }
}

fn first_yaw_rate(kind: TrackerKind, steering: SteeringType, start: Pose) -> f64 {
    let mut ctrl = make(kind);
    ctrl.set_config(config(OutputUnits::Physical));
    let path = line_path(0.0, 10.0, 0.5);
    ctrl.set_path(path.clone());
    let goal = goal_at_end(&path, 0.4, PI);
    let c = constraints(steering);
    let state = RobotState {
        pose: start,
        velocity: ondrive::Velocity {
            linear: 0.5,
            ..Default::default()
        },
        allow_move: true,
        ..Default::default()
    };
    let cmd = ctrl.compute_control(&state, &goal, &c, 0.1, None);
    assert!(cmd.valid, "{kind:?}: {}", cmd.status_message);
    assert!(cmd.linear_velocity > 0.0, "{kind:?}: should drive forward");
    cmd.angular_velocity
}

#[test]
fn geometric_followers_steer_toward_the_path() {
    for steering in [SteeringType::Differential, SteeringType::Ackermann] {
        for kind in [
            TrackerKind::PurePursuit,
            TrackerKind::Stanley,
            TrackerKind::Lqr,
            TrackerKind::Flc,
            TrackerKind::Mpc,
            TrackerKind::Ilqr,
            TrackerKind::RegulatedPursuit,
        ] {
            let left = first_yaw_rate(kind, steering, pose(2.0, 0.6, 0.0));
            let right = first_yaw_rate(kind, steering, pose(2.0, -0.6, 0.0));
            assert!(left < -1e-3, "{kind:?} {steering:?}: left of path should turn right, got {left:.4}");
            assert!(right > 1e-3, "{kind:?} {steering:?}: right of path should turn left, got {right:.4}");
            let heading_left = first_yaw_rate(kind, steering, pose(2.0, 0.0, 0.4));
            assert!(heading_left < -1e-3, "{kind:?} {steering:?}: heading left should turn right");
        }
    }
}

#[test]
fn pure_pursuit_reverses_along_a_path_behind_it() {
    let path = line_path(0.0, -8.0, 0.5);
    for steering in [SteeringType::Differential, SteeringType::Ackermann] {
        let mut ctrl = PurePursuitFollower::new();
        let mut cfg = config(OutputUnits::Physical);
        cfg.allow_reverse = true;
        ctrl.set_config(cfg);
        ctrl.set_path(path.clone());
        let goal = goal_at_end(&path, 0.4, PI);
        let c = constraints(steering);
        let mut saw_reverse = false;
        let run = simulate(&mut ctrl, &goal, &c, pose(0.0, 0.2, 0.0), 600, 0.1, |_, cmd, _| {
            if cmd.linear_velocity < -0.05 {
                saw_reverse = true;
            }
        });
        assert!(saw_reverse, "{steering:?}: never drove in reverse");
        assert!(run.reached, "{steering:?}: did not reach the end in reverse");
        assert!(
            run.pose.point.x < -7.0,
            "{steering:?}: ended at x={:.2}",
            run.pose.point.x
        );
    }
}

#[test]
fn pure_pursuit_turns_around_when_reverse_is_forbidden() {
    let path = line_path(0.0, -8.0, 0.5);
    let mut ctrl = PurePursuitFollower::new();
    ctrl.set_config(config(OutputUnits::Physical));
    ctrl.set_path(path.clone());
    let goal = goal_at_end(&path, 0.4, PI);
    let c = constraints(SteeringType::Differential);
    let run = simulate(&mut ctrl, &goal, &c, pose(0.0, 0.2, 0.0), 800, 0.1, |_, cmd, _| {
        assert!(cmd.linear_velocity >= -1e-9, "reverse command without permission");
    });
    assert!(run.reached);
}

#[test]
fn pid_aligns_to_goal_orientation_when_it_can_turn_in_place() {
    let mut ctrl = PidFollower::new();
    let mut cfg = config(OutputUnits::Physical);
    cfg.goal_tolerance = 0.15;
    ctrl.set_config(cfg);
    let goal = Goal {
        target_pose: pose(3.0, 0.0, FRAC_PI_2),
        tolerance_position: 0.15,
        tolerance_orientation: 0.1,
        ..Default::default()
    };
    let c = constraints(SteeringType::Differential);
    let run = simulate(&mut ctrl, &goal, &c, pose(0.0, 0.0, 0.0), 1500, 0.05, |_, _, _| {});
    assert!(run.reached, "PID did not reach the pose goal");
    let yaw = run.pose.rotation.to_euler().yaw;
    assert!((yaw - FRAC_PI_2).abs() < 0.15, "final yaw {yaw:.3} not aligned");
}

#[test]
fn pid_ackermann_reports_arrival_on_position() {
    let mut ctrl = PidFollower::new();
    let mut cfg = config(OutputUnits::Physical);
    cfg.goal_tolerance = 0.2;
    ctrl.set_config(cfg);
    let goal = Goal {
        target_pose: pose(4.0, 0.0, FRAC_PI_2),
        tolerance_position: 0.2,
        tolerance_orientation: 0.1,
        ..Default::default()
    };
    let c = constraints(SteeringType::Ackermann);
    let run = simulate(&mut ctrl, &goal, &c, pose(0.0, 0.0, 0.0), 1500, 0.05, |_, _, _| {});
    assert!(run.reached, "Ackermann PID should stop on position tolerance");
}

#[test]
fn pid_integral_does_not_wind_up() {
    let mut ctrl = PidFollower::new();
    let mut cfg = config(OutputUnits::Physical);
    cfg.ki_linear = 1.0;
    cfg.ki_angular = 1.0;
    ctrl.set_config(cfg);
    let goal = Goal {
        target_pose: pose(5.0, 0.0, 0.0),
        ..Default::default()
    };
    let c = constraints(SteeringType::Differential);
    let stuck = RobotState {
        pose: pose(0.0, 1.0, 0.0),
        allow_move: true,
        ..Default::default()
    };
    for _ in 0..2000 {
        let cmd = ctrl.compute_control(&stuck, &goal, &c, 0.05, None);
        assert!(cmd.valid && cmd.linear_velocity <= c.max_linear_velocity + 1e-9);
    }
    let near = RobotState {
        pose: pose(4.95, 0.0, 0.0),
        allow_move: true,
        ..Default::default()
    };
    let cmd = ctrl.compute_control(&near, &goal, &c, 0.05, None);
    assert!(cmd.valid);
    assert!(ctrl.get_status().goal_reached, "arrival must not be delayed by windup");
}

#[test]
fn tracker_derives_goal_from_path_end() {
    let mut t = Tracker::new(TrackerKind::PurePursuit);
    t.set_config(config(OutputUnits::Physical));
    t.init(constraints(SteeringType::Differential));
    t.set_path(line_path(0.0, 6.0, 0.5));
    let mut state = RobotState {
        pose: pose(0.0, 0.0, 0.0),
        allow_move: true,
        ..Default::default()
    };
    let dt = 0.1;
    let mut reached = false;
    for _ in 0..400 {
        let cmd = t.tick(&state, dt, None);
        assert!(cmd.valid, "{}", cmd.status_message);
        let yaw = state.pose.rotation.to_euler().yaw;
        state.pose.point.x += cmd.linear_velocity * yaw.cos() * dt;
        state.pose.point.y += cmd.linear_velocity * yaw.sin() * dt;
        state.pose.rotation =
            Quaternion::from_euler(Euler::new(0.0, 0.0, yaw + cmd.angular_velocity * dt));
        state.velocity.linear = cmd.linear_velocity;
        if t.is_goal_reached() {
            reached = true;
            break;
        }
    }
    assert!(reached, "path without explicit goal was not completed");
    assert!(t.is_path_completed());
    let cmd = t.tick(&state, dt, None);
    assert!(cmd.valid && cmd.linear_velocity == 0.0);
}

#[test]
fn tracker_walks_point_controller_through_waypoints() {
    let mut t = Tracker::new(TrackerKind::Pid);
    let mut cfg = config(OutputUnits::Physical);
    cfg.goal_tolerance = 0.3;
    cfg.angular_tolerance = PI;
    t.set_config(cfg);
    t.init(constraints(SteeringType::Differential));
    let mut path = Path::default();
    path.waypoints = vec![
        pose(0.0, 0.0, 0.0),
        pose(3.0, 0.0, 0.0),
        pose(3.0, 3.0, 0.0),
        pose(0.0, 3.0, 0.0),
    ];
    t.set_path(path);
    let mut state = RobotState {
        pose: pose(0.0, 0.0, 0.0),
        allow_move: true,
        ..Default::default()
    };
    let mut targets_seen = Vec::new();
    let dt = 0.05;
    for _ in 0..4000 {
        let cmd = t.tick(&state, dt, None);
        assert!(cmd.valid, "{}", cmd.status_message);
        if let Some(target) = t.current_target()
            && targets_seen.last() != Some(&(target.x, target.y))
        {
            targets_seen.push((target.x, target.y));
        }
        if t.is_path_completed() {
            break;
        }
        let yaw = state.pose.rotation.to_euler().yaw;
        state.pose.point.x += cmd.linear_velocity * yaw.cos() * dt;
        state.pose.point.y += cmd.linear_velocity * yaw.sin() * dt;
        state.pose.rotation =
            Quaternion::from_euler(Euler::new(0.0, 0.0, yaw + cmd.angular_velocity * dt));
    }
    assert!(t.is_path_completed(), "targets seen: {targets_seen:?}");
    assert!(targets_seen.contains(&(3.0, 0.0)) && targets_seen.contains(&(3.0, 3.0)));
    assert!(state.pose.point.distance_to_2d(Point::new(0.0, 3.0, 0.0)) < 0.5);
}

#[test]
fn goal_tolerances_override_config() {
    let mut ctrl = PidFollower::new();
    let mut cfg = config(OutputUnits::Physical);
    cfg.goal_tolerance = 0.05;
    ctrl.set_config(cfg);
    let goal = Goal {
        target_pose: pose(1.0, 0.0, 0.0),
        tolerance_position: 0.5,
        tolerance_orientation: 0.5,
        ..Default::default()
    };
    let state = RobotState {
        pose: pose(0.7, 0.0, 0.0),
        allow_move: true,
        ..Default::default()
    };
    let c = constraints(SteeringType::Differential);
    ctrl.compute_control(&state, &goal, &c, 0.05, None);
    assert!(ctrl.get_status().goal_reached);
}

#[test]
fn dwa_reaches_goal_behind_a_differential_robot() {
    let mut ctrl = make(TrackerKind::Dwa);
    ctrl.set_config(config(OutputUnits::Physical));
    let goal = Goal {
        target_pose: pose(-3.0, 0.5, 0.0),
        tolerance_position: 0.4,
        tolerance_orientation: PI,
        ..Default::default()
    };
    let c = constraints(SteeringType::Differential);
    let run = simulate(ctrl.as_mut(), &goal, &c, pose(0.0, 0.0, 0.0), 600, 0.1, |_, _, _| {});
    assert!(run.reached, "DWA ended at ({:.2},{:.2})", run.pose.point.x, run.pose.point.y);
}

#[test]
fn seeded_sampling_controllers_are_deterministic() {
    let path = sine_path(20);
    let goal = goal_at_end(&path, 0.4, PI);
    let c = constraints(SteeringType::Differential);
    let state = RobotState {
        pose: pose(0.0, 0.3, 0.0),
        allow_move: true,
        ..Default::default()
    };
    for kind in [TrackerKind::Mppi, TrackerKind::Mca, TrackerKind::Soc] {
        let mut a = make(kind);
        let mut b = make(kind);
        for ctrl in [&mut a, &mut b] {
            ctrl.set_config(config(OutputUnits::Physical));
            ctrl.set_path(path.clone());
        }
        let ca = a.compute_control(&state, &goal, &c, 0.1, None);
        let cb = b.compute_control(&state, &goal, &c, 0.1, None);
        assert_eq!(ca.linear_velocity, cb.linear_velocity, "{kind:?}");
        assert_eq!(ca.angular_velocity, cb.angular_velocity, "{kind:?}");
    }
}

#[test]
fn path_projection_geometry() {
    let wps = vec![pose(0.0, 0.0, 0.0), pose(2.0, 0.0, 0.0), pose(2.0, 2.0, 0.0)];
    let cum = cumulative_lengths(&wps);
    assert_eq!(cum, vec![0.0, 2.0, 4.0]);

    let p = project(&wps, &cum, Point::new(1.0, 0.5, 0.0), 0, 2, usize::MAX).unwrap();
    assert_eq!(p.segment, 0);
    assert!((p.t - 0.5).abs() < 1e-12);
    assert!((p.lateral_error - 0.5).abs() < 1e-12, "left of +x is positive");
    assert!((p.heading).abs() < 1e-12);
    assert!((p.arc_length - 1.0).abs() < 1e-12);
    assert!(!p.beyond_end);

    let p = project(&wps, &cum, Point::new(3.0, 1.0, 0.0), 0, 2, usize::MAX).unwrap();
    assert_eq!(p.segment, 1);
    assert!((p.lateral_error + 1.0).abs() < 1e-12, "right of +y is negative");
    assert!((p.heading - FRAC_PI_2).abs() < 1e-12);
    assert!((p.arc_length - 3.0).abs() < 1e-12);

    let p = project(&wps, &cum, Point::new(2.5, 3.0, 0.0), 0, 2, usize::MAX).unwrap();
    assert!(p.beyond_end);
    assert!((p.t - 1.0).abs() < 1e-12);

    let single = vec![pose(1.0, 1.0, 0.0)];
    let cum1 = cumulative_lengths(&single);
    let p = project(&single, &cum1, Point::new(3.0, 1.0, 0.0), 5, 2, usize::MAX).unwrap();
    assert!((p.distance - 2.0).abs() < 1e-12 && p.beyond_end);
}
