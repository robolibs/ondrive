//! Regressions for defects found by the independent audit of the rewrite.

#![allow(clippy::field_reassign_with_default)]

use datapod::{Euler, Point, Pose, Quaternion};
use ondrive::core::path::{cumulative_lengths, project};
use ondrive::{
    Controller, ControllerConfig, DwaConfig, DwaFollower, GaussianMode, Goal, MpcConfig,
    MpcFollower, MppiConfig, MppiFollower, Obstacle, OutputUnits, Path, PurePursuitFollower,
    RobotConstraints, RobotState, StanleyFollower, SteeringType, TebConfig, TebFollower, Tracker,
    TrackerKind, Velocity, WorldConstraints,
};
use std::f64::consts::PI;

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

fn constraints(steering: SteeringType) -> RobotConstraints {
    let mut c = RobotConstraints::default();
    c.steering_type = steering;
    c.max_linear_velocity = 1.0;
    c.min_linear_velocity = -0.6;
    c.max_angular_velocity = 2.0;
    c.max_linear_acceleration = 1.0;
    c.max_angular_acceleration = 4.0;
    c.max_steering_angle = 0.6;
    c.min_turning_radius = 0.0;
    c.wheelbase = 0.5;
    c.robot_width = 0.4;
    c.robot_length = 0.6;
    c
}

fn config() -> ControllerConfig {
    let mut cfg = ControllerConfig::default();
    cfg.output_units = OutputUnits::Physical;
    cfg.goal_tolerance = 0.4;
    cfg.angular_tolerance = PI;
    cfg.lookahead_distance = 1.0;
    cfg.kp_linear = 1.5;
    cfg.kp_angular = 2.0;
    cfg.k_cross_track = 1.5;
    cfg.k_heading = 1.0;
    cfg
}

fn goal_at_end(path: &Path) -> Goal {
    Goal {
        target_pose: *path.waypoints.last().unwrap(),
        tolerance_position: 0.4,
        tolerance_orientation: PI,
        ..Default::default()
    }
}

fn integrate(state: &mut RobotState, cmd: &ondrive::VelocityCommand, dt: f64) {
    let yaw = state.pose.rotation.to_euler().yaw;
    state.pose.point.x += cmd.linear_velocity * yaw.cos() * dt;
    state.pose.point.y += cmd.linear_velocity * yaw.sin() * dt;
    state.pose.rotation =
        Quaternion::from_euler(Euler::new(0.0, 0.0, yaw + cmd.angular_velocity * dt));
    state.velocity.linear = cmd.linear_velocity;
    state.velocity.angular = cmd.angular_velocity;
}

#[test]
fn stanley_converges_while_reversing_from_either_side() {
    let path = line_path(0.0, -8.0, 0.5);
    for steering in [SteeringType::Ackermann, SteeringType::Differential] {
        for y0 in [0.5, -0.5] {
            let mut ctrl = StanleyFollower::new();
            let mut cfg = config();
            cfg.allow_reverse = true;
            ctrl.set_config(cfg);
            ctrl.set_path(path.clone());
            let goal = goal_at_end(&path);
            let c = constraints(steering);
            let mut state = RobotState {
                pose: pose(0.0, y0, 0.0),
                velocity: Velocity {
                    linear: -0.3,
                    ..Default::default()
                },
                allow_move: true,
                ..Default::default()
            };
            let first = ctrl.compute_control(&state, &goal, &c, 0.1, None);
            assert!(first.valid && first.linear_velocity < 0.0, "{steering:?} y0={y0}: not reversing");
            let cte = ctrl.get_status().cross_track_error;
            assert!(
                first.angular_velocity.signum() == -cte.signum(),
                "{steering:?} y0={y0}: omega {:.3} must oppose lateral error {:.3}",
                first.angular_velocity,
                cte
            );
            let mut reached = false;
            for i in 0..600 {
                let cmd = ctrl.compute_control(&state, &goal, &c, 0.1, None);
                assert!(cmd.valid);
                integrate(&mut state, &cmd, 0.1);
                if i > 50 {
                    assert!(
                        ctrl.get_status().cross_track_error.abs() < 0.15,
                        "{steering:?} y0={y0}: diverged, cte={:.3} at tick {i}",
                        ctrl.get_status().cross_track_error
                    );
                }
                if ctrl.get_status().goal_reached {
                    reached = true;
                    break;
                }
            }
            assert!(reached, "{steering:?} y0={y0}: did not reach the end in reverse");
        }
    }
}

#[test]
fn pure_pursuit_reverses_toward_an_offset_path_behind_it() {
    let mut path = Path::default();
    let bearing = 120.0_f64.to_radians();
    path.waypoints = (0..=16)
        .map(|i| {
            let s = i as f64 * 0.5;
            pose(s * bearing.cos(), s * bearing.sin(), bearing)
        })
        .collect();
    let mut ctrl = PurePursuitFollower::new();
    let mut cfg = config();
    cfg.allow_reverse = true;
    ctrl.set_config(cfg);
    ctrl.set_path(path.clone());
    let goal = goal_at_end(&path);
    let c = constraints(SteeringType::Differential);
    let mut state = RobotState {
        pose: pose(0.0, 0.0, 0.0),
        allow_move: true,
        ..Default::default()
    };
    let first = ctrl.compute_control(&state, &goal, &c, 0.1, None);
    assert!(first.linear_velocity < 0.0 && first.angular_velocity < 0.0,
        "first command v={:.3} w={:.3}: must reverse while swinging the rear clockwise",
        first.linear_velocity, first.angular_velocity);
    let mut reached = false;
    for _ in 0..800 {
        let cmd = ctrl.compute_control(&state, &goal, &c, 0.1, None);
        assert!(cmd.linear_velocity <= 1e-9, "switched to forward driving");
        integrate(&mut state, &cmd, 0.1);
        if ctrl.get_status().goal_reached {
            reached = true;
            break;
        }
    }
    assert!(reached);
}

#[test]
fn teb_avoids_obstacle_on_its_path() {
    let path = line_path(0.0, 12.0, 0.5);
    let mut ctrl = TebFollower::with_teb_config(TebConfig::default());
    ctrl.set_config(config());
    ctrl.set_path(path.clone());
    let goal = goal_at_end(&path);
    let c = constraints(SteeringType::Differential);
    let world = WorldConstraints {
        obstacles: vec![Obstacle {
            id: 0,
            radius: 0.3,
            modes: vec![GaussianMode {
                weight: 1.0,
                mean_x: vec![5.0],
                mean_y: vec![0.0],
                std_x: vec![0.1],
                std_y: vec![0.1],
            }],
        }],
        ..Default::default()
    };
    let mut state = RobotState {
        pose: pose(0.0, 0.0, 0.0),
        allow_move: true,
        ..Default::default()
    };
    let contact = 0.3 + 0.5 * (0.4_f64).hypot(0.6);
    let mut min_d = f64::MAX;
    for _ in 0..800 {
        let cmd = ctrl.compute_control(&state, &goal, &c, 0.1, Some(&world));
        assert!(cmd.valid, "{}", cmd.status_message);
        integrate(&mut state, &cmd, 0.1);
        min_d = min_d.min((state.pose.point.x - 5.0).hypot(state.pose.point.y));
        if ctrl.get_status().goal_reached || state.pose.point.x > 11.0 {
            break;
        }
    }
    assert!(min_d > contact, "TEB came within {min_d:.3} m of the obstacle centre (contact at {contact:.3})");
    assert!(state.pose.point.x > 9.0, "TEB stalled at x={:.2}", state.pose.point.x);
}

#[test]
fn teb_respects_acceleration_from_rest_and_follows_goal_changes() {
    let mut t = Tracker::new(TrackerKind::Teb);
    t.set_config(config());
    let c = constraints(SteeringType::Differential);
    t.init(c.clone());
    t.set_goal(Goal {
        target_pose: pose(4.0, 0.0, 0.0),
        ..Default::default()
    });
    let mut state = RobotState {
        pose: pose(0.0, 0.0, 0.0),
        allow_move: true,
        ..Default::default()
    };
    let dt = 0.1;
    let first = t.tick(&state, dt, None);
    assert!(first.valid);
    assert!(
        first.linear_velocity <= c.max_linear_acceleration * dt + 1e-9,
        "first command {:.3} exceeds the acceleration limit",
        first.linear_velocity
    );
    for _ in 0..30 {
        let cmd = t.tick(&state, dt, None);
        integrate(&mut state, &cmd, dt);
    }
    t.set_goal(Goal {
        target_pose: pose(0.0, 4.0, 0.0),
        ..Default::default()
    });
    let mut reached = false;
    for _ in 0..800 {
        let cmd = t.tick(&state, dt, None);
        assert!(cmd.valid);
        integrate(&mut state, &cmd, dt);
        if t.is_goal_reached() {
            reached = true;
            break;
        }
    }
    assert!(reached, "TEB did not follow the new goal; ended at ({:.2},{:.2})", state.pose.point.x, state.pose.point.y);
    assert!(state.pose.point.distance_to_2d(Point::new(0.0, 4.0, 0.0)) < 0.5);
}

#[test]
fn dwa_escapes_from_inside_the_obstacle_margin() {
    let mut ctrl = DwaFollower::with_dwa_config(DwaConfig::default());
    ctrl.set_config(config());
    let c = constraints(SteeringType::Differential);
    let world = WorldConstraints {
        obstacles: vec![Obstacle {
            id: 0,
            radius: 0.3,
            modes: vec![GaussianMode {
                weight: 1.0,
                mean_x: vec![-0.75],
                mean_y: vec![0.0],
                std_x: vec![0.1],
                std_y: vec![0.1],
            }],
        }],
        ..Default::default()
    };
    let goal = Goal {
        target_pose: pose(10.0, 0.0, 0.0),
        ..Default::default()
    };
    let state = RobotState {
        pose: pose(0.0, 0.0, 0.0),
        allow_move: true,
        ..Default::default()
    };
    let cmd = ctrl.compute_control(&state, &goal, &c, 0.1, Some(&world));
    assert!(cmd.valid && cmd.linear_velocity > 0.0, "DWA froze inside the margin: v={:.3}", cmd.linear_velocity);
}

#[test]
fn predictive_controllers_honour_acceleration_at_high_control_rates() {
    let path = line_path(0.0, 10.0, 0.5);
    let goal = goal_at_end(&path);
    let c = constraints(SteeringType::Differential);
    let dt = 0.02;
    let mut mpc = MpcFollower::with_mpc_config(MpcConfig::default());
    let mut mppi = MppiFollower::with_seed(MppiConfig::default(), 5);
    let controllers: Vec<(&str, &mut dyn Controller)> = vec![("mpc", &mut mpc), ("mppi", &mut mppi)];
    for (name, ctrl) in controllers {
        ctrl.set_config(config());
        ctrl.set_path(path.clone());
        let mut state = RobotState {
            pose: pose(0.0, 0.0, 0.0),
            allow_move: true,
            ..Default::default()
        };
        let mut prev_v = 0.0;
        for i in 0..60 {
            let cmd = ctrl.compute_control(&state, &goal, &c, dt, None);
            assert!(cmd.valid);
            let dv = (cmd.linear_velocity - prev_v).abs();
            assert!(
                dv <= c.max_linear_acceleration * dt + 1e-6,
                "{name} tick {i}: speed step {dv:.4} exceeds a*dt={:.4}",
                c.max_linear_acceleration * dt
            );
            prev_v = cmd.linear_velocity;
            integrate(&mut state, &cmd, dt);
        }
    }
}

#[test]
fn predictive_ackermann_with_pose_ahead_of_rear_axle_still_tracks() {
    let mut path = Path::default();
    path.waypoints = (0..40)
        .map(|i| {
            let x = i as f64 * 0.4;
            pose(x, (x * 0.4).sin(), 0.0)
        })
        .collect();
    let goal = goal_at_end(&path);
    let mut c = constraints(SteeringType::Ackermann);
    c.rear_wheelbase = 0.35;
    let mut ctrl = MpcFollower::with_mpc_config(MpcConfig::default());
    ctrl.set_config(config());
    ctrl.set_path(path.clone());
    let mut state = RobotState {
        pose: pose(0.0, 0.3, 0.0),
        allow_move: true,
        ..Default::default()
    };
    let mut max_cte: f64 = 0.0;
    let mut reached = false;
    for i in 0..800 {
        let cmd = ctrl.compute_control(&state, &goal, &c, 0.1, None);
        assert!(cmd.valid);
        integrate(&mut state, &cmd, 0.1);
        if i > 30 {
            max_cte = max_cte.max(ctrl.get_status().cross_track_error.abs());
        }
        if ctrl.get_status().goal_reached {
            reached = true;
            break;
        }
    }
    assert!(reached && max_cte < 0.5, "reached={reached} max_cte={max_cte:.3}");
}

#[test]
fn set_path_clears_stale_goal_reached() {
    let path = line_path(0.0, 2.0, 0.5);
    let goal = goal_at_end(&path);
    let c = constraints(SteeringType::Differential);
    let state = RobotState {
        pose: pose(2.0, 0.0, 0.0),
        allow_move: true,
        ..Default::default()
    };
    let mut ctrl = PurePursuitFollower::new();
    ctrl.set_config(config());
    ctrl.set_path(path.clone());
    ctrl.compute_control(&state, &goal, &c, 0.1, None);
    assert!(ctrl.get_status().goal_reached);
    ctrl.set_path(line_path(2.0, 8.0, 0.5));
    assert!(!ctrl.get_status().goal_reached, "stale goal_reached survived set_path");
}

#[test]
fn duplicated_final_waypoint_still_reports_beyond_end() {
    let mut wps = vec![pose(0.0, 0.0, 0.0), pose(2.0, 0.0, 0.0)];
    wps.push(pose(2.0, 0.0, 0.0));
    let cum = cumulative_lengths(&wps);
    let p = project(&wps, &cum, Point::new(2.5, 0.1, 0.0), 0, 2, usize::MAX).unwrap();
    assert!(p.beyond_end, "projection past a duplicated end must flag beyond_end");
}

#[test]
fn tracker_point_controller_stays_on_a_circle_path() {
    let mut path = Path::default();
    let pts: Vec<(f64, f64)> = (0..=56)
        .map(|i| {
            let a = i as f64 * 0.25 / 3.0;
            (3.0 * a.sin(), 3.0 * (1.0 - a.cos()))
        })
        .collect();
    for w in pts.windows(2) {
        let h = (w[1].1 - w[0].1).atan2(w[1].0 - w[0].0);
        path.waypoints.push(pose(w[0].0, w[0].1, h));
    }
    let last = pts[pts.len() - 1];
    path.waypoints.push(pose(last.0, last.1, 0.0));

    let mut t = Tracker::new(TrackerKind::Pid);
    t.set_config(config());
    t.init(constraints(SteeringType::Ackermann));
    t.set_path(path.clone());
    let cum = cumulative_lengths(&path.waypoints);
    let mut state = RobotState {
        pose: pose(0.0, 0.0, 0.0),
        allow_move: true,
        ..Default::default()
    };
    let mut max_off = 0.0_f64;
    let mut reached = false;
    for _ in 0..3000 {
        let cmd = t.tick(&state, 0.05, None);
        assert!(cmd.valid, "{}", cmd.status_message);
        integrate(&mut state, &cmd, 0.05);
        let p = project(&path.waypoints, &cum, state.pose.point, 0, 0, usize::MAX).unwrap();
        max_off = max_off.max(p.distance);
        if t.is_goal_reached() {
            reached = true;
            break;
        }
    }
    assert!(reached, "PID via tracker did not finish the circle");
    assert!(max_off < 0.5, "PID cut across the circle: max offset {max_off:.2} m");
}

fn obstacle_world(x: f64, y: f64) -> WorldConstraints {
    WorldConstraints {
        obstacles: vec![Obstacle {
            id: 0,
            radius: 0.3,
            modes: vec![GaussianMode {
                weight: 1.0,
                mean_x: vec![x; 40],
                mean_y: vec![y; 40],
                std_x: vec![0.05; 40],
                std_y: vec![0.05; 40],
            }],
        }],
        ..Default::default()
    }
}

#[test]
fn sampling_planners_pass_an_obstacle_beside_the_path_on_ackermann() {
    let mut path = Path::default();
    path.waypoints = (0..=60)
        .map(|i| {
            let x = i as f64 * 0.25;
            pose(x, (x * 0.35).sin() * 1.2, 0.0)
        })
        .collect();
    let goal = goal_at_end(&path);
    let (ox, oy) = (8.0, (8.0_f64 * 0.35).sin() * 1.2 + 0.4);
    let world = obstacle_world(ox, oy);
    let c = constraints(SteeringType::Ackermann);
    let contact = 0.3 + 0.5 * (0.4_f64).hypot(0.6);

    for seed in [4_u64, 6, 9] {
        let mut soc_cfg = ondrive::SocConfig::default();
        soc_cfg.horizon_steps = 14;
        soc_cfg.guide_samples = 24;
        soc_cfg.num_samples = 160;
        soc_cfg.svgd_iterations = 2;
        let mut soc = ondrive::SocFollower::with_seed(soc_cfg, seed);
        let mut mca_cfg = ondrive::McaConfig::default();
        mca_cfg.horizon_steps = 14;
        mca_cfg.num_samples = 200;
        mca_cfg.num_mc_samples = 60;
        let mut mca = ondrive::McaFollower::with_seed(mca_cfg, seed);
        let planners: Vec<(&str, &mut dyn Controller)> = vec![("soc", &mut soc), ("mca", &mut mca)];
        for (name, ctrl) in planners {
            ctrl.set_config(config());
            ctrl.set_path(path.clone());
            let mut state = RobotState {
                pose: pose(0.0, 0.0, 0.0),
                allow_move: true,
                ..Default::default()
            };
            let mut min_d = f64::MAX;
            let mut reached = false;
            for _ in 0..800 {
                let cmd = ctrl.compute_control(&state, &goal, &c, 0.1, Some(&world));
                assert!(cmd.valid, "{name} seed {seed}: {}", cmd.status_message);
                integrate(&mut state, &cmd, 0.1);
                min_d = min_d.min((state.pose.point.x - ox).hypot(state.pose.point.y - oy));
                if ctrl.get_status().goal_reached {
                    reached = true;
                    break;
                }
            }
            assert!(
                reached,
                "{name} seed {seed}: stalled at ({:.2},{:.2})",
                state.pose.point.x, state.pose.point.y
            );
            assert!(min_d > contact, "{name} seed {seed}: came within {min_d:.2} m of the obstacle");
        }
    }
}
