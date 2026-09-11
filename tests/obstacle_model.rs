//! Shared obstacle model: occupancy grid distance field, footprint-aware
//! clearance, and the controllers that consume them.

#![allow(clippy::field_reassign_with_default)]

use datapod::{Euler, Point, Pose, Quaternion};
use ondrive::core::obstacles::CollisionChecker;
use ondrive::{
    Controller, ControllerConfig, DwaConfig, DwaFollower, Footprint, GaussianMode, Goal,
    Obstacle, OccupancyGrid, OutputUnits, Path, PoseReachFollower, RobotConstraints, RobotState,
    SteeringType, TebConfig, TebFollower, WorldConstraints,
};
use std::f64::consts::PI;

fn pose(x: f64, y: f64, yaw: f64) -> Pose {
    Pose { point: Point::new(x, y, 0.0), rotation: Quaternion::from_euler(Euler::new(0.0, 0.0, yaw)) }
}

fn constraints(steering: SteeringType) -> RobotConstraints {
    let mut c = RobotConstraints::default();
    c.steering_type = steering;
    c.max_linear_velocity = 1.0;
    c.min_linear_velocity = -0.6;
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

fn config() -> ControllerConfig {
    let mut cfg = ControllerConfig::default();
    cfg.output_units = OutputUnits::Physical;
    cfg.goal_tolerance = 0.3;
    cfg.angular_tolerance = PI;
    cfg.lookahead_distance = 1.0;
    cfg
}

/// 12 m x 6 m grid at 0.1 m with a block occupied at `[x0,x1] x [y0,y1]`.
fn grid_with_block(x0: f64, x1: f64, y0: f64, y1: f64) -> OccupancyGrid {
    let (w, h, res) = (120, 60, 0.1);
    let (ox, oy) = (0.0, -3.0);
    let mut occ = vec![false; w * h];
    for iy in 0..h {
        for ix in 0..w {
            let x = ox + (ix as f64 + 0.5) * res;
            let y = oy + (iy as f64 + 0.5) * res;
            if x >= x0 && x <= x1 && y >= y0 && y <= y1 {
                occ[iy * w + ix] = true;
            }
        }
    }
    OccupancyGrid::new(ox, oy, res, w, h, occ)
}

#[test]
fn grid_distance_field_measures_distance_to_occupied_cells() {
    let g = grid_with_block(5.0, 5.1, -0.1, 0.1);
    assert!(g.is_occupied(5.05, 0.0));
    assert!(!g.is_occupied(4.0, 0.0));
    let d = g.distance_to_occupied(4.0, 0.0);
    assert!((d - 1.0).abs() < 0.08, "distance {d:.3}, expected about 1.0");
    let d2 = g.distance_to_occupied(5.05, 1.0);
    assert!((d2 - 0.95).abs() < 0.1, "distance {d2:.3}, expected about 0.95");
    assert!(g.distance_to_occupied(50.0, 50.0).is_infinite());
}

#[test]
fn polygon_footprint_clearance_depends_on_heading() {
    let world = WorldConstraints {
        obstacles: vec![Obstacle {
            id: 0,
            radius: 0.1,
            modes: vec![GaussianMode { weight: 1.0, mean_x: vec![1.0], mean_y: vec![0.0], std_x: vec![0.05], std_y: vec![0.05] }],
        }],
        ..Default::default()
    };
    let mut c = constraints(SteeringType::Differential);
    c.footprint = Footprint::Polygon { points: vec![(0.5, 0.2), (-0.5, 0.2), (-0.5, -0.2), (0.5, -0.2)] };
    let checker = CollisionChecker::new(Some(&world), &c, 0.0);
    let facing = checker.clearance(0, 0.0, 0.0, 0.0);
    let sideways = checker.clearance(0, 0.0, 0.0, PI / 2.0);
    assert!((facing - 0.4).abs() < 0.02, "facing clearance {facing:.3}");
    assert!((sideways - 0.7).abs() < 0.02, "sideways clearance {sideways:.3}");
    let disc = CollisionChecker::new(Some(&world), &constraints(SteeringType::Differential), 0.0);
    let d = disc.clearance(0, 0.0, 0.0, 0.0);
    assert!((d - (0.9 - 0.5 * (0.4_f64).hypot(0.6))).abs() < 1e-9);
}

fn integrate(state: &mut RobotState, cmd: &ondrive::VelocityCommand, dt: f64) {
    let yaw = state.pose.rotation.to_euler().yaw;
    state.pose.point.x += cmd.linear_velocity * yaw.cos() * dt;
    state.pose.point.y += cmd.linear_velocity * yaw.sin() * dt;
    state.pose.rotation = Quaternion::from_euler(Euler::new(0.0, 0.0, yaw + cmd.angular_velocity * dt));
    state.velocity.linear = cmd.linear_velocity;
    state.velocity.angular = cmd.angular_velocity;
}

fn line(len: f64) -> Path {
    let mut p = Path::default();
    p.waypoints = (0..=(len / 0.25) as usize).map(|i| pose(i as f64 * 0.25, 0.0, 0.0)).collect();
    p
}

#[test]
fn dwa_and_teb_avoid_a_grid_block_on_the_path() {
    let world = WorldConstraints { grid: Some(grid_with_block(5.0, 5.6, -0.4, 0.4)), ..Default::default() };
    let c = constraints(SteeringType::Differential);
    let checker = CollisionChecker::new(Some(&world), &c, 0.0);
    let goal = Goal { target_pose: pose(10.0, 0.0, 0.0), tolerance_position: 0.3, tolerance_orientation: PI, ..Default::default() };

    let mut dwa = DwaFollower::with_dwa_config(DwaConfig::default());
    dwa.set_config(config());
    let mut teb = TebFollower::with_teb_config(TebConfig::default());
    teb.set_config(config());
    teb.set_path(line(10.0));
    let ctrls: Vec<(&str, &mut dyn Controller)> = vec![("dwa", &mut dwa), ("teb", &mut teb)];
    for (name, ctrl) in ctrls {
        let mut state = RobotState { pose: pose(0.0, 0.0, 0.0), allow_move: true, ..Default::default() };
        let mut reached = false;
        let mut min_clear = f64::MAX;
        for _ in 0..800 {
            let cmd = ctrl.compute_control(&state, &goal, &c, 0.1, Some(&world));
            assert!(cmd.valid, "{name}: {}", cmd.status_message);
            integrate(&mut state, &cmd, 0.1);
            let yaw = state.pose.rotation.to_euler().yaw;
            min_clear = min_clear.min(checker.clearance(0, state.pose.point.x, state.pose.point.y, yaw));
            if ctrl.get_status().goal_reached {
                reached = true;
                break;
            }
        }
        assert!(reached, "{name}: stalled at ({:.2},{:.2})", state.pose.point.x, state.pose.point.y);
        assert!(min_clear > 0.0, "{name}: footprint entered the grid block (clearance {min_clear:.3})");
    }
}

#[test]
fn pose_planner_picks_a_collision_free_curve() {
    let world = WorldConstraints { grid: Some(grid_with_block(1.5, 2.5, -0.6, 0.6)), ..Default::default() };
    let c = constraints(SteeringType::Ackermann);
    let checker = CollisionChecker::new(Some(&world), &c, 0.0);
    let mut ctrl = PoseReachFollower::new();
    let mut cfg = config();
    cfg.goal_tolerance = 0.2;
    cfg.angular_tolerance = 0.2;
    cfg.allow_reverse = true;
    ctrl.set_config(cfg);
    let goal = Goal { target_pose: pose(4.0, 0.0, 0.0), tolerance_position: 0.2, tolerance_orientation: 0.2, ..Default::default() };
    let state = RobotState { pose: pose(0.0, 0.0, 0.0), allow_move: true, ..Default::default() };
    let cmd = ctrl.compute_control(&state, &goal, &c, 0.05, Some(&world));
    assert!(cmd.valid, "{}", cmd.status_message);
    let plan = ctrl.plan();
    assert!(plan.waypoints.len() > 2);
    for w in &plan.waypoints {
        assert!(!checker.collides(0, w.point.x, w.point.y, w.rotation.to_euler().yaw), "plan passes through the block at ({:.2},{:.2})", w.point.x, w.point.y);
    }
}
