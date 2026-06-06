#![allow(clippy::field_reassign_with_default)]

use approx::assert_relative_eq;
use datapod::{Euler, Point, Pose, Quaternion};
use nalgebra::{Matrix4, Vector4};
use ondrive::{
    ControllerConfig, Goal, OutputUnits, Path, RobotConstraints, RobotState, Tracker, TrackerKind,
};

fn pose_at(x: f64, y: f64, yaw: f64) -> Pose {
    Pose {
        point: Point::new(x, y, 0.0),
        rotation: Quaternion::from_euler(Euler::new(0.0, 0.0, yaw)),
    }
}

/// Re-implement the DARE solver locally to verify convergence (the one inside
/// the crate is private; this test mirrors the same iteration to sanity-check
/// that the Riccati residual is small at convergence).
fn solve_dare(a: &Matrix4<f64>, b: &Vector4<f64>, q: &Matrix4<f64>, r: f64) -> Matrix4<f64> {
    let mut p = *q;
    let at = a.transpose();
    for _ in 0..500 {
        let at_p = at * p;
        let at_p_a = at_p * a;
        let pb = p * b;
        let bt_p_b = b.dot(&pb);
        let denom = r + bt_p_b;
        let inv_denom = 1.0 / denom;
        let at_p_b = at_p * b;
        let bt_p_a = (b.transpose() * p * a).transpose();
        let term = at_p_b * bt_p_a.transpose() * inv_denom;
        let p_next = at_p_a - term + q;
        if (p_next - p).norm() < 1e-8 {
            p = p_next;
            break;
        }
        p = p_next;
    }
    p
}

#[test]
fn dare_residual_is_small_at_convergence() {
    let dt = 0.1;
    let velocity = 1.0;
    let wheelbase = 0.5;

    let a = Matrix4::new(
        1.0, dt, 0.0, 0.0, 0.0, 1.0, velocity, 0.0, 0.0, 0.0, 1.0, dt, 0.0, 0.0, 0.0, 1.0,
    );
    let b = Vector4::new(0.0, 0.0, 0.0, velocity / wheelbase);
    let mut q = Matrix4::zeros();
    q[(0, 0)] = 1.0;
    q[(2, 2)] = 0.5;
    let r = 0.5;

    let p = solve_dare(&a, &b, &q, r);

    // Check the discrete-time Riccati residual:
    //   R(P) = A'PA − A'PB(R+B'PB)⁻¹B'PA + Q − P
    let at_p = a.transpose() * p;
    let at_p_a = at_p * a;
    let pb = p * b;
    let bt_p_b = b.dot(&pb);
    let inv_denom = 1.0 / (r + bt_p_b);
    let at_p_b = at_p * b;
    let bt_p_a = (b.transpose() * p * a).transpose();
    let term = at_p_b * bt_p_a.transpose() * inv_denom;
    let residual = (at_p_a - term + q - p).norm();

    assert_relative_eq!(residual, 0.0, epsilon = 1e-5);
}

#[test]
fn lqr_tracks_straight_line() {
    let mut path = Path::default();
    path.waypoints = (0..40).map(|i| pose_at(i as f64 * 0.4, 0.0, 0.0)).collect();
    let final_wp = *path.waypoints.last().unwrap();

    let mut tracker = Tracker::new(TrackerKind::Lqr);
    let mut cfg = ControllerConfig::default();
    cfg.goal_tolerance = 0.4;
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
        tolerance_position: 0.4,
        tolerance_orientation: 1.0,
        ..Default::default()
    });

    // Start offset laterally: LQR should pull us back onto the line.
    let mut state = RobotState {
        pose: pose_at(0.0, 0.3, 0.0),
        allow_move: true,
        ..Default::default()
    };

    let dt = 0.05;
    let mut max_cte: f64 = 0.0;
    let mut reached = false;
    let mut t = 0.0;
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
        // After a brief settle window, CTE should stay small.
        if t > 2.0 {
            max_cte = max_cte.max(tracker.get_status().cross_track_error);
        }
        if tracker.get_status().goal_reached {
            reached = true;
            break;
        }
    }
    assert!(reached, "LQR did not reach goal (t={t:.2})");
    assert!(max_cte < 0.4, "post-settle CTE too large: {max_cte:.3}");
}
