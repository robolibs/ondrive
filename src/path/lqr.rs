use crate::controller::{Controller, ControllerBase, is_goal_reached};
use crate::core::math::normalize_angle;
use crate::types::{
    Goal, OutputUnits, RobotConstraints, RobotState, SteeringType, VelocityCommand,
    WorldConstraints,
};
use datapod::Point;
use nalgebra::{Matrix4, Vector4};

const DARE_MAX_ITERATIONS: usize = 150;
const DARE_TOLERANCE: f64 = 1e-5;

fn solve_dare(a: &Matrix4<f64>, b: &Vector4<f64>, q: &Matrix4<f64>, r: f64) -> Matrix4<f64> {
    let mut p = *q;
    let at = a.transpose();

    for _ in 0..DARE_MAX_ITERATIONS {
        let at_p = at * p;
        let at_p_a = at_p * a;

        let pb = p * b;
        let bt_p_b = b.dot(&pb);
        let mut denom = r + bt_p_b;
        if denom.abs() < 1e-9 {
            denom = if denom >= 0.0 { 1e-9 } else { -1e-9 };
        }
        let inv_denom = 1.0 / denom;

        let at_p_b = at_p * b;
        let bt_p_a = (b.transpose() * p * a).transpose();

        let term = at_p_b * bt_p_a.transpose() * inv_denom;

        let p_next = at_p_a - term + q;

        let diff_norm = (p_next - p).norm();
        p = p_next;

        if diff_norm < DARE_TOLERANCE {
            break;
        }
    }

    p
}

fn compute_lqr_gain(
    velocity: f64,
    constraints: &RobotConstraints,
    dt: f64,
    is_diff_drive: bool,
) -> Vector4<f64> {
    let a = Matrix4::new(
        1.0, dt, 0.0, 0.0,
        0.0, 1.0, velocity, 0.0,
        0.0, 0.0, 1.0, dt,
        0.0, 0.0, 0.0, 1.0,
    );

    let b_last = if is_diff_drive {
        1.0
    } else {
        velocity / constraints.wheelbase
    };
    let b = Vector4::new(0.0, 0.0, 0.0, b_last);

    let mut q = Matrix4::zeros();
    q[(0, 0)] = 1.0;
    q[(2, 2)] = 0.5;

    let r = 0.5;
    let p = solve_dare(&a, &b, &q, r);

    let pb = p * b;
    let bt_p_b = b.dot(&pb);
    let mut denom = r + bt_p_b;
    if denom.abs() < 1e-9 {
        denom = if denom >= 0.0 { 1e-9 } else { -1e-9 };
    }
    let inv_denom = 1.0 / denom;

    let k_row = (b.transpose() * p * a) * inv_denom;
    Vector4::new(k_row[0], k_row[1], k_row[2], k_row[3])
}

struct PathError {
    nearest_index: usize,
    nearest_point: Point,
    path_heading: f64,
    lateral_error: f64,
    heading_error: f64,
    path_curvature: f64,
}

#[derive(Clone, Debug, Default)]
pub struct LqrFollower {
    pub base: ControllerBase,
    previous_lateral_error: f64,
    previous_heading_error: f64,
}

impl LqrFollower {
    pub fn new() -> Self {
        Self::default()
    }

    fn calculate_path_error(&mut self, state: &RobotState) -> PathError {
        let waypoints = &self.base.path.waypoints;
        let mut min_distance = f64::MAX;
        let mut nearest_idx = self.base.path_index;

        for i in self.base.path_index..waypoints.len() {
            let dist = state.pose.point.distance_to(waypoints[i].point);
            if dist < min_distance {
                min_distance = dist;
                nearest_idx = i;
            }
            if i > self.base.path_index && dist > min_distance * 1.5 {
                break;
            }
        }

        self.base.path_index = nearest_idx;
        let nearest_point = waypoints[nearest_idx].point;

        let path_heading = if nearest_idx + 1 < waypoints.len() {
            let next = waypoints[nearest_idx + 1].point;
            (next.y - nearest_point.y).atan2(next.x - nearest_point.x)
        } else {
            waypoints[nearest_idx].rotation.to_euler().yaw
        };

        let dx = state.pose.point.x - nearest_point.x;
        let dy = state.pose.point.y - nearest_point.y;
        let lateral_error = -dx * path_heading.sin() + dy * path_heading.cos();
        let heading_error =
            normalize_angle(state.pose.rotation.to_euler().yaw - path_heading);

        let path_curvature = if nearest_idx > 0 && nearest_idx + 1 < waypoints.len() {
            let prev = waypoints[nearest_idx - 1].point;
            let curr = waypoints[nearest_idx].point;
            let next = waypoints[nearest_idx + 1].point;
            let heading1 = (curr.y - prev.y).atan2(curr.x - prev.x);
            let heading2 = (next.y - curr.y).atan2(next.x - curr.x);
            let heading_change = normalize_angle(heading2 - heading1);
            let arc_length = prev.distance_to(curr) + curr.distance_to(next);
            heading_change / (arc_length / 2.0)
        } else {
            0.0
        };

        PathError {
            nearest_index: nearest_idx,
            nearest_point,
            path_heading,
            lateral_error,
            heading_error,
            path_curvature,
        }
    }
}

impl Controller for LqrFollower {
    fn compute_control(
        &mut self,
        state: &RobotState,
        goal: &Goal,
        constraints: &RobotConstraints,
        dt: f64,
        _world: Option<&WorldConstraints>,
    ) -> VelocityCommand {
        if self.base.path.waypoints.is_empty() {
            return VelocityCommand::invalid("no path");
        }

        let cfg = self.base.config.clone();
        let (reached, dist, yaw_diff) = is_goal_reached(
            &state.pose,
            &goal.target_pose,
            cfg.goal_tolerance,
            cfg.angular_tolerance,
        );
        self.base.status.distance_to_goal = dist;
        self.base.status.heading_error = yaw_diff;

        if reached {
            self.base.status.goal_reached = true;
            self.base.status.mode = "stopped".into();
            return VelocityCommand {
                valid: true,
                status_message: "Goal reached".into(),
                ..VelocityCommand::default()
            };
        }

        let error = self.calculate_path_error(state);
        let _ = error.nearest_index;
        let _ = error.nearest_point;
        let _ = error.path_heading;

        let lateral_error_rate = (error.lateral_error - self.previous_lateral_error) / dt;
        let heading_error_rate = (error.heading_error - self.previous_heading_error) / dt;
        self.previous_lateral_error = error.lateral_error;
        self.previous_heading_error = error.heading_error;

        let is_diff = matches!(
            constraints.steering_type,
            SteeringType::Differential | SteeringType::SkidSteer
        );

        if is_diff && state.turn_first && error.heading_error.abs() > cfg.angular_tolerance {
            let kp_angular = 2.0;
            let angular_control = (kp_angular * error.heading_error).clamp(
                -constraints.max_angular_velocity,
                constraints.max_angular_velocity,
            );

            self.base.status.distance_to_goal =
                state.pose.point.distance_to(goal.target_pose.point);
            self.base.status.cross_track_error = error.lateral_error.abs();
            self.base.status.heading_error = error.heading_error.abs();
            self.base.status.goal_reached = false;
            self.base.status.mode = "turning".into();

            let (linear, angular) = match cfg.output_units {
                OutputUnits::Normalized => {
                    (0.0, angular_control / constraints.max_angular_velocity)
                }
                OutputUnits::Physical => (0.0, angular_control),
            };

            return VelocityCommand {
                valid: true,
                status_message: "Turning to align".into(),
                linear_velocity: linear,
                angular_velocity: angular,
                ..VelocityCommand::default()
            };
        }

        let mut velocity = state.velocity.linear;
        if velocity.abs() < 0.01 {
            velocity = 0.1;
        }

        let k = compute_lqr_gain(velocity, constraints, dt, is_diff);
        let state_error = Vector4::new(
            error.lateral_error,
            lateral_error_rate,
            error.heading_error,
            heading_error_rate,
        );
        let feedback_control = -k.dot(&state_error);

        self.base.status.distance_to_goal =
            state.pose.point.distance_to(goal.target_pose.point);
        self.base.status.cross_track_error = error.lateral_error.abs();
        self.base.status.heading_error = error.heading_error.abs();
        self.base.status.goal_reached = false;
        self.base.status.mode = "lqr".into();

        let linear_physical = constraints.max_linear_velocity;

        let angular_physical = if is_diff {
            let feedforward_omega = velocity * error.path_curvature;
            (feedforward_omega + feedback_control).clamp(
                -constraints.max_angular_velocity,
                constraints.max_angular_velocity,
            )
        } else {
            let feedforward_steering =
                (constraints.wheelbase * error.path_curvature).atan2(1.0);
            let steering_angle = (feedforward_steering + feedback_control).clamp(
                -constraints.max_steering_angle,
                constraints.max_steering_angle,
            );
            let kp_steer = 2.0;
            (kp_steer * steering_angle).clamp(
                -constraints.max_angular_velocity,
                constraints.max_angular_velocity,
            )
        };

        let (linear, angular) = match cfg.output_units {
            OutputUnits::Normalized => (
                linear_physical / constraints.max_linear_velocity,
                angular_physical / constraints.max_angular_velocity,
            ),
            OutputUnits::Physical => (linear_physical, angular_physical),
        };

        VelocityCommand {
            valid: true,
            status_message: "LQR tracking".into(),
            linear_velocity: linear,
            angular_velocity: angular,
            ..VelocityCommand::default()
        }
    }

    fn reset(&mut self) {
        self.base.path.waypoints.clear();
        self.base.path_index = 0;
        self.base.status = Default::default();
        self.previous_lateral_error = 0.0;
        self.previous_heading_error = 0.0;
    }

    fn get_type(&self) -> &'static str {
        "lqr_follower"
    }

    fn base(&self) -> &ControllerBase {
        &self.base
    }

    fn base_mut(&mut self) -> &mut ControllerBase {
        &mut self.base
    }
}
