use crate::controller::{Controller, ControllerBase, is_goal_reached};
use crate::core::math::normalize_angle;
use crate::types::{
    Goal, OutputUnits, RobotConstraints, RobotState, SteeringType, VelocityCommand,
    WorldConstraints,
};
use datapod::Point;
use std::f64::consts::PI;

#[derive(Clone, Debug, Default)]
pub struct PurePursuitFollower {
    pub base: ControllerBase,
}

impl PurePursuitFollower {
    pub fn new() -> Self {
        Self::default()
    }

    fn find_lookahead_point(
        &mut self,
        rear_axle: Point,
        lookahead: f64,
        progress: f64,
    ) -> Option<Point> {
        let waypoints = &self.base.path.waypoints;
        if waypoints.is_empty() {
            return None;
        }

        while self.base.path_index + 1 < waypoints.len() {
            let current = waypoints[self.base.path_index].point;
            let next = waypoints[self.base.path_index + 1].point;

            let dist_to_current = rear_axle.distance_to(current);
            let dx_to_current = current.x - rear_axle.x;
            let dy_to_current = current.y - rear_axle.y;
            let dx_path = next.x - current.x;
            let dy_path = next.y - current.y;
            let dot = dx_to_current * dx_path + dy_to_current * dy_path;

            if dist_to_current < progress || dot < 0.0 {
                self.base.path_index += 1;
            } else {
                break;
            }
        }

        let mut min_diff = f64::MAX;
        let mut best: Option<Point> = None;

        for i in self.base.path_index..waypoints.len() {
            let wp = waypoints[i].point;
            let dist = rear_axle.distance_to(wp);
            let diff = (dist - lookahead).abs();
            if diff < min_diff && dist >= lookahead * 0.5 {
                min_diff = diff;
                best = Some(wp);
            }

            if i + 1 < waypoints.len() {
                let next = waypoints[i + 1].point;
                if let Some(intersection) =
                    find_circle_segment_intersection(rear_axle, lookahead, wp, next)
                {
                    let d = rear_axle.distance_to(intersection);
                    let diff = (d - lookahead).abs();
                    if diff < min_diff {
                        min_diff = diff;
                        best = Some(intersection);
                    }
                }
            }

            if dist > lookahead * 2.0 {
                break;
            }
        }

        if best.is_none() && self.base.path_index < waypoints.len() {
            best = Some(waypoints.last().unwrap().point);
        }
        best
    }
}

fn find_circle_segment_intersection(
    center: Point,
    radius: f64,
    start: Point,
    end: Point,
) -> Option<Point> {
    let mut dx = end.x - start.x;
    let mut dy = end.y - start.y;
    let seg_len = dx.hypot(dy);
    if seg_len < 1e-6 {
        return None;
    }
    dx /= seg_len;
    dy /= seg_len;

    let fx = center.x - start.x;
    let fy = center.y - start.y;
    let projection = fx * dx + fy * dy;

    let closest_x = start.x + projection * dx;
    let closest_y = start.y + projection * dy;
    let dist_to_line = (center.x - closest_x).hypot(center.y - closest_y);
    if dist_to_line > radius {
        return None;
    }

    let half_chord = (radius * radius - dist_to_line * dist_to_line).sqrt();
    let t1 = projection - half_chord;
    let t2 = projection + half_chord;

    if t2 >= 0.0 && t2 <= seg_len {
        Some(Point::new(start.x + t2 * dx, start.y + t2 * dy, 0.0))
    } else if t1 >= 0.0 && t1 <= seg_len {
        Some(Point::new(start.x + t1 * dx, start.y + t1 * dy, 0.0))
    } else {
        None
    }
}

impl Controller for PurePursuitFollower {
    fn compute_control(
        &mut self,
        state: &RobotState,
        goal: &Goal,
        constraints: &RobotConstraints,
        _dt: f64,
        _world: Option<&WorldConstraints>,
    ) -> VelocityCommand {
        let cfg = self.base.config.clone();

        let wheelbase = if constraints.wheelbase > 0.0 {
            constraints.wheelbase
        } else {
            1.0
        };

        let yaw = state.pose.rotation.to_euler().yaw;
        let rear_x = state.pose.point.x - (wheelbase / 2.0) * yaw.cos();
        let rear_y = state.pose.point.y - (wheelbase / 2.0) * yaw.sin();
        let rear_axle = Point::new(rear_x, rear_y, 0.0);

        let k_lookahead = 0.1;
        let base_lookahead = cfg.lookahead_distance;
        let lookahead = (k_lookahead * state.velocity.linear.abs() + base_lookahead)
            .clamp(base_lookahead, base_lookahead * 3.0);

        let progress = wheelbase * 1.5;
        let has_path = !self.base.path.waypoints.is_empty();
        let target_point = if has_path {
            self.find_lookahead_point(rear_axle, lookahead, progress)
                .unwrap_or(goal.target_pose.point)
        } else {
            goal.target_pose.point
        };

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

        let is_diff = matches!(
            constraints.steering_type,
            SteeringType::Differential | SteeringType::SkidSteer
        );

        let dx = target_point.x - rear_x;
        let dy = target_point.y - rear_y;
        let alpha = normalize_angle(dy.atan2(dx) - yaw);

        self.base.status.distance_to_goal =
            state.pose.point.distance_to(goal.target_pose.point);
        self.base.status.cross_track_error = (alpha.sin() * dx.hypot(dy)).abs();
        self.base.status.goal_reached = false;
        self.base.status.mode = "pure_pursuit".into();

        let kp_angular = 2.5;

        if is_diff && state.turn_first && alpha.abs() > cfg.angular_tolerance {
            let angular_physical = (kp_angular * alpha).clamp(
                -constraints.max_angular_velocity,
                constraints.max_angular_velocity,
            );
            let (linear, angular) = match cfg.output_units {
                OutputUnits::Normalized => (0.0, angular_physical / constraints.max_angular_velocity),
                OutputUnits::Physical => (0.0, angular_physical),
            };
            self.base.status.mode = "turning".into();
            return VelocityCommand {
                valid: true,
                status_message: "Turning to align".into(),
                linear_velocity: linear,
                angular_velocity: angular,
                ..VelocityCommand::default()
            };
        }

        let angular_physical = (kp_angular * alpha).clamp(
            -constraints.max_angular_velocity,
            constraints.max_angular_velocity,
        );

        let alpha_mag = alpha.abs();
        let linear_physical = if alpha_mag > PI * 0.66 {
            constraints.max_linear_velocity * 0.3
        } else if alpha_mag > PI / 3.0 {
            constraints.max_linear_velocity * 0.6
        } else {
            constraints.max_linear_velocity
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
            status_message: "Following path".into(),
            linear_velocity: linear,
            angular_velocity: angular,
            ..VelocityCommand::default()
        }
    }

    fn get_type(&self) -> &'static str {
        "pure_pursuit_follower"
    }

    fn base(&self) -> &ControllerBase {
        &self.base
    }

    fn base_mut(&mut self) -> &mut ControllerBase {
        &mut self.base
    }
}
