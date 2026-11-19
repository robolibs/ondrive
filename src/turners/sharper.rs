use super::geometry::{Point, Point2Ext, angle_difference, normalize_angle, point_distance, point_xy};

use super::Pose2D;

#[derive(Debug, Clone, PartialEq)]
pub struct SharpTurnPath {
    pub waypoints: Vec<Pose2D>,
    pub segment_types: Vec<String>,
    pub total_length: f64,
    pub pattern_name: String,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sharper {
    radius: f64,
    machine_length: f64,
    machine_width: f64,
}

impl Sharper {
    pub fn new(min_turning_radius: f64, machine_length: f64, machine_width: f64) -> Self {
        Self {
            radius: min_turning_radius,
            machine_length,
            machine_width: if machine_width > 0.0 {
                machine_width
            } else {
                machine_length * 0.4
            },
        }
    }

    pub fn plan_sharp_turn_at_point(
        &self,
        turning_point: Point,
        old_heading: f64,
        new_heading: f64,
        pattern: &str,
    ) -> SharpTurnPath {
        self.plan_sharp_turn(
            Pose2D::from_point(turning_point, old_heading),
            Pose2D::from_point(turning_point, new_heading),
            pattern,
        )
    }

    pub fn plan_sharp_turn(&self, start: Pose2D, end: Pose2D, pattern: &str) -> SharpTurnPath {
        let turn_angle = angle_difference(start.yaw, end.yaw);
        match pattern {
            "three_point" => self.generate_three_point_turn(start, end),
            "bulb" => self.generate_bulb_turn(start, end, turn_angle),
            "fishtail" => self.generate_fishtail_turn(start, end, turn_angle),
            _ => self.select_best_pattern(start, end, turn_angle),
        }
    }

    fn select_best_pattern(&self, start: Pose2D, end: Pose2D, turn_angle: f64) -> SharpTurnPath {
        let angle_deg = turn_angle.abs().to_degrees();
        let dist = point_distance(start.point, end.point);
        let same_point_turn = dist < 0.1 * self.machine_length;

        if same_point_turn || angle_deg > 120.0 {
            self.generate_three_point_turn(start, end)
        } else if angle_deg > 60.0 {
            self.generate_bulb_turn(start, end, turn_angle)
        } else {
            self.generate_fishtail_turn(start, end, turn_angle)
        }
    }

    fn generate_three_point_turn(&self, start: Pose2D, end: Pose2D) -> SharpTurnPath {
        let forward = Pose2D::from_point(
            point_xy(
                start.point.x() + self.machine_length * start.yaw.cos(),
                start.point.y() + self.machine_length * start.yaw.sin(),
            ),
            start.yaw,
        );
        let reverse = Pose2D::from_point(
            point_xy(
                start.point.x() - self.machine_length * end.yaw.cos(),
                start.point.y() - self.machine_length * end.yaw.sin(),
            ),
            end.yaw,
        );
        finish_path(
            "three_point",
            vec![start, forward, reverse, end],
            vec![
                "start".to_string(),
                "forward_along_old_heading".to_string(),
                "reverse_with_turn".to_string(),
                "forward_to_turning_point".to_string(),
            ],
        )
    }

    fn generate_bulb_turn(&self, start: Pose2D, end: Pose2D, turn_angle: f64) -> SharpTurnPath {
        let mut waypoints = vec![start];
        let mut segment_types = vec!["start".to_string()];
        let bulb_radius = self.machine_length * 0.8;
        let count = 5;
        for i in 1..=count {
            let t = i as f64 / (count + 1) as f64;
            let current_angle = normalize_angle(start.yaw + turn_angle * t);
            let bulb_factor = (std::f64::consts::PI * t).sin() * bulb_radius;
            let perpendicular = current_angle + std::f64::consts::FRAC_PI_2;
            let point = point_xy(
                start.point.x()
                    + self.radius * t * current_angle.cos()
                    + bulb_factor * perpendicular.cos(),
                start.point.y()
                    + self.radius * t * current_angle.sin()
                    + bulb_factor * perpendicular.sin(),
            );
            waypoints.push(Pose2D::from_point(point, current_angle));
            segment_types.push("bulb_arc".to_string());
        }
        waypoints.push(end);
        segment_types.push("end".to_string());
        finish_path("bulb", waypoints, segment_types)
    }

    fn generate_fishtail_turn(&self, start: Pose2D, end: Pose2D, turn_angle: f64) -> SharpTurnPath {
        let extend = self.machine_length * 1.2;
        let approach = Pose2D::from_point(
            point_xy(
                start.point.x() + 0.5 * self.machine_length * start.yaw.cos(),
                start.point.y() + 0.5 * self.machine_length * start.yaw.sin(),
            ),
            start.yaw,
        );
        let tail_angle = normalize_angle(start.yaw - turn_angle * 0.4);
        let tail = Pose2D::from_point(
            point_xy(
                approach.point.x() + extend * tail_angle.cos(),
                approach.point.y() + extend * tail_angle.sin(),
            ),
            tail_angle,
        );
        let transition_angle = normalize_angle(start.yaw + turn_angle * 0.7);
        let transition = Pose2D::from_point(
            point_xy(
                tail.point.x() + self.machine_length * transition_angle.cos(),
                tail.point.y() + self.machine_length * transition_angle.sin(),
            ),
            transition_angle,
        );
        let final_approach = Pose2D::from_point(
            point_xy(
                end.point.x() - 0.5 * self.machine_length * end.yaw.cos(),
                end.point.y() - 0.5 * self.machine_length * end.yaw.sin(),
            ),
            end.yaw,
        );

        finish_path(
            "fishtail",
            vec![start, approach, tail, transition, final_approach, end],
            vec![
                "start".to_string(),
                "approach".to_string(),
                "tail_out".to_string(),
                "transition".to_string(),
                "final_approach".to_string(),
                "end".to_string(),
            ],
        )
    }
}

fn finish_path(
    pattern_name: &str,
    waypoints: Vec<Pose2D>,
    segment_types: Vec<String>,
) -> SharpTurnPath {
    let total_length = waypoints
        .windows(2)
        .map(|pair| point_distance(pair[0].point, pair[1].point))
        .sum();
    SharpTurnPath {
        waypoints,
        segment_types,
        total_length,
        pattern_name: pattern_name.to_string(),
    }
}
