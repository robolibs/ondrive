use super::geometry::{Point, point_xy};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pose2D {
    pub point: Point,
    pub yaw: f64,
}

impl Pose2D {
    pub fn new(x: f64, y: f64, yaw: f64) -> Self {
        Self {
            point: point_xy(x, y),
            yaw,
        }
    }

    pub fn from_point(point: Point, yaw: f64) -> Self {
        Self { point, yaw }
    }
}
