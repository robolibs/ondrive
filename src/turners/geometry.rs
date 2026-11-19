//! Plane geometry the turners work in: `datapod` points read by their x
//! and y, distances, and angles wrapped to (−π, π].

pub use datapod::Point;

/// The plane coordinates of a point.
pub trait Point2Ext {
    fn x(&self) -> f64;
    fn y(&self) -> f64;
}

impl Point2Ext for Point {
    fn x(&self) -> f64 {
        self.x
    }

    fn y(&self) -> f64 {
        self.y
    }
}

/// The point at `x`, `y` on the ground plane.
pub fn point_xy(x: f64, y: f64) -> Point {
    Point::new(x, y, 0.0)
}

/// Distance between `a` and `b` in the plane.
pub fn point_distance(a: Point, b: Point) -> f64 {
    (a.x - b.x).hypot(a.y - b.y)
}

/// Whether `a` and `b` lie within `epsilon` of each other in the plane.
pub fn points_equal(a: Point, b: Point, epsilon: f64) -> bool {
    point_distance(a, b) < epsilon
}

/// `angle` wrapped to (−π, π].
pub fn normalize_angle(mut angle: f64) -> f64 {
    while angle > std::f64::consts::PI {
        angle -= 2.0 * std::f64::consts::PI;
    }
    while angle < -std::f64::consts::PI {
        angle += 2.0 * std::f64::consts::PI;
    }
    angle
}

/// The turn from heading `from` to heading `to`, wrapped to (−π, π].
pub fn angle_difference(from: f64, to: f64) -> f64 {
    normalize_angle(to - from)
}
