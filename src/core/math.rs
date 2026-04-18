use datapod::{Point, Pose};
use std::f64::consts::PI;

pub const TAU: f64 = 2.0 * PI;

pub fn normalize_angle(mut a: f64) -> f64 {
    while a > PI {
        a -= TAU;
    }
    while a < -PI {
        a += TAU;
    }
    a
}

pub fn distance(a: Point, b: Point) -> f64 {
    a.distance_to(b)
}

pub fn distance_2d(a: Point, b: Point) -> f64 {
    a.distance_to_2d(b)
}

pub fn yaw_of(pose: &Pose) -> f64 {
    pose.rotation.to_euler().yaw
}

pub fn heading_error(current: &Pose, target: Point) -> f64 {
    let dx = target.x - current.point.x;
    let dy = target.y - current.point.y;
    let desired = dy.atan2(dx);
    normalize_angle(desired - yaw_of(current))
}

pub fn yaw_error(current: &Pose, target: &Pose) -> f64 {
    normalize_angle(yaw_of(target) - yaw_of(current))
}
