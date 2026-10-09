//! Pose reach demo: an Ackermann car planning Reeds-Shepp curves to successive waypoint poses.

#[path = "common/viz.rs"]
mod viz;
#[path = "common/demo.rs"]
mod demo;

use ondrive::{SteeringType, TrackerKind};


fn main() {

    demo::run(demo::Demo {
        name: "pose_reach",
        kind: TrackerKind::PoseReach,
        path: demo::path_through(&[(0.0, 0.0), (3.0, 1.5), (5.0, 1.5), (6.0, -1.0)]),
        constraints: demo::constraints(SteeringType::Ackermann, 0.6),
        config: demo::config(),
        start: demo::pose(0.0, 0.0, 0.0),
        world: None,
        dt: 0.05,
        max_time: 120.0,
    });
}
