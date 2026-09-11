//! Pose regulator demo: a differential robot regulating onto successive waypoint poses.

#[path = "common/viz.rs"]
mod viz;
#[path = "common/demo.rs"]
mod demo;

use ondrive::{SteeringType, TrackerKind};


fn main() {

    demo::run(demo::Demo {
        name: "pose_regulator",
        kind: TrackerKind::PoseRegulator,
        path: demo::path_through(&[(0.0, 0.0), (3.0, 1.0), (3.0, 4.0), (0.0, 4.0)]),
        constraints: demo::constraints(SteeringType::Differential, 0.8),
        config: demo::config(),
        start: demo::pose(0.0, 0.0, 0.0),
        world: None,
        dt: 0.05,
        max_time: 120.0,
    });
}
