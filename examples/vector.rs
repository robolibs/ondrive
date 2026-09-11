//! Vector Pursuit demo: converging from a 1 m offset with orientation-aware pursuit.

#[path = "common/viz.rs"]
mod viz;
#[path = "common/demo.rs"]
mod demo;

use ondrive::{SteeringType, TrackerKind};


fn main() {

    demo::run(demo::Demo {
        name: "vector",
        kind: TrackerKind::VectorPursuit,
        path: demo::sine(20.0, 1.2, 0.35),
        constraints: demo::constraints(SteeringType::Ackermann, 1.0),
        config: demo::config(),
        start: demo::pose(0.0, 1.0, 0.0),
        world: None,
        dt: 0.05,
        max_time: 120.0,
    });
}
