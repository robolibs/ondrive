//! Carrot demo: a differential robot chasing zig-zag waypoints.

#[path = "common/viz.rs"]
mod viz;
#[path = "common/demo.rs"]
mod demo;

use ondrive::{SteeringType, TrackerKind};

fn main() {
    let path = demo::path_through(&[(0.0, 0.0), (3.0, 2.0), (6.0, 0.0), (9.0, 2.0), (12.0, 0.0)]);
    demo::run(demo::Demo {
        name: "carrot",
        kind: TrackerKind::Carrot,
        path,
        constraints: demo::constraints(SteeringType::Differential, 0.6),
        config: demo::config(),
        start: demo::pose(0.0, 0.0, 0.0),
        world: None,
        dt: 0.05,
        max_time: 120.0,
    });
}
