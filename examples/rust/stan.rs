//! Stanley demo: an Ackermann car starting 0.6 m left of a sine path.

#[path = "common/viz.rs"]
mod viz;
#[path = "common/demo.rs"]
mod demo;

use ondrive::{SteeringType, TrackerKind};

fn main() {
    demo::run(demo::Demo {
        name: "stanley",
        kind: TrackerKind::Stanley,
        path: demo::sine(20.0, 1.2, 0.35),
        constraints: demo::constraints(SteeringType::Ackermann, 0.7),
        config: demo::config(),
        start: demo::pose(0.0, 0.6, 0.0),
        world: None,
        dt: 0.05,
        max_time: 120.0,
    });
}
