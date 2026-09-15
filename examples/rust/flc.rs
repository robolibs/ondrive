//! FLC demo: a differential robot starting 0.5 m off a sine path.

#[path = "common/viz.rs"]
mod viz;
#[path = "common/demo.rs"]
mod demo;

use ondrive::{SteeringType, TrackerKind};

fn main() {
    demo::run(demo::Demo {
        name: "flc",
        kind: TrackerKind::Flc,
        path: demo::sine(18.0, 1.0, 0.4),
        constraints: demo::constraints(SteeringType::Differential, 0.8),
        config: demo::config(),
        start: demo::pose(0.0, 0.5, 0.0),
        world: None,
        dt: 0.05,
        max_time: 120.0,
    });
}
