//! Kanayama demo: a differential robot on a sine path (spatial reference).

#[path = "common/viz.rs"]
mod viz;
#[path = "common/demo.rs"]
mod demo;

use ondrive::{SteeringType, TrackerKind};


fn main() {

    demo::run(demo::Demo {
        name: "kanayama",
        kind: TrackerKind::Kanayama,
        path: demo::sine(18.0, 1.0, 0.4),
        constraints: demo::constraints(SteeringType::Differential, 0.8),
        config: demo::config(),
        start: demo::pose(0.0, 0.5, 0.0),
        world: None,
        dt: 0.05,
        max_time: 120.0,
    });
}
