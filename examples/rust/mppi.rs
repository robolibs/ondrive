//! MPPI demo: a differential robot on a sine path, with the sampled plan
//! drawn in magenta.

#[path = "common/viz.rs"]
mod viz;
#[path = "common/demo.rs"]
mod demo;

use ondrive::{SteeringType, TrackerKind};

fn main() {
    demo::run(demo::Demo {
        name: "mppi",
        kind: TrackerKind::Mppi,
        path: demo::sine(20.0, 1.2, 0.35),
        constraints: demo::constraints(SteeringType::Differential, 1.0),
        config: demo::config(),
        start: demo::pose(0.0, 0.2, 0.0),
        world: None,
        dt: 0.1,
        max_time: 120.0,
    });
}
