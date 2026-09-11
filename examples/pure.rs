//! Pure Pursuit demo: an Ackermann car on a sine path.

#[path = "common/viz.rs"]
mod viz;
#[path = "common/demo.rs"]
mod demo;

use ondrive::{SteeringType, TrackerKind};

fn main() {
    let mut config = demo::config();
    config.lookahead_distance = 1.2;
    demo::run(demo::Demo {
        name: "pure",
        kind: TrackerKind::PurePursuit,
        path: demo::sine(20.0, 1.2, 0.35),
        constraints: demo::constraints(SteeringType::Ackermann, 1.0),
        config,
        start: demo::pose(0.0, 0.0, 0.0),
        world: None,
        dt: 0.05,
        max_time: 120.0,
    });
}
