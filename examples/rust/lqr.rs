//! LQR demo: an Ackermann car starting 0.4 m off a sine path.

#[path = "common/viz.rs"]
mod viz;
#[path = "common/demo.rs"]
mod demo;

use ondrive::{SteeringType, TrackerKind};

fn main() {
    demo::run(demo::Demo {
        name: "lqr",
        kind: TrackerKind::Lqr,
        path: demo::sine(20.0, 1.0, 0.4),
        constraints: demo::constraints(SteeringType::Ackermann, 0.6),
        config: demo::config(),
        start: demo::pose(0.0, 0.4, 0.0),
        world: None,
        dt: 0.05,
        max_time: 120.0,
    });
}
