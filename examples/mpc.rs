//! MPC demo: an Ackermann car on a sine path, with the planned horizon
//! drawn in magenta.

#[path = "common/viz.rs"]
mod viz;
#[path = "common/demo.rs"]
mod demo;

use ondrive::{SteeringType, TrackerKind};

fn main() {
    demo::run(demo::Demo {
        name: "mpc",
        kind: TrackerKind::Mpc,
        path: demo::sine(16.0, 0.8, 0.4),
        constraints: demo::constraints(SteeringType::Ackermann, 1.0),
        config: demo::config(),
        start: demo::pose(0.0, 0.3, 0.0),
        world: None,
        dt: 0.1,
        max_time: 120.0,
    });
}
