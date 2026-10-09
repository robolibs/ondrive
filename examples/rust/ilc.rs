//! Iterative learning demo: Pure Pursuit wrapped in ILC on a 2.5 m circle (one pass; call set_path again to learn).

#[path = "common/viz.rs"]
mod viz;
#[path = "common/demo.rs"]
mod demo;

use ondrive::{SteeringType, TrackerKind};


fn main() {

    demo::run(demo::Demo {
        name: "ilc",
        kind: TrackerKind::Ilc,
        path: demo::curve(|s| { let a = s / 2.5; (2.5 * a.sin(), 2.5 * (1.0 - a.cos())) }, 2.5 * 1.5 * std::f64::consts::PI, 0.25),
        constraints: demo::constraints(SteeringType::Ackermann, 1.0),
        config: demo::config(),
        start: demo::pose(0.0, 0.0, 0.0),
        world: None,
        dt: 0.05,
        max_time: 120.0,
    });
}
