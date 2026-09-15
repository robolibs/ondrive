//! MCA demo: an Ackermann car on a straight path with two obstacles on and
//! beside it. Obstacles are red discs, the plan is magenta.

#[path = "common/viz.rs"]
mod viz;
#[path = "common/demo.rs"]
mod demo;

use ondrive::{SteeringType, TrackerKind};

fn static_obstacle(x: f64, y: f64, radius: f64) -> ondrive::Obstacle {
    ondrive::Obstacle {
        id: 0,
        radius,
        modes: vec![ondrive::GaussianMode {
            weight: 1.0,
            mean_x: vec![x; 40],
            mean_y: vec![y; 40],
            std_x: vec![0.05; 40],
            std_y: vec![0.05; 40],
        }],
    }
}

fn main() {
    let world = ondrive::WorldConstraints {
        obstacles: vec![static_obstacle(6.0, 0.3, 0.3), static_obstacle(11.0, -0.4, 0.3)],
        ..Default::default()
    };
    demo::run(demo::Demo {
        name: "mca",
        kind: TrackerKind::Mca,
        path: demo::curve(|s| (s, 0.0), 16.0, 0.25),
        constraints: demo::constraints(SteeringType::Ackermann, 1.0),
        config: demo::config(),
        start: demo::pose(0.0, 0.0, 0.0),
        world: Some(world),
        dt: 0.1,
        max_time: 120.0,
    });
}
