//! DWA demo: a differential robot walked through waypoints with an obstacle
//! sitting between two of them.

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
        obstacles: vec![static_obstacle(6.0, 1.5, 0.3)],
        ..Default::default()
    };
    demo::run(demo::Demo {
        name: "dwa",
        kind: TrackerKind::Dwa,
        path: demo::path_through(&[(0.0, 0.0), (4.0, 0.0), (8.0, 3.0), (12.0, 3.0)]),
        constraints: demo::constraints(SteeringType::Differential, 1.0),
        config: demo::config(),
        start: demo::pose(0.0, 0.0, 0.0),
        world: Some(world),
        dt: 0.1,
        max_time: 120.0,
    });
}
