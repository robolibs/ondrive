//! PID point-to-point demo: a differential robot walked through sparse
//! waypoints by the tracker. Streams to a running rerun viewer.

#[path = "common/viz.rs"]
mod viz;
#[path = "common/demo.rs"]
mod demo;

use ondrive::{SteeringType, TrackerKind};

fn main() {
    let path = demo::path_through(&[(0.0, 0.0), (3.0, 0.5), (5.0, 2.5), (8.0, 3.0), (10.0, 5.0)]);
    demo::run(demo::Demo {
        name: "pid",
        kind: TrackerKind::Pid,
        path,
        constraints: demo::constraints(SteeringType::Differential, 0.6),
        config: demo::config(),
        start: demo::pose(0.0, 0.0, 0.0),
        world: None,
        dt: 0.05,
        max_time: 120.0,
    });
}
