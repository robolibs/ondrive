# ondrive

`ondrive` is a Rust motion-control and path-tracking library for mobile
robots. A single `Tracker` facade dispatches to twelve interchangeable
controllers across four algorithm families, all operating on the same
`datapod::spatial` pose and velocity types.

## Algorithms

| Family              | Controller         | Notes                                                           |
|---------------------|--------------------|-----------------------------------------------------------------|
| Point-to-point      | **PID**            | Dual-loop (distance + heading), deadband, anti-windup           |
|                     | **Carrot**         | Proportional with speed-vs-turn scaling                         |
| Path following      | **Pure Pursuit**   | Adaptive lookahead, circle-segment intersection, Ackermann/diff |
|                     | **Stanley**        | Signed CTE + heading, reverse-motion support                    |
|                     | **LQR**            | Full DARE solver on `nalgebra::Matrix4`, curvature feedforward  |
| Predictive / optimal| **MPC**            | Projected gradient descent + Savitzky-Golay + decel taper       |
|                     | **MPPI**           | 1000-sample importance-weighted path integral, warm-start       |
|                     | **MCA** (DRA-MPPI) | MPPI + Monte Carlo collision probability over Gaussian modes    |
|                     | **SOC** (SVG-MPPI) | Stein Variational Gradient Descent + adaptive variance MPPI    |
|                     | **DWA**            | Fox 1997 dynamic-window grid search with obstacle rejection     |
|                     | **TEB**            | Timed Elastic Band with projected GD on poses + Δt_i            |
| Fuzzy               | **FLC**            | Mamdani with 7 triangular terms and a 49-rule additive base     |

All controllers accept the four standard kinematic models through
`SteeringType`: differential, Ackermann, holonomic, skid-steer. Obstacle-
aware controllers (MCA, DWA, TEB, SOC) consume `WorldConstraints` with
Gaussian-mode obstacle predictions.

## Install

```toml
[dependencies]
ondrive = { path = "../ondrive" }
```

The crate depends on `datapod` (geometry types) via a local path and
`stateup` from Codeberg.

## Quick start

```rust
use datapod::{Euler, Point, Pose, Quaternion};
use ondrive::{
    ControllerConfig, Goal, Path, RobotConstraints, RobotState, Tracker,
    TrackerKind, smoothen_path,
};

// Build a path.
let mut path = Path::default();
for i in 0..20 {
    path.waypoints.push(Pose {
        point: Point::new(i as f64 * 0.5, 0.0, 0.0),
        rotation: Quaternion::from_euler(Euler::new(0.0, 0.0, 0.0)),
    });
}
smoothen_path(&mut path, 0.25);

// Configure a Pure Pursuit tracker.
let mut tracker = Tracker::new(TrackerKind::PurePursuit);
let mut cfg = ControllerConfig::default();
cfg.lookahead_distance = 1.2;
cfg.goal_tolerance = 0.4;
cfg.output_units = ondrive::OutputUnits::Physical;
tracker.set_config(cfg);

let mut cons = RobotConstraints::default();
cons.max_linear_velocity = 1.0;
cons.max_angular_velocity = 2.0;
cons.wheelbase = 0.5;
tracker.init(cons);

let final_wp = *path.waypoints.last().unwrap();
tracker.set_path(path);
tracker.set_goal(Goal {
    target_pose: final_wp,
    tolerance_position: 0.4,
    tolerance_orientation: 1.0,
    ..Default::default()
});

// Tick.
let state = RobotState {
    pose: Pose {
        point: Point::new(0.0, 0.2, 0.0),
        rotation: Quaternion::default(),
    },
    allow_move: true,
    ..Default::default()
};
let cmd = tracker.tick(&state, 0.05, None);
assert!(cmd.valid);
```

## Tracker facade

The `Tracker` provides the high-level API used by every example and
binding:

| Method                           | Purpose                                            |
|----------------------------------|----------------------------------------------------|
| `new(kind)` / `with_config(...)` | Construct with a `TrackerKind`                     |
| `init(constraints)`              | Set robot constraints                              |
| `set_config(cfg)` / `get_config` | Controller config (gains, tolerances, units)       |
| `set_goal(g)` / `clear_goal`     | Target point/pose (all controllers)                |
| `set_path(p)` / `clear_path`     | Waypoint sequence (path-following controllers)     |
| `smoothen(max_segment_m)`        | Densify the active path (linear + slerp)           |
| `tick(state, dt, world)`         | Produce `VelocityCommand` for one control period   |
| `emergency_stop()`               | Clears state, returns zero command                 |
| `get_status()` / `is_goal_reached()` / `current_target()` | Telemetry accessors    |

Predictive controllers expose their specialised configs
(`MpcConfig`, `MppiConfig`, `McaConfig`, `SocConfig`, `DwaConfig`,
`TebConfig`, `FlcConfig`) on the underlying follower types — construct
with `MpcFollower::with_mpc_config(...)` etc. when you need more control
than the generic `ControllerConfig` provides.

## Obstacles (MCA / DWA / SOC / TEB)

```rust
use ondrive::{GaussianMode, Obstacle, WorldConstraints};

let world = WorldConstraints {
    obstacles: vec![Obstacle {
        id: 0,
        radius: 0.3,
        modes: vec![GaussianMode {
            weight: 1.0,
            mean_x: vec![6.0; 10],
            mean_y: vec![0.0; 10],
            std_x: vec![0.1; 10],
            std_y: vec![0.1; 10],
        }],
    }],
    ..Default::default()
};
let cmd = tracker.tick(&state, 0.05, Some(&world));
```

## C ABI

`libondrive` is built as a `cdylib` and exposes a full C API through
[`include/ondrive.h`](include/ondrive.h). Opaque handles (`OndrivePathHandle`,
`OndriveWorldHandle`, `OndriveTrackerHandle`) wrap state, and plain
`#[repr(C)]` POD structs carry values by copy.

```bash
make -C examples/c_abi run
```

See [`examples/c_abi/demo.c`](examples/c_abi/demo.c) for the full usage
walk-through.

Error handling follows a consistent pattern: boolean-returning functions
signal failure with `false`, and `ondrive_last_error_message()` returns
the most recent thread-local message.

## Python bindings

Built with `pyo3` + `maturin`. Enable the `python-extension` feature
when building with maturin; the ergonomic wrappers live in
[`src/python.rs`](src/python.rs) and cover `Tracker`, `Path`, `World`,
`Goal`, `RobotState`, `RobotConstraints`, `ControllerConfig`,
`VelocityCommand`, and `ControllerStatus`.

```bash
make -C examples/python_binding basic     # PID point-goal demo
make -C examples/python_binding main      # Pure Pursuit sine-path demo
```

```python
import math, ondrive

tracker = ondrive.Tracker("pure_pursuit")
cfg = ondrive.ControllerConfig.default_()
cfg.lookahead_distance = 1.2
cfg.output_units = "physical"
tracker.set_config(cfg)

cons = ondrive.RobotConstraints.default_()
cons.steering_type = "ackermann"
cons.max_linear_velocity = 1.0
cons.wheelbase = 0.5
tracker.init(cons)

path = ondrive.Path()
for i in range(20):
    path.add_waypoint_xy(i * 0.5, 0.0)
tracker.set_path(path)
tracker.set_goal(
    ondrive.Goal(target_pose=path.waypoint(len(path) - 1))
)

state = ondrive.RobotState(pose=((0.0, 0.2, 0.0), 0.0), allow_move=True)
cmd = tracker.tick(state, 0.05)
```

## Examples

Each controller has an example under `examples/`. `examples/common/viz.rs`
provides rerun helpers (path/robot/goal/status) shared across demos —
rerun is a dev-dependency only, so the library itself stays
visualisation-agnostic.

```bash
make run EXAMPLE=pid
make run EXAMPLE=pure
make run EXAMPLE=mpc
make run EXAMPLE=mca
# ... carrot, stan, lqr, mppi, soc
```

## Development

```bash
nix develop             # provisions rustc, cargo, maturin, python3
make build              # cargo build --lib --examples
make test               # 29 tests across library + FFI smoke
make c-demo             # build and run the C demo
```

Test coverage:

- **Library (22 tests)**: each controller drives a canonical scenario;
  LQR test includes a numeric Riccati-residual check; MPPI/MCA/SOC tests
  use seeded RNGs for reproducibility.
- **FFI (7 tests)**: every entry point is smoke-tested for null-pointer
  rejection, unknown kind rejection, full lifecycle (new → init →
  set_path → set_goal → tick-to-goal → emergency_stop → free),
  obstacle round-tripping, and a 100-iteration create/free stress loop.
- **Tracker facade (5 tests)**: emergency stop, current_target priority,
  smoothen densification, free-function smoothen, is_goal_reached
  lifecycle.

See [`PLAN.md`](PLAN.md) for the design overview and the algorithm-by-
algorithm translation notes from the C++ source.

## Status

- Git-dependency based; not configured for crates.io publication.
- Pinned to Rust 2024 edition.
- License: MIT.
