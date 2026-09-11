# ondrive

`ondrive` is a Rust motion-control and path-tracking library for mobile
robots. A single `Tracker` facade dispatches to twelve interchangeable
controllers across four algorithm families, all operating on the same
`datapod` pose and velocity types and all emitting the same body-frame
twist.

## Algorithms

| Family              | Controller         | Notes                                                                 |
|---------------------|--------------------|-----------------------------------------------------------------------|
| Point-to-point      | **PID**            | Dual-loop (distance + heading), clamped integrals, wrapped derivative |
|                     | **Carrot**         | Proportional bearing control with distance / heading speed scaling    |
| Path following      | **Pure Pursuit**   | Rear-axle arc-length lookahead, `κ = 2 sin α / L_d`, reverse support  |
|                     | **Stanley**        | Front-axle segment projection, `δ = θ_e + atan(k e / (k_soft + v))`   |
|                     | **LQR**            | Kinematic `[e, θ_e]` error model, DARE solved every tick, curvature FF|
| Predictive / optimal| **MPC**            | Projected gradient descent with line search, warm start, decel taper  |
|                     | **MPPI**           | Williams-style weights incl. control cost term, warm start            |
|                     | **MCA** (DRA-MPPI) | MPPI + Monte Carlo collision probability over Gaussian-mixture modes  |
|                     | **SOC** (SVG-MPPI) | Stein Variational Gradient Descent guides + adaptive-variance MPPI    |
|                     | **DWA**            | Fox 1997 window with braking admissibility and time-indexed obstacles |
|                     | **TEB**            | Timed Elastic Band with non-holonomic residual, projected GD on Δt_i  |
| Fuzzy               | **FLC**            | Mamdani, 7 triangular terms, 49-rule additive base, curvature FF      |

All controllers accept the four kinematic models through `SteeringType`:
differential, Ackermann, holonomic, skid-steer. Obstacle-aware controllers
(MCA, DWA, TEB, SOC) consume `WorldConstraints` with Gaussian-mode obstacle
predictions.

## Command contract

Every controller returns a `VelocityCommand` with the same meaning,
regardless of algorithm or steering type:

- `linear_velocity` is the body forward speed (m/s), negative when reversing.
- `angular_velocity` is the body yaw rate (rad/s). Integrating
  `yaw += angular_velocity * dt` is always correct.
- `steering_angle` is the equivalent Ackermann front-wheel angle (rad),
  consistent with the two values above; zero for other steering types.
- Ackermann commands never exceed the curvature allowed by
  `max_steering_angle` and `min_turning_radius`, and never request a yaw
  rate at zero speed.
- `ControllerConfig::output_units` defaults to `Physical`. `Normalized`
  divides by `max_linear_velocity` and `max_angular_velocity`.
- Reverse motion is only commanded when `ControllerConfig::allow_reverse`
  or `RobotState::allow_reverse` is set.
- A non-positive or non-finite `dt` yields an invalid command; the tracker
  also returns a valid zero command when `RobotState::allow_move` is false.

Errors use one convention: lateral error is positive when the robot is to
the left of the path, heading error is positive when the path heading is
to the left of the robot heading.

`RobotConstraints::rear_wheelbase` is the distance from the pose origin
back to the rear axle (0 = the pose is the rear axle). On Ackermann
platforms Pure Pursuit and Stanley use it to place their reference points
and the predictive controllers use it in their rollout model.

## Goals and arrival

`Goal::tolerance_position` and `tolerance_orientation` override the
config tolerances when set (> 0). An orientation tolerance of `π` or more
disables the orientation check. On arrival:

- inside position tolerance with orientation satisfied (or not required):
  stop, `goal_reached = true`;
- inside position tolerance but misaligned, on a platform that can turn in
  place: rotate toward the goal yaw (mode `aligning`);
- inside position tolerance on an Ackermann platform: stop and report
  reached, since the orientation cannot be corrected in place.

Path followers additionally report arrival when they pass the end of the
path within twice the position tolerance.

## Install

```toml
[dependencies]
ondrive = { git = "https://codeberg.org/robolibs/ondrive.git" }
```

The crate depends on the sibling `datapod` crate for geometry types.

## Quick start

```rust
use datapod::{Euler, Point, Pose, Quaternion};
use ondrive::{
    ControllerConfig, Path, RobotConstraints, RobotState, SteeringType, Tracker, TrackerKind,
    smoothen_path,
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
tracker.set_config(cfg);

let mut cons = RobotConstraints::default();
cons.steering_type = SteeringType::Ackermann;
cons.max_linear_velocity = 1.0;
cons.max_angular_velocity = 2.0;
cons.max_steering_angle = 0.6;
cons.wheelbase = 0.5;
tracker.init(cons);

// Without an explicit goal the tracker aims at the path end.
tracker.set_path(path);

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

| Method                           | Purpose                                                    |
|----------------------------------|------------------------------------------------------------|
| `new(kind)` / `with_config(...)` | Construct with a `TrackerKind`                             |
| `init(constraints)`              | Set robot constraints                                      |
| `set_config(cfg)` / `get_config` | Controller config (gains, tolerances, units)               |
| `set_goal(g)` / `clear_goal`     | Explicit target pose                                       |
| `set_path(p)` / `clear_path`     | Waypoint sequence, optional per-waypoint speeds            |
| `smoothen(max_segment_m)`        | Densify the active path (linear + slerp, speeds interpolated) |
| `tick(state, dt, world)`         | Produce `VelocityCommand` for one control period           |
| `emergency_stop()`               | Clears state, returns zero command                         |
| `get_status()` / `is_goal_reached()` / `is_path_completed()` / `current_target()` | Telemetry |

Tracker semantics when no explicit goal is set:

- path followers aim at the last waypoint with the path's end tangent as
  the target orientation;
- point controllers (PID, Carrot, DWA) are aimed at the waypoint one
  `lookahead_distance` ahead of the robot's projection on the path (passed
  waypoints are skipped, and waypoints inside an obstacle footprint from
  `WorldConstraints` are skipped too); only the last waypoint uses
  `goal_tolerance`. `is_path_completed()` turns true after it and further
  ticks return a zero command.

`Path::speeds`, when present, caps the commanded speed per waypoint.

Predictive controllers expose their specialised configs (`MpcConfig`,
`MppiConfig`, `McaConfig`, `SocConfig`, `DwaConfig`, `TebConfig`,
`FlcConfig`) on the follower types; construct with
`MpcFollower::with_mpc_config(...)` etc. and drive them through the
`Controller` trait when you need more control than the generic
`ControllerConfig` provides. Sampling controllers take a seed via
`with_seed` for reproducible runs.

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

Mode vectors are indexed by horizon step; a prediction shorter than the
horizon holds its last entry. The robot footprint is a disc of radius
`hypot(robot_width, robot_length) / 2` (0.3 m when unset).

## C ABI

`libondrive` is built as a `cdylib` and exposes a full C API through
[`include/ondrive.h`](include/ondrive.h). Opaque handles (`OndrivePathHandle`,
`OndriveWorldHandle`, `OndriveTrackerHandle`) wrap state, and plain
`#[repr(C)]` POD structs carry values by copy.

```bash
make -C examples/c_abi run
```

See [`examples/c_abi/demo.c`](examples/c_abi/demo.c) for the full usage
walk-through. Boolean-returning functions signal failure with `false`, and
`ondrive_last_error_message()` returns the most recent thread-local message.

## Python bindings

Built with `pyo3` + `maturin`. Enable the `python` feature when building
with maturin; the wrappers live in [`src/python/mod.rs`](src/python/mod.rs)
and cover `Tracker`, `Path`, `World`, `Goal`, `RobotState`,
`RobotConstraints`, `ControllerConfig`, `VelocityCommand`, and
`ControllerStatus`.

```bash
make -C examples/python_binding basic     # PID point-goal demo
make -C examples/python_binding main      # Pure Pursuit sine-path demo
```

## Examples

Every controller has a demo under `examples/` built on the shared
scaffold in `examples/common/demo.rs` (rerun is a dev-dependency only).
Each one streams the path, current target, robot, predicted plan,
obstacles and status to a running rerun viewer at real-time speed.

```bash
make run EXAMPLE=pid      # pid, carrot, pure, stan, lqr, mpc, mppi,
                          # mca, soc, dwa, teb, flc
ONDRIVE_NO_VIZ=1 cargo run --release --example teb   # headless
```

`examples/verify.rs` is a headless check of every controller on both
steering models across straight, offset, heading-offset, sine, 90-degree
corner, circle, hairpin, reverse and obstacle scenarios. It reports
arrival, settled cross-track error, limit violations and NaNs per case:

```bash
cargo run --release --example verify
DT=0.02 cargo run --release --example verify   # other control periods
```

## Development

```bash
nix develop             # provisions rustc, cargo, maturin, python3
make build              # cargo build --lib
make test               # cargo test --all-targets
make bind               # regenerate the C header + build the Python wheel
make -C examples/c_abi run
```

Tests:

- **Per-controller** (`tests/<controller>.rs`): each controller drives a
  canonical scenario to its goal; sampling controllers use fixed seeds.
- **Cross-controller** (`tests/correctness.rs`): `dt` and `allow_move`
  guards, unit normalisation, Ackermann curvature feasibility and
  steering-angle consistency, steering direction from either side of the
  path, reverse driving, goal orientation handling, integral anti-windup,
  tracker path semantics and path-projection geometry.
- **FFI** (`tests/ffi_smoke.rs`): null-pointer rejection, unknown kinds,
  full lifecycle, obstacle round-tripping, create/free stress.

See [`PLAN.md`](PLAN.md) for the controller roadmap.

## Status

- Git-dependency based; not configured for crates.io publication.
- Pinned to Rust 2024 edition.
- License: MIT.
