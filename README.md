# ondrive

`ondrive` is a Rust motion-control and path-tracking library for mobile
robots. A single `Tracker` facade dispatches to twenty interchangeable
controllers, all operating on the same `datapod` pose and velocity types
and all emitting the same body-frame twist.

## Algorithms

| Family              | Controller               | Notes                                                                 |
|---------------------|--------------------------|-----------------------------------------------------------------------|
| Point-to-point      | **PID**                  | Dual-loop (distance + heading), clamped integrals, wrapped derivative |
|                     | **Carrot**               | Proportional bearing control with distance / heading speed scaling    |
|                     | **PoseRegulator**        | Astolfi polar law `(rho, alpha, beta)`; converges to position and yaw |
|                     | **PoseReach**            | Reeds-Shepp / Dubins curve to a goal pose, tracked run by run         |
|                     | **APF**                  | Potential field fallback; reports local minima instead of hiding them |
| Path following      | **Pure Pursuit**         | Rear-axle arc-length lookahead, `κ = 2 sin α / L_d`, reverse, cusps   |
|                     | **RegulatedPursuit**     | Nav2-style: curvature / proximity speed regulation, arc collision check|
|                     | **VectorPursuit**        | Screw-theory pursuit using the lookahead point's orientation          |
|                     | **Stanley**              | Front-axle segment projection, `δ = θ_e + atan(k e / (k_soft + v))`   |
|                     | **LQR**                  | Kinematic `[e, θ_e]` error model, DARE solved every tick, curvature FF|
|                     | **Kanayama**             | Robot-frame error law against a moving (timed or previewed) reference |
|                     | **ILC**                  | Wraps any follower; shifts the reference to cancel learned error      |
| Predictive / optimal| **MPC**                  | Projected gradient descent with line search, warm start, decel taper  |
|                     | **iLQR**                 | Analytic-Jacobian iterative LQR with regularised backward pass        |
|                     | **MPPI**                 | Williams-style weights incl. control cost term, warm start            |
|                     | **MCA** (DRA-MPPI)       | MPPI + Monte Carlo collision probability over Gaussian-mixture modes  |
|                     | **SOC** (SVG-MPPI)       | Stein Variational Gradient Descent guides + adaptive-variance MPPI    |
|                     | **DWA**                  | Fox 1997 window, braking admissibility, recovery rotation            |
|                     | **TEB**                  | Timed Elastic Band, common-arc residual (skipped if holonomic), GD on Δt_i|
| Fuzzy               | **FLC**                  | Mamdani, 7 triangular terms, 49-rule additive base, curvature FF      |

All controllers accept the four kinematic models through `SteeringType`.
Three families of motion model back them:

- **Ackermann** — the true bicycle model. A single virtual front wheel,
  curvature `= tan(steer) / wheelbase`, integrated at the rear axle
  (offset by `rear_wheelbase`), steering and turning-radius limits enforced
  by `finalize`.
- **Differential / SkidSteer** — the unicycle model. `v` and `omega` are
  commanded independently with no wheelbase constraint; the platform can
  turn in place.
- **Holonomic** — full `(vx, vy, omega)` body-frame control, translation and
  rotation fully decoupled, output through `finalize_holonomic` /
  `VelocityCommand::lateral_velocity`. Supported end to end by every
  controller: the point controllers (PID, Carrot, PoseRegulator, APF), the
  geometric path followers (Pure Pursuit, RegulatedPursuit, VectorPursuit,
  Stanley, LQR, Kanayama, FLC), DWA's sampling window, the shared
  3-control/5-state predictive core (MPC, iLQR, MPPI, MCA, SOC), TEB, and
  PoseReach, which bypasses its Reeds-Shepp/Dubins planner entirely on a
  holonomic platform and translates/rotates straight to the goal.

Obstacle-aware controllers (MCA, SOC, DWA, TEB, RegulatedPursuit, PoseReach,
APF) consume `WorldConstraints`: Gaussian-mode obstacle predictions and/or
an occupancy grid, through one footprint-aware collision checker.

## Command contract

Every controller returns a `VelocityCommand` with the same meaning,
regardless of algorithm or steering type:

- `linear_velocity` is the body forward speed (m/s), negative when reversing.
- `angular_velocity` is the body yaw rate (rad/s). Integrating
  `yaw += angular_velocity * dt` is always correct.
- `lateral_velocity` is the body leftward speed (m/s); zero except from a
  holonomic-capable controller with `SteeringType::Holonomic`.
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
platforms the pursuit followers and Stanley use it to place their reference
points and the predictive controllers use it in their rollout model.

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

`PoseReach` and `PoseRegulator` are the exceptions: they treat the
orientation tolerance strictly and manoeuvre until both are met.
Path followers additionally report arrival when they pass the end of the
path within twice the position tolerance.

## Paths, trajectories and cusps

`Path` is a waypoint polyline with optional per-waypoint `speeds`. A
negative speed marks a segment driven in reverse; pursuit followers never
look ahead across such a cusp and taper speed into it. `smoothen_path`
densifies a path (linear + slerp, speeds interpolated).

`Trajectory` adds a time per pose. `Tracker::set_trajectory` installs it,
restarts the tracker clock (advanced by `dt`, or by `RobotState::timestamp`
when provided) and hands the time to controllers that track it: Kanayama,
MPC, iLQR, MPPI, MCA and SOC follow a time-indexed reference; every other
controller follows its poses as a path. `Trajectory::from_path` builds one
at constant speed.

## Install

```toml
[dependencies]
ondrive = { git = "https://github.com/robolibs/ondrive" }
```

The crate depends on the sibling `datapod` crate for geometry types.

## Quick start

```rust
use datapod::{Euler, Point, Pose, Quaternion};
use ondrive::{
    ControllerConfig, Path, RobotConstraints, RobotState, SteeringType, Tracker, TrackerKind,
    smoothen_path,
};

let mut path = Path::default();
for i in 0..20 {
    path.waypoints.push(Pose {
        point: Point::new(i as f64 * 0.5, 0.0, 0.0),
        rotation: Quaternion::from_euler(Euler::new(0.0, 0.0, 0.0)),
    });
}
smoothen_path(&mut path, 0.25);

let mut tracker = Tracker::new(TrackerKind::RegulatedPursuit);
let mut cfg = ControllerConfig::default();
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

| Method                                     | Purpose                                                   |
|--------------------------------------------|-----------------------------------------------------------|
| `new(kind)` / `with_config(...)`           | Construct with a `TrackerKind`                            |
| `init(constraints)`                        | Set robot constraints (limits, footprint)                 |
| `set_config(cfg)` / `get_config`           | Controller config (gains, tolerances, units, lookahead)   |
| `set_goal(g)` / `clear_goal`               | Explicit target pose                                      |
| `set_path(p)` / `clear_path`               | Waypoint sequence, optional per-waypoint speeds           |
| `set_trajectory(t)` / `trajectory_time()`  | Timed trajectory and the tracker clock                    |
| `smoothen(max_segment_m)`                  | Densify the active path                                   |
| `tick(state, dt, world)`                   | Produce `VelocityCommand` for one control period          |
| `emergency_stop()`                         | Clears state, returns zero command                        |
| `predicted_trajectory()`                   | Planned trajectory of predictive controllers, if any      |
| `get_status()` / `is_goal_reached()` / `is_path_completed()` / `current_target()` | Telemetry |

Tracker semantics when no explicit goal is set:

- path followers aim at the last waypoint with the path's end tangent as
  the target orientation;
- point controllers (PID, Carrot, DWA, PoseReach, PoseRegulator, APF) are
  aimed at the waypoint one `lookahead_distance` ahead of the robot's
  projection on the path (passed waypoints are skipped, and waypoints whose
  footprint collides with `WorldConstraints` are skipped too); only the last
  waypoint uses `goal_tolerance`. `is_path_completed()` turns true after it
  and further ticks return a zero command.

Specialised configs (`MpcConfig`, `IlqrConfig`, `MppiConfig`, `McaConfig`,
`SocConfig`, `DwaConfig`, `TebConfig`, `FlcConfig`,
`RegulatedPursuitConfig`, `VectorPursuitConfig`, `PoseReachConfig`,
`ApfConfig`, `IlcConfig`) live on the follower types; construct with
`MpcFollower::with_mpc_config(...)` etc. and drive them through the
`Controller` trait. Sampling controllers take a seed via `with_seed`.
`IlcFollower::new(Box::new(inner))` wraps any follower; a pass is learned
each time the same path is installed again.

## Adaptive lookahead

`ControllerConfig::lookahead_time` (default 0.3 s) adds lookahead per unit
of speed for Pure Pursuit and Vector Pursuit; both also shrink the
lookahead on tight path curvature. Stanley's cross-track gain is already
speed-softened by its law. `RegulatedPursuitConfig` has its own
lookahead-time and regulation settings.

## Obstacles

```rust
use ondrive::{Footprint, GaussianMode, Obstacle, OccupancyGrid, WorldConstraints};

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
    grid: Some(OccupancyGrid::new(0.0, -5.0, 0.1, 200, 100, vec![false; 20_000])),
    ..Default::default()
};
let cmd = tracker.tick(&state, 0.05, Some(&world));
```

Mode vectors are indexed by horizon step; a prediction shorter than the
horizon holds its last entry. `OccupancyGrid::new` precomputes a
continuous distance-to-occupied field. `RobotConstraints::footprint` is a
disc by default (radius `hypot(width, length) / 2`, 0.3 m when unset) or a
`Footprint::Polygon`; the shared `CollisionChecker` in `core::obstacles`
evaluates clearance for both against both obstacle kinds.

## C ABI

`libondrive` is built as a `cdylib` and exposes a full C API through
[`include/ondrive.h`](include/ondrive.h): opaque path, world and tracker
handles, `#[repr(C)]` POD structs, `ondrive_tracker_set_trajectory`,
`ondrive_world_set_grid`, and one `ONDRIVE_KIND_*` value per controller.

```bash
make -C examples/c_abi run
```

Boolean-returning functions signal failure with `false`, and
`ondrive_last_error_message()` returns the most recent thread-local message.

## Python bindings

Built with `pyo3` + `maturin` (`python` feature). Wrappers cover `Tracker`
(including `set_trajectory`, `trajectory_time`, `is_path_completed`),
`Path`, `World` (including `set_grid`), `Goal`, `RobotState`,
`RobotConstraints`, `ControllerConfig`, `VelocityCommand` and
`ControllerStatus`; `tracker_kinds()` lists the kind names.

```bash
make -C examples/python_binding basic     # PID point-goal demo
make -C examples/python_binding main      # Pure Pursuit sine-path demo
```

## Examples and verification

Every controller has a demo under `examples/` built on the shared scaffold
in `examples/common/demo.rs` (rerun is a dev-dependency only). Each one
streams the path, current target, robot, predicted plan, obstacles and
status to a running rerun viewer at real-time speed.

```bash
make run EXAMPLE=pid      # pid, carrot, pure, rpp, vector, stan, lqr,
                          # kanayama, flc, mpc, ilqr, mppi, mca, soc, dwa,
                          # teb, ilc, pose_reach, pose_regulator, apf
ONDRIVE_NO_VIZ=1 cargo run --release --example teb   # headless
```

`examples/verify.rs` is a headless check of every controller on both
steering models across straight, offset, heading-offset, sine, 90-degree
corner, circle, hairpin, reverse and obstacle scenarios, plus pose-goal
manoeuvres (parallel park, 90-degree offset, goal behind) and timed
circles for the time-tracking controllers. It reports arrival, settled
error, limit violations and NaNs per case:

```bash
cargo run --release --example verify
DT=0.02 cargo run --release --example verify          # other control periods
KIND=Stanley SCEN=corner90 cargo run --release --example verify
```

`benches/tick.rs` times one tick per controller at default settings:

```bash
cargo bench --bench tick
```

## Development

```bash
nix develop             # provisions rustc, cargo, maturin, python3
make build              # cargo build --lib
make test               # cargo test --all-targets
make bind               # regenerate the C header + build the Python wheel
```

## Status

- Git-dependency based; not configured for crates.io publication.
- Pinned to Rust 2024 edition.
- License: MIT.
