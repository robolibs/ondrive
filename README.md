# ondrive

Rust motion-control / path-tracking library for mobile robots.

Provides a uniform `Controller` trait and a high-level `Tracker` facade
dispatching to one of twelve algorithms:

- **Point-to-point:** PID, Carrot
- **Path-following:** Pure Pursuit, Stanley, LQR
- **Predictive / optimal:** MPC, MPPI, MCA (risk-aware MPPI), SOC
  (SVG-MPPI), DWA (Dynamic Window Approach), TEB (Timed Elastic Band)
- **Fuzzy:** FLC (Mamdani fuzzy controller)

Supports differential, Ackermann, holonomic, and skid-steer kinematics,
obstacle-aware planning via `WorldConstraints`, and optional rerun
visualisation in examples.

## Build

```bash
cargo build
cargo test
```

## C ABI

```bash
make -C examples/c_abi run
```

## Python bindings

```bash
make -C examples/python_binding basic
```

See `PLAN.md` for the full design.
