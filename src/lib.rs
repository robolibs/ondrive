//! ondrive — motion-control / path-tracking library for mobile robots.
//!
//! See `PLAN.md` at the crate root for the design overview. The public
//! surface is kept flat at `ondrive::*` via re-exports below.

pub mod core;

pub mod controller;
pub mod ffi;
pub mod fuzzy;
pub mod path;
pub mod point;
pub mod pred;
pub mod tracker;
pub mod types;

#[cfg(feature = "python")]
pub mod python;

pub use crate::controller::{Controller, ControllerBase, is_goal_reached};
pub use crate::core::error::{Error, Result};
pub use crate::core::math::{
    distance, distance_2d, heading_error, normalize_angle, yaw_error, yaw_of,
};
pub use crate::tracker::{Tracker, TrackerKind, smoothen_path};
pub use crate::types::{
    ControllerConfig, ControllerStatus, GaussianMode, Goal, Obstacle, OutputType, OutputUnits,
    Path, RobotConstraints, RobotState, SteeringType, Velocity, VelocityCommand, WorldConstraints,
    Zone,
};

pub use crate::fuzzy::{FlcConfig, FlcFollower, Term};
pub use crate::path::{
    LqrFollower, PurePursuitFollower, RegulatedPursuitConfig, RegulatedPursuitFollower,
    StanleyFollower,
};
pub use crate::point::{CarrotFollower, PidFollower, PoseReachConfig, PoseReachFollower};
pub use crate::pred::{
    DwaConfig, DwaFollower, IlqrConfig, IlqrFollower, McaConfig, McaFollower, MpcConfig,
    MpcFollower, MppiConfig, MppiFollower, SocConfig, SocFollower, TebConfig, TebFollower,
};
