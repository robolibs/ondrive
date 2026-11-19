//! Turns between two poses for a machine with a tightest turning radius:
//! Dubins curves driven forward, Reeds–Shepp curves that may reverse, and
//! the sharp-turn patterns (three-point, bulb, fishtail). Shared with
//! `maptrax`, which plans its headland turns with them.

mod dubins;
mod geometry;
mod pose;
mod reeds_shepp;
mod sharper;

pub use dubins::{Dubins, DubinsPath, DubinsSegment, DubinsSegmentType};
pub use geometry::{Point2Ext, angle_difference, normalize_angle, point_distance, point_xy, points_equal};
pub use pose::Pose2D;
pub use reeds_shepp::{ReedsShepp, ReedsSheppPath, ReedsSheppSegment, ReedsSheppSegmentType};
pub use sharper::{SharpTurnPath, Sharper};
