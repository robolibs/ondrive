//! Iterative learning control for repeated routes. Wraps any follower;
//! records the lateral error along the original path during a pass and,
//! when the same path is installed again, shifts the reference handed to
//! the inner follower sideways by the accumulated correction so systematic
//! tracking error shrinks pass after pass.

use crate::controller::{Controller, ControllerBase};
use crate::core::path::{PathCursor, segment_heading};
use crate::path::PurePursuitFollower;
use crate::types::{
    ControllerConfig, ControllerStatus, Goal, Path, RobotConstraints, RobotState, Trajectory,
    VelocityCommand, WorldConstraints,
};
use datapod::{Point, Pose};

#[derive(Clone, Debug)]
pub struct IlcConfig {
    /// Fraction of the recorded error applied as correction per pass.
    pub gain: f64,
    /// Largest lateral shift of the reference (metres).
    pub max_correction: f64,
    /// Moving-average half-width (waypoints) applied to the correction.
    pub smoothing: usize,
}

impl Default for IlcConfig {
    fn default() -> Self {
        Self { gain: 0.7, max_correction: 0.5, smoothing: 2 }
    }
}

/// Identity of a path: waypoint count and its two end points.
#[derive(Clone, Copy, Debug, PartialEq)]
struct PathKey {
    n: usize,
    first: (f64, f64),
    last: (f64, f64),
}

fn key_of(path: &Path) -> Option<PathKey> {
    let f = path.waypoints.first()?;
    let l = path.waypoints.last()?;
    Some(PathKey { n: path.waypoints.len(), first: (f.point.x, f.point.y), last: (l.point.x, l.point.y) })
}

pub struct IlcFollower {
    pub base: ControllerBase,
    pub ilc_config: IlcConfig,
    inner: Box<dyn Controller>,
    cursor: PathCursor,
    key: Option<PathKey>,
    corrections: Vec<f64>,
    error_sum: Vec<f64>,
    error_count: Vec<u32>,
    passes: usize,
}

impl Default for IlcFollower {
    fn default() -> Self {
        Self::new(Box::new(PurePursuitFollower::new()))
    }
}

impl IlcFollower {
    pub fn new(inner: Box<dyn Controller>) -> Self {
        Self {
            base: ControllerBase::default(),
            ilc_config: IlcConfig::default(),
            inner,
            cursor: PathCursor::default(),
            key: None,
            corrections: Vec::new(),
            error_sum: Vec::new(),
            error_count: Vec::new(),
            passes: 0,
        }
    }

    pub fn with_ilc_config(inner: Box<dyn Controller>, cfg: IlcConfig) -> Self {
        Self { ilc_config: cfg, ..Self::new(inner) }
    }

    /// Passes completed on the current path.
    pub fn passes(&self) -> usize {
        self.passes
    }

    /// Current lateral correction per waypoint (positive = left).
    pub fn corrections(&self) -> &[f64] {
        &self.corrections
    }

    /// Fold the errors of the pass just driven into the corrections.
    fn learn(&mut self) {
        let n = self.corrections.len();
        let mut update = vec![0.0; n];
        for (i, u) in update.iter_mut().enumerate() {
            if self.error_count[i] > 0 {
                *u = self.error_sum[i] / self.error_count[i] as f64;
            }
        }
        // Fill waypoints never visited from their neighbours.
        for i in 0..n {
            if self.error_count[i] == 0 {
                let prev = (0..i).rev().find(|j| self.error_count[*j] > 0).map(|j| update[j]);
                let next = (i + 1..n).find(|j| self.error_count[*j] > 0).map(|j| update[j]);
                update[i] = match (prev, next) {
                    (Some(a), Some(b)) => 0.5 * (a + b),
                    (Some(a), None) | (None, Some(a)) => a,
                    _ => 0.0,
                };
            }
        }
        let k = self.ilc_config.smoothing;
        let smoothed: Vec<f64> = (0..n)
            .map(|i| {
                let lo = i.saturating_sub(k);
                let hi = (i + k + 1).min(n);
                update[lo..hi].iter().sum::<f64>() / (hi - lo) as f64
            })
            .collect();
        let cap = self.ilc_config.max_correction.abs();
        for (c, s) in self.corrections.iter_mut().zip(smoothed.iter()) {
            *c = (*c - self.ilc_config.gain * s).clamp(-cap, cap);
        }
        self.passes += 1;
    }

    fn corrected_path(&self) -> Path {
        let mut path = self.base.path.clone();
        let n = path.waypoints.len();
        for i in 0..n {
            let h = segment_heading(&self.base.path.waypoints, i.min(n.saturating_sub(2)));
            let c = self.corrections.get(i).copied().unwrap_or(0.0);
            let p = self.base.path.waypoints[i].point;
            path.waypoints[i] = Pose {
                point: Point::new(p.x - c * h.sin(), p.y + c * h.cos(), p.z),
                rotation: self.base.path.waypoints[i].rotation,
            };
        }
        path
    }
}

impl Controller for IlcFollower {
    fn compute_control(
        &mut self,
        state: &RobotState,
        goal: &Goal,
        constraints: &RobotConstraints,
        dt: f64,
        world: Option<&WorldConstraints>,
    ) -> VelocityCommand {
        let cmd = self.inner.compute_control(state, goal, constraints, dt, world);
        let mut status = self.inner.get_status();
        if let Some(proj) = self.cursor.project(&self.base.path.waypoints, state.pose.point, self.base.path_index) {
            self.base.path_index = proj.segment;
            let idx = if proj.t < 0.5 { proj.segment } else { proj.segment + 1 };
            if let Some(slot) = self.error_sum.get_mut(idx) {
                *slot += proj.lateral_error;
                self.error_count[idx] += 1;
            }
            status.cross_track_error = proj.lateral_error;
        }
        self.base.status = status;
        cmd
    }

    fn set_path(&mut self, path: Path) {
        let key = key_of(&path);
        let same = key.is_some() && key == self.key && self.corrections.len() == path.waypoints.len();
        if same {
            self.learn();
        } else {
            self.corrections = vec![0.0; path.waypoints.len()];
            self.passes = 0;
            self.key = key;
        }
        self.error_sum = vec![0.0; path.waypoints.len()];
        self.error_count = vec![0; path.waypoints.len()];
        self.cursor.set_path(&path.waypoints);
        self.base.path = path;
        self.base.path_index = 0;
        self.base.status = ControllerStatus::default();
        let corrected = self.corrected_path();
        self.inner.set_path(corrected);
    }

    fn set_trajectory(&mut self, trajectory: Trajectory) {
        self.set_path(trajectory.to_path());
    }

    fn reset(&mut self) {
        self.inner.reset();
        self.base.path = Path::default();
        self.base.path_index = 0;
        self.base.status = ControllerStatus::default();
        self.key = None;
        self.corrections.clear();
        self.error_sum.clear();
        self.error_count.clear();
        self.passes = 0;
    }

    fn get_status(&self) -> ControllerStatus {
        self.base.status.clone()
    }

    fn set_config(&mut self, config: ControllerConfig) {
        self.base.config = config.clone();
        self.inner.set_config(config);
    }

    fn get_config(&self) -> ControllerConfig {
        self.inner.get_config()
    }

    fn get_type(&self) -> &'static str {
        "ilc_follower"
    }

    fn get_path_index(&self) -> usize {
        self.inner.get_path_index()
    }

    fn predicted_trajectory(&self) -> Vec<Point> {
        self.inner.predicted_trajectory()
    }

    fn base(&self) -> &ControllerBase {
        &self.base
    }

    fn base_mut(&mut self) -> &mut ControllerBase {
        &mut self.base
    }
}
