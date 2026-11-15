//! Obstacle tracking: detections (positions only) folded over frames into a
//! constant-velocity Kalman filter per obstacle, whose state and covariance
//! give each obstacle's predicted motion, its spread growing with what the
//! filter does not know.
//!
//! Detections carrying an id stay with that id's track; those without are
//! matched to the nearest track inside a Mahalanobis gate, and any left
//! over start tracks of their own. Tracks unseen for `max_age` seconds are
//! dropped.

use crate::types::{GaussianMode, Obstacle, WorldConstraints};

/// One obstacle as seen in a frame: where its centre is and how wide it is.
#[derive(Clone, Copy, Debug, Default)]
pub struct Detection {
    /// Identity from the detector, if it keeps one.
    pub id: Option<u64>,
    pub x: f64,
    pub y: f64,
    pub radius: f64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct TrackingConfig {
    /// Standard deviation, m/s², of the unmodelled acceleration.
    pub acceleration_noise: f64,
    /// Standard deviation, metres, of a detected position.
    pub position_noise: f64,
    /// Standard deviation, m/s, of a new track's unknown velocity.
    pub initial_speed_spread: f64,
    /// Squared Mahalanobis distance within which a detection without an id
    /// may join a track (chi-square, two degrees of freedom).
    pub gate: f64,
    /// Seconds a track is kept without being seen.
    pub max_age: f64,
    /// Prediction steps given out, and seconds between them.
    pub horizon_steps: usize,
    pub step: f64,
    /// Standard deviation, metres per metre an obstacle is predicted to
    /// travel, added to its predicted spread.
    pub motion_spread: f64,
    /// Share of each detection's radius taken into the smoothed radius.
    pub radius_gain: f64,
}

impl Default for TrackingConfig {
    fn default() -> Self {
        Self {
            acceleration_noise: 0.5,
            position_noise: 0.05,
            initial_speed_spread: 1.5,
            gate: 13.8,
            max_age: 1.0,
            horizon_steps: 30,
            step: 0.2,
            motion_spread: 0.15,
            radius_gain: 0.2,
        }
    }
}

/// State `[x, y, vx, vy]` and its covariance.
#[derive(Clone, Debug)]
struct Track {
    id: u64,
    keyed: bool,
    s: [f64; 4],
    p: [[f64; 4]; 4],
    radius: f64,
    seen: f64,
    hits: u32,
}

impl Track {
    fn new(id: u64, keyed: bool, d: &Detection, time: f64, cfg: &TrackingConfig) -> Self {
        let (r, v) = (cfg.position_noise.powi(2), cfg.initial_speed_spread.powi(2));
        let mut p = [[0.0; 4]; 4];
        (p[0][0], p[1][1], p[2][2], p[3][3]) = (r, r, v, v);
        Track { id, keyed, s: [d.x, d.y, 0.0, 0.0], p, radius: d.radius.max(0.0), seen: time, hits: 1 }
    }

    /// State and covariance `dt` seconds on, under constant velocity with
    /// white acceleration noise.
    fn ahead(&self, dt: f64, q: f64) -> ([f64; 4], [[f64; 4]; 4]) {
        let s = [self.s[0] + dt * self.s[2], self.s[1] + dt * self.s[3], self.s[2], self.s[3]];
        let mut f = [[0.0; 4]; 4];
        for (i, row) in f.iter_mut().enumerate() {
            row[i] = 1.0;
        }
        (f[0][2], f[1][3]) = (dt, dt);
        let mut fp = [[0.0; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                fp[i][j] = (0..4).map(|k| f[i][k] * self.p[k][j]).sum();
            }
        }
        let mut p = [[0.0; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                p[i][j] = (0..4).map(|k| fp[i][k] * f[j][k]).sum();
            }
        }
        let (a, b, c) = (q * dt.powi(4) / 4.0, q * dt.powi(3) / 2.0, q * dt * dt);
        for (pos, vel) in [(0, 2), (1, 3)] {
            p[pos][pos] += a;
            p[pos][vel] += b;
            p[vel][pos] += b;
            p[vel][vel] += c;
        }
        (s, p)
    }

    /// Innovation and its covariance for a detection at `(x, y)`.
    fn innovation(&self, x: f64, y: f64, r: f64) -> ([f64; 2], [[f64; 2]; 2]) {
        ([x - self.s[0], y - self.s[1]], [[self.p[0][0] + r, self.p[0][1]], [self.p[1][0], self.p[1][1] + r]])
    }

    /// Squared Mahalanobis distance of a detection at `(x, y)`.
    fn distance2(&self, x: f64, y: f64, r: f64) -> f64 {
        let (e, s) = self.innovation(x, y, r);
        let det = s[0][0] * s[1][1] - s[0][1] * s[1][0];
        if det <= 0.0 {
            return f64::INFINITY;
        }
        (s[1][1] * e[0] * e[0] - (s[0][1] + s[1][0]) * e[0] * e[1] + s[0][0] * e[1] * e[1]) / det
    }

    /// The Kalman update with a detection at `(x, y)`.
    fn correct(&mut self, x: f64, y: f64, r: f64) {
        let (e, s) = self.innovation(x, y, r);
        let det = s[0][0] * s[1][1] - s[0][1] * s[1][0];
        if det <= 0.0 {
            return;
        }
        let inv = [[s[1][1] / det, -s[0][1] / det], [-s[1][0] / det, s[0][0] / det]];
        let mut k = [[0.0; 2]; 4];
        for (i, row) in k.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                *cell = (0..2).map(|m| self.p[i][m] * inv[m][j]).sum();
            }
        }
        for (i, si) in self.s.iter_mut().enumerate() {
            *si += k[i][0] * e[0] + k[i][1] * e[1];
        }
        let mut p = self.p;
        for i in 0..4 {
            for j in 0..4 {
                p[i][j] -= k[i][0] * self.p[0][j] + k[i][1] * self.p[1][j];
            }
        }
        self.p = p;
    }
}

/// Obstacles tracked from detections over frames.
#[derive(Clone, Debug, Default)]
pub struct ObstacleTracker {
    pub config: TrackingConfig,
    tracks: Vec<Track>,
    next_id: u64,
    time: Option<f64>,
}

/// First id given to tracks started from detections without one.
const OWN_IDS: u64 = 1 << 48;

impl ObstacleTracker {
    pub fn new(config: TrackingConfig) -> Self {
        Self { config, tracks: Vec::new(), next_id: OWN_IDS, time: None }
    }

    /// Folds the detections of the frame at `time` seconds into the tracks.
    pub fn update(&mut self, time: f64, detections: &[Detection]) {
        let cfg = self.config.clone();
        let (q, r) = (cfg.acceleration_noise.powi(2), cfg.position_noise.powi(2));
        let dt = self.time.map_or(0.0, |t| (time - t).max(0.0));
        if dt > 0.0 {
            for t in &mut self.tracks {
                (t.s, t.p) = t.ahead(dt, q);
            }
        }
        self.time = Some(time);
        let mut taken = vec![false; self.tracks.len()];
        let mut loose: Vec<&Detection> = Vec::new();
        for d in detections.iter().filter(|d| d.x.is_finite() && d.y.is_finite()) {
            match d.id {
                Some(id) => match self.tracks.iter().position(|t| t.keyed && t.id == id) {
                    Some(k) => {
                        taken[k] = true;
                        self.see(k, d, time);
                    }
                    None => {
                        self.tracks.push(Track::new(id, true, d, time, &cfg));
                        taken.push(true);
                    }
                },
                None => loose.push(d),
            }
        }
        // Detections without an id join the nearest free track in the gate,
        // nearest pairs first.
        let mut pairs: Vec<(f64, usize, usize)> = Vec::new();
        for (j, d) in loose.iter().enumerate() {
            for (k, t) in self.tracks.iter().enumerate() {
                if !t.keyed && !taken[k] {
                    let m = t.distance2(d.x, d.y, r);
                    if m <= cfg.gate {
                        pairs.push((m, j, k));
                    }
                }
            }
        }
        pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut used = vec![false; loose.len()];
        for (_, j, k) in pairs {
            if !used[j] && !taken[k] {
                (used[j], taken[k]) = (true, true);
                self.see(k, loose[j], time);
            }
        }
        for (j, d) in loose.iter().enumerate() {
            if !used[j] {
                let id = self.next_id;
                self.next_id += 1;
                self.tracks.push(Track::new(id, false, d, time, &cfg));
            }
        }
        self.tracks.retain(|t| time - t.seen <= cfg.max_age);
    }

    fn see(&mut self, k: usize, d: &Detection, time: f64) {
        let (r, gain) = (self.config.position_noise.powi(2), self.config.radius_gain.clamp(0.0, 1.0));
        let t = &mut self.tracks[k];
        t.correct(d.x, d.y, r);
        t.radius += gain * (d.radius.max(0.0) - t.radius);
        t.seen = time;
        t.hits += 1;
    }

    /// Number of obstacles being tracked.
    pub fn len(&self) -> usize {
        self.tracks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tracks.is_empty()
    }

    /// Velocity estimated for the obstacle with `id`, if tracked.
    pub fn velocity(&self, id: u64) -> Option<(f64, f64)> {
        self.tracks.iter().find(|t| t.id == id).map(|t| (t.s[2], t.s[3]))
    }

    /// Every tracked obstacle, predicted `horizon_steps` of `step` seconds
    /// from the last frame: the filter's state carried on at constant
    /// velocity. Its spread is what the filter's position and velocity
    /// uncertainty become over that time, plus `motion_spread` of the
    /// distance it covers.
    pub fn obstacles(&self) -> Vec<Obstacle> {
        let cfg = &self.config;
        let step = cfg.step.max(1e-3);
        self.tracks
            .iter()
            .map(|t| {
                let at: Vec<([f64; 4], [[f64; 4]; 4])> = (0..=cfg.horizon_steps).map(|k| t.ahead(k as f64 * step, 0.0)).collect();
                let speed = t.s[2].hypot(t.s[3]);
                let grown = |k: usize, var: f64| var.max(0.0).sqrt() + cfg.motion_spread.max(0.0) * speed * k as f64 * step;
                let mode = GaussianMode {
                    weight: 1.0,
                    mean_x: at.iter().map(|a| a.0[0]).collect(),
                    mean_y: at.iter().map(|a| a.0[1]).collect(),
                    std_x: at.iter().enumerate().map(|(k, a)| grown(k, a.1[0][0])).collect(),
                    std_y: at.iter().enumerate().map(|(k, a)| grown(k, a.1[1][1])).collect(),
                };
                Obstacle { id: t.id, radius: t.radius, dt: step, modes: vec![mode] }
            })
            .collect()
    }

    /// `world` with its obstacles replaced by the tracked ones.
    pub fn fill(&self, world: &mut WorldConstraints) {
        world.obstacles = self.obstacles();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noise(seed: &mut u64) -> f64 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        ((*seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 0.1
    }

    #[test]
    fn learns_a_walker_s_velocity_from_noisy_positions() {
        let mut tr = ObstacleTracker::new(TrackingConfig::default());
        let mut seed = 7u64;
        for k in 0..40 {
            let t = 0.05 * k as f64;
            let d = Detection { id: Some(3), x: 2.0 + 0.8 * t + noise(&mut seed), y: -1.0 - 0.6 * t + noise(&mut seed), radius: 0.4 };
            tr.update(t, &[d]);
        }
        let (vx, vy) = tr.velocity(3).expect("tracked");
        assert!((vx - 0.8).abs() < 0.15 && (vy + 0.6).abs() < 0.15, "velocity ({vx:.2}, {vy:.2})");
        let o = &tr.obstacles()[0];
        let m = &o.modes[0];
        let (x3, y3) = (2.0 + 0.8 * (1.95 + 3.0), -1.0 - 0.6 * (1.95 + 3.0));
        assert!((m.mean_x[15] - x3).abs() < 0.5 && (m.mean_y[15] - y3).abs() < 0.5, "3 s on at ({:.2}, {:.2})", m.mean_x[15], m.mean_y[15]);
        assert!(m.std_x[30] > m.std_x[0], "the spread grows with time");
    }

    #[test]
    fn a_standing_post_settles_to_no_motion() {
        let mut tr = ObstacleTracker::new(TrackingConfig::default());
        let mut seed = 11u64;
        for k in 0..60 {
            tr.update(0.05 * k as f64, &[Detection { id: Some(1), x: 5.0 + noise(&mut seed), y: 2.0 + noise(&mut seed), radius: 0.5 }]);
        }
        let (vx, vy) = tr.velocity(1).expect("tracked");
        assert!(vx.hypot(vy) < 0.1, "speed {:.3}", vx.hypot(vy));
    }

    #[test]
    fn a_new_track_is_less_certain_than_a_long_seen_one() {
        let mut tr = ObstacleTracker::new(TrackingConfig::default());
        tr.update(0.0, &[Detection { id: Some(1), x: 0.0, y: 0.0, radius: 0.4 }]);
        let fresh = tr.obstacles()[0].modes[0].std_x[10];
        for k in 1..40 {
            tr.update(0.05 * k as f64, &[Detection { id: Some(1), x: 0.0, y: 0.0, radius: 0.4 }]);
        }
        let known = tr.obstacles()[0].modes[0].std_x[10];
        assert!(known < 0.5 * fresh, "spread 2 s on: fresh {fresh:.2}, long seen {known:.2}");
    }

    #[test]
    fn walkers_without_ids_keep_their_tracks_as_they_pass() {
        let mut tr = ObstacleTracker::new(TrackingConfig::default());
        for k in 0..60 {
            let t = 0.05 * k as f64;
            let a = Detection { id: None, x: -2.0 + 1.0 * t, y: 0.5, radius: 0.3 };
            let b = Detection { id: None, x: 2.0 - 1.0 * t, y: -0.5, radius: 0.3 };
            tr.update(t, &[a, b]);
        }
        assert_eq!(tr.len(), 2);
        let speeds: Vec<f64> = tr.obstacles().iter().map(|o| (o.modes[0].mean_x[1] - o.modes[0].mean_x[0]) / o.dt).collect();
        assert!(speeds.iter().any(|v| (v - 1.0).abs() < 0.2) && speeds.iter().any(|v| (v + 1.0).abs() < 0.2), "speeds {speeds:?}");
    }

    #[test]
    fn tracks_unseen_for_long_are_dropped() {
        let mut tr = ObstacleTracker::new(TrackingConfig::default());
        tr.update(0.0, &[Detection { id: Some(1), x: 0.0, y: 0.0, radius: 0.4 }]);
        tr.update(0.5, &[]);
        assert_eq!(tr.len(), 1);
        tr.update(1.6, &[]);
        assert!(tr.is_empty());
    }
}
