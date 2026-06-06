//! DWA (Dynamic Window Approach) — Fox et al. 1997.
//!
//! Samples (v, ω) over the feasible dynamic window, forward-simulates each
//! candidate, and picks the minimum-cost pair based on heading alignment,
//! obstacle clearance, forward velocity, and goal attraction.

use crate::controller::{Controller, ControllerBase, is_goal_reached};
use crate::core::math::normalize_angle;
use crate::types::{
    Goal, RobotConstraints, RobotState, SteeringType, VelocityCommand, WorldConstraints,
};
use std::f64::consts::PI;

#[derive(Clone, Debug)]
pub struct DwaConfig {
    pub predict_time: f64,
    pub dt: f64,
    pub v_samples: usize,
    pub w_samples: usize,
    pub max_accel: f64,
    pub max_angular_accel: f64,
    pub weight_heading: f64,
    pub weight_distance: f64,
    pub weight_velocity: f64,
    pub weight_clearance: f64,
    pub target_velocity: f64,
    /// Samples that get within this distance of any obstacle are rejected
    /// (infinite cost). Measured in metres at the robot centre.
    pub obstacle_margin: f64,
}

impl Default for DwaConfig {
    fn default() -> Self {
        Self {
            predict_time: 1.0,
            dt: 0.1,
            v_samples: 10,
            w_samples: 20,
            max_accel: 2.0,
            max_angular_accel: 3.0,
            weight_heading: 1.0,
            weight_distance: 2.0,
            weight_velocity: 0.5,
            weight_clearance: 1.0,
            target_velocity: 1.0,
            obstacle_margin: 0.2,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct DwaFollower {
    pub base: ControllerBase,
    pub dwa_config: DwaConfig,
}

impl DwaFollower {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_dwa_config(cfg: DwaConfig) -> Self {
        Self {
            base: ControllerBase::default(),
            dwa_config: cfg,
        }
    }

    pub fn set_dwa_config(&mut self, cfg: DwaConfig) {
        self.dwa_config = cfg;
    }
}

impl Controller for DwaFollower {
    fn compute_control(
        &mut self,
        state: &RobotState,
        goal: &Goal,
        constraints: &RobotConstraints,
        _dt: f64,
        world: Option<&WorldConstraints>,
    ) -> VelocityCommand {
        let cfg = self.base.config.clone();
        let dwa = &self.dwa_config;

        let (reached, dist_to_goal, yaw_diff) = is_goal_reached(
            &state.pose,
            &goal.target_pose,
            cfg.goal_tolerance,
            cfg.angular_tolerance,
        );
        self.base.status.distance_to_goal = dist_to_goal;
        self.base.status.heading_error = yaw_diff;
        if reached {
            self.base.status.goal_reached = true;
            self.base.status.mode = "stopped".into();
            return VelocityCommand {
                valid: true,
                status_message: "Goal reached".into(),
                ..VelocityCommand::default()
            };
        }

        let is_diff = matches!(
            constraints.steering_type,
            SteeringType::Differential | SteeringType::SkidSteer
        );
        let v_now = state.velocity.linear;
        let w_now = state.velocity.angular;

        // Dynamic window: intersection of kinematic bounds and reachable
        // (v, ω) within one control period.
        let window_dt = dwa.dt;
        let v_min = (v_now - dwa.max_accel * window_dt).max(if cfg.allow_reverse {
            constraints.min_linear_velocity
        } else {
            0.0
        });
        let v_max = (v_now + dwa.max_accel * window_dt).min(constraints.max_linear_velocity);
        let w_min =
            (w_now - dwa.max_angular_accel * window_dt).max(-constraints.max_angular_velocity);
        let w_max =
            (w_now + dwa.max_angular_accel * window_dt).min(constraints.max_angular_velocity);

        if v_max < v_min || w_max < w_min {
            return VelocityCommand::invalid("DWA: empty dynamic window");
        }

        let v_step = if dwa.v_samples > 1 {
            (v_max - v_min) / (dwa.v_samples - 1) as f64
        } else {
            0.0
        };
        let w_step = if dwa.w_samples > 1 {
            (w_max - w_min) / (dwa.w_samples - 1) as f64
        } else {
            0.0
        };

        let goal_point = goal.target_pose.point;

        let obstacles: Vec<(f64, f64, f64)> = world
            .map(|w| {
                w.obstacles
                    .iter()
                    .filter_map(|o| {
                        o.modes.first().and_then(|m| {
                            if m.mean_x.is_empty() || m.mean_y.is_empty() {
                                None
                            } else {
                                Some((m.mean_x[0], m.mean_y[0], o.radius))
                            }
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();

        let steps = (dwa.predict_time / dwa.dt).max(1.0) as usize;

        let mut best_cost = f64::INFINITY;
        let mut best_v = 0.0;
        let mut best_w = 0.0;

        // Normalization: first pass collects per-sample scores, second pass
        // combines them. We accumulate unnormalized heading / distance /
        // velocity / clearance, then pick the lowest weighted sum.
        let mut samples: Vec<(f64, f64, f64, f64, f64, f64)> = Vec::new();
        //                   (v, w, heading, distance, velocity, clearance)

        for vi in 0..dwa.v_samples {
            let v = v_min + vi as f64 * v_step;
            for wi in 0..dwa.w_samples {
                let w = w_min + wi as f64 * w_step;

                // Forward simulate with constant (v, w); for Ackermann,
                // treat w as direct yaw-rate (the DWA is a kinematic
                // planner, not a steering controller).
                let mut x = state.pose.point.x;
                let mut y = state.pose.point.y;
                let mut yaw = state.pose.rotation.to_euler().yaw;
                let mut min_clearance = f64::INFINITY;
                let mut crashed = false;
                for _ in 0..steps {
                    x += v * yaw.cos() * dwa.dt;
                    y += v * yaw.sin() * dwa.dt;
                    yaw = normalize_angle(yaw + w * dwa.dt);
                    for (ox, oy, r) in &obstacles {
                        let d = ((x - ox).powi(2) + (y - oy).powi(2)).sqrt() - r;
                        if d < dwa.obstacle_margin {
                            crashed = true;
                            break;
                        }
                        if d < min_clearance {
                            min_clearance = d;
                        }
                    }
                    if crashed {
                        break;
                    }
                }
                if crashed {
                    continue;
                }

                // Heading: smaller = better (aligned with goal direction).
                let dx = goal_point.x - x;
                let dy = goal_point.y - y;
                let desired_heading = dy.atan2(dx);
                let heading_err = normalize_angle(desired_heading - yaw).abs();

                // Distance: residual to goal.
                let distance = (dx * dx + dy * dy).sqrt();

                // Velocity: encourage higher forward motion.
                let velocity_score = (dwa.target_velocity - v.max(0.0)).abs();

                // Clearance: larger clearance is better (lower cost).
                let clearance = if min_clearance.is_finite() {
                    1.0 / (min_clearance + 1e-3)
                } else {
                    0.0
                };

                samples.push((v, w, heading_err, distance, velocity_score, clearance));
            }
        }

        if samples.is_empty() {
            return VelocityCommand::invalid("DWA: all trajectories unsafe");
        }

        // Normalize each column to [0, 1] before weighting, so the weights
        // behave intuitively regardless of magnitude.
        let max_of = |i: usize| {
            samples
                .iter()
                .map(|s| match i {
                    0 => s.2,
                    1 => s.3,
                    2 => s.4,
                    3 => s.5,
                    _ => 0.0,
                })
                .fold(0.0_f64, f64::max)
                .max(1e-9)
        };
        let m_heading = max_of(0);
        let m_distance = max_of(1);
        let m_velocity = max_of(2);
        let m_clearance = max_of(3);

        for (v, w, heading, distance, velocity, clearance) in samples {
            let cost = dwa.weight_heading * (heading / m_heading)
                + dwa.weight_distance * (distance / m_distance)
                + dwa.weight_velocity * (velocity / m_velocity)
                + dwa.weight_clearance * (clearance / m_clearance);
            if cost < best_cost {
                best_cost = cost;
                best_v = v;
                best_w = w;
            }
        }

        let angular_output = if is_diff {
            best_w.clamp(
                -constraints.max_angular_velocity,
                constraints.max_angular_velocity,
            )
        } else {
            // Ackermann: convert yaw rate to steering angle if needed —
            // for the simple DWA we pass yaw rate through, the kinematic
            // model handles Ackermann in the simulator.
            best_w.clamp(
                -constraints.max_angular_velocity,
                constraints.max_angular_velocity,
            )
        };

        self.base.status.distance_to_goal = dist_to_goal;
        self.base.status.heading_error = yaw_diff;
        let _ = PI; // silence unused-import in case of future cleanup
        self.base.status.goal_reached = false;
        self.base.status.mode = "dwa".into();

        VelocityCommand {
            valid: true,
            status_message: "DWA tracking".into(),
            linear_velocity: best_v,
            angular_velocity: angular_output,
            ..VelocityCommand::default()
        }
    }

    fn get_type(&self) -> &'static str {
        "dwa_follower"
    }

    fn base(&self) -> &ControllerBase {
        &self.base
    }

    fn base_mut(&mut self) -> &mut ControllerBase {
        &mut self.base
    }
}
