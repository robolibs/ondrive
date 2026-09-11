#ifndef ONDRIVE_H
#define ONDRIVE_H

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define ONDRIVE_KIND_PID 0

#define ONDRIVE_KIND_CARROT 1

#define ONDRIVE_KIND_PURE_PURSUIT 2

#define ONDRIVE_KIND_STANLEY 3

#define ONDRIVE_KIND_LQR 4

#define ONDRIVE_KIND_MPC 5

#define ONDRIVE_KIND_MPPI 6

#define ONDRIVE_KIND_MCA 7

#define ONDRIVE_KIND_SOC 8

#define ONDRIVE_KIND_DWA 9

#define ONDRIVE_KIND_TEB 10

#define ONDRIVE_KIND_FLC 11

#define ONDRIVE_STEERING_DIFFERENTIAL 0

#define ONDRIVE_STEERING_ACKERMANN 1

#define ONDRIVE_STEERING_HOLONOMIC 2

#define ONDRIVE_STEERING_SKID_STEER 3

#define ONDRIVE_UNITS_NORMALIZED 0

#define ONDRIVE_UNITS_PHYSICAL 1

typedef struct OndrivePath OndrivePath;

typedef struct OndriveTracker OndriveTracker;

typedef struct OndriveWorld OndriveWorld;

typedef struct {
  uint32_t output_units;
  double kp_linear;
  double ki_linear;
  double kd_linear;
  double kp_angular;
  double ki_angular;
  double kd_angular;
  double lookahead_distance;
  double k_cross_track;
  double k_heading;
  bool allow_reverse;
  double goal_tolerance;
  double angular_tolerance;
} OndriveControllerConfig;

typedef struct {
  uint32_t steering_type;
  double wheelbase;
  double track_width;
  double wheel_radius;
  double max_linear_velocity;
  double min_linear_velocity;
  double max_angular_velocity;
  double max_linear_acceleration;
  double max_angular_acceleration;
  double max_steering_angle;
  double max_steering_rate;
  double min_turning_radius;
  double rear_wheelbase;
  double max_rear_steering_angle;
  double robot_width;
  double robot_length;
} OndriveRobotConstraints;

typedef struct {
  double x;
  double y;
  double z;
  double w;
} OndriveQuat;

typedef struct {
  double x;
  double y;
  double z;
} OndriveVec3;

typedef struct {
  OndriveVec3 position;
  OndriveQuat rotation;
} OndrivePose;

typedef struct {
  double linear;
  double angular;
  double lateral;
} OndriveVelocity;

typedef struct {
  OndrivePose target_pose;
  bool has_target_velocity;
  OndriveVelocity target_velocity;
  double tolerance_position;
  double tolerance_orientation;
} OndriveGoal;

typedef struct {
  OndrivePose pose;
  OndriveVelocity velocity;
  double timestamp;
  bool allow_reverse;
  bool turn_first;
  bool allow_move;
  bool has_trailer;
  OndrivePose trailer_pose;
} OndriveRobotState;

typedef struct {
  bool valid;
  double linear_velocity;
  double angular_velocity;
  double lateral_velocity;
  /**
   * Ackermann steering angle (rad) consistent with `angular_velocity`.
   */
  double steering_angle;
  uint32_t output_type;
} OndriveVelocityCommand;

typedef struct {
  bool goal_reached;
  double distance_to_goal;
  double cross_track_error;
  double heading_error;
} OndriveControllerStatus;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

const char *ondrive_last_error_message(void);

OndriveControllerConfig ondrive_default_config(void);

OndriveRobotConstraints ondrive_default_constraints(void);

OndriveQuat ondrive_quaternion_from_yaw(double yaw);

double ondrive_quaternion_yaw(OndriveQuat q);

OndrivePath *ondrive_path_new(void);

void ondrive_path_free(OndrivePath *h);

bool ondrive_path_add_waypoint(OndrivePath *h, OndrivePose pose, double speed);

bool ondrive_path_add_waypoint_xy(OndrivePath *h, double x, double y, double yaw, double speed);

uintptr_t ondrive_path_len(const OndrivePath *h);

bool ondrive_path_waypoint(const OndrivePath *h, uintptr_t idx, OndrivePose *out);

bool ondrive_path_clear(OndrivePath *h);

bool ondrive_path_set_closed(OndrivePath *h, bool closed);

bool ondrive_path_smoothen(OndrivePath *h, double max_segment_m);

OndriveWorld *ondrive_world_new(void);

void ondrive_world_free(OndriveWorld *h);

bool ondrive_world_clear(OndriveWorld *h);

uintptr_t ondrive_world_obstacle_count(const OndriveWorld *h);

/**
 * Add an obstacle with a single Gaussian mode whose mean stays at (x, y)
 * for `horizon_steps` time steps (standard deviations `std_x`, `std_y`).
 */
bool ondrive_world_add_static_gaussian_obstacle(OndriveWorld *h,
                                                uint64_t id,
                                                double x,
                                                double y,
                                                double radius,
                                                double std_x,
                                                double std_y,
                                                uintptr_t horizon_steps);

/**
 * Add an obstacle with a provided mean trajectory and fixed std across the
 * horizon. `mean_x` / `mean_y` must point to `horizon_steps` doubles.
 */
bool ondrive_world_add_trajectory_obstacle(OndriveWorld *h,
                                           uint64_t id,
                                           double radius,
                                           const double *mean_x,
                                           const double *mean_y,
                                           double std_x,
                                           double std_y,
                                           uintptr_t horizon_steps);

OndriveTracker *ondrive_tracker_new(uint32_t kind);

void ondrive_tracker_free(OndriveTracker *h);

bool ondrive_tracker_init(OndriveTracker *h, OndriveRobotConstraints constraints);

bool ondrive_tracker_set_config(OndriveTracker *h, OndriveControllerConfig config);

bool ondrive_tracker_get_config(const OndriveTracker *h, OndriveControllerConfig *out);

bool ondrive_tracker_set_goal(OndriveTracker *h, OndriveGoal goal);

bool ondrive_tracker_clear_goal(OndriveTracker *h);

bool ondrive_tracker_set_path(OndriveTracker *h, const OndrivePath *path);

bool ondrive_tracker_clear_path(OndriveTracker *h);

bool ondrive_tracker_reset(OndriveTracker *h);

bool ondrive_tracker_smoothen(OndriveTracker *h, double max_segment_m);

bool ondrive_tracker_tick(OndriveTracker *h,
                          OndriveRobotState state,
                          double dt,
                          const OndriveWorld *world,
                          OndriveVelocityCommand *out_cmd);

bool ondrive_tracker_emergency_stop(OndriveTracker *h, OndriveVelocityCommand *out_cmd);

bool ondrive_tracker_status(const OndriveTracker *h, OndriveControllerStatus *out);

const char *ondrive_tracker_status_mode(void);

const char *ondrive_tracker_last_command_message(void);

bool ondrive_tracker_is_goal_reached(const OndriveTracker *h);

/**
 * True once a path driven without an explicit goal has been consumed.
 */
bool ondrive_tracker_is_path_completed(const OndriveTracker *h);

bool ondrive_tracker_current_target(const OndriveTracker *h, OndriveVec3 *out);

uint32_t ondrive_tracker_kind(const OndriveTracker *h);

bool ondrive_tracker_constraints(const OndriveTracker *h, OndriveRobotConstraints *out);

const char *ondrive_version(void);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  /* ONDRIVE_H */
