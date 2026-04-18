#ifndef ONDRIVE_H
#define ONDRIVE_H

/*
 * C ABI for the ondrive motion-control / path-tracking library.
 *
 * Linking: the Rust crate is built as `cdylib`, producing libondrive.so
 * (or libondrive.dylib / ondrive.dll). Link with `-londrive`.
 *
 * Error reporting: functions that return `bool` indicate success/failure;
 * on failure, call `ondrive_last_error_message` for a human-readable
 * description.
 *
 * Memory: opaque handles (Path, World, Tracker) are owned by the caller
 * and must be freed with the matching *_free function.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------------------------------------------------------------------
 * Enum constants (mirror the Rust `TrackerKind`, `SteeringType`, `OutputUnits`).
 * ------------------------------------------------------------------------- */

#define ONDRIVE_KIND_PID           0u
#define ONDRIVE_KIND_CARROT        1u
#define ONDRIVE_KIND_PURE_PURSUIT  2u
#define ONDRIVE_KIND_STANLEY       3u
#define ONDRIVE_KIND_LQR           4u
#define ONDRIVE_KIND_MPC           5u
#define ONDRIVE_KIND_MPPI          6u
#define ONDRIVE_KIND_MCA           7u
#define ONDRIVE_KIND_SOC           8u
#define ONDRIVE_KIND_DWA           9u
#define ONDRIVE_KIND_TEB          10u
#define ONDRIVE_KIND_FLC          11u

#define ONDRIVE_STEERING_DIFFERENTIAL  0u
#define ONDRIVE_STEERING_ACKERMANN     1u
#define ONDRIVE_STEERING_HOLONOMIC     2u
#define ONDRIVE_STEERING_SKID_STEER    3u

#define ONDRIVE_UNITS_NORMALIZED  0u
#define ONDRIVE_UNITS_PHYSICAL    1u

/* ---------------------------------------------------------------------------
 * POD structs (pass by value).
 * ------------------------------------------------------------------------- */

typedef struct {
    double x;
    double y;
    double z;
} OndriveVec3;

typedef struct {
    double x;
    double y;
    double z;
    double w;
} OndriveQuat;

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
    OndrivePose     pose;
    OndriveVelocity velocity;
    double          timestamp;
    bool            allow_reverse;
    bool            turn_first;
    bool            allow_move;
    bool            has_trailer;
    OndrivePose     trailer_pose;
} OndriveRobotState;

typedef struct {
    OndrivePose     target_pose;
    bool            has_target_velocity;
    OndriveVelocity target_velocity;
    double          tolerance_position;
    double          tolerance_orientation;
} OndriveGoal;

typedef struct {
    uint32_t steering_type;
    double   wheelbase;
    double   track_width;
    double   wheel_radius;
    double   max_linear_velocity;
    double   min_linear_velocity;
    double   max_angular_velocity;
    double   max_linear_acceleration;
    double   max_angular_acceleration;
    double   max_steering_angle;
    double   max_steering_rate;
    double   min_turning_radius;
    double   rear_wheelbase;
    double   max_rear_steering_angle;
    double   robot_width;
    double   robot_length;
} OndriveRobotConstraints;

typedef struct {
    uint32_t output_units;
    double   kp_linear;
    double   ki_linear;
    double   kd_linear;
    double   kp_angular;
    double   ki_angular;
    double   kd_angular;
    double   lookahead_distance;
    double   k_cross_track;
    double   k_heading;
    bool     allow_reverse;
    double   goal_tolerance;
    double   angular_tolerance;
} OndriveControllerConfig;

typedef struct {
    bool   goal_reached;
    double distance_to_goal;
    double cross_track_error;
    double heading_error;
} OndriveControllerStatus;

typedef struct {
    bool     valid;
    double   linear_velocity;
    double   angular_velocity;
    double   lateral_velocity;
    uint32_t output_type;
} OndriveVelocityCommand;

/* ---------------------------------------------------------------------------
 * Opaque handles.
 * ------------------------------------------------------------------------- */

typedef struct OndrivePathHandle    OndrivePathHandle;
typedef struct OndriveWorldHandle   OndriveWorldHandle;
typedef struct OndriveTrackerHandle OndriveTrackerHandle;

/* ---------------------------------------------------------------------------
 * Error + helpers.
 * ------------------------------------------------------------------------- */

const char* ondrive_last_error_message(void);
const char* ondrive_version(void);

OndriveControllerConfig  ondrive_default_config(void);
OndriveRobotConstraints  ondrive_default_constraints(void);

OndriveQuat ondrive_quaternion_from_yaw(double yaw);
double      ondrive_quaternion_yaw(OndriveQuat q);

/* ---------------------------------------------------------------------------
 * Path handle.
 * ------------------------------------------------------------------------- */

OndrivePathHandle* ondrive_path_new(void);
void               ondrive_path_free(OndrivePathHandle* h);

bool    ondrive_path_add_waypoint(OndrivePathHandle* h, OndrivePose pose, double speed);
bool    ondrive_path_add_waypoint_xy(
            OndrivePathHandle* h, double x, double y, double yaw, double speed);
size_t  ondrive_path_len(const OndrivePathHandle* h);
bool    ondrive_path_waypoint(const OndrivePathHandle* h, size_t idx, OndrivePose* out);
bool    ondrive_path_clear(OndrivePathHandle* h);
bool    ondrive_path_set_closed(OndrivePathHandle* h, bool closed);
bool    ondrive_path_smoothen(OndrivePathHandle* h, double max_segment_m);

/* ---------------------------------------------------------------------------
 * World (obstacles) handle.
 * ------------------------------------------------------------------------- */

OndriveWorldHandle* ondrive_world_new(void);
void                ondrive_world_free(OndriveWorldHandle* h);
bool                ondrive_world_clear(OndriveWorldHandle* h);
size_t              ondrive_world_obstacle_count(const OndriveWorldHandle* h);

/* Single Gaussian mode pinned at (x, y) for `horizon_steps`. */
bool ondrive_world_add_static_gaussian_obstacle(
    OndriveWorldHandle* h,
    uint64_t id,
    double x,
    double y,
    double radius,
    double std_x,
    double std_y,
    size_t horizon_steps);

/* Gaussian mode with a per-step mean trajectory and a fixed std. */
bool ondrive_world_add_trajectory_obstacle(
    OndriveWorldHandle* h,
    uint64_t id,
    double radius,
    const double* mean_x,
    const double* mean_y,
    double std_x,
    double std_y,
    size_t horizon_steps);

/* ---------------------------------------------------------------------------
 * Tracker handle — the main entry point.
 * ------------------------------------------------------------------------- */

OndriveTrackerHandle* ondrive_tracker_new(uint32_t kind);
void                  ondrive_tracker_free(OndriveTrackerHandle* h);

bool ondrive_tracker_init(
    OndriveTrackerHandle* h, OndriveRobotConstraints constraints);

bool ondrive_tracker_set_config(
    OndriveTrackerHandle* h, OndriveControllerConfig config);
bool ondrive_tracker_get_config(
    const OndriveTrackerHandle* h, OndriveControllerConfig* out);

bool ondrive_tracker_set_goal(OndriveTrackerHandle* h, OndriveGoal goal);
bool ondrive_tracker_clear_goal(OndriveTrackerHandle* h);

bool ondrive_tracker_set_path(
    OndriveTrackerHandle* h, const OndrivePathHandle* path);
bool ondrive_tracker_clear_path(OndriveTrackerHandle* h);
bool ondrive_tracker_smoothen(OndriveTrackerHandle* h, double max_segment_m);

bool ondrive_tracker_reset(OndriveTrackerHandle* h);

bool ondrive_tracker_tick(
    OndriveTrackerHandle* h,
    OndriveRobotState state,
    double dt,
    const OndriveWorldHandle* world, /* may be NULL */
    OndriveVelocityCommand* out_cmd);

bool ondrive_tracker_emergency_stop(
    OndriveTrackerHandle* h, OndriveVelocityCommand* out_cmd);

bool ondrive_tracker_status(
    const OndriveTrackerHandle* h, OndriveControllerStatus* out);

/* Textual status fields live in thread-local storage and are refreshed
 * after `ondrive_tracker_status` / `ondrive_tracker_tick` /
 * `ondrive_tracker_emergency_stop`. The returned pointer is valid until
 * the next call that refreshes the corresponding slot. */
const char* ondrive_tracker_status_mode(void);
const char* ondrive_tracker_last_command_message(void);

bool ondrive_tracker_is_goal_reached(const OndriveTrackerHandle* h);
bool ondrive_tracker_current_target(const OndriveTrackerHandle* h, OndriveVec3* out);

uint32_t ondrive_tracker_kind(const OndriveTrackerHandle* h);
bool     ondrive_tracker_constraints(
            const OndriveTrackerHandle* h, OndriveRobotConstraints* out);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ONDRIVE_H */
