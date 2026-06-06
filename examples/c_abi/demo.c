#include <math.h>
#include <stdbool.h>
#include <stdio.h>

#include "../../include/ondrive.h"

static void fail(const char* what) {
    fprintf(stderr, "%s failed: %s\n", what, ondrive_last_error_message());
}

int main(void) {
    printf("ondrive version: %s\n", ondrive_version());

    /* ---------------- Build a straight path and smoothen it ------------- */
    OndrivePath* path = ondrive_path_new();
    for (int i = 0; i < 6; ++i) {
        if (!ondrive_path_add_waypoint_xy(path, (double)i * 2.0, 0.0, 0.0, 0.5)) {
            fail("path_add_waypoint_xy");
            ondrive_path_free(path);
            return 1;
        }
    }
    printf("path before smoothen: %zu waypoints\n", ondrive_path_len(path));
    ondrive_path_smoothen(path, 0.5);
    printf("path after smoothen:  %zu waypoints\n", ondrive_path_len(path));

    /* ---------------- Configure a tracker ------------------------------- */
    OndriveTracker* t = ondrive_tracker_new(ONDRIVE_KIND_PURE_PURSUIT);
    if (t == NULL) {
        fail("tracker_new");
        ondrive_path_free(path);
        return 1;
    }

    OndriveRobotConstraints cons = ondrive_default_constraints();
    cons.steering_type         = ONDRIVE_STEERING_ACKERMANN;
    cons.max_linear_velocity   = 1.0;
    cons.max_angular_velocity  = 2.0;
    cons.max_steering_angle    = 0.6;
    cons.wheelbase             = 0.5;
    ondrive_tracker_init(t, cons);

    OndriveControllerConfig cfg = ondrive_default_config();
    cfg.lookahead_distance = 1.2;
    cfg.goal_tolerance     = 0.4;
    cfg.angular_tolerance  = 1.0;
    cfg.output_units       = ONDRIVE_UNITS_PHYSICAL;
    ondrive_tracker_set_config(t, cfg);

    ondrive_tracker_set_path(t, path);

    /* Goal = last waypoint */
    OndrivePose final_pose;
    ondrive_path_waypoint(path, ondrive_path_len(path) - 1, &final_pose);
    OndriveGoal goal = {
        .target_pose           = final_pose,
        .has_target_velocity   = false,
        .target_velocity       = {0, 0, 0},
        .tolerance_position    = 0.4,
        .tolerance_orientation = 1.0,
    };
    ondrive_tracker_set_goal(t, goal);

    /* ---------------- Simulate ------------------------------------------ */
    OndriveRobotState state;
    state.pose.position       = (OndriveVec3){0.0, 0.2, 0.0};
    state.pose.rotation       = ondrive_quaternion_from_yaw(0.0);
    state.velocity            = (OndriveVelocity){0.0, 0.0, 0.0};
    state.timestamp           = 0.0;
    state.allow_reverse       = false;
    state.turn_first          = false;
    state.allow_move          = true;
    state.has_trailer         = false;
    state.trailer_pose        = state.pose;

    const double dt = 0.05;
    double t_sim = 0.0;
    bool reached = false;

    for (int i = 0; i < 2000; ++i) {
        OndriveVelocityCommand cmd;
        if (!ondrive_tracker_tick(t, state, dt, NULL, &cmd)) {
            fail("tick");
            break;
        }
        if (!cmd.valid) {
            fprintf(stderr, "invalid command: %s\n",
                    ondrive_tracker_last_command_message());
            break;
        }

        /* Kinematic integration. */
        double yaw = ondrive_quaternion_yaw(state.pose.rotation);
        state.pose.position.x += cmd.linear_velocity * cos(yaw) * dt;
        state.pose.position.y += cmd.linear_velocity * sin(yaw) * dt;
        state.pose.rotation    =
            ondrive_quaternion_from_yaw(yaw + cmd.angular_velocity * dt);
        state.velocity.linear  = cmd.linear_velocity;
        state.velocity.angular = cmd.angular_velocity;
        t_sim += dt;

        if (i % 40 == 0) {
            OndriveControllerStatus st;
            ondrive_tracker_status(t, &st);
            printf(
                "t=%5.2f pos=(%5.2f,%5.2f) v=%.2f w=%.2f cte=%.3f mode=%s\n",
                t_sim,
                state.pose.position.x, state.pose.position.y,
                cmd.linear_velocity, cmd.angular_velocity,
                st.cross_track_error,
                ondrive_tracker_status_mode());
        }

        if (ondrive_tracker_is_goal_reached(t)) {
            reached = true;
            break;
        }
    }

    if (reached) {
        printf("goal reached at t=%.2f\n", t_sim);
    } else {
        printf("budget exhausted at t=%.2f, pos=(%.2f,%.2f)\n",
               t_sim, state.pose.position.x, state.pose.position.y);
    }

    /* ---------------- Check kind round-trips ---------------------------- */
    printf("tracker kind = %u (expected %u for PurePursuit)\n",
           ondrive_tracker_kind(t), ONDRIVE_KIND_PURE_PURSUIT);

    OndriveVec3 target;
    if (ondrive_tracker_current_target(t, &target)) {
        printf("current target = (%.2f, %.2f)\n", target.x, target.y);
    }

    /* ---------------- Emergency stop ------------------------------------ */
    OndriveVelocityCommand stop;
    ondrive_tracker_emergency_stop(t, &stop);
    printf("emergency stop: v=%.2f w=%.2f msg=%s\n",
           stop.linear_velocity, stop.angular_velocity,
           ondrive_tracker_last_command_message());

    /* ---------------- World / obstacle demo ----------------------------- */
    OndriveWorld* world = ondrive_world_new();
    ondrive_world_add_static_gaussian_obstacle(world, 0, 3.0, 0.2, 0.3, 0.1, 0.1, 10);
    printf("world obstacles: %zu\n", ondrive_world_obstacle_count(world));

    ondrive_world_free(world);
    ondrive_tracker_free(t);
    ondrive_path_free(path);
    return 0;
}
