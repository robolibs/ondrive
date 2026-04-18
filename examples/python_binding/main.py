"""Path-following demo — drive Pure Pursuit on a sine-wave path."""

import math

import ondrive


def sine_path(n: int) -> ondrive.Path:
    path = ondrive.Path()
    for i in range(n):
        x = i * 0.4
        y = math.sin(x * 0.4) * 0.8
        path.add_waypoint_xy(x, y, yaw=0.0)
    return path


def main() -> None:
    path = sine_path(40)
    print(f"raw path:       {len(path)} waypoints")
    path.smoothen(0.15)
    print(f"after smoothen: {len(path)} waypoints")

    tracker = ondrive.Tracker("pure_pursuit")

    cfg = ondrive.ControllerConfig.default_()
    cfg.lookahead_distance = 1.2
    cfg.goal_tolerance = 0.4
    cfg.angular_tolerance = 1.0
    cfg.output_units = "physical"
    tracker.set_config(cfg)

    cons = ondrive.RobotConstraints.default_(
    )
    cons.steering_type = "ackermann"
    cons.max_linear_velocity = 1.0
    cons.max_angular_velocity = 2.0
    cons.max_steering_angle = 0.6
    cons.wheelbase = 0.5
    tracker.init(cons)

    tracker.set_path(path)

    final_wp = path.waypoint(len(path) - 1)
    tracker.set_goal(
        ondrive.Goal(
            target_pose=final_wp,
            tolerance_position=0.4,
            tolerance_orientation=1.0,
        )
    )

    # An obstacle world the tracker can optionally respect.
    world = ondrive.World()
    world.add_static_gaussian_obstacle(id=0, x=7.0, y=0.0, radius=0.3, horizon_steps=20)
    print(f"world has {world.obstacle_count()} obstacles")

    state = ondrive.RobotState(pose=((0.0, 0.2, 0.0), 0.0), allow_move=True)

    dt = 0.05
    t = 0.0
    for step in range(2000):
        cmd = tracker.tick(state, dt)  # world not passed — Pure Pursuit ignores it
        if not cmd.valid:
            print("invalid:", cmd.status_message)
            break

        (x, y, z), yaw = state.pose
        x += cmd.linear_velocity * math.cos(yaw) * dt
        y += cmd.linear_velocity * math.sin(yaw) * dt
        yaw += cmd.angular_velocity * dt
        state.pose = ((x, y, z), yaw)
        state.velocity = (cmd.linear_velocity, cmd.angular_velocity, 0.0)
        t += dt

        if step % 20 == 0:
            status = tracker.get_status()
            print(
                f"t={t:5.2f}  pos=({x:5.2f},{y:5.2f})  yaw={yaw:5.2f}  "
                f"v={cmd.linear_velocity:.2f} w={cmd.angular_velocity:.2f}  "
                f"cte={status.cross_track_error:.3f}"
            )

        if tracker.is_goal_reached():
            print(f"goal reached at t={t:.2f}")
            break

    stop = tracker.emergency_stop()
    print("emergency stop:", stop)


if __name__ == "__main__":
    main()
