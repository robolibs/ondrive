"""Minimal Python binding demo — drive a PID follower to a point goal."""

import math

import ondrive


def main() -> None:
    print("ondrive version:", ondrive.__version__)
    print("available kinds:", ondrive.available_tracker_kinds())
    print("available steering types:", ondrive.available_steering_types())

    tracker = ondrive.Tracker("pid")

    cfg = ondrive.ControllerConfig.default_()
    cfg.kp_linear = 2.0
    cfg.kp_angular = 1.5
    cfg.goal_tolerance = 0.2
    cfg.angular_tolerance = 0.4
    tracker.set_config(cfg)

    cons = ondrive.RobotConstraints.default_()
    cons.max_linear_velocity = 0.5
    cons.max_angular_velocity = 1.5
    tracker.init(cons)

    # Goal at (5, 0) with yaw = 0.
    tracker.set_goal(
        ondrive.Goal(
            target_pose=((5.0, 0.0, 0.0), 0.0),
            tolerance_position=0.2,
            tolerance_orientation=0.4,
        )
    )

    state = ondrive.RobotState(pose=((0.0, 0.0, 0.0), 0.0), allow_move=True)

    dt = 0.05
    t = 0.0
    for step in range(1000):
        cmd = tracker.tick(state, dt)
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
                f"v={cmd.linear_velocity:.3f} w={cmd.angular_velocity:.3f}  "
                f"mode={status.mode}"
            )

        if tracker.is_goal_reached():
            print(f"goal reached at t={t:.2f}")
            break

    print("current target:", tracker.current_target())


if __name__ == "__main__":
    main()
