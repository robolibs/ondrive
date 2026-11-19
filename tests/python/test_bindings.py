"""Python bindings: construction, ticking, controller parameters, seeds,
obstacles, zones, footprints, input checks and threads.

    make test-py
"""

import math
import threading
import unittest

import ondrive


def line(n=40, step=0.25):
    path = ondrive.Path()
    for i in range(n):
        path.add_waypoint_xy(i * step, 0.0, yaw=0.0)
    return path


def tracker(kind, steering="differential"):
    t = ondrive.Tracker(kind)
    c = ondrive.RobotConstraints.default_()
    c.steering_type = steering
    t.init(c)
    t.set_path(line())
    return t


def state(x=0.0, y=0.0, yaw=0.0, **kw):
    return ondrive.RobotState(pose=((x, y, 0.0), yaw), allow_move=True, **kw)


class Bindings(unittest.TestCase):
    def test_every_kind_ticks_a_valid_command(self):
        for kind in ondrive.available_tracker_kinds():
            cmd = tracker(kind).tick(state(), 0.05)
            self.assertTrue(cmd.valid, f"{kind}: {cmd.status_message}")

    def test_controller_parameters_round_trip(self):
        t = ondrive.Tracker("mppi")
        self.assertIn("horizon_steps", t.params())
        t.set_params({"horizon_steps": 12, "num_samples": 64})
        self.assertEqual(t.params()["horizon_steps"], 12)
        with self.assertRaises(ValueError):
            t.set_params({"no_such_thing": 1})
        self.assertEqual(ondrive.Tracker("pid").params(), {})

    def test_a_seed_makes_sampling_controllers_repeatable(self):
        def run(seed):
            t = tracker("mppi")
            t.set_params({"num_samples": 64})
            t.set_seed(seed)
            return [round(t.tick(state(0.1 * i, 0.2), 0.05).angular_velocity, 12) for i in range(5)]

        self.assertEqual(run(3), run(3))

    def test_predicted_trajectory_is_exposed(self):
        t = tracker("mppi")
        t.tick(state(), 0.05)
        plan = t.predicted_trajectory()
        self.assertGreater(len(plan), 2)
        self.assertEqual(len(plan[0]), 3)

    def test_debug_frames_show_rollouts_and_guides(self):
        t = tracker("mppi")
        t.set_params({"num_samples": 128})
        world = ondrive.World()
        world.add_obstacle(0, 0.4, [(1.0, [3.0] * 10, [0.0] * 10, 0.05, 0.05)], dt=0.2)
        t.tick(state(), 0.05, world)
        self.assertEqual(t.debug_frame(), {"rollouts": [], "costs": [], "guides": []})
        t.set_debug("full")
        t.tick(state(), 0.05, world)
        frame = t.debug_frame()
        self.assertEqual(len(frame["rollouts"]), len(frame["costs"]))
        self.assertGreater(len(frame["rollouts"]), 0)
        self.assertGreater(len(frame["guides"]), 0)
        self.assertEqual(len(frame["rollouts"][0][0]), 2)
        with self.assertRaises(ValueError):
            t.set_debug("everything")

    def test_timed_obstacle_mixtures_and_zones(self):
        world = ondrive.World()
        world.add_obstacle(1, 0.4, [(1.0, [2.0, 2.0], [0.0, 0.0], 0.05, [0.05, 0.1])], dt=0.2)
        world.add_zone([(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)], 0.2)
        self.assertEqual(world.obstacle_count(), 1)
        t = tracker("pure_pursuit")
        cmd = t.tick(state(velocity=(1.0, 0.0, 0.0)), 0.05, world)
        self.assertTrue(cmd.valid)
        self.assertLessEqual(abs(cmd.linear_velocity), 0.2 + 1e-9)
        with self.assertRaises(ValueError):
            world.add_obstacle(2, 0.4, [(1.0, [0.0, 1.0], [0.0], 0.1, 0.1)])

    def test_velocity_is_measured_only_when_given(self):
        self.assertIsNone(state().velocity)
        self.assertEqual(state(velocity=(0.0, 0.0, 0.0)).velocity, (0.0, 0.0, 0.0))

    def test_footprints(self):
        c = ondrive.RobotConstraints.default_()
        c.set_footprint(radius=0.5)
        c.set_footprint(points=[(0.5, 0.3), (-0.5, 0.3), (-0.5, -0.3), (0.5, -0.3)])
        with self.assertRaises(ValueError):
            c.set_footprint()

    def test_bad_input_is_an_invalid_command_not_an_exception(self):
        cmd = tracker("stanley").tick(state(x=math.nan), 0.05)
        self.assertFalse(cmd.valid)

    def test_trackers_tick_in_parallel_threads(self):
        results = []

        def work():
            t = tracker("mca")
            t.set_params({"num_samples": 64})
            results.append(all(t.tick(state(0.1 * i), 0.05).valid for i in range(10)))

        threads = [threading.Thread(target=work) for _ in range(4)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        self.assertEqual(results, [True] * 4)


if __name__ == "__main__":
    unittest.main()
