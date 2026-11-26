import contextlib
import io
import math
import unittest
from unittest.mock import Mock, patch

from physics_bench import MotionModel, RouteDevice


class GeometricMissionTests(unittest.TestCase):
    def simulate(self, task, values, *, drop_event=False, start=(3,-2,0), height=1.5):
        import geometry_missions as missions
        model = MotionModel(start)
        self.device = RouteDevice(model, drop_event)
        self.segments = []
        self.targets = []
        original = self.device.go_to_local_point

        def record(x,y,z):
            self.segments.append(len(model.samples))
            self.targets.append((x,y,z))
            return original(x,y,z)

        land = self.device.land
        def record_landing():
            self.segments.append(len(model.samples))
            return land()
        self.device.land = record_landing
        self.device.go_to_local_point = record
        self.stdout = io.StringIO()
        with contextlib.redirect_stdout(self.stdout), patch.object(missions, "monotonic", lambda: model.data.time), \
             patch.object(missions, "sleep", model.advance):
            missions.run(self.device, task, values, height=height, timeout=5)
        return model

    def test_circle_and_eight_do_full_orbits_and_land(self):
        for task, values, count in (("circle", [0.6], 64), ("eight", [], 128)):
            with self.subTest(task=task):
                model = self.simulate(task, values)
                self.assertEqual(self.device.landings, 1)
                self.assertEqual(self.device.events, count + 1)
                self.assertLess(math.dist(model.position, (3,-2,0)), 0.2)
                self.assertEqual(self.targets[0], self.targets[-1])
                self.assertTrue(all(y == -2 for _,y,_ in self.targets))
                self.assertGreater(max(z for _,_,z in self.targets),2)
                self.assertEqual(self.device.phase, "ground")

    def test_spherical_target_is_relative_to_launch_not_airborne_height(self):
        model = self.simulate("spherical", [2,0,30], start=(3,-2,0.4))
        target = (3+math.sqrt(3), -2, 1.4)
        self.assertLess(math.dist(target, self.targets[0]), 1e-8)
        self.assertEqual(self.targets[-1], (3,-2,1.9))
        import geometry
        self.assertGreaterEqual(geometry.longest_hold(model.samples, target,.02), 3.0)
        self.assertEqual(self.device.landings, 1)

    def test_drop_event_has_only_one_point_and_one_landing_request(self):
        with self.assertRaises(TimeoutError):
            self.simulate("circle", [0.5], drop_event=True)
        self.assertEqual(len(self.targets), 1)
        self.assertEqual(self.device.landings, 1)

    def test_reject_invalid_or_below_floor_before_arm(self):
        import geometry_missions as missions
        for task, values, height in (("eight", [], .8), ("spherical", [2,90,-30], 1.5),
                                     ("circle", [-1], 1.5), ("eight", [], math.nan)):
            drone = Mock()
            drone.get_local_position_lps.return_value = (2, 3, .1)
            with self.subTest(task=task, height=height), self.assertRaises(ValueError):
                missions.run(drone, task, values, height=height, timeout=5)
            drone.arm.assert_not_called()

    def test_hold_needs_three_consecutive_seconds_after_telemetry_dropout(self):
        import geometry_missions as missions
        position = (1,2,3)
        clock = [0.0]
        def advance(seconds):
            clock[0] += seconds
        def sample():
            return None if 1.4 <= clock[0] < 1.6 else position
        drone = Mock()
        drone.get_local_position_lps.side_effect = sample
        with patch.object(missions,"monotonic",lambda: clock[0]), patch.object(missions,"sleep",advance):
            missions.hold(drone,position,6)
        self.assertGreaterEqual(clock[0], 4.6)

    def test_measured_path_stays_in_tube_and_turns_in_opposite_directions(self):
        for task, values in (("circle",[.6]),("eight",[])):
            model = self.simulate(task,values)
            radius = values[0] if values else 1
            centers = [(3,1.5+radius)] if values else [(2,1.5),(4,1.5)]
            turns = []
            for i,(cx,cz) in enumerate(centers):
                a,b = self.segments[1+64*i], self.segments[65+64*i]
                angles = []
                for _,(x,y,z) in model.samples[a:b]:
                    error = math.hypot(math.hypot(x-cx,z-cz)-radius,y+2)
                    self.assertLessEqual(error,0.2)
                    angles.append(math.atan2(z-cz,x-cx))
                turn = sum(math.remainder(b-a, math.tau) for a,b in zip(angles,angles[1:]))
                self.assertLess(abs(abs(turn)-math.tau),.15)
                turns.append(turn)
            if task == "eight":
                self.assertLess(turns[0]*turns[1],0)