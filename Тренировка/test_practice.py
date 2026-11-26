"""Критерии домашних упражнений. Входы камеры синтетические, SDK заменён."""
import math
import unittest
from unittest.mock import Mock

import cv2
import numpy as np


class GeometryTests(unittest.TestCase):
    def test_circle_lies_in_xz_and_makes_one_turn(self):
        from geometry import circle
        for radius in (0.1, 1.0, 3.7):
            path = circle((2, -4, 0.5), radius)
            self.assertEqual(path[0], path[-1])
            self.assertEqual(path[0], (2, -4, 0.5))
            for x, y, z in path:
                self.assertEqual(y, -4)
                self.assertAlmostEqual(math.hypot(x - 2, z - 0.5 - radius), radius)
            self.check_turn(path, (2, 0.5 + radius))
            for a, b in zip(path, path[1:]):
                chord = math.dist(a, b)
                self.assertLessEqual(radius - math.sqrt(radius**2 - (chord / 2)**2), 0.2)

    def check_turn(self, path, center):
        angles = [math.atan2(z - center[1], x - center[0]) for x, _, z in path]
        deltas = [math.remainder(b - a, math.tau) for a, b in zip(angles, angles[1:])]
        self.assertAlmostEqual(sum(abs(d) for d in deltas), math.tau)
        self.assertAlmostEqual(abs(sum(deltas)), math.tau)
        return sum(deltas)

    def test_figure_eight_tangency_and_opposite_turns(self):
        from geometry import figure_eight
        start = (4, -3, 2)
        path = figure_eight(start)
        middle = len(path) // 2
        self.assertEqual([i for i, p in enumerate(path) if p == start], [0, middle, len(path)-1])
        turns = []
        for half, cx in ((path[:middle+1], 3), (path[middle:], 5)):
            for x, y, z in half:
                self.assertEqual(y, -3)
                self.assertAlmostEqual(math.hypot(x-cx, z-2), 1)
            turns.append(self.check_turn(half, (cx, 2)))
        self.assertLess(turns[0]*turns[1], 0)

    def test_spherical_relative_to_start(self):
        from geometry import spherical_target
        for az, el, offset in [(0, 0, (2,0,0)), (90, 0, (0,2,0)),
                               (180, 0, (-2,0,0)), (-90, 0, (0,-2,0)),
                               (47, 90, (0,0,2)), (23, -90, (0,0,-2)),
                               (45, 30, (math.sqrt(1.5), math.sqrt(1.5), 1))]:
            start = (4, -3, 1)
            actual = spherical_target(start, 2, az, el)
            for a, origin, delta in zip(actual, start, offset):
                self.assertAlmostEqual(a, origin+delta)

    def test_geometry_bad_inputs(self):
        from geometry import circle, spherical_target
        for r in (0, -1, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                circle((0,0,1), r)
        for args in [((1,2), 1, 0, 0), ((1,2,3), -1, 0, 0),
                     ((1,2,3), 1, float("nan"), 0), ((1,2,3), 1, 0, 91)]:
            with self.assertRaises(ValueError):
                spherical_target(*args)


class FrameTests(unittest.TestCase):
    def test_bright_zone_moves_and_can_be_ambiguous(self):
        from frames import bright_zone
        frame = np.zeros((180, 260, 3), np.uint8)
        self.assertIsNone(bright_zone(frame, 180, 500))
        for x in (35, 170):
            with self.subTest(x=x):
                sample = frame.copy()
                cv2.circle(sample, (x,85), 22, (0,255,255), -1)
                sample[12,12] = 255
                self.assertEqual(bright_zone(sample, 180, 500), (x,85))
        cv2.circle(sample, (35,85), 22, (255,255,255), -1)
        self.assertIsNone(bright_zone(sample, 180, 500))

    def test_line_cross_section_lost_or_forked(self):
        from frames import line_x
        im = np.full((60, 160, 3), 255, np.uint8)
        self.assertIsNone(line_x(im, 35, 80, 4))
        im[30:40, 50:61] = 0
        im[35, 5] = 0
        self.assertEqual(line_x(im, 35, 80, 4), 55)
        self.assertIsNone(line_x(im, 10, 80, 4))
        im[30:40, 90:100] = 0
        self.assertIsNone(line_x(im, 35, 80, 4))

    def test_gate_requires_opening_and_selected_color(self):
        from frames import gate_center
        im = np.zeros((180, 330, 3), np.uint8)
        cv2.rectangle(im, (20,25), (100,115), (255,0,0), 6)
        cv2.rectangle(im, (130,20), (230,140), (0,0,255), 6)
        cv2.rectangle(im, (250,40), (315,120), (0,0,255), -1)
        self.assertEqual(gate_center(im, "red", 300), (180,80))
        self.assertEqual(gate_center(im, "blue", 300), (60,70))
        self.assertIsNone(gate_center(im, "green", 300))
        cv2.rectangle(im, (252,50), (312,110), (0,0,0), -1)
        self.assertIsNone(gate_center(im, "red", 300))

    def test_real_aruco_decode_selection_and_finish_priority(self):
        from frames import markers, choose_marker, marker_action
        for name in ("DICT_4X4_50", "DICT_ARUCO_ORIGINAL"):
            dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))
            sheet = np.full((160, 370), 255, np.uint8)
            self.assertEqual(markers(cv2.cvtColor(sheet, cv2.COLOR_GRAY2BGR), name), [])
            for ident, x in ((8,25), (15,185)):
                sheet[30:130,x:x+100] = cv2.aruco.generateImageMarker(dictionary, ident, 100)
            found = markers(cv2.cvtColor(sheet, cv2.COLOR_GRAY2BGR), name)
            self.assertEqual(sorted(i for i,_ in found), [8,15])
            np.testing.assert_allclose(choose_marker(found,15), (234.5,79.5), atol=1)
            self.assertIsNone(choose_marker(found,31))
            self.assertIsNone(choose_marker(found+found,8))
        self.assertEqual([marker_action(i,15) for i in (8,7,15)], ["left","right","finish"])

    def test_invalid_frame_parameters(self):
        from frames import line_x, gate_center, markers
        frame = np.zeros((10, 20, 3), np.uint8)
        for operation in (lambda: line_x(frame, -1, 80, 2),
                          lambda: line_x(frame, 2, 256, 2),
                          lambda: gate_center(frame, "purple", 5),
                          lambda: markers(frame, "not_a_dictionary")):
            with self.assertRaises(ValueError):
                operation()


class MonitorTests(unittest.TestCase):
    def test_each_request_reads_again_and_labels_missing_fields(self):
        from monitor import telemetry
        drone = Mock(spec=["get_local_position_lps","get_altitude","get_battery_status","get_orientation"])
        drone.get_local_position_lps.side_effect = [(1,2,3), (-1,0,2)]
        drone.get_battery_status.side_effect = [(11.4,25), (None,24)]
        drone.get_orientation.side_effect = [(0,1,-30), None]
        drone.get_altitude.side_effect = [2.5, float("nan")]
        a, b = telemetry(drone), telemetry(drone)
        self.assertEqual(a["xyz_m"], (1,2,3))
        self.assertEqual(a["altitude_m"], 2.5)
        self.assertEqual(a["battery_V"], 11.4)
        self.assertEqual(a["roll_pitch_yaw"], (0,1,-30))
        self.assertEqual(b["xyz_m"], (-1,0,2))
        for field in ("altitude_m", "battery_V", "roll_pitch_yaw"):
            self.assertIsNone(b[field])
        for getter in drone._mock_methods:
            self.assertEqual(getattr(drone, getter).call_count, 2)

    def test_map_scale_bounds_and_age(self):
        from monitor import MapTrace
        trace = MapTrace((-2, 2, -1, 1), 50)
        self.assertEqual(trace.pixel((-2,-1)), (0,100))
        self.assertEqual(trace.pixel((0,0)), (100,50))
        self.assertEqual(trace.pixel((2,1)), (200,0))
        trace.record(1, (0,0))
        trace.record(2, (1,0))
        self.assertTrue(trace.fresh(4))
        self.assertFalse(trace.fresh(4.001))
        self.assertFalse(trace.fresh(1))
        im = trace.render(2)
        self.assertEqual(im.shape, (101,201,3))
        self.assertFalse(np.array_equal(im[50,125], im[48,125]))
        for stamp, point in ((1,(0,0)), (3,(float("nan"),0)), (3,(0,))):
            with self.assertRaises(ValueError):
                trace.record(stamp,point)

    def test_map_breaks_trace_after_missing_positions(self):
        from monitor import MapTrace
        trace = MapTrace((0,4,0,4), 20)
        trace.record(0,(1,2))
        trace.record(3,(3,2))
        im = trace.render(3)
        self.assertTrue(np.array_equal(im[40,40], im[38,40]))
        self.assertFalse(np.array_equal(im[40,60], im[20,60]))
