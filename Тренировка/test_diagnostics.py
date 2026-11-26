"""Регрессии по наблюдаемым отказам домашних стендов."""
import contextlib
import io
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
from pioneer_sdk2 import Pioneer

import frames
import monitor
import shortest_route


class CameraBench(unittest.TestCase):
    def test_overexposure_and_cropped_zone(self):
        for image in (np.full((200,300,3), 255, np.uint8), np.zeros((200,300,3), np.uint8)):
            cv2.circle(image, (5,70), 30, (255,255,255), -1)
            self.assertIsNone(frames.bright_zone(image, 160, 300))

    def test_blurred_compressed_marker(self):
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        image = np.full((240,320), 185, np.uint8)
        image[70:170,100:200] = cv2.aruco.generateImageMarker(dictionary, 21, 100)
        image = cv2.GaussianBlur(image, (5,5), 1.2)
        ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 45])
        self.assertTrue(ok)
        found = frames.markers(cv2.imdecode(jpeg, cv2.IMREAD_COLOR), "DICT_4X4_50")
        np.testing.assert_allclose(frames.choose_marker(found, 21), (149.5,119.5), atol=2)

    def test_single_hue_cannot_match_two_colors(self):
        img = np.zeros((100,150,3), np.uint8)
        cv2.rectangle(img, (20,20), (110,80), (35,250,240), 6)
        bgr = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        matches = [color for color in frames.HUES if frames.gate_center(bgr, color, 200)]
        self.assertEqual(len(matches), 1)


class WaypointBench(unittest.TestCase):
    def setUp(self):
        self.drone = Mock()
        self.time = 0.0
        self.enterContext(patch.object(shortest_route, "monotonic", lambda: self.time))
        self.enterContext(patch.object(shortest_route, "sleep", self.advance))
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))
        self.drone.point_reached.return_value = True
        self.drone.get_local_position_lps.return_value = (1,0,1)
        self.drone.go_to_local_point.return_value = None

    def advance(self, seconds):
        self.time += seconds
        self.assertLess(self.time, 3)

    def visit(self):
        shortest_route.visit_points(self.drone, [(0,0),(1,0)], 1, 0.5)

    def test_late_position_must_not_confirm_point(self):
        def late():
            self.advance(0.6)
            return (1,0,1)
        self.drone.get_local_position_lps.side_effect = late
        with self.assertRaises(TimeoutError):
            self.visit()

    def test_partial_sample_then_recovery(self):
        self.drone.get_local_position_lps.side_effect = [(1,), (None,0,1), (1,0,1)]
        self.visit()
        self.assertEqual(self.drone.get_local_position_lps.call_count, 3)
        self.drone.go_to_local_point.assert_called_once()

    def test_explicit_rejection_ends_route_and_requests_landing(self):
        self.drone.go_to_local_point.return_value = False
        with self.assertRaisesRegex(RuntimeError, "отклонил"):
            shortest_route.fly(self.drone, [(1,2),(2,1),(2,2)], 1, 0.5)
        self.drone.go_to_local_point.assert_called_once()
        self.drone.land.assert_called_once()


class SDKTelemetryBench(unittest.TestCase):
    def test_real_sdk_getters_convert_raw_fields_before_display(self):
        # Реальные методы установленной библиотеки, подставлены только поля транспорта.
        sdk = Pioneer.__new__(Pioneer)
        self.addCleanup(setattr, sdk, "messenger", None)
        sdk._logger = False
        values = {"lps": {"x": 1250, "y": -250, "z": 2100},
                  "board": {"altitude": 1930, "roll": 150, "pitch": -200, "yaw": 9000},
                  "battery": {"voltage": 11350, "temperature": 27}}
        sdk.messenger = {group: {k: Mock(read=Mock(return_value=v)) for k,v in fields.items()}
                         for group,fields in values.items()}
        self.assertEqual(monitor.telemetry(sdk),
                         {"xyz_m": (1.25,-0.25,2.1), "altitude_m": 1.93,
                          "roll_pitch_yaw": (1.5,-2,90), "battery_V": 11.35})
        sdk.messenger["lps"]["y"].read.side_effect = OSError("fixture disconnect")
        result = monitor.telemetry(sdk)
        self.assertIsNone(result["xyz_m"])
        self.assertEqual(result["battery_V"], 11.35)
