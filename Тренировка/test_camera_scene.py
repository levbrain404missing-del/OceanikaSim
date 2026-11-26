"""Проверка заданий на изображениях из виртуальной камеры MuJoCo."""
import unittest

import frames


class RenderedCameraBench(unittest.TestCase):
    def test_bright_zone_moves_across_camera(self):
        from camera_scene import render_pair
        first, second = render_pair("zone")
        cx, cy = frames.bright_zone(first, 145, 200)
        dx, dy = frames.bright_zone(second, 145, 200)
        self.assertGreater(cx-dx, 12)
        self.assertAlmostEqual(cy, dy, delta=2)

    def test_dark_line_moves_across_camera(self):
        from camera_scene import render_pair
        first, second = render_pair("line")
        x = frames.line_x(first, 120, 80, 4)
        y = frames.line_x(second, 120, 80, 4)
        self.assertGreater(x-y, 12)

    def test_two_distinct_openings_on_same_frame(self):
        from camera_scene import render_pair
        for frame in render_pair("gate"):
            red = frames.gate_center(frame, "red", 200)
            blue = frames.gate_center(frame, "blue", 200)
            self.assertLess(red[0], blue[0])
            self.assertIsNone(frames.gate_center(frame, "green", 200))

    def test_both_marker_ids_on_both_views(self):
        from camera_scene import render_pair
        for frame in render_pair("markers"):
            found = frames.markers(frame, "DICT_4X4_50")
            self.assertEqual(sorted(idx for idx,_ in found), [8,15])
            self.assertLess(frames.choose_marker(found,8)[0], frames.choose_marker(found,15)[0])
            self.assertEqual(frames.marker_action(8,15), "left")
            self.assertEqual(frames.marker_action(15,15), "finish")
