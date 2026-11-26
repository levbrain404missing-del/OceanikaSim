import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np


FOLDER = Path(__file__).resolve().parent


def cli(script, *args):
    proc = subprocess.run([sys.executable, "-B", str(FOLDER / script), *map(str, args)],
                          capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10)
    if proc.returncode:
        raise AssertionError(proc.stderr)
    return proc.stdout


class ReplayTests(unittest.TestCase):
    def test_long_map_draws_each_segment_once(self):
        from unittest.mock import patch
        from monitor import MapTrace
        trace = MapTrace((0,8,0,4), 30)
        with patch("monitor.cv2.line", wraps=cv2.line) as line:
            for i in range(301):
                trace.record(i / 10, (i / 50, 2))
                trace.render(i / 10)
        self.assertEqual(line.call_count, 300)

    def test_telemetry_terminal_two_requests_then_disconnect(self):
        from contextlib import redirect_stdout
        from io import StringIO
        from unittest.mock import Mock, patch
        from monitor import main
        drone = Mock(spec=["get_local_position_lps","get_altitude","get_battery_status",
                           "get_orientation","close_connection"])
        drone.get_local_position_lps.side_effect = [(0,0,1), (4,1,2)]
        drone.get_altitude.return_value = 2
        drone.get_battery_status.return_value = (11.1,25)
        drone.get_orientation.return_value = (0,0,90)
        buffer = StringIO()
        with patch.object(sys,"argv",["monitor.py","telemetry"]), patch("pioneer_sdk2.Pioneer",return_value=drone), \
             patch("builtins.input",side_effect=["","","q"]), redirect_stdout(buffer):
            main()
        rows = [json.loads(row) for row in buffer.getvalue().splitlines()]
        self.assertEqual([r["xyz_m"] for r in rows], [[0,0,1],[4,1,2]])
        drone.close_connection.assert_called_once()

    def test_video_lost_line_and_reacquisition(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "line.avi"
            out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (120,80))
            self.assertTrue(out.isOpened())
            try:
                for x in (20, None, 80):
                    frame = np.full((80,120,3), 255, np.uint8)
                    if x is not None:
                        frame[:,x-4:x+5] = 0
                    out.write(frame)
            finally:
                out.release()
            rows = [json.loads(row) for row in cli("frames.py", "--video", path, "line", 40, 80, 5).splitlines()]
        self.assertEqual([r["frame"] for r in rows], [0,1,2])
        self.assertEqual([r["result"] for r in rows], [20.0,None,80.0])

    def test_map_window_receives_two_distinct_positions_without_hardware(self):
        from unittest.mock import Mock, patch
        from monitor import show_map
        drone = Mock(spec=["get_local_position_lps"])
        drone.get_local_position_lps.side_effect = [(1,1,2),(2,2,2)]
        with patch("monitor.cv2.namedWindow"), patch("monitor.cv2.imshow") as draw, \
             patch("monitor.cv2.waitKey", side_effect=[-1,27]), \
             patch("monitor.cv2.getWindowProperty", return_value=1), patch("monitor.cv2.destroyAllWindows") as close:
            show_map(drone, (0,4,0,4), 50)
        self.assertEqual(draw.call_count, 2)
        for call, (x,y) in zip(draw.call_args_list, [(50,150),(100,100)]):
            self.assertGreater(call.args[1][y,x,2], call.args[1][y,x,0])
        close.assert_called_once()

    def test_spherical_hold_resets_on_dropout_or_outside_position(self):
        from geometry import longest_hold
        target = (0,0,1)
        positions = [(t/2,(0.2,0.2,1.2)) for t in range(7)]
        self.assertEqual(longest_hold(positions,target,0.5), 3)
        for wrong in ((0.201,0,1), None, (0,0,float("nan"))):
            broken = positions.copy()
            broken[3] = (1.5, wrong)
            self.assertEqual(longest_hold(broken,target,0.5), 1)
        self.assertEqual(longest_hold([positions[0],positions[-1]],target,0.5), 0)
        for samples in (positions[::-1], [positions[0],positions[0]]):
            with self.assertRaises(ValueError):
                longest_hold(samples,target,0.5)

    def test_hold_cli_reads_local_xyz_and_seconds(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hold.csv"
            with path.open("w", newline="",encoding="utf-8") as stream:
                rows = csv.writer(stream)
                rows.writerow(["t_s","x_m","y_m","z_m"])
                for i in range(9):
                    rows.writerow([i/2, 0, 1, 2])
            result = json.loads(cli("geometry.py","--start",0,0,2,"hold",1,90,0,path,"--max-gap",0.5))
        self.assertTrue(result["held_3s"])
        self.assertEqual(result["observed_s"], 4)
