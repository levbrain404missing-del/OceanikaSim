"""Критерии № 3–7 на измерениях модели с движущейся камерой."""
import unittest


class MovingCameraTests(unittest.TestCase):
    def run_course(self, number, **options):
        from visual_mujoco import simulate
        result = simulate(number, **options)
        self.assertEqual(result["landings"], 1, result)
        self.assertTrue(result["floor_contact_after_landing"], result)
        self.assertGreater(result["physics_steps"], 100)
        return result

    def test_landing_footprint_in_random_zone_or_requested_marker(self):
        for number, marker in ((3,12),(7,7),(7,15)):
            for seed in (2,7):
                with self.subTest(number=number, seed=seed, marker=marker):
                    r = self.run_course(number, seed=seed, target_id=marker)
                    self.assertTrue(r["passed"], r)
                    self.assertGreaterEqual(r["edge_clearance_m"], 0)
                    self.assertGreaterEqual(r["distinct_camera_poses"], 2)

    def test_full_lap_and_motion_beyond_start_until_external_stop(self):
        r = self.run_course(4,stop_after=19)
        self.assertTrue(r["passed"],r)
        self.assertEqual(r["moves"],19)
        self.assertEqual(r["laps"],1)
        self.assertGreater(r["path_length_m"],16.5)
        self.assertLess(r["max_line_error_m"],.1)
        self.assertGreater(r["in_motion_line_frames"],19)
        self.assertEqual(r["in_motion_line_misses"],0)
        self.assertFalse(self.run_course(4,stop_after=2)["passed"])

    def test_two_marker_turns_and_final_touchdown(self):
        r = self.run_course(5)
        self.assertTrue(r["passed"],r)
        self.assertEqual(r["actions"],["left","right","finish"])
        self.assertLess(r["finish_error_m"],.15)

    def test_crosses_requested_3d_gate_with_clearance(self):
        for seed,color in ((1,"red"),(1,"blue"),(5,"red"),(5,"blue")):
            with self.subTest(seed=seed,color=color):
                r = self.run_course(6,seed=seed,color=color)
                self.assertTrue(r["passed"],r)
                self.assertGreater(r['moves'],0)
                self.assertEqual(r["crossed"],color)
                self.assertEqual(r["frame_contacts"],0)
                self.assertGreater(r["edge_clearance_m"],0)
                self.assertGreater(r["exit_clearance_m"],0)
                self.assertGreaterEqual(r["distinct_camera_poses"],3)

    def test_camera_dropout_at_second_read(self):
        for number in range(3,8):
            with self.subTest(number=number):
                r = self.run_course(number,drop_at=2)
                self.assertFalse(r["passed"],r)
                self.assertEqual(r["reason"],"camera unavailable")
                self.assertEqual(r["frames"],2)

    def test_unavailable_mark_and_wrong_color(self):
        self.assertFalse(self.run_course(7,target_id=49)["passed"])
        self.assertFalse(self.run_course(6,color="green")["passed"])

    def test_no_control_decision_uses_discrete_camera_crop(self):
        from unittest.mock import patch
        with patch("cv2.getRectSubPix",side_effect=AssertionError("2D crop")):
            self.assertTrue(self.run_course(3,seed=2)["passed"])

    def test_lost_line_during_a_move_stops_and_records_failed_run(self):
        import numpy as np
        from unittest.mock import patch
        from visual_mujoco import MovingField
        original = MovingField.capture_frame
        def cover_camera(field):
            frame = original(field)
            return np.full_like(frame,255) if field.moves else frame
        with patch.object(MovingField,"capture_frame",cover_camera):
            r = self.run_course(4)
        self.assertFalse(r["passed"])
        self.assertEqual(r["reason"],"line lost in motion")
        self.assertEqual(r["in_motion_line_misses"],1)
        self.assertEqual(r["moves"],1)

    def test_judge_rejects_wrong_gate_despite_wrong_positive_detections(self):
        from unittest.mock import patch
        import frames
        actual = frames.gate_center
        with patch.object(frames,"gate_center",side_effect=lambda image,color,area: actual(image,"blue",area)):
            r = self.run_course(6,color="red")
        self.assertFalse(r["passed"])
        self.assertEqual(r["crossed"],"blue")
        self.assertEqual(r["frame_contacts"],0)

    def test_landing_finishes_on_ground_with_low_speed(self):
        r = self.run_course(3,seed=2)
        self.assertGreaterEqual(r["settled_contact_s"],.14)
        self.assertLess(r["final_speed_m_s"],.02)

    def test_missing_id_falsely_reported_by_detector_fails_independent_judge(self):
        from unittest.mock import patch
        with patch("frames.choose_marker",return_value=(50,50)):
            r = self.run_course(7,target_id=49)
        self.assertFalse(r["passed"])
        self.assertFalse(r["feet_inside"])

    def test_frame_contact_prevents_gate_success(self):
        from unittest.mock import patch
        import xml.etree.ElementTree as ET
        serialize = ET.tostring
        def oversized_body(scene, **options):
            # Физика рассчитывает коллизии увеличенного корпуса с момента компиляции сцены.
            scene.find(".//geom[@name='craft']").set("size", ".38 .08 .05")
            return serialize(scene,**options)
        with patch("visual_mujoco.ET.tostring",side_effect=oversized_body):
            r = self.run_course(6)
        self.assertFalse(r["passed"])
        self.assertGreater(r["frame_contacts"],0)

    def test_cli_all_visual_tasks_emit_json_results(self):
        import json
        from pathlib import Path
        import subprocess
        import sys
        script = Path(__file__).with_name("visual_mujoco.py")
        for number in range(3,8):
            with self.subTest(number=number):
                run = subprocess.run([sys.executable,"-B","-X","utf8",str(script),str(number),"--seed","7"],
                                     capture_output=True,text=True,encoding="utf-8",timeout=30)
                self.assertEqual(run.returncode,0,run.stderr)
                self.assertIs(json.loads(run.stdout)["passed"],True)
