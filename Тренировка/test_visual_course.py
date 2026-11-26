import unittest


class VisualCourseTests(unittest.TestCase):
    def test_both_landing_tasks_use_scene_and_keep_feet_inside(self):
        from visual_course import exercise
        placements = set()
        for number in (3,7):
            for seed in range(5):
                with self.subTest(number=number,seed=seed):
                    result = exercise(number,seed=seed,target_id=12)
                    self.assertTrue(result["passed"])
                    self.assertTrue(result["feet_inside"])
                    self.assertEqual(result["landings"],1)
                    placements.add(tuple(result["cell"]))
        self.assertGreater(len(placements),2)

    def test_unknown_marker_and_blanked_camera_end_without_claiming_target(self):
        from visual_course import exercise
        for number in (3,4,5,6,7):
            with self.subTest(number=number):
                result = exercise(number, blank=True)
                self.assertFalse(result["passed"])
                self.assertEqual(result["landings"],1)
        self.assertFalse(exercise(7,target_id=49)["passed"])

    def test_full_lap_continues_until_external_stop(self):
        from visual_course import exercise
        for steps in (17,23):
            result = exercise(4,stop_after=steps)
            self.assertTrue(result["passed"])
            self.assertGreaterEqual(result["laps"],1)
            self.assertEqual(result["moves"],steps)
            self.assertEqual(result["landings"],1)
        early = exercise(4,stop_after=2)
        self.assertFalse(early["passed"])
        self.assertEqual(early["moves"],2)

    def test_both_turns_and_once_per_marker_encounter(self):
        from visual_course import exercise
        result = exercise(5)
        self.assertTrue(result["passed"])
        self.assertEqual(result["actions"],["left","right","finish"])
        self.assertEqual(result["landings"],1)

    def test_requested_gate_and_full_clearance(self):
        from visual_course import exercise
        for color in ("red","blue"):
            for seed in range(3):
                result = exercise(6,seed=seed,color=color)
                self.assertTrue(result["passed"])
                self.assertEqual(result["crossed"],color)
                self.assertTrue(result["fully_clear"])
                self.assertEqual(result["landings"],1)
        self.assertFalse(exercise(6,color="green")["passed"])

    def test_camera_disconnect_stops_before_any_further_motion(self):
        from visual_course import exercise
        # Второй кадр теряется после начала распознавания.
        for number in (3,4,5,6,7):
            with self.subTest(number=number):
                result = exercise(number,drop_at=2)
                self.assertFalse(result["passed"])
                self.assertEqual(result["frames"],2)
                self.assertEqual(result["reason"],"camera unavailable")
                self.assertEqual(result["landings"],1)
        self.assertEqual(exercise(5,drop_at=2)["moves"],0)
