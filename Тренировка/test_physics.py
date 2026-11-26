"""Критерии сценариев в MuJoCo. Запуск без сети и оборудования."""
import contextlib
import io
import math
import unittest


class PhysicsBench(unittest.TestCase):
    def test_actual_motion_has_lag_and_settles(self):
        from physics_bench import MotionModel
        model = MotionModel((1,-1,0))
        model.set_target((2,-1,0))
        model.advance(0.05)
        self.assertGreater(model.position[0], 1)
        self.assertLess(model.position[0], 1.5)
        model.advance(2)
        self.assertLess(math.dist(model.position, (2,-1,0)), 0.025)

    def test_route_returns_after_three_points_and_lands(self):
        from physics_bench import run_route
        with contextlib.redirect_stdout(io.StringIO()):
            result = run_route(drop_event=False)
        self.assertFalse(result["timeout"])
        self.assertEqual(result["waypoints_sent"], 4)
        self.assertEqual(result["confirmed"], 4)
        self.assertLess(result["return_xy_error_m"], 0.2)
        self.assertLess(abs(result["final_xyz_m"][2]), 0.025)
        self.assertEqual(result["landing_requests"], 1)

    def test_event_loss_stops_route_and_lands(self):
        from physics_bench import run_route
        with contextlib.redirect_stdout(io.StringIO()):
            result = run_route(drop_event=True)
        self.assertTrue(result["timeout"])
        self.assertEqual(result["waypoints_sent"], 1)
        self.assertEqual(result["confirmed"], 0)
        self.assertEqual(result["landing_requests"], 1)
        self.assertLess(abs(result["final_xyz_m"][2]), 0.025)

    def test_two_figures_follow_geometry_in_physics_engine(self):
        from physics_bench import run_figure
        for name in ("circle","eight"):
            with self.subTest(name=name):
                result = run_figure(name)
                self.assertLess(result["max_radial_error_m"], 0.2)
                self.assertLess(result["return_error_m"], 0.2)
                self.assertGreater(result["samples"], 100)
                self.assertGreater(result["max_radial_error_m"], 0)

    def test_spherical_hold_with_offset_origin(self):
        from physics_bench import run_hold
        result = run_hold()
        self.assertGreaterEqual(result["observed_hold_s"], 3)
        self.assertLess(result["return_error_m"], 0.2)

    def test_map_tracks_motion_samples_from_engine(self):
        from monitor import MapTrace
        from physics_bench import MotionModel
        model = MotionModel((0,0,1))
        model.set_target((1,-0.5,1))
        model.advance(2)
        trace = MapTrace((-2,2,-2,2), 50)
        for stamp, pos in model.samples:
            trace.record(stamp, pos[:2])
        image = trace.render(stamp)
        # Независимая привязка цели (1, -0.5) к пикселю (150, 125).
        self.assertEqual(image[125,150].tolist(), [35,35,215])
        self.assertFalse(trace.fresh(stamp + 2.01))

    def test_rendered_trace_follows_clocked_physics_sample_under_two_seconds(self):
        from time import perf_counter
        from physics_bench import MotionModel
        from monitor import MapTrace
        model = MotionModel((0,0,1))
        trace = MapTrace((-2,2,-2,2),100)
        model.set_target((1,1,1))
        for _ in range(25):
            begin = perf_counter()
            model.advance(.05)
            x,y,_ = model.position
            trace.record(begin,(x,y))
            image = trace.render(perf_counter())
            self.assertEqual(tuple(image[round((2-y)*100),round((2+x)*100)]),(35,35,215))
            self.assertLess(perf_counter()-begin,2)
