import io
import math
import random
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from functools import cache
from pathlib import Path
from unittest.mock import Mock, patch

import shortest_route as mission


def minimum_length(start, points):
    # Независимый эталон: минимум для каждого подмножества оставшихся точек.
    @cache
    def finish(current, remaining):
        if not remaining:
            return math.dist(current, start)
        return min(
            math.dist(current, points[i]) + finish(points[i], remaining - {i})
            for i in remaining
        )

    return finish(start, frozenset(range(len(points))))


class RouteTests(unittest.TestCase):
    def test_300_routes_against_independent_minimum(self):
        rng = random.Random(2511)
        for size in (3, 4, 5):
            for case in range(100):
                start, *points = [
                    (rng.randrange(-10, 11) / 2, rng.randrange(-10, 11) / 2)
                    for _ in range(size + 1)
                ]
                with self.subTest(size=size, case=case):
                    route, length = mission.shortest_route(start, points)
                    self.assertEqual((route[0], route[-1]), (start, start))
                    self.assertCountEqual(route[1:-1], points)
                    self.assertAlmostEqual(length, minimum_length(start, points))
                    self.assertAlmostEqual(length, sum(map(math.dist, route, route[1:])))

    def test_coincident_points(self):
        route, length = mission.shortest_route((1, 2), [(1, 2)] * 3)
        self.assertEqual(route, ((1, 2),) * 5)
        self.assertEqual(length, 0)

    def test_invalid_inputs(self):
        for points in ([(0, 0)] * 2, [(0, 0)] * 6, [(0, 0)] * 2 + [(1,)],
                       [(math.nan, 0)] * 3, [(0, math.inf)] * 3):
            with self.subTest(points=points), self.assertRaises(ValueError):
                mission.shortest_route((0, 0), points)

    def test_normal_cli_without_hardware(self):
        result = subprocess.run(
            [sys.executable, '-X', 'utf8', mission.__file__],
            input='0 0\n3\n1 1\n1 0\n0 1\n', encoding='utf-8',
            capture_output=True, timeout=5, check=True, cwd=Path(mission.__file__).parent,
        )
        self.assertIn('4.0000 м', result.stdout)


class FlightTests(unittest.TestCase):
    points = ((1, 1), (0, 1), (1, 0))

    def setUp(self):
        self.drone = Mock()
        self.position = (0, 0, 0)
        self.drone.get_local_position_lps.side_effect = lambda: self.position
        self.drone.go_to_local_point.side_effect = self.arrive
        self.drone.point_reached.return_value = True
        self.clock = 0
        self.enterContext(patch.object(mission, 'sleep', self.advance))
        self.enterContext(patch.object(mission, 'monotonic', lambda: self.clock))
        self.enterContext(redirect_stdout(io.StringIO()))

    def advance(self, seconds):
        self.clock += seconds
        self.assertLess(self.clock, 5, 'Стенд обнаружил бесконечное ожидание')

    def arrive(self, x, y, z):
        self.position = (x, y, z)

    def fly(self):
        mission.fly(self.drone, self.points, height=1, timeout=0.5)

    def test_success_sequence(self):
        self.fly()
        calls = self.drone.go_to_local_point.call_args_list
        self.assertCountEqual([(c.kwargs['x'], c.kwargs['y']) for c in calls[:-1]], self.points)
        self.assertEqual(calls[-1].kwargs, dict(x=0, y=0, z=1))
        self.assertTrue(all(c.kwargs['z'] == 1 for c in calls))
        actions = [c[0] for c in self.drone.method_calls if c[0] in
                   ('arm', 'takeoff', 'go_to_local_point', 'land')]
        self.assertEqual(actions, ['arm', 'takeoff'] + ['go_to_local_point'] * 4 + ['land'])

    def test_missing_start(self):
        self.position = None
        with self.assertRaises(RuntimeError):
            self.fly()
        self.drone.arm.assert_not_called()

    def test_arm_refused(self):
        self.drone.arm.return_value = False
        with self.assertRaises(RuntimeError):
            self.fly()
        self.drone.takeoff.assert_not_called()

    def test_takeoff_refused(self):
        self.drone.takeoff.return_value = False
        with self.assertRaises(RuntimeError):
            self.fly()
        self.drone.go_to_local_point.assert_not_called()
        self.drone.land.assert_called_once()

    def test_missing_event(self):
        self.drone.point_reached.return_value = False
        with self.assertRaises(TimeoutError):
            self.fly()
        self.drone.go_to_local_point.assert_called_once()
        self.drone.land.assert_called_once()

    def test_stale_event_wrong_position(self):
        self.drone.go_to_local_point.side_effect = None
        with self.assertRaises(TimeoutError):
            self.fly()
        self.drone.go_to_local_point.assert_called_once()
        self.drone.land.assert_called_once()

    def test_missing_telemetry_after_takeoff(self):
        self.drone.get_local_position_lps.side_effect = lambda: (
            None if self.drone.takeoff.called else (0, 0, 0)
        )
        with self.assertRaises(TimeoutError):
            self.fly()
        self.drone.go_to_local_point.assert_called_once()
        self.drone.land.assert_called_once()

    def test_movement_error(self):
        self.drone.go_to_local_point.side_effect = ConnectionError
        with self.assertRaises(ConnectionError):
            self.fly()
        self.drone.go_to_local_point.assert_called_once()
        self.drone.land.assert_called_once()

    def test_unconfirmed_landing(self):
        self.drone.land.return_value = False
        with self.assertRaisesRegex(RuntimeError, 'Посадка не подтверждена'):
            self.fly()


if __name__ == '__main__':
    unittest.main()
