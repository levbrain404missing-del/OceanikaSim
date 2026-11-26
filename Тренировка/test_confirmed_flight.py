"""Настоящие методы SDK 0.16.1 с локальным транспортом вместо борта."""
import contextlib
import io
import time
import unittest
from unittest.mock import MagicMock, patch

from pioneer_sdk2 import Event
from pioneer_sdk2.piosdk2 import FlyState, proto

from shortest_route import airborne


class ConfirmationTests(unittest.TestCase):
    def setUp(self):
        from confirmed_flight import connect
        self.link = MagicMock()
        self.writes = []
        self.events = {proto.BoardState.ARM: Event.ENGINES_STARTED,
                       proto.BoardState.TAKEOFF: Event.TAKEOFF_COMPLETE,
                       proto.BoardState.LAND: Event.COPTER_LANDED}
        self.enterContext(patch.object(proto, "NetworkStream"))
        self.enterContext(patch.object(proto, "FlightAPI", return_value=self.link))
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))
        self.link["board"]["state"].write.side_effect = self.write
        self.drone = connect(0.08)
        self.addCleanup(self.drone.close_connection)

    def write(self, command):
        self.writes.append(command)
        if command in self.events:
            self.link.on_event(self.events[command].value)

    def test_confirmed_order_and_native_guards(self):
        self.assertTrue(self.drone.wait_callback and self.drone.safety_command)
        with self.assertRaises(RuntimeError):
            self.drone.takeoff()
        self.assertEqual(self.writes, [])
        with airborne(self.drone):
            self.assertEqual(self.drone.get_fly_state(), FlyState.IN_SKY)
        self.assertEqual(self.drone.get_fly_state(), FlyState.ON_LAND)
        self.assertEqual(self.writes,[proto.BoardState.ARM,proto.BoardState.TAKEOFF,proto.BoardState.LAND])

    def test_lost_takeoff_aborts_and_requests_landing_once(self):
        del self.events[proto.BoardState.TAKEOFF]
        start = time.monotonic()
        with self.assertRaisesRegex(TimeoutError,"взлёта"):
            with airborne(self.drone):
                self.fail("Маршрут продолжился без подтверждения")
        self.assertLess(time.monotonic()-start, 1)
        self.assertEqual(self.writes,[proto.BoardState.ARM,proto.BoardState.TAKEOFF,proto.BoardState.LAND])

    def test_stale_landing_flag_does_not_count(self):
        del self.events[proto.BoardState.LAND]
        with self.assertRaisesRegex(TimeoutError,"посадки"):
            with airborne(self.drone):
                self.drone.callback_events[Event.COPTER_LANDED.value].set()
        self.assertEqual(self.writes.count(proto.BoardState.LAND),1)
        self.assertEqual(self.drone.get_fly_state(),FlyState.IN_SKY)

    def test_async_confirmation_succeeds(self):
        import threading
        self.assertTrue(self.drone.arm())
        def later(command):
            timer = threading.Timer(.015,self.link.on_event,args=(self.events[command].value,))
            self.addCleanup(timer.join)
            timer.start()
        self.link["board"]["state"].write.side_effect = later
        self.assertTrue(self.drone.takeoff())
        self.assertTrue(self.drone.land())

    def test_invalid_timeout_precedes_connection(self):
        from confirmed_flight import connect
        with patch.object(proto,"NetworkStream") as stream:
            for value in (0,-1,float("nan"),float("inf")):
                with self.assertRaises(ValueError):
                    connect(value)
            stream.assert_not_called()

    def test_flight_entry_points_select_timeout_and_close_after_failure(self):
        import sys
        import geometry_missions
        import shortest_route
        for module, function, args in [
            (geometry_missions, "run", ["circle", "0.6", "--height", "1.5", "--fly"]),
            (shortest_route, "fly", ["--fly-at","1.5","--point-timeout","3"]),
        ]:
            with self.subTest(module=module.__name__), patch.object(sys,"argv",[
                module.__name__,*args,"--confirmation-timeout","7.5"
            ]), patch("confirmed_flight.connect") as open_drone, patch.object(
                module,function,side_effect=TimeoutError
            ), patch("builtins.input",side_effect=["3","1 0","1 1","0 1"]):
                with self.assertRaises(TimeoutError):
                    module.main()
                open_drone.assert_called_once_with(7.5)
                open_drone.return_value.close_connection.assert_called_once()

    def test_mujoco_takeoff_and_landing_confirmed_via_sdk(self):
        from physics_bench import MotionModel
        model = MotionModel((0,0,0))
        def simulate(command):
            self.writes.append(command)
            if command == proto.BoardState.TAKEOFF:
                model.set_target((0,0,1))
                model.advance(2)
                self.assertGreater(model.position[2], .98)
            elif command == proto.BoardState.LAND:
                model.set_target((0,0,0))
                model.advance(2)
                self.assertLess(model.position[2], .02)
            self.link.on_event(self.events[command].value)
        self.link["board"]["state"].write.side_effect = simulate
        with airborne(self.drone):
            self.assertGreater(model.position[2], .98)
        self.assertEqual(self.drone.get_fly_state(),FlyState.ON_LAND)
        self.assertEqual(self.writes.count(proto.BoardState.LAND),1)

    def test_late_confirmation_after_transport_returns_is_not_success(self):
        self.assertTrue(self.drone.arm())
        def delayed_write(command):
            time.sleep(.10)
            self.link.on_event(self.events[command].value)
        self.link["board"]["state"].write.side_effect = delayed_write
        with self.assertRaisesRegex(TimeoutError,"взлёта"):
            self.drone.takeoff()
