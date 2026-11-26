"""Ограниченное ожидание двух подтверждений в Pioneer SDK 2 версии 0.16.1."""
import math
from threading import Event as ThreadEvent
from time import monotonic

from pioneer_sdk2 import Event, Pioneer


class Confirmation(ThreadEvent):
    def begin(self, seconds):
        self.clear()
        self.expired = False
        self.deadline = monotonic() + seconds

    def wait(self, timeout=None):
        remaining = self.deadline - monotonic()
        if timeout is not None:
            remaining = min(remaining, timeout)
        if remaining > 0 and super().wait(remaining) and monotonic() < self.deadline:
            return True
        self.expired = True
        # SDK игнорирует возвращённое False, но перехватывает исключение.
        raise TimeoutError("Нет свежего подтверждения команды в отведённое время.")


class ConfirmedPioneer(Pioneer):
    def _confirmed(self, event, name, command):
        flag = self.callback_events[event.value]
        flag.begin(self.confirmation_timeout)
        result = command()
        if flag.expired:
            raise TimeoutError(f"Нет подтверждения {name} за {self.confirmation_timeout:g} с.")
        return result

    def takeoff(self):
        return self._confirmed(Event.TAKEOFF_COMPLETE, "взлёта", super().takeoff)

    def land(self):
        return self._confirmed(Event.COPTER_LANDED, "посадки", super().land)


def connect(timeout):
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Нужен конечный положительный тайм-аут подтверждения.")
    drone = ConfirmedPioneer()
    drone.confirmation_timeout = timeout
    # Обработчик SDK доставляет события через эту таблицу. Обе замены до ARM.
    for event in (Event.TAKEOFF_COMPLETE, Event.COPTER_LANDED):
        drone.callback_events[event.value] = Confirmation()
    return drone
