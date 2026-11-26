"""№ 10–11. Чтение телеметрии по Enter или показ координат на карте."""
import argparse
import json
import math
from time import perf_counter

import cv2
import numpy as np


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def vector(value, count):
    if value is not None and len(value) == count and all(map(finite, value)):
        return tuple(value)
    return None


def telemetry(drone):
    position = vector(drone.get_local_position_lps(), 3)
    altitude = drone.get_altitude()
    battery = drone.get_battery_status()
    voltage = battery[0] if battery else None
    return {"xyz_m": position, "altitude_m": altitude if finite(altitude) else None,
            "battery_V": voltage if finite(voltage) else None,
            "roll_pitch_yaw": vector(drone.get_orientation(), 3)}


class MapTrace:
    """Метрический масштаб общий для X и Y. Время измерений в секундах."""
    def __init__(self, bounds, scale):
        if not vector(bounds, 4) or not finite(scale) or scale <= 0:
            raise ValueError("Проверьте границы карты и масштаб.")
        self.xmin, xmax, ymin, self.ymax = bounds
        if xmax <= self.xmin or self.ymax <= ymin:
            raise ValueError("Границы карты перепутаны.")
        self.scale = scale
        self.width = 1 + round((xmax-self.xmin)*scale)
        self.height = 1 + round((self.ymax-ymin)*scale)
        if min(self.width, self.height) < 2 or self.width*self.height > 16000000:
            raise ValueError("Неподходящий размер изображения карты.")
        self.trail = np.full((self.height, self.width, 3), 245, np.uint8)
        self.last_stamp = None
        self.last_pixel = None

    def pixel(self, xy):
        x, y = xy
        return round((x-self.xmin)*self.scale), round((self.ymax-y)*self.scale)

    def record(self, stamp, xy):
        if not finite(stamp) or not vector(xy, 2) or (self.last_stamp is not None and stamp <= self.last_stamp):
            raise ValueError("Нужны конечные координаты и возрастающее время.")
        point = self.pixel(xy)
        if self.last_stamp is not None and stamp - self.last_stamp <= 2:
            cv2.line(self.trail, self.last_pixel, point, (155,80,30), 1)
        self.last_pixel = point
        self.last_stamp = stamp

    def fresh(self, now):
        return self.last_stamp is not None and 0 <= now - self.last_stamp <= 2

    def render(self, now):
        canvas = self.trail.copy()
        if self.fresh(now):
            cv2.circle(canvas, self.last_pixel, 3, (35,35,215), -1)
        return canvas


def show_map(drone, bounds, scale):
    trace = MapTrace(bounds, scale)
    title = "Pioneer XY"
    cv2.namedWindow(title)
    try:
        while True:
            read_started = perf_counter()
            pos = vector(drone.get_local_position_lps(), 3)
            if pos:
                trace.record(read_started, pos[:2])
            now = perf_counter()
            canvas = trace.render(now)
            state = "XY" if trace.fresh(now) else "NO RECENT XY"
            cv2.putText(canvas, f"{state}   {scale:g} px/m   X >   Y up", (5,17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30,30,30), 1)
            cv2.imshow(title, canvas)
            if cv2.waitKey(50) & 0xff in (27, ord("q")) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    tasks = parser.add_subparsers(dest="task", required=True)
    tasks.add_parser("telemetry")
    xy = tasks.add_parser("map")
    xy.add_argument("bounds", type=float, nargs=4, metavar="BOUND", help="xmin xmax ymin ymax, метры")
    xy.add_argument("--scale", type=float, default=80, help="Пикселей на метр")
    args = parser.parse_args()
    from pioneer_sdk2 import Pioneer
    drone = Pioneer()
    try:
        if args.task == "map":
            show_map(drone, args.bounds, args.scale)
        else:
            while input("Enter — запрос, q — выход: ").strip().lower() != "q":
                print(json.dumps(telemetry(drone), ensure_ascii=False), flush=True)
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        drone.close_connection()


if __name__ == "__main__":
    main()
