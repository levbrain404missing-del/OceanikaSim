"""Учебные сценарии № 1, 2 и 9. По умолчанию используется MuJoCo."""
import argparse
import json
import math
from time import monotonic, sleep

import geometry
from shortest_route import airborne, valid_position, visit_point


def hold(drone, target, timeout):
    deadline = monotonic() + timeout
    first = last = None
    while monotonic() < deadline:
        pos = drone.get_local_position_lps()
        now = monotonic()
        if now >= deadline:
            break
        if valid_position(pos) and all(abs(a-b) <= .2 + 1e-12 for a,b in zip(pos,target)):
            if first is None or last is None or now-last > .25:
                first = now
            if now-first >= 3:
                return now-first
        else:
            first = None
        last = now
        sleep(min(.05, deadline-now))
    raise TimeoutError("Нет трёх секунд подряд внутри допуска.")


def run(drone, task, values, *, height, timeout):
    if not all(math.isfinite(v) and v > 0 for v in (height,timeout)):
        raise ValueError("Высота и тайм-аут должны быть положительными.")
    start = drone.get_local_position_lps()
    if not valid_position(start):
        raise RuntimeError("Нет локальных координат старта.")
    x, y, floor = start
    entry = (x, y, floor + height)
    if task == "circle" and len(values) == 1:
        points = geometry.circle(entry, values[0])
    elif task == "eight" and not values:
        points = geometry.figure_eight(entry)
    elif task == "spherical" and len(values) == 3:
        target = geometry.spherical_target(start, *values)
        print("XYZ (м):", *target, flush=True)
        points = [target, entry]
    else:
        raise ValueError("Для circle нужен радиус, для eight нет параметров, для spherical нужны три числа.")
    if any(not valid_position(point) or point[2] <= floor for point in points):
        raise ValueError("После пересчёта часть пути лежит на уровне старта на земле или ниже.")
    observed = None
    with airborne(drone):
        for i, point in enumerate(points):
            visit_point(drone, point, timeout, clock=monotonic, pause=sleep)
            if task == "spherical" and i == 0:
                observed = hold(drone, point, timeout + 3)
    return {"points": len(points), "observed_hold_s": observed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("circle","eight","spherical"))
    parser.add_argument("values", nargs="*", type=float, help="Радиус или дистанция, азимут, возвышение")
    parser.add_argument("--height", type=float, required=True, help="Высота старта фигуры или возврата над местом взлёта, м")
    parser.add_argument("--timeout", type=float, default=5, help="Время подтверждения точки, с")
    parser.add_argument("--fly", action="store_true", help="Использовать подключённый Pioneer SDK 2")
    parser.add_argument("--confirmation-timeout", type=float, default=15,
                        help="Лимит подтверждения взлёта и посадки в SDK, с (по умолчанию 15)")
    args = parser.parse_args()
    if args.fly:
        from confirmed_flight import connect
        drone = connect(args.confirmation_timeout)
        try:
            result = run(drone, args.task, args.values, height=args.height, timeout=args.timeout)
        finally:
            drone.close_connection()
    else:
        import sys
        from unittest.mock import patch
        from physics_bench import MotionModel, RouteDevice
        model = MotionModel((0,0,0))
        drone = RouteDevice(model, False)
        module = sys.modules[__name__]
        with patch.object(module,"monotonic",lambda: model.data.time), patch.object(module,"sleep",model.advance):
            result = run(drone, args.task, args.values, height=args.height, timeout=args.timeout)
        result.update(simulation="MuJoCo", final_xyz=model.position, landing_requests=drone.landings)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
