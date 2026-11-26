"""Расчёт замкнутого маршрута в плоскости XY, координаты в метрах."""

import argparse
from contextlib import contextmanager
from itertools import pairwise, permutations
from math import dist, fsum, isfinite
from time import monotonic, sleep


def shortest_route(start, points):
    """Вернуть маршрут со стартом и возвратом, а также его длину в метрах."""
    if not 3 <= len(points) <= 5:
        raise ValueError("Требуется от 3 до 5 контрольных точек.")
    for point in (start, *points):
        if len(point) != 2 or not all(isfinite(value) for value in point):
            raise ValueError("Для каждой точки нужны два конечных числа X Y.")

    best_length = float("inf")
    best_route = None
    for order in permutations(points):
        route = (start, *order, start)
        length = fsum(dist(a, b) for a, b in pairwise(route))
        if length < best_length:
            best_route, best_length = route, length
    if best_route is None:
        raise ValueError("Длина маршрута выходит за диапазон чисел.")
    return best_route, best_length


def read_point(prompt):
    fields = input(prompt).split()
    if len(fields) != 2:
        raise ValueError("Введите X и Y через пробел.")
    return tuple(map(float, fields))


def valid_position(position):
    return (isinstance(position, (tuple, list)) and len(position) == 3
            and all(isinstance(v, (int, float)) and isfinite(v) for v in position))


def visit_point(drone, target, timeout, *, axes=3, clock=None, pause=None):
    clock = clock or monotonic
    pause = pause or sleep
    deadline = clock() + timeout
    drone.point_reached()  # Чтение сбрасывает оставшееся событие старой точки.
    x, y, z = target
    if drone.go_to_local_point(x=x, y=y, z=z) is False:
        raise RuntimeError("Автопилот отклонил команду перехода.")
    reached = False
    while clock() < deadline:
        reached = reached or drone.point_reached()
        if reached:
            position = drone.get_local_position_lps()
            if valid_position(position) and clock() < deadline:
                if dist(position[:axes], target[:axes]) <= 0.2:
                    return
        remaining = deadline - clock()
        if remaining > 0:
            pause(min(0.05, remaining))
    raise TimeoutError(f"Точка {target} не подтверждена за {timeout:g} с.")


def visit_points(drone, route, height, timeout):
    for x, y in route[1:]:
        visit_point(drone, (x, y, height), timeout, axes=2)


def print_route(start, points):
    route, length = shortest_route(start, points)
    print("Маршрут:", " → ".join(f"({x:g}, {y:g})" for x, y in route))
    print(f"Длина: {length:.4f} м", flush=True)
    return route


@contextmanager
def airborne(drone):
    if not drone.arm():
        raise RuntimeError("Запуск двигателей не подтверждён.")
    try:
        if not drone.takeoff():
            raise RuntimeError("Взлёт не подтверждён.")
        yield
    finally:
        print("Запрашиваю штатную посадку.", flush=True)
        if not drone.land():
            raise RuntimeError("Посадка не подтверждена. Сообщите пилоту.")


def fly(drone, points, height, timeout):
    position = drone.get_local_position_lps()
    if not valid_position(position):
        raise RuntimeError("Нет локальных координат старта.")
    route = print_route(position[:2], points)
    with airborne(drone):
        visit_points(drone, route, height, timeout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fly-at", type=float, metavar="Z", help="Полёт на локальной отметке Z (м)")
    parser.add_argument("--point-timeout", type=float, metavar="SEC", help="Лимит ожидания точки (с)")
    parser.add_argument("--confirmation-timeout", type=float, default=15,
                        help="Лимит подтверждения взлёта и посадки в SDK, с (по умолчанию 15)")
    args = parser.parse_args()
    if (args.fly_at is None) != (args.point_timeout is None):
        parser.error("Для полёта укажите вместе --fly-at и --point-timeout.")
    if args.fly_at is not None and not all(
        isfinite(value) and value > 0 for value in (args.fly_at, args.point_timeout)
    ):
        parser.error("Высота и время должны быть положительными конечными числами.")
    start = read_point("Старт X Y (м): ") if args.fly_at is None else None
    count = int(input("Сколько контрольных точек (3–5)? "))
    if not 3 <= count <= 5:
        raise ValueError("Требуется от 3 до 5 контрольных точек.")
    points = [read_point(f"Точка {i + 1}, X Y (м): ") for i in range(count)]
    if start is not None:
        print_route(start, points)
        return
    from confirmed_flight import connect

    drone = connect(args.confirmation_timeout)
    try:
        fly(drone, points, args.fly_at, args.point_timeout)
    finally:
        drone.close_connection()


if __name__ == "__main__":
    try:
        main()
    except (ValueError, EOFError) as error:
        raise SystemExit(f"Ошибка ввода: {error}") from error
