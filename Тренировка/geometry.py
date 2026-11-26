"""Учебные расчёты для № 1, 2, 9. Локальные XYZ в метрах, углы ввода в градусах."""
import argparse
import csv
import json
import math


def check_start(start):
    if len(start) != 3 or not all(map(math.isfinite, start)):
        raise ValueError("Нужны три конечные координаты старта.")


def circle(start, radius):
    """Старт фигуры в нижней точке, центр выше на radius."""
    check_start(start)
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("Радиус должен быть положительным и конечным.")
    # Разбиение для домашнего расчёта. Защита от непрактичного размера вывода.
    segments = 64
    while radius * 2 * math.sin(math.pi / (2 * segments)) ** 2 > 0.2:
        segments *= 2
        if segments > 65536:
            raise ValueError("Радиус слишком велик для стенда.")
    x, y, z = start
    path = [tuple(start)]
    for i in range(1, segments):
        angle = math.tau * i / segments
        path.append((x + radius * math.sin(angle), y, z + radius * (1 - math.cos(angle))))
    return [*path, tuple(start)]


def figure_eight(start):
    """Старт в точке касания, центры на расстоянии метра слева и справа."""
    check_start(start)
    x, y, z = start
    result = [tuple(start)]
    for side in (-1, 1):
        for i in range(1, 64):
            angle = math.tau * i / 64
            result.append((x + side * (1 - math.cos(angle)), y, z + math.sin(angle)))
        result.append(tuple(start))
    return result


def spherical_target(start, distance, azimuth, elevation):
    check_start(start)
    if not all(map(math.isfinite, (distance, azimuth, elevation))) or distance < 0 or abs(elevation) > 90:
        raise ValueError("Проверьте дистанцию и углы. Возвышение должно лежать в пределах ±90°.")
    az = math.radians(azimuth % 360)
    el = math.radians(elevation)
    horizontal = distance * math.cos(el)
    x, y, z = start
    target = x + horizontal * math.cos(az), y + horizontal * math.sin(az), z + distance * math.sin(el)
    check_start(target)
    return target


def longest_hold(samples, target, max_gap):
    """Самая длинная подтверждённая серия внутри допуска по каждой оси."""
    check_start(target)
    if not math.isfinite(max_gap) or max_gap <= 0:
        raise ValueError("Укажите положительный допустимый интервал измерений.")
    previous = since = None
    longest = 0.0
    for stamp, position in samples:
        if not math.isfinite(stamp) or (previous is not None and stamp <= previous):
            raise ValueError("Отметки времени должны строго возрастать.")
        if previous is not None and stamp - previous > max_gap:
            since = None
        previous = stamp
        inside = position is not None and len(position) == 3 and all(
            math.isfinite(v) and abs(v - expected) <= 0.2 + 1e-12
            for v, expected in zip(position, target)
        )
        if inside:
            if since is None:
                since = stamp
            longest = max(longest, stamp - since)
        else:
            since = None
    return longest


def read_samples(stream):
    for row in csv.DictReader(stream):
        fields = [row[key] for key in ("x_m", "y_m", "z_m")]
        yield float(row["t_s"]), tuple(map(float, fields)) if all(fields) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=float, nargs=3, default=(0, 0, 0), metavar=("X","Y","Z"))
    tasks = parser.add_subparsers(dest="task", required=True)
    tasks.add_parser("circle").add_argument("radius", type=float)
    tasks.add_parser("eight")
    tasks.add_parser("spherical").add_argument("values", type=float, nargs=3, metavar="NUMBER")
    hold = tasks.add_parser("hold")
    hold.add_argument("values", type=float, nargs=3, metavar="NUMBER")
    hold.add_argument("csv_file")
    hold.add_argument("--max-gap", type=float, required=True, help="Допустимый промежуток между отсчётами, с")
    args = parser.parse_args()
    if args.task == "hold":
        target = spherical_target(args.start, *args.values)
        with open(args.csv_file, encoding="utf-8-sig", newline="") as stream:
            duration = longest_hold(read_samples(stream), target, args.max_gap)
        print(json.dumps({"observed_s": duration, "held_3s": duration >= 3}))
        return
    if args.task == "spherical":
        rows = [spherical_target(args.start, *args.values)]
    elif args.task == "eight":
        rows = figure_eight(args.start)
    else:
        rows = circle(args.start, args.radius)
    print("x_m,y_m,z_m")
    for x, y, z in rows:
        print(f"{x:.6f},{y:.6f},{z:.6f}")


if __name__ == "__main__":
    main()
