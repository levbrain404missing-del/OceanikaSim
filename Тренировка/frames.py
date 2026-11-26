"""Разбор одного сохранённого кадра для упражнений № 3–7."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def gray(frame):
    if frame is None or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0:
        raise ValueError("Ожидается непустой цветной кадр BGR uint8.")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def check_threshold(threshold):
    if not 0 <= threshold <= 255:
        raise ValueError("Порог яркости должен лежать между 0 и 255.")


def single_center(contours, min_area):
    if min_area <= 0:
        raise ValueError("Площадь должна быть положительной.")
    selected = [c for c in contours if cv2.contourArea(c) >= min_area]
    if len(selected) == 1:
        m = cv2.moments(selected[0])
        return round(m["m10"] / m["m00"]), round(m["m01"] / m["m00"])
    return None


def bright_zone(frame, threshold, min_area):
    check_threshold(threshold)
    mask = cv2.inRange(gray(frame), threshold, 255)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = mask.shape
    visible = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if x > 0 and y > 0 and x + w < width and y + h < height:
            visible.append(contour)
    return single_center(visible, min_area)


def line_x(frame, row, threshold, min_width):
    check_threshold(threshold)
    image = gray(frame)
    if not 0 <= row < len(image) or min_width < 1:
        raise ValueError("Проверьте строку кадра и минимальную ширину полосы.")
    dark = image[row] <= threshold
    changes = np.diff(np.asarray([False, *dark, False], dtype=np.int8))
    spans = [(a,b) for a,b in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1))
             if b-a >= min_width]
    if len(spans) == 1:
        left, right = spans[0]
        return float((left + right - 1) / 2)
    return None


HUES = {"red": [(0,10),(170,179)], "green": [(35,85)], "blue": [(95,135)], "yellow": [(20,34)]}


def gate_center(frame, color, min_area):
    if color not in HUES:
        raise ValueError("Неизвестный цвет.")
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for low, high in HUES[color]:
        mask |= cv2.inRange(hsv, (low,80,60), (high,255,255))
    contours, tree = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    holes = [] if tree is None else [c for c, h in zip(contours, tree[0]) if h[3] >= 0]
    return single_center(holes, min_area)


DICTIONARIES = ("DICT_4X4_50", "DICT_ARUCO_ORIGINAL")


def markers(frame, dictionary_name):
    if dictionary_name not in DICTIONARIES:
        raise ValueError("Выберите словарь из учебных материалов.")
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
    corners, ids, _ = cv2.aruco.ArucoDetector(dictionary).detectMarkers(gray(frame))
    if ids is None:
        return []
    return [(int(idx), tuple(map(float, box.reshape(4,2).mean(axis=0)))) for idx, box in zip(ids.ravel(), corners)]


def choose_marker(found, target_id):
    centers = [center for idx, center in found if idx == target_id]
    return centers[0] if len(centers) == 1 else None


def marker_action(marker_id, finish_id):
    if marker_id == finish_id:
        return "finish"
    return "left" if marker_id % 2 == 0 else "right"


def analyze(frame, args):
    if args.task == "zone":
        return bright_zone(frame, args.threshold, args.min_area)
    if args.task == "line":
        return line_x(frame, args.row, args.threshold, args.min_width)
    if args.task == "gate":
        return gate_center(frame, args.color, args.min_area)
    found = markers(frame, args.dictionary)
    if args.task == "sign":
        return [(idx, marker_action(idx, args.id)) for idx, _ in found]
    return choose_marker(found, args.id)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--video", action="store_true", help="Обработать все кадры записанного видео")
    tasks = parser.add_subparsers(dest="task", required=True)
    zone = tasks.add_parser("zone")
    zone.add_argument("threshold", type=int)
    zone.add_argument("min_area", type=int)
    line = tasks.add_parser("line")
    line.add_argument("row", type=int)
    line.add_argument("threshold", type=int)
    line.add_argument("min_width", type=int)
    gate = tasks.add_parser("gate")
    gate.add_argument("color", choices=HUES)
    gate.add_argument("min_area", type=int)
    for name in ("sign","marker"):
        task = tasks.add_parser(name)
        task.add_argument("dictionary", choices=DICTIONARIES)
        task.add_argument("id", type=int, help="ID финиша для sign, искомый ID для marker")
    args = parser.parse_args()
    if args.video:
        if not args.image.is_file():
            parser.error("Видеофайл отсутствует.")
        capture = cv2.VideoCapture(str(args.image))
        index = 0
        try:
            while capture.isOpened():
                ok, frame = capture.read()
                if not ok:
                    break
                print(json.dumps({"frame": index, "result": analyze(frame, args)}))
                index += 1
        finally:
            capture.release()
        if not index:
            parser.error("Не удалось прочитать ни одного кадра.")
        return
    data = args.image.read_bytes()
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) if data else None
    if frame is None:
        parser.error("Файл не удалось прочитать как изображение.")
    result = analyze(frame, args)
    print(result if result is not None else "Нет однозначного результата")


if __name__ == "__main__":
    main()
