"""Снимки через API камеры для повторения ошибок распознавания на ПК."""
import argparse
import math
from pathlib import Path

import cv2


def capture(out, count, timeout, open_camera):
    if count < 1 or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Укажите число кадров и положительный тайм-аут.")
    out.mkdir()
    camera = open_camera()
    try:
        for index in range(1, count+1):
            image = camera.get_cv_frame(timeout=timeout)
            ok, png = cv2.imencode(".png", image)
            if not ok:
                raise OSError("Ошибка кодирования PNG.")
            (out / f"frame-{index:04}.png").write_bytes(png.tobytes())
    finally:
        camera.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder",type=Path,help="Новая папка для кадров")
    parser.add_argument("--count",type=int,default=1)
    parser.add_argument("--timeout",type=float,default=5)
    parser.add_argument("--address",help="IP:порт для RTSP-драйвера SDK")
    args = parser.parse_args()
    from pioneer_sdk2 import Camera
    def open_camera():
        return Camera(camera_ip=args.address) if args.address else Camera()
    capture(args.folder,args.count,args.timeout,open_camera)


if __name__ == "__main__":
    main()
