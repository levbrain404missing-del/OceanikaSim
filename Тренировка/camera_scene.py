"""Воспроизводимые виды учебных объектов через камеру MuJoCo."""
import argparse
from pathlib import Path

import cv2
import mujoco


def tile(x, y, half_width, half_height, color):
    return (f'<geom type="box" pos="{x} {y} .01" size="{half_width} {half_height} .005" '
            f'rgba="{color} 1"/>')


def objects(scene):
    if scene == "zone":
        return [tile(.25, 0, .2, .14, "1 1 1")]
    if scene == "line":
        return [tile(0, 0, .035, 2, "0 0 0")]
    geoms = []
    if scene == "gate":
        for x, color in ((-.48,"1 0 0"), (.48,"0 0 1")):
            for sign in (-1,1):
                geoms.append(tile(x + sign*.22, 0, .025, .225, color))
                geoms.append(tile(x, sign*.2, .22, .025, color))
    elif scene == "markers":
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        for x, idx in ((-.48,8), (.48,15)):
            cells = cv2.aruco.generateImageMarker(dictionary, idx, 6)
            for row, col in zip(*((cells == 0).nonzero())):
                geoms.append(tile(x + (col-2.5)*.08, (2.5-row)*.08, .04, .04, "0 0 0"))
    else:
        raise ValueError(scene)
    return geoms


def render_pair(scene):
    floor = "0.05 0.05 0.05" if scene == "zone" else "1 1 1"
    xml = f"""
    <mujoco><visual><headlight ambient=".85 .85 .85" diffuse=".15 .15 .15" specular="0 0 0"/></visual>
    <worldbody><geom type="plane" size="4 4 .1" rgba="{floor} 1"/>
    <camera name="first" pos="0 0 2" fovy="60"/>
    <camera name="second" pos=".25 0 2" fovy="60"/>
    {''.join(objects(scene))}
    </worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    frames = []
    with mujoco.Renderer(model, height=240, width=320) as renderer:
        for camera in ("first", "second"):
            renderer.update_scene(data, camera=camera)
            frames.append(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
    return frames


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", choices=("zone","line","gate","markers"))
    parser.add_argument("output", type=Path, help="Новая папка для двух PNG")
    args = parser.parse_args()
    images = render_pair(args.scene)
    args.output.mkdir()
    for index, frame in enumerate(images, start=1):
        ok, png = cv2.imencode(".png", frame)
        if not ok:
            raise RuntimeError("Не удалось записать PNG")
        (args.output / f"{args.scene}-{index}.png").write_bytes(png.tobytes())
