"""Домашние сценарии MuJoCo для № 1, 2, 8 и 9. Запуск без оборудования."""
import argparse
import csv
import json
import math
from unittest.mock import patch

import mujoco

import geometry
import shortest_route


# Учебное тело перемещается по трём осям с инерцией и демпфированием.
# Масса и приводы выбраны для стенда. Параметры «Мини 2» ими не задаются.
MODEL = """
<mujoco>
  <option timestep=".01" gravity="0 0 0" integrator="implicitfast"/>
  <worldbody>
    <body><geom type="sphere" size=".05" mass="1"/>
      <joint name="x" type="slide" axis="1 0 0" damping="12"/>
      <joint name="y" type="slide" axis="0 1 0" damping="12"/>
      <joint name="z" type="slide" axis="0 0 1" damping="12"/>
    </body>
  </worldbody>
  <actuator>
    <position joint="x" kp="36"/>
    <position joint="y" kp="36"/>
    <position joint="z" kp="36"/>
  </actuator>
</mujoco>
"""


class MotionModel:
    def __init__(self, start, writer=None):
        self.model = mujoco.MjModel.from_xml_string(MODEL)
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:] = start
        self.data.ctrl[:] = start
        mujoco.mj_forward(self.model, self.data)
        self.writer = writer
        self.samples = []

    @property
    def position(self):
        return tuple(self.data.qpos.tolist())

    def set_target(self, xyz):
        self.data.ctrl[:] = xyz

    def advance(self, seconds):
        steps = max(1, round(seconds / self.model.opt.timestep))
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)
            xyz = self.position
            self.samples.append((self.data.time, xyz))
            if self.writer is not None:
                self.writer.writerow((self.data.time, *xyz, *self.data.ctrl))


class RouteDevice:
    """Адаптер команд для запуска существующей fly() с модельным временем."""
    def __init__(self, model, drop_event):
        self.model = model
        self.drop_event = drop_event
        self.pending = None
        self.waypoints = self.events = self.landings = 0
        self.phase = "ground"

    def arm(self):
        if self.phase != "ground":
            return False
        self.phase = "armed"
        return True

    def takeoff(self):
        if self.phase != "armed":
            return False
        x, y, _ = self.model.position
        self.model.set_target((x,y,1))
        self.model.advance(2)
        self.phase = "air"
        return True

    def go_to_local_point(self, x, y, z):
        if self.phase != "air":
            return False
        self.pending = (x,y,z)
        self.model.set_target(self.pending)
        self.waypoints += 1

    def point_reached(self):
        if self.drop_event or self.pending is None or math.dist(self.model.position, self.pending) > .025:
            return False
        self.pending = None
        self.events += 1
        return True

    def get_local_position_lps(self):
        return self.model.position

    def land(self):
        self.landings += 1
        x, y, _ = self.model.position
        self.model.set_target((x,y,0))
        self.model.advance(2)
        self.phase = "ground"
        return abs(self.model.position[2]) < .025


def run_route(drop_event, writer=None):
    start = (1,-1,0)
    model = MotionModel(start, writer)
    device = RouteDevice(model, drop_event)
    timed_out = False
    with patch.object(shortest_route, "monotonic", lambda: model.data.time), \
         patch.object(shortest_route, "sleep", model.advance):
        try:
            shortest_route.fly(device, [(2,-1), (2,0), (1,0)], 1, 3)
        except TimeoutError:
            if not drop_event:
                raise
            timed_out = True
    return {"timeout": timed_out, "waypoints_sent": device.waypoints, "confirmed": device.events,
            "landing_requests": device.landings, "final_xyz_m": model.position,
            "return_xy_error_m": math.dist(model.position[:2], start[:2])}


def run_figure(name, writer=None):
    start = (0,0,1.5)
    if name == "circle":
        points = geometry.circle(start, 1)
    elif name == "eight":
        points = geometry.figure_eight(start)
    else:
        raise ValueError(name)
    model = MotionModel(start, writer)
    max_error = 0.0
    for i, point in enumerate(points[1:], 1):
        if name == "circle":
            cx, cz = 0, 2.5
        else:
            cx, cz = (-1 if i <= len(points) // 2 else 1), 1.5
        before = len(model.samples)
        model.set_target(point)
        model.advance(.25)
        for _, (x, y, z) in model.samples[before:]:
            radial_error = math.hypot(x-cx, z-cz) - 1
            max_error = max(max_error, math.hypot(y, radial_error))
    model.advance(1)
    return {"max_radial_error_m": max_error, "return_error_m": math.dist(model.position, start),
            "samples": len(model.samples)}


def run_hold(writer=None):
    start = (1,-2,1)
    model = MotionModel(start, writer)
    target = geometry.spherical_target(start, 2, 45, 30)
    model.set_target(target)
    model.advance(5)
    observed = geometry.longest_hold(model.samples, target, .02)
    model.set_target(start)
    model.advance(2)
    return {"observed_hold_s": observed, "return_error_m": math.dist(model.position, start)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", choices=("circle","eight","route","lost-event","hold"))
    parser.add_argument("--csv", help="Сохранить отсчёты в новый файл")
    args = parser.parse_args()
    stream = open(args.csv, "x", newline="", encoding="utf-8") if args.csv else None
    try:
        writer = csv.writer(stream) if stream else None
        if writer:
            writer.writerow(("t_s","x_m","y_m","z_m","target_x_m","target_y_m","target_z_m"))
        if args.scenario in ("circle", "eight"):
            result = run_figure(args.scenario, writer)
        elif args.scenario == "hold":
            result = run_hold(writer)
        else:
            result = run_route(args.scenario == "lost-event", writer)
        print(json.dumps({"mujoco": mujoco.__version__, "scenario": args.scenario, **result}, indent=2))
    finally:
        if stream:
            stream.close()


if __name__ == "__main__":
    main()
