"""Домашняя площадка № 3–7 в MuJoCo. Камера закреплена на подвижном теле."""
import argparse
import json
import math
import random
import xml.etree.ElementTree as ET

import cv2
import mujoco
import numpy as np

import frames
from physics_bench import MODEL
from visual_course import Board, CourseInterrupted, HALF_BODY, TILE, run_board


UNIT = .01  # Один пиксель исходного плана площадки в метрах модели.
CRUISE = .5 / math.tan(math.radians(30)) + .06
BODY_HEIGHT = .05
BODY_WIDTH = HALF_BODY * UNIT
GATE_Y = 1.6
GATE_HALF_X, GATE_HALF_Z, BAR = .29, .32, .025


def box(world, center, half_size, color, **attrs):
    ET.SubElement(world, "geom", type="box", pos=" ".join(map(str,center)),
                  size=" ".join(map(str,half_size)), rgba=" ".join(map(str,(*color,1))), **attrs)


def draw_floor(world, bitmap, background):
    """Перенос прямоугольных участков разметки в геометрию неподвижной сцены."""
    height, width = bitmap.shape[:2]
    changed_rows = np.flatnonzero(np.any(bitmap[1:] != bitmap[:-1],axis=(1,2))) + 1
    rows = [0,*changed_rows,height]
    for top,bottom in zip(rows,rows[1:]):
        row = bitmap[top]
        changed_cols = np.flatnonzero(np.any(row[1:] != row[:-1],axis=1)) + 1
        cols = [0,*changed_cols,width]
        for left,right in zip(cols,cols[1:]):
            if np.all(row[left] == background):
                continue
            box(world,((left+right)*UNIT/2,-(top+bottom)*UNIT/2,.001),
                ((right-left)*UNIT/2,(bottom-top)*UNIT/2,.001),
                row[left,::-1]/255, contype="0", conaffinity="0")


class MovingField(Board):
    def __init__(self, number, drop_at):
        side = 100 if number == 6 else 500
        self.background = 25 if number in (3,6) else 255
        super().__init__(side,side,self.background,False,drop_at)
        self.number = number
        self.renderer = None
        self.gates = {}
        self.steps = 0
        self.contact_steps = 0
        self.path = []
        self.camera_poses = set()
        self.touchdown = False
        self.settled_s = 0.0
        self.line_checks = self.line_misses = 0

    def takeoff(self):
        tree = ET.fromstring(MODEL)
        lighting = ET.SubElement(tree,"visual")
        ET.SubElement(lighting,"headlight",ambient=".9 .9 .9",diffuse=".1 .1 .1",specular="0 0 0")
        world = tree.find("worldbody")
        grey = self.background/255
        ET.SubElement(world,"geom",type="plane",name="ground",size="6 6 .1",rgba=f"{grey} {grey} {grey} 1")
        body = world.find("body")
        body.find("geom").attrib.update(type="box",size=f"{BODY_WIDTH} {BODY_WIDTH} {BODY_HEIGHT}",name="craft")
        if self.number == 6:
            for color, x in self.gates.items():
                rgb = (1,0,0) if color == "red" else (0,0,1)
                for sign in (-1,1):
                    box(world,(x+sign*(GATE_HALF_X+BAR),GATE_Y,CRUISE),
                        (BAR,BAR,GATE_HALF_Z+2*BAR),rgb,name=f"gate_{color}_side_{sign}")
                    box(world,(x,GATE_Y,CRUISE+sign*(GATE_HALF_Z+BAR)),
                        (GATE_HALF_X,BAR,BAR),rgb,name=f"gate_{color}_bar_{sign}")
            ET.SubElement(body,"camera",name="sensor",pos="0 .09 0",xyaxes="1 0 0 0 0 1",fovy="60")
        else:
            draw_floor(world,self.image,self.background)
            ET.SubElement(body,"camera",name="sensor",pos="0 0 -.06",fovy="60")
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(tree,encoding="unicode"))
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:] = (self.x*UNIT,-self.y*UNIT,BODY_HEIGHT)
        self.data.ctrl[:] = self.data.qpos
        mujoco.mj_forward(self.model,self.data)
        self.renderer = mujoco.Renderer(self.model,height=TILE,width=TILE)
        super().takeoff()
        self.travel((self.data.qpos[0],self.data.qpos[1],CRUISE),"takeoff")

    def _step(self, phase):
        mujoco.mj_step(self.model,self.data)
        self.steps += 1
        floor_contact = False
        ids = [(c.geom1,c.geom2) for c in self.data.contact]
        for a,b in ids:
            names = {self.model.geom(int(a)).name,self.model.geom(int(b)).name}
            if "craft" in names and any(n.startswith("gate_") for n in names):
                self.contact_steps += 1
            if phase == "landing" and {"craft","ground"} == names:
                floor_contact = True
        if phase == "landing":
            if floor_contact and np.linalg.norm(self.data.qvel) < .02:
                self.settled_s += self.model.opt.timestep
            else:
                self.settled_s = 0.0
            self.touchdown = self.settled_s >= .15
        if phase == "move":
            self.path.append(tuple(self.data.qpos))

    def travel(self,target,phase):
        self.data.ctrl[:] = target
        for step in range(300):
            self._step(phase)
            if phase == "move" and self.number == 4 and step in (12,30,52):
                frame = self.camera()
                self.line_checks += 1
                if not np.any(frames.gray(frame)<70):
                    self.line_misses += 1
                    raise CourseInterrupted("line lost in motion")
            if phase == "landing" and self.touchdown:
                return
            if phase != "landing" and np.linalg.norm(self.data.qpos-target) < .001:
                return
        raise TimeoutError(f"Модель не завершила этап {phase}")

    def gate_move(self, lane, depth):
        self.moves += 1
        self.travel((lane,depth,CRUISE),"move")

    def move(self,x,y):
        if not super().move(x,y):
            return False
        self.travel((x*UNIT,-y*UNIT,CRUISE),"move")
        return True

    def footprint_center(self):
        x,y,_ = self.data.qpos
        return float(x)/UNIT,-float(y)/UNIT

    def capture_frame(self):
        mujoco.mj_forward(self.model,self.data)
        self.camera_poses.add(tuple(np.round(self.data.qpos,3)))
        self.renderer.update_scene(self.data,camera="sensor")
        return cv2.cvtColor(self.renderer.render(),cv2.COLOR_RGB2BGR)

    def land(self):
        if self.airborne:
            x,y,_ = self.data.qpos
            self.travel((x,y,BODY_HEIGHT-.025),"landing")
            super().land()


def gate_exercise(board,rng,color):
    # Игровые ворота занимают две фиксированные полосы; цвет полос меняется.
    lanes = [.08,.92]
    rng.shuffle(lanes)
    board.gates = dict(zip(("red","blue"),lanes))
    board.x,board.y = 50,0
    board.takeoff()
    seen = frames.gate_center(board.camera(),color,80)
    if seen is None:
        return {"passed":False,"reason":"gate missing"}
    lane = min(lanes) if seen[0] < TILE/2 else max(lanes)
    board.gate_move(lane,0)
    seen = frames.gate_center(board.camera(),color,80)
    if seen is None or abs(seen[0]-TILE/2) > 3:
        return {"passed":False,"reason":"lane not confirmed"}
    # До входа камера повторно подтверждает проём. В рамке и за ней вид меняется.
    for stop in (.5,.8,1.15,1.5,1.85,2.05):
        board.gate_move(lane,stop)
        image = board.camera()
        if stop <= .8 and frames.gate_center(image,color,80) is None:
            return {"passed":False,"reason":"gate lost before entry"}
    cross_section = [(x,z) for x,y,z in board.path if abs(y-GATE_Y) <= BODY_WIDTH+BAR]
    exit_clearance = float(board.data.qpos[1])-GATE_Y-BAR-BODY_WIDTH
    if not cross_section:
        return {"passed":False,"reason":"gate plane not crossed"}
    crossed = None
    clearance = -math.inf
    for name,cx in board.gates.items():
        margin = min(min(GATE_HALF_X-abs(x-cx)-BODY_WIDTH,GATE_HALF_Z-abs(z-CRUISE)-BODY_HEIGHT)
                     for x,z in cross_section)
        if margin > clearance:
            crossed,clearance = name,float(margin)
    return {"passed":crossed == color and clearance>0 and exit_clearance>0 and board.contact_steps==0,
            "crossed":crossed, "edge_clearance_m":clearance,"exit_clearance_m":exit_clearance,
            "frame_contacts":board.contact_steps}


def simulate(number, *, seed=0, target_id=12, color="red", stop_after=17, drop_at=None):
    if number not in range(3,8) or stop_after < 0 or (drop_at is not None and drop_at < 1):
        raise ValueError("Проверьте номер задания, остановку и номер потерянного кадра.")
    board = MovingField(number,drop_at)
    try:
        result = run_board(board,number,random.Random(seed),target_id,color,stop_after,gate_exercise)
        result.update(physics_steps=board.steps, distinct_camera_poses=len(board.camera_poses),
                      floor_contact_after_landing=board.touchdown,settled_contact_s=board.settled_s,
                      final_speed_m_s=float(np.linalg.norm(board.data.qvel)))
        if number == 6:
            result['frame_contacts'] = board.contact_steps
        if "edge_clearance_px" in result:
            result["edge_clearance_m"] = result["edge_clearance_px"]*UNIT
        if number == 4 and board.path:
            xy = np.array(board.path)[:,:2]
            # Для прямоугольника с углами (0.5,-0.5) и (4.5,-4.5)
            # ближайшая точка каждой стороны вычисляется по измеренной позиции тела.
            x,y = xy.T
            dx = np.maximum.reduce([.5-x,x-4.5,np.zeros_like(x)])
            dy = np.maximum.reduce([-4.5-y,y+.5,np.zeros_like(y)])
            vertical = np.hypot(np.minimum(abs(x-.5),abs(x-4.5)),dy)
            horizontal = np.hypot(dx,np.minimum(abs(y+.5),abs(y+4.5)))
            result.update(max_line_error_m=float(np.minimum(vertical,horizontal).max()),
                          path_length_m=float(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum()),
                          in_motion_line_frames=board.line_checks,in_motion_line_misses=board.line_misses)
            result["passed"] &= result["max_line_error_m"] < .1 and board.line_misses == 0
        if number == 5:
            result["finish_error_m"] = math.dist(board.data.qpos[:2],(4.5,-1.5))
            result["passed"] &= result["finish_error_m"] < .15
        result["passed"] &= board.touchdown
        return result
    finally:
        if board.renderer is not None:
            board.renderer.close()


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("number",type=int,choices=range(3,8))
    cli.add_argument("--seed",type=int,default=0)
    cli.add_argument("--id",type=int,default=12,dest="target_id")
    cli.add_argument("--color",choices=frames.HUES,default="red")
    cli.add_argument("--stop-after",type=int,default=17)
    cli.add_argument("--drop-at",type=int)
    result = simulate(**vars(cli.parse_args()))
    print(json.dumps(result,ensure_ascii=False))
    raise SystemExit(0 if result["passed"] else 1)
