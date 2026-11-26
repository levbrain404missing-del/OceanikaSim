"""Настольная модель заданий № 3–7. Координаты учебного поля заданы в пикселях."""
import argparse
import json
import random

import cv2
import numpy as np

import frames


TILE = 100
HALF_BODY = 8
DICTIONARY = "DICT_4X4_50"
DIRECTIONS = ((0,-1), (1,0), (0,1), (-1,0))


class CourseInterrupted(RuntimeError):
    pass


class Board:
    def __init__(self, width, height, background, blank, drop_at):
        self.image = np.full((height,width,3), background, np.uint8)
        self.x = self.y = TILE // 2
        self.airborne = False
        self.landings = self.moves = self.reads = 0
        self.blank, self.drop_at = blank, drop_at

    def takeoff(self):
        self.airborne = True

    def land(self):
        if self.airborne:
            self.airborne = False
            self.landings += 1

    def move(self, x, y):
        if not self.airborne:
            raise RuntimeError("Попытка движения после посадки")
        h, w = self.image.shape[:2]
        if not (HALF_BODY <= x < w-HALF_BODY and HALF_BODY <= y < h-HALF_BODY):
            return False
        self.x, self.y = x, y
        self.moves += 1
        return True

    def camera(self):
        self.reads += 1
        if self.drop_at is not None and self.reads >= self.drop_at:
            raise ConnectionError('camera unavailable')
        if self.blank:
            return np.full((TILE,TILE,3), 120, np.uint8)
        return self.capture_frame()

    def footprint_center(self):
        return self.x, self.y

    def capture_frame(self):
        return cv2.getRectSubPix(self.image, (TILE,TILE), (float(self.x)-.5,float(self.y)-.5))


def draw_marker(board, center, ident):
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker = cv2.aruco.generateImageMarker(d, ident, 42)
    x,y = center
    board.image[y-21:y+21,x-21:x+21] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)


def landing(board, rng, number, requested):
    cells = [(x,y) for y in range(3) for x in (range(4) if y%2 == 0 else reversed(range(4)))]
    placements = rng.sample(cells, 3 if number == 7 else 1)
    centers = {}
    for idx, (col,row) in zip((12,7,15), placements):
        center = col*TILE+50+rng.randint(-9,9), row*TILE+50+rng.randint(-9,9)
        centers[idx] = center
        x,y = center
        if number == 7:
            draw_marker(board, center, idx)
        else:
            cv2.rectangle(board.image,(x-22,y-22),(x+22,y+22),(240,240,240),-1)
    board.takeoff()
    def detect():
        image = board.camera()
        if number == 3:
            return frames.bright_zone(image,180,500)
        return frames.choose_marker(frames.markers(image,DICTIONARY),requested)
    for col,row in cells:
        board.move(col*TILE+50,row*TILE+50)
        seen = detect()
        if seen is None:
            continue
        board.move(round(board.x + seen[0]-50), round(board.y + seen[1]-50))
        confirm = detect()
        if confirm is None or max(abs(v-50) for v in confirm) > 2:
            break
        board.land()
        chosen = centers.get(requested if number == 7 else 12)
        if chosen is None:
            return {"passed":False,"feet_inside":False,"reason":"requested ID absent from scene"}
        cx,cy = chosen
        px,py = board.footprint_center()
        clearance = 21 - HALF_BODY - max(abs(px-cx),abs(py-cy))
        inside = clearance >= 0
        return {"passed": inside, "feet_inside": inside, "edge_clearance_px": clearance, "cell": (col,row)}
    return {"passed": False, "feet_inside": False, "reason": "no confirmed zone"}


def line_course(board, stop_requested):
    # Кольцо существует на изображении площадки. Наблюдателю доступен текущий участок.
    corners = [(50,50),(450,50),(450,450),(50,450)]
    cv2.polylines(board.image,[np.array(corners,np.int32)],True,(0,0,0),9)
    board.takeoff()
    previous = None
    laps = 0
    while not stop_requested(board.moves):
        image = board.camera()
        # После каждого поворота изображения верхняя строка соответствует другому краю.
        exits = [i for i in range(4)
                 if frames.line_x(np.rot90(image,i).copy(), 2, 70, 4) is not None]
        options = [i for i in exits if previous is None or i != (previous+2)%4]
        if previous is None and len(options) == 2:
            options = options[:1]
        if len(options) != 1:
            return {"passed": False, "laps": laps, "reason": "lost or ambiguous line"}
        previous = options[0]
        dx,dy = DIRECTIONS[previous]
        if not board.move(board.x+TILE*dx,board.y+TILE*dy):
            return {"passed": False, "laps": laps, "reason": "outside board"}
        if (board.x,board.y) == corners[0]:
            laps += 1
    return {"passed": laps >= 1, "laps": laps, "reason": "external stop"}


def sign_course(board):
    board.x, board.y = 50,350
    finish_id = 14
    signs = {(250,350):8, (250,150):7, (450,150):finish_id}
    for center, ident in signs.items():
        draw_marker(board, center, ident)
    board.takeoff()
    direction = 1
    actions = []
    last_id = None
    while True:
        # Два наблюдения одной стоянки проверяют защиту от повторного поворота.
        for _ in range(2):
            found = frames.markers(board.camera(), DICTIONARY)
            if len(found) > 1:
                return {"passed": False, "actions": actions, "reason": "ambiguous markers"}
            ident = found[0][0] if found else None
            if ident is not None and ident != last_id:
                action = frames.marker_action(ident,finish_id)
                actions.append(action)
                if action == "finish":
                    board.land()
                    return {"passed": actions == ["left","right","finish"] and
                            signs.get((board.x,board.y)) == finish_id, "actions": actions}
                direction = (direction + (1 if action == "right" else -1))%4
            last_id = ident
        dx,dy = DIRECTIONS[direction]
        if not board.move(board.x+TILE*dx,board.y+TILE*dy) or board.moves >= 25:
            return {"passed": False,"actions": actions,"reason": "finish missing"}


def gates(board, rng, requested):
    # Фронтальная проекция двух рам и отдельная шкала глубины игрового поля.
    centers = [27,73]
    rng.shuffle(centers)
    openings = dict(zip(("red","blue"), centers))
    for name, cx in openings.items():
        color = (0,0,230) if name == "red" else (230,0,0)
        cv2.rectangle(board.image,(cx-17,50-23),(cx+17,50+23),color,4)
    board.takeoff()
    def observe():
        return frames.gate_center(board.camera(), requested, 250)
    seen = observe()
    if seen is None:
        return {"passed": False,"fully_clear": False,"reason": "opening missing"}
    if not board.move(round(board.x+seen[0]-50),board.y):
        return {"passed": False,"fully_clear": False,"reason": "outside board"}
    depth, plane = 0, 30
    crossed = None
    while depth-HALF_BODY <= plane:
        seen = observe()
        if seen is None or max(abs(v-50) for v in seen) > 2:
            return {"passed": False,"fully_clear": False,"reason": "opening lost"}
        depth += 3
        if abs(depth-plane) <= HALF_BODY:
            crossed = next((name for name,x in openings.items()
                            if abs(board.x-x)+HALF_BODY < 14 and abs(board.y-50)+HALF_BODY < 20),None)
            if crossed is None:
                return {"passed": False,"fully_clear": False,"reason": "frame contact"}
    board.land()
    return {"passed": crossed == requested, "fully_clear": depth-HALF_BODY > plane, "crossed": crossed}


def exercise(number, *, seed=0, target_id=12, color="red", stop_after=17, blank=False, drop_at=None):
    if number not in (3,4,5,6,7) or stop_after < 0:
        raise ValueError("Проверьте номер и число шагов.")
    rng = random.Random(seed)
    side = 100 if number == 6 else 500
    board = Board(side,side,25 if number in (3,6) else 255,blank,drop_at)
    return run_board(board, number, rng, target_id, color, stop_after)


def run_board(board, number, rng, target_id, color, stop_after, gate_task=gates):
    try:
        if number in (3,7):
            report = landing(board,rng,number,target_id)
        elif number == 4:
            report = line_course(board,lambda moves: moves >= stop_after)
        elif number == 5:
            report = sign_course(board)
        else:
            report = gate_task(board,rng,color)
    except (ConnectionError, CourseInterrupted, TimeoutError) as error:
        report = dict(passed=False, reason=str(error))
    finally:
        board.land()
    return dict(report, landings=board.landings, moves=board.moves, frames=board.reads)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("number",type=int,choices=range(3,8))
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--id",type=int,dest="target_id",default=12)
    parser.add_argument("--color",choices=frames.HUES,default="red")
    parser.add_argument("--stop-after",type=int,default=17,help="Внешний сигнал стенда после N шагов")
    parser.add_argument("--drop-at",type=int,help="Отключить изображение с указанного кадра")
    args = parser.parse_args()
    result = exercise(**vars(args))
    print(json.dumps(result))
    raise SystemExit(0 if result["passed"] else 1)
