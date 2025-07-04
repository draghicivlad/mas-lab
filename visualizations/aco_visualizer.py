import sys
from pathlib import Path

# Ensure project root on sys.path so that 'algorithms' package is importable when
# this script is run directly (its working directory is visualizations/).
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import math
import time
import random
from typing import List, Tuple, Dict, Any

import pygame
import socketio
import requests
from algorithms.grid_astar import AStarGrid

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
BACKEND_URL = "http://localhost:5000"
WINDOW_SIZE = (1000, 800)
FPS = 60

# Visual parameters
TASK_RADIUS = 10
ROBOT_RADIUS = 12
PATH_COLOR = [(200, 0, 0), (0, 150, 0), (0, 0, 200)]
OBSTACLE_COLOR = (80, 80, 80)

random.seed(42)

# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def lerp(a: Tuple[float, float], b: Tuple[float, float], t: float):
    return a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t


def dist(a: Tuple[float, float], b: Tuple[float, float]):
    return math.hypot(a[0] - b[0], a[1] - b[1])


# --------------------------------------------------
# Path planning helper for visualisation
# --------------------------------------------------

def compute_astar_path(start: Tuple[float, float], goal: Tuple[float, float]):
    """Return list of points (world coords) along grid A* path avoiding obstacles."""
    cell_size = 20
    cols = WINDOW_SIZE[0] // cell_size
    rows = WINDOW_SIZE[1] // cell_size
    occ = [[False for _ in range(cols)] for _ in range(rows)]
    for ox, oy, w, h in obstacles_data:
        x0 = int(ox // cell_size)
        y0 = int(oy // cell_size)
        x1 = int((ox + w) // cell_size)
        y1 = int((oy + h) // cell_size)
        for cy in range(y0, min(rows, y1 + 1)):
            for cx in range(x0, min(cols, x1 + 1)):
                occ[cy][cx] = True
    grid = AStarGrid(occ)
    def to_grid(p):
        return int(p[0] // cell_size), int(p[1] // cell_size)
    def to_world(g):
        return (g[0] + 0.5) * cell_size, (g[1] + 0.5) * cell_size

    s_grid = to_grid(start)
    g_grid = to_grid(goal)
    cells = grid.shortest_path(s_grid, g_grid)
    if cells is None:
        return [start, goal]
    path_pts = [to_world(c) for c in cells]
    # ensure last point is exact goal center
    path_pts[-1] = goal
    path_pts[0] = start
    return path_pts


# --------------------------------------------------
# Build demo scenario (15 tasks, 3 robots, dependencies)
# --------------------------------------------------

def generate_demo_scenario():
    tasks = []
    margin = 100
    cols = 5
    rows = 3
    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx >= 15:
                break
            x = margin + c * ( (WINDOW_SIZE[0]-2*margin)/(cols-1) ) + random.uniform(-20,20)
            y = margin + r * ( (WINDOW_SIZE[1]-2*margin)/(rows-1) ) + random.uniform(-20,20)
            tasks.append({"id": idx, "x": x, "y": y})
            idx += 1

    # simple dependency graph example
    dependencies = {
        1: [0],
        2: [1],
        4: [3, 5],
        7: [2],
        9: [8],
        10: [9],
        13: [12, 11]
    }

    robots = [
        {"id": 0, "x": 50, "y": 50},
        {"id": 1, "x": WINDOW_SIZE[0]-50, "y": 60},
        {"id": 2, "x": 60, "y": WINDOW_SIZE[1]-60},
    ]
    return tasks, dependencies, robots


# --------------------------------------------------
# Pygame Visualizer
# --------------------------------------------------

pygame.init()
screen = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption("ACO Task Allocation Visualizer")
clock = pygame.time.Clock()
FONT = pygame.font.SysFont("arial", 16)

# Socket.IO client
sio = socketio.Client()

tasks_data, deps_data, robots_data, obstacles_data = (None, None, None, None)


class RobotAnim:
    def __init__(self, path: List[Tuple[float, float]], color):
        self.color = color
        self.path = path
        self.segment_idx = 0
        self.t = 0.0  # progress along current segment (0..1)
        self.speed = 100.0  # pixels per second

    def update(self, dt):
        if self.segment_idx >= len(self.path) - 1:
            return  # finished
        a = self.path[self.segment_idx]
        b = self.path[self.segment_idx + 1]
        seg_len = dist(a, b)
        if seg_len < 1e-6:
            self.segment_idx += 1
            self.t = 0
            return
        move = self.speed * dt
        seg_progress = move / seg_len
        self.t += seg_progress
        while self.t >= 1.0 and self.segment_idx < len(self.path) - 1:
            self.t -= 1.0
            self.segment_idx += 1
            if self.segment_idx < len(self.path) - 1:
                a = self.path[self.segment_idx]
                b = self.path[self.segment_idx + 1]
                seg_len = dist(a, b)
                seg_progress = self.t * seg_len / seg_len  # noop

    def pos(self):
        if self.segment_idx >= len(self.path) - 1:
            return self.path[-1]
        a = self.path[self.segment_idx]
        b = self.path[self.segment_idx + 1]
        return lerp(a, b, self.t)

    def draw(self, surf):
        # draw entire planned path
        pygame.draw.lines(surf, self.color, False, self.path, 2)
        # draw robot
        x, y = self.pos()
        pygame.draw.circle(surf, self.color, (int(x), int(y)), ROBOT_RADIUS)


class ACOVisualizer:
    def __init__(self):
        self.robots_anim: List[RobotAnim] = []
        self.running_sim = False

    def start(self):
        # send request to backend
        try:
            requests.post(f"{BACKEND_URL}/api/run/aco", json={
                "tasks": tasks_data,
                "dependencies": deps_data,
                "robots": robots_data,
                "obstacles": obstacles_data,
                "map_size": list(WINDOW_SIZE),
                "n_iter": 200,
                "snapshot_interval": 20
            })
            print("[Visualizer] Requested ACO run...")
        except Exception as e:
            print("Failed to request ACO:", e)

    def handle_result(self, data):
        history = data["history"]
        final_snap = history[-1]
        sol: Dict[str, Any] = final_snap["best_solution"]
        self.robots_anim = []
        for ridx, task_ids in sol.items():
            ridx = int(ridx)  # keys might be strings from JSON
            color = PATH_COLOR[ridx % len(PATH_COLOR)]
            start_pos = (robots_data[ridx]["x"], robots_data[ridx]["y"])
            # build polyline via A* between each consecutive waypoint
            waypoints = [start_pos] + [(tasks_data[t]["x"], tasks_data[t]["y"]) for t in task_ids]
            full_path = []
            for a, b in zip(waypoints[:-1], waypoints[1:]):
                segment = compute_astar_path(a, b)
                if full_path:
                    full_path.extend(segment[1:])
                else:
                    full_path.extend(segment)
            path = full_path
            self.robots_anim.append(RobotAnim(path, color))
        self.running_sim = True
        self.final_cost = data.get("best_cost", 0)
        self.allocation_info = data.get("allocation", {})
        self.finish_times = data.get("finish_times", {})
        print("[Visualizer] Received ACO result, starting animation.")

    def update(self, dt):
        if not self.running_sim:
            return
        for ra in self.robots_anim:
            ra.update(dt)

    def draw(self, surf):
        surf.fill((240, 240, 240))
        # draw dependencies as arrows
        arrow_color = (150, 150, 150)
        for tgt, prereqs in deps_data.items():
            tx, ty = tasks_data[tgt]["x"], tasks_data[tgt]["y"]
            for src in prereqs:
                sx, sy = tasks_data[src]["x"], tasks_data[src]["y"]
                # compute direction and shorten line by TASK_RADIUS
                v = (tx - sx, ty - sy)
                L = math.hypot(*v)
                if L < 1e-3:
                    continue
                ux, uy = v[0]/L, v[1]/L
                end_x = tx - ux * TASK_RADIUS
                end_y = ty - uy * TASK_RADIUS
                start_x = sx + ux * TASK_RADIUS
                start_y = sy + uy * TASK_RADIUS
                pygame.draw.line(surf, arrow_color, (start_x, start_y), (end_x, end_y), 1)
                # arrow head
                left = (end_x - 10*ux + 5*uy, end_y - 10*uy - 5*ux)
                right = (end_x - 10*ux - 5*uy, end_y - 10*uy + 5*ux)
                pygame.draw.polygon(surf, arrow_color, [(end_x, end_y), left, right])
        # draw tasks
        for t in tasks_data:
            x, y = t["x"], t["y"]
            pygame.draw.circle(surf, (0, 120, 0), (int(x), int(y)), TASK_RADIUS)
            label = FONT.render(str(t["id"]), True, (255, 255, 255))
            surf.blit(label, (x - 6, y - 6))
        # draw robots anim
        for ra in self.robots_anim:
            ra.draw(surf)
        # draw obstacles
        for ox, oy, w, h in obstacles_data:
            pygame.draw.rect(surf, OBSTACLE_COLOR, (ox, oy, w, h))
        # HUD
        hud1 = FONT.render("SPACE=start  R=reset env", True, (0, 0, 0))
        surf.blit(hud1, (10, WINDOW_SIZE[1]-40))
        # dependencies text
        dep_str = ", ".join([f"{t}:{deps_data[t]}" for t in sorted(deps_data.keys())])
        hud2 = FONT.render(f"Deps: {dep_str}", True, (0, 0, 0))
        surf.blit(hud2, (10, WINDOW_SIZE[1]-20))
        # Allocation panel (grouped by robot)
        if hasattr(self, "allocation_info") and hasattr(self, "finish_times"):
            panel_x = WINDOW_SIZE[0] - 260
            y_off = 10
            surf.blit(FONT.render("Allocation (R -> tasks, finish)", True, (0, 0, 0)), (panel_x, y_off))
            y_off += 18
            for rob_id in sorted(self.allocation_info, key=lambda r: int(r)):
                task_list = self.allocation_info[rob_id]
                surf.blit(FONT.render(f"R{rob_id}:", True, (0, 0, 200)), (panel_x, y_off))
                y_off += 16
                for t in task_list:
                    ft = self.finish_times.get(str(t), self.finish_times.get(t, 0))
                    txt_surf = FONT.render(f"  T{t}  t={ft:.1f}", True, (0, 0, 0))
                    surf.blit(txt_surf, (panel_x + 20, y_off))
                    y_off += 14


viz = ACOVisualizer()

# ---------------- Socket.IO events -----------------

@sio.event
def connect():
    print("Socket connected")

@sio.on("aco_result")
def on_aco_result(data):
    viz.handle_result(data)

@sio.event
def disconnect():
    print("Socket disconnected")


# ---------------------------------------------------
# Main loop
# ---------------------------------------------------

def main():
    try:
        sio.connect(BACKEND_URL)
    except Exception as e:
        print("Failed to connect to backend:", e)
        sys.exit(1)

    running = True
    last_time = time.time()
    while running:
        now = time.time()
        dt = now - last_time
        last_time = now
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_SPACE:
                    viz.start()
                elif ev.key == pygame.K_r:
                    regenerate_environment()
                    viz.robots_anim.clear()
                    if hasattr(viz, "final_cost"):
                        delattr(viz, "final_cost")
        viz.update(dt)
        viz.draw(screen)
        pygame.display.flip()
        clock.tick(FPS)

    sio.disconnect()
    pygame.quit()
    sys.exit()


def regenerate_environment():
    global tasks_data, deps_data, robots_data, obstacles_data
    # random non-overlapping obstacles rectangles
    obstacles_data = []
    attempts = 0
    while len(obstacles_data) < 6 and attempts < 500:
        attempts += 1
        w = random.randint(80, 150)
        h = random.randint(60, 120)
        x = random.randint(150, WINDOW_SIZE[0] - w - 150)
        y = random.randint(150, WINDOW_SIZE[1] - h - 150)
        rect = pygame.Rect(x, y, w, h)
        if any(rect.colliderect(pygame.Rect(ox, oy, ow, oh)) for ox, oy, ow, oh in obstacles_data):
            continue
        obstacles_data.append((x, y, w, h))

    # generate tasks outside obstacles
    tasks_data = []
    attempts = 0
    while len(tasks_data) < 15 and attempts < 1000:
        attempts += 1
        x = random.randint(50, WINDOW_SIZE[0]-50)
        y = random.randint(50, WINDOW_SIZE[1]-50)
        # check distance to obstacles and tasks
        min_dist = 40  # pixels
        if any(pygame.Rect(ox - min_dist, oy - min_dist, w + 2*min_dist, h + 2*min_dist).collidepoint(x, y) for ox, oy, w, h in obstacles_data):
            continue
        if any(math.hypot(x - t["x"], y - t["y"]) < min_dist for t in tasks_data):
            continue
        tasks_data.append({"id": len(tasks_data), "x": x, "y": y})

    # robots fixed corners
    robots_data = [
        {"id": 0, "x": 50, "y": 50},
        {"id": 1, "x": WINDOW_SIZE[0]-50, "y": 60},
        {"id": 2, "x": 60, "y": WINDOW_SIZE[1]-60},
    ]

    # random dependency DAG
    deps_data = {}
    for t in tasks_data:
        tid = t["id"]
        # with small probability add up to 2 prerequisites with smaller id
        prereq_candidates = list(range(tid))
        random.shuffle(prereq_candidates)
        k = random.randint(0, min(2, len(prereq_candidates)))
        if k:
            deps_data[tid] = prereq_candidates[:k]


# initialize first environment
regenerate_environment()


if __name__ == "__main__":
    main() 