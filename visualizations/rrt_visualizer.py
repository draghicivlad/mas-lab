import sys
import math
import time
from typing import List, Tuple, Dict, Any

import pygame
import socketio
import requests

# ----------------------------- CONFIG ---------------------------------
BACKEND_URL = 'http://localhost:5000'  # change if backend on different host
WINDOW_SIZE = (900, 700)
FPS = 30

# keybindings help string
HELP_TEXT = "SPACE play/pause | →/← step | +/- speed | R start RRT*"

# ----------------------------- PYGAME SETUP ---------------------------
pygame.init()
screen = pygame.display.set_mode(WINDOW_SIZE)
pygame.display.set_caption('MAS-Lab Visualizer')
clock = pygame.time.Clock()
FONT = pygame.font.SysFont('arial', 16)

# ----------------------------- SOCKET.IO ------------------------------
sio = socketio.Client()

# ----------------------------- STATE ----------------------------------
class RRTVisualizer:
    def __init__(self):
        self.history: List[Dict[str, Any]] = []
        self.frame_idx: int = 0
        self.playing: bool = False
        self.last_advance: float = time.time()
        self.play_speed: float = 0.05  # seconds per frame

        # default config mirrors web
        self.start = (20, 20)
        self.goal = (780, 580)
        self.bounds = ((0, 0), (800, 600))
        self.obstacles = [
            pygame.Rect(300, 0, 50, 400),
            pygame.Rect(500, 200, 60, 400),
            pygame.Rect(100, 300, 150, 40),
            pygame.Rect(650, 0, 40, 250),
        ]

    # ------------------------------------------------------------------
    def handle_history(self, data):
        deltas = data['history']
        tree: List[Any] = []
        frames = []
        for d in deltas:
            idx, x, y, parent = d['new']
            # ensure list size
            if idx >= len(tree):
                tree.extend([None]*(idx - len(tree) + 1))
            tree[idx] = [x, y, parent]
            for pair in d.get('rewired', []):
                child, new_parent = pair
                if child < len(tree) and tree[child] is not None:
                    tree[child][2] = new_parent
            # make deep-ish copy (up to tuples)
            frames.append({
                'iteration': d['iteration'],
                'nodes': [n[:] if n is not None else None for n in tree],
                'path': d.get('path')
            })

        self.history = frames
        self.frame_idx = 0
        self.playing = False
        print(f"Received RRT history: {len(self.history)} frames")

    # ------------------------------------------------------------------
    def update(self):
        if self.playing and time.time() - self.last_advance >= self.play_speed:
            if self.frame_idx + 1 >= len(self.history)-1:
                self.frame_idx = len(self.history)-1
                self.playing = False  # stop at last frame
            else:
                self.frame_idx += 1
            self.last_advance = time.time()

    # ------------------------------------------------------------------
    def draw(self, surf):
        # clear
        surf.fill((250, 250, 250))

        # draw obstacles
        for ob in self.obstacles:
            pygame.draw.rect(surf, (100, 100, 100), ob)

        if not self.history:
            return
        snapshot = self.history[self.frame_idx]
        nodes = snapshot['nodes']

        # draw final path if present in snapshot
        if snapshot.get('path'):
            pts = snapshot['path']
            if pts:
                pygame.draw.lines(surf,(255,0,0),False,pts,3)

        # build unique edges (child-parent)
        edges = set()
        for idx, n in enumerate(nodes):
            if n is None:
                continue
            p = n[2]
            if p is None or p < 0:
                continue
            key = tuple(sorted((idx, p)))
            edges.add(key)

        # draw edges
        for (a, b) in edges:
            na = nodes[a]
            nb = nodes[b]
            if na is None or nb is None:
                continue
            pygame.draw.line(surf, (150, 150, 150), (na[0], na[1]), (nb[0], nb[1]), 1)

        # draw nodes (sample)
        step = max(1, len(nodes) // 1200)
        for i in range(0, len(nodes), step):
            n = nodes[i]
            if n is None:
                continue
            pygame.draw.circle(surf, (120, 120, 120), (int(n[0]), int(n[1])), 2)

        # start / goal
        pygame.draw.circle(surf, (0, 200, 0), self.start, 6)
        pygame.draw.circle(surf, (0, 0, 200), self.goal, 6)

        # frame counter
        txt = FONT.render(f"Frame {self.frame_idx+1}/{len(self.history)} | speed {self.play_speed:.2f}s", True, (0,0,0))
        surf.blit(txt, (10, 610))

# --------------------------------------------------------
# GLOBAL STATE
# --------------------------------------------------------

rrt_vis = RRTVisualizer()

# ----------------------------- SOCKET EVENTS --------------------------
@sio.event
def connect():
    print("Connected to backend")

@sio.on('rrt_history')
def on_rrt_history(data):
    rrt_vis.handle_history(data)

@sio.event
def disconnect():
    print("Disconnected from backend")

# ----------------------------- MAIN LOOP ------------------------------

def main():
    try:
        sio.connect(BACKEND_URL)
    except Exception as e:
        print("Failed to connect to backend:", e)
        sys.exit(1)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    rrt_vis.playing = not rrt_vis.playing
                elif event.key == pygame.K_RIGHT:
                    rrt_vis.frame_idx = min(rrt_vis.frame_idx + 1, max(0, len(rrt_vis.history) - 1))
                elif event.key == pygame.K_LEFT:
                    rrt_vis.frame_idx = max(rrt_vis.frame_idx - 1, 0)
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    rrt_vis.play_speed = max(0.01, rrt_vis.play_speed - 0.02)
                elif event.key == pygame.K_MINUS:
                    rrt_vis.play_speed = rrt_vis.play_speed + 0.02
                elif event.key == pygame.K_r:
                    # trigger RRT run via HTTP
                    try:
                        requests.post(f"{BACKEND_URL}/api/run/rrtstar", json={
                            "start": rrt_vis.start,
                            "goal": rrt_vis.goal,
                            "bounds": [list(rrt_vis.bounds[0]), list(rrt_vis.bounds[1])],
                            "obstacles": [{"x": ob.x, "y": ob.y, "w": ob.w, "h": ob.h} for ob in rrt_vis.obstacles],
                            "step_size": 15,
                            "max_iter": 3000,
                            "snapshot_interval": 10
                        })
                    except Exception as e:
                        print("Failed to start RRT:", e)

        rrt_vis.update()
        rrt_vis.draw(screen)

        # help text
        help_txt=FONT.render(HELP_TEXT,True,(0,0,0))
        screen.blit(help_txt,(10,650))

        pygame.display.flip()
        clock.tick(FPS)

    sio.disconnect()
    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main() 