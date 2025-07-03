"""RRT* implementation for 2-D rectangular world with axis-aligned rectangular obstacles.

This version is fully resolved (for instructor/demo). Students will later receive a version
with TODO markers replacing the core methods.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import List, Tuple, Iterable, Optional

import numpy as np
from shapely.geometry import Polygon, LineString, Point


@dataclass
class Node:
    x: float
    y: float
    parent: Optional[int]  # index in tree list
    cost: float = 0.0

    def as_tuple(self) -> Tuple[float, float]:
        return self.x, self.y


@dataclass
class RRTStar:
    start: Tuple[float, float]
    goal: Tuple[float, float]
    bounds: Tuple[Tuple[float, float], Tuple[float, float]]  # ((xmin, ymin), (xmax, ymax))
    obstacles: List[Polygon]
    step_size: float = 20.0
    goal_radius: float = 25.0
    max_iter: int = 5000
    gamma: float = 40.0  # radius factor for neighbor search
    snapshot_interval: int = 20
    seed: Optional[int] = None

    # internal state
    tree: List[Node] = field(init=False, default_factory=list)
    best_goal_idx: Optional[int] = field(init=False, default=None)

    def __post_init__(self):
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
        self.tree.append(Node(self.start[0], self.start[1], parent=None, cost=0.0))

    # ---------------------------------------------------------------------
    # Utility geometry helpers
    # ---------------------------------------------------------------------
    def _in_obstacle(self, point: Tuple[float, float]) -> bool:
        p = Point(point)
        # intersect (includes boundary) not just contains
        return any(ob.intersects(p) for ob in self.obstacles)

    def _segment_collision(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> bool:
        line = LineString([p1, p2])
        return any(ob.intersects(line) for ob in self.obstacles)

    # ---------------------------------------------------------------------
    # Core RRT* components
    # ---------------------------------------------------------------------
    def sample_free(self) -> Tuple[float, float]:
        """Uniform random sample in free space."""
        (xmin, ymin), (xmax, ymax) = self.bounds
        while True:
            x = random.uniform(xmin, xmax)
            y = random.uniform(ymin, ymax)
            if not self._in_obstacle((x, y)):
                return x, y

    def nearest(self, q_rand: Tuple[float, float]) -> int:
        """Return index of nearest node in tree to q_rand."""
        distances = [np.hypot(n.x - q_rand[0], n.y - q_rand[1]) for n in self.tree]
        return int(np.argmin(distances))

    def steer(self, q_near: Tuple[float, float], q_rand: Tuple[float, float]) -> Tuple[float, float]:
        """Move from q_near towards q_rand by at most step_size."""
        dx = q_rand[0] - q_near[0]
        dy = q_rand[1] - q_near[1]
        dist = np.hypot(dx, dy)
        if dist <= self.step_size:
            return q_rand
        else:
            theta = np.arctan2(dy, dx)
            return q_near[0] + self.step_size * np.cos(theta), q_near[1] + self.step_size * np.sin(theta)

    def near(self, q_new: Tuple[float, float], radius: float) -> List[int]:
        """Return indices of nodes within radius of q_new."""
        indices = []
        for idx, n in enumerate(self.tree):
            if np.hypot(n.x - q_new[0], n.y - q_new[1]) <= radius:
                indices.append(idx)
        return indices

    def obstacle_free(self, q1: Tuple[float, float], q2: Tuple[float, float]) -> bool:
        return not self._segment_collision(q1, q2)

    # ---------------------------------------------------------------------
    def rewire(self, q_new_idx: int, neighbor_indices: List[int]):
        q_new = self.tree[q_new_idx]
        for idx in neighbor_indices:
            neighbor = self.tree[idx]
            if idx == q_new.parent:
                continue
            if not self.obstacle_free(q_new.as_tuple(), neighbor.as_tuple()):
                continue
            new_cost = q_new.cost + np.hypot(neighbor.x - q_new.x, neighbor.y - q_new.y)
            if new_cost < neighbor.cost:
                neighbor.parent = q_new_idx
                neighbor.cost = new_cost

    # ---------------------------------------------------------------------
    def _reconstruct_path(self, goal_idx: int) -> List[Tuple[float, float]]:
        path = []
        idx = goal_idx
        while idx is not None:
            n = self.tree[idx]
            path.append((n.x, n.y))
            idx = n.parent
        return path[::-1]

    # ---------------------------------------------------------------------
    def run_iter(self) -> Iterable[dict]:
        """Generator yielding snapshots during planning."""
        goal_found = False
        for k in range(1, self.max_iter + 1):
            q_rand = self.sample_free()
            nearest_idx = self.nearest(q_rand)
            q_near = self.tree[nearest_idx]
            q_new_point = self.steer(q_near.as_tuple(), q_rand)
            if not self.obstacle_free(q_near.as_tuple(), q_new_point):
                continue

            # choose parent with minimal cost within radius
            radius = min(self.gamma * (np.log(len(self.tree)) / len(self.tree)) ** 0.5, self.step_size * 5)
            neighbor_indices = self.near(q_new_point, radius)
            min_cost = q_near.cost + np.hypot(q_new_point[0] - q_near.x, q_new_point[1] - q_near.y)
            min_parent = nearest_idx
            for idx in neighbor_indices:
                neighbor = self.tree[idx]
                if not self.obstacle_free(neighbor.as_tuple(), q_new_point):
                    continue
                cost = neighbor.cost + np.hypot(q_new_point[0] - neighbor.x, q_new_point[1] - neighbor.y)
                if cost < min_cost:
                    min_cost = cost
                    min_parent = idx
            # add new node
            new_idx = len(self.tree)
            self.tree.append(Node(q_new_point[0], q_new_point[1], parent=min_parent, cost=min_cost))

            # rewire neighbors
            self.rewire(new_idx, neighbor_indices)

            # check goal
            if np.hypot(q_new_point[0] - self.goal[0], q_new_point[1] - self.goal[1]) <= self.goal_radius:
                goal_found = True
                if self.best_goal_idx is None or self.tree[new_idx].cost < self.tree[self.best_goal_idx].cost:
                    self.best_goal_idx = new_idx

            # snapshot
            if k % self.snapshot_interval == 0 or k == self.max_iter or (goal_found and self.best_goal_idx == new_idx):
                snapshot = {
                    "iteration": k,
                    "nodes": [(n.x, n.y, n.parent) for n in self.tree],
                    "goal_reached": self.best_goal_idx is not None,
                    "path": self._reconstruct_path(self.best_goal_idx) if self.best_goal_idx is not None else None,
                }
                yield snapshot
            if goal_found and k >= self.max_iter:
                break

    # property for compatibility
    @property
    def best_path(self) -> List[Tuple[float, float]]:
        if self.best_goal_idx is None:
            return []
        return self._reconstruct_path(self.best_goal_idx) 