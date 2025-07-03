"""Conflict-Based Search (CBS) implementation for Multi-Agent Path Finding on a 2-D grid.

This version is complete (instructor / demo). Students will later receive a
variant with TODO markers in core methods.

The solver works with a rectangular grid world. Obstacles are axis-aligned
rectangles converted to blocked cells. Each agent is a point occupying a grid
cell. Time is discretised in unit steps. An agent can move in 4-neighbourhood
(N,E,S,W) or stay. A conflict occurs if two agents occupy the same cell in the
same timestep (vertex conflict) or they swap cells in consecutive timesteps
(edge conflict).

Returned paths are lists of grid coordinates (x,y) for every timestep until all
agents reach their goals (agents then wait in place).
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Iterable, Set

import math
import itertools


# ---------------------------------------------------------------------------
# Grid representation helpers
# ---------------------------------------------------------------------------
Coord = Tuple[int, int]
MOVE_DIRS: List[Coord] = [(1, 0), (-1, 0), (0, 1), (0, -1), (0, 0)]  # 4-neigh + wait


class GridWorld:
    def __init__(self, width: int, height: int, blocked: Set[Coord]):
        self.W = width
        self.H = height
        self.blocked = blocked

    def in_bounds(self, p: Coord) -> bool:
        x, y = p
        return 0 <= x < self.W and 0 <= y < self.H

    def traversable(self, p: Coord) -> bool:
        return self.in_bounds(p) and p not in self.blocked

    def neighbors(self, p: Coord) -> List[Coord]:
        res = []
        for dx, dy in MOVE_DIRS:
            q = (p[0] + dx, p[1] + dy)
            if self.traversable(q):
                res.append(q)
        return res


# ---------------------------------------------------------------------------
# Low-level search: constraint-aware A*
# ---------------------------------------------------------------------------

Constraint = Tuple[str, Coord, int]  # (type, coord, t) type="v" vertex or "e" edge from->to encoded later
EdgeConstraint = Tuple[str, Coord, Coord, int]  # ("e", from, to, t)


def heuristic(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def build_constraint_table(constraints: List[Constraint | EdgeConstraint], agent_id: int):
    vertex: Dict[int, Set[Coord]] = {}
    edge: Dict[int, Set[Tuple[Coord, Coord]]] = {}
    for c in constraints:
        if c[0] == "v":
            _, coord, t = c
            vertex.setdefault(t, set()).add(coord)
        else:  # edge
            _, from_c, to_c, t = c  # type: ignore
            edge.setdefault(t, set()).add((from_c, to_c))
    return vertex, edge


def is_constrained(curr: Coord, nxt: Coord, t: int, vtab, etab):
    # vertex constraint for next position at t+1
    if (t + 1) in vtab and nxt in vtab[t + 1]:
        return True
    # edge constraint between curr->nxt during [t,t+1]
    if (t + 1) in etab and (curr, nxt) in etab[t + 1]:
        return True
    return False


def a_star_constrained(world: GridWorld, start: Coord, goal: Coord, constraints, max_t=256) -> Optional[List[Coord]]:
    vtab, etab = build_constraint_table(constraints, 0)
    start_state = (heuristic(start, goal), 0, start, 0, [start])
    open_heap = [start_state]  # (f, g, coord, t, path)
    visited = {}
    while open_heap:
        f, g, curr, t, path = heapq.heappop(open_heap)
        if (curr, t) in visited and visited[(curr, t)] <= g:
            continue
        visited[(curr, t)] = g

        if curr == goal and t >= max_t // 4:
            # found; extend path by waiting until max(len(path), horizon)
            return path
        if t > max_t:
            continue
        for nxt in world.neighbors(curr):
            if is_constrained(curr, nxt, t, vtab, etab):
                continue
            g2 = g + 1
            h2 = heuristic(nxt, goal)
            f2 = g2 + h2
            path2 = path + [nxt]
            heapq.heappush(open_heap, (f2, g2, nxt, t + 1, path2))
        # wait move (nxt == curr) already included in neighbors due to (0,0)
    return None


# ---------------------------------------------------------------------------
# CBS high-level search
# ---------------------------------------------------------------------------

@dataclass(order=True)
class CBSNode:
    priority: int
    cost: int = field(compare=False)
    constraints: List[Constraint | EdgeConstraint] = field(compare=False, default_factory=list)
    paths: List[List[Coord]] = field(compare=False, default_factory=list)
    id_counter: int = field(compare=False, default=0)


class CBSSolver:
    def __init__(self, world: GridWorld, starts: List[Coord], goals: List[Coord]):
        assert len(starts) == len(goals)
        self.world = world
        self.n_agents = len(starts)
        self.starts = starts
        self.goals = goals

    # --------------------------------------------------------
    def detect_conflict(self, paths: List[List[Coord]]):
        max_len = max(len(p) for p in paths)
        for t in range(max_len):
            positions = {}
            for idx, path in enumerate(paths):
                pos = path[t] if t < len(path) else path[-1]
                # vertex conflict
                if pos in positions:
                    return {
                        "type": "vertex",
                        "time": t,
                        "agents": (positions[pos], idx),
                        "loc": pos,
                    }
                positions[pos] = idx
            # edge conflict
            for (i, p_i) in enumerate(paths):
                if t + 1 >= len(p_i):
                    pos_i_next = p_i[-1]
                else:
                    pos_i_next = p_i[t + 1]
                for j in range(i + 1, self.n_agents):
                    p_j = paths[j]
                    pos_j = p_j[t] if t < len(p_j) else p_j[-1]
                    pos_j_next = p_j[t + 1] if t + 1 < len(p_j) else p_j[-1]
                    if pos_i_next == pos_j and pos_j_next == pos_i_next:
                        return {
                            "type": "edge",
                            "time": t + 1,
                            "agents": (i, j),
                            "loc": (positions.get(pos_i_next, i), positions.get(pos_j_next, j)),
                            "edge": ((pos_i_next, pos_j_next),),
                        }
        return None

    # --------------------------------------------------------
    def initial_paths(self):
        paths = []
        for s, g in zip(self.starts, self.goals):
            p = a_star_constrained(self.world, s, g, [])
            if p is None:
                return None
            paths.append(p)
        return paths

    # --------------------------------------------------------
    def solve(self) -> Tuple[List[List[Coord]], List[dict]]:
        """Return (paths, snapshots) list if found; else ([], snapshots)."""
        snapshots = []
        root_paths = self.initial_paths()
        if root_paths is None:
            return [], snapshots
        root_cost = sum(len(p) for p in root_paths)
        open_heap: List[CBSNode] = []
        root_node = CBSNode(priority=root_cost, cost=root_cost, constraints=[], paths=root_paths)
        heapq.heappush(open_heap, root_node)
        iteration = 0
        while open_heap:
            node = heapq.heappop(open_heap)
            iteration += 1
            conflict = self.detect_conflict(node.paths)
            snapshots.append({
                "iteration": iteration,
                "paths": node.paths,
                "conflict": conflict,
            })
            if conflict is None:
                return node.paths, snapshots
            a1, a2 = conflict["agents"]
            if conflict["type"] == "vertex":
                loc = conflict["loc"]
                t = conflict["time"]
                constraints = [
                    ("v", loc, t),  # forbid agent at vertex at time
                ]
                for agent in (a1, a2):
                    new_constraints = node.constraints + constraints  # type: ignore
                    new_paths = node.paths.copy()
                    path = a_star_constrained(self.world, self.starts[agent], self.goals[agent], [c for c in new_constraints if c[0] == "v" or len(c) == 4 and agent in conflict["agents"]])
                    if path is None:
                        continue
                    new_paths[agent] = path
                    cost = sum(len(p) for p in new_paths)
                    heapq.heappush(open_heap, CBSNode(priority=cost, cost=cost, constraints=new_constraints, paths=new_paths))
            else:  # edge conflict
                t = conflict["time"]
                (p_i, p_j) = conflict["loc"]  # we didn't store edges exactly
                edge1 = (node.paths[a1][t - 1], node.paths[a1][t])
                edge2 = (node.paths[a2][t - 1], node.paths[a2][t])
                constraint1: EdgeConstraint = ("e", edge1[0], edge1[1], t)
                constraint2: EdgeConstraint = ("e", edge2[0], edge2[1], t)
                for agent, constr in [(a1, constraint1), (a2, constraint2)]:
                    new_constraints = node.constraints + [constr]
                    new_paths = node.paths.copy()
                    path = a_star_constrained(self.world, self.starts[agent], self.goals[agent], [c for c in new_constraints if c[0] == "v" or len(c) == 4 and agent == agent])
                    if path is None:
                        continue
                    new_paths[agent] = path
                    cost = sum(len(p) for p in new_paths)
                    heapq.heappush(open_heap, CBSNode(priority=cost, cost=cost, constraints=new_constraints, paths=new_paths))
        return [], snapshots

    # --------------------------------------------------------
    def run_iter(self) -> Iterable[dict]:
        paths, snapshots = self.solve()
        for snap in snapshots:
            yield snap
        # final snapshot (done) is already last snapshot with conflict None


# ---------------------------------------------------------------------------
# Helper to build GridWorld from bounding box + obstacles
# ---------------------------------------------------------------------------

def build_grid_from_rects(bounds: Tuple[Tuple[float, float], Tuple[float, float]], obstacles_rects: List[dict], cell_size: int):
    (xmin, ymin), (xmax, ymax) = bounds
    width = math.ceil((xmax - xmin) / cell_size)
    height = math.ceil((ymax - ymin) / cell_size)
    blocked: Set[Coord] = set()
    for rect in obstacles_rects:
        rx, ry, rw, rh = rect["x"], rect["y"], rect["w"], rect["h"]
        # convert to cell coords
        for i in range(width):
            for j in range(height):
                cx = xmin + i * cell_size + cell_size / 2
                cy = ymin + j * cell_size + cell_size / 2
                if rx <= cx <= rx + rw and ry <= cy <= ry + rh:
                    blocked.add((i, j))
    return GridWorld(width, height, blocked) 