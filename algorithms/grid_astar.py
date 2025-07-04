from __future__ import annotations

"""Simple A* search on a 2-D occupancy grid (4-neighbour connectivity).

Grid coordinates are integer indices (col, row).  Each free cell has cost 1.
Returned path length is number of steps × cell_size (passed by caller).
"""

from typing import List, Tuple, Optional, Set, Dict
import heapq

Coord = Tuple[int, int]


class AStarGrid:
    def __init__(self, occupancy: List[List[bool]]):
        """occupancy[r][c] == True  => cell blocked"""
        self.occ = occupancy
        self.h = len(occupancy)
        self.w = len(occupancy[0]) if self.h else 0

    # --------------------------------------------------
    def in_bounds(self, p: Coord) -> bool:
        x, y = p
        return 0 <= x < self.w and 0 <= y < self.h

    def passable(self, p: Coord) -> bool:
        return not self.occ[p[1]][p[0]]

    def neighbours(self, p: Coord):
        x, y = p
        # 8 directions with costs
        dirs = [
            (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
            (1, 1, 1.41421356237), (-1, 1, 1.41421356237), (1, -1, 1.41421356237), (-1, -1, 1.41421356237)
        ]
        for dx, dy, cost in dirs:
            np = (x + dx, y + dy)
            if self.in_bounds(np) and self.passable(np):
                yield np, cost

    # --------------------------------------------------
    def heuristic(self, a: Coord, b: Coord) -> float:
        # Octile distance suitable for 8-neighbour grids
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        return (dx + dy) + (1.41421356237 - 2) * min(dx, dy)

    # --------------------------------------------------
    def path_length(self, start: Coord, goal: Coord) -> Optional[int]:
        if not (self.passable(start) and self.passable(goal)):
            return None
        frontier: List[Tuple[int, Coord]] = []
        heapq.heappush(frontier, (0, start))
        g_score: Dict[Coord, int] = {start: 0}
        came: Dict[Coord, Coord] = {}

        while frontier:
            _, current = heapq.heappop(frontier)
            if current == goal:
                return g_score[current]
            for nei, cost in self.neighbours(current):
                tentative = g_score[current] + cost
                if tentative < g_score.get(nei, 1_000_000):
                    g_score[nei] = tentative
                    came[nei] = current
                    f = tentative + self.heuristic(nei, goal)
                    heapq.heappush(frontier, (f, nei))
        return None  # no path

    # ------------------------------------------------------------------
    def shortest_path(self, start: Coord, goal: Coord) -> Optional[List[Coord]]:
        """Return list of cells from start to goal (inclusive) or None if unreachable."""
        if not (self.passable(start) and self.passable(goal)):
            return None
        frontier: List[Tuple[int, Coord]] = []
        heapq.heappush(frontier, (0, start))
        g_score: Dict[Coord, int] = {start: 0}
        came_from: Dict[Coord, Coord] = {}

        while frontier:
            _, current = heapq.heappop(frontier)
            if current == goal:
                # reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path
            for nei, cost in self.neighbours(current):
                tentative = g_score[current] + cost
                if tentative < g_score.get(nei, 1_000_000):
                    g_score[nei] = tentative
                    came_from[nei] = current
                    f = tentative + self.heuristic(nei, goal)
                    heapq.heappush(frontier, (f, nei))
        return None 