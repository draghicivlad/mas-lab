from __future__ import annotations

"""Ant Colony Optimization for multi-robot task allocation with task dependencies.

This implementation is intentionally compact and documented so that students can
read and extend it.  The core ACO routine is provided, but there are several
"TODO" markers where students are encouraged to experiment with parameters or
alternative cost functions.

Problem model
-------------
* There are `n_tasks` tasks located at given 2-D coordinates.
* There are `m` robots, each starting at its own (x, y) start position.
* A directed acyclic graph (DAG) defines *dependencies* between tasks – a task
  can only start once all predecessors have finished.
* Every task takes a fixed *service time* (default 0) once the robot arrives.

A *solution* is an assignment of every task to a robot **and** an order for the
robot to perform its tasks such that all dependencies are respected.  The cost
of a solution is the *makespan* (total time until the last robot finishes).

ACO overview (very much simplified):
1. For each iteration, every *ant* incrementally builds a feasible schedule.
2. A schedule is built by repeatedly selecting one of the *currently available*
   tasks (all predecessors finished) for some robot.
3. Selection probability is guided by a *pheromone* value τ and a heuristic η
   (here: inverse added travel distance).
4. At the end of the iteration pheromone trails evaporate and the best ant of
   the iteration deposits additional pheromone ∆τ proportional to solution
   quality.
5. The best global schedule found across all iterations is returned.

Limitations / Simplifications
----------------------------
* Robots travel with unit speed and Euclidean distance is used as travel time.
* Waiting due to dependencies is ignored in cost (students can extend this).
* Pheromone is stored *per task* (not per (task,robot) pair) which already
  biases ants towards good *ordering* but not necessarily robot-task affinity.
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import random
import math
import numpy as np


Position = Tuple[float, float]
TaskID = int
RobotID = int


def euclid(a: Position, b: Position) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


@dataclass
class ACOTaskAllocation:
    task_positions: List[Position]
    dependencies: Dict[TaskID, List[TaskID]]  # DAG
    robot_starts: List[Position]
    # New params
    obstacles: List[Tuple[float, float, float, float]] | None = None  # list of rects (x,y,w,h)
    map_size: Tuple[int, int] | None = None  # (width, height)
    cell_size: int = 20

    # --- ACO hyper-parameters (can be tweaked by students) ---
    n_ants: int = 40
    n_iter: int = 200
    alpha: float = 1.0  # pheromone influence
    beta: float = 2.0   # heuristic influence
    rho: float = 0.1    # evaporation rate
    q: float = 100.0    # pheromone deposit factor
    seed: Optional[int] = None

    def __post_init__(self):
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        self.n_tasks = len(self.task_positions)
        self.n_robots = len(self.robot_starts)

        # Make sure all tasks exist in dependency dict
        for t in range(self.n_tasks):
            self.dependencies.setdefault(t, [])

        if self.obstacles and self.map_size:
            self._compute_distance_with_astar()
        else:
            self._compute_euclidean_distances()

        # Initialise pheromone τ_i  (same for every robot)
        self.tau = np.ones(self.n_tasks)

        # Keep track of best overall solution
        self.best_cost: float = float("inf")
        self.best_solution: Dict[RobotID, List[TaskID]] = {}

    # ------------------------------------------------------------------
    def _heuristic(self, robot: RobotID, task: TaskID, last_task: Optional[TaskID]) -> float:
        """η (eta) – heuristic desirability for assigning *task* next for *robot*.

        If robot hasn't done any task yet we use distance from start; otherwise
        distance from last_task.
        """
        if last_task is None:
            d = self._dist_robot_task[robot, task]
        else:
            d = self._dist_task_task[last_task, task]
        # Avoid division by zero – add small epsilon
        return 1.0 / (d + 1e-6)

    # ------------------------------------------------------------------
    def _build_schedule(self) -> Tuple[Dict[RobotID, List[TaskID]], float]:
        """Construct one feasible schedule and return (solution, cost)."""
        # state trackers
        remaining = set(range(self.n_tasks))
        completed: set[TaskID] = set()
        solution: Dict[RobotID, List[TaskID]] = {r: [] for r in range(self.n_robots)}
        last_task: Dict[RobotID, Optional[TaskID]] = {r: None for r in range(self.n_robots)}

        while remaining:
            # Build list of tasks whose dependencies are satisfied
            available = [t for t in remaining if all(dep in completed for dep in self.dependencies[t])]
            if not available:
                # Should not happen if dependencies form a DAG
                raise ValueError("Cyclic dependencies detected or bad state")

            # For each available (task, robot) compute selection probability
            probs = []
            pairs = []  # (robot, task)
            for task in available:
                tau_t = self.tau[task] ** self.alpha
                for r in range(self.n_robots):
                    eta = self._heuristic(r, task, last_task[r]) ** self.beta
                    pairs.append((r, task))
                    probs.append(tau_t * eta)

            # Remove zero-weight pairs to avoid all-zero probabilities
            probs_nonzero = [(p, pair) for p, pair in zip(probs, pairs) if p > 0]
            if not probs_nonzero:
                # fallback: choose random robot & task ignoring distance
                r_sel = random.choice(range(self.n_robots))
                t_sel = random.choice(available)
            else:
                probs_arr = np.array([p for p, _ in probs_nonzero])
                probs_arr = probs_arr / probs_arr.sum()
                idx = np.random.choice(len(probs_arr), p=probs_arr)
                r_sel, t_sel = probs_nonzero[idx][1]

            # Assign task to the chosen robot
            solution[r_sel].append(t_sel)
            remaining.remove(t_sel)
            completed.add(t_sel)
            last_task[r_sel] = t_sel

        cost = self._schedule_cost(solution)
        return solution, cost

    # ------------------------------------------------------------------
    def _schedule_cost(self, sol: Dict[RobotID, List[TaskID]]) -> float:
        """Compute makespan while enforcing dependency finish times.

        Each robot travels with unit speed; completion time for task = arrival time.
        If any dependency is violated (a task finishes after a dependent task starts),
        return a huge penalty cost (1e9).
        """
        task_finish: Dict[int, float] = {}
        max_time = 0.0

        # For each robot compute cumulative travel time and finish times
        for r, tasks in sol.items():
            if not tasks:
                continue
            t_acc = self._dist_robot_task[r, tasks[0]]
            task_finish[tasks[0]] = t_acc
            for i in range(1, len(tasks)):
                prev = tasks[i-1]
                cur = tasks[i]
                t_acc += self._dist_task_task[prev, cur]
                task_finish[cur] = t_acc
            max_time = max(max_time, t_acc)

        # Validate dependencies
        for task, prereqs in self.dependencies.items():
            for pre in prereqs:
                if task not in task_finish or pre not in task_finish:
                    return 1e9  # should not happen
                if task_finish[pre] > task_finish[task]:
                    return 1e9  # violation

        # store for external inspection
        self.last_task_finish = task_finish
        return max_time

    # ------------------------------------------------------------------
    def compute_finish_times(self, sol: Dict[RobotID, List[TaskID]]) -> Dict[int, float]:
        """Public helper to compute per-task finish times (without penalty check)."""
        task_finish: Dict[int, float] = {}
        for r, tasks in sol.items():
            if not tasks:
                continue
            t_acc = self._dist_robot_task[r, tasks[0]]
            task_finish[tasks[0]] = t_acc
            for i in range(1, len(tasks)):
                prev = tasks[i-1]
                cur = tasks[i]
                t_acc += self._dist_task_task[prev, cur]
                task_finish[cur] = t_acc
        return task_finish

    # ------------------------------------------------------------------
    def run(self, snapshot_interval: int = 10):
        """Iterate ACO and yield snapshot dicts every `snapshot_interval` iterations."""
        for it in range(1, self.n_iter + 1):
            best_it_cost = float("inf")
            best_it_solution = None

            # Construct solutions with each ant
            for _ in range(self.n_ants):
                sol, cost = self._build_schedule()
                if cost < best_it_cost:
                    best_it_cost = cost
                    best_it_solution = sol

            # Pheromone evaporation
            self.tau *= (1 - self.rho)
            # Deposit pheromone from best ant of iteration
            for tasks in best_it_solution.values():
                for t in tasks:
                    self.tau[t] += self.q / best_it_cost

            # Update global best
            if best_it_cost < self.best_cost:
                self.best_cost = best_it_cost
                self.best_solution = best_it_solution

            if it % snapshot_interval == 0:
                yield {
                    "iteration": it,
                    "best_cost": self.best_cost,
                    "best_solution": self.best_solution,
                    "tau": self.tau.copy().tolist(),
                }

        # final snapshot
        yield {
            "iteration": self.n_iter,
            "best_cost": self.best_cost,
            "best_solution": self.best_solution,
            "tau": self.tau.copy().tolist(),
            "done": True,
        }

    # ------------------------------------------------------------------
    def _compute_euclidean_distances(self):
        """Fallback distance metric without obstacles."""
        self._dist_task_task = np.zeros((self.n_tasks, self.n_tasks))
        for i in range(self.n_tasks):
            for j in range(self.n_tasks):
                self._dist_task_task[i, j] = euclid(self.task_positions[i], self.task_positions[j])
        self._dist_robot_task = np.zeros((self.n_robots, self.n_tasks))
        for r in range(self.n_robots):
            for t in range(self.n_tasks):
                self._dist_robot_task[r, t] = euclid(self.robot_starts[r], self.task_positions[t])

    # ------------------------------------------------------------------
    def _compute_distance_with_astar(self):
        from algorithms.grid_astar import AStarGrid

        width, height = self.map_size
        cols = width // self.cell_size
        rows = height // self.cell_size
        # Build occupancy grid
        occ = [[False for _ in range(cols)] for _ in range(rows)]
        for (x, y, w, h) in self.obstacles:
            # mark cells covered by obstacle as True
            x0 = int(x // self.cell_size)
            y0 = int(y // self.cell_size)
            x1 = int((x + w) // self.cell_size)
            y1 = int((y + h) // self.cell_size)
            for cy in range(y0, min(rows, y1 + 1)):
                for cx in range(x0, min(cols, x1 + 1)):
                    occ[cy][cx] = True

        astar = AStarGrid(occ)

        def to_grid(pos: Position) -> Tuple[int, int]:
            return int(pos[0] // self.cell_size), int(pos[1] // self.cell_size)

        self._dist_task_task = np.zeros((self.n_tasks, self.n_tasks))
        for i in range(self.n_tasks):
            for j in range(self.n_tasks):
                if i == j:
                    self._dist_task_task[i, j] = 0.0
                    continue
                start = to_grid(self.task_positions[i])
                goal = to_grid(self.task_positions[j])
                steps = astar.path_length(start, goal)
                if steps is None:
                    self._dist_task_task[i, j] = 1e9
                else:
                    self._dist_task_task[i, j] = steps * self.cell_size

        self._dist_robot_task = np.zeros((self.n_robots, self.n_tasks))
        for r in range(self.n_robots):
            for t in range(self.n_tasks):
                start = to_grid(self.robot_starts[r])
                goal = to_grid(self.task_positions[t])
                steps = astar.path_length(start, goal)
                self._dist_robot_task[r, t] = 1e9 if steps is None else steps * self.cell_size 