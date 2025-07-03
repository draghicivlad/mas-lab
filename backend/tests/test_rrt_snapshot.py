import math
from shapely.geometry import Polygon, LineString
from backend.algorithms.rrt_star import RRTStar


def segment_valid(p1, p2, obstacles):
    seg = LineString([p1, p2])
    return all(not obs.intersects(seg) for obs in obstacles)


def test_rrt_snapshot_integrity():
    """Ensure snapshots emitted by RRT* respect tree properties and obstacle constraints."""
    bounds = ((0, 0), (400, 300))
    start, goal = (10, 10), (390, 290)
    obstacles = [Polygon([(150, 0), (250, 0), (250, 250), (150, 250)])]

    planner = RRTStar(
        start=start,
        goal=goal,
        bounds=bounds,
        obstacles=obstacles,
        step_size=15,
        max_iter=800,
        goal_radius=20,
        seed=42,
        snapshot_interval=50,
    )

    for snap in planner.run_iter():
        nodes = snap["nodes"]  # list of (x, y, parentIdx)
        # 1. unique identity check (no duplicates in list)
        assert len(nodes) == len(set(map(id, nodes)))
        # 2. each node (except root) has exactly one valid parent within bounds
        for idx, n in enumerate(nodes):
            parent_idx = n[2]
            if parent_idx is None:
                assert idx == 0  # only root has no parent
            else:
                assert 0 <= parent_idx < len(nodes)
        # 3. edge validity w.r.t obstacles
        for n in nodes[1:]:
            parent = nodes[n[2]]
            p1 = (n[0], n[1])
            p2 = (parent[0], parent[1])
            assert segment_valid(p1, p2, obstacles)

    # Path produced should start near start and end near goal
    final_path = planner.best_path
    assert math.hypot(final_path[0][0]-start[0], final_path[0][1]-start[1]) < 1e-3
    assert math.hypot(final_path[-1][0]-goal[0], final_path[-1][1]-goal[1]) <= planner.goal_radius 