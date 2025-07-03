import math
from shapely.geometry import Polygon, Point
from backend.algorithms.rrt_star import RRTStar


def test_rrt_star_simple_world():
    bounds = ((0, 0), (400, 300))
    start = (10, 10)
    goal = (390, 290)
    obstacles = [
        Polygon([(150, 0), (250, 0), (250, 250), (150, 250)]),
    ]
    planner = RRTStar(
        start=start,
        goal=goal,
        bounds=bounds,
        obstacles=obstacles,
        max_iter=1500,
        step_size=15,
        goal_radius=20,
        seed=42,
    )
    for _ in planner.run_iter():
        pass
    path = planner.best_path
    assert path, "Planner should find a path"
    # path should start and end correctly
    assert math.isclose(path[0][0], start[0], abs_tol=1e-6)
    assert math.isclose(path[0][1], start[1], abs_tol=1e-6)
    end = path[-1]
    assert math.hypot(end[0] - goal[0], end[1] - goal[1]) <= planner.goal_radius
    # ensure path segments are collision free
    for p1, p2 in zip(path[:-1], path[1:]):
        assert not any(obst.intersects(Point(p1).buffer(1).union(Point(p2).buffer(1))) for obst in obstacles) 