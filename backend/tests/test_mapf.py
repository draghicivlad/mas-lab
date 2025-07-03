from backend.algorithms.mapf import build_grid_from_rects, CBSSolver


def test_cbs_two_agents():
    bounds = ((0, 0), (100, 100))
    obstacles = []
    cell_size = 10
    world = build_grid_from_rects(bounds, obstacles, cell_size)
    starts = [(0, 0), (9, 0)]  # cell coords because world width ~10
    goals = [(9, 9), (0, 9)]
    solver = CBSSolver(world, starts, goals)
    paths = None
    for snap in solver.run_iter():
        if snap["conflict"] is None:
            paths = snap["paths"]
    assert paths is not None
    # Paths length moderate
    assert len(paths) == 2
    # check final positions
    assert paths[0][-1] == goals[0]
    assert paths[1][-1] == goals[1]
    # ensure no vertex conflicts
    max_len = max(len(p) for p in paths)
    for t in range(max_len):
        occupied = set()
        for path in paths:
            pos = path[t] if t < len(path) else path[-1]
            assert pos not in occupied
            occupied.add(pos) 