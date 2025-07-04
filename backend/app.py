"""Flask backend serving RRT* snapshots via SocketIO.

This is the solved instructor version. It exposes minimal HTTP endpoints to start
algorithms and streams incremental snapshots through WebSocket events using
Flask-SocketIO (which falls back to polling if WebSocket is unavailable).
"""
from __future__ import annotations

from threading import Lock
from typing import Dict, Any

from flask import Flask, request, jsonify
from flask_socketio import SocketIO
from flask_cors import CORS
from shapely.geometry import Polygon

from algorithms.rrt_star import RRTStar

app = Flask(__name__)
app.config["SECRET_KEY"] = "change-me"  # in production override via env
socketio = SocketIO(app, cors_allowed_origins="*")

# Enable CORS for /api/* endpoints so that frontend localhost:5173 can POST
CORS(app, resources={r"/api/*": {"origins": "*"}})

# keep track of running planners (id -> generator)
_planners: Dict[str, Any] = {}
_planners_lock = Lock()


def _make_obstacles(obs_raw):
    """Create Shapely polygons from list of rectangles.

    Each rectangle: {"x": , "y": , "w": , "h": }
    Assumes axis aligned with bottom-left origin.
    """
    obstacles = []
    for rect in obs_raw:
        x, y, w, h = rect["x"], rect["y"], rect["w"], rect["h"]
        obstacles.append(Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)]).buffer(0))
    return obstacles


@app.route("/api/run/rrtstar", methods=["POST"])
def run_rrtstar():
    data = request.get_json(force=True)

    # parse world configuration
    start = tuple(data["start"])  # [x, y]
    goal = tuple(data["goal"])
    bounds = tuple(map(tuple, data["bounds"]))  # [[xmin,ymin], [xmax,ymax]]
    obstacles = _make_obstacles(data.get("obstacles", []))

    planner = RRTStar(
        start=start,
        goal=goal,
        bounds=bounds,
        obstacles=obstacles,
        step_size=data.get("step_size", 20.0),
        goal_radius=data.get("goal_radius", 25.0),
        max_iter=data.get("max_iter", 2000),
        snapshot_interval=data.get("snapshot_interval", 1),
        min_sample_dist=10,
        seed=data.get("seed"),
    )

    run_id = data.get("run_id", f"rrt_{id(planner)}")

    def _background_task():
        history = []
        iter_found = None
        for delta in planner.run_iter():
            history.append(delta)
            if delta.get("goal_reached") and iter_found is None:
                iter_found = delta["iteration"]
                # stop planning once first valid path found
                break

        # Emit entire history at once (deltas only)
        socketio.emit("rrt_history", {"run_id": run_id, "history": history})

        final_path = planner.best_path
        socketio.emit("rrt_done", {"run_id": run_id, "path": final_path, "iterations": iter_found})

    socketio.start_background_task(_background_task)

    with _planners_lock:
        _planners[run_id] = planner

    return jsonify({"status": "started", "run_id": run_id})


@app.route("/api/runs", methods=["GET"])
def list_runs():
    with _planners_lock:
        return jsonify(list(_planners.keys()))


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True) 