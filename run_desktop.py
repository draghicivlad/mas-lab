"""Launcher script to run backend server and pygame visualizer in one go.

Usage:
    python run_desktop.py

It spawns the backend in a child process, waits a couple seconds for it to
bind to the port, then starts the Pygame visualizer in the foreground. When
the visualizer window is closed, the backend process is terminated.
"""

import subprocess
import sys
import time
import os
import signal
from pathlib import Path

ROOT = Path(__file__).parent.resolve()

def main():
    # Spawn backend
    backend_cmd = [sys.executable, "-m", "backend.app"]
    print("Starting backend:", " ".join(backend_cmd))
    backend_proc = subprocess.Popen(backend_cmd, cwd=str(ROOT))

    try:
        # small delay to give backend time to start
        time.sleep(2)
        # Launch visualizer (blocking)
        viz_cmd = [sys.executable, str(ROOT / "visualizations" / "rrt_visualizer.py")]
        print("Launching visualizer...")
        subprocess.call(viz_cmd)
    finally:
        print("Shutting down backend...")
        if os.name == 'nt':
            backend_proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            backend_proc.terminate()
        try:
            backend_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            backend_proc.kill()

if __name__ == "__main__":
    main() 