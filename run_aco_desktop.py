"""Launch backend server and ACO visualizer together.

Usage:
    python run_aco_desktop.py
"""

import subprocess
import sys
import time
import os
import signal
from pathlib import Path

ROOT = Path(__file__).parent.resolve()


def main():
    backend_cmd = [sys.executable, "-m", "backend.app"]
    print("Starting backend:", " ".join(backend_cmd))
    backend_proc = subprocess.Popen(backend_cmd, cwd=str(ROOT))

    try:
        time.sleep(2)  # allow backend to initialise
        viz_cmd = [sys.executable, str(ROOT / "visualizations" / "aco_visualizer.py")]
        print("Launching ACO visualizer...")
        subprocess.call(viz_cmd)
    finally:
        print("Shutting down backend...")
        if os.name == "nt":
            backend_proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            backend_proc.terminate()
        try:
            backend_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            backend_proc.kill()


if __name__ == "__main__":
    main() 