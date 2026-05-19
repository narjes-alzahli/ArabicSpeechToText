#!/usr/bin/env python3
"""Watchdog: runs app.py and restarts it automatically on crash."""

import subprocess
import sys
import time
import socket
from pathlib import Path

SCRIPT = Path(__file__).parent / "app.py"
PYTHON = Path(__file__).parent / ".venv" / "Scripts" / "python.exe"
if not PYTHON.exists():
    PYTHON = sys.executable

PORT = 7860
RESTART_DELAY = 3  # seconds between restarts


def free_port():
    """Kill any process holding PORT so the restart can bind successfully."""
    try:
        result = subprocess.run(
            ["netstat", "-ano"],
            capture_output=True, text=True
        )
        for line in result.stdout.splitlines():
            if f":{PORT}" in line and "LISTENING" in line:
                pid = line.strip().split()[-1]
                if pid.isdigit():
                    subprocess.run(["taskkill", "/F", "/PID", pid],
                                   capture_output=True)
                    print(f"[watchdog] Freed port {PORT} (killed PID {pid})", flush=True)
    except Exception as e:
        print(f"[watchdog] Could not free port: {e}", flush=True)


def run():
    attempt = 0
    while True:
        attempt += 1
        free_port()
        time.sleep(1)
        print(f"\n[watchdog] Starting app.py (attempt {attempt}) ...\n", flush=True)
        try:
            proc = subprocess.run([str(PYTHON), str(SCRIPT)])
            code = proc.returncode
        except KeyboardInterrupt:
            print("\n[watchdog] Interrupted — shutting down.", flush=True)
            break

        if code == 0:
            print("[watchdog] app.py exited cleanly. Stopping.", flush=True)
            break

        print(f"[watchdog] app.py exited with code {code}. Restarting in {RESTART_DELAY}s ...", flush=True)
        try:
            time.sleep(RESTART_DELAY)
        except KeyboardInterrupt:
            print("\n[watchdog] Interrupted during restart delay — shutting down.", flush=True)
            break

if __name__ == "__main__":
    run()
