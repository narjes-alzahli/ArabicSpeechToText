#!/usr/bin/env python3
"""Run the Gradio app and restart on .py changes (watchdog)."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SKIP_DIRS = {".venv", ".git", ".gradio_temp", "__pycache__", ".certs"}


def _should_watch(path: Path) -> bool:
    if path.suffix != ".py":
        return False
    try:
        rel = path.resolve().relative_to(ROOT)
    except ValueError:
        return False
    return rel.parts and rel.parts[0] not in SKIP_DIRS


def main() -> None:
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer

    cmd = [sys.executable, str(ROOT / "app.py")]

    proc: subprocess.Popen | None = None
    debounce = 0.0

    def start() -> subprocess.Popen:
        print(f"Starting: {' '.join(cmd)}", file=sys.stderr, flush=True)
        return subprocess.Popen(cmd, cwd=str(ROOT))

    def stop() -> None:
        nonlocal proc
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=3)
        proc = None

    class Handler(FileSystemEventHandler):
        def _schedule(self, path: str) -> None:
            nonlocal debounce
            if not _should_watch(Path(path)):
                return
            debounce = time.time()

        def on_modified(self, event):  # noqa: N802
            if not event.is_directory:
                self._schedule(event.src_path)

        def on_created(self, event):  # noqa: N802
            if not event.is_directory:
                self._schedule(event.src_path)

    proc = start()
    handler = Handler()
    observer = Observer()
    observer.schedule(handler, str(ROOT), recursive=True)
    observer.start()
    print("Watchdog: watching .py files in project root", file=sys.stderr, flush=True)

    last_restart = 0.0
    try:
        while True:
            time.sleep(0.4)
            if debounce and time.time() - debounce > 0.8:
                if time.time() - last_restart < 1.0:
                    debounce = 0.0
                    continue
                debounce = 0.0
                last_restart = time.time()
                print("Restarting after code change...", file=sys.stderr, flush=True)
                stop()
                time.sleep(1)
                proc = start()
            if proc and proc.poll() is not None:
                code = proc.returncode
                if debounce:
                    time.sleep(0.5)
                    continue
                print(f"App exited ({code}); restarting in 2s...", file=sys.stderr, flush=True)
                time.sleep(2)
                proc = start()
    except KeyboardInterrupt:
        pass
    finally:
        observer.stop()
        observer.join()
        stop()


if __name__ == "__main__":
    main()
