"""a spawned pool's workers end when their parent is killed (:func:`threetears.core.processes.exit_with_parent`)."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

_PARENT = """
import multiprocessing, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from threetears.core.processes import exit_with_parent

def pid(_):
    time.sleep(0.2)
    return os.getpid()

if __name__ == "__main__":
    initializer = exit_with_parent if sys.argv[1] == "watch" else None
    pool = ProcessPoolExecutor(max_workers=3, mp_context=multiprocessing.get_context("spawn"), initializer=initializer)
    pids = sorted(set(pool.map(pid, range(12))))
    print(" ".join(map(str, pids)), flush=True)
    time.sleep(600)
"""


def _alive(pid: int) -> bool:
    """whether ``pid`` is a live process."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _orphans(mode: str, tmp_path: Path) -> list[int]:
    """start a parent holding a spawn pool, SIGKILL it, and return its workers still alive 5 s later."""
    script = tmp_path / "parent.py"
    script.write_text(_PARENT, encoding="utf-8")
    parent = subprocess.Popen([sys.executable, str(script), mode], stdout=subprocess.PIPE, text=True)
    try:
        assert parent.stdout is not None
        workers = [int(pid) for pid in parent.stdout.readline().split()]
        assert workers, "the pool started no worker"
        os.kill(parent.pid, signal.SIGKILL)
        parent.wait()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and any(_alive(pid) for pid in workers):
            time.sleep(0.1)
        return [pid for pid in workers if _alive(pid)]
    finally:
        if parent.poll() is None:
            parent.kill()


def test_workers_exit_within_5_seconds_of_their_parent_s_sigkill(tmp_path: Path) -> None:
    """Flips if a spawn pool's worker started with exit_with_parent outlives its SIGKILLed parent by 5 s."""
    left = _orphans("watch", tmp_path)
    for pid in left:
        os.kill(pid, signal.SIGKILL)
    assert left == [], f"workers outlived their parent: {left}"
