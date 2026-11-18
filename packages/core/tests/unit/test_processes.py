"""a pool's workers end when the process they work for is killed, and a pool started through the fork server is
forked by it, never by that process (:mod:`threetears.core.processes`)."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

#: a parent holding a pool of three workers: ``spawn`` (its workers its children), or ``server`` (forked by the fork
#: server, watching the parent's pid); prints the parent's pid, then each worker's pid and its parent's, then waits
_PARENT = """
import multiprocessing, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from threetears.core.processes import exit_with_parent, start_worker_server

def pid(_):
    time.sleep(0.2)
    return os.getpid(), os.getppid()

if __name__ == "__main__":
    if sys.argv[1] == "server":
        context, initargs = start_worker_server(), (0.2, os.getpid())
    else:
        context, initargs = multiprocessing.get_context("spawn"), (0.2,)
    pool = ProcessPoolExecutor(max_workers=3, mp_context=context, initializer=exit_with_parent, initargs=initargs)
    workers = sorted(set(pool.map(pid, range(12))))
    print(os.getpid(), " ".join(f"{worker}:{parent}" for worker, parent in workers), flush=True)
    time.sleep(600)
"""

#: a parent whose fork server preloads ``tabnanny`` (or nothing): prints whether a worker holds it before its job
#: imports anything
_PRELOADED = """
import os, sys
from concurrent.futures import ProcessPoolExecutor
from threetears.core.processes import start_worker_server

def holds(name):
    return name in sys.modules

if __name__ == "__main__":
    preload = sys.argv[1:]
    with ProcessPoolExecutor(max_workers=1, mp_context=start_worker_server(preload)) as pool:
        print(pool.submit(holds, "tabnanny").result(), flush=True)
"""


#: a parent that starts one worker (``spawn``, or forked by the fork ``server``) and prints its pid; the worker
#: prints its own pid, waits until the file named by ``argv[2]`` exists (the parent killed meanwhile: a spawn's
#: module import, 1.6 s in the hub's read worker) and only then calls exit_with_parent, naming no serving process
_LATE = """
import multiprocessing, os, sys, time
from pathlib import Path
from threetears.core.processes import exit_with_parent, start_worker_server

def work(go):
    print(os.getpid(), flush=True)
    while not Path(go).exists():
        time.sleep(0.05)
    exit_with_parent(0.2)
    time.sleep(600)

if __name__ == "__main__":
    context = start_worker_server() if sys.argv[1] == "server" else multiprocessing.get_context("spawn")
    print(os.getpid(), flush=True)
    context.Process(target=work, args=(sys.argv[2],)).start()
    time.sleep(600)
"""


def _alive(pid: int) -> bool:
    """whether ``pid`` is a live process."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _left_after_kill(mode: str, tmp_path: Path) -> tuple[int, list[tuple[int, int]], list[int]]:
    """start a parent holding a pool, SIGKILL it, and return its pid, its workers (pid, parent), and the workers
    and their parents still alive 5 s later."""
    script = tmp_path / "parent.py"
    script.write_text(_PARENT, encoding="utf-8")
    parent = subprocess.Popen([sys.executable, str(script), mode], stdout=subprocess.PIPE, text=True)
    workers: list[tuple[int, int]] = []
    try:
        assert parent.stdout is not None
        said, *pairs = parent.stdout.readline().split()
        workers = [(int(worker), int(server)) for worker, server in (pair.split(":") for pair in pairs)]
        assert workers, "the pool started no worker"
        os.kill(parent.pid, signal.SIGKILL)
        parent.wait()
        watched = {pid for pair in workers for pid in pair} - {parent.pid}
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and any(_alive(pid) for pid in watched):
            time.sleep(0.1)
        return int(said), workers, sorted(pid for pid in watched if _alive(pid))
    finally:
        if parent.poll() is None:
            parent.kill()
        for worker, server in workers:
            for pid in (worker, server):
                if pid != parent.pid and _alive(pid):
                    os.kill(pid, signal.SIGKILL)


def test_workers_exit_within_5_seconds_of_their_parent_s_sigkill(tmp_path: Path) -> None:
    """Flips if a spawn pool's worker started with exit_with_parent outlives its SIGKILLed parent by 5 s."""
    _, _, left = _left_after_kill("spawn", tmp_path)
    assert left == [], f"workers outlived their parent: {left}"


def test_a_fork_server_s_workers_and_the_server_exit_within_5_seconds_of_the_serving_process_s_sigkill(
    tmp_path: Path,
) -> None:
    """Flips if a worker the fork server forked outlives the SIGKILLed serving process it watches by 5 s, or the
    server outlives its last worker: the server lives while any worker holds its liveness pipe, so a worker watching
    its parent (the server) alone outlived the serving process for ever."""
    serving, workers, left = _left_after_kill("server", tmp_path)
    assert {server for _, server in workers} != {serving}, "the serving process forked its workers itself"
    assert left == [], f"workers or their server outlived the serving process: {left}"


def _left_after_late_watch(mode: str, tmp_path: Path) -> bool:
    """start a parent and its worker, SIGKILL the parent before the worker calls exit_with_parent, then let it
    call it: return whether the worker still lives 5 s later."""
    script = tmp_path / "late.py"
    script.write_text(_LATE, encoding="utf-8")
    go = tmp_path / "go"
    parent = subprocess.Popen([sys.executable, str(script), mode, str(go)], stdout=subprocess.PIPE, text=True)
    worker = 0
    try:
        assert parent.stdout is not None
        assert int(parent.stdout.readline()) == parent.pid
        worker = int(parent.stdout.readline())
        os.kill(parent.pid, signal.SIGKILL)
        parent.wait()
        go.write_text("", encoding="utf-8")
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and _alive(worker):
            time.sleep(0.1)
        return _alive(worker)
    finally:
        if parent.poll() is None:
            parent.kill()
        if worker and _alive(worker):
            os.kill(worker, signal.SIGKILL)


def test_a_spawned_worker_whose_parent_died_before_it_watched_exits_within_5_seconds(tmp_path: Path) -> None:
    """Flips if exit_with_parent takes the parent a worker has when it is called for the one that spawned it: a
    parent killed while the worker still imported left it adopted by init, which the watch then waited on for ever
    (the hub's read worker, red 4 of 4, 2026-11-18)."""
    assert not _left_after_late_watch("spawn", tmp_path), "the worker outlived the parent that spawned it"


def test_a_fork_server_s_worker_watching_no_named_process_exits_within_5_seconds_of_its_spawner(
    tmp_path: Path,
) -> None:
    """Flips if a worker the fork server forked, calling exit_with_parent with no ``serving``, watches the server
    (its parent, alive while the worker is) rather than the process that asked for it."""
    assert not _left_after_late_watch("server", tmp_path), "the worker outlived the process that started it"


def test_the_fork_server_s_workers_hold_its_preloaded_modules(tmp_path: Path) -> None:
    """Flips if a module named in preload is not imported by the fork server before it forks a worker (each worker
    importing it again), or one not named is."""
    script = tmp_path / "preloaded.py"
    script.write_text(_PRELOADED, encoding="utf-8")

    def said(*preload: str) -> str:
        done = subprocess.run(
            [sys.executable, str(script), *preload], capture_output=True, text=True, timeout=120, check=False
        )
        assert done.returncode == 0, done.stderr
        return done.stdout.strip().splitlines()[-1]

    assert said("tabnanny") == "True"
    assert said() == "False"
