"""CPU work in worker processes, plain data in and out (:mod:`threetears.core.process_workers`)."""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import time
from collections.abc import AsyncIterator
from pathlib import Path
from concurrent.futures.process import BrokenProcessPool
from typing import Any

import pytest

from threetears.core.process_workers import ProcessWorkers, require_plain


def _pid_and(value: bytes) -> tuple[int, bytes]:
    """
    :param value: anything plain
    :ptype value: bytes
    :return: the process that ran it, and the value reversed
    :rtype: tuple[int, bytes]
    """
    return os.getpid(), value[::-1]


def _comm() -> str:
    """
    :return: this process's name as Linux says it (``/proc/self/comm``); empty elsewhere
    :rtype: str
    """
    comm = Path("/proc/self/comm")
    return comm.read_text(encoding="utf-8").strip() if comm.exists() else ""


def _an_object() -> object:
    """
    :return: an object, which no worker may hand back
    :rtype: object
    """
    return object()


def _dies() -> None:
    """end the worker's process, as a worker killed mid-job would.

    :return: never
    :rtype: None
    """
    os._exit(1)


def _sleeps(seconds: float) -> int:
    """
    :param seconds: how long the job runs
    :ptype seconds: float
    :return: the process that ran it
    :rtype: int
    """
    time.sleep(seconds)
    return os.getpid()


def _raises(message: str) -> None:
    """
    :param message: the error's text
    :ptype message: str
    :return: never
    :rtype: None
    :raises ValueError: always
    """
    raise ValueError(message)


@pytest.fixture
async def workers() -> AsyncIterator[ProcessWorkers]:
    """
    :return: two workers, shut down after the test
    :rtype: AsyncIterator[ProcessWorkers]
    """
    made = ProcessWorkers(workers=2, name="test")
    try:
        yield made
    finally:
        await made.close()


async def test_a_pool_s_workers_carry_the_name_it_gives_them() -> None:
    """Flips if a worker of a pool given ``process_name`` does not name itself so as it starts (Linux, cut to 15
    bytes), or a worker of a pool given none takes a name: every worker the fork server forks has its command line,
    so the name is what an operator's fault aims at one pool's workers by."""
    named = ProcessWorkers(workers=1, name="test", process_name="tt-a-named-worker")
    plain = ProcessWorkers(workers=1, name="test")
    try:
        comms = (await named.run(_comm), await plain.run(_comm))
    finally:
        await named.close()
        await plain.close()
    if sys.platform.startswith("linux"):
        assert comms[0] == "tt-a-named-work"
        assert comms[1] != comms[0]
    else:
        assert comms == ("", "")


class TestPlainData:
    @pytest.mark.parametrize("value", [b"x", "x", 1, 1.5, True, None, (b"a", ["b", frozenset({"c"})]), bytes])
    def test_plain_data_passes(self, value: Any) -> None:
        require_plain(value, what="a value")

    @pytest.mark.parametrize("value", [object(), {"a": 1}, (b"a", [object()]), {1, 2}])
    def test_an_object_is_refused(self, value: Any) -> None:
        with pytest.raises(TypeError, match="plain data"):
            require_plain(value, what="a value")

    async def test_an_argument_that_is_an_object_is_refused_before_any_worker_starts(
        self, workers: ProcessWorkers
    ) -> None:
        with pytest.raises(TypeError, match="arguments of _pid_and"):
            await workers.run(_pid_and, object())

    async def test_an_answer_that_is_an_object_is_refused(self, workers: ProcessWorkers) -> None:
        with pytest.raises(TypeError, match="answer of _an_object"):
            await workers.run(_an_object)


class TestWhereItRuns:
    async def test_a_job_runs_in_another_process(self, workers: ProcessWorkers) -> None:
        pid, value = await workers.run(_pid_and, b"abc")

        assert pid != os.getpid() and value == b"cba"

    async def test_an_inline_job_runs_here_by_the_same_function(self, workers: ProcessWorkers) -> None:
        assert await workers.run(_pid_and, b"abc", inline=True) == (os.getpid(), b"cba")

    async def test_each_run_answers_in_its_order(self, workers: ProcessWorkers) -> None:
        answers = await workers.run_each(_pid_and, [(bytes([n]),) for n in range(8)])

        assert [value for _, value in answers] == [bytes([n]) for n in range(8)]
        assert len({pid for pid, _ in answers}) <= workers.workers

    async def test_a_jobs_error_reaches_the_caller(self, workers: ProcessWorkers) -> None:
        with pytest.raises(ValueError, match="no good"):
            await workers.run(_raises, "no good")

    async def test_a_worker_is_forked_by_the_fork_server_never_by_this_process(self, workers: ProcessWorkers) -> None:
        """Flips if a worker is this process's own child (a spawn: a fork_exec of the whole serving process under the
        interpreter lock)."""
        assert await workers.run(os.getppid) != os.getpid()


class TestTheWorkersLifetime:
    async def test_a_dead_worker_fails_its_job_and_the_next_job_gets_a_fresh_pool(
        self, workers: ProcessWorkers
    ) -> None:
        with pytest.raises(BrokenProcessPool):
            await workers.run(_dies)

        pid, _ = await workers.run(_pid_and, b"a")
        assert pid != os.getpid()

    async def test_closed_workers_are_started_again_by_the_next_job(self, workers: ProcessWorkers) -> None:
        first, _ = await workers.run(_pid_and, b"a")
        await workers.close()

        second, _ = await workers.run(_pid_and, b"a")

        assert second != first

    async def test_close_without_waiting_returns_at_once_and_ends_the_running_job_s_worker(
        self, workers: ProcessWorkers
    ) -> None:
        """Flips if ``close(wait=False)`` waits for a running job (a stop bound waiting minutes on a fold), or leaves
        its worker running, or fails to start a fresh pool for the next job."""
        worker = await workers.run(os.getpid)
        running = asyncio.ensure_future(workers.run(_sleeps, 30.0))
        await asyncio.sleep(0.2)  # the job is in its worker
        began = time.monotonic()
        await workers.close(wait=False)
        took = time.monotonic() - began
        with pytest.raises(BrokenProcessPool):
            await asyncio.wait_for(running, 5)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and _alive(worker):
            await asyncio.sleep(0.05)
        assert took < 0.5, f"close(wait=False) took {took:.2f} s"
        assert not _alive(worker), "the worker outlived close(wait=False)"
        assert await workers.run(os.getpid) != worker, "the next job starts a fresh pool"

    def test_a_worker_exits_within_5_seconds_of_the_serving_process_s_sigkill(self, tmp_path: Any) -> None:
        """Flips if a worker outlives its SIGKILLed serving process by 5 s: its parent, the fork server, lives while
        any worker holds the server's liveness pipe, so only a watch on the serving process ends it."""
        serving = subprocess.Popen([sys.executable, "-c", _SERVING], stdout=subprocess.PIPE, text=True)
        worker = 0
        try:
            assert serving.stdout is not None
            worker = int(serving.stdout.readline())
            os.kill(serving.pid, signal.SIGKILL)
            serving.wait()
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and _alive(worker):
                time.sleep(0.1)
            left = _alive(worker)
        finally:
            if serving.poll() is None:
                serving.kill()
            if worker and _alive(worker):
                os.kill(worker, signal.SIGKILL)
        assert not left, f"worker {worker} outlived the serving process"

    def test_a_pool_needs_a_worker(self) -> None:
        with pytest.raises(ValueError, match="needs a worker"):
            ProcessWorkers(workers=0, name="none")


#: a serving process holding one worker: prints the worker's pid, then waits to be killed
_SERVING = """
import asyncio, os, time
from threetears.core.process_workers import ProcessWorkers
async def main():
    workers = ProcessWorkers(workers=1, name="killed")
    print(await workers.run(os.getpid), flush=True)
    time.sleep(600)
asyncio.run(main())
"""


def _alive(pid: int) -> bool:
    """whether ``pid`` is a live process."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


#: a process at the lowest priority there is starts a pool and runs a job on it; prints whether a worker ran it
_AT_THE_LOWEST_PRIORITY = """
import asyncio, os
from threetears.core.process_workers import ProcessWorkers
os.setpriority(os.PRIO_PROCESS, 0, 20)
async def main():
    workers = ProcessWorkers(workers=1, name="lowest-priority")
    try:
        print(await workers.run(os.getpid) != os.getpid())
    finally:
        await workers.close()
asyncio.run(main())
"""

#: what macOS answers a process at the lowest priority asking to go lower (Linux clamps instead): installed in every
#: process of the run, the spawned workers included, so the refusal is the same on every platform
_NICE_REFUSED = """
import os
def _refused(increment):
    raise PermissionError(1, "Operation not permitted")
os.nice = _refused
"""


class TestAWorkerAtTheLowestPriority:
    """a worker that cannot be scheduled any lower keeps its priority and runs its jobs, never breaking its pool."""

    def test_a_pool_started_at_the_lowest_priority_runs_a_job(self, tmp_path: Any) -> None:
        (tmp_path / "sitecustomize.py").write_text(_NICE_REFUSED, encoding="utf-8")
        path = os.pathsep.join(filter(None, [str(tmp_path), os.environ.get("PYTHONPATH", "")]))
        done = subprocess.run(
            [sys.executable, "-c", _AT_THE_LOWEST_PRIORITY],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
            env={**os.environ, "PYTHONPATH": path},
        )

        assert done.returncode == 0, done.stderr
        assert done.stdout.strip().splitlines()[-1] == "True", done.stdout
