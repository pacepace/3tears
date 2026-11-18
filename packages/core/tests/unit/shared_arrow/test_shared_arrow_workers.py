"""A shared Arrow pool's workers: forked by the fork server, never by the serving process, each exiting when the
serving process dies; and handed out a job at a time, a reader's job ahead of every background job waiting."""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import textwrap
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import pytest

from packages.core.tests.unit.shared_arrow import jobs
from threetears.core.shared_arrow import SharedArrowPool, SharedArrowStore

#: a serving process holding a pool of two workers over a store at ``argv[1]``: prints its pid, then each worker's
#: pid and its parent's, then waits to be killed
_SERVING = textwrap.dedent(
    """
    import asyncio
    import os
    import sys
    import time

    from threetears.core.shared_arrow import SharedArrowPool, SharedArrowStore


    def pids(context, pause):
        time.sleep(pause)
        return os.getpid(), os.getppid()


    async def main(root):
        pool = SharedArrowPool(SharedArrowStore(root, max_bytes=16 << 20), workers=2, private_bytes=1 << 40, name="s")
        ran = await asyncio.gather(*(pool.run(pids, 0.3, read=False) for _ in range(4)))
        workers = sorted({result.value for result in ran})
        print(os.getpid(), " ".join(f"{worker}:{server}" for worker, server in workers), flush=True)
        time.sleep(600)


    if __name__ == "__main__":
        asyncio.run(main(sys.argv[1]))
    """
)


def _alive(pid: int) -> bool:
    """whether ``pid`` is a live process."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


async def _until(condition: Callable[[], bool], what: str, seconds: float = 30.0) -> None:
    """wait until ``condition()`` holds."""
    deadline = time.monotonic() + seconds
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError(f"never: {what}")
        await asyncio.sleep(0.01)


@pytest.fixture
def where(tmp_path: Path) -> Path:
    """the test's directory: ``shm`` the store's root, ``signals`` the jobs' signal files."""
    (tmp_path / "signals").mkdir()
    return tmp_path


@pytest.fixture
async def single(where: Path) -> AsyncIterator[SharedArrowPool]:
    """one worker over a store, shut down after the test."""
    made = SharedArrowPool(
        SharedArrowStore(where / "shm", max_bytes=16 << 20), workers=1, private_bytes=1 << 40, name="single"
    )
    yield made
    await made.close()


class TestWorkersAreForkedByTheServer:
    async def test_every_worker_and_every_replacement_is_forked_by_the_fork_server(self, where: Path) -> None:
        """Flips if a pool's worker, its first or one replacing a worker past its bound, is a child of the serving
        process (a spawn: a fork_exec of the whole serving process under the interpreter lock)."""
        pool = SharedArrowPool(
            SharedArrowStore(where / "shm", max_bytes=16 << 20), workers=1, private_bytes=0, name="replaced"
        )
        try:
            ran = [(await pool.run(jobs.process_and_parent, read=False)).value for _ in range(3)]
        finally:
            await pool.close()

        assert pool.replaced >= 2, "every job's worker was past its bound: its pool should have been replaced"
        assert len({worker for worker, _ in ran}) == 3, ran
        assert os.getpid() not in {parent for _, parent in ran}, ran

    def test_the_workers_and_their_server_exit_within_5_seconds_of_the_serving_process_s_sigkill(
        self, where: Path
    ) -> None:
        """Flips if a pool's worker outlives its SIGKILLed serving process by 5 s (an orphan holding its memory), or
        the fork server outlives its last worker."""
        script = where / "serving.py"
        script.write_text(_SERVING, encoding="utf-8")
        serving = subprocess.Popen([sys.executable, str(script), str(where / "shm")], stdout=subprocess.PIPE, text=True)
        watched: set[int] = set()
        try:
            assert serving.stdout is not None
            said, *pairs = serving.stdout.readline().split()
            workers = [tuple(int(pid) for pid in pair.split(":")) for pair in pairs]
            assert workers, "the pool started no worker"
            assert int(said) == serving.pid
            watched = {pid for pair in workers for pid in pair}
            assert serving.pid not in watched, "the serving process forked its workers itself"
            os.kill(serving.pid, signal.SIGKILL)
            serving.wait()
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and any(_alive(pid) for pid in watched):
                time.sleep(0.1)
            left = sorted(pid for pid in watched if _alive(pid))
        finally:
            if serving.poll() is None:
                serving.kill()
            for pid in watched:
                if _alive(pid):
                    os.kill(pid, signal.SIGKILL)
        assert left == [], f"workers or their server outlived the serving process: {left}"


class TestReadersGoFirst:
    async def test_readers_submitted_behind_passes_start_before_them_in_their_order(
        self, single: SharedArrowPool, where: Path
    ) -> None:
        """Flips if a reader's job waits behind background jobs submitted before it, or either kind runs out of the
        order it was submitted in."""
        signals, started = where / "signals", where / "started"
        busy = asyncio.ensure_future(
            single.run(jobs.note_start_then_wait, str(started), "busy", str(signals), read=False)
        )
        await _until(lambda: started.exists(), "the busy job started")
        queued = []
        submitted = [("pass-0", False), ("pass-1", False), ("reader-0", True), ("pass-2", False), ("reader-1", True)]
        for name, reader in submitted:
            run = single.run(jobs.note_start, str(started), name, read=False, reader=reader)
            queued.append(asyncio.ensure_future(run))
            # the run asks for its turn at its first step: each asks in the order submitted
            await asyncio.sleep(0)

        (signals / "go").touch()
        await asyncio.gather(busy, *queued)

        assert started.read_text(encoding="utf-8").split() == [
            "busy",
            "reader-0",
            "reader-1",
            "pass-0",
            "pass-1",
            "pass-2",
        ]

    async def test_a_waiting_job_whose_caller_stops_waiting_never_runs_and_holds_no_worker(
        self, single: SharedArrowPool, where: Path
    ) -> None:
        """Flips if a job cancelled while waiting for a worker still runs, or keeps the turn it never used, so the
        jobs after it wait for ever."""
        signals, started = where / "signals", where / "started"
        busy = asyncio.ensure_future(
            single.run(jobs.note_start_then_wait, str(started), "busy", str(signals), read=False)
        )
        await _until(lambda: started.exists(), "the busy job started")
        dropped = asyncio.ensure_future(single.run(jobs.note_start, str(started), "dropped", read=False, reader=True))
        await asyncio.sleep(0)
        kept = asyncio.ensure_future(single.run(jobs.note_start, str(started), "kept", read=False))
        await asyncio.sleep(0)
        dropped.cancel()
        with pytest.raises(asyncio.CancelledError):
            await dropped

        (signals / "go").touch()
        await asyncio.wait_for(asyncio.gather(busy, kept), timeout=30)
        await asyncio.wait_for(single.run(jobs.note_start, str(started), "after", read=False), timeout=30)

        assert started.read_text(encoding="utf-8").split() == ["busy", "kept", "after"]
