"""The CPU-clock timing rule a performance test leans on: ``best_cpu_seconds``, ``thread_cpu_seconds``, ``CpuStopwatch``.

A fake clock stands in for the CPU clock, so what is proven is the rule itself (warm-up discarded, fastest run kept,
a slow run cannot fail it, a slow code path still does), not this machine's speed. Moved from survey's
``tests/support/cpu_budget.py`` (d93c8691) with its tests.
"""

from __future__ import annotations

import threading

import pytest

from threetears.core.testing.calibrate import CpuStopwatch, best_cpu_seconds, cpu_seconds, thread_cpu_seconds


class _ScriptedOperation:
    """an operation whose every await advances a fake clock by the next scripted cost."""

    def __init__(self, costs: list[float]) -> None:
        self._costs = list(costs)
        self.now = 0.0
        self.awaits = 0

    def clock(self) -> float:
        return self.now

    async def __call__(self) -> None:
        self.now += self._costs[self.awaits]
        self.awaits += 1


class TestBestCpuSeconds:
    """the fastest timed run is the cost; the warm-up and one-off stalls are not."""

    async def test_the_warm_up_is_run_and_not_counted(self) -> None:
        operation = _ScriptedOperation([5.0, 0.002, 0.002, 0.002])

        best = await best_cpu_seconds(operation, runs=3, clock=operation.clock)

        assert operation.awaits == 4
        assert best == pytest.approx(0.002)

    async def test_one_stalled_run_does_not_make_the_cost(self) -> None:
        # a full collection or another thread's CPU landing in one run: 0.149s against 0.0004s
        operation = _ScriptedOperation([0.0013, 0.149, 0.0004, 0.0005])

        best = await best_cpu_seconds(operation, runs=3, clock=operation.clock)

        assert best == pytest.approx(0.0004)

    async def test_a_slow_operation_is_slow_in_every_run_and_still_reads_slow(self) -> None:
        operation = _ScriptedOperation([0.3, 0.2, 0.25, 0.21])

        best = await best_cpu_seconds(operation, runs=3, clock=operation.clock)

        assert best == pytest.approx(0.2)

    async def test_no_runs_is_refused(self) -> None:
        operation = _ScriptedOperation([0.1])

        with pytest.raises(ValueError, match="at least 1"):
            await best_cpu_seconds(operation, runs=0, clock=operation.clock)

        assert operation.awaits == 0


class TestThreadCpuSeconds:
    """the calling thread's clock does not count another thread's CPU."""

    def test_another_threads_work_is_not_counted(self) -> None:
        def burn() -> None:
            total = 0
            for i in range(3_000_000):
                total += i

        process_began = cpu_seconds()
        began = thread_cpu_seconds()
        worker = threading.Thread(target=burn)
        worker.start()
        worker.join()
        spent = thread_cpu_seconds() - began
        process_spent = cpu_seconds() - process_began

        # the worker spends on the order of 100ms, which the process clock sees; this thread only started and joined it
        assert process_spent > 0.03, f"the worker spent only {process_spent * 1000:.1f}ms, too little to tell"
        assert spent < 0.02, f"the calling thread was charged {spent * 1000:.1f}ms for another thread's work"


class TestCpuStopwatch:
    """the CPU spent inside the block, in seconds and milliseconds."""

    def test_the_block_s_cpu_is_read_in_seconds_and_milliseconds(self) -> None:
        with CpuStopwatch() as spent:
            total = 0
            for i in range(1_000_000):
                total += i

        assert spent.seconds > 0
        assert spent.ms == pytest.approx(spent.seconds * 1000)
