"""The interrupt rule every holder of runs keeps (``kernel.offload.drain_then_cut``): a first interrupt asks the work to
start nothing more and waits for it to settle; a further cancel, or a second Ctrl+C delivered into the wait, cuts it,
and the wait still runs to its end.

Held here, on the rule alone (the engine's launches, sweeps and foreground waits, and the quick path's comparisons, are
held where each is tested):

- a wait no further interrupt meets asks once and never cuts;
- a cancel of the waiting task cuts, and the wait goes on to its end;
- a ``KeyboardInterrupt`` landing as the first interrupt is asked cuts, and is returned for the caller to raise;
- a failing settle step is recorded, and the next one still runs.
"""

from __future__ import annotations

import asyncio

import pytest

from threetears.evals.kernel.offload import drain_then_cut


async def test_a_wait_no_further_interrupt_meets_asks_once_and_never_cuts() -> None:
    asked: list[str] = []
    settled = asyncio.Event()

    async def settle() -> None:
        await asyncio.sleep(0.01)
        settled.set()

    second = await drain_then_cut(lambda: asked.append("ask"), lambda: asked.append("cut"), [(settle, "settling")])

    assert second is None and asked == ["ask"] and settled.is_set()


async def test_a_cancel_of_the_waiting_task_cuts_and_the_wait_goes_on_to_its_end() -> None:
    asked: list[str] = []
    release = asyncio.Event()

    def cut() -> None:
        asked.append("cut")
        release.set()

    async def settle() -> None:
        await release.wait()

    waiting = asyncio.ensure_future(drain_then_cut(lambda: asked.append("ask"), cut, [(settle, "settling")]))
    await asyncio.sleep(0.01)
    assert not waiting.done(), "the first interrupt lets the work finish"
    waiting.cancel()
    assert await waiting is None
    assert asked == ["ask", "cut"]


async def test_a_ctrl_c_landing_as_the_first_is_asked_cuts_and_is_returned_for_the_caller() -> None:
    asked: list[str] = []
    landed = KeyboardInterrupt()

    def ask() -> None:
        asked.append("ask")
        raise landed

    async def settle() -> None:
        return None

    second = await drain_then_cut(ask, lambda: asked.append("cut"), [(settle, "settling")])

    assert second is landed and asked == ["ask", "cut"]


async def test_a_failing_settle_step_is_recorded_and_the_next_still_runs(caplog: pytest.LogCaptureFixture) -> None:
    ran: list[str] = []

    async def fails() -> None:
        ran.append("fails")
        raise OSError("the store went away")

    async def after() -> None:
        ran.append("after")

    second = await drain_then_cut(lambda: None, lambda: None, [(fails, "the wait failed"), (after, "the next failed")])

    assert second is None and ran == ["fails", "after"]
    assert "the wait failed" in caplog.text
