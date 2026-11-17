"""A quick-path comparison interrupted twice by real ``SIGINT`` signals, as a person's two Ctrl+C reach it under
``asyncio.run``: the cells in flight are cut, the campaign is filed, the person sees ``KeyboardInterrupt``, and the
comparison run again runs only the cells no run recorded an answer for.

The first signal cancels the comparison's task (``asyncio.run``'s handler), which drains. The second raises
``KeyboardInterrupt`` wherever the main thread is, and that is where the cases differ:

- the loop idle, awaiting I/O (the usual case for a model candidate): it leaves the loop from its selector, and
  ``asyncio.run`` cancels every task once before it reports it;
- a cell's own step running on the loop thread (a CPU-busy candidate): the cell records itself cut, then the
  interrupt ends its task and leaves the loop from there.

Each signal is sent after an event the comparison says on its stdout or stderr, never after a wait: the child is
``fixtures/interrupted_comparison.py``. The one wait for a state rather than a line is the idle case's: the main
thread is read from ``/proc`` to be blocked in the selector's ``epoll`` call before the second signal goes.
"""

from __future__ import annotations

import platform
import queue
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import IO, Any

import pytest

from threetears.evals import Eval
from threetears.evals.analysis import get_campaign, list_campaigns
from threetears.evals.kernel.candidate_kind import QUICK_PATH_FILER
from threetears.evals.kernel.result_condition import runs_again
from threetears.evals.quick import as_host
from threetears.evals.quick.one_call import launched_as_run
from threetears.evals.run import list_results

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="real signals to a child, and /proc, are Linux's")

_CHILD = Path(__file__).parent / "fixtures" / "interrupted_comparison.py"
#: Every event's bound: a fail-safe for a child that never says it, not a timing the test depends on.
_EVENT_BOUND_S = 120.0
#: The selector's wait, by the syscall number ``/proc/<pid>/syscall`` names it with: ``epoll_wait``, ``epoll_pwait``
#: and ``epoll_pwait2``, as each architecture numbers them.
_EPOLL_WAITS = {"x86_64": {232, 281, 441}, "aarch64": {22, 441}}


class _Child:
    """The comparison's process, its stdout and stderr read a line at a time as they come."""

    def __init__(self, mode: str, store: Path) -> None:
        self.process = subprocess.Popen(
            [sys.executable, str(_CHILD), mode, str(store)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.lines: queue.Queue[tuple[str, str]] = queue.Queue()
        self.seen: list[tuple[str, str]] = []
        assert self.process.stdout is not None and self.process.stderr is not None
        self.pumps = [
            threading.Thread(target=self._pump, args=(stream, name), daemon=True)
            for stream, name in ((self.process.stdout, "out"), (self.process.stderr, "err"))
        ]
        for pump in self.pumps:
            pump.start()

    def _pump(self, stream: IO[str], name: str) -> None:
        for line in stream:
            self.lines.put((name, line.rstrip("\n")))

    def until(self, said: Callable[[str, str], bool]) -> None:
        """Read lines until one that ``said`` holds of."""
        bound = time.monotonic() + _EVENT_BOUND_S
        while True:
            name, line = self.lines.get(timeout=max(0.0, bound - time.monotonic()))
            self.seen.append((name, line))
            if said(name, line):
                return

    def finish(self) -> tuple[int, list[str]]:
        """Wait for the process to end and its output to be read; its exit code, and its stderr's lines."""
        code = self.process.wait(timeout=_EVENT_BOUND_S)
        for pump in self.pumps:
            pump.join(timeout=_EVENT_BOUND_S)
        while not self.lines.empty():
            self.seen.append(self.lines.get())
        return code, [line for name, line in self.seen if name == "err"]

    def cells(self, verb: str) -> set[tuple[int, str]]:
        """The cells the child said ``verb`` of, as ``(case, arm)``."""
        return {
            (int(n), arm)
            for name, line in self.seen
            if name == "out" and line.startswith(f"{verb} ")
            for n, arm in [line.split()[1:3]]
        }


def _in_the_selector(pid: int) -> bool:
    """Whether the process's main thread is blocked in its event loop's ``epoll`` wait."""
    fields = Path(f"/proc/{pid}/task/{pid}/syscall").read_text().split()
    return bool(fields) and fields[0].isdigit() and int(fields[0]) in _EPOLL_WAITS[platform.machine()]


def a(case: Mapping[str, Any]) -> str:
    """The child's arm ``a``, by its name."""
    return "right"


def b(case: Mapping[str, Any]) -> str:
    """The child's arm ``b``, by its name."""
    return "right"


def correct(case: Mapping[str, Any], answer: str) -> bool:
    """Whether the answer is right: the child's grader, by its name."""
    return answer == "right"


def _recorded(store: Path) -> tuple[list[str], dict[tuple[int, str], list[tuple[str, bool]]]]:
    """The comparison's filings under the child's name, and each cell's rows over every run they file, as
    ``(termination, runs again)``, read from the store."""
    host = as_host([Eval([{"n": 0}], {"a": a, "b": b}, [correct], k=1, name="interrupted")], store=store)
    campaigns = list_campaigns(host.storage, "s", subject_id="interrupted")
    assert all(campaign.created_by == QUICK_PATH_FILER for campaign in campaigns)
    # Each launch files the comparison over every run it has so far, so a run is in each filing after its own.
    run_ids = {
        run_id
        for campaign in campaigns
        for aspect in get_campaign(host.storage, campaign.id, "s").aspects
        for run_id in aspect.run_ids
    }
    rows: dict[tuple[int, str], list[tuple[str, bool]]] = {}
    for run_id in sorted(run_ids):
        run = host.storage.load_eval_run(run_id, "s")
        assert run is not None
        arm = str(launched_as_run(run)[0])
        for result in list_results(host.storage, run.id, "s"):
            case = int(result.test_case_id.rsplit("-", 1)[1])
            rows.setdefault((case, arm), []).append((str(result.termination), runs_again(result)))
    return [campaign.id for campaign in campaigns], rows


@pytest.mark.parametrize("where", ["idle", "busy"])
def test_two_sigints_cut_the_cells_in_flight_file_the_comparison_and_raise_keyboard_interrupt(
    where: str, tmp_path: Path
) -> None:
    """Interrupted twice by ``SIGINT``, the second landing with the loop idle or inside a cell's running step, the
    comparison files its campaign, each cell cut is recorded ``cancelled``, the process ends by the ``KeyboardInterrupt``
    the person gave; run again, it runs every cell no run recorded an answer for, each once, and no other."""
    if where == "idle" and platform.machine() not in _EPOLL_WAITS:
        pytest.fail(f"no epoll syscall numbers known for {platform.machine()}; add them to _EPOLL_WAITS")
    store = tmp_path / "evals.sqlite"
    child = _Child(where, store)
    # Two cells awaiting what never comes, so one is left to cut beside a cell the second signal lands in.
    child.until(lambda name, line: name == "out" and len(child.cells("waiting")) >= 2)
    child.process.send_signal(signal.SIGINT)
    child.until(lambda name, line: name == "err" and line.startswith("Interrupted: finishing"))
    spinning: set[tuple[int, str]] = set()
    if where == "busy":
        assert child.process.stdin is not None
        child.process.stdin.write("go\n")
        child.process.stdin.flush()
        child.until(lambda name, line: name == "out" and line.startswith("spinning "))
        spinning = child.cells("spinning")
    else:
        bound = time.monotonic() + _EVENT_BOUND_S
        while not _in_the_selector(child.process.pid):
            assert time.monotonic() < bound, "the comparison's loop never waited in its selector"
            time.sleep(0.001)
    child.process.send_signal(signal.SIGINT)
    code, stderr = child.finish()

    # The person's interrupt is what ended the process: asyncio.run reported the KeyboardInterrupt, uncaught.
    assert code == -signal.SIGINT and stderr[-1] == "KeyboardInterrupt", stderr
    filed, rows = _recorded(store)
    assert len(filed) == 1, stderr
    answered, waiting = child.cells("answered"), child.cells("waiting")
    assert answered and all(rows[cell] == [("completed", False)] for cell in answered), rows
    # Every cell in flight when the cut came is recorded cut, the cell the interrupt itself landed in among them, and
    # each runs again: none recorded an answer.
    assert waiting - spinning and spinning <= waiting, (waiting, spinning)
    assert all(rows.get(cell) == [("cancelled", True)] for cell in waiting), rows

    again = _Child("continue", store)
    code, stderr = again.finish()
    assert code == 0, stderr
    every = {(n, arm) for n in range(12) for arm in "ab"}
    # Run again, it runs the cut cells and the cells never recorded, each once, and no cell that recorded an answer.
    assert again.cells("answered") == every - answered
    _, rows = _recorded(store)
    assert all([kept for kept in rows[cell] if not kept[1]] == [("completed", False)] for cell in every), rows
