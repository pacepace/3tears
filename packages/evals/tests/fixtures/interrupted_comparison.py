"""A quick-path comparison run under ``asyncio.run``, as a person runs one, for a test to interrupt with real signals.

``python interrupted_comparison.py <mode> <store>`` runs a comparison of two arms, ``a`` and ``b``, over twelve cases,
into the store file at ``<store>``, scope ``s``, under the name ``interrupted``. It says on stdout, a line each, what
its cells do, so the test can send each signal after the event it waits for and never after a wait of its own:

- ``answered <n> <arm>``: the cell answered at once (the first four cells to start do);
- ``waiting <n> <arm>``: the cell is awaiting I/O that does not come, the loop idle;
- ``spinning <n> <arm>``: the cell is running on the loop thread (mode ``busy`` only, once ``go`` is read on stdin).

``<mode>`` is ``idle``: every later cell waits until it is cut; ``busy``: the same, and the first cell to read ``go``
spins on the loop thread, yielding never, until a signal lands in it; or ``continue``: every cell answers at once, the
comparison run again, and the cells it ran are said as ``answered`` lines.
"""

from __future__ import annotations

import asyncio
import sys
import time
from collections.abc import Callable, Mapping
from typing import Any

from threetears.evals import Eval

#: The cells that answer at once before the rest wait.
ANSWERED_AT_ONCE = 4
#: A spinning cell's own bound, so a signal that never comes fails the test rather than hanging it.
SPIN_BOUND_S = 120.0

MODE, STORE = sys.argv[1], sys.argv[2]


def _say(line: str) -> None:
    sys.stdout.write(f"{line}\n")
    sys.stdout.flush()


def correct(case: Mapping[str, Any], answer: str) -> bool:
    """Whether the answer is right."""
    return answer == "right"


def _arm(name: str, go: asyncio.Event, started: list[str], spun: list[str]) -> Callable[[Mapping[str, Any]], Any]:
    async def run(case: Mapping[str, Any]) -> str:
        cell = f"{case['n']} {name}"
        started.append(cell)
        if MODE == "continue" or len(started) <= ANSWERED_AT_ONCE:
            _say(f"answered {cell}")
            return "right"
        _say(f"waiting {cell}")
        await go.wait()
        if not spun:
            spun.append(cell)
            _say(f"spinning {cell}")
            bound = time.monotonic() + SPIN_BOUND_S
            while time.monotonic() < bound:
                pass
        await asyncio.Event().wait()
        return "right"

    run.__name__ = name
    return run


async def main() -> None:
    go = asyncio.Event()
    if MODE == "busy":

        def read_go() -> None:
            if sys.stdin.readline().strip() == "go":
                go.set()

        asyncio.get_running_loop().add_reader(sys.stdin.fileno(), read_go)
    started: list[str] = []
    spun: list[str] = []
    arms = {name: _arm(name, go, started, spun) for name in "ab"}
    cases = [{"n": n} for n in range(12)]
    await Eval(cases, arms, [correct], k=1, name="interrupted").run(store=STORE, scope="s")


if __name__ == "__main__":
    asyncio.run(main())
