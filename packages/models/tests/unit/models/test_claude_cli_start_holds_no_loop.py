"""A Claude CLI start never holds the event loop, however long the spawn takes.

Under uvloop the SDK's ``anyio.open_process`` forks the whole worker with the interpreter lock held, and
in metallm's ~500 MB workers that froze every thread for 2-12 s until uvicorn killed the worker
(production, 2026-10-14..21: 134 of 208 freeze reports in ``PooledCliSession.start``). The CLI is now
started by ``subprocess.Popen`` (vfork) in a worker thread (:mod:`threetears.models._claude_cli_spawn`).

Every check here drives a real pooled start, through the SDK's own transport, against a stand-in CLI.
The stall check makes the spawn itself slow -- ``subprocess.Popen`` sleeps before starting the child:
a loop that spawns on its own thread stalls for the whole sleep, one that hands the spawn to a thread
keeps ticking.
"""

from __future__ import annotations

import asyncio
import importlib.util
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from threetears.models.claude_cli_pool import ClaudeCliSessionError, PooledCliSession

pytest.importorskip("claude_agent_sdk")

#: How long the patched spawn blocks the thread that calls it, in seconds.
_SPAWN_BLOCKS = 0.5
#: The longest the loop may stand still while a CLI starts, in seconds: a tenth of a second, far
#: under the blocked spawn, far over a scheduling hiccup.
_LOOP_MAY_STALL = 0.1

#: A stand-in for the Claude CLI: answers every control request (the SDK's ``initialize`` among them)
#: with success, and exits when its input closes.
_STAND_IN_CLI = f"""#!{sys.executable}
import json, sys
for line in sys.stdin:
    message = json.loads(line)
    if message.get("type") == "control_request":
        reply = {{"type": "control_response",
                  "response": {{"subtype": "success", "request_id": message["request_id"], "response": {{}}}}}}
        sys.stdout.write(json.dumps(reply) + "\\n")
        sys.stdout.flush()
"""


@pytest.fixture
def stand_in_cli(tmp_path: Path) -> Path:
    path = tmp_path / "claude"
    path.write_text(_STAND_IN_CLI, encoding="utf-8")
    path.chmod(0o755)
    return path


@pytest.fixture
def slow_spawn(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[subprocess.Popen[bytes]]]:
    """``subprocess.Popen`` blocks its caller for :data:`_SPAWN_BLOCKS` before it starts the child --
    every caller, asyncio's own subprocess transport included."""
    real_popen = subprocess.Popen
    started: list[subprocess.Popen[bytes]] = []

    def blocking_popen(*args: Any, **kwargs: Any) -> subprocess.Popen[bytes]:
        time.sleep(_SPAWN_BLOCKS)
        popen = real_popen(*args, **kwargs)
        started.append(popen)
        return popen

    monkeypatch.setattr(subprocess, "Popen", blocking_popen)
    yield started


class _LoopStall:
    """The longest the running loop went without running a 1 ms ticker."""

    def __init__(self) -> None:
        self.longest = 0.0
        self._task: asyncio.Task[None] | None = None

    async def _tick(self) -> None:
        last = time.perf_counter()
        while True:
            await asyncio.sleep(0.001)
            now = time.perf_counter()
            self.longest = max(self.longest, now - last)
            last = now

    async def __aenter__(self) -> _LoopStall:
        self._task = asyncio.create_task(self._tick())
        await asyncio.sleep(0.01)
        return self

    async def __aexit__(self, *exc: object) -> None:
        assert self._task is not None
        self._task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await self._task


def _options(cli: Path) -> Any:
    from claude_agent_sdk import ClaudeAgentOptions

    return ClaudeAgentOptions(cli_path=str(cli))


def test_the_sdk_starts_its_cli_through_its_module_global_anyio() -> None:
    """The premise of the fix, true of claude_agent_sdk 0.2.163: the transport calls ``open_process`` on
    its module's ``anyio``. An SDK that spawns some other way must fail here, not freeze workers again."""
    # read as text: the transport module is the SDK's own private module
    spec = importlib.util.find_spec("claude_agent_sdk")
    assert spec is not None and spec.origin is not None
    source = (Path(spec.origin).parent / "_internal" / "transport" / "subprocess_cli.py").read_text(encoding="utf-8")
    assert "\nimport anyio\n" in source
    assert "self._process = await anyio.open_process(" in source


async def test_a_pooled_start_keeps_the_loop_running_while_the_spawn_blocks(
    stand_in_cli: Path, slow_spawn: list[subprocess.Popen[bytes]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK", "1")
    async with _LoopStall() as stall:
        session = await PooledCliSession.start(_options(stand_in_cli), key="k")
    try:
        assert [popen.pid for popen in slow_spawn] == [session.pid]  # the start went through the blocked spawn
        assert stall.longest < _LOOP_MAY_STALL, (
            f"the loop stood still {stall.longest:.3f} s while a CLI started (the spawn blocked {_SPAWN_BLOCKS} s)"
        )
    finally:
        await session.dispose(grace_seconds=1.0)
    # the CLI read its input closing and exited on its own, and was reaped
    assert slow_spawn[0].returncode == 0


async def test_a_missing_cli_is_a_start_failure_and_not_a_hang(tmp_path: Path) -> None:
    from claude_agent_sdk import CLINotFoundError

    with pytest.raises(ClaudeCliSessionError) as failure:
        await PooledCliSession.start(_options(tmp_path / "no-such-claude"), key="k")
    assert isinstance(failure.value.__cause__, CLINotFoundError)


async def test_a_start_cancelled_while_spawning_kills_the_child_it_started(
    stand_in_cli: Path, slow_spawn: list[subprocess.Popen[bytes]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK", "1")
    starting = asyncio.create_task(PooledCliSession.start(_options(stand_in_cli), key="k"))
    await asyncio.sleep(_SPAWN_BLOCKS / 5)
    starting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await starting
    # the cancelled start returned at once; the spawn finishes in its thread, and the child it started is killed
    deadline = time.monotonic() + 5.0
    while not (slow_spawn and slow_spawn[0].returncode is not None) and time.monotonic() < deadline:
        await asyncio.sleep(0.01)
    [child] = slow_spawn
    assert child.returncode == -signal.SIGKILL
