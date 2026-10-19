"""A pooled CLI start never runs the SDK's ``claude -v`` check on the event loop.

``claude_agent_sdk``'s transport spawns ``claude -v`` before every CLI start, unless
``CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK`` is set in the process environment. The spawn is an
``asyncio.create_subprocess_exec`` on the loop, and in a large worker (metallm's are about 700 MB) it froze
the loop long enough for uvicorn to kill the worker: every faulthandler dump of a frozen worker had the main
thread in that spawn under ``PooledCliSession.start`` (metallm, production, 2026-10-19). The check only warns
about an old CLI, so a pooled start skips it, the first included.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any

import pytest

from threetears.models.claude_cli_pool import PooledCliSession

_SKIP = "CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK"


class _Options:
    """the launch options the pool hands the SDK: an environment, and no agents."""

    def __init__(self) -> None:
        self.env: dict[str, str] = {}
        self.agents: dict[str, Any] | None = None
        self.model: str | None = None


# parity-exempt: only connect and disconnect are driven; connect records whether the SDK would skip its version check
class _RecordingClient:
    seen: list[str | None] = []

    def __init__(self, options: Any) -> None:
        self.options = options

    async def connect(self) -> None:
        _RecordingClient.seen.append(os.environ.get(_SKIP))

    async def disconnect(self) -> None:
        return None


def test_the_sdk_reads_the_skip_flag_from_the_process_environment() -> None:
    """The premise this fix rests on, true of claude_agent_sdk 0.2.163: an SDK that reads the flag elsewhere
    (from the client's options, say) must fail here rather than silently spawn on the loop again."""
    # read as text: the transport module is the SDK's own private module, so nothing here imports it
    spec = importlib.util.find_spec("claude_agent_sdk")
    assert spec is not None and spec.origin is not None
    transport = Path(spec.origin).parent / "_internal" / "transport" / "subprocess_cli.py"
    assert f'os.environ.get("{_SKIP}")' in transport.read_text(encoding="utf-8")


async def test_every_pooled_start_skips_the_version_check_the_first_included(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import claude_agent_sdk

    monkeypatch.setenv(_SKIP, "")  # unset as far as the SDK is concerned; restored after the test
    monkeypatch.setattr(claude_agent_sdk, "ClaudeSDKClient", _RecordingClient)
    _RecordingClient.seen = []

    for n in range(3):
        await PooledCliSession.start(_Options(), key=f"k{n}")

    assert _RecordingClient.seen == ["1", "1", "1"]
