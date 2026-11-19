"""The offline guard's failure names every refused connect's maker, whole.

A pool run keeps a failed ``attempts == []``'s values only as pytest shortened them (``Connect(hugg...the guard in)``),
so the frames that made a connect never reached the log. :func:`assert_no_connects` fails with every record as the
failure's own message, which pytest keeps as written.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

from huggingface_hub import constants as hub_constants
import pytest

from packages.evals.tests.offline_support import GUARD_THREAD, Connect, assert_no_connects

#: frames deeper than any shortened comparison keeps
_FRAMES = tuple(
    f"/work/packages/evals/src/threetears/evals/maker_{depth}.py:{depth} in step_{depth}" for depth in range(40)
)


def test_no_connect_passes() -> None:
    assert_no_connects([])


def test_a_refused_connect_fails_with_every_frame_of_its_record() -> None:
    made = Connect(
        address=("99.86.101.56", 443), host="huggingface.co", thread="MainThread", origin=GUARD_THREAD, stack=_FRAMES
    )

    with pytest.raises(pytest.fail.Exception) as failed:
        assert_no_connects([made, made])

    said = str(failed.value)
    assert said.startswith("2 connect(s) tried while every connect was refused"), said
    assert "a connect to huggingface.co ('99.86.101.56', 443) from thread 'MainThread'" in said, said
    for frame in _FRAMES:
        assert said.count(frame) == 2, frame


#: A process that loaded ``huggingface_hub`` (as ``voyageai`` does), then froze the clock with every connect refused,
#: as an offline test under :func:`packages.evals.tests.clock_support.frozen_clock` does. It prints each host it was
#: asked to resolve or reach, so it needs no network either way: the lookup is refused before any packet leaves.
_FREEZE_BEHIND_THE_GUARD = """
import socket
from datetime import UTC, datetime

from huggingface_hub import constants as hub_constants
from freezegun import freeze_time

tried = []

def resolve(host, *args, **kwargs):
    tried.append(host)
    raise socket.gaierror("refused")

def connect(self, address):
    tried.append(address)
    raise ConnectionRefusedError("refused")

socket.getaddrinfo = resolve
socket.socket.connect = connect
socket.socket.connect_ex = connect
with freeze_time(datetime(2026, 7, 20, tzinfo=UTC), real_asyncio=True):
    pass
print(tried)
"""


def _hosts_tried(env: dict[str, str]) -> list[object]:
    """What the freeze above tried to reach, run in a fresh process with ``env``."""
    done = subprocess.run(
        [sys.executable, "-c", _FREEZE_BEHIND_THE_GUARD], env=env, capture_output=True, text=True, check=True
    )
    tried: list[object] = ast.literal_eval(done.stdout.strip())
    return tried


def test_a_freeze_behind_the_guard_reaches_no_hub_even_with_no_registry_cached(tmp_path: Path) -> None:
    """A frozen clock behind the guard failed on CI on a connect to huggingface.co, whichever offline test was first in
    its process to freeze once ``huggingface_hub`` was loaded: freezegun's walk reads the package's lazy ``cli``,
    whose import fetches the Hub's agent-harness registry unless HF_HOME holds a copy under a day old. The session sets
    HF_HUB_OFFLINE (the root ``conftest.py``), so with an empty HF_HOME the freeze tries no host; without it, the same
    freeze tries the Hub, which is the failure the setting removes."""
    assert hub_constants.HF_HUB_OFFLINE, (
        "the session is not offline: the root conftest must set HF_HUB_OFFLINE before anything imports huggingface_hub"
    )
    session = {**os.environ, "HF_HOME": str(tmp_path / "hf-home")}
    online = {name: value for name, value in session.items() if name not in {"HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"}}

    assert _hosts_tried(session) == []
    assert "huggingface.co" in _hosts_tried(online)
