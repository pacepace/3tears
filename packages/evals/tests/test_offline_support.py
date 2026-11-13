"""The offline guard's failure names every refused connect's maker, whole.

A pool run keeps a failed ``attempts == []``'s values only as pytest shortened them (``Connect(hugg...the guard in)``),
so the frames that made a connect never reached the log. :func:`assert_no_connects` fails with every record as the
failure's own message, which pytest keeps as written.
"""

from __future__ import annotations

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
