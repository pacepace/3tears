"""falling behind on a table logs what the judgement saw, so it names its own cause.

A hub on cobalt dev judged its own ``namespaces`` table missed every few seconds with nothing in the log to say
whether an advance went unheard, arrived out of order, or was its own write left unaccounted. A catch-up that
stops short is logged at error with the epoch it could not apply, the generation read and the mark.
"""

from __future__ import annotations

import logging

import pytest

from threetears.core.collections.generation import GenerationMarks, GenerationVerdict
from threetears.core.collections.registry import CollectionRegistry


def test_an_advance_heard_past_a_gap_waits_as_pending() -> None:
    marks = GenerationMarks()
    marks.follow("namespaces")
    marks.settle("namespaces", "inc:1")
    marks.hear("namespaces", "inc:3", 1)
    assert marks.pending("namespaces") == ("inc:3 (1/1)",)
    assert marks.pending("not_followed") == ()
    assert not marks.heard_in_full("namespaces", "inc:2")
    assert marks.heard_in_full("namespaces", "inc:3")


async def test_falling_behind_logs_the_epoch_the_generation_read_and_the_mark(
    caplog: pytest.LogCaptureFixture,
) -> None:
    registry = CollectionRegistry()
    registry.follow_generation("namespaces")
    await registry.catch_up_generation("namespaces", "inc:1", changes=None)
    registry.generation_marks.hear("namespaces", "inc:3", 1)

    with caplog.at_level(logging.INFO):
        verdict = await registry.catch_up_generation("namespaces", "inc:3", changes=None)

    assert verdict is GenerationVerdict.MISSED
    assert registry.behind_on("namespaces")
    (seen,) = [
        r.__dict__.get("extra_data", {}) for r in caplog.records if r.getMessage().startswith("behind on a table")
    ]
    assert seen == {
        "table": "namespaces",
        "reason": "record_missing",
        "epoch": "inc:2",
        "read": "inc:3",
        "recorded": "inc:1",
    }


async def test_the_writers_own_advance_is_never_judged_behind() -> None:
    registry = CollectionRegistry()
    registry.follow_generation("namespaces")
    await registry.catch_up_generation("namespaces", "inc:1", changes=None)
    registry.account_generation("namespaces", "inc:2")
    assert await registry.catch_up_generation("namespaces", "inc:2", changes=None) is GenerationVerdict.CURRENT
    assert not registry.behind_on("namespaces")
