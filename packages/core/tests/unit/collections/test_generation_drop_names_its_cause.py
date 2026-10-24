"""a drop for a generation judged behind logs what the judgement saw, so the drop names its own cause.

A hub on cobalt dev dropped its own ``namespaces`` table as missed every few seconds with nothing in the log to say
whether an advance went unheard, arrived out of order, or was its own write left unaccounted.
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


def test_a_missed_drop_logs_the_generation_read_the_mark_and_what_waits(caplog: pytest.LogCaptureFixture) -> None:
    registry = CollectionRegistry()
    registry.follow_generation("namespaces")
    registry.settle_generation("namespaces", "inc:1")
    registry.generation_marks.hear("namespaces", "inc:3", 1)

    with caplog.at_level(logging.INFO):
        verdict = registry.settle_generation("namespaces", "inc:3")

    assert verdict is GenerationVerdict.MISSED
    judged = [
        r.__dict__.get("extra_data", {})
        for r in caplog.records
        if r.getMessage() == "write generation judged: the table is dropped"
    ]
    (seen,) = [extra for extra in judged if extra.get("verdict") == "missed"]
    assert seen == {
        "table": "namespaces",
        "verdict": "missed",
        "read": "inc:3",
        "recorded": "inc:1",
        "pending": ["inc:3 (1/1)"],
    }


def test_the_writers_own_advance_is_never_judged_behind() -> None:
    registry = CollectionRegistry()
    registry.follow_generation("namespaces")
    registry.settle_generation("namespaces", "inc:1")
    registry.account_generation("namespaces", "inc:2")
    assert registry.settle_generation("namespaces", "inc:2") is GenerationVerdict.CURRENT
