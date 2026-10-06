"""A kind's result payloads: the engine's record of background work, and the kind's own opaque report.

Two carriers leave a kind on :class:`~threetears.evals.contracts.candidate_kind.CandidateOutput` and
land on :class:`~threetears.evals.contracts.models.EvalResult`:

- ``async_deliveries`` — one :class:`~threetears.evals.contracts.models.AsyncDelivery` per piece of
  background work (a scout sent ahead: acknowledged at once, delivered turns later on a model of its
  own). Engine vocabulary, so the engine reads it, and its refusals are what keep one entry telling
  one story.
- ``kind_payload`` — what only the kind can name, stored verbatim and read by nothing in the engine.

The runner-dispatch half (each carrier lands, the payload is never unpacked, the payload survives a
storage round trip) is in ``test_candidate_kind.py``. This file holds the record's own rules, a real
kind's payload end to end, and the deadline arm, which builds its record without the kind's return
value.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from pydantic import ValidationError

from threetears.evals.contracts.candidate_kind import CandidateOutput, CellSink
from threetears.evals.contracts.models import AsyncDelivery, EvalTestCase
from threetears.evals.run.runner import RunnerOptions, execute_run
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.fixtures.toyhost.kind import (
    TOY_EXTRACTOR_KIND,
    TOY_SCRIPTS,
    ScriptedExtractionClient,
    ToyExtractorInstance,
    ToyExtractorKind,
)
from packages.evals.tests.fixtures.toyhost.run import (
    RUN_MODELS,
    execute_toyhost_run,
    toyhost_run,
    toyhost_template,
    toyhost_test_cases,
)

# =============================================================================
# AsyncDelivery — one entry, one story
# =============================================================================


def _delivery(**overrides: Any) -> AsyncDelivery:
    """A scout delivered two turns after it was acknowledged, live on its own model, with overrides.

    Args:
        **overrides: Field overrides.

    Returns:
        The entry.
    """
    fields: dict[str, Any] = {
        "tool": "scout_ahead",
        "requested_by": "gm",
        "status": "delivered",
        "model": "scout/model",
        "acknowledged_turn": 1,
        "delivered_turn": 3,
        "elapsed_ms": 4200.0,
        "delivered_items": 2,
        "summary": "two goblins behind the ridge",
        "substituted": False,
    }
    fields.update(overrides)
    return AsyncDelivery(**fields)


@pytest.mark.parametrize(
    "fields",
    [
        pytest.param({}, id="delivered-live"),
        pytest.param({"model": None, "substituted": True}, id="delivered-from-a-seed"),
        pytest.param({"delivered_items": 0}, id="delivered-nothing-is-a-delivery"),
        pytest.param(
            {
                "status": "failed",
                "error": "scout timed out",
                "delivered_turn": None,
                "delivered_items": None,
                "summary": None,
            },
            id="failed",
        ),
        pytest.param(
            {"status": "undelivered", "delivered_turn": None, "delivered_items": None, "summary": None},
            id="undelivered-when-the-cell-ended",
        ),
        pytest.param({"acknowledged_turn": None, "delivered_turn": None}, id="a-kind-that-does-not-converse"),
        pytest.param({"acknowledged_turn": 2, "delivered_turn": 2}, id="delivered-on-the-turn-it-was-asked"),
    ],
)
def test_every_coherent_shape_of_background_work_is_accepted(fields: dict[str, Any]) -> None:
    entry = _delivery(**fields)
    assert entry.tool == "scout_ahead"


@pytest.mark.parametrize(
    ("fields", "names"),
    [
        pytest.param({"error": "boom"}, "error is set exactly when status is 'failed'", id="error-on-a-delivery"),
        pytest.param(
            {"status": "failed", "delivered_turn": None, "delivered_items": None, "summary": None},
            "error is set exactly when status is 'failed'",
            id="failure-without-an-error",
        ),
        pytest.param(
            {"status": "undelivered", "delivered_items": None, "summary": None},
            "delivered_turn is set on a 'undelivered' entry",
            id="undelivered-with-a-delivery-turn",
        ),
        pytest.param(
            {"status": "failed", "error": "x", "delivered_turn": None, "summary": None},
            "delivered_items is set on a 'failed' entry",
            id="failed-with-items",
        ),
        pytest.param(
            {"status": "undelivered", "delivered_turn": None, "delivered_items": None},
            "summary is set on a 'undelivered' entry",
            id="undelivered-with-a-summary",
        ),
        pytest.param(
            {"acknowledged_turn": 3, "delivered_turn": 1},
            "delivered_turn 1 precedes acknowledged_turn 3",
            id="delivered-before-it-was-asked",
        ),
        pytest.param({"substituted": True}, "a substituted delivery ran on no model", id="seeded-yet-names-a-model"),
    ],
)
def test_an_entry_telling_two_stories_is_refused_naming_the_contradiction(fields: dict[str, Any], names: str) -> None:
    with pytest.raises(ValidationError, match=names):
        _delivery(**fields)


def test_substituted_has_no_default_because_live_and_seeded_are_different_evidence() -> None:
    """A forgotten flag must not report a seeded run's cost as production's."""
    with pytest.raises(ValidationError, match="substituted"):
        AsyncDelivery(tool="scout_ahead", status="undelivered")


@pytest.mark.parametrize(
    "fields",
    [
        pytest.param({"tool": ""}, id="blank-tool"),
        pytest.param({"acknowledged_turn": -1}, id="negative-turn"),
        pytest.param({"elapsed_ms": -1.0}, id="negative-duration"),
        pytest.param({"delivered_items": -1}, id="negative-count"),
        pytest.param({"status": "pending"}, id="status-outside-the-three"),
    ],
)
def test_out_of_range_values_are_refused(fields: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        _delivery(**fields)


# =============================================================================
# A real kind's payload, end to end through the runner and the store
# =============================================================================


async def test_every_toy_cell_stores_the_fields_its_extraction_missed() -> None:
    """The toy kind names its misses on ``kind_payload``; each stored result carries exactly them."""
    path = await execute_toyhost_run(host=toyhost_host())

    results = path.results
    assert results, "the drive produced no cells"
    misses = {(script.model, doc): list(fields) for script in TOY_SCRIPTS for doc, fields in script.misses.items()}
    for result in results:
        trace = path.trace(result)
        assert trace is not None
        document_id = trace.trace[0]["document_id"]
        expected = misses.get((result.model, document_id), [])
        assert result.kind_payload is not None
        assert sorted(result.kind_payload["missed_fields"]) == sorted(expected)
        # The kind starts no background work, and says so by reporting nothing rather than [].
        assert result.async_deliveries is None
    assert any(result.kind_payload and result.kind_payload["missed_fields"] for result in results), (
        "no cell missed a field, so this drive could not tell a stored payload from an empty one"
    )


# =============================================================================
# The deadline arm: the record is built from the kind's reading, not its return value
# =============================================================================


class _ScoutStillOutKind(ToyExtractorKind):
    """A toy extractor whose turn acknowledges a scout, reports it, and never finishes.

    The cell's deadline cancels ``invoke`` from outside, so nothing it would have returned exists;
    what the record keeps is what the kind registered as its reading so far.
    """

    async def invoke(self, instance: ToyExtractorInstance, test_case: EvalTestCase, sink: CellSink) -> CandidateOutput:
        """Report a scout in flight and a payload, then wait past any deadline.

        Args:
            instance: Ignored.
            test_case: Ignored.
            sink: The cell's sink, which the reading is registered on.

        Returns:
            Never.
        """
        sink.report_progress(
            lambda: CandidateOutput(
                output=[{"content": "the party waits at the ford"}],
                async_deliveries=[
                    AsyncDelivery(
                        tool="scout_ahead",
                        requested_by="gm",
                        status="undelivered",
                        model="scout/model",
                        acknowledged_turn=0,
                        substituted=False,
                    )
                ],
                kind_payload={"phase": "waiting on the scout"},
            )
        )
        sink.waiting_on("background_work")
        await asyncio.Event().wait()
        raise AssertionError("unreachable: the cell deadline cancels this wait")


async def test_a_cell_cut_off_by_its_deadline_keeps_the_kinds_payload_and_deliveries() -> None:
    host = toyhost_host()
    world = host.profile.world
    assert world is not None
    template = toyhost_template()
    kind = _ScoutStillOutKind(
        client=ScriptedExtractionClient(), world=world, goal_checks=tuple(template.goal_state_checks)
    )
    (case, *_) = toyhost_test_cases(template)
    run = toyhost_run(model=RUN_MODELS[0], template=template, kind=kind, world=world).model_copy(
        update={"k_runs": 1, "test_case_ids": [case.id]}
    )
    host.storage.save_eval_run(run)

    await execute_run(
        host,
        run=run,
        template=template,
        test_cases=[case],
        judge_service=None,
        options=RunnerOptions(candidate_kinds={TOY_EXTRACTOR_KIND: lambda _cell: kind}, cell_timeout_s=0.05),
    )

    (result,) = host.storage.query_eval_results_by_run(run.id, run.scope_id)
    assert result.termination == "cell_timeout"
    assert result.kind_payload == {"phase": "waiting on the scout"}
    assert result.async_deliveries is not None
    assert [(entry.tool, entry.status) for entry in result.async_deliveries] == [("scout_ahead", "undelivered")]
    # Waiting on the candidate's own background work is the candidate's failure, not the rig's.
    assert result.candidate_error is not None and "background work the candidate started" in result.candidate_error
