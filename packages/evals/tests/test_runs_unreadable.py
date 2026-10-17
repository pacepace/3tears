"""Which stored runs this build cannot read, across a scope, with why (discodon#2498).

The boot sweep counts the non-terminal runs whose stored rows the current model refuses
(``AbandonedRunSweepReport.unreadable``), and that count reached only the boot log: nothing an operator
could call said which runs they were, or why. The ``runs_unreadable`` operation and action read every
stored run of the scope through the same isolating read the sweep's scan goes through, and name each
row the model refuses — ``stranded`` when it was stored pending or running, the rows the sweep counts.
"""

from __future__ import annotations

from typing import Any

import pytest

from threetears.evals.actions import eval_catalogue, standard_tools
from threetears.evals.kernel import StorageError
from threetears.evals.ops import UnreadableRunsListing, runs_unreadable
from threetears.evals.run import sweep_abandoned_runs
from threetears.evals.storage import InMemoryDocumentStore
from packages.evals.tests.factories import make_eval_run
from packages.evals.tests.ops_support import CALLER, TOYHOST_SCOPE, OpsFixture, ops_fixture
from packages.evals.tests.off_loop import off_loop

RETIRED = "retired_knob: Extra inputs are not permitted"
MISSING = "candidate_model: Field required"


class _AnsweringStore(InMemoryDocumentStore):
    """An in-memory store that stops answering for runs once told to, as a database that has gone does."""

    def __init__(self) -> None:
        super().__init__()
        self.down = False

    def by_doc_type(self, doc_type: str, scope_id: str, *, field_in: Any = None, **kwargs: Any) -> list[dict[str, Any]]:
        if self.down and doc_type == "eval_run":
            raise ConnectionError("the database went away")
        return super().by_doc_type(doc_type, scope_id, field_in=field_in, **kwargs)


def _fixture() -> tuple[OpsFixture, _AnsweringStore]:
    """The toy operations host, and the store under it, which a test writes rows to as no model would."""
    store = _AnsweringStore()
    return ops_fixture(store=store), store


def _unreadable_rows(store: InMemoryDocumentStore) -> None:
    """Three rows this build's model refuses, two of them stored not ended, and one readable run left lapsed."""
    for run_id, status in (("run-stranded-running", "running"), ("run-stranded-pending", "pending")):
        run = make_eval_run(id=run_id, scope_id=TOYHOST_SCOPE, status=status)
        store.upsert(run.to_dict() | {"retired_knob": 1})
    ended = make_eval_run(id="run-ended", scope_id=TOYHOST_SCOPE, status="completed").to_dict()
    del ended["candidate_model"]
    store.upsert(ended)
    store.upsert(make_eval_run(id="run-lapsed", scope_id=TOYHOST_SCOPE, status="running").to_dict())


def test_the_operation_names_each_run_the_model_refuses_with_why_and_no_readable_one() -> None:
    fixture, store = _fixture()
    _unreadable_rows(store)

    listing = runs_unreadable(fixture.host.eval_host, TOYHOST_SCOPE)

    assert listing.runs_read == len(fixture.campaign.run_ids) + 4
    assert [(line.run_id, line.status, line.stranded) for line in listing.runs] == [
        ("run-stranded-pending", "pending", True),
        ("run-stranded-running", "running", True),
        ("run-ended", "completed", False),
    ]
    assert [line.reasons for line in listing.runs] == [[RETIRED], [RETIRED], [MISSING]]
    assert listing.stranded == 2


def test_a_scope_whose_every_run_reads_names_none() -> None:
    fixture, _store = _fixture()

    listing = runs_unreadable(fixture.host.eval_host, TOYHOST_SCOPE)

    assert (listing.runs_read, listing.stranded, listing.runs) == (len(fixture.campaign.run_ids), 0, [])
    assert len(fixture.campaign.run_ids) > 0


def test_a_scope_holding_no_runs_names_none() -> None:
    fixture, _store = _fixture()

    listing = runs_unreadable(fixture.host.eval_host, "another-scope")

    assert (listing.scope_id, listing.runs_read, listing.stranded, listing.runs) == ("another-scope", 0, 0, [])


def test_the_boot_sweep_counts_unreadable_exactly_the_rows_the_operation_calls_stranded() -> None:
    fixture, store = _fixture()
    _unreadable_rows(store)
    storage = fixture.host.eval_host.storage

    listing = runs_unreadable(fixture.host.eval_host, TOYHOST_SCOPE)
    report = sweep_abandoned_runs(storage, [TOYHOST_SCOPE])

    assert report.unreadable == listing.stranded == 2
    assert report.cancelled_run_ids == ["run-lapsed"], "the readable run the sweep could settle is not counted"
    assert runs_unreadable(fixture.host.eval_host, TOYHOST_SCOPE) == listing, "the sweep settled none of them"


def test_a_store_that_fails_to_answer_is_an_error_never_a_scope_with_none() -> None:
    fixture, store = _fixture()
    store.down = True

    with pytest.raises(StorageError, match="the database went away"):
        runs_unreadable(fixture.host.eval_host, TOYHOST_SCOPE)


async def test_the_action_reads_the_same_and_says_it() -> None:
    fixture, store = off_loop(_fixture)
    off_loop(_unreadable_rows, store)
    (evals, *_rest) = eval_catalogue().mount_all(standard_tools())

    outcome = await evals.call({"action": "runs_unreadable"}, host=fixture.host, caller=CALLER)

    assert not outcome.is_error, outcome.text
    listing = UnreadableRunsListing.model_validate(outcome.structured)
    assert listing == off_loop(runs_unreadable, fixture.host.eval_host, TOYHOST_SCOPE)
    assert {f"runs_read: {listing.runs_read}", "stranded: 2", "runs (3):"} <= set(outcome.text.splitlines())
    assert f"- run_id=run-stranded-running; status=running; stranded=True; reasons=[{RETIRED}]" in outcome.text
    assert f"- run_id=run-ended; status=completed; stranded=False; reasons=[{MISSING}]" in outcome.text
    assert fixture.campaign.run_ids[0] not in outcome.text


async def test_the_action_refuses_when_the_store_fails_to_answer() -> None:
    fixture, store = off_loop(_fixture)
    store.down = True
    (evals, *_rest) = eval_catalogue().mount_all(standard_tools())

    outcome = await evals.call({"action": "runs_unreadable"}, host=fixture.host, caller=CALLER)

    assert outcome.is_error and "failed to read the eval runs" in outcome.text, outcome.text
