"""A run this build cannot read is refused or disclosed by every surface that reads it, never raised raw.

``runs_list`` failed with the model's own ``ValidationError`` through the action catalogue when the scope held
one run its stored row no longer validated (a corpus that outlived the model that wrote it), as did every other
read of the scope's runs: a defect's crash, where an agent needed a refusal naming what to do. Each surface now
answers one of two ways, chosen by what its answer is for:

- **The inventory discloses.** ``runs_list`` lists every run that reads and counts the rest, naming
  ``runs_unreadable``: an operator opening it is looking for a run, and one row nothing can read must not hide
  every other.
- **Everything else refuses.** A run's own read, its job, its curation, and every reader whose answer would be
  wrong without a run it matched (a comparison, a campaign's members, a cost) refuses with
  :class:`~threetears.evals.kernel.UnreadableRunError`, whose remedy names ``runs_unreadable``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from threetears.evals.actions import MountedTool, eval_catalogue, standard_tools
from threetears.evals.kernel import EvalStorage, UnreadableRunError, ValidationFailedError
from threetears.evals.kernel.campaign import EvalSweep, SweepArmRecord
from threetears.evals.kernel.remedies import FIND_UNREADABLE_RUNS, UNREADABLE_JOB_RECORD
from threetears.evals.ops import RunDeleted, RunListing, run_delete, runs_list, runs_unreadable
from threetears.evals.run import EvalActionJob, EvalLaunch
from threetears.evals.schema import JobLease
from threetears.evals.schema.models import EVAL_DOC_TYPES, eval_trace_doc_id
from threetears.evals.storage import InMemoryDocumentStore
from packages.evals.tests.factories import make_analysis_attempt, make_eval_run, make_eval_trace
from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_SUBJECT
from packages.evals.tests.fixtures.toyhost.run import toyhost_template
from packages.evals.tests.ops_support import CALLER, TOYHOST_SCOPE, OpsFixture, ops_fixture
from packages.evals.tests.off_loop import off_loop

RETIRED = "retired_knob: Extra inputs are not permitted"


def _fixture() -> tuple[OpsFixture, str, str]:
    """The toy operations host with one of its campaign's runs rewritten as no model would, and the other intact.

    Returns:
        The fixture, the unreadable run's id, and the readable one's.
    """
    store = InMemoryDocumentStore()
    fixture = ops_fixture(store=store)
    unreadable, readable = fixture.campaign.run_ids[:2]
    stored = store.get(unreadable, TOYHOST_SCOPE)
    assert stored is not None
    store.upsert(stored | {"retired_knob": 1})
    return fixture, unreadable, readable


def _tools() -> dict[str, MountedTool]:
    """Each engine action, by name, on the standard tool that carries it."""
    return {action.name: tool for tool in eval_catalogue().mount_all(standard_tools()) for action in tool.actions}


# =============================================================================
# The inventory discloses
# =============================================================================


def test_runs_list_lists_every_run_that_reads_and_counts_the_one_that_does_not() -> None:
    fixture, unreadable, readable = _fixture()

    listing = runs_list(fixture.host.eval_host, TOYHOST_SCOPE)

    assert [line.id for line in listing.runs] == [readable]
    assert listing.unreadable == 1


def test_runs_list_counts_an_unreadable_row_only_under_the_status_it_was_stored_with() -> None:
    fixture, _unreadable, _readable = _fixture()

    completed = runs_list(fixture.host.eval_host, TOYHOST_SCOPE, status="completed")
    running = runs_list(fixture.host.eval_host, TOYHOST_SCOPE, status="running")

    assert (len(completed.runs), completed.unreadable) == (1, 1)
    assert (running.runs, running.unreadable) == ([], 0)


def test_runs_list_of_a_scope_whose_every_run_reads_counts_none_unreadable() -> None:
    fixture = ops_fixture()

    listing = runs_list(fixture.host.eval_host, TOYHOST_SCOPE)

    assert (len(listing.runs), listing.unreadable) == (len(fixture.campaign.run_ids), 0)


async def test_the_runs_list_action_says_how_many_it_left_out_and_where_they_are_named() -> None:
    fixture, unreadable, readable = off_loop(_fixture)

    outcome = await _tools()["runs_list"].call({"action": "runs_list"}, host=fixture.host, caller=CALLER)

    assert not outcome.is_error, outcome.text
    assert RunListing.model_validate(outcome.structured).unreadable == 1
    assert readable in outcome.text and unreadable not in outcome.text
    assert outcome.text.splitlines()[-1] == (
        "not listed: 1 stored run(s) this build cannot read; runs_unreadable names each with why"
    )


# =============================================================================
# Everything else refuses
# =============================================================================


def _calls(fixture: OpsFixture, unreadable: str, readable: str) -> dict[str, dict[str, Any]]:
    """Each action that reads runs, called as it would reach the unreadable one."""
    campaign = fixture.campaign.id
    return {
        "run_get": {"run_id": unreadable},
        "results_list": {"run_id": unreadable},
        "job_poll": {"job_id": f"run:{unreadable}"},
        "job_cancel": {"job_id": f"run:{unreadable}"},
        "run_archive": {"run_id": unreadable},
        "run_label": {"run_id": unreadable, "run_labels": ["kept"]},
        "runs_compare": {"baseline_run_id": readable, "candidate_run_id": unreadable},
        "runs_bisect": {"baseline_run_id": readable, "candidate_run_id": unreadable},
        "scope_pivot": {"row_factor": "template_id", "column_factor": "model", "metric": "cost_usd"},
        "scope_history": {"metric": "cost_usd"},
        "scope_frontier": {},
        "scope_export": {"export_format": "csv", "export_run_ids": [readable, unreadable]},
        "report_read": {"campaign_id": campaign},
        "bars_propose": {"campaign_id": campaign},
        "analysis_estimate": {"campaign_id": campaign},
        "campaign_create": {
            "name": "again",
            "subject_id": TOYHOST_SUBJECT.subject_id,
            "behavior": "extraction",
            "exploratory": True,
            "run_ids": [readable, unreadable],
        },
        "judge_second_estimate": {"run_id": unreadable, "second_judge_model": "judge/b", "sample_fraction": 0.5},
        "launch_estimate": {
            "template_id": toyhost_template().id,
            "subject_id": TOYHOST_SUBJECT.subject_id,
            "models": ["extractor-v2"],
        },
    }


#: Every action :func:`_calls` reaches the unreadable run through, by name, for the parametrize ids.
_REFUSING = [
    "run_get",
    "results_list",
    "job_poll",
    "job_cancel",
    "run_archive",
    "run_label",
    "runs_compare",
    "runs_bisect",
    "scope_pivot",
    "scope_history",
    "scope_frontier",
    "scope_export",
    "report_read",
    "bars_propose",
    "analysis_estimate",
    "campaign_create",
    "judge_second_estimate",
    "launch_estimate",
]


@pytest.mark.parametrize("action", _REFUSING)
async def test_an_action_that_cannot_answer_without_the_run_refuses_naming_runs_unreadable(action: str) -> None:
    fixture, unreadable, readable = off_loop(_fixture)
    arguments = _calls(fixture, unreadable, readable)[action]

    outcome = await _tools()[action].call({"action": action, **arguments}, host=fixture.host, caller=CALLER)

    assert outcome.is_error and outcome.refusal is not None, outcome.text
    assert (outcome.refusal.kind, outcome.refusal.status) == ("refused", 500), outcome.text
    assert unreadable in outcome.text and RETIRED in outcome.text, outcome.text
    assert f"remedy: {FIND_UNREADABLE_RUNS}" in outcome.text, outcome.text


# =============================================================================
# The store's run reads
# =============================================================================


def _store_with_two_unreadable() -> tuple[EvalStorage, str]:
    store = InMemoryDocumentStore()
    storage = EvalStorage(store)
    readable = make_eval_run(id="run-reads", scope_id="s", status="completed")
    storage.save_eval_run(readable)
    for run_id in ("run-a", "run-b"):
        store.upsert(make_eval_run(id=run_id, scope_id="s", status="completed").to_dict() | {"retired_knob": 1})
    return storage, readable.id


def test_a_strict_run_read_refuses_naming_every_unreadable_row_it_matched() -> None:
    storage, _readable = _store_with_two_unreadable()

    with pytest.raises(UnreadableRunError) as refused:
        storage.query_eval_runs("s")

    assert sorted(refused.value.run_ids) == ["run-a", "run-b"]
    assert refused.value.remedy == FIND_UNREADABLE_RUNS and refused.value.status_code == 500
    assert "run-a: " + RETIRED in refused.value.message


@pytest.mark.parametrize(
    "read",
    [
        lambda storage: storage.load_eval_run("run-a", "s"),
        lambda storage: storage.load_eval_runs(["run-reads", "run-a"], "s"),
    ],
    ids=["load_eval_run", "load_eval_runs"],
)
def test_a_point_or_batch_read_of_an_unreadable_run_refuses_as_the_listing_does(read: Any) -> None:
    storage, _readable = _store_with_two_unreadable()

    with pytest.raises(UnreadableRunError) as refused:
        read(storage)

    assert refused.value.run_ids == ["run-a"]


def test_the_isolating_listing_returns_what_reads_and_names_what_does_not() -> None:
    storage, readable = _store_with_two_unreadable()

    read = storage.query_eval_runs_isolating("s")

    assert [run.id for run in read.runs] == [readable]
    assert sorted((row.row_id, tuple(row.reasons)) for row in read.unreadable) == [
        ("run-a", (RETIRED,)),
        ("run-b", (RETIRED,)),
    ]


# =============================================================================
# The way out: run_delete removes a run this build cannot read
# =============================================================================


def _with_a_trace_and_unreadable_results(fixture: OpsFixture, store: InMemoryDocumentStore, run_id: str) -> list[str]:
    """Give the run's first result a trace, then rewrite every result of it as no model would; return their ids."""
    storage = fixture.host.eval_host.storage
    results = storage.query_eval_results_by_run(run_id, TOYHOST_SCOPE)
    first = results[0]
    storage.save_eval_result(first, make_eval_trace(result_id=first.id, scope_id=TOYHOST_SCOPE, eval_run_id=run_id))
    for result in results:
        stored = store.get(result.id, TOYHOST_SCOPE)
        assert stored is not None
        store.upsert(stored | {"retired_knob": 1})
    return [result.id for result in results]


def _deleted_fixture(*, readable: bool) -> tuple[RunDeleted, InMemoryDocumentStore, OpsFixture, str, list[str]]:
    """Delete the campaign's first run — read whole, or rewritten as no model would — and return what is left."""
    store = InMemoryDocumentStore()
    fixture = ops_fixture(store=store)
    run_id = fixture.campaign.run_ids[0]
    results = _with_a_trace_and_unreadable_results(fixture, store, run_id) if not readable else []
    if readable:
        storage = fixture.host.eval_host.storage
        first = storage.query_eval_results_by_run(run_id, TOYHOST_SCOPE)[0]
        storage.save_eval_result(first, make_eval_trace(result_id=first.id, scope_id=TOYHOST_SCOPE, eval_run_id=run_id))
        results = [result.id for result in storage.query_eval_results_by_run(run_id, TOYHOST_SCOPE)]
    else:
        stored = store.get(run_id, TOYHOST_SCOPE)
        assert stored is not None
        store.upsert(stored | {"retired_knob": 1})
    deleted = run_delete(fixture.host.eval_host, run_id, TOYHOST_SCOPE, confirm=run_id)
    return deleted, store, fixture, run_id, results


def test_an_unreadable_run_is_deleted_with_its_results_traces_and_campaign_membership() -> None:
    deleted, store, fixture, run_id, results = _deleted_fixture(readable=False)

    assert (deleted.run_id, deleted.results_deleted, deleted.campaigns_detached) == (
        run_id,
        len(results),
        [fixture.campaign.id],
    )
    assert len(results) > 0
    assert store.get(run_id, TOYHOST_SCOPE) is None
    assert [store.get(result_id, TOYHOST_SCOPE) for result_id in results] == [None] * len(results)
    assert store.get(eval_trace_doc_id(results[0]), TOYHOST_SCOPE) is None
    campaign = fixture.host.eval_host.storage.load_campaign(fixture.campaign.id, TOYHOST_SCOPE)
    assert campaign is not None and run_id not in campaign.run_ids
    assert runs_unreadable(fixture.host.eval_host, TOYHOST_SCOPE).runs == []


def test_an_unreadable_run_s_delete_removes_exactly_what_a_readable_run_s_does() -> None:
    unreadable, unreadable_store, *_ = _deleted_fixture(readable=False)
    readable, readable_store, *_ = _deleted_fixture(readable=True)

    assert readable == unreadable
    assert _ids(unreadable_store) == _ids(readable_store)


def _ids(store: InMemoryDocumentStore) -> dict[str, list[str]]:
    """Every document left in the toy scope, by type, by id."""
    return {
        doc_type: sorted(row["id"] for row in store.by_doc_type(doc_type, TOYHOST_SCOPE)) for doc_type in EVAL_DOC_TYPES
    }


def test_an_unreadable_run_s_delete_still_asks_for_the_id_to_be_echoed() -> None:
    fixture, unreadable, _readable = _fixture()

    with pytest.raises(ValidationFailedError):
        run_delete(fixture.host.eval_host, unreadable, TOYHOST_SCOPE, confirm="not-it")

    assert len(runs_unreadable(fixture.host.eval_host, TOYHOST_SCOPE).runs) == 1


def _stranded(lease: JobLease | None) -> tuple[OpsFixture, str]:
    """A run stored running under ``lease`` that this build cannot read."""
    store = InMemoryDocumentStore()
    fixture = ops_fixture(store=store)
    run = make_eval_run(id="run-stranded", scope_id=TOYHOST_SCOPE, status="running", lease=lease)
    store.upsert(run.to_dict() | {"retired_knob": 1})
    return fixture, run.id


@pytest.mark.parametrize(
    "lease",
    [None, JobLease(holder="gone", expires_at="2020-01-01T00:00:00+00:00")],
    ids=["no lease", "a lapsed lease"],
)
def test_a_stranded_unreadable_run_is_deleted_since_no_process_holds_it(lease: JobLease | None) -> None:
    fixture, run_id = _stranded(lease)

    deleted = run_delete(fixture.host.eval_host, run_id, TOYHOST_SCOPE, confirm=run_id)

    assert deleted.run_id == run_id
    assert runs_unreadable(fixture.host.eval_host, TOYHOST_SCOPE).runs == []


def test_an_unreadable_run_a_live_lease_still_holds_is_refused_and_left_whole() -> None:
    live = JobLease(holder="another-build", expires_at=(datetime.now(UTC) + timedelta(hours=1)).isoformat())
    fixture, run_id = _stranded(live)

    with pytest.raises(ValidationFailedError, match="lease held until"):
        run_delete(fixture.host.eval_host, run_id, TOYHOST_SCOPE, confirm=run_id)

    assert [line.run_id for line in runs_unreadable(fixture.host.eval_host, TOYHOST_SCOPE).runs] == [run_id]


async def test_once_the_unreadable_run_is_deleted_launch_pricing_answers_again() -> None:
    fixture, unreadable, readable = off_loop(_fixture)
    tools = _tools()
    estimate = _calls(fixture, unreadable, readable)["launch_estimate"]

    refused = await tools["launch_estimate"].call(
        {"action": "launch_estimate", **estimate}, host=fixture.host, caller=CALLER
    )
    removed = await tools["run_delete"].call(
        {"action": "run_delete", "run_id": unreadable, "confirm": unreadable}, host=fixture.host, caller=CALLER
    )
    priced = await tools["launch_estimate"].call(
        {"action": "launch_estimate", **estimate}, host=fixture.host, caller=CALLER
    )

    assert refused.is_error and "run_delete removes one" in refused.text, refused.text
    assert not removed.is_error, removed.text
    assert not priced.is_error, priced.text


# =============================================================================
# A job record this build cannot read
# =============================================================================


def _job_records() -> dict[str, tuple[Any, str]]:
    """One record of each kind job_poll reads beside a run's, with the job id that polls it."""
    sweep = EvalSweep(
        scope_id=TOYHOST_SCOPE, campaign_id="c-1", template_id="t-1", subject_id="s-1",
        arms=[SweepArmRecord(label="a", model="m")], max_concurrent_arms=1,
    )  # fmt: skip
    launch = EvalLaunch(
        id="launch-1", scope_id=TOYHOST_SCOPE, template_id="t-1", subject_id="s-1", models=["m"], k_runs=1,
        n_variations=0,
    )  # fmt: skip
    attempt = make_analysis_attempt(scope_id=TOYHOST_SCOPE, campaign_id="c-1")
    action = EvalActionJob(scope_id=TOYHOST_SCOPE, action="rubric_propose")
    return {
        "sweep": (sweep, f"sweep:{sweep.id}"),
        "launch": (launch, f"launch:{launch.id}"),
        "analysis": (attempt, f"analysis:c-1:{attempt.id}"),
        "action": (action, f"action:{action.id}"),
    }


@pytest.mark.parametrize("kind", ["sweep", "launch", "analysis", "action"])
async def test_a_job_whose_record_this_build_cannot_read_is_refused_naming_the_record(kind: str) -> None:
    store = InMemoryDocumentStore()
    fixture = off_loop(ops_fixture, store=store)
    record, job_id = _job_records()[kind]
    store.upsert(record.to_dict() | {"retired_knob": 1})

    outcome = await _tools()["job_poll"].call(
        {"action": "job_poll", "job_id": job_id}, host=fixture.host, caller=CALLER
    )

    assert outcome.is_error and outcome.refusal is not None, outcome.text
    assert (outcome.refusal.kind, outcome.refusal.status) == ("refused", 500), outcome.text
    assert record.id in outcome.text and RETIRED in outcome.text, outcome.text
    assert f"remedy: {UNREADABLE_JOB_RECORD}" in outcome.text, outcome.text
