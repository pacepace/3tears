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

from typing import Any

import pytest

from threetears.evals.actions import MountedTool, eval_catalogue, standard_tools
from threetears.evals.kernel import EvalStorage, UnreadableRunError
from threetears.evals.kernel.remedies import FIND_UNREADABLE_RUNS
from threetears.evals.ops import RunListing, runs_list
from threetears.evals.storage import InMemoryDocumentStore
from packages.evals.tests.factories import make_eval_run
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
        "run_delete": {"run_id": unreadable, "confirm": unreadable},
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
    "run_delete",
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
