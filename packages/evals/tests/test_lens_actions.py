"""The read lenses as actions: ``scope_pivot``, ``scope_history``, ``scope_export`` and ``launch_estimate``.

Driven over the toy host through :meth:`MountedTool.call`, the path every transport takes. Pinned here:

- **Each action is a thin binding of its lens.** Its structured result is the lens's typed model as the
  operation returns it, value for value, and validates as that model — a pivot is a
  :class:`PivotTable`, never a ``dict`` a surface re-types.
- **An estimate is the launch's own price.** ``launch_estimate`` plans and prices each arm as the launch
  does, and an arm it reports refused is refused by the launch in the same words; a kind's request-level
  refusal arrives ahead of any "cannot be priced".
- **An estimate becomes a pivot's plan.** ``launch_estimate``'s structured result, handed back as
  ``predicted_cost``, puts each priced arm's prediction beside the cost it observed.
- **An export is the serializer's bytes.** A CSV body keeps its final row terminator, and the counts a
  CSV has no place for ride beside it.
- **Every refusal names what was wrong**: a case count below one,
  a status no run carries, a format there is not, a plan that is not an estimate, the campaign's
  ``run_ids`` on an export.

Mutations that turn this file red (each run against a saved copy and restored from it):

- ``run.launch._judged``: either refusal reworded for the launch alone (the estimate then disagrees with it).
- ``actions.engine``: ``CaseCount`` without ``ge=1``; ``RunStatusFilter`` as a plain ``str``;
  ``ExportFormat`` as a plain ``str``; a handler dropping ``subject_filter``, ``run_status`` or
  ``predicted_cost`` on the way to its operation.
- ``contracts.base``: ``ScoreExport.body`` or ``ReportDocument.body`` as a plain ``str`` (the stance strips
  the trailing terminator).
- ``ops.lenses``: the exclusion line left out of a pivot's text.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import replace
from typing import Any

import pytest

from threetears.evals.actions import Caller, MountedTool, eval_catalogue, standard_tools
from threetears.evals.analysis import (
    HistoryResult,
    PivotTable,
    ScoreExport,
    campaign_report,
)
from threetears.evals.analysis.reporting import ScoreProjection
from threetears.evals.contracts import ValidationFailedError
from threetears.evals.ops import (
    LaunchArguments,
    LaunchEstimate,
    history_launch_pricer,
    launch_estimate,
    run_launch,
    report_read,
    scope_export,
    scope_history,
    scope_pivot,
    serialize_report,
)
from packages.evals.tests.factories import make_eval_result, make_eval_run
from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_COST_CEILING_USD
from packages.evals.tests.fixtures.toyhost.launch import TOYHOST_REVIEWER_POOL
from packages.evals.tests.fixtures.toyhost.run import toyhost_template, toyhost_test_cases
from packages.evals.tests.ops_support import CALLER, TOYHOST_SCOPE, TOYHOST_SUBJECT, OpsFixture, ops_fixture

#: The model the priced history is for, and the costs its three results record.
PRICED_MODEL = "extractor-v2"
PRICED_COSTS = (0.10, 0.20, 0.30)
#: The cases the toy kind plans every arm at.
CASES = len(toyhost_test_cases(toyhost_template()))


@pytest.fixture
def evals() -> MountedTool:
    return eval_catalogue().mount_all(standard_tools())[0]


def _priced(fixture: OpsFixture) -> OpsFixture:
    """Store a run of the toy template launched as a toy arm will be, with three priced results; price by history.

    Returns:
        The fixture over a host whose launch pricer is the engine's history pricer.
    """
    storage = fixture.host.eval_host.storage
    template = toyhost_template()
    run = make_eval_run(
        scope_id=TOYHOST_SCOPE,
        template_id=template.id,
        candidate_kind=template.candidate_kind,
        candidate_model=PRICED_MODEL,
        status="completed",
        # The toy kind's standing rig, which every toy arm is set up with, so the run was launched as one is.
        apparatus_settings={"reviewer_pool": TOYHOST_REVIEWER_POOL},
    )
    storage.save_eval_run(run)
    for index, cost in enumerate(PRICED_COSTS):
        storage.save_eval_result(
            make_eval_result(
                id=f"priced-{index}",
                eval_run_id=run.id,
                scope_id=TOYHOST_SCOPE,
                model=PRICED_MODEL,
                test_case_id=f"case-{index % 2}",
                cost_usd=cost,
            )
        )
    launch = replace(fixture.host.launch, launch_pricer=history_launch_pricer(fixture.host.eval_host))
    return OpsFixture(replace(fixture.host, launch=launch), fixture.campaign, fixture.writers)


def _unstamped(estimate: LaunchEstimate) -> dict[str, Any]:
    """An estimate's JSON form without its ``computed_at``, which is the clock's, not the lens's."""
    dumped = estimate.model_dump(mode="json")
    dumped.pop("computed_at")
    return dumped


def _launching(template_id: str, *models: str, **arguments: Any) -> dict[str, Any]:
    """A launch's own arguments — what ``run_launch`` and ``launch_estimate`` both take."""
    return {"template_id": template_id, "subject_id": TOYHOST_SUBJECT.subject_id, "models": list(models), **arguments}


async def _call(tool: MountedTool, fixture: OpsFixture, arguments: dict[str, Any], caller: Caller = CALLER) -> Any:
    return await tool.call(arguments, host=fixture.host, caller=caller)


# =============================================================================
# Each action is its lens
# =============================================================================


async def test_scope_pivot_returns_the_lens_table(evals: MountedTool) -> None:
    fixture = ops_fixture()
    arguments = {"row_factor": "test_case_id", "column_factor": "model", "metric": "cost_usd"}

    outcome = await _call(evals, fixture, {"action": "scope_pivot", **arguments})

    assert not outcome.is_error
    direct = scope_pivot(fixture.host.eval_host, TOYHOST_SCOPE, **arguments)
    assert outcome.structured == direct.model_dump(mode="json")
    table = PivotTable.model_validate(outcome.structured)
    assert table.cells and table.n_observations == sum(cell.n for cell in table.cells)
    assert outcome.text.startswith("pivot of cost_usd by test_case_id (rows) x model (columns)")
    assert "- doc-01 / extractor-v2: " in outcome.text


async def test_scope_pivot_reads_the_subject_and_status_it_is_given(evals: MountedTool) -> None:
    fixture = ops_fixture()
    base = {"action": "scope_pivot", "row_factor": "test_case_id", "column_factor": "model", "metric": "cost_usd"}

    elsewhere = await _call(evals, fixture, {**base, "subject_filter": "no-such-subject"})
    failed_only = await _call(evals, fixture, {**base, "run_status": "failed"})

    assert PivotTable.model_validate(elsewhere.structured).cells == []
    assert PivotTable.model_validate(elsewhere.structured).n_filtered_out > 0
    assert PivotTable.model_validate(failed_only.structured).cells == []
    assert PivotTable.model_validate(failed_only.structured).exclusions.results_outside_queried_runs > 0


async def test_a_pivot_says_what_it_left_out(evals: MountedTool) -> None:
    fixture = ops_fixture()
    archived = fixture.campaign.run_ids[0]
    await _call(evals, fixture, {"action": "run_archive", "run_id": archived, "archived": True})

    outcome = await _call(
        evals, fixture, {"action": "scope_pivot", "row_factor": "test_case_id", "column_factor": "model"}
    )

    table = PivotTable.model_validate(outcome.structured)
    assert table.exclusions.results_from_archived_runs > 0
    assert f"excluded: {table.exclusions.total} observation(s)" in outcome.text
    assert f"{table.exclusions.results_from_archived_runs} from archived runs" in outcome.text


async def test_scope_history_returns_the_lens_series(evals: MountedTool) -> None:
    fixture = ops_fixture()
    arguments = {"metric": "cost_usd", "min_relative_change": 0.1}

    outcome = await _call(evals, fixture, {"action": "scope_history", **arguments})

    assert not outcome.is_error
    direct = scope_history(fixture.host.eval_host, TOYHOST_SCOPE, **arguments)
    assert outcome.structured == direct.model_dump(mode="json")
    result = HistoryResult.model_validate(outcome.structured)
    assert result.min_relative_change == 0.1 and result.series
    assert outcome.text.startswith("history of cost_usd (lower is better)")
    for series in result.series:
        for point in series.points:
            assert point.run_id in outcome.text


async def test_scope_export_is_the_serializers_bytes_with_its_counts_beside_it(evals: MountedTool) -> None:
    fixture = ops_fixture()

    outcome = await _call(evals, fixture, {"action": "scope_export"})

    assert not outcome.is_error
    export = ScoreExport.model_validate(outcome.structured)
    assert export == scope_export(fixture.host.eval_host, TOYHOST_SCOPE)
    assert export.format == "csv"
    assert export.body.endswith("\r\n"), "the CSV's final row terminator is the serializer's, kept"
    assert len(list(csv.DictReader(io.StringIO(export.body)))) == export.n_records > 0
    assert outcome.text.startswith(f"export (csv, {export.n_records} row(s))")
    assert outcome.text.endswith(export.body)


async def test_a_json_export_carries_the_projection(evals: MountedTool) -> None:
    fixture = ops_fixture()
    named = fixture.campaign.run_ids[0]

    outcome = await _call(
        evals, fixture, {"action": "scope_export", "export_format": "json", "export_run_ids": [named]}
    )

    export = ScoreExport.model_validate(outcome.structured)
    projection = ScoreProjection.model_validate(json.loads(export.body))
    assert export.format == "json" and export.n_records == len(projection.records) > 0
    assert {record.run_id for record in projection.records} == {named}


async def test_launch_estimate_prices_the_launch_by_the_launchs_own_rule(evals: MountedTool) -> None:
    fixture = _priced(ops_fixture())
    arguments = _launching(toyhost_template().id, PRICED_MODEL, "unheard-of", k_runs=2)

    outcome = await _call(evals, fixture, {"action": "launch_estimate", **arguments})

    assert not outcome.is_error, outcome.text
    estimate = LaunchEstimate.model_validate(outcome.structured)
    direct = await launch_estimate(fixture.host, LaunchArguments(**arguments), TOYHOST_SCOPE)
    assert _unstamped(estimate) == _unstamped(direct)
    priced, unheard = estimate.arms
    assert (priced.case_count, priced.case_source, priced.n_observations) == (CASES, "stored", CASES * 2)
    assert priced.central_usd == pytest.approx(0.20 * CASES * 2), "the history's mean x the plan's cases x repeats"
    assert priced.predicted_usd is not None and priced.predicted_usd > priced.central_usd, "held to the upper end"
    assert unheard.predicted_usd is None and unheard.outcome == "refused" and estimate.total_predicted_usd is None
    assert unheard.refusal is not None and "cannot be priced: no priced past result" in unheard.refusal
    assert (estimate.cap_usd, estimate.cap_origin, estimate.would_launch) == (
        TOYHOST_COST_CEILING_USD,
        "inherited",
        False,
    )
    # One rule: the launch refuses the first refused arm in the estimate's own words.
    (first_refused,) = [arm.refusal for arm in estimate.arms if arm.refusal is not None][:1]
    with pytest.raises(ValidationFailedError) as refused:
        await run_launch(fixture.host, LaunchArguments(**arguments), TOYHOST_SCOPE)
    assert str(refused.value) == first_refused
    assert outcome.text.startswith(f"estimate: template {toyhost_template().id}, subject {TOYHOST_SUBJECT.subject_id}")
    assert "would be refused" in outcome.text


async def test_an_estimate_raises_what_the_launch_refuses_before_pricing_a_kinds_plan_first(evals: MountedTool) -> None:
    """A kind's request-level refusal reaches the operator ahead of "cannot be priced", on a host that prices nothing."""
    fixture = ops_fixture()
    unpriced = OpsFixture(
        replace(fixture.host, launch=replace(fixture.host.launch, launch_pricer=None)), fixture.campaign, []
    )

    outcome = await _call(evals, unpriced, {"action": "launch_estimate", **_launching(toyhost_template().id)})
    with pytest.raises(ValidationFailedError) as refused:
        await run_launch(unpriced.host, LaunchArguments(**_launching(toyhost_template().id)), TOYHOST_SCOPE)

    assert outcome.is_error and "has no default candidate model; name one" in outcome.text
    assert "cannot be priced" not in outcome.text
    assert str(refused.value) == f"kind '{toyhost_template().candidate_kind}' has no default candidate model; name one"


async def test_an_estimate_handed_back_as_predicted_cost_sits_beside_the_cost_observed(evals: MountedTool) -> None:
    fixture = _priced(ops_fixture())
    template_id = toyhost_template().id
    estimated = await _call(evals, fixture, {"action": "launch_estimate", **_launching(template_id, PRICED_MODEL)})
    arguments = {"row_factor": "template_id", "column_factor": "model", "metric": "cost_usd"}

    outcome = await _call(
        evals, fixture, {"action": "scope_pivot", **arguments, "predicted_cost": estimated.structured}
    )

    assert not outcome.is_error
    direct = scope_pivot(
        fixture.host.eval_host,
        TOYHOST_SCOPE,
        **arguments,
        predicted_cost=LaunchEstimate.model_validate(estimated.structured),
    )
    assert outcome.structured == direct.model_dump(mode="json")
    (cell,) = [cell for cell in direct.cells if cell.row == template_id]
    assert cell.value == pytest.approx(0.20)
    assert cell.predicted is not None and cell.predicted.method_id == "usage-history"
    assert f"- {template_id} / {PRICED_MODEL}: 0.2 (measured; n=3, 2 case(s)" in outcome.text
    assert "; predicted 0.2 [" in outcome.text


# =============================================================================
# Refusals that teach
# =============================================================================


async def test_a_case_count_below_one_is_refused_naming_it(evals: MountedTool) -> None:
    outcome = await _call(
        evals,
        ops_fixture(),
        {"action": "launch_estimate", **_launching("t", PRICED_MODEL), "n_test_cases": 0},
    )
    assert outcome.is_error and "launch_estimate was called with values it cannot take" in outcome.text
    assert "- n_test_cases:" in outcome.text


@pytest.mark.parametrize("action", ["run_launch", "launch_estimate"])
async def test_a_cap_above_the_hosts_is_refused_by_the_operation_and_the_action(
    evals: MountedTool, action: str
) -> None:
    """``max_cost_usd`` may only lower the host's ceiling — on the operation and on the action an agent calls."""
    fixture = _priced(ops_fixture())
    above = TOYHOST_COST_CEILING_USD + 1.0
    arguments = _launching(toyhost_template().id, PRICED_MODEL, max_cost_usd=above)
    operation = {"run_launch": run_launch, "launch_estimate": launch_estimate}[action]

    with pytest.raises(ValidationFailedError, match=f"max_cost_usd={above} is above the host's ceiling"):
        await operation(fixture.host, LaunchArguments(**arguments), TOYHOST_SCOPE)
    outcome = await _call(evals, fixture, {"action": action, **arguments})

    assert outcome.is_error and "a launch may only lower the host's ceiling" in outcome.text
    assert fixture.host.launch.job_manager.admitted_count == 0


async def test_a_status_no_run_carries_is_refused_naming_it(evals: MountedTool) -> None:
    outcome = await _call(evals, ops_fixture(), {"action": "scope_history", "run_status": "done"})
    assert outcome.is_error and "scope_history was called with values it cannot take" in outcome.text
    assert "- run_status" in outcome.text


async def test_an_export_format_there_is_not_is_refused_naming_it(evals: MountedTool) -> None:
    outcome = await _call(evals, ops_fixture(), {"action": "scope_export", "export_format": "parquet"})
    assert outcome.is_error and "scope_export was called with values it cannot take" in outcome.text
    assert "- export_format:" in outcome.text


async def test_a_plan_that_is_not_an_estimate_is_refused(evals: MountedTool) -> None:
    outcome = await _call(
        evals,
        ops_fixture(),
        {
            "action": "scope_pivot",
            "row_factor": "template_id",
            "column_factor": "model",
            "metric": "cost_usd",
            "predicted_cost": {"cells": "nope"},
        },
    )
    assert outcome.is_error and outcome.text.startswith(
        "refused: scope_pivot: predicted_cost is neither a launch estimate nor a cost estimate"
    )


async def test_a_lens_refusal_is_a_refused_call_with_the_lens_reason(evals: MountedTool) -> None:
    outcome = await _call(evals, ops_fixture(), {"action": "scope_history", "metric": "banana"})
    assert outcome.is_error and outcome.text.startswith("refused: scope_history: ")
    assert "banana" in outcome.text


async def test_the_campaigns_run_ids_are_not_an_exports(evals: MountedTool) -> None:
    """``run_ids`` means a campaign's membership on this tool; an export names its runs ``export_run_ids``."""
    outcome = await _call(evals, ops_fixture(), {"action": "scope_export", "run_ids": ["r"]})
    assert outcome.is_error and outcome.text.startswith("refused: scope_export does not take run_ids.")
    assert "- export_run_ids (" in outcome.text


def test_no_action_takes_the_scope_as_a_parameter() -> None:
    """The scope is the caller's, resolved by the host; naming one is never a way to reach another."""
    for action in eval_catalogue().actions:
        assert not {"scope_id", "scope"} & set(action.params.model_fields), action.name


# =============================================================================
# A serialized body is kept as written
# =============================================================================


def test_a_reports_body_is_the_serializers_bytes() -> None:
    fixture = ops_fixture()
    host = fixture.host.eval_host
    written = serialize_report(campaign_report(host, fixture.campaign.id, TOYHOST_SCOPE), "markdown")

    document = report_read(host, fixture.campaign.id, TOYHOST_SCOPE, format="markdown")

    assert written.endswith("\n"), "the serializer's report must end on a newline for this to test anything"
    assert document.body.endswith("\n")
    assert document.body.splitlines()[0] == written.splitlines()[0]
