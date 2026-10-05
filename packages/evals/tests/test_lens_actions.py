"""The read lenses as actions: ``scope_pivot``, ``scope_history``, ``scope_export`` and ``launch_estimate``.

Driven over the toy host through :meth:`MountedTool.call`, the path every transport takes. Pinned here:

- **Each action is a thin binding of its lens.** Its structured result is the lens's typed model as the
  operation returns it, value for value, and validates as that model — a pivot is a
  :class:`PivotTable`, never a ``dict`` a surface re-types.
- **An estimate becomes a pivot's plan.** ``launch_estimate``'s structured result, handed back as
  ``predicted_cost``, puts each planned model's prediction beside the cost it observed.
- **An export is the serializer's bytes.** A CSV body keeps its final row terminator, and the counts a
  CSV has no place for ride beside it.
- **Every refusal names what was wrong**: a host that counts no template cases, a case count below one,
  a status no run carries, a format there is not, a plan that is not an estimate, the campaign's
  ``run_ids`` on an export.

Mutations that turn this file red (each run against a saved copy and restored from it):

- ``ops.launch_estimate``: removing the refusal of a host with no ``count_template_cases``.
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
from typing import Any

import pytest

from threetears.evals.actions import Caller, MountedTool, eval_catalogue, standard_tools
from threetears.evals.analysis import (
    CostEstimate,
    CostEstimateCell,
    HistoryResult,
    PivotTable,
    ScoreExport,
    campaign_report,
)
from threetears.evals.analysis.reporting import ScoreProjection
from threetears.evals.ops import (
    launch_estimate,
    report_read,
    scope_export,
    scope_history,
    scope_pivot,
    serialize_report,
)
from packages.evals.tests.factories import make_eval_result, make_eval_run, make_test_case
from packages.evals.tests.fixtures.toyhost.run import toyhost_template
from packages.evals.tests.ops_support import CALLER, TOYHOST_SCOPE, OpsFixture, ops_fixture

#: The model the priced history is for, and the costs its three results record.
PRICED_MODEL = "extractor-v2"
PRICED_COSTS = (0.10, 0.20, 0.30)


@pytest.fixture
def evals() -> MountedTool:
    return eval_catalogue().mount_all(standard_tools())[0]


def _priced(fixture: OpsFixture) -> str:
    """Store two of the toy template's cases and a run of it with three priced results; return the template id."""
    storage = fixture.host.eval_host.storage
    template = toyhost_template()
    for index in range(2):
        storage.save_test_case(make_test_case(id=f"case-{index}", scope_id=TOYHOST_SCOPE, template_id=template.id))
    run = make_eval_run(
        scope_id=TOYHOST_SCOPE,
        template_id=template.id,
        candidate_kind=template.candidate_kind,
        candidate_model=PRICED_MODEL,
        status="completed",
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
    return template.id


def _unstamped(estimate: CostEstimate) -> dict[str, Any]:
    """An estimate's JSON form without each prediction's ``computed_at``, which is the clock's, not the lens's."""
    dumped = estimate.model_dump(mode="json")
    for cell in dumped["cells"]:
        if cell["predicted"] is not None:
            cell["predicted"].pop("computed_at")
    return dumped


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


async def test_launch_estimate_prices_the_launch_from_the_scopes_history(evals: MountedTool) -> None:
    fixture = ops_fixture()
    template_id = _priced(fixture)
    arguments = {"template_id": template_id, "models": [PRICED_MODEL, "unheard-of"], "k_runs": 2}

    outcome = await _call(evals, fixture, {"action": "launch_estimate", **arguments})

    assert not outcome.is_error
    estimate = CostEstimate.model_validate(outcome.structured)
    assert _unstamped(estimate) == _unstamped(launch_estimate(fixture.host, TOYHOST_SCOPE, **arguments))
    assert (estimate.n_test_cases, estimate.n_test_cases_source, estimate.k_runs) == (2, "derived", 2)
    priced, unheard = estimate.cells
    assert isinstance(priced, CostEstimateCell), "the estimate's types are the analysis root's to import"
    assert priced.n_historical == len(PRICED_COSTS) and priced.predicted is not None
    assert priced.predicted.value == pytest.approx(0.20 * 2 * 2)
    assert unheard.basis == "no_history" and estimate.n_uncovered_models == 1
    assert outcome.text.startswith("estimate: 2 case(s) (derived) x k_runs 2")
    assert f"- {PRICED_MODEL}: 4 observation(s) at $0.2 each, from 3 priced in history" in outcome.text
    assert "1 model(s) have no history and are not in the total" in outcome.text


async def test_a_launch_estimate_reads_the_subject_it_is_given(evals: MountedTool) -> None:
    fixture = ops_fixture()
    template_id = _priced(fixture)

    outcome = await _call(
        evals,
        fixture,
        {"action": "launch_estimate", "template_id": template_id, "models": [PRICED_MODEL], "subject_filter": "nobody"},
    )

    estimate = CostEstimate.model_validate(outcome.structured)
    assert estimate.subject_id == "nobody" and estimate.cells[0].basis == "no_history"


async def test_an_estimate_handed_back_as_predicted_cost_sits_beside_the_cost_observed(evals: MountedTool) -> None:
    fixture = ops_fixture()
    template_id = _priced(fixture)
    estimated = await _call(
        evals, fixture, {"action": "launch_estimate", "template_id": template_id, "models": [PRICED_MODEL]}
    )
    arguments = {"row_factor": "template_id", "column_factor": "model", "metric": "cost_usd"}

    outcome = await _call(
        evals, fixture, {"action": "scope_pivot", **arguments, "predicted_cost": estimated.structured}
    )

    assert not outcome.is_error
    direct = scope_pivot(
        fixture.host.eval_host,
        TOYHOST_SCOPE,
        **arguments,
        predicted_cost=CostEstimate.model_validate(estimated.structured),
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


async def test_a_host_that_counts_no_template_cases_refuses_an_estimate(evals: MountedTool) -> None:
    fixture = ops_fixture(counts_cases=False)

    outcome = await _call(
        evals,
        fixture,
        {"action": "launch_estimate", "template_id": toyhost_template().id, "models": [PRICED_MODEL]},
    )

    assert outcome.is_error and outcome.structured is None
    assert outcome.text.startswith("refused: launch_estimate: this host does not estimate launches here")
    assert "OpsHost.count_template_cases" in outcome.text


async def test_a_case_count_below_one_is_refused_naming_it(evals: MountedTool) -> None:
    outcome = await _call(
        evals,
        ops_fixture(),
        {"action": "launch_estimate", "template_id": "t", "models": [PRICED_MODEL], "n_test_cases": 0},
    )
    assert outcome.is_error and "launch_estimate was called with values it cannot take" in outcome.text
    assert "- n_test_cases:" in outcome.text


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
    assert outcome.is_error and outcome.text.startswith("refused: scope_pivot: predicted_cost is not a cost estimate")


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
