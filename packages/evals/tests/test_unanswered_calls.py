"""The calls a role sent that went unanswered: counted on its usage row, carried to the report beside the arm's cost.

A usage row counts its role's calls that went unanswered (``RoleUsage.unanswered_retried``, ``unanswered_lost``,
``unanswered_reasons``): a forced tool choice refused and then answered unforced in the same call, and a call sent and
never answered. They stand beside the row's spend and never inside it. Contracts held here, hop by hop:

- **the row**: a count nobody made is None and is left out of the stored row, which is then the row it was before
  the fields existed; a zero is a count, and kept; reasons need a count beside them, count at least one call each,
  and never account for more calls than the counts do;
- **the runner**: a kind's candidate rows reach the stored result with their counts as the kind reported them
  (``CandidateTelemetry`` passes them through), and the cell's cost is the answered calls' alone;
- **the bundle**: each arm's census sums each role's counts over the rows that counted them, with the answered calls
  beside them and their share of every call sent; a trial whose row counted nothing is ``results_uncounted``, never a
  zero; a role no row counted in has no entry, and an arm with none carries no ``unanswered`` key at all;
- **the report input**: each scoreboard row copies its roles' counts, and leaves the key out where none counted;
- **the keys**: ``calls.unanswered.<arm>`` reads retried and lost together, by field beside it, on the lead aspect
  and by aspect; an arm no role counted for is no key; a share over two roles is computed, and marked derived;
- **beside the cost**: the analyst's overview cites them on the arm's row, and the options table flags them on its
  cost cell ("1 call unanswered", "3 calls unanswered"), only where any went unanswered;
- **the scoreboard block** (#1250: it summed the roles' records, ``int + Unanswered``, and ended a production report):
  its cell states the calls answered and unanswered in the options table's words, never a sum, a zero included; a bar
  of them is refused, so a page draws the table;
- **not counted is never zero**: where a role counted only one kind, the bare count and the share are refused as not
  counted, and the kind it counted reads alone; where some trials counted none, every count the cell and the flag
  state reads "at least";
- **a row standing for no answer** (``RoleUsage.answers_nothing``, the one statement of the rule, which a reading of
  the row carries) is never read as an answer: it names no model that served a call, no model a judge
  shares, no request settings left unverified, no token count left unreported, and no model a role's lever ran on.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
from pydantic import ValidationError

from packages.evals.tests.bundle_support import one_batch_bundle
from packages.evals.tests.factories import (
    DEFAULT_KIND,
    make_eval_result,
    make_eval_run,
    make_template,
    make_test_case,
)
from packages.evals.tests.fixtures.report_inputs import dataset
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.run_facts_support import ASPECT, CASES, aspect_bundle, campaign_over, result_of, run_of
from threetears.evals.analysis.analyst.blocks import block_render
from threetears.evals.analysis.bundle.config import effective_config, exercised
from threetears.evals.analysis.reporting import pooled_served_models, request_settings_readings, self_judging_models
from threetears.evals.analysis.analyst.facts import campaign_overview
from threetears.evals.analysis.errors import UnresolvableReference, UnstatedFigure
from threetears.evals.analysis.report.figure_keys import format_figure, resolve
from threetears.evals.analysis.report.block_spec import block_look, complete_chart
from threetears.evals.analysis.report.parts.resolve import resolve_chart
from threetears.evals.analysis.report.parts.tables import arms_table
from threetears.evals.analysis.report.fact_arms import fact_arms
from threetears.evals.analysis.report_input import ReportInput, Unanswered
from threetears.evals.analysis.report_input_build import report_input
from threetears.evals.analysis.report.forms import PageForms
from threetears.evals.kernel.candidate_kind import CandidateOutput, CandidateTelemetry, CellSink, VariantConfig
from threetears.evals.kernel.cassettes import CellCassettes
from threetears.evals.kernel import EvalStorage, RoleUsageLedger, UsageReading
from threetears.evals.kernel.errors import ValidationFailedError
from threetears.evals.kernel.host.sweepables import SHARED_CORE
from threetears.evals.kernel.result_reading import SharedValues
from threetears.evals.kernel.chart_query import ChartQuery, PlannedChart
from threetears.evals.kernel.usage_capture import blended_cost
from threetears.evals.run.judge import JUDGE_REQUEST_SETTINGS
from threetears.evals.run.remeasure import attest_served_model
from threetears.evals.run.runner import RunnerOptions, execute_run
from threetears.evals.schema.models import EvalResult, EvalRun, JudgedArtifact, RoleUsage
from threetears.evals.storage import InMemoryDocumentStore

FAST = "fast"
BASE = "base"


def _counting(
    calls: int = 3,
    *,
    retried: int | None,
    lost: int | None,
    reasons: dict[str, int] | None = None,
    role: str = "candidate",
) -> RoleUsage:
    return RoleUsage(
        role=role,
        model=FAST,
        cost_usd=0.002,
        call_count=calls,
        unanswered_retried=retried,
        unanswered_lost=lost,
        unanswered_reasons=reasons,
    )


def _trials(fast_usage: list[list[RoleUsage]]) -> list[tuple[EvalRun, list[EvalResult]]]:
    """The control's trials, each one priced call, and the candidate's, one trial per case with the usage given."""
    base, fast = run_of(BASE), run_of(FAST)
    return [
        (base, [result_of(base, index) for index in range(CASES)]),
        (fast, [result_of(fast, index, usage=usage) for index, usage in enumerate(fast_usage)]),
    ]


def _counted_input() -> ReportInput:
    """Two of the candidate's trials lost a call, one to a timeout; every trial retried one; each counted three."""
    usage = [
        [_counting(retried=1, lost=1 if index < 2 else 0, reasons={"timeout": 1} if index == 0 else None)]
        for index in range(CASES)
    ]
    return report_input(campaign_over(_trials(usage)))


# --- the row --------------------------------------------------------------------------------------------------------


def test_a_row_that_counted_nothing_stores_as_before_and_a_zero_is_kept() -> None:
    uncounted = RoleUsage(role="candidate", model=FAST, cost_usd=0.002, call_count=1).to_dict()
    assert not {"unanswered_retried", "unanswered_lost", "unanswered_reasons"} & set(uncounted)
    counted = _counting(retried=0, lost=0).to_dict()
    assert (counted["unanswered_retried"], counted["unanswered_lost"]) == (0, 0)
    assert "unanswered_reasons" not in counted
    assert RoleUsage.from_dict(counted) == _counting(retried=0, lost=0)


@pytest.mark.parametrize(
    ("retried", "lost", "reasons", "message"),
    [
        (None, None, {"timeout": 1}, "the row counts none"),
        (1, None, {"timeout": 0}, "at least one"),
        (1, 1, {"timeout": 2, "cancelled": 1}, "more than the 2"),
    ],
    ids=["no-count", "a-reason-of-none", "more-reasons-than-calls"],
)
def test_reasons_count_the_calls_the_counts_count(
    retried: int | None, lost: int | None, reasons: dict[str, int], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        _counting(retried=retried, lost=lost, reasons=reasons)


def test_a_count_below_zero_is_refused() -> None:
    with pytest.raises(ValidationError):
        _counting(retried=-1, lost=None)


# --- the ledger -----------------------------------------------------------------------------------------------------


def test_the_ledger_counts_unanswered_calls_on_the_row_of_the_calls_they_sit_beside() -> None:
    ledger = RoleUsageLedger(role="candidate")
    ledger.add(
        model="m",
        served_model="m-1",
        price_source="openrouter",
        prompt_tokens=10,
        completion_tokens=5,
        reasoning_tokens=None,
        cost_usd=0.02,
        calls=3,
    )
    ledger.add_unanswered(model="m", served_model="m-1", price_source="openrouter", retried=1, reasons={"refused": 1})
    ledger.add_unanswered(model="m", lost=2, reasons={"timeout": 2})
    ledger.add_unanswered(model="m", lost=0)

    # The lost calls named no served model and no price source; they fold into their model's answered row, beside its
    # call_count (owner, 2026-10-27: unanswered calls add nothing to cost and never unprice it). A separate row of
    # unanswered calls alone, as this test once expected, unpriced the cell's cost.
    (row,) = ledger.rows()

    assert (row.call_count, row.cost_usd, row.unanswered_retried, row.unanswered_lost) == (3, 0.02, 1, 2), (
        "the unanswered counts sit beside the answered calls and move none of their spend; a counted zero adds"
    )
    assert row.unanswered_reasons == {"refused": 1, "timeout": 2}


def test_a_lost_call_recorded_before_its_models_answered_calls_folds_into_their_row() -> None:
    ledger = RoleUsageLedger(role="candidate")
    ledger.add_unanswered(model="m", lost=1, reasons={"timeout": 1})
    ledger.add(model="m", served_model="m-1", prompt_tokens=1, completion_tokens=1, reasoning_tokens=None, cost_usd=0.5)

    (row,) = ledger.rows()

    assert (row.served_model, row.call_count, row.cost_usd, row.unanswered_lost) == ("m-1", 1, 0.5, 1)


def test_unanswered_calls_never_unprice_a_cells_cost() -> None:
    ledger = RoleUsageLedger(role="candidate")
    ledger.add(model="m", prompt_tokens=1, completion_tokens=1, reasoning_tokens=None, cost_usd=0.25)
    ledger.add_unanswered(model="m", served_model="other", lost=2)
    ledger.add_unanswered(model="never-answered", lost=1)

    rows = ledger.rows()
    alone = [row for row in rows if row.answers_nothing]

    assert [row.model for row in alone] == ["never-answered"], "a model that answered nothing keeps its own row"
    assert (alone[0].call_count, alone[0].cost_usd) == (None, None), "its spend stays unreported, never 0"
    assert blended_cost(rows, {"candidate"}) == 0.25, "the cell's cost is its answered calls', still priced"


def test_unanswered_calls_land_on_a_row_keyed_by_provider_and_actor_as_add_keyed_it() -> None:
    ledger = RoleUsageLedger(role="simulator")
    ledger.add(
        model="m",
        provider="p",
        provider_unit="call",
        actor_id="a1",
        purpose="utterance",
        prompt_tokens=1,
        completion_tokens=1,
        reasoning_tokens=None,
        cost_usd=0.01,
    )
    ledger.add_unanswered(model="m", provider="p", provider_unit="call", actor_id="a1", purpose="utterance", lost=1)

    (row,) = ledger.rows()

    assert (row.actor_id, row.provider, row.call_count, row.unanswered_lost) == ("a1", "p", 1, 1)
    with pytest.raises(ValueError, match="attribute simulator calls"):
        RoleUsageLedger(role="candidate").add_unanswered(model="m", actor_id="a1", lost=1)


def test_a_ledger_that_counted_no_unanswered_call_builds_its_rows_as_before() -> None:
    ledger = RoleUsageLedger(role="candidate")
    ledger.add(model="m", prompt_tokens=1, completion_tokens=1, reasoning_tokens=None, cost_usd=0.01)

    (row,) = ledger.rows()

    assert (row.unanswered_retried, row.unanswered_lost, row.unanswered_reasons) == (None, None, None)


@pytest.mark.parametrize(
    ("given", "says"),
    [
        ({"lost": -1}, "0 or more"),
        ({"reasons": {"timeout": 1}}, "none are counted here"),
        ({"lost": 1, "reasons": {"timeout": 0}}, "at least one"),
        ({"lost": 1, "reasons": {"timeout": 2}}, "more than the 1 counted"),
    ],
    ids=["negative", "reasons-alone", "zero-reason", "reasons-past-counts"],
)
def test_the_ledger_refuses_counts_a_row_could_not_hold(given: dict[str, Any], says: str) -> None:
    with pytest.raises(ValueError, match=says):
        RoleUsageLedger(role="candidate").add_unanswered(model="m", **given)


# --- the runner -----------------------------------------------------------------------------------------------------


# parity-with: threetears.evals.kernel.candidate_kind.CandidateKind
class _FakeCountingKind:
    """Answers every case with one priced candidate row counting one retried and one lost call, and one row of
    another model that counted nothing."""

    judged_artifact = JudgedArtifact.UNJUDGED

    async def prepare(
        self,
        *,
        subject_snapshot: Any,
        variant_config: VariantConfig,
        world_seed: Any,
        span_window: Any,
        cassettes: CellCassettes | None,
        world: Any,
    ) -> object:
        return object()

    async def invoke(self, instance: object, test_case: Any, sink: CellSink) -> CandidateOutput:
        return CandidateOutput(
            output=[{"answered": True}],
            telemetry=CandidateTelemetry(
                usage=[
                    _counting(2, retried=1, lost=1, reasons={"tool_choice_refused": 1, "timeout": 1}),
                    RoleUsage(role="candidate", model="other", cost_usd=0.001, call_count=1),
                ]
            ),
        )


async def test_the_kinds_counts_reach_the_stored_row_and_the_cost_counts_the_answered_calls() -> None:
    host = toyhost_host()
    template = make_template(id="tpl-counted", candidate_kind=DEFAULT_KIND, goal_state_checks=[])
    case = make_test_case(id="case-1", template_id=template.id)
    run = make_eval_run(
        template_id=template.id, test_case_ids=[case.id], candidate_kind=DEFAULT_KIND, candidate_model=FAST
    )
    await execute_run(
        host,
        run=run,
        template=template,
        test_cases=[case],
        judge_service=None,
        options=RunnerOptions(candidate_kinds={DEFAULT_KIND: lambda _cell: _FakeCountingKind()}),
    )
    (result,) = off_loop(host.storage.query_eval_results_by_run, run.id, run.scope_id)

    counted, other = result.usage or []
    assert (counted.unanswered_retried, counted.unanswered_lost) == (1, 1)
    assert counted.unanswered_reasons == {"tool_choice_refused": 1, "timeout": 1}
    assert (other.unanswered_retried, other.unanswered_lost, other.unanswered_reasons) == (None, None, None)
    assert result.cost_usd == pytest.approx(0.003)


# --- the bundle -----------------------------------------------------------------------------------------------------


def test_the_census_sums_each_counting_role_with_the_answered_calls_and_their_share() -> None:
    usage = [[_counting(retried=1, lost=1, reasons={"timeout": 1})]]
    usage += [[_counting(retried=1, lost=0)] for _ in range(CASES - 1)]
    _, bundle, _ = aspect_bundle(_trials(usage))
    board = {row.variant_key: row for row in bundle.scoreboard}
    (fast,) = [row for row in bundle.scoreboard if row.unanswered]
    (entry,) = fast.unanswered
    assert (entry.role, entry.retried, entry.lost, entry.reasons) == ("candidate", CASES, 1, {"timeout": 1})
    assert entry.calls == 3 * CASES
    assert entry.share == pytest.approx((CASES + 1) / (3 * CASES + CASES + 1))
    assert entry.results_uncounted == 0
    assert [row.unanswered for row in board.values() if row is not fast] == [[]]


def test_a_trial_whose_row_counted_nothing_is_uncounted_never_a_zero() -> None:
    usage = [[_counting(retried=None, lost=2)], *([[_counting(retried=None, lost=None)]] * (CASES - 1))]
    _, bundle, _ = aspect_bundle(_trials(usage))
    (fast,) = [row for row in bundle.scoreboard if row.unanswered]
    (entry,) = fast.unanswered
    assert (entry.retried, entry.lost, entry.results_uncounted) == (None, 2, CASES - 1)
    assert entry.calls == 3


def test_a_counting_row_that_did_not_count_its_calls_states_no_share() -> None:
    usage = [[RoleUsage(role="candidate", model=FAST, cost_usd=0.002, unanswered_lost=1)]] * CASES
    _, bundle, _ = aspect_bundle(_trials(usage))
    (fast,) = [row for row in bundle.scoreboard if row.unanswered]
    assert (fast.unanswered[0].calls, fast.unanswered[0].share) == (None, None)


def test_an_arm_no_row_counted_for_carries_no_key_and_its_census_dumps_as_before() -> None:
    _, bundle, _ = aspect_bundle(_trials([[_counting(retried=None, lost=None)]] * CASES))
    assert all(not row.unanswered for row in bundle.scoreboard)
    assert all("unanswered" not in row.to_dict() for row in bundle.scoreboard)


def test_a_bundle_over_counting_rows_reports_neither_count_as_a_lost_measurement_nor_summarises_it() -> None:
    """Both names are core measures beside ``call_count``: the walk reads a counting row's leaves as described, so it
    reports no lost observation of either, and neither has a direction, so no measure surface summarises them."""
    usage = [[_counting(retried=1, lost=1, reasons={"timeout": 1})]] * CASES
    _, bundle, _ = aspect_bundle(_trials(usage))
    collections = [
        bundle.telemetry.measures,
        *(summary.measures for summary in bundle.run_summaries),
        *(cell.measures for cell in bundle.cell_measures),
    ]
    names = ("unanswered_retried", "unanswered_lost")
    assert not [
        observation
        for collection in collections
        for observation in collection.unreported_observations
        if observation.startswith(names)
    ]
    assert not [measure.name for collection in collections for measure in collection.measures if measure.name in names]


def test_a_role_one_of_whose_rows_counted_one_kind_states_no_share_and_no_bare_count() -> None:
    """Row by row: one counted retried alone, the rest lost alone. The sums are both set, yet no row counted both, so
    the bundle stores no share and the keys refuse the two together as not counted."""
    usage = [[_counting(retried=1, lost=None)], *([[_counting(retried=None, lost=1)]] * (CASES - 1))]
    _, bundle, _ = aspect_bundle(_trials(usage))
    (fast,) = [row for row in bundle.scoreboard if row.unanswered]
    assert (fast.unanswered[0].retried, fast.unanswered[0].lost, fast.unanswered[0].share) == (1, CASES - 1, None)
    held = report_input(campaign_over(_trials(usage)))
    for key in (f"calls.unanswered.{FAST}", f"calls.unanswered.{FAST}.share"):
        with pytest.raises(UnstatedFigure, match="not counted"):
            resolve(key, held)
    assert format_figure(resolve(f"calls.unanswered.{FAST}.lost", held), forms=PageForms.of(held)) == str(CASES - 1)


# --- the report input and its keys ----------------------------------------------------------------------------------


def test_the_input_copies_each_roles_counts_and_leaves_the_key_out_where_none_counted() -> None:
    held = _counted_input()
    (copied,) = (held.scoreboard[FAST].unanswered or {}).items()
    assert copied[0] == "candidate"
    assert copied[1].model_dump() == {
        "retried": CASES,
        "lost": 2,
        "reasons": {"timeout": 1},
        "calls": 3 * CASES,
        "share": pytest.approx((CASES + 2) / (3 * CASES + CASES + 2)),
        "uncounted": 0,
    }
    assert held.scoreboard[BASE].unanswered is None
    dumped = held.model_dump(mode="json", by_alias=True)
    assert "unanswered" in dumped["scoreboard"][FAST] and "unanswered" not in dumped["scoreboard"][BASE]
    assert ReportInput.model_validate(dumped) == held


@pytest.mark.parametrize(
    ("key", "reads"),
    [
        (f"calls.unanswered.{FAST}", "8"),
        (f"calls.unanswered.{FAST}.retried", "6"),
        (f"calls.unanswered.{FAST}.lost", "2"),
        (f"calls.unanswered.{FAST}.calls", "18"),
        (f"calls.unanswered.{FAST}.share", "30.8%"),
        (f"calls.unanswered.{FAST}.reasons", "timeout 1"),
        (f"calls.unanswered.{FAST}.uncounted", "0"),
        (f"aspects.{ASPECT}.calls.unanswered.{FAST}", "8"),
        (f"aspects.{ASPECT}.calls.unanswered.{FAST}.lost", "2"),
    ],
)
def test_the_keys_read_the_counts_beside_the_cost(key: str, reads: str) -> None:
    held = _counted_input()
    assert format_figure(resolve(key, held), forms=PageForms.of(held)) == reads
    assert not resolve(key, held).derived


def test_an_arm_no_role_counted_for_is_no_key() -> None:
    with pytest.raises(UnresolvableReference, match="calls.unanswered"):
        resolve(f"calls.unanswered.{BASE}", _counted_input())


def test_a_share_over_two_roles_is_computed_from_their_counts_and_marked_derived() -> None:
    held = _counted_input()
    row = held.scoreboard[FAST]
    two = row.model_copy(
        update={
            "unanswered": {
                **(row.unanswered or {}),
                "inner_agent": Unanswered(retried=0, lost=2, calls=10, share=2 / 12),
            }
        }
    )
    widened = held.model_copy(update={"scoreboard": {**held.scoreboard, FAST: two}})
    share = resolve(f"calls.unanswered.{FAST}.share", widened)
    assert share.derived
    assert share.value == pytest.approx((CASES + 2 + 2) / (3 * CASES + 10 + CASES + 2 + 2))
    assert format_figure(resolve(f"calls.unanswered.{FAST}", widened), forms=PageForms.of(widened)) == "10"


# --- beside the cost ------------------------------------------------------------------------------------------------


def test_the_overview_cites_them_on_the_arms_row_only_where_any_went_unanswered() -> None:
    held = _counted_input()
    rows = {row.arm.key: row for row in campaign_overview(held, "snap").arms}
    assert [figure.key for figure in rows[f"arm:{FAST}"].unanswered] == [
        f"calls.unanswered.{FAST}",
        *(f"calls.unanswered.{FAST}.{field}" for field in ("retried", "lost", "share", "reasons", "uncounted")),
    ]
    assert rows[f"arm:{BASE}"].unanswered == []
    assert "unanswered" not in rows[f"arm:{BASE}"].model_dump()


def test_the_overview_of_an_arm_whose_calls_all_came_back_cites_none() -> None:
    held = report_input(campaign_over(_trials([[_counting(retried=0, lost=0)]] * CASES)))
    assert all(not row.unanswered for row in campaign_overview(held, "snap").arms)


def _cost_flags(held: ReportInput) -> list[str | None]:
    """Each arm's cost flag in a report document's options table (the arms table) over ``held``."""
    table = arms_table(held, block_look(held, fact_arms(held)), (), forms=PageForms.of(held))
    return [row["cost_per_1k"].flag for row in table.rows]


def _flag_changes(unanswered: Unanswered) -> list[tuple[str | None, str | None]]:
    """On the Kairo round 2 input, whose options table has a cost column: each cost flag that changes when its first
    arm is given ``unanswered``, before and after."""
    held = dataset.report_input_of("r5")
    arm = next(iter(held.scoreboard))
    counted = held.scoreboard[arm].model_copy(update={"unanswered": {"candidate": unanswered}})
    flags = _cost_flags(held.model_copy(update={"scoreboard": {**held.scoreboard, arm: counted}}))
    before = _cost_flags(held)
    return [(old, new) for old, new in zip(before, flags, strict=True) if old != new]


@pytest.mark.parametrize(
    ("unanswered", "said"),
    [
        (Unanswered(retried=0, lost=3, calls=297, share=0.01), "3 calls unanswered"),
        (Unanswered(retried=0, lost=1, calls=299, share=1 / 300), "1 call unanswered"),
        (Unanswered(retried=None, lost=2, calls=298), "2 calls lost unanswered"),
        (Unanswered(retried=0, lost=3, calls=297, share=0.01, uncounted=4), "at least 3 calls unanswered"),
    ],
    ids=["several", "one", "lost-only", "trials-uncounted"],
)
def test_the_options_table_flags_them_on_the_cost_cell(unanswered: Unanswered, said: str) -> None:
    ((old, new),) = _flag_changes(unanswered)
    assert new is not None and new.endswith(said)
    assert old is None or new == f"{old}; {said}"


def test_the_options_table_flags_nothing_where_none_went_unanswered() -> None:
    assert _flag_changes(Unanswered(retried=0, lost=0, calls=300, share=0.0)) == []


def test_the_bare_count_and_the_share_are_not_counted_where_a_role_counted_one_kind() -> None:
    usage = [[_counting(retried=None, lost=1)]] * CASES
    held = report_input(campaign_over(_trials(usage)))
    for key in (f"calls.unanswered.{FAST}", f"calls.unanswered.{FAST}.share"):
        with pytest.raises(UnstatedFigure, match="not counted"):
            resolve(key, held)
    assert format_figure(resolve(f"calls.unanswered.{FAST}.lost", held), forms=PageForms.of(held)) == str(CASES)
    with pytest.raises(UnstatedFigure, match="no role of the arm counted"):
        resolve(f"calls.unanswered.{FAST}.retried", held)
    _, bundle, _ = aspect_bundle(_trials(usage))
    (fast,) = [row for row in bundle.scoreboard if row.unanswered]
    assert (fast.unanswered[0].calls, fast.unanswered[0].share) == (3 * CASES, None)
    rows = {row.arm.key: row for row in campaign_overview(held, "snap").arms}
    assert [figure.key for figure in rows[f"arm:{FAST}"].unanswered] == [
        f"calls.unanswered.{FAST}.lost",
        f"calls.unanswered.{FAST}.uncounted",
    ]


# --- the scoreboard block (#1250) -----------------------------------------------------------------------------------

#: A scoreboard query naming nothing: every figure the rows hold, the unanswered calls among them.
_SCOREBOARD_QUERY: dict[str, Any] = {
    "source": "scoreboard",
    "aspect": "",
    "measures": [],
    "arms": [],
    "strata": [],
    "limit": 0,
    "transform": "none",
    "small_multiples_by": "none",
}


def _unanswered_cells(held: ReportInput, **query: Any) -> dict[str, str]:
    """Each arm's Unanswered cell in the scoreboard block, by arm label, drawn as the analyst's and the writer's
    ``block_render`` draws it (through ``resolve_chart``)."""
    drawn = block_render(held, block_kind="table", query=ChartQuery.model_validate({**_SCOREBOARD_QUERY, **query}))
    return {str(row.cells[0]): str(row.cells[2]) for row in drawn.rows if row.cells[1] == "Unanswered"}


@pytest.mark.parametrize("aspect", ["", ASPECT], ids=["input", "aspect"])
def test_the_scoreboard_block_states_the_calls_answered_and_unanswered_and_never_sums_them(aspect: str) -> None:
    """Today's code summed the arm's records by role (``int + Unanswered``) and raised, which ended a report."""
    cells = _unanswered_cells(_counted_input(), aspect=aspect)
    # Each trial counted three answered calls and retried one; two of them lost one more.
    assert list(cells.values()) == [f"{3 * CASES} calls answered; {CASES + 2} calls unanswered"]
    (arm,) = cells
    assert arm.endswith(FAST), "an arm no role counted for states none"


@pytest.mark.parametrize(
    ("unanswered", "said"),
    [
        ({"candidate": Unanswered(retried=0, lost=0, calls=300, share=0.0)}, "300 calls answered; 0 calls unanswered"),
        ({"candidate": Unanswered(retried=None, lost=2, calls=298)}, "298 calls answered; 2 calls lost unanswered"),
        ({"candidate": Unanswered(retried=1, lost=0)}, "1 call unanswered"),
        (
            {
                "candidate": Unanswered(retried=1, lost=1, calls=10, share=2 / 12),
                "inner_agent": Unanswered(retried=0, lost=2, calls=20, share=2 / 22),
            },
            "30 calls answered; 4 calls unanswered",
        ),
        (
            {"candidate": Unanswered(retried=1, lost=1, calls=10, share=2 / 12, uncounted=2)},
            "at least 10 calls answered; at least 2 calls unanswered",
        ),
        (
            {
                "candidate": Unanswered(retried=1, lost=1, calls=10, share=2 / 12),
                "inner_agent": Unanswered(retried=None, lost=1, calls=20, uncounted=1),
            },
            "at least 30 calls answered; at least 2 calls lost unanswered; at least 1 call retried unanswered",
        ),
    ],
    ids=[
        "none-unanswered",
        "one-kind-counted",
        "answered-uncounted",
        "two-roles",
        "trials-uncounted",
        "one-role-uncounted",
    ],
)
def test_the_scoreboard_cell_reads_each_count_its_key_reads(unanswered: dict[str, Unanswered], said: str) -> None:
    held = _counted_input()
    counted = held.scoreboard[FAST].model_copy(update={"unanswered": unanswered})
    cells = _unanswered_cells(held.model_copy(update={"scoreboard": {**held.scoreboard, FAST: counted}}))
    assert list(cells.values()) == [said]


def test_a_cell_over_trials_that_counted_none_reads_each_count_as_at_least() -> None:
    """One trial lost two calls and five counted none (``calls.unanswered.<arm>.uncounted``): each count is a lower
    bound, as the keys' rule says, so the cell reads "at least" where it stated whole counts."""
    usage = [[_counting(retried=None, lost=2)], *([[_counting(retried=None, lost=None)]] * (CASES - 1))]
    cells = _unanswered_cells(report_input(campaign_over(_trials(usage))))
    assert list(cells.values()) == ["at least 3 calls answered; at least 2 calls lost unanswered"]


#: The second aspect's id, whose own scoreboard holds counts the lead's do not.
_SECOND = "speed"


def _two_aspects(second: Unanswered) -> ReportInput:
    """The counted input with a second aspect after the lead, whose scoreboard holds the candidate's ``second``
    counts, where the lead's (the input's own) hold its six trials'."""
    held = _counted_input()
    lead = held.aspects[0]
    board = {
        arm: row.model_copy(update={"unanswered": {"candidate": second}}) if arm == FAST else row
        for arm, row in held.scoreboard.items()
    }
    return held.model_copy(update={"aspects": [lead, lead.model_copy(update={"id": _SECOND, "scoreboard": board})]})


def test_each_aspects_scoreboard_cell_reads_its_own_aspects_counts() -> None:
    """A cell of an aspect other than the lead reads its own keys (``aspects.<id>.calls.unanswered.<arm>``): a wrong
    prefix printed the lead's counts under its scoreboard."""
    held = _two_aspects(Unanswered(retried=1, lost=0, calls=7, share=1 / 8))
    assert list(_unanswered_cells(held, aspect=_SECOND).values()) == ["7 calls answered; 1 call unanswered"]
    lead = [f"{3 * CASES} calls answered; {CASES + 2} calls unanswered"]
    assert list(_unanswered_cells(held, aspect=ASPECT).values()) == lead


def test_a_bar_of_the_calls_answered_and_unanswered_is_refused_so_the_page_draws_its_table() -> None:
    """A page draws a block it completed (``doc_page.build``) before its checks name the measure, and a refusal is
    what it catches to draw the table instead: today's code raised ``TypeError`` there."""
    held = _counted_input()
    query = {**_SCOREBOARD_QUERY, "measures": ["unanswered"]}
    planned = PlannedChart.model_validate(
        {"kind": "bars", "title": "", "query": query, "reference": [], "emphasis": ""}
    )
    spec, faults = complete_chart(planned, held, "block", forms=PageForms.of(held))
    assert not faults
    with pytest.raises(UnresolvableReference, match="'unanswered' holds no one number per arm"):
        resolve_chart(spec, block_look(held, fact_arms(held)), title="", forms=PageForms.of(held))


# --- a row standing for no answer (#1238's class) -------------------------------------------------------------------

#: A candidate call to a fallback model, sent and never answered: a row no answered call reached.
_FALLBACK = "m-fallback"
_UNANSWERED_CANDIDATE = RoleUsage(role="candidate", model=_FALLBACK, unanswered_lost=1)
_ANSWERED_CANDIDATE = RoleUsage(
    role="candidate",
    model=FAST,
    served_model=f"{FAST}-2026",
    cost_usd=0.002,
    call_count=1,
    prompt_tokens=10,
    completion_tokens=5,
)


#: A judge row counting one call lost unanswered and nothing else: the row the owner's rule says answered nothing.
_LOST: dict[str, Any] = {"role": "judge", "model": "j", "unanswered_lost": 1}


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        # The unanswered clause: either count, a zero included, makes the row one of unanswered calls; neither is a
        # row that counted nothing, which is no row of unanswered calls.
        (_LOST, True),
        ({"role": "judge", "model": "j", "unanswered_retried": 1}, True),
        ({"role": "judge", "model": "j", "unanswered_retried": 0}, True),
        ({"role": "judge", "model": "j", "unanswered_retried": 1, "unanswered_lost": 2}, True),
        ({"role": "judge", "model": "j"}, False),
        # Each answered clause alone, beside a counted unanswered call: the row answered something.
        ({**_LOST, "call_count": 1}, False),
        ({**_LOST, "cost_usd": 0.0}, False),
        ({**_LOST, "prompt_tokens": 1}, False),
        ({**_LOST, "completion_tokens": 1}, False),
        ({**_LOST, "reasoning_tokens": 1}, False),
        ({**_LOST, "provider_unit": "credits", "provider_units": 1}, False),
        # An external row is held to the same rule: its units are its answered clause.
        ({"role": "external", "provider": "p", "unanswered_lost": 1}, True),
        (
            {
                "role": "external",
                "provider": "p",
                "provider_unit": "credits",
                "provider_units": 3,
                "unanswered_lost": 1,
            },
            False,
        ),
    ],
    ids=[
        "lost",
        "retried",
        "retried-zero",
        "retried-and-lost",
        "counted-nothing",
        "calls-beside",
        "dollars-beside",
        "prompt-tokens-beside",
        "completion-tokens-beside",
        "reasoning-tokens-beside",
        "units-beside",
        "external-lost",
        "external-units-beside",
    ],
)
def test_a_row_answers_nothing_only_where_it_counts_unanswered_calls_and_nothing_answered(
    fields: dict[str, Any], expected: bool
) -> None:
    """``RoleUsage.answers_nothing`` is the one statement of the rule, each clause varied alone; a reading of the row
    carries its value, so a reader of readings reads a row as no answer exactly where the stored row says."""
    row = RoleUsage.model_validate(fields)
    assert row.answers_nothing is expected
    assert UsageReading.of(row, SharedValues()).answers_nothing is expected


def test_a_row_that_answered_nothing_names_no_call_whose_served_model_went_unrecorded() -> None:
    """It names no served model because no model served it: read as a call whose response named none, it made a
    contestant answered by one named model read ``unrecorded``, and a cell split on it apart."""
    reading = pooled_served_models([make_eval_result(usage=[_ANSWERED_CANDIDATE, _UNANSWERED_CANDIDATE])])
    assert reading is not None
    assert (reading.served_models, reading.n_unrecorded, reading.state) == ([f"{FAST}-2026"], 0, "one")
    assert pooled_served_models([make_eval_result(usage=[_UNANSWERED_CANDIDATE])]) is None


def test_a_model_that_answered_nothing_is_no_candidate_a_judge_shares() -> None:
    run = make_eval_run(candidate_model=FAST, judge_model=_FALLBACK)
    result = make_eval_result(usage=[_ANSWERED_CANDIDATE, _UNANSWERED_CANDIDATE])
    assert self_judging_models(run, [result]) == []


def test_a_simulator_row_that_answered_nothing_leaves_its_results_settings_verified() -> None:
    """It sent no request a reply came back under, so it carries no settings: read as a row that did not report
    them, it made the result's settings unverified."""
    run = make_eval_run(simulator_model="vendor/sim", simulator_asked_settings=JUDGE_REQUEST_SETTINGS)
    answered = RoleUsage(role="simulator", model="vendor/sim", call_count=1, sent_settings=[JUDGE_REQUEST_SETTINGS])
    unanswered = RoleUsage(role="simulator", model="vendor/sim", unanswered_lost=1)
    result = make_eval_result(eval_run_id=run.id, rubric_scores=[], usage=[answered, unanswered])
    readings = request_settings_readings([run], [result])
    assert [(reading.role, reading.state) for reading in readings] == [("simulator", "as_asked")]
    # The rig a run's results fall in reads the same rows by the same rule: the settings it sent, never undecidable.
    alone = make_eval_result(eval_run_id=run.id, rubric_scores=[], usage=[answered])
    sent = SHARED_CORE.read_all(run, [result])["simulator_sent_settings"]
    assert sent is not None and sent == SHARED_CORE.read_all(run, [alone])["simulator_sent_settings"]


def test_a_row_that_answered_nothing_leaves_no_token_count_unreported() -> None:
    """No call it counts was answered, so no token was spent to report: read as an unreported count, it made the
    run's token sums read "at least"."""
    unanswered = RoleUsage(role="judge", model="j", unanswered_lost=1)
    tokens = one_batch_bundle(
        [make_eval_result(usage=[_ANSWERED_CANDIDATE, unanswered])], profile=toyhost_profile()
    ).telemetry.tokens
    assert tokens is not None
    assert (tokens.prompt_tokens, tokens.completion_tokens, tokens.n_results_tokens_unreported) == (10, 5, 0)


#: A lever read off the model a role ran on, as a host's research model is.
_OBSERVED = "retrieval.rerank_depth"


def test_a_model_that_answered_nothing_is_no_level_of_the_lever_its_role_is_read_on() -> None:
    """A lever recovered from a role is read off the models the role spent tokens on: a call to a model that never
    answered is no observation of it, so the run neither exercises the lever nor inherits that model as its level."""
    profile = replace(toyhost_profile(tunable_retrieval=True), observed_model_levers={_OBSERVED: "inner_agent"})
    run = make_eval_run()
    lost = RoleUsage(role="inner_agent", model="research-model-1", unanswered_lost=1)
    results = [make_eval_result(eval_run_id=run.id, usage=[_ANSWERED_CANDIDATE, lost])]
    assert exercised([run], {run.id: results}, profile=profile) == {_OBSERVED: {run.id: 0}}
    assert _OBSERVED not in effective_config(run, results, profile=profile)


async def test_a_row_that_answered_nothing_is_no_silent_row_a_statement_of_the_served_model_covers() -> None:
    """A statement of the model that answered covers the candidate rows whose responses named none: a row no model
    answered names none for that reason, so a run whose every answered row named its model holds nothing to state."""
    run = run_of(FAST)

    def stored() -> EvalStorage:
        storage = EvalStorage(InMemoryDocumentStore())
        storage.save_eval_run(run)
        storage.save_eval_result(result_of(run, 0, usage=[_ANSWERED_CANDIDATE, _UNANSWERED_CANDIDATE]))
        return storage

    storage = off_loop(stored)
    with pytest.raises(ValidationFailedError, match="holds no candidate usage row that names no served model"):
        await attest_served_model(
            toyhost_host(storage=storage), [run.id], run.scope_id, f"{FAST}-2026", attested_by="person:host"
        )


def test_a_row_that_answered_nothing_leaves_the_arms_pricing_counted() -> None:
    """It counted no answered call and priced nothing: read as a production row that left its calls uncounted, it
    made the arm's priced share not counted."""
    usage = [[_ANSWERED_CANDIDATE, _UNANSWERED_CANDIDATE]] * CASES
    _, bundle, _ = aspect_bundle(_trials(usage))
    (fast,) = [row for row in bundle.scoreboard if row.unanswered]
    pricing = fast.pricing
    assert pricing is not None and (pricing.calls, pricing.priced_calls) == (CASES, CASES)


def test_a_row_that_answered_nothing_adds_no_answered_call_and_leaves_the_share_counted() -> None:
    """Its answered calls are none, never uncounted: read as a row that did not count them, it left the arm's answered
    calls and its share of every call sent not counted."""
    lost = RoleUsage(role="candidate", model=_FALLBACK, unanswered_retried=0, unanswered_lost=1)
    _, bundle, _ = aspect_bundle(_trials([[_counting(retried=0, lost=0), lost]] * CASES))
    (fast,) = [row for row in bundle.scoreboard if row.unanswered]
    (entry,) = fast.unanswered
    assert (entry.calls, entry.lost, entry.results_uncounted) == (3 * CASES, CASES, 0)
    assert entry.share == pytest.approx(CASES / (3 * CASES + CASES))
