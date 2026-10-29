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
- **not counted is never zero**: where a role counted only one kind, the bare count and the share are refused as not
  counted, and the kind it counted reads alone.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from packages.evals.tests.factories import DEFAULT_KIND, make_eval_run, make_template, make_test_case
from packages.evals.tests.fixtures.report_plans import dataset
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.run_facts_support import ASPECT, CASES, aspect_bundle, campaign_over, result_of, run_of
from threetears.evals.analysis.analyst.facts import campaign_overview
from threetears.evals.analysis.errors import UnresolvableReference, UnstatedFigure
from threetears.evals.analysis.report.figure_keys import format_figure, resolve
from threetears.evals.analysis.report.plan_page.build import plan_page
from threetears.evals.analysis.report_input import ReportInput, Unanswered
from threetears.evals.analysis.report_input_build import report_input
from threetears.evals.kernel.candidate_kind import CandidateOutput, CandidateTelemetry, CellSink, VariantConfig
from threetears.evals.kernel.cassettes import CellCassettes
from threetears.evals.kernel import RoleUsageLedger
from threetears.evals.run.runner import RunnerOptions, execute_run
from threetears.evals.schema.models import EvalResult, EvalRun, JudgedArtifact, RoleUsage

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

    answered, lost = ledger.rows()

    assert (answered.call_count, answered.cost_usd, answered.unanswered_retried, answered.unanswered_lost) == (
        3,
        0.02,
        1,
        None,
    ), "a retried count sits beside the answered calls and moves none of their spend"
    assert answered.unanswered_reasons == {"refused": 1}
    assert (lost.served_model, lost.call_count, lost.cost_usd, lost.unanswered_retried) == (None, None, None, None)
    assert (lost.unanswered_lost, lost.unanswered_reasons) == (2, {"timeout": 2}), "a counted zero adds to the count"


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
    assert format_figure(resolve(f"calls.unanswered.{FAST}.lost", held)) == str(CASES - 1)


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
    assert format_figure(resolve(key, held)) == reads
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
    assert format_figure(resolve(f"calls.unanswered.{FAST}", widened)) == "10"


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


def _flag_changes(unanswered: Unanswered) -> list[tuple[str | None, str | None]]:
    """On the Kairo reference plan, whose options table has a cost column: each cost flag that changes when its first
    arm is given ``unanswered``, before and after."""
    plan = dataset.load_plan(dataset.HERE / "r5" / "anchor.plan.json")
    held = dataset.report_input_of("r5")
    arm = next(iter(held.scoreboard))
    counted = held.scoreboard[arm].model_copy(update={"unanswered": {"candidate": unanswered}})
    flagged = plan_page(plan, held.model_copy(update={"scoreboard": {**held.scoreboard, arm: counted}}))
    unflagged = plan_page(plan, held)
    flags = [row["cost_per_1k"].flag for row in flagged.options.table.rows]
    before = [row["cost_per_1k"].flag for row in unflagged.options.table.rows]
    return [(old, new) for old, new in zip(before, flags, strict=True) if old != new]


@pytest.mark.parametrize(
    ("unanswered", "said"),
    [
        (Unanswered(retried=0, lost=3, calls=297, share=0.01), "3 calls unanswered"),
        (Unanswered(retried=0, lost=1, calls=299, share=1 / 300), "1 call unanswered"),
        (Unanswered(retried=None, lost=2, calls=298), "2 calls lost unanswered"),
    ],
    ids=["several", "one", "lost-only"],
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
    assert format_figure(resolve(f"calls.unanswered.{FAST}.lost", held)) == str(CASES)
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
