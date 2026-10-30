"""Which ``EvalResult`` attributes each reader family reads, measured and pinned.

The compact per-result reading (``ResultReading``) carries exactly the attributes the readers access. This file
measures that set: each reader family runs over the oracle's campaigns with every result it reads served through
the access recorder (:mod:`~packages.evals.tests.result_access_recorder`), and the attribute paths it read are
compared with the family's pinned set (:data:`PINNED`), less what that campaign leaves unread (:data:`UNREAD`).

- **bundle:** ``assemble_context_bundle`` (through ``assemble_campaign_bundle``, every aspect) and ``report_read``
  code-only, in every form with both trial details;
- **aggregates:** ``pivot``, ``history``, ``frontier``, ``program_budget``, ``orphaned_runs``, ``comparison_sets``,
  and the regression watch's ``scope_history`` and judged tiers (checked on a step that regressed, so the tiers are
  read);
- **planning:** ``detectable_effects``, which reads its history through ``planning_readings``.

The campaigns: the production-shaped corpus at ``scale=1`` and its two-aspect variant, the toy host's corpus campaign,
the courier's planner bake-off (read back from the oracle's capture), and two built here. The **enriched** campaign is
the corpus at four cases a run with every record list filled (re-judges, repeats, second judgings, background
deliveries, preconditions, world events, simulator usage, latency) and every run's asked request settings equal to
those sent, so each reader's walk over those items, and the settings tally's equal branch, runs. The **regressed**
campaign is the corpus with one regressed step added, on which the regression watch reads its judged tiers.

**A red names the attribute and where it was read.** A family reading an attribute outside its pinned set means the
reading must add it, deliberately; a pinned attribute no campaign reads any more means the reading carries one it
need not. Either way the pin moves with the reading, never alone.

**Judge prose is pinned by where it is read** (:data:`PROSE_SITES`): a score's reasoning, the judge's top-level
reasoning and the judge's "cannot tell" notes, each as a read of its text or as a truth test (``bool(...)``), at
``module function``. A read at a site not pinned fails. The bundle's trials section reads prose only of the trials
the bundle lists.

**One test per family and campaign**, since a recorded read of a production-shaped campaign is minutes of work, and
the recorder pays for a frame read on every attribute read.

**The recorder changes nothing a reader computes**: every family's output under the recorder is compared, byte for
byte, with the same calls' output without it.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from threetears.evals.analysis import comparison_sets, orphaned_runs, program_budget
from threetears.evals.analysis import service as analysis_service
from threetears.evals.analysis.reporting import METRIC_COMPOSITE, METRIC_OUTCOME, METRIC_TRANSCRIPT
from threetears.evals.analysis.service import assemble_campaign_bundle
from threetears.evals.kernel.host import EvalHost
from threetears.evals.ops import report_read
from threetears.evals.ops.lenses import detectable_effects, scope_frontier, scope_history, scope_pivot
from threetears.evals.ops.regressions import RegressionAlert, RegressionWatch
from threetears.evals.run.reads import list_runs
from threetears.evals.schema.models import (
    TRANSCRIPT_DIM_ID,
    AsyncDelivery,
    ClientRequestSettings,
    EvalResult,
    EvalRun,
    JudgeRepeat,
    JudgeRescore,
    LatencyMetrics,
    PreconditionOutcome,
    RepeatedScore,
    RubricScore,
    SecondJudge,
    SecondJudgeScore,
    SecondJudging,
    WorldEvent,
)
from packages.evals.tests.factories import make_eval_result, memory_storage
from packages.evals.tests.fixtures.courierhost import COURIER_SCOPE, COURIER_TEMPLATE_ID, courier_cases
from packages.evals.tests.fixtures.toyhost.run import toyhost_template
from packages.evals.tests.hot_path_support import (
    ARM_MODELS,
    CASES_PER_RUN,
    HOT_PATH_SCOPE,
    K_RUNS,
    campaign_id_of,
    courierhost_from,
    hot_path_corpus,
)
from packages.evals.tests.ops_support import TOYHOST_SCOPE, ops_fixture
from packages.evals.tests.result_access_recorder import ResultAccess, recording, recording_results

#: The instant every code-only page is assembled at (the oracle's).
REPORT_NOW = "2026-10-30T09:00:00+00:00"

#: The courier's campaign as the oracle captured it: a run mints its ids, so it is read back from the capture.
COURIER_CAPTURE = Path(__file__).parent / "fixtures" / "hot_path_oracle" / "inputs" / "courierhost.json"

#: The pages a report is read as, ``(format, trials)``. A page reads nothing of a result that its bundle's assembly
#: did not (every form and both trial details were measured to read the same set over the scaled campaign), so two
#: pages cover ``report_read``'s own path at a fraction of the cost.
PAGES = (("markdown", "all"), ("json", "table"))

#: The scope of the campaign built for the regression watch: the scaled corpus with one regressed step added.
REGRESSED_SCOPE = "hot-path-regressed"

#: The scope of the campaign whose results carry every record list the production-shaped corpus leaves empty.
ENRICHED_SCOPE = "hot-path-enriched"

#: The cases each run of the enriched campaign measures: small, since what it adds is fields, not rows.
ENRICHED_CASES = 4

#: What the enriched campaign's simulated user was asked to send, and sent.
SIMULATOR_SENT = ClientRequestSettings(max_tokens=2048, reasoning_effort=None, reasoning_max_tokens=None)

#: The measures the history and the regression watch are read on: every one the history series (the composite, both
#: judged axes, spend and latency).
MEASURES = (METRIC_COMPOSITE, METRIC_TRANSCRIPT, METRIC_OUTCOME, "cost_usd", "total_ms")

#: The pivot's measures: every metric it accepts (it pivots no latency).
PIVOT_MEASURES = (METRIC_COMPOSITE, METRIC_TRANSCRIPT, METRIC_OUTCOME, "goal_state", "cost_usd", "score")


@dataclass(frozen=True)
class Campaign:
    """One campaign a family is read over: its host, its id and scope, and the launch its planning block plans."""

    host: EvalHost
    campaign_id: str
    scope_id: str
    template_id: str
    n_cases: int
    k_runs: int
    n_arms: int


@functools.cache
def campaign(name: str) -> Campaign:
    """The campaign ``name``, built or loaded once per process."""
    if name in ("scaled", "scaled_two_aspects"):
        corpus = hot_path_corpus(scale=1, two_aspects=name == "scaled_two_aspects")
        return Campaign(
            corpus.host,
            corpus.campaign.id,
            HOT_PATH_SCOPE,
            corpus.template_ids[0],
            CASES_PER_RUN,
            K_RUNS,
            len(ARM_MODELS),
        )
    if name == "regressed":
        corpus = hot_path_corpus(scale=1, scope_id=REGRESSED_SCOPE)
        add_a_regressed_step(corpus.host, REGRESSED_SCOPE)
        return Campaign(
            corpus.host,
            corpus.campaign.id,
            REGRESSED_SCOPE,
            corpus.template_ids[0],
            CASES_PER_RUN,
            K_RUNS,
            len(ARM_MODELS),
        )
    if name == "enriched":
        corpus = hot_path_corpus(scale=1, scope_id=ENRICHED_SCOPE, cases_per_run=ENRICHED_CASES)
        enrich(corpus.host, ENRICHED_SCOPE)
        return Campaign(
            corpus.host,
            corpus.campaign.id,
            ENRICHED_SCOPE,
            corpus.template_ids[0],
            ENRICHED_CASES,
            K_RUNS,
            len(ARM_MODELS),
        )
    if name == "toyhost":
        fixture = ops_fixture()
        return Campaign(fixture.host.eval_host, fixture.campaign.id, TOYHOST_SCOPE, toyhost_template().id, 12, 3, 2)
    if name == "courierhost":
        documents: list[dict[str, Any]] = json.loads(COURIER_CAPTURE.read_text())
        return Campaign(
            courierhost_from(documents),
            campaign_id_of(documents),
            COURIER_SCOPE,
            COURIER_TEMPLATE_ID,
            len(courier_cases()),
            2,
            2,
        )
    raise KeyError(name)


CAMPAIGNS = ("scaled", "scaled_two_aspects", "toyhost", "courierhost", "enriched")


def enrich(host: EvalHost, scope_id: str) -> None:
    """Give every result of the scope each record list the production-shaped corpus leaves empty, and every run the
    request settings it asked for, so each reader's walk over those records runs.

    Each run asks its judge for exactly what its scores say was sent, and its simulated user for
    :data:`SIMULATOR_SENT`, which a simulator usage row reports sending: the settings tally's equal branch. Each
    result gains its latency, a precondition, a world event and two background deliveries (one delivered, one
    failed and substituted), and every fourth is marked read under concurrency (its latency withheld). Each judged
    result also gains a re-judge, a repeat of every score under the result's own judge configs (so each pairs with
    its first answer), a repeat round that did not answer (a can't-tell or an error, alternately), and a second
    judge's pass.
    """
    storage = host.storage
    for run in list_runs(host, scope_id):
        results = storage.query_eval_results_by_run(run.id, scope_id)
        sent = next((score.judge_sent_settings for result in results for score in result.judge_scores()), None)
        storage.save_eval_run(
            run.model_copy(update={"judge_asked_settings": sent, "simulator_asked_settings": SIMULATOR_SENT})
        )
        for index, result in enumerate(results):
            storage.save_eval_result(result.model_copy(update=_enriched(result, index)))


def _enriched(result: EvalResult, index: int) -> dict[str, Any]:
    """The fields :func:`enrich` gives ``result``, the ``index``-th of its run."""
    judge_row = next(row for row in result.usage if row.role == "judge")
    simulator_row = judge_row.model_copy(
        update={"role": "simulator", "model": "simulator-model", "sent_settings": [SIMULATOR_SENT]}
    )
    fields: dict[str, Any] = {
        "usage": [*result.usage, simulator_row],
        # Every fourth result was read while other cells ran beside it, so its latency is withheld (contention).
        "covariates": {**result.covariates, "execution_mode": "concurrent"} if index % 4 == 0 else result.covariates,
        "latency": LatencyMetrics(
            total_ms=4_000.0 + index, llm_ms=3_000.0, tool_ms=400.0, async_wait_ms=200.0, judge_ms=300.0
        ),
        "precondition_outcomes": [
            PreconditionOutcome(expression="state.ticket.status == 'open'", presumes="an open ticket", held=True)
        ],
        "world_events": [
            WorldEvent(
                kind="event",
                dimension="payment_hold",
                condition="refund_posted",
                caused_by="world",
                event="posting-hold",
                turn=2,
            )
        ],
        "async_deliveries": [
            AsyncDelivery(
                tool="lookup_order",
                status="delivered",
                model="scout-model",
                served_model="scout-model-2026-08",
                acknowledged_turn=1,
                delivered_turn=2,
                elapsed_ms=900.0 + index,
                delivered_items=3,
                substituted=False,
                input_tokens=1_200,
                output_tokens=300,
                llm_calls=1,
                cost_usd=0.002,
                price_source="provider",
            ),
            AsyncDelivery(
                tool="issue_refund", status="failed", elapsed_ms=150.0, error="the refund API refused", substituted=True
            ),
        ],
    }
    scores = result.judge_scores()
    if not scores:
        return fields
    first = scores[0]
    fields["judge_rescores"] = [
        JudgeRescore(
            dims=[first.dim],
            prior_judge_error="the judge's reply did not parse as the scoring schema",
            scores={first.dim: first.score},
            judge_model="judge-model",
            usage=[judge_row],
            cost_usd=0.004,
        )
    ]
    unanswered = RepeatedScore(
        dim=first.dim,
        scale=first.scale,
        first_score=first.score,
        first_served_model=first.served_model,
        first_judge_config_id=result.judge_config_ids.get(first.dim),
        first_judge_temperature=first.judge_temperature,
        first_judge_sent_settings=first.judge_sent_settings,
        **(
            {"cannot_tell": "the transcript ends before the refund is decided"}
            if index % 2
            else {"error": "the judge's repeat timed out"}
        ),
    )
    fields["judge_repeats"] = [
        JudgeRepeat(
            judge_model="judge-model",
            judge_config_ids=dict(result.judge_config_ids),
            scores=[
                RepeatedScore(
                    dim=score.dim,
                    scale=score.scale,
                    first_score=score.score,
                    first_served_model=score.served_model,
                    first_judge_config_id=result.judge_config_ids.get(score.dim),
                    first_judge_temperature=score.judge_temperature,
                    first_judge_sent_settings=score.judge_sent_settings,
                    repeat=score.model_copy(update={"score": max(1, score.score - index % 2)}),
                )
                for score in scores
            ],
        ),
        JudgeRepeat(judge_model="judge-model", judge_config_ids=dict(result.judge_config_ids), scores=[unanswered]),
    ]
    fields["judge_seconds"] = [
        SecondJudging(
            pass_id="second-pass",
            judge=SecondJudge(model="second-judge-model"),
            sample_fraction=1.0,
            sample_seed=0,
            scores=[
                SecondJudgeScore(
                    dim=score.dim,
                    scale=score.scale,
                    first_score=score.score,
                    first_served_model=score.served_model,
                    first_judge_config_id=result.judge_config_ids.get(score.dim),
                    first_judge_temperature=score.judge_temperature,
                    first_judge_sent_settings=score.judge_sent_settings,
                    second=score.model_copy(
                        update={"score": min(5, score.score + index % 2), "served_model": "second-judge-model"}
                    ),
                )
                for score in scores
            ],
        )
    ]
    return fields


def add_a_regressed_step(host: EvalHost, scope_id: str) -> None:
    """Repeat the control arm's latest run once more, later, with every judged score at the scale's floor.

    No corpus the oracle reads has a step its history reads ``regressed``, and the regression watch reads its judged
    tiers only on one: this is that step, every other field of each result as the run it repeats recorded it.
    """
    storage = host.storage
    control = max(
        (run for run in list_runs(host, scope_id) if run.candidate_model == ARM_MODELS[0]),
        key=lambda run: run.created_at,
    )
    repeat = control.model_copy(update={"id": f"{control.id}--regressed", "created_at": "2026-09-03T08:00:00+00:00"})
    storage.save_eval_run(repeat)
    for result in storage.query_eval_results_by_run(control.id, scope_id):
        floored = {
            "rubric_scores": [score.model_copy(update={"score": 1}) for score in result.rubric_scores],
            "transcript_score": result.transcript_score and result.transcript_score.model_copy(update={"score": 1}),
            "outcome_score": result.outcome_score and result.outcome_score.model_copy(update={"score": 1}),
        }
        storage.save_eval_result(
            result.model_copy(
                update={"id": f"{result.id}--regressed", "eval_run_id": repeat.id, "has_trace": False, **floored}
            )
        )


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _lister(host: EvalHost) -> Callable[..., list[EvalRun]]:
    return lambda scope_id, **kwargs: list_runs(host, scope_id, **kwargs)


def read_bundle(read: Campaign) -> list[str]:
    """The bundle family: every aspect's bundle and its fingerprint, the campaign's frame, then the code-only pages."""
    bundle = assemble_campaign_bundle(read.host, read.campaign_id, read.scope_id)
    outputs = [_json(aspect.to_dict()) + aspect.fingerprint() for aspect in bundle.aspects]
    outputs.append(_json(bundle.frame()) + bundle.fingerprint())
    for form, trials in PAGES:
        outputs.append(report_read(read.host, read.campaign_id, read.scope_id, format=form, trials=trials).body)
    return outputs


def regressed_steps(read: Campaign) -> list[tuple[str, str]]:
    """Each tracked measure's first run whose step into it reads ``regressed``, as ``(measure, run_id)``.

    Found off the history itself, outside any recording, so the watch below is checked where its judged tiers are read.
    """
    found = []
    for measure in MEASURES:
        history = scope_history(read.host, read.scope_id, metric=measure)
        step = next(
            (
                point.run_id
                for series in history.series
                for point in series.points
                if point.regression is not None and point.regression.label == "regressed"
            ),
            None,
        )
        if step is not None:
            found.append((measure, step))
    return found


def read_aggregates(read: Campaign, steps: list[tuple[str, str]], *, lenses: bool = True) -> list[str]:
    """The aggregates family: the regression watch on ``steps``, then each lens over the campaign's scope (unless
    ``lenses`` is off)."""
    host, scope_id = read.host, read.scope_id
    outputs = []
    for measure, run_id in steps:
        watch = RegressionWatch(host=host, sink=_unused_sink, measures=(measure,))
        outputs.append(_json([alert.model_dump(mode="json") for alert in watch.check(run_id, scope_id)]))
    if not lenses:
        return outputs
    for measure in PIVOT_MEASURES:
        pivoted = scope_pivot(host, scope_id, row_factor="model", column_factor="test_case_id", metric=measure)
        outputs.append(pivoted.model_dump_json())
    for measure in MEASURES:
        outputs.append(scope_history(host, scope_id, metric=measure).model_dump_json())
    outputs.append(scope_frontier(host, scope_id).model_dump_json())
    outputs.append(scope_frontier(host, scope_id, bar=0.5).model_dump_json())
    outputs.append(_json(program_budget(host.storage, scope_id, list_runs=_lister(host))))
    outputs.append(_json(orphaned_runs(host.storage, scope_id, list_runs=_lister(host))))
    outputs.append(_json(comparison_sets(host.storage, scope_id, list_runs=_lister(host), profile=host.profile)))
    outputs.append(
        _json(
            comparison_sets(
                host.storage, scope_id, list_runs=_lister(host), profile=host.profile, campaign_id=read.campaign_id
            )
        )
    )
    return outputs


async def _unused_sink(alert: RegressionAlert) -> None:
    """The watch's sink: never called, since the watch is only asked to ``check``."""
    raise AssertionError(f"check delivers nothing, yet {alert} was delivered")


def read_planning(read: Campaign) -> list[str]:
    """The planning family: the block a launch estimate carries."""
    block = detectable_effects(
        read.host, read.scope_id, read.template_id, n_cases=read.n_cases, k_runs=read.k_runs, n_arms=read.n_arms
    )
    return [block.model_dump_json()]


def recorded(read: Campaign, family: Callable[..., list[str]], *args: Any) -> tuple[ResultAccess, list[str], list[str]]:
    """``family`` over ``read``: what it read of the results, its outputs recorded, and its outputs unrecorded."""
    plain = family(read, *args)
    with recording_results(), recording() as access:
        outputs = family(read, *args)
    return access, outputs, plain


#: Each family's reader, and the campaigns it is read over (the planning block over those the oracle plans for).
FAMILIES: dict[str, tuple[Callable[..., list[str]], tuple[str, ...]]] = {
    "bundle": (read_bundle, CAMPAIGNS),
    "aggregates": (read_aggregates, (*CAMPAIGNS, "regressed")),
    "planning": (read_planning, ("scaled", "toyhost", "courierhost", "enriched")),
}


#: Every ``(family, campaign)`` read, one test each: a recorded read of a production-shaped campaign is minutes of work.
READS = tuple((family, name) for family, (_, names) in FAMILIES.items() for name in names)


@dataclass(frozen=True)
class Measured:
    """One family read over one campaign: what it read, whether the recorder moved its output, how many regressed
    steps the regression watch was checked on, and the results its bundles list as trials (the bundle family only)."""

    access: ResultAccess
    moved: bool
    regressed_steps: int
    listed_trials: frozenset[str] | None


def listed_trials(outputs: list[str], n_aspects: int) -> frozenset[str]:
    """The ids of the results the bundle family's aspect bundles list as trials (its first ``n_aspects`` outputs, each
    an aspect's JSON then its fingerprint)."""
    decoder = json.JSONDecoder()
    return frozenset(
        str(row["result_id"])
        for output in outputs[:n_aspects]
        for row in decoder.raw_decode(output)[0].get("trials", [])
    )


@functools.cache
def measured(family: str, name: str) -> Measured:
    """``family`` read over the campaign ``name``, recorded, once per process."""
    reader, _ = FAMILIES[family]
    read = campaign(name)
    args: tuple[Any, ...] = ()
    steps: list[tuple[str, str]] = []
    if family == "aggregates":
        steps = regressed_steps(read)
        args = (steps,)
    family_reader = functools.partial(reader, lenses=False) if name == "regressed" else reader
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(analysis_service, "utc_now_iso", lambda: REPORT_NOW)
        access, outputs, plain = recorded(read, family_reader, *args)
    # The bundle family's outputs are each aspect's bundle, then the campaign's frame, then each page.
    listed = listed_trials(outputs, len(outputs) - 1 - len(PAGES)) if family == "bundle" else None
    return Measured(access, outputs != plain, len(steps), listed)


def prose_pins(sites: dict[str, frozenset[str]]) -> frozenset[str]:
    """Each pinned prose read as ``"<path> at <module function>"``."""
    return frozenset(f"{path} at {where}" for path, wheres in sites.items() for where in wheres)


def _where(site: str) -> str:
    """A call site as a prose pin names it: its module and function, without the line that moves with every edit."""
    location, function = site.split(" ", 1)
    return f"{location.rsplit(':', 1)[0]} {function}"


#: Every attribute path each family reads of a result, over every campaign it is read over (see the module
#: docstring for the path syntax). Sizes: bundle 187, aggregates 100, planning 124.
#:
#: The measure walk reads each model type's declared fields, never the judge's prose
#: (``analysis/bundle/measures.py``), so no family reads ``judge_reasoning``, ``world_events`` or ``kind_payload``
#: through it, and planning reads no score's ``reasoning``.
#:
#: Every record list a result carries is filled in the enriched campaign, so its items' fields are measured, with
#: two exceptions: ``judge_rescores[].usage[]`` (no family reads a re-judge's own usage rows, so the reading leaves
#: them out) and ``judge_cut_short{}`` (carried whole by the reading; no campaign
#: carries a cut-short judge answer, so its items' fields are unmeasured; one item read is known by reading the code,
#: ``.sent_settings`` at ``analysis/bundle/cell_reads.py``).
PINNED: dict[str, frozenset[str]] = {
    "bundle": frozenset(
        {
            "async_deliveries",
            "async_deliveries[].acknowledged_turn",
            "async_deliveries[].cost_usd",
            "async_deliveries[].delivered_items",
            "async_deliveries[].delivered_turn",
            "async_deliveries[].elapsed_ms",
            "async_deliveries[].error",
            "async_deliveries[].input_tokens",
            "async_deliveries[].llm_calls",
            "async_deliveries[].model",
            "async_deliveries[].model_copy()",
            "async_deliveries[].output_tokens",
            "async_deliveries[].price_source",
            "async_deliveries[].reasoning_tokens",
            "async_deliveries[].requested_by",
            "async_deliveries[].served_model",
            "async_deliveries[].status",
            "async_deliveries[].substituted",
            "async_deliveries[].summary",
            "async_deliveries[].tool",
            "candidate_error",
            "candidate_instance_id",
            "candidate_kind",
            "cost_roles",
            "cost_usd",
            "covariates",
            "doc_type",
            "eval_run_id",
            "goal_state_outcomes",
            "goal_state_outcomes[].detail",
            "goal_state_outcomes[].expression",
            "goal_state_outcomes[].passed",
            "has_trace",
            "host_measures",
            "id",
            "identity_version",
            "infra_error",
            "judge_cannot_tell",
            "judge_cannot_tell_boundary",
            "judge_config_ids",
            "judge_cut_short",
            "judge_error",
            "judge_model",
            "judge_repeats",
            "judge_repeats[].judge_config_ids",
            "judge_repeats[].scores",
            "judge_repeats[].scores[].cannot_tell",
            "judge_repeats[].scores[].dim",
            "judge_repeats[].scores[].first_judge_config_id",
            "judge_repeats[].scores[].first_judge_sent_settings",
            "judge_repeats[].scores[].first_judge_sent_settings.model_dump()",
            "judge_repeats[].scores[].first_judge_temperature",
            "judge_repeats[].scores[].first_score",
            "judge_repeats[].scores[].first_served_model",
            "judge_repeats[].scores[].repeat",
            "judge_repeats[].scores[].repeat.judge_sent_settings",
            "judge_repeats[].scores[].repeat.judge_sent_settings.==",
            "judge_repeats[].scores[].repeat.judge_temperature",
            "judge_repeats[].scores[].repeat.score",
            "judge_repeats[].scores[].repeat.served_model",
            "judge_repeats[].scores[].scale",
            "judge_rescores",
            "judge_rescores[].cost_usd",
            "judge_score()",
            "judge_scores()",
            "judge_seconds",
            "judge_seconds[].judge",
            "judge_seconds[].judge.model",
            "judge_seconds[].judge.temperature",
            "judge_seconds[].judge_config_ids",
            "judge_seconds[].pass_id",
            "judge_seconds[].scores",
            "judge_seconds[].scores[].dim",
            "judge_seconds[].scores[].first_judge_config_id",
            "judge_seconds[].scores[].first_judge_sent_settings",
            "judge_seconds[].scores[].first_judge_sent_settings.model_dump()",
            "judge_seconds[].scores[].first_judge_temperature",
            "judge_seconds[].scores[].first_score",
            "judge_seconds[].scores[].first_served_model",
            "judge_seconds[].scores[].scale",
            "judge_seconds[].scores[].second",
            "judge_seconds[].scores[].second.judge_sent_settings",
            "judge_seconds[].scores[].second.judge_sent_settings.model_dump()",
            "judge_seconds[].scores[].second.score",
            "judge_seconds[].scores[].second.served_model",
            "k_iteration",
            "latency",
            "latency.async_wait_ms",
            "latency.judge_ms",
            "latency.llm_ms",
            "latency.tool_ms",
            "latency.total_ms",
            "lease_epoch",
            "metered_calls",
            "metered_calls_refused",
            "model",
            "model_copy()",
            "outcome_score",
            "outcome_score.axis",
            "outcome_score.criterion_fingerprint",
            "outcome_score.dim",
            "outcome_score.judge_sent_settings",
            "outcome_score.judge_sent_settings.==",
            "outcome_score.judge_sent_settings.max_tokens",
            "outcome_score.judge_sent_settings.model_dump()",
            "outcome_score.judge_sent_settings.reasoning_effort",
            "outcome_score.judge_sent_settings.reasoning_max_tokens",
            "outcome_score.judge_sent_settings.strict_output",
            "outcome_score.judge_sent_settings.unsent_reasoning_max_tokens",
            "outcome_score.judge_temperature",
            "outcome_score.label_key",
            "outcome_score.output_fingerprint",
            "outcome_score.reasoning",
            "outcome_score.scale",
            "outcome_score.score",
            "outcome_score.served_model",
            "phase_timings",
            "precondition_outcomes",
            "precondition_outcomes[].detail",
            "precondition_outcomes[].expression",
            "precondition_outcomes[].presumes",
            "rubric_scores",
            "rubric_scores[].axis",
            "rubric_scores[].clears()",
            "rubric_scores[].criterion_fingerprint",
            "rubric_scores[].dim",
            "rubric_scores[].judge_sent_settings",
            "rubric_scores[].judge_sent_settings.==",
            "rubric_scores[].judge_sent_settings.model_dump()",
            "rubric_scores[].judge_sent_settings.strict_output",
            "rubric_scores[].judge_temperature",
            "rubric_scores[].label_key",
            "rubric_scores[].normalized",
            "rubric_scores[].output_fingerprint",
            "rubric_scores[].reasoning",
            "rubric_scores[].scale",
            "rubric_scores[].score",
            "rubric_scores[].served_model",
            "runner_error",
            "schema_minor",
            "schema_version",
            "scope_id",
            "scored_at",
            "stop_cause",
            "subject_id",
            "superseded_spend_usd",
            "termination",
            "test_case_id",
            "transcript_score",
            "transcript_score.axis",
            "transcript_score.criterion_fingerprint",
            "transcript_score.dim",
            "transcript_score.judge_sent_settings",
            "transcript_score.judge_sent_settings.==",
            "transcript_score.judge_sent_settings.model_dump()",
            "transcript_score.judge_sent_settings.strict_output",
            "transcript_score.judge_temperature",
            "transcript_score.label_key",
            "transcript_score.output_fingerprint",
            "transcript_score.reasoning",
            "transcript_score.scale",
            "transcript_score.score",
            "transcript_score.served_model",
            "turns_delivered",
            "usage",
            "usage[].actor_id",
            "usage[].call_count",
            "usage[].completion_tokens",
            "usage[].cost_usd",
            "usage[].model",
            "usage[].price_source",
            "usage[].prompt_tokens",
            "usage[].provider",
            "usage[].provider_unit",
            "usage[].provider_units",
            "usage[].purpose",
            "usage[].reasoning_tokens",
            "usage[].role",
            "usage[].sent_settings",
            "usage[].sent_settings[].==",
            "usage[].sent_settings[].model_dump()",
            "usage[].sent_settings[].reasoning_max_tokens",
            "usage[].sent_settings[].strict_output",
            "usage[].served_model",
            "usage[].unanswered_lost",
            "usage[].unanswered_retried",
            "variant_key",
        }
    ),
    "aggregates": frozenset(
        {
            "async_deliveries",
            "async_deliveries[].model_copy()",
            "async_deliveries[].substituted",
            "candidate_error",
            "candidate_kind",
            "cost_roles",
            "cost_usd",
            "covariates",
            "eval_run_id",
            "goal_state_outcomes",
            "goal_state_outcomes[].expression",
            "goal_state_outcomes[].passed",
            "host_measures",
            "id",
            "identity_version",
            "infra_error",
            "judge_cannot_tell",
            "judge_cannot_tell_boundary",
            "judge_config_ids",
            "judge_cut_short",
            "judge_error",
            "judge_model",
            "judge_repeats",
            "judge_repeats[].judge_config_ids",
            "judge_repeats[].scores",
            "judge_repeats[].scores[].cannot_tell",
            "judge_repeats[].scores[].dim",
            "judge_repeats[].scores[].first_judge_config_id",
            "judge_repeats[].scores[].first_judge_sent_settings",
            "judge_repeats[].scores[].first_judge_sent_settings.model_dump()",
            "judge_repeats[].scores[].first_judge_temperature",
            "judge_repeats[].scores[].first_score",
            "judge_repeats[].scores[].first_served_model",
            "judge_repeats[].scores[].repeat",
            "judge_repeats[].scores[].repeat.judge_sent_settings",
            "judge_repeats[].scores[].repeat.judge_sent_settings.==",
            "judge_repeats[].scores[].repeat.judge_temperature",
            "judge_repeats[].scores[].repeat.score",
            "judge_repeats[].scores[].repeat.served_model",
            "judge_repeats[].scores[].scale",
            "judge_score()",
            "judge_scores()",
            "k_iteration",
            "latency",
            "latency.async_wait_ms",
            "latency.judge_ms",
            "latency.llm_ms",
            "latency.tool_ms",
            "latency.total_ms",
            "model",
            "model_copy()",
            "outcome_score",
            "outcome_score.criterion_fingerprint",
            "outcome_score.dim",
            "outcome_score.judge_sent_settings",
            "outcome_score.judge_sent_settings.model_dump()",
            "outcome_score.judge_temperature",
            "outcome_score.label_key",
            "outcome_score.output_fingerprint",
            "outcome_score.scale",
            "outcome_score.score",
            "outcome_score.served_model",
            "rubric_scores",
            "rubric_scores[].axis",
            "rubric_scores[].clears()",
            "rubric_scores[].criterion_fingerprint",
            "rubric_scores[].dim",
            "rubric_scores[].judge_sent_settings",
            "rubric_scores[].judge_sent_settings.model_dump()",
            "rubric_scores[].judge_temperature",
            "rubric_scores[].label_key",
            "rubric_scores[].normalized",
            "rubric_scores[].output_fingerprint",
            "rubric_scores[].scale",
            "rubric_scores[].score",
            "rubric_scores[].served_model",
            "scope_id",
            "scored_at",
            "subject_id",
            "termination",
            "test_case_id",
            "transcript_score",
            "transcript_score.criterion_fingerprint",
            "transcript_score.dim",
            "transcript_score.judge_sent_settings",
            "transcript_score.judge_sent_settings.model_dump()",
            "transcript_score.judge_temperature",
            "transcript_score.label_key",
            "transcript_score.output_fingerprint",
            "transcript_score.scale",
            "transcript_score.score",
            "transcript_score.served_model",
            "turns_delivered",
            "usage",
            "usage[].cost_usd",
            "usage[].role",
            "usage[].sent_settings",
            "usage[].sent_settings[].model_dump()",
            "usage[].served_model",
            "variant_key",
        }
    ),
    "planning": frozenset(
        {
            "async_deliveries",
            "async_deliveries[].acknowledged_turn",
            "async_deliveries[].cost_usd",
            "async_deliveries[].delivered_items",
            "async_deliveries[].delivered_turn",
            "async_deliveries[].elapsed_ms",
            "async_deliveries[].error",
            "async_deliveries[].input_tokens",
            "async_deliveries[].llm_calls",
            "async_deliveries[].model",
            "async_deliveries[].model_copy()",
            "async_deliveries[].output_tokens",
            "async_deliveries[].price_source",
            "async_deliveries[].reasoning_tokens",
            "async_deliveries[].requested_by",
            "async_deliveries[].served_model",
            "async_deliveries[].status",
            "async_deliveries[].substituted",
            "async_deliveries[].summary",
            "async_deliveries[].tool",
            "candidate_error",
            "candidate_instance_id",
            "candidate_kind",
            "cost_roles",
            "cost_usd",
            "covariates",
            "doc_type",
            "eval_run_id",
            "goal_state_outcomes",
            "goal_state_outcomes[].detail",
            "goal_state_outcomes[].expression",
            "goal_state_outcomes[].passed",
            "host_measures",
            "id",
            "identity_version",
            "infra_error",
            "judge_cannot_tell",
            "judge_cannot_tell_boundary",
            "judge_config_ids",
            "judge_cut_short",
            "judge_error",
            "judge_model",
            "k_iteration",
            "latency",
            "latency.async_wait_ms",
            "latency.judge_ms",
            "latency.llm_ms",
            "latency.tool_ms",
            "latency.total_ms",
            "lease_epoch",
            "metered_calls",
            "metered_calls_refused",
            "model",
            "model_copy()",
            "outcome_score",
            "outcome_score.axis",
            "outcome_score.criterion_fingerprint",
            "outcome_score.dim",
            "outcome_score.judge_sent_settings",
            "outcome_score.judge_sent_settings.model_dump()",
            "outcome_score.judge_temperature",
            "outcome_score.output_fingerprint",
            "outcome_score.scale",
            "outcome_score.score",
            "outcome_score.served_model",
            "phase_timings",
            "precondition_outcomes",
            "precondition_outcomes[].detail",
            "precondition_outcomes[].expression",
            "precondition_outcomes[].presumes",
            "rubric_scores",
            "rubric_scores[].axis",
            "rubric_scores[].criterion_fingerprint",
            "rubric_scores[].dim",
            "rubric_scores[].judge_sent_settings",
            "rubric_scores[].judge_sent_settings.model_dump()",
            "rubric_scores[].judge_temperature",
            "rubric_scores[].normalized",
            "rubric_scores[].output_fingerprint",
            "rubric_scores[].scale",
            "rubric_scores[].score",
            "rubric_scores[].served_model",
            "runner_error",
            "schema_minor",
            "schema_version",
            "scope_id",
            "scored_at",
            "stop_cause",
            "subject_id",
            "superseded_spend_usd",
            "termination",
            "test_case_id",
            "transcript_score",
            "transcript_score.axis",
            "transcript_score.criterion_fingerprint",
            "transcript_score.dim",
            "transcript_score.judge_sent_settings",
            "transcript_score.judge_sent_settings.model_dump()",
            "transcript_score.judge_temperature",
            "transcript_score.output_fingerprint",
            "transcript_score.scale",
            "transcript_score.score",
            "transcript_score.served_model",
            "turns_delivered",
            "usage",
            "usage[].actor_id",
            "usage[].call_count",
            "usage[].completion_tokens",
            "usage[].cost_usd",
            "usage[].model",
            "usage[].price_source",
            "usage[].prompt_tokens",
            "usage[].provider",
            "usage[].provider_unit",
            "usage[].provider_units",
            "usage[].purpose",
            "usage[].reasoning_tokens",
            "usage[].role",
            "usage[].sent_settings",
            "usage[].sent_settings[].model_dump()",
            "usage[].served_model",
            "usage[].unanswered_lost",
            "usage[].unanswered_retried",
            "variant_key",
        }
    ),
}

#: What each campaign leaves unread of its family's pinned set: a campaign whose results carry no latency, no
#: judge, no record list or no host measure reads none of those attributes; a campaign missing here reads them
#: all.
UNREAD: dict[str, dict[str, frozenset[str]]] = {
    "bundle": {
        "scaled": frozenset(
            {
                "async_deliveries[].acknowledged_turn",
                "async_deliveries[].cost_usd",
                "async_deliveries[].delivered_items",
                "async_deliveries[].delivered_turn",
                "async_deliveries[].elapsed_ms",
                "async_deliveries[].error",
                "async_deliveries[].input_tokens",
                "async_deliveries[].llm_calls",
                "async_deliveries[].model",
                "async_deliveries[].model_copy()",
                "async_deliveries[].output_tokens",
                "async_deliveries[].price_source",
                "async_deliveries[].reasoning_tokens",
                "async_deliveries[].requested_by",
                "async_deliveries[].served_model",
                "async_deliveries[].status",
                "async_deliveries[].substituted",
                "async_deliveries[].summary",
                "async_deliveries[].tool",
                "judge_repeats[].judge_config_ids",
                "judge_repeats[].scores",
                "judge_repeats[].scores[].cannot_tell",
                "judge_repeats[].scores[].dim",
                "judge_repeats[].scores[].first_judge_config_id",
                "judge_repeats[].scores[].first_judge_sent_settings",
                "judge_repeats[].scores[].first_judge_sent_settings.model_dump()",
                "judge_repeats[].scores[].first_judge_temperature",
                "judge_repeats[].scores[].first_score",
                "judge_repeats[].scores[].first_served_model",
                "judge_repeats[].scores[].repeat",
                "judge_repeats[].scores[].repeat.judge_sent_settings",
                "judge_repeats[].scores[].repeat.judge_sent_settings.==",
                "judge_repeats[].scores[].repeat.judge_temperature",
                "judge_repeats[].scores[].repeat.score",
                "judge_repeats[].scores[].repeat.served_model",
                "judge_repeats[].scores[].scale",
                "judge_rescores[].cost_usd",
                "judge_seconds[].judge",
                "judge_seconds[].judge.model",
                "judge_seconds[].judge.temperature",
                "judge_seconds[].judge_config_ids",
                "judge_seconds[].pass_id",
                "judge_seconds[].scores",
                "judge_seconds[].scores[].dim",
                "judge_seconds[].scores[].first_judge_config_id",
                "judge_seconds[].scores[].first_judge_sent_settings",
                "judge_seconds[].scores[].first_judge_sent_settings.model_dump()",
                "judge_seconds[].scores[].first_judge_temperature",
                "judge_seconds[].scores[].first_score",
                "judge_seconds[].scores[].first_served_model",
                "judge_seconds[].scores[].scale",
                "judge_seconds[].scores[].second",
                "judge_seconds[].scores[].second.judge_sent_settings",
                "judge_seconds[].scores[].second.judge_sent_settings.model_dump()",
                "judge_seconds[].scores[].second.score",
                "judge_seconds[].scores[].second.served_model",
                "latency.async_wait_ms",
                "latency.judge_ms",
                "latency.llm_ms",
                "latency.tool_ms",
                "latency.total_ms",
                "model_copy()",
                "precondition_outcomes[].detail",
                "precondition_outcomes[].expression",
                "precondition_outcomes[].presumes",
                "usage[].sent_settings",
                "usage[].sent_settings[].==",
                "usage[].sent_settings[].model_dump()",
                "usage[].sent_settings[].reasoning_max_tokens",
                "usage[].sent_settings[].strict_output",
            }
        ),
        "scaled_two_aspects": frozenset(
            {
                "async_deliveries[].acknowledged_turn",
                "async_deliveries[].cost_usd",
                "async_deliveries[].delivered_items",
                "async_deliveries[].delivered_turn",
                "async_deliveries[].elapsed_ms",
                "async_deliveries[].error",
                "async_deliveries[].input_tokens",
                "async_deliveries[].llm_calls",
                "async_deliveries[].model",
                "async_deliveries[].model_copy()",
                "async_deliveries[].output_tokens",
                "async_deliveries[].price_source",
                "async_deliveries[].reasoning_tokens",
                "async_deliveries[].requested_by",
                "async_deliveries[].served_model",
                "async_deliveries[].status",
                "async_deliveries[].substituted",
                "async_deliveries[].summary",
                "async_deliveries[].tool",
                "judge_repeats[].judge_config_ids",
                "judge_repeats[].scores",
                "judge_repeats[].scores[].cannot_tell",
                "judge_repeats[].scores[].dim",
                "judge_repeats[].scores[].first_judge_config_id",
                "judge_repeats[].scores[].first_judge_sent_settings",
                "judge_repeats[].scores[].first_judge_sent_settings.model_dump()",
                "judge_repeats[].scores[].first_judge_temperature",
                "judge_repeats[].scores[].first_score",
                "judge_repeats[].scores[].first_served_model",
                "judge_repeats[].scores[].repeat",
                "judge_repeats[].scores[].repeat.judge_sent_settings",
                "judge_repeats[].scores[].repeat.judge_sent_settings.==",
                "judge_repeats[].scores[].repeat.judge_temperature",
                "judge_repeats[].scores[].repeat.score",
                "judge_repeats[].scores[].repeat.served_model",
                "judge_repeats[].scores[].scale",
                "judge_rescores[].cost_usd",
                "judge_seconds[].judge",
                "judge_seconds[].judge.model",
                "judge_seconds[].judge.temperature",
                "judge_seconds[].judge_config_ids",
                "judge_seconds[].pass_id",
                "judge_seconds[].scores",
                "judge_seconds[].scores[].dim",
                "judge_seconds[].scores[].first_judge_config_id",
                "judge_seconds[].scores[].first_judge_sent_settings",
                "judge_seconds[].scores[].first_judge_sent_settings.model_dump()",
                "judge_seconds[].scores[].first_judge_temperature",
                "judge_seconds[].scores[].first_score",
                "judge_seconds[].scores[].first_served_model",
                "judge_seconds[].scores[].scale",
                "judge_seconds[].scores[].second",
                "judge_seconds[].scores[].second.judge_sent_settings",
                "judge_seconds[].scores[].second.judge_sent_settings.model_dump()",
                "judge_seconds[].scores[].second.score",
                "judge_seconds[].scores[].second.served_model",
                "latency.async_wait_ms",
                "latency.judge_ms",
                "latency.llm_ms",
                "latency.tool_ms",
                "latency.total_ms",
                "model_copy()",
                "precondition_outcomes[].detail",
                "precondition_outcomes[].expression",
                "precondition_outcomes[].presumes",
                "usage[].sent_settings",
                "usage[].sent_settings[].==",
                "usage[].sent_settings[].model_dump()",
                "usage[].sent_settings[].reasoning_max_tokens",
                "usage[].sent_settings[].strict_output",
            }
        ),
        "toyhost": frozenset(
            {
                "async_deliveries[].acknowledged_turn",
                "async_deliveries[].cost_usd",
                "async_deliveries[].delivered_items",
                "async_deliveries[].delivered_turn",
                "async_deliveries[].elapsed_ms",
                "async_deliveries[].error",
                "async_deliveries[].input_tokens",
                "async_deliveries[].llm_calls",
                "async_deliveries[].model",
                "async_deliveries[].model_copy()",
                "async_deliveries[].output_tokens",
                "async_deliveries[].price_source",
                "async_deliveries[].reasoning_tokens",
                "async_deliveries[].requested_by",
                "async_deliveries[].served_model",
                "async_deliveries[].status",
                "async_deliveries[].substituted",
                "async_deliveries[].summary",
                "async_deliveries[].tool",
                "judge_cut_short",
                "judge_repeats[].judge_config_ids",
                "judge_repeats[].scores",
                "judge_repeats[].scores[].cannot_tell",
                "judge_repeats[].scores[].dim",
                "judge_repeats[].scores[].first_judge_config_id",
                "judge_repeats[].scores[].first_judge_sent_settings",
                "judge_repeats[].scores[].first_judge_sent_settings.model_dump()",
                "judge_repeats[].scores[].first_judge_temperature",
                "judge_repeats[].scores[].first_score",
                "judge_repeats[].scores[].first_served_model",
                "judge_repeats[].scores[].repeat",
                "judge_repeats[].scores[].repeat.judge_sent_settings",
                "judge_repeats[].scores[].repeat.judge_sent_settings.==",
                "judge_repeats[].scores[].repeat.judge_temperature",
                "judge_repeats[].scores[].repeat.score",
                "judge_repeats[].scores[].repeat.served_model",
                "judge_repeats[].scores[].scale",
                "judge_rescores[].cost_usd",
                "judge_seconds[].judge",
                "judge_seconds[].judge.model",
                "judge_seconds[].judge.temperature",
                "judge_seconds[].judge_config_ids",
                "judge_seconds[].pass_id",
                "judge_seconds[].scores",
                "judge_seconds[].scores[].dim",
                "judge_seconds[].scores[].first_judge_config_id",
                "judge_seconds[].scores[].first_judge_sent_settings",
                "judge_seconds[].scores[].first_judge_sent_settings.model_dump()",
                "judge_seconds[].scores[].first_judge_temperature",
                "judge_seconds[].scores[].first_score",
                "judge_seconds[].scores[].first_served_model",
                "judge_seconds[].scores[].scale",
                "judge_seconds[].scores[].second",
                "judge_seconds[].scores[].second.judge_sent_settings",
                "judge_seconds[].scores[].second.judge_sent_settings.model_dump()",
                "judge_seconds[].scores[].second.score",
                "judge_seconds[].scores[].second.served_model",
                "model_copy()",
                "outcome_score.axis",
                "outcome_score.criterion_fingerprint",
                "outcome_score.dim",
                "outcome_score.judge_sent_settings",
                "outcome_score.judge_sent_settings.==",
                "outcome_score.judge_sent_settings.max_tokens",
                "outcome_score.judge_sent_settings.model_dump()",
                "outcome_score.judge_sent_settings.reasoning_effort",
                "outcome_score.judge_sent_settings.reasoning_max_tokens",
                "outcome_score.judge_sent_settings.strict_output",
                "outcome_score.judge_sent_settings.unsent_reasoning_max_tokens",
                "outcome_score.judge_temperature",
                "outcome_score.label_key",
                "outcome_score.output_fingerprint",
                "outcome_score.reasoning",
                "outcome_score.scale",
                "outcome_score.score",
                "outcome_score.served_model",
                "precondition_outcomes[].detail",
                "precondition_outcomes[].expression",
                "precondition_outcomes[].presumes",
                "rubric_scores[].judge_sent_settings.==",
                "rubric_scores[].judge_sent_settings.model_dump()",
                "rubric_scores[].judge_sent_settings.strict_output",
                "transcript_score.axis",
                "transcript_score.criterion_fingerprint",
                "transcript_score.dim",
                "transcript_score.judge_sent_settings",
                "transcript_score.judge_sent_settings.==",
                "transcript_score.judge_sent_settings.model_dump()",
                "transcript_score.judge_sent_settings.strict_output",
                "transcript_score.judge_temperature",
                "transcript_score.label_key",
                "transcript_score.output_fingerprint",
                "transcript_score.reasoning",
                "transcript_score.scale",
                "transcript_score.score",
                "transcript_score.served_model",
                "usage[].sent_settings",
                "usage[].sent_settings[].==",
                "usage[].sent_settings[].model_dump()",
                "usage[].sent_settings[].reasoning_max_tokens",
                "usage[].sent_settings[].strict_output",
            }
        ),
        "courierhost": frozenset(
            {
                "async_deliveries[].acknowledged_turn",
                "async_deliveries[].cost_usd",
                "async_deliveries[].delivered_items",
                "async_deliveries[].delivered_turn",
                "async_deliveries[].elapsed_ms",
                "async_deliveries[].error",
                "async_deliveries[].input_tokens",
                "async_deliveries[].llm_calls",
                "async_deliveries[].model",
                "async_deliveries[].model_copy()",
                "async_deliveries[].output_tokens",
                "async_deliveries[].price_source",
                "async_deliveries[].reasoning_tokens",
                "async_deliveries[].requested_by",
                "async_deliveries[].served_model",
                "async_deliveries[].status",
                "async_deliveries[].substituted",
                "async_deliveries[].summary",
                "async_deliveries[].tool",
                "goal_state_outcomes[].detail",
                "goal_state_outcomes[].expression",
                "goal_state_outcomes[].passed",
                "judge_cut_short",
                "judge_repeats[].judge_config_ids",
                "judge_repeats[].scores",
                "judge_repeats[].scores[].cannot_tell",
                "judge_repeats[].scores[].dim",
                "judge_repeats[].scores[].first_judge_config_id",
                "judge_repeats[].scores[].first_judge_sent_settings",
                "judge_repeats[].scores[].first_judge_sent_settings.model_dump()",
                "judge_repeats[].scores[].first_judge_temperature",
                "judge_repeats[].scores[].first_score",
                "judge_repeats[].scores[].first_served_model",
                "judge_repeats[].scores[].repeat",
                "judge_repeats[].scores[].repeat.judge_sent_settings",
                "judge_repeats[].scores[].repeat.judge_sent_settings.==",
                "judge_repeats[].scores[].repeat.judge_temperature",
                "judge_repeats[].scores[].repeat.score",
                "judge_repeats[].scores[].repeat.served_model",
                "judge_repeats[].scores[].scale",
                "judge_rescores[].cost_usd",
                "judge_score()",
                "judge_scores()",
                "judge_seconds[].judge",
                "judge_seconds[].judge.model",
                "judge_seconds[].judge.temperature",
                "judge_seconds[].judge_config_ids",
                "judge_seconds[].pass_id",
                "judge_seconds[].scores",
                "judge_seconds[].scores[].dim",
                "judge_seconds[].scores[].first_judge_config_id",
                "judge_seconds[].scores[].first_judge_sent_settings",
                "judge_seconds[].scores[].first_judge_sent_settings.model_dump()",
                "judge_seconds[].scores[].first_judge_temperature",
                "judge_seconds[].scores[].first_score",
                "judge_seconds[].scores[].first_served_model",
                "judge_seconds[].scores[].scale",
                "judge_seconds[].scores[].second",
                "judge_seconds[].scores[].second.judge_sent_settings",
                "judge_seconds[].scores[].second.judge_sent_settings.model_dump()",
                "judge_seconds[].scores[].second.score",
                "judge_seconds[].scores[].second.served_model",
                "latency.async_wait_ms",
                "latency.judge_ms",
                "latency.llm_ms",
                "latency.tool_ms",
                "latency.total_ms",
                "model_copy()",
                "outcome_score.axis",
                "outcome_score.criterion_fingerprint",
                "outcome_score.dim",
                "outcome_score.judge_sent_settings",
                "outcome_score.judge_sent_settings.==",
                "outcome_score.judge_sent_settings.max_tokens",
                "outcome_score.judge_sent_settings.model_dump()",
                "outcome_score.judge_sent_settings.reasoning_effort",
                "outcome_score.judge_sent_settings.reasoning_max_tokens",
                "outcome_score.judge_sent_settings.strict_output",
                "outcome_score.judge_sent_settings.unsent_reasoning_max_tokens",
                "outcome_score.judge_temperature",
                "outcome_score.label_key",
                "outcome_score.output_fingerprint",
                "outcome_score.reasoning",
                "outcome_score.scale",
                "outcome_score.score",
                "outcome_score.served_model",
                "precondition_outcomes[].detail",
                "precondition_outcomes[].expression",
                "precondition_outcomes[].presumes",
                "rubric_scores[].axis",
                "rubric_scores[].clears()",
                "rubric_scores[].criterion_fingerprint",
                "rubric_scores[].dim",
                "rubric_scores[].judge_sent_settings",
                "rubric_scores[].judge_sent_settings.==",
                "rubric_scores[].judge_sent_settings.model_dump()",
                "rubric_scores[].judge_sent_settings.strict_output",
                "rubric_scores[].judge_temperature",
                "rubric_scores[].label_key",
                "rubric_scores[].normalized",
                "rubric_scores[].output_fingerprint",
                "rubric_scores[].reasoning",
                "rubric_scores[].scale",
                "rubric_scores[].score",
                "rubric_scores[].served_model",
                "transcript_score.axis",
                "transcript_score.criterion_fingerprint",
                "transcript_score.dim",
                "transcript_score.judge_sent_settings",
                "transcript_score.judge_sent_settings.==",
                "transcript_score.judge_sent_settings.model_dump()",
                "transcript_score.judge_sent_settings.strict_output",
                "transcript_score.judge_temperature",
                "transcript_score.label_key",
                "transcript_score.output_fingerprint",
                "transcript_score.reasoning",
                "transcript_score.scale",
                "transcript_score.score",
                "transcript_score.served_model",
                "usage[].sent_settings",
                "usage[].sent_settings[].==",
                "usage[].sent_settings[].model_dump()",
                "usage[].sent_settings[].reasoning_max_tokens",
                "usage[].sent_settings[].strict_output",
            }
        ),
        "enriched": frozenset(
            {
                "outcome_score.judge_sent_settings.max_tokens",
                "outcome_score.judge_sent_settings.reasoning_effort",
                "outcome_score.judge_sent_settings.unsent_reasoning_max_tokens",
            }
        ),
    },
    "aggregates": {
        "scaled": frozenset(
            {
                "async_deliveries[].model_copy()",
                "async_deliveries[].substituted",
                "judge_model",
                "judge_repeats[].judge_config_ids",
                "judge_repeats[].scores",
                "judge_repeats[].scores[].cannot_tell",
                "judge_repeats[].scores[].dim",
                "judge_repeats[].scores[].first_judge_config_id",
                "judge_repeats[].scores[].first_judge_sent_settings",
                "judge_repeats[].scores[].first_judge_sent_settings.model_dump()",
                "judge_repeats[].scores[].first_judge_temperature",
                "judge_repeats[].scores[].first_score",
                "judge_repeats[].scores[].first_served_model",
                "judge_repeats[].scores[].repeat",
                "judge_repeats[].scores[].repeat.judge_sent_settings",
                "judge_repeats[].scores[].repeat.judge_sent_settings.==",
                "judge_repeats[].scores[].repeat.judge_temperature",
                "judge_repeats[].scores[].repeat.score",
                "judge_repeats[].scores[].repeat.served_model",
                "judge_repeats[].scores[].scale",
                "latency.async_wait_ms",
                "latency.judge_ms",
                "latency.llm_ms",
                "latency.tool_ms",
                "latency.total_ms",
                "model_copy()",
                "usage[].sent_settings",
                "usage[].sent_settings[].model_dump()",
            }
        ),
        "scaled_two_aspects": frozenset(
            {
                "async_deliveries[].model_copy()",
                "async_deliveries[].substituted",
                "judge_model",
                "judge_repeats[].judge_config_ids",
                "judge_repeats[].scores",
                "judge_repeats[].scores[].cannot_tell",
                "judge_repeats[].scores[].dim",
                "judge_repeats[].scores[].first_judge_config_id",
                "judge_repeats[].scores[].first_judge_sent_settings",
                "judge_repeats[].scores[].first_judge_sent_settings.model_dump()",
                "judge_repeats[].scores[].first_judge_temperature",
                "judge_repeats[].scores[].first_score",
                "judge_repeats[].scores[].first_served_model",
                "judge_repeats[].scores[].repeat",
                "judge_repeats[].scores[].repeat.judge_sent_settings",
                "judge_repeats[].scores[].repeat.judge_sent_settings.==",
                "judge_repeats[].scores[].repeat.judge_temperature",
                "judge_repeats[].scores[].repeat.score",
                "judge_repeats[].scores[].repeat.served_model",
                "judge_repeats[].scores[].scale",
                "latency.async_wait_ms",
                "latency.judge_ms",
                "latency.llm_ms",
                "latency.tool_ms",
                "latency.total_ms",
                "model_copy()",
                "usage[].sent_settings",
                "usage[].sent_settings[].model_dump()",
            }
        ),
        "toyhost": frozenset(
            {
                "async_deliveries[].model_copy()",
                "async_deliveries[].substituted",
                "judge_cut_short",
                "judge_model",
                "judge_repeats[].judge_config_ids",
                "judge_repeats[].scores",
                "judge_repeats[].scores[].cannot_tell",
                "judge_repeats[].scores[].dim",
                "judge_repeats[].scores[].first_judge_config_id",
                "judge_repeats[].scores[].first_judge_sent_settings",
                "judge_repeats[].scores[].first_judge_sent_settings.model_dump()",
                "judge_repeats[].scores[].first_judge_temperature",
                "judge_repeats[].scores[].first_score",
                "judge_repeats[].scores[].first_served_model",
                "judge_repeats[].scores[].repeat",
                "judge_repeats[].scores[].repeat.judge_sent_settings",
                "judge_repeats[].scores[].repeat.judge_sent_settings.==",
                "judge_repeats[].scores[].repeat.judge_temperature",
                "judge_repeats[].scores[].repeat.score",
                "judge_repeats[].scores[].repeat.served_model",
                "judge_repeats[].scores[].scale",
                "latency.async_wait_ms",
                "latency.judge_ms",
                "latency.llm_ms",
                "latency.tool_ms",
                "model_copy()",
                "outcome_score.criterion_fingerprint",
                "outcome_score.dim",
                "outcome_score.judge_sent_settings",
                "outcome_score.judge_sent_settings.model_dump()",
                "outcome_score.judge_temperature",
                "outcome_score.label_key",
                "outcome_score.output_fingerprint",
                "outcome_score.scale",
                "outcome_score.score",
                "outcome_score.served_model",
                "rubric_scores[].judge_sent_settings.model_dump()",
                "termination",
                "transcript_score.criterion_fingerprint",
                "transcript_score.dim",
                "transcript_score.judge_sent_settings",
                "transcript_score.judge_sent_settings.model_dump()",
                "transcript_score.judge_temperature",
                "transcript_score.label_key",
                "transcript_score.output_fingerprint",
                "transcript_score.scale",
                "transcript_score.score",
                "transcript_score.served_model",
                "turns_delivered",
                "usage[].sent_settings",
                "usage[].sent_settings[].model_dump()",
            }
        ),
        "courierhost": frozenset(
            {
                "async_deliveries[].model_copy()",
                "async_deliveries[].substituted",
                "goal_state_outcomes[].expression",
                "goal_state_outcomes[].passed",
                "judge_cut_short",
                "judge_repeats[].judge_config_ids",
                "judge_repeats[].scores",
                "judge_repeats[].scores[].cannot_tell",
                "judge_repeats[].scores[].dim",
                "judge_repeats[].scores[].first_judge_config_id",
                "judge_repeats[].scores[].first_judge_sent_settings",
                "judge_repeats[].scores[].first_judge_sent_settings.model_dump()",
                "judge_repeats[].scores[].first_judge_temperature",
                "judge_repeats[].scores[].first_score",
                "judge_repeats[].scores[].first_served_model",
                "judge_repeats[].scores[].repeat",
                "judge_repeats[].scores[].repeat.judge_sent_settings",
                "judge_repeats[].scores[].repeat.judge_sent_settings.==",
                "judge_repeats[].scores[].repeat.judge_temperature",
                "judge_repeats[].scores[].repeat.score",
                "judge_repeats[].scores[].repeat.served_model",
                "judge_repeats[].scores[].scale",
                "judge_score()",
                "judge_scores()",
                "latency.async_wait_ms",
                "latency.judge_ms",
                "latency.llm_ms",
                "latency.tool_ms",
                "latency.total_ms",
                "model_copy()",
                "outcome_score.criterion_fingerprint",
                "outcome_score.dim",
                "outcome_score.judge_sent_settings",
                "outcome_score.judge_sent_settings.model_dump()",
                "outcome_score.judge_temperature",
                "outcome_score.label_key",
                "outcome_score.output_fingerprint",
                "outcome_score.scale",
                "outcome_score.score",
                "outcome_score.served_model",
                "rubric_scores[].axis",
                "rubric_scores[].clears()",
                "rubric_scores[].criterion_fingerprint",
                "rubric_scores[].dim",
                "rubric_scores[].judge_sent_settings",
                "rubric_scores[].judge_sent_settings.model_dump()",
                "rubric_scores[].judge_temperature",
                "rubric_scores[].label_key",
                "rubric_scores[].normalized",
                "rubric_scores[].output_fingerprint",
                "rubric_scores[].scale",
                "rubric_scores[].score",
                "rubric_scores[].served_model",
                "termination",
                "transcript_score.criterion_fingerprint",
                "transcript_score.dim",
                "transcript_score.judge_sent_settings",
                "transcript_score.judge_sent_settings.model_dump()",
                "transcript_score.judge_temperature",
                "transcript_score.label_key",
                "transcript_score.output_fingerprint",
                "transcript_score.scale",
                "transcript_score.score",
                "transcript_score.served_model",
                "turns_delivered",
                "usage[].sent_settings",
                "usage[].sent_settings[].model_dump()",
            }
        ),
        "enriched": frozenset(
            {
                "judge_model",
                "usage[].cost_usd",
            }
        ),
        "regressed": frozenset(
            {
                "async_deliveries",
                "async_deliveries[].model_copy()",
                "async_deliveries[].substituted",
                "candidate_kind",
                "cost_roles",
                "cost_usd",
                "goal_state_outcomes",
                "goal_state_outcomes[].expression",
                "goal_state_outcomes[].passed",
                "host_measures",
                "judge_cut_short",
                "judge_model",
                "judge_repeats[].judge_config_ids",
                "judge_repeats[].scores",
                "judge_repeats[].scores[].cannot_tell",
                "judge_repeats[].scores[].dim",
                "judge_repeats[].scores[].first_judge_config_id",
                "judge_repeats[].scores[].first_judge_sent_settings",
                "judge_repeats[].scores[].first_judge_sent_settings.model_dump()",
                "judge_repeats[].scores[].first_judge_temperature",
                "judge_repeats[].scores[].first_score",
                "judge_repeats[].scores[].first_served_model",
                "judge_repeats[].scores[].repeat",
                "judge_repeats[].scores[].repeat.judge_sent_settings",
                "judge_repeats[].scores[].repeat.judge_sent_settings.==",
                "judge_repeats[].scores[].repeat.judge_temperature",
                "judge_repeats[].scores[].repeat.score",
                "judge_repeats[].scores[].repeat.served_model",
                "judge_repeats[].scores[].scale",
                "k_iteration",
                "latency",
                "latency.async_wait_ms",
                "latency.judge_ms",
                "latency.llm_ms",
                "latency.tool_ms",
                "latency.total_ms",
                "model_copy()",
                "rubric_scores[].clears()",
                "scope_id",
                "scored_at",
                "termination",
                "turns_delivered",
                "usage[].cost_usd",
                "usage[].sent_settings",
                "usage[].sent_settings[].model_dump()",
            }
        ),
    },
    "planning": {
        "scaled": frozenset(
            {
                "async_deliveries[].acknowledged_turn",
                "async_deliveries[].cost_usd",
                "async_deliveries[].delivered_items",
                "async_deliveries[].delivered_turn",
                "async_deliveries[].elapsed_ms",
                "async_deliveries[].error",
                "async_deliveries[].input_tokens",
                "async_deliveries[].llm_calls",
                "async_deliveries[].model",
                "async_deliveries[].model_copy()",
                "async_deliveries[].output_tokens",
                "async_deliveries[].price_source",
                "async_deliveries[].reasoning_tokens",
                "async_deliveries[].requested_by",
                "async_deliveries[].served_model",
                "async_deliveries[].status",
                "async_deliveries[].substituted",
                "async_deliveries[].summary",
                "async_deliveries[].tool",
                "latency.async_wait_ms",
                "latency.judge_ms",
                "latency.llm_ms",
                "latency.tool_ms",
                "latency.total_ms",
                "precondition_outcomes[].detail",
                "precondition_outcomes[].expression",
                "precondition_outcomes[].presumes",
                "usage[].sent_settings",
                "usage[].sent_settings[].model_dump()",
            }
        ),
        "toyhost": frozenset(
            {
                "async_deliveries",
                "async_deliveries[].acknowledged_turn",
                "async_deliveries[].cost_usd",
                "async_deliveries[].delivered_items",
                "async_deliveries[].delivered_turn",
                "async_deliveries[].elapsed_ms",
                "async_deliveries[].error",
                "async_deliveries[].input_tokens",
                "async_deliveries[].llm_calls",
                "async_deliveries[].model",
                "async_deliveries[].model_copy()",
                "async_deliveries[].output_tokens",
                "async_deliveries[].price_source",
                "async_deliveries[].reasoning_tokens",
                "async_deliveries[].requested_by",
                "async_deliveries[].served_model",
                "async_deliveries[].status",
                "async_deliveries[].substituted",
                "async_deliveries[].summary",
                "async_deliveries[].tool",
                "candidate_error",
                "candidate_instance_id",
                "candidate_kind",
                "cost_roles",
                "cost_usd",
                "covariates",
                "doc_type",
                "eval_run_id",
                "goal_state_outcomes",
                "goal_state_outcomes[].detail",
                "goal_state_outcomes[].expression",
                "goal_state_outcomes[].passed",
                "host_measures",
                "id",
                "identity_version",
                "infra_error",
                "judge_cannot_tell",
                "judge_cannot_tell_boundary",
                "judge_config_ids",
                "judge_cut_short",
                "judge_error",
                "judge_model",
                "k_iteration",
                "latency",
                "latency.async_wait_ms",
                "latency.judge_ms",
                "latency.llm_ms",
                "latency.tool_ms",
                "latency.total_ms",
                "lease_epoch",
                "metered_calls",
                "metered_calls_refused",
                "model",
                "model_copy()",
                "outcome_score",
                "outcome_score.axis",
                "outcome_score.criterion_fingerprint",
                "outcome_score.dim",
                "outcome_score.judge_sent_settings",
                "outcome_score.judge_sent_settings.model_dump()",
                "outcome_score.judge_temperature",
                "outcome_score.output_fingerprint",
                "outcome_score.scale",
                "outcome_score.score",
                "outcome_score.served_model",
                "phase_timings",
                "precondition_outcomes",
                "precondition_outcomes[].detail",
                "precondition_outcomes[].expression",
                "precondition_outcomes[].presumes",
                "rubric_scores",
                "rubric_scores[].axis",
                "rubric_scores[].criterion_fingerprint",
                "rubric_scores[].dim",
                "rubric_scores[].judge_sent_settings",
                "rubric_scores[].judge_sent_settings.model_dump()",
                "rubric_scores[].judge_temperature",
                "rubric_scores[].normalized",
                "rubric_scores[].output_fingerprint",
                "rubric_scores[].scale",
                "rubric_scores[].score",
                "rubric_scores[].served_model",
                "runner_error",
                "schema_minor",
                "schema_version",
                "scope_id",
                "scored_at",
                "stop_cause",
                "subject_id",
                "superseded_spend_usd",
                "termination",
                "test_case_id",
                "transcript_score",
                "transcript_score.axis",
                "transcript_score.criterion_fingerprint",
                "transcript_score.dim",
                "transcript_score.judge_sent_settings",
                "transcript_score.judge_sent_settings.model_dump()",
                "transcript_score.judge_temperature",
                "transcript_score.output_fingerprint",
                "transcript_score.scale",
                "transcript_score.score",
                "transcript_score.served_model",
                "turns_delivered",
                "usage",
                "usage[].actor_id",
                "usage[].call_count",
                "usage[].completion_tokens",
                "usage[].cost_usd",
                "usage[].model",
                "usage[].price_source",
                "usage[].prompt_tokens",
                "usage[].provider",
                "usage[].provider_unit",
                "usage[].provider_units",
                "usage[].purpose",
                "usage[].reasoning_tokens",
                "usage[].role",
                "usage[].sent_settings",
                "usage[].sent_settings[].model_dump()",
                "usage[].served_model",
                "usage[].unanswered_lost",
                "usage[].unanswered_retried",
                "variant_key",
            }
        ),
        "courierhost": frozenset(
            {
                "async_deliveries[].acknowledged_turn",
                "async_deliveries[].cost_usd",
                "async_deliveries[].delivered_items",
                "async_deliveries[].delivered_turn",
                "async_deliveries[].elapsed_ms",
                "async_deliveries[].error",
                "async_deliveries[].input_tokens",
                "async_deliveries[].llm_calls",
                "async_deliveries[].model",
                "async_deliveries[].model_copy()",
                "async_deliveries[].output_tokens",
                "async_deliveries[].price_source",
                "async_deliveries[].reasoning_tokens",
                "async_deliveries[].requested_by",
                "async_deliveries[].served_model",
                "async_deliveries[].status",
                "async_deliveries[].substituted",
                "async_deliveries[].summary",
                "async_deliveries[].tool",
                "goal_state_outcomes[].detail",
                "goal_state_outcomes[].expression",
                "goal_state_outcomes[].passed",
                "judge_cut_short",
                "latency.async_wait_ms",
                "latency.judge_ms",
                "latency.llm_ms",
                "latency.tool_ms",
                "latency.total_ms",
                "model_copy()",
                "outcome_score.axis",
                "outcome_score.criterion_fingerprint",
                "outcome_score.dim",
                "outcome_score.judge_sent_settings",
                "outcome_score.judge_sent_settings.model_dump()",
                "outcome_score.judge_temperature",
                "outcome_score.output_fingerprint",
                "outcome_score.scale",
                "outcome_score.score",
                "outcome_score.served_model",
                "precondition_outcomes[].detail",
                "precondition_outcomes[].expression",
                "precondition_outcomes[].presumes",
                "rubric_scores[].axis",
                "rubric_scores[].criterion_fingerprint",
                "rubric_scores[].dim",
                "rubric_scores[].judge_sent_settings",
                "rubric_scores[].judge_sent_settings.model_dump()",
                "rubric_scores[].judge_temperature",
                "rubric_scores[].normalized",
                "rubric_scores[].output_fingerprint",
                "rubric_scores[].scale",
                "rubric_scores[].score",
                "rubric_scores[].served_model",
                "transcript_score.axis",
                "transcript_score.criterion_fingerprint",
                "transcript_score.dim",
                "transcript_score.judge_sent_settings",
                "transcript_score.judge_sent_settings.model_dump()",
                "transcript_score.judge_temperature",
                "transcript_score.output_fingerprint",
                "transcript_score.scale",
                "transcript_score.score",
                "transcript_score.served_model",
                "usage[].sent_settings",
                "usage[].sent_settings[].model_dump()",
            }
        ),
    },
}

#: Where each family reads judge prose: each prose path (a text read, or ``bool(...)`` for a truth test) and the
#: ``module function`` sites that read it, over every campaign.
PROSE_SITES: dict[str, dict[str, frozenset[str]]] = {
    "bundle": {
        "judge_cannot_tell": frozenset(
            {
                "analysis/bundle/trials.py _row",
                "analysis/reporting.py project_score_records",
                "kernel/result_condition.py judged_on_every_dim",
                "kernel/scoring.py has_pass_criterion",
            }
        ),
        "judge_repeats[].scores[].cannot_tell": frozenset(
            {
                "analysis/agreement.py self_agreement_of_repeats",
            }
        ),
        "outcome_score.reasoning": frozenset(
            {
                "analysis/bundle/trials.py _row",
            }
        ),
        "rubric_scores[].reasoning": frozenset(
            {
                "analysis/bundle/trials.py _row",
            }
        ),
        "transcript_score.reasoning": frozenset(
            {
                "analysis/bundle/trials.py _row",
            }
        ),
    },
    "aggregates": {
        "judge_cannot_tell": frozenset(
            {
                "analysis/reporting.py project_score_records",
                "kernel/result_condition.py judged_on_every_dim",
                "kernel/scoring.py has_pass_criterion",
            }
        ),
        "judge_repeats[].scores[].cannot_tell": frozenset(
            {
                "analysis/agreement.py self_agreement_of_repeats",
            }
        ),
    },
    "planning": {
        "judge_cannot_tell": frozenset(
            {
                "analysis/reporting.py project_score_records",
                "kernel/result_condition.py judged_on_every_dim",
            }
        ),
    },
}

#: What each campaign leaves unread of its family's prose pins (:func:`prose_pins`), as :data:`UNREAD` does.
PROSE_UNREAD: dict[str, dict[str, frozenset[str]]] = {
    "bundle": {
        "scaled": frozenset(
            {
                "judge_repeats[].scores[].cannot_tell at analysis/agreement.py self_agreement_of_repeats",
            }
        ),
        "scaled_two_aspects": frozenset(
            {
                "judge_repeats[].scores[].cannot_tell at analysis/agreement.py self_agreement_of_repeats",
            }
        ),
        "toyhost": frozenset(
            {
                "judge_repeats[].scores[].cannot_tell at analysis/agreement.py self_agreement_of_repeats",
                "outcome_score.reasoning at analysis/bundle/trials.py _row",
                "transcript_score.reasoning at analysis/bundle/trials.py _row",
            }
        ),
        "courierhost": frozenset(
            {
                "judge_repeats[].scores[].cannot_tell at analysis/agreement.py self_agreement_of_repeats",
                "outcome_score.reasoning at analysis/bundle/trials.py _row",
                "rubric_scores[].reasoning at analysis/bundle/trials.py _row",
                "transcript_score.reasoning at analysis/bundle/trials.py _row",
            }
        ),
    },
    "aggregates": {
        "scaled": frozenset(
            {
                "judge_repeats[].scores[].cannot_tell at analysis/agreement.py self_agreement_of_repeats",
            }
        ),
        "scaled_two_aspects": frozenset(
            {
                "judge_repeats[].scores[].cannot_tell at analysis/agreement.py self_agreement_of_repeats",
            }
        ),
        "toyhost": frozenset(
            {
                "judge_repeats[].scores[].cannot_tell at analysis/agreement.py self_agreement_of_repeats",
            }
        ),
        "courierhost": frozenset(
            {
                "judge_repeats[].scores[].cannot_tell at analysis/agreement.py self_agreement_of_repeats",
            }
        ),
        "regressed": frozenset(
            {
                "judge_cannot_tell at analysis/reporting.py project_score_records",
                "judge_cannot_tell at kernel/scoring.py has_pass_criterion",
                "judge_repeats[].scores[].cannot_tell at analysis/agreement.py self_agreement_of_repeats",
            }
        ),
    },
    "planning": {
        "toyhost": frozenset(
            {
                "judge_cannot_tell at analysis/reporting.py project_score_records",
                "judge_cannot_tell at kernel/result_condition.py judged_on_every_dim",
            }
        ),
    },
}


class TestTheRecorderRecordsWhatIsReadAndChangesNothing:
    """The recorder's own contract, on one stored result: each read is recorded at its path, a truth test of prose apart
    from a read of its text, a whole-object read by its operation, and nothing outside a recording block."""

    def test_each_read_is_recorded_at_its_path(self) -> None:
        storage, _ = memory_storage()
        stored = make_eval_result(
            id="r-1",
            judge_reasoning="the reply cites the refund window",
            transcript_score=RubricScore(
                dim=TRANSCRIPT_DIM_ID, score=4, scale="ordinal", axis="capability", reasoning="courteous throughout"
            ),
        )
        storage.save_eval_result(stored)

        with recording_results():
            (result,) = storage.query_eval_results(stored.scope_id)
            outside = result.eval_run_id
            with recording() as access:
                score = result.rubric_scores[0].score
                narrated = "narrated" if result.judge_reasoning else "silent"
                words = result.judge_reasoning.split()
                transcript = result.transcript_score
                assert transcript is not None
                reason = transcript.reasoning.upper()
                dumped = result.model_dump()
                copied = result.model_copy()
                copied_case = copied.test_case_id

        assert isinstance(result, EvalResult)
        assert (outside, score, narrated, words[0], reason, copied_case) == (
            stored.eval_run_id,
            4,
            "narrated",
            "the",
            "COURTEOUS THROUGHOUT",
            stored.test_case_id,
        )
        assert dumped == stored.model_dump()
        assert access.paths == {
            "rubric_scores",
            "rubric_scores[].score",
            "bool(judge_reasoning)",
            "judge_reasoning",
            "transcript_score",
            "transcript_score.reasoning",
            "model_dump()",
            "model_copy()",
            "test_case_id",
        }, "eval_run_id was read before the block opened, so it is not one of the block's reads"
        assert set(access.whole_object_reads()) == {"model_dump()", "model_copy()"}
        assert {path: set(by_site) for path, by_site in access.prose.items()} == {
            "bool(judge_reasoning)": {next(iter(access.sites["bool(judge_reasoning)"]))},
            "judge_reasoning": {next(iter(access.sites["judge_reasoning"]))},
            "transcript_score.reasoning": {next(iter(access.sites["transcript_score.reasoning"]))},
        }
        assert all(reads.results == {"r-1"} for by_site in access.prose.values() for reads in by_site.values())
        assert all(
            site.startswith("test_result_reading_fields.py:") for sites in access.sites.values() for site in sites
        )

    def test_a_recorded_model_compares_and_prints_as_the_model_it_records(self) -> None:
        """Equal to an equal unrecorded model, and to an equal one recorded at another path, both ways round; unequal
        to an unequal one; printed under its own class's name. Each recorded path is a class of its own, and both
        pydantic's equality and the core shapes' require one class on both sides."""
        sent = ClientRequestSettings(max_tokens=1024, reasoning_effort="high", reasoning_max_tokens=None)
        other = ClientRequestSettings(max_tokens=2048, reasoning_effort="high", reasoning_max_tokens=None)
        storage, _ = memory_storage()
        stored = make_eval_result(
            id="r-1",
            rubric_scores=[
                RubricScore(
                    dim="conversation.tone", score=4, scale="ordinal", axis="capability", judge_sent_settings=sent
                )
            ],
            transcript_score=RubricScore(
                dim=TRANSCRIPT_DIM_ID, score=4, scale="ordinal", axis="capability", judge_sent_settings=sent
            ),
        )
        storage.save_eval_result(stored)

        with recording_results():
            (result,) = storage.query_eval_results(stored.scope_id)
        transcript = result.transcript_score
        assert transcript is not None
        at_transcript = transcript.judge_sent_settings
        at_rubric = result.rubric_scores[0].judge_sent_settings
        assert at_transcript is not None and at_rubric is not None

        assert type(at_transcript) is not type(at_rubric) and type(at_transcript) is not ClientRequestSettings
        assert at_transcript == sent and sent == at_transcript and not at_transcript != sent
        assert at_transcript == at_rubric and at_rubric == at_transcript
        assert at_transcript != other and other != at_transcript
        assert at_transcript != transcript and transcript != at_transcript, "a model of another class is unequal"
        assert result == stored and stored == result
        assert repr(at_transcript) == repr(sent) and str(at_transcript) == str(sent)
        assert repr(result).startswith("EvalResult(")

    def test_a_result_read_outside_the_recorder_is_the_stored_class(self) -> None:
        storage, _ = memory_storage()
        storage.save_eval_result(make_eval_result(id="r-1"))

        (result,) = storage.query_eval_results("uni-1")

        assert type(result) is EvalResult


@pytest.mark.parametrize(("family", "name"), READS, ids=[f"{family}-{name}" for family, name in READS])
def test_each_family_reads_exactly_its_pinned_attributes(family: str, name: str) -> None:
    """Over each campaign a family reads exactly its pinned set less what that campaign leaves unread
    (:data:`UNREAD`), so over its campaigns together it reads exactly its pinned set; its judge prose only where
    pinned; and (the bundle) its trials section reads the prose of the trials it lists and of no other result (at most
    ``MAX_TRIALS`` per aspect, never the whole campaign)."""
    measurement = measured(family, name)
    read = measurement.access.paths
    expected = PINNED[family] - UNREAD[family].get(name, frozenset())

    assert not measurement.moved, f"the recorder moved {family}'s output over {name}"
    if name == "regressed":
        assert measurement.regressed_steps > 0, "the regression watch was never checked on a regressed step"
    problems = [
        f"{family} reads {path} over {name}, outside its pinned set, at "
        f"{', '.join(sorted(measurement.access.sites[path]))}: the reading must carry it, and the pin add it, "
        "deliberately"
        for path in sorted(read - PINNED[family])
    ]
    problems += [
        f"{family} reads {path} over {name}, pinned as unread there (UNREAD), at "
        f"{', '.join(sorted(measurement.access.sites[path]))}"
        for path in sorted((read & PINNED[family]) - expected)
    ]
    problems += [
        f"{family} no longer reads {path} over {name}: unpin it there (UNREAD), or from the family if no campaign "
        "reads it, so the reading carries nothing it need not"
        for path in sorted(expected - read)
    ]
    prose = frozenset(
        f"{path} at {_where(site)}" for path, by_site in measurement.access.prose.items() for site in by_site
    )
    expected_prose = prose_pins(PROSE_SITES[family]) - PROSE_UNREAD[family].get(name, frozenset())
    problems += [
        f"{family} reads judge prose {pin} over {name}, unpinned there" for pin in sorted(prose - expected_prose)
    ]
    problems += [
        f"{family} no longer reads judge prose {pin} over {name}: unpin it there (PROSE_UNREAD), or from the family"
        for pin in sorted(expected_prose - prose)
    ]
    if measurement.listed_trials is not None:
        listed = measurement.listed_trials
        if not listed:
            problems.append(f"{name}'s bundle lists no trial, so its trials section's prose reads check nothing")
        for path, by_site in measurement.access.prose.items():
            for site, reads in by_site.items():
                if _where(site).startswith("analysis/bundle/trials.py ") and not reads.results <= listed:
                    problems.append(f"{name}: {path} read at {site} of {len(reads.results - listed)} unlisted results")
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_every_pin_is_read_by_some_campaign(family: str) -> None:
    """A path, or a prose site, unread by every campaign is a pin the reading need not carry: the unread tables name only
    what each campaign leaves out of what some other campaign reads."""
    _, names = FAMILIES[family]
    never = frozenset.intersection(*(UNREAD[family].get(name, frozenset()) for name in names))

    never_prose = frozenset.intersection(*(PROSE_UNREAD[family].get(name, frozenset()) for name in names))

    assert not never, f"{family}: pinned but unread by every campaign: {sorted(never)}"
    assert not never_prose, f"{family}: prose pinned but read by no campaign: {sorted(never_prose)}"
    assert set(UNREAD[family]) <= set(names) and set(PROSE_UNREAD[family]) <= set(names)
    assert all(paths <= PINNED[family] for paths in UNREAD[family].values())
    assert all(pins <= prose_pins(PROSE_SITES[family]) for pins in PROSE_UNREAD[family].values())
