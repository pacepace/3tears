"""Which ``EvalResult`` attributes each reader family reads, measured and pinned (W4b P1).

The compact per-result reading (``ResultReading``) carries exactly the attributes the readers access. This file
measures that set: each reader family runs over the oracle's campaigns with every result it reads served through
the access recorder (:mod:`~packages.evals.tests.result_access_recorder`), and the attribute paths it read are
compared with the family's pinned set (:data:`PINNED`).

- **bundle:** ``assemble_context_bundle`` (through ``assemble_campaign_bundle``, every aspect) and ``report_read``
  code-only, in every form with both trial details;
- **aggregates:** ``pivot``, ``history``, ``frontier``, ``program_budget``, ``orphaned_runs``, ``comparison_sets``,
  and the regression watch's ``scope_history`` and judged tiers (checked on a step that regressed, so the tiers are
  read);
- **planning:** ``detectable_effects``, which reads its history through ``planning_readings``.

The campaigns: the production-shaped corpus at ``scale=1`` and its two-aspect variant, the toy host's corpus campaign,
and the courier's planner bake-off (read back from the oracle's capture).

**A red names the attribute and where it was read.** A family reading an attribute outside its pinned set means the
reading must add it, deliberately; a pinned attribute no campaign reads any more means the reading carries one it
need not. Either way the pin moves with the reading, never alone.

**Judge prose is pinned by where it is read** (:data:`PROSE_SITES`): a score's reasoning, the judge's top-level
reasoning and the judge's "cannot tell" notes, each as a read of its text or as a truth test (``bool(...)``), at
``module function``.

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
from threetears.evals.schema.models import TRANSCRIPT_DIM_ID, EvalResult, EvalRun, RubricScore
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


CAMPAIGNS = ("scaled", "scaled_two_aspects", "toyhost", "courierhost")


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
    "planning": (read_planning, ("scaled", "toyhost", "courierhost")),
}


@dataclass(frozen=True)
class Measured:
    """One family read over its campaigns: what it read, over all of them and over each, each campaign whose output
    the recorder moved, and how many regressed steps the regression watch was checked on."""

    access: ResultAccess
    by_campaign: dict[str, ResultAccess]
    moved: tuple[str, ...]
    regressed_steps: int
    listed_trials: dict[str, set[str]]


def listed_trials(outputs: list[str], n_aspects: int) -> set[str]:
    """The ids of the results the bundle family's aspect bundles list as trials (its first ``n_aspects`` outputs, each
    an aspect's JSON then its fingerprint)."""
    decoder = json.JSONDecoder()
    return {
        str(row["result_id"])
        for output in outputs[:n_aspects]
        for row in decoder.raw_decode(output)[0].get("trials", [])
    }


@functools.cache
def measured(family: str) -> Measured:
    """``family`` read over every campaign it is read over, recorded, once per process."""
    reader, names = FAMILIES[family]
    access = ResultAccess()
    by_campaign: dict[str, ResultAccess] = {}
    listed: dict[str, set[str]] = {}
    moved = []
    steps_read = 0
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(analysis_service, "utc_now_iso", lambda: REPORT_NOW)
        for name in names:
            read = campaign(name)
            args: tuple[Any, ...] = ()
            if family == "aggregates":
                steps = regressed_steps(read)
                steps_read += len(steps)
                args = (steps,)
            family_reader = functools.partial(reader, lenses=False) if name == "regressed" else reader
            read_access, outputs, plain = recorded(read, family_reader, *args)
            access.merge(read_access)
            by_campaign[name] = read_access
            if family == "bundle":
                # The family's outputs are each aspect's bundle, then the campaign's frame, then each page.
                listed[name] = listed_trials(outputs, len(outputs) - 1 - len(PAGES))
            if outputs != plain:
                moved.append(name)
    return Measured(access, by_campaign, tuple(moved), steps_read, listed)


def _where(site: str) -> str:
    """A call site as a prose pin names it: its module and function, without the line that moves with every edit."""
    location, function = site.split(" ", 1)
    return f"{location.rsplit(':', 1)[0]} {function}"


#: Every attribute path each family reads of a result, over every campaign it is read over (see the module
#: docstring for the path syntax). Sizes: bundle 129, aggregates 75, planning 109.
PINNED: dict[str, frozenset[str]] = {
    "bundle": frozenset(
        {
            "account_refused",
            "async_deliveries",
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
            "judge_reasoning",
            "judge_repeats",
            "judge_rescores",
            "judge_score()",
            "judge_scores()",
            "judge_seconds",
            "k_iteration",
            "kind_payload",
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
            "usage[].served_model",
            "usage[].unanswered_lost",
            "usage[].unanswered_reasons",
            "usage[].unanswered_retried",
            "usage[].unpriced_reasons",
            "variant_key",
            "world_events",
        }
    ),
    "aggregates": frozenset(
        {
            "async_deliveries",
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
            "judge_rescores",
            "judge_score()",
            "judge_scores()",
            "k_iteration",
            "latency",
            "latency.total_ms",
            "model",
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
            "usage[].served_model",
            "variant_key",
        }
    ),
    "planning": frozenset(
        {
            "account_refused",
            "async_deliveries",
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
            "judge_reasoning",
            "judge_repeats",
            "judge_rescores",
            "judge_seconds",
            "k_iteration",
            "kind_payload",
            "latency",
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
            "outcome_score.reasoning",
            "outcome_score.scale",
            "outcome_score.score",
            "outcome_score.served_model",
            "phase_timings",
            "precondition_outcomes",
            "rubric_scores",
            "rubric_scores[].axis",
            "rubric_scores[].criterion_fingerprint",
            "rubric_scores[].dim",
            "rubric_scores[].judge_sent_settings",
            "rubric_scores[].judge_sent_settings.model_dump()",
            "rubric_scores[].judge_temperature",
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
            "transcript_score.judge_sent_settings.model_dump()",
            "transcript_score.judge_temperature",
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
            "usage[].served_model",
            "usage[].unanswered_lost",
            "usage[].unanswered_reasons",
            "usage[].unanswered_retried",
            "usage[].unpriced_reasons",
            "variant_key",
            "world_events",
        }
    ),
}

#: Where each family reads judge prose: each prose path (a text read, or ``bool(...)`` for a truth test) and the
#: ``module function`` sites that read it.
PROSE_SITES: dict[str, dict[str, frozenset[str]]] = {
    "bundle": {
        "judge_cannot_tell": frozenset(
            {
                "analysis/bundle/measures.py _carrier_leaves",
                "analysis/bundle/measures.py _scalar_leaves",
                "analysis/bundle/trials.py _row",
                "analysis/reporting.py project_score_records",
                "kernel/result_condition.py judged_on_every_dim",
                "kernel/scoring.py has_pass_criterion",
            }
        ),
        "judge_reasoning": frozenset(
            {
                "analysis/bundle/measures.py _carrier_leaves",
                "analysis/bundle/measures.py _scalar_leaves",
            }
        ),
        "outcome_score.reasoning": frozenset(
            {
                "analysis/bundle/measures.py _scalar_leaves",
                "analysis/bundle/trials.py _row",
            }
        ),
        "rubric_scores[].reasoning": frozenset(
            {
                "analysis/bundle/measures.py _scalar_leaves",
                "analysis/bundle/trials.py _row",
            }
        ),
        "transcript_score.reasoning": frozenset(
            {
                "analysis/bundle/measures.py _scalar_leaves",
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
    },
    "planning": {
        "judge_cannot_tell": frozenset(
            {
                "analysis/bundle/measures.py _carrier_leaves",
                "analysis/bundle/measures.py _scalar_leaves",
                "analysis/reporting.py project_score_records",
                "kernel/result_condition.py judged_on_every_dim",
            }
        ),
        "judge_reasoning": frozenset(
            {
                "analysis/bundle/measures.py _carrier_leaves",
                "analysis/bundle/measures.py _scalar_leaves",
            }
        ),
        "outcome_score.reasoning": frozenset(
            {
                "analysis/bundle/measures.py _scalar_leaves",
            }
        ),
        "rubric_scores[].reasoning": frozenset(
            {
                "analysis/bundle/measures.py _scalar_leaves",
            }
        ),
        "transcript_score.reasoning": frozenset(
            {
                "analysis/bundle/measures.py _scalar_leaves",
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

    def test_a_result_read_outside_the_recorder_is_the_stored_class(self) -> None:
        storage, _ = memory_storage()
        storage.save_eval_result(make_eval_result(id="r-1"))

        (result,) = storage.query_eval_results("uni-1")

        assert type(result) is EvalResult


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_each_family_reads_exactly_its_pinned_attributes(family: str) -> None:
    measurement = measured(family)
    read = measurement.access.paths

    assert not measurement.moved, f"the recorder moved {family}'s output over {measurement.moved}"
    if family == "aggregates":
        assert measurement.regressed_steps > 0, "the regression watch was never checked on a regressed step"
    problems = [
        f"{family} reads {path}, outside its pinned set, at {', '.join(sorted(measurement.access.sites[path]))}: "
        "the reading must carry it, and the pin add it, deliberately"
        for path in sorted(read - PINNED[family])
    ]
    problems += [
        f"{family}'s pinned set holds {path}, which no campaign read: the reading need not carry it"
        for path in sorted(PINNED[family] - read)
    ]
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_judge_prose_is_read_only_where_it_is_pinned(family: str) -> None:
    measurement = measured(family)
    observed = {path: frozenset(_where(site) for site in by_site) for path, by_site in measurement.access.prose.items()}

    assert observed == PROSE_SITES[family]


def test_the_trials_section_reads_judge_prose_only_of_the_trials_it_lists() -> None:
    """The bundle's trials carry each listed trial's judge reasoning (at most ``MAX_TRIALS`` per aspect): it is read of
    those results and no other, never of the whole campaign's."""
    measurement = measured("bundle")
    problems = []
    for name, access in measurement.by_campaign.items():
        listed = measurement.listed_trials[name]
        assert listed, f"{name}'s bundle lists no trial, so this checks nothing there"
        for path, by_site in access.prose.items():
            for site, reads in by_site.items():
                if _where(site).startswith("analysis/bundle/trials.py ") and not reads.results <= listed:
                    problems.append(f"{name}: {path} read at {site} of {len(reads.results - listed)} unlisted results")
    assert not problems, "\n".join(problems)
