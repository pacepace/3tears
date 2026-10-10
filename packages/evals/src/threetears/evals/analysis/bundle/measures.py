"""The reflective measure walk: every measure a campaign's results carry, pooled into the bundle's measure surface.

:func:`_collect_measures` walks each result's scalar leaves, record carriers, derived and lineage leaves and open
maps, keeps the measures the registry describes as reportable, and summarises each over its own population
(:func:`_measure_summary`). :func:`goal_check_proofs_of` reads the goal-check proofs the results recorded.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Iterator, Sequence
from itertools import chain
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from threetears.evals.analysis.confusion import label_statistics
from threetears.evals.analysis.latency_partition import decompose_total_ms
from threetears.evals.analysis.stats import (
    clustered_standard_error,
    observed_mean_interval,
    proportion_interval,
    small_sample_case_means,
)
from threetears.evals.kernel.analysis_measures import MeasureCollection, MeasureSummary
from threetears.evals.kernel.scoring import median_unbiased_quantile
from threetears.evals.kernel.host.profile import HostProfile
from threetears.evals.kernel.metrics import (
    ACCURACY_MEASURE,
    CONFUSION_CELL_MEASURE,
    MATCH_MEASURE,
    AttributionScope,
    ClassifierStatistic,
    MeasurePopulation,
    MetricDescriptor,
    classifier_label_measure,
    confusion_of,
    describe_measure,
    describe_phase_timing,
    goal_check_measure,
    is_code_graded,
    summary_population,
    undeclarable_host_measures,
)
from threetears.evals.kernel.covariates import undeclarable_covariates

# At runtime for its field set, which tells a result-level measure from a row-level one.
from threetears.evals.schema.models import (
    goal_check_proofs_as_read,
    stale_goal_check_proofs,
    EvalResult,
    GoalCheckProof,
)
from threetears.evals.kernel.result_condition import (
    ResultOutcome,
    classify_result,
    counted_goal_verdicts,
    delivered_a_turn,
    harness_faulted,
)
from threetears.evals.kernel.usage_capture import (
    count_substituted_deliveries,
    production_replicating_cost,
    spend_observed,
)
from threetears.evals.analysis.bundle.schema import GoalCheckProofReading

if TYPE_CHECKING:  # runtime models — TYPE_CHECKING-only to keep the runtime import graph minimal.
    from threetears.evals.kernel.host.measures import MeasureRegistry
    from threetears.evals.schema.models import EvalRun


# The registry's full attribution-scope vocabulary, so a scope that observed nothing can
# be named as absent rather than being missing. Ordered end-to-end first, matching how a
# reader narrows: the whole run, then the part of it under test.
_ATTRIBUTION_SCOPES: tuple[AttributionScope, ...] = ("end_to_end", "subsystem")

# The observation unit of a measure the result carries at most once — its own scalars, its
# open maps, and any single sub-model's leaves. Everything else is named for the list it
# rides on, so two units compare equal only when one observation of each describes the same
# thing. See ``_collect_measures``.
_PER_RESULT = "result"

# The result's blended spend, as a measure — the one lineage leaf that is read only where it was observed.
_COST_MEASURE = "cost_usd"

#: The spend belonging to the roles production runs — the candidate's cost, without the judge's. The one cost
#: a contrast between arms is tested on (:func:`_per_case_values`); ``cost_usd`` is what it cost to measure.
_CANDIDATE_SPEND = "production_replicating_cost"


def _percentile(sorted_values: list[float], q: float) -> float | None:
    """A measure's ``q`` quantile, median-unbiased, or ``None`` where its sample cannot give one.

    :func:`~threetears.evals.kernel.scoring.median_unbiased_quantile` (Hyndman–Fan type 8), the rule the
    run summary's ``p95_total_ms`` reads too, so a tail figure means one thing on every surface. It replaced
    linear interpolation (numpy's default), which at the sizes a campaign's cells have sat below the true
    95th percentile 0.84 (n=5), 0.73 (n=15) and 0.68 (n=30) of the time — a tail figure that understates
    the tail. The median is unchanged by the switch: type 8 and linear interpolation place it alike.

    Args:
        sorted_values: Ascending-sorted values, at least one element.
        q: Quantile in (0, 1) — NOT 0-100.

    Returns:
        The estimate, or ``None`` for ``p05``/``p95`` below 13 observations, where no estimate is
        median-unbiased; ``max`` beside it is then the worst case seen, under its own name.
    """
    return median_unbiased_quantile(sorted_values, q)


def _is_reportable(descriptor: MetricDescriptor, measures: MeasureRegistry) -> bool:
    """Whether a measure belongs on the bundle's measure surfaces — what the generator reads and ranks from.

    Three filters, each excluding a class of measure that would otherwise mislead:

    - **Seeded only.** ``family is None`` marks a name nobody has described; the
      registry itself refuses to guess at one, and pooling it here would be that same
      guess made silently.
    - **Code-graded families only** (:func:`~threetears.evals.kernel.metrics.is_code_graded`, the
      predicate the bar-name resolver asks too, so the two cannot disagree). The generator ranks on
      mechanism, never on judged quality — and quality already has a home in the ``reporting`` lenses.
      This is a filter on registry *metadata*, not on a subject or scenario type. A host's own family
      is admitted exactly when the host declared it ``graded_by="code"``.

      **``classifier`` is admitted beside ``mechanical``, and the omission was a real
      defect**: the descriptive-telemetry rule's ranking half is about JUDGE scores — "no judge can score
      a run that produced nothing to grade" — and a classifier's grade has no judge
      anywhere in it. It is code compared against an expected label, which is the same
      kind of fact ``mechanical`` names. While the classifier was a second path its
      measures never reached a bundle and the proxy cost nothing; folding it in made
      every classifier campaign assemble a bundle carrying the parse-failure rates and
      **not accuracy, precision or recall** — the figures the campaign exists to produce.
      The remaining exclusions: ``rubric`` and ``dual_axis`` are the judge-mediated ones
      that rule actually names, and they stay out permanently. ``composite`` stays out because
      **no producer puts one on a bundle surface**, so admitting it would be widening on a
      case nothing exercises — which is how the ``mechanical``-only proxy came to be written.

      **``goal_state`` is admitted, on the classifier's argument**:
      a check's verdict is code compared against what the candidate did. Its exclusion had
      rested only on the no-producer claim, and that claim was about the check's own text —
      ``GoalStateOutcome`` carries ``expression`` / ``passed`` / ``detail``, so the generic
      carrier walk yields names and never a verdict. :func:`_goal_check_leaves` is the
      producer: one 0/1 per check per observation, named by
      :func:`~threetears.evals.kernel.metrics.goal_check_measure`, so each check's pass rate reaches every
      measure surface whether or not a bar names it.
    - **A numeric measure must have a direction, unless it is a declared diagnostic.**
      ``higher_is_better is None`` marks a raw count with no better end — per-role token
      counts, call counts. Nothing can be ranked on one, and pooling a per-role count
      across the candidate, judge and simulator rows produces a distribution of nothing in
      particular. A measure whose descriptor declares
      :attr:`~threetears.evals.kernel.metrics.MetricDescriptor.diagnostic` is the one exception,
      and a declared one — the engine's own (the candidate provider's output rate) and a host's
      (a signed error against what was asked) alike, through this one predicate: it has no better
      end either, but it explains a movement and a reader needs it beside the measures it
      explains. Declared rather than inferred, because nothing else on a descriptor tells a
      diagnostic from a count, and a guess that admitted counts would reopen the pooling defect.
      It is carried here so the run summaries, the catalog
      and the divergence lens see it; its missing direction is what keeps every direction-
      reading surface — a superlative, a bar — from treating it as a merit. Categorical
      measures are kept without a direction: they carry the *how did it conclude* signal
      (a forced-vs-voluntary split) that is ranked on as a rate, not a value. **Boolean and text
      measures are kept too**: a boolean is summarised as a rate with an interval and a text
      measure is listed as evidence, and neither is ever averaged into a distribution.
    """
    if not is_code_graded(descriptor, measures):
        return False
    if descriptor.data_type in ("categorical", "boolean", "text"):
        return True
    if descriptor.data_type != "numeric":
        return False
    return descriptor.higher_is_better is not None or descriptor.diagnostic


def _value_fits(descriptor: MetricDescriptor, value: float | str) -> bool:
    """Whether an observation's Python type agrees with what its descriptor claims.

    ``covariates`` is an open ``dict[str, str | float]``, so a value's type is not
    guaranteed to match the descriptor its key resolves to. Without this check a string
    arriving under a seeded-numeric name reaches ``float(value)`` and raises out of
    ``assemble_context_bundle``, destroying the whole analysis over one bad observation.
    Dropping the observation instead keeps the blast radius at one measure — and the drop
    is reported rather than silent (see ``MeasureCollection.unreported_observations``).
    """
    if descriptor.name == CONFUSION_CELL_MEASURE:
        # A confusion cell is categorical AND has a format: one that does not split into its two labels
        # could not be counted into the matrix the per-label statistics are derived from.
        return isinstance(value, str) and confusion_of(value) is not None
    if descriptor.data_type in ("categorical", "text"):
        return isinstance(value, str)
    if descriptor.data_type == "boolean":
        return isinstance(value, bool)
    return not isinstance(value, (str, bool))


def _scalar_leaves(model: BaseModel) -> Iterator[tuple[str, float | str]]:
    """Yield ``(field_name, value)`` for each present numeric/string scalar on a model.

    Booleans are excluded despite being ints in Python: a boolean measure is a
    condition, and averaging one into a percentile reads as a rate nobody computed.
    """
    for field_name in type(model).model_fields:
        value = getattr(model, field_name, None)
        if value is None or isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            yield field_name, float(value)
        elif isinstance(value, str):
            yield field_name, value


#: Result fields whose sub-models RECORD something done to the result rather than something
#: measured in its cell, so the measure walk must not read their scalars as observations. A
#: re-judge's record carries its own ``cost_usd`` — spend an operator paid after the run —
#: and walking it would pool that as a second cost observation of the cell, beside the prose
#: of its timestamp, prior error and model reported as measurements the registry lost. A repeat
#: of the judge's scores is a measurement of the JUDGE, read by ``judge_self_agreement``; walking
#: it would pool a repeat's score as a second observation of the candidate's cell. A second judge's scores are the
#: same: a measurement of the judges, read by ``inter_judge_agreement`` and ``judge_drift``.
RECORD_CARRIERS: frozenset[str] = frozenset({"judge_rescores", "judge_repeats", "judge_seconds"})


def _carrier_leaves(result: EvalResult, *, profile: HostProfile) -> Iterator[tuple[str, float | str, bool, str, str]]:
    """Yield ``(name, value, report_gaps, carrier, observation_unit)`` for every sub-model's leaves.

    This is the whole subject-agnostic mechanism: a result announces what it measured
    by *carrying* it, and the registry says what each name means. A subsystem that
    later lands its own lifecycle carrier is reported here with no change to this
    module — which is the property that makes the surface genuinely subject-agnostic
    rather than shaped for one tool with the name filed off.

    The carrier identity yielded last is the **field name** on the result, not the record
    class. Two subsystems can share a record type, subclass one, or land a generic
    lifecycle record; they cannot share a field, so the field is what distinguishes
    "these are two different quantities" from "these are more observations of one".

    ``report_gaps`` says whether an *undescribed* leaf of this carrier is worth reporting
    as a lost measurement. It is True only when the registry already describes at least one
    of the carrier's leaves, which separates the two ways a carrier can be undescribed:

    - **A described carrier that grew an undescribed field** — the likely drift, and a real
      loss. Reported.
    - **A carrier whose names are an OPEN name space** — judged rubric dimensions and
      goal-state expressions, which ``metrics.py`` deliberately does not enumerate and
      resolves through ``describe_rubric_dim`` / ``describe_goal_state`` instead. Nothing
      is lost, and reporting them would bury the real signal in permanent false positives.

    The blind spot is deliberate: a carrier with *no* described leaf at all — a wholly
    unregistered subsystem — reports nothing here, because it is indistinguishable at
    runtime from the open-name-space case.

    The last value yielded is the leaf's **observation unit** — what one observation of it
    describes. A single sub-model contributes at most one observation per result, so its
    measures are per-result (``'result'``) and may be differenced against each other; a LIST
    contributes one per element, so its measures are means per element of that list
    (``'usage[]'``, ``'async_deliveries[]'``) and differencing one against a per-result
    mean is wrong by however many elements a result carried. The two are indistinguishable
    once the values are pooled — both arrive in the measure's own honest unit — so the walk
    is the only place the difference can be observed, and it says so by name rather than
    leaving each comparison site to re-derive it.
    """
    for field_name in type(result).model_fields:
        if field_name in RECORD_CARRIERS:
            continue
        value = getattr(result, field_name, None)
        repeatable = isinstance(value, list)
        carriers: list[Any] = [value] if isinstance(value, BaseModel) else value if isinstance(value, list) else []
        observation_unit = f"{field_name}[]" if repeatable else _PER_RESULT
        for item in carriers:
            if not isinstance(item, BaseModel):
                continue
            leaves = list(_scalar_leaves(item))
            report_gaps = any(describe_measure(name, profile.measures).family is not None for name, _ in leaves)
            for name, leaf in leaves:
                yield name, leaf, report_gaps, field_name, observation_unit


def _derived_leaves(result: EvalResult) -> Iterator[tuple[str, float | str, bool, str, str]]:
    """Yield the measures a result implies but does not carry, in the carrier-leaf shape.

    Two. ``candidate_output_tokens_per_s`` is the candidate provider's output rate — see
    :func:`_candidate_output_throughput` — which is what separates a latency difference a
    lever caused from one the provider's load caused. The other is ``orchestration_ms``, the
    named remainder that closes the ``total_ms`` partition. Both are derived rather than
    captured on purpose — their inputs already settle them, so persisting either would be a
    second answer to one question — but a derived measure the bundle never walks is one the
    analysis cannot rank on, and this surface is the only way a subsystem reaches a report at all.

    Without it, a whole-run latency movement that lived in orchestration could only be
    reported as ``total_ms`` moving by more than ``llm_ms`` and ``tool_ms`` account for, which
    is indistinguishable in a report from a measurement fault. With it the movement has
    a component to be attributed to, and — because the registry declares it
    ``contained_by: total_ms`` — the divergence lens will difference it against the
    whole rather than refusing.

    Both are yielded under the ``latency`` carrier and at the per-result observation unit,
    matching the components they are computed from: a remainder pooled over a different
    unit from its own parts would be exactly the mismatch this decomposition exists to remove.
    """
    partition = decompose_total_ms(result.latency)
    if partition.orchestration_ms is not None:
        # `report_gaps=True`: the latency carrier's other leaves are all described, so
        # an undescribed one here would be real drift rather than an open name space.
        yield "orchestration_ms", partition.orchestration_ms, True, "latency", _PER_RESULT
    if (throughput := _candidate_output_throughput(result)) is not None:
        yield "candidate_output_tokens_per_s", throughput, True, "latency", _PER_RESULT


def goal_check_proofs_of(runs: Sequence[EvalRun], results: Iterable[EvalResult]) -> list[GoalCheckProofReading]:
    """Each goal check the runs' results graded, with the proof its runs froze at launch, in the order first met.

    Args:
        runs: The member runs.
        results: Their results.

    Returns:
        One reading per check.
    """
    graded: dict[str, list[str]] = {}
    for result in results:
        for outcome in result.goal_state_outcomes:
            runs_of = graded.setdefault(outcome.expression, [])
            if result.eval_run_id not in runs_of:
                runs_of.append(result.eval_run_id)
    by_id = {run.id: run for run in runs}
    refusals: dict[str, tuple[str, list[str]]] = {}
    for run in runs:
        for check, reason in (run.refused_goal_checks or {}).items():
            refusals.setdefault(check, (reason, []))[1].append(run.id)
    readings = []
    for check, run_ids in graded.items():
        members = [by_id[run_id] for run_id in run_ids if run_id in by_id]
        # As read under the current proof rules: a `proven` an older rule stamped is unproven (#665).
        recorded = [goal_check_proofs_as_read(run) for run in members]
        proofs = [None if record is None else record.get(check, "unproven") for record in recorded]
        refused = refusals.get(check)
        proof: GoalCheckProof = (
            "refuted"
            if "refuted" in proofs or refused is not None
            else "proven"
            if proofs and all(each == "proven" for each in proofs)
            else "unproven"
        )
        readings.append(
            GoalCheckProofReading(
                check=check,
                measure_id=goal_check_measure(check),
                proof=proof,
                runs=max(len(run_ids), 1),
                unrecorded=sum(1 for each in proofs if each is None),
                stale=sum(1 for run in members if check in stale_goal_check_proofs(run)),
                refused=None if refused is None else refused[0],
            )
        )
    # A check the grammar refused is graded on no cell, so no result names it; it is read from the runs.
    readings.extend(
        GoalCheckProofReading(
            check=check,
            measure_id=goal_check_measure(check),
            proof="refuted",
            runs=len(run_ids),
            unrecorded=0,
            refused=reason,
        )
        for check, (reason, run_ids) in refusals.items()
        if check not in graded
    )
    return readings


def _goal_check_leaves(result: EvalResult) -> Iterator[tuple[str, float | str, bool, str, str]]:
    """Yield each goal-state check's verdict on this observation, 1.0 passed and 0.0 not.

    The mechanical tier's verdicts, which the bundle carries unconditionally: a bar is a
    threshold someone chose to hold a check to, never the condition for the check's pass rate
    reaching a reader. The generic carrier walk cannot supply them — a ``GoalStateOutcome``'s
    scalars are its expression and detail as TEXT, and ``passed`` is a boolean the walk skips —
    so they are yielded here, one per check, under the name the registry mints for a check's
    measure. Per result, since an observation evaluates each of its checks once. Each verdict is
    the one :func:`~threetears.evals.kernel.result_condition.counted_goal_verdicts` says every rate counts, so
    this and the pivot cannot disagree: none from a harness-faulted result, whose checks read the
    harness (the cells already skip it through :func:`_non_faulted`, but run summaries, the
    telemetry rollup and the scope divergences collect over every result, and without this they
    would report a second rate for the same check), and a failure for every check on a candidate
    failure.
    """
    counted = counted_goal_verdicts(result)
    if counted is None:
        return
    for outcome, passed in counted:
        yield goal_check_measure(outcome.expression), 1.0 if passed else 0.0, False, "goal_state_outcomes", _PER_RESULT


def _failures_as_misses(results_by_run: dict[str, list[EvalResult]]) -> dict[str, list[EvalResult]]:
    """Each run's results, with a classifier's failure that landed no verdict read as a miss.

    A classifier kind lands ``match`` on every classification, and a call its model refused lands nothing
    unless the kind says otherwise — the quick callable kind does, a host's kind may not. Left absent, the
    failure was in no rate: not the match rate, not ``accuracy``, not a comparison's per-case values, so an
    arm that refused the cases it would have got wrong read MORE accurate than one that answered them, and
    one that refused everything had no accuracy to compare at all. So a candidate failure carrying no
    ``match`` is read with ``match`` False — the rule
    :func:`~threetears.evals.kernel.result_condition.counted_goal_verdicts` keeps for every goal check —
    when both of these hold:

    - **its case is a classification**: some result in the campaign landed ``match`` on that test case. A
      case nothing classified is not one a failure could have missed, whatever kind ran it.
    - **its run classifies**: no result the run delivered without failing lacks ``match`` on a case that is
      a classification. A run that answers a classified case without landing a verdict grades by something
      else (a scorer-only run of the same callable kind, beside a classifier run over the same cases), and
      giving its failures an accuracy would invent one. A run that delivered nothing has shown no such
      evidence, and its refusals are misses.

    Copied, never written back: what the kind stored is untouched.

    Args:
        results_by_run: Each run's stored results.

    Returns:
        The same results, each such failure replaced by a copy carrying ``match`` False.
    """
    classified_cases = {
        result.test_case_id
        for members in results_by_run.values()
        for result in members
        if isinstance(result.host_measures.get(MATCH_MEASURE), bool)
    }
    grading_otherwise = {
        run_id
        for run_id, members in results_by_run.items()
        if any(
            result.test_case_id in classified_cases
            and MATCH_MEASURE not in result.host_measures
            and classify_result(result) is ResultOutcome.OK
            for result in members
        )
    }

    def read(run_id: str, result: EvalResult) -> EvalResult:
        if (
            run_id not in grading_otherwise
            and result.test_case_id in classified_cases
            and MATCH_MEASURE not in result.host_measures
            and classify_result(result) is ResultOutcome.CANDIDATE_FAIL
        ):
            return result.model_copy(update={"host_measures": {**result.host_measures, MATCH_MEASURE: False}})
        return result

    return {run_id: [read(run_id, result) for result in members] for run_id, members in results_by_run.items()}


def _accuracy_leaves(result: EvalResult) -> Iterator[tuple[str, float | str, bool, str, str]]:
    """Yield the observation's classifier accuracy, 1.0 matched and 0.0 not, derived from its ``match``.

    ``match`` is the boolean a classifier kind lands, and a boolean summarises as a rate with no
    per-case mean, so it cannot carry the classifier's reading on the quality axis into a family of
    comparisons. ``accuracy`` is that reading, derived here from the one carried verdict — so a host
    lands one measure and every surface sees one comparison, rather than a host minting its own
    numeric copy beside ``match`` and every family testing the same verdict twice. A kind may not
    land ``accuracy`` itself (:func:`~threetears.evals.run.runner.refuse_engine_derived_host_measures`).
    Nothing is yielded for an observation that carries no ``match``, or one whose ``match`` is not a
    bool — that value is the walk's to drop and report, under ``match``'s own name.

    **A candidate failure is a miss**, whatever its ``match`` says — the rule
    :func:`~threetears.evals.kernel.result_condition.counted_goal_verdicts` keeps for every goal check,
    carried to a classifier's accuracy. A refused call answered nothing; read as anything but a miss, an arm
    that refused the cases it would have got wrong read MORE accurate than one that answered them. A
    classifier kind lands ``match`` on a failure for this reason — the quick callable kind lands it False
    with the expected label's confusion cell — so a failure is in the accuracy, the match rate and the
    per-label counts alike; and a host kind's failure that landed none is read with ``match`` False before
    the walk sees it (:func:`_failures_as_misses`). A failed observation still carrying no ``match`` is of a
    kind that classifies nothing, and yields nothing: an accuracy for it would be invented.

    Args:
        result: The observation.

    Yields:
        ``("accuracy", 1.0 | 0.0, True, "host_measures", "result")`` at most once.
    """
    matched = result.host_measures.get(MATCH_MEASURE)
    if isinstance(matched, bool):
        hit = matched and classify_result(result) is not ResultOutcome.CANDIDATE_FAIL
        yield ACCURACY_MEASURE, 1.0 if hit else 0.0, True, "host_measures", _PER_RESULT


def _candidate_output_throughput(result: EvalResult) -> float | None:
    """The candidate's output tokens per second of its own model-call time, or None.

    Both halves are the CANDIDATE's: ``llm_ms`` sums the candidate's model-call spans, and the
    tokens are summed over its own usage rows alone — a judge's or an inner agent's tokens were
    produced on a different clock and would inflate a rate over time they did not spend. Absent,
    never zero, when either half went unmeasured: a result with no candidate token count has no
    rate, and zero would read as a provider that produced nothing.

    Args:
        result: The observation.

    Returns:
        Tokens per second, or None when the candidate's output tokens or a non-zero ``llm_ms``
        is missing.
    """
    llm_ms = result.latency.llm_ms if result.latency is not None else None
    counted = [
        row.completion_tokens for row in result.usage if row.role == "candidate" and row.completion_tokens is not None
    ]
    if not llm_ms or not counted:
        return None
    return sum(counted) / (llm_ms / 1000.0)


def _withheld_derived(results: list[EvalResult], reported: Collection[str]) -> list[str]:
    """Name a derived measure no result could produce, with the reason it could not.

    A derived measure is the one kind that can vanish from the bundle without anything
    noticing. A CARRIED measure that is absent had no instrument; a derived one may have
    had every instrument and still be unreportable because an input was unmeasured — and
    reporting neither the value nor the reason hands the analysis exactly the unexplained
    gap the measure was added to close.

    Silence is still correct in two cases, and both are checked rather than assumed. When
    the measure IS reported, the mean carries its own ``n``, so partial coverage is
    already disclosed the way every other measure discloses it. When no result timed
    anything at all, nothing was withheld — the campaign simply has no latency, which
    ``absent_scopes`` is the right place for.

    Args:
        results: The results the collection was built from.
        reported: Measure names the walk successfully summarised.

    Returns:
        Entries of the form ``name (reason)``, empty when nothing was withheld. The
        reason is carried only when every withholding result gave the SAME one —
        otherwise the name alone, since a single reason chosen from several would be a
        claim about results it does not describe.
    """
    if "orchestration_ms" in reported:
        return []
    reasons = {
        p.withheld for result in results if (p := decompose_total_ms(result.latency)).withheld and result.latency
    }
    if not reasons:
        return []
    return [f"orchestration_ms ({reasons.pop()})" if len(reasons) == 1 else "orchestration_ms"]


def _lineage_leaves(result: EvalResult, *, profile: HostProfile) -> Iterator[tuple[str, float | str, MetricDescriptor]]:
    """Yield the result's own top-level scalars.

    Mostly lineage and provenance — ids, a schema version, the k index — with a couple of
    genuine measures mixed in (``cost_usd``). The registry sorts the two apart, so nothing
    here needs a hand-written exclusion list; but an *undescribed* name here is a version
    field rather than a lost measurement, which is why the caller does not report gaps
    from this source.

    **``cost_usd`` is an observation only where the result observed spend**
    (:func:`~threetears.evals.kernel.usage_capture.spend_observed`): a row in its cost roles carrying
    dollars. Without one the stored 0.0 is the sum of nothing — a candidate that reported no spend, not one
    that spent none — so it is left out here, and with it out of every cell, run, case and stratum this
    walk summarises. Decided per result, so every slicing of the same results agrees; a cell where no
    result observed spend carries no ``cost_usd`` reading at all, so nothing charts or tests it, and
    :attr:`AnalysisContextBundle.cost_unmeasured` says why.
    """
    observed = spend_observed(result.usage, result.cost_roles)
    for name, value in _scalar_leaves(result):
        if name == _COST_MEASURE and not observed:
            continue
        yield name, value, describe_measure(name, profile.measures)
    # The candidate's own spend, where the result measured one: what the arm costs, beside ``cost_usd``, what
    # it cost to measure (the judge's and simulator's spend included). Derived here rather than stored, so every
    # slicing the walk serves — a cell, a run, a case, a stratum — reads the one figure a contrast on cost tests.
    candidate_spend = production_replicating_cost(
        result.usage, substituted_deliveries=count_substituted_deliveries(result)
    )
    if candidate_spend is not None:
        yield _CANDIDATE_SPEND, candidate_spend, describe_measure(_CANDIDATE_SPEND, profile.measures)


def _open_map_leaves(
    result: EvalResult, *, profile: HostProfile
) -> Iterator[tuple[str, float | str, MetricDescriptor]]:
    """Yield the result's covariate, phase-timing and host-measure entries — all open key spaces.

    Phase timings resolve through :func:`describe_phase_timing` rather than
    :func:`describe_measure`, because a phase key is indistinguishable *by name* from a
    run-summary statistic and the bare resolver says so explicitly. The caller here
    knows which map it is reading, so it uses the resolver that knows too.

    Keys here are operator- and tool-authored rather than model fields, so a blank one is
    reachable, and the registry resolvers refuse it (correctly: an empty string is the
    absence of a name, not a name that failed to resolve). Skip rather than raise, so one
    malformed key cannot destroy the analysis it appears in.
    """
    # A core-named covariate no covariate writer lands — a result stored by another writer, or before the rule —
    # is dropped and named, never pooled, as an undeclarable host measure is (`_undeclarable_host_entries`).
    stray = set(undeclarable_covariates(result.covariates))
    for name, value in result.covariates.items():
        if name.strip() and name not in stray:
            yield name, value, describe_measure(name, profile.measures)
    for name, value in result.phase_timings.items():
        if name.strip():
            yield name, float(value), describe_phase_timing(name)
    # Host-declared measures resolve through `describe_measure`, which consults the
    # host's measure registry between the seeded core and the goal-state branch. So a name the host
    # declared arrives with the host's own descriptor — merit axis and direction included — and
    # one it did not stays undescribed and is reported as a lost measurement, exactly as an
    # unrecognised covariate is.
    #
    # The core wins a tie on the DESCRIPTOR, and only on the descriptor. A host reporting
    # `cost_usd` here would not redefine what the word means — but its numbers WOULD pool into
    # the engine's own spend distribution under that core descriptor, with `n` inflated,
    # because `record()` appends into one bucket per name at this level. So a host may not
    # DECLARE a measure named like a core one: `MeasureRegistry._defects` refuses it, and
    # `run_eval` refuses a scorer so named. A core name still arrives here legitimately — the
    # classifier track lands `match` and `confusion_cell` as host measures — and those pass. Any
    # other engine-owned key is one no host could have declared: the runner refuses a kind landing
    # one, and a result stored before that refusal has it dropped here and named as unreported
    # (`_undeclarable_host_entries`), never pooled into the engine's own observations of the name.
    smuggled = set(undeclarable_host_measures(result.host_measures))
    for name, value in result.host_measures.items():
        if name.strip() and name not in smuggled:
            yield name, value, describe_measure(name, profile.measures)


def _undeclarable_host_entries(results: Sequence[EvalResult]) -> list[str]:
    """The unreported-observation entries for host-measure keys the walk dropped as engine-owned.

    Each entry is the key with its reason in parentheses, the form
    :attr:`~threetears.evals.kernel.analysis_measures.MeasureCollection.unreported_observations` reads —
    the plain name could not carry it, because the engine's own measure of that name is usually pooled
    beside it and the bare name would read as a gap in the engine's reading rather than a drop of the host's.

    Args:
        results: The results the walk read.

    Returns:
        One entry per dropped key, sorted.
    """
    names = {name for result in results for name in undeclarable_host_measures(result.host_measures)}
    covariates = {name for result in results for name in undeclarable_covariates(result.covariates)}
    return [
        f"{name} (a host kind reported it on host_measures, where only the engine measures it; dropped, not pooled)"
        for name in sorted(names)
    ] + [
        f"{name} (a result carried it as a covariate, which no covariate writer lands; dropped, not pooled)"
        for name in sorted(covariates)
    ]


def in_population(population: MeasurePopulation, result: EvalResult) -> bool:
    """Whether a result is an observation of a measure read over ``population``.

    The one membership rule the measure walk applies, per measure and per result: a result the harness
    faulted is in ``all_observed`` only; ``delivered`` holds exactly the turns the candidate took
    (:func:`~threetears.evals.kernel.result_condition.delivered_a_turn`), so a failure that took no turn
    — a refusal, a model error — is in every population but that one, and a failure that took a turn (one
    its budget ended, its output cap cut, its deadline struck mid-call) is in all three.

    Args:
        population: The population the measure's summary is computed over
            (:func:`~threetears.evals.kernel.metrics.summary_population`).
        result: The result.

    Returns:
        True when the result's observations of the measure count.
    """
    if population == "delivered":
        return delivered_a_turn(result)
    if population == "scored":
        return not harness_faulted(result)
    return True


#: One measure's pooled observations: its descriptor, its values, and each value's test case.
_PooledMeasure = tuple[MetricDescriptor, list[float | str], list[str]]


def _measure_collection(
    results: list[EvalResult], *, profile: HostProfile, undeclared: MeasurePopulation
) -> MeasureCollection:
    """Build the scope-tagged measure surface over a set of results.

    See :func:`_collect_measures`, which this wraps for the callers that need only the
    collection and not the provenance of each measure.
    """
    return _collect_measures(results, profile=profile, undeclared=undeclared)[0]


def _collect_measures(
    results: list[EvalResult], *, profile: HostProfile, undeclared: MeasurePopulation
) -> tuple[MeasureCollection, dict[str, str], dict[str, _PooledMeasure]]:
    """Build the measure surface, and say what one observation of each measure describes.

    **Each measure is computed over its own population** (``MetricDescriptor.population``): a
    ``scored`` measure leaves out every result the harness faulted, an ``all_observed`` one keeps
    them, a ``delivered`` one holds only the turns the candidate took
    (:func:`~threetears.evals.kernel.result_condition.delivered_a_turn`), and every summary states which
    it was. ``results`` is therefore EVERY result in scope, faulted and failed ones included — the walk
    does the excluding, per measure, so a cell, a bar, a run summary and a divergence lens reporting
    one measure name report it over one population. A measure that declares none is computed over
    ``undeclared``, the population of the surface asking: ``scored`` for the decision surface's cells
    and bars, ``all_observed`` for a run's summary and the rollups, which is what each of those always
    computed — except a cost or latency measure, which every surface reads over ``delivered``
    (:func:`~threetears.evals.kernel.metrics.summary_population`): a refused call is a failure every
    rate counts against its arm, and not a 50 ms turn costing nothing on any of them.

    The second return value maps every pooled measure to its **observation unit**:
    ``'result'`` for the result's own scalars, its open maps, and the leaves of any single
    sub-model — at most one observation each — against one element of a repeatable LIST
    record (``'usage[]'``, ``'async_deliveries[]'``), where a result contributes as many
    observations as it carried elements. Two means over the same unit may be differenced; a
    per-result mean and a per-delivery mean may not, and nothing in the values themselves
    says which is which (both are honestly in the measure's own unit). Only the walk knows,
    so the walk is what names it — a count-based guess would misread the ordinary case of a
    measure simply missing from one result.

    A name observed at BOTH levels takes the outer unit, for the same reason the outer
    values win: a per-role ``cost_usd`` row decomposes the result-level ``cost_usd``, and
    what the collection publishes for that name is the result-level measure. That
    precedence is why the unit cannot be a property of the measure REGISTRY — one
    descriptor, keyed by name, would have to answer for both levels at once.

    Two levels are walked and merged with **the outermost winning any name collision**:
    a result's own observations, then the leaves of the carriers it holds. The
    precedence matters for exactly one real case — a per-role ``cost_usd`` row
    decomposes the result-level ``cost_usd`` rather than independently observing it, so
    pooling both would count the same spend twice at two different granularities.

    Every scope in the registry's vocabulary that ends up with no measure is named in
    ``absent_scopes`` instead of being silently missing, and an observation the walk reached
    but could not summarise is named in ``unreported_observations`` — see that field for why
    silence there was the dangerous case.

    The third return value is the pooled observations each summary was computed from, beside their
    cases, for a lens that tests between levels over per-case values rather than reading summaries.
    """
    # Each observation is pooled beside its test case. `n` alone cannot distinguish 15 independent
    # observations from 5 cases repeated 3 times, and the two license very different intervals —
    # pooling k repeats as independent draws narrows every interval by roughly sqrt(k). So the
    # summary computes its spread over the cases and counts them, from the very observations it
    # pooled: a name's two levels never contribute cases to each other.
    outer: dict[str, _PooledMeasure] = {}
    inner: dict[str, _PooledMeasure] = {}
    unreported: set[str] = set()
    carriers_by_name: dict[str, set[str]] = {}
    inner_units: dict[str, str] = {}

    def record(
        level: dict[str, _PooledMeasure],
        name: str,
        value: float | str,
        descriptor: MetricDescriptor,
        *,
        report_gaps: bool,
        case_id: str,
        result: EvalResult,
        carrier: str | None = None,
    ) -> None:
        # Outside its population before anything else: a faulted result is not an observation of a
        # `scored` or `delivered` measure at all, nor a failure that took no turn one of a `delivered`
        # measure, so it can neither contribute a value nor be reported as one lost.
        if not in_population(summary_population(descriptor, undeclared), result):
            return
        if not _is_reportable(descriptor, profile.measures):
            # An undescribed NUMBER from a telemetry source is the loss worth reporting: a
            # measurement the code emits that the registry cannot explain. Three things stay
            # quiet, each for its own reason — an undescribed string (almost always an
            # identifier: a trace id, a role, a price source), a measure a deliberate filter
            # excluded (a directionless count, a judged family), and anything from the
            # result's lineage fields, where "undescribed" means "not a measure" rather than
            # "a measure nobody described".
            if report_gaps and descriptor.family is None and not isinstance(value, str):
                unreported.add(name)
            return
        if not _value_fits(descriptor, value):
            unreported.add(name)
            return
        if carrier is not None:
            carriers_by_name.setdefault(name, set()).add(carrier)
        _, values, cases = level.setdefault(name, (descriptor, [], []))
        values.append(value)
        cases.append(case_id)

    for result in results:
        case_id = result.test_case_id
        for name, value, descriptor in _lineage_leaves(result, profile=profile):
            record(outer, name, value, descriptor, report_gaps=False, case_id=case_id, result=result)
        for name, value, descriptor in _open_map_leaves(result, profile=profile):
            record(outer, name, value, descriptor, report_gaps=True, case_id=case_id, result=result)
        for name, value, report_gaps, carrier, observation_unit in chain(
            _carrier_leaves(result, profile=profile),
            _derived_leaves(result),
            _goal_check_leaves(result),
            _accuracy_leaves(result),
        ):
            record(
                inner,
                name,
                value,
                describe_measure(name, profile.measures),
                report_gaps=report_gaps,
                case_id=case_id,
                result=result,
                carrier=carrier,
            )
            inner_units[name] = observation_unit

    # Two DIFFERENT carriers using the same leaf name are two different quantities that
    # happen to share a word — an inner agent's elapsed_ms and some future subsystem's are
    # not one distribution, and averaging them would report a number describing neither,
    # under whichever descriptor's prose won the name. Refuse to pool and say so, rather
    # than producing a plausible number nobody can trace.
    ambiguous = {name for name, carriers in carriers_by_name.items() if len(carriers) > 1 and name not in outer}
    for name in ambiguous:
        inner.pop(name, None)
        unreported.add(name)

    # Outer wins outright — both the values and the metadata — so a colliding inner
    # name can never contribute half of a measure. A name that is a field of the result itself
    # is the result-level measure even where no result here observed one — an unpriced result's
    # ``cost_usd`` is None — so the per-role rows that decompose it never stand in for it: a
    # sum of the priced rows under ``cost_usd`` is exactly the partial figure unpriced spend
    # must never become.
    pooled = {**{name: entry for name, entry in inner.items() if name not in EvalResult.model_fields}, **outer}

    measures = [
        _measure_summary(*pooled[name], population=summary_population(pooled[name][0], undeclared))
        for name in sorted(pooled)
    ]
    confusion = next((measure for measure in measures if measure.name == CONFUSION_CELL_MEASURE), None)
    if confusion is not None:
        _, cells, cell_cases = pooled[CONFUSION_CELL_MEASURE]
        observations = [(str(cell), case) for cell, case in zip(cells, cell_cases)]
        measures = sorted(
            [*measures, *_classifier_label_summaries(confusion, observations)], key=lambda measure: measure.name
        )
    present = {measure.attribution_scope for measure in measures}
    collection = MeasureCollection(
        measures=measures,
        absent_scopes=[scope for scope in _ATTRIBUTION_SCOPES if scope not in present],
        unreported_observations=sorted(
            (unreported - set(pooled))
            | set(_withheld_derived(results, pooled))
            | set(_undeclarable_host_entries(results))
        ),
    )
    # Outer names describe the result itself by construction; an inner name keeps whatever
    # the walk saw carrying it, and loses to the outer level on a collision — the same
    # precedence the values follow, so a measure's unit always describes the values pooled
    # under it. Names dropped along the way (unreportable, ambiguous) are excluded, so the
    # map is exactly the collection's own vocabulary.
    units = {**inner_units, **dict.fromkeys(outer, _PER_RESULT)}
    return collection, {name: units[name] for name in pooled}, pooled


def _classifier_label_summaries(
    confusion: MeasureSummary, observations: Sequence[tuple[str, str]]
) -> list[MeasureSummary]:
    """Each label's precision, recall and F1, derived from a cell's confusion matrix.

    The matrix is the ``confusion_cell`` measure's observations — one ``expected → predicted`` pair
    each, beside its test case — so the per-label statistics are counted from what the walk already
    pooled, over the same population, never re-read from the results, and counted by
    :func:`~threetears.evals.analysis.confusion.label_statistics`, the one count the run summary reads
    too. Precision and recall are
    proportions, so each is a boolean-shaped summary: its rate, the count behind it, and its interval
    over the cases (:func:`~threetears.evals.analysis.stats.proportion_interval`). F1 is not a proportion of anything, so it is a numeric summary with a mean and no
    spread — it has none by construction, at any n. It is the harmonic mean of precision and recall, so a
    label missing either has no F1 either, rather than an F1 of 0.0 stated over no evidence; its ``n`` is
    the label's support across both — the observations predicted or expected as it
    (``predicted + expected - hits``), which is what its value is computed over.

    Args:
        confusion: The ``confusion_cell`` summary.
        observations: The ``(confusion_cell, test_case_id)`` observations it summarises.

    Returns:
        The derived summaries, named by :func:`~threetears.evals.kernel.metrics.classifier_label_measure`.
        A label never predicted has no precision; one never expected has no recall; either has no F1.
    """
    derived: list[MeasureSummary] = []
    for statistics in label_statistics(observations):
        rates: tuple[tuple[ClassifierStatistic, int, int, float | None, tuple[float, float] | None], ...] = (
            (
                "precision",
                statistics.predicted,
                statistics.predicted_cases,
                statistics.precision,
                statistics.precision_interval,
            ),
            ("recall", statistics.expected, statistics.expected_cases, statistics.recall, statistics.recall_interval),
        )
        for statistic, n, cases, rate, interval in rates:
            if rate is not None:
                derived.append(
                    MeasureSummary(
                        name=classifier_label_measure(statistic, statistics.label),
                        attribution_scope=confusion.attribution_scope,
                        higher_is_better=True,
                        population=confusion.population,
                        n=n,
                        n_independent=cases,
                        rate=rate,
                        n_true=statistics.correct,
                        ci_low=None if interval is None else interval[0],
                        ci_high=None if interval is None else interval[1],
                    )
                )
        if statistics.f1 is not None:
            derived.append(
                MeasureSummary(
                    name=classifier_label_measure("f1", statistics.label),
                    attribution_scope=confusion.attribution_scope,
                    higher_is_better=True,
                    population=confusion.population,
                    n=statistics.predicted + statistics.expected - statistics.correct,
                    mean=statistics.f1,
                )
            )
    return derived


def _measure_summary(
    descriptor: MetricDescriptor,
    values: list[float | str],
    cases: list[str],
    *,
    population: MeasurePopulation,
) -> MeasureSummary:
    """Summarise one measure's observations in the shape its data type takes.

    A spread is computed over the test cases, never over the observations as if each were its own
    draw: the SEM is :func:`~threetears.evals.analysis.stats.clustered_standard_error`, and the interval
    is read on ``n_independent - 1`` degrees of freedom. Where every case was observed once, both are
    the unclustered forms exactly.

    Args:
        descriptor: The measure's registry descriptor, carried onto the summary.
        values: Its observations, at least one.
        cases: Each observation's test case, aligned with ``values``.
        population: The population those observations were drawn from, stated on the summary.

    Returns:
        A categorical summary (counts), a boolean one (rate + interval), a text one (every
        observation listed, nothing aggregated) or a numeric one (distribution + SEM + interval).
    """
    shape: dict[str, Any]
    if descriptor.data_type == "text":
        shape = {"texts": [str(value) for value in values]}
    elif descriptor.data_type == "boolean":
        n_true = sum(1 for value in values if value is True)
        interval = proportion_interval([value is True for value in values], cases)
        shape = {
            "rate": n_true / len(values),
            "n_true": n_true,
            "ci_low": None if interval is None else interval[0],
            "ci_high": None if interval is None else interval[1],
            "case_means": small_sample_case_means([1.0 if value is True else 0.0 for value in values], cases),
        }
    elif descriptor.data_type == "categorical":
        counts: dict[str, int] = {}
        for value in values:
            counts[str(value)] = counts.get(str(value), 0) + 1
        shape = {"categories": counts}
    else:
        observed = [float(value) for value in values]
        numeric = sorted(observed)
        mean = sum(numeric) / len(numeric)
        sem = clustered_standard_error(observed, cases)
        interval = observed_mean_interval(
            observed, cases=cases, value_range=descriptor.value_range, floor=descriptor.interval_floor
        )
        shape = {
            "mean": mean,
            "p05": _percentile(numeric, 0.05),
            "p50": _percentile(numeric, 0.50),
            "p95": _percentile(numeric, 0.95),
            "max": numeric[-1],
            "sem": sem,
            "n_zero": sum(1 for value in numeric if value == 0.0),
            # An interval on the MEAN at `stats.INTERVAL_LEVEL`, carried structurally so the generator never
            # has to derive one. The viz contract requires a `ci` on the distribution
            # and null_result payloads while forbidding the generator from inventing a
            # statistic the bundle does not give it — with only `sem` here, both of
            # those payloads were unfillable, so the findings that most need them (a
            # null result draws the arms' intervals, and a distribution its spread)
            # silently came back with no visualization at all. Interval of the mean,
            # not of the observations: it
            # answers "where does this arm's average sit", which is the question a
            # null result asks. Over the cases, not the observations, so k repeats of a case are not
            # k draws. None below n=2, or over a single case, where `sem` itself is unestimable. Inside
            # the measure's declared scale, and a 0/1 measure's is its proportion's interval — one rule,
            # `stats.observed_mean_interval`, so `accuracy` and the `match` it is derived from agree.
            "ci_low": None if interval is None else interval[0],
            "ci_high": None if interval is None else interval[1],
            # Below the band floor a chart draws the cases rather than the interval, so it needs them.
            "case_means": small_sample_case_means(observed, cases),
        }
    return MeasureSummary(
        name=descriptor.name,
        attribution_scope=descriptor.attribution_scope,
        higher_is_better=descriptor.higher_is_better,
        population=population,
        n=len(values),
        n_independent=len(set(cases)),
        **shape,
    )


__all__ = [
    "goal_check_proofs_of",
]
