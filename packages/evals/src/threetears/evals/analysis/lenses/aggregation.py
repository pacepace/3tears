"""The aggregation vocabulary the read lenses share: weighting modes, observation-to-aggregate names, and the roll-up.

:func:`resolve_measure_name` maps an aggregate name back to the observation it rolls up, :func:`metric_help` glosses
each, and :func:`_aggregate` rolls a cell's observations into one number under a weighting mode. The pivot, the
frontier and the history all aggregate through here.
"""

from __future__ import annotations

from threetears.evals.kernel.host.profile import HostProfile
from threetears.evals.kernel.metrics import MetricDescriptor, describe_measure
from threetears.evals.analysis.reporting import (
    METRIC_COMPOSITE,
    METRIC_COST_USD,
    METRIC_GOAL_STATE,
    METRIC_OUTCOME,
    METRIC_SCORE,
    METRIC_TOTAL_MS,
    METRIC_TRANSCRIPT,
    SCOPED_METRICS,
)


# Weighting modes for rolling a cell's observations into one number.
#
# The default is equal-per-scenario: case counts in this corpus reflect budget
# history rather than importance, and k is unbalanced (k=1 vs k=3) across the
# same suite, so sample-weighting lets an artifact of how much we could afford
# to run drive the ranking. Sample-weighted stays available as an explicit,
# labeled toggle — it is the right answer when the question really is "what did
# the whole population do", and the cost of the default (a thin scenario gets an
# equal vote) is exactly what the per-cell `n` is displayed to expose.
#: Weighting mode: every scenario (case) gets an equal vote in a cell's number, however many observations it has.
WEIGHTING_EQUAL_PER_SCENARIO = "equal_per_scenario"
#: Weighting mode: every observation gets an equal vote, so a case with more repeats counts for more.
WEIGHTING_SAMPLE_WEIGHTED = "sample_weighted"
WEIGHTINGS = (WEIGHTING_EQUAL_PER_SCENARIO, WEIGHTING_SAMPLE_WEIGHTED)
#: The weighting a pivot uses when none is named: equal per scenario.
DEFAULT_WEIGHTING = WEIGHTING_EQUAL_PER_SCENARIO

# Observation-level metric name -> the registry name of the AGGREGATE that an
# aggregating surface (`pivot` cell, `history` series point) actually reports.
#
# It has a second job: inverted, it is the operator-facing ALIAS table
# (`_OBSERVATION_OF_AGGREGATE`), so every name `list_metrics` publishes for one of
# these quantities is a name the surfaces accept. Both jobs need the same pairs,
# which is why they are one table and not two that can drift.
#
# A row metric is never its own cell's descriptor: a row carries one observation,
# while `mean_composite` / `mean_cost_usd` / `mean_score` describe the mean OVER
# rows, which is what a cell holds. That is true whether or not the row metric is
# itself a registry key — `score` is one (the registry describes the raw 1-5 judge
# score) and `composite` / `cost_usd` are not, and all three map here regardless,
# because what a cell needs described is the aggregate.
#
# The mapping has to be explicit because `describe_measure` is total — an unmapped
# name resolves silently to whatever the registry happens to hold for it, which for
# an unseeded name is the unclassified arm (family=None, held at the strictest
# transferability class) and for `score` would be the per-observation 1-5 descriptor
# on a cell holding a mean. Adding a row metric means adding its aggregate here, or
# accepting the fallthrough knowingly.
_AGGREGATE_OF_OBSERVATION = {
    METRIC_COMPOSITE: "mean_composite",
    METRIC_COST_USD: "mean_cost_usd",
    # The registry declares `mean_score` `judge_mediated` on a (1.0, 5.0) range.
    # The cross-subject pooling refusal would fire without this entry — the unclassified arm holds
    # at `scenario_bound`, which is stricter still — but it would fire naming a
    # class nothing vetted, and every honest cell would report `family=None` with
    # no scale beside a number whose whole point is that it is 1-5 and not 0-1.
    METRIC_SCORE: "mean_score",
    # `history` only — `pivot` refuses `total_ms` (it is not projected). Without
    # this entry a latency series described itself with the `total_ms` descriptor,
    # which describes ONE result's wall-clock and its llm/tool/orchestration
    # partition — a partition a mean over cases does not have.
    METRIC_TOTAL_MS: "mean_total_ms",
    # The dual-score axes. Their per-observation descriptors are the registry entries
    # under these same names (1-5, family `dual_axis`); a cell holds the MEAN of those,
    # which is a different quantity and needs its own entry — the fallthrough would
    # otherwise describe a mean with the descriptor of one observation, the exact defect
    # the `score` entry above exists to avoid. Not mapped to `mean_score`, though the
    # arithmetic is identical: this table is inverted into the operator-facing alias map,
    # so three metrics sharing one aggregate name would make the inverse answer with
    # whichever came last.
    METRIC_TRANSCRIPT: "mean_transcript_score",
    METRIC_OUTCOME: "mean_outcome_score",
    # A cell of 1/0 check rows means a pass rate — of ONE check when the cell is keyed on goal_check.
    METRIC_GOAL_STATE: "goal_state_pass_rate",
}

# The inverse: registry aggregate name -> the observation-level name the surfaces
# key rows on. This is what closes the seam an operator hits when they read
# `list_metrics` and use what it says.
#
# `list_metrics` publishes the AGGREGATE names (`mean_composite`, `mean_cost_usd`,
# `mean_total_ms`, `mean_score`) because those describe what an aggregating surface
# returns. The `metric` parameter of `pivot` / `history` names the ROW being
# aggregated, so it took the short observation names — and `mean_composite`, the one
# name the catalog gives that quantity, was refused as unknown. Accepting both is the
# fix, in this direction and not the other:
#
#   - Renaming the row constants to the registry keys is forbidden at their definition
#     site (see the comment above `METRIC_COMPOSITE`) and would be wrong anyway: an
#     export row IS one observation, so `metric=composite` on a row is its honest name.
#   - Seeding a bare `composite` measure into the registry would publish a second
#     composite-quality entry describing a row nobody aggregates, and would leave
#     `mean_cost_usd` / `mean_total_ms` still refused.
#
# So the catalog name resolves to the row name at the surface boundary, and the row
# name keeps working. Derived rather than written out, so a new pair cannot be added
# to one direction only; `_metric_vocabulary` renders both names in every refusal, so
# an operator who guesses wrong is told the other one exists.
_OBSERVATION_OF_AGGREGATE = {aggregate: observation for observation, aggregate in _AGGREGATE_OF_OBSERVATION.items()}


def resolve_measure_name(metric: str) -> str:
    """Resolve a catalog (aggregate) measure name to the observation name rows carry.

    Args:
        metric: A measure name as an operator supplied it — either the
            observation-level row name (``composite``) or the registry name of its
            aggregate as ``list_metrics`` publishes it (``mean_composite``).

    Returns:
        The observation-level name, unchanged for anything this table does not
        cover — an unknown name must reach its surface's own refusal so the caller
        is told what IS accepted, rather than being rewritten into something else.
    """
    return _OBSERVATION_OF_AGGREGATE.get(metric, metric)


def _metric_vocabulary(accepted: frozenset[str]) -> str:
    """Render an accepted-measure set naming each row name beside its catalog name.

    Args:
        accepted: The observation-level names a surface accepts.

    Returns:
        A sorted, comma-joined list in which every measure that has a published
        aggregate name shows it too — a refusal that named only the row names is
        what sent an operator reading ``list_metrics`` in a circle.
    """
    return ", ".join(
        f"{name} (or {_AGGREGATE_OF_OBSERVATION[name]})" if name in _AGGREGATE_OF_OBSERVATION else name
        for name in sorted(accepted)
    )


# What each surface-accepted metric is, in a clause, for every help text that lists them. Keyed on
# exactly the metrics some surface accepts, and checked at import, so a metric added to one of the
# accepted sets without a gloss fails loudly instead of reaching a help text that never mentions it.
_METRIC_GLOSS: dict[str, str] = {
    METRIC_COMPOSITE: "0-1 quality",
    METRIC_COST_USD: "measuring spend in USD: the candidate's and the judge's and simulator's",
    METRIC_SCORE: "the raw judge score: 1-5, or 1/0 on a pass/fail dimension",
    METRIC_TOTAL_MS: "one result's wall-clock",
    METRIC_TRANSCRIPT: "dual-score axis, 1-5 and NOT a dimension",
    METRIC_OUTCOME: (
        "dual-score axis, 1-5 and NOT a dimension; its divergence from the transcript axis is what separates a "
        "worse agent from a changed world"
    ),
    METRIC_GOAL_STATE: "1 passed / 0 not, code-graded",
}


def metric_help(accepted: frozenset[str]) -> str:
    """Render the metrics a surface accepts, each glossed and beside its catalog name, for its help text.

    The one producer of that list for the pivot and history help on REST and MCP alike, so a help
    text cannot name a metric its surface refuses or leave out one it accepts.

    Args:
        accepted: The observation-level names the surface accepts.

    Returns:
        A semicolon-joined list, in name order.
    """
    return "; ".join(
        f"'{name}' ({_METRIC_GLOSS[name]}"
        + (f"; or its catalog name '{_AGGREGATE_OF_OBSERVATION[name]}'" if name in _AGGREGATE_OF_OBSERVATION else "")
        + ")"
        for name in sorted(accepted)
    )


def _describe_aggregate(metric: str, *, profile: HostProfile) -> MetricDescriptor:
    """Describe what a pivot CELL holds, which is an aggregate over rows.

    Args:
        metric: The observation-level metric name carried on the rows.
        profile: The host whose vocabulary this reads.

    Returns:
        The aggregate's descriptor when one is registered, else the honest
        unclassified answer for the observation name itself.
    """
    return describe_measure(_AGGREGATE_OF_OBSERVATION.get(metric, metric), profile.measures)


def _effective_formula(metric: str, weighting: str, *, scoped: bool = True) -> str:
    """State exactly what the reported number is, under this weighting and scope.

    Two things can make the registry's static formula describe a different number
    than the cell shows, and both are stated here rather than left to the reader.
    The weighting is the first. The second is *scope*: the registry describes
    ``mean_score`` as the mean "for one rubric dimension", which is what a cell
    holds only while ``rubric_dim`` is pinned. With it pinned by neither axis nor
    filter, a cell averages every dimension the judge scored — a real question,
    and roughly the composite on the 1-5 scale, but not the one the descriptor
    names. Saying so is the same obligation the weighting clause discharges: an
    unqualified formula beside a qualified number is how a reader checks the
    wrong arithmetic and concludes the table is right.

    Args:
        metric: The observation-level metric being aggregated.
        weighting: The weighting mode the value was computed under.
        scoped: Whether the metric's scope coordinate (:data:`SCOPED_METRICS`) is pinned by an axis or
            filter. Only meaningful for a scoped metric; every other measure is whole-result and has
            nothing to pool. Both pinnings count here because both scope the cell, but only the axis is
            reachable from the REST and MCP surfaces — see the emitted caveat below.

    Returns:
        A one-line formula an operator can read at point of use.
    """
    if weighting == WEIGHTING_EQUAL_PER_SCENARIO:
        formula = f"mean over test cases of the per-case mean of {metric} (each case weighted equally)"
    else:
        formula = f"mean of {metric} over every contributing observation (cases weighted by their observation count)"
    if metric in SCOPED_METRICS and not scoped:
        scope_field, scope_noun, row_noun = SCOPED_METRICS[metric]
        # Names the AXIS only, though `scoped` also honours a filter.
        # This string ships to operators, and no operator can reach the filter:
        # `reads.pivot` composes `filters` itself from `subject_id`, and
        # neither surface accepts a `rubric_dim` filter — MCP would swallow it in
        # its kwarg union and REST declares no such query parameter, so the
        # advice would be followed, silently ignored, and answered with the same
        # pooled table still carrying this caveat. A remedy a reader cannot
        # perform is the false-capability claim this measure was added to end.
        formula += f", POOLED ACROSS EVERY {scope_noun.upper()} (put '{scope_field}' on an axis for one)"
        # The pooling also changes what the cell's EVIDENCE fields count, and
        # qualifying the value while leaving `n` unqualified is the same defect
        # one field over. Pooled, a row is a (result, dimension) pair: a judged
        # result contributes one row per dimension and an unjudged result exactly
        # one unmeasured row. So four judged 3-dim results beside four unjudged
        # ones report n=12 with 4 unmeasured — which reads as 75% measured when
        # half the results were never scored. With `rubric_dim` on an axis each
        # cell holds one dimension and `n` is a result count again.
        #
        # The DISPERSION is named per weighting rather than lumped in, because
        # `_aggregate` branches: under `equal_per_scenario` (the default) the SEM
        # is over test-case means — pooling changes those means, and how many
        # observations each averages, but not how MANY MEANS the SEM is taken
        # over — so claiming the dispersion
        # rides on rows would send an operator reconstructing an interval to `n`
        # instead of the case count and hand them one too narrow by ~sqrt(n/n_cases).
        # `sample_weighted` averages the flat rows, but its SEM is clustered by test case
        # (`_aggregate`), so its dispersion does not ride on rows either.
        formula += f"; n and the outcome counts are over (result x {row_noun}) rows here, not results"
        if weighting == WEIGHTING_SAMPLE_WEIGHTED:
            formula += "; the dispersion is clustered by test case, so the rows of one case are not independent draws"
        else:
            formula += "; the dispersion is over test-case means — a BASIS pooling does not change, though the means themselves do"
    return formula


def _aggregate(values_by_case: dict[str, list[float]], weighting: str) -> tuple[float, float | None]:
    """Roll one cell's observations into a value plus its dispersion.

    Args:
        values_by_case: Observed values grouped by test case.
        weighting: Which of :data:`WEIGHTINGS` to apply.

    Returns:
        ``(value, sem)``. The SEM is over whatever the value averages, so the
        two always describe the same estimate, and it counts cases, not observations:
        under ``sample_weighted`` the flat mean takes the cluster-robust SEM over the
        cases (:func:`~threetears.evals.analysis.stats.clustered_standard_error`), since
        a case's repeats are not independent draws.
    """
    from threetears.evals.analysis.stats import clustered_standard_error, standard_error_of_mean

    if weighting == WEIGHTING_EQUAL_PER_SCENARIO:
        case_means = [sum(vals) / len(vals) for vals in values_by_case.values()]
        return sum(case_means) / len(case_means), standard_error_of_mean(case_means)
    flat = [v for vals in values_by_case.values() for v in vals]
    cases = [case for case, vals in values_by_case.items() for _ in vals]
    return sum(flat) / len(flat), clustered_standard_error(flat, cases)


__all__ = [
    "DEFAULT_WEIGHTING",
    "metric_help",
    "resolve_measure_name",
    "WEIGHTING_EQUAL_PER_SCENARIO",
    "WEIGHTING_SAMPLE_WEIGHTED",
    "WEIGHTINGS",
]
