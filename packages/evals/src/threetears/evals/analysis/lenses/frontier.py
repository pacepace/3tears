"""Frontier: the verdict surface — the cheapest variant clearing the bar, per subject.

:func:`compute_frontier` ranks each subject's contestants on pass^k against production-replicating cost and
latency, decides dominance between them by test rather than by point estimate (:func:`_decide_dominance`),
names the cost ties it cannot separate, and checks the boundary pillar.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Iterable, Mapping, Sequence
from fractions import Fraction
from typing import TYPE_CHECKING, ClassVar, Literal, NamedTuple

from threetears.evals.analysis.contention import (
    contended_latency_sentence,
    withheld_latency,
    withhold_contended_latency,
)
from threetears.evals.analysis.numbers import format_number
from threetears.evals.analysis.stats import (
    SIGNIFICANCE_ALPHA,
    case_rate_interval,
    contrast_samples,
    exact_decimal,
    guardrail_decision,
    holm_adjust,
    interval_clears,
    separation_test,
)
from threetears.evals.kernel.analysis_measures import BarDecision
from threetears.evals.schema.base import EvalBaseModel, EvalDocumentModel
from threetears.evals.kernel.host.profile import HostProfile
from threetears.evals.schema.values import PooledProductionFooting, ProductionFooting
from threetears.evals.schema.models import (
    SCALES,
    RubricScale,
)
from threetears.evals.kernel.surface import FrontierDominance, GuardrailDecision
from threetears.evals.kernel.result_condition import (
    ResultOutcome,
    classify_result,
    delivered_a_turn,
    harness_faulted,
)
from threetears.evals.kernel.scoring import (
    CompositeBasis,
    PassHatPoint,
    case_pass_hat_k,
    pass_hat_k_at,
    pass_hat_k_cell,
    pool_pass_hat_k,
    pool_pass_hat_k_attempts,
    result_composite,
)
from threetears.evals.analysis.completeness import completeness_disclosure
from threetears.evals.analysis.reporting import (
    cassette_mode_disclosure,
    place_results,
    pooled_composite_basis,
    pooled_cost_compositions,
    pooled_served_models,
    ProjectionExclusions,
    ServedModelReading,
)
from threetears.evals.analysis.lenses.aggregation import _aggregate, WEIGHTING_EQUAL_PER_SCENARIO
from threetears.evals.analysis.lenses.contestants import (
    _contestant_key,
    _identity_span_disclosure,
    _identity_version_disclosure,
    _identity_version_span,
    ContestantKey,
)

if TYPE_CHECKING:
    from threetears.evals.schema.models import (
        EvalResult,
        EvalRun,
    )


# A frontier point is one (subject, variant, identity version): the resolved contestant stack that
# would ship if it won, keyed on `variant_key`. Grouping by model alone would
# pool two tool-config or prompt-override contestants of the same model into one
# point — the same silent-pooling error refused across subjects — so the
# variant is the unit. `model` rides along as the label.
#
# The corollary, which is not optional: because the key is the shipping stack, the
# CONDITIONS a variant was measured under do not partition its points. Every scenario
# template and every cassette mode it ran under pools into the one point. Both are
# disclosed on the point and carried onto the verdict rather than split out — keying a
# condition would restamp and re-cohort every stored result to close a rendering
# gap. `comparison_sets` is where a reader goes to group by template.
#
# Domination is over three axes: pass^k (higher better), production-replicating
# cost (lower better), and mean total latency (lower better). mean-composite is shown
# beside pass^k as the secondary quality read but is NOT a domination
# axis — ranking on two correlated quality measures would double-weight quality.
#
# Domination is a claim that one contestant is worse, so it is DECIDED, by test, never read off
# point estimates: two contestants drawn from one distribution differ on every continuous axis,
# and comparing the numbers flagged one of them dominated about a third of the time. See
# `_dominance_p` for the rule. Latency is ranked on the MEAN, not a tail, for the same reason:
# a ranking axis must carry a test, and at the sizes a frontier sees a 95th percentile has
# none — a 95% interval on a p95 has no upper end below 72 observations — while a mean has a paired test over
# cases. The tail is read beside it (the run summary's `p95_total_ms` / `max_total_ms`).
#
# Cost is restricted to the production-replicating roles (candidate + inner_agent
# + external). A frontier computed on judge/simulator-inclusive cost ranks a
# variant by money the operator would never spend in production, which is a defect.


class FrontierError(ValueError):
    """A frontier was requested that cannot be answered honestly.

    Raised rather than returned, the way :class:`PivotError` is, so a bad request
    refuses identically on both surfaces. A bar outside the quality range, or one
    that is not a number, is the case that reaches it: silently clamping or
    dropping it would produce a verdict against a threshold the caller never chose.
    """


def normalize_bar(value: float | str | None) -> float | None:
    """Parse a caller's quality bar to a float, identically on every surface.

    Lives here, in one place as
    :func:`~threetears.evals.kernel.status_filter.normalize_status_filter` does,
    so REST (which hands a validated float) and MCP (which hands the raw string a
    tool argument arrives as) reach :func:`compute_frontier` through one coercion, rather
    than each parsing it and disagreeing about what a blank or a non-number means
    — the divergence class the parity gate exists to catch.

    Args:
        value: Raw bar from the caller. ``None`` or a blank string means the bar
            was not supplied.

    Returns:
        The bar as a float, or ``None`` when it was not supplied.

    Raises:
        FrontierError: The value is present but not a number.
    """
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError as e:
            raise FrontierError(f"bar {value!r} is not a number in [0, 1]") from e
    return float(value)


class FrontierBoundaryCheck(EvalBaseModel):
    """One boundary (guardrail) dimension of one contestant, held against the control the frontier was given.

    Decided by the one guardrail rule (:func:`~threetears.evals.analysis.stats.guardrail_decision`) the bundle's
    guardrails are decided by, over the samples every contrast reads
    (:func:`~threetears.evals.analysis.stats.contrast_samples`): ``held`` when the contestant is shown no worse than
    the control by more than the dimension's declared margin, ``breached`` when it is shown worse, ``undecided``
    otherwise.
    """

    dimension: str
    decision: GuardrailDecision
    #: The interval on ``mean(contestant) − mean(control)`` the decision read; None when none exists.
    interval: tuple[float, float] | None = None
    #: The margin it was held to — the campaign's declared one for the dimension — or None (held at zero change).
    margin: float | None = None
    paired: bool = False
    n_cases_control: int = 0
    n_cases: int = 0
    #: Why it is undecided, when it is; None otherwise.
    undecided_reason: str | None = None


def _template_span_entries(templates_by_run: Mapping[str, str | None]) -> list[str]:
    """The suites behind one frontier point, one ``"<template> (n runs)"`` entry per suite.

    The half of the span that VARIES between two contestants: which suites, and how many
    runs each contributed. Split out from :func:`_template_span_disclosure`, whose other
    ninety words are the same on every marked point in a scope — a surface rendering
    several marked contestants owes the RULE once and the sets once each, the collapse
    a host's model-disparity lines make for a uniform judge set and for its reason: a
    caveat repeated per row trains the reader to skip the line, precisely where a verdict
    is being named.

    The disclosure is composed from this list rather than re-deriving it, so a surface
    printing the sentence and a surface printing the sets cannot disagree about whether a
    point spans, or about which suites it spanned when it does.

    Args:
        templates_by_run: One ``template_id`` per contributing run, keyed by run id. A
            ``None`` is a real observed state — an ad-hoc run assembled from explicit
            test-case ids, per :attr:`~threetears.evals.schema.models.EvalRun.template_id` — so it
            counts as its own suite rather than being dropped or merged into a neighbour.

    Returns:
        One entry per distinct template, ad-hoc last, or an empty list when fewer than two
        runs contributed or every one of them ran the same template — "this point does not
        span", the same predicate the disclosure reports as ``None``.
    """
    if len(templates_by_run) < 2:
        return []
    by_template: dict[str | None, list[str]] = {}
    for run_id, template_id in templates_by_run.items():
        by_template.setdefault(template_id, []).append(run_id)
    if len(by_template) < 2:
        return []
    return [
        f"{template_id or 'ad-hoc (no template)'} ({len(run_ids)} run{'' if len(run_ids) == 1 else 's'})"
        for template_id, run_ids in sorted(by_template.items(), key=lambda kv: (kv[0] is None, kv[0] or ""))
    ]


def _template_span_disclosure(span_entries: Sequence[str]) -> str | None:
    """The sentence a frontier point carries when its observations span scenario suites.

    A point is one ``(subject, variant_key, identity version)``, and ``variant_key`` digests the resolved
    contestant stack — everything that would ship, and nothing about the conditions it was
    measured under. ``template_id`` is one of those conditions: it is hashed into the
    CONTEXT key, and :func:`compute_comparison_sets` groups on it. So every scenario suite a
    variant was ever entered into pools into this one point, and ``pass_hat_k``,
    ``mean_composite`` and ``mean_total_ms`` are each one number over cases drawn from all
    of them.

    Pool-and-disclose, not key-it — the position the cassette span already records for its
    axis (:func:`cassette_mode_disclosure`), taken here for the same two reasons. The
    variant key is what would SHIP and a scenario suite ships nothing; and re-keying it
    would restamp and re-cohort every stored result to close a rendering gap.

    It matters more on this surface than on any other, because this one names a VERDICT.
    Two suites need not be of equal difficulty, so adding an easy suite to a scope
    raises the pooled pass^k of every variant entered into it, without a single contestant
    changing — and two
    contestants measured over different mixes are not being ranked on the same cases at
    all.

    What it does NOT claim: that one template id is one suite. A template is mutable and
    versioned, so two runs sharing an id can still have executed different cases;
    :func:`compute_comparison_sets`' ``case_set_differs`` badge owns that axis, and silence here is
    not evidence against it.

    Descriptive, never a verdict, on the rule :func:`cassette_mode_disclosure` follows: no
    comparison is refused and no number is adjusted. Measuring one variant across several
    suites is a legitimate thing to do.

    Args:
        span_entries: This point's suites as :func:`_template_span_entries` renders them.
            Taken rather than re-derived, so the sentence and the per-point list a surface
            may print instead are one predicate over one grouping.

    Returns:
        The disclosure, rendered verbatim by a surface showing this point ALONE, or
        ``None`` when the point does not span. A surface rendering several marked points
        states this rule once and prints ``span_entries`` per point: the only thing that
        differs between two of these sentences is the parenthetical.
    """
    if not span_entries:
        return None
    detail = ", ".join(span_entries)
    return (
        f"This contestant's observations span {len(span_entries)} scenario templates ({detail}). pass^k, "
        "composite and latency here are each ONE number over cases pooled from all of them, and two suites "
        "need not be of equal difficulty — so this is a weighted average over whichever mix happened to run, "
        "not a score on any one suite, and a contestant measured over a different mix is not being ranked on "
        "the same cases. Group by template (`comparison_sets` does) before reading a gap between rows as a "
        "difference between the contestants."
    )


class FrontierDominator(EvalBaseModel):
    """One contestant shown to beat another point on every axis it measured, named as a ROW is named.

    The identity triple of :class:`FrontierPoint`, carried rather than projected down to
    ``model``. A point is one ``(subject, variant, identity version)`` and every surface renders it as
    ``<model> · <variant_key>``, so a dominator list of bare model names is strictly less
    identifying than the thing it points at: two variants of one model collapse into one
    entry, and the surviving string is the dominated row's OWN model, which reads as a row
    dominating itself: a point ``model-a · key1`` rendered as "dominated by model-a" while
    two distinct variants of ``model-a``, ``key2`` and ``key3``, beat it.

    **Identity, not a rendered label.** How much of the key to show is a surface's
    decision — MCP truncates it to match its own variant column — and a pre-formatted
    string here would either mismatch that column or deny a web surface the full key.
    ``frontier``'s whole job is *which variant should I move to*, and that answer is the
    key.
    """

    variant_key: str
    model: str
    #: The identity version stamped beside ``variant_key``, carried for the same reason the
    #: key is: since the lenses partition on it, two points can now share a model AND a key
    #: and differ only here — and under a context-side bump they DO, since the shared counter
    #: moves the variant stamp without moving the digest. Without this the self-domination defect above returns in a new form —
    #: `<model> · <key>` identifies BOTH rows, so the dominator label again names something
    #: indistinguishable from the row it dominates. Cross-version domination is correct and
    #: is not blocked: the gate says these are two contestants, and two contestants beating
    #: each other on measured axes is what a frontier is for. Only the LABEL needed fixing.
    variant_identity_version: int
    #: The Holm-adjusted p the domination was decided on, below :data:`~threetears.evals.analysis.stats.SIGNIFICANCE_ALPHA`
    #: (see :func:`_dominance_p`). Carried because a verdict a reader cannot check against its statistic is
    #: an assertion. ``None`` on a dominator stored before domination was tested: that one was read off
    #: point estimates.
    p_value: float | None = None


#: How a frontier verdict's pick stands on cost against the other contestants that cleared the bar with a
#: cost — see :attr:`FrontierVerdict.cost_decision`.
FrontierCostDecision = Literal["shown_cheapest", "not_separated", "untested", "only_cleared"]


class FrontierCostTie(EvalBaseModel):
    """A contestant that cleared the bar with a cost, which the verdict's pick was NOT shown cheaper than.

    Named as a row is named (the identity triple :class:`FrontierDominator` carries, for its reasons), with
    the cost it was read at and the p the comparison reached, so a reader can check why it stays in the set.
    """

    variant_key: str
    model: str
    variant_identity_version: int
    production_replicating_cost: float
    #: The Holm-adjusted p of the test that the pick costs less than this contestant, at or above
    #: :data:`~threetears.evals.analysis.stats.SIGNIFICANCE_ALPHA`. ``None`` when no test could decide — fewer
    #: than two cases carried a cost on a side, or every case differed by one amount over too few cases for the
    #: exact test to reach α — which is untested, not a tie the data showed.
    p_value: float | None = None


class FrontierPoint(EvalBaseModel):
    """One contestant's position on quality x cost x latency, within a subject.

    A point is one ``(subject, variant, predicate version)``. ``model`` is the human
    label; ``variant_key`` is the grouping identity, and it is qualified by the
    identity version stamped beside it — two keys carrying different stamps are ranked as
    separate contestants, since the stamp cannot say which predicate moved and merging on
    an unverifiable digest is the direction nothing downstream undoes.

    Every axis carries its own denominator (``n_pass_cases``, ``n_composite_cases``,
    ``n_cost``, ``n_latency``) because the axes are independently sourced: a
    point's cost mean can rest on fewer results than its quality when some observed no
    production-role cost, and a mean over 2 of 30 that reads like a mean over 30
    is exactly the misreading these counts exist to prevent.
    """

    variant_key: str
    model: str
    #: The predicate version that minted ``variant_key``. Carried onto the row because
    #: this surface GATES on it — two keys carrying different stamps are ranked separately,
    #: because the stamp cannot say which predicate moved and a wrong merge is the direction
    #: nothing downstream undoes — and a reader who meets that split needs the number that
    #: caused it.
    variant_identity_version: int
    #: Set when that predicate is not this build's.
    identity_version_disclosure: str | None = None

    # Quality — pass^k is the headline (and the domination axis), mean-composite
    # the secondary read shown alongside it. Both always travel together.
    # pass^k is the chance that `k` attempts at a case ALL pass (τ-bench; never pass@k, the
    # chance that at least one does), estimated without bias per case and averaged over the
    # cases measured at least `k` times — a case's attempts pooled across every run of one
    # cell (`scoring.pool_pass_hat_k`), so a repeat run adds depth rather than a second copy.
    # `k` is the subject's: every point is ranked at the same depth (see SubjectFrontier.k).
    # None when no case was scored that deep: nothing was measured,
    # which is not a pass^k of zero, and an unmeasured axis never wins a domination.
    pass_hat_k: float | None
    #: The depth ``pass_hat_k`` is read at — the subject's, so every point is ranked at one.
    k: int = 1
    #: The cases ``pass_hat_k`` averages over: case-within-cell units scored at least ``k`` times.
    n_pass_cases: int = 0
    #: pass^1..pass^K for this point, each with its own case count. pass^1 is the per-case pass
    #: rate; the deeper points show how fast reliability decays with repetition.
    pass_hat_k_curve: list[PassHatPoint] = []
    #: The interval on ``pass_hat_k`` at :data:`~threetears.evals.analysis.stats.INTERVAL_LEVEL`, over its
    #: cases (:func:`~threetears.evals.analysis.stats.case_rate_interval`): the cases are the draws, and each
    #: case's own estimate is noisy too. ``None`` below two cases, where none is estimable, and on a point
    #: stored before pass^k carried one.
    pass_hat_k_ci_low: float | None = None
    #: The high end of that interval; ``None`` exactly when ``pass_hat_k_ci_low`` is.
    pass_hat_k_ci_high: float | None = None
    #: Attempts behind this point with nothing for pass^k to conjoin — no goal-state check and no judge — left
    #: out of pass^k and its curve rather than read as failures (#688).
    n_pass_no_criterion: int = 0
    #: Why ``pass_hat_k`` is None when the reason is that no attempt carried a pass criterion: the point has
    #: no pass^k, not one of 0, and so no cost per acceptable outcome either. None otherwise.
    pass_hat_k_unmeasured_reason: str | None = None
    #: How this point's pass^k reads against the bar, decided by its interval the way every campaign bar is
    #: (:func:`~threetears.evals.analysis.stats.interval_clears`): ``cleared``, ``missed``, ``undecided`` (the
    #: interval straddles the bar — neither a pass nor a failure), ``no_interval`` (fewer than two cases, not
    #: read) or ``no_data``. ``None`` when no bar was supplied.
    bar_decision: BarDecision | None = None
    #: Each boundary dimension of the subject, held against the control (:class:`FrontierBoundaryCheck`). Empty for
    #: the control itself, when the subject scores no boundary dimension, and when no control was given — then
    #: :attr:`SubjectFrontier.boundary_pillar` says the pillar was not checked.
    boundary_checks: list[FrontierBoundaryCheck] = []
    #: The boundary dimensions this contestant breached: it is disqualified, named by them, and never the pick.
    disqualified_by: list[str] = []
    mean_composite: float | None = None
    composite_sem: float | None = None
    n_composite_cases: int = 0
    #: What ``mean_composite`` was meaned over (:func:`pooled_composite_basis`), ``ragged`` when this point's
    #: results were scored on different dimension sets (#638). ``None`` when no composite was pooled, and on a
    #: point stored before the basis was carried — which says nothing about whether that pool was ragged.
    composite_basis: CompositeBasis | None = None

    # Cost — production-replicating only. ``None`` (never 0) when no
    # result at this point reported one; ``cost_is_partial`` when some did and some did
    # not — a result contributes nothing here when its production roles observed no cost,
    # when it carries no usage rows at all, or when a substituted delivery withheld the
    # figure (its background dollars were never spent, so the observed sum would understate
    # production), or when it is no turn the candidate took: a call its model refused, or a cell
    # the harness faulted, whose shortened spend would let the rig make the point look cheaper.
    production_replicating_cost: float | None = None
    n_cost: int = 0
    cost_is_partial: bool = False
    #: The role sets this point's cost mean was drawn over. More than one entry means the
    #: contestant's own results priced different things — and where two POINTS disagree,
    #: their costs are not comparable, so ranking one cheaper than the other is ranking a
    #: convention rather than a config. Empty when no result recorded a composition.
    cost_compositions: list[list[str]] = []
    # cost / pass-probability — the alternate denominator. The probability is pass^1, the
    # chance ONE attempt passes, because the cost is the mean of one attempt: dividing it by
    # pass^k would price a k-attempt streak at one attempt's cost. ``None`` when
    # cost is unknown or pass^1 is zero (dividing by a zero pass rate is undefined,
    # not "infinitely expensive").
    cost_per_acceptable_outcome: float | None = None
    #: What the runs behind ``production_replicating_cost`` set away from the subject's production configuration
    #: (#571), each run's own footing read off the host's sweepable declarations: the cost is what production would
    #: spend only where no run moved anything (``moved_nothing``). ``None`` when the frontier was computed without
    #: the host's declarations, and on a point stored before it was read — nobody checked, never "nothing moved".
    production_footing: PooledProductionFooting | None = None

    # Latency — total wall-clock ms, the performance axis. Mean over the turns the candidate took
    # (`delivered_a_turn`) that harvested a total; ``None`` when none did. The mean, not a tail: it is
    # the latency statistic a domination can be TESTED on at these sizes (see the section comment).
    mean_total_ms: float | None = None
    n_latency: int = 0
    #: Results whose candidate's model refused or errored with no turn taken — failures pass^k counts,
    #: left out of the cost and latency above. Equal to ``n_results`` (less any faulted) when every call
    #: failed that way: then the absent cost and latency mean every result failed, not that nothing was
    #: measured, and the point can dominate nothing on either axis.
    n_no_turn: int = 0

    # What this point rests on across all axes.
    n_results: int = 0
    n_cases: int = 0

    #: Set when this point's observations did NOT all record one cassette mode. The
    #: variant key is the stack that would SHIP, and cassette mode is apparatus rather
    #: than product, so it is deliberately absent from that key (it lives in the context
    #: key) — which means a capture arm and a replay arm of one contestant pool into this
    #: single point, correctly by that design and silently by this surface's rendering.
    #: The pooling is disclosed rather than split, because splitting would re-answer the
    #: variant/context identity split as a side effect of a rendering fix and would move
    #: every stored variant's identity with it. It qualifies EVERY axis here, not only
    #: cost: ``cost_is_partial`` already says a replayed arm withheld its cost, and
    #: ``pass_hat_k`` / ``mean_composite`` / ``cost_per_acceptable_outcome`` carry no such
    #: mark while resting on the same mixed pool — the ratio worst of all, since it
    #: divides a partial cost by a confounded pass rate and the two errors do not cancel.
    cassette_mode_disclosure: str | None = None

    #: Set when this point's observations did NOT all come from one scenario template. The
    #: same shape of pooling as ``cassette_mode_disclosure``, one axis over, and for the same
    #: reason: ``template_id`` is a measurement CONDITION (it is in the context key, and
    #: ``comparison_sets`` groups on it) while the variant key is the stack that would ship,
    #: so every suite a variant entered pools into this one point — correctly by that design
    #: and silently by this surface's rendering. It qualifies every axis here, not one:
    #: ``pass_hat_k``, ``mean_composite`` and ``mean_total_ms`` are each a weighted average
    #: over a mix of suites of unequal difficulty, and ``cost_per_acceptable_outcome``
    #: divides by that same pooled pass rate. Without it, adding an easy suite to a scope
    #: lifts the headline of every variant entered into it, with nothing on the row to show it.
    template_span_disclosure: str | None = None

    #: The suites behind that disclosure, one ``"<template> (n runs)"`` entry each — the only
    #: part of it that differs between two marked contestants. Carried beside the sentence
    #: rather than only inside it because a surface showing N marked points would otherwise
    #: print the same ninety-word rule N times, and a caveat printed N times is one nobody
    #: reads. Set exactly when ``template_span_disclosure`` is (both come from
    #: :func:`_template_span_entries`), so a surface can key on either without the two
    #: disagreeing about whether this point spans.
    template_span: list[str] = []

    #: ``run_id -> DEGRADED sentence`` for the runs behind this point that measured
    #: less than the matrix they promised. The third pooling disclosure on this model,
    #: and the one whose axis is not a measurement CONDITION but the measurement's own
    #: extent: a short run's cells are fewer, so every denominator on this row —
    #: ``n_pass_cases``, ``n_composite_cases``, ``n_cost``, ``n_latency`` — is partly
    #: made of a matrix that never finished. It moves the verdict rather than only
    #: qualifying it: admitting two truncated runs can take one contestant's
    #: pass^k from 1.00 over 3 cases to 1.00 over 4, lower its mean cost by a third,
    #: and flip its rival from *on frontier* to *dominated*.
    completeness_disclosures: dict[str, str] = {}

    #: Which models the provider's responses named as having answered this contestant's candidate calls (#684):
    #: ``one``, ``pooled`` — the contestant asked for one model id (a floating alias, say) and several models
    #: answered, so every axis here is a mixture of them — or ``unrecorded``, where some response named none and
    #: whether one model answered cannot be established. A contestant is keyed at launch, before any response
    #: names a model, so the mixture is disclosed rather than split. ``None`` when no result's candidate made a
    #: call, and on a point stored before it was read.
    served_models: ServedModelReading | None = None

    # Domination — a dominated point is grayed, never dropped: silently removing a
    # cheap-but-brittle variant looks identical to it never having run.
    #: True only when another point is SHOWN to beat this one (:attr:`dominance` ``dominated``).
    #: False is not a claim that nothing beats it. On a point stored before domination was tested
    #: (``dominance`` is ``None``), this was read off point estimates.
    dominated: bool = False
    #: ``dominated`` — some other point is shown better on every axis this one measured, by the test
    #: :func:`_dominance_p` states; ``not_separated`` — this point was tested against at least one other
    #: and no domination was shown, which says nothing about whether one exists; ``untested`` — no
    #: test against any other point could decide (fewer than two cases on an axis, no shared axis, or an axis
    #: on which every case differed by one amount over too few cases for the exact test to reach α — the
    #: reading :func:`~threetears.evals.analysis.stats.level_difference` also calls untested).
    #: ``None`` on a point stored before domination was tested.
    dominance: FrontierDominance | None = None
    #: Every point shown to beat this one on every axis, each carrying the identity a row
    #: carries and the p it was shown at. In the order the points are sorted, which is the order
    #: the table prints them, so a reader scanning for a named dominator meets them in that order.
    dominated_by: list[FrontierDominator] = []


class FrontierVerdict(EvalBaseModel):
    """The cheapest variant clearing the operator's bar for one subject — or, where the data cannot pick
    one, the set it is among.

    Present only when a bar was supplied AND at least one point both cleared it — its pass^k
    interval wholly at or above the bar, the rule every campaign bar is read by — and reported a
    production-replicating cost — a pick cannot be named cheapest
    on a cost nobody observed. "Cheapest" is decided by test, as domination is: the lowest point cost
    is named the cheapest only when it is shown cheaper than each rival (:attr:`cost_decision`), and
    otherwise the verdict names it beside every rival it was not shown cheaper than (:attr:`tied_with`).
    Two contestants drawn from one distribution always differ in point cost, so the lowest one alone
    would crown one of them every time. ``cost_is_partial`` rides along so a pick made on a
    partially-observed cost basis says so, and ``cassette_mode_disclosure`` for the
    stronger version of the same duty: a verdict is a RECOMMENDATION, so one drawn from
    a pool in which some observations replayed their third-party half has to carry that
    on the verdict itself, not only on a table row the reader may never scroll back to.
    ``template_span_disclosure`` rides along on exactly that rule — the pass^k this pick
    was named cheapest-above-the-bar on may be an average across scenario suites of
    different difficulty, which is a fact about the recommendation and not about a row.
    ``completeness_disclosures`` rides along for the sharpest version of it: the bar was
    applied to a pass^k, and the cheapest-by comparison to a cost, that partly rest on
    runs which never finished their matrix.
    """

    #: The contestant with the lowest point cost among those that cleared the bar with a cost. It is
    #: named THE cheapest only when :attr:`cost_decision` is ``shown_cheapest`` (or ``only_cleared``);
    #: otherwise it is one member, listed first, of the set :attr:`tied_with` completes.
    variant_key: str
    model: str
    pass_hat_k: float
    #: The depth ``pass_hat_k`` was read at — the subject's, the same as every point's.
    k: int = 1
    #: The interval the bar was decided on (:attr:`FrontierPoint.pass_hat_k_ci_low`). ``None`` only on a
    #: verdict stored before the bar read intervals: that one compared the point pass^k with the bar.
    pass_hat_k_ci_low: float | None = None
    pass_hat_k_ci_high: float | None = None
    production_replicating_cost: float | None = None
    cost_is_partial: bool = False
    #: The picked point's :attr:`FrontierPoint.production_footing`: the pick is made on a cost that is
    #: production's only where its runs moved nothing, so what they moved rides on the recommendation (#571).
    production_footing: PooledProductionFooting | None = None
    cassette_mode_disclosure: str | None = None
    template_span_disclosure: str | None = None
    completeness_disclosures: dict[str, str] = {}
    #: Rides along on the rule the four above ride on: a verdict is a RECOMMENDATION, so
    #: a pick whose key was minted by a superseded predicate says so where the
    #: recommendation is, not only on a table row the reader may never scroll back to.
    identity_version_disclosure: str | None = None
    #: The picked point's :attr:`FrontierPoint.served_models`, carried for the reason the disclosures above are:
    #: a recommendation of a contestant several models answered is a recommendation of their mixture.
    served_models: ServedModelReading | None = None
    #: The version stamped beside ``variant_key``, carried for the reason
    #: :attr:`FrontierDominator.variant_identity_version` is: since the lenses partition on
    #: it, ``(model, variant_key)`` no longer identifies a row, and a verdict a reader
    #: cannot match back to its row names an ambiguous pick. It is also what lets every
    #: reader of this surface decide "superseded?" from ONE signal — the version compared
    #: against the build's — rather than the row-renderers using the version and the verdict
    #: renderer using whether the sentence above is set, which are two predicates for one
    #: question and only agree while one producer keeps them in step.
    variant_identity_version: int
    #: Whether the pick is SHOWN cheaper than every other contestant that cleared the bar with a cost, by the
    #: separation test the frontier's dominance reads, one comparison per rival, Holm-adjusted together
    #: (:func:`_cost_ties`). ``shown_cheapest`` — every comparison separated in the pick's favour;
    #: ``not_separated`` — at least one rival could not be shown dearer, so the data says only that the
    #: cheapest is among the pick and :attr:`tied_with`, and the verdict names that set rather than a winner;
    #: ``untested`` — no test against any rival left in the set could decide (too few priced cases, or every
    #: case differing by one amount over too few cases for the exact test to reach α); ``only_cleared`` — no
    #: other contestant cleared the bar with a cost. ``None`` on a verdict stored before the pick was tested:
    #: that one was the lowest point cost, a winner the data may not have shown.
    cost_decision: FrontierCostDecision | None = None
    #: The rivals the pick was not shown cheaper than, ordered by point cost. Empty when ``cost_decision``
    #: is ``shown_cheapest`` or ``only_cleared``.
    tied_with: list[FrontierCostTie] = []
    #: The subject's boundary dimensions the pick was NOT held against a control on, because the frontier was given
    #: no control: the verdict then rests on the capability pillar alone, and says so here. Empty when every
    #: boundary dimension was checked, or the subject scores none.
    boundary_unchecked: list[str] = []


class SubjectFrontier(EvalBaseModel):
    """One subject's frontier: its points, and the verdict over them.

    Per-subject always: composite quality is derived from the subject's own rubric
    and is not comparable across subjects, so two subjects are two frontiers, never one.
    ``n_cleared_bar`` is carried so an absent verdict distinguishes "no variant cleared
    the bar" from "some cleared it but none has a known cost to be cheapest by", and
    ``n_undecided_bar`` so "none cleared" distinguishes "shown short" from "too few cases to say".
    """

    subject_id: str
    subject_label: str = ""
    #: The depth every point's ``pass_hat_k`` is read at, the bar applied to and domination
    #: decided on: the smallest ``k_runs`` among the subject's ranked runs, the depth every run
    #: here was commissioned to. One depth for the subject because pass^3 and pass^1 are
    #: different quantities, and ranking one contestant on each ranks the depths. A contestant
    #: measured deeper keeps its extra depth as precision (and on its curve), not as a harder bar.
    k: int = 1
    points: list[FrontierPoint] = []
    verdict: FrontierVerdict | None = None
    n_cleared_bar: int = 0
    #: Points whose pass^k interval straddles the bar: neither cleared nor missed.
    n_undecided_bar: int = 0
    #: The boundary (guardrail) dimensions any of the subject's results was scored on, sorted.
    boundary_dimensions: list[str] = []
    #: Points disqualified on a boundary dimension (:attr:`FrontierPoint.disqualified_by`).
    n_disqualified: int = 0
    #: Points not shown to hold every boundary dimension and not shown to breach one: never the pick, never
    #: disqualified.
    n_boundary_undecided: int = 0
    #: How the boundary pillar was read, as a sentence: checked against which control, or not checked and why.
    #: None when the subject scores no boundary dimension.
    boundary_pillar: str | None = None


class FrontierResult(EvalDocumentModel):
    """The verdict surface across every subject, plus its disclosures.

    ``bar`` is echoed back so the answer names the threshold its verdicts were
    made against (the bar is the caller's, never invented). ``control_variant_key``
    names the control every contestant's boundary dimensions were held against; each
    subject says how its boundary pillar was read. ``exclusions`` and the
    ``n_*`` counts keep an all-excluded corpus from rendering as an empty one,
    exactly as :class:`PivotTable` does.

    ``two_pillar`` (``TwoPillarDisclosure``, the statement that the boundary pillar was descoped) is retired: the
    pillar is decided (#613). A frozen bundle carrying it reads with it discarded — what it said is no longer true
    of any answer this build gives.
    """

    __retired_fields__: ClassVar[dict[str, str | None]] = {"two_pillar": None}

    bar: float | None = None
    #: The control the boundary pillar held every contestant against — the campaign's control arm on the bundle's
    #: frontier, the caller's choice on the scope's. None when none was given: no boundary dimension was checked.
    control_variant_key: str | None = None
    #: The 1–5 level a capability criterion had to reach for an attempt to pass, in every pass^k here
    #: (#642): the behavior's declared threshold
    #: (:meth:`~threetears.evals.kernel.host.BarRegistry.pass_threshold`) where the caller had one, else 3.
    #: A frontier stored before this was recorded defaults to 3, which is the threshold every pass^k was
    #: computed at then.
    rubric_threshold: int = 3
    subjects: list[SubjectFrontier] = []
    n_results: int = 0
    n_filtered_out: int = 0
    exclusions: ProjectionExclusions = ProjectionExclusions()
    #: ``run_id -> DEGRADED sentence`` for every run behind this frontier that came up
    #: short of its matrix, corpus-wide. The per-point copies say which contestants they
    #: land on; this one is the set, so a surface can state the rule once.
    completeness_disclosures: dict[str, str] = {}
    #: How many of ``n_results`` came from those runs — the weight of the caveat.
    n_degraded_observations: int = 0
    #: The distinct identity versions this answer's keyed observations were stamped at,
    #: ascending. Partitioning stops two stampings of one stack
    #: being RANKED together; it cannot stop them appearing as two rows, so the span and
    #: the sentence below are what keep a doubled row from reading as a mystery.
    identity_version_span: list[int] = []
    #: The one-sentence form of that span, or ``None`` when the answer rests on a single
    #: predicate. Computed from the placed rows rather than from the assembled
    #: points, so an answer whose points were all filtered out still says what
    #: it spanned.
    identity_span_disclosure: str | None = None
    #: That latency read under concurrency was left out of the latency axis — and so of every
    #: dominance test and "fastest" — and how much; ``None`` when every latency ranked was read
    #: serially (:mod:`~threetears.evals.analysis.contention`).
    contended_latency_disclosure: str | None = None


def _frontier_point(
    results: list[EvalResult],
    *,
    contestant: ContestantKey,
    k: int,
    cell_of_run: Mapping[str, Hashable],
    rubric_threshold: int,
    cassette_modes_by_run: Mapping[str, str],
    templates_by_run: Mapping[str, str | None],
    degraded_by_run: Mapping[str, str],
    footing_by_run: Mapping[str, ProductionFooting | None] | None,
) -> tuple[FrontierPoint, _ContestantCases]:
    """Aggregate one contestant's results into a single frontier point, and the per-case values behind it.

    Every axis aggregates over its own measured subset — pass^k reuses
    :func:`~threetears.evals.kernel.scoring.pool_pass_hat_k` (which drops infra-excluded
    iterations, and pools a case's attempts across the runs of one cell), composite drops the nulls :func:`~threetears.evals.kernel.scoring.result_composite`
    returns for infra-excluded and no-rubric results, cost sums only the
    production-replicating roles per result and means over the results that
    reported one, and latency means over the results that harvested a total. The
    subsets can legitimately differ in size, so each carries its own denominator.

    Args:
        results: One contestant's observations (all sharing a model, and a
            ``variant_key`` when captured). Never empty.
        contestant: The :func:`_contestant_key` these results were grouped under. Passed
            rather than re-derived off ``results[0]``: the key is what placed them
            together, so reading the identity back out of it makes "no row in this pool
            can disagree" structural instead of a claim a later change to the key could
            silently falsify — the same reason the gate lives IN the group key rather than
            beside it.
        k: The subject's depth, at which ``pass_hat_k`` is read (:attr:`SubjectFrontier.k`).
        cell_of_run: Run id → the cell its attempts are repeats of
            (:func:`~threetears.evals.kernel.scoring.pass_hat_k_cell`), so two runs of one
            configuration pool their attempts at a case and two configurations never do.
        rubric_threshold: Pass threshold forwarded to pass^k.
        cassette_modes_by_run: Every candidate run's recorded ``cassette_mode``, keyed by
            run id; narrowed here to the runs these results came from. **Required and
            deliberately without a default**, for the reason
            :func:`~threetears.evals.kernel.usage_capture.production_replicating_cost` refuses one:
            the map's absence and a genuinely uniform pool both yield no disclosure, so a
            default would let a caller that never supplied it render a live-versus-replayed
            pool as clean — the single failure this parameter exists to make impossible.
        templates_by_run: Every candidate run's ``template_id``, keyed by run id; narrowed
            here to the runs these results came from. **Required and deliberately without a
            default**, for the reason its neighbour above is: an absent map and a genuinely
            single-suite contestant both yield no disclosure, so a default would let a
            caller that never supplied it render a mixed-difficulty pool as clean.
        degraded_by_run: ``run_id -> DEGRADED sentence`` for the candidate runs that
            measured less than their matrix, from :func:`degraded_run_disclosures`;
            narrowed here to the runs these results came from. **Required and without a
            default**, on the rule its two neighbours state: an unsupplied map and a pool
            of genuinely complete runs both yield no disclosure, so a default would let a
            caller that forgot it rank a truncated contestant as if it were whole.
        footing_by_run: Each candidate run's production footing (``None`` for a run nobody could check), or
            ``None`` when the frontier was given no host declarations to read one with; narrowed here to the
            runs these results came from.

    Returns:
        A :class:`FrontierPoint` with quality (and its interval), cost, latency, and every
        denominator, and the per-case values each axis averages, which a domination test pairs
        on. Domination and the bar are filled by the caller, which needs the subject's other
        points and the bar to decide them.
    """
    from threetears.evals.kernel.usage_capture import count_substituted_deliveries, production_replicating_cost

    model = results[0].model
    # Unpacked from the group key, never re-derived off a row.
    variant_key, identity_version = contestant

    # pass^k — the canonical estimator, over this contestant's attempts pooled per case within
    # each cell: a repeat run of one configuration deepens its cases, while the same case under
    # another context (a second template, a replayed cassette) stays a case of its own beside it.
    pooled = pool_pass_hat_k(results, cell_of_run=cell_of_run, k=k, rubric_threshold=rubric_threshold)
    pass_hat_k = pooled["pass_hat_k"]
    pass_hat_1 = pass_hat_k_at(pooled["pass_hat_k_curve"], 1)["pass_hat_k"]
    # The same cases the headline averages, one estimate each: the interval runs over them, and a
    # domination test pairs two contestants on the test cases both measured.
    attempts, _ = pool_pass_hat_k_attempts(results, cell_of_run=cell_of_run, rubric_threshold=rubric_threshold)
    unit_estimates: list[float] = []
    unit_attempts = 0
    pass_by_case: dict[str, list[float]] = {}
    for (_, _, _, _, test_case_id), case_attempts in attempts.items():
        estimate = case_pass_hat_k(case_attempts, k)
        if estimate is None:
            continue
        unit_estimates.append(estimate)
        unit_attempts += len(case_attempts)
        pass_by_case.setdefault(test_case_id, []).append(estimate)
    pass_interval = case_rate_interval(unit_estimates, max_effective_n=unit_attempts / k) if unit_estimates else None

    # composite — case-weighted mean of the 0-1 scores, dropping the nulls that
    # mark infra-excluded and no-rubric results (a candidate failure scores 0.0
    # and stays in, depressing the mean rather than vanishing).
    values_by_case: dict[str, list[float]] = {}
    for result in results:
        composite = result_composite(result)
        if composite is not None:
            values_by_case.setdefault(result.test_case_id, []).append(composite)
    if values_by_case:
        mean_composite, composite_sem = _aggregate(values_by_case, WEIGHTING_EQUAL_PER_SCENARIO)
    else:
        mean_composite, composite_sem = None, None

    # cost — production-replicating roles only, meaned over the results that
    # reported one. None when none did; partial when some did and some did not.
    # Reads the RAW persisted `usage` where the run-summary aggregate reads
    # `resolve_result_usage`. The two once answered differently — a result whose
    # decomposition could be reconstructed at read counted toward that denominator and not
    # this one — which is why this site was deliberately left divergent: this is the surface
    # that RANKS contestants, and moving it was a ranking decision rather than a reporting
    # one. That reconstruction no longer exists, and with it gone the resolver returns rows
    # this call cannot distinguish from the raw field: identical on `captured`, `None` on
    # both sides when nothing was observed, and an empty list on a cancelled cell, which
    # sums to `None` exactly as the absent field does. So the two agree here in every case,
    # and reading the raw field is no longer a divergence to justify — it is the shorter way
    # to the same number. Aligning the call sites would be tidying, not a fix.
    #
    # The substitution disclosure is NOT part of that divergence and is read per result
    # regardless: a seeded or replayed delivery spent neither inner-agent dollars nor
    # tool credits, so letting it contribute its observed sum here would rank a
    # substituted contestant as cheaper than a live one on a difference in apparatus
    # rather than in configuration. Withholding is what keeps the ranking a comparison.
    #
    # The population is the turns the candidate took (`delivered_a_turn`), the one every comparison cost reads
    # (#619). A call the candidate's model refused or errored on took no turn, so its dollars are no turn's
    # spend, and averaged in they rank a refusing contestant cheap. A cell an apparatus fault cut short spent
    # less than a whole one, so averaged in it lets the rig make a contestant look cheaper. Both dollars stay
    # in program spend (`cost_usd`), which is accounting rather than a comparison.
    priced = [
        (r.test_case_id, c)
        for r, c in (
            (r, production_replicating_cost(r.usage, substituted_deliveries=count_substituted_deliveries(r)))
            for r in results
            if delivered_a_turn(r)
        )
        if c is not None
    ]
    observed_costs = [c for _, c in priced]
    n_cost = len(observed_costs)
    prod_cost = math.fsum(observed_costs) / n_cost if n_cost else None
    cost_is_partial = 0 < n_cost < len(results)
    # What that mean is made of. The frontier RANKS on cost, so a contestant whose run
    # priced its search calls against one whose run did not is a comparison of conventions
    # wearing the clothes of a comparison of configs — and nothing else on the point shows
    # it. Disclosed rather than corrected: which of the two is "right" depends on what the
    # operator is deciding, and the numbers are each honest about their own run.
    cost_compositions = pooled_cost_compositions(results)

    # latency — total wall-clock ms over the results that harvested a total, MINUS the cells a
    # harness failure produced. The same predicate `compute_pass_hat_k` and `compute_dimension_summary`
    # use, and for the same reason one step further on: this point is RANKED. Domination is
    # decided over pass^k x prod cost x total latency, and an infra-excluded cell carries a real
    # but truncated `LatencyMetrics` (unlike the timeout path, whose `_degraded_capture_fields`
    # leaves latency None), so pooling it here lets an apparatus fault push a contestant into or
    # out of the dominated set. Cost drops them too, above. A call the model refused or
    # errored on is out as well: it took no turn, and its round trip ranked an all-refusing contestant the fastest
    # on the subject, dominating the arms that answered. `delivered_a_turn` is that predicate — the
    # cells' own — and it keeps a turn the budget ended or the deadline struck, which really took
    # that long.
    timed = [
        (r.test_case_id, r.latency.total_ms)
        for r in results
        if r.latency is not None and r.latency.total_ms is not None and delivered_a_turn(r)
    ]
    totals = [total for _, total in timed]
    n_latency = len(totals)
    mean_total_ms = round(sum(totals) / n_latency, 3) if n_latency else None

    # cost / pass-probability — undefined when cost is unknown or nothing passed.
    # One attempt's mean cost over one attempt's pass probability, pass^1 — see the field.
    cost_per_acceptable_outcome = (prod_cost / pass_hat_1) if (prod_cost is not None and pass_hat_1) else None

    # Does this contestant's pool span cassette modes? Narrowed from the caller's whole
    # map to the runs these results actually came from, so a point can never be qualified
    # by a mode no observation in it recorded. Restricted to run ids the map knows, on the
    # rule every comparison here follows: a run whose mode was not supplied is not evidence
    # that it recorded a different one.
    contributing_runs = sorted({r.eval_run_id for r in results})
    contributing_modes = {
        run_id: cassette_modes_by_run[run_id] for run_id in contributing_runs if run_id in cassette_modes_by_run
    }

    # Does this contestant's pool span scenario suites? Narrowed the same way and on the
    # same rule: a run the map does not know is not evidence that it ran a different
    # template. A known run whose `template_id` is None IS evidence — that value is the
    # ad-hoc case, a real recorded state rather than an absence — so it stays in.
    contributing_templates = {
        run_id: templates_by_run[run_id] for run_id in contributing_runs if run_id in templates_by_run
    }
    # Grouped once and carried both ways: the sentence for a surface showing this point
    # alone, the entries for one showing it beside its siblings. Two derivations could
    # disagree; one cannot.
    template_span = _template_span_entries(contributing_templates)

    cases = _ContestantCases(
        pass_hat_k=_case_means((case, value) for case, values in pass_by_case.items() for value in values),
        production_replicating_cost=_case_means(priced),
        mean_total_ms=_case_means(timed),
    )
    point = FrontierPoint(
        cassette_mode_disclosure=cassette_mode_disclosure(contributing_modes),
        template_span=template_span,
        template_span_disclosure=_template_span_disclosure(template_span),
        # Narrowed to the contributing runs like the two above, and for the reason
        # they are: a caveat naming a run that put no observation on this row points
        # the reader at a denominator this point never used.
        completeness_disclosures={
            run_id: degraded_by_run[run_id] for run_id in contributing_runs if run_id in degraded_by_run
        },
        variant_key=variant_key,
        model=model,
        variant_identity_version=identity_version,
        identity_version_disclosure=_identity_version_disclosure(identity_version),
        pass_hat_k=pass_hat_k,
        k=k,
        n_pass_cases=pooled["n_cases_at_k"],
        pass_hat_k_curve=pooled["pass_hat_k_curve"],
        pass_hat_k_ci_low=None if pass_interval is None else pass_interval[0],
        pass_hat_k_ci_high=None if pass_interval is None else pass_interval[1],
        n_pass_no_criterion=pooled["n_no_criterion_excluded"],
        pass_hat_k_unmeasured_reason=pooled["pass_hat_k_unmeasured_reason"],
        mean_composite=mean_composite,
        composite_sem=composite_sem,
        composite_basis=pooled_composite_basis(results) if mean_composite is not None else None,
        n_composite_cases=len(values_by_case),
        production_replicating_cost=prod_cost,
        n_cost=n_cost,
        cost_is_partial=cost_is_partial,
        cost_compositions=cost_compositions,
        cost_per_acceptable_outcome=cost_per_acceptable_outcome,
        production_footing=(
            PooledProductionFooting(runs={run_id: footing_by_run.get(run_id) for run_id in contributing_runs})
            if footing_by_run is not None
            else None
        ),
        mean_total_ms=mean_total_ms,
        n_latency=n_latency,
        n_no_turn=sum(
            1 for r in results if classify_result(r) is ResultOutcome.CANDIDATE_FAIL and not delivered_a_turn(r)
        ),
        n_results=len(results),
        n_cases=len({r.test_case_id for r in results}),
        served_models=pooled_served_models(results),
    )
    return point, cases


class _ContestantCases(NamedTuple):
    """One contestant's per-test-case value on each domination axis: what a test between two pairs on.

    Keyed by ``test_case_id``, each the mean of the contestant's values at that case — its unit pass^k
    estimates (one per cell the case was measured under), or its observations' cost or total latency. An
    axis the contestant never measured is empty.
    """

    pass_hat_k: dict[str, Fraction]
    production_replicating_cost: dict[str, Fraction]
    mean_total_ms: dict[str, Fraction]


#: The domination axes, each a :class:`FrontierPoint` headline with its :class:`_ContestantCases` field of
#: the same name, and whether higher is better on it.
_DOMINATION_AXES: tuple[tuple[Literal["pass_hat_k", "production_replicating_cost", "mean_total_ms"], bool], ...] = (
    ("pass_hat_k", True),
    ("production_replicating_cost", False),
    ("mean_total_ms", False),
)


#: The domination axes with a known range: a case's pass^k is a rate. Cost and latency have none.
_AXIS_RANGES: dict[str, tuple[float, float]] = {"pass_hat_k": (0.0, 1.0)}


def _case_means(pairs: Iterable[tuple[str, float]]) -> dict[str, Fraction]:
    """The mean value at each case, from ``(case, value)`` pairs, exactly.

    Each value is read as the decimal it is written as (:func:`~threetears.evals.analysis.stats.exact_decimal`)
    and averaged over rationals — the arithmetic :func:`~threetears.evals.analysis.stats.level_difference`
    reads — so a constant per-case shift between two contestants stays a constant the separation test reads
    exactly, rather than acquiring a float residue its t-test would read as a tiny, perfectly consistent
    spread.
    """
    grouped: dict[str, list[Fraction]] = {}
    for case, value in pairs:
        grouped.setdefault(case, []).append(exact_decimal(value))
    return {case: sum(values, Fraction(0)) / len(values) for case, values in grouped.items()}


def _dominance_p(
    a: FrontierPoint, a_cases: _ContestantCases, b: FrontierPoint, b_cases: _ContestantCases
) -> float | None:
    """The p of the test that point ``a`` dominates point ``b``, or ``None`` where none can run.

    ``a`` dominates ``b`` when it is SHOWN better on every axis ``b`` measured — an intersection–union
    test (Berger 1982): each axis is tested on its own, and the claim stands only if every one rejects, so
    its p is the largest of theirs. Requiring every axis is what makes the conjunction a level-α test with
    no correction across axes; correcting across them would only make it stricter than α.

    Each axis is the engine's separation test (:func:`~threetears.evals.analysis.stats.separation_test`):
    paired over the test cases both contestants measured where they share at least two, Welch over each
    side's cases otherwise, two-sided, and counting only in ``a``'s favour — so on one axis a false call
    of "better" happens at most α/2 of the time.

    **"Better or equal" collapses to "better".** Showing a contestant no worse than another by at most a
    margin is an equivalence test, and these axes declare no margin; showing it no worse by zero is
    showing it better. So a tie on an axis — two contestants that each passed every case — blocks the
    claim rather than satisfying it: the data cannot say which is better there, and absence of evidence
    is not a claim.

    The missing-axis rule is the one point estimates were held to. If ``b`` measured an axis ``a`` did not,
    ``a`` cannot dominate: its unknown value there could be worse, and an unmeasured axis is never the
    winning "fastest" or "cheapest". The reverse does not block: an axis only ``a`` measured is skipped,
    which is what lets a working variant dominate one that failed everywhere (measured pass^k, no turn
    taken, so no cost or latency to rescue it).

    Args:
        a: The candidate dominator.
        a_cases: Its per-case values.
        b: The point tested for being dominated.
        b_cases: Its per-case values.

    Returns:
        The p: below α only when every axis ``b`` measured separates in ``a``'s favour, and 1.0 when an
        axis was tested and did not — a tie, a separation the other way, or no separation. ``None`` when
        no test could decide: ``b`` measured no axis, ``a`` lacks one ``b`` measured, or an axis has no test
        that can decide (:func:`_axis_p`).
    """
    measured = [(axis, higher) for axis, higher in _DOMINATION_AXES if getattr(b, axis) is not None]
    if not measured:
        return None
    largest = 0.0
    for axis, higher_is_better in measured:
        if getattr(a, axis) is None:
            return None
        p = _axis_p(
            getattr(a_cases, axis),
            getattr(b_cases, axis),
            higher_is_better=higher_is_better,
            value_range=_AXIS_RANGES.get(axis),
        )
        if p is None:
            return None
        largest = max(largest, p)
    return largest


def _axis_p(
    a_values: Mapping[str, Fraction],
    b_values: Mapping[str, Fraction],
    *,
    higher_is_better: bool,
    value_range: tuple[float, float] | None = None,
) -> float | None:
    """The p of the test that ``a`` is better than ``b`` on one axis, counting only in ``a``'s favour.

    The engine's separation test (:func:`~threetears.evals.analysis.stats.separation_test`): paired over the
    test cases both sides measured where they share at least two, Welch over each side's cases otherwise,
    two-sided. Read in one direction: where the tested means do not favour ``a`` the p is 1.0, so a false
    call of "better" happens at most α/2 of the time. The one axis test both :func:`_dominance_p` and the
    verdict's cost comparison (:func:`_cost_ties`) read.

    Where every case moved by one amount no t exists, and the bounded test on the axis's range reads it: pass^k
    per case lies in [0, 1]. Cost and latency declare no range, so no test of the mean can show such a move,
    and the axis reads 1.0, not separated in ``a``'s favour (the exact sign-flip test it once read asks about
    symmetry, not the mean — #597).

    Args:
        a_values: ``a``'s per-case values on the axis.
        b_values: ``b``'s.
        higher_is_better: Which way is better on the axis.
        value_range: The axis's inclusive bounds, or None where it has none.

    Returns:
        The p — 1.0 where the means do not favour ``a``, or where there is no spread on an axis with no range —
        or ``None`` where no test can decide: fewer than two cases on a side, or a spread that vanishes in floating
        point (:func:`~threetears.evals.analysis.stats.separation_test`).
    """
    if len(a_values) < 2 or len(b_values) < 2:
        return None
    shared = sorted(set(a_values) & set(b_values))
    paired = len(shared) >= 2
    a_side = [a_values[case] for case in shared] if paired else list(a_values.values())
    b_side = [b_values[case] for case in shared] if paired else list(b_values.values())
    tested = separation_test(a_side, b_side, paired=paired, value_range=value_range)
    if tested.refusal is not None:
        # No spread on an axis with no range: no test of the mean can show `a` better there, so the claim on this
        # axis is not shown — p 1, which holds α trivially — rather than left untested (#597).
        return 1.0
    p = tested.p
    if p is None:
        return None
    gap = sum(a_side, Fraction(0)) / len(a_side) - sum(b_side, Fraction(0)) / len(b_side)
    a_better = gap > 0 if higher_is_better else gap < 0
    return p if a_better else 1.0


def _decide_dominance(points: list[FrontierPoint], cases: list[_ContestantCases]) -> None:
    """Fill every point's ``dominance``, ``dominated`` and ``dominated_by``, by test.

    Every pair of contestants in the subject is one comparison, with the smaller p of its two directions
    (:func:`_dominance_p`). Each direction's p is built from two-sided axis tests read in one direction
    only, so it errs at most α/2; the two directions are disjoint claims, so the smaller of the two is the
    pair's two-sided p, as the engine's every separation test is. The pairs are one family, Holm-adjusted together
    (:func:`~threetears.evals.analysis.stats.holm_adjust`), so the chance that ANY point in the subject is
    flagged dominated when it is not stays within α — the rule every family of between-arm claims in
    the engine is read by. A domination is shown when its pair's adjusted p is below α in that direction.

    Args:
        points: The subject's points, in display order. Mutated.
        cases: Each point's per-case values, aligned with ``points``.
    """
    family: list[tuple[int, int, float]] = []
    tested: set[int] = set()
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            forward = _dominance_p(points[i], cases[i], points[j], cases[j])
            backward = _dominance_p(points[j], cases[j], points[i], cases[i])
            if forward is not None:
                tested.add(j)
            if backward is not None:
                tested.add(i)
            if forward is None and backward is None:
                continue
            if backward is None or (forward is not None and forward <= backward):
                family.append((i, j, forward if forward is not None else 1.0))
            else:
                family.append((j, i, backward))
    adjusted = holm_adjust([p for _, _, p in family])
    dominators: dict[int, list[tuple[int, float]]] = {}
    for (winner, loser, _), p_adjusted in zip(family, adjusted, strict=True):
        if p_adjusted < SIGNIFICANCE_ALPHA:
            dominators.setdefault(loser, []).append((winner, p_adjusted))
    for index, point in enumerate(points):
        # Built by walking the sorted points rather than by collecting a SET of names: a set over
        # `model` alone silently merges two variants of one model and leaves behind the dominated
        # row's own model name, which reads as a row dominating itself.
        found = sorted(dominators.get(index, []))
        point.dominated_by = [
            FrontierDominator(
                variant_key=points[winner].variant_key,
                model=points[winner].model,
                variant_identity_version=points[winner].variant_identity_version,
                p_value=p_adjusted,
            )
            for winner, p_adjusted in found
        ]
        point.dominated = bool(found)
        point.dominance = "dominated" if found else "not_separated" if index in tested else "untested"


def _cost_ties(
    pick: int, rivals: Sequence[int], points: Sequence[FrontierPoint], cases: Sequence[_ContestantCases]
) -> tuple[FrontierCostDecision, list[FrontierCostTie]]:
    """Whether the pick is shown cheaper than each rival, and the rivals it is not.

    One comparison per rival, each the frontier's own axis test on production-replicating cost
    (:func:`_axis_p`, counting only in the pick's favour), Holm-adjusted together
    (:func:`~threetears.evals.analysis.stats.holm_adjust`): dropping a rival from the set is a claim that it
    is dearer, and the claims are one family. The pick is the lowest point cost, so on identical
    contestants every comparison already leans its way; requiring each to separate is what keeps a
    "cheapest" named on noise near α.

    Args:
        pick: The index of the lowest point cost among the cleared, priced points.
        rivals: The other cleared, priced points' indices, ordered by point cost.
        points: The subject's points.
        cases: Each point's per-case values, aligned with ``points``.

    Returns:
        The decision and the rivals left in the set, each with its adjusted p (``None`` where untested).
    """
    if not rivals:
        return "only_cleared", []
    raw = [
        _axis_p(
            cases[pick].production_replicating_cost, cases[rival].production_replicating_cost, higher_is_better=False
        )
        for rival in rivals
    ]
    tested = [p for p in raw if p is not None]
    adjusted = iter(holm_adjust(tested))
    ties: list[FrontierCostTie] = []
    for rival, p in zip(rivals, raw, strict=True):
        p_adjusted = None if p is None else next(adjusted)
        if p_adjusted is not None and p_adjusted < SIGNIFICANCE_ALPHA:
            continue
        point = points[rival]
        assert point.production_replicating_cost is not None  # Only priced points are rivals.
        ties.append(
            FrontierCostTie(
                variant_key=point.variant_key,
                model=point.model,
                variant_identity_version=point.variant_identity_version,
                production_replicating_cost=point.production_replicating_cost,
                p_value=p_adjusted,
            )
        )
    if not ties:
        return "shown_cheapest", []
    if all(tie.p_value is None for tie in ties):
        return "untested", ties
    return "not_separated", ties


def _bar_decision(point: FrontierPoint, bar: float) -> BarDecision:
    """How a point's pass^k reads against the frontier's bar — the five words a campaign bar's verdict uses."""
    if point.pass_hat_k is None:
        return "no_data"
    if point.pass_hat_k_ci_low is None or point.pass_hat_k_ci_high is None:
        return "no_interval"
    cleared = interval_clears(
        (point.pass_hat_k_ci_low, point.pass_hat_k_ci_high), bar, margin=None, higher_is_better=True
    )
    return "undecided" if cleared is None else "cleared" if cleared else "missed"


def _boundary_values(results: Sequence[EvalResult]) -> dict[str, dict[str, float]]:
    """One contestant's per-case mean on each boundary dimension it was scored on — results the harness spoiled left out."""
    scores: dict[str, dict[str, list[int]]] = {}
    for result in results:
        if harness_faulted(result):
            continue
        for score in result.rubric_scores:
            if score.axis == "boundary":
                scores.setdefault(score.dim, {}).setdefault(result.test_case_id, []).append(score.score)
    return {dim: {case: sum(v) / len(v) for case, v in cases.items()} for dim, cases in scores.items()}


def _boundary_scales(results: Iterable[EvalResult]) -> dict[str, RubricScale]:
    """The scale each boundary dimension was scored on."""
    return {score.dim: score.scale for result in results for score in result.rubric_scores if score.axis == "boundary"}


def _boundary_eligible(point: FrontierPoint, control_variant_key: str | None) -> bool:
    """Whether a point may be the pick on the boundary pillar: the control, unchecked, or every dimension held."""
    if control_variant_key is None or point.variant_key == control_variant_key:
        return True
    return not point.disqualified_by and all(check.decision == "held" for check in point.boundary_checks)


def _decide_boundary_pillar(
    points: Sequence[FrontierPoint],
    members: Sequence[Sequence[EvalResult]],
    control_variant_key: str | None,
    margins: Mapping[str, float],
) -> tuple[list[str], str | None]:
    """Hold every contestant's boundary dimensions against the control, in place, and say how the pillar was read.

    Returns:
        The subject's boundary dimensions, sorted, and the sentence for :attr:`SubjectFrontier.boundary_pillar`.
    """
    scales = _boundary_scales(result for group in members for result in group)
    dimensions = sorted(scales)
    if not dimensions:
        return [], None
    named = ", ".join(dimensions)
    if control_variant_key is None:
        return dimensions, (
            f"not checked: no control was named, so no contestant was held against one on {named}; a verdict here "
            "rests on the capability pillar alone"
        )
    control = [index for index, point in enumerate(points) if point.variant_key == control_variant_key]
    if not control:
        return dimensions, (
            f"not checked: the control {control_variant_key} has no point in this subject, so no contestant was held "
            f"against it on {named}"
        )
    reference = _boundary_values(members[control[0]])
    for index, point in enumerate(points):
        if index in control:
            continue
        values = _boundary_values(members[index])
        checks = []
        for dim in dimensions:
            low, high = SCALES[scales[dim]].scores
            a, b, paired = contrast_samples(reference.get(dim, {}), values.get(dim, {}))
            margin = margins.get(dim)
            verdict = guardrail_decision(
                a, b, paired=paired, margin=margin, higher_is_better=True, value_range=(float(low), float(high))
            )
            reason = None
            if verdict.decision == "undecided":
                reason = verdict.refusal or (
                    "a side carries fewer than two cases scored on it"
                    if len(a) < 2 or len(b) < 2
                    else "no interval on the difference exists"
                    if verdict.interval is None
                    else "the interval on the difference reaches both sides of "
                    + (f"the declared margin {format_number(margin)}" if margin else "zero change (no margin declared)")
                )
            checks.append(
                FrontierBoundaryCheck(
                    dimension=dim,
                    decision=verdict.decision,
                    interval=verdict.interval,
                    margin=margin,
                    paired=paired,
                    n_cases_control=len(a),
                    n_cases=len(b),
                    undecided_reason=reason,
                )
            )
        point.boundary_checks = checks
        point.disqualified_by = [check.dimension for check in checks if check.decision == "breached"]
    return dimensions, f"checked: every contestant held against the control {control_variant_key} on {named}"


def compute_frontier(
    runs: list[EvalRun],
    results: list[EvalResult],
    *,
    bar: float | None = None,
    subject_id: str | None = None,
    rubric_threshold: int = 3,
    known_run_ids: set[str] | None = None,
    archived_run_ids: set[str] | None,
    profile: HostProfile | None = None,
    control_variant_key: str | None = None,
    guardrail_margins: Mapping[str, float] | None = None,
) -> FrontierResult:
    """Rank each subject's variants on quality x cost x latency and pick the cheapest above bar.

    One subject is one frontier: composite quality is derived from each
    subject's own rubric and is never comparable across subjects, so the subjects
    partition the answer at the top and are never pooled. Within a subject each
    contestant — one ``variant_key`` under one identity version, the resolved stack
    that would ship — becomes a point carrying pass^k (headline), mean-composite
    (secondary), production-replicating cost, total latency, and every denominator.
    The version is part of the contestant because one counter backs both key
    predicates and cannot say which moved, so pooling on the digest alone would merge
    on unverifiable provenance. The non-dominated set
    is computed and dominated points are flagged rather than dropped.

    The subjects partition; the CONDITIONS do not. ``variant_key`` names the shipping
    stack and nothing about how it was measured, so every scenario template and every
    cassette mode a variant was measured under pools into its single point. That is the
    recorded design (pool-and-disclose, not key-it), and both spans are disclosed on the
    point and carried onto the verdict: see ``FrontierPoint.template_span_disclosure`` and
    ``cassette_mode_disclosure``. A pooled pass^k over suites of unequal difficulty is
    still a real number about the variant — it is just not a score on any one suite, which
    is exactly what the disclosure says.

    **One depth per subject.** Every point's pass^k is read at :attr:`SubjectFrontier.k`, the
    shallowest ``k_runs`` among the subject's ranked runs, and estimated without bias from each
    case's attempts pooled across the runs of one cell
    (:func:`~threetears.evals.kernel.scoring.pool_pass_hat_k`): two runs of one variant under
    one ``context_key`` are more attempts at the same cases, while the same case measured under
    another context is a separate case beside it. So a repeat run sharpens a contestant's estimate
    rather than doubling its case count, and no contestant is ranked on a different depth than
    its rivals.

    When ``bar`` is supplied it gates pass^k, read by each point's pass^k interval the way every
    campaign bar is read (:attr:`FrontierPoint.bar_decision`): cleared only when the whole interval is at
    or above the bar, undecided when it straddles it. The cleared variant with the lowest known cost is
    the verdict's pick, and it is named the cheapest only when it is shown cheaper than every other cleared,
    priced variant by the dominance test's own cost comparison, Holm-adjusted over them; otherwise the
    verdict names the set the data cannot order (:attr:`FrontierVerdict.cost_decision`). An unsupplied bar
    yields the full frontier with no verdict, because inventing a quality threshold would editorialize.

    **Domination is decided by test, never read off point estimates** (:func:`_dominance_p`): a point is
    flagged dominated only when another is shown better on every axis it measured, the subject's pairs
    Holm-adjusted together, and otherwise reads ``not_separated`` or ``untested``. pass^k and the composite read
    capability dims only.

    **The boundary pillar.** A boundary (guardrail) rubric dimension is held per contestant against
    ``control_variant_key`` by the guardrail rule the bundle decides guardrails by
    (:class:`FrontierBoundaryCheck`), at the dimension's margin in ``guardrail_margins``. A contestant that
    breaches one is disqualified, named by the dimensions it breached (:attr:`FrontierPoint.disqualified_by`),
    and is never the pick; one not shown to hold every one is never the pick either. The pick must clear the bar
    on both pillars. With no control there is nothing to hold a contestant against: no boundary dimension is
    checked, :attr:`SubjectFrontier.boundary_pillar` says so, and a verdict names the dimensions it was not
    checked on (:attr:`FrontierVerdict.boundary_unchecked`).

    The two skips :func:`project_score_records` makes — a result whose run is
    absent, and one whose run captured no subject — are mirrored here and returned
    as :class:`ProjectionExclusions`, so an all-excluded corpus discloses that its
    data exists but cannot be placed rather than rendering as an empty scope.

    **A run that measured less than its matrix is admitted and disclosed, never
    refused** — the same pool-and-disclose position the cassette and template spans
    take, and for a stronger reason: its cells are real measurements of the same
    contestant, so dropping them would be the silent removal this surface already
    refuses for a dominated point. What it costs is the denominator, and that is
    what :attr:`FrontierPoint.completeness_disclosures` carries onto every affected
    row and onto the verdict. **The predicate is the completeness record, not the
    status:** a ``completed`` run with an infra-excluded cell is short too, and a
    surface that reasoned from ``status`` would pool it by default with nothing said.

    Args:
        runs: The runs supplying subject identity. Order is irrelevant.
        results: The observations to rank.
        bar: The pass^k threshold the verdict is made against, in ``[0, 1]``.
            ``None`` computes the frontier without a verdict.
        subject_id: When set, restrict to this subject; other subjects' results
            are counted as ``n_filtered_out`` rather than dropped silently.
        rubric_threshold: Forwarded to pass^k — a rubric score at or above it
            counts as a passing dimension — and recorded on the answer
            (:attr:`FrontierResult.rubric_threshold`). A campaign's bundle passes its
            behavior's declared threshold.
        known_run_ids: Every run id in the corpus, so a result excluded by the
            caller's own run filter (in ``known_run_ids`` but not ``runs``) is
            counted as filtered-on-request rather than unplaceable. See
            :func:`project_score_records`.
        archived_run_ids: The corpus run ids the operator has archived, so an
            archival exclusion is counted as ``results_from_archived_runs``
            rather than under the caller's ``status`` filter. See
            :func:`place_results`.
        profile: The host whose sweepable declarations each point's cost is read against: with it, every point
            and verdict names what each of its runs set away from the subject's production configuration
            (:attr:`FrontierPoint.production_footing`, #571). Read off the runs as given, so a run handed with its
            host payload elided reads as unchecked. ``None`` leaves that field ``None`` — nobody checked.
        control_variant_key: The variant every contestant's boundary dimensions are held against — the campaign's
            control arm. ``None`` checks none, and says so.
        guardrail_margins: The margin each boundary dimension is held to, by dimension name
            (``CampaignDesign.guardrail_margins``); a dimension absent is held at zero change.

    Returns:
        A :class:`FrontierResult` — one :class:`SubjectFrontier` per subject,
        sorted by subject id, plus the corpus-level exclusion accounting and the
        completeness disclosures of every short run it ranked over.

    Raises:
        FrontierError: ``bar`` is outside ``[0, 1]`` — refused here so both
            surfaces refuse identically rather than one clamping it.
    """
    if bar is not None and not (0.0 <= bar <= 1.0):
        raise FrontierError(f"bar {bar!r} is outside the pass^k range [0, 1] — pass^k is a probability")

    # Latency read under concurrency is never ranked: removed before any point is built, so the latency
    # axis, every dominance test and every "fastest" read only latency taken serially (#701).
    measures = profile.measures if profile is not None else None
    contended = {result.id for result in withheld_latency(results, measures)}
    results = withhold_contended_latency(results, measures)

    placed, exclusions = place_results(
        runs, results, known_run_ids, source="frontier", archived_run_ids=archived_run_ids
    )

    by_subject: dict[str, dict[ContestantKey, list[EvalResult]]] = {}
    subject_labels: dict[str, str] = {}
    # Every placed run's recorded cassette mode, collected here because a point is
    # built from RESULTS and the mode lives on the RUN — a result carries no cassette
    # field of its own to stand in for it.
    cassette_modes_by_run: dict[str, str] = {}
    # Every placed run's template, collected for the same reason and on the same seam: a
    # point is built from RESULTS and the template lives on the RUN. `EvalResult` carries no
    # template of its own, and its `test_case_id` is not a substitute: resolving a case id to
    # the template that owns it needs the case documents, which this function is never given
    # and which an ad-hoc run (`template_id is None`) would not answer anyway.
    templates_by_run: dict[str, str | None] = {}
    # Every candidate run that came up short of its matrix, collected on the same seam
    # and for the same reason as the two maps above: a point is built from RESULTS and
    # completeness lives on the RUN. Keyed on the run rather than derived from its
    # STATUS, because a `completed` run can be degraded too — one cell infra-excluded
    # is a shorter denominator that no status filter can see.
    degraded_by_run: dict[str, str] = {}
    n_filtered_out = 0
    n_considered = 0
    n_degraded_observations = 0
    # The observations this answer actually rests on. Collected here rather than derived
    # from the assembled points for two reasons: an answer whose points were all dropped
    # downstream still spanned what it spanned, and narrowing to what SURVIVED the subject
    # filter keeps the span from naming a predicate no row in this answer was keyed under
    # — the same rule the cassette-mode and template maps below follow.
    considered: list[EvalResult] = []
    # Which configuration each run measured, so a contestant's repeat runs pool their attempts at
    # a case (more depth) and runs under different conditions never do. Collected on the same seam
    # as the maps above: a point is built from RESULTS and the measurement context lives on the RUN.
    cell_of_run: dict[str, Hashable] = {}
    # The depth each subject is ranked at: the shallowest k any of its ranked runs was
    # commissioned to (see SubjectFrontier.k).
    k_by_subject: dict[str, int] = {}
    # Each placed run and its results, for the production footing a point's cost names (#571).
    placed_runs: dict[str, EvalRun] = {}
    results_of_run: dict[str, list[EvalResult]] = {}

    for run, result, resolved in placed:
        if subject_id is not None and resolved != subject_id:
            n_filtered_out += 1
            continue

        n_considered += 1
        considered.append(result)
        by_subject.setdefault(resolved, {}).setdefault(_contestant_key(result), []).append(result)
        subject_labels[resolved] = run.subject_snapshot.subject_label
        cassette_modes_by_run[run.id] = run.cassette_mode
        templates_by_run[run.id] = run.template_id
        cell_of_run[run.id] = pass_hat_k_cell(run)
        placed_runs[run.id] = run
        results_of_run.setdefault(run.id, []).append(result)
        k_by_subject[resolved] = min(k_by_subject.get(resolved, run.k_runs), run.k_runs)
        if (short := completeness_disclosure(run.completeness)) is not None:
            degraded_by_run[run.id] = short
            n_degraded_observations += 1

    footing_by_run: dict[str, ProductionFooting | None] | None = (
        {
            run_id: None
            if run.elided_payload_paths
            else profile.sweepables.production_footing(run, results_of_run[run_id])
            for run_id, run in placed_runs.items()
        }
        if profile is not None
        else None
    )

    subjects: list[SubjectFrontier] = []
    for resolved in sorted(by_subject):
        groups = by_subject[resolved]
        subject_k = k_by_subject[resolved]
        built = [
            (
                *_frontier_point(
                    groups[key],
                    contestant=key,
                    k=subject_k,
                    cell_of_run=cell_of_run,
                    rubric_threshold=rubric_threshold,
                    cassette_modes_by_run=cassette_modes_by_run,
                    templates_by_run=templates_by_run,
                    degraded_by_run=degraded_by_run,
                    footing_by_run=footing_by_run,
                ),
                groups[key],
            )
            for key in sorted(groups, key=lambda k: (str(k), ""))
        ]
        # The version is in the sort key because the partition made a TIE on the first two
        # reachable: two points can share a model and a key and differ only in predicate.
        # Sort stability alone would have made that order deterministic but arbitrary —
        # and the two rows sit adjacent, which is where an unexplained order reads as noise.
        built.sort(key=lambda entry: (entry[0].model, entry[0].variant_key, entry[0].variant_identity_version))
        points = [point for point, _, _ in built]
        point_cases = [cases for _, cases, _ in built]
        _decide_dominance(points, point_cases)
        boundary_dimensions, boundary_pillar = _decide_boundary_pillar(
            points, [members for _, _, members in built], control_variant_key, guardrail_margins or {}
        )

        verdict: FrontierVerdict | None = None
        n_cleared_bar = 0
        n_undecided_bar = 0
        if bar is not None:
            # The bar is read by the point's pass^k INTERVAL, the rule every campaign bar is read by
            # (`stats.interval_clears`): cleared only when the whole interval is at or above it, and a
            # straddle is undecided — neither a pass nor a failure, and never the cheapest pick.
            for point in points:
                point.bar_decision = _bar_decision(point, bar)
            cleared = [p for p in points if p.bar_decision == "cleared"]
            n_cleared_bar = len(cleared)
            n_undecided_bar = sum(1 for p in points if p.bar_decision == "undecided")
            # Both pillars: the bar cleared on pass^k, and every boundary dimension shown held against the control
            # (or no control to hold it against, which the verdict names). The control is the reference.
            # The cleared, priced points by point cost. The lowest is the pick, and it is named THE cheapest
            # only when it is shown cheaper than each of the rest (`_cost_ties`); otherwise the verdict names
            # it beside every rival it could not be shown cheaper than.
            costed = sorted(
                (
                    index
                    for index, p in enumerate(points)
                    if p.bar_decision == "cleared"
                    and _boundary_eligible(p, control_variant_key)
                    and p.production_replicating_cost is not None
                    and p.pass_hat_k is not None
                ),
                key=lambda i: (points[i].production_replicating_cost, -(points[i].pass_hat_k or 0.0), points[i].model),
            )
            if costed:
                pick = points[costed[0]]
                pick_cost, pick_pass_hat_k = pick.production_replicating_cost, pick.pass_hat_k
                assert pick_pass_hat_k is not None  # Only points with a pass^k are costed.
                cost_decision, tied_with = _cost_ties(costed[0], costed[1:], points, point_cases)
                verdict = FrontierVerdict(
                    variant_key=pick.variant_key,
                    model=pick.model,
                    pass_hat_k=pick_pass_hat_k,
                    k=subject_k,
                    pass_hat_k_ci_low=pick.pass_hat_k_ci_low,
                    pass_hat_k_ci_high=pick.pass_hat_k_ci_high,
                    production_replicating_cost=pick_cost,
                    cost_is_partial=pick.cost_is_partial,
                    production_footing=pick.production_footing,
                    # Carried from the picked point rather than re-derived: one predicate,
                    # so the verdict and the row it names cannot disagree about whether
                    # that contestant's observations spanned cassette modes.
                    cassette_mode_disclosure=pick.cassette_mode_disclosure,
                    # Carried from the picked point for the same one-predicate reason: the
                    # verdict and the row it names cannot disagree about which suites the
                    # pass^k it was picked on was averaged over.
                    template_span_disclosure=pick.template_span_disclosure,
                    # And on the same rule again: the bar was applied to this pass^k and the
                    # pick made on this cost, so a recommendation resting partly on runs that
                    # never finished their matrix has to say so where the recommendation is.
                    completeness_disclosures=pick.completeness_disclosures,
                    # Carried, not re-derived, for the one-predicate reason the four above
                    # are: the verdict and the row it names cannot disagree about which
                    # predicate minted the key this recommendation is addressed by.
                    identity_version_disclosure=pick.identity_version_disclosure,
                    served_models=pick.served_models,
                    variant_identity_version=pick.variant_identity_version,
                    cost_decision=cost_decision,
                    tied_with=tied_with,
                    boundary_unchecked=boundary_dimensions if control_variant_key is None else [],
                )

        subjects.append(
            SubjectFrontier(
                subject_id=resolved,
                subject_label=subject_labels.get(resolved, ""),
                k=subject_k,
                points=points,
                verdict=verdict,
                n_cleared_bar=n_cleared_bar,
                n_undecided_bar=n_undecided_bar,
                boundary_dimensions=boundary_dimensions,
                n_disqualified=sum(1 for p in points if p.disqualified_by),
                n_boundary_undecided=sum(
                    1
                    for p in points
                    if not p.disqualified_by and any(check.decision == "undecided" for check in p.boundary_checks)
                ),
                boundary_pillar=boundary_pillar,
            )
        )

    identity_versions = _identity_version_span(considered)

    return FrontierResult(
        bar=bar,
        control_variant_key=control_variant_key,
        rubric_threshold=rubric_threshold,
        subjects=subjects,
        n_results=n_considered,
        n_filtered_out=n_filtered_out,
        exclusions=exclusions,
        completeness_disclosures=degraded_by_run,
        n_degraded_observations=n_degraded_observations,
        identity_version_span=identity_versions,
        identity_span_disclosure=_identity_span_disclosure(identity_versions),
        contended_latency_disclosure=contended_latency_sentence(
            sum(1 for result in considered if result.id in contended), len(considered)
        ),
    )


__all__ = [
    "compute_frontier",
    "FrontierBoundaryCheck",
    "FrontierCostDecision",
    "FrontierCostTie",
    "FrontierDominator",
    "FrontierError",
    "FrontierPoint",
    "FrontierResult",
    "FrontierVerdict",
    "normalize_bar",
    "SubjectFrontier",
]
