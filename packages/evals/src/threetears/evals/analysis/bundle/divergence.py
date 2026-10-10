"""The divergence lens: lever changes where the whole run moved by a different amount than the part under test.

:func:`_scope_divergences` tests each level pair's remainder (whole minus part, per case) between the two levels,
corrected together by Holm's method; :func:`_comparable_pairs` and :func:`_unsound_subtraction` decide which
cross-scope measure pairs may be subtracted at all, and say why one is withheld.
"""

from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
from collections.abc import Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, NamedTuple

from threetears.evals.analysis.stats import (
    SIGNIFICANCE_ALPHA,
    LevelDifference,
    exact_decimal,
    holm_adjust,
    level_difference,
)
from threetears.evals.kernel.analysis_measures import MeasureCollection, MeasureSummary
from threetears.evals.kernel.host.profile import HostProfile
from threetears.evals.kernel.metrics import (
    AttributionScope,
    MetricDescriptor,
    materiality,
    partition_components,
    remainder_withheld_reason,
)
from threetears.evals.schema.models import EvalResult
from threetears.evals.analysis.bundle.caps import _MAX_DIVERGENCES
from threetears.evals.analysis.bundle.schema import (
    MeasureMovement,
    MovementDirection,
    RealizedDesign,
    ScopeDivergence,
)
from threetears.evals.analysis.bundle.measures import (
    _collect_measures,
    _PooledMeasure,
)
from threetears.evals.analysis.bundle.design import (
    _lever_cohort,
    _lever_levels,
    _SurfaceFolds,
)
from threetears.evals.analysis.bundle.confounds import _uncontrolled_dimensions
from threetears.evals.analysis.bundle.mechanisms import (
    _MechanismObservations,
    _observed_mechanism_confounds,
    _served_model_confounds,
    _ServedModels,
)

if TYPE_CHECKING:  # runtime models — TYPE_CHECKING-only to keep the runtime import graph minimal.
    from threetears.evals.schema.models import EvalRun


# How a divergence is decided. It is a test of the DIFFERENCE between the two movements, never two
# movements graded apart and set side by side: "the whole moved" beside "the part did not" is the
# difference between a significant and a non-significant result, which is not itself significant
# (Gelman & Stern 2006), and with no divergence at all it published one 11% to 33% of the time. So
# each case's remainder (its whole minus its part, per-case means) is tested between the two levels
# by the engine's between-level test (`stats.level_difference`), and the lever's tests are corrected
# together by Holm's method. A movement graded on its own still reads `improved`, `regressed`,
# `not_separated` or `equivalent`, but it is context: no verdict on one movement decides a divergence.

# The distinguishing clause of each reason a cross-scope difference is withheld. The reason
# reaches the generator as a SENTENCE (see ``ScopeDivergence.unattributed_withheld``), and that
# prose gets tuned for its reader — so the clause that identifies WHICH condition fired is named
# here and shared with the tests. Rewording the sentence around it is then free, while changing
# what a test actually pins takes editing this line.
WITHHELD_OPPOSITE_DIRECTIONS = "have opposite better-directions"
WITHHELD_DIFFERENT_POPULATIONS = "are averaged over different populations"
WITHHELD_UNKNOWN_POPULATION = "has no observation unit"


#: Measure name -> case id -> the exact mean of that case's observations of it, at one level.
_PerCaseMeasures = dict[str, dict[str, Fraction]]


def _per_case_measures(pooled: Mapping[str, _PooledMeasure]) -> _PerCaseMeasures:
    """Each numeric measure's per-case means at one level, exact — the unit a between-level test reads.

    Repeats of a case are averaged first, so a case repeated three times is one case, and averaged
    exactly (:func:`~threetears.evals.analysis.stats.exact_decimal`) so a constant read at unequal repeats
    stays one constant rather than acquiring a float residue a test would read as spread.

    Args:
        pooled: The level's pooled observations, from :func:`_collect_measures`.

    Returns:
        ``{name: {case id: mean}}`` for every numeric measure; text, boolean and categorical measures are absent.
    """
    per_case: _PerCaseMeasures = {}
    for name, (descriptor, values, cases) in pooled.items():
        if descriptor.data_type in ("text", "boolean", "categorical"):
            continue
        by_case: dict[str, list[Fraction]] = defaultdict(list)
        for value, case in zip(values, cases):
            by_case[case].append(exact_decimal(float(value)))
        per_case[name] = {
            case: sum(case_values, Fraction(0)) / len(case_values) for case, case_values in sorted(by_case.items())
        }
    return per_case


def measure_movement(
    descriptor: MetricDescriptor, at_a: Mapping[str, Fraction], at_b: Mapping[str, Fraction]
) -> MeasureMovement:
    """Test one measure's movement between two levels against its own noise, and read it against what matters.

    Public as the one reading the scope-divergence lens grades a whole, a part and each component by, so
    :func:`component_carrier` can be handed movements read the way the lens reads them.

    Args:
        descriptor: The measure's descriptor — its name, scope, better direction, and the materiality
            threshold that is both the delta's label and the margin an equivalence test runs against.
        at_a: Its per-case means at the first level.
        at_b: Its per-case means at the second level.

    Returns:
        The movement, read by :func:`~threetears.evals.analysis.stats.level_difference`.
    """
    tested = level_difference(
        at_a, at_b, equivalence_margin=descriptor.materiality_threshold, value_range=descriptor.value_range
    )
    assert tested.mean_a is not None and tested.mean_b is not None and tested.delta is not None
    direction: MovementDirection
    if tested.separated is None:
        direction = "untested"
    elif tested.separated:
        direction = "improved" if (tested.delta > 0) == bool(descriptor.higher_is_better) else "regressed"
    elif tested.equivalent:
        direction = "equivalent"
    else:
        direction = "not_separated"
    return MeasureMovement(
        name=descriptor.name,
        scope=descriptor.attribution_scope,
        mean_a=tested.mean_a,
        mean_b=tested.mean_b,
        delta=tested.delta,
        se_of_delta=tested.se,
        test=tested.test,
        n_a=tested.n_a,
        n_b=tested.n_b,
        direction=direction,
        materiality=materiality(descriptor.materiality_threshold, tested.delta),
        not_separated_reason=tested.not_separated_reason,
    )


def _difference_range(
    minuend: MetricDescriptor | None, subtrahend: MetricDescriptor | None
) -> tuple[float, float] | None:
    """The range one measure minus another can take, from their declared ranges, or None unless both declare one.

    A remainder or a gap between two measures is what the scope lens tests, and a move with no spread is read on a
    declared range (:func:`~threetears.evals.analysis.stats.separation_test`): ``[low₁ − high₂, high₁ − low₂]``.
    """
    if minuend is None or subtrahend is None or minuend.value_range is None or subtrahend.value_range is None:
        return None
    (low_1, high_1), (low_2, high_2) = minuend.value_range, subtrahend.value_range
    return low_1 - high_2, high_1 - low_2


def _remainders(whole: Mapping[str, Fraction], part: Mapping[str, Fraction]) -> dict[str, Fraction]:
    """Each case's whole minus its part, at one level, over the cases carrying both — what a divergence tests."""
    return {case: value - part[case] for case, value in whole.items() if case in part}


def _comparable_pairs(
    level_a: MeasureCollection,
    level_b: MeasureCollection,
    catalog: dict[str, MetricDescriptor],
) -> Iterator[tuple[str, MeasureSummary, MeasureSummary, MeasureSummary, MeasureSummary]]:
    """Yield ``(unit, e_at_a, e_at_b, s_at_a, s_at_b)`` for each cross-scope pair sharing a unit.

    Only numeric, directional measures present at BOTH levels qualify: a measure observed at
    one level and not the other has no difference to grade, and one with no better direction
    cannot be said to have improved.

    Args:
        level_a: Measures at the first level.
        level_b: Measures at the second level.
        catalog: Descriptors by measure name — the source of each measure's unit.

    Yields:
        One tuple per comparable cross-scope pair.
    """
    at_a = {m.name: m for m in level_a.measures}
    at_b = {m.name: m for m in level_b.measures}
    shared = [
        name
        for name in sorted(at_a.keys() & at_b.keys())
        if at_a[name].mean is not None and at_b[name].mean is not None
    ]
    by_scope: dict[AttributionScope, list[str]] = {"end_to_end": [], "subsystem": []}
    for name in shared:
        if at_a[name].higher_is_better is not None and catalog.get(name) is not None and catalog[name].unit:
            by_scope[at_a[name].attribution_scope].append(name)
    for whole in by_scope["end_to_end"]:
        for part in by_scope["subsystem"]:
            unit = catalog[whole].unit
            if unit and unit == catalog[part].unit:
                yield unit, at_a[whole], at_b[whole], at_a[part], at_b[part]


def _unsound_subtraction(
    *,
    whole: MeasureSummary,
    part: MeasureSummary,
    catalog: dict[str, MetricDescriptor],
    observation_units: list[dict[str, str]],
    profile: HostProfile,
) -> str | None:
    """Say why differencing these two measures would describe nothing, or None if it would not.

    Four ways a subtraction goes wrong while both sides stay individually true, so the
    answer is a SENTENCE rather than a flag: the returned string is what a reader is owed
    in place of the number, and it is the bundle's job to state it rather than the
    generator's to guess it from a null.

    **Two clauses about containment, and they are different questions.** The first is
    whether the part is inside the whole at all. Sharing a unit makes two measures
    comparable, not nested: background-tool phases, the drain wait and the judge phase are all
    milliseconds that fall OUTSIDE the turn spans ``total_ms`` sums — background work by
    design, so a long background run cannot inflate turn latency, and judging because it
    scores after the turns have ended — so subtracting one from the other once produced a
    ~95-second "unattributed" swing that described no stretch of wall-clock at all. Containment is
    therefore declared on the measure (``MetricDescriptor.contained_by``) and defaults to
    absent, which fails closed: an undeclared part is reported beside its whole with both
    movements intact and no arithmetic between them.

    The second is whether the part EXHAUSTS the whole, and it is the one this function was
    missing. Containment was treated as sufficient, so a part that is one of three declared
    components earned the subtraction, and the leftover — the sum of the OTHER components'
    movements — was published as "unattributed". That word is a claim that nothing accounts
    for it, and here the catalog names exactly what does. Disjointness was handled and
    partition was not, which is why the sibling case (``async_wait_ms``, undeclared) was
    correctly withheld in the same analysis that got this one wrong.

    Args:
        whole: The end-to-end measure's summary at the first level.
        part: The subsystem measure's summary at the first level.
        catalog: Descriptors by measure name — the source of the containment declaration.
        observation_units: Each compared level's measure → observation-unit map.
        profile: The host whose vocabulary this reads.

    Returns:
        A sentence naming the reason, or None when the difference is sound.
    """
    if whole.higher_is_better != part.higher_is_better:
        return (
            f"{whole.name} and {part.name} {WITHHELD_OPPOSITE_DIRECTIONS}, so adding their movements "
            "would sum two changes that mean opposite things."
        )
    seen = {units.get(name) for units in observation_units for name in (whole.name, part.name)}
    if None in seen:
        # Defensive: the unit map is built from the same walk as the measures it accompanies,
        # so a measure present in the collection has one. Withholding beats assuming — and
        # beats a KeyError, which would take the whole analysis down over one measure.
        return f"{whole.name} or {part.name} {WITHHELD_UNKNOWN_POPULATION} on one of the two levels."
    if len(seen) > 1:
        return (
            f"{whole.name} and {part.name} {WITHHELD_DIFFERENT_POPULATIONS} "
            f"({' vs '.join(sorted(str(unit) for unit in seen))}), so their difference is off by however "
            "many observations each result contributed."
        )
    # Containment and exhaustion are the rule every remainder site shares, so they are asked of
    # the one predicate that states them — the attribution chart compiler asks the same question.
    return remainder_withheld_reason(part.name, whole.name, catalog.get(part.name), catalog, measures=profile.measures)


def _carried_by(
    whole: MeasureMovement,
    level_a: _PerCaseMeasures,
    level_b: _PerCaseMeasures,
    catalog: dict[str, MetricDescriptor],
    *,
    profile: HostProfile,
) -> tuple[list[MeasureMovement], str | None, float | None]:
    """Grade each declared component of a whole between two levels, and name the one carrying it.

    The components are read from the same declarations :func:`_unsound_subtraction` reads —
    :func:`~threetears.evals.kernel.metrics.partition_components` over the whole describable measure space —
    so the partition this names and the partition that withholds a remainder are one partition.
    What this adds is where the movement went, which a withheld remainder deliberately leaves
    unsaid: the lens's own rule is to report the other components rather than a remainder, and a
    reader handed only that rule and no components has nothing to report.

    Args:
        whole: The whole-run measure's movement.
        level_a: Per-case measures at the first level.
        level_b: Per-case measures at the second level.
        catalog: Descriptors by measure name.
        profile: The host whose vocabulary this reads.

    Returns:
        ``(components, carried_by, carried_share)``, the carrier decided by :func:`component_carrier`.
    """
    components = [
        measure_movement(catalog[name], level_a[name], level_b[name])
        for name in partition_components(whole.name, catalog, measures=profile.measures)
        if level_a.get(name) and level_b.get(name)
    ]
    carrier = component_carrier(whole, components, level_a, level_b)
    if carrier is None:
        return components, None, None
    return components, carrier.name, carrier.delta / whole.delta


def component_carrier(
    whole: MeasureMovement,
    components: Sequence[MeasureMovement],
    level_a: Mapping[str, Mapping[str, Fraction]],
    level_b: Mapping[str, Mapping[str, Fraction]],
) -> MeasureMovement | None:
    """The component SHOWN to carry the whole's movement, or None where the data cannot name one.

    Public so the rule the scope-divergence lens names a ``carried_by`` component by can be read, and tested for its
    false-naming rate, on per-case values directly: the lens itself calls exactly this, over movements read by
    :func:`measure_movement`.

    The candidate is the component whose delta, in the whole's direction, is largest. It is named only
    when two things are shown, each by the engine's between-level test
    (:func:`~threetears.evals.analysis.stats.level_difference`): its own movement separates in the whole's
    direction, and it moved further that way than every other component — each case's difference between the
    candidate and that component, tested between the levels, Holm-adjusted over the other components. Named on
    the largest delta alone, two components moved alike would hand the carrier to whichever noise favoured.

    Args:
        whole: The whole-run measure's movement.
        components: Each component's movement.
        level_a: Per-case measures at the first level.
        level_b: Per-case measures at the second level.

    Returns:
        The carrier, or None when the whole's movement does not separate, no component moved its way, or the
        largest mover is not shown to move further than every other.
    """
    if whole.direction not in ("improved", "regressed") or whole.delta == 0.0:
        return None
    sign = 1.0 if whole.delta > 0 else -1.0
    moving = [component for component in components if component.delta * sign > 0]
    if not moving:
        return None
    top = max(moving, key=lambda component: (component.delta * sign, component.name))
    if top.direction != whole.direction:
        return None
    p_values: list[float] = []
    for other in components:
        if other.name == top.name:
            continue
        gap_a = {
            case: value - level_a[other.name][case]
            for case, value in level_a[top.name].items()
            if case in level_a[other.name]
        }
        gap_b = {
            case: value - level_b[other.name][case]
            for case, value in level_b[top.name].items()
            if case in level_b[other.name]
        }
        tested = level_difference(gap_a, gap_b)
        if tested.p_value is None or tested.delta is None or tested.delta * sign <= 0:
            return None
        p_values.append(tested.p_value)
    return top if all(p < SIGNIFICANCE_ALPHA for p in holm_adjust(p_values)) else None


class _DivergenceCount(NamedTuple):
    """How many whole-and-part pairs the divergence lens tested, and how many it could not."""

    tested: int
    untested: int


def _scope_divergences(
    runs: list[EvalRun],
    results_by_run: dict[str, list[EvalResult]],
    catalog: dict[str, MetricDescriptor],
    apparatus_levels: dict[str, dict[str, str | None]],
    design: RealizedDesign,
    *,
    folds: _SurfaceFolds,
    observations: _MechanismObservations,
    served: _ServedModels,
    profile: HostProfile,
) -> tuple[list[ScopeDivergence], int, _DivergenceCount]:
    """Find the lever changes where the whole run moved by a different amount than the part under test.

    Each level of each swept lever gets its own measure collection, built from that level's
    results by the same walk the rest of the bundle uses — so the observations compared here
    are the real ones, not summaries of summaries. Levels are then compared pairwise, and for
    each whole-and-part pair sharing a unit **the divergence itself is tested**: each case's
    whole minus its part (per-case means) between the two levels, by the engine's between-level
    test (:func:`~threetears.evals.analysis.stats.level_difference`). Every such test of one lever
    is one family, Holm-corrected at the engine's alpha, and a divergence is published only where
    its adjusted p is below it. Grading the whole and the part apart and publishing where their
    verdicts differ is NOT a test of the difference (Gelman & Stern 2006): a whole that
    separates beside a part that does not is ordinary noise, and that rule published a divergence
    that did not exist 11–33% of the time.

    Two honesty constraints ride along, because a comparison this cheap to produce is easy
    to over-read. The cohorts are grouped by ONE lever, so they also differ in whatever else
    moved — every such dimension is named in ``confounded_by``, and the reason a change in it
    matters is in the bundle's ``confound_catalog``.
    And the unattributed swing is stated only when the difference is sound at all — see
    :func:`_unsound_subtraction`, which names the reason when it is not, so a reader gets a
    sentence rather than a null.

    Args:
        runs: The campaign's resolved runs.
        results_by_run: Each run's results, keyed by run id.
        catalog: Descriptors by measure name, for the unit that makes a pair comparable.
        apparatus_levels: Dimension → run id → level key, from :func:`_apparatus_levels`.
        design: The derived design, narrowing each lever's cohort the same way the coverage
            map narrows it — a divergence read over pooled cells would disagree with the
            coverage entry for the same lever, and a reader has no way to tell which is right.
        folds: The campaign's :class:`_SurfaceFolds`. A resolved surface whose movement across
            its cohort is its swept members' is not compared as a lever of its own, for the reason
            the coverage map drops its row: its levels are the members' levels under another name,
            so every divergence it produced would restate one a member already reports.
        observations: The campaign's mechanism observations, compared across each divergence's two levels
            for the observed-mechanism confounds it names.
        served: Which model answered each result's candidate calls, for the served-model confound.
        profile: The host whose vocabulary this reads.

    Returns:
        The divergences to report (strongest first, capped), the count dropped by the cap, and how
        many pairs were tested and could not be.
    """
    found: list[tuple[float, ScopeDivergence]] = []
    n_tested = n_untested = 0
    all_run_ids = [run.id for run in runs]
    lever_levels = _lever_levels(runs, results_by_run, profile=profile)
    for lever, campaign_wide in lever_levels.items():
        cohort = set(_lever_cohort(lever, design, all_run_ids))
        by_level = {
            level: members for level, raw in campaign_wide.items() if (members := [r for r in raw if r in cohort])
        }
        if len(by_level) < 2 or folds.folds_away(lever, cohort):
            continue
        levels = sorted(by_level)
        collected = {
            level: _collect_measures(
                [result for run_id in by_level[level] for result in results_by_run.get(run_id, [])],
                profile=profile,
                undeclared="all_observed",
            )
            for level in levels
        }
        collections = {level: collection for level, (collection, _, _) in collected.items()}
        per_case = {level: _per_case_measures(pooled) for level, (_, _, pooled) in collected.items()}
        result_ids = {
            level: {result.id for run_id in by_level[level] for result in results_by_run.get(run_id, [])}
            for level in levels
        }
        units = {level: observation_units for level, (_, observation_units, _) in collected.items()}
        # Every test this lever's comparisons carried, with what a published divergence needs beside it.
        family: list[tuple[LevelDifference, dict[str, Any]]] = []
        for index, level_a in enumerate(levels):
            for level_b in levels[index + 1 :]:
                confounded = _uncontrolled_dimensions(
                    lever,
                    by_level[level_a] + by_level[level_b],
                    lever_levels,
                    apparatus_levels,
                    folds=folds,
                    profile=profile,
                ) + _observed_mechanism_confounds(
                    lever, {level: result_ids[level] for level in (level_a, level_b)}, observations, profile=profile
                )
                confounded += _served_model_confounds(result_ids[level_a] | result_ids[level_b], served)
                at_a, at_b = per_case[level_a], per_case[level_b]
                for unit, e_a, _e_b, s_a, _s_b in _comparable_pairs(
                    collections[level_a], collections[level_b], catalog
                ):
                    tested = level_difference(
                        _remainders(at_a[e_a.name], at_a[s_a.name]),
                        _remainders(at_b[e_a.name], at_b[s_a.name]),
                        value_range=_difference_range(catalog.get(e_a.name), catalog.get(s_a.name)),
                    )
                    if tested.p_value is None:
                        n_untested += 1
                        continue
                    whole = measure_movement(catalog[e_a.name], at_a[e_a.name], at_b[e_a.name])
                    part = measure_movement(catalog[s_a.name], at_a[s_a.name], at_b[s_a.name])
                    withheld = _unsound_subtraction(
                        whole=e_a,
                        part=s_a,
                        catalog=catalog,
                        observation_units=[units[level_a], units[level_b]],
                        profile=profile,
                    )
                    components, carried_by, carried_share = _carried_by(whole, at_a, at_b, catalog, profile=profile)
                    assert tested.test is not None
                    family.append(
                        (
                            tested,
                            {
                                "lever": lever,
                                "level_a": level_a,
                                "level_b": level_b,
                                "unit": unit,
                                "end_to_end": whole,
                                "subsystem": part,
                                "test": tested.test,
                                "n_cases_a": tested.n_a,
                                "n_cases_b": tested.n_b,
                                "p_raw": tested.p_value,
                                "unattributed_delta": None if withheld else whole.delta - part.delta,
                                "unattributed_withheld": withheld,
                                # Read from the same catalog `_unsound_subtraction` consulted, so the
                                # published fact and the decision made from it have one source.
                                "contained_by": (
                                    described.contained_by if (described := catalog.get(part.name)) else None
                                ),
                                "confounded_by": confounded,
                                "whole_components": components,
                                "carried_by": carried_by,
                                "carried_share": carried_share,
                            },
                        )
                    )
        n_tested += len(family)
        adjusted = holm_adjust([tested.p_value or 0.0 for tested, _ in family])
        for (tested, fields), p_adjusted in zip(family, adjusted):
            if p_adjusted >= SIGNIFICANCE_ALPHA:
                continue
            divergence = ScopeDivergence(**fields, p_adjusted=p_adjusted, family_size=len(family))
            # Rank by how far the whole's movement and the part's differ — the difference the test
            # read — as a fraction of the whole's own scale: the question a reader opens a divergence
            # to answer, and scale-free so milliseconds and dollars can be ordered against each other.
            # The scale takes both levels so a measure starting near zero cannot manufacture an
            # unbounded score.
            whole = divergence.end_to_end
            scale = max(abs(whole.mean_a), abs(whole.mean_b), 1e-9)
            strength = abs(tested.delta or 0.0) / scale
            found.append((strength, divergence))

    found.sort(
        key=lambda item: (
            -item[0],
            item[1].lever,
            item[1].level_a,
            item[1].level_b,
            item[1].end_to_end.name,
            item[1].subsystem.name,
        )
    )
    return (
        [divergence for _, divergence in found[:_MAX_DIVERGENCES]],
        max(0, len(found) - _MAX_DIVERGENCES),
        _DivergenceCount(n_tested, n_untested),
    )


__all__ = [
    "component_carrier",
    "measure_movement",
]
