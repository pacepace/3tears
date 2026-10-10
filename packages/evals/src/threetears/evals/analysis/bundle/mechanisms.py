"""Mechanism checks and served-model readings: whether a lever's declared mechanism measurably moved.

:func:`_mechanism_check` tests a swept lever's ``acts_on`` measure across its levels,
:func:`_observed_mechanism_confounds` and :func:`_served_model_confounds` name the mechanisms and served models
that moved with a model contrast, and :func:`_arm_mechanisms` / :func:`_arm_served_models` read both per arm.
"""

from __future__ import annotations

import math
from collections import defaultdict
from fractions import Fraction
from collections.abc import Collection, Iterable, Mapping, Sequence
from itertools import chain
from typing import Literal, NamedTuple


from threetears.evals.analysis.reporting import (
    pool_served_readings,
    ResultServedReading,
    served_reading,
)
from threetears.evals.analysis.stats import (
    SIGNIFICANCE_ALPHA,
    exact_decimal,
    holm_adjust,
    level_difference,
)
from threetears.evals.kernel.host.profile import HostProfile
from threetears.evals.schema.values import PooledProductionFooting
from threetears.evals.kernel.metrics import (
    describe_measure,
    summary_population,
)

# At runtime for its field set, which tells a result-level measure from a row-level one.
from threetears.evals.schema.models import EvalResult


from threetears.evals.analysis.bundle.schema import (
    _OBSERVED_MECHANISMS,
    ArmMechanismReading,
    ArmServedModel,
    CANDIDATE_SERVED_MODEL_CONFOUND,
    Confound,
    MechanismCheck,
    MechanismUncheckedReason,
    observed_mechanism_key,
    RealizedDesign,
)


from threetears.evals.analysis.bundle.config import _CANDIDATE_MODEL_LEVER


from threetears.evals.analysis.bundle.measures import (
    _carrier_leaves,
    _derived_leaves,
    _lineage_leaves,
    _open_map_leaves,
    _PER_RESULT,
    in_population,
)

from threetears.evals.analysis.bundle.design import _CampaignArms


#: Measure name -> result id -> ``(case id, value)``: that result's one observation of it, and the case
#: it observed. Built once per bundle by :func:`_mechanism_observations` and read by every lens that
#: compares a mechanism across levels. The case is kept because the separation test's unit is the case.
_MechanismObservations = dict[str, dict[str, tuple[str, float]]]


def _mechanism_value(result: EvalResult, name: str, *, profile: HostProfile) -> float | None:
    """One result's observation of ``name``, read off the result itself, or None when it carries none.

    Walks the sources the measure surface walks — the result's own scalars, its covariate and host
    measure maps, its single sub-models and the measures it implies — and reads a value only where the
    result carries exactly one numeric observation of the name at the per-result unit. A row-level
    leaf (one per element of a list, such as a usage row per role) is not read: pooling one role's
    count with another's describes no mechanism. Two observations of one name is ambiguity, and an
    ambiguous result contributes nothing rather than one of its values chosen by order.

    Args:
        result: The result to read.
        name: A measure or covariate name.
        profile: The host whose measure registry resolves the name's population.

    Returns:
        The value, or None — a result outside the measure's population (a faulted one outside a ``scored``
        measure's; one that took no turn outside a ``delivered`` one's, which every cost or latency measure
        is read as unless it declares ``all_observed``), no observation, an ambiguous one, or one that is
        not a finite number. Any other measure that declares no population is read over every result here,
        as it always was.
    """
    # Over every result where the measure declares nothing, as a mechanism always read — except a turn's
    # time or spend, which every reader takes over the turns taken (`summary_population`).
    if not in_population(summary_population(describe_measure(name, profile.measures), "all_observed"), result):
        return None
    outer = [
        value
        for leaf, value, _descriptor in chain(
            _lineage_leaves(result, profile=profile), _open_map_leaves(result, profile=profile)
        )
        if leaf == name
    ]
    inner = [
        value
        for leaf, value, _gaps, _carrier, unit in chain(
            _carrier_leaves(result, profile=profile), _derived_leaves(result)
        )
        if leaf == name and unit == _PER_RESULT
    ]
    found = outer or inner
    if len(found) != 1:
        return None
    value = found[0]
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        return None
    return float(value)


def _mechanism_observations(results: Sequence[EvalResult], *, profile: HostProfile) -> _MechanismObservations:
    """Every result's observation of every mechanism a lens here compares across levels.

    The names are each declared lever's ``acts_on`` and every covariate read as an observed mechanism
    (:data:`_OBSERVED_MECHANISMS`). Read once, so a coverage row, a divergence, a contrast and an arm
    reading of one measure read one set of values.

    Args:
        results: Every resolved result in the campaign.
        profile: The host whose declarations name the mechanisms.

    Returns:
        ``{name: {result id: (case id, value)}}``; a result that observed nothing of a name is absent under it.
    """
    names = {measure for _lever, measure in profile.sweepables.mechanisms}
    names |= set(_OBSERVED_MECHANISMS)
    observed: _MechanismObservations = {}
    for name in sorted(names):
        observed[name] = {
            result.id: (result.test_case_id, value)
            for result in results
            if (value := _mechanism_value(result, name, profile=profile)) is not None
        }
    return observed


def _per_case_means(values: Mapping[str, tuple[str, float]], result_ids: Collection[str]) -> dict[str, Fraction]:
    """One level's exact per-case means of one mechanism — the unit the separation test reads.

    Repeats of a case are averaged first, as :func:`_per_case_values` averages them for a family comparison,
    and averaged EXACTLY (:func:`~threetears.evals.analysis.stats.exact_decimal`): a float mean of three
    0.1s is 0.10000000000000002, so a constant measure read at three repeats a case on one level and one on
    another would differ by float noise with no spread, which the separation test counts as a gap.

    Args:
        values: Result id -> that result's ``(case id, value)``.
        result_ids: The level's result ids.

    Returns:
        Case id -> the exact mean of its results' observations; a case none of whose results observed it is
        absent.
    """
    by_case: dict[str, list[Fraction]] = defaultdict(list)
    for result_id in result_ids:
        if result_id in values:
            case_id, value = values[result_id]
            by_case[case_id].append(exact_decimal(value))
    return {
        case_id: sum(case_values, Fraction(0)) / len(case_values) for case_id, case_values in sorted(by_case.items())
    }


def _level_value(per_case: Mapping[str, Fraction]) -> Fraction:
    """A level's value of one mechanism: the exact mean of its per-case means.

    **The one derivation of "this level's value"**, read by the mechanism check, the observed-mechanism
    confound and the arm readings alike. Per case first, because the separation test runs over per-case
    means and a case repeated three times is still one case; a mean over results would weigh it thrice and
    show a different number for the same arm wherever the repeats are unequal.

    Args:
        per_case: The level's per-case means, from :func:`_per_case_means`; not empty.
    """
    return sum(per_case.values(), Fraction(0)) / len(per_case)


def _level_means(
    values: Mapping[str, tuple[str, float]], result_ids_by_level: Mapping[str, Collection[str]]
) -> dict[str, Fraction]:
    """Each level's value of one mechanism (:func:`_level_value`), for every level that observed it.

    Args:
        values: Result id -> that result's ``(case id, value)``.
        result_ids_by_level: The comparison's level -> the result ids at that level.

    Returns:
        The exact value per level that observed any; a level that observed none is absent.
    """
    return {
        level: _level_value(per_case)
        for level in sorted(result_ids_by_level)
        if (per_case := _per_case_means(values, result_ids_by_level[level]))
    }


class _LevelsSeparate(NamedTuple):
    """What :func:`_levels_separate` found over every pair of a lever's levels."""

    separated: bool
    #: Some pair had fewer than two cases on a side, or a spread that vanishes in floating point.
    untestable: bool
    #: Some pair shifted every case alike on a measure with no declared range, so no test of the mean ran on it.
    needs_range: bool


def _levels_separate(
    per_case: Mapping[str, Mapping[str, Fraction]], value_range: tuple[float, float] | None = None
) -> _LevelsSeparate:
    """Whether any pair of levels separates on per-case values, and whether any pair could not be tested.

    The engine's between-level test applied to every pair of levels
    (:func:`~threetears.evals.analysis.stats.level_difference`): paired over the cases both levels ran when
    they share at least two, else Welch's over each level's per-case values, the pairs Holm-corrected as one
    family and read against the same alpha. A gap with no spread at all (every case moved by the same nonzero
    amount, or two different constants) is read by the bounded test on the measure's declared range, and with
    none is not tested (the exact sign-flip test it once read asks about symmetry, not the mean — #597).

    Args:
        per_case: Level -> its exact per-case means, for every level that observed the measure.
        value_range: The measure's declared inclusive bounds, or None when it declares none.

    Returns:
        The :class:`_LevelsSeparate`.
    """
    raw: list[float] = []
    untestable = needs_range = False
    levels = sorted(per_case)
    for index, level_a in enumerate(levels):
        for level_b in levels[index + 1 :]:
            tested = level_difference(per_case[level_a], per_case[level_b], value_range=value_range)
            if tested.not_separated_reason is not None:
                needs_range = True
            elif tested.p_value is None:
                untestable = True
            else:
                raw.append(tested.p_value)
    separated = bool(raw) and min(holm_adjust(raw)) < SIGNIFICANCE_ALPHA
    return _LevelsSeparate(separated, untestable, needs_range)


def _mechanism_check(
    measure: str | None,
    result_ids_by_level: Mapping[str, Collection[str]],
    observations: _MechanismObservations,
    value_range: tuple[float, float] | None = None,
) -> MechanismCheck:
    """Decide whether a swept lever's declared mechanism measurably moved across its levels.

    Args:
        measure: The lever's ``acts_on``, or None when it declares none.
        result_ids_by_level: The lever's level -> the result ids at that level, over the row's cohort.
        observations: The campaign's mechanism observations.
        value_range: The mechanism measure's declared inclusive bounds, or None when it declares none.

    Returns:
        ``moved`` when some pair of observed levels separates; otherwise ``inert`` when every level observed
        the measure and every pair was tested; otherwise ``unchecked``, with the reason.
    """
    if measure is None:
        return MechanismCheck(state="unchecked", reason="not_declared")
    values = observations.get(measure, {})
    per_case = {
        level: cases
        for level in sorted(result_ids_by_level)
        if (cases := _per_case_means(values, result_ids_by_level[level]))
    }
    unobserved = [level for level in sorted(result_ids_by_level) if level not in per_case]
    state: Literal["moved", "inert", "unchecked"]
    reason: MechanismUncheckedReason | None = None
    found = _levels_separate(per_case, value_range)
    if len(result_ids_by_level) < 2:
        state, reason = "unchecked", "not_swept"
    elif found.separated:
        state = "moved"
    elif unobserved or len(per_case) < 2:
        state, reason = "unchecked", "levels_unobserved"
    elif found.untestable:
        state, reason = "unchecked", "too_few_observations"
    elif found.needs_range:
        state, reason = "unchecked", "uniform_move_needs_range"
    else:
        state = "inert"
    return MechanismCheck(
        state=state,
        measure=measure,
        level_means={level: float(_level_value(cases)) for level, cases in per_case.items()},
        level_n={level: len(cases) for level, cases in per_case.items()},
        unobserved_levels=unobserved,
        reason=reason,
    )


def _raises_observed_mechanism(lever: str, covariate: str, *, profile: HostProfile) -> bool:
    """Whether a comparison grouped by ``lever`` may name ``covariate`` as an observed-mechanism confound.

    **The one answer to that question**, asked by every lens that raises one — the coverage row, a
    divergence, and each pairwise contrast between two models — so no two of them can disagree about the
    same lever and covariate. Yes only for the candidate model, the comparison a pinned effort word fails to
    equalise; on any other lever the covariate moving is what the lever was swept to do. And never for the
    lever's own declared mechanism (``acts_on``): it moving is the lever's effect, not a rival to it.

    Args:
        lever: The lever the comparison varies.
        covariate: An observed-mechanism covariate.
        profile: The host whose declaration of the lever names its mechanism.
    """
    if lever != _CANDIDATE_MODEL_LEVER:
        return False
    return profile.sweepables.acts_on(lever) != covariate


def _observed_mechanism_confounds(
    lever: str,
    result_ids_by_level: Mapping[str, Collection[str]],
    observations: _MechanismObservations,
    *,
    profile: HostProfile,
) -> list[Confound]:
    """Name every observed mechanism whose levels' values diverged by at least its threshold.

    The third kind of confound, and the one no setting records: the levels compared ran under the same
    configuration as far as any lever says, and still did measurably different things. It qualifies the
    comparison and never suppresses it. A level that measured none of the covariate is left out of the
    comparison rather than read as zero, so a covariate nothing measured names no confound — the arm
    readings (``arm_mechanisms``) say it went unmeasured. Which covariates a lever may name at all is
    :func:`_raises_observed_mechanism`'s answer, asked here so every caller gets it.

    Args:
        lever: The lever the comparison varies.
        result_ids_by_level: The comparison's level -> the result ids at that level.
        observations: The campaign's mechanism observations.
        profile: The host whose declaration of the lever names its mechanism.

    Returns:
        One ``observed_mechanism`` confound per diverged covariate, sorted by dimension; empty for a lever
        the predicate excludes.
    """
    confounds: list[Confound] = []
    for covariate, mechanism in sorted(_OBSERVED_MECHANISMS.items()):
        if not _raises_observed_mechanism(lever, covariate, profile=profile):
            continue
        means = _level_means(observations.get(covariate, {}), result_ids_by_level)
        if len(means) < 2 or max(means.values()) - min(means.values()) < exact_decimal(mechanism.threshold):
            continue
        confounds.append(
            Confound(
                dimension=observed_mechanism_key(covariate),
                kind="observed_mechanism",
                level_values={level: float(mean) for level, mean in means.items()},
                threshold=mechanism.threshold,
            )
        )
    return confounds


def _model_contrast_confounds(
    side_a: Sequence[EvalResult],
    side_b: Sequence[EvalResult],
    observations: _MechanismObservations,
    *,
    profile: HostProfile,
) -> list[Confound]:
    """The observed-mechanism confounds of one pairwise contrast, when its two sides ran different models.

    Keyed by each side's model, so the confound names the two arms' values in the words a reader compares
    them in. A side whose results name no single model is no model contrast, and names nothing.

    Args:
        side_a: One side's results.
        side_b: The other side's results.
        observations: The campaign's mechanism observations.
        profile: The host whose declaration of the model lever names its mechanism.

    Returns:
        The confounds, or an empty list when the sides share a model or either names no single model.
    """
    models_a = {result.model for result in side_a}
    models_b = {result.model for result in side_b}
    if len(models_a) != 1 or len(models_b) != 1 or models_a == models_b:
        return []
    (model_a,), (model_b,) = models_a, models_b
    if model_a is None or model_b is None:
        return []
    return _observed_mechanism_confounds(
        _CANDIDATE_MODEL_LEVER,
        {model_a: {result.id for result in side_a}, model_b: {result.id for result in side_b}},
        observations,
        profile=profile,
    )


def _arm_production_footings(
    arms: _CampaignArms, results_by_run: Mapping[str, list[EvalResult]], *, profile: HostProfile
) -> dict[str, PooledProductionFooting]:
    """Each arm's production footing: what each of its runs set away from production (#571).

    An arm's production-replicating cost (its cells, contrasts and bars on that measure) pools its runs, and
    each run's footing is read off the host's sweepable declarations as :class:`RunSummary` reads it — so the
    arm carries every run's own, and says where they disagree, rather than one merged claim.

    Args:
        arms: The campaign's arms. A run that resolved no arm is in none.
        results_by_run: Each run's results.
        profile: The host whose declarations are read.

    Returns:
        Variant key -> the arm's pooled footing, for every arm.
    """
    return {
        variant_key: PooledProductionFooting(
            runs={
                run.id: None
                if run.elided_payload_paths
                else profile.sweepables.production_footing(run, results_by_run.get(run.id, []))
                for run in members
            }
        )
        for variant_key, members in sorted(arms.keyed.items())
    }


def _design_with_mechanism_confounds(
    design: RealizedDesign,
    results_by_run: Mapping[str, list[EvalResult]],
    observations: _MechanismObservations,
    *,
    served: _ServedModels,
    profile: HostProfile,
) -> RealizedDesign:
    """The design with each contrast arm's observed confounds against the control arm.

    Args:
        design: The derived design.
        results_by_run: Each run's results.
        observations: The campaign's mechanism observations.
        served: Which model answered each result's candidate calls, from :func:`_served_models`.
        profile: The host whose declaration of the model lever names its mechanism.

    Returns:
        The design, its contrasts qualified where they ran another model and a mechanism diverged, and where
        one requested model id was answered by more than one model across the two arms; unchanged when no
        control resolved.
    """
    if design.control_arm is None:
        return design
    control = [result for run_id in design.control_arm.run_ids for result in results_by_run.get(run_id, [])]
    contrasts = []
    for arm in design.contrasts:
        side = [result for run_id in arm.run_ids for result in results_by_run.get(run_id, [])]
        confounds = _model_contrast_confounds(control, side, observations, profile=profile)
        confounds += _served_model_confounds((result.id for result in (*control, *side)), served)
        contrasts.append(arm.model_copy(update={"mechanism_confounds": confounds}))
    return design.model_copy(update={"contrasts": contrasts})


def _arm_mechanisms(
    arms: _CampaignArms, results_by_run: Mapping[str, list[EvalResult]], observations: _MechanismObservations
) -> list[ArmMechanismReading]:
    """Each arm's mean of every observed-mechanism covariate, saying so where none was measured.

    Args:
        arms: The campaign's arms. A run that resolved no arm is in none, and so in no reading.
        results_by_run: Each run's results.
        observations: The campaign's mechanism observations.

    Returns:
        One reading per arm and covariate, sorted by arm then covariate.
    """
    readings: list[ArmMechanismReading] = []
    for variant_key, members in sorted(arms.keyed.items()):
        result_ids = {result.id for run in members for result in results_by_run.get(run.id, [])}
        for covariate in sorted(_OBSERVED_MECHANISMS):
            values = observations.get(covariate, {})
            means = _level_means(values, {variant_key: result_ids})
            readings.append(
                ArmMechanismReading(
                    variant_key=variant_key,
                    covariate=covariate,
                    mean=float(means[variant_key]) if variant_key in means else None,
                    n_measured=sum(1 for result_id in result_ids if result_id in values),
                    n_results=len(result_ids),
                )
            )
    return readings


#: Result id -> what its candidate calls say about the model that answered them. A result whose candidate
#: left no usage row is absent: nothing was called, so nothing answered, and no claim is made about it.
_ServedModels = dict[str, ResultServedReading]


def _served_models(results: Iterable[EvalResult]) -> _ServedModels:
    """Read which model answered each result's candidate calls, off its candidate usage rows.

    ``RoleUsage.served_model`` only — what the provider's response named — and never ``RoleUsage.model``
    or the run's ``candidate_model``, which are what the launch asked for and, for a floating alias, name
    the pointer rather than the model behind it.

    Args:
        results: The campaign's results.

    Returns:
        Each result's reading, for the results whose candidate left a usage row.
    """
    return {result.id: reading for result in results if (reading := served_reading(result)) is not None}


def _served_model_confounds(result_ids: Iterable[str], served: _ServedModels) -> list[Confound]:
    """Name the candidate's served model as a confound where one requested id was answered by more than one model.

    The served model is EXPECTED to move with the requested one — a comparison between two model ids is
    a comparison between the models that answered them — so a difference between arms that asked for
    different ids is the lever, not a confound. What is a confound is one requested id answered by two
    models across the runs compared: within one arm, its numbers are a mixture; between two arms that
    asked for the same id, the arms differ by a model nobody set. The rule is the fold the engine applies
    to a resolved surface (:class:`_SurfaceFolds`): the served model folds into the requested one where it
    is constant within each requested id, and only there.

    Args:
        result_ids: The results under comparison, every side together.
        served: The campaign's readings, from :func:`_served_models`.

    Returns:
        One ``served_model`` confound, ``varied`` where some requested id was answered by two or more named
        models, ``undecided`` where none was but some candidate call named no model — which model answered
        cannot be established there, and unknown is never read as one model. Empty otherwise, including where
        no result under comparison called its candidate.
    """
    by_requested: dict[str, set[str]] = {}
    unrecorded = False
    for result_id in result_ids:
        if (reading := served.get(result_id)) is not None:
            by_requested.setdefault(reading.requested, set()).update(reading.served)
            unrecorded = unrecorded or reading.unrecorded
    if any(len(models) > 1 for models in by_requested.values()):
        return [Confound(dimension=CANDIDATE_SERVED_MODEL_CONFOUND, kind="served_model")]
    if unrecorded:
        return [Confound(dimension=CANDIDATE_SERVED_MODEL_CONFOUND, kind="served_model", status="undecided")]
    return []


def _arm_served_models(
    arms: _CampaignArms, results_by_run: Mapping[str, list[EvalResult]], served: _ServedModels
) -> list[ArmServedModel]:
    """Each arm's served models, as the provider's responses named them.

    Args:
        arms: The campaign's arms. A run that resolved no arm is in none, and so in no reading.
        results_by_run: Each run's results.
        served: The campaign's readings, from :func:`_served_models`.

    Returns:
        One reading per arm whose candidate left a usage row, sorted by arm.
    """
    readings: list[ArmServedModel] = []
    for variant_key, members in sorted(arms.keyed.items()):
        pooled = pool_served_readings(
            served.get(result.id) for run in members for result in results_by_run.get(run.id, [])
        )
        if pooled is not None:
            readings.append(ArmServedModel(variant_key=variant_key, **pooled.model_dump()))
    return readings


__all__: list[str] = []
