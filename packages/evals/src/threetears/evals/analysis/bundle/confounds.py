"""The confound scan: what was not held fixed across a cohort, and why each one matters.

:func:`_uncontrolled_dimensions` names every swept lever, resolved surface and apparatus or world dimension that
varied inside a cohort, and :func:`_confound_catalog` collects the reason for every confound dimension the bundle
names, once. The reason sentences are kept here beside the scan that states them.
"""

from __future__ import annotations

from collections.abc import Collection


from threetears.evals.kernel.host.profile import HostProfile


from threetears.evals.analysis.bundle.schema import (
    _OBSERVED_MECHANISMS,
    _SERVED_MODEL_CONFOUNDS,
    AnalysisContextBundle,
    Confound,
    OBSERVED_MECHANISM_PREFIX,
    UNDECIDED_CONFOUND_PREFIX,
    UNVERIFIED_FOLD_PREFIX,
)


from threetears.evals.analysis.bundle.design import (
    _SurfaceFolds,
    world_dimension_key,
)


# Why another swept lever varying inside a cohort clouds the comparison. Generic on
# purpose: which lever it is says nothing extra here, because a campaign sweeping it
# already believes it can move the numbers — that belief is what makes it a lever.
_SWEPT_LEVER_CONFOUNDS = (
    "another lever this campaign swept, which also took more than one value across these runs, "
    "so part of the movement may belong to it"
)

# Why a RESOLVED SURFACE still clouds a comparison after its family's swept members are taken
# back out of it. It reaches a confound list only in that state — where the residuals agree, the
# surface's movement IS the members' movement seen a second time and it is not named at all — so
# the sentence can say what the remaining difference means rather than hedging between the two.
_UNEXPLAINED_SURFACE_CONFOUNDS = (
    "the resolved surface that {family}'s members are written into, and it differed across these runs "
    "by more than the members swept here account for — something besides the swept knobs changed, so "
    "part of the movement may belong to whatever that was. What this dimension is: {prose}"
)

# Why a surface folded into a FIXED knob without a check still qualifies the comparison. The surface is
# in the variant key, so every run of an arm resolves one surface; with one arm per level of the knob the
# fold holds by construction, and a second change made exactly where the knob changed would fold with it.
# Named on every comparison that folds it that way, so a memo cannot present it as a checked non-confound.
_UNVERIFIED_FOLD_CONFOUNDS = (
    "folded, unverified: {surface} is the resolved surface {knob} is written into, and it moved with {knob} "
    "here — but every level of {knob} in these runs was run by one arm only, so nothing in them could have "
    "shown {surface} moving apart from {knob}. It is reported as the same change as {knob} without having been "
    "checked, and part of the movement may belong to anything else written into it; two arms at one level of "
    "{knob} would test it. What {surface} is: {prose}"
)

# The same, for a surface a FIXED lever is written into (a kind's overlay marked ``ResolvesInto``).
# It has no members to take out, so the sentence states the fold rule's own two ways of failing:
# runs that held the knob at one level carried different surfaces, which the knob cannot have done,
# or a run did not record the surface at all (the confound's ``undecided`` status says which).
_UNEXPLAINED_WRITTEN_SURFACE_CONFOUNDS = (
    "the resolved surface that {family} is written into, and across these runs it was not shown to have "
    "moved only where {family} did — it differed between runs that held {family} at one level, or some run "
    "did not record it — so part of the movement may belong to something besides that knob. What this "
    "dimension is: {prose}"
)


# What a world dimension placed differently across a cohort does to the comparison. The engine
# owns this clause because it is identical for every host and every dimension — one run seeded the
# state and another let the subject witness whatever was there, so the two were not measured
# against the same starting world. The host's own ``matters`` prose is appended to it rather than
# used as it, because the two answer different questions in different registers: ``matters`` says
# why a scenario would presume the dimension, and this slot says what a difference in it costs a
# reader. Handing one to the other produced a sentence that read as neither.
_WORLD_PLACEMENT_CONFOUND = (
    "the subject was placed in a different world across these runs — one seeded this dimension and "
    "another let the subject witness whatever was there — so they did not start from the same state"
)


def _world_confound_reason(matters: str) -> str:
    """Compose one world dimension's confound reason: the engine's clause, then the host's why.

    Args:
        matters: The dimension's required ``matters`` prose.

    Returns:
        The reason a confound catalog renders for it.
    """
    return f"{_WORLD_PLACEMENT_CONFOUND}. What this dimension is: {matters}"


def _apparatus_confound_reasons(profile: HostProfile) -> dict[str, str]:
    """Apparatus dimension -> why a change in it clouds the measurement, for one host.

    Read straight off the host's own declarations, so a dimension a host registers reaches the
    confound scan with its reason attached and cannot arrive as a bare name.

    **World dimensions join the sweepable apparatus here**, because a world that moved across a
    campaign is the same defect one axis over: a dimension the subject perceives and one run
    seeded while another merely witnessed is a rival explanation for whatever moved, and one that
    silently widens the variance of every run in the campaign. Their reason is composed by
    :func:`_world_confound_reason` — the engine's generic clause about what a placement difference
    costs, plus the host's own ``matters`` prose about what the dimension is.

    **Built per call from the profile handed in rather than at import**, which is not a style
    choice: two hosts can assemble bundles in one process, and a module-level dict would freeze
    whichever host's declarations it was first built from and scan every other host against them.

    Args:
        profile: The host whose apparatus and world declarations the reasons come from.

    Returns:
        ``{dimension: reason}`` for every apparatus sweepable the host declares, plus every
        world dimension under :func:`world_dimension_key`.
    """
    reasons = {
        declared.name: declared.confounds
        for declared in profile.sweepables.declarations
        if declared.role == "apparatus" and declared.confounds is not None
    }
    if profile.world is not None:
        reasons.update(
            {
                world_dimension_key(declared.name): _world_confound_reason(declared.matters)
                for declared in profile.world.declarations
            }
        )
    return reasons


def _uncontrolled_dimensions(
    lever: str,
    cohort_run_ids: list[str],
    lever_levels: dict[str, dict[str, list[str]]],
    apparatus_levels: dict[str, dict[str, str | None]],
    *,
    folds: _SurfaceFolds,
    profile: HostProfile,
) -> list[Confound]:
    """Name everything that was not held fixed across a cohort, and why each one matters.

    Grouping runs by one lever leaves every other swept lever free to vary inside the
    groups, so the comparison is marginal — averaged over whatever else moved — rather
    than controlled. That is the honest thing a campaign of this size can offer, and it is
    only misleading when it goes unsaid: a reader who assumes a clean A/B will attribute
    the whole movement to the one lever named.

    The apparatus dimensions are the half a lever scan structurally cannot see, because
    they are run attributes rather than overlays — and they are the more serious half.
    A swept lever varying is at least a knob somebody chose to turn; a template or a judge
    varying means the two arms were measured by different instruments, which no amount of
    replication fixes.

    An apparatus dimension that was never recorded on some run here is reported too, with
    ``status='undecided'``. Silence would read as "it held still", which is an observation
    nobody made — and the state is structural rather than worded into the reason, so a
    consumer never has to read prose to find out which of the three cases it is looking at.

    Args:
        lever: The lever being compared (excluded from its own confound list).
        cohort_run_ids: Every run in the cohort under comparison.
        lever_levels: The full lever → level → run-ids map.
        apparatus_levels: Dimension → run id → level key, from :func:`_apparatus_levels`.
        folds: The campaign's :class:`_SurfaceFolds`. A resolved surface that varied only because
            its swept members did is the lever under comparison (or its sibling members) seen a
            second time, so it is not named — and likewise a surface a fixed knob is written into,
            where it held one level within each of the knob's. One whose residual or surface could
            not be read is named as ``undecided``, because whether anything besides the swept knob
            moved is exactly what no run recorded.
        profile: The host whose vocabulary this reads.

    Returns:
        The swept levers that varied within the cohort (sorted), then the apparatus
        dimensions that varied or could not be decided (in declaration order).
    """
    cohort = set(cohort_run_ids)
    confounds: list[Confound] = []
    for other, by_level in sorted(lever_levels.items()):
        if other == lever or sum(1 for members in by_level.values() if cohort.intersection(members)) <= 1:
            continue
        if other in folds.surfaces:
            verdict = folds.fold(other, cohort_run_ids)
            if verdict == "explained":
                continue
            if verdict == "unverified":
                # Folded — the knob names the comparison — but marked, because these runs could not
                # have shown the surface moving apart from its knob, and an untested fold is no pass.
                confounds.append(Confound(dimension=f"{UNVERIFIED_FOLD_PREFIX}{other}", kind="unverified_fold"))
                continue
            if verdict == "undetermined":
                confounds.append(Confound(dimension=other, kind="swept_lever", status="undecided"))
                continue
        confounds.append(Confound(dimension=other, kind="swept_lever"))
    return confounds + _apparatus_confounds(cohort_run_ids, apparatus_levels, profile=profile)


def _apparatus_confounds(
    cohort_run_ids: Collection[str],
    apparatus_levels: dict[str, dict[str, str | None]],
    *,
    profile: HostProfile,
) -> list[Confound]:
    """Name every apparatus dimension that moved, or could not be shown to have held, in a cohort.

    The single producer of an apparatus confound. Three callers ask this question over three
    different cohorts — a lever's runs, a divergence's two arms, and the whole campaign — and
    they must answer it identically, because the same dimension reported as varied under one
    and silent under another reads to a generator as a fact about the comparison rather than
    about which run set it was asked over.

    Args:
        cohort_run_ids: The runs under comparison.
        apparatus_levels: Dimension → run id → level key, from :func:`_apparatus_levels`.
        profile: The host whose vocabulary this reads.

    Returns:
        One entry per dimension that varied or is undecided, **sorted by dimension**. A
        dimension that held still is absent — that absence is the claim, so it is only ever
        made about runs that recorded a value.

    Note:
        **Sorted rather than emitted in declaration order, because this list is fingerprinted.**
        ``apparatus_confounds`` and every ``coverage[].confounded_by`` are lists, ``to_dict()``
        preserves list order, and ``canonical_digest`` sorts dict keys only — so with declaration
        order the fingerprint moved whenever the registry was rearranged, over evidence that had
        not changed. A host's declaration order is a layout choice and must not be an input to an
        identity: the second host to register makes that unavoidable rather than merely untidy,
        since two hosts have no shared order to agree on.
    """
    cohort = set(cohort_run_ids)
    confounds: list[Confound] = []
    reasons = _apparatus_confound_reasons(profile=profile)
    for dimension in sorted(reasons):
        observed = {key for run_id, key in apparatus_levels.get(dimension, {}).items() if run_id in cohort}
        if None in observed:
            confounds.append(Confound(dimension=dimension, kind="apparatus", status="undecided"))
        elif len(observed) > 1:
            confounds.append(Confound(dimension=dimension, kind="apparatus"))
    return confounds


def _confound_catalog(bundle: AnalysisContextBundle, *, profile: HostProfile) -> dict[str, str]:
    """Collect the reason for every confound dimension appearing anywhere in the bundle.

    The reason belongs in the payload — a name a reader cannot judge is a label — but it
    belongs there ONCE. The same dimension is re-stated by every lever that names it, by
    every divergence, and by the campaign-wide scan, so an inlined multi-sentence reason is
    the identical paragraph repeated emitters × dimensions times inside the bundle that IS
    the paid one-shot prompt. Same normalisation, same reason, as ``measure_catalog``.

    Built from the assembled bundle rather than alongside it, so the catalog cannot claim a
    dimension the lenses never emitted, and cannot miss one they did. **All three emitters
    are read**, not the two that came first: a dimension named only by the campaign-wide scan
    — the case a campaign that swept nothing produces — would otherwise reach a report as a
    bare name with no reason attached, which is the one thing this catalog exists to prevent.

    Args:
        bundle: The assembled bundle, with coverage and divergences already populated.
        profile: The host whose vocabulary this reads.

    Returns:
        ``{dimension: why}`` for every dimension named anywhere in the bundle's confound
        lists. A swept lever's reason is generic by design — a campaign sweeping a lever
        already believes it moves the numbers, which is what makes it a lever. An observed
        mechanism's reason is the engine's own, registered beside its threshold.
    """
    catalog: dict[str, str] = {}
    apparatus_reasons = _apparatus_confound_reasons(profile=profile)
    sweepables = profile.sweepables
    # A resolved surface reaches a confound list only once the knob written into it is accounted for
    # and something is still left over, so its reason says that rather than the generic sentence
    # about another knob: "another lever this campaign swept" would be false of a surface nobody
    # swept, and silent about the one thing a reader needs — that a change no knob names rode in.
    surface_reasons = {
        surface: (
            _UNEXPLAINED_SURFACE_CONFOUNDS
            if claimant.open_family is not None
            else _UNEXPLAINED_WRITTEN_SURFACE_CONFOUNDS
        ).format(family=claimant.name, prose=sweepables.reader_prose(surface))
        for surface, claimant in sweepables.resolution_surfaces.items()
    }
    unverified_reasons = {
        f"{UNVERIFIED_FOLD_PREFIX}{surface}": _UNVERIFIED_FOLD_CONFOUNDS.format(
            knob=claimant.name, surface=surface, prose=sweepables.reader_prose(surface)
        )
        for surface, claimant in sweepables.resolution_surfaces.items()
        if claimant.open_family is None
    }
    emitted = [confound for entry in bundle.coverage for confound in entry.confounded_by]
    emitted += [confound for divergence in bundle.scope_divergences for confound in divergence.confounded_by]
    emitted += bundle.apparatus_confounds
    emitted += [confound for arm in bundle.design.contrasts for confound in arm.mechanism_confounds]
    emitted += [
        confound
        for family in bundle.multiple_comparisons.families
        for comparison in family.comparisons
        for confound in comparison.mechanism_confounds
    ]
    for confound in emitted:
        # Branch on ``kind``, which the scan sets, rather than on whether the name happens to
        # be an apparatus key. A name lookup with a fallback makes two wrong answers
        # expressible — a swept lever colliding with an apparatus name silently takes the
        # apparatus reason, and an apparatus dimension missing from the map silently takes the
        # swept-lever sentence instead of failing. Neither is reachable today; both would be
        # invisible if they became so.
        if confound.kind == "apparatus":
            reason = apparatus_reasons[confound.dimension]
        elif confound.kind == "unverified_fold":
            reason = unverified_reasons[confound.dimension]
        elif confound.kind == "observed_mechanism":
            reason = _OBSERVED_MECHANISMS[confound.dimension.removeprefix(OBSERVED_MECHANISM_PREFIX)].reason
        elif confound.kind == "served_model":
            reason = _SERVED_MODEL_CONFOUNDS
        else:
            reason = surface_reasons.get(confound.dimension, _SWEPT_LEVER_CONFOUNDS)
        catalog[confound.dimension] = (
            f"{UNDECIDED_CONFOUND_PREFIX}; if it did, {reason}" if confound.status == "undecided" else reason
        )
    return catalog


__all__: list[str] = []
