"""The coverage map: the per-lever structural coverage the analysis is built on, and the factor-aliasing scan.

:func:`_coverage_map` builds one row per reportable lever (its levels, repeats, k floor, confounds and mechanism
check), :func:`_declared_crossing` and :func:`_declared_level_coverage` say which declared cells and levels
were run, and :func:`_factor_aliasing` groups the factors that moved in lockstep.
"""

from __future__ import annotations

from itertools import chain, product
from typing import TYPE_CHECKING, Literal


from threetears.evals.analysis.numbers import format_number
from threetears.evals.analysis.reporting import (
    METRIC_COMPOSITE,
    ScoreRecord,
    pooled_composite_basis,
)
from threetears.evals.analysis.stats import clustered_standard_error
from threetears.evals.kernel.declaration import (
    CampaignDesign,
    SweptAxis,
)
from threetears.evals.schema.hashing import canonical_digest
from threetears.evals.kernel.host.profile import HostProfile
from threetears.evals.kernel.metrics import describe_measure

# At runtime for its field set, which tells a result-level measure from a row-level one.
from threetears.evals.schema.models import EvalResult

from threetears.evals.analysis.bundle.caps import (
    _capped,
    _CELL_STATES,
    _MAX_DECLARED_CELLS,
    _MAX_FACTOR_PAIR_PIVOTS,
)

from threetears.evals.analysis.bundle.schema import (
    AliasedFactors,
    DeclaredCellCoverage,
    DeclaredCrossing,
    DeclaredLevelCoverage,
    FactorPairCell,
    FactorPairPivot,
    FactorPairScan,
    LeverCoverageInput,
    RealizedDesign,
)


from threetears.evals.analysis.bundle.config import (
    _CANDIDATE_MODEL_LEVER,
    _lever_value,
    _resolve_config,
    EffectiveLever,
)


from threetears.evals.analysis.bundle.design import (
    _arm_repeats,
    _campaign_arms,
    _CampaignArms,
    _is_control_referenced,
    _lever_cohort,
    _lever_levels,
    _SurfaceFolds,
)

from threetears.evals.analysis.bundle.confounds import _uncontrolled_dimensions

from threetears.evals.analysis.bundle.mechanisms import (
    _mechanism_check,
    _MechanismObservations,
    _observed_mechanism_confounds,
    _served_model_confounds,
    _ServedModels,
)


if TYPE_CHECKING:  # runtime models — TYPE_CHECKING-only to keep the runtime import graph minimal.
    from threetears.evals.schema.models import EvalRun


# A cell needs at least this many repeats to read as measured rather than thin.
# k=1 is noise-dominated on a single template (a standing project learning:
# confirm finalists at k>=3+), so k<3 downgrades a swept lever to ``thin``.
_MEASURED_K_FLOOR = 3


def _within_level_dispersion(
    composite_records: list[ScoreRecord], lever: str, effective_by_run: dict[str, dict[str, EffectiveLever]]
) -> str:
    """A lever's within-level composite spread — the estimate's measurement noise.

    Groups the composite observations by the lever's own level, takes the standard
    error of the mean *within* each level (the core ``stats`` helper — no new
    statistic), and averages across levels. This is the noise the point estimate at
    a level carries — the "is this ±0.10 or ±0.01" signal the analysis wants — and it is
    lever-specific (a different grouping per lever), unlike a spread pooled across
    every observation. The standard error is over test cases
    (``clustered_standard_error``), since a level's repeats of one case are not independent
    draws, and it needs ≥2 cases, so a level with a lone case contributes nothing; if no
    level clears that bar the spread is unestimable and the field reads ``"unscored"``
    (honest, not a fabricated 0).

    Args:
        composite_records: Score records for the composite metric, ``value`` present.
        lever: The lever to group by.
        effective_by_run: Each run's resolved levers, keyed by run id.

    **A ragged pool says so in the text** (#638): where the composites behind the spread were meaned over
    different dimension sets (:func:`~threetears.evals.analysis.reporting.pooled_composite_basis`), the spread
    is partly the difference between those sets, and the text carries the sets beside the number.

    Returns:
        ``"±"`` and the mean within-level SEM in :func:`~threetears.evals.analysis.numbers.format_number`'s spelling,
        followed by the ragged-composite disclosure in parentheses where the pool is ragged, or ``"unscored"``.
    """
    by_level: dict[str, tuple[list[float], list[str]]] = {}
    pooled: list[ScoreRecord] = []
    for record in composite_records:
        if record.value is not None and (level := _lever_value(record, lever, effective_by_run)) is not None:
            values, cases = by_level.setdefault(level, ([], []))
            values.append(record.value)
            cases.append(record.test_case_id)
            pooled.append(record)
    sems = [sem for values, cases in by_level.values() if (sem := clustered_standard_error(values, cases)) is not None]
    if not sems:
        return "unscored"
    basis = pooled_composite_basis(pooled)
    ragged = f" ({basis.disclosure()})" if basis is not None and basis.ragged else ""
    return f"±{format_number(sum(sems) / len(sems))}{ragged}"


def _lever_k_floor(
    lever: str,
    records: list[ScoreRecord],
    k_by_arm: dict[str, int],
    group_of_run: dict[str, str],
    effective_by_run: dict[str, dict[str, EffectiveLever]],
) -> int:
    """Repeat depth available for comparing one lever's levels.

    Comparing levels needs replication at *each* level, so the weakest level binds the
    comparison; within a level, the best-replicated ARM is what a reader can lean on.
    Hence the minimum, across levels, of the maximum arm ``k`` observed at that level — an arm's
    repeats per case (:func:`_arm_repeats`), so two runs of one arm over one case set at k=3 are one
    arm at k=6.

    A campaign-wide ``min(k_runs)`` was the earlier answer, and it coupled every lever to
    the weakest run anywhere in the campaign: one k=1 exploratory run dropped *all* of them
    to ``thin``, including levers swept three-deep at every level, and no later replication
    could lift them because a minimum only falls. The per-level maximum localises the
    penalty to the lever that actually lacks repeats — a model swept once stays thin while
    a fanout swept three-deep at every level reads as measured.

    **This is not monotonic in campaign size, and should not be read as if it were.** A run
    that opens a NEW level still lowers the floor for its own lever, because that level
    genuinely has one repeat behind it — see the sibling test that pins exactly this. What
    the change removes is *spurious* coupling: a run can no longer degrade a lever it says
    nothing about.

    Args:
        lever: A lever name or a declared coordinate name.
        records: The projected score records (they carry the level and the run).
        k_by_arm: Repeats per case per arm group, keyed as :meth:`_CampaignArms.groups` keys them.
        group_of_run: Run id → the arm group it belongs to.
        effective_by_run: Each run's resolved levers, keyed by run id.

    Returns:
        The binding repeat depth, or 0 when no record carries a resolvable run.
    """
    best_at_level: dict[str, int] = {}
    for record in records:
        level = _lever_value(record, lever, effective_by_run)
        if level is None:
            continue
        k = k_by_arm.get(group_of_run.get(record.run_id, ""), 0)
        best_at_level[level] = max(best_at_level.get(level, 0), k)
    return min(best_at_level.values(), default=0)


def _reportable_levers(
    resolved: set[str],
    records: list[ScoreRecord],
    effective_by_run: dict[str, dict[str, EffectiveLever]],
    declared_design: CampaignDesign | None,
    engaged: set[str],
) -> set[str]:
    """Which levers earn a coverage row: the ones that MOVED, plus the ones the campaign declared.

    Reading every declared lever through the host's registry hands this
    lens the whole registry rather than one host's overlay carrier — which is the fix, and which
    also means an always-constant lever now reaches it. A host can have ten: a subject's
    components resolve a level on every run and move only when somebody edits the subject
    between arms. A row apiece would put ten constant lines in every memo a paid generator
    writes, and would invite a reader to sweep an axis whose levels are not contrastable.

    So a lever is reportable when the campaign **engaged** with it, in any of four ways:

    - it **moved** — more than one level across the campaign, read through the same
      :func:`_lever_value` the row's own ``levels`` are;
    - ``declared_design`` **names it as an axis**;
    - the **launch named it**, as an open family's member;
    - **observation recovered it** — the host declared a recovery rule and the role ran.

    The last two are what keep a HELD-FIXED lever reporting, and they are not decoration: a
    campaign that pinned one inner-agent model across every arm, and a single-arm campaign that
    never moved its candidate model, both get an ``unswept`` row rather than silence. "Held at
    one value" and "not a lever in this campaign" are different answers, and the silence is what
    once let a single-arm campaign's configuration name an inner-agent model and nothing else.
    The declared clause is load-bearing for a different reason: the completeness check
    (``generator._reject_incomplete_axis_coverage``) excuses silence about a declared axis only
    through a ``thin``/``unswept`` row, so a declared axis that resolved nothing must still get
    one — ``unswept``, at whatever the records bin to (every record at ``'—'`` where nothing
    resolved the lever, so ``cells=1``, not zero). "Held at one level, and that level is nothing
    anyone recorded" is the honest reading, and the refusal it would otherwise raise is not.

    **What this leaves out** is the one remaining case: a lever nothing but the run record speaks
    to, that never moved. Those are the contestant's own properties — a subject's backstory
    resolves a level on every run and moves only when somebody edits the subject between arms —
    and they are what the campaign held constant rather than measured. They are on NEITHER
    surface: ``_run_summary`` narrows ``config`` to this same set, deliberately, because ten
    content hashes per run each stamped ``overridden`` would tell the design lens every run
    departed from ten things it never touched. They reappear the moment one of them actually
    moves, which is the case worth seeing.

    Args:
        resolved: Every lever any run resolved, plus the candidate model where records disagree.
        records: The projected score records — the levels are read over these.
        effective_by_run: Each run's resolved levers, keyed by run id.
        declared_design: What the campaign SET OUT to sweep, or None when it declared nothing.
        engaged: Levers the launch named or observation recovered, across every run.

    Returns:
        The levers to build rows for.
    """
    declared = {axis.axis_id for axis in declared_design.axes} if declared_design else set()
    moved = {
        lever
        for lever in resolved
        if len({level for record in records if (level := _lever_value(record, lever, effective_by_run)) is not None})
        > 1
    }
    return moved | declared | (engaged & resolved)


def _run_axis_identities(
    axis_id: str, run: EvalRun, results: list[EvalResult], *, profile: HostProfile
) -> set[str] | None:
    """The identities a run's level on one declared axis can be joined to a declared level by, or None if unknown.

    Its variant coordinate where it has one (the engine's own and the host's), else the value the host's registry
    resolves for it, as its canonical digest and, for a string that may itself be a digest, as itself.

    Args:
        axis_id: The declared axis.
        run: The run.
        results: The run's results.
        profile: The host whose vocabulary this reads.

    Returns:
        The identities, or None when the run's level on the axis cannot be established.
    """
    coordinates = {
        **profile.engine_levels(run),
        **(profile.variant_levers(run) if profile.variant_levers is not None else {}),
    }
    if (coordinate := coordinates.get(axis_id)) is not None:
        return {coordinate.content_hash}
    value = profile.sweepables.resolve_levers(run, results).values.get(axis_id)
    if value is None:
        return None
    return {canonical_digest(value), value} if isinstance(value, str) else {canonical_digest(value)}


def _declared_crossing(
    design: CampaignDesign | None,
    runs: list[EvalRun],
    results_by_run: dict[str, list[EvalResult]],
    *,
    profile: HostProfile,
) -> DeclaredCrossing | None:
    """Mark every cell of a design that says which combinations it meant to run: ran, missing, or skipped by design.

    A cell is one declared level of every declared axis. A run sits at a cell when each axis's level joins to
    that cell's level (:func:`_run_axis_identities`). An unrun cell the design left out on purpose
    (:meth:`~threetears.evals.kernel.declaration.CampaignDesign.skipped_by_design`) is ``skipped_by_design``
    and is never a gap; an unrun cell it meant to run is ``not_run``, unless some run's level on an axis cannot
    be established, when it may be sitting there and the cell is ``undetermined``.

    Args:
        design: The campaign's declaration.
        runs: The campaign's resolved runs.
        results_by_run: Each run's results.
        profile: The host whose vocabulary this reads.

    Returns:
        The crossing, or None when there is no design or it says nothing about combinations — today's reading,
        where an unrun combination is neither skipped nor missing.
    """
    if design is None or not design.declares_cells():
        return None
    ran: set[tuple[str, ...]] = set()
    unestablished = False
    for run in runs:
        levels: list[str | None] = []
        for axis in design.axes:
            identities = _run_axis_identities(axis.axis_id, run, results_by_run.get(run.id, []), profile=profile)
            if identities is None:
                unestablished = True
                levels.append(None)
                continue
            matched = [level.content_hash for level in axis.values if level.content_hash in identities]
            levels.append(matched[0] if matched else None)
        if all(level is not None for level in levels):
            ran.add(tuple(level for level in levels if level is not None))
    cells = []
    for combination in product(*(axis.values for axis in design.axes)):
        identity = tuple(level.content_hash for level in combination)
        state: Literal["ran", "not_run", "skipped_by_design", "undetermined"]
        if identity in ran:
            state = "ran"
        elif design.skipped_by_design(identity):
            state = "skipped_by_design"
        else:
            state = "undetermined" if unestablished else "not_run"
        cells.append(
            DeclaredCellCoverage(
                levels={axis.axis_id: level.display for axis, level in zip(design.axes, combination, strict=True)},
                state=state,
            )
        )
    counts = {state: sum(1 for cell in cells if cell.state == state) for state in _CELL_STATES}
    kept, omitted = _capped(
        cells, _MAX_DECLARED_CELLS, weight=lambda cell: {"not_run": 2, "undetermined": 1}.get(cell.state, 0)
    )
    planned = len(cells) - counts["skipped_by_design"]
    sentence = (
        f"The design declares {len(cells)} cell(s) over its {len(design.axes)} axis(es) and meant to run {planned}: "
        f"{counts['ran']} ran"
        + (f", {counts['not_run']} never ran (a gap)" if counts["not_run"] else "")
        + (f", {counts['undetermined']} cannot be established" if counts["undetermined"] else "")
        + (
            f"; {counts['skipped_by_design']} were skipped by design, which is no gap"
            if counts["skipped_by_design"]
            else ""
        )
        + (f"; {omitted} cell(s) are left out of the list, gaps last to go" if omitted else "")
        + "."
    )
    return DeclaredCrossing(
        crossing=design.crossing,
        n_cells=len(cells),
        n_ran=counts["ran"],
        n_not_run=counts["not_run"],
        n_skipped_by_design=counts["skipped_by_design"],
        n_undetermined=counts["undetermined"],
        cells=kept,
        cells_omitted=omitted,
        sentence=sentence,
    )


def _declared_level_coverage(
    axis: SweptAxis,
    runs: list[EvalRun],
    results_by_run: dict[str, list[EvalResult]],
    *,
    profile: HostProfile,
) -> list[DeclaredLevelCoverage]:
    """Mark each level ``axis`` declares as ran, not run, or undetermined, joined on content identity.

    A run's level on the axis is its variant coordinate where it has one (the engine's own and the
    host's), else the value the host's registry resolves for it — joined to a declared level by
    ``content_hash``, the declaration's identity. A resolved value that is itself a digest is
    compared as one. A run whose level cannot be established (no coordinate, nothing resolved)
    blocks a ``not_run`` claim on every level no other run matched: it may be sitting at one.

    Args:
        axis: The declared axis.
        runs: The campaign's resolved runs.
        results_by_run: Each run's results.
        profile: The host whose vocabulary this reads.

    Returns:
        One entry per declared level, in the declaration's order.
    """
    observed: set[str] = set()
    unestablished = False
    for run in runs:
        identities = _run_axis_identities(axis.axis_id, run, results_by_run.get(run.id, []), profile=profile)
        if identities is None:
            unestablished = True
        else:
            observed |= identities
    return [
        DeclaredLevelCoverage(
            display=level.display,
            content_hash=level.content_hash,
            state="ran" if level.content_hash in observed else "undetermined" if unestablished else "not_run",
        )
        for level in axis.values
    ]


def _coverage_map(
    runs: list[EvalRun],
    records: list[ScoreRecord],
    apparatus_levels: dict[str, dict[str, str | None]],
    design: RealizedDesign,
    results_by_run: dict[str, list[EvalResult]],
    declared_design: CampaignDesign | None,
    *,
    folds: _SurfaceFolds,
    observations: _MechanismObservations,
    served: _ServedModels,
    arms: _CampaignArms | None = None,
    profile: HostProfile,
) -> list[LeverCoverageInput]:
    """Build the per-lever structural coverage map (the analysis's spine).

    A **lever** is whatever the host declares as one, resolved per run through
    :func:`_effective_config`. Which of them earns a row is :func:`_reportable_levers`: the ones
    that moved, plus the ones the campaign declared. A DECLARED axis that nothing resolved still
    gets a row reading ``unswept`` rather than no row at all — "held at one value" and "not a
    lever in this campaign" are different answers, and only the row can say which.
    For each, coverage reports how finely it was swept
    (``cells`` = distinct observed levels), a repeat floor (``k``, see
    :func:`_lever_k_floor` — per-lever and per arm, never a campaign-wide minimum),
    the samples informing it (``n`` = distinct results), the within-level composite
    spread (``dispersion``, see :func:`_within_level_dispersion`), and a coarse
    ``status``:

    - ``unswept`` — a single observed level (held fixed / not explored).
    - ``thin`` — swept, but ``k < 3`` or fewer than ~2 samples per cell (a k=1
      point estimate is noise-dominated).
    - ``measured`` — swept with an adequate repeat floor.

    Each lever also carries what did NOT hold still behind it (``confounded_by``). Most
    findings are formed per lever from this map rather than from the divergence lens, so a
    confound list that reached only the divergences left the common path unqualified: the
    generator was handed a lever's coverage with no way to know the campaign had also
    changed template and judge underneath it.

    **Each row states its own cohort in ``cohort_scope``, because the campaign-wide case is
    not simply "no control was designated".** Under a control, a lever some arm moved is read
    over the control plus those arms (:func:`_lever_cohort` says why pooling the rest
    manufactures a confound) — but a lever NO arm moved stays campaign-wide even then, since
    there is no contrast to narrow it to. The two rows are the same
    shape and differ only in what they may be read as, which is why the scope is a field
    rather than something a reader re-derives from the design.

    Deterministic: levers and their levels are sorted, so identical inputs give an
    identical map (and fingerprint).

    Args:
        runs: The campaign's resolved runs (for each run's ``k_runs`` and case set, counted per arm by :func:`_arm_repeats`).
        records: The projected score records (for levels, n, and composite spread).
        apparatus_levels: Dimension → run id → level key, from :func:`_apparatus_levels`.
        design: The derived design — what each lever's comparison is drawn from. Without a
            control every lever is read over the whole campaign, which is the marginal
            comparison this map has always reported.
        results_by_run: Each run's results, keyed by run id — the observation side of
            effective-config resolution.
        declared_design: What the campaign SET OUT to sweep. A declared axis earns a row even
            when nothing resolved it — see :func:`_reportable_levers`.
        folds: The campaign's :class:`_SurfaceFolds`. A resolved surface earns no row of its own
            where its movement across its cohort is its swept members' movement — the member rows
            already report that change, and a second row levelled by content hash would count one
            override as two swept levers (and the same for a fixed knob's surface that moved only
            where the knob did). It keeps its row where the residuals disagree or cannot be read, or a
            fixed knob held at one level saw it differ, because then it carries something no knob's
            row does. A declared axis is never
            dropped this way: the completeness check needs its row whatever it resolved to.
        observations: The campaign's mechanism observations, which each row's ``mechanism`` check and its
            observed-mechanism confounds compare across the row's levels.
        served: Which model answered each result's candidate calls, for the served-model confound.
        arms: :func:`_campaign_arms`'s answer, when the caller already has it; derived otherwise.
        profile: The host whose vocabulary this reads.

    Returns:
        One :class:`LeverCoverageInput` per reportable lever, sorted by lever name.
    """
    if not records:
        return []

    # k is per ARM: the runs repeating one arm pool their repeats into it.
    groups = (arms if arms is not None else _campaign_arms(runs, results_by_run, profile=profile)).groups()
    group_of_run = {run.id: key for key, members in groups.items() for run in members}
    k_by_arm = {key: _arm_repeats(members) for key, members in groups.items()}
    resolved_by_run = {run.id: _resolve_config(run, results_by_run.get(run.id, []), profile=profile) for run in runs}
    effective_by_run = {run_id: flat for run_id, (flat, _engaged) in resolved_by_run.items()}
    # The launch-named and observation-recovered halves of "engaged", carried out of the same pass
    # that resolved the levels rather than re-derived: `overridden` is written both by a family
    # member and by a fixed declaration's own reader, so provenance cannot tell them apart.
    engaged = {name for _flat, names in resolved_by_run.values() for name in names}
    # The lever set comes from the registry, through `_effective_config`, and never from
    # `record.factors`. `factors` is the raw launch-overlay flattening in one host's carrier
    # spelling, so drawing the set from it re-created the second vocabulary that made a
    # declared axis unmatchable to its own coverage row.
    resolved = {lever for config in effective_by_run.values() for lever in config}
    # Observation is the primary source for the candidate model, and the records are its
    # floor: a result carrying no `candidate` usage row recovers nothing, yet every record
    # still knows which model produced it. `resolved` is a set, so the two sources name one
    # coverage row, never two.
    if len({record.model for record in records}) > 1:
        resolved.add(_CANDIDATE_MODEL_LEVER)
    lever_levels = _lever_levels(runs, results_by_run, profile=profile)
    all_run_ids = [run.id for run in runs]
    levers = _reportable_levers(resolved, records, effective_by_run, declared_design, engaged)

    coverage: list[LeverCoverageInput] = []
    declared_axes = {axis.axis_id: axis for axis in declared_design.axes} if declared_design else {}
    declared = set(declared_axes)
    for lever in sorted(levers):
        # Every number below is read over this lever's own cohort, which under a designated
        # control is usually narrower than the campaign — reporting a campaign-wide n or spread
        # beside a control-vs-cell contrast would describe a comparison that was never made.
        # Usually, not always: a lever no cell moved stays campaign-wide, and cohort_scope below
        # is what tells the reader which of the two this row is.
        cohort = set(_lever_cohort(lever, design, all_run_ids))
        if lever not in declared and folds.folds_away(lever, cohort):
            continue
        # A cohort with no records is emitted, not skipped: the lever exists in this campaign,
        # and dropping its row would tell the generator it does not. It comes out at cells=0
        # / n=0 / unswept, which is the honest reading — nothing here measured it.
        cohort_records = [record for record in records if record.run_id in cohort]
        composite_records = [r for r in cohort_records if r.metric == METRIC_COMPOSITE and r.value is not None]
        n_distinct_results = len({record.result_id for record in cohort_records})
        levels = sorted(
            {level for record in cohort_records if (level := _lever_value(record, lever, effective_by_run)) is not None}
        )
        cells = len(levels)
        # The same records the levels were read from, so a mechanism is compared over exactly the
        # levels and cohort every other number on the row describes.
        result_ids_by_level: dict[str, set[str]] = {}
        for record in cohort_records:
            if (level := _lever_value(record, lever, effective_by_run)) is not None:
                result_ids_by_level.setdefault(level, set()).add(record.result_id)
        k = _lever_k_floor(lever, cohort_records, k_by_arm, group_of_run, effective_by_run)
        # A declared axis is asked the authoring gate's own question. A campaign stored before that
        # gate, or past it, can still declare an apparatus or label input, and its row is then
        # `unswept` for a reason no run could change — said here, in the host's words, so the memo
        # can state the cause rather than report a sweep that never happened (#675).
        controllable = profile.controllable(lever) if lever in declared else None
        if cells <= 1:
            status: Literal["measured", "thin", "unswept"] = "unswept"
        elif k < _MEASURED_K_FLOOR or n_distinct_results < 2 * cells:
            status = "thin"
        else:
            status = "measured"
        coverage.append(
            LeverCoverageInput(
                name=lever,
                levels=levels,
                cells=cells,
                k=k,
                n=n_distinct_results,
                dispersion=_within_level_dispersion(composite_records, lever, effective_by_run),
                status=status,
                # Read from the same predicate the cohort was built with, never inferred from
                # its size: a single-lever star has every cell moving the one lever, so the
                # cohort spans every run while still being the contrast the campaign exists to
                # draw. Sizing it would label that row 'campaign' and tell the generator, two
                # paragraphs after "compare each cell to the control", that no contrast exists.
                cohort_scope="control_referenced" if _is_control_referenced(lever, design) else "campaign",
                declared_levels=(
                    _declared_level_coverage(declared_axes[lever], runs, results_by_run, profile=profile)
                    if lever in declared_axes
                    else []
                ),
                cannot_be_an_arm=(
                    controllable.reason if controllable is not None and controllable.state != "covered" else None
                ),
                confounded_by=_uncontrolled_dimensions(
                    lever, sorted(cohort), lever_levels, apparatus_levels, folds=folds, profile=profile
                )
                + _observed_mechanism_confounds(lever, result_ids_by_level, observations, profile=profile)
                + _served_model_confounds(chain.from_iterable(result_ids_by_level.values()), served),
                mechanism=_mechanism_check(
                    acts_on := profile.sweepables.acts_on(lever),
                    result_ids_by_level,
                    observations,
                    # The mechanism measure's declared range, which a shift of every case alike is read on.
                    describe_measure(acts_on, profile.measures).value_range if acts_on is not None else None,
                ),
            )
        )
    return coverage


def _factor_aliasing(
    run_ids: list[str],
    lever_levels: dict[str, dict[str, list[str]]],
    apparatus_levels: dict[str, dict[str, str | None]],
) -> tuple[list[AliasedFactors], FactorPairScan]:
    """Group the factors that moved in lockstep, and check every pair of factors for co-varying.

    A factor is anything that varied across the campaign's runs: a swept lever or the candidate model, as
    :func:`_lever_levels` reads them, and an apparatus dimension every run recorded at two or more levels (one some
    run never recorded is undecided, not a partition, and is named in ``apparatus_confounds``). Each factor
    splits the runs it was read on into groups, one per level. **Factors whose splits are identical are aliased**
    — the same runs, grouped the same way, whatever the levels are called — and are reported as one group.

    A pair **co-varies** when neither can be compared holding the other fixed: across the runs at any one level
    of one, the other takes a single level. Every pair is checked; a co-varying pair whose factors are not in one
    group gets a pivot crossing the two (a group stands in for each of its members, so its mates share one
    pivot), with every combination no run sat at as ``not_run``. Only interactions go unchecked
    (:data:`INTERACTION_ALIASING_UNCHECKED`).

    Args:
        run_ids: The campaign's resolved runs.
        lever_levels: Lever → level → run ids, from :func:`_lever_levels`.
        apparatus_levels: Dimension → run id → level key, from :func:`_apparatus_levels`.

    Returns:
        The lockstep groups (sorted by their first factor), and the pair scan.
    """
    by_factor: dict[str, dict[str, str]] = {}
    for lever, by_level in lever_levels.items():
        by_factor[lever] = {run_id: level for level, members in by_level.items() for run_id in members}
    for dimension, by_run in apparatus_levels.items():
        levels = [by_run.get(run_id) for run_id in run_ids]
        if dimension in by_factor or any(level is None for level in levels) or len(set(levels)) < 2:
            continue
        by_factor[dimension] = {run_id: str(by_run[run_id]) for run_id in run_ids}
    factors = sorted(name for name, by_run in by_factor.items() if len(set(by_run.values())) >= 2)

    def split(name: str) -> frozenset[frozenset[str]]:
        blocks: dict[str, set[str]] = {}
        for run_id, level in by_factor[name].items():
            blocks.setdefault(level, set()).add(run_id)
        return frozenset(frozenset(block) for block in blocks.values())

    by_split: dict[frozenset[frozenset[str]], list[str]] = {}
    for name in factors:
        by_split.setdefault(split(name), []).append(name)
    group_of = {name: members[0] for members in by_split.values() for name in members}
    groups = []
    for blocks, members in sorted(by_split.items(), key=lambda item: item[1][0]):
        if len(members) < 2:
            continue
        n_runs = sum(len(block) for block in blocks)
        across = f"all {n_runs} runs" if n_runs == len(run_ids) else f"the {n_runs} runs that recorded them"
        groups.append(
            AliasedFactors(
                factors=members,
                n_runs=n_runs,
                n_levels=len(blocks),
                sentence=(
                    f"{_listed_names(members)} move together across {across}, splitting them into the same "
                    f"{len(blocks)} groups; no comparison separates them, so a difference across them belongs to all "
                    f"{len(members)} at once."
                ),
            )
        )

    def varies_within(name: str, other: str) -> bool:
        shared = by_factor[name].keys() & by_factor[other].keys()
        seen: dict[str, set[str]] = {}
        for run_id in shared:
            seen.setdefault(by_factor[other][run_id], set()).add(by_factor[name][run_id])
        return any(len(levels) >= 2 for levels in seen.values())

    n_pairs = n_covarying = n_in_groups = 0
    pivots: dict[tuple[str, str], FactorPairPivot] = {}
    for index, row in enumerate(factors):
        for column in factors[index + 1 :]:
            n_pairs += 1
            if varies_within(row, column) and varies_within(column, row):
                continue
            n_covarying += 1
            if group_of[row] == group_of[column]:
                n_in_groups += 1
                continue
            first, second = sorted((group_of[row], group_of[column]))
            key = (first, second)
            if key not in pivots:
                pivots[key] = _factor_pair_pivot(key[0], key[1], by_factor, group_of)
    ordered = [pivots[key] for key in sorted(pivots)]
    kept, omitted = _capped(
        ordered,
        _MAX_FACTOR_PAIR_PIVOTS,
        weight=lambda pivot: sum(1 for cell in pivot.cells if cell.status == "not_run"),
    )
    outside = n_covarying - n_in_groups
    if len(factors) < 2:
        completeness = f"{len(factors)} factor varied, so no pair of factors could co-vary."
    else:
        completeness = (
            f"{n_pairs} factor pair(s) examined over the {len(factors)} factors that varied; {n_covarying} co-vary"
            + (f", {n_in_groups} of them inside a group that moves in lockstep" if n_in_groups else "")
            + (
                f"; the {outside} outside any group are shown in {len(ordered)} pivot(s), a group's members sharing one"
                if outside
                else ""
            )
            + (f", of which {omitted} with the fewest unrun combinations are left out" if omitted else "")
            + "."
        )
    return groups, FactorPairScan(
        factors=factors,
        n_pairs_examined=n_pairs,
        n_covarying=n_covarying,
        n_covarying_in_groups=n_in_groups,
        pivots=kept,
        pivots_omitted=omitted,
        completeness=completeness,
    )


def _factor_pair_pivot(
    row: str,
    column: str,
    by_factor: dict[str, dict[str, str]],
    group_of: dict[str, str],
) -> FactorPairPivot:
    """Cross two factors' observed levels over the runs both were read on, counting the runs at each combination."""
    shared = sorted(by_factor[row].keys() & by_factor[column].keys())
    counts: dict[tuple[str, str], int] = {}
    for run_id in shared:
        combination = (by_factor[row][run_id], by_factor[column][run_id])
        counts[combination] = counts.get(combination, 0) + 1
    rows = sorted({by_factor[row][run_id] for run_id in shared})
    columns = sorted({by_factor[column][run_id] for run_id in shared})
    mates = {name: [other for other in group_of if group_of[other] == name and other != name] for name in (row, column)}
    return FactorPairPivot(
        row_factor=row,
        column_factor=column,
        row_aliases=sorted(mates[row]),
        column_aliases=sorted(mates[column]),
        cells=[
            FactorPairCell(
                row_level=row_level,
                column_level=column_level,
                n_runs=counts.get((row_level, column_level), 0),
                status="ran" if (row_level, column_level) in counts else "not_run",
            )
            for row_level in rows
            for column_level in columns
        ],
    )


def _listed_names(names: list[str]) -> str:
    """``a``, ``a and b``, ``a, b and c``."""
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + f" and {names[-1]}"


__all__: list[str] = []
