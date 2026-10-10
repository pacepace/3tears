"""Campaign design and cohorts: the realized design, its arms, and the levels of each lever.

:func:`_campaign_design` derives the design from the campaign's arms and the declared control variant,
:func:`_lever_cohort` and :func:`_lever_levels` group runs by one lever's level, and :class:`_SurfaceFolds`
answers whether a resolved surface moved on its own. :func:`_name_arms` names the arms a reader sees.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Collection
from functools import partial
from typing import TYPE_CHECKING, Any, Literal, NamedTuple


from threetears.evals.analysis.cells import Observation
from threetears.evals.analysis.reporting import lever_level
from threetears.evals.kernel.campaign import VariantIndexEntry
from threetears.evals.kernel.declaration import CampaignDesign
from threetears.evals.schema.hashing import canonical_json
from threetears.evals.kernel.host.profile import HostProfile
from threetears.evals.schema.values import SweepableValue

# At runtime for its field set, which tells a result-level measure from a row-level one.
from threetears.evals.schema.models import EvalResult


from threetears.evals.analysis.bundle.schema import (
    DesignArm,
    RealizedDesign,
)


from threetears.evals.analysis.bundle.config import (
    _CANDIDATE_MODEL_LEVER,
    _effective_config,
    _effective_values,
    _INHERITED_DEFAULT_LEVEL,
)

from threetears.evals.analysis.bundle.observations import variant_key_of_run


if TYPE_CHECKING:  # runtime models — TYPE_CHECKING-only to keep the runtime import graph minimal.
    from threetears.evals.schema.models import EvalRun


#: How a world dimension is named where it sits beside the sweepable apparatus dimensions.
#:
#: Prefixed rather than carried under its bare name, so the two registries' entries sit in disjoint
#: namespaces of one dimension map. ``HostProfile`` refuses a name registered in both, and the
#: prefix keeps the map's correctness from resting on that refusal: bare names that met would
#: collide and one fact would overwrite the other, which is the wrong-merge direction nothing
#: downstream can undo. The two kinds of entry say different things: a sweepable says which level
#: a run swept an input to, a world dimension says whether a run seeded it at all.
_WORLD_DIMENSION_PREFIX = "world:"


def world_dimension_key(dimension: str) -> str:
    """Name ``dimension`` for the apparatus maps, where sweepables and world dimensions share a namespace.

    Args:
        dimension: A world dimension name, as its host declared it.

    Returns:
        The prefixed key. See :data:`_WORLD_DIMENSION_PREFIX`.
    """
    return f"{_WORLD_DIMENSION_PREFIX}{dimension}"


#: What the runs showed about a resolved surface's movement, over one cohort.
#:
#: ``explained`` — the surface moved only because the knob written into it did, so it is the same
#: change seen twice: for an open family, every run's residual (the surface with the swept members
#: taken back out) agrees; for a fixed lever, the surface held one level within each of the lever's
#: levels, and some level was held by two or more arms, so the runs could have shown otherwise.
#: ``unverified`` — a fixed lever's fold that the runs could NOT have refuted: the surface held one
#: level within each of the lever's, but every level was held by one arm only, and every run of an
#: arm resolves the arm's one surface (the surface is in the variant key), so the dependency holds by
#: construction. The surface is still folded — the knob names the arm — and every lens that folds it
#: marks the comparison with an ``unverified_fold`` confound, because a check that could not run is
#: not a pass. ``unexplained`` — something besides the knob changed it: the residuals disagree, two
#: runs that held the lever at one level carried different surfaces, or a run the lever does not apply
#: to (another kind's) carried a surface no run of the lever's own kind in the cohort carries. ``undetermined`` — some run's
#: residual or surface could not be read, so neither can be shown; the surface is kept, because
#: folding it would be an inference.
SurfaceFold = Literal["explained", "unverified", "unexplained", "undetermined"]

#: The verdicts under which a surface is folded into its knob: one checked, one that could not be.
_FOLDED: frozenset[SurfaceFold] = frozenset({"explained", "unverified"})


class _SurfaceFolds:
    """The ONE answer to "did this resolved surface move on its own, across these runs".

    A host may register a knob AND the surface it is merged into as levers
    (:attr:`~threetears.evals.kernel.host.sweepables.Sweepable.resolves_into`): an open family's
    members and the tool configuration they are written into, or a kind's ``reasoning_effort``
    overlay and the resolved model parameters it is written into
    (:class:`~threetears.evals.kernel.host.kinds.ResolvesInto`). One turn of the knob then reaches
    every lens twice — as the knob and as the surface's content hash — and a lens that counted both
    reported a one-knob arm as ``multi_factor`` and each lever as confounded by the other. Every lens
    that decides what a comparison moved or what confounds it asks this object, over the cohort it is
    comparing, so two lenses over one cohort cannot come to different answers about one surface. Lenses
    over different cohorts can: a knob's coverage row pools every arm that moved the knob, while a design
    contrast reads only the arms its own departures cover, so each answer is about the runs it names.

    **Two rules, one per kind of knob, because only one of them has anything to take out.** An open
    family's members are names a host can remove from its surface, so the family's own residual
    reader answers. A fixed lever is one value with nothing to remove, and the surface without it is
    not something the engine could ask for — so the runs answer instead, by functional dependency:
    the surface folds where every level of the lever carries one level of the surface across the
    cohort, which is what "it moved only where the knob did" means when nothing else can be read.
    **What that cannot see:** the surface is in the variant key, so every run of one arm resolves
    one surface, and repeats of an arm can never disagree with it. A cohort with one ARM per level of
    the lever therefore satisfies the rule by construction, and a second change that rode in exactly
    where the knob changed folds with it. Only a level of the lever held by two or more arms can show
    anything else wrote into the surface — which is exactly the shape a sweep that changed something
    besides the knob produces, and the shape the rule keeps as a confound. A fold no such level
    tested is ``unverified``: still folded, so the knob names the arm, and marked on every comparison
    that folds it (:data:`UNVERIFIED_FOLD_PREFIX`), so it is never read as a checked non-confound.

    **A fixed lever's level is the level its variant coordinate carries**
    (:meth:`~threetears.evals.kernel.host.profile.HostProfile.engine_levels` and the host's
    variant-lever reader), so the fold and the variant key cannot disagree about whether two runs sat
    at one level — in particular, a run of another kind sits at that kind's "not this kind" level,
    never at a ``None`` a run of the lever's own kind can also hold. The lever does not apply to such
    a run and cannot have written its surface, so where such a run carries a surface no run of the
    lever's own kind in the cohort carries, the kind's change moved it and nothing is folded.

    **Per cohort, never campaign-wide**, because the answer depends on which runs are compared. A
    surface can be explained across the whole campaign — every member any run named taken out — and
    unexplained inside a contrast that swept only one of them, where a second key moved with nothing
    naming it. That second key is exactly what the rule exists to surface.

    Built once per assembly and memoised per ``(run, removed members)``, because a residual read
    is host code over the run's own payload and every lever × cohort asks.
    """

    def __init__(
        self, runs: list[EvalRun], results_by_run: dict[str, list[EvalResult]], *, profile: HostProfile
    ) -> None:
        """Capture the campaign's runs, which members each family resolved on each, and each fixed knob's level.

        Args:
            runs: The campaign's resolved runs.
            results_by_run: Each run's results, keyed by run id.
            profile: The host whose vocabulary this reads.
        """
        self._registry = profile.sweepables
        self._surfaces = self._registry.resolution_surfaces
        self._runs = {run.id: run for run in runs}
        self._results = results_by_run
        resolutions = (
            {run.id: self._registry.resolve_levers(run, results_by_run.get(run.id, [])) for run in runs}
            if self._surfaces
            else {}
        )
        self._members: dict[str, dict[str, frozenset[str]]] = {
            run_id: dict(resolution.members_by_family) for run_id, resolution in resolutions.items()
        }
        self._member_values: dict[str, dict[str, Any]] = {
            run_id: {member: resolution.values.get(member) for member in resolution.overlaid}
            for run_id, resolution in resolutions.items()
        }
        # A fixed knob's level, as the variant key carries it, and its surface's raw value, both as
        # comparable keys. The knob is read off the variant coordinate so a run of another kind sits at
        # that kind's own level, apart from any value a run of the knob's kind holds. The surface is read
        # off the resolution, because its ``None`` is a run that did not record it, and that must stay
        # ``None`` so it can only ever read as "cannot say".
        claimants = {
            surface: claimant.name for surface, claimant in self._surfaces.items() if claimant.open_family is None
        }
        self._fixed_levels: dict[str, dict[str, str | None]] = {}
        self._inapplicable: dict[str, frozenset[str]] = {}
        for run in runs if claimants else ():
            coordinates = {
                **profile.engine_levels(run),
                **(profile.variant_levers(run) if profile.variant_levers is not None else {}),
            }
            values = resolutions[run.id].values
            levels: dict[str, str | None] = {}
            inapplicable: set[str] = set()
            for surface, knob in claimants.items():
                raw = values.get(surface)
                levels[surface] = None if raw is None else canonical_json(raw)
                coordinate = coordinates.get(knob)
                if coordinate is None:
                    levels[knob] = canonical_json(values.get(knob))
                else:
                    levels[knob] = coordinate.content_hash
                    if coordinate.not_of_kind is not None:
                        inapplicable.add(knob)
            self._fixed_levels[run.id] = levels
            self._inapplicable[run.id] = frozenset(inapplicable)
        # Which arm each run measured: a fixed knob's fold is testable only where one of its levels
        # was held by two or more arms. A run no observation keys stands for an arm of its own.
        self._arm_of: dict[str, str] = {
            run.id: variant_key_of_run(results_by_run.get(run.id, [])) or f"run:{run.id}" for run in runs
        }
        self._residuals: dict[tuple[str, str, frozenset[str]], str | None] = {}

    @property
    def surfaces(self) -> frozenset[str]:
        """Every lever name a knob resolves into — the only names this object ever folds."""
        return frozenset(self._surfaces)

    def fold(self, surface: str, cohort_run_ids: Collection[str]) -> SurfaceFold:
        """Decide whether ``surface`` moved on its own across a cohort.

        Args:
            surface: A name in :attr:`surfaces`.
            cohort_run_ids: The runs under comparison.

        Returns:
            The verdict — see :data:`SurfaceFold`.
        """
        claimant = self._surfaces[surface]
        cohort = [run_id for run_id in dict.fromkeys(cohort_run_ids) if run_id in self._runs]
        if claimant.open_family is None:
            return self._fold_by_level(surface, claimant.name, cohort)
        swept = frozenset().union(*(self._members.get(run_id, {}).get(claimant.name, frozenset()) for run_id in cohort))
        residuals = [self._residual(surface, run_id, swept) for run_id in cohort]
        if any(residual is None for residual in residuals):
            return "undetermined"
        return "explained" if len(set(residuals)) <= 1 else "unexplained"

    def folds_away(self, lever: str, cohort_run_ids: Collection[str]) -> bool:
        """True when ``lever`` is a resolved surface folded into its knob across the cohort.

        The question every call site actually asks, so none of them re-derives it from
        :meth:`fold` with a comparison that could drift. ``unverified`` folds as well as
        ``explained``; the comparison is then marked by :func:`_uncontrolled_dimensions`.

        Args:
            lever: Any lever name.
            cohort_run_ids: The runs under comparison.

        Returns:
            True only for a surface the cohort shows moved with its knob alone, checked or not; every
            other lever is False.
        """
        return lever in self._surfaces and self.fold(lever, cohort_run_ids) in _FOLDED

    def folds_away_in_contrast(
        self, lever: str, pair: Collection[str], contrast_cohort: Callable[[str], Collection[str]]
    ) -> bool:
        """:meth:`folds_away` for one contrast against the control, over the cohort that can decide it.

        An open family's residual is a check two runs can make, so a contrast asks it over its own
        pair. A fixed knob's fold is not: over two runs it reduces to "did the knob also move", which
        can never come out ``unexplained``, so the design would fold a surface every other lens calls a
        varying confound. It is decided over the contrast's cohort instead — the control and every
        contrast whose departures, the surface aside, fall within this one's — where a second arm at
        one of the knob's levels can show the surface moving on its own, and an arm that moved
        something else cannot make this contrast's knob answer for it.

        Args:
            lever: Any lever name.
            pair: The control's run and the contrast's.
            contrast_cohort: The contrast's cohort for a surface, asked only for a fixed knob's.

        Returns:
            Whether the contrast's movement of ``lever`` is its knob's, seen a second time.
        """
        if lever not in self._surfaces:
            return False
        fixed = self._surfaces[lever].open_family is None
        return self.folds_away(lever, contrast_cohort(lever) if fixed else pair)

    def swept_members(self, surface: str, run_ids: Collection[str]) -> dict[str, Any]:
        """The members of ``surface``'s family these runs overlaid, with the value each named.

        What an arm is named by once its surface folds away. The runs are one arm's: they share a
        variant key, so they resolved one surface, and a member one of them names while another
        inherits it resolved to the same value in both — which is why the union is the arm's, and
        why two of them cannot name one member at two values.

        A surface a FIXED lever is written into has no members: the lever names the arm under its
        own name, as the coordinate it carries in the arm's ``levers`` (registration refuses one
        that carries none), so nothing is added beside it.

        Args:
            surface: A name in :attr:`surfaces`.
            run_ids: The runs that carried one arm.

        Returns:
            Member name → the value named, sorted by name; empty when none of them overlaid a member,
            and for a surface a fixed lever is written into.
        """
        claimant = self._surfaces[surface]
        if claimant.open_family is None:
            return {}
        named: dict[str, Any] = {}
        for run_id in dict.fromkeys(run_ids):
            values = self._member_values.get(run_id, {})
            for member in self._members.get(run_id, {}).get(claimant.name, frozenset()):
                named[member] = values.get(member)
        return dict(sorted(named.items()))

    def _fold_by_level(self, surface: str, lever: str, cohort: list[str]) -> SurfaceFold:
        """Whether ``surface`` held one level within each of ``lever``'s levels across the cohort.

        Args:
            surface: The surface a fixed lever is written into.
            lever: That lever.
            cohort: The runs under comparison, deduplicated, each one this object holds.

        Returns:
            ``undetermined`` when some run did not record the surface; ``unexplained`` when two runs at
            one of the lever's levels carried different surfaces, or a run the lever does not apply to
            carried a surface no run of its own kind here carries; otherwise ``explained`` when some level of the
            lever was held by two or more arms, and ``unverified`` when none was.
        """
        levels = [self._fixed_levels.get(run_id, {}) for run_id in cohort]
        if any(level.get(surface) is None for level in levels):
            return "undetermined"
        own_kind = {
            level.get(surface)
            for run_id, level in zip(cohort, levels, strict=True)
            if lever not in self._inapplicable.get(run_id, frozenset())
        }
        if any(
            level.get(surface) not in own_kind
            for run_id, level in zip(cohort, levels, strict=True)
            if lever in self._inapplicable.get(run_id, frozenset())
        ):
            # A run of another kind carries the knob's "not this kind" level, so the knob did not write its
            # surface. Where that surface is one no run of the knob's own kind carries here, the kind's
            # change moved it, and the knob cannot answer for it.
            return "unexplained"
        surface_at: dict[str | None, str | None] = {}
        arms_at: dict[str | None, set[str]] = defaultdict(set)
        for run_id, level in zip(cohort, levels, strict=True):
            if surface_at.setdefault(level.get(lever), level.get(surface)) != level.get(surface):
                return "unexplained"
            arms_at[level.get(lever)].add(self._arm_of.get(run_id, f"run:{run_id}"))
        return "explained" if any(len(arms) > 1 for arms in arms_at.values()) else "unverified"

    def _residual(self, surface: str, run_id: str, removed: frozenset[str]) -> str | None:
        """One run's residual, as a comparable key, memoised.

        Args:
            surface: The surface to read.
            run_id: The run to read it off.
            removed: The family members to take out.

        Returns:
            The residual's canonical JSON, or None when the run does not carry the surface.
        """
        key = (surface, run_id, removed)
        if key not in self._residuals:
            content = self._registry.read_residual(surface, self._runs[run_id], self._results.get(run_id, []), removed)
            self._residuals[key] = None if content is None else canonical_json(content)
        return self._residuals[key]


class _CampaignArms(NamedTuple):
    """The campaign's runs grouped by the arm each measured — the ONE grouping every lens reads.

    A run carries exactly one arm (assembly refuses a member carrying several models), so a run's
    variant key is a property of the whole run and the lenses group whole runs by it.

    Attributes:
        keyed: Variant key → its runs, arms in order of their first run and runs in bundle order.
        unplaced: Runs none of whose observations resolved a variant key, in bundle order.
    """

    keyed: dict[str, list[EvalRun]]
    unplaced: list[EvalRun]

    def groups(self) -> dict[str, list[EvalRun]]:
        """Every arm, plus each unplaced run as a group of its own under ``run:<id>``.

        An unplaced run cannot pool with any other: treating two of them as one arm would invent a
        repeat nobody ran.

        Returns:
            Group key → its runs.
        """
        return {**self.keyed, **{f"run:{run.id}": [run] for run in self.unplaced}}


def _campaign_arms(
    runs: list[EvalRun], results_by_run: dict[str, list[EvalResult]], *, profile: HostProfile
) -> _CampaignArms:
    """Group the campaign's runs by the arm each measured.

    Args:
        runs: The campaign's resolved runs, in bundle order.
        results_by_run: Each run's results, keyed by run id — where a variant key resolves.
        profile: The host whose vocabulary this reads.

    Returns:
        The grouping.
    """
    keyed: dict[str, list[EvalRun]] = {}
    unplaced: list[EvalRun] = []
    for run in runs:
        key = variant_key_of_run(results_by_run.get(run.id, []))
        if key is not None:
            keyed.setdefault(key, []).append(run)
        else:
            unplaced.append(run)
    return _CampaignArms(keyed=keyed, unplaced=unplaced)


def _arm_repeats(members: list[EvalRun]) -> int:
    """How many times an arm's runs repeated each of its cases, at the case repeated least.

    Per CASE, never a sum over runs: two runs of one arm over the same cases are that arm at twice
    the repeats, while two over different cases are each case once — summing their ``k_runs`` would
    claim replication that never happened. A run that recorded no case set is its own unnamed set,
    which no other run can be shown to share.

    Read off each run's planned ``k_runs`` and case set, as the coverage map's ``k`` always has been;
    a run that delivered fewer results than it planned is disclosed by ``short_runs``, not here.

    Args:
        members: The arm's runs.

    Returns:
        The minimum, over every case any member ran, of the ``k_runs`` summed across the members that ran it;
        0 for no members.
    """
    per_case: dict[str, int] = defaultdict(int)
    for run in members:
        for case in run.test_case_ids or [f"unrecorded:{run.id}"]:
            per_case[case] += run.k_runs
    return min(per_case.values(), default=0)


def _campaign_design(
    runs: list[EvalRun],
    control_variant: str | None,
    *,
    results_by_run: dict[str, list[EvalResult]],
    arms: _CampaignArms | None = None,
    archived_control_variants: Collection[str] = (),
    has_unresolved_members: bool = False,
    folds: _SurfaceFolds | None = None,
    profile: HostProfile,
) -> RealizedDesign:
    """Derive the campaign's design from its arms and the DECLARED control variant.

    An arm's moved levers are the keys where its EFFECTIVE configuration differs from the control
    arm's, plus ``model`` when the candidate differs — the same notion of "lever" the rest of this
    module uses, read against one reference arm instead of pooled across all of them. Effective,
    not merely declared: comparing launch overlays alone reports a lever as moved whenever one arm
    names a value the other inherits, which is the normal shape of a sweep against an un-overridden
    control.

    **Arms, not runs.** Every run of one arm carries the same configuration, so an arm is read off
    its first run and its other runs are its repeats. A re-run of the control therefore adds to the
    control's ``k`` rather than appearing as a contrast that moved nothing.

    Args:
        runs: The campaign's resolved runs.
        control_variant: The declared control variant key, or None.
        results_by_run: Each run's results, keyed by run id — the observation side of
            effective-config resolution, and where a variant key resolves.
        arms: :func:`_campaign_arms`'s answer, when the caller already has it; derived otherwise.
        archived_control_variants: The variant keys carried by member runs an operator archived.
            Supplied by the caller because archived runs are held out of every lens and their
            results are not otherwise read; it is what tells a deliberate exclusion apart from a
            variant nobody ever ran.
        has_unresolved_members: Whether any member run failed to load at all. Such a run's
            observations cannot be read, so whether it carried the control is unknowable — and
            reporting that as "never run" is the conflation the ``unresolved`` arm exists to
            prevent. A membership outliving its run is a recurring state in a real store, not a hypothetical.
        folds: The campaign's :class:`_SurfaceFolds`, shared with the other lenses when the caller
            has one; built from ``runs`` and ``results_by_run`` otherwise. A resolved surface whose
            movement between the control and a contrast is its swept members' is not a second moved
            lever — see :class:`_SurfaceFolds`.
        profile: The host whose vocabulary this reads.

    Returns:
        The design. ``shape`` is ``undesignated`` whenever no control resolved. With no contrasts
        it is vacuously ``one_factor_at_a_time`` — nothing is being compared, and the empty
        ``contrasts`` list is what a reader acts on, not the shape word.
    """
    grouping = arms if arms is not None else _campaign_arms(runs, results_by_run, profile=profile)
    arms_by_key = grouping.keyed
    unplaced = [run.id for run in grouping.unplaced]

    def arm(key: str, moved: dict[str, str] | None = None) -> DesignArm:
        members = arms_by_key[key]
        return DesignArm(
            variant_key=key, run_ids=[run.id for run in members], k=_arm_repeats(members), moved=moved or {}
        )

    if not control_variant or control_variant not in arms_by_key:
        excluded: Literal["archived", "unresolved", "unobserved"] | None = None
        if control_variant:
            if control_variant in set(archived_control_variants):
                # Certain, and the most actionable: a carrier exists and somebody removed it.
                excluded = "archived"
            elif has_unresolved_members:
                # NOT certain, and that is the answer. An unloadable member's observations
                # cannot be keyed, so "nothing carries it" is unproven — claiming `unobserved`
                # here sends an operator to re-run an experiment that may already have run.
                excluded = "unresolved"
            else:
                excluded = "unobserved"
        return RealizedDesign(control_excluded=excluded, unplaced_run_ids=unplaced)

    folds = folds if folds is not None else _SurfaceFolds(runs, results_by_run, profile=profile)
    control = arms_by_key[control_variant][0]
    control_overlays = _effective_values(control, results_by_run.get(control.id, []), profile=profile)
    control_model = control.candidate_model
    # Every contrast's departures from the control BEFORE any surface is folded: a fixed knob's fold is
    # decided over the contrasts whose departures fall within this one's, so they are needed for all of
    # them first. One run stands for each arm: its runs share a variant key, so they share a configuration.
    #
    # KNOWN LIMIT: this maps both "the lever did not apply" and "it applied but resolved
    # ambiguously" to the inherited-default level, so an ambiguous control reads as a
    # moved lever. That surfaces as a claim a reader can check against
    # `RunSummary.config_provenance`, rather than as a silently dropped comparison, which
    # is why it is accepted here. `_lever_levels` keeps the two apart because a cohort a
    # run cannot support is worse than a contrast it cannot make.
    #
    # The candidate model is read off the run's declared model rather than the effective
    # resolution, which reports a run whose candidate role left no usage row as carrying no
    # model at all — and that absence would read as the arm having moved its model to the
    # inherited default.
    departures: dict[str, dict[str, str]] = {}
    for key, members in arms_by_key.items():
        if key == control_variant:
            continue
        first = members[0]
        overlays = _effective_values(first, results_by_run.get(first.id, []), profile=profile)
        departed = {
            lever: overlays.get(lever) or _INHERITED_DEFAULT_LEVEL
            for lever in sorted((set(overlays) | set(control_overlays)) - {_CANDIDATE_MODEL_LEVER})
            if (overlays.get(lever) or _INHERITED_DEFAULT_LEVEL)
            != (control_overlays.get(lever) or _INHERITED_DEFAULT_LEVEL)
        }
        if first.candidate_model != control_model:
            departed[_CANDIDATE_MODEL_LEVER] = first.candidate_model
        departures[key] = departed
    control_runs = [run.id for run in arms_by_key[control_variant]]

    def contrast_cohort(surface: str, key: str) -> list[str]:
        """The runs a fixed knob's fold on ``surface`` is decided over for the contrast ``key``.

        The control and every contrast whose departures, ``surface`` aside, fall within this one's —
        this contrast itself, a second arm at one of its knob's levels, an arm that moved less. Those
        are the arms that can show the surface moving apart from the knob WITHIN this comparison. An
        arm that moved something this contrast did not (another lever, another kind) is a different
        comparison, and its surface moving would be blamed on this contrast's knob otherwise.
        """
        own = set(departures[key]) - {surface}
        return control_runs + [
            run.id
            for other, departed in departures.items()
            if set(departed) - {surface} <= own
            for run in arms_by_key[other]
        ]

    contrasts: list[DesignArm] = []
    for key, departed in departures.items():
        # A resolved surface is left out of `moved` only where it moved with its knob alone — the
        # pair's residuals agree, or, for a fixed knob, the contrasts that moved nothing beyond this
        # one show the surface held one level within each of the knob's (a pair alone cannot show
        # that: see `folds_away_in_contrast`). Then its new hash is the knob it was written from,
        # counted a second time, and keeping it would make every one-knob arm `multi_factor`. Where
        # it moved otherwise it stays, because the arm really did move something no swept knob names.
        pair = (control.id, arms_by_key[key][0].id)
        moved = {
            lever: level
            for lever, level in departed.items()
            if not folds.folds_away_in_contrast(lever, pair, partial(contrast_cohort, key=key))
        }
        contrasts.append(arm(key, moved))

    shape: Literal["one_factor_at_a_time", "multi_factor"] = (
        "one_factor_at_a_time" if all(len(c.moved) <= 1 for c in contrasts) else "multi_factor"
    )
    return RealizedDesign(control_arm=arm(control_variant), contrasts=contrasts, unplaced_run_ids=unplaced, shape=shape)


def _is_control_referenced(lever: str, design: RealizedDesign) -> bool:
    """True when the design makes this lever a contrast against the control.

    The single answer to "is this row a contrast or a marginal comparison", read by both the
    cohort builder and the label that describes what it built. Two derivations would be free
    to disagree, and one way to get there is tempting and wrong: inferring the answer from
    whether the cohort came out smaller than the campaign. Those coincide only by accident —
    a single-lever star, where every arm moved the one lever, has a cohort spanning every run
    AND is exactly the control-referenced contrast the campaign exists to draw.

    Args:
        lever: The lever under comparison.
        design: The derived design.

    Returns:
        True when a control resolved and at least one arm moved this lever off it.
    """
    return design.control_arm is not None and any(lever in c.moved for c in design.contrasts)


def _lever_cohort(lever: str, design: RealizedDesign, all_run_ids: list[str]) -> list[str]:
    """The runs a comparison on ``lever`` is actually drawn from.

    Without a control every run is in every lever's cohort, because nothing says which runs
    were meant to be read together — that is the marginal comparison the coverage map has
    always reported, and it is honest about being one.

    With a control, an arm that moved a DIFFERENT lever belongs to a different contrast and is
    not evidence about this one. Pooling it in is what invents a confound: the comparison's side
    at the control's level of this lever then also contains that arm's own moved lever at two
    values, and the scan correctly reports a difference the comparison never had.

    **A lever no arm moved is not narrowed at all**: every arm holds it at the control's level, so
    there is no contrast to narrow to, and the whole campaign is the honest cohort — which is what
    it was before any control was designated.

    Args:
        lever: The lever under comparison.
        design: The derived design.
        all_run_ids: Every resolved run id, in bundle order.

    Returns:
        The cohort's run ids in ``all_run_ids`` order: every run of the control arm, of each arm that
        moved this lever, and of any arm that moved no lever at all. A run that measured no keyable
        arm belongs to no contrast and is left out.
    """
    if not _is_control_referenced(lever, design) or design.control_arm is None:
        return all_run_ids
    keep = set(design.control_arm.run_ids)
    for contrast in design.contrasts:
        if lever in contrast.moved or not contrast.moved:
            keep |= set(contrast.run_ids)
    return [run_id for run_id in all_run_ids if run_id in keep]


def _lever_levels(
    runs: list[EvalRun], results_by_run: dict[str, list[EvalResult]], *, profile: HostProfile
) -> dict[str, dict[str, list[str]]]:
    """Map each swept lever to its levels, and each level to the runs that sat at it.

    A lever is any axis a run's EFFECTIVE configuration names — which is whatever the
    host declares, under the host's own names — plus the candidate ``model`` when it actually
    varied. The same set the coverage map draws from, before that map narrows it to the
    levers the campaign engaged with (:func:`_reportable_levers`); this lens needs the wider set,
    because a confound is a dimension that moved whether or not anyone was comparing on it.

    **Effective, not declared — but absence still means something, and the two cases are
    different.** For a lever whose value is RECOVERABLE (see :func:`_observed_model_levers`),
    binning an un-named run at ``'—'`` reads absence as a level of its own and splits one
    cohort in two whenever a run inherits the very value another run names — the ordinary
    shape of a sweep against an un-overridden control. Those runs are pooled on the value
    that ran.

    For a lever that is NOT recoverable — a round budget, a token budget, anything no
    record pins — ``'—'`` is retained and is a real cohort: "ran at the subject's own
    setting", which every un-overriding run shares. Dropping those runs instead would delete
    the comparison, leaving a two-arm sweep with one level and no divergence to find.

    Only a lever that APPLIED and whose value is genuinely ambiguous (two models observed
    under one role) leaves a run out: there, no level can be claimed without inventing one.

    Levers observed at a single level are dropped: there is nothing to compare, so they can
    produce no divergence.

    **The candidate model is binned on the run's DECLARED model, not on its effective value**,
    which is the one place this lens deliberately parts from the resolution above: a run whose
    candidate role left no usage row resolves no model, and dropping it would delete a level the
    run plainly ran. A run carries exactly one model, so binning whole runs bins whole arms.

    Args:
        runs: The campaign's resolved runs.
        results_by_run: Each run's results, keyed by run id — the observation side of
            effective-config resolution.
        profile: The host whose vocabulary this reads.

    Returns:
        ``{lever: {level: [run_id, ...]}}`` for every lever with at least two levels.
    """
    levels: dict[str, dict[str, list[str]]] = {}
    configs = {run.id: _effective_config(run, results_by_run.get(run.id, []), profile=profile) for run in runs}
    lever_names = {key for flat in configs.values() for key in flat}
    # Observation is not the only way the model becomes a lever: a campaign whose results
    # carry no `candidate` usage row recovers nothing, yet the declared sets still differ.
    # `lever_names` is a set, so the two sources name one lever, never two.
    if len({run.candidate_model for run in runs}) > 1:
        lever_names.add(_CANDIDATE_MODEL_LEVER)
    for lever in sorted(lever_names):
        by_level: dict[str, list[str]] = {}
        for run in runs:
            if lever == _CANDIDATE_MODEL_LEVER:
                level: str = run.candidate_model
            else:
                effective = configs[run.id].get(lever)
                if effective is None:
                    # The lever does not apply to this run: it named no override and nothing
                    # recovers a value. That is the "ran at the subject's own setting" cohort,
                    # which every un-overriding run shares — dropping them would delete the
                    # contrast a sweep against an un-overridden control exists to make.
                    level = _INHERITED_DEFAULT_LEVEL
                elif effective.value is None:
                    # Applied, but ambiguous. No level can be claimed without inventing one.
                    continue
                else:
                    level = effective.value
            by_level.setdefault(level, []).append(run.id)
        if len(by_level) > 1:
            levels[lever] = by_level
    return levels


def _apparatus_levels(
    runs: list[EvalRun], results_by_run: dict[str, list[EvalResult]], *, profile: HostProfile
) -> dict[str, dict[str, str | None]]:
    """Read each run's value for every apparatus dimension, as comparable level keys.

    The apparatus is everything a campaign is *not* tuning — the template, the judge, the
    simulated user, the cassette corpus, the subject behind it all. A swept lever moving is
    the experiment; the apparatus moving is the experiment quietly becoming a different one,
    which is why these are scanned separately from the overlays and reported distinguishably.

    Values are canonicalised to strings because the raw ones are not all hashable (two of
    these are sets) and because two structurally identical values must key the same.

    Args:
        runs: The campaign's resolved runs.
        results_by_run: Each run's results, for the dimensions observed per result.
        profile: The host whose vocabulary this reads.

    Returns:
        ``{dimension: {run_id: level_key}}`` over the host's apparatus sweepables and — under
        :func:`world_dimension_key` — its world dimensions, whose level is the placement this run
        derived for each. A ``None`` level key means the value
        cannot be compared at all — the dimension declares that a blank means "nobody
        recorded this" rather than "this is the recorded value". Which inputs those are is
        decided at the declaration, never here: a blank is a real level on some of them (an
        ad-hoc run genuinely has no template) and an absence on others, and only the
        declaration knows which. An input whose reader emits a sentinel for a recorded state
        — a run that pinned no judge config, say — never reaches this function as a blank at
        all, which is how a level that looks empty stops being read as an absence.

        A dimension these runs do not HAVE — pinned to a rig seat no kind among them fills
        (:attr:`~threetears.evals.kernel.host.kinds.KindContract.seats`) — is absent from the
        result entirely rather than present with a ``None`` level. That absence is the claim the
        confound scan already makes about a dimension that held still, and it is what keeps the
        dimension out of the apparatus class as well, since that is built from these keys.
    """
    levels: dict[str, dict[str, str | None]] = {}
    sweepables = profile.sweepables
    values_by_run = {run.id: sweepables.read_all(run, results_by_run.get(run.id, [])) for run in runs}
    apparatus = [declared for declared in sweepables.declarations if declared.role == "apparatus"]
    # ONE decision per dimension, taken over EVERY arm's value before any of them is stored.
    #
    # A dimension this host does not HAVE is omitted rather than read, so it reaches neither the
    # confound scan nor `apparatus_class_of` — whose dimension set is `set(apparatus_levels)`,
    # which is why the cell partition needs no separate rule. Omitting is the only honest answer
    # available for a blank: the declaration says a blank here is an absence, and reporting
    # "nobody recorded the simulated user" about a host that simulates nobody is a fact about the
    # engine's vocabulary rather than about the runs.
    #
    # **Across the cohort, not per run**, which is the whole of what `omits_apparatus` asks for:
    # "one arm recording a level is enough to refute the claim". Deciding inside the run loop
    # dropped a real apparatus DIFFERENCE — the arm that recorded a level was kept and the blank
    # arm was skipped, so the dimension held exactly one level, and a scan that sees one level and
    # no blank emits nothing. That is the failure the declaration-versus-data rule exists to
    # close, re-created one loop in. It also fired the contradiction warning once per run rather
    # than once per dimension.
    omitted = {
        declared.name
        for declared in apparatus
        if runs
        and profile.omits_apparatus(declared.name, [(run, values_by_run[run.id][declared.name]) for run in runs])
    }
    for run in runs:
        values = values_by_run[run.id]
        for declared in apparatus:
            if declared.name in omitted:
                continue
            # A run whose rig had no such seat reads at UNSEATED_LEVEL — a level, not an unknown — beside
            # a run that had it, so a cohort mixing judged and code-only runs of one kind does not read
            # the code-only runs' judge as undecided.
            value = profile.apparatus_level(run, declared.name, values[declared.name])
            undecided = sweepables.is_indeterminate(declared.name, value)
            levels.setdefault(declared.name, {})[run.id] = None if undecided else canonical_json(value)
        # The world this run placed the subject in, on the same axis for the same reason. A run
        # that recorded no placements reads UNDECIDED rather than as having placed nothing: the
        # record is what says what a run did with its world, and absence of a record is an
        # observation nobody made, which is the state that blocks a merge instead of faking one.
        if profile.world is not None:
            placements = run.world_placements
            for dimension in profile.world.declarations:
                key = world_dimension_key(dimension.name)
                levels.setdefault(key, {})[run.id] = None if placements is None else placements.get(dimension.name)
    return levels


def _name_arms(
    index: list[VariantIndexEntry],
    observations: list[Observation],
    folds: _SurfaceFolds,
    run_ids: Collection[str],
) -> list[VariantIndexEntry]:
    """Name each arm by the knobs it swept wherever its resolved surface folds away.

    An arm's ``levers`` are its key's pre-image, and a host that registers an open family's
    resolved surface as a lever puts that surface there rather than the members — so every arm of
    a one-key tool-config sweep carried one opaque surface hash and no mention of the key it
    swept, and an arm table built from it rendered identical rows and could place none of the
    memo's swept-knob coordinates. The same fold every other lens asks decides it here: where the
    surface's residual agrees across the campaign, the members name the arm and the surface is
    recorded as ``folded``; where it does not, the surface moved on its own and stays the arm's
    name, exactly as it stays a moved lever and a confound elsewhere. A surface a fixed knob is
    written into folds the same way and adds nothing to ``swept``: the knob is already one of the
    arm's ``levers``, and names it once the surface is set aside.

    **Campaign-wide, because an arm's name is read against every other arm.** A surface explained
    only within some contrasts would name some arms by their members and others by an opaque hash,
    and the table would compare two vocabularies.

    Args:
        index: The variant index, one entry per keyed variant.
        observations: The observations the index was built from — each names its variant and the
            run it came from, which is what says which runs carried an arm.
        folds: The campaign's :class:`_SurfaceFolds`.
        run_ids: Every resolved member run — the cohort a name is read against.

    Returns:
        The index, each entry carrying ``swept`` and ``folded`` where a surface folds. Entries whose
        levels are unavailable, and every entry when nothing folds, are returned as they were.
    """
    explained = sorted(surface for surface in folds.surfaces if folds.folds_away(surface, run_ids))
    if not explained:
        return index
    runs_of: dict[str, dict[str, None]] = {}
    for observation in observations:
        # An inline-apparatus observation names no run, and so contributes no run's overlays.
        if observation.apparatus_ref is not None:
            runs_of.setdefault(observation.variant_key, {})[observation.apparatus_ref] = None
    named = []
    for entry in index:
        swept: dict[str, SweepableValue] = {}
        folded: list[str] = []
        for surface in explained:
            if surface not in entry.levers:
                continue
            members = folds.swept_members(surface, runs_of.get(entry.variant_key, {}))
            folded.append(surface)
            swept.update(
                {member: SweepableValue.of(value, display=lever_level(value)) for member, value in members.items()}
            )
        if not folded:
            named.append(entry)
            continue
        named.append(
            VariantIndexEntry(
                variant_key=entry.variant_key,
                levers=entry.levers,
                levels_unavailable=entry.levels_unavailable,
                swept=dict(sorted(swept.items())),
                folded=folded,
            )
        )
    return named


def _declared_level_names(index: list[VariantIndexEntry], declared: CampaignDesign | None) -> list[VariantIndexEntry]:
    """Name each arm's levels by what the campaign declared them as, where it declared them.

    A host displays a level by what it can see of it, and for a long text — a prompt in a prompt sweep — that is a
    fingerprint (``cognitive_style: 2304 chars · 539ef3``), so two arms that differ only in their text read as two
    digests. A campaign that declared the level (``SweptAxis.values``) said what to call it: "current text",
    "optimised text". That name replaces the host's display on every entry carrying the level, so every surface
    reading the index — the arm names, the report's tables and charts, and the writer's arm list — prints it.

    **Joined on ``(axis_id, content_hash)``, never on a display**: the hash is the level's identity, and a display
    is what is being replaced. The first declaration wins where one hash is declared twice on one axis, as an
    author reading the declaration top to bottom would expect. A level with no declared name keeps the host's
    display, and so does a lever's "not a run of this kind" level, which is the engine's and no declaration's.
    Only ``display`` changes: the variant key digests content hashes alone, so no key moves and no cell regroups.

    Args:
        index: The variant index, its arms already named by :func:`_name_arms`.
        declared: The campaign's declaration, or None when it declared nothing.

    Returns:
        The index, each declared level displayed by its declared name; entries no declaration names unchanged.
    """
    if declared is None:
        return index
    names: dict[tuple[str, str], str] = {}
    for axis in declared.axes:
        for value in axis.values:
            names.setdefault((axis.axis_id, value.content_hash), value.display)

    def named(levers: dict[str, SweepableValue]) -> dict[str, SweepableValue]:
        return {
            axis: level.model_copy(update={"display": name})
            if level.not_of_kind is None and (name := names.get((axis, level.content_hash))) is not None
            else level
            for axis, level in levers.items()
        }

    renamed: list[VariantIndexEntry] = []
    for entry in index:
        levers, swept = named(entry.levers), named(entry.swept)
        if levers == entry.levers and swept == entry.swept:
            renamed.append(entry)
            continue
        renamed.append(
            VariantIndexEntry(
                variant_key=entry.variant_key,
                levers=levers,
                levels_unavailable=entry.levels_unavailable,
                swept=swept,
                folded=entry.folded,
            )
        )
    return renamed


__all__: list[str] = []
