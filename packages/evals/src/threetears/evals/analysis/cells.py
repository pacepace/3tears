"""Variant-primary cells — what pools with what, and what a refusal to pool costs.

A cell is ``(variant_key, apparatus_class_id)``: the resolved contestant stack, measured
under one rig. It is deliberately **not** a run. A run is a batch — one arm's trials,
commissioned together — and using it as the unit of comparison produces the defect this
module exists to remove: a re-run of an arm becomes a second thing to compare rather than
more of the same arm, and an arm measured under two rigs cannot be told apart from itself.

**Three rules decide every merge here, and each of them refuses in the safe direction.**

*Identical recorded apparatus, or no merge.* Two observations pool when their variant
matches and their apparatus falls in the same class. A dimension recorded at two values
is a rival explanation for whatever moved, so the cells stay separate and the reason is
stated rather than averaged away.

*An unrecorded dimension blocks the merge too.* ``unknown`` is neither agreement nor
difference: it is an observation nobody made, and reading it as either asserts a fact
that does not exist. Blocking is the conservative side, and it has a real cost, though
not the one it is easy to assume: two observations that BOTH failed to record a
dimension are *identically* silent, so they share a class and pool normally, carrying
the gap forward in ``unknown_dimensions``. What refuses is the ASYMMETRIC case — one
side recorded it and the other did not — which is the common shape when a registry entry
is newer than some of the data. That cost is paid deliberately, and :class:`NextExperiment`
is what keeps it from being pure refusal: the engine says what recording would unblock,
and how many more observations it would pool.

*Commissioned and witnessed never pool.* Whether an apparatus was set before the fact or
found after it is the difference between an experiment and a log, and an analysis that
cannot see which it is holding will describe one as the other. It is a property of the RIG,
so it lives on the apparatus class and enters the class id: a witnessed observation and a
commissioned one under identical recorded dimensions are two classes, and so two cells,
whatever reads a cell by its two coordinates downstream. The same distinction runs
one axis over, and lands on the FIRST rule rather than this one: a state dimension the
subject perceives that one run seeded and another merely witnessed is an apparatus
dimension recorded at two values, so it refuses to merge and names itself.

**Nothing here names a host concept.** The dimensions, the levers and the measures are
whatever the host's profile registered; this module does arithmetic over names it never
interprets.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection, Iterable, Mapping, Sequence
from typing import Literal

from pydantic import Field, model_validator

from threetears.evals.schema.hashing import canonical_digest
from threetears.evals.schema.base import EvalDocumentModel
from threetears.evals.schema.models import ApparatusProvenance

#: The cell model this module implements, stamped onto every analysis generated under it
#: (``GenerationProvenance.cell_model_version``).
#:
#: A stored analysis is a claim about which observations pooled, and that claim is only legible
#: beside the rule that produced it: two analyses whose cells were computed under different
#: definitions are not comparable on n, on k, or on any per-cell number. Bump it whenever what a
#: cell pools or says changes through the PACKAGE — a core dimension joining the apparatus or world
#: coordinate, a change to the pooling rule, or a renamed count. A host adding, removing or renaming
#: one of its own declarations moves the coordinate too, but it is not a reason to bump this and a
#: host cannot: that move is recorded beside it, as the host declarations digest the bundle and the
#: generation's provenance carry (``AnalysisContextBundle.host_declarations_digest``). Why each
#: earlier version moved is in this file's history.
CELL_MODEL_VERSION: int = 11


class ApparatusClass(EvalDocumentModel):
    """One configuration of the measuring rig, and what it could not say about itself.

    The class id digests the recorded map **and the set of dimensions that were not
    recorded** — the names only, never a value standing in for an absence. Hashing an absence
    as a value is what would make two observations that recorded nothing look like two that
    agreed; hashing the *names* of what went unmeasured does the opposite, because which axes
    a measurement could not speak to is a real property of that measurement.

    **The id must carry it, and it is not enough for the merge rule to.** :func:`pool_observations`
    groups by this id before any rule is consulted, so two classes sharing an id are one group
    and never reach :func:`_refusal` at all. While the id digested `recorded` alone, an
    observation that recorded `{judge: j-1}` and one that recorded `{judge: j-1}` while never
    recording `ocr` hashed identically, pooled into a single cell of two observations, and emitted no
    ``RefusedMerge`` — a wrong MERGE, which is the one outcome nothing downstream can undo, and
    exactly what this module exists to refuse.

    **Provenance is a third half, for the same reason.** Every surface downstream of the cell
    algebra keys a cell by ``(variant_key, apparatus_class_id)``. While provenance rode beside the
    class rather than inside it, a witnessed and a commissioned observation of one variant under
    the same recorded rig were two cells sharing both coordinates, and every per-cell surface
    keyed on them silently kept one and dropped the other.
    """

    apparatus_class_id: str = Field(
        min_length=1,
        description=(
            "sha256 over three parts under distinct keys: the sorted map of recorded dimension → level, the "
            "sorted NAMES of the dimensions that were not recorded, and the provenance. None is optional — "
            "pooling groups by this id before any merge rule is consulted, so an id leaving one out lets two "
            "classes that measured different amounts, or one set and one found, share a cell."
        ),
    )
    recorded: dict[str, str] = Field(
        default_factory=dict,
        description="Dimension → canonical level, for every apparatus dimension this observation actually recorded.",
    )
    unknown_dimensions: list[str] = Field(
        default_factory=list,
        description=(
            "Apparatus dimensions this observation never recorded, sorted. Not an empty value and not a "
            "disagreement — a state nothing observed, which blocks a merge and generates a next-experiment."
        ),
    )
    provenance: ApparatusProvenance = Field(
        description=(
            "commissioned = the rig was set before the observations were made. witnessed = it was found "
            "afterwards. Read off the run that carried it (`EvalRun.apparatus_provenance`). Digested into the "
            "id, so cells never pool across the two: the distinction is what separates an experiment from a "
            "log, and an analysis that cannot see it will describe one as the other."
        )
    )

    @model_validator(mode="after")
    def _unknown_and_recorded_are_disjoint(self) -> ApparatusClass:
        """A dimension cannot be both recorded and unrecorded.

        Returns:
            The validated class.

        Raises:
            ValueError: A dimension appears in both maps, which means two producers disagreed
                about whether it was observed and the class can no longer say which is true.
        """
        if overlap := sorted(set(self.recorded) & set(self.unknown_dimensions)):
            raise ValueError(f"dimensions are both recorded and unknown: {', '.join(overlap)}")
        return self


class Observation(EvalDocumentModel):
    """One measurement, at one cell, under one rig.

    The unit the engine actually pools. It carries its cell coordinates rather than a link to
    a run, which is what makes a runless host expressible: an observational consumer supplies
    apparatus per observation and declares no batch at all, and nothing downstream can tell
    the difference.

    ``apparatus_ref`` names the batch this observation was commissioned under when there was
    one, and is ``None`` when the apparatus was carried inline. It is provenance for a reader,
    never an input to the cell — two observations from different batches with identical
    apparatus are one cell, which is the whole point of pooling across batches.

    It carries no provenance of its own: whether its rig was set or found is a property of the rig,
    so it is read off the class its ``apparatus_class_id`` names, and there is no second field for
    the two to disagree in.
    """

    id: str = Field(min_length=1, description="Stable id of this observation.")
    scope_id: str = Field(min_length=1, description="The scope this observation was read under.")
    variant_key: str = Field(min_length=1, description="The resolved contestant stack — see compute_variant_key.")
    apparatus_class_id: str = Field(min_length=1, description="Which rig configuration measured it.")
    apparatus_ref: str | None = Field(
        default=None,
        description="The batch this was commissioned under, or None when the apparatus was carried inline.",
    )
    measures: dict[str, float] = Field(
        default_factory=dict,
        description="Measure id → value, in the host's registered measure vocabulary.",
    )
    case_ref: str | None = Field(
        default=None,
        description="The battery case this exercised, when there is a battery. None is a real state, not a gap.",
    )
    observed_at: str = Field(default="", description="When the measurement was taken (ISO-8601).")


class Cell(EvalDocumentModel):
    """Every observation sharing one variant and one apparatus class.

    Replaces a model in which a cell **was** a run and ``moved`` was derived by diffing that
    run's overlays against the control's. Three things that model could not say, and this one
    can: a re-run of the control is the same cell with more observations rather than a second
    cell that "moved nothing"; the control is a variant rather than a run id, so curating one run out no longer
    destroys the design if another observation carries the same variant; and a variant's
    sample is its own arithmetic rather than something a chart arm has to invent.

    **How many observations pooled is not how replicated the cell is**, and the cell says both
    rather than one standing for the other. Fifteen observations over five cases is five cases
    run three times — five independent draws, not fifteen — and fifteen over fifteen cases is
    fifteen draws with no repeat at all. Only the cases the observations exercised separate the
    two, so the cell states ``n_cases`` and the repeats per case beside the pooled count. The
    repeats are a MIN and a MAX because a cell can hold a case fewer times than its neighbours —
    a lost observation, or a repeat run covering part of the battery — and one number would
    have to round that away.
    """

    variant_key: str = Field(min_length=1, description="The cell's variant coordinate.")
    apparatus_class_id: str = Field(min_length=1, description="The cell's apparatus coordinate.")
    provenance: ApparatusProvenance = Field(
        description="The provenance of the cell's apparatus class, restated so a reader need not resolve the class.",
    )
    n_observations: int = Field(
        ge=1,
        description=(
            "How many observations pooled here, counted over every case and every repeat. NOT a repeat "
            "count: fifteen observations can be five cases run three times or fifteen cases run once, and "
            "`n_cases` with `repeats_per_case_min`/`repeats_per_case_max` is what says which. A cell with "
            "no observations is not a cell."
        ),
    )
    n_cases: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Distinct cases the pooled observations exercised — the independent draws behind this cell. "
            "None when some observation carries no case reference, which is a real shape (a host with no "
            "battery) rather than a gap: cases are then not a coordinate this cell can be counted on."
        ),
    )
    repeats_per_case_min: int | None = Field(
        default=None,
        ge=1,
        description=(
            "The fewest observations any one case contributed — how many times the least-repeated case ran "
            "in this cell. Pooling several runs of one setting adds their repeats, so this can exceed any one "
            "run's `k_runs`. None exactly when `n_cases` is."
        ),
    )
    repeats_per_case_max: int | None = Field(
        default=None,
        ge=1,
        description=(
            "The most observations any one case contributed. Equal to `repeats_per_case_min` when every case "
            "was repeated alike, which is the ordinary shape; None exactly when `n_cases` is."
        ),
    )
    observation_ids: list[str] = Field(
        default_factory=list,
        description="The pooled observations, sorted. What a reader checks the arithmetic against.",
    )
    unknown_dimensions: list[str] = Field(
        default_factory=list,
        description="Apparatus dimensions none of these observations recorded — why this cell did not pool further.",
    )

    @model_validator(mode="after")
    def _the_counts_agree_with_each_other(self) -> Cell:
        """Every count is arithmetic over the evidence, never a second source of truth for it.

        Returns:
            The validated cell.

        Raises:
            ValueError: ``n_observations`` disagrees with ``observation_ids``, which would let a
                cell report a sample size no list of evidence supports; the case count and the
                repeats are not all present or all absent; or ``n_cases`` cases repeated between
                the stated bounds cannot add up to ``n_observations``, which would let a cell
                state a replication its own sample contradicts.
        """
        if self.n_observations != len(self.observation_ids):
            raise ValueError(
                f"n_observations={self.n_observations} disagrees with {len(self.observation_ids)} observation ids"
            )
        n_cases, low, high = self.n_cases, self.repeats_per_case_min, self.repeats_per_case_max
        if n_cases is None or low is None or high is None:
            if (n_cases, low, high) != (None, None, None):
                raise ValueError(
                    "n_cases, repeats_per_case_min and repeats_per_case_max are stated together or not at all — "
                    f"got {(n_cases, low, high)}"
                )
            return self
        if low > high:
            raise ValueError(f"repeats_per_case_min={low} exceeds repeats_per_case_max={high}")
        if not n_cases * low <= self.n_observations <= n_cases * high:
            raise ValueError(
                f"{n_cases} cases repeated {low}..{high} times cannot pool {self.n_observations} observations"
            )
        return self


class RefusedMerge(EvalDocumentModel):
    """Two cells that share a variant and did not pool, and which rule stopped them.

    Stated rather than silent because a refusal is evidence. A reader looking at two cells of
    three observations where they expected one of six is owed the reason, and the reason is
    actionable in exactly one of the three cases — which is why ``reason`` is an enum a
    consumer can branch on rather than a sentence it has to parse.
    """

    variant_key: str = Field(min_length=1, description="The variant both cells share.")
    apparatus_class_ids: list[str] = Field(
        min_length=2,
        max_length=2,
        description=(
            "The apparatus classes of the two cells that stayed apart, sorted. Always two: provenance is "
            "digested into the class id, so two same-variant cells are two classes even when they share "
            "every recorded dimension and differ only in whether the rig was set or found."
        ),
    )
    reason: Literal["apparatus_differs", "apparatus_unknown", "provenance_differs"] = Field(
        description=(
            "apparatus_differs = a dimension was recorded at two values, so a rival explanation exists. "
            "apparatus_unknown = a dimension was never recorded, so whether it varied is undecidable — the "
            "only one of the three that recording can fix. provenance_differs = one side was commissioned and "
            "the other witnessed."
        )
    )
    dimensions: list[str] = Field(
        default_factory=list,
        description="Which dimensions carried the refusal, sorted. Empty for provenance_differs, which names no dimension.",
    )


class NextExperiment(EvalDocumentModel):
    """What recording one dimension would buy, in pooled observations.

    Both counts are :attr:`Cell.n_observations`, maxed or summed over the cells in question, and
    are named for it: a replication claim reads a run's ``k_runs``, never these.

    The mitigation that keeps a conservative merge from being pure refusal. Generated
    mechanically — no model is asked — because it is arithmetic: these observations already
    share a variant, and the only thing keeping them in separate cells is a dimension nobody
    wrote down. Recording it either merges them or proves they never should have merged, and
    both answers are worth more than the current silence.
    """

    variant_key: str = Field(min_length=1, description="The variant whose pooling is being held back.")
    dimension: str = Field(min_length=1, description="The unrecorded apparatus dimension.")
    n_observations_now: int = Field(
        ge=1,
        description=(
            "The largest `n_observations` among the cells this one recording would merge — the baseline "
            "the gain is measured against. Scoped to those cells rather than to the whole variant, because "
            "a variant-wide baseline would compare this merge's result against a cell it cannot reach and "
            "report a real gain as none."
        ),
    )
    n_observations_if_recorded: int = Field(
        ge=1,
        description=(
            "The `n_observations` those cells reach together once the dimension is recorded AND turns out "
            "to agree. The optimistic branch, deliberately: the pessimistic one — they disagree and stay "
            "apart — needs no action, so the number worth showing is the one that says what is at stake."
        ),
    )
    observation_ids: list[str] = Field(
        default_factory=list,
        description="The observations missing the dimension — where the recording has to happen, sorted.",
    )

    @model_validator(mode="after")
    def _recording_can_only_help(self) -> NextExperiment:
        """A next-experiment that promises no gain is noise a reader must still read.

        Returns:
            The validated entry.

        Raises:
            ValueError: The promised observation count does not exceed today's, so there is nothing to act on.
        """
        if self.n_observations_if_recorded <= self.n_observations_now:
            raise ValueError(
                f"n_observations_if_recorded={self.n_observations_if_recorded} does not exceed "
                f"n_observations_now={self.n_observations_now}"
            )
        return self


class SubjectKeyInstability(EvalDocumentModel):
    """A subject key and its label disagreeing about how many things there are.

    A population-level check, not a judgement about any single observation: one key seen
    under two labels means something was renamed mid-campaign (or two things collided onto
    one key), and one label seen under two keys means a thing a reader believes is single is
    actually two. Either way a reader grouping on the label — which is what a reader does,
    because the label is what is legible — gets a grouping the keys do not support.

    This warns and never refuses. Both states are legitimate often enough (a rename IS a
    rename) that blocking on them would stop honest analyses, and the point is that the memo
    says so rather than that the engine decides.
    """

    kind: Literal["one_key_many_labels", "one_label_many_keys"] = Field(
        description="Which direction the disagreement runs — they have different remedies and must not pool."
    )
    key: str = Field(default="", description="The subject key, for one_key_many_labels.")
    label: str = Field(default="", description="The subject label, for one_label_many_keys.")
    counterparts: list[str] = Field(
        description="The two-or-more values observed on the other side, sorted.", min_length=2
    )


def apparatus_class_of(
    recorded: Mapping[str, str | None],
    *,
    provenance: ApparatusProvenance,
    dimensions: Collection[str] = (),
    id_neutral: Collection[str] = (),
) -> ApparatusClass:
    """Classify one observation's rig from the dimensions it recorded.

    Args:
        recorded: Dimension → canonical level. A ``None`` level, or a dimension named in
            ``dimensions`` and absent here, is an unrecorded dimension.
        dimensions: Every apparatus dimension the host declares. Supplied so a dimension the
            registry knows about and this observation never mentioned reads as unrecorded
            rather than as a dimension that does not exist — those are different facts, and only the
            declaration can tell them apart.
        provenance: Whether the rig was set (``commissioned``) or found (``witnessed``). Required:
            only the writer of the observations knows, and a default would let a log read as an
            experiment.
        id_neutral: Dimensions this observation's class lists but its id does not digest, because the
            caller has established that leaving them out cannot make two different classes share an id
            (the bundle's :data:`~threetears.evals.analysis.bundle.CELL_ID_NEUTRAL` says when). They stay on
            the class — in ``recorded`` or ``unknown_dimensions`` — so the merge rule still refuses to pool
            an unrecorded dimension with a recorded one; only the id is spared a dimension that says
            nothing new, which is what keeps a cell minted before the dimension existed addressable.

    Returns:
        The class. Its id digests the recorded map, the names of the unrecorded dimensions and the
        provenance — see :class:`ApparatusClass` for why none of the three can be dropped — less the
        ``id_neutral`` dimensions.
    """
    known = {name: level for name, level in recorded.items() if level is not None}
    unknown = sorted((set(dimensions) | set(recorded)) - set(known))
    neutral = set(id_neutral)
    return ApparatusClass(
        # Every part under its own key so a dimension NAMED as unknown can never collide with one
        # RECORDED at a value that happens to equal its name.
        apparatus_class_id=canonical_digest(
            {
                "provenance": provenance,
                "recorded": {name: level for name, level in known.items() if name not in neutral},
                "unknown": [name for name in unknown if name not in neutral],
            }
        ),
        recorded=known,
        unknown_dimensions=unknown,
        provenance=provenance,
    )


#: What separates a cell reference's two coordinates. Neither coordinate is a hex digest's
#: alphabet member, so the split is unambiguous whichever half a reader is after.
_CELL_REF_SEPARATOR = ":"


def cell_ref(variant_key: str, apparatus_class_id: str) -> str:
    """Render a cell's two coordinates as the one string a citation can carry.

    A cell IS ``(variant_key, apparatus_class_id)`` and a reference to one has to name both:
    the same contestant stack measured under two rigs is two cells, and a reference carrying
    only the variant half would place a number on whichever of them a reader met first.

    Minted here rather than formatted at each call site so that this function and
    :func:`variant_of_cell_ref` are the only two places the format exists — a reference
    produced one way and parsed another is a join that silently finds nothing.

    The producer is :mod:`threetears.evals.analysis.references`, which keys the decision surface's
    cells by this string and resolves the short alias a writer names a cell by
    (:func:`~threetears.evals.analysis.references.cell_of_alias`) back to it, so a stored analysis
    carries this identity and never the alias.

    Args:
        variant_key: The cell's variant coordinate.
        apparatus_class_id: The cell's apparatus coordinate.

    Returns:
        The reference.
    """
    return f"{variant_key}{_CELL_REF_SEPARATOR}{apparatus_class_id}"


def variant_of_cell_ref(ref: str) -> str | None:
    """Read the variant coordinate out of a cell reference, or refuse it.

    Args:
        ref: A reference minted by :func:`cell_ref`.

    Returns:
        The variant key, or ``None`` when the string is not a cell reference. ``None`` is a
        refusal rather than a fallback: a caller that guessed — reading an unparseable ref as
        a bare variant key, say — would place a measurement at a coordinate nobody stated,
        and a number at the wrong arm is the one error a reader cannot see.
    """
    variant, separator, apparatus = ref.partition(_CELL_REF_SEPARATOR)
    if not separator or not variant or not apparatus:
        return None
    return variant


def _refusal(
    a: ApparatusClass, b: ApparatusClass
) -> tuple[Literal["apparatus_differs", "apparatus_unknown"], list[str]] | None:
    """Decide whether two apparatus classes may merge, and on what grounds they may not.

    Args:
        a: One class.
        b: The other.

    Returns:
        ``(reason, dimensions)`` when the merge is refused, or None when the two are one class.

        **A definite difference is reported ahead of an unknown, and only dimensions recorded
        on BOTH sides can be different.** Both halves of that matter and they pull opposite
        ways. A dimension unrecorded on one side and recorded on the other is undecidable, not
        different — calling it a difference claims an observation nobody made — so it is
        excluded from the difference scan. But once some dimension genuinely differs, the
        merge is closed whatever else is unknown, and saying ``apparatus_unknown`` there sends
        an operator to record a dimension that cannot change the outcome. Reporting the
        decisive fact first is what keeps the actionable reason actionable.
    """
    undecidable = set(a.unknown_dimensions) | set(b.unknown_dimensions)
    comparable = (set(a.recorded) & set(b.recorded)) - undecidable
    if differing := sorted(k for k in comparable if a.recorded[k] != b.recorded[k]):
        return "apparatus_differs", differing
    if unknown := sorted(undecidable):
        return "apparatus_unknown", unknown
    return None


def pool_observations(
    observations: Sequence[Observation],
    classes: Mapping[str, ApparatusClass],
) -> tuple[list[Cell], list[RefusedMerge], list[NextExperiment]]:
    """Pool observations into variant-primary cells, and say what did not pool and why.

    The single producer of all three answers, because they are one traversal of one grouping
    and three producers would be free to disagree about which observations were in which
    cell.

    Args:
        observations: Every observation under analysis. Order does not matter; outputs are
            sorted so the bundle they land in stays fingerprintable.
        classes: Apparatus class id → the class, for every id the observations reference.

    Returns:
        ``(cells, refused_merges, next_experiments)``.

    Raises:
        KeyError: An observation references an apparatus class not in ``classes``, which
            would silently drop it from every cell.
    """
    grouped: dict[tuple[str, str], list[Observation]] = defaultdict(list)
    for obs in observations:
        if obs.apparatus_class_id not in classes:
            raise KeyError(f"observation {obs.id} references unknown apparatus class {obs.apparatus_class_id}")
        grouped[(obs.variant_key, obs.apparatus_class_id)].append(obs)

    cells = [
        Cell(
            variant_key=variant,
            apparatus_class_id=class_id,
            provenance=classes[class_id].provenance,
            n_observations=len(members),
            **_replication_of(members),
            observation_ids=sorted(o.id for o in members),
            unknown_dimensions=classes[class_id].unknown_dimensions,
        )
        for (variant, class_id), members in sorted(grouped.items())
    ]

    refused = _refused_merges(cells, classes)
    return cells, refused, _next_experiments(cells, classes)


def _replication_of(members: Sequence[Observation]) -> dict[str, int | None]:
    """Count one cell's cases and how many times each was repeated, from the observations alone.

    Read off ``case_ref`` rather than off any run's ``k_runs``, because a cell is not a run: it
    pools every run of one setting, so its repeats per case are the sum of theirs, and a run's
    launch count describes only the run.

    Args:
        members: The observations pooled into the cell.

    Returns:
        ``n_cases``, ``repeats_per_case_min`` and ``repeats_per_case_max``, all None when any
        observation carries no case reference — a cell partly on a battery and partly off one
        has no single count of cases to state, and counting only the half that has one would
        report a replication for evidence it does not describe.
    """
    refs = [obs.case_ref for obs in members]
    if any(ref is None for ref in refs):
        return {"n_cases": None, "repeats_per_case_min": None, "repeats_per_case_max": None}
    per_case: dict[str | None, int] = defaultdict(int)
    for ref in refs:
        per_case[ref] += 1
    return {
        "n_cases": len(per_case),
        "repeats_per_case_min": min(per_case.values()),
        "repeats_per_case_max": max(per_case.values()),
    }


def _refused_merges(cells: Sequence[Cell], classes: Mapping[str, ApparatusClass]) -> list[RefusedMerge]:
    """Name every pair of same-variant cells that stayed apart, and the rule that kept them.

    Args:
        cells: The pooled cells.
        classes: Apparatus class id → the class.

    Returns:
        One entry per refused pair, sorted. Pairs are unordered — a refusal is symmetric, and
        emitting both directions would double every count a reader makes off this list.
    """
    by_variant: dict[str, list[Cell]] = defaultdict(list)
    for cell in cells:
        by_variant[cell.variant_key].append(cell)

    refused: list[RefusedMerge] = []
    for variant, members in sorted(by_variant.items()):
        for i, left in enumerate(members):
            for right in members[i + 1 :]:
                if left.provenance != right.provenance:
                    refused.append(
                        RefusedMerge(
                            variant_key=variant,
                            apparatus_class_ids=sorted({left.apparatus_class_id, right.apparatus_class_id}),
                            reason="provenance_differs",
                        )
                    )
                    continue
                verdict = _refusal(classes[left.apparatus_class_id], classes[right.apparatus_class_id])
                if verdict is not None:
                    reason, dimensions = verdict
                    refused.append(
                        RefusedMerge(
                            variant_key=variant,
                            apparatus_class_ids=sorted({left.apparatus_class_id, right.apparatus_class_id}),
                            reason=reason,
                            dimensions=dimensions,
                        )
                    )
    return refused


def _next_experiments(cells: Sequence[Cell], classes: Mapping[str, ApparatusClass]) -> list[NextExperiment]:
    """Compute what recording each unrecorded dimension would buy, per variant.

    A recording is only worth asking for when the dimension is **the last thing** separating
    the cells it would merge. That is the whole difficulty here, and getting it wrong produces
    confident false advice rather than silence: cells kept apart by six unrecorded dimensions
    each look, dimension by dimension, like they are one recording away from merging, so a
    naive scan emits six entries that each promise a gain and none of which delivers one. A
    host that inherits rig dimensions it structurally never records — an observational
    consumer with no judge and no simulator — hits that case on every pair it has.

    So a cell is a candidate for dimension ``D`` only when ``D`` is its ONLY unknown, and the
    cells that would then pool are those agreeing on every other recorded dimension and
    sharing a provenance. Anything else is a merge this recording cannot complete.

    Args:
        cells: The pooled cells.
        classes: Apparatus class id → the class.

    Returns:
        One entry per (variant, dimension) that recording could actually unblock, sorted. A
        dimension whose recording would merge nothing is omitted — an entry promising no gain
        is worse than no entry, because a reader acts on it.
    """
    by_variant: dict[str, list[Cell]] = defaultdict(list)
    for cell in cells:
        by_variant[cell.variant_key].append(cell)

    entries: list[NextExperiment] = []
    for variant, members in sorted(by_variant.items()):
        if len(members) < 2:
            continue
        dimensions = sorted({dim for cell in members for dim in classes[cell.apparatus_class_id].unknown_dimensions})
        for dimension in dimensions:
            # Only a cell whose sole remaining unknown is this dimension can be unblocked by
            # recording it. A cell carrying a second unknown stays put whatever happens here.
            candidates = [c for c in members if set(classes[c.apparatus_class_id].unknown_dimensions) <= {dimension}]
            if len(candidates) < 2:
                continue
            for group in _mergeable_groups(candidates, classes, dimension):
                reachable = _reachable_once_recorded(group, classes, dimension)
                if len(reachable) < 2:
                    continue
                n_now = max(c.n_observations for c in reachable)
                n_if_recorded = sum(c.n_observations for c in reachable)
                if n_if_recorded <= n_now:
                    continue
                group = reachable
                entries.append(
                    NextExperiment(
                        variant_key=variant,
                        dimension=dimension,
                        n_observations_now=n_now,
                        n_observations_if_recorded=n_if_recorded,
                        observation_ids=sorted(
                            oid
                            for c in group
                            if dimension in classes[c.apparatus_class_id].unknown_dimensions
                            for oid in c.observation_ids
                        ),
                    )
                )
    return entries


def _reachable_once_recorded(
    group: Sequence[Cell],
    classes: Mapping[str, ApparatusClass],
    dimension: str,
) -> list[Cell]:
    """Narrow a group to what recording ``dimension`` could ACTUALLY merge.

    :func:`_mergeable_groups` strips ``dimension`` from every signature so a cell that recorded
    it and one that did not can group together — which is right, because the hypothesis is that
    the second one records it and matches. But stripping it from BOTH sides of a pair that each
    recorded it, at DIFFERENT values, groups two cells that recording can never join: they
    already disagree on the one thing under hypothesis, and no further recording changes that.

    Left unnarrowed, the group's observations are summed over all of them and the promise is arithmetic
    nobody can keep — with an empty ``observation_ids``, because no cell in it is actually
    missing the dimension. That entry reaches the analysis bundle and then the generator.

    Args:
        group: Cells that agree on every recorded dimension OTHER than this one.
        classes: Apparatus class id → the class.
        dimension: The dimension whose recording is hypothesised.

    Returns:
        The cells one recording could pool: every cell missing the dimension, plus the LARGEST
        set of cells already agreeing on a single recorded value for it. Largest because the
        unknowns may turn out to match any of the recorded values, and the optimistic branch is
        the one worth showing — but only one of them, since the recorders cannot all be right.
    """
    unknown = [c for c in group if dimension in classes[c.apparatus_class_id].unknown_dimensions]
    if not unknown:
        return []

    by_value: dict[str, list[Cell]] = defaultdict(list)
    for cell in group:
        recorded = classes[cell.apparatus_class_id].recorded.get(dimension)
        if recorded is not None:
            by_value[recorded].append(cell)

    if not by_value:
        return unknown
    largest = max(by_value.values(), key=lambda cells: (sum(c.n_observations for c in cells), len(cells)))
    return unknown + largest


def _mergeable_groups(
    candidates: Sequence[Cell],
    classes: Mapping[str, ApparatusClass],
    dimension: str,
) -> list[list[Cell]]:
    """Group cells that would become one cell once ``dimension`` is recorded and agrees.

    Args:
        candidates: Cells whose only unknown is ``dimension`` (or which have none).
        classes: Apparatus class id → the class.
        dimension: The dimension whose recording is being hypothesised.

    Returns:
        The groups, each in cell order. Two cells share a group when they agree on every
        recorded dimension OTHER than the one being hypothesised and share a provenance —
        ``dimension`` itself is excluded from the comparison because the hypothesis is that it
        gets recorded, so a side that already recorded it must not count as disagreeing with a
        side that has not.
    """
    grouped: dict[str, list[Cell]] = defaultdict(list)
    for cell in candidates:
        klass = classes[cell.apparatus_class_id]
        signature = canonical_digest(
            {
                "provenance": cell.provenance,
                "recorded": {k: v for k, v in klass.recorded.items() if k != dimension},
            }
        )
        grouped[signature].append(cell)
    return [group for _, group in sorted(grouped.items())]


def subject_key_instabilities(observed: Iterable[tuple[str, str]]) -> list[SubjectKeyInstability]:
    """Report subject keys and labels that disagree about how many subjects there are.

    Args:
        observed: ``(subject_key, subject_label)`` for every observation, duplicates included.

    Returns:
        Every instability, keys before labels and sorted within each. Empty when the two
        agree, which is the ordinary case and the one that must produce no noise.
    """
    labels_by_key: dict[str, set[str]] = defaultdict(set)
    keys_by_label: dict[str, set[str]] = defaultdict(set)
    for key, label in observed:
        labels_by_key[key].add(label)
        keys_by_label[label].add(key)

    warnings = [
        SubjectKeyInstability(kind="one_key_many_labels", key=key, counterparts=sorted(labels))
        for key, labels in sorted(labels_by_key.items())
        if len(labels) > 1
    ]
    warnings += [
        SubjectKeyInstability(kind="one_label_many_keys", label=label, counterparts=sorted(keys))
        for label, keys in sorted(keys_by_label.items())
        if len(keys) > 1
    ]
    return warnings


__all__ = [
    "CELL_MODEL_VERSION",
    "ApparatusClass",
    "Cell",
    "NextExperiment",
    "Observation",
    "RefusedMerge",
    "SubjectKeyInstability",
    "apparatus_class_of",
    "cell_ref",
    "pool_observations",
    "subject_key_instabilities",
    "variant_of_cell_ref",
]
