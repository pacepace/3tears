"""The decision surface — the campaign's per-cell numbers, as code computed them.

An analysis has two layers, and this module is the one no model writes. The interpretation — the
memo, the findings' verdicts — is prose, and whether it is right is an eval's question. The
numbers it rests on are arithmetic over the campaign's cells, and a number a model transcribes
reaches the reader with the same authority as one code computed while nothing checks it. So the
per-cell numbers are computed once, at bundle assembly, and frozen onto the analysis here.

**Every number on the page resolves here.** On an analysis generated with references, a finding's
evidence rows, chart payloads, merit claims and caveat magnitudes name a cell and a measure, and
code fills the figure from these facts (:mod:`threetears.evals.analysis.references`). An older
analysis's figures were typed by the model and are marked as such.

**Persisted: the facts. Derived: the table.** A :class:`CellFacts` holds everything measured in one
cell — every measure, every judged dimension, its replication, the notes on its runs — and nothing
about how to present it. Which arms group together, their order, and which measures stand in for
cost and latency (read off each measure's frozen ``merit_axis``) are derived on every read, once,
by :mod:`threetears.evals.analysis.surface_table`, and served to every surface — so a reader can pivot
without a stored table deciding for them, and no two surfaces derive it differently. That is the
same line :mod:`threetears.evals.analysis.arms` draws, for the same reason.

**Every measure is kept, not only the ones a surface shows today**, because a reference the
interpretation makes — a measure at a cell — must resolve against the analysis it was written in,
and the bundle it was written over is not stored (only its fingerprint is).

A leaf module: the analysis models import it, so it imports neither them nor the bundle.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, field_validator, model_validator

from threetears.evals.contracts.analysis_measures import BarAdjudication, MeasureCollection
from threetears.evals.contracts.metrics import MeasurePopulation, MeritAxis
from threetears.evals.contracts.base import EvalDocumentModel
from threetears.evals.contracts.evidence_tiers import JudgedEvidenceTier
from threetears.evals.contracts.models import DimName


class JudgedReading(EvalDocumentModel):
    """One judged dimension's scores in one cell.

    Kept beside the measures rather than inside them because a judged dimension is not a measure:
    it never ranks, and it reaches the bundle through the score projection, where its name
    survives, rather than through the measure walk, where it does not.
    """

    dimension: DimName = Field(min_length=1, description="The dimension, spelled exactly as the judge stamped it.")
    mean: float | None = Field(default=None, description="Mean score on the dimension's own scale. None when n is 0.")
    sem: float | None = Field(default=None, description="Standard error of that mean. None below n=2.")
    n: int = Field(ge=0, description="Scores contributing to the mean — one per scored, non-faulted observation.")
    n_independent: int = Field(ge=0, description="Distinct test cases behind those scores.")
    n_infra_excluded: int = Field(
        default=0, ge=0, description="Scores on observations the harness faulted, left out of n and the mean."
    )
    n_cannot_tell: int = Field(
        default=0,
        ge=0,
        description="Observations the judge could not score on this dimension, left out of n and the mean.",
    )
    evidence_tier: JudgedEvidenceTier = Field(
        description=(
            "What these scores can bear, decided by code from the judge's measured reliability "
            "(`threetears.evals.contracts.evidence_tiers`): the weakest tier among the judges that served them. "
            "`undetermined` when the evidence decides no tier — never a default standing in for one."
        )
    )


class MeasureFacts(EvalDocumentModel):
    """What one measure IS, frozen beside its values — the catalogue entry a reader needs to read them.

    Frozen rather than looked up at render for two reasons: a browser has no measure registry to
    ask, and a registry edit after generation would otherwise change how an old analysis's numbers
    read — which axis a measure counts on, which end is better — under an interpretation written
    against the old description.
    """

    unit: str | None = Field(default=None, description="The measure's unit, e.g. 'ms' or 'usd'. None when unitless.")
    merit_axis: MeritAxis | None = Field(
        default=None,
        description="Which axis of merit the measure is read on — how a surface picks its cost and latency columns.",
    )
    higher_is_better: bool | None = Field(
        default=None, description="Which end is better. None for a categorical measure."
    )
    materiality_threshold: float | None = Field(
        default=None,
        ge=0.0,
        description=(
            "The magnitude, in the measure's unit, below which a difference between two cells is labelled "
            "immaterial. None when the host declared none, and every difference is then material."
        ),
    )
    population: MeasurePopulation | None = Field(
        default=None,
        description=(
            "The population the measure declares, when it declares one; every cell's summary states the population "
            "it was actually computed over."
        ),
    )


class JudgedDimensionFacts(EvalDocumentModel):
    """What one judged dimension IS, frozen beside its scores — :class:`MeasureFacts`' judged sibling.

    A reference to a judged score needs its polarity to say whether a move improved anything and
    its scale to draw it, and the judge configuration that declared both is not stored with the
    analysis. A judged dimension's merit axis is always quality — a judge scores how good an
    output is — so none is stored.
    """

    higher_is_better: bool = Field(default=True, description="Which end of the scale is better.")
    value_range: tuple[float, float] | None = Field(
        default=None, description="The scale the scores are on, when declared."
    )


#: The fewest distinct cases a stratum holds before its figures are read on their own. Below it a
#: stratum's figures are still stated, with their intervals and n, and the report says the stratum is too
#: small to read: at ten cases a 90% rate's Wilson interval runs from about 60% to 98%, which is the floor
#: the classifier guide sets per label (``docs/designing-classifier-evals.md`` § 4). Flagged, never a
#: reason to leave a stratum out.
STRATUM_MIN_CASES = 10


def _no_turn_delivered(n_observations: int, n_infra_excluded: int, n_no_turn: int | None) -> bool:
    """Whether no observation the harness did not fault took a turn — every one a failure that took none.

    The one reading of the counts :attr:`CellFacts.all_failed` and :attr:`StratumFacts.all_failed` share.
    False when nothing was counted at all (every observation faulted), and when the count was not kept.
    """
    if not n_no_turn:
        return False
    return n_observations - n_infra_excluded - n_no_turn == 0


def _check_failures_fit(
    n_observations: int, n_infra_excluded: int, n_candidate_failed: int | None, n_no_turn: int | None
) -> None:
    """Refuse failure counts that cannot come from the observations they partition.

    Raises:
        ValueError: One count was kept without the other; the faulted and failed counts sum to more than
            ``n_observations``; or more failures took no turn than failed at all.
    """
    if (n_candidate_failed is None) != (n_no_turn is None):
        raise ValueError("n_candidate_failed and n_no_turn are kept together, or neither is")
    if n_candidate_failed is None or n_no_turn is None:
        return
    if n_infra_excluded + n_candidate_failed > n_observations:
        raise ValueError(
            f"{n_infra_excluded} faulted and {n_candidate_failed} failed observations cannot come from "
            f"{n_observations}: a result is faulted, failed or neither, never two of them"
        )
    if n_no_turn > n_candidate_failed:
        raise ValueError(
            f"{n_no_turn} failures that took no turn cannot come from {n_candidate_failed} failures: they are a "
            "subset of them"
        )


#: How :attr:`CellFacts.n_no_turn` and :attr:`StratumFacts.n_no_turn` read.
_N_NO_TURN = (
    "Of `n_candidate_failed`, the failures that took no turn — the candidate's model refused or errored before "
    "the cell's deadline (`delivered_a_turn` is False) — left out of every cost and latency measure, which "
    "describe the turns the candidate took. A failure that took a turn (its budget ended it, its output cap "
    "cut it, its deadline struck mid-call) is not here: its time and spend stay in. None exactly when "
    "`n_candidate_failed` is."
)

#: How :attr:`CellFacts.n_candidate_failed` and :attr:`StratumFacts.n_candidate_failed` read.
_N_CANDIDATE_FAILED = (
    "Observations the candidate failed (`classify_result`'s `candidate_fail`), for any cause: counted against "
    "the arm in every rate, bar and judged score below. None on an analysis frozen before the count was kept "
    "— whose cost and latency were read over every non-faulted result, failures included — never 0 standing "
    "in for a count nobody took."
)


class StratumFacts(EvalDocumentModel):
    """Everything measured in one stratum of one cell — the cell's figures again, over one kind of case.

    The same population and the same rules as the cell's own: the stratum's results are a subset of the
    cell's, summarised by the walk that summarised the cell (classifier per-label statistics included,
    derived from the stratum's own confusion matrix) and judged by the same transposition. So a stratum's
    figure and the cell's pooled figure differ only in which observations they read.
    """

    stratum: str | None = Field(
        description=(
            "The stratum the cases declare (`EvalTestCase.stratum`). None for the cases in this cell that declare "
            "none — or whose case document no longer resolves, so nothing can say which stratum they were — kept "
            "as their own entry so the strata still add up to the cell."
        ),
    )
    n_observations: int = Field(
        ge=1, description="Observations of this stratum's cases in the cell, over every repeat — faulted ones included."
    )
    n_cases: int = Field(
        ge=1,
        description=(
            f"Distinct cases behind those observations — the independent draws every figure here rests on. Below "
            f"{STRATUM_MIN_CASES} the stratum is too small to read on its own: its figures are stated, and its "
            "intervals are wide."
        ),
    )
    n_infra_excluded: int = Field(
        default=0, ge=0, description="Observations the harness faulted, excluded from every value below."
    )
    n_candidate_failed: int | None = Field(default=None, ge=0, description=_N_CANDIDATE_FAILED)
    n_no_turn: int | None = Field(default=None, ge=0, description=_N_NO_TURN)
    measures: MeasureCollection = Field(
        default_factory=MeasureCollection,
        description="Every measure over the stratum's observations, by the cell's own rules.",
    )
    judged: list[JudgedReading] = Field(
        default_factory=list, description="Every judged dimension scored in this stratum, sorted by dimension."
    )

    @property
    def all_failed(self) -> bool:
        """Whether no result of the stratum the harness did not fault took a turn — see :attr:`CellFacts.all_failed`."""
        return _no_turn_delivered(self.n_observations, self.n_infra_excluded, self.n_no_turn)

    @model_validator(mode="after")
    def _failures_fit(self) -> StratumFacts:
        """Refuse failure counts the stratum cannot hold — see :meth:`CellFacts._failures_fit`."""
        _check_failures_fit(self.n_observations, self.n_infra_excluded, self.n_candidate_failed, self.n_no_turn)
        return self


class CellFacts(EvalDocumentModel):
    """Everything measured in one cell — one arm, under one rig.

    The population is the cell's observations the harness did not fault, the same one every bar is
    adjudicated over, so a measure read here and a bar verdict on the same cell describe the same
    observations. How many were left out is stated beside them rather than folded in.

    **A cost or latency measure is read over fewer: the turns the candidate took**
    (:func:`~threetears.evals.contracts.result_condition.delivered_a_turn`). Every candidate failure is a
    result of the arm, and every rate, bar and judged score counts it against the arm; but a call the model
    refused or errored on straight away took no turn, so its round trip is not a turn's latency and its
    empty usage not a spend anyone observed. Averaged in, a classifier arm whose every call was refused
    read 53 ms and $0 on the surface — the fastest, cheapest arm — and nothing said its every result had
    failed. A failure that took a turn — ended by the turn budget, cut by the output cap, struck by the
    deadline mid-call — stays in: its time and spend are what failing cost the arm. So both are counted
    here (``n_candidate_failed``, and the ``n_no_turn`` of them left out of cost and latency), and a cell
    where no result took a turn says so (:attr:`all_failed`) where it would otherwise have no cost or
    latency reading and look merely unmeasured.
    """

    variant_key: str = Field(min_length=1, description="The arm's variant — its key into the analysis's variant_index.")
    apparatus_class_id: str = Field(
        min_length=1, description="The rig it was measured under — the other half of its cell."
    )
    run_ids: list[str] = Field(min_length=1, description="The member runs whose observations this cell pooled, sorted.")
    n_observations: int = Field(
        ge=1, description="Observations pooled here, over every case and every repeat — faulted ones included."
    )
    n_cases: int | None = Field(
        default=None,
        ge=1,
        description="Distinct cases those observations exercised — the independent draws. None when a case is unrecorded.",
    )
    repeats_per_case_min: int | None = Field(default=None, ge=1, description="Fewest times any one case ran here.")
    repeats_per_case_max: int | None = Field(default=None, ge=1, description="Most times any one case ran here.")
    n_infra_excluded: int = Field(
        default=0, ge=0, description="Observations the harness faulted, excluded from every value below."
    )
    n_candidate_failed: int | None = Field(default=None, ge=0, description=_N_CANDIDATE_FAILED)
    n_no_turn: int | None = Field(default=None, ge=0, description=_N_NO_TURN)
    measures: MeasureCollection = Field(
        default_factory=MeasureCollection,
        description=(
            "Every measure over the cell's non-faulted observations — a cost or latency measure over the turns the "
            "candidate took, leaving out the `n_no_turn` failures that took none."
        ),
    )
    judged: list[JudgedReading] = Field(
        default_factory=list, description="Every judged dimension scored in this cell, sorted by dimension."
    )
    short_runs: dict[str, str] = Field(
        default_factory=dict,
        description="Member runs of this cell that ran short of their design, each with the bundle's sentence saying how.",
    )
    incomplete_runs: dict[str, str] = Field(
        default_factory=dict, description="Member runs of this cell that did not complete, each with its status."
    )
    strata: list[StratumFacts] = Field(
        default_factory=list,
        description=(
            "The cell read again per stratum of its cases, beside the pooled figures above: named strata in name "
            "order, then the cases that declare none, when some do. Every observation of the cell is in exactly "
            "one entry. Empty when none of the cell's cases declares a stratum, and on a time-axis position, whose "
            "cells are not broken down by stratum."
        ),
    )

    @property
    def all_failed(self) -> bool:
        """Whether no result of this cell the harness did not fault took a turn — each one a failure that took none.

        Such a cell carries no cost or latency reading, and that absence is not "unmeasured": every result
        it holds is a failure, which its rates and bars count. Derived from the counts rather than stored,
        so it cannot disagree with them. A cell whose every result failed but some took a turn (its budget
        ended them) is not ``all_failed``: it has the cost and latency of those turns to state, and
        ``n_candidate_failed`` says every result failed. False when no result was counted at all (every one
        faulted, which ``n_infra_excluded`` says), and when the count was not kept (``n_no_turn`` None).
        """
        return _no_turn_delivered(self.n_observations, self.n_infra_excluded, self.n_no_turn)

    @model_validator(mode="after")
    def _failures_fit(self) -> CellFacts:
        """Refuse failure counts the cell cannot hold.

        A result is faulted, failed or neither (``classify_result``), so the faulted and failed counts
        together are at most the observations; the failures that took no turn are some of the failures; and
        the two failure counts are kept together or not at all. A cell claiming otherwise is describing a
        population that does not exist.

        Raises:
            ValueError: Any of those does not hold.
        """
        _check_failures_fit(self.n_observations, self.n_infra_excluded, self.n_candidate_failed, self.n_no_turn)
        return self

    @field_validator("strata")
    @classmethod
    def _strata_are_distinct_and_named(cls, strata: list[StratumFacts]) -> list[StratumFacts]:
        """Refuse a stratum listed twice, or a breakdown whose only entry is the cases that declare none.

        Raises:
            ValueError: Two entries share a stratum, or every entry is the undeclared one — a cell none of
                whose cases declares a stratum is not broken down at all, and says so by an empty list.
        """
        names = [stratum.stratum for stratum in strata]
        if len(set(names)) != len(names):
            raise ValueError(f"a cell lists each stratum once; got {names}")
        if strata and all(name is None for name in names):
            raise ValueError("a cell whose cases declare no stratum carries no strata, not one undeclared stratum")
        return strata

    @model_validator(mode="after")
    def _strata_add_up_to_the_cell(self) -> CellFacts:
        """Refuse strata that do not partition the cell's observations and cases.

        A stratum left out would leave its observations in the pooled figure and in no stratum, and a
        reader adding the strata up would meet a cell larger than its parts with nothing saying why.

        Raises:
            ValueError: The strata's observations, faulted observations, failed observations or cases do not
                sum to the cell's.
        """
        if not self.strata:
            return self
        if sum(stratum.n_observations for stratum in self.strata) != self.n_observations:
            raise ValueError("a cell's strata hold every one of its observations, each in one stratum")
        if sum(stratum.n_infra_excluded for stratum in self.strata) != self.n_infra_excluded:
            raise ValueError("a cell's strata hold every one of its faulted observations, each in one stratum")
        for name, total, counts in (
            ("failed", self.n_candidate_failed, [stratum.n_candidate_failed for stratum in self.strata]),
            ("no-turn", self.n_no_turn, [stratum.n_no_turn for stratum in self.strata]),
        ):
            if total is None:
                if any(count is not None for count in counts):
                    raise ValueError(f"a cell that kept no {name} count has strata that kept none either")
            elif None in counts or sum(count or 0 for count in counts) != total:
                raise ValueError(f"a cell's strata hold every one of its {name} observations, each in one stratum")
        if self.n_cases is not None and sum(stratum.n_cases for stratum in self.strata) != self.n_cases:
            raise ValueError("a cell's strata hold every one of its cases, each in one stratum")
        return self


def all_failed_sentence(n_all_failed: int, n_cells: int) -> str:
    """The one sentence every surface says about cells where no result took a turn (:attr:`CellFacts.all_failed`).

    Stated once, here, because two readers say it — the bundle the analysis is written from, and the
    decision-surface table every report renders — and a rewording in one would leave them describing one
    cell two ways. The sentence names no cell: the bundle lists the cells beside it, and the table names
    their arms after it when only some cells failed.

    Args:
        n_all_failed: The cells where no result took a turn, at least one.
        n_cells: The cells on the surface.

    Returns:
        The sentence.
    """
    if n_all_failed == n_cells:
        return (
            "Every result failed: the candidate's model refused or errored on every call and took no turn, so there "
            "is no cost or latency to read — not a fast, free arm — and every failure counts against its arm in "
            "each rate, bar and judged score."
        )
    return (
        f"Every result failed in {n_all_failed} of {n_cells} cells: the candidate's model refused or errored on "
        "every call there and took no turn, so those cells have no cost or latency to read — not a fast, free arm "
        "— and every failure counts against its arm in each rate, bar and judged score."
    )


#: What a campaign's time positions are: the builds a host labels (``release``), or the UTC days its runs
#: started on (``date``).
TimeAxisBasis = Literal["release", "date"]


class TimePosition(EvalDocumentModel):
    """One point on a campaign's time axis — the runs at one build or on one day, and what they measured.

    ``cells`` are computed exactly as the surface's own cells are, over this position's runs alone: the same
    pooling, the same non-faulted population and the same judged transposition, so a cell's reading here and
    its reading on the whole surface are one rule applied to two sets of observations.
    """

    key: str = Field(
        min_length=1, description="The position's name: the build's label value, or the day as YYYY-MM-DD."
    )
    first_run_at: str = Field(description="When the earliest run at this position was created, ISO-8601.")
    last_run_at: str = Field(description="When the latest run at this position was created, ISO-8601.")
    run_ids: list[str] = Field(min_length=1, description="The runs at this position that measured something, sorted.")
    cells: list[CellFacts] = Field(
        min_length=1,
        description="Every cell measured at this position, ordered by (variant_key, apparatus_class_id).",
    )


class TimeAxis(EvalDocumentModel):
    """The campaign's runs placed in time — present only when they span two builds or two days.

    **Ordered by when each position first ran**, never by its name: a build label is the host's string and
    nothing here can sort it ("0.10" against "0.9"), while the order the runs happened in is a fact every run
    records. A date axis is in calendar order by the same rule.
    """

    basis: TimeAxisBasis = Field(
        description=(
            "`release` when the positions are the builds the host labels; `date` when they are the UTC days the "
            "runs started on — the fallback when the host labels no build, a run recorded none, or every run "
            "recorded the same one."
        )
    )
    release_label: str | None = Field(
        default=None,
        description="The host label the positions name, on a `release` axis; None on a `date` axis.",
    )
    basis_reason: str | None = Field(
        default=None,
        description=(
            "Why the positions are days rather than builds, on a `date` axis — the host labels no build, every run "
            "recorded the same one, or which runs recorded none — so a fallback from builds to days is stated "
            "rather than silent. None on a `release` axis, which needs no reason."
        ),
    )
    positions: list[TimePosition] = Field(
        min_length=2, description="The positions, earliest first. Two or more, or there is no axis."
    )

    @field_validator("positions")
    @classmethod
    def _positions_are_distinct(cls, positions: list[TimePosition]) -> list[TimePosition]:
        """Refuse two positions under one name — a reader could not tell which point is which."""
        keys = [position.key for position in positions]
        if repeated := sorted({key for key in keys if keys.count(key) > 1}):
            raise ValueError(f"time positions must be distinct; repeated: {', '.join(repeated)}")
        return positions

    @model_validator(mode="after")
    def _release_names_its_label(self) -> TimeAxis:
        """A release axis says which label it reads, and a date axis reads none."""
        if (self.basis == "release") != (self.release_label is not None):
            raise ValueError("a `release` time axis names its release_label, and a `date` axis names none")
        return self

    @model_validator(mode="after")
    def _a_date_axis_says_why_it_is_not_builds(self) -> TimeAxis:
        """A `date` axis states why it fell back from builds, and a `release` axis states no fallback.

        Days are the fallback, and a host that labels its builds reads a date axis as its builds unless
        something says otherwise — which runs lacked the label is the one thing that sends them to fix it.
        """
        reason = (self.basis_reason or "").strip()
        if self.basis == "date" and not reason:
            raise ValueError("a `date` time axis states its basis_reason: why the positions are days, not builds")
        if self.basis == "release" and self.basis_reason is not None:
            raise ValueError("a `release` time axis carries no basis_reason; it fell back from nothing")
        return self


class DecisionSurface(EvalDocumentModel):
    """The campaign's measured cells and the bars they were held to — frozen at generation.

    Copied from the bundle the analysis was generated over, never recomputed from the store: runs
    added to or archived from the campaign afterwards would otherwise change the numbers under an
    interpretation written about the old ones.
    """

    control_variant_key: str | None = Field(
        default=None, description="The declared control's variant, or None when the campaign declared none."
    )
    cells: list[CellFacts] = Field(
        default_factory=list, description="One entry per cell, ordered by (variant_key, apparatus_class_id)."
    )
    bars: list[BarAdjudication] = Field(
        default_factory=list, description="Every bar the campaign is held to, each adjudicated against every cell."
    )
    measures: dict[str, MeasureFacts] = Field(
        default_factory=dict,
        description="What each measure named in `cells` is, keyed by measure name — every name there has an entry.",
    )
    dimensions: dict[DimName, JudgedDimensionFacts] = Field(
        default_factory=dict,
        description=(
            "What each judged dimension named in `cells` is, keyed by dimension — every name there has an entry. "
            "Empty when no cell names a judged dimension."
        ),
    )
    time_axis: TimeAxis | None = Field(
        default=None,
        description=(
            "The campaign's runs placed in time, each position carrying its own cells; None when the runs share one "
            "build and one day. What a `timeseries` chart draws, and the only thing it can draw."
        ),
    )


__all__ = [
    "STRATUM_MIN_CASES",
    "CellFacts",
    "DecisionSurface",
    "JudgedDimensionFacts",
    "JudgedReading",
    "MeasureFacts",
    "StratumFacts",
    "TimeAxis",
    "TimeAxisBasis",
    "TimePosition",
    "all_failed_sentence",
]
