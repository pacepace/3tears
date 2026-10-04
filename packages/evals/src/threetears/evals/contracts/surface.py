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

from pydantic import Field

from threetears.evals.contracts.analysis_measures import BarAdjudication, MeasureCollection
from threetears.evals.contracts.metrics import MeritAxis
from threetears.evals.contracts.base import EvalDocumentModel
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


class CellFacts(EvalDocumentModel):
    """Everything measured in one cell — one arm, under one rig.

    The population is the cell's observations the harness did not fault, the same one every bar is
    adjudicated over, so a measure read here and a bar verdict on the same cell describe the same
    observations. How many were left out is stated beside them rather than folded in.
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
    measures: MeasureCollection = Field(
        default_factory=MeasureCollection, description="Every measure over the cell's non-faulted observations."
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


__all__ = [
    "CellFacts",
    "DecisionSurface",
    "JudgedDimensionFacts",
    "JudgedReading",
    "MeasureFacts",
]
