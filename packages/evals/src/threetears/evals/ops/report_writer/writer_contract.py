"""The report writer's kind contract: its levers, the seats its runs fill, and the measures each run lands.

The report writer evaluated as a candidate (:mod:`~threetears.evals.ops.report_writer.writer_kind`): each case is one
frozen report snapshot, and each arm a skill version, a reasoning effort and a revision count on the arm's candidate
model. What the host adds to launch one is only what the host alone knows: the model each arm runs on, built with its
key, and which skill versions a launch may run. Everything else is the same for every host, so it is declared here:

- :class:`WriterOverlays`, the arm's levers (``report_writer.skill``, ``report_writer.reasoning_effort``,
  ``report_writer.revisions``), so two arms that differ in any of them are keyed apart;
- :data:`WRITER_KIND_CONTRACT`, the kind's :class:`~threetears.evals.kernel.host.KindContract`, and
  :func:`writer_kind_contract` for a host that seats apparatus of its own;
- :data:`WRITER_KIND_MEASURES`, the measures a writer run lands on ``host_measures``, for a host to declare on its
  measure registry, as it declares the judge kind's.

**The skill lever is the skill's content digest, defaults included.** An arm naming no skill runs the shipped one, and
the frozen overlays record the shipped skill's digest (:attr:`AnalystSkill.digest
<threetears.evals.analysis.analyst.skill.AnalystSkill.digest>`), never a bare "shipped": a skill file changed between
two builds then keys two arms apart rather than pooling them as repeats of one condition. A run recorded under an
older build reads as departing from the current skill, which it does.

**The model is not an overlay.** It is the arm's candidate model, which the engine already keys a variant by. Its
reasoning effort is a lever here, honoured by the host's model factory (:data:`WriterModels`), which builds the
arm's model at the effort the lever names; a lever a host declares of its own (a quantization, a served route) rides
on its own sweepables beside these, so nothing here assumes the lever under test is text.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Annotated, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from threetears.evals.analysis.analyst.skill import load_skill
from threetears.evals.kernel.host.kinds import KindContract, Ordinal
from threetears.evals.kernel.metrics import MetricDescriptor
from threetears.evals.ops.report_writer.loop import TurnModel, WriterEnding

#: The ``candidate_kind`` a report writer template declares.
WRITER_KIND: Final = "report_writer"

#: The skill lever's name, which the arm's subject carries as a component of the same name (the kind's
#: ``writer_subject_components``), so deriving the variant identity checks the subject against the lever.
SKILL_LEVER: Final = f"{WRITER_KIND}.skill"

#: An effort word, as the shipped model client takes it (``negotiation.EFFORT_LADDER``), lowest first. Declared here
#: rather than imported: ``ops`` may not import ``negotiation``, and a test holds the two equal.
EffortWord = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]

#: The host's model factory: the arm's candidate model, built at the reasoning effort the arm's lever names, or at the
#: model's own default for ``None``. Called once per cell, so each cell's sessions start from a model of their own.
WriterModels = Callable[[EffortWord | None], TurnModel]

#: The most revisions a report is given, as the writing loop holds it (``WriterLimits.revisions``).
_MOST_REVISIONS: Final = 2


def _shipped_skill() -> str:
    """The digest of the skill this build ships: what an arm naming no skill runs."""
    return load_skill().digest


class WriterOverlays(BaseModel):
    """What one arm of a report writer campaign turns besides its model: the skill, the effort and the revisions.

    Each field is a lever (``report_writer.<field>``) that keys the arm's variant. ``skill`` is a content digest, so
    two skill texts that differ by one line are two levels however they were named.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    skill: str = Field(
        default_factory=_shipped_skill,
        pattern=r"^[0-9a-f]{64}$",
        description="the analyst skill both sessions start from, by its content digest (every file's path and text); "
        "none named runs the skill this build ships, recorded by its digest",
    )
    reasoning_effort: Annotated[EffortWord, Ordinal()] | None = Field(
        default=None,
        description="the reasoning effort every turn is sent at, as the host's model factory builds the arm's model; "
        "none sends the model's own default",
    )
    revisions: int = Field(
        default=_MOST_REVISIONS,
        ge=0,
        le=_MOST_REVISIONS,
        description="the revisions the analyst is given after a review that does not clear its report",
    )


#: The seats every report writer run fills: it is judged (its final page is scored on the template's rubric) and it
#: spends under a ceiling (every turn is paid for). It drives no simulated user.
_WRITER_SEATS: frozenset[str] = frozenset({"judge", "max_cost_usd"})


def writer_kind_contract(host_seats: Iterable[str] = ()) -> KindContract:
    """The report writer's contract, also seating apparatus dimensions of the host's own that its runs record.

    :param host_seats: The host's own apparatus dimensions, or pinned roles, a writer run fills.
    :ptype host_seats: Iterable[str]
    :return: The contract: :class:`WriterOverlays`, no spec, and the engine's seats with ``host_seats``.
    :rtype: KindContract
    """
    return KindContract(WRITER_KIND, overlays=WriterOverlays, seats=_WRITER_SEATS | frozenset(host_seats))


#: The report writer's contract, for a profile's ``kinds``: its levers are :class:`WriterOverlays`, and it seats the
#: judge and the spend ceiling.
WRITER_KIND_CONTRACT = writer_kind_contract()

#: How the writing loop ended, on every cell the loop ran.
ENDING_MEASURE: Final = f"{WRITER_KIND}.ending"
#: Whether a review cleared the report, so it is stored finished.
FINISHED_MEASURE: Final = f"{WRITER_KIND}.finished"
#: Submissions sent back for the document's validity on the snapshot.
VALIDITY_BOUNCES_MEASURE: Final = f"{WRITER_KIND}.validity_bounces"
#: Submissions sent back for an invariant.
INVARIANT_BOUNCES_MEASURE: Final = f"{WRITER_KIND}.invariant_bounces"
#: ``must_fix`` findings open on the stored document.
MUST_FIX_OPEN_MEASURE: Final = f"{WRITER_KIND}.must_fix_open"
#: ``should_fix`` findings open on the stored document.
SHOULD_FIX_OPEN_MEASURE: Final = f"{WRITER_KIND}.should_fix_open"
#: The analyst's rounds begun: the draft, then each revision.
ANALYST_ROUNDS_MEASURE: Final = f"{WRITER_KIND}.analyst_rounds"
#: The editor's review rounds begun.
EDITOR_ROUNDS_MEASURE: Final = f"{WRITER_KIND}.editor_rounds"
#: Every turn sent, a turn whose reply never came included.
TURNS_MEASURE: Final = f"{WRITER_KIND}.turns"
#: Turns sent that nothing priced, so the cell's cost is unknown rather than low.
UNPRICED_TURNS_MEASURE: Final = f"{WRITER_KIND}.unpriced_turns"

#: Every ending the writing loop has, in the loop's order.
_ENDINGS: tuple[WriterEnding, ...] = (
    "stored",
    "stored_with_findings",
    "over_budget",
    "refused",
    "review_failed",
    "failed",
)


def _count(name: str, reader_name: str, description: str, *, fewer_is_better: bool) -> MetricDescriptor:
    """A count the run itself reports: lower is better, or a diagnostic with no better end."""
    return MetricDescriptor(
        name=name,
        reader_name=reader_name,
        data_type="numeric",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="end_to_end",
        higher_is_better=False if fewer_is_better else None,
        diagnostic=not fewer_is_better,
        nonnegative=True,
        scale="ratio",
        description=description,
    )


#: The measures a report writer run lands on ``host_measures``, for a host to declare on its measure registry. Each is
#: read off the run by code; the run's cost is its usage rows' (``cost_usd``, which the engine derives), never a measure
#: of its own. A measure about the stored document (the open findings) is absent on a run that stored none, never 0.
WRITER_KIND_MEASURES: tuple[MetricDescriptor, ...] = (
    MetricDescriptor(
        name=ENDING_MEASURE,
        reader_name="Writing ended",
        data_type="categorical",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="end_to_end",
        categories=_ENDINGS,
        description="How the writing loop ended: stored finished, stored a draft, or stored nothing, and why.",
    ),
    MetricDescriptor(
        name=FINISHED_MEASURE,
        reader_name="Report finished",
        data_type="boolean",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="end_to_end",
        higher_is_better=True,
        description="Whether a review cleared the report, so it is stored finished rather than as a draft or not at all.",
    ),
    _count(
        VALIDITY_BOUNCES_MEASURE,
        "Validity bounces",
        "Submissions sent back to the analyst because the document was not valid on its snapshot.",
        fewer_is_better=True,
    ),
    _count(
        INVARIANT_BOUNCES_MEASURE,
        "Invariant bounces",
        "Submissions sent back to the analyst because the document broke an invariant.",
        fewer_is_better=True,
    ),
    _count(
        MUST_FIX_OPEN_MEASURE,
        "Must-fix findings open",
        "The editor's must_fix findings still open on the stored report.",
        fewer_is_better=True,
    ),
    _count(
        SHOULD_FIX_OPEN_MEASURE,
        "Should-fix findings open",
        "The editor's should_fix findings still open on the stored report.",
        fewer_is_better=True,
    ),
    _count(
        ANALYST_ROUNDS_MEASURE,
        "Analyst rounds",
        "The analyst's rounds begun: the draft, then each revision.",
        fewer_is_better=False,
    ),
    _count(EDITOR_ROUNDS_MEASURE, "Editor rounds", "The editor's review rounds begun.", fewer_is_better=False),
    _count(
        TURNS_MEASURE,
        "Writer turns",
        "Every model turn the report's sessions sent, a turn whose reply never came included.",
        fewer_is_better=False,
    ),
    _count(
        UNPRICED_TURNS_MEASURE,
        "Unpriced turns",
        "Turns nothing priced, so the run's cost is unknown rather than low.",
        fewer_is_better=False,
    ),
)


__all__ = [
    "ANALYST_ROUNDS_MEASURE",
    "EDITOR_ROUNDS_MEASURE",
    "ENDING_MEASURE",
    "FINISHED_MEASURE",
    "INVARIANT_BOUNCES_MEASURE",
    "MUST_FIX_OPEN_MEASURE",
    "SHOULD_FIX_OPEN_MEASURE",
    "SKILL_LEVER",
    "TURNS_MEASURE",
    "UNPRICED_TURNS_MEASURE",
    "VALIDITY_BOUNCES_MEASURE",
    "WRITER_KIND",
    "WriterModels",
    "WriterOverlays",
]
