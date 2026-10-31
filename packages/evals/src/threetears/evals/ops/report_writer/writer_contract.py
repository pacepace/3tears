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
  measure registry, as it declares the judge kind's;
- :func:`writer_scoring_rubric`, the rubric a writer template's judge scores, and
  :func:`refuse_undeliverable_writer_template`, a template asking for what a writer cannot honour or giving its judge
  nothing to score;
- :func:`refuse_unfixed_scoring_judge`, a campaign whose arms were not scored by one judge on one rubric.

**The skill lever is the skill's content digest, defaults included.** An arm naming no skill runs the shipped one, and
the frozen overlays record the shipped skill's digest (:attr:`AnalystSkill.digest
<threetears.evals.analysis.analyst.skill.AnalystSkill.digest>`), never a bare "shipped": a skill file changed between
two builds then keys two arms apart rather than pooling them as repeats of one condition. A run recorded under an
older build reads as departing from the current skill, which it does.

**The model is not an overlay.** It is the arm's candidate model, which the engine already keys a variant by. Its
reasoning effort is a lever here, honoured by the host's model factory (:data:`WriterModels`), which builds the
arm's model at the effort the lever names; a lever a host declares of its own (a quantization, a served route) rides
on its own sweepables beside these, so nothing here assumes the lever under test is text.

**The scoring judge is fixed; the editor is the arm's.** The editor reviews each report inside the loop, from the
arm's own ``editor.md``, so it is part of what an arm runs. The page an arm stores is then scored by the template's
judge on the editor rubric's seven items, as the shipped skill states them when the template is created
(:func:`writer_scoring_rubric`): the dimensions are copied into the stored template, so neither an arm's ``editor.md``
nor a later build's moves what a campaign's judge asks. Its model is the run's judge pin, which the launch resolves
once for every arm it starts. A campaign whose arms were scored by another judge or rubric is no comparison of arms,
and :func:`refuse_unfixed_scoring_judge` refuses it. Until people rate pages, the judge's scores read as
uncalibrated through the engine's judge evidence tiers, as any judge's do.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Annotated, Final, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field

from threetears.evals.analysis.analyst.skill import load_skill
from threetears.evals.kernel.errors import ValidationFailedError
from threetears.evals.kernel.host.kinds import KindContract, Ordinal
from threetears.evals.kernel.metrics import MetricDescriptor
from threetears.evals.ops.report_writer.loop import MOST_REVISIONS, TurnModel, WriterEnding
from threetears.evals.ops.report_writer.rubric import shipped_rubric
from threetears.evals.ops.report_writer.tally import BounceKind
from threetears.evals.schema.models import EvalRun, EvalTemplate, RubricDim

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
        default=MOST_REVISIONS,
        ge=0,
        le=MOST_REVISIONS,
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
#: Submissions sent back, one measure per check that sent them (``report_writer.<kind>_bounces``), read off the
#: tally's typed counts (:meth:`~threetears.evals.ops.report_writer.tally.Tally.bounces`): every kind the analyst
#: bounces for is measured, and a kind added to :data:`BounceKind` is measured with it.
BOUNCE_MEASURES: Final[Mapping[BounceKind, str]] = {
    kind: f"{WRITER_KIND}.{kind}_bounces" for kind in get_args(BounceKind)
}
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


#: Each bounce kind's measure, in words: its reader name, and why the submissions it counts were sent back.
_BOUNCES: Final[Mapping[BounceKind, tuple[str, str]]] = {
    "unparsed": ("Unparsed", "Submissions sent back to the analyst because they held no single fenced JSON document."),
    "shape": ("Shape", "Submissions sent back to the analyst because the document did not fit the report's model."),
    "validity": (
        "Validity",
        "Submissions sent back to the analyst because the document was not valid on its snapshot.",
    ),
    "invariant": ("Invariant", "Submissions sent back to the analyst because the document broke an invariant."),
    "max_tokens": ("Cut-off", "Submissions sent back to the analyst because the reply was cut at its output cap."),
}


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
    *(
        _count(BOUNCE_MEASURES[kind], f"{reader} bounces", why, fewer_is_better=True)
        for kind, (reader, why) in _BOUNCES.items()
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


def writer_scoring_rubric() -> list[RubricDim]:
    """The rubric a report writer template's judge scores: the editor rubric's items, as the shipped skill states them.

    One ordinal dimension per item, ``report_writer.<item name>``, its description the item's name and definition,
    and its scoring guide what the editor's 2, 1 and 0 mean, at 5, 3 and 1. Read from the shipped ``editor.md``
    (:func:`~threetears.evals.ops.report_writer.rubric.shipped_rubric`), never from an arm's skill. A host seeds a
    writer template's ``rubric`` from it when it creates the template; from then on the stored template holds the
    text, and a campaign's runs freeze it (``EvalRun.template_digest``).

    Each dimension declares the engine's built-in judge prompt for its scale, so a judge config authored for it is
    refused at launch rather than scoring in its place: the prompt is held fixed with the rubric.

    :return: The dimensions, in the rubric's order.
    :rtype: list[RubricDim]
    """
    rubric = shipped_rubric()
    guide = {"5": rubric.top, "3": rubric.middle, "1": rubric.bottom}
    return [
        RubricDim(
            name=f"{WRITER_KIND}.{item.name.lower().replace(' ', '_')}",
            description=f"{item.name}. {item.definition}",
            scale="ordinal",
            scoring_guide=dict(guide),
            judge="builtin",
        )
        for item in rubric.items
    ]


#: A writer template's fields no report writer honours, each with what it would ask of one.
_UNDELIVERABLE: Mapping[str, str] = {
    "conversation": "a simulated conversation (a writer writes one report per case)",
    "goal_state_checks": "goal checks (a writer acts on no world; its page is judged)",
    "preconditions": "preconditions (a writer acts on no world)",
    "tools_allowed": "tools (a writer's tools are its own, mounted by the kind)",
    "tools_required": "tools (a writer's tools are its own, mounted by the kind)",
    "variation_axes": "variation axes (a writer case is frozen from a report snapshot, never generated)",
}


def refuse_undeliverable_writer_template(template: EvalTemplate) -> None:
    """Refuse a report writer template that asks for apparatus a writer cannot honour, or gives its judge nothing to
    score.

    A writer cell is one report over a frozen snapshot, its final page judged on the template's rubric. So a
    conversation, goal checks, preconditions, tools, variation axes or a world seed would each be declared and
    silently never run; and a template with no rubric would have every report written and paid for with nothing to
    judge it on. For a host's ``refuse_undeliverable_template`` check on a writer template, and at every writer launch.

    :param template: A template declaring the report writer kind.
    :ptype template: EvalTemplate
    :raises ValidationFailedError: The template declares any of those, or no rubric; every defect is named at once.
    """
    asked = [why for name, why in _UNDELIVERABLE.items() if getattr(template, name)]
    if template.world_seed.namespaces:
        asked.append("a world seed (a writer acts on no world)")
    if not template.rubric:
        asked.append("no rubric (a writer's page is judged, so its template must say on what: writer_scoring_rubric)")
    if asked:
        raise ValidationFailedError(
            f"report writer template {template.name!r} declares " + "; ".join(asked),
            remedy="author the template with what each of those names",
        )


def _judged_by(run: EvalRun) -> tuple[str, ...]:
    """The judge a run's scores were asked of, as one level: its pin, its configs and its temperature."""
    configs = ", ".join(f"{dim}={config}" for dim, config in sorted((run.judge_config_ids or {}).items()))
    return (f"model {run.judge_model}", f"configs {{{configs}}}", f"temperature {run.judge_temperature}")


def refuse_unfixed_scoring_judge(runs: Sequence[EvalRun]) -> None:
    """Refuse a writer campaign whose arms were not all scored by one judge on one rubric.

    The arms of a writer campaign differ in their levers, and their judged scores compare them only while the judge
    holds still: a page scored by another model, prompt or temperature, or on a rubric edited between launches, would
    move with the instrument and read as the arm's. The engine records each as it happens (the judge's pins on each
    run, what its template measured as ``template_digest``) and reports a judge change as a confound; a promotion is
    a decision, so it is refused instead. A run that recorded no judge or no template digest is refused too: nothing
    shows it was scored as its siblings were.

    :param runs: The campaign's member runs.
    :ptype runs: Sequence[EvalRun]
    :raises ValidationFailedError: The runs name more than one template, rubric revision or judge, or a run recorded
        none; each is named with its runs.
    """
    faults = [
        f"run {run.id} recorded no {what}"
        for run in runs
        for what, recorded in (("judge", run.judge_model), ("template digest", run.template_digest))
        if recorded is None
    ]
    for what, level_of in (
        ("templates", lambda run: (f"template {run.template_id}",)),
        ("rubric revisions", lambda run: (f"template digest {run.template_digest}",)),
        ("judges", _judged_by),
    ):
        levels: dict[tuple[str, ...], list[str]] = {}
        for run in runs:
            levels.setdefault(level_of(run), []).append(run.id)
        if len(levels) > 1:
            named = "; ".join(f"{', '.join(level)} ({', '.join(sorted(ids))})" for level, ids in sorted(levels.items()))
            faults.append(f"the arms were scored under {len(levels)} {what}: {named}")
    if faults:
        raise ValidationFailedError(
            "a writer campaign's arms are compared on one scoring judge and one rubric, and " + "; ".join(faults),
            remedy="launch every arm against one writer template with one judge pin, and edit the template only "
            "between campaigns",
        )


__all__ = [
    "ANALYST_ROUNDS_MEASURE",
    "BOUNCE_MEASURES",
    "EDITOR_ROUNDS_MEASURE",
    "ENDING_MEASURE",
    "FINISHED_MEASURE",
    "MUST_FIX_OPEN_MEASURE",
    "SHOULD_FIX_OPEN_MEASURE",
    "SKILL_LEVER",
    "TURNS_MEASURE",
    "UNPRICED_TURNS_MEASURE",
    "WRITER_KIND",
    "WriterModels",
    "WriterOverlays",
    "refuse_undeliverable_writer_template",
    "refuse_unfixed_scoring_judge",
    "writer_scoring_rubric",
]
