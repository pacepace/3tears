"""Launching a report writer arm: every refusal it meets before it is priced, its plan, and its cell deadline.

The writer's launch registry entry (``actions.report_writing.launchable_report_writer_kind``) builds what only it can
build, the tools each cell mounts; everything it decides is here, so its plan and its launcher read one admission
(:func:`admit_writer_launch`):

- the template, held to what a writer honours and given a rubric to score on
  (:func:`~threetears.evals.ops.report_writer.writer_contract.refuse_undeliverable_writer_template`);
- the arm's skill, one of the versions the launch was given;
- the scoring judge, named by the launch and of another model family than the writer, wherever the host can tell a
  model's family;
- the cases: the launch's case set when it names one (a writer bank's ``writer-dev`` or ``writer-held-out``,
  resolved by the dispatch), else the template's live writer cases; each must carry a writer case this build reads,
  and none may be retired.

**The judge's family is the host's to tell.** The engine knows a model only by the id a launch names: a provider's
own id (``claude-haiku-5-5``) and a routed one (``anthropic/claude-haiku-5.5``) of one model share nothing it could
compare, and no table of vendors it kept would stay true. So the host answers it (:data:`ModelFamily`), and a model
the host cannot place is admitted with the check recorded on the run as not made, never guessed.

**A writer cell runs far past the engine's default deadline.** Its report is tens of turns sent back to back, then
the judge's phase, so :func:`writer_launch_deadline` derives the cell's deadline from the arm's limits, the report
budget's turn cap and the host's request ceiling
(:func:`~threetears.evals.ops.report_writer.writer_kind.writer_cell_timeout_s`), and the launcher wires it
(``KindWiring.cell_timeout_s``).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from threetears.evals.analysis.analyst.skill import AnalystSkill, load_skill
from threetears.evals.kernel.errors import ValidationFailedError
from threetears.evals.kernel.host.eval_host import EvalHost
from threetears.evals.ops.report_writer.loop import WriterLimits
from threetears.evals.ops.report_writer.tally import ReportBudget
from threetears.evals.ops.report_writer.writer_bank import writer_case_bank
from threetears.evals.ops.report_writer.writer_contract import (
    WriterOverlays,
    refuse_undeliverable_writer_template,
)
from threetears.evals.ops.report_writer.writer_kind import writer_cell_timeout_s
from threetears.evals.run.judge import JUDGE_CALL_ATTEMPTS, JUDGE_MAX_TOKENS
from threetears.evals.run.launch import ArmPlan, LaunchArgument, LaunchRequest, plan_judge, require_candidate_model
from threetears.evals.schema.completion import RequestCeiling
from threetears.evals.schema.logs import get_logger
from threetears.evals.schema.models import EvalTestCase, JudgedArtifact, scored_dim_ids

log = get_logger(__name__)

#: The host's reading of a model's family, by the id a launch names it by: the vendor that trained it
#: (``anthropic``, ``google``), or ``None`` for a model the host cannot place.
ModelFamily = Callable[[str], str | None]

#: The launch arguments a writer arm cannot honour: it drives no simulated user, calls no tool a cassette could
#: record, and generates no case (each is minted from a host's own campaign).
WRITER_UNHONOURED_LAUNCH_ARGUMENTS: Final[frozenset[LaunchArgument]] = frozenset(
    {"simulator_model", "cassette_mode", "n_variations"}
)


@dataclass(frozen=True, kw_only=True)
class WriterAdmitted:
    """What :func:`admit_writer_launch` resolved of a writer arm it can honour.

    :ivar overlays: The arm's levers, as the dispatch validated them.
    :ivar skill: The skill the arm runs, the one its ``skill`` lever names.
    :ivar candidate_model: The model the arm writes with.
    :ivar judge_model: The scoring judge's pin, as the launch named it.
    :ivar judge_family: The family check as made: both families when the host placed both, else which model it
        could not place. Recorded on the run.
    :ivar cases: The cases the arm runs, in run order.
    """

    overlays: WriterOverlays
    skill: AnalystSkill
    candidate_model: str
    judge_model: str
    judge_family: str
    cases: tuple[EvalTestCase, ...]


def _skill(overlays: WriterOverlays, skills: Mapping[str, AnalystSkill]) -> AnalystSkill:
    """The skill the arm's lever names, refused unless the launch was given it."""
    shipped = load_skill()
    given = {shipped.digest: shipped, **skills}
    skill = given.get(overlays.skill)
    if skill is None or skill.digest != overlays.skill:
        raise ValidationFailedError(
            f"the arm runs skill {overlays.skill}, which this launch was not given; the skills it may run are "
            f"{', '.join(sorted(given))}",
            remedy="name one of those skills, or give the launch the skill the arm names",
        )
    return skill


def _judge_family(candidate_model: str, judge_model: str, model_family: ModelFamily) -> str:
    """The family check, refused when the writer and its scoring judge are of one family; else what was checked."""
    writer, judge = model_family(candidate_model), model_family(judge_model)
    if writer is not None and writer == judge:
        raise ValidationFailedError(
            f"the scoring judge {judge_model!r} is of the writer {candidate_model!r}'s family ({writer}), and a judge "
            "scoring its own family's pages favours them",
            remedy="name a judge of another model family",
        )
    unplaced = [model for model, family in ((candidate_model, writer), (judge_model, judge)) if family is None]
    if unplaced:
        log.warning(
            "eval.writer_judge_family unchecked: the host places no family for %s", ", ".join(map(repr, unplaced))
        )
        return f"unchecked: the host places no family for {', '.join(map(repr, unplaced))}"
    return f"writer {writer}, judge {judge}"


def _cases(request: LaunchRequest, stored: Sequence[EvalTestCase]) -> tuple[EvalTestCase, ...]:
    """The arm's cases: its case set's, else the template's live ones; each a writer case this build reads, none
    retired."""
    cases = request.cases_or([case for case in stored if not case.archived])
    where = f"case set {request.case_set.ref.label!r}" if request.case_set is not None else "the template"
    if not cases:
        raise ValidationFailedError(
            f"{where} holds no writer case to run", remedy="mint some into it (writer_case_mint)"
        )
    bank = writer_case_bank(cases)
    read, unreadable = {case.id for case, _ in bank.cases}, dict(bank.unreadable)
    faults: list[str] = []
    for case in cases:
        if case.id in unreadable:
            faults.append(f"{case.id!r} ({unreadable[case.id]})")
        elif case.id not in read:
            faults.append(f"{case.id!r} (carries no writer case)")
        elif case.archived:
            faults.append(f"{case.id!r} (retired)")
    if faults:
        raise ValidationFailedError(
            f"{where} names case(s) no writer can run: {', '.join(faults)}",
            remedy="mint a version of the set without them (writer_case_mint lists a source's new case without its "
            "retired one, case_set_mint any other), or launch without case_set",
        )
    return tuple(cases)


def admit_writer_launch(
    request: LaunchRequest,
    stored: Sequence[EvalTestCase],
    *,
    skills: Mapping[str, AnalystSkill],
    model_family: ModelFamily,
) -> WriterAdmitted:
    """Every refusal a writer arm meets before it is priced, and what it resolved to: for its plan and its launcher.

    In order: the template
    (:func:`~threetears.evals.ops.report_writer.writer_contract.refuse_undeliverable_writer_template`), the arm's
    skill, its model, the scoring judge and its family, then the cases (the module docstring).

    :param request: The arm.
    :ptype request: LaunchRequest
    :param stored: The template's stored cases in the request's scope, read when the launch names no case set.
    :ptype stored: Sequence[EvalTestCase]
    :param skills: The skill versions the launch may run, by digest, beside the shipped one.
    :ptype skills: Mapping[str, AnalystSkill]
    :param model_family: The host's reading of a model's family.
    :ptype model_family: ModelFamily
    :return: What the arm resolved to.
    :rtype: WriterAdmitted
    :raises ValidationFailedError: The template asks for what a writer cannot honour or has no rubric; the arm's skill
        was not given; the arm names no model or no judge; the judge is of the writer's family; or a case is
        missing, unreadable, carries no writer case or is retired.
    """
    refuse_undeliverable_writer_template(request.template)
    overlays = request.overlays_as(WriterOverlays) if request.overlays is not None else WriterOverlays()
    skill = _skill(overlays, skills)
    candidate_model = require_candidate_model(request, None)
    if request.judge_model is None:
        raise ValidationFailedError(
            "a writer arm's pages are scored by one judge for the whole campaign, and this launch names none",
            remedy="name the scoring judge (judge_model), and name the same one for every arm of the campaign",
        )
    judge_family = _judge_family(candidate_model, request.judge_model, model_family)
    return WriterAdmitted(
        overlays=overlays,
        skill=skill,
        candidate_model=candidate_model,
        judge_model=request.judge_model,
        judge_family=judge_family,
        cases=_cases(request, stored),
    )


def plan_writer_arm(host: EvalHost, request: LaunchRequest, admitted: WriterAdmitted) -> ArmPlan:
    """A writer arm's plan: its admitted cases on its model, scored on the page by its judge, with no simulator.

    :param host: The host, where the judge configs are read.
    :ptype host: EvalHost
    :param request: The arm.
    :ptype request: LaunchRequest
    :param admitted: What :func:`admit_writer_launch` resolved of it.
    :ptype admitted: WriterAdmitted
    :return: The plan.
    :rtype: ArmPlan
    :raises ValidationFailedError: Anything :func:`~threetears.evals.run.plan_judge` refuses.
    """
    return ArmPlan(
        case_count=len(admitted.cases),
        candidate_model=admitted.candidate_model,
        judge=plan_judge(
            host,
            request.template,
            admitted.judge_model,
            request.judge_config_ids,
            judged_artifact=JudgedArtifact.DOCUMENT,
        ),
        simulator_model=None,
    )


def writer_launch_deadline(
    request: LaunchRequest, admitted: WriterAdmitted, *, budget: ReportBudget, request_s: RequestCeiling
) -> float:
    """The deadline every cell of a writer arm runs under: its whole report, then its page's judge phase.

    :param request: The arm, whose template's rubric the judge scores and whose settings say how many judge calls a
        cell makes at once.
    :ptype request: LaunchRequest
    :param admitted: What the arm resolved to; its revisions bound the report's rounds.
    :ptype admitted: WriterAdmitted
    :param budget: Each report's budget; its turn cap bounds the report's turns.
    :ptype budget: ReportBudget
    :param request_s: The host's ceiling for one request on its client, by output cap.
    :ptype request_s: RequestCeiling
    :return: The cell deadline in seconds
        (:func:`~threetears.evals.ops.report_writer.writer_kind.writer_cell_timeout_s`).
    :rtype: float
    """
    return writer_cell_timeout_s(
        WriterLimits(revisions=admitted.overlays.revisions),
        request_s=request_s,
        turns=budget.turns,
        judge_dims=len(scored_dim_ids([dim.name for dim in request.template.rubric], JudgedArtifact.DOCUMENT)),
        judge_concurrency=request.settings.judge_concurrency,
        judge_call_attempts=JUDGE_CALL_ATTEMPTS,
        judge_max_tokens=JUDGE_MAX_TOKENS,
    )


__all__ = [
    "WRITER_UNHONOURED_LAUNCH_ARGUMENTS",
    "ModelFamily",
    "WriterAdmitted",
    "admit_writer_launch",
    "plan_writer_arm",
    "writer_launch_deadline",
]
