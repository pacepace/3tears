"""The report writer as a candidate kind: its skill and settings measured like any other candidate's.

A change to the writer (a line of its skill, its reasoning effort, its revisions) is a decision the engine's corrected
comparison makes, so the writer runs as a kind (:data:`~threetears.evals.ops.report_writer.writer_contract.WRITER_KIND`)
through the ordinary run path, beside the analysis reporter.

- **A case** (:class:`WriterCase`) is one report snapshot, frozen whole into the case's ``host_payload``, and the
  exemplar notes its analyst is not given. A snapshot is derived and regenerable: a schema wipe sweeps it, and freezing
  the campaign again after one builds it from evidence that may have moved. So the case carries the snapshot document
  itself (its report input, the bundle it was built from and its digests), as a reporter case carries its bundle, and
  a cell puts it back in the run's scope before any turn (``report_snapshot_restore``): its id is derived from its
  evidence, so the snapshot restored is the one the case froze, byte for byte, and the analyst tools read it as they
  read any. A snapshot this build no longer reads excludes the cell as the harness's, naming the remedy: freeze the
  case again. Withholding is per case, so a blind run withholds what the case froze and nothing a launch chose.
- **An arm** is a :class:`~threetears.evals.ops.report_writer.writer_contract.WriterOverlays` on the arm's candidate
  model. :class:`ReportWriterKind` is built per arm by the launcher, with the arm's overlays, the skill versions the
  launch may run and the host's model factory. The arm's subject carries its skill as a component under the skill
  lever's own name (:func:`writer_subject_components`), so deriving the variant identity refuses a subject whose
  skill is not the lever's, and :meth:`ReportWriterKind.prepare` refuses a subject that names another.
- **A cell** writes the report over the case by driving the writing loop
  (:func:`~threetears.evals.ops.report_writer.loop.write_rounds`) directly: the loop is the one ``report_write``
  runs, and the cell keeps what ``report_write`` adds around it out of the host's scope. It stores no report
  document and no analyst gap (its tools are mounted with the gap note that stores nothing); the one thing it writes
  is the case's snapshot, which a freeze of the same evidence would write too.
- **The output** is the final page's text, which the judge reads beside the facts code computed from the snapshot
  (the editor's). The run's own measures land on ``host_measures``
  (:data:`~threetears.evals.ops.report_writer.writer_contract.WRITER_KIND_MEASURES`), and its cost is the turns'
  usage rows, each turn's dollars as its client priced them and an unpriced turn as unpriced, never ``0``.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from threetears.evals.analysis.analyst.skill import AnalystSkill, SkillText, load_skill
from threetears.evals.analysis.report.doc_page.build import build
from threetears.evals.analysis.report.doc_page.text import doc_page_text
from threetears.evals.analysis.report_doc_checks import DocVerdict
from threetears.evals.analysis.report_doc_store import doc_verdicts
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.kernel.candidate_kind import (
    CandidateOutput,
    CandidatePreparationFailed,
    CandidateTelemetry,
    CellSink,
    CellSpanWindow,
    VariantConfig,
)
from threetears.evals.kernel.errors import NotFoundError, ValidationFailedError
from threetears.evals.kernel.inner_ceilings import judge_phase_ceiling_s, writer_ceiling_s
from threetears.evals.kernel.failed_call import failed_call_output
from threetears.evals.kernel.provider import describe_failure, log_provider_failure
from threetears.evals.kernel.report_doc import ReportDoc
from threetears.evals.kernel.report_snapshot import EvalReportSnapshot
from threetears.evals.kernel.usage_capture import RoleUsageLedger
from threetears.evals.ops.analyst import report_snapshot_restore, snapshot_input
from threetears.evals.ops.host import OpsHost
from threetears.evals.ops.report_writer.editor import computed_facts
from threetears.evals.ops.report_writer.loop import TurnModel, WriterLimits, WritingOutcome, write_rounds
from threetears.evals.ops.report_writer.session import ToolMount
from threetears.evals.ops.report_writer.tally import ReportBudget, Tally
from threetears.evals.ops.report_writer.writer_contract import (
    ANALYST_ROUNDS_MEASURE,
    BOUNCE_MEASURES,
    EDITOR_ROUNDS_MEASURE,
    ENDING_MEASURE,
    FINISHED_MEASURE,
    MUST_FIX_OPEN_MEASURE,
    SHOULD_FIX_OPEN_MEASURE,
    SKILL_LEVER,
    TURNS_MEASURE,
    UNPRICED_TURNS_MEASURE,
    WriterModels,
    WriterOverlays,
)
from threetears.evals.schema.completion import RequestCeiling
from threetears.evals.schema.logs import get_logger
from threetears.evals.schema.models import EvalTestCase, JudgedArtifact, JudgeEvidence
from threetears.evals.schema.values import SweepableValue

log = get_logger(__name__)

#: The key under which a writer case lives inside ``EvalTestCase.host_payload``.
WRITER_CASE_KEY = "writer_case"

#: Why a turn counted as sent has no usage: its reply never came.
_UNRETURNED = "the turn was sent and its reply never came"

#: The writer case's payload version: a snapshot frozen whole, the notes withheld, and where the case came from. A
#: later shape takes the next version, and a case of a version this build does not know does not read.
WRITER_CASE_VERSION: Final = 1

#: Which side of a case bank's split a case sits on: the cases a change is tuned on, or the ones it is decided on.
WriterSide = Literal["dev", "held_out"]

#: Where a minted case's snapshot came from: a campaign's evidence frozen at mint, a stored analysis's frozen bundle,
#: or a snapshot the host had already frozen.
WriterCaseSource = Literal["campaign", "analysis", "snapshot"]


class WriterCaseOrigin(BaseModel):
    """Where a writer case came from, and the side of the split it was put on: what its mint recorded.

    A case's evidence is its snapshot; this is everything else a reader of a bank asks of one: which host minted it
    and who, from what, and on which side of the split, under which rule. Recorded once, since a stored case never
    changes.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    host_id: str = Field(min_length=1, description="The minting host, as its profile names it (``host_id``).")
    minted_by: str = Field(min_length=1, description="Who minted it, as the host names the caller.")
    source: WriterCaseSource = Field(description="What the mint named: a campaign, a stored analysis or a snapshot.")
    campaign_id: str | None = Field(description="The campaign the mint named; None when it named another source.")
    analysis_id: str | None = Field(description="The analysis the mint named; None when it named another source.")
    group: str = Field(
        min_length=1,
        description="The split group: cases of one group sit on one side. The host's name for it, or the case's "
        "source key when it named none.",
    )
    side: WriterSide = Field(description="The side of the split the case was put on.")
    case_set: str = Field(min_length=1, description="The named case set the mint put it in, the side's.")
    held_out_fraction: float = Field(
        gt=0, lt=1, description="The share of groups the split rule put on the held-out side when it placed this one."
    )
    skill: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="The digest of the shipped skill the withheld notes were checked against at mint.",
    )


class WriterCase(BaseModel):
    """What one writer case pins: the report snapshot it is written over, whole, and the notes its analyst is not given.

    Plain ``BaseModel`` because ``snapshot`` is a document whose strings are data, not input to tidy.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_version: Literal[1] = Field(
        default=WRITER_CASE_VERSION, description="The payload's version (:data:`WRITER_CASE_VERSION`)."
    )
    snapshot: dict[str, Any] = Field(  # Any: the snapshot document's JSON
        description="The report snapshot, frozen whole: ``EvalReportSnapshot.model_dump(mode='json')``."
    )
    withhold: tuple[str, ...] = Field(
        default=(),
        description="The exemplar notes, by file name, the analyst is not given: the note reasoned on this case's own "
        "campaign, for a blind run. Fixed per case, never chosen at launch.",
    )
    origin: WriterCaseOrigin | None = Field(
        default=None,
        description="Where the case came from and the split side its mint put it on; None for a case no mint wrote, "
        "which no bank's split holds.",
    )

    @classmethod
    def of(
        cls, snapshot: EvalReportSnapshot, *, withhold: Collection[str] = (), origin: WriterCaseOrigin | None = None
    ) -> WriterCase:
        """Freeze a stored snapshot into a case.

        :param snapshot: The snapshot the report is written over.
        :ptype snapshot: EvalReportSnapshot
        :param withhold: The notes the analyst is not given.
        :ptype withhold: Collection[str]
        :param origin: Where the case came from, as its mint records it; ``None`` for a case no mint writes.
        :ptype origin: WriterCaseOrigin | None
        :return: The case.
        :rtype: WriterCase
        :raises TypeError: ``withhold`` is a single string rather than a collection of names.
        """
        if isinstance(withhold, str):
            raise TypeError(f"withhold takes a collection of note file names: pass [{withhold!r}], not {withhold!r}")
        return cls(snapshot=snapshot.model_dump(mode="json"), withhold=tuple(withhold), origin=origin)

    @property
    def source_key(self) -> str:
        """What makes two mints one case: the campaign the snapshot froze, or its report input for a bare one.

        :return: ``campaign:<id>``, or ``report-input:<digest>`` for a snapshot with no campaign behind it.
        :rtype: str
        """
        campaign_id = self.snapshot.get("campaign_id")
        if isinstance(campaign_id, str) and campaign_id:
            return f"campaign:{campaign_id}"
        return f"report-input:{self.snapshot.get('report_input_digest')}"


def writer_case_payload(case: WriterCase) -> dict[str, Any]:  # Any: the case's JSON
    """The ``host_payload`` a writer ``EvalTestCase`` carries.

    :param case: The case to store.
    :ptype case: WriterCase
    :return: The payload, keyed so :func:`writer_case_of` can read it back.
    :rtype: dict[str, Any]
    """
    return {WRITER_CASE_KEY: case.model_dump(mode="json")}


def writer_case_of(test_case: EvalTestCase) -> WriterCase | None:
    """Read a writer case back off a stored test case.

    :param test_case: Any stored case.
    :ptype test_case: EvalTestCase
    :return: The case, or ``None`` when this test case carries no writer case.
    :rtype: WriterCase | None
    :raises ValidationError: It carries one that does not read.
    """
    payload = test_case.host_payload
    if not isinstance(payload, dict) or WRITER_CASE_KEY not in payload:
        return None
    return WriterCase.model_validate(payload[WRITER_CASE_KEY])


def writer_subject_components(skill: AnalystSkill) -> dict[str, SweepableValue]:
    """The components a writer arm's subject carries: its skill, under the skill lever's name, by its digest.

    The skill's digest addresses every file's path and text, the analyst's modules and notes and the editor's text
    alike, so one component carries both roles' texts. It is keyed exactly as the lever keys the arm's
    ``skill`` overlay, so the variant identity's derivation accepts it only beside an arm that runs that skill.

    :param skill: The skill the arm runs.
    :ptype skill: AnalystSkill
    :return: ``{SKILL_LEVER: the digest as a level}``.
    :rtype: dict[str, SweepableValue]
    """
    return {SKILL_LEVER: SweepableValue.of(skill.digest, display=f"skill {skill.digest[:12]}")}


def writer_cell_timeout_s(
    limits: WriterLimits,
    *,
    request_s: RequestCeiling,
    turns: int | None,
    judge_dims: int,
    judge_concurrency: int,
    judge_call_attempts: int,
    judge_max_tokens: int,
) -> float:
    """The wall-clock ceiling of one writer cell: every turn the writing loop can send, then the judge phase.

    A writer cell is a whole report, tens of turns run back to back, so the engine's default per-cell deadline cuts it
    off mid-report; the launcher wires this instead (``KindWiring.cell_timeout_s``). The report's ceiling is
    :func:`~threetears.evals.kernel.inner_ceilings.writer_ceiling_s` over the arm's limits and the report budget's turn
    cap, which a host sets to bound it; the judge's is
    :func:`~threetears.evals.kernel.inner_ceilings.judge_phase_ceiling_s`.

    Not covered: restoring the case's snapshot and the tool calls the turns make, which the host's headroom covers.

    :param limits: What the arm's writing loop is held to.
    :ptype limits: WriterLimits
    :param request_s: The host's ceiling for one request on its client, by output cap.
    :ptype request_s: RequestCeiling
    :param turns: The report budget's turn cap, or ``None`` when it declares none.
    :ptype turns: int | None
    :param judge_dims: Rubric dimensions the run's judge scores on each cell.
    :ptype judge_dims: int
    :param judge_concurrency: Judge requests in flight at once.
    :ptype judge_concurrency: int
    :param judge_call_attempts: Requests one judge dimension can make, its parse retries included.
    :ptype judge_call_attempts: int
    :param judge_max_tokens: The judge's output cap.
    :ptype judge_max_tokens: int
    :return: The cell ceiling in seconds.
    :rtype: float
    """
    judging_s = judge_phase_ceiling_s(
        judge_dims=judge_dims,
        judge_concurrency=judge_concurrency,
        judge_call_attempts=judge_call_attempts,
        judge_max_tokens=judge_max_tokens,
        request_s=request_s,
    )
    return writer_ceiling_s(limits, request_s=request_s, turns=turns) + judging_s


@dataclass
class PreparedWriter:
    """One cell's writer: the model its sessions run on, the skill they start from, the limits its loop is held to and
    the cell's tracing windows.

    Per cell rather than on the kind, because one kind instance drives every cell of an arm.
    """

    model: TurnModel
    skill: AnalystSkill
    limits: WriterLimits
    span_window: CellSpanWindow


def _refused(message: str) -> CandidatePreparationFailed:
    """A cell this kind cannot run, logged and excluded before anything is sent."""
    log.error("Report writer cell refused: %s", message)
    return CandidatePreparationFailed(f"apparatus: {message}", termination="factory_failed")


class ReportWriterKind:
    """One writer arm: writes a report over each case's snapshot with the arm's skill, effort and revisions.

    One instance per arm, built by the launcher; a cell's state is what :meth:`prepare` returns and the tally
    :meth:`invoke` keeps.
    """

    #: The final page, judged against the facts code computed from its snapshot.
    judged_artifact = JudgedArtifact.DOCUMENT
    #: No limit of the kind's own: each cell's model comes from the factory, its tally and budget are its own, and the
    #: mounts read a frozen snapshot.
    max_concurrent_cells: int | None = None

    def __init__(
        self,
        *,
        host: OpsHost,
        scope_id: str,
        overlays: WriterOverlays,
        models: WriterModels,
        budget: ReportBudget,
        analyst_mount: ToolMount,
        editor_mount: ToolMount,
        skills: Mapping[str, AnalystSkill] = MappingProxyType({}),
    ) -> None:
        """Bind the arm's levers and what its cells are written with.

        :param host: The host whose store the analyst tools read, and whose failure describer reads what a model
            client raises.
        :ptype host: OpsHost
        :param scope_id: The scope the mounts read: each case's snapshot is restored there.
        :ptype scope_id: str
        :param overlays: The arm's overlays, as the launch validated and froze them.
        :ptype overlays: WriterOverlays
        :param models: The host's model factory: the arm's candidate model at a reasoning effort.
        :ptype models: WriterModels
        :param budget: Each report's budget; every cell is written under a fresh one of the same caps.
        :ptype budget: ReportBudget
        :param analyst_mount: The ``evals_analyst`` tool on the analyst's cut, read-only, its gap note storing nothing.
        :ptype analyst_mount: ToolMount
        :param editor_mount: The tool on the editor's cut, bound the same way.
        :ptype editor_mount: ToolMount
        :param skills: The skill versions the launch may run, by digest. The shipped skill is always one of them.
        :ptype skills: Mapping[str, AnalystSkill]
        :raises ValueError: A skill is keyed by a digest other than its own.
        """
        if misfiled := sorted(digest for digest, skill in skills.items() if skill.digest != digest):
            raise ValueError(f"skills are keyed by their own digest, and {', '.join(misfiled)} is not")
        shipped = load_skill()
        self._skills: dict[str, AnalystSkill] = {shipped.digest: shipped, **skills}
        self._host = host
        self._scope_id = scope_id
        self._overlays = overlays
        self._models = models
        self._budget = budget
        self._analyst_mount = analyst_mount
        self._editor_mount = editor_mount

    async def prepare(
        self,
        *,
        subject_snapshot: Any,
        variant_config: VariantConfig,
        world_seed: Any,
        span_window: CellSpanWindow,
        cassettes: Any,
        world: Any,
    ) -> PreparedWriter:
        """Bind this cell's skill and model, refusing a cell that would measure another arm than the one it records.

        :param subject_snapshot: The arm's subject, which must carry the skill lever's component, at the arm's skill.
        :ptype subject_snapshot: SubjectSnapshot | None
        :param variant_config: This cell's contestant stack; its model is the one the factory must build.
        :ptype variant_config: VariantConfig
        :param world_seed: Unread: a writer acts on no world.
        :ptype world_seed: Any
        :param span_window: This cell's tracing windows, opened in :meth:`invoke`.
        :ptype span_window: CellSpanWindow
        :param cassettes: Unwired: the writer's turns are recorded by its model (``turn_recording``), not a cassette.
        :ptype cassettes: CellCassettes | None
        :param world: Unread: a writer acts on no world.
        :ptype world: Any
        :return: The cell's model, skill and windows.
        :rtype: PreparedWriter
        :raises CandidatePreparationFailed: The arm's revisions are not ones the loop runs; the arm's skill is not one
            the launch was given; the subject does not carry it; the factory's model is not the cell's; or, under a
            dollar budget, the model's turns cannot be priced. Each before anything is sent.
        """
        try:
            limits = WriterLimits(revisions=self._overlays.revisions)
        except ValueError as refused:
            raise _refused(f"the arm's settings are not ones the writing loop runs: {refused}") from refused
        digest = self._overlays.skill
        skill = self._skills.get(digest)
        if skill is None:
            raise _refused(
                f"the arm runs skill {digest}, which this launch was not given; the skills it may run are "
                f"{', '.join(sorted(self._skills))}"
            )
        carried = subject_snapshot.components.get(SKILL_LEVER) if subject_snapshot is not None else None
        expected = writer_subject_components(skill)[SKILL_LEVER]
        if carried is None or carried.content_hash != expected.content_hash:
            raise _refused(
                f"the arm runs skill {digest}, and its subject carries "
                + ("no skill" if carried is None else f"another ({carried.display})")
                + ": capture the subject with writer_subject_components of the skill the arm runs"
            )
        try:
            model = self._models(self._overlays.reasoning_effort)
        except ValueError as refused:
            raise _refused(f"the host's model factory refused the arm's settings: {refused}") from refused
        if model.model_name != variant_config.candidate_model:
            raise _refused(
                f"this cell is recorded against model {variant_config.candidate_model!r} but the model factory built "
                f"{model.model_name!r}, so running it would report one candidate's page under another's name"
            )
        if self._budget.usd is not None and (unpriced := await model.turn_pricing_refusal()) is not None:
            raise _refused(f"its {self._budget.spelled} budget admits each turn on its price ceiling, and {unpriced}")
        return PreparedWriter(model=model, skill=skill, limits=limits, span_window=span_window)

    async def invoke(self, instance: PreparedWriter, test_case: EvalTestCase, sink: CellSink) -> CandidateOutput:
        """Write one report over the case's snapshot, and hand the judge its final page.

        Never raises. A case that does not read, a withheld note the arm's skill has no note of, and a snapshot that
        cannot be restored are the harness's (``infra_errors``, the cell excluded), each before any turn. A loop that
        ends with no document (refused, its review unreadable, its budget spent before a review) is the
        candidate's (``candidate_errors``); one whose client failed is the account's or the provider's when the host's
        describer says so, as a reporter cell's is, and the candidate's otherwise. Every turn's spend is on the
        output's usage and on the sink's reading, so a cell cut off at its deadline keeps what it paid for.

        :param instance: What :meth:`prepare` bound.
        :ptype instance: PreparedWriter
        :param test_case: The cell's case, carrying the writer case in its ``host_payload``.
        :ptype test_case: EvalTestCase
        :param sink: This cell's sink, handed the tally's reading.
        :ptype sink: CellSink
        :return: The final page, its judge evidence, the run's measures and its usage.
        :rtype: CandidateOutput
        """
        with instance.span_window.identity():
            prepared = await self._case_input(instance, test_case)
            if isinstance(prepared, str):
                log.error("Report writer cell cannot write: %s", prepared)
                return CandidateOutput(infra_errors=[prepared])
            snapshot_id, report_input, analyst_skill, editor_skill = prepared
            tally = Tally()
            model = instance.model
            sink.report_progress(lambda: CandidateOutput(telemetry=_telemetry(tally, model.model_name)))

            def check(doc: ReportDoc) -> list[DocVerdict]:
                return doc_verdicts(doc, report_input)

            with instance.span_window.collecting():
                outcome = await write_rounds(
                    model=model,
                    snapshot=snapshot_id,
                    report_input=report_input,
                    analyst_mount=self._analyst_mount,
                    editor_mount=self._editor_mount,
                    analyst_skill=analyst_skill,
                    editor_skill=editor_skill,
                    check=check,
                    tally=tally,
                    budget=ReportBudget(usd=self._budget.usd, tokens=self._budget.tokens, turns=self._budget.turns),
                    limits=instance.limits,
                )
        return self._output(outcome, tally, instance, report_input, analyst_skill, editor_skill)

    async def _case_input(
        self, instance: PreparedWriter, test_case: EvalTestCase
    ) -> tuple[str, ReportInput, SkillText, SkillText] | str:
        """The snapshot id, its report input and both roles' texts a cell writes with; or the infra error, with no
        turn sent."""
        try:
            case = writer_case_of(test_case)
        except ValidationError as unread:
            return f"apparatus: test case {test_case.id} carries an unreadable writer case: {unread}"
        if case is None:
            return f"apparatus: test case {test_case.id} carries no writer case under host_payload[{WRITER_CASE_KEY!r}]"
        try:
            analyst_skill = instance.skill.analyst(withhold=case.withhold)
        except ValueError as unknown:
            # The case withholds a note this arm's skill has none of: the blind run the case asks for cannot be run.
            return (
                f"apparatus: test case {test_case.id} withholds a note skill {instance.skill.digest} lacks: {unknown}"
            )
        try:
            snapshot = EvalReportSnapshot.model_validate(case.snapshot)
        except ValidationError as unread:
            return (
                f"apparatus: test case {test_case.id}'s frozen snapshot does not read under this build "
                f"({unread.error_count()} fault(s)); freeze the case again"
            )
        try:
            restored = await report_snapshot_restore(self._host, self._scope_id, snapshot)
            report_input = await snapshot_input(self._host, self._scope_id, restored.snapshot_id)
        except (NotFoundError, ValidationFailedError) as unusable:
            return f"apparatus: test case {test_case.id}'s frozen snapshot cannot be used: {unusable}"
        return restored.snapshot_id, report_input, analyst_skill, instance.skill.editor()

    def _output(
        self,
        outcome: WritingOutcome,
        tally: Tally,
        instance: PreparedWriter,
        report_input: ReportInput,
        analyst_skill: SkillText,
        editor_skill: SkillText,
    ) -> CandidateOutput:
        """The cell's output for how the loop ended: the page and its evidence when a document was stored."""
        telemetry = _telemetry(tally, instance.model.model_name)
        measures = _measures(outcome, tally)
        payload: dict[str, JsonValue] = {
            "skill": instance.skill.digest,
            "analyst_text": analyst_skill.digest,
            "editor_text": editor_skill.digest,
            "ending": outcome.ending,
            "reason": outcome.reason,
            "budget": self._budget.spelled,
        }
        if outcome.doc is not None:
            page = doc_page_text(build(outcome.doc, report_input, strict=True, findings=outcome.findings, made=""))
            return CandidateOutput(
                output=[outcome.doc.model_dump(mode="json", by_alias=True)],
                judge_evidence=JudgeEvidence(
                    case_material="Computed facts, from the snapshot the report was written over. Code computed each; "
                    "none is a verdict:\n" + computed_facts(outcome.doc, report_input),
                    artifact=page,
                ),
                telemetry=telemetry,
                host_measures=measures,
                kind_payload=payload,
            )
        if outcome.ending == "failed" and outcome.failure is not None:
            return self._failed(outcome.failure, telemetry, measures, payload)
        return CandidateOutput(
            candidate_errors=[f"report_writer: the writing ended {outcome.ending} with no report: {outcome.reason}"],
            telemetry=telemetry,
            host_measures=measures,
            kind_payload=payload,
        )

    def _failed(
        self,
        failure: Exception,
        telemetry: CandidateTelemetry,
        measures: dict[str, bool | float | str],
        payload: dict[str, JsonValue],
    ) -> CandidateOutput:
        """A loop the client's failure ended, split by whose failure the host's describer says it was."""
        described = describe_failure(
            self._host.eval_host.failure_describer, failure, logger=log, where="Report writer turn"
        )
        return failed_call_output(
            described.owner,
            described.description,
            refused="report_writer: a turn was refused for the calling account",
            declined="report_writer: a turn was declined by the provider",
            failed="report_writer: a turn failed",
            log=lambda what, level: log_provider_failure(log, described, failure, "%s", what, level=level),
            telemetry=telemetry,
            host_measures=measures,
            kind_payload=payload,
        )


def _telemetry(tally: Tally, model: str) -> CandidateTelemetry:
    """Every turn the cell sent, as the candidate role's usage: priced as its client priced it, an unpriced turn and a
    turn whose reply never came as unpriced, with why."""
    ledger = RoleUsageLedger(role="candidate")
    for turn in tally.turns:
        ledger.add(
            model=turn.model,
            served_model=turn.served_model,
            prompt_tokens=turn.input_tokens,
            completion_tokens=turn.output_tokens,
            reasoning_tokens=turn.reasoning_tokens,
            cost_usd=turn.cost_usd,
            price_source=turn.price_source,
            unpriced_reason=turn.pricing_failure,
        )
    for _ in range(tally.unreturned_turns()):
        ledger.add(
            model=model,
            prompt_tokens=None,
            completion_tokens=None,
            reasoning_tokens=None,
            cost_usd=None,
            unpriced_reason=_UNRETURNED,
        )
    return CandidateTelemetry(usage=ledger.rows(), turns_delivered=len(tally.turns))


def _measures(outcome: WritingOutcome, tally: Tally) -> dict[str, bool | float | str]:
    """The run's measures, read off the loop's outcome and the tally; the open findings only where a document was
    stored, since a run that stored none has no findings open on anything."""
    measures: dict[str, bool | float | str] = {
        ENDING_MEASURE: outcome.ending,
        FINISHED_MEASURE: outcome.ending == "stored",
        **{BOUNCE_MEASURES[kind]: float(count) for kind, count in tally.bounces().items()},
        ANALYST_ROUNDS_MEASURE: float(outcome.analyst_rounds),
        EDITOR_ROUNDS_MEASURE: float(outcome.editor_rounds),
        TURNS_MEASURE: float(tally.sent),
        UNPRICED_TURNS_MEASURE: float(tally.unpriced_turns()),
    }
    if outcome.doc is not None:
        measures[MUST_FIX_OPEN_MEASURE] = float(sum(1 for f in outcome.findings if f.severity == "must_fix"))
        measures[SHOULD_FIX_OPEN_MEASURE] = float(sum(1 for f in outcome.findings if f.severity == "should_fix"))
    return measures


__all__ = [
    "ReportWriterKind",
    "WriterCase",
    "WriterCaseOrigin",
    "WriterCaseSource",
    "WriterSide",
    "writer_case_of",
    "writer_case_payload",
    "writer_cell_timeout_s",
    "writer_subject_components",
]
