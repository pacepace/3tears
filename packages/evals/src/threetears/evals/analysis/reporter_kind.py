"""The analysis reporter as a candidate kind — its memos judged like any other LLM output.

Code checks the STRUCTURE of an analysis and never its prose; whether the
prose is grounded, useful, insightful, relevant, calibrated and readable is measured here, by
evaluating the reporter the way every other candidate is evaluated: its own cases, rubric and
judge, through the ordinary run path.

- **A case** is one campaign's analysis bundle, FROZEN into the case's ``host_payload``. Bundles
  are re-assembled on demand everywhere else, and the schema moves, so without the freeze a case
  would silently change under every candidate scored on it. The bundle already carries the
  campaign's declaration as the aspect reads it (``declared``: its question and bars).
- **A candidate** is a (prompt preset, generator model) pair — :class:`ReporterKind` — or the
  memo the campaign actually got — :class:`AsRecordedReporterKind` — so calibrating the rubric
  against a reviewer's read of a stored memo goes through the same path and judge as every later
  candidate.
- **The judge** sees the evidence each memo's own writer read — a generated memo the message its
  generator was sent, the recorded memo the writer message the case froze
  (:func:`judge_case_material` decides which) — and the memo as the model WROTE it (its words, in
  one canonical reader's layout), never as a surface rendered it — a rendering defect is the
  surface's, not the reporter's.
- **Labels** — a reviewer's written read of a recorded memo, mapped to a rubric dimension and
  carrying the criterion text it was written against — ride on the case for the calibration
  comparison and are never shown to the judge.
- **A plan case** (:data:`PLAN_CASE_VERSION`) judges what the report planner wrote. It freezes the bundle frozen with
  the planned analysis, the report input built from it, the plan in key form and the planner's prompt version; the
  judged text is the plan with every figure key written out against that input (:func:`render_plan_for_judge`, as
  the campaign's page writes it), and the judge's material what the planner read of it
  (:func:`~threetears.evals.analysis.planner.plan_user_message`). Both are frozen as text in the fields a memo case
  uses (``recorded_memo``, ``writer_message``), so the replay, the excerpt check and the writer-message check read the
  two kinds alike. A memo case (:data:`MEMO_CASE_VERSION`, every case stored before) is read and judged as it was.

Generic by construction: nothing here names a subject's shape, a product or a domain. It implements
the run-side ``CandidateKind`` seam from inside the analysis package. Where it belongs is already
decided — the kind goes runner-side, while the case model and the memo rendering stay in analysis —
and moving it is the remaining half of that split.
"""

from __future__ import annotations

import json
import logging
import re
import types
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Annotated, Any, Final, Literal, Union, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field, model_validator

from threetears.evals.analysis.campaign_bundle import (
    CampaignBundle,
    build_campaign_user_message,
    campaign_evidence_digest,
)
from threetears.evals.analysis.errors import GenerationError
from threetears.evals.analysis.generator import (
    first_request,
    generate_analysis,
    refuse_an_undescribable_arm_table,
    user_message_digest,
)
from threetears.evals.analysis.planned_analysis import report_input_document_digest
from threetears.evals.analysis.report.figure_keys import render
from threetears.evals.analysis.report.md_parts import markdown_table, table_lines
from threetears.evals.analysis.report.plan_page.drawable import plan_look
from threetears.evals.analysis.report.plan_page.tables import arms_key
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.analysis.report_plan_facts import plan_facts
from threetears.evals.analysis.report.words import (
    CONFIDENCE_WORDS,
    evidence_columns,
    arm_namer,
    evidence_rows,
    positions,
    stands_on_words,
)
from threetears.evals.kernel.campaign import ConfidenceTier, EvalAnalysis, FindingResolution
from threetears.evals.kernel.candidate_kind import (
    CandidateOutput,
    CandidatePreparationFailed,
    CandidateTelemetry,
    CellSink,
    CellSpanWindow,
    VariantConfig,
)
from threetears.evals.kernel.capped_completion import CappedCompletion, call_ceiling
from threetears.evals.kernel.errors import RefusedInput
from threetears.evals.kernel.cassettes import CellCassettes
from threetears.evals.kernel.inner_ceilings import judge_phase_ceiling_s
from threetears.evals.kernel.report_plan import PlannedReport, ReportPlan
from threetears.evals.kernel.host.eval_host import EvalHost
from threetears.evals.schema.models import (
    DimName,
    EvalTestCase,
    JudgedArtifact,
    JudgeEvidence,
    RubricDim,
    RubricScale,
)
from threetears.evals.schema.completion import CompletionGenerator, CompletionResult, RequestCeiling
from threetears.evals.kernel.provider import describe_failure, log_provider_failure
from threetears.evals.kernel.text_block import one_line, under
from threetears.evals.kernel.usage_capture import CallUsage, RoleUsageLedger
from threetears.evals.schema.logs import contain, get_logger
from threetears.evals.schema.prose import field_is_prose

log = get_logger(__name__)

#: The ``candidate_kind`` a reporter template declares.
REPORTER_KIND = "analysis_reporter"

#: The candidate model name the as-recorded candidate runs under. Not a provider model — a
#: reporter run launched with it replays each case's recorded memo and calls no generator.
AS_RECORDED_MODEL = "as-recorded"

#: The key under which a reporter case lives inside ``EvalTestCase.host_payload``.
REPORTER_CASE_KEY = "reporter_case"

#: The reporter kind's version for a case judging a MEMO: the memo writer's output, rendered as written
#: (:func:`render_memo_as_written`) and judged against the writer message the case froze. Every case stored before
#: plans were judged is one, and is read and judged exactly as it was; it is never migrated.
MEMO_CASE_VERSION: Final = 1

#: The reporter kind's version for a case judging a PLAN: the report planner's output, its figure keys written out
#: against the report input the case froze (:func:`render_plan_for_judge`), judged against what the planner read
#: (:func:`~threetears.evals.analysis.planner.plan_user_message`). The subject changed, so the version did: a plan
#: case's scores are never pooled with a memo case's (:func:`~threetears.evals.analysis.reporter_bank.read_calibration`).
PLAN_CASE_VERSION: Final = 2

#: Every reporter kind version this build reads, oldest first.
ReporterKindVersion = Literal[1, 2]

#: The limit a case states when the message it freezes is not the one its recorded writer was sent, by what the case
#: judges (``ReporterCase.kind_version``): a memo's writer read a rendered bundle, a plan's planner its report input's
#: view. ``{recorded}`` is the digest the generation recorded.
WRITER_MESSAGE_DIFFERS: Mapping[ReporterKindVersion, str] = types.MappingProxyType(
    {
        MEMO_CASE_VERSION: (
            "The bundle reproduces the one the recorded memo was written from, but the writer message frozen here "
            "is not the one its generator was sent (recorded digest {recorded}): how a bundle is rendered changed "
            "between the generation and this freeze. The memo is judged against a message its writer did not read, "
            "so a claim resting on something only the older rendering showed scores as ungrounded here."
        ),
        PLAN_CASE_VERSION: (
            "The planner's message frozen here is not the one the planner was sent (recorded digest {recorded}): how "
            "the planner is shown its report input changed between the generation and this freeze, or the input "
            "itself did, which a limit on the report input then states. The plan is judged against a message its "
            "planner did not read, so a claim resting on something only the older message showed scores as "
            "ungrounded here."
        ),
    }
)


#: Requests the memo writer's generation can make, which a reporter cell measures: the memo, and one repair
#: round-trip a refused output buys (:func:`~threetears.evals.analysis.generator.generate_analysis`).
MEMO_WRITER_CALLS = 2


def reporter_cell_timeout_s(
    *,
    judge_dims: int,
    judge_concurrency: int,
    generator_max_tokens: int,
    judge_call_attempts: int,
    judge_max_tokens: int,
    request_s: RequestCeiling,
) -> float:
    """The wall-clock ceiling of one reporter cell, derived from the ceilings it wraps.

    A reporter cell is one memo writer's generation, :data:`MEMO_WRITER_CALLS` requests at the generator's cap (an
    analysis generation is the planner's, :func:`~threetears.evals.kernel.inner_ceilings.generation_ceiling_s`, and
    makes more), then its rubric's judge phase
    (:func:`judge_phase_ceiling_s`). Every request in either is bounded by the host's own answer for one
    request at its output cap (``request_s``), so the ceiling sits above the slowest finishing cell
    rather than at a sampled duration: the engine's generic 600s cancelled a generation that was still
    writing inside its own output cap.

    Every operational input is a parameter, so the host passes the values its launch read and the
    ceiling is derived from the same generator cap the run's clients are built with. The judge's limits
    are parameters too: they belong to the judge, which the run package owns, and analysis imports
    nothing from run.

    :param judge_dims: Rubric dimensions the run's judge scores on each cell.
    :ptype judge_dims: int
    :param judge_concurrency: Judge requests in flight at once.
    :ptype judge_concurrency: int
    :param generator_max_tokens: The generator's output cap, as its clients are built with.
    :ptype generator_max_tokens: int
    :param judge_call_attempts: Requests one judge dimension can make, its parse retries included.
    :ptype judge_call_attempts: int
    :param judge_max_tokens: The judge's output cap, as its requests are built with.
    :ptype judge_max_tokens: int
    :param request_s: The host's ceiling for one request on its client, by output cap.
    :ptype request_s: RequestCeiling
    :return: The cell ceiling in seconds: the generation's ceiling plus the judge phase's.
    :rtype: float
    """
    generation_s = MEMO_WRITER_CALLS * request_s(generator_max_tokens)
    judging_s = judge_phase_ceiling_s(
        judge_dims=judge_dims,
        judge_concurrency=judge_concurrency,
        judge_call_attempts=judge_call_attempts,
        judge_max_tokens=judge_max_tokens,
        request_s=request_s,
    )
    return generation_s + judging_s


#: Where a person reading a memo places it on a rubric dimension, low to high.
LabelDirection = Literal["low", "low_mid", "mid", "mid_high", "high"]

#: What a person reading a memo says of it on a pass/fail rubric dimension.
LabelVerdict = Literal["pass", "fail"]

#: The 1-5 judge scores each label direction agrees with, inclusive. A calibration read counts an
#: ordinal dimension as agreeing when the judge's score falls inside its label's band.
LABEL_BANDS: dict[LabelDirection, tuple[int, int]] = {
    "low": (1, 2),
    "low_mid": (2, 3),
    "mid": (3, 3),
    "mid_high": (3, 4),
    "high": (4, 5),
}


class LabelCriterion(BaseModel):
    """The rubric criterion a label was written against, frozen as text when its case was frozen.

    A label says where a memo sits on a dimension, and "where" is only meaningful against the
    words that define the dimension. A template's rubric is store-mastered and editable, so
    without this copy a reworded criterion would silently change what every stored label means.
    Plain ``BaseModel`` because the words are the rubric author's, verbatim.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    description: str = Field(description="The dimension's description, as the template stated it at freeze time.")
    scale: RubricScale = Field(
        description=(
            "How the dimension was answered at freeze time: 'ordinal' (1-5) or 'pass_fail'. A stored criterion naming "
            "any other scale is refused on read, so its case is listed as unreadable by name rather than read as one "
            "it is not."
        )
    )
    scoring_guide: dict[str, str] = Field(
        default_factory=dict,
        description="The scoring guide, level to descriptor, at freeze time. Empty when the dimension had none.",
    )

    @classmethod
    def of(cls, dim: RubricDim) -> LabelCriterion:
        """Copy the criterion a rubric dimension states.

        :param dim: The template's dimension.
        :ptype dim: RubricDim
        :return: Its description, scale and scoring guide.
        :rtype: LabelCriterion
        """
        return cls(description=dim.description, scale=dim.scale, scoring_guide=dict(dim.scoring_guide))


class ReporterLabel(BaseModel):
    """One reader's written verdict on a recorded memo, mapped to the dimension it bears on.

    Its form is its dimension's scale: a ``direction`` on an ordinal (1-5) dimension, never a number, since the reads
    these come from never gave one; a ``verdict`` of pass or fail on a pass/fail dimension. Exactly one is set. The
    freeze refuses a form that does not match the scale its criterion states; a label stored before verdicts existed
    holds a direction and reads as it did. Plain ``BaseModel`` because ``quote`` and ``memo_excerpt`` are words
    verbatim.

    A verdict is read as a person's (the judge's agreement with readers), so the ``reporter_case_freeze``
    operation names its ``reader`` only as the host attests them (``EvalHost.attests_person``) and records the caller
    that relayed it (``collected_by``). A host's own code freezing through ``freeze_reporter_case`` is trusted to
    name its readers, or none. Neither field is part of what makes two freezes the same case, and neither is written
    to the stored payload while None, so a label stored before them reads, and is stored, as it was.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    dimension: DimName = Field(
        description="The rubric dimension name the verdict bears on, e.g. 'reporter.groundedness'."
    )
    direction: LabelDirection | None = Field(
        default=None,
        description="Where the reader places the memo on an ordinal dimension; None on a label carrying a verdict.",
    )
    verdict: LabelVerdict | None = Field(
        default=None,
        description="Whether the reader passes or fails the memo on a pass/fail dimension; None on a label carrying a "
        "direction.",
    )
    quote: str = Field(description="The reader's own words, verbatim — the evidence for the direction or verdict.")
    memo_excerpt: str | None = Field(
        default=None,
        description=(
            "The passage of the recorded memo the verdict is about, verbatim. The freeze refuses one that is not in "
            "the case's recorded memo, compared after collapsing each run of whitespace to one space on both sides "
            "and trimming the excerpt's ends. None = the label quotes no passage. No part of the verdict: a re-freeze "
            "quoting other passages than the pair's live case is refused unless it supersedes that case."
        ),
    )
    note: str | None = Field(default=None, description="Why the quote maps to this dimension, when not self-evident.")
    reader: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Who gave the verdict, as the host names them: attested by the host for a verdict relayed through the "
            "reporter_case_freeze action; None = not recorded (a host's own freeze naming no one, or a label stored "
            "before readers were). Not part of what makes two freezes the same case."
        ),
    )
    collected_by: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "The calling identity that relayed the verdict, which the host attested for its reader; None = written by "
            "the host's own code, or stored before relays were recorded. Not part of what makes two freezes the same "
            "case."
        ),
    )
    criterion: LabelCriterion | None = Field(
        default=None,
        description=(
            "The criterion the verdict was written against, stamped by the freeze from the template's rubric — a "
            "caller never supplies it, and a freeze refuses a label that arrives with one. None only on a label as "
            "a caller submits it: every label a case stores carries one (``ReporterCase`` refuses one that does not)."
        ),
    )

    @model_validator(mode="after")
    def _a_relay_names_its_reader(self) -> ReporterLabel:
        """Refuse a label holding both a direction and a verdict, or neither; a blank reader, relay or excerpt; and a
        relay with no reader.

        ``quote`` and ``memo_excerpt`` are kept verbatim, so this model strips nothing; a name that is only whitespace
        names no one, and an excerpt that is only whitespace quotes nothing. A relay records who carried a named
        reader's verdict, so it has no meaning without one, as a rating's ``source`` and ``collected_by`` are set
        together.

        :raises ValueError: ``direction`` and ``verdict`` are both set or both unset; ``reader``, ``collected_by`` or
            ``memo_excerpt`` is blank; or ``collected_by`` is set without ``reader``.
        """
        if (self.direction is None) == (self.verdict is None):
            held = "both a direction and a verdict" if self.direction is not None else "neither"
            raise ValueError(
                f"a reporter label on {self.dimension!r} holds a direction (an ordinal dimension) or a verdict (a "
                f"pass/fail one), exactly one, and this one holds {held}"
            )
        for name, value in (("reader", self.reader), ("collected_by", self.collected_by)):
            if value is not None and not value.strip():
                raise ValueError(f"a reporter label's {name} names who it is, and {value!r} is blank")
        if self.memo_excerpt is not None and not self.memo_excerpt.strip():
            raise ValueError(f"a reporter label's memo_excerpt quotes a passage, and {self.memo_excerpt!r} is blank")
        if self.collected_by is not None and self.reader is None:
            raise ValueError(
                f"a reporter label relayed by {self.collected_by!r} names the reader whose verdict it relays"
            )
        return self


class ReporterCase(BaseModel):
    """What one reporter case pins: a frozen bundle, optionally the memo (or plan) it got, and its labels.

    ``kind_version`` says which: a memo case freezes no plan field; a plan case freezes every one, with its judged text
    in ``recorded_memo`` and what its planner read in ``writer_message``.

    Plain ``BaseModel`` because ``bundle`` and ``recorded_memo`` are documents whose strings are
    data, not input to tidy.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    bundle: dict[str, Any] = Field(
        description="The campaign's bundle, frozen whole, every aspect in it: ``CampaignBundle.to_dict()``."
    )
    bundle_fingerprint: str = Field(
        description="The frozen bundle's fingerprint, so a case can be matched and re-checked."
    )
    bundle_assembled_at: str = Field(
        description="When the frozen bundle was assembled (ISO-8601) — the insight ledger's as-of."
    )
    recorded_analysis_id: str | None = Field(
        default=None,
        description="The id of the analysis the campaign actually got, or ``None`` when the case pins none.",
    )
    recorded_memo: str | None = Field(
        default=None,
        description=(
            "That analysis's memo as the judge reads it, frozen as TEXT when the case is frozen — so a label stays "
            "attached to exactly the memo it judged, whatever shape a stored analysis later takes."
        ),
    )
    writer_message: str | None = Field(
        default=None,
        description=(
            "The user message the recorded memo is judged against, frozen as TEXT — so a change to how a bundle is "
            "rendered cannot move the evidence under a label. Rendered from the frozen bundle when the case was "
            "frozen, because a stored analysis keeps no copy of the message its writer was sent — only a digest of "
            "it (``recorded_writer_message_digest``), against which the freeze checks this text "
            "(:func:`writer_message_check`). Generated candidates never read it — each is judged against the "
            "message it was itself sent. None exactly when the case pins no memo."
        ),
    )
    recorded_writer_message_digest: str | None = Field(
        default=None,
        description=(
            "The digest of the user message the recorded memo's generator was sent, copied at freeze time from its "
            "analysis (``GenerationProvenance.user_message_digest``). Held so whether ``writer_message`` is what that "
            "writer read stays checkable from the case alone (:func:`writer_message_check`). None exactly when the "
            "case pins no memo."
        ),
    )
    labels: list[ReporterLabel] = Field(
        default_factory=list, description="Reader verdicts on the recorded memo. Never shown to the judge."
    )
    limits: list[str] = Field(
        default_factory=list,
        description="What this case cannot test, stated when frozen (e.g. arms that re-assemble without their levels).",
    )
    supersedes: list[str] = Field(
        default_factory=list,
        description=(
            "The stored test-case ids this case replaces — cases of the same campaign and recorded memo, frozen "
            "with other labels or from another re-assembly of the bundle, normally exactly one. More than one only "
            "when it resolves a pair that held several live cases. Each replaced case is kept, so runs measured "
            "against it still read it, and no launch runs it again."
        ),
    )
    kind_version: ReporterKindVersion = Field(
        default=MEMO_CASE_VERSION,
        description=(
            "What the case judges: 1, a memo (every case stored before plans were judged); 2, a report plan. A plan "
            "case's ``recorded_memo`` is the plan rendered for the judge (:func:`render_plan_for_judge`), and its "
            "``writer_message`` what the planner read (``plan_user_message``) over the frozen ``report_input``."
        ),
    )
    report_input: dict[str, Any] | None = Field(
        default=None,
        description=(
            "A plan case's report input, frozen whole (``ReportInput`` JSON by alias), built from the frozen bundle: "
            "what the plan's figure keys resolve against and what the judge's material is rendered from. None on a "
            "memo case."
        ),
    )
    report_input_digest: str | None = Field(
        default=None,
        description="The frozen report input's digest (``report_input_digest``), re-checked on every read. None on a memo case.",
    )
    recorded_report_input_digest: str | None = Field(
        default=None,
        description=(
            "The digest of the report input the recorded plan was written against, copied from its stored plan "
            "(``EvalReportPlan.report_input_digest``), so whether the frozen input is that one stays checkable from "
            "the case alone. None on a memo case."
        ),
    )
    recorded_plan: dict[str, Any] | None = Field(
        default=None,
        description=(
            "The recorded plan in key form, frozen whole (``ReportPlan`` JSON by alias): its prose with figure keys, "
            "not numbers. None on a memo case."
        ),
    )
    planner_prompt_version: str | None = Field(
        default=None,
        description=(
            "The prompt version the recorded plan's planner ran (``ReportPlanProvenance.prompt_version``): what a "
            "calibration groups plan cases by beside the kind version. None on a memo case."
        ),
    )

    @model_validator(mode="after")
    def _a_recorded_memo_travels_with_what_its_writer_read(self) -> ReporterCase:
        """Refuse a case whose recorded memo, writer message and message digest are not all present or all absent.

        The freeze writes the three together from one analysis: the memo, the message its writer was
        sent (rendered from the frozen bundle) and the digest the generation recorded of it. A case
        holding one without the others is not a shape any freeze produces — a memo with no frozen
        message would be judged against whatever the running build renders, which is the evidence
        moving under its labels; a message with no memo describes a writer that does not exist.
        Every label must carry the criterion the freeze stamped, for the same reason: a label with
        none says nothing about what it was written against.
        """
        present = {
            "recorded_memo": self.recorded_memo is not None,
            "writer_message": self.writer_message is not None,
            "recorded_writer_message_digest": self.recorded_writer_message_digest is not None,
        }
        if len(set(present.values())) > 1:
            held = ", ".join(name for name, has in present.items() if has)
            missing = ", ".join(name for name, has in present.items() if not has)
            raise ValueError(
                f"a reporter case pins a recorded memo, the writer message it was written from and that message's "
                f"digest together or not at all; this case holds {held} without {missing}"
            )
        if unstamped := [label.dimension for label in self.labels if label.criterion is None]:
            raise ValueError(
                f"every label a reporter case stores carries the criterion its freeze stamped; the label(s) on "
                f"{', '.join(unstamped)} carry none"
            )
        planned = {
            "report_input": self.report_input is not None,
            "report_input_digest": self.report_input_digest is not None,
            "recorded_report_input_digest": self.recorded_report_input_digest is not None,
            "recorded_plan": self.recorded_plan is not None,
            "planner_prompt_version": self.planner_prompt_version is not None,
            "recorded_memo": self.recorded_memo is not None,
        }
        if self.kind_version == PLAN_CASE_VERSION and not all(planned.values()):
            missing = ", ".join(name for name, has in planned.items() if not has)
            raise ValueError(
                f"a plan reporter case (kind version {PLAN_CASE_VERSION}) freezes its recorded plan, the report input "
                f"it resolves against, their digests, the planner's prompt version and the plan as the judge reads it "
                f"together; this case lacks {missing}"
            )
        if self.report_input is not None and self.report_input_digest is not None:
            # Checked on the raw document, never through the running build's ReportInput: the frozen input is the
            # evidence the labels were written against, and a schema move must not make a case unreadable.
            if (held_digest := report_input_document_digest(self.report_input)) != self.report_input_digest:
                raise ValueError(
                    f"the case's frozen report input digests to {held_digest} and the case recorded "
                    f"{self.report_input_digest}: it does not hold the input it names"
                )
        if self.kind_version == MEMO_CASE_VERSION and (
            stray := [name for name, has in planned.items() if has and name != "recorded_memo"]
        ):
            raise ValueError(
                f"a memo reporter case (kind version {MEMO_CASE_VERSION}) freezes no plan, and this case holds "
                f"{', '.join(stray)}"
            )
        return self


#: Whether a case's frozen writer message is the one its recorded memo's generator was sent — see
#: :func:`writer_message_check`.
WriterMessageCheck = Literal["verified", "differs"]


def message_check(writer_message: str, recorded_digest: str) -> WriterMessageCheck:
    """Compare a writer message about to be frozen with the digest its generation recorded.

    The one comparison: the freeze calls it before the case exists (its answer decides a limit) and
    :func:`writer_message_check` calls it on a stored case, so the two cannot disagree.

    :param writer_message: The message the case freezes.
    :ptype writer_message: str
    :param recorded_digest: ``GenerationProvenance.user_message_digest`` of the recorded memo's analysis.
    :ptype recorded_digest: str
    :return: ``verified`` when the message digests to what the generation recorded, ``differs`` when it
        does not.
    :rtype: WriterMessageCheck
    """
    return "verified" if user_message_digest(writer_message) == recorded_digest else "differs"


def writer_message_check(case: ReporterCase) -> WriterMessageCheck | None:
    """Whether the case's frozen writer message is what its recorded memo's generator was sent.

    A freeze renders the message from the bundle it pins, so it is the writer's own message only if
    the bundle reproduces and the rendering has not changed since the generation. The recorded digest
    settles both at once: ``verified`` means byte-identical, and ``differs`` means the memo is judged
    against a message its writer never read. A differing freeze is not refused — ``limits`` states it.

    :param case: The stored case.
    :ptype case: ReporterCase
    :return: The check's answer, or ``None`` when the case pins no memo and there is nothing to check.
    :rtype: WriterMessageCheck | None
    """
    if case.writer_message is None or case.recorded_writer_message_digest is None:
        return None
    return message_check(case.writer_message, case.recorded_writer_message_digest)


def reporter_case_payload(case: ReporterCase) -> dict[str, Any]:
    """The ``host_payload`` a reporter ``EvalTestCase`` carries.

    A label's ``reader`` and ``collected_by`` are written only when set, so a label naming neither is stored exactly
    as a build at durable v16.0 stored it, and that build, whose ``ReporterLabel`` refuses a key it does not declare,
    still reads the case: ``host_payload`` is opaque, so no newer-minor tolerance reaches it. ``direction``,
    ``verdict`` and ``memo_excerpt`` are written only when set on the same terms, so a direction label quoting no
    passage is stored exactly as v16.1 stored it.

    :param case: The case to store.
    :ptype case: ReporterCase
    :return: The payload, keyed so :func:`reporter_case_of` can read it back.
    :rtype: dict[str, Any]
    """
    frozen = case.model_dump(mode="json")
    for label in frozen["labels"]:
        for added in ("direction", "verdict", "memo_excerpt", "reader", "collected_by"):
            if label[added] is None:
                del label[added]
    # A memo case is stored exactly as a build before plan cases stored it: no kind version, no plan fields.
    if case.kind_version == MEMO_CASE_VERSION:
        for planned in (
            "kind_version",
            "report_input",
            "report_input_digest",
            "recorded_report_input_digest",
            "recorded_plan",
            "planner_prompt_version",
        ):
            del frozen[planned]
    return {REPORTER_CASE_KEY: frozen}


def reporter_case_of(test_case: EvalTestCase) -> ReporterCase | None:
    """Read a reporter case back off a stored test case.

    :param test_case: Any stored case.
    :ptype test_case: EvalTestCase
    :return: The case, or ``None`` when this test case carries no reporter case.
    :rtype: ReporterCase | None
    """
    payload = test_case.host_payload
    if not isinstance(payload, dict) or REPORTER_CASE_KEY not in payload:
        return None
    return ReporterCase.model_validate(payload[REPORTER_CASE_KEY])


def _losses(frozen: Any, rebuilt: Any, path: str, added: list[str]) -> list[str]:
    """Name every frozen value the rebuild dropped or changed, collecting the keys it only ADDED.

    :param frozen: A node of the frozen document.
    :ptype frozen: Any
    :param rebuilt: The same node of the rebuilt bundle's ``to_dict()``.
    :ptype rebuilt: Any
    :param path: Where the node sits, for the message.
    :ptype path: str
    :param added: Receives the path of every key the rebuild has and the frozen document lacks.
    :ptype added: list[str]
    :return: One entry per lost or changed value; empty when every frozen value survived.
    :rtype: list[str]
    """
    if isinstance(frozen, dict) and isinstance(rebuilt, dict):
        losses: list[str] = []
        for key, value in frozen.items():
            if key in rebuilt:
                losses.extend(_losses(value, rebuilt[key], f"{path}.{key}", added))
            else:
                losses.append(f"{path}.{key} dropped")
        added.extend(f"{path}.{key}" for key in rebuilt if key not in frozen)
        return losses
    if isinstance(frozen, list) and isinstance(rebuilt, list):
        if len(frozen) != len(rebuilt):
            return [f"{path} held {len(frozen)} items and rebuilds with {len(rebuilt)}"]
        losses = []
        for index, (old, new) in enumerate(zip(frozen, rebuilt, strict=True)):
            losses.extend(_losses(old, new, f"{path}[{index}]", added))
        return losses
    return [] if frozen == rebuilt else [f"{path} changed from {_clipped(frozen)} to {_clipped(rebuilt)}"]


def _clipped(value: Any) -> str:
    """A value's repr, cut to a length a refusal message can carry."""
    shown = repr(value)
    return shown if len(shown) <= 80 else shown[:77] + "..."


def rebuild_bundle(case: ReporterCase) -> CampaignBundle:
    """Rebuild the frozen campaign bundle, refusing one that did not survive the round-trip intact.

    A case frozen under an older bundle shape could hand the generator and the judge a bundle
    that is not the one the case pins. Three checks make that loud:

    - the frozen document must still digest to the fingerprint stored beside it, in that fingerprint's version
      (an unprefixed one predates rounding floats), so the case holds the bundle it says it holds;
    - it must validate strictly — a field the bundle model no longer declares is refused, as on
      every stored read;
    - every frozen value must survive the rebuild unchanged — none altered by validation.

    **A campaign bundle frozen at an earlier shape this build reads as its own is rebuilt at its own**
    (:meth:`~threetears.evals.analysis.campaign_bundle.CampaignBundle.from_dict`): shape 2 lacks only the figures
    across aspects, which the rebuild reads off the frozen readings and aspects' bundles, so a case frozen at 2 keeps
    generating, calibrating and fingerprinting as the evidence it froze, and its labels stand. Its recorded
    fingerprint is checked against the document as frozen, and the shape is the one value the rebuild moves.

    **A bundle frozen before one memo covered a campaign is refused**, before any of this: a case frozen over one
    aspect's bundle (before the campaign rewrite), or over a bundle of a shape before 59, holds no campaign bundle, so no rebuild can
    hand the writer what it now reads, and the case is frozen again from its campaign instead.

    **From shape 59 on, a field the bundle model gained after the freeze is NOT a loss, and that tolerance is
    deliberate** — the one place a read here accepts an older shape. The frozen bundle is the
    EVIDENCE the case's labels were written against, not a stored document to be dropped and
    rewritten across a change: a re-freeze would re-assemble today's evidence, which is not what
    the labels judged, so refusing the older shape would leave no way to keep the case honest. The
    frozen document never held the new field, so the rebuild can only fill it from the model's
    own default, and every value the case pins is still there. The added fields are logged by path, and the generated
    analysis records the rebuilt bundle's own fingerprint, so a memo generated over the wider bundle is distinguishable
    from one generated at freeze time.

    :param case: The case whose frozen bundle to rebuild.
    :ptype case: ReporterCase
    :return: The bundle, carrying every frozen value.
    :rtype: CampaignBundle
    :raises RefusedInput: The frozen bundle is not a campaign bundle of a shape this build reads, or an aspect's bundle
        in it is of a shape from before a bundle read one aspect (59).
    :raises ValueError: The frozen bundle does not validate, no longer digests to the recorded
        fingerprint, or loses or changes a frozen value in the round-trip.
    """
    frozen = case.bundle
    what = f"the reporter case frozen over bundle {case.bundle_fingerprint}"
    remedy = "freeze the case again from its campaign, so its bundle is the campaign bundle its writer now reads"
    CampaignBundle.refuse_another_shape(frozen, what=what, remedy=remedy)
    frozen_digest = campaign_evidence_digest(frozen, like=case.bundle_fingerprint)
    if frozen_digest != case.bundle_fingerprint:
        raise ValueError(
            f"the case's frozen bundle digests to {frozen_digest} and the case recorded "
            f"{case.bundle_fingerprint} — the case does not hold the bundle it names, so neither the "
            "generator nor the judge would see the bundle this case pins"
        )
    bundle = CampaignBundle.from_dict(frozen, what=what, remedy=remedy)
    added: list[str] = []
    # A bundle frozen at an earlier shape this build reads is rebuilt at this one; its shape is the one value the
    # rebuild moves on purpose, and every other frozen value must survive.
    losses = _losses({**frozen, "schema_version": bundle.schema_version}, bundle.to_dict(), "bundle", added)
    if losses:
        shown = "; ".join(losses[:5]) + (f"; and {len(losses) - 5} more" if len(losses) > 5 else "")
        raise ValueError(
            f"the case's frozen bundle does not survive the round-trip: {shown} — so neither the "
            "generator nor the judge would see the bundle this case pins"
        )
    if added:
        log.info(
            "Frozen bundle %s rebuilt with %d field(s) the bundle model gained since the freeze, at their defaults: %s",
            case.bundle_fingerprint[:12],
            len(added),
            ", ".join(sorted({_indexless(path) for path in added})),
        )
    return bundle


def _indexless(path: str) -> str:
    """A path with its list indices folded, so one added field reads once, not once per row."""
    return re.sub(r"\[\d+\]", "[]", path)


def render_case_material(case: ReporterCase) -> str:
    """The case's frozen bundle rendered by this build — byte-identical to what a generator is sent over it.

    One renderer, two readers: this is the generator's own
    :func:`~threetears.evals.analysis.campaign_bundle.build_campaign_user_message` over the rebuilt bundle, never
    a copy of it, so the judge cannot ground against a bundle the generator was not given. Which
    memo is judged against this render and which against the case's frozen writer message is
    decided in one place, :func:`judge_case_material`.

    :param case: The case whose frozen bundle to render.
    :ptype case: ReporterCase
    :return: The generator's own user message over the frozen bundle.
    :rtype: str
    :raises ValueError: The frozen bundle cannot be rebuilt intact (see :func:`rebuild_bundle`).
    """
    return build_campaign_user_message(rebuild_bundle(case))


def judge_case_material(case: ReporterCase, *, generated_over: CampaignBundle | None) -> str:
    """The evidence the judge scores a memo against: what that memo's own writer read — the one place it is decided.

    - **A generated memo** is judged against the message its generator was sent: this build's render
      of the bundle it was generated over, ``generated_over``.
    - **The case's recorded memo** (``generated_over=None``) is judged against the writer message the
      case froze, so a later change to how a bundle is rendered cannot move the evidence under its
      labels — plus the case's limits (:func:`replay_case_material`). The bundle is not read at all
      then, so a recorded memo stays judgeable after a bundle change that orphans the frozen
      document for generation.

    :param case: The cell's case.
    :ptype case: ReporterCase
    :param generated_over: The bundle a generated memo was written over, or ``None`` for the recorded memo.
    :ptype generated_over: CampaignBundle | None
    :return: The judge's case material.
    :rtype: str
    :raises ValueError: ``generated_over`` is ``None`` and the case pins no recorded memo, so there is no
        memo, and no message its writer read, to judge.
    """
    if generated_over is not None:
        return build_campaign_user_message(generated_over)
    if case.writer_message is None:
        raise ValueError(
            "this reporter case pins no recorded memo, so there is no writer message to judge one against; only "
            "generated candidates run it"
        )
    return replay_case_material(case, case.writer_message)


# --- The memo as written: the model's words, in one reader's layout -------------------------------
#
# The judge scores the memo as the model WROTE it, so what is laid out here is
# the model's authored words and nothing a surface would add. What this layout decides is only the
# SHAPE: headings and lines a reader follows, never the field names of a serialisation — a judge
# shown ``posture.tier`` or ``cell_ref`` docks readability for vocabulary no reader ever meets.


def _confidence(confidence: ConfidenceTier) -> str:
    """A confidence tier in words."""
    return CONFIDENCE_WORDS[confidence]


def _read_case(test_case: EvalTestCase) -> ReporterCase | str:
    """Read a cell's case, or say why it cannot be — an infra fault.

    :param test_case: The cell's stored case.
    :ptype test_case: EvalTestCase
    :return: The case, or the infra error to report.
    :rtype: ReporterCase | str
    """
    try:
        case = reporter_case_of(test_case)
    except ValueError as exc:
        return f"apparatus: test case {test_case.id} carries an unreadable reporter case: {exc}"
    if case is None:
        return f"apparatus: test case {test_case.id} carries no reporter case under host_payload[{REPORTER_CASE_KEY!r}]"
    return case


def _unusable_bundle(test_case: EvalTestCase, exc: ValueError) -> str:
    """The infra error for a case whose frozen bundle cannot be rebuilt intact."""
    return f"apparatus: test case {test_case.id}'s frozen bundle cannot be used: {exc}"


def _generation_input(test_case: EvalTestCase) -> tuple[ReporterCase, CampaignBundle] | str:
    """The case and the bundle a cell generates over, or the infra error that excludes it with no call made.

    The one reading of a case a generator is sent, shared by :meth:`ReporterKind.invoke`, which generates over it,
    and :meth:`ReporterKind.price_case`, which prices the call before launch, so the two cannot read one case two
    ways. The request built from the bundle is :func:`~threetears.evals.analysis.generator.first_request`, which
    the generation sends.

    :param test_case: The cell's stored case.
    :ptype test_case: EvalTestCase
    :return: ``(case, bundle)``, or the infra error to report.
    :rtype: tuple[ReporterCase, CampaignBundle] | str
    """
    case = _read_case(test_case)
    if isinstance(case, str):
        return case
    if case.kind_version != MEMO_CASE_VERSION:
        # A generating candidate here is a memo writer, and a plan case's subject is a plan: its output would be
        # scored against a rubric and evidence that judge another writer.
        return (
            f"apparatus: test case {test_case.id} judges a report plan (reporter kind version {case.kind_version}), "
            "and a generating reporter candidate writes memos; a plan case runs as recorded"
        )
    try:
        bundle = rebuild_bundle(case)
    except ValueError as exc:
        return _unusable_bundle(test_case, exc)
    try:
        # The generator's own precondition, asked before the candidate is: a bundle in which no arm can be
        # described fails EVERY candidate identically, so it is the case's fault and must not be scored against
        # the candidate.
        refuse_an_undescribable_arm_table(bundle)
    except GenerationError as exc:
        return f"apparatus: test case {test_case.id} cannot be generated over: {exc}"
    return case, bundle


@dataclass
class PreparedReporter:
    """One cell's reporter candidate: the model it binds, and the cell's tracing windows.

    Per cell rather than on the kind, because one kind instance drives every cell of a run.
    """

    model: str
    span_window: CellSpanWindow


def _refuse_a_model_mismatch(variant_config: VariantConfig, bound_model: str, *, what: str) -> None:
    """Refuse a cell recorded against one model while this kind would run another.

    The run records ``EvalResult.model`` from the cell's variant, and the memo comes from
    whatever this kind is bound to; if the two disagree the run scores one candidate under
    another's name. The cell is cleanly excluded rather than measured.

    :param variant_config: The cell's contestant stack.
    :ptype variant_config: VariantConfig
    :param bound_model: The model this kind actually runs.
    :ptype bound_model: str
    :param what: How to name this kind's binding in the message.
    :ptype what: str
    :raises CandidatePreparationFailed: The two models differ.
    """
    if variant_config.candidate_model == bound_model:
        return
    log.error(
        "Reporter kind asked for a cell on model %r while %s is %r — refusing the cell",
        variant_config.candidate_model,
        what,
        bound_model,
    )
    raise CandidatePreparationFailed(
        f"apparatus: this cell is recorded against model {variant_config.candidate_model!r} but {what} is "
        f"{bound_model!r}, so running it would report one candidate's memo under another's name — wire one "
        "reporter kind per candidate model",
        termination="factory_failed",
    )


@dataclass
class _RecordingClient:
    """The generator's client, with every completion it returned — and the one that raised — kept.

    Per cell, so the spend it reports is this cell's. A generation that is refused has still
    been billed, so the telemetry is read off what the provider returned, not off the analysis
    that may never have been built.
    """

    inner: CompletionGenerator
    results: list[CompletionResult] = field(default_factory=list)
    failure: BaseException | None = None

    async def generate(
        self, *, system: str, user: str, response_format: dict[str, Any] | None = None
    ) -> CompletionResult:
        """Forward one call and keep its result.

        :param system: The system prompt.
        :ptype system: str
        :param user: The user message.
        :ptype user: str
        :param response_format: The provider directive.
        :ptype response_format: dict[str, Any] | None
        :return: The provider's completion, unchanged.
        :rtype: CompletionResult
        """
        try:
            result = await self.inner.generate(system=system, user=user, response_format=response_format)
        # prawduct:ok-broad-except — not swallowed: recorded so invoke can tell the candidate's own call failing from a harness fault, then re-raised
        except Exception as exc:
            self.failure = exc
            raise
        self.results.append(result)
        return result

    def telemetry(self, bound_model: str) -> CandidateTelemetry:
        """What every call this cell made cost, as the candidate role's usage, and how many returned.

        Each generator call that returned is a turn the candidate delivered (``turns_delivered``), so a memo
        refused for soundness, cut at its output cap, or failed in its repair call AFTER a call returned and
        was billed keeps that call's time and spend in the arm's cost and latency
        (:func:`~threetears.evals.kernel.result_condition.delivered_a_turn`). Zero when the first call
        itself raised: nothing was delivered, and nothing is averaged in.

        :param bound_model: The model to attribute a call to when the provider named none.
        :ptype bound_model: str
        :return: The cell's telemetry.
        :rtype: CandidateTelemetry
        """
        ledger = RoleUsageLedger(role="candidate")
        for result in self.results:
            # The spend alone: a candidate's row carries no sent settings (only the judge's and the simulator's are
            # disclosed), so what the generator's completion says it was sent with is never read, and a value of the
            # wrong type there cannot cost the record of a call that was billed.
            read = CallUsage.billed(result)
            # The bound model stands in for an unnamed ``model`` only, which attributes spend; the
            # served model stays what the response said, unrecorded when it said nothing.
            ledger.add_llm_result(replace(read, model=read.model or bound_model))
        return CandidateTelemetry(usage=ledger.rows(), turns_delivered=len(self.results))

    def progress(self, bound_model: str) -> Callable[[], CandidateOutput]:
        """The reading a cell registers with its sink: every call returned so far, and its cost.

        Reads :attr:`results` when it is called rather than when it is built, so a generation
        cut off in its repair call is recorded with the initial call's spend.

        :param bound_model: The model to attribute a call to when the provider named none.
        :ptype bound_model: str
        :return: The candidate side so far — spend only, since no memo exists until ``invoke`` returns.
        :rtype: Callable[[], CandidateOutput]
        """
        return lambda: CandidateOutput(telemetry=self.telemetry(bound_model))


class ReporterKind:
    """A (prompt preset, generator model) candidate: generates a memo over each case's bundle.

    Calls the pure :func:`~threetears.evals.analysis.generator.generate_analysis` and never the
    service method, so an eval run stores no analysis and mints no insight a later bundle would
    read. One instance drives a whole run; a cell's state lives on what :meth:`prepare` returns.
    """

    #: The memo, judged against the bundle it was written over.
    judged_artifact = JudgedArtifact.DOCUMENT
    #: No limit of the kind's own: a cell's state is what :meth:`prepare` returns and the per-cell recording
    #: client :meth:`invoke` builds, and the one generator client every cell shares is a completion client,
    #: which the port requires to be safe to call from several cells at once
    #: (:class:`~threetears.evals.schema.completion.CompletionClient`).
    max_concurrent_cells: int | None = None

    def __init__(
        self,
        *,
        prompt: str,
        prompt_id: str,
        prompt_version: str | None,
        client: Any,
        model: str,
        host: EvalHost,
    ) -> None:
        """Bind the prompt and the generator client this candidate runs, for the host whose memos it writes.

        :param prompt: The resolved ``eval_analysis_gen`` prompt text.
        :ptype prompt: str
        :param prompt_id: The registry key it was resolved under.
        :ptype prompt_id: str
        :param prompt_version: Its version, or ``None`` to let the generator derive a content version.
        :ptype prompt_version: str | None
        :param client: The completion client, built by the launcher for ``model``.
        :ptype client: Any
        :param model: The generator model id.
        :ptype model: str
        :param host: The host the frozen bundles were assembled for: its vocabulary is what a memo
            over them is written and checked in, and its failure describer is the only reading
            of what ``client`` raises that can say a failed call was refused for the calling
            account, since the engine names none of the host's error types.
        :ptype host: EvalHost
        """
        self._prompt = prompt
        self._prompt_id = prompt_id
        self._prompt_version = prompt_version
        self._client = client
        self._model = model
        self._host = host

    async def prepare(
        self,
        *,
        subject_snapshot: Any,
        variant_config: VariantConfig,
        world_seed: Any,
        span_window: CellSpanWindow,
        cassettes: CellCassettes | None,
        world: Any,
    ) -> PreparedReporter:
        """Bind this cell's model and windows, refusing a cell that would measure another model.

        The case is not read here: it arrives with the test case in :meth:`invoke`.

        :param subject_snapshot: Unread — the candidate is the prompt and model this kind holds.
        :ptype subject_snapshot: Any
        :param variant_config: This cell's contestant stack; its model is the one it must measure.
        :ptype variant_config: VariantConfig
        :param world_seed: Unread — a reporter perceives no world.
        :ptype world_seed: Any
        :param span_window: This cell's tracing windows, opened in :meth:`invoke`.
        :ptype span_window: CellSpanWindow
        :param cassettes: Unwired — a reporter calls no tool a cassette could record, so a cassette run
            of one is refused by the engine rather than run live under a replay.
        :ptype cassettes: CellCassettes | None
        :param world: Unread — a reporter perceives no world, so it opens none and its cells record none.
        :ptype world: Any
        :return: The cell's bound model and windows.
        :rtype: PreparedReporter
        :raises CandidatePreparationFailed: The cell's model is not the one this kind's client runs.
        """
        _refuse_a_model_mismatch(variant_config, self._model, what="the reporter's generator client")
        return PreparedReporter(model=self._model, span_window=span_window)

    def price_case(self, test_case: EvalTestCase) -> float | None:
        """The ceiling :meth:`invoke` reserves for ``test_case``'s first generator call, built and priced as it is.

        What a launch prices an arm's largest call by before anything is spent
        (:func:`~threetears.evals.ops.price_largest_reporter_call`): a call whose ceiling is above the run's cap
        on its own is one the run could never make. The request is the one the generation sends first
        (:func:`~threetears.evals.analysis.generator.first_request`, over the bundle :meth:`invoke` reads), priced
        on this kind's client as :class:`~threetears.evals.kernel.capped_completion.CappedCompletion` prices it.

        The repair round-trip a refused output buys is not priced here: its message carries the first call's
        output, which does not exist before launch, so it stays held to the cap when it is reserved.

        :param test_case: The cell's case, carrying the reporter case in its ``host_payload``.
        :ptype test_case: EvalTestCase
        :return: The ceiling in dollars; ``None`` for a case :meth:`invoke` makes no call for (it is excluded as the
            harness's) or one this kind's client cannot price.
        :rtype: float | None
        :raises ValueError: The client priced the call at an amount no reservation could hold.
        """
        generation_input = _generation_input(test_case)
        if isinstance(generation_input, str):
            return None
        _case, bundle = generation_input
        system, user, contract = first_request(bundle, self._prompt, self._host.profile)
        return call_ceiling(self._client, system=system, user=user, response_format=contract)

    async def invoke(self, instance: PreparedReporter, test_case: EvalTestCase, sink: CellSink) -> CandidateOutput:
        """Generate one memo over the case's frozen bundle and hand the judge both.

        Never raises. An unusable case and a bundle no candidate could generate over are the
        harness's (``infra_errors``, the cell excluded); a generation the
        engine refused, or the candidate's own provider call failing, is the candidate's
        (``candidate_errors``, the cell failed, no judge) — unless the host's describer says the
        call was the account's, which excludes the cell and stops the run (``account_refused``), or
        the provider's (it declined the call: a rate limit, or the provider overloaded or unavailable),
        which excludes the cell and lets the run go on. Spend
        is reported on every exit that reached the provider, because a refused generation was still
        billed — and to ``sink`` as each call returns, because a
        generation is up to two calls and a deadline in the second must not lose the first. Each call is
        counted on the sink's candidate total as it returns; the run's cost cap is read between cells, so no
        call is refused inside the cell.

        :param instance: What :meth:`prepare` bound.
        :ptype instance: PreparedReporter
        :param test_case: The cell's case, carrying the reporter case in its ``host_payload``.
        :ptype test_case: EvalTestCase
        :param sink: This cell's sink, handed the recording client's reading.
        :ptype sink: CellSink
        :return: The analysis as one document, its judge evidence and what the generation cost.
        :rtype: CandidateOutput
        """
        scopes = instance.span_window
        with scopes.identity():
            generation_input = _generation_input(test_case)
            if isinstance(generation_input, str):
                log.error("Reporter cell cannot generate: %s", generation_input)
                return CandidateOutput(infra_errors=[generation_input])
            case, bundle = generation_input

            # Every generator call's spend is reported to the sink as it returns — a generation is up to two
            # calls, and the repair's prompt exists only once the first has returned.
            recorder = _RecordingClient(inner=CappedCompletion(self._client, sink=sink, source="candidate"))
            sink.report_progress(recorder.progress(instance.model))
            try:
                with scopes.collecting():
                    analysis, _insights = await generate_analysis(
                        bundle,
                        prompt=self._prompt,
                        model=instance.model,
                        client=recorder,
                        prompt_id=self._prompt_id,
                        bundle_assembled_at=case.bundle_assembled_at,
                        prompt_version=self._prompt_version,
                        profile=self._host.profile,
                        # A reporter run is how a host learns which writers to allow, so it measures any.
                        measuring_writers=True,
                    )
            except GenerationError as exc:
                # A refused or truncated generation, or a prompt that does not ask for the memo contract:
                # all the candidate's. SoundnessRefusal is a subclass and lands here too.
                return CandidateOutput(
                    candidate_errors=[f"reporter: {exc}"], telemetry=recorder.telemetry(instance.model)
                )
            # prawduct:ok-broad-except — invoke must not raise (an exception here kills a run that has already spent); split by whether the candidate's own provider call raised it
            except Exception as exc:
                telemetry = recorder.telemetry(instance.model)
                if recorder.failure is exc:
                    failure = describe_failure(
                        self._host.failure_describer, exc, logger=log, where="Reporter generator call"
                    )
                    # A call after one that returned is the repair the first reply's refusal bought: named, so a
                    # reader of an excluded cell knows the candidate had already written a memo that was refused.
                    call = (
                        "generator's repair call, after its first reply was refused,"
                        if recorder.results
                        else "generator call"
                    )
                    if failure.owner == "account":
                        # Every model behind the key gets the same answer, so this cell measured the
                        # account, not the candidate: excluded, and ``account_refused`` stops the RUN.
                        log_provider_failure(
                            log, failure, exc, "Reporter candidate's %s was refused for the calling account", call
                        )
                        return CandidateOutput(
                            infra_errors=[
                                f"apparatus: reporter: the {call} was refused for the calling account: {failure.description}"
                            ],
                            account_refused=True,
                            telemetry=telemetry,
                        )
                    if failure.owner == "provider":
                        # The provider declined to serve the call at that moment, as it would any request: the
                        # cell measured the provider, not the candidate, so it is excluded and the run goes on.
                        log_provider_failure(
                            log,
                            failure,
                            exc,
                            "Reporter candidate's %s was declined by the provider",
                            call,
                            level=logging.WARNING,
                        )
                        return CandidateOutput(
                            infra_errors=[
                                f"apparatus: reporter: the {call} was declined by the provider: {failure.description}"
                            ],
                            telemetry=telemetry,
                        )
                    log_provider_failure(
                        log, failure, exc, "Reporter candidate's %s failed", call, level=logging.WARNING
                    )
                    return CandidateOutput(
                        candidate_errors=[f"reporter: the {call} failed: {failure.description}"],
                        telemetry=telemetry,
                    )
                # Everything else is the harness's: a defect below the kind. Failing the candidate for either would score it for the rig.
                contain(
                    exc,
                    log=log,
                    what="Reporter cell failed outside the candidate's own call",
                    level="error",
                )
                return CandidateOutput(infra_errors=[f"apparatus: {type(exc).__name__}: {exc}"], telemetry=telemetry)

        return CandidateOutput(
            output=[analysis.model_dump(mode="json")],
            judge_evidence=JudgeEvidence(
                case_material=judge_case_material(case, generated_over=bundle),
                artifact=render_memo_as_written(analysis),
            ),
            telemetry=recorder.telemetry(instance.model),
        )


#: The heading the as-recorded candidate puts over a case's limits in what the judge reads.
REPLAY_NOTES_HEADING = "# NOTES ON THIS EVIDENCE — not seen by the memo's author"


def render_memo_as_written(analysis: EvalAnalysis) -> str:
    """The memo as authored: the model's words, code's figures, in one canonical layout a reader follows.

    Markdown, in the order the campaign's page lays it out: the summary across every aspect, the recommendation (its
    arm, reasons, confidence, the findings it rests on and every broken floor it is made over), each aspect's summary,
    the findings in the order the model wrote them, and what to run next — each list section omitted when the model
    wrote nothing for it. Findings are numbered ``F1`` onward, and every link the model made by position is printed
    that way, a finding's trials by their ids. The memo has no headline or chart: the report's warnings and readings
    are code's, so the judge scores what the writer wrote, its summary among it. Arms are named by
    :func:`~threetears.evals.analysis.arms.arm_label`, numbers by
    :func:`~threetears.evals.analysis.numbers.format_number`.

    The numbers code resolved from the model's own readings are printed beside them, because they are what the
    model's claims are about. Deterministic: one analysis renders to one string.

    :param analysis: The analysis to render.
    :ptype analysis: EvalAnalysis
    :return: The text the judge reads as the output under review.
    :rtype: str
    :raises RefusedInput: The analysis was written by the report planner, so it holds no memo.
    """
    document = analysis.document
    if document is None:
        raise RefusedInput(
            f"analysis {analysis.id} was written by the report planner and holds no memo to render",
            remedy="render a memo the memo writer wrote; a planned analysis is read through its plan",
        )
    arm = arm_namer(analysis)
    resolutions: list[FindingResolution | None] = (
        list(analysis.resolutions) if analysis.resolutions else [None] * len(document.findings)
    )
    # The question, then the read across it, as the campaign's page opens.
    question = analysis.design_snapshot.question
    opening = ["## Summary", ""]
    if question:
        opening += [f"Question: {one_line(question)}", ""]
    sections: list[list[str]] = [[*opening, document.summary.strip()]]

    recommended = document.recommended()
    recommendation = ["## Recommendation", ""]
    if recommended is None:
        recommendation.append("- None: the memo recommends no arm.")
    else:
        rests = f" Rests on {positions(recommended.rests_on)}." if recommended.rests_on else ""
        recommendation.append(
            f"- **{one_line(recommended.arm)}** — confidence {_confidence(recommended.confidence)}.{rests}"
        )
        for reason in recommended.reasons:
            recommendation += under(reason, "    ", first="  - ")
        for override in recommended.floors_overridden:
            recommendation += under(override.why, "    ", first=f"  - Over the broken floor {override.floor}: ")
        for behind in recommended.behind_on:
            recommendation += under(
                behind.why, "    ", first=f"  - Though {behind.ahead_arm} is ahead on {behind.aspect}: "
            )
        for missed in recommended.bars_missed:
            recommendation += under(
                missed.why, "    ", first=f"  - Below the bar on {missed.measure} ({missed.aspect}): "
            )
    sections.append(recommendation)

    if document.aspect_summaries:
        aspects = ["## Aspects"]
        for summary in document.aspect_summaries:
            aspects += ["", f"### {one_line(summary.aspect)}", ""]
            aspects += under(summary.what_changed, "  ", first="- What changed: ")
            aspects += under(summary.how_sure, "  ", first="- How sure: ")
            aspects += under(summary.means, "  ", first="- What it means: ")
        sections.append(aspects)

    if document.findings:
        findings = ["## Findings"]
        for position, (finding, resolution) in enumerate(zip(document.findings, resolutions, strict=True)):
            findings += ["", f"### F{position + 1}. {one_line(finding.title)}", ""]
            facts = [f"Confidence: {_confidence(finding.confidence)}."]
            if resolution is not None:
                facts.append(f"Stands on: {stands_on_words(analysis, resolution.evidence_tier)}.")
            if finding.axes:
                facts.append(f"About: {', '.join(finding.axes)}.")
            if finding.trials:
                facts.append(f"Trials: {', '.join(finding.trials)}.")
            if finding.invalidates:
                facts.append(f"Invalidates {positions(finding.invalidates)}.")
            findings.append(" ".join(facts))
            if finding.body.strip():
                findings += ["", finding.body.strip()]
            if resolution is not None and resolution.evidence:
                rows = evidence_rows(analysis, resolution.evidence, arm)
                table = markdown_table(
                    [header for _, header in evidence_columns(analysis)],
                    [[row[key].text for key, _ in evidence_columns(analysis)] for row in rows],
                )
                findings += ["", "Evidence:", "", *table]
            if finding.caveats:
                findings += ["", "Caveats:"]
                for caveat in finding.caveats:
                    findings += under(caveat.text, "  ", first=f"- {caveat.kind}: ")
            if finding.durable.strip():
                findings += ["", f"Carried forward: {finding.durable.strip()}"]
        sections.append(findings)

    if document.next:
        steps = ["## What to run next", ""]
        for number, step in enumerate(document.next, start=1):
            lever = f" ({step.lever})" if step.lever else ""
            steps.append(f"{number}. **{one_line(step.title)}** — {step.leverage} leverage{lever}")
            if step.why.strip():
                steps += under(step.why, "     ", first="   - Why: ")
        sections.append(steps)

    return "\n\n".join("\n".join(section) for section in sections) + "\n"


def _model_in(annotation: Any) -> type[BaseModel] | None:
    """The model a field's annotation holds, directly or inside a container, ``Annotated`` or union; None for none."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    origin = get_origin(annotation)
    if origin is None:
        return None
    arguments = get_args(annotation)
    if origin is Annotated:
        arguments = arguments[:1]
    elif origin not in (Union, types.UnionType) and not (isinstance(origin, type) and issubclass(origin, Sequence)):
        return None
    return next((model for argument in arguments if (model := _model_in(argument)) is not None), None)


def _as_planned(node: BaseModel, shape: type[BaseModel], written: Callable[[str], str]) -> dict[str, Any]:
    """``node`` read as ``shape``, one of the models it extends: only ``shape``'s fields, at every depth, and every
    prose field written out."""
    projected: dict[str, Any] = {}
    for name, info in shape.model_fields.items():
        value = getattr(node, name)
        nested = _model_in(info.annotation)
        if nested is not None and isinstance(value, BaseModel):
            projected[name] = _as_planned(value, nested, written)
        elif nested is not None and isinstance(value, Sequence) and not isinstance(value, str):
            projected[name] = [_as_planned(item, nested, written) for item in value]
        elif isinstance(value, str) and field_is_prose(info):
            projected[name] = written(value)
        elif isinstance(value, BaseModel):
            # A field the stored plan holds as a model where the planner's part holds a plain value (an answer's
            # warning: its position as the planner chose it, the warning code resolved it to as stored): the stored
            # model, which names what the planner pointed at.
            projected[name] = value.model_dump(mode="json")
        else:
            projected[name] = value
    return projected


def render_plan_for_judge(
    plan: ReportPlan, report_input: ReportInput, arm_names: Mapping[str, str] | None = None
) -> str:
    """The plan as the planner wrote it, with every figure key written out: the text the reporter eval judges.

    The planner's own part of the plan (:class:`~threetears.evals.kernel.report_plan.PlannedReport`, what it chose
    and wrote), as indented JSON in its field order, with nothing code filled in beside it (the signals, the
    masthead, the options columns, a chart's unit). Every prose field has its figure keys and ``{arm:<id>}``
    references written out by :func:`~threetears.evals.analysis.report.figure_keys.render`, the renderer the
    campaign's page writes the plan's prose with, against the same report input, so each sentence reads as a
    reader sees it. Deterministic: one plan and input render to one string.

    **The arms key leads, where the page shows one.** An arm is written ``{name} ({tag})``, and the tag means
    something only through the key the page prints once (``plan.show_arms_key``). The key is built by the page's own
    :func:`~threetears.evals.analysis.report.plan_page.tables.arms_key` over the page's own look, so judge and reader
    see one table, and it comes BEFORE the JSON so the judge has read what each tag stands for when it meets the
    first sentence that cites one. With it the text is the key block, a blank line, then the JSON; a plan whose page
    shows no key renders as the bare JSON.

    :param plan: The stored plan.
    :ptype plan: ReportPlan
    :param report_input: The report input it was written against.
    :ptype report_input: ReportInput
    :param arm_names: Each arm's computed short name, by id, as :func:`~threetears.evals.analysis.report.figure_keys.render`
        takes it; None for none, as the campaign's page passes.
    :ptype arm_names: Mapping[str, str] | None
    :return: The text the judge reads as the output under review.
    :rtype: str
    :raises UnresolvableReference: A figure key does not resolve against ``report_input``.
    """
    names = dict(arm_names or {})
    projected = _as_planned(plan, PlannedReport, lambda text: render(text, report_input, names))
    body = json.dumps(projected, indent=2, ensure_ascii=False) + "\n"
    if not plan.show_arms_key:
        return body
    key = arms_key(report_input, plan_look(plan, report_input, plan_facts(report_input)))
    return "\n".join([f"## {key.eyebrow}", "", key.takeaway, "", *table_lines(key.table), "", body])


def replay_case_material(case: ReporterCase, material: str) -> str:
    """The evidence a RECORDED memo is judged against: the frozen bundle, plus what its author could not see.

    A recorded memo was written against an earlier assembly of its campaign's evidence; the case
    freezes today's, and ``case.limits`` says where the two can differ (a bundle that no longer
    reproduces, arms that re-assemble without their levels). Without these notes a judge charges
    the memo for not mentioning an artifact of re-assembly — a calibration round saw exactly that:
    groundedness AND decision usefulness docked for "levels unavailable" arms the author never saw.
    The claims are still checked against the evidence above; what changes
    is that an OMISSION of a re-assembly artifact is not a defect. A generated memo is written
    against this very bundle, so it gets no notes, and a case with no limits reads byte-identically
    to what the generator was sent.

    :param case: The frozen case.
    :ptype case: ReporterCase
    :param material: The message the recorded memo is judged against, as :func:`judge_case_material` chose it.
    :ptype material: str
    :return: The judge's case material for the as-recorded candidate.
    :rtype: str
    """
    if not case.limits:
        return material
    notes = "\n".join(f"- {limit}" for limit in case.limits)
    if case.kind_version == PLAN_CASE_VERSION:
        return (
            f"{material}\n\n{REPLAY_NOTES_HEADING.replace('memo', 'plan')}\n"
            "The evidence above is rendered by a later build than the one the plan under review was written "
            "against, and differs from what its planner read as follows. Judge every claim the plan makes against "
            "the evidence above, but do not count it against the plan that it does not mention any of these:"
            f"\n{notes}"
        )
    return (
        f"{material}\n\n{REPLAY_NOTES_HEADING}\n"
        "The memo under review was written against an EARLIER assembly of this evidence. The differences "
        "below are artifacts of re-assembling it later, which its author could not have seen. Judge every "
        "claim the memo makes against the evidence above, but do not count it against the memo that it does "
        f"not mention any of these:\n{notes}"
    )


class AsRecordedReporterKind:
    """The memo each case's campaign actually got, replayed — the calibration candidate.

    Runs under :data:`AS_RECORDED_MODEL`, calls no generator and costs nothing, and hands the
    judge the recorded memo through the same render a generated memo gets, against the writer
    message the case froze (:func:`judge_case_material`) — so calibrating the rubric against a
    reviewer's read of a stored memo is the same measurement as scoring any later candidate, over
    the evidence the reviewer's read was of.
    """

    #: The recorded memo, judged exactly as a generated one is.
    judged_artifact = JudgedArtifact.DOCUMENT
    #: No limit of the kind's own: it calls nothing and holds nothing between cells.
    max_concurrent_cells: int | None = None

    async def prepare(
        self,
        *,
        subject_snapshot: Any,
        variant_config: VariantConfig,
        world_seed: Any,
        span_window: CellSpanWindow,
        cassettes: CellCassettes | None,
        world: Any,
    ) -> PreparedReporter:
        """Bind this cell's windows, refusing a cell recorded against a generator model.

        :param subject_snapshot: Unread.
        :ptype subject_snapshot: Any
        :param variant_config: This cell's contestant stack; it must name :data:`AS_RECORDED_MODEL`.
        :ptype variant_config: VariantConfig
        :param world_seed: Unread.
        :ptype world_seed: Any
        :param span_window: This cell's tracing windows.
        :ptype span_window: CellSpanWindow
        :param cassettes: Unwired, as for the generating reporter.
        :ptype cassettes: CellCassettes | None
        :param world: Unread, as for the generating reporter.
        :ptype world: Any
        :return: The cell's binding.
        :rtype: PreparedReporter
        :raises CandidatePreparationFailed: The cell names a real model, whose score this replay
            would otherwise be recorded under.
        """
        _refuse_a_model_mismatch(variant_config, AS_RECORDED_MODEL, what="the as-recorded replay")
        return PreparedReporter(model=AS_RECORDED_MODEL, span_window=span_window)

    async def invoke(self, instance: PreparedReporter, test_case: EvalTestCase, sink: CellSink) -> CandidateOutput:
        """Replay the case's recorded memo through the judge path. Never raises.

        Only the identity window is opened: nothing the candidate does here takes time worth
        measuring, and a collection window around a replay would claim otherwise.

        :param instance: What :meth:`prepare` bound.
        :ptype instance: PreparedReporter
        :param test_case: The cell's case, carrying the reporter case in its ``host_payload``.
        :ptype test_case: EvalTestCase
        :param sink: This cell's sink. Unused: a replay awaits nothing and spends nothing, so there
            is no progress to report.
        :ptype sink: CellSink
        :return: The recorded analysis as one document and its judge evidence, at zero cost — or an
            infra error when the case carries no readable recorded memo.
        :rtype: CandidateOutput
        """
        with instance.span_window.identity():
            case = _read_case(test_case)
            if isinstance(case, str):
                log.error("As-recorded reporter cell has no usable case: %s", case)
                return CandidateOutput(infra_errors=[case])
            if case.recorded_memo is None:
                return CandidateOutput(
                    infra_errors=[
                        f"apparatus: test case {test_case.id} carries no recorded memo for the as-recorded candidate"
                    ]
                )
            # A case pinning a memo pins the message its writer read (``ReporterCase`` refuses
            # one without), so the material is the frozen text and the bundle is not read.
            material = judge_case_material(case, generated_over=None)

        recorded: dict[str, Any] = {"recorded_analysis_id": case.recorded_analysis_id, "memo": case.recorded_memo}
        if case.kind_version == PLAN_CASE_VERSION:
            # The plan as rendered for the judge rides under ``memo``, where every reader of a replay finds the judged
            # text; the plan in key form beside it.
            recorded |= {"kind_version": case.kind_version, "plan": case.recorded_plan}
        return CandidateOutput(
            output=[recorded],
            judge_evidence=JudgeEvidence(case_material=material, artifact=case.recorded_memo),
            telemetry=CandidateTelemetry(
                usage=[], untimed_reason="a replay of the recorded memo — nothing was generated"
            ),
        )


__all__ = [
    "AS_RECORDED_MODEL",
    "LABEL_BANDS",
    "MEMO_CASE_VERSION",
    "PLAN_CASE_VERSION",
    "WRITER_MESSAGE_DIFFERS",
    "REPLAY_NOTES_HEADING",
    "REPORTER_CASE_KEY",
    "REPORTER_KIND",
    "AsRecordedReporterKind",
    "LabelCriterion",
    "LabelDirection",
    "LabelVerdict",
    "PreparedReporter",
    "ReporterCase",
    "ReporterKind",
    "ReporterKindVersion",
    "ReporterLabel",
    "WriterMessageCheck",
    "judge_case_material",
    "message_check",
    "rebuild_bundle",
    "render_case_material",
    "render_memo_as_written",
    "render_plan_for_judge",
    "replay_case_material",
    "reporter_case_of",
    "reporter_case_payload",
    "reporter_cell_timeout_s",
    "writer_message_check",
]
