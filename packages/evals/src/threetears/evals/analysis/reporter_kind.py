"""The analysis reporter as a candidate kind — its memos judged like any other LLM output.

Code checks the STRUCTURE of an analysis and never its prose; whether the
prose is grounded, useful, insightful, relevant, calibrated and readable is measured here, by
evaluating the reporter the way every other candidate is evaluated: its own cases, rubric and
judge, through the ordinary run path.

- **A case** is one campaign's analysis bundle, FROZEN into the case's ``host_payload``. Bundles
  are re-assembled on demand everywhere else, and the schema moves, so without the freeze a case
  would silently change under every candidate scored on it. The bundle already carries the
  campaign's declaration (``declared_design``: its questions and bars).
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

Generic by construction: nothing here names a subject's shape, a product or a domain. It implements
the run-side ``CandidateKind`` seam from inside the analysis package. Where it belongs is already
decided — the kind goes runner-side, while the case model and the memo rendering stay in analysis —
and moving it is the remaining half of that split.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from threetears.evals.analysis.bundle import AnalysisContextBundle
from threetears.evals.analysis.errors import GenerationError
from threetears.evals.analysis.generator import (
    build_user_message,
    generate_analysis,
    generation_ceiling_s,
    refuse_an_undescribable_arm_table,
    user_message_digest,
)
from threetears.evals.analysis.report.serialize_md import markdown_table
from threetears.evals.analysis.report.words import (
    CONFIDENCE_WORDS,
    EVIDENCE_COLUMNS,
    arm_namer,
    evidence_rows,
    positions,
    stands_on_words,
)
from threetears.evals.contracts.authored import NO_CHART
from threetears.evals.contracts.base import EvalDocumentModel
from threetears.evals.contracts.campaign import (
    ConfidenceTier,
    EvalAnalysis,
    FindingResolution,
)
from threetears.evals.contracts.candidate_kind import (
    CandidateOutput,
    CandidatePreparationFailed,
    CandidateTelemetry,
    CellSink,
    CellSpanWindow,
    VariantConfig,
)
from threetears.evals.contracts.cassettes import CellCassettes
from threetears.evals.contracts.hashing import canonical_digest
from threetears.evals.contracts.host.eval_host import EvalHost
from threetears.evals.contracts.models import DimName, EvalTestCase, JudgedArtifact, JudgeEvidence, RubricDim
from threetears.evals.contracts.provider import (
    CompletionGenerator,
    CompletionResult,
    RequestCeiling,
    describe_failure,
    log_provider_failure,
)
from threetears.evals.contracts.usage_capture import CallUsage, RoleUsageLedger
from threetears.observe import get_logger

log = get_logger(__name__)

#: The ``candidate_kind`` a reporter template declares.
REPORTER_KIND = "analysis_reporter"

#: The candidate model name the as-recorded candidate runs under. Not a provider model — a
#: reporter run launched with it replays each case's recorded memo and calls no generator.
AS_RECORDED_MODEL = "as-recorded"

#: The key under which a reporter case lives inside ``EvalTestCase.host_payload``.
REPORTER_CASE_KEY = "reporter_case"


def judge_phase_ceiling_s(
    *,
    judge_dims: int,
    judge_concurrency: int,
    judge_call_attempts: int,
    judge_max_tokens: int,
    request_s: RequestCeiling,
) -> float:
    """The wall-clock ceiling of one cell's judge phase, derived from the ceilings of its requests.

    The judge scores ``judge_dims`` dimensions in waves of ``judge_concurrency``, each dimension up to
    ``judge_call_attempts`` requests capped at ``judge_max_tokens``, each bounded by the host's own answer
    for one such request (``request_s``), which counts every provider call and retry wait the host's
    client spends on it. Every kind whose cell ends in a judge phase adds this to its ceiling, since the
    cell's deadline bounds the judging too, and a ceiling without it cuts a slow cell off while it is scored.

    Args:
        judge_dims: Dimensions the judge scores on each cell.
        judge_concurrency: Judge requests in flight at once.
        judge_call_attempts: Requests one judge dimension can make, its parse retries included.
        judge_max_tokens: The judge's output cap, as its requests are built with.
        request_s: The host's ceiling for one request on its client, by output cap.

    Returns:
        The judge phase's ceiling in seconds: ``waves * judge_call_attempts * request_s(judge_max_tokens)``,
        with ``waves`` the dimensions over the concurrency, rounded up.
    """
    judge_waves = -(-max(judge_dims, 0) // max(judge_concurrency, 1))
    return judge_waves * judge_call_attempts * request_s(judge_max_tokens)


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

    A reporter cell is one generation (:func:`~threetears.evals.analysis.generator.generation_ceiling_s`,
    so the generation's request count lives in one place), then its rubric's judge phase
    (:func:`judge_phase_ceiling_s`). Every request in either is bounded by the host's own answer for one
    request at its output cap (``request_s``), so the ceiling sits above the slowest finishing cell
    rather than at a sampled duration: the engine's generic 600s cancelled a generation that was still
    writing inside its own output cap.

    Every operational input is a parameter, so the host passes the values its launch read and the
    ceiling is derived from the same generator cap the run's clients are built with. The judge's limits
    are parameters too: they belong to the judge, which the run package owns, and analysis imports
    nothing from run.

    Args:
        judge_dims: Rubric dimensions the run's judge scores on each cell.
        judge_concurrency: Judge requests in flight at once.
        generator_max_tokens: The generator's output cap, as its clients are built with.
        judge_call_attempts: Requests one judge dimension can make, its parse retries included.
        judge_max_tokens: The judge's output cap, as its requests are built with.
        request_s: The host's ceiling for one request on its client, by output cap.

    Returns:
        The cell ceiling in seconds: the generation's ceiling plus the judge phase's.
    """
    generation_s = generation_ceiling_s(request_s=request_s, generator_max_tokens=generator_max_tokens)
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

#: The 1-5 judge scores each label direction agrees with, inclusive. A calibration read counts a
#: dimension as agreeing when the judge's score falls inside its label's band.
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
    scale: str = Field(description="How the dimension was answered at freeze time: 'ordinal' (1-5) or 'pass_fail'.")
    scoring_guide: dict[str, str] = Field(
        default_factory=dict,
        description="The scoring guide, level to descriptor, at freeze time. Empty when the dimension had none.",
    )

    @classmethod
    def of(cls, dim: RubricDim) -> LabelCriterion:
        """Copy the criterion a rubric dimension states.

        Args:
            dim: The template's dimension.

        Returns:
            Its description, scale and scoring guide.
        """
        return cls(description=dim.description, scale=dim.scale, scoring_guide=dict(dim.scoring_guide))


class ReporterLabel(BaseModel):
    """One reader's written verdict on a recorded memo, mapped to the dimension it bears on.

    A direction, never a number: the reads these come from never gave one. Plain ``BaseModel``
    because ``quote`` is someone's words verbatim.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    dimension: DimName = Field(
        description="The rubric dimension name the verdict bears on, e.g. 'reporter.groundedness'."
    )
    direction: LabelDirection = Field(description="Where the reader places the memo on that dimension.")
    quote: str = Field(description="The reader's own words, verbatim — the evidence for the direction.")
    note: str | None = Field(default=None, description="Why the quote maps to this dimension, when not self-evident.")
    criterion: LabelCriterion | None = Field(
        default=None,
        description=(
            "The criterion the verdict was written against, stamped by the freeze from the template's rubric — a "
            "caller never supplies it, and a freeze refuses a label that arrives with one. None only on a label as "
            "a caller submits it: every label a case stores carries one (``ReporterCase`` refuses one that does not)."
        ),
    )


class ReporterCase(BaseModel):
    """What one reporter case pins: a frozen bundle, optionally the memo it got, and its labels.

    Plain ``BaseModel`` because ``bundle`` and ``recorded_memo`` are documents whose strings are
    data, not input to tidy.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    bundle: dict[str, Any] = Field(description="The campaign's bundle, frozen: ``AnalysisContextBundle.to_dict()``.")
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
        return self


#: Whether a case's frozen writer message is the one its recorded memo's generator was sent — see
#: :func:`writer_message_check`.
WriterMessageCheck = Literal["verified", "differs"]


def message_check(writer_message: str, recorded_digest: str) -> WriterMessageCheck:
    """Compare a writer message about to be frozen with the digest its generation recorded.

    The one comparison: the freeze calls it before the case exists (its answer decides a limit) and
    :func:`writer_message_check` calls it on a stored case, so the two cannot disagree.

    Args:
        writer_message: The message the case freezes.
        recorded_digest: ``GenerationProvenance.user_message_digest`` of the recorded memo's analysis.

    Returns:
        ``verified`` when the message digests to what the generation recorded, ``differs`` when it
        does not.
    """
    return "verified" if user_message_digest(writer_message) == recorded_digest else "differs"


def writer_message_check(case: ReporterCase) -> WriterMessageCheck | None:
    """Whether the case's frozen writer message is what its recorded memo's generator was sent.

    A freeze renders the message from the bundle it pins, so it is the writer's own message only if
    the bundle reproduces and the rendering has not changed since the generation. The recorded digest
    settles both at once: ``verified`` means byte-identical, and ``differs`` means the memo is judged
    against a message its writer never read. A differing freeze is not refused — ``limits`` states it.

    Args:
        case: The stored case.

    Returns:
        The check's answer, or ``None`` when the case pins no memo and there is nothing to check.
    """
    if case.writer_message is None or case.recorded_writer_message_digest is None:
        return None
    return message_check(case.writer_message, case.recorded_writer_message_digest)


def reporter_case_payload(case: ReporterCase) -> dict[str, Any]:
    """The ``host_payload`` a reporter ``EvalTestCase`` carries.

    Args:
        case: The case to store.

    Returns:
        The payload, keyed so :func:`reporter_case_of` can read it back.
    """
    return {REPORTER_CASE_KEY: case.model_dump(mode="json")}


def reporter_case_of(test_case: EvalTestCase) -> ReporterCase | None:
    """Read a reporter case back off a stored test case.

    Args:
        test_case: Any stored case.

    Returns:
        The case, or ``None`` when this test case carries no reporter case.
    """
    payload = test_case.host_payload
    if not isinstance(payload, dict) or REPORTER_CASE_KEY not in payload:
        return None
    return ReporterCase.model_validate(payload[REPORTER_CASE_KEY])


def _renamed_keys() -> dict[str, str]:
    """Every key a document model renamed within the current schema version: old name → new name.

    Read off the models' own ``__retired_fields__``, so a frozen bundle's renamed key is followed by the
    same declaration its stored read follows. Renames only: a key removed outright is a frozen value the
    rebuild does not carry, and stays a loss here.
    """
    renamed: dict[str, str] = {}
    pending: list[type[EvalDocumentModel]] = [EvalDocumentModel]
    while pending:
        model = pending.pop()
        pending.extend(model.__subclasses__())
        renamed.update({old: new for old, new in model.__retired_fields__.items() if new is not None})
    return renamed


def _losses(frozen: Any, rebuilt: Any, path: str, added: list[str], renamed: dict[str, str] | None = None) -> list[str]:
    """Name every frozen value the rebuild dropped or changed, collecting the keys it only ADDED.

    A frozen key a model renamed within the schema version (``__retired_fields__``) is compared with the
    rebuilt value under its new name: the stored read moved it there, so it was carried, not dropped.

    Args:
        frozen: A node of the frozen document.
        rebuilt: The same node of the rebuilt bundle's ``to_dict()``.
        path: Where the node sits, for the message.
        added: Receives the path of every key the rebuild has and the frozen document lacks.
        renamed: Old key → new key for every rename; read once from the models when omitted.

    Returns:
        One entry per lost or changed value; empty when every frozen value survived.
    """
    renames = _renamed_keys() if renamed is None else renamed
    if isinstance(frozen, dict) and isinstance(rebuilt, dict):
        losses: list[str] = []
        moved: set[str] = set()
        for key, value in frozen.items():
            if key in rebuilt:
                losses.extend(_losses(value, rebuilt[key], f"{path}.{key}", added, renames))
            elif (renamed_to := renames.get(key)) is not None and renamed_to in rebuilt and renamed_to not in frozen:
                moved.add(renamed_to)
                losses.extend(_losses(value, rebuilt[renamed_to], f"{path}.{renamed_to}", added, renames))
            else:
                losses.append(f"{path}.{key} dropped")
        added.extend(f"{path}.{key}" for key in rebuilt if key not in frozen and key not in moved)
        return losses
    if isinstance(frozen, list) and isinstance(rebuilt, list):
        if len(frozen) != len(rebuilt):
            return [f"{path} held {len(frozen)} items and rebuilds with {len(rebuilt)}"]
        losses = []
        for index, (old, new) in enumerate(zip(frozen, rebuilt, strict=True)):
            losses.extend(_losses(old, new, f"{path}[{index}]", added, renames))
        return losses
    return [] if frozen == rebuilt else [f"{path} changed from {_clipped(frozen)} to {_clipped(rebuilt)}"]


def _clipped(value: Any) -> str:
    """A value's repr, cut to a length a refusal message can carry."""
    shown = repr(value)
    return shown if len(shown) <= 80 else shown[:77] + "..."


def rebuild_bundle(case: ReporterCase) -> AnalysisContextBundle:
    """Rebuild the frozen bundle, refusing one that did not survive the round-trip intact.

    A case frozen under an older bundle shape could hand the generator and the judge a bundle
    that is not the one the case pins. Three checks make that loud:

    - the frozen document must still digest to the fingerprint stored beside it, so the case
      holds the bundle it says it holds;
    - it must validate strictly — a field the bundle model no longer declares is refused, as on
      every stored read;
    - every frozen value must survive the rebuild unchanged — none altered by validation. A value under
      a key renamed within the schema version (``__retired_fields__``) survives under its new name.

    **A field the bundle model gained after the freeze is NOT a loss, and that tolerance is
    deliberate** — the one place a read here accepts an older shape. The frozen bundle is the
    EVIDENCE the case's labels were written against, not a stored document to be dropped and
    rewritten across a change: a re-freeze would re-assemble today's evidence, which is not what
    the labels judged, so refusing the older shape would leave no way to keep the case honest. The
    frozen document never held the new field, so the rebuild can only fill it from the model's
    own default, and every value the case pins is still there. The added fields are logged by path, and the generated analysis records the rebuilt bundle's own
    fingerprint, so a memo generated over the wider bundle is distinguishable from one generated
    at freeze time.

    Args:
        case: The case whose frozen bundle to rebuild.

    Returns:
        The bundle, carrying every frozen value.

    Raises:
        ValueError: The frozen bundle does not validate, no longer digests to the recorded
            fingerprint, or loses or changes a frozen value in the round-trip.
    """
    frozen_digest = canonical_digest(case.bundle)
    if frozen_digest != case.bundle_fingerprint:
        raise ValueError(
            f"the case's frozen bundle digests to {frozen_digest} and the case recorded "
            f"{case.bundle_fingerprint} — the case does not hold the bundle it names, so neither the "
            "generator nor the judge would see the bundle this case pins"
        )
    bundle = AnalysisContextBundle.from_dict(case.bundle)
    added: list[str] = []
    losses = _losses(case.bundle, bundle.to_dict(), "bundle", added)
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
    :func:`~threetears.evals.analysis.generator.build_user_message` over the rebuilt bundle, never
    a copy of it, so the judge cannot ground against a bundle the generator was not given. Which
    memo is judged against this render and which against the case's frozen writer message is
    decided in one place, :func:`judge_case_material`.

    Args:
        case: The case whose frozen bundle to render.

    Returns:
        The generator's own user message over the frozen bundle.

    Raises:
        ValueError: The frozen bundle cannot be rebuilt intact (see :func:`rebuild_bundle`).
    """
    return build_user_message(rebuild_bundle(case))


def judge_case_material(case: ReporterCase, *, generated_over: AnalysisContextBundle | None) -> str:
    """The evidence the judge scores a memo against: what that memo's own writer read — the one place it is decided.

    - **A generated memo** is judged against the message its generator was sent: this build's render
      of the bundle it was generated over, ``generated_over``.
    - **The case's recorded memo** (``generated_over=None``) is judged against the writer message the
      case froze, so a later change to how a bundle is rendered cannot move the evidence under its
      labels — plus the case's limits (:func:`replay_case_material`). The bundle is not read at all
      then, so a recorded memo stays judgeable after a bundle change that orphans the frozen
      document for generation.

    Args:
        case: The cell's case.
        generated_over: The bundle a generated memo was written over, or ``None`` for the recorded memo.

    Returns:
        The judge's case material.

    Raises:
        ValueError: ``generated_over`` is ``None`` and the case pins no recorded memo, so there is no
            memo, and no message its writer read, to judge.
    """
    if generated_over is not None:
        return build_user_message(generated_over)
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


def _one_line(text: str) -> str:
    """Text a heading or a bold lead carries, on one line — a newline would end either mid-sentence."""
    return " ".join(text.splitlines())


def _confidence(confidence: ConfidenceTier) -> str:
    """A confidence tier in words."""
    return CONFIDENCE_WORDS[confidence]


def _read_case(test_case: EvalTestCase) -> ReporterCase | str:
    """Read a cell's case, or say why it cannot be — an infra fault.

    Args:
        test_case: The cell's stored case.

    Returns:
        The case, or the infra error to report.
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

    Args:
        variant_config: The cell's contestant stack.
        bound_model: The model this kind actually runs.
        what: How to name this kind's binding in the message.

    Raises:
        CandidatePreparationFailed: The two models differ.
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

        Args:
            system: The system prompt.
            user: The user message.
            response_format: The provider directive.

        Returns:
            The provider's completion, unchanged.
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
        (:func:`~threetears.evals.contracts.result_condition.delivered_a_turn`). Zero when the first call
        itself raised: nothing was delivered, and nothing is averaged in.

        Args:
            bound_model: The model to attribute a call to when the provider named none.

        Returns:
            The cell's telemetry.
        """
        ledger = RoleUsageLedger(role="candidate")
        for result in self.results:
            read = CallUsage.of(result)
            # The bound model stands in for an unnamed ``model`` only, which attributes spend; the
            # served model stays what the response said, unrecorded when it said nothing.
            ledger.add_llm_result(replace(read, model=read.model or bound_model))
        return CandidateTelemetry(usage=ledger.rows(), turns_delivered=len(self.results))

    def progress(self, bound_model: str) -> Callable[[], CandidateOutput]:
        """The reading a cell registers with its sink: every call returned so far, and its cost.

        Reads :attr:`results` when it is called rather than when it is built, so a generation
        cut off in its repair call is recorded with the initial call's spend.

        Args:
            bound_model: The model to attribute a call to when the provider named none.

        Returns:
            The candidate side so far — spend only, since no memo exists until ``invoke`` returns.
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

        Args:
            prompt: The resolved ``eval_analysis_gen`` prompt text.
            prompt_id: The registry key it was resolved under.
            prompt_version: Its version, or ``None`` to let the generator derive a content version.
            client: The completion client, built by the launcher for ``model``.
            model: The generator model id.
            host: The host the frozen bundles were assembled for: its vocabulary is what a memo
                over them is written and checked in, and its failure describer is the only reading
                of what ``client`` raises that can say a failed call was refused for the calling
                account, since the engine names none of the host's error types.
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

        Args:
            subject_snapshot: Unread — the candidate is the prompt and model this kind holds.
            variant_config: This cell's contestant stack; its model is the one it must measure.
            world_seed: Unread — a reporter perceives no world.
            span_window: This cell's tracing windows, opened in :meth:`invoke`.
            cassettes: Unwired — a reporter calls no tool a cassette could record, so a cassette run
                of one is refused by the engine rather than run live under a replay.
            world: Unread — a reporter perceives no world, so it opens none and its cells record none.

        Returns:
            The cell's bound model and windows.

        Raises:
            CandidatePreparationFailed: The cell's model is not the one this kind's client runs.
        """
        _refuse_a_model_mismatch(variant_config, self._model, what="the reporter's generator client")
        return PreparedReporter(model=self._model, span_window=span_window)

    async def invoke(self, instance: PreparedReporter, test_case: EvalTestCase, sink: CellSink) -> CandidateOutput:
        """Generate one memo over the case's frozen bundle and hand the judge both.

        Never raises. An unusable case and a bundle no candidate could generate over are the
        harness's (``infra_errors``, the cell excluded); a generation the
        engine refused, or the candidate's own provider call failing, is the candidate's
        (``candidate_errors``, the cell failed, no judge) — unless the host's describer says the
        call was refused for the calling account, which excludes the cell and stops the run
        (``account_refused``). Spend is reported on every exit that reached the provider, because
        a refused generation was still billed — and to ``sink`` as each call returns, because a
        generation is up to two calls and a deadline in the second must not lose the first.

        Args:
            instance: What :meth:`prepare` bound.
            test_case: The cell's case, carrying the reporter case in its ``host_payload``.
            sink: This cell's sink, handed the recording client's reading.

        Returns:
            The analysis as one document, its judge evidence and what the generation cost.
        """
        scopes = instance.span_window
        with scopes.identity():
            case = _read_case(test_case)
            if isinstance(case, str):
                log.error("Reporter cell has no usable case: %s", case)
                return CandidateOutput(infra_errors=[case])
            try:
                bundle = rebuild_bundle(case)
            except ValueError as exc:
                log.error("Reporter cell's frozen bundle cannot be used: %s", exc)
                return CandidateOutput(infra_errors=[_unusable_bundle(test_case, exc)])
            try:
                # The generator's own precondition, asked before the candidate is: a bundle in
                # which no arm can be described fails EVERY candidate identically, so it is the
                # case's fault and must not be scored against the candidate.
                refuse_an_undescribable_arm_table(bundle)
            except GenerationError as exc:
                log.error("Reporter cell's case cannot be generated over: %s", exc)
                return CandidateOutput(
                    infra_errors=[f"apparatus: test case {test_case.id} cannot be generated over: {exc}"]
                )

            recorder = _RecordingClient(inner=self._client)
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
                    if failure.account_refused:
                        # Every model behind the key gets the same answer, so this cell measured the
                        # account, not the candidate: excluded, and ``account_refused`` stops the RUN.
                        log_provider_failure(
                            log, failure, exc, "Reporter candidate's generator call was refused for the calling account"
                        )
                        return CandidateOutput(
                            infra_errors=[
                                f"apparatus: reporter: the generator call was refused for the calling account: {failure.description}"
                            ],
                            account_refused=True,
                            telemetry=telemetry,
                        )
                    log_provider_failure(
                        log, failure, exc, "Reporter candidate's generator call failed", level=logging.WARNING
                    )
                    return CandidateOutput(
                        candidate_errors=[f"reporter: the generator call failed: {failure.description}"],
                        telemetry=telemetry,
                    )
                # Everything else is the harness's: a defect below the kind. Failing the candidate for either would score it for the rig.
                log.exception("Reporter cell failed outside the candidate's own call")
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

    Markdown, in the order a reader acts on it: the headline, the summary, where each declared
    question stands, the decisions, the findings in the order the model wrote them, and what to run
    next — each section omitted when the model wrote nothing for it. Findings are numbered from one,
    and every link the model made by position is printed that way. Arms are named by
    :func:`~threetears.evals.analysis.arms.arm_label`, numbers by :func:`~threetears.evals.analysis.numbers.format_number`.

    The numbers code resolved from the model's own readings are printed beside them, because they are
    what the model's claims are about; a chart is its type and its caption, never the figures it
    draws; a chart dropped as undrawable says so. Deterministic: one analysis renders to one string.

    Args:
        analysis: The analysis to render.

    Returns:
        The text the judge reads as the output under review.
    """
    document = analysis.document
    arm = arm_namer(analysis)
    resolutions: list[FindingResolution | None] = (
        list(analysis.resolutions) if analysis.resolutions else [None] * len(document.findings)
    )
    sections: list[list[str]] = [[f"# {_one_line(document.headline) or '(blank headline)'}"]]
    if document.summary.strip():
        sections.append([document.summary.strip()])

    if document.questions:
        questions = ["## Declared questions", ""]
        for answer in document.questions:
            rests = f" (rests on finding {positions(answer.rests_on)})" if answer.rests_on else ""
            asked = (
                analysis.design_snapshot.question_words(answer.question_id)
                if analysis.design_snapshot is not None
                else answer.question_id
            )
            questions.append(f"- {_one_line(asked)} — {answer.resolution}: {_one_line(answer.answer)}{rests}")
        sections.append(questions)

    if document.decisions:
        decisions = ["## Decisions", ""]
        for decision in document.decisions:
            rests = f" Rests on finding {positions(decision.rests_on)}." if decision.rests_on else ""
            decisions.append(
                f"- **{_one_line(decision.proposal)}** — {decision.disposition}; "
                f"confidence {_confidence(decision.confidence)}.{rests}"
            )
            if decision.revisit_when:
                decisions.append(f"  - Revisit when: {decision.revisit_when}")
        sections.append(decisions)

    if document.findings:
        findings = ["## Findings"]
        for position, (finding, resolution) in enumerate(zip(document.findings, resolutions, strict=True)):
            findings += ["", f"### {position + 1}. {_one_line(finding.title)}", ""]
            facts = [f"Confidence: {_confidence(finding.confidence)}."]
            if resolution is not None:
                facts.append(f"Stands on: {stands_on_words(analysis, resolution.evidence_tier)}.")
            if finding.axes:
                facts.append(f"About: {', '.join(finding.axes)}.")
            if finding.invalidates:
                facts.append(f"Invalidates finding {positions(finding.invalidates)}.")
            findings.append(" ".join(facts))
            if finding.body.strip():
                findings += ["", finding.body.strip()]
            if resolution is not None and resolution.evidence:
                rows = evidence_rows(analysis, resolution.evidence, arm)
                table = markdown_table(
                    [header for _, header in EVIDENCE_COLUMNS],
                    [[row[key] for key, _ in EVIDENCE_COLUMNS] for row in rows],
                )
                findings += ["", "Evidence:", "", *table]
            if finding.chart.type != NO_CHART:
                if resolution is not None and resolution.chart_note:
                    findings += ["", resolution.chart_note]
                else:
                    caption = f": {_one_line(finding.chart.caption)}" if finding.chart.caption.strip() else ""
                    note = f" — {_one_line(finding.chart.note)}" if finding.chart.note.strip() else ""
                    findings += ["", f"Chart ({finding.chart.type}){caption}{note}"]
            if finding.caveats:
                findings += ["", "Caveats:", *(f"- {caveat.kind}: {caveat.text}" for caveat in finding.caveats)]
            if finding.durable.strip():
                findings += ["", f"Carried forward: {finding.durable.strip()}"]
        sections.append(findings)

    if document.next:
        steps = ["## What to run next", ""]
        for number, step in enumerate(document.next, start=1):
            lever = f" ({step.lever})" if step.lever else ""
            steps.append(f"{number}. **{_one_line(step.title)}** — {step.leverage} leverage{lever}")
            if step.why.strip():
                steps.append(f"   - Why: {step.why.strip()}")
        sections.append(steps)

    return "\n\n".join("\n".join(section) for section in sections) + "\n"


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

    Args:
        case: The frozen case.
        material: The message the recorded memo is judged against, as :func:`judge_case_material` chose it.

    Returns:
        The judge's case material for the as-recorded candidate.
    """
    if not case.limits:
        return material
    notes = "\n".join(f"- {limit}" for limit in case.limits)
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

        Args:
            subject_snapshot: Unread.
            variant_config: This cell's contestant stack; it must name :data:`AS_RECORDED_MODEL`.
            world_seed: Unread.
            span_window: This cell's tracing windows.
            cassettes: Unwired, as for the generating reporter.
            world: Unread, as for the generating reporter.

        Returns:
            The cell's binding.

        Raises:
            CandidatePreparationFailed: The cell names a real model, whose score this replay
                would otherwise be recorded under.
        """
        _refuse_a_model_mismatch(variant_config, AS_RECORDED_MODEL, what="the as-recorded replay")
        return PreparedReporter(model=AS_RECORDED_MODEL, span_window=span_window)

    async def invoke(self, instance: PreparedReporter, test_case: EvalTestCase, sink: CellSink) -> CandidateOutput:
        """Replay the case's recorded memo through the judge path. Never raises.

        Only the identity window is opened: nothing the candidate does here takes time worth
        measuring, and a collection window around a replay would claim otherwise.

        Args:
            instance: What :meth:`prepare` bound.
            test_case: The cell's case, carrying the reporter case in its ``host_payload``.
            sink: This cell's sink. Unused: a replay awaits nothing and spends nothing, so there
                is no progress to report.

        Returns:
            The recorded analysis as one document and its judge evidence, at zero cost — or an
            infra error when the case carries no readable recorded memo.
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

        return CandidateOutput(
            output=[{"recorded_analysis_id": case.recorded_analysis_id, "memo": case.recorded_memo}],
            judge_evidence=JudgeEvidence(case_material=material, artifact=case.recorded_memo),
            telemetry=CandidateTelemetry(
                usage=[], untimed_reason="a replay of the recorded memo — nothing was generated"
            ),
        )


__all__ = [
    "AS_RECORDED_MODEL",
    "LABEL_BANDS",
    "REPLAY_NOTES_HEADING",
    "REPORTER_CASE_KEY",
    "REPORTER_KIND",
    "AsRecordedReporterKind",
    "LabelCriterion",
    "LabelDirection",
    "PreparedReporter",
    "ReporterCase",
    "ReporterKind",
    "ReporterLabel",
    "WriterMessageCheck",
    "judge_case_material",
    "judge_phase_ceiling_s",
    "message_check",
    "rebuild_bundle",
    "render_case_material",
    "render_memo_as_written",
    "replay_case_material",
    "reporter_case_of",
    "reporter_case_payload",
    "reporter_cell_timeout_s",
    "writer_message_check",
]
