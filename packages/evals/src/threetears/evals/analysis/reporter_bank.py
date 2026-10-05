"""The reporter's case bank: what a frozen case cannot test, and how a run reads against its labels.

Pure reads of the reporter's case bank, and none calls a model.

- **Which case is live**, :func:`reporter_case_bank` — the ONE derivation, over a template's stored
  cases, of which reporter case each (campaign, recorded memo) pair launches, which were superseded
  and by what, which an operator retired (archived), which cases this build cannot read, and which
  pairs hold more than one live case.
  The launch, its price, a freeze, a calibration read and a retirement's restore all ask it
  through :func:`decidable_reporter_case_bank`, so no two of them can disagree about a case one of
  them cannot read or a pair two freezes raced on.

- **Before the run**, :func:`case_limits` states what a frozen bundle cannot support a judgement
  about, in sentences, so the reader of a calibration knows which of its labels the evidence the
  judge saw could even reach. A case is frozen from TODAY's re-assembly of a campaign, and some of
  what the original memo saw may not re-assemble (an arm whose levels this build cannot describe,
  a variant with no per-arm cell) — a judge scoring a memo's claim about that arm is scoring it
  against evidence that no longer says anything about it.
- **After the run**, :func:`read_calibration` sets each labelled dimension's judge score beside the
  reader's written verdict and says whether they agree under :data:`LABEL_BANDS`, and whether the
  criterion the verdict was written against still reads as it did. The run's calibration ratings —
  people's scores of its results — are read beside the labels through the same agreement the bundle
  reads (:func:`~threetears.evals.analysis.agreement.judge_agreement`). It is a READ: every number
  comes from stored results and ratings, and nothing here re-scores or re-judges.

Generic by construction (extraction target ``3tears-eval-analysis``): it names no subject, product
or domain, reads a case only through :func:`reporter_case_of`, and reads a result only through the
fields the engine records on every result.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from threetears.evals.analysis.agreement import JudgeAgreement, judge_agreement
from threetears.evals.analysis.reporter_kind import (
    LABEL_BANDS,
    LabelCriterion,
    LabelDirection,
    ReporterCase,
    ReporterLabel,
    WriterMessageCheck,
    message_check,
    reporter_case_of,
    writer_message_check,
)
from threetears.evals.contracts.errors import ValidationFailedError

if TYPE_CHECKING:
    from threetears.evals.analysis.bundle import AnalysisContextBundle
    from threetears.evals.contracts.models import CalibrationRating, EvalResult, EvalTestCase, RubricScore


def case_limits(
    bundle: AnalysisContextBundle,
    *,
    recorded_fingerprint: str | None = None,
    writer_message: str | None = None,
    recorded_message_digest: str | None = None,
) -> list[str]:
    """State, as sentences, what a judgement grounded on this bundle cannot reach.

    Computed from the bundle rather than written by hand, because the reason a case exists is
    that it is reproducible: the same bundle always yields the same limits, and a limit an
    operator forgot to write would read as a case with none.

    Five conditions, each one sentence per instance so a reader can match a sentence to a claim:

    - **the re-assembly does not reproduce the memo's bundle** — only when a recorded memo is
      pinned: its generation fingerprinted the bundle it read, and today's re-assembly differs, so
      the judge grounds that memo against evidence it did not see (moved evidence, or a bundle
      schema that moved under unchanged evidence — nothing stored says which). A claim that was
      true of the old bundle and is false of this one is scored as false here.
    - **the bundle reproduces and its rendering does not** — the frozen writer message digests to
      something other than what the memo's generation recorded sending, over the same bundle, so
      the way bundles are rendered changed in between. Stated only when the bundle reproduces:
      when it does not, the message differs as a consequence and the sentence above already says so.
    - **an arm whose levels are unavailable** — its variant was keyed by a predicate this build
      cannot reproduce, so the bundle shows it measured and pooled but cannot say what it RAN. A
      memo's claim naming that arm's configuration cannot be checked against this case.
    - **a variant with no per-arm cell** — the bundle indexes it but carries no ``cell_measures``
      entry for it, so a per-arm number the memo quotes for it has nothing to be checked against.
    - **no keyed variant at all** — nothing in the bundle is addressable per arm, so every
      per-arm claim is uncheckable. Stated rather than left as an empty list, because an empty
      list is also what a bundle with nothing wrong produces.

    Args:
        bundle: The bundle being frozen.
        recorded_fingerprint: The pinned memo's ``generation.bundle_fingerprint``, or ``None`` when
            the case pins no memo.
        writer_message: The writer message the case freezes, or ``None`` when it freezes none.
        recorded_message_digest: The pinned memo's ``generation.user_message_digest``, or ``None``
            when the case pins no memo.

    Returns:
        The limits — the reproduction or rendering sentence first, then per arm in
        ``variant_index`` order; empty when the bundle and its rendering reproduce and every
        variant is described and has a cell.
    """
    limits: list[str] = []
    fingerprint = bundle.fingerprint()
    if recorded_fingerprint is not None and recorded_fingerprint != fingerprint:
        limits.append(
            f"This bundle is today's re-assembly and does not reproduce the one the recorded memo was written "
            f"from (recorded {recorded_fingerprint}, now {fingerprint}): either the evidence moved or the bundle's "
            "shape did, and nothing stored says which. The memo is judged against evidence it did not see, so a "
            "claim true of the old bundle and false of this one scores as false here."
        )
    elif (
        writer_message is not None
        and recorded_message_digest is not None
        and message_check(writer_message, recorded_message_digest) == "differs"
    ):
        limits.append(
            f"The bundle reproduces the one the recorded memo was written from, but the writer message frozen here "
            f"is not the one its generator was sent (recorded digest {recorded_message_digest}): how a bundle is "
            "rendered changed between the generation and this freeze. The memo is judged against a message its "
            "writer did not read, so a claim resting on something only the older rendering showed scores as "
            "ungrounded here."
        )
    if not bundle.variant_index:
        limits.append(
            "The bundle keys no variant, so no claim the memo makes about a particular arm can be checked "
            "against this case — every per-arm statement is ungrounded here whatever it says."
        )
        return limits
    with_cells = {cell.variant_key for cell in bundle.cell_measures}
    for entry in bundle.variant_index:
        if entry.levels_unavailable is not None:
            limits.append(
                f"Arm {entry.variant_key} re-assembles without its levels ({entry.levels_unavailable}), so the "
                "judge sees it measured but cannot see what it ran — a claim naming that arm's configuration "
                "cannot be checked against this case."
            )
        if entry.variant_key not in with_cells:
            limits.append(
                f"Arm {entry.variant_key} has no per-arm cell in this bundle, so a number the memo attributes to "
                "that arm alone has nothing here to be checked against."
            )
    return limits


def case_pair(case: ReporterCase) -> tuple[str | None, str | None]:
    """The (campaign, recorded memo) pair a case freezes — what supersession and liveness are scoped to.

    **One memo judged against its campaign's evidence is ONE case, whatever bundle it was frozen
    from.** A freeze fingerprints TODAY's re-assembly, so re-freezing one campaign's memo after
    anything moved the bundle (a run archived, a bundle schema change) yields a new fingerprint for
    the same memo. Keying a case on the fingerprint would make that re-freeze a second LIVE case of
    the memo — a launch would judge it twice and every calibration denominator would count it
    twice, with nothing reading it as ambiguous. So the pair is the campaign and the memo, and a
    re-freeze after the evidence moved SUPERSEDES the earlier freeze of the pair like any other
    revision. The fingerprint is not identity: it is recorded on the case, and when it no longer
    reproduces what the memo read, :func:`case_limits` states that as a limit.

    A case pinning no memo is keyed on its campaign alone, on the same terms: a generating
    candidate over one campaign's evidence is one case per campaign, not one per re-assembly.

    Args:
        case: A reporter case.

    Returns:
        ``(campaign id the frozen bundle names, recorded analysis id)`` — either ``None`` when the
        frozen document carries none.
    """
    return _str_or_none(case.bundle.get("campaign_id")), case.recorded_analysis_id


@dataclass(frozen=True)
class AmbiguousPair:
    """A (campaign, recorded memo) pair holding more than one live case.

    Attributes:
        campaign_id: The campaign the pair's frozen bundles name, or ``None`` when they name none.
        recorded_analysis_id: The pair's recorded memo, or ``None`` when it pins none.
        live_case_ids: Every live case of the pair, sorted — each one a launch would run.
    """

    campaign_id: str | None
    recorded_analysis_id: str | None
    live_case_ids: tuple[str, ...]


@dataclass(frozen=True)
class ReporterCaseBank:
    """One template's stored reporter cases, with which of them is live — derived once, read by every consumer.

    **Supersession is a pointer on the SUCCESSOR**, so whether a case is live is a fact about
    every OTHER stored case of the template. That is why it is derived here once rather than by
    each consumer walking the store: the launch, its price, the freeze and the calibration read
    each used to build their own superseded set, and they disagreed about a case they could not
    read (one refused, two skipped) and about a pair holding two live cases (one refused, three
    counted both).

    **A case this build cannot read makes liveness undecidable, not merely incomplete**: it may
    supersede a case that would otherwise read as live. So it is carried in ``unreadable`` rather
    than dropped, and a consumer that decides anything from liveness refuses while it is non-empty.

    **A retired case is neither live nor unreadable.** An operator retires one (``archived`` on the
    stored case) when it can no longer measure anything — its frozen bundle orphaned by a schema
    change nothing can migrate, say — so no launch runs it and no price counts it, while it stays
    readable for every run already measured against it. Its own ``supersedes`` still hold: retiring
    a revision does not bring back the labels it replaced.

    Attributes:
        cases: Every readable reporter case of the template, ``(stored case, reporter case)`` in
            storage order, retired ones included. A stored case carrying no reporter case is not
            here (a template repointed to this kind can hold cases written for another).
        unreadable: ``(case id, why)`` for every stored case carrying a reporter case this build
            cannot read, in storage order.
        superseded_by: Case id -> the readable cases that list it in ``supersedes``, sorted.
        archived: Case id -> the operator's reason (or ``None``) for every readable case retired.
    """

    cases: tuple[tuple[EvalTestCase, ReporterCase], ...]
    unreadable: tuple[tuple[str, str], ...]
    superseded_by: Mapping[str, tuple[str, ...]]
    archived: Mapping[str, str | None]

    def is_live(self, case_id: str) -> bool:
        """Whether a launch runs ``case_id``: nothing readable supersedes it, and no operator retired it.

        Args:
            case_id: A stored case id.

        Returns:
            ``True`` when nothing supersedes it and it is not retired. Meaningful only while
            ``unreadable`` is empty.
        """
        return case_id not in self.superseded_by and case_id not in self.archived

    @property
    def live(self) -> list[tuple[EvalTestCase, ReporterCase]]:
        """Every live readable case, in storage order."""
        return [(stored, case) for stored, case in self.cases if self.is_live(stored.id)]

    def pair(
        self, campaign_id: str | None, recorded_analysis_id: str | None
    ) -> list[tuple[EvalTestCase, ReporterCase]]:
        """Every readable case freezing one (campaign, recorded memo) pair, live or not, in storage order.

        Args:
            campaign_id: The pair's campaign.
            recorded_analysis_id: The pair's recorded memo, or ``None``.

        Returns:
            The pair's cases, whichever bundle each was frozen from.
        """
        wanted = (campaign_id, recorded_analysis_id)
        return [(stored, case) for stored, case in self.cases if case_pair(case) == wanted]

    @property
    def ambiguous(self) -> list[AmbiguousPair]:
        """Every pair holding more than one live case, in order of its first live case."""
        by_pair: dict[tuple[str | None, str | None], list[str]] = {}
        for stored, case in self.live:
            by_pair.setdefault(case_pair(case), []).append(stored.id)
        return [
            AmbiguousPair(campaign_id=campaign_id, recorded_analysis_id=analysis_id, live_case_ids=tuple(sorted(ids)))
            for (campaign_id, analysis_id), ids in by_pair.items()
            if len(ids) > 1
        ]


def reporter_case_bank(stored: Iterable[EvalTestCase]) -> ReporterCaseBank:
    """Derive which of a template's stored reporter cases is live — the one derivation of it.

    Args:
        stored: Every stored case of ONE template in one partition, in storage order. Cases
            carrying no reporter case are skipped.

    Returns:
        The bank: readable cases, unreadable ones, and who supersedes whom.
    """
    cases: list[tuple[EvalTestCase, ReporterCase]] = []
    unreadable: list[tuple[str, str]] = []
    superseded_by: dict[str, list[str]] = {}
    archived: dict[str, str | None] = {}
    for test_case in stored:
        try:
            case = reporter_case_of(test_case)
        except ValueError as e:
            unreadable.append((test_case.id, str(e)))
            continue
        if case is None:
            continue
        cases.append((test_case, case))
        if test_case.archived:
            archived[test_case.id] = test_case.archived_reason
        for replaced in case.supersedes:
            superseded_by.setdefault(replaced, []).append(test_case.id)
    return ReporterCaseBank(
        cases=tuple(cases),
        unreadable=tuple(unreadable),
        superseded_by={case_id: tuple(sorted(ids)) for case_id, ids in superseded_by.items()},
        archived=archived,
    )


def decidable_reporter_case_bank(stored: Iterable[EvalTestCase], *, template_id: str) -> ReporterCaseBank:
    """Derive the bank for a consumer that DECIDES something from liveness, refusing when that is undecidable.

    Every such consumer reads the bank through this — the launch and its price, the freeze, the
    calibration read and a retirement's restore — so they cannot disagree about which case is live,
    least of all about a case this build cannot read.

    **A case this build cannot read is refused, not skipped.** It may supersede a case that would
    otherwise read as live, so skipping it would let a launch run a case whose labels were revised,
    a freeze revise the wrong case, a calibration read call a revised case live, and a restore make
    a second live case of a pair. Such a case is this kind's own in a shape this build does not
    know — written by a newer build, or damaged — and the refusal names it.

    Args:
        stored: Every stored case of the template in one partition, in storage order.
        template_id: The reporter template, named in the refusal.

    Returns:
        The bank, whose ``unreadable`` is empty.

    Raises:
        ValidationFailedError: A stored case of the template carries a reporter case this build
            cannot read.
    """
    bank = reporter_case_bank(stored)
    if bank.unreadable:
        raise ValidationFailedError(
            f"template {template_id!r} holds reporter case(s) this build cannot read — "
            + "; ".join(f"{case_id!r}: {why}" for case_id, why in bank.unreadable)
            + ". One of them may supersede a case that would otherwise read as live, so which case each bundle "
            "and memo launches cannot be decided here; read them with the build that wrote them."
        )
    return bank


class FrozenReporterCase(BaseModel):
    """What a reporter-case freeze stored — the receipt every surface renders.

    A projection of the stored case rather than the case itself: the frozen bundle and the
    recorded memo are whole documents a caller asked to freeze, not something the receipt
    should echo back, while everything a reader decides from — which case, which memo, the
    fingerprint to match against, the labels and the limits — is here.
    Plain ``BaseModel`` because ``labels`` carry someone's words verbatim.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    test_case_id: str = Field(description="The stored case — the existing one when the freeze matched it.")
    template_id: str = Field(description="The reporter template the case is stored under.")
    scope_id: str = Field(description="The partition the case (and the campaign's runs) live in.")
    source_campaign_id: str | None = Field(
        description="The campaign whose bundle the case froze, as the bundle records it."
    )
    recorded_analysis_id: str | None = Field(
        description="The analysis whose memo the case pins as recorded, or None when it pins none (only generating "
        "candidates can then run it)."
    )
    bundle_fingerprint: str = Field(description="The frozen bundle's fingerprint.")
    bundle_assembled_at: str = Field(
        description="When the frozen bundle was assembled (ISO-8601) — the insight ledger's as-of."
    )
    labels: list[ReporterLabel] = Field(description="The reader verdicts the case carries, in stored order.")
    limits: list[str] = Field(
        description="What the case cannot test, computed when it was frozen. Empty when nothing limits it."
    )
    supersedes: list[str] = Field(
        description="The cases whose labels this one replaced — normally one, several only when it resolved a pair "
        "holding more than one live case, empty when it replaced none. Each is kept for the runs measured against "
        "it, and no launch runs it again."
    )
    writer_message_check: WriterMessageCheck | None = Field(
        description=(
            "Whether the frozen writer message is the one the recorded memo's generator was sent: 'verified' (it "
            "digests to what the generation recorded) or 'differs' (it does not — the case's limits say why). None "
            "when the case pins no memo, and so freezes no writer message."
        )
    )
    archived: bool = Field(
        description="Whether an operator retired the case: no launch runs it and no price counts it; it stays readable."
    )
    archived_reason: str | None = Field(
        description="Why it was retired, in the operator's words — None when it is live, or was retired without one."
    )


def frozen_case_receipt(test_case: EvalTestCase) -> FrozenReporterCase:
    """Project a stored reporter case onto the receipt a freeze answers with.

    Args:
        test_case: The stored case a freeze returned.

    Returns:
        Its receipt.

    Raises:
        ValueError: ``test_case`` carries no reporter case. A freeze stores and returns only
            cases it can read back, so this is a defect in the caller rather than an input error.
    """
    case = reporter_case_of(test_case)
    if case is None:
        raise ValueError(f"test case {test_case.id!r} carries no reporter case")
    return FrozenReporterCase(
        test_case_id=test_case.id,
        template_id=test_case.template_id,
        scope_id=test_case.scope_id,
        source_campaign_id=_str_or_none(case.bundle.get("campaign_id")),
        recorded_analysis_id=case.recorded_analysis_id,
        bundle_fingerprint=case.bundle_fingerprint,
        bundle_assembled_at=case.bundle_assembled_at,
        labels=list(case.labels),
        limits=list(case.limits),
        supersedes=list(case.supersedes),
        writer_message_check=writer_message_check(case),
        archived=test_case.archived,
        archived_reason=test_case.archived_reason,
    )


#: How a label's criterion compares with the template's today — see :attr:`LabelReading.criterion_drift`.
CriterionDrift = Literal["unchanged", "changed", "no_live_criterion"]


def criterion_drift(label: ReporterLabel, live: Mapping[str, LabelCriterion] | None) -> CriterionDrift:
    """Compare the criterion a label was written against with the template's criterion today.

    Args:
        label: A stored label, carrying the criterion its freeze stamped.
        live: The template's criteria by dimension today, or ``None`` when the template is gone.

    Returns:
        ``no_live_criterion`` when the template is gone or no longer scores the dimension; otherwise
        ``unchanged`` or ``changed``.

    Raises:
        ValueError: ``label`` carries no criterion — a label as a caller submits it, never one a case
            stores (``ReporterCase`` refuses that), so nothing says what it was written against.
    """
    if label.criterion is None:
        raise ValueError(
            f"the label on {label.dimension!r} carries no criterion; only a label a case stored, which its freeze "
            "stamped, can be compared with the template"
        )
    today = None if live is None else live.get(label.dimension)
    if today is None:
        return "no_live_criterion"
    return "unchanged" if today == label.criterion else "changed"


class LabelReading(BaseModel):
    """One reader's verdict on a dimension, beside what the judge scored there.

    Plain ``BaseModel`` because ``quote`` and ``reasoning`` are someone's words verbatim.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    dimension: str = Field(description="The rubric dimension the label bears on.")
    direction: LabelDirection = Field(description="Where the reader placed the memo on it.")
    quote: str = Field(description="The reader's own words, verbatim.")
    note: str | None = Field(default=None, description="Why the quote maps to this dimension, when recorded.")
    score: int | None = Field(
        description="The judge's stored score on this dimension (1-5, or 1/0 on pass/fail), or None when it scored none."
    )
    score_label: str | None = Field(
        description="The score as a reader reads it: the 1-5 integer, or pass/fail. None when it scored none."
    )
    reasoning: str | None = Field(description="The judge's reasoning for that score, or None when it scored none.")
    agrees: bool | None = Field(
        description=(
            "Whether the score falls inside the label's band (LABEL_BANDS, inclusive). None when there is no "
            "score to compare, or the dimension is pass/fail and no band applies — never read as disagreement."
        )
    )
    criterion_drift: CriterionDrift = Field(
        description=(
            "Whether the criterion the label was written against still reads as the template states it today: "
            "`unchanged`; `changed` — the dimension was reworded since, so a disagreement is not evidence against "
            "the judge; `no_live_criterion` — the template is gone or no longer scores the dimension. Compared with "
            "the template as read NOW: a run records no criterion text, so a wording change between the run and "
            "this read is not visible here."
        )
    )


class DimensionReading(BaseModel):
    """A dimension the judge scored that no label speaks to — reported, never compared.

    Plain ``BaseModel`` because ``reasoning`` is the judge's words verbatim.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    dimension: str = Field(description="The rubric dimension.")
    score: int | None = Field(
        description="The judge's stored score (1-5, or 1/0 on pass/fail), or None when it scored none."
    )
    score_label: str | None = Field(
        description="The score as a reader reads it: the 1-5 integer, or pass/fail. None when it scored none."
    )
    reasoning: str | None = Field(description="The judge's reasoning, or None when it scored none.")


class CalibrationCell(BaseModel):
    """One result of one case: a single (model, repeat) pass, read against the case's labels."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    result_id: str = Field(description="The stored result this reading comes from.")
    model: str = Field(description="The candidate model the result recorded — the as-recorded sentinel included.")
    k_iteration: int = Field(description="Which repeat of this (case, model) it was.")
    termination: str | None = Field(
        default=None,
        description="How the cell ended when it ended abnormally (e.g. an apparatus fault); None for an ordinary cell.",
    )
    labelled: list[LabelReading] = Field(default_factory=list, description="One reading per label the case carries.")
    unlabelled: list[DimensionReading] = Field(
        default_factory=list, description="Every rubric dimension no label speaks to, with what the judge scored there."
    )


class CalibrationCase(BaseModel):
    """One frozen case of the run, with everything its results say against its labels."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    test_case_id: str = Field(description="The case.")
    source_campaign_id: str | None = Field(
        description="The campaign whose bundle the case froze, as the bundle records it."
    )
    source_analysis_id: str | None = Field(
        description="The analysis whose memo the case pins as recorded, or None when it pins none."
    )
    bundle_fingerprint: str = Field(description="The frozen bundle's fingerprint.")
    limits: list[str] = Field(default_factory=list, description="What the case cannot test, stated when it was frozen.")
    supersedes: list[str] = Field(
        default_factory=list,
        description="The cases whose labels this one replaced — normally one; empty when it replaced none.",
    )
    superseded_by: list[str] = Field(
        default_factory=list,
        description=(
            "The stored cases that have since replaced this case's labels — normally one, empty while it is live. "
            "When set, this run is read against labels a reader later revised, and launches no longer run this case."
        ),
    )
    archived: bool = Field(
        default=False, description="Whether an operator has since retired the case, so no launch runs it again."
    )
    archived_reason: str | None = Field(
        default=None, description="Why it was retired, in the operator's words — None when live or retired without one."
    )
    writer_message_check: WriterMessageCheck | None = Field(
        description=(
            "Whether the writer message the case froze — what its recorded memo's cells were judged against — is "
            "the one that memo's generator was sent: 'verified', or 'differs' (the memo was judged against a "
            "message its writer did not read). None when the case pins no recorded memo."
        )
    )
    cells: list[CalibrationCell] = Field(
        default_factory=list,
        description="One entry per stored result of this case, ordered by (model, repeat). Empty when none was stored.",
    )


class ReporterCalibration(BaseModel):
    """A reporter run read against its cases' labels — the rubric's calibration, as data.

    A read over stored results: no model is called and nothing is written.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(description="The run read.")
    template_id: str = Field(description="The template it ran.")
    status: str = Field(description="The run's status when read — a running run reads what it has stored so far.")
    dimensions: list[str] = Field(description="The rubric dimensions the run scores, in the template's order.")
    judge_model: str | None = Field(
        description=(
            "The run's judge pin — the model a dimension falls back to when no judge config names its own. "
            "None when the run was not judged."
        )
    )
    effective_judges: dict[str, str] | None = Field(
        description=(
            "The model that scored each dimension, as the run recorded it at launch — what every score in this "
            "read came from, so agreement is agreement WITH THAT JUDGE. None when the run recorded no attribution."
        )
    )
    cases: list[CalibrationCase] = Field(default_factory=list, description="One entry per reporter case, in run order.")
    missing_case_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Cases the run froze that no longer load, in run order. Named rather than dropped: their labels "
            "cannot be read, so the calibration covers fewer cases than the run measured."
        ),
    )
    rating_agreement: JudgeAgreement = Field(
        default_factory=JudgeAgreement,
        description=(
            "The run's results read against people's calibration ratings of them, per dimension and judge — the "
            "same read the bundle's `judge_agreement` makes. A label places a memo in a band and a rating gives it "
            "a score, so the two are read side by side rather than pooled: labels in `cases`, ratings here."
        ),
    )


def label_agrees(direction: LabelDirection, score: RubricScore) -> bool | None:
    """Whether a judge score falls inside a label direction's band.

    Args:
        direction: Where the label placed the memo.
        score: The judge's score.

    Returns:
        ``True`` when the score is inside the direction's inclusive band; ``None`` for a pass/fail
        dimension. The bands are 1-5 levels, and mapping five directions onto two answers would be a
        calibration rule nobody ratified — and one pass/fail dimension must not abort the read of the
        rest, which a raise here would do after the run was paid for.
    """
    if score.scale != "ordinal":
        return None
    low, high = LABEL_BANDS[direction]
    return low <= score.score <= high


def read_calibration(
    *,
    run_id: str,
    template_id: str,
    status: str,
    dimensions: Sequence[str],
    cases: Sequence[tuple[str, ReporterCase]],
    results: Sequence[EvalResult],
    superseded_by: Mapping[str, Sequence[str]],
    archived: Mapping[str, str | None],
    live_criteria: Mapping[str, LabelCriterion] | None,
    judge_model: str | None,
    effective_judges: Mapping[str, str] | None,
    ratings: Sequence[CalibrationRating],
    missing_case_ids: Sequence[str] = (),
) -> ReporterCalibration:
    """Set every labelled dimension's judge score beside its label, per case and per result.

    Args:
        run_id: The run.
        template_id: Its template.
        status: Its status when read.
        dimensions: The rubric dimensions the run scores, in order. A label naming a dimension
            outside this list is still read — the judge simply scored nothing there, which the
            reading says with ``score=None``.
        cases: ``(test_case_id, case)`` for every reporter case of the run, in run order.
        results: Every stored result of the run. A result whose case is not among ``cases`` is
            ignored.
        superseded_by: Case id → the stored cases that replaced its labels since, for every case
            of the template some stored case supersedes. A case absent from it is live.
        archived: Case id → the operator's reason (or ``None``) for every case of the template
            retired since.
        live_criteria: The template's criteria by dimension as it reads now, or ``None`` when the
            template is gone — what each label's frozen criterion is compared with.
        judge_model: The run's judge pin, carried through so the read names its judge.
        effective_judges: The run's recorded per-dimension judges, carried through likewise.
        ratings: Every calibration rating of the run's results, read against ``results`` into
            ``rating_agreement``.
        missing_case_ids: Cases the run froze that did not load, carried through for disclosure.

    Returns:
        The calibration read.
    """
    by_case: dict[str, list[EvalResult]] = {}
    for result in results:
        by_case.setdefault(result.test_case_id, []).append(result)

    read_cases: list[CalibrationCase] = []
    for test_case_id, case in cases:
        labelled_dims = {label.dimension for label in case.labels}
        cells: list[CalibrationCell] = []
        for result in sorted(by_case.get(test_case_id, []), key=lambda r: (r.model, r.k_iteration)):
            scored = {score.dim: score for score in result.rubric_scores}
            readings = []
            for label in case.labels:
                hit = scored.get(label.dimension)
                readings.append(
                    LabelReading(
                        dimension=label.dimension,
                        direction=label.direction,
                        quote=label.quote,
                        note=label.note,
                        score=hit.score if hit is not None else None,
                        score_label=hit.label if hit is not None else None,
                        reasoning=hit.reasoning if hit is not None else None,
                        agrees=label_agrees(label.direction, hit) if hit is not None else None,
                        criterion_drift=criterion_drift(label, live_criteria),
                    )
                )
            unlabelled = [
                DimensionReading(
                    dimension=dim,
                    score=scored[dim].score if dim in scored else None,
                    score_label=scored[dim].label if dim in scored else None,
                    reasoning=scored[dim].reasoning if dim in scored else None,
                )
                for dim in dimensions
                if dim not in labelled_dims
            ]
            cells.append(
                CalibrationCell(
                    result_id=result.id,
                    model=result.model,
                    k_iteration=result.k_iteration,
                    termination=_termination_of(result),
                    labelled=readings,
                    unlabelled=unlabelled,
                )
            )
        read_cases.append(
            CalibrationCase(
                test_case_id=test_case_id,
                source_campaign_id=_str_or_none(case.bundle.get("campaign_id")),
                source_analysis_id=case.recorded_analysis_id,
                bundle_fingerprint=case.bundle_fingerprint,
                limits=list(case.limits),
                supersedes=list(case.supersedes),
                superseded_by=sorted(superseded_by.get(test_case_id, ())),
                archived=test_case_id in archived,
                archived_reason=archived.get(test_case_id),
                writer_message_check=writer_message_check(case),
                cells=cells,
            )
        )
    return ReporterCalibration(
        run_id=run_id,
        template_id=template_id,
        status=status,
        dimensions=list(dimensions),
        judge_model=judge_model,
        effective_judges=None if effective_judges is None else dict(effective_judges),
        cases=read_cases,
        missing_case_ids=list(missing_case_ids),
        rating_agreement=judge_agreement(ratings, results),
    )


def _termination_of(result: EvalResult) -> str | None:
    """Name how a cell ended, when it ended abnormally.

    Args:
        result: The stored result.

    Returns:
        The termination, or ``None`` for a cell that completed.
    """
    return None if result.termination == "completed" else result.termination


def _str_or_none(value: Any) -> str | None:
    """Pass a string through, and anything else as absent.

    Args:
        value: A value read off a frozen document.

    Returns:
        The string, or ``None``.
    """
    return value if isinstance(value, str) and value else None


__all__ = [
    "AmbiguousPair",
    "CalibrationCase",
    "CalibrationCell",
    "CriterionDrift",
    "DimensionReading",
    "FrozenReporterCase",
    "LabelReading",
    "ReporterCalibration",
    "ReporterCaseBank",
    "case_limits",
    "case_pair",
    "criterion_drift",
    "frozen_case_receipt",
    "label_agrees",
    "read_calibration",
    "reporter_case_bank",
]
