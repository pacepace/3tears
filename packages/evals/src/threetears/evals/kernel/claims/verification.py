"""Checking each listed claim against numbered evidence: the verify step of claims grading, and the verdict it gives.

:class:`ClaimVerifier` takes the claims :class:`~threetears.evals.kernel.claims.extraction.ClaimExtractor` listed and
asks a model, in batches, which numbered pieces of evidence support or contradict each one. It never asks for a
verdict: the verdict is derived in code from the citations that pass the checks (:func:`truth_verdict`).

**Two steps, two kinds of evidence.** The *truth* step sees only what is the same for every arm: the case key's facts,
numbered ``K1..``, and passages of the case's own documents, numbered ``S1..``. It never sees a document the candidate
was given or fetched itself (a ``run`` document): those never nominate a claim and never decide one. The *ground* step,
asked only when the criterion names ``grounded_in``, sees only passages of documents of those provenances, the
candidate's own, and cites support alone. A key fact never grounds a claim, since the candidate was never shown it.

**Passages are chosen by code, under the page budget, the same way every time.** Each document is cut into tiles of
``window_chars`` from its start. Each tile is ranked by the anchors it holds: the folded words, of at least four
characters or holding a digit, of the cell's world claims and the key's names and aliases, each weighted by one over
the number of the document's tiles that hold it, so a rare name outweighs a common word. Tiles are then taken in rounds
across the documents, in their order, each document's best remaining tile per round, while the document's
``per_document_chars`` and the step's ``total_chars`` allow; adjacent tiles join into one passage. A document not read
whole is recorded as truncated. Each step's selection is bounded on its own, so the documents a candidate fetched never
take the case's share: truth reads the same case passages whatever an arm fetched, given the same claims.

**A citation names a value and its subject, in one sentence or row.** A passage citation quotes at most
:data:`~threetears.evals.schema.models.CLAIM_QUOTE_CHARS` characters, found by
:func:`~threetears.evals.kernel.claims.verbatim.verbatim_span` inside the passage the call was shown and cited (never
elsewhere in its document), and names, inside that quote, the words that name the claim's subject and the words that
state the value. Both are found inside the stored quote, and no sentence end (``.``, ``!`` or ``?`` before a space or
the end), semicolon or line break may lie between them. That narrows the neighbouring-value error, a value taken from
the sentence or row next to the one about this subject, to one sentence or row. It does not prove the value belongs to
that subject: a sentence stating two values for two subjects ("the pamphlet in 1931 and the bound volume in 1932")
stays the model's to bind, and calibration measures what remains. What is stored is the passage's own slice, never the
model's strings. A key fact is cited whole, with no quote.

**Batches.** One call per unit, holding every world claim of that unit, and one more for the claims outside any unit.
A restatement is not asked again: it rides in its original's batch, shown beside it, and takes the original's
citations, suspicion and verdict, since it states the same fact. Session claims are about the output itself and are
never verified here; their finding is None.

**Repair, not refusal.** A reply that will not parse, or does not answer exactly the claims asked, is asked again in
full, naming why, up to three times, then fails (``unreadable``). A claim whose answer fails a check (a ref to no
evidence shown, a quote not in the passage cited or over the length, a missing or misplaced span, a subject and value
in different sentences, a suspect claim that cites evidence) is asked again alone, with its failure named, within the
repair budget: its rounds, and the most items one cell may ask again, which each step holds on its own. Only failing
claims are asked again. Still failing, or past the budget, fails (``repair_budget_spent``); a reply cut short fails at
once (``cut_short``).

**Truth and groundedness fail apart.** A truth failure raises
:class:`~threetears.evals.kernel.claims.failures.ClaimGradingFailed`, costing the cell this criterion. A ground
failure never touches truth: the documents an arm fetched differ by arm, so they must not decide whether its cell
keeps a truth reading. The verification comes back with every truth finding, no groundedness, and the ground step's
loss on :attr:`Verification.ground_loss`, for the judge phase to record against the groundedness reading alone.

**It records what it spent, and how it read.** Every completion is folded into the cell's ``judge`` ledger and
recorded as a :class:`~threetears.evals.schema.models.ClaimCall`. :attr:`ClaimVerifier.identity` names the model and a
digest of everything that shapes a verdict: the prompts with the host's guidance, the reply schemas, the page budget
and the selection rule, the batch rule, the repair budget, ``grounded_in``, the quote cap, the attempts and the
verbatim fold's identity.

Engine-internal: the judge phase builds and calls it, and nothing here is exported.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType
from typing import Any, Final, Literal

from pydantic import Field, ValidationError

from threetears.evals.kernel.authored import strict_response_format
from threetears.evals.kernel.claims.extraction import ExtractedClaim
from threetears.evals.kernel.claims.failures import ClaimGradingFailed
from threetears.evals.kernel.claims.verbatim import VERBATIM_FOLD, verbatim_span
from threetears.evals.kernel.provider import describe_incomplete_completion, extract_json
from threetears.evals.kernel.usage_capture import RoleUsageLedger
from threetears.evals.schema.base import EvalBaseModel
from threetears.evals.schema.completion import (
    CompletionClient,
    CompletionProtocolDefect,
    CompletionResult,
    sent_settings_of,
)
from threetears.evals.schema.hashing import canonical_digest
from threetears.evals.schema.models import (
    CLAIM_QUOTE_CHARS,
    MODEL_DEFAULT_TEMPERATURE,
    ClaimCall,
    ClaimCitation,
    EvidenceDocument,
    EvidenceProvenance,
    EvidenceRef,
    GradedClaim,
    JudgeTemperature,
    KnownClaimLoss,
    PageBudget,
    RepairBudget,
    TextSpan,
)
from threetears.evals.schema.reference import ReferenceFact, ReferenceKey

#: How many times one batch is asked in full: the first call and two more when a finished reply will not parse.
_ATTEMPTS: Final = 3

#: The ledger every verify call is billed to: it reads the candidate's output, as a judge does.
_ROLE: Final = "judge"

#: Something other than whitespace.
_NOT_BLANK: Final = r"\S"

#: The fewest characters a folded word needs to anchor a passage, unless it holds a digit.
_ANCHOR_MIN_CHARS: Final = 4

#: A word, once folded.
_WORD: Final = re.compile(r"\w+")

#: The page-selection rule, by revision and in words: part of the identity, raised with any change to the code below.
_SELECTION_RULE: Final = (
    "passages-v1: tiles of window_chars from each document's start; each ranked by the anchors it holds (folded words "
    "of the world claims and the key's names and aliases, of at least 4 characters or holding a digit), each weighted "
    "1 / the number of the document's tiles holding it, ties to the earlier tile; tiles taken in rounds across the "
    "documents in order, one per document per round, while per_document_chars and the step's total_chars allow; "
    "adjacent tiles joined; each step selected on its own"
)

#: The batch rule, in words: part of the identity.
_BATCH_RULE: Final = (
    "one batch per unit in unit order, then one for claims outside any unit; a restatement rides with its original "
    "and takes its answer; session claims are not verified"
)

#: What a claim is, for a check: the same fact about the same subject. Shared by every verify prompt.
_CITING_RULES: Final = (
    "- Bind the value to its subject. Evidence that states a value for a neighbouring subject (another item, edition, "
    "date, person or place in the same passage) bears on nothing here, even when its value matches the claim's.\n"
    "- An omission contradicts nothing: evidence that does not state the claim's fact, or states other facts about "
    "its subject, is not cited.\n"
    "- For a passage (S), quote the one sentence or table row that states it, copied exactly as it is written, at most "
    f"{CLAIM_QUOTE_CHARS} characters, never shortened with an ellipsis. Give subject: the words of that quote that "
    "name the claim's subject; and value: the words of that quote that state the value at issue.\n"
    "- Cite each piece of evidence at most once for a claim."
)

_TRUTH_RULES: Final = (
    "Read each claim against the numbered evidence given, and nothing else: not what you remember.\n"
    "- Cite every piece of evidence that bears on a claim, by its ref (K1, S1, ...), with its stance: supports when "
    "it states this same fact about this same subject; contradicts when it states a different value for this same "
    "subject and relation. A contradiction's value is the value the evidence gives instead.\n"
    + _CITING_RULES
    + "\n- For a key fact (K), give quote, subject and value as null: a key fact is cited whole.\n"
    "- suspect: true only when no evidence given bears on the claim and you doubt that it is true; false otherwise. "
    "A claim you cite evidence for is never suspect."
)

_GROUND_RULES: Final = (
    "Read each claim against the numbered passages given, which come from documents the text's author was given, and "
    "nothing else: not what you remember.\n"
    "- Cite every passage that supports a claim, by its ref (S1, S2, ...): one that states this same fact about this "
    "same subject. Never cite a passage that only mentions the subject, or that states something else.\n"
    + _CITING_RULES
)

_GUIDANCE_HEADING: Final = "\n\n# Notes on reading claims here\n\n"

_ANSWER_LINE: Final = "\n\nAnswer with JSON only, in the shape the response format states."

_TRUTH_SYSTEM: Final = (
    "You check the factual claims a text states against numbered evidence. Each claim is numbered; answer for every "
    "claim asked, by its number.\n\n" + _TRUTH_RULES + _ANSWER_LINE
)

_GROUND_SYSTEM: Final = (
    "You check which factual claims a text states are supported by numbered passages. Each claim is numbered; answer "
    "for every claim asked, by its number.\n\n" + _GROUND_RULES + _ANSWER_LINE
)

_TRUTH_REPAIR_SYSTEM: Final = (
    "You correct your answer for one claim checked against numbered evidence. You are given the evidence, the claim, "
    "your answer and what is wrong with it. Answer for the claim again with that fixed.\n\n"
    + _TRUTH_RULES
    + _ANSWER_LINE
)

_GROUND_REPAIR_SYSTEM: Final = (
    "You correct your answer for one claim checked against numbered passages. You are given the passages, the claim, "
    "your answer and what is wrong with it. Answer for the claim again with that fixed.\n\n"
    + _GROUND_RULES
    + _ANSWER_LINE
)

_USER_TEMPLATE: Final = "# Evidence\n\n{evidence}\n\n# Claims\n\n{claims}"

_FACT_TEMPLATE: Final = "{ref}. A key fact about {entity}: {claim}"

_PASSAGE_TEMPLATE: Final = "{ref}. From {title}, characters {start} to {end}:\n{text}"

_NO_EVIDENCE: Final = "None."

_CLAIM_TEMPLATE: Final = 'Claim {position}, {place}: {claim}\nIt is said as: "{excerpt}"'

_RESTATED_TEMPLATE: Final = 'It is said again, as claim {position}: "{excerpt}"'

_AGAIN_TEMPLATE: Final = "{user}\n\n# Your last reply\n\nIt could not be used: {failure}. Answer again, in full."

_REPAIR_TEMPLATE: Final = (
    "# Evidence\n\n{evidence}\n\n# The claim\n\n{claim}\n\n# Your answer\n\n{answer}\n\n# What is wrong\n\n{failure}"
)

#: What breaks one sentence or table row from the next, between a citation's subject and its value: a sentence end
#: (``.``, ``!`` or ``?`` before a space or the end), a semicolon or a line break.
_SENTENCE_BREAK: Final = re.compile(r"[.!?](?=\s|$)|[;\n\r]")

#: Every failure a repair or a full re-ask names to the model, as the templates it is written from: what the judge is
#: shown, so part of the identity.
_FAULTS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "twice": "it cites {refs} more than once; cite each piece of evidence once",
        "suspect": "it is marked suspect and cites evidence; suspect means no evidence given bears on the claim",
        "unknown_ref": "it cites {ref}, which is no evidence shown here; cite the refs listed under Evidence",
        "no_quote": "its citation of {ref} quotes nothing; a passage is cited with the sentence or row that states it",
        "not_shown": (
            "its quote of {ref} is not in passage {ref} as shown; a quote is one unbroken stretch of the passage it "
            "cites, copied as it is written"
        ),
        "too_long": "its quote of {ref} runs {length} characters; a quote is one sentence or row of at most {cap}",
        "no_span": "its citation of {ref} names no {what}; give the words of its quote that state the {what}",
        "span_outside": "its {what} for {ref} is not in its quote; the {what} is words of the quoted sentence itself",
        "apart": (
            "its subject and value for {ref} stand in different sentences or rows of its quote; quote the one sentence "
            "or row that states this value for this subject"
        ),
        "correction": "{failure}; and the last correction could not be used: {why}",
        "shape": "it is not the shape the response format states ({problems})",
        "answers": "it must answer claims {asked} once each, and it {problems}",
        "missing": "leaves out claims {missing}",
        "extra": "answers claims {extra} that were not asked, or more than once",
    }
)

_NULLABLE_STRING: Final[dict[str, Any]] = {"anyOf": [{"type": "string"}, {"type": "null"}]}

_TRUTH_CITATION: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "ref": {"type": "string"},
        "stance": {"type": "string", "enum": ["supports", "contradicts"]},
        "quote": _NULLABLE_STRING,
        "subject": _NULLABLE_STRING,
        "value": _NULLABLE_STRING,
    },
}

_GROUND_CITATION: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "ref": {"type": "string"},
        "quote": _NULLABLE_STRING,
        "subject": _NULLABLE_STRING,
        "value": _NULLABLE_STRING,
    },
}

type _Step = Literal["truth", "ground"]

#: The steps in the order their batches run and their failing items are asked again.
_STEPS: Final[tuple[_Step, ...]] = ("truth", "ground")

#: A web check's outcome as the verdict table reads it: what the page said, or that the cell's budget skipped it.
type WebOutcome = Literal["true", "false", "inconclusive", "skipped_cap"]

type Verdict = Literal["supported", "contradicted", "unverifiable", "true_ungrounded"]


class _Cited(EvalBaseModel):
    """One citation as a truth reply gives it."""

    ref: str = Field(pattern=_NOT_BLANK)
    stance: Literal["supports", "contradicts"]
    quote: str | None
    subject: str | None
    value: str | None


class _GroundCited(EvalBaseModel):
    """One citation as a ground reply gives it: support, so it carries no stance."""

    ref: str = Field(pattern=_NOT_BLANK)
    quote: str | None
    subject: str | None
    value: str | None

    def as_cited(self) -> _Cited:
        return _Cited(stance="supports", **self.model_dump())


class _Answer(EvalBaseModel):
    """One claim's answer under either step. A ground answer is never suspect."""

    citations: list[_Cited]
    suspect: bool


class _TruthItem(EvalBaseModel):
    claim: int = Field(ge=0)
    citations: list[_Cited]
    suspect: bool


class _TruthBatch(EvalBaseModel):
    claims: list[_TruthItem]


class _GroundItem(EvalBaseModel):
    claim: int = Field(ge=0)
    citations: list[_GroundCited]


class _GroundBatch(EvalBaseModel):
    claims: list[_GroundItem]


class _TruthRepaired(EvalBaseModel):
    citations: list[_Cited]
    suspect: bool


class _GroundRepaired(EvalBaseModel):
    citations: list[_GroundCited]


@dataclass(frozen=True)
class ClaimFinding:
    """What the verify step found for one world claim: the citations that passed their checks, and the call behind.

    :ivar truth_citations: Case evidence cited for or against it, each checked.
    :ivar suspect: No case evidence bears on it, and the model doubts it: the judge's nomination for the web check.
    :ivar truth_call: The call whose answer stands for it, by its place in :attr:`Verification.calls`: its batch's,
        or the repair that fixed it. A restatement's is its original's.
    :ivar ground_citations: Documents the candidate was given that support it, each checked; empty when not asked.
    :ivar grounded: Whether one supports it; None when the criterion names no ``grounded_in``, or the ground step was
        lost.
    """

    truth_citations: tuple[ClaimCitation, ...]
    suspect: bool
    truth_call: int
    ground_citations: tuple[ClaimCitation, ...] = ()
    grounded: bool | None = None


@dataclass(frozen=True)
class Verification:
    """What the verify step read and found for one cell.

    :ivar evidence: Every piece of evidence a call was shown, numbered as the calls cited it.
    :ivar documents_truncated: The documents a shown selection did not read whole, by id, in document order.
    :ivar calls: Every call made, in order.
    :ivar findings: One per claim, in the claims' order: a :class:`ClaimFinding` for a world claim, None for a session
        claim.
    :ivar ground_loss: Why the ground step was lost (``cut_short``, ``unreadable`` or ``repair_budget_spent``), its
        own loss and never the criterion's: the truth findings stand, and every finding's ``grounded`` reads None.
        None when the step was not lost, or not asked.
    :ivar ground_failure: What failed in the ground step, named, beside :attr:`ground_loss`; None with it.
    """

    evidence: tuple[EvidenceRef, ...]
    documents_truncated: tuple[str, ...]
    calls: tuple[ClaimCall, ...]
    findings: tuple[ClaimFinding | None, ...]
    ground_loss: KnownClaimLoss | None = None
    ground_failure: str | None = None


@dataclass(frozen=True)
class TruthVerdict:
    """A world claim's truth verdict, derived from its case citations, its suspicion and a web outcome.

    :ivar verdict: ``supported``, ``contradicted``, ``unverifiable`` or ``true_ungrounded``.
    :ivar graded_against: ``case`` or ``web``, what decided it; None for ``unverifiable``.
    :ivar evidence_conflict: Case evidence both supports and contradicts it; the support stands.
    :ivar nominated_by: ``judge`` when no case evidence bears on it and the model doubts it; else None.
    """

    verdict: Verdict
    graded_against: Literal["case", "web"] | None
    evidence_conflict: bool
    nominated_by: Literal["judge"] | None


def truth_verdict(citations: Sequence[ClaimCitation], *, suspect: bool, web: WebOutcome | None = None) -> TruthVerdict:
    """The verdict table: case support always reads supported, case contradiction contradicted, and the web decides
    only a claim no case evidence bears on that the model doubts.

    :param citations: The claim's checked case citations.
    :param suspect: Whether the model doubts it, with no case evidence bearing on it.
    :param web: What the web check found for it; None when it was not checked (web off, or not yet run).
    :return: The verdict, what decided it, whether case evidence conflicts, and who nominated it.
    :raises ValueError: A web outcome is given for a claim nobody nominated: the web never overrules case evidence and
        never checks a claim the model did not doubt.
    """
    supports = any(citation.stance == "supports" for citation in citations)
    contradicts = any(citation.stance == "contradicts" for citation in citations)
    nominated = not citations and suspect
    if web is not None and not nominated:
        raise ValueError(
            "a web outcome was given for a claim nobody nominated: the web checks only a claim no case evidence bears "
            "on that the judge doubts, and never overrules case evidence"
        )
    if supports:
        return TruthVerdict("supported", "case", contradicts, None)
    if contradicts:
        return TruthVerdict("contradicted", "case", False, None)
    if not nominated:
        return TruthVerdict("unverifiable", None, False, None)
    if web == "true":
        return TruthVerdict("true_ungrounded", "web", False, "judge")
    if web == "false":
        return TruthVerdict("contradicted", "web", False, "judge")
    return TruthVerdict("unverifiable", None, False, "judge")


def graded_claims(
    claims: Sequence[ExtractedClaim], verification: Verification, *, calls_before: int
) -> tuple[GradedClaim, ...]:
    """Each claim as the grade stores it, with no web check run: a nominated claim reads ``unverifiable``.

    :param claims: The cell's claims, as listed.
    :param verification: What :meth:`ClaimVerifier.verify` found for them.
    :param calls_before: How many of the grade's calls precede the verify step's (the extraction's), so each
        ``decided_by`` is a place in the grade's calls.
    :return: One graded claim per claim, in order.
    :raises ValueError: The verification holds a finding for another number of claims.
    """
    if len(verification.findings) != len(claims):
        raise ValueError(
            f"a verification holds {len(verification.findings)} findings, and there are {len(claims)} claims"
        )
    graded: list[GradedClaim] = []
    for claim, finding in zip(claims, verification.findings, strict=True):
        # The extractor's fields are the stored claim's own: where it stands, the proposition and its tags.
        base: dict[str, Any] = claim.model_dump()
        if finding is None:
            graded.append(GradedClaim.model_validate(base))
            continue
        verdict = truth_verdict(finding.truth_citations, suspect=finding.suspect)
        graded.append(
            GradedClaim.model_validate(
                {
                    **base,
                    "truth_citations": finding.truth_citations,
                    "ground_citations": finding.ground_citations,
                    "suspect": finding.suspect,
                    "nominated_by": verdict.nominated_by,
                    "verdict": verdict.verdict,
                    "graded_against": verdict.graded_against,
                    "grounded": finding.grounded,
                    "evidence_conflict": verdict.evidence_conflict,
                    "decided_by": calls_before + finding.truth_call,
                }
            )
        )
    return tuple(graded)


@dataclass(frozen=True)
class _Passage:
    """One numbered passage: a stretch of a document, by offsets in its text."""

    document: EvidenceDocument
    start: int
    end: int


@dataclass(frozen=True)
class _Shown:
    """What one step's calls are shown: its numbered facts and passages, rendered once."""

    facts: tuple[tuple[str, str, ReferenceFact], ...]
    passages: tuple[tuple[str, _Passage], ...]

    @property
    def refs(self) -> frozenset[str]:
        return frozenset([ref for ref, _, _ in self.facts] + [ref for ref, _ in self.passages])

    def passage(self, ref: str) -> _Passage | None:
        return next((passage for shown, passage in self.passages if shown == ref), None)

    def rendered(self) -> str:
        lines = [_FACT_TEMPLATE.format(ref=ref, entity=entity, claim=fact.claim) for ref, entity, fact in self.facts]
        lines += [
            _PASSAGE_TEMPLATE.format(
                ref=ref,
                title=json.dumps(passage.document.title, ensure_ascii=False),
                start=passage.start,
                end=passage.end,
                text=passage.document.text[passage.start : passage.end],
            )
            for ref, passage in self.passages
        ]
        return "\n\n".join(lines) or _NO_EVIDENCE


def _anchors(texts: Iterable[str]) -> frozenset[str]:
    """The folded words of ``texts`` that anchor a passage: at least :data:`_ANCHOR_MIN_CHARS` long, or holding a
    digit. A retrieval key, never a check of what any text says."""
    words: set[str] = set()
    for text in texts:
        folded = VERBATIM_FOLD.fold(text)[0]
        words.update(
            word for word in _WORD.findall(folded) if len(word) >= _ANCHOR_MIN_CHARS or any(c.isdigit() for c in word)
        )
    return frozenset(words)


def select_passages(
    documents: Sequence[EvidenceDocument], anchors: frozenset[str], budget: PageBudget
) -> tuple[tuple[_Passage, ...], tuple[str, ...]]:
    """The passages one step reads under ``budget``, and the documents it did not read whole (the selection rule,
    :data:`_SELECTION_RULE`).

    :param documents: The step's documents, in order.
    :param anchors: The folded words a passage is ranked by.
    :param budget: The criterion's page budget.
    :return: The passages in document order and, within one, by offset; and the ids of the documents truncated.
    """
    width = budget.window_chars
    ranked: list[list[tuple[int, int]]] = []
    for document in documents:
        text = document.text
        tiles = [(start, min(start + width, len(text))) for start in range(0, len(text), width)]
        held = [_anchors([text[start:end]]) & anchors for start, end in tiles]
        spread = Counter(word for words in held for word in words)
        weight = [sum((Fraction(1, spread[word]) for word in words), Fraction(0)) for words in held]
        order = sorted(range(len(tiles)), key=lambda at: (-weight[at], at))
        ranked.append([tiles[at] for at in order])
    taken: list[list[tuple[int, int]]] = [[] for _ in documents]
    document_left = [budget.per_document_chars] * len(documents)
    total_left = budget.total_chars
    for rank in range(max((len(tiles) for tiles in ranked), default=0)):
        for place, tiles in enumerate(ranked):
            if rank >= len(tiles):
                continue
            start, end = tiles[rank]
            if end - start <= min(document_left[place], total_left):
                taken[place].append((start, end))
                document_left[place] -= end - start
                total_left -= end - start
    passages: list[_Passage] = []
    truncated: list[str] = []
    for document, spans in zip(documents, taken, strict=True):
        joined: list[list[int]] = []
        for start, end in sorted(spans):
            if joined and joined[-1][1] == start:
                joined[-1][1] = end
            else:
                joined.append([start, end])
        passages.extend(_Passage(document, start, end) for start, end in joined)
        if sum(end - start for start, end in joined) < len(document.text):
            truncated.append(document.id)
    return tuple(passages), tuple(truncated)


def _batches(claims: Sequence[ExtractedClaim]) -> list[tuple[int, ...]]:
    """The world claims asked, by position, one batch per unit in unit order and one for claims outside any unit.

    A restatement is not asked: it takes its original's answer.
    """
    units: dict[int, list[int]] = {}
    outside: list[int] = []
    for position, claim in enumerate(claims):
        # prose-canary: allow — `scope` is the closed Literal world/session the extractor checked, not free text
        if claim.scope != "world" or claim.restates is not None:
            continue
        if claim.unit is None:
            outside.append(position)
        else:
            units.setdefault(claim.unit, []).append(position)
    batches = [tuple(units[unit]) for unit in sorted(units)]
    return [*batches, tuple(outside)] if outside else batches


def _sent_temperature(completion: CompletionResult) -> JudgeTemperature:
    """The temperature a completion was sent at, as its client reports it.

    :raises CompletionProtocolDefect: The completion carries no ``temperature``.
    """
    try:
        sent = completion.temperature
    except AttributeError as missing:
        raise CompletionProtocolDefect(
            completion, "temperature", "a claim's grade records the temperature each call was sent at"
        ) from missing
    return MODEL_DEFAULT_TEMPERATURE if sent is None else float(sent)


def _unreadable(refused: ValueError) -> str:
    """Why a reply could not be read, in a line a model can act on."""
    if isinstance(refused, ValidationError):
        problems = "; ".join(
            f"{'/'.join(str(part) for part in error['loc']) or 'the reply'}: {error['msg']}"
            for error in refused.errors()[:5]
        )
        return _FAULTS["shape"].format(problems=problems)
    return str(refused)


@dataclass
class _Checked:
    """One claim's answer under one step: the citations that passed, what is wrong with the rest, and its call."""

    answer: _Answer
    call: int
    citations: tuple[ClaimCitation, ...]
    fault: str | None


@dataclass(frozen=True)
class ClaimVerifier:
    """A model step that checks a cell's world claims against numbered evidence, in batches, and repairs what fails.

    :ivar client: The completion client the step calls. It must name its model (``model_name``).
    :ivar pages: How much of the documents each step reads.
    :ivar repairs: How often, and how many items, a cell may ask again.
    :ivar grounded_in: The provenances of the documents the candidate itself was given, whose support grounds a claim;
        None asks no ground step.
    :ivar guidance: The host's own notes on reading claims in its domain, appended to every prompt. Empty by default.
    """

    client: CompletionClient
    pages: PageBudget
    repairs: RepairBudget
    grounded_in: tuple[EvidenceProvenance, ...] | None = None
    guidance: str = ""

    def __post_init__(self) -> None:
        """Refuse a client that names no model, or an empty ``grounded_in``.

        :raises ValueError: One of those.
        """
        model = getattr(self.client, "model_name", None)
        if not isinstance(model, str) or not model.strip():
            raise ValueError(
                f"a claim verifier's client names its model (model_name), and {type(self.client).__name__} names "
                "none: the grade records which model checked the claims, so an unnamed one would pool with any other"
            )
        if self.grounded_in is not None and not self.grounded_in:
            raise ValueError("grounded_in names at least one provenance; None asks no ground step")

    @property
    def model_name(self) -> str:
        """The model the client calls."""
        # Checked present and a non-blank string at construction.
        return str(getattr(self.client, "model_name", ""))

    @property
    def truth_response_format(self) -> dict[str, Any]:
        """The strict directive a truth batch sends: ``{"claims": [{"claim", "citations", "suspect"}]}``."""
        item = {
            "type": "object",
            "properties": {
                "claim": {"type": "integer"},
                "citations": {"type": "array", "items": _TRUTH_CITATION},
                "suspect": {"type": "boolean"},
            },
        }
        schema = {"type": "object", "properties": {"claims": {"type": "array", "items": item}}}
        return strict_response_format("claim_truth", schema)

    @property
    def ground_response_format(self) -> dict[str, Any]:
        """The strict directive a ground batch sends: ``{"claims": [{"claim", "citations"}]}``, supports only."""
        item = {
            "type": "object",
            "properties": {
                "claim": {"type": "integer"},
                "citations": {"type": "array", "items": _GROUND_CITATION},
            },
        }
        schema = {"type": "object", "properties": {"claims": {"type": "array", "items": item}}}
        return strict_response_format("claim_ground", schema)

    @property
    def truth_repair_response_format(self) -> dict[str, Any]:
        """The strict directive a truth repair sends: one claim's ``citations`` and ``suspect``."""
        schema = {
            "type": "object",
            "properties": {
                "citations": {"type": "array", "items": _TRUTH_CITATION},
                "suspect": {"type": "boolean"},
            },
        }
        return strict_response_format("claim_truth_repair", schema)

    @property
    def ground_repair_response_format(self) -> dict[str, Any]:
        """The strict directive a ground repair sends: one claim's ``citations``."""
        schema = {"type": "object", "properties": {"citations": {"type": "array", "items": _GROUND_CITATION}}}
        return strict_response_format("claim_ground_repair", schema)

    @property
    def identity(self) -> str:
        """``<model>|claimverify:<sha12>``: the model, and a digest of the prompts (with the host's guidance), the
        reply schemas, the failures a repair is told, the sentence-break rule, the page budget and selection rule, the
        batch rule, the repair budget, ``grounded_in``, the quote cap, the attempts and the verbatim fold's
        identity."""
        digest = canonical_digest(
            {
                "truth_system": self._system("truth"),
                "ground_system": self._system("ground"),
                "truth_repair_system": self._repair_system("truth"),
                "ground_repair_system": self._repair_system("ground"),
                "templates": [
                    _USER_TEMPLATE,
                    _FACT_TEMPLATE,
                    _PASSAGE_TEMPLATE,
                    _NO_EVIDENCE,
                    _CLAIM_TEMPLATE,
                    _RESTATED_TEMPLATE,
                    _AGAIN_TEMPLATE,
                    _REPAIR_TEMPLATE,
                ],
                "schemas": [
                    self.truth_response_format,
                    self.ground_response_format,
                    self.truth_repair_response_format,
                    self.ground_repair_response_format,
                ],
                "pages": self.pages.model_dump(mode="json"),
                "faults": dict(_FAULTS),
                "sentence_break": _SENTENCE_BREAK.pattern,
                "selection": [_SELECTION_RULE, _ANCHOR_MIN_CHARS, _WORD.pattern],
                "batches": _BATCH_RULE,
                "repairs": self.repairs.model_dump(mode="json"),
                "grounded_in": None if self.grounded_in is None else sorted(set(self.grounded_in)),
                "quote_chars": CLAIM_QUOTE_CHARS,
                "attempts": _ATTEMPTS,
                "fold": VERBATIM_FOLD.identity,
            }
        )
        return f"{self.model_name}|claimverify:{digest[:12]}"

    def _with_guidance(self, prompt: str) -> str:
        return prompt + _GUIDANCE_HEADING + self.guidance.strip() if self.guidance.strip() else prompt

    def _system(self, step: _Step) -> str:
        return self._with_guidance(_TRUTH_SYSTEM if step == "truth" else _GROUND_SYSTEM)

    def _repair_system(self, step: _Step) -> str:
        return self._with_guidance(_TRUTH_REPAIR_SYSTEM if step == "truth" else _GROUND_REPAIR_SYSTEM)

    async def verify(
        self,
        claims: Sequence[ExtractedClaim],
        *,
        key: ReferenceKey | None,
        documents: Sequence[EvidenceDocument] | None,
        ledger: RoleUsageLedger,
    ) -> Verification:
        """Check every world claim in ``claims`` against the case's evidence, and against the candidate's own
        documents where ``grounded_in`` names them.

        No world claim makes no call. What the client itself raises (a provider failure, a recording that holds no
        answer) is not caught: it is the caller's to classify.

        :param claims: The cell's claims, as the extractor listed them: each ``restates`` names an earlier claim of the
            same scope that restates nothing.
        :param key: The case's reference key, whose facts are numbered ``K1..``; None when the case has none.
        :param documents: The cell's evidence documents, in order: ``()`` when the candidate was given none, None when
            undeclared.
        :param ledger: The cell's ``judge`` ledger.
        :return: The evidence shown, the documents truncated, the calls made, one finding per claim, and the ground
            step's loss when it was lost.
        :raises ValueError: ``ledger`` is not the ``judge`` role's; ``grounded_in`` is named and the documents are
            undeclared; or a ``restates`` names no earlier original of its scope.
        :raises ClaimGradingFailed: A truth reply was cut short, a truth batch's reply did not parse in its attempts, or
            a claim still failed its truth checks within the repair budget. The same in the ground step never raises:
            it is :attr:`Verification.ground_loss`.
        """
        if ledger.role != _ROLE:
            raise ValueError(
                f"a claim verifier's calls are billed to the {_ROLE!r} role, and the ledger is {ledger.role!r}'s"
            )
        if self.grounded_in is not None and documents is None:
            raise ValueError(
                "a criterion with grounded_in reads the documents the candidate was given, and this cell declares "
                "none; declare them, empty when it was given none"
            )
        cell = tuple(claims)
        for position, claim in enumerate(cell):
            original = claim.restates
            if original is not None and not (
                original < position and cell[original].scope == claim.scope and cell[original].restates is None
            ):
                raise ValueError(
                    f"claim {position} restates claim {original}, which is no earlier original of its scope"
                )
        batches = _batches(cell)
        if not batches:
            return Verification((), (), (), tuple(None for _ in cell))
        # prose-canary: allow — `scope` is the closed Literal world/session the extractor checked, not free text
        world = [claim for claim in cell if claim.scope == "world"]
        facts = [] if key is None else [(entity.name, fact) for entity in key.fact_entities for fact in entity.facts]
        names = [] if key is None else [name for entity in key.fact_entities for name in (entity.name, *entity.aliases)]
        anchors = _anchors([*names, *(claim.claim for claim in world), *(claim.excerpt for claim in world)])
        given = tuple(documents or ())
        truth_passages, truth_cut = select_passages(
            [document for document in given if document.provenance == "case"], anchors, self.pages
        )
        grounding = (
            [document for document in given if document.provenance in self.grounded_in]
            if self.grounded_in is not None
            else []
        )
        ground_passages, ground_cut = select_passages(grounding, anchors, self.pages) if grounding else ((), ())

        numbered: dict[tuple[str, int, int], str] = {}

        def number(passages: Sequence[_Passage]) -> tuple[tuple[str, _Passage], ...]:
            for passage in passages:
                numbered.setdefault((passage.document.id, passage.start, passage.end), f"S{len(numbered) + 1}")
            return tuple((numbered[(p.document.id, p.start, p.end)], p) for p in passages)

        shown: dict[_Step, _Shown] = {
            "truth": _Shown(
                tuple((f"K{at}", entity, fact) for at, (entity, fact) in enumerate(facts, start=1)),
                number(truth_passages),
            )
        }
        if ground_passages:
            shown["ground"] = _Shown((), number(ground_passages))
        calls: list[ClaimCall] = []
        checked: dict[tuple[_Step, int], _Checked] = {}
        await self._run_step("truth", batches, checked, cell, shown, calls, ledger)
        lost: ClaimGradingFailed | None = None
        if "ground" in shown:
            try:
                await self._run_step("ground", batches, checked, cell, shown, calls, ledger)
            except ClaimGradingFailed as failed:
                # Recorded, never swallowed: the loss rides on the verification for the groundedness reading alone.
                lost = failed
        return Verification(
            evidence=_index(shown),
            documents_truncated=tuple(document.id for document in given if document.id in {*truth_cut, *ground_cut}),
            calls=tuple(calls),
            findings=_findings(cell, checked, grounded=self.grounded_in is not None and lost is None),
            ground_loss=None if lost is None else lost.loss,
            ground_failure=None if lost is None else str(lost),
        )

    async def _run_step(
        self,
        step: _Step,
        batches: Sequence[tuple[int, ...]],
        checked: dict[tuple[_Step, int], _Checked],
        cell: tuple[ExtractedClaim, ...],
        shown: Mapping[_Step, _Shown],
        calls: list[ClaimCall],
        ledger: RoleUsageLedger,
    ) -> None:
        """Ask every batch under ``step``, check each answer, and repair what fails within the step's own budget.

        :raises ClaimGradingFailed: A reply was cut short, a batch could not be used, or a claim still failed.
        """
        for batch in batches:
            answers, call = await self._ask_batch(step, batch, cell, shown[step], calls, ledger)
            for position, answer in answers.items():
                checked[(step, position)] = self._check(answer, call, shown[step])
        await self._repair(step, checked, cell, shown, calls, ledger)

    async def _complete(
        self,
        *,
        role: Literal["truth", "ground", "repair"],
        system: str,
        user: str,
        response_format: dict[str, Any],
        calls: list[ClaimCall],
        ledger: RoleUsageLedger,
    ) -> tuple[str, int]:
        """One billed and recorded call's finished reply, and its place among the calls.

        :raises ClaimGradingFailed: The reply was cut short.
        """
        completion = await self.client.generate(system=system, user=user, response_format=response_format)
        ledger.add_llm_result(completion)
        calls.append(
            ClaimCall(
                role=role,
                served_model=getattr(completion, "served_model", None) or None,
                judge_temperature=_sent_temperature(completion),
                judge_sent_settings=sent_settings_of(completion),
            )
        )
        cut_short = describe_incomplete_completion(completion)
        if cut_short is not None:
            raise ClaimGradingFailed(f"the claim verifier's {role} reply was cut short: {cut_short}", loss="cut_short")
        return completion.content, len(calls) - 1

    async def _ask_batch(
        self,
        step: _Step,
        batch: tuple[int, ...],
        cell: tuple[ExtractedClaim, ...],
        shown: _Shown,
        calls: list[ClaimCall],
        ledger: RoleUsageLedger,
    ) -> tuple[dict[int, _Answer], int]:
        """One batch's answers, by claim position, and the call that gave them, asked again in full, naming why,
        while a finished reply will not parse or does not answer exactly the claims asked.

        :raises ClaimGradingFailed: A reply was cut short, or none of the attempts' replies could be used.
        """
        first = _USER_TEMPLATE.format(evidence=shown.rendered(), claims=_rendered_claims(batch, cell))
        user = first
        refused: ValueError | None = None
        for _attempt in range(_ATTEMPTS):
            content, call = await self._complete(
                role=step,
                system=self._system(step),
                user=user,
                response_format=self.truth_response_format if step == "truth" else self.ground_response_format,
                calls=calls,
                ledger=ledger,
            )
            try:
                return _read_batch(step, content, batch), call
            except ValueError as unusable:
                refused = unusable
                user = _AGAIN_TEMPLATE.format(user=first, failure=_unreadable(unusable))
        raise ClaimGradingFailed(
            f"the claim verifier's {step} reply for claims {list(batch)} could not be used in {_ATTEMPTS} attempts",
            loss="unreadable",
        ) from refused

    @staticmethod
    def _check(answer: _Answer, call: int, shown: _Shown) -> _Checked:
        """``answer``'s citations, checked: each that passes, resolved to stored offsets, and every failure named."""
        faults: list[str] = []
        citations: list[ClaimCitation] = []
        refs = [cited.ref.strip() for cited in answer.citations]
        if twice := sorted({ref for ref in refs if refs.count(ref) > 1}):
            faults.append(_FAULTS["twice"].format(refs=", ".join(twice)))
        if answer.suspect and answer.citations:
            faults.append(_FAULTS["suspect"])
        for ref, cited in zip(refs, answer.citations, strict=True):
            if ref not in shown.refs:
                faults.append(_FAULTS["unknown_ref"].format(ref=ref))
                continue
            passage = shown.passage(ref)
            if passage is None:
                # A key fact is cited whole: what a reply quotes of it is not read.
                citations.append(ClaimCitation(ref=ref, stance=cited.stance))
                continue
            citation, fault = _passage_citation(ref, cited, passage)
            if fault is not None:
                faults.append(fault)
            elif citation is not None:
                citations.append(citation)
        return _Checked(answer=answer, call=call, citations=tuple(citations), fault="; ".join(faults) or None)

    async def _repair(
        self,
        step: _Step,
        checked: dict[tuple[_Step, int], _Checked],
        cell: tuple[ExtractedClaim, ...],
        shown: Mapping[_Step, _Shown],
        calls: list[ClaimCall],
        ledger: RoleUsageLedger,
    ) -> None:
        """Ask every claim failing under ``step`` again alone, with its failure named, within that step's own repair
        budget; keep each answer that passes.

        :raises ClaimGradingFailed: The next round would ask more items than the step's budget has left, a claim still
            fails after the rounds, or a repair reply was cut short.
        """
        failing = sorted(item for item, found in checked.items() if item[0] == step and found.fault is not None)
        asked = 0
        for _round in range(self.repairs.rounds):
            if not failing:
                return
            if asked + len(failing) > self.repairs.max_items_per_cell:
                raise ClaimGradingFailed(
                    f"the claim verifier would ask {len(failing)} {step} claim(s) again with {asked} of the step's "
                    f"{self.repairs.max_items_per_cell} repairs spent: {_named(failing, checked)}",
                    loss="repair_budget_spent",
                )
            for item in failing:
                position = item[1]
                asked += 1
                before = checked[item]
                content, call = await self._complete(
                    role="repair",
                    system=self._repair_system(step),
                    user=_REPAIR_TEMPLATE.format(
                        evidence=shown[step].rendered(),
                        claim=_rendered_claim(position, cell),
                        answer=json.dumps(_answer_json(step, before.answer), ensure_ascii=False),
                        failure=before.fault,
                    ),
                    response_format=(
                        self.truth_repair_response_format if step == "truth" else self.ground_repair_response_format
                    ),
                    calls=calls,
                    ledger=ledger,
                )
                try:
                    answer = _read_repair(step, content)
                except ValueError as refused:
                    checked[item] = _Checked(
                        answer=before.answer,
                        call=before.call,
                        citations=before.citations,
                        fault=_FAULTS["correction"].format(failure=before.fault, why=_unreadable(refused)),
                    )
                    continue
                checked[item] = self._check(answer, call, shown[step])
            failing = [item for item in failing if checked[item].fault is not None]
        if failing:
            raise ClaimGradingFailed(
                f"the claim verifier could not repair {len(failing)} claim(s) in {self.repairs.rounds} round(s): "
                f"{_named(failing, checked)}",
                loss="repair_budget_spent",
            )


def _named(failing: Sequence[tuple[_Step, int]], checked: Mapping[tuple[_Step, int], _Checked]) -> str:
    return "; ".join(f"{step} claim {position}: {checked[(step, position)].fault}" for step, position in failing)


def _place(claim: ExtractedClaim) -> str:
    return "outside any unit" if claim.unit is None else f"in unit {claim.unit}"


def _rendered_claim(position: int, cell: tuple[ExtractedClaim, ...]) -> str:
    """One asked claim, with every restatement of it shown beside it."""
    claim = cell[position]
    lines = [_CLAIM_TEMPLATE.format(position=position, place=_place(claim), claim=claim.claim, excerpt=claim.excerpt)]
    lines += [
        _RESTATED_TEMPLATE.format(position=other, excerpt=again.excerpt)
        for other, again in enumerate(cell)
        if again.restates == position
    ]
    return "\n".join(lines)


def _rendered_claims(batch: tuple[int, ...], cell: tuple[ExtractedClaim, ...]) -> str:
    return "\n\n".join(_rendered_claim(position, cell) for position in batch)


def _answer_json(step: _Step, answer: _Answer) -> dict[str, Any]:
    """An answer as its step's reply shape gives it, to show a repair what it said."""
    if step == "truth":
        return answer.model_dump(mode="json")
    return {"citations": [cited.model_dump(mode="json", exclude={"stance"}) for cited in answer.citations]}


def _read_batch(step: _Step, content: str, batch: tuple[int, ...]) -> dict[int, _Answer]:
    """One batch reply's answers, by claim position.

    :raises ValueError: The reply is not its step's shape, or does not answer each claim asked exactly once.
    """
    payload = extract_json(content)
    if step == "truth":
        answers = [
            (item.claim, _Answer(citations=item.citations, suspect=item.suspect))
            for item in _TruthBatch.model_validate(payload).claims
        ]
    else:
        answers = [
            (item.claim, _Answer(citations=[cited.as_cited() for cited in item.citations], suspect=False))
            for item in _GroundBatch.model_validate(payload).claims
        ]
    named = [position for position, _ in answers]
    if sorted(named) != sorted(batch):
        missing = sorted(set(batch) - set(named))
        extra = sorted({position for position in named if position not in batch or named.count(position) > 1})
        problems = [
            *([_FAULTS["missing"].format(missing=missing)] if missing else []),
            *([_FAULTS["extra"].format(extra=extra)] if extra else []),
        ]
        raise ValueError(_FAULTS["answers"].format(asked=list(batch), problems="; ".join(problems)))
    return dict(answers)


def _read_repair(step: _Step, content: str) -> _Answer:
    """One repair reply's answer.

    :raises ValueError: It is not its step's shape.
    """
    payload = extract_json(content)
    if step == "truth":
        repaired = _TruthRepaired.model_validate(payload)
        return _Answer(citations=repaired.citations, suspect=repaired.suspect)
    ground = _GroundRepaired.model_validate(payload)
    return _Answer(citations=[cited.as_cited() for cited in ground.citations], suspect=False)


def _passage_citation(ref: str, cited: _Cited, passage: _Passage) -> tuple[ClaimCitation | None, str | None]:
    """A passage citation checked: its quote a stretch of the passage shown, its subject and value inside that quote
    and in one sentence or row of it.

    :return: The stored citation, or the failure named.
    """
    if cited.quote is None or not cited.quote.strip():
        return None, _FAULTS["no_quote"].format(ref=ref)
    shown = passage.document.text[passage.start : passage.end]
    span = verbatim_span(cited.quote, shown)
    if span is None:
        return None, _FAULTS["not_shown"].format(ref=ref)
    quote = shown[span[0] : span[1]]
    if len(quote) > CLAIM_QUOTE_CHARS:
        return None, _FAULTS["too_long"].format(ref=ref, length=len(quote), cap=CLAIM_QUOTE_CHARS)
    spans: dict[str, TextSpan] = {}
    for what, words in (("subject", cited.subject), ("value", cited.value)):
        if words is None or not words.strip():
            return None, _FAULTS["no_span"].format(ref=ref, what=what)
        found = verbatim_span(words, quote)
        if found is None:
            return None, _FAULTS["span_outside"].format(ref=ref, what=what)
        spans[what] = TextSpan(start=found[0], end=found[1])
    first, second = sorted((spans["subject"], spans["value"]), key=lambda found: found.start)
    # The quote is the passage's own text, not a model's: where its sentences break is structure, read by position.
    if second.start > first.end and _SENTENCE_BREAK.search(quote[first.end : second.start]):
        return None, _FAULTS["apart"].format(ref=ref)
    return (
        ClaimCitation(
            ref=ref, stance=cited.stance, quote=quote, subject_span=spans["subject"], value_span=spans["value"]
        ),
        None,
    )


def _index(shown: Mapping[_Step, _Shown]) -> tuple[EvidenceRef, ...]:
    """Every piece of evidence shown, once each: the key facts, then the passages in the order they were numbered."""
    refs: dict[str, EvidenceRef] = {}
    for step in _STEPS:
        if step not in shown:
            continue
        for ref, entity, fact in shown[step].facts:
            refs.setdefault(ref, EvidenceRef(ref=ref, provenance="case", entity=entity, fact=fact))
        for ref, passage in shown[step].passages:
            refs.setdefault(
                ref,
                EvidenceRef(
                    ref=ref,
                    provenance=passage.document.provenance,
                    document=passage.document.id,
                    span=TextSpan(start=passage.start, end=passage.end),
                ),
            )
    return tuple(sorted(refs.values(), key=lambda entry: (entry.ref[0] != "K", int(entry.ref[1:]))))


def _findings(
    cell: tuple[ExtractedClaim, ...], checked: Mapping[tuple[_Step, int], _Checked], *, grounded: bool
) -> tuple[ClaimFinding | None, ...]:
    """One finding per claim: a world original's own, a restatement its original's, a session claim None."""
    findings: list[ClaimFinding | None] = []
    for position, claim in enumerate(cell):
        # prose-canary: allow — `scope` is the closed Literal world/session the extractor checked, not free text
        if claim.scope != "world":
            findings.append(None)
            continue
        original = position if claim.restates is None else claim.restates
        truth = checked[("truth", original)]
        ground = checked.get(("ground", original)) if grounded else None
        ground_citations = () if ground is None else ground.citations
        findings.append(
            ClaimFinding(
                truth_citations=truth.citations,
                suspect=truth.answer.suspect,
                truth_call=truth.call,
                ground_citations=ground_citations,
                grounded=bool(ground_citations) if grounded else None,
            )
        )
    return tuple(findings)


__all__: list[str] = []
