"""The claim verifier: the step that checks each listed claim against numbered evidence, on a scripted client.

Every behaviour is driven through a client that answers from a script and records what it was sent, so nothing is
spent and each reply is chosen. The truth step sees the case key's facts (``K1..``) and passages of the case's own
documents (``S1..``) and never a document the candidate fetched; the ground step, asked only under ``grounded_in``,
sees only those documents. Code derives every verdict: case support always reads supported, case contradiction
contradicted, and only a claim no case evidence bears on that the model doubts is nominated for the web. Each
citation's quote is a stretch of the evidence shown, a passage or a key fact's line, with its subject and value found
inside it and no sentence end or line break between them: that narrows a neighbouring value to one sentence or row (a
semicolon joins clauses of one sentence), keeps a key fact from supporting a value it does not state, and what one
sentence leaves to the model calibration measures. A case document may override others: a
citation an overriding document's opposite stance meets is overruled, the call is shown which document overrides
which, and a key fact sits outside precedence. Restatements ride with their original and take its answer; session claims
are never verified. A failing answer is asked again alone within its step's repair budget; an unreadable batch is
asked again in full; a cut-short reply fails at once. Truth and groundedness fail apart: a lost ground step leaves
every truth finding as it was. Passages are chosen by code under the page budget, the same way every time, with
truncation recorded. The identity moves with the budgets, ``grounded_in``, the guidance and the model, and is pinned
for the shipped prompts, schemas and selection rule. The text is invented: a small press's history of one book, a
table's house rule, a source a summary must stay faithful to, and a library's lending policy with the update that
amends it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from threetears.evals.kernel import RoleUsageLedger
from threetears.evals.kernel.claims.extraction import ClaimExtractor, ExtractedClaim
from threetears.evals.kernel.claims.failures import ClaimGradingFailed
from threetears.evals.kernel.claims.verification import (
    ClaimVerifier,
    Verification,
    WebOutcome,
    graded_claims,
    truth_verdict,
)
from threetears.evals.schema import Segment
from threetears.evals.schema.models import (
    ClaimCitation,
    ClaimGrade,
    EvidenceDocument,
    PageBudget,
    RepairBudget,
)
from threetears.evals.schema.reference import ReferenceEntity, ReferenceFact, ReferenceKey

_MODEL = "scripted/verify"

_SEGMENTS = (
    Segment(text="Good evening, readers."),
    Segment(text="The bound volume of Harrow Tales appeared in 1931, from Lantern Works.", unit=0),
    Segment(text="Ida Venn wrote it. Harrow Tales has twelve chapters.", unit=1),
)

_KEY = ReferenceKey(
    core=(
        ReferenceEntity(
            name="Harrow Tales",
            aliases=("The Harrow Book",),
            facts=(ReferenceFact(claim="Harrow Tales was written by Ida Venn.", source_url="https://example.org/ht"),),
        ),
    )
)

#: Two values on one page, each bound to its own subject: the pamphlet's year and the bound volume's.
_PRESS = EvidenceDocument(
    id="press-history",
    provenance="case",
    title="A history of Lantern Works",
    text=(
        "Lantern Works printed small runs for local writers. The pamphlet edition of Harrow Tales appeared in 1931. "
        "The bound volume of Harrow Tales followed in 1932."
    ),
)

#: A budget that reads every document here whole.
_WHOLE = PageBudget(per_document_chars=4000, total_chars=8000, window_chars=2000)

_REPAIRS = RepairBudget(rounds=2, max_items_per_cell=10)


@dataclass(frozen=True)
class _Reply:
    """One completion, shaped as the engine's completion port describes it."""

    content: str
    stop_reason: str = "end_turn"
    model: str = _MODEL
    served_model: str | None = _MODEL
    input_tokens: int | None = 500
    output_tokens: int | None = 120
    reasoning_tokens: int | None = None
    cost_usd: float | None = 0.003
    price_source: str | None = "scripted"
    temperature: float | None = None


# parity-with: threetears.evals.schema.CompletionClient
@dataclass
class _FakeScriptedClient:
    """Answers each call with the next scripted reply, and records every prompt pair and response format sent."""

    replies: list[_Reply]
    model_name: str = _MODEL
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def generate(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> _Reply:
        self.calls.append({"system": system, "user": user, "response_format": response_format})
        return self.replies[len(self.calls) - 1]

    async def aclose(self) -> None:
        """Nothing to release."""

    async def __aenter__(self) -> _FakeScriptedClient:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self.aclose()


# parity-with: threetears.evals.schema.CompletionClient
class _FakeUnnamedClient:
    """A client that names no model."""

    async def generate(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> _Reply:
        raise AssertionError("an unnamed verifier is refused before any call")

    async def aclose(self) -> None:
        """Nothing to release."""

    async def __aenter__(self) -> _FakeUnnamedClient:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self.aclose()


def _claim(
    segments: tuple[Segment, ...],
    segment: int,
    claim: str,
    excerpt: str,
    *,
    scope: str = "world",
    restates: int | None = None,
    hedged: bool = False,
) -> ExtractedClaim:
    start = segments[segment].text.index(excerpt)
    return ExtractedClaim(
        segment=segment,
        unit=segments[segment].unit,
        claim=claim,
        excerpt=excerpt,
        start=start,
        end=start + len(excerpt),
        specific=True,
        hedged=hedged,
        attributed=False,
        scope=scope,  # type: ignore[arg-type]
        restates=restates,
    )


_BOUND_YEAR = _claim(
    _SEGMENTS,
    1,
    "The bound volume of Harrow Tales appeared in 1931.",
    "The bound volume of Harrow Tales appeared in 1931",
)
_AUTHOR = _claim(_SEGMENTS, 2, "Ida Venn wrote Harrow Tales.", "Ida Venn wrote it")
_CHAPTERS = _claim(_SEGMENTS, 2, "Harrow Tales has twelve chapters.", "Harrow Tales has twelve chapters")
_CLAIMS = (_BOUND_YEAR, _AUTHOR, _CHAPTERS)


def _cite(
    ref: str,
    stance: str = "supports",
    *,
    quote: str | None = None,
    subject: str | None = None,
    value: str | None = None,
) -> dict[str, Any]:
    return {"ref": ref, "stance": stance, "quote": quote, "subject": subject, "value": value}


def _ground_cite(ref: str, *, quote: str, subject: str, value: str) -> dict[str, Any]:
    return {"ref": ref, "quote": quote, "subject": subject, "value": value}


def _answer(claim: int, *citations: dict[str, Any], suspect: bool = False) -> dict[str, Any]:
    return {"claim": claim, "citations": list(citations), "suspect": suspect}


def _batch(*answers: dict[str, Any]) -> _Reply:
    return _Reply(content=json.dumps({"claims": list(answers)}))


def _ground(*answers: tuple[int, list[dict[str, Any]]]) -> _Reply:
    return _Reply(content=json.dumps({"claims": [{"claim": c, "citations": cites} for c, cites in answers]}))


def _repair(*citations: dict[str, Any], suspect: bool = False) -> _Reply:
    return _Reply(content=json.dumps({"citations": list(citations), "suspect": suspect}))


#: The key's one fact, cited as support for the authorship claim, quoting its line as a call is shown it.
_WRITTEN_BY_IDA_VENN = _cite(
    "K1", quote="Harrow Tales was written by Ida Venn.", subject="Harrow Tales", value="Ida Venn"
)

_CONTRADICTED_BY_THE_BOUND_VOLUME = _cite(
    "S1",
    "contradicts",
    quote="The bound volume of Harrow Tales followed in 1932.",
    subject="The bound volume of Harrow Tales",
    value="1932",
)


def _verifier(client: Any, **options: Any) -> ClaimVerifier:
    options.setdefault("pages", _WHOLE)
    options.setdefault("repairs", _REPAIRS)
    return ClaimVerifier(client=client, **options)


async def _verify(
    client: _FakeScriptedClient,
    claims: tuple[ExtractedClaim, ...] = _CLAIMS,
    *,
    key: ReferenceKey | None = _KEY,
    documents: tuple[EvidenceDocument, ...] | None = (_PRESS,),
    ledger: RoleUsageLedger | None = None,
    **options: Any,
) -> Verification:
    return await _verifier(client, **options).verify(
        claims, key=key, documents=documents, ledger=ledger or RoleUsageLedger(role="judge")
    )


def _grade(claims: tuple[ExtractedClaim, ...], verification: Verification) -> ClaimGrade:
    """The stored grade the findings make, which validates every reference they hold."""
    return ClaimGrade(
        criterion_fingerprint="0" * 64,
        claim_evidence_digest="1" * 64,
        calls=verification.calls,
        evidence=verification.evidence,
        documents_truncated=verification.documents_truncated,
        claims=graded_claims(claims, verification, calls_before=0),
    )


# --- the verdict table ----------------------------------------------------------------------------------------------


async def test_each_claim_reads_the_verdict_its_case_citations_give_and_the_grade_stores() -> None:
    client = _FakeScriptedClient(
        [
            _batch(_answer(0, _CONTRADICTED_BY_THE_BOUND_VOLUME)),
            _batch(_answer(1, _WRITTEN_BY_IDA_VENN), _answer(2)),
        ]
    )

    verification = await _verify(client)
    grade = _grade(_CLAIMS, verification)

    assert [(claim.verdict, claim.graded_against) for claim in grade.claims] == [
        ("contradicted", "case"),
        ("supported", "case"),
        ("unverifiable", None),
    ]
    assert [claim.decided_by for claim in grade.claims] == [0, 1, 1]
    assert [call.role for call in grade.calls] == ["truth", "truth"]
    assert grade.calls[0].served_model == _MODEL
    assert grade.calls[0].judge_temperature == "model_default"
    assert all(claim.nominated_by is None and claim.grounded is None for claim in grade.claims)


async def test_the_key_facts_are_numbered_first_then_the_case_passages() -> None:
    client = _FakeScriptedClient([_batch(_answer(0)), _batch(_answer(1), _answer(2))])

    verification = await _verify(client)

    k1, s1 = verification.evidence
    assert (k1.ref, k1.provenance, k1.entity, k1.fact) == ("K1", "case", "Harrow Tales", _KEY.core[0].facts[0])
    assert (s1.ref, s1.provenance, s1.document) == ("S1", "case", "press-history")
    assert s1.span is not None and (s1.span.start, s1.span.end) == (0, len(_PRESS.text))
    assert verification.documents_truncated == ()
    user = client.calls[0]["user"]
    assert "K1. A key fact about Harrow Tales: Harrow Tales was written by Ida Venn." in user
    assert 'S1. From "A history of Lantern Works", characters 0 to' in user
    assert _PRESS.text in user


async def test_a_neighbouring_value_on_the_same_page_is_bound_to_its_own_subject() -> None:
    client = _FakeScriptedClient(
        [_batch(_answer(0, _CONTRADICTED_BY_THE_BOUND_VOLUME)), _batch(_answer(1), _answer(2))]
    )

    verification = await _verify(client)

    (citation,) = verification.findings[0].truth_citations  # type: ignore[union-attr]
    assert citation.quote == "The bound volume of Harrow Tales followed in 1932."
    assert citation.subject_span is not None and citation.value_span is not None
    assert citation.quote[citation.subject_span.start : citation.subject_span.end] == "The bound volume of Harrow Tales"
    assert citation.quote[citation.value_span.start : citation.value_span.end] == "1932"
    assert "neighbouring subject" in client.calls[0]["system"]


async def test_the_pamphlets_year_cited_for_the_bound_volume_is_asked_again_and_the_repair_stands() -> None:
    pamphlet = _cite(
        "S1",
        quote="The pamphlet edition of Harrow Tales appeared in 1931.",
        subject="The bound volume of Harrow Tales",
        value="1931",
    )
    client = _FakeScriptedClient(
        [
            _batch(_answer(0, pamphlet)),
            _batch(_answer(1), _answer(2)),
            _repair(_CONTRADICTED_BY_THE_BOUND_VOLUME),
        ]
    )

    verification = await _verify(client)
    grade = _grade(_CLAIMS, verification)

    assert grade.claims[0].verdict == "contradicted"
    assert grade.claims[0].decided_by == 2, "the repair's answer is the one that stands"
    repair = client.calls[2]
    assert "its subject for S1 is not in its quote" in repair["user"]
    assert "Claim 0, in unit 0" in repair["user"]
    assert "Claim 1" not in repair["user"], "only the failing claim is asked again"
    assert [call.role for call in grade.calls] == ["truth", "truth", "repair"]


async def test_a_quote_spanning_two_sentences_cannot_bind_the_neighbours_value_and_is_asked_again() -> None:
    both = _cite(
        "S1",
        quote="The pamphlet edition of Harrow Tales appeared in 1931. The bound volume of Harrow Tales followed in 1932.",
        subject="The bound volume of Harrow Tales",
        value="1931",
    )
    client = _FakeScriptedClient(
        [
            _batch(_answer(0, both)),
            _batch(_answer(1), _answer(2)),
            _repair(_CONTRADICTED_BY_THE_BOUND_VOLUME),
        ]
    )

    grade = _grade(_CLAIMS, await _verify(client))

    assert "its subject and value for S1 stand in different sentences or rows of its quote" in client.calls[2]["user"]
    assert (grade.claims[0].verdict, grade.claims[0].decided_by) == ("contradicted", 2)


async def test_a_line_break_between_subject_and_value_is_asked_again() -> None:
    text = "The pamphlet edition of Harrow Tales appeared in 1931\nthe bound volume of Harrow Tales followed in 1932."
    rows = EvidenceDocument(id="rows", provenance="case", title="Rows", text=text)
    client = _FakeScriptedClient(
        [
            _batch(_answer(0, _cite("S1", quote=text, subject="the bound volume of Harrow Tales", value="1931"))),
            _batch(_answer(1), _answer(2)),
            _repair(
                _cite(
                    "S1",
                    "contradicts",
                    quote="the bound volume of Harrow Tales followed in 1932.",
                    subject="the bound volume of Harrow Tales",
                    value="1932",
                )
            ),
        ]
    )

    grade = _grade(_CLAIMS, await _verify(client, documents=(rows,)))

    assert "stand in different sentences or rows" in client.calls[2]["user"]
    assert grade.claims[0].verdict == "contradicted"


_RENEWAL = (Segment(text="Renewing a loan costs two dollars.", unit=0),)
_RENEWAL_COST = _claim(_RENEWAL, 0, "Renewing a loan costs two dollars.", "Renewing a loan costs two dollars")


@pytest.mark.parametrize(
    ("text", "repaired"),
    [
        ("Members may renew a loan once; doing so costs two dollars.", False),
        ("Members may renew a loan once. Doing so costs two dollars.", True),
    ],
    ids=["semicolon-one-sentence", "full-stop-two-sentences"],
)
async def test_a_semicolon_joins_clauses_of_one_sentence_and_a_full_stop_still_parts_two(
    text: str, repaired: bool
) -> None:
    """The subject in one clause and its value in the next, joined by a semicolon, bind in one sentence and need no
    repair; the same two clauses as two sentences are still asked again."""
    rules = EvidenceDocument(id="lending", provenance="case", title="Lending rules", text=text)
    cited = _cite("S1", quote=text, subject="renew a loan", value="two dollars")
    client = _FakeScriptedClient([_batch(_answer(0, cited)), _repair()])

    grade = _grade((_RENEWAL_COST,), await _verify(client, (_RENEWAL_COST,), key=None, documents=(rules,)))

    if repaired:
        assert len(client.calls) == 2
        assert "its subject and value for S1 stand in different sentences or rows" in client.calls[1]["user"]
        assert grade.claims[0].verdict == "unverifiable"
    else:
        assert len(client.calls) == 1, "no repair call"
        (citation,) = grade.claims[0].truth_citations
        assert (citation.quote, grade.claims[0].verdict) == (text, "supported")


async def test_a_misbinding_across_a_semicolon_passes_every_code_check_and_is_left_to_the_model() -> None:
    """A semicolon joins clauses of one sentence, so the pamphlet's year cited for the bound volume across one is no
    code failure: the binding is the model's reading, as within any one sentence, and calibration measures it. The
    accepted trade-off: fewer forced repairs on rule-style sentences, for a weaker code check."""
    text = "The pamphlet edition of Harrow Tales appeared in 1931; the bound volume of Harrow Tales followed in 1932."
    joined = EvidenceDocument(id="joined", provenance="case", title="Joined", text=text)
    client = _FakeScriptedClient(
        [
            _batch(_answer(0, _cite("S1", quote=text, subject="the bound volume of Harrow Tales", value="1931"))),
            _batch(_answer(1), _answer(2)),
        ]
    )

    grade = _grade(_CLAIMS, await _verify(client, documents=(joined,)))

    assert len(client.calls) == 2, "no repair call"
    assert grade.claims[0].verdict == "supported", "a misbinding across a semicolon passes every code check"


async def test_one_sentence_naming_two_subjects_is_left_to_the_model_to_bind() -> None:
    """Code narrows a neighbouring value to one sentence; inside one, which value belongs to which subject is the
    model's reading, and calibration measures it."""
    shared = EvidenceDocument(
        id="shared",
        provenance="case",
        title="Shared",
        text="Lantern Works issued the pamphlet in 1931 and the bound volume of Harrow Tales in 1932.",
    )
    client = _FakeScriptedClient(
        [
            _batch(
                _answer(0, _cite("S1", quote=shared.text, subject="the bound volume of Harrow Tales", value="1931"))
            ),
            _batch(_answer(1), _answer(2)),
        ]
    )

    grade = _grade(_CLAIMS, await _verify(client, documents=(shared,)))

    assert grade.claims[0].verdict == "supported", "a misbinding within one sentence passes every code check"


async def test_a_fact_the_key_omits_contradicts_nothing() -> None:
    client = _FakeScriptedClient([_batch(_answer(0)), _batch(_answer(1, _WRITTEN_BY_IDA_VENN), _answer(2))])

    verification = await _verify(client)
    chapters = _grade(_CLAIMS, verification).claims[2]

    assert (chapters.verdict, chapters.graded_against, chapters.nominated_by) == ("unverifiable", None, None)
    assert "An omission contradicts nothing" in client.calls[1]["system"]


async def test_case_support_with_a_case_contradiction_reads_supported_with_a_conflict() -> None:
    other = EvidenceDocument(
        id="catalogue", provenance="case", title="A catalogue", text="Harrow Tales: written by Ida Venn, 1932."
    )
    client = _FakeScriptedClient(
        [
            _batch(_answer(0)),
            _batch(
                _answer(
                    1,
                    _WRITTEN_BY_IDA_VENN,
                    _cite(
                        "S2",
                        "contradicts",
                        quote="Harrow Tales: written by Ida Venn",
                        subject="Harrow Tales",
                        value="Ida Venn",
                    ),
                ),
                _answer(2),
            ),
        ]
    )

    grade = _grade(_CLAIMS, await _verify(client, documents=(_PRESS, other)))

    assert (grade.claims[1].verdict, grade.claims[1].evidence_conflict) == ("supported", True)


async def test_a_claim_no_evidence_covers_that_the_model_doubts_is_nominated_and_reads_unverifiable_with_web_off() -> (
    None
):
    client = _FakeScriptedClient([_batch(_answer(0)), _batch(_answer(1), _answer(2, suspect=True))])

    grade = _grade(_CLAIMS, await _verify(client))

    chapters = grade.claims[2]
    assert (chapters.suspect, chapters.nominated_by, chapters.verdict, chapters.web) == (
        True,
        "judge",
        "unverifiable",
        "none",
    )


_SUPPORT = ClaimCitation(ref="K1", stance="supports")
_CONTRADICTION = ClaimCitation(ref="K2", stance="contradicts")


@pytest.mark.parametrize(
    ("citations", "suspect", "web", "expected"),
    [
        ((_SUPPORT,), False, None, ("supported", "case", False, None)),
        ((_SUPPORT, _CONTRADICTION), False, None, ("supported", "case", True, None)),
        ((_CONTRADICTION,), False, None, ("contradicted", "case", False, None)),
        ((), True, "true", ("true_ungrounded", "web", False, "judge")),
        ((), True, "false", ("contradicted", "web", False, "judge")),
        ((), True, "inconclusive", ("unverifiable", None, False, "judge")),
        ((), True, "skipped_cap", ("unverifiable", None, False, "judge")),
        ((), True, None, ("unverifiable", None, False, "judge")),
        ((), False, None, ("unverifiable", None, False, None)),
    ],
)
def test_the_verdict_table(
    citations: tuple[ClaimCitation, ...], suspect: bool, web: WebOutcome | None, expected: tuple[Any, ...]
) -> None:
    verdict = truth_verdict(citations, suspect=suspect, web=web)

    assert (verdict.verdict, verdict.graded_against, verdict.evidence_conflict, verdict.nominated_by) == expected


@pytest.mark.parametrize("citations", [(_SUPPORT,), (_CONTRADICTION,), ()])
def test_the_web_never_decides_a_claim_nobody_nominated(citations: tuple[ClaimCitation, ...]) -> None:
    with pytest.raises(ValueError, match="never overrules case evidence"):
        truth_verdict(citations, suspect=False, web="false")


# --- precedence among case documents ------------------------------------------------------------------------------

_LOANS = (
    Segment(text="A loan runs two weeks, and members may renew it once.", unit=0),
    Segment(text="A loan now runs three weeks.", unit=1),
)
_TWO_WEEKS = _claim(_LOANS, 0, "A loan runs two weeks.", "A loan runs two weeks")
_RENEW_ONCE = _claim(_LOANS, 0, "Members may renew a loan once.", "members may renew it once")
_THREE_WEEKS = _claim(_LOANS, 1, "A loan runs three weeks.", "A loan now runs three weeks")
_LOAN_CLAIMS = (_TWO_WEEKS, _RENEW_ONCE, _THREE_WEEKS)

_POLICY = EvidenceDocument(
    id="policy",
    provenance="case",
    title="Lending policy",
    text="A loan runs two weeks. Members may renew a loan once.",
)
_UPDATE = EvidenceDocument(
    id="update",
    provenance="case",
    title="Policy update",
    text="From March, a loan runs three weeks.",
    overrides=("policy",),
)
_ERRATUM = EvidenceDocument(
    id="erratum",
    provenance="case",
    title="Erratum",
    text="The update is corrected: a loan runs ten days.",
    overrides=("update",),
)

_TWO_WEEKS_IN_THE_POLICY = _cite("S1", quote="A loan runs two weeks.", subject="A loan", value="two weeks")
_RENEW_ONCE_IN_THE_POLICY = _cite(
    "S1", quote="Members may renew a loan once.", subject="Members may renew a loan", value="once"
)
_THREE_WEEKS_IN_THE_UPDATE = _cite(
    "S2", quote="From March, a loan runs three weeks.", subject="a loan", value="three weeks"
)


def _stance(cited: dict[str, Any], stance: str) -> dict[str, Any]:
    return {**cited, "stance": stance}


def _loan_replies(*first_unit: dict[str, Any], second_unit: tuple[dict[str, Any], ...] = ()) -> list[_Reply]:
    return [_batch(*first_unit), _batch(_answer(2, *second_unit))]


@pytest.mark.parametrize("overridden", [True, False], ids=["update-overrides", "no-override"])
async def test_support_from_an_overridden_document_its_override_contradicts_reads_contradicted(
    overridden: bool,
) -> None:
    update = _UPDATE if overridden else _UPDATE.model_copy(update={"overrides": ()})
    client = _FakeScriptedClient(
        _loan_replies(
            _answer(0, _TWO_WEEKS_IN_THE_POLICY, _stance(_THREE_WEEKS_IN_THE_UPDATE, "contradicts")),
            _answer(1, _RENEW_ONCE_IN_THE_POLICY),
        )
    )

    grade = _grade(_LOAN_CLAIMS, await _verify(client, _LOAN_CLAIMS, key=None, documents=(_POLICY, update)))

    two_weeks, renew_once, _ = grade.claims
    expected = ("contradicted", "case", False) if overridden else ("supported", "case", True)
    assert (two_weeks.verdict, two_weeks.graded_against, two_weeks.evidence_conflict) == expected
    assert [c.stance for c in two_weeks.truth_citations] == ["supports", "contradicts"], "both stay on the record"
    assert (renew_once.verdict, renew_once.evidence_conflict) == ("supported", False), (
        "where the overriding document is silent, the overridden one's support stands"
    )
    user = client.calls[0]["user"]
    if overridden:
        assert '- "Policy update" (S2) overrides "Lending policy" (S1).' in user
        assert "never leave a passage out because another overrides it" in user
    else:
        assert "overrides" not in user, "a cell whose documents override none is shown what it was shown before"


async def test_support_from_the_overriding_document_stands_over_the_overridden_ones_contradiction() -> None:
    client = _FakeScriptedClient(
        _loan_replies(
            _answer(0),
            _answer(1),
            second_unit=(_THREE_WEEKS_IN_THE_UPDATE, _stance(_TWO_WEEKS_IN_THE_POLICY, "contradicts")),
        )
    )

    grade = _grade(_LOAN_CLAIMS, await _verify(client, _LOAN_CLAIMS, key=None, documents=(_POLICY, _UPDATE)))

    three_weeks = grade.claims[2]
    assert (three_weeks.verdict, three_weeks.graded_against, three_weeks.evidence_conflict) == (
        "supported",
        "case",
        False,
    )


async def test_precedence_follows_the_chain_past_a_silent_document_between() -> None:
    erratum = _cite(
        "S3", "contradicts", quote="The update is corrected: a loan runs ten days.", subject="a loan", value="ten days"
    )
    client = _FakeScriptedClient(_loan_replies(_answer(0, _TWO_WEEKS_IN_THE_POLICY, erratum), _answer(1)))

    verification = await _verify(client, _LOAN_CLAIMS, key=None, documents=(_POLICY, _UPDATE, _ERRATUM))
    grade = _grade(_LOAN_CLAIMS, verification)

    assert (grade.claims[0].verdict, grade.claims[0].evidence_conflict) == ("contradicted", False)
    assert dict(verification.outranks) == {"S2": frozenset({"S1"}), "S3": frozenset({"S1", "S2"})}
    user = client.calls[0]["user"]
    assert '- "Policy update" (S2) overrides "Lending policy" (S1).' in user
    assert '- "Erratum" (S3) overrides "Lending policy" (S1), "Policy update" (S2).' in user


async def test_a_key_fact_sits_outside_precedence() -> None:
    """A key fact is no document: an overriding document's contradiction does not overrule it, and it reads as the
    conflict it is."""
    key = ReferenceKey(
        core=(
            ReferenceEntity(
                name="A loan",
                facts=(ReferenceFact(claim="A loan runs two weeks.", source_url="https://example.org/loans"),),
            ),
        )
    )
    client = _FakeScriptedClient(
        _loan_replies(
            _answer(
                0,
                _cite("K1", quote="A loan runs two weeks.", subject="A loan", value="two weeks"),
                _stance(_THREE_WEEKS_IN_THE_UPDATE, "contradicts"),
            ),
            _answer(1),
        )
    )

    grade = _grade(_LOAN_CLAIMS, await _verify(client, _LOAN_CLAIMS, key=key, documents=(_POLICY, _UPDATE)))

    assert (grade.claims[0].verdict, grade.claims[0].evidence_conflict) == ("supported", True)


async def test_an_override_of_a_document_the_call_was_not_shown_is_not_stated() -> None:
    """The precedence lines name only documents the call reads: an overridden document the page budget left unread
    leaves its override unstated, and nothing it could cite to overrule."""
    unread = PageBudget(per_document_chars=60, total_chars=60, window_chars=60)
    client = _FakeScriptedClient(_loan_replies(_answer(0), _answer(1)))

    verification = await _verify(client, _LOAN_CLAIMS, key=None, documents=(_UPDATE, _POLICY), pages=unread)

    assert [entry.document for entry in verification.evidence] == ["update"]
    assert "overrides" not in client.calls[0]["user"] and dict(verification.outranks) == {}


# parity-with: threetears.evals.schema.completion.BoundCompletionClient
@dataclass
class _FakeBytePricedClient(_FakeScriptedClient):
    """Prices a call by the bytes it sends, so evidence a priced prompt leaves out shows as a ceiling too low."""

    def price_ceiling(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> float:
        del response_format
        return (len(system.encode()) + len(user.encode())) * 1e-6


async def test_the_price_ceiling_bounds_the_precedence_block_every_truth_call_may_carry() -> None:
    client = _FakeBytePricedClient([])
    verifier = _verifier(client)
    plain = (_POLICY, _UPDATE.model_copy(update={"overrides": ()}))

    added = verifier.price_ceiling(_LOANS, key=None, documents=(_POLICY, _UPDATE))
    before = verifier.price_ceiling(_LOANS, key=None, documents=plain)
    scripted = _FakeScriptedClient(_loan_replies(_answer(0), _answer(1)))
    await _verify(scripted, _LOAN_CLAIMS, key=None, documents=(_POLICY, _UPDATE))
    shown = scripted.calls[0]["user"]
    block = shown[shown.index("Where passages") : shown.index("# Claims")]

    assert added is not None and before is not None
    # Each of the two batches may be asked three times, each time carrying the block.
    assert added - before >= 2 * 3 * len(block.encode()) * 1e-6


_S_SUPPORT = ClaimCitation(ref="S1", stance="supports")
_S_CONTRADICTION = ClaimCitation(ref="S2", stance="contradicts")
_S_OVERRIDES = {"S2": frozenset({"S1"})}


@pytest.mark.parametrize(
    ("citations", "outranks", "expected"),
    [
        ((_S_SUPPORT, _S_CONTRADICTION), _S_OVERRIDES, ("contradicted", False)),
        ((_S_SUPPORT, _S_CONTRADICTION), {}, ("supported", True)),
        ((_S_SUPPORT, _S_CONTRADICTION), {"S1": frozenset({"S2"})}, ("supported", False)),
        ((_S_SUPPORT,), _S_OVERRIDES, ("supported", False)),
        ((_S_CONTRADICTION,), _S_OVERRIDES, ("contradicted", False)),
        ((_SUPPORT, _S_CONTRADICTION), _S_OVERRIDES, ("supported", True)),
        ((_S_SUPPORT, ClaimCitation(ref="S2", stance="supports")), _S_OVERRIDES, ("supported", False)),
    ],
    ids=[
        "overridden-support-overruled",
        "no-override-conflict",
        "overriding-support-stands",
        "overriding-silent",
        "overriding-alone",
        "key-fact-outside-precedence",
        "agreeing-documents",
    ],
)
def test_precedence_in_the_verdict_table(
    citations: tuple[ClaimCitation, ...], outranks: dict[str, frozenset[str]], expected: tuple[str, bool]
) -> None:
    verdict = truth_verdict(citations, suspect=False, outranks=outranks)

    assert (verdict.verdict, verdict.evidence_conflict) == expected
    assert verdict.graded_against == "case"


# --- house rules, summaries and the candidate's own documents ---------------------------------------------------------

_TABLE = (Segment(text="At this table a natural 20 on a skill check always succeeds.", unit=0),)
_NATURAL_20 = _claim(
    _TABLE, 0, "A natural 20 on a skill check always succeeds.", "a natural 20 on a skill check always succeeds"
)
_HOUSE_RULES = EvidenceDocument(
    id="house-rules",
    provenance="case",
    title="House rules in force",
    text="House rule: a natural 20 on a skill check always succeeds.",
)
_FETCHED_RULEBOOK = EvidenceDocument(
    id="rulebook-page",
    provenance="run",
    title="A public rulebook page",
    url="https://example.org/rules/checks",
    text="Under the standard rules, a natural 20 on a skill check has no special effect.",
)


async def test_a_house_rule_truth_reads_supported_by_its_case_document_over_a_contradicting_fetched_passage() -> None:
    client = _FakeScriptedClient(
        [
            _batch(
                _answer(
                    0,
                    _cite(
                        "S1",
                        quote="House rule: a natural 20 on a skill check always succeeds.",
                        subject="a natural 20 on a skill check",
                        value="always succeeds",
                    ),
                )
            ),
            _ground((0, [])),
        ]
    )

    verification = await _verify(
        client, (_NATURAL_20,), key=None, documents=(_HOUSE_RULES, _FETCHED_RULEBOOK), grounded_in=("run",)
    )
    (graded,) = _grade((_NATURAL_20,), verification).claims

    truth, ground = client.calls
    assert _FETCHED_RULEBOOK.text not in truth["user"], "the truth step never sees a document the candidate fetched"
    assert _FETCHED_RULEBOOK.text in ground["user"] and _HOUSE_RULES.text not in ground["user"]
    assert (graded.verdict, graded.graded_against, graded.grounded) == ("supported", "case", False)
    assert [(entry.ref, entry.provenance) for entry in verification.evidence] == [("S1", "case"), ("S2", "run")]
    finding = verification.findings[0]
    assert finding is not None
    with pytest.raises(ValueError, match="never overrules case evidence"):
        truth_verdict(finding.truth_citations, suspect=finding.suspect, web="false")


_SOURCE = EvidenceDocument(
    id="source",
    provenance="case",
    title="The source",
    text="The committee met twice in March. It approved the new bridge budget at its second meeting.",
)
_SUMMARY = (Segment(text="The committee approved the bridge budget at its second March meeting.", unit=0),)
_APPROVED = _claim(
    _SUMMARY,
    0,
    "The committee approved the bridge budget at its second March meeting.",
    "The committee approved the bridge budget at its second March meeting",
)


async def test_a_source_faithful_summary_claim_reads_supported_and_grounded_on_one_numbered_passage() -> None:
    quote = "It approved the new bridge budget at its second meeting."
    client = _FakeScriptedClient(
        [
            _batch(_answer(0, _cite("S1", quote=quote, subject="It", value="approved the new bridge budget"))),
            _ground((0, [_ground_cite("S1", quote=quote, subject="It", value="approved the new bridge budget")])),
        ]
    )

    verification = await _verify(client, (_APPROVED,), key=None, documents=(_SOURCE,), grounded_in=("case",))
    (graded,) = _grade((_APPROVED,), verification).claims

    assert (graded.verdict, graded.graded_against, graded.grounded) == ("supported", "case", True)
    assert [entry.ref for entry in verification.evidence] == ["S1"], "both steps read the one passage, numbered once"
    assert [call.role for call in verification.calls] == ["truth", "ground"]
    assert graded.ground_citations[0].quote == quote


async def test_no_documents_given_reads_ungrounded_with_no_ground_call() -> None:
    client = _FakeScriptedClient([_batch(_answer(0))])

    verification = await _verify(client, (_NATURAL_20,), key=None, documents=(_HOUSE_RULES,), grounded_in=("run",))

    assert len(client.calls) == 1
    assert verification.findings[0] is not None and verification.findings[0].grounded is False


async def test_a_ground_citation_of_a_case_passage_is_no_evidence_shown_there() -> None:
    support = _ground_cite(
        "S1", quote=_HOUSE_RULES.text, subject="a natural 20 on a skill check", value="always succeeds"
    )
    client = _FakeScriptedClient(
        [_batch(_answer(0)), _ground((0, [support])), _Reply(content=json.dumps({"citations": []}))]
    )

    verification = await _verify(
        client, (_NATURAL_20,), key=None, documents=(_HOUSE_RULES, _FETCHED_RULEBOOK), grounded_in=("run",)
    )

    assert "it cites S1, which is no evidence shown here" in client.calls[2]["user"]
    assert verification.findings[0] is not None and verification.findings[0].grounded is False


_MESSY_PAGE = EvidenceDocument(
    id="messy-page",
    provenance="run",
    title="A fetched page",
    text="Lantern Works printed Harrow Tales. The bound volume of Harrow Tales followed in 1932.",
)
_BAD_GROUND = _ground_cite("S7", quote="nothing", subject="x", value="y")


def _ground_repair(*citations: dict[str, Any]) -> _Reply:
    return _Reply(content=json.dumps({"citations": list(citations)}))


async def _truth_only() -> Verification:
    return await _verify(
        _FakeScriptedClient(
            [
                _batch(_answer(0, _BAD)),
                _batch(_answer(1, _WRITTEN_BY_IDA_VENN), _answer(2)),
                _repair(_CONTRADICTED_BY_THE_BOUND_VOLUME),
            ]
        )
    )


async def test_a_ground_step_past_its_own_repair_budget_loses_only_groundedness() -> None:
    client = _FakeScriptedClient(
        [
            _batch(_answer(0, _BAD)),
            _batch(_answer(1, _WRITTEN_BY_IDA_VENN), _answer(2)),
            _repair(_CONTRADICTED_BY_THE_BOUND_VOLUME),
            _ground((0, [_BAD_GROUND])),
            _ground((1, [_BAD_GROUND]), (2, [_BAD_GROUND])),
        ]
    )

    verification = await _verify(
        client,
        documents=(_PRESS, _MESSY_PAGE),
        grounded_in=("run",),
        repairs=RepairBudget(rounds=2, max_items_per_cell=1),
    )
    alone = await _truth_only()

    assert verification.ground_loss == "repair_budget_spent"
    assert verification.ground_failure is not None and "would ask 3 ground claim(s)" in verification.ground_failure
    assert [f.truth_citations if f else None for f in verification.findings] == [
        f.truth_citations if f else None for f in alone.findings
    ]
    assert all(f is not None and f.grounded is None and f.ground_citations == () for f in verification.findings)
    graded = _grade(_CLAIMS, verification).claims
    assert [claim.verdict for claim in graded] == ["contradicted", "supported", "unverifiable"]
    assert [call.role for call in verification.calls] == ["truth", "truth", "repair", "ground", "ground"]


@pytest.mark.parametrize(
    ("ground_replies", "loss"),
    [
        ([_Reply(content="not json")] * 3, "unreadable"),
        ([_Reply(content='{"claims": [', stop_reason="max_tokens")], "cut_short"),
        (
            [
                _ground((0, [_BAD_GROUND])),
                _ground((1, []), (2, [])),
                _ground_repair(_BAD_GROUND),
                _ground_repair(_BAD_GROUND),
            ],
            "repair_budget_spent",
        ),
    ],
)
async def test_a_lost_ground_step_leaves_every_truth_verdict_as_it_was(ground_replies: list[_Reply], loss: str) -> None:
    client = _FakeScriptedClient(
        [
            _batch(_answer(0, _BAD)),
            _batch(_answer(1, _WRITTEN_BY_IDA_VENN), _answer(2)),
            _repair(_CONTRADICTED_BY_THE_BOUND_VOLUME),
            *ground_replies,
        ]
    )

    verification = await _verify(client, documents=(_PRESS, _MESSY_PAGE), grounded_in=("run",))
    alone = await _truth_only()

    assert verification.ground_loss == loss
    assert [claim.verdict for claim in _grade(_CLAIMS, verification).claims] == [
        claim.verdict for claim in _grade(_CLAIMS, alone).claims
    ]
    assert all(f is not None and f.grounded is None for f in verification.findings)


# --- the code checks, each a repair item ----------------------------------------------------------------------------

_LONG = EvidenceDocument(
    id="long-sentence",
    provenance="case",
    title="A long account",
    text="The bound volume of Harrow Tales, "
    + "which the press set by hand over a long winter, " * 7
    + "came out in 1932.",
)


@pytest.mark.parametrize(
    ("first", "failure"),
    [
        (_cite("S9", quote="anything", subject="a", value="b"), "it cites S9, which is no evidence shown here"),
        (_cite("S1"), "its citation of S1 quotes nothing"),
        (
            _cite("S1", quote="The bound volume came out in 1930.", subject="The bound volume", value="1930"),
            "its quote of S1 is not in passage S1 as shown",
        ),
        (
            _cite("S1", "contradicts", quote="The bound volume of Harrow Tales followed in 1932.", value="1932"),
            "its citation of S1 names no subject",
        ),
        (
            _cite(
                "S1", "contradicts", quote="The bound volume of Harrow Tales followed in 1932.", subject="Harrow Tales"
            ),
            "its citation of S1 names no value",
        ),
        (
            _cite(
                "S1",
                "contradicts",
                quote="The bound volume of Harrow Tales followed in 1932.",
                subject="Harrow Tales",
                value="1931",
            ),
            "its value for S1 is not in its quote",
        ),
        (
            _cite("K1", "contradicts"),
            "it is marked suspect and cites evidence",
        ),
    ],
)
async def test_an_answer_failing_a_check_is_asked_again_alone_with_its_failure_named(
    first: dict[str, Any], failure: str
) -> None:
    suspect = first["ref"] == "K1"
    client = _FakeScriptedClient(
        [
            _batch(_answer(0, first, suspect=suspect)),
            _batch(_answer(1), _answer(2)),
            _repair(_CONTRADICTED_BY_THE_BOUND_VOLUME),
        ]
    )

    verification = await _verify(client)

    assert failure in client.calls[2]["user"]
    assert json.dumps(first["ref"]) in client.calls[2]["user"], "the repair is shown its own last answer"
    assert _grade(_CLAIMS, verification).claims[0].verdict == "contradicted"


async def test_a_ref_cited_twice_is_asked_again() -> None:
    client = _FakeScriptedClient(
        [
            _batch(_answer(0, _CONTRADICTED_BY_THE_BOUND_VOLUME, _CONTRADICTED_BY_THE_BOUND_VOLUME)),
            _batch(_answer(1), _answer(2)),
            _repair(_CONTRADICTED_BY_THE_BOUND_VOLUME),
        ]
    )

    await _verify(client)

    assert "it cites S1 more than once" in client.calls[2]["user"]


async def test_a_quote_over_the_cap_is_asked_again() -> None:
    long_quote = _LONG.text
    assert len(long_quote) > 300
    client = _FakeScriptedClient(
        [
            _batch(_answer(0, _cite("S1", "contradicts", quote=long_quote, subject="The bound volume", value="1932"))),
            _batch(_answer(1), _answer(2)),
            _repair(_cite("S1", "contradicts", quote="came out in 1932.", subject="came out", value="1932")),
        ]
    )

    await _verify(client, documents=(_LONG,))

    assert f"its quote of S1 runs {len(long_quote)} characters" in client.calls[2]["user"]


async def test_a_quote_is_stored_as_the_documents_own_slice() -> None:
    curly = EvidenceDocument(
        id="curly", provenance="case", title="Curly", text="The press’s bound volume of Harrow Tales followed in 1932."
    )
    client = _FakeScriptedClient(
        [
            _batch(
                _answer(
                    0,
                    _cite(
                        "S1",
                        "contradicts",
                        quote="the press's bound volume of harrow tales followed in 1932",
                        subject="bound volume of harrow tales",
                        value="1932",
                    ),
                )
            ),
            _batch(_answer(1), _answer(2)),
        ]
    )

    verification = await _verify(client, documents=(curly,))

    (citation,) = verification.findings[0].truth_citations  # type: ignore[union-attr]
    assert citation.quote == "The press’s bound volume of Harrow Tales followed in 1932"
    assert citation.subject_span is not None
    assert citation.quote[citation.subject_span.start : citation.subject_span.end] == "bound volume of Harrow Tales"


# --- a key fact is quoted for the value it is cited for -------------------------------------------------------------

_CORPUS = Path(__file__).resolve().parent / "fixtures" / "claim_corpus"


def _corpus_lines(name: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (_CORPUS / name).read_text(encoding="utf-8").splitlines() if line.strip()]


@dataclass(frozen=True)
class _CorpusCase:
    """One invented corpus row as a cell: its excerpt the only segment, its claim, and the key its pointer names."""

    claim: ExtractedClaim
    key: ReferenceKey
    label: str
    entity: str
    fact: str


def _synthetic(claim_id: str) -> _CorpusCase:
    (row,) = [line for line in _corpus_lines("claimset/synthetic.jsonl") if line["claim_id"] == claim_id]
    (pointer,) = [line for line in _corpus_lines("keys/pointers_synthetic.jsonl") if line["claim_id"] == claim_id]
    entity, fact = pointer["key_fact"]["entity"], pointer["key_fact"]["claim"]
    segments = (Segment(text=row["utterance_excerpt"], unit=0),)
    key = ReferenceKey(
        core=(ReferenceEntity(name=entity, facts=(ReferenceFact(claim=fact, source_url=row["evidence_url"]),)),)
    )
    return _CorpusCase(_claim(segments, 0, row["claim"], row["utterance_excerpt"]), key, row["label"], entity, fact)


#: A key naming an artist's true home town, and a claim naming another town for the same artist.
_TOWN = _synthetic("syn-origin-01")

_CONTRADICTED_BY_THE_KEY = _cite("K1", "contradicts", quote=_TOWN.fact, subject=_TOWN.entity, value="Port Imber")


@pytest.mark.parametrize(
    ("first", "failure"),
    [
        (_cite("K1"), "its citation of K1 quotes nothing"),
        (_cite("K1", quote=_TOWN.fact, subject=_TOWN.entity), "its citation of K1 names no value"),
        (
            _cite("K1", quote=_TOWN.fact, subject=_TOWN.entity, value="Saltmarsh"),
            "its value for K1 is not in its quote",
        ),
    ],
    ids=["quotes-nothing", "names-the-subject-only", "names-a-value-the-fact-does-not-state"],
)
async def test_a_wrong_home_town_cited_as_the_keys_support_is_asked_again_and_reads_contradicted(
    first: dict[str, Any], failure: str
) -> None:
    """Support from a key fact names, inside the fact as shown, the words that state the value: a citation that quotes
    nothing, names only the subject, or names the claim's own value where the fact states another is asked again."""
    client = _FakeScriptedClient([_batch(_answer(0, first)), _repair(_CONTRADICTED_BY_THE_KEY)])

    verification = await _verify(client, (_TOWN.claim,), key=_TOWN.key, documents=())

    assert failure in client.calls[1]["user"]
    assert _grade((_TOWN.claim,), verification).claims[0].verdict == _TOWN.label == "contradicted"
    assert "A key fact (K) is quoted" in client.calls[0]["system"]


async def test_a_key_fact_cited_as_support_for_a_value_it_does_not_state_never_reads_supported() -> None:
    insists = _cite("K1", quote=_TOWN.fact, subject=_TOWN.entity, value="Saltmarsh")
    client = _FakeScriptedClient([_batch(_answer(0, insists)), _repair(insists), _repair(insists)])

    with pytest.raises(ClaimGradingFailed, match="its value for K1 is not in its quote") as failed:
        await _verify(client, (_TOWN.claim,), key=_TOWN.key, documents=())

    assert failed.value.loss == "repair_budget_spent"


async def test_a_key_citation_is_stored_as_the_key_facts_own_slice_with_its_spans() -> None:
    client = _FakeScriptedClient(
        [
            _batch(_answer(0)),
            _batch(
                _answer(
                    1, _cite("K1", quote="Harrow Tales was written by ida venn", subject="Harrow Tales", value="Venn")
                ),
                _answer(2),
            ),
        ]
    )

    verification = await _verify(client)

    assert verification.findings[1] is not None
    (citation,) = verification.findings[1].truth_citations
    assert citation.quote == "Harrow Tales was written by Ida Venn"
    assert citation.subject_span is not None and citation.value_span is not None
    assert citation.quote[citation.subject_span.start : citation.subject_span.end] == "Harrow Tales"
    assert citation.quote[citation.value_span.start : citation.value_span.end] == "Venn"


async def test_a_key_fact_that_does_not_name_its_subject_takes_the_name_its_line_gives_it() -> None:
    key = ReferenceKey(
        core=(
            ReferenceEntity(
                name=_TOWN.entity,
                facts=(ReferenceFact(claim="Born and raised in Port Imber.", source_url="https://example.org/tv"),),
            ),
        )
    )
    cited = _cite(
        "K1",
        "contradicts",
        quote="Tamsin Vell: Born and raised in Port Imber.",
        subject="Tamsin Vell",
        value="Port Imber",
    )
    client = _FakeScriptedClient([_batch(_answer(0, cited))])

    verification = await _verify(client, (_TOWN.claim,), key=key, documents=())

    assert _grade((_TOWN.claim,), verification).claims[0].verdict == "contradicted"
    assert "K1. A key fact about Tamsin Vell: Born and raised in Port Imber." in client.calls[0]["user"]


# --- repair budget and unreadable replies ---------------------------------------------------------------------------

_BAD = _cite("S9", quote="anything", subject="a", value="b")


async def test_an_unreadable_repair_reply_is_named_in_the_next_round() -> None:
    client = _FakeScriptedClient(
        [
            _batch(_answer(0, _BAD)),
            _batch(_answer(1), _answer(2)),
            _Reply(content="not json"),
            _repair(_CONTRADICTED_BY_THE_BOUND_VOLUME),
        ]
    )

    await _verify(client)

    assert "the last correction could not be used" in client.calls[3]["user"]


async def test_a_claim_still_failing_after_its_rounds_raises_naming_it_and_every_call_is_billed() -> None:
    client = _FakeScriptedClient(
        [_batch(_answer(0, _BAD)), _batch(_answer(1), _answer(2)), _repair(_BAD), _repair(_BAD)]
    )
    ledger = RoleUsageLedger(role="judge")

    with pytest.raises(
        ClaimGradingFailed, match=r"could not repair 1 claim\(s\) in 2 round\(s\): truth claim 0"
    ) as failed:
        await _verify(client, ledger=ledger)

    assert failed.value.loss == "repair_budget_spent"
    (row,) = ledger.rows()
    assert row.call_count == 4


async def test_more_failing_items_than_the_cells_repair_budget_raises_before_asking() -> None:
    client = _FakeScriptedClient([_batch(_answer(0, _BAD)), _batch(_answer(1, _BAD), _answer(2))])

    with pytest.raises(ClaimGradingFailed, match="would ask 2 truth claim") as failed:
        await _verify(client, repairs=RepairBudget(rounds=2, max_items_per_cell=1))

    assert failed.value.loss == "repair_budget_spent"
    assert len(client.calls) == 2


async def test_no_repair_rounds_makes_a_failing_item_the_criterions_loss_at_once() -> None:
    client = _FakeScriptedClient([_batch(_answer(0, _BAD)), _batch(_answer(1), _answer(2))])

    with pytest.raises(ClaimGradingFailed, match="in 0 round") as failed:
        await _verify(client, repairs=RepairBudget(rounds=0, max_items_per_cell=5))

    assert failed.value.loss == "repair_budget_spent"
    assert len(client.calls) == 2


async def test_an_unusable_batch_reply_is_asked_again_in_full_naming_why_then_refused() -> None:
    client = _FakeScriptedClient(
        [
            _Reply(content="not json"),
            _batch(_answer(1)),  # answers a claim not asked, and leaves claim 0 out
            _batch({"claim": 0, "citations": []}),  # suspect is missing
        ]
    )
    ledger = RoleUsageLedger(role="judge")

    with pytest.raises(ClaimGradingFailed, match="could not be used in 3 attempts") as failed:
        await _verify(client, ledger=ledger)

    assert failed.value.loss == "unreadable"
    first, second, third = client.calls
    assert "# Your last reply" not in first["user"]
    assert "No valid JSON object" in second["user"]
    assert "leaves out claims [0]; answers claims [1] that were not asked" in third["user"]
    (row,) = ledger.rows()
    assert row.call_count == 3


async def test_a_reply_cut_short_is_refused_at_once_and_still_billed() -> None:
    client = _FakeScriptedClient([_Reply(content='{"claims": [', stop_reason="max_tokens")])
    ledger = RoleUsageLedger(role="judge")

    with pytest.raises(ClaimGradingFailed, match="truth reply was cut short") as failed:
        await _verify(client, ledger=ledger)

    assert failed.value.loss == "cut_short"
    (row,) = ledger.rows()
    assert row.call_count == 1


async def test_an_extraction_failure_names_its_loss_too() -> None:
    client = _FakeScriptedClient([_Reply(content='{"claims": [', stop_reason="max_tokens")])

    with pytest.raises(ClaimGradingFailed) as failed:
        await ClaimExtractor(client=client).extract(_SEGMENTS, ledger=RoleUsageLedger(role="judge"))

    assert failed.value.loss == "cut_short"


# --- batches, restatements and session claims -----------------------------------------------------------------------

_SPREAD = (
    Segment(text="Tonight: two books.", unit=None),
    Segment(text="Harrow Tales came from Lantern Works.", unit=0),
    Segment(text="Its bound volume followed in 1932.", unit=1),
    Segment(text="Lantern Works, again, printed Harrow Tales.", unit=0),
)


async def test_one_call_per_unit_in_unit_order_then_one_outside_any_unit() -> None:
    claims = (
        _claim(_SPREAD, 0, "There are two books tonight.", "two books", scope="session"),
        _claim(_SPREAD, 1, "Lantern Works printed Harrow Tales.", "Harrow Tales came from Lantern Works"),
        _claim(_SPREAD, 2, "The bound volume of Harrow Tales followed in 1932.", "Its bound volume followed in 1932"),
        _claim(
            _SPREAD, 3, "Lantern Works printed Harrow Tales.", "Lantern Works, again, printed Harrow Tales", restates=1
        ),
        _claim(_SPREAD, 0, "Two books are reviewed.", "Tonight: two books"),
    )
    client = _FakeScriptedClient(
        [
            _batch(
                _answer(
                    1,
                    _cite(
                        "S1",
                        quote="Lantern Works printed small runs for local writers.",
                        subject="Lantern Works",
                        value="printed small runs",
                    ),
                )
            ),
            _batch(_answer(2, _CONTRADICTED_BY_THE_BOUND_VOLUME)),
            _batch(_answer(4)),
        ]
    )

    verification = await _verify(client, claims)
    grade = _grade(claims, verification)

    unit0, unit1, outside = client.calls
    assert "Claim 1, in unit 0" in unit0["user"]
    assert 'It is said again, as claim 3: "Lantern Works, again, printed Harrow Tales"' in unit0["user"]
    assert "Claim 3," not in unit0["user"], "a restatement is shown beside its original, never asked"
    assert "Claim 2, in unit 1" in unit1["user"]
    assert "Claim 4, outside any unit" in outside["user"]
    assert all("Claim 0," not in call["user"] for call in client.calls), "a session claim is never verified"
    assert verification.findings[0] is None
    assert grade.claims[0].verdict is None and grade.claims[0].scope == "session"
    assert verification.findings[3] == verification.findings[1], "a restatement takes its original's answer"
    assert grade.claims[3].verdict == "supported" and grade.claims[3].decided_by == 0


async def test_a_cell_of_session_claims_only_makes_no_call() -> None:
    claims = (_claim(_SPREAD, 0, "There are two books tonight.", "two books", scope="session"),)
    client = _FakeScriptedClient([])

    verification = await _verify(client, claims)

    assert client.calls == []
    assert verification == Verification((), (), (), (None,))


async def test_a_restatement_of_no_earlier_original_is_refused_before_any_call() -> None:
    claims = (_BOUND_YEAR, _claim(_SEGMENTS, 2, "Ida Venn wrote it.", "Ida Venn wrote it", restates=1))
    client = _FakeScriptedClient([])

    with pytest.raises(ValueError, match="restates claim 1, which is no earlier original"):
        await _verify(client, claims)

    assert client.calls == []


# --- page selection -------------------------------------------------------------------------------------------------

_WIDTH = 120


def _tile(text: str) -> str:
    assert len(text) <= _WIDTH
    return text.ljust(_WIDTH)


_FILLER = _tile("The cellar kept ledgers and spare type cases for many seasons, stacked by the stairs.")

#: Four tiles: filler, filler, the two editions, filler.
_ARCHIVE = EvidenceDocument(
    id="archive",
    provenance="case",
    title="The archive",
    text=_FILLER
    + _FILLER
    + _tile("The pamphlet edition of Harrow Tales appeared in 1931. The bound volume of Harrow Tales followed in 1932.")
    + _FILLER,
)

_LEDGER = EvidenceDocument(
    id="ledger", provenance="case", title="A ledger", text=_FILLER + _tile("Harrow Tales sold well in 1932.") + _FILLER
)

_TIGHT = PageBudget(per_document_chars=2 * _WIDTH, total_chars=3 * _WIDTH, window_chars=_WIDTH)


def _spans(verification: Verification) -> list[tuple[str, str | None, int, int]]:
    return [
        (entry.ref, entry.document, entry.span.start, entry.span.end)
        for entry in verification.evidence
        if entry.span is not None
    ]


async def test_the_page_budget_reads_the_anchored_tile_first_and_records_the_truncation() -> None:
    client = _FakeScriptedClient([_batch(_answer(0)), _batch(_answer(1), _answer(2))])

    verification = await _verify(client, documents=(_ARCHIVE,), pages=_TIGHT)

    assert _spans(verification) == [("S1", "archive", 0, _WIDTH), ("S2", "archive", 2 * _WIDTH, 3 * _WIDTH)]
    assert verification.documents_truncated == ("archive",)
    assert "The bound volume of Harrow Tales followed in 1932." in client.calls[0]["user"]


async def test_the_total_budget_is_shared_in_rounds_across_documents() -> None:
    client = _FakeScriptedClient([_batch(_answer(0)), _batch(_answer(1), _answer(2))])

    verification = await _verify(client, documents=(_ARCHIVE, _LEDGER), pages=_TIGHT)

    assert _spans(verification) == [
        ("S1", "archive", 0, _WIDTH),
        ("S2", "archive", 2 * _WIDTH, 3 * _WIDTH),
        ("S3", "ledger", _WIDTH, 2 * _WIDTH),
    ]
    assert verification.documents_truncated == ("archive", "ledger")


async def test_adjacent_tiles_join_into_one_passage() -> None:
    client = _FakeScriptedClient([_batch(_answer(0)), _batch(_answer(1), _answer(2))])
    wide = PageBudget(per_document_chars=4 * _WIDTH, total_chars=4 * _WIDTH, window_chars=_WIDTH)

    verification = await _verify(client, documents=(_ARCHIVE,), pages=wide)

    assert _spans(verification) == [("S1", "archive", 0, 4 * _WIDTH)]
    assert verification.documents_truncated == ()


async def test_the_same_inputs_give_the_same_selection_and_prompts() -> None:
    replies = [_batch(_answer(0)), _batch(_answer(1), _answer(2))]
    first, second = _FakeScriptedClient(list(replies)), _FakeScriptedClient(list(replies))

    one = await _verify(first, documents=(_ARCHIVE, _LEDGER), pages=_TIGHT)
    two = await _verify(second, documents=(_ARCHIVE, _LEDGER), pages=_TIGHT)

    assert one.evidence == two.evidence and one.documents_truncated == two.documents_truncated
    assert first.calls == second.calls


async def test_a_fetched_document_never_moves_what_the_truth_step_reads() -> None:
    fetched = EvidenceDocument(id="fetched", provenance="run", title="Fetched", text=_FILLER * 6)
    without = _FakeScriptedClient([_batch(_answer(0)), _batch(_answer(1), _answer(2))])
    with_it = _FakeScriptedClient(
        [_batch(_answer(0)), _batch(_answer(1), _answer(2)), _ground((0, [])), _ground((1, []), (2, []))]
    )

    alone = await _verify(without, documents=(_ARCHIVE,), pages=_TIGHT)
    # Fetched first: under one budget shared across both steps' documents, it would take the archive's second tile.
    beside = await _verify(with_it, documents=(fetched, _ARCHIVE), pages=_TIGHT, grounded_in=("run",))

    def case_spans(verification: Verification) -> list[tuple[str, str | None, int, int]]:
        return [
            (entry.ref, entry.document, entry.span.start, entry.span.end)
            for entry in verification.evidence
            if entry.span is not None and entry.provenance == "case"
        ]

    assert (
        case_spans(beside)
        == case_spans(alone)
        == [
            ("S1", "archive", 0, _WIDTH),
            ("S2", "archive", 2 * _WIDTH, 3 * _WIDTH),
        ]
    )
    assert [call["user"] for call in with_it.calls[:2]] == [call["user"] for call in without.calls]


async def test_a_quote_from_a_part_of_the_document_the_call_was_not_shown_is_asked_again() -> None:
    narrow = PageBudget(per_document_chars=_WIDTH, total_chars=_WIDTH, window_chars=_WIDTH)
    unshown = _FILLER.strip()
    client = _FakeScriptedClient(
        [
            _batch(_answer(0)),
            _batch(
                _answer(1, _cite("S1", "contradicts", quote=unshown, subject="The cellar", value="ledgers")), _answer(2)
            ),
            _repair(_cite("S1", quote="Harrow Tales sold well in 1932.", subject="Harrow Tales", value="sold well")),
        ]
    )

    verification = await _verify(client, documents=(_LEDGER,), pages=narrow)

    assert _spans(verification) == [("S1", "ledger", _WIDTH, 2 * _WIDTH)]
    assert unshown in _LEDGER.text
    assert "its quote of S1 is not in passage S1 as shown" in client.calls[2]["user"]


# --- refusals and identity ------------------------------------------------------------------------------------------


async def test_a_ledger_of_another_role_is_refused() -> None:
    with pytest.raises(ValueError, match="billed to the 'judge' role"):
        await _verify(_FakeScriptedClient([]), ledger=RoleUsageLedger(role="candidate"))


async def test_grounded_in_with_undeclared_documents_asks_no_ground_step_and_loses_it_alone() -> None:
    """Nothing says what the candidate was given, so no ground call is made: the ground step is lost as
    ``documents_undeclared`` and every truth finding is what the same cell reads with no ground step asked."""

    def script() -> _FakeScriptedClient:
        return _FakeScriptedClient([_batch(_answer(0)), _batch(_answer(1, _WRITTEN_BY_IDA_VENN), _answer(2))])

    undeclared = await _verify(script(), documents=None, grounded_in=("run",))
    alone = await _verify(script(), documents=None)

    assert undeclared.ground_loss == "documents_undeclared"
    assert undeclared.ground_failure is not None and "declares no documents" in undeclared.ground_failure
    assert [call.role for call in undeclared.calls] == ["truth", "truth"], "no ground call is asked"
    assert [f.truth_citations if f else None for f in undeclared.findings] == [
        f.truth_citations if f else None for f in alone.findings
    ]
    assert all(f is not None and f.grounded is None for f in undeclared.findings)
    assert [c.verdict for c in _grade(_CLAIMS, undeclared).claims] == [c.verdict for c in _grade(_CLAIMS, alone).claims]
    nothing = await _verify(_FakeScriptedClient([]), (), documents=None, grounded_in=("run",))
    assert (nothing.calls, nothing.ground_loss) == ((), "documents_undeclared"), "a cell with no claim loses it too"


def test_a_client_that_names_no_model_is_refused() -> None:
    with pytest.raises(ValueError, match="names its model"):
        _verifier(_FakeUnnamedClient())


async def test_the_hosts_guidance_reaches_every_prompt() -> None:
    guidance = "A printing's year is the year its copies were sold."
    client = _FakeScriptedClient(
        [_batch(_answer(0, _BAD)), _batch(_answer(1), _answer(2)), _repair(_CONTRADICTED_BY_THE_BOUND_VOLUME)]
    )

    await _verify(client, guidance=guidance)

    assert all(call["system"].endswith(guidance) for call in client.calls)


#: The identity of :func:`_verifier` over a client of :data:`_MODEL`, with :data:`_WHOLE` and :data:`_REPAIRS`, from
#: the shipped prompts, schemas, selection and batch rules, quote cap, attempts and fold. Produced by
#: ``ClaimVerifier.identity`` itself; it moves with any change to them, and with the fold's tables or Unicode
#: version, which is a new definition: write the new value here deliberately.
_PINNED_IDENTITY = "scripted/verify|claimverify:fdeb461d61f3"


def test_the_identity_names_the_model_and_moves_with_every_budget_and_setting() -> None:
    client = _FakeScriptedClient([])
    base = _verifier(client)

    assert base.identity == _PINNED_IDENTITY
    assert base.identity == _verifier(_FakeScriptedClient([])).identity
    others = [
        _verifier(client, guidance="A printing's year is the year its copies were sold.").identity,
        _verifier(_FakeScriptedClient([], model_name="scripted/other")).identity,
        _verifier(client, pages=PageBudget(per_document_chars=4000, total_chars=8000, window_chars=1000)).identity,
        _verifier(client, pages=PageBudget(per_document_chars=3000, total_chars=8000, window_chars=2000)).identity,
        _verifier(client, pages=PageBudget(per_document_chars=4000, total_chars=9000, window_chars=2000)).identity,
        _verifier(client, repairs=RepairBudget(rounds=1, max_items_per_cell=10)).identity,
        _verifier(client, repairs=RepairBudget(rounds=2, max_items_per_cell=9)).identity,
        _verifier(client, grounded_in=("run",)).identity,
        _verifier(client, grounded_in=("case",)).identity,
        _verifier(client, grounded_in=("case", "run")).identity,
    ]
    assert base.identity not in others
    assert len(set(others)) == len(others)
    assert (
        _verifier(client, grounded_in=("run", "case")).identity
        == _verifier(client, grounded_in=("case", "run")).identity
    )
    assert _verifier(client, guidance="  ").identity == base.identity, "blank guidance is no guidance"
