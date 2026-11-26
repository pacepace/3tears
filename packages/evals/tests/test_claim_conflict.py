"""A case conflict is asked again, and never reads supported by default; a support covers every value a claim states.

Every behaviour is driven through the claim verifier on a scripted client, so nothing is spent and each reply is
chosen. A world claim whose case citations, once precedence is read, both support and contradict it joins the cell's
one re-ask call, shown exactly the evidence it cited on both sides, and every citation check applies. Only support
cited reads ``supported``; only contradiction ``contradicted``; both, or nothing, ``unverifiable`` with
``evidence_conflict``, which the web never checks. That is the opposite of a doubted claim's re-ask, where citing
nothing leaves the batch's answer standing, and both rules hold in one cell.

The three conflicts reproduced first are a broadcast pass's, scripted with invented names: in each, one key fact gives
another value for what the claim states, and the batch cites a key fact about another property as its support.

The compound cases hold the truth and re-ask rule that a claim stating several values is supported only where the
evidence cited as support states every one of them. Code cannot read whether a fact covers a value, so a scripted
reply here shows only that the code reads each reply right and that the rule reaches each prompt; whether a model
obeys it is measured on a paid pass, never here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest
from pydantic import ValidationError

from threetears.evals.kernel import RoleUsageLedger
from threetears.evals.kernel.claims.extraction import ExtractedClaim
from threetears.evals.kernel.claims.failures import ClaimGradingFailed
from threetears.evals.kernel.claims.verification import ClaimVerifier, Verification, graded_claims
from threetears.evals.schema import Segment
from threetears.evals.schema.models import (
    ClaimCitation,
    ClaimGrade,
    ClaimJudgeCounts,
    EvidenceDocument,
    GradedClaim,
    PageBudget,
    RepairBudget,
)
from threetears.evals.schema.reference import ReferenceEntity, ReferenceFact, ReferenceKey

_MODEL = "scripted/verify"

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
    sent_settings: Any = None


# parity-with: threetears.evals.schema.CompletionClient
@dataclass
class _FakeScriptedClient:
    """Answers each call with the next scripted reply, or raises it, and records every prompt pair and response format
    sent."""

    replies: list[_Reply | Exception]
    model_name: str = _MODEL
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def generate(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> _Reply:
        """The next scripted reply; a scripted exception is raised, as a provider's failure is."""
        self.calls.append({"system": system, "user": user, "response_format": response_format})
        reply = self.replies[len(self.calls) - 1]
        if isinstance(reply, Exception):
            raise reply
        return reply

    async def aclose(self) -> None:
        """Nothing to release."""


# parity-with: threetears.evals.schema.completion.BoundCompletionClient
@dataclass
class _FakeBytePricedClient(_FakeScriptedClient):
    """Prices a call by the bytes it sends, so evidence a priced prompt leaves out shows as a ceiling too low."""

    def price_ceiling(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> float:
        del response_format
        return (len(system.encode()) + len(user.encode())) * 1e-6


def _cite(ref: str, stance: str = "supports", *, quote: str, subject: str, value: str) -> dict[str, Any]:
    return {"ref": ref, "stance": stance, "quote": quote, "subject": subject, "value": value}


def _answer(claim: int, *citations: dict[str, Any], suspect: bool = False) -> dict[str, Any]:
    return {"claim": claim, "citations": list(citations), "suspect": suspect}


def _batch(*answers: dict[str, Any]) -> _Reply:
    return _Reply(content=json.dumps({"claims": list(answers)}))


def _reasked(*items: tuple[int, list[dict[str, Any]]]) -> _Reply:
    """The re-ask's reply: each claim's citations, and no suspicion."""
    return _Reply(content=json.dumps({"claims": [{"claim": c, "citations": cites} for c, cites in items]}))


def _repair(*citations: dict[str, Any], suspect: bool = False) -> _Reply:
    return _Reply(content=json.dumps({"citations": list(citations), "suspect": suspect}))


def _schema(call: dict[str, Any]) -> str:
    return str(call["response_format"]["json_schema"]["name"])


def _key(*facts: tuple[str, str]) -> ReferenceKey:
    """A key of one entity per name given, each holding the facts given for it, numbered K1.. in this order."""
    named: dict[str, list[str]] = {}
    for name, fact in facts:
        named.setdefault(name, []).append(fact)
    return ReferenceKey(
        core=tuple(
            ReferenceEntity(
                name=name,
                facts=tuple(ReferenceFact(claim=fact, source_url="https://example.org/key") for fact in held),
            )
            for name, held in named.items()
        )
    )


def _ref(key: ReferenceKey, fact: str) -> str:
    """The ref a verify call numbers ``fact`` by: its place among the key's facts, in the key's own order."""
    facts = [held.claim for entity in key.fact_entities for held in entity.facts]
    return f"K{facts.index(fact) + 1}"


def _claims(*texts: str, restates: dict[int, int] | None = None) -> tuple[ExtractedClaim, ...]:
    """One world claim per text, all said in one segment of unit 0, each its own proposition and excerpt."""
    segment = Segment(text=" ".join(texts), unit=0)
    listed: list[ExtractedClaim] = []
    for position, text in enumerate(texts):
        excerpt = text.rstrip(".")
        start = segment.text.index(excerpt, sum(len(t) + 1 for t in texts[:position]))
        listed.append(
            ExtractedClaim(
                segment=0,
                unit=0,
                claim=text,
                excerpt=excerpt,
                start=start,
                end=start + len(excerpt),
                specific=True,
                hedged=False,
                attributed=False,
                scope="world",
                restates=(restates or {}).get(position),
            )
        )
    return tuple(listed)


async def _verify(
    client: _FakeScriptedClient,
    claims: tuple[ExtractedClaim, ...],
    *,
    key: ReferenceKey | None,
    documents: tuple[EvidenceDocument, ...] = (),
    repairs: RepairBudget = _REPAIRS,
) -> Verification:
    verifier = ClaimVerifier(client=client, pages=_WHOLE, repairs=repairs)
    return await verifier.verify(claims, key=key, documents=documents, ledger=RoleUsageLedger(role="judge"))


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


def _read(claim: GradedClaim) -> tuple[Any, ...]:
    return (claim.verdict, claim.graded_against, claim.evidence_conflict, claim.nominated_by, claim.web)


# --- three conflicts from a broadcast pass, reproduced --------------------------------------------------------------


@dataclass(frozen=True)
class _Conflict:
    """A claim, the key fact that gives another value for what it states, and a fact about another property of its
    subject that the batch cites as its support."""

    entity: str
    claim: str
    against: str
    against_value: str
    beside: str
    beside_value: str


_PASS_CONFLICTS = {
    "began-in-another-city": _Conflict(
        entity="Rook Vale",
        claim="Rook Vale began in Glasgow with Fen Tribe.",
        against="Rook Vale's first band, in Edinburgh, was Fen Tribe.",
        against_value="Edinburgh",
        beside="Rook Vale played guitar in Fen Tribe.",
        beside_value="guitar",
    ),
    "a-record-of-another-year": _Conflict(
        entity="Ghostline",
        claim="Ghostline is a Delta Hum record from 1994.",
        against="Ghostline came out in 1991.",
        against_value="1991",
        beside="'Burnt' is on Ghostline.",
        beside_value="Burnt",
    ),
    "public-attention-in-another-year": _Conflict(
        entity="Mal Rennick",
        claim="Mal Rennick came to public attention in 1979 as the Pale Hours' manager.",
        against="Mal Rennick came to public attention in 1976 as the Pale Hours' manager.",
        against_value="1976",
        beside="Mal Rennick was the Pale Hours' manager; he named the band.",
        beside_value="manager",
    ),
}

#: A fact about the same entity that bears on neither side: never shown in the re-ask, since the claim cited none of it.
_UNCITED = "was born in a river town."


@pytest.mark.parametrize("case", list(_PASS_CONFLICTS.values()), ids=list(_PASS_CONFLICTS))
async def test_a_case_conflict_from_the_broadcast_pass_is_asked_again_and_reads_contradicted(case: _Conflict) -> None:
    """The batch cites the fact giving another value as a contradiction and a fact about another property as support.
    The claim is asked again in the re-ask, shown the two facts it cited and no other, and the re-ask's contradiction
    reads contradicted, from the case."""
    claims = _claims(case.claim)
    key = _key((case.entity, case.against), (case.entity, case.beside), (case.entity, _UNCITED))
    at, by = _ref(key, case.against), _ref(key, case.beside)
    against = _cite(at, "contradicts", quote=case.against, subject=case.entity, value=case.against_value)
    beside = _cite(by, quote=case.beside, subject=case.entity, value=case.beside_value)
    client = _FakeScriptedClient([_batch(_answer(0, against, beside)), _reasked((0, [against]))])

    (claim,) = _grade(claims, await _verify(client, claims, key=key)).claims

    assert _read(claim) == ("contradicted", "case", False, None, "none")
    assert claim.decided_by == 1, "the re-ask decided it"
    assert [_schema(call) for call in client.calls] == ["claim_truth", "claim_reask"]
    asked = client.calls[1]["user"]
    assert f"{at}. A key fact about {case.entity}: {case.against}" in asked
    assert f"{by}. A key fact about {case.entity}: {case.beside}" in asked
    assert f"Its evidence: {', '.join(sorted((at, by)))}" in asked
    assert _UNCITED not in asked, "the re-ask is shown the evidence the claim cited, not its entity's other facts"


# --- the conflict's outcome, adversarially ---------------------------------------------------------------------------

_LIVES = "Ann Lee lives in York."
_BORN = "Ann Lee was born in Leeds."
_LIVES_SUPPORT = _cite("K1", quote=_LIVES, subject="Ann Lee", value="York")
_BORN_MISREAD = _cite("K2", "contradicts", quote=_BORN, subject="Ann Lee", value="Leeds")
_LEE_KEY = _key(("Ann Lee", _LIVES), ("Ann Lee", _BORN))


async def test_a_true_claim_whose_batch_misreads_a_contradiction_reads_supported_when_the_reask_cites_only_support() -> (
    None
):
    """A key fact on another property, cited as a contradiction beside a real support, makes a conflict; a re-ask
    that cites only the support reads supported, with no conflict left."""
    claims = _claims(_LIVES)
    client = _FakeScriptedClient([_batch(_answer(0, _LIVES_SUPPORT, _BORN_MISREAD)), _reasked((0, [_LIVES_SUPPORT]))])

    (claim,) = _grade(claims, await _verify(client, claims, key=_LEE_KEY)).claims

    assert _read(claim) == ("supported", "case", False, None, "none")
    assert [c.ref for c in claim.truth_citations] == ["K1"], "the re-ask's citations stand for it"


async def test_a_conflict_the_reask_still_cites_both_sides_of_reads_unverifiable_with_the_conflict() -> None:
    claims = _claims(_LIVES)
    both = [_LIVES_SUPPORT, _BORN_MISREAD]
    client = _FakeScriptedClient([_batch(_answer(0, *both)), _reasked((0, both))])

    (claim,) = _grade(claims, await _verify(client, claims, key=_LEE_KEY)).claims

    assert _read(claim) == ("unverifiable", None, True, None, "none")
    assert not claim.suspect, "a conflict is never nominated: the web never checks it"
    assert claim.decided_by == 1


async def test_a_conflict_whose_reask_cites_nothing_reads_unverifiable_never_the_batchs_supported() -> None:
    """Unlike a doubted claim, whose batch answer stands when its re-ask cites nothing, a conflict's batch answer
    never stands as supported: its citations stay on record, and it reads unverifiable with the conflict."""
    claims = _claims(_LIVES)
    client = _FakeScriptedClient([_batch(_answer(0, _LIVES_SUPPORT, _BORN_MISREAD)), _reasked((0, []))])

    (claim,) = _grade(claims, await _verify(client, claims, key=_LEE_KEY)).claims

    assert _read(claim) == ("unverifiable", None, True, None, "none")
    assert [(c.ref, c.stance) for c in claim.truth_citations] == [("K1", "supports"), ("K2", "contradicts")]
    assert claim.decided_by == 1, "the re-ask, which cited nothing, decided it"


_POLICY = EvidenceDocument(
    id="policy", provenance="case", title="Lending policy", text="A loan runs two weeks. Members may renew once."
)
_UPDATE = EvidenceDocument(
    id="update",
    provenance="case",
    title="Policy update",
    text="From March, a loan runs three weeks.",
    overrides=("policy",),
)
_TWO_WEEKS = _cite("S1", quote="A loan runs two weeks.", subject="A loan", value="two weeks")
_THREE_WEEKS = _cite("S2", quote="From March, a loan runs three weeks.", subject="a loan", value="three weeks")


def _stance(cited: dict[str, Any], stance: str) -> dict[str, Any]:
    return {**cited, "stance": stance}


async def test_precedence_decides_first_and_a_conflict_it_settles_is_never_asked_again() -> None:
    """A case document's support over an overridden document's contradiction reads supported, and the reverse
    contradicted, each from the case with no conflict, and neither is asked again."""
    claims = _claims("A loan runs three weeks.", "A loan runs two weeks.")
    client = _FakeScriptedClient(
        [
            _batch(
                _answer(0, _THREE_WEEKS, _stance(_TWO_WEEKS, "contradicts")),
                _answer(1, _TWO_WEEKS, _stance(_THREE_WEEKS, "contradicts")),
            )
        ]
    )

    grade = _grade(claims, await _verify(client, claims, key=None, documents=(_POLICY, _UPDATE)))

    assert [_read(claim) for claim in grade.claims] == [
        ("supported", "case", False, None, "none"),
        ("contradicted", "case", False, None, "none"),
    ]
    assert [_schema(call) for call in client.calls] == ["claim_truth"], "no re-ask"


async def test_a_conflict_between_case_passages_is_shown_its_passages_and_their_precedence_and_no_key_is_needed() -> (
    None
):
    """With no override between them, two case documents' passages on opposite sides are a conflict: the re-ask is
    shown the two passages the claim cited, and a re-ask citing only one side reads from it."""
    plain = _UPDATE.model_copy(update={"overrides": ()})
    claims = _claims("A loan runs two weeks.")
    client = _FakeScriptedClient(
        [
            _batch(_answer(0, _TWO_WEEKS, _stance(_THREE_WEEKS, "contradicts"))),
            _reasked((0, [_stance(_THREE_WEEKS, "contradicts")])),
        ]
    )

    (claim,) = _grade(claims, await _verify(client, claims, key=None, documents=(_POLICY, plain))).claims

    assert _read(claim) == ("contradicted", "case", False, None, "none")
    asked = client.calls[1]["user"]
    assert 'S1. From "Lending policy"' in asked and 'S2. From "Policy update"' in asked
    assert "Its evidence: S1, S2" in asked


async def test_a_conflict_and_a_doubted_claim_share_the_one_reask_and_each_is_read_by_its_own_rule() -> None:
    """Both re-ask items cite nothing: the doubted claim's batch answer stands, so it stays nominated for the web;
    the conflict's never does, so it reads unverifiable with the conflict and is nominated for nothing."""
    claims = _claims(_LIVES, "Bo Finch lives in Ely.")
    key = _key(("Ann Lee", _LIVES), ("Ann Lee", _BORN), ("Bo Finch", "Bo Finch lives in Hull."))
    client = _FakeScriptedClient(
        [
            _batch(_answer(0, _LIVES_SUPPORT, _BORN_MISREAD), _answer(1, suspect=True)),
            _reasked((0, []), (1, [])),
        ]
    )

    grade = _grade(claims, await _verify(client, claims, key=key))

    assert [_schema(call) for call in client.calls] == ["claim_truth", "claim_reask"], "one re-ask for the cell"
    asked = client.calls[1]["user"]
    assert "Its evidence: K1, K2" in asked and "Its evidence: K3" in asked
    conflict, doubted = grade.claims
    assert _read(conflict) == ("unverifiable", None, True, None, "none")
    assert _read(doubted) == ("unverifiable", None, False, "judge", "none")
    assert doubted.suspect and doubted.decided_by == 0, "a doubted claim's batch answer stands"


async def test_a_conflict_and_a_doubted_claim_each_take_the_reasks_citations() -> None:
    claims = _claims(_LIVES, "Bo Finch lives in Ely.")
    hull = "Bo Finch lives in Hull."
    key = _key(("Ann Lee", _LIVES), ("Ann Lee", _BORN), ("Bo Finch", hull))
    in_hull = _cite("K3", "contradicts", quote=hull, subject="Bo Finch", value="Hull")
    client = _FakeScriptedClient(
        [
            _batch(_answer(0, _LIVES_SUPPORT, _BORN_MISREAD), _answer(1, suspect=True)),
            _reasked((0, [_LIVES_SUPPORT]), (1, [in_hull])),
        ]
    )

    grade = _grade(claims, await _verify(client, claims, key=key))

    assert [_read(claim) for claim in grade.claims] == [
        ("supported", "case", False, None, "none"),
        ("contradicted", "case", False, None, "none"),
    ]
    assert len(client.calls) == 2


async def test_a_reask_item_citing_evidence_shown_only_for_another_claim_fails_the_check() -> None:
    """The re-ask showed K3 for the doubted claim alone: cited for the conflict, it is no evidence shown for it, so
    the item is repaired, shown the conflict's own evidence, and never reads from K3."""
    claims = _claims(_LIVES, "Bo Finch lives in Ely.")
    hull = "Bo Finch lives in Hull."
    key = _key(("Ann Lee", _LIVES), ("Ann Lee", _BORN), ("Bo Finch", hull))
    borrowed = _cite("K3", "contradicts", quote=hull, subject="Bo Finch", value="Hull")
    client = _FakeScriptedClient(
        [
            _batch(_answer(0, _LIVES_SUPPORT, _BORN_MISREAD), _answer(1, suspect=True)),
            _reasked((0, [borrowed]), (1, [])),
            _repair(),
        ]
    )

    grade = _grade(claims, await _verify(client, claims, key=key))

    repair = client.calls[2]["user"]
    assert "it cites K3, which is no evidence shown here" in repair
    assert hull not in repair.split("# The claim")[0], "the repair is shown the conflict's own evidence"
    assert _BORN in repair.split("# The claim")[0]
    conflict = grade.claims[0]
    assert _read(conflict) == ("unverifiable", None, True, None, "none")
    assert conflict.decided_by == 2, "the repair decided it"


async def test_a_restatement_takes_its_originals_conflict_outcome() -> None:
    claims = _claims(_LIVES, "Ann Lee lives in York.", restates={1: 0})
    client = _FakeScriptedClient(
        [_batch(_answer(0, _LIVES_SUPPORT, _BORN_MISREAD)), _reasked((0, [_stance(_LIVES_SUPPORT, "supports")]))]
    )

    grade = _grade(claims, await _verify(client, claims, key=_LEE_KEY))

    assert [(_read(claim), claim.decided_by) for claim in grade.claims] == [
        (("supported", "case", False, None, "none"), 1)
    ] * 2


@pytest.mark.parametrize(
    ("replies", "raised", "loss"),
    [
        ([RuntimeError("the provider refused the call")], RuntimeError, None),
        ([_Reply(content='{"claims": [', stop_reason="max_tokens")], ClaimGradingFailed, "cut_short"),
        ([_Reply(content="not json")] * 3, ClaimGradingFailed, "unreadable"),
    ],
    ids=["provider", "cut-short", "unreadable"],
)
async def test_a_conflicts_reask_that_fails_is_the_callers_loss_never_a_verdict(
    replies: list[_Reply | Exception], raised: type[Exception], loss: str | None
) -> None:
    claims = _claims(_LIVES)
    client = _FakeScriptedClient([_batch(_answer(0, _LIVES_SUPPORT, _BORN_MISREAD)), *replies])

    with pytest.raises(raised) as failed:
        await _verify(client, claims, key=_LEE_KEY)

    if loss is not None:
        assert isinstance(failed.value, ClaimGradingFailed) and failed.value.loss == loss


# --- what it costs ---------------------------------------------------------------------------------------------------


async def test_a_cell_with_a_conflict_and_no_doubted_claim_makes_one_reask_call_where_it_made_none() -> None:
    claims = _claims(_LIVES)
    client = _FakeScriptedClient([_batch(_answer(0, _LIVES_SUPPORT, _BORN_MISREAD)), _reasked((0, [_LIVES_SUPPORT]))])

    await _verify(client, claims, key=_LEE_KEY)

    assert [_schema(call) for call in client.calls] == ["claim_truth", "claim_reask"]


async def test_the_price_ceiling_bounds_the_reask_of_a_conflict_between_case_passages_with_no_key() -> None:
    """A cell whose key states nothing can still hold a conflict between case documents: its re-ask is priced, asked in
    full up to its attempts, at the truth step's evidence."""
    repairs = RepairBudget(rounds=0, max_items_per_cell=0)
    plain = _UPDATE.model_copy(update={"overrides": ()})
    segments = (Segment(text="A loan runs two weeks.", unit=0),)
    priced = ClaimVerifier(client=_FakeBytePricedClient([]), pages=_WHOLE, repairs=repairs)
    ceiling = priced.price_ceiling(segments, key=None, documents=(_POLICY, plain))
    bare = priced.price_ceiling(segments, key=None, documents=())
    client = _FakeScriptedClient(
        [
            _batch(_answer(0, _TWO_WEEKS, _stance(_THREE_WEEKS, "contradicts"))),
            *[_Reply(content="not json")] * 2,
            _reasked((0, [_TWO_WEEKS])),
        ]
    )

    await _verify(client, _claims("A loan runs two weeks."), key=None, documents=(_POLICY, plain), repairs=repairs)

    reasks = [call for call in client.calls if _schema(call) == "claim_reask"]
    spent = sum((len(call["system"].encode()) + len(call["user"].encode())) * 1e-6 for call in reasks)
    assert ceiling is not None and bare is not None
    assert len(reasks) == 3 and spent <= ceiling - bare


# --- compound claims: a support states every value --------------------------------------------------------------------

_CONNOR = "Dace Connor"


async def test_the_every_value_rule_reaches_the_truth_its_repair_and_the_reask_prompts() -> None:
    claims = _claims(_LIVES, "Bo Finch lives in Ely.")
    key = _key(("Ann Lee", _LIVES), ("Ann Lee", _BORN), ("Bo Finch", "Bo Finch lives in Hull."))
    client = _FakeScriptedClient(
        [
            _batch(_answer(0, _LIVES_SUPPORT, _BORN_MISREAD), _answer(1, suspect=True)),
            _reasked((0, [_cite("K3", quote="x", subject="x", value="x")]), (1, [])),
            _repair(),
        ]
    )

    await _verify(client, claims, key=key)

    rule = "states every value the claim states"
    assert [rule in call["system"] for call in client.calls] == [True, True, True]
    assert [_schema(call) for call in client.calls] == ["claim_truth", "claim_reask", "claim_truth_repair"]


_MINISTRY = f"{_CONNOR} was in Ironhall and performed drums and vocals on Mind Lathe."
_KEYBOARDS_AND_SANG = f"{_CONNOR} played keyboards and sang in Ironhall (Mind Lathe era)."


@pytest.mark.parametrize(
    ("replies", "expected"),
    [
        (
            [
                _batch(
                    _answer(
                        0, _cite("K1", "contradicts", quote=_KEYBOARDS_AND_SANG, subject=_CONNOR, value="keyboards")
                    )
                )
            ],
            ("contradicted", "case", False, None, "none"),
        ),
        ([_batch(_answer(0))], ("unverifiable", None, False, None, "none")),
        (
            [
                _batch(_answer(0, suspect=True)),
                _reasked(
                    (0, [_cite("K1", "contradicts", quote=_KEYBOARDS_AND_SANG, subject=_CONNOR, value="keyboards")])
                ),
            ],
            ("contradicted", "case", False, None, "none"),
        ),
    ],
    ids=["cited-as-a-contradiction", "cited-for-nothing", "doubted-then-contradicted-in-the-reask"],
)
async def test_a_compound_whose_true_half_a_fact_states_reads_anything_but_supported(
    replies: list[_Reply], expected: tuple[Any, ...]
) -> None:
    """The fact states the vocals and another instrument: under the rule it is never cited as support for drums and
    vocals, and each reading of a reply that obeys it is something other than supported."""
    claims = _claims(_MINISTRY)
    client = _FakeScriptedClient(replies)

    (claim,) = _grade(claims, await _verify(client, claims, key=_key((_CONNOR, _KEYBOARDS_AND_SANG)))).claims

    assert _read(claim) == expected


async def test_a_compound_one_fact_states_whole_reads_supported() -> None:
    fact = f"{_CONNOR} played keyboards and sang in Ironhall."
    claims = _claims(fact)
    client = _FakeScriptedClient(
        [_batch(_answer(0, _cite("K1", quote=fact, subject=_CONNOR, value="keyboards and sang")))]
    )

    (claim,) = _grade(claims, await _verify(client, claims, key=_key((_CONNOR, fact)))).claims

    assert _read(claim) == ("supported", "case", False, None, "none")


async def test_a_compound_two_facts_state_one_value_each_reads_supported_with_no_conflict() -> None:
    keyboards, sang = f"{_CONNOR} played keyboards in Ironhall.", f"{_CONNOR} sang in Ironhall."
    claims = _claims(f"{_CONNOR} played keyboards and sang in Ironhall.")
    client = _FakeScriptedClient(
        [
            _batch(
                _answer(
                    0,
                    _cite("K1", quote=keyboards, subject=_CONNOR, value="keyboards"),
                    _cite("K2", quote=sang, subject=_CONNOR, value="sang"),
                )
            )
        ]
    )

    (claim,) = _grade(claims, await _verify(client, claims, key=_key((_CONNOR, keyboards), (_CONNOR, sang)))).claims

    assert _read(claim) == ("supported", "case", False, None, "none")
    assert len(client.calls) == 1, "nothing conflicts, so nothing is asked again"


async def test_a_joint_value_reads_supported_as_one_value() -> None:
    fact = "Low Tide was written by Ivo Marsh and Tess Holt."
    claims = _claims("Ivo Marsh and Tess Holt wrote Low Tide.")
    client = _FakeScriptedClient(
        [_batch(_answer(0, _cite("K1", quote=fact, subject="Low Tide", value="Ivo Marsh and Tess Holt")))]
    )

    (claim,) = _grade(claims, await _verify(client, claims, key=_key(("Low Tide", fact)))).claims

    assert _read(claim) == ("supported", "case", False, None, "none")


async def test_a_compound_with_one_value_contradicted_and_one_supported_is_a_conflict_and_takes_the_reask() -> None:
    keyboards, sang = f"{_CONNOR} played keyboards in Ironhall.", f"{_CONNOR} sang in Ironhall."
    claims = _claims(f"{_CONNOR} played drums and sang in Ironhall.")
    against = _cite("K1", "contradicts", quote=keyboards, subject=_CONNOR, value="keyboards")
    client = _FakeScriptedClient(
        [
            _batch(_answer(0, against, _cite("K2", quote=sang, subject=_CONNOR, value="sang"))),
            _reasked((0, [against])),
        ]
    )

    (claim,) = _grade(claims, await _verify(client, claims, key=_key((_CONNOR, keyboards), (_CONNOR, sang)))).claims

    assert [_schema(call) for call in client.calls] == ["claim_truth", "claim_reask"]
    assert "Its evidence: K1, K2" in client.calls[1]["user"]
    assert _read(claim) == ("contradicted", "case", False, None, "none")


# --- the stored shape ------------------------------------------------------------------------------------------------


def _graded(**changes: Any) -> GradedClaim:
    fields: dict[str, Any] = {
        "segment": 0,
        "unit": 0,
        "claim": _LIVES,
        "excerpt": "Ann Lee lives in York",
        "start": 0,
        "end": 21,
        "scope": "world",
        "specific": True,
        "hedged": False,
        "attributed": False,
        "truth_citations": (
            ClaimCitation(ref="K1", stance="supports"),
            ClaimCitation(ref="K2", stance="contradicts"),
        ),
        "verdict": "unverifiable",
        "graded_against": None,
        "evidence_conflict": True,
        "decided_by": 1,
        **changes,
    }
    return GradedClaim(**fields)


def test_a_stored_claim_holds_a_conflict_as_unverifiable_or_as_a_grade_stored_before_read_it_supported() -> None:
    assert _graded().evidence_conflict
    assert _graded(verdict="supported", graded_against="case").evidence_conflict, "a grade stored before still reads"
    with pytest.raises(ValidationError, match="conflict reads supported .* or unverifiable, not 'contradicted'"):
        _graded(verdict="contradicted", graded_against="case")
    with pytest.raises(ValidationError, match="conflict reads supported .* or unverifiable, not 'true_ungrounded'"):
        _graded(verdict="true_ungrounded", graded_against="web", suspect=True, nominated_by="judge", web="checked")


def _counts(**changes: Any) -> ClaimJudgeCounts:
    """One asserted claim, unverifiable, its case evidence in conflict."""
    zero = dict.fromkeys(
        (
            "hedged",
            "attributed",
            "contradicted",
            "asserted_contradicted",
            "hedged_contradicted",
            "attributed_contradicted",
            "grounded_specific",
            "true_ungrounded",
            "web_decided",
            "web_checked",
            "web_skipped",
        ),
        0,
    )
    fields: dict[str, Any] = {
        "served_model": None,
        "judge_temperature": None,
        "judge_sent_settings": None,
        **zero,
        "claims": 1,
        "asserted": 1,
        "asserted_specific": 1,
        "unverifiable": 1,
        "evidence_conflicts": 1,
        **changes,
    }
    return ClaimJudgeCounts(**fields)


def test_a_tally_counts_a_conflict_among_the_unverifiable_claims_and_never_beyond_them_and_the_supported() -> None:
    assert _counts().evidence_conflicts == 1
    with pytest.raises(ValidationError, match="do not nest: evidence_conflicts"):
        _counts(evidence_conflicts=2)
    with pytest.raises(ValidationError, match="do not nest: evidence_conflicts"):
        _counts(unverifiable=0, contradicted=1, asserted_contradicted=1)
