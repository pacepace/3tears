"""The claim extractor: the step that lists the factual claims a cell's segments state, on a scripted client.

Every behaviour is driven through a client that answers from a script and records what it was sent, so nothing is
spent and each reply is chosen. The claims come back ordered by segment, each excerpt the segment's own slice at
offsets the verbatim check found, ``restates`` resolved to the first claim of its chain and renumbered with the order.
Repair, not refusal: a reply that will not parse is asked again in full naming why; an item that fails a check (an
excerpt not in its segment, a ``restates`` pointing forward or nowhere) is asked again alone with its failure named,
for two rounds, then the extraction raises. Every claim carries a scope, ``world`` or ``session``: a back-announce
line splits into a session claim and a world claim, and a ``restates`` crossing scopes is repaired; a cut-short reply raises at once. The failure is ``ClaimGradingFailed``,
no ``ApparatusError``. Every completion is billed to the ``judge`` ledger. A cell's units are numbered 0..n-1, each
once, or the cell is refused before any call; a segment's tags never reach a prompt. The identity names the model,
moves with the host's guidance, and is pinned for the shipped prompts, schemas, fold and budgets. The text is
invented: a walking-tour guide's lines about made-up places.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest
from pydantic import ValidationError

from threetears.evals.kernel import RoleUsageLedger
from threetears.evals.kernel.claims.extraction import ClaimExtractor, ExtractedClaim
from threetears.evals.kernel.claims.failures import ClaimGradingFailed
from threetears.evals.kernel.host import ApparatusError
from threetears.evals.schema import Segment

_MODEL = "scripted/claims"

_CELL = (
    Segment(text="Welcome aboard! Today we walk the old harbour."),
    Segment(text="First stop: the Varn Bridge, built in 1894 by Hela Orsk. It’s a beauty.", unit=0),
    Segment(text="Next, the mill. Hela Orsk built this too, I believe, around 1901.", unit=1),
)


@dataclass(frozen=True)
class _Reply:
    """One completion, shaped as the engine's completion port describes it."""

    content: str
    stop_reason: str = "end_turn"
    model: str = _MODEL
    served_model: str | None = _MODEL
    input_tokens: int | None = 300
    output_tokens: int | None = 80
    reasoning_tokens: int | None = None
    cost_usd: float | None = 0.002
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
        raise AssertionError("an unnamed extractor is refused before any call")

    async def aclose(self) -> None:
        """Nothing to release."""

    async def __aenter__(self) -> _FakeUnnamedClient:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self.aclose()


def _item(
    segment: int | None,
    claim: str,
    excerpt: str,
    *,
    specific: bool = True,
    hedged: bool = False,
    attributed: bool = False,
    scope: str = "world",
    restates: int | None = None,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "claim": claim,
        "excerpt": excerpt,
        "specific": specific,
        "hedged": hedged,
        "attributed": attributed,
        "scope": scope,
        "restates": restates,
    }
    if segment is not None:
        item["segment"] = segment
    return item


def _listing(*items: dict[str, Any]) -> _Reply:
    return _Reply(content=json.dumps({"claims": list(items)}))


def _repair(claim: str, excerpt: str, **tags: Any) -> _Reply:
    return _Reply(content=json.dumps(_item(None, claim, excerpt, **tags)))


_BRIDGE_YEAR = _item(1, "The Varn Bridge was built in 1894.", "the Varn Bridge, built in 1894")
_BRIDGE_BUILDER = _item(1, "Hela Orsk built the Varn Bridge.", "built in 1894 by Hela Orsk")
_MILL_BUILDER = _item(2, "Hela Orsk built the mill.", "Hela Orsk built this too", hedged=True)
_MILL_YEAR = _item(2, "The mill was built around 1901.", "around 1901", hedged=True)


def _extractor(client: _FakeScriptedClient, **options: Any) -> ClaimExtractor:
    return ClaimExtractor(client=client, **options)


async def test_the_claims_come_back_checked_with_each_excerpt_the_segments_own_slice() -> None:
    client = _FakeScriptedClient(
        [
            _listing(
                _BRIDGE_YEAR,
                # The model straightens the apostrophe and changes case; the kept excerpt is the segment's.
                _item(1, "The guide likes the bridge.", "it's a BEAUTY", specific=False),
                _MILL_BUILDER,
            )
        ]
    )

    claims = await _extractor(client).extract(_CELL, ledger=RoleUsageLedger(role="judge"))

    text = _CELL[1].text
    start = text.index("It’s")
    assert claims[1] == ExtractedClaim(
        segment=1,
        unit=0,
        claim="The guide likes the bridge.",
        excerpt="It’s a beauty",
        start=start,
        end=start + len("It’s a beauty"),
        specific=False,
        hedged=False,
        attributed=False,
        scope="world",
        restates=None,
    )
    assert [claim.unit for claim in claims] == [0, 0, 1]
    assert [claim.hedged for claim in claims] == [False, False, True]
    for claim in claims:
        assert _CELL[claim.segment].text[claim.start : claim.end] == claim.excerpt
    (call,) = client.calls
    for index, segment in enumerate(_CELL):
        assert f"Segment {index}:\n{segment.text}" in call["user"]
    schema = call["response_format"]["json_schema"]
    assert schema["strict"] is True
    item = schema["schema"]["properties"]["claims"]["items"]
    assert item["required"] == [
        "segment",
        "claim",
        "excerpt",
        "specific",
        "hedged",
        "attributed",
        "scope",
        "restates",
    ]
    assert item["properties"]["scope"]["enum"] == ["world", "session"]


async def test_a_rumour_comes_back_attributed() -> None:
    cell = (Segment(text="Some say the Varn Bridge was first planned in stone.", unit=0),)
    client = _FakeScriptedClient(
        [_listing(_item(0, "The Varn Bridge was first planned in stone.", "first planned in stone", attributed=True))]
    )

    (claim,) = await _extractor(client).extract(cell, ledger=RoleUsageLedger(role="judge"))

    assert (claim.attributed, claim.hedged) == (True, False)
    assert "attributed: true when the text reports the claim without vouching for it" in client.calls[0]["system"]


async def test_a_statement_backed_by_a_source_comes_back_asserted() -> None:
    cell = (Segment(text="According to the harbour records, the Varn Bridge opened in 1894.", unit=0),)
    client = _FakeScriptedClient(
        [_listing(_item(0, "The Varn Bridge opened in 1894.", "the Varn Bridge opened in 1894", attributed=False))]
    )

    (claim,) = await _extractor(client).extract(cell, ledger=RoleUsageLedger(role="judge"))

    assert (claim.attributed, claim.hedged) == (False, False)
    assert "backs with a source" in client.calls[0]["system"]
    assert "is the text's own and not attributed" in client.calls[0]["system"]


async def test_an_instruction_that_states_a_rule_carries_its_claim() -> None:
    cell = (Segment(text="To cross at high tide, take the upper path: the lower one floods.", unit=0),)
    client = _FakeScriptedClient([_listing(_item(0, "The lower path floods at high tide.", "the lower one floods"))])

    (claim,) = await _extractor(client).extract(cell, ledger=RoleUsageLedger(role="judge"))

    assert claim.scope == "world"
    system = client.calls[0]["system"]
    assert "a bare instruction with no factual content" in system
    assert "An instruction or how-to that states a rule or a fact" in system


async def test_a_back_announce_splits_into_a_session_claim_and_a_world_claim() -> None:
    cell = (Segment(text="That was the Varn Bridge, from Hela Orsk's harbour plan. Next up, the mill.", unit=0),)
    client = _FakeScriptedClient(
        [
            _listing(
                _item(0, "The tour just visited the Varn Bridge.", "That was the Varn Bridge", scope="session"),
                _item(
                    0,
                    "The Varn Bridge is part of Hela Orsk's harbour plan.",
                    "the Varn Bridge, from Hela Orsk's harbour plan",
                ),
                _item(0, "The mill is the tour's next stop.", "Next up, the mill", scope="session"),
            )
        ]
    )

    claims = await _extractor(client).extract(cell, ledger=RoleUsageLedger(role="judge"))

    assert [(claim.scope, claim.excerpt) for claim in claims] == [
        ("session", "That was the Varn Bridge"),
        ("world", "the Varn Bridge, from Hela Orsk's harbour plan"),
        ("session", "Next up, the mill"),
    ]
    system = client.calls[0]["system"]
    assert "about this output or session itself" in system
    assert "A sentence that does both is split" in system
    assert len(client.calls) == 1


async def test_a_restates_crossing_scopes_is_asked_again() -> None:
    cell = (
        Segment(text="That was the Varn Bridge.", unit=0),
        Segment(text="The Varn Bridge, as we just saw, spans the harbour.", unit=1),
    )
    client = _FakeScriptedClient(
        [
            _listing(
                _item(0, "The tour just visited the Varn Bridge.", "That was the Varn Bridge", scope="session"),
                _item(1, "The tour just visited the Varn Bridge.", "as we just saw", scope="session", restates=0),
                _item(1, "The Varn Bridge spans the harbour.", "spans the harbour", restates=0),
            ),
            _repair("The Varn Bridge spans the harbour.", "spans the harbour", restates=None),
        ]
    )

    claims = await _extractor(client).extract(cell, ledger=RoleUsageLedger(role="judge"))

    assert [(claim.scope, claim.restates) for claim in claims] == [("session", None), ("session", 0), ("world", None)]
    repair = client.calls[1]
    assert "position 2" in repair["user"]
    assert "its restates names claim 0, a session claim, and this one is a world claim" in repair["user"]


async def test_a_repair_that_changes_a_targets_scope_sends_its_restatement_to_repair() -> None:
    cell = (
        Segment(text="That was the Varn Bridge.", unit=0),
        Segment(text="The Varn Bridge, the one we just crossed, opened in 1894.", unit=1),
    )
    client = _FakeScriptedClient(
        [
            _listing(
                # Claim 0 fails on its excerpt; claim 1 restates it, both world claims, so claim 1 passes at first.
                _item(0, "The tour just visited the Varn Bridge.", "That was the bridge"),
                _item(1, "The tour just visited the Varn Bridge.", "the one we just crossed", restates=0),
                _item(1, "The Varn Bridge opened in 1894.", "opened in 1894"),
            ),
            # The repair fixes the excerpt and corrects the scope to session: claim 1 now restates across scopes.
            _repair("The tour just visited the Varn Bridge.", "That was the Varn Bridge", scope="session"),
            _repair("The tour just crossed the Varn Bridge.", "the one we just crossed", scope="session", restates=0),
        ]
    )

    claims = await _extractor(client).extract(cell, ledger=RoleUsageLedger(role="judge"))

    assert [(claim.scope, claim.restates) for claim in claims] == [("session", None), ("session", 0), ("world", None)]
    assert len(client.calls) == 3
    second_round = client.calls[2]["user"]
    assert "position 1" in second_round
    assert "its restates names claim 0, a session claim, and this one is a world claim" in second_round


async def test_a_segments_tags_never_reach_a_prompt() -> None:
    cell = (Segment(text="The Varn Bridge was built in 1894.", unit=0, tags={"entity": "landmark-zq7"}),)
    client = _FakeScriptedClient(
        [
            _listing(_item(0, "The Varn Bridge was built in 1894.", "built in 1893")),
            _repair("The Varn Bridge was built in 1894.", "built in 1894"),
        ]
    )

    await _extractor(client).extract(cell, ledger=RoleUsageLedger(role="judge"))

    assert len(client.calls) == 2
    assert not any("landmark-zq7" in call["system"] + call["user"] for call in client.calls)


async def test_the_claims_are_ordered_by_segment_and_restates_is_renumbered_with_them() -> None:
    client = _FakeScriptedClient(
        [
            _listing(
                _MILL_BUILDER,
                _BRIDGE_BUILDER,
                # The mill line restates the builder, listed out of order: position 1 appears first.
                _item(2, "Hela Orsk built the mill.", "Hela Orsk built this too", restates=1),
            )
        ]
    )

    claims = await _extractor(client).extract(_CELL, ledger=RoleUsageLedger(role="judge"))

    assert [claim.segment for claim in claims] == [1, 2, 2]
    assert claims[0].claim == "Hela Orsk built the Varn Bridge."
    assert claims[2].restates == 0
    assert len(client.calls) == 1, "an order the model chose is no failure: nothing is asked again"


async def test_a_restates_chain_is_resolved_to_its_first_claim() -> None:
    client = _FakeScriptedClient(
        [
            _listing(
                _BRIDGE_BUILDER,
                _item(2, "Hela Orsk built the Varn Bridge.", "Hela Orsk built this too", restates=0),
                _item(2, "Hela Orsk built the Varn Bridge.", "Hela Orsk", restates=1),
            )
        ]
    )

    claims = await _extractor(client).extract(_CELL, ledger=RoleUsageLedger(role="judge"))

    assert [claim.restates for claim in claims] == [None, 0, 0]


async def test_an_excerpt_not_in_its_segment_is_asked_again_alone_with_the_failure_named() -> None:
    client = _FakeScriptedClient(
        [
            _listing(_BRIDGE_YEAR, _item(2, "Hela Orsk built the mill.", "Orsk designed the mill"), _MILL_YEAR),
            _repair("Hela Orsk built the mill.", "Hela Orsk built this too", hedged=True),
        ]
    )
    ledger = RoleUsageLedger(role="judge")

    claims = await _extractor(client).extract(_CELL, ledger=ledger)

    assert [claim.excerpt for claim in claims] == [
        "the Varn Bridge, built in 1894",
        "Hela Orsk built this too",
        "around 1901",
    ]
    listing, repair = client.calls
    assert repair["system"] != listing["system"]
    assert "position 1" in repair["user"]
    assert "Orsk designed the mill" in repair["user"]
    assert "its excerpt is not in segment 2" in repair["user"]
    assert _CELL[2].text in repair["user"]
    assert _CELL[1].text not in repair["user"], "a failing item is asked about alone, with its own segment"
    assert "0: The Varn Bridge was built in 1894." in repair["user"], "the claims listed before it are shown"
    assert "segment" not in repair["response_format"]["json_schema"]["schema"]["properties"]
    (row,) = ledger.rows()
    assert row.call_count == 2


async def test_an_elided_excerpt_is_asked_again_as_one_not_in_its_segment() -> None:
    client = _FakeScriptedClient(
        [
            _listing(_item(1, "Hela Orsk built the Varn Bridge.", "the Varn Bridge… by Hela Orsk")),
            _repair("Hela Orsk built the Varn Bridge.", "built in 1894 by Hela Orsk"),
        ]
    )

    (claim,) = await _extractor(client).extract(_CELL, ledger=RoleUsageLedger(role="judge"))

    assert claim.excerpt == "built in 1894 by Hela Orsk"
    assert "its excerpt is not in segment 1; an excerpt is one unbroken stretch" in client.calls[1]["user"]


async def test_an_excerpt_through_an_ellipsis_the_text_holds_is_kept() -> None:
    cell = (Segment(text="The bridge opened… in 1894, late.", unit=0),)
    client = _FakeScriptedClient([_listing(_item(0, "The bridge opened in 1894.", "opened... in 1894"))])

    (claim,) = await _extractor(client).extract(cell, ledger=RoleUsageLedger(role="judge"))

    assert claim.excerpt == "opened… in 1894"
    assert len(client.calls) == 1


async def test_a_restates_pointing_forward_is_asked_again_and_the_correction_is_kept() -> None:
    client = _FakeScriptedClient(
        [
            _listing(
                _item(1, "Hela Orsk built the Varn Bridge.", "built in 1894 by Hela Orsk", restates=1), _MILL_BUILDER
            ),
            _repair("Hela Orsk built the Varn Bridge.", "built in 1894 by Hela Orsk", restates=None),
        ]
    )

    claims = await _extractor(client).extract(_CELL, ledger=RoleUsageLedger(role="judge"))

    assert [claim.restates for claim in claims] == [None, None]
    repair = client.calls[1]
    assert "its restates names claim 1, which appears after it" in repair["user"]
    assert "position 0" in repair["user"]
    assert "# Claims listed before it\n\nnone" in repair["user"]


@pytest.mark.parametrize("restates", [7, -1, 0])
async def test_a_restates_naming_no_other_claim_is_asked_again(restates: int) -> None:
    client = _FakeScriptedClient(
        [
            _listing(_item(1, "Hela Orsk built the Varn Bridge.", "built in 1894 by Hela Orsk", restates=restates)),
            _repair("Hela Orsk built the Varn Bridge.", "built in 1894 by Hela Orsk"),
        ]
    )

    (claim,) = await _extractor(client).extract(_CELL, ledger=RoleUsageLedger(role="judge"))

    assert claim.restates is None
    assert f"its restates names claim {restates}, and the list has no such other claim" in client.calls[1]["user"]


async def test_an_item_still_failing_after_two_repair_rounds_raises_naming_it_and_every_call_is_billed() -> None:
    wrong = _item(2, "Hela Orsk built the mill.", "Orsk designed the mill")
    client = _FakeScriptedClient(
        [
            _listing(_BRIDGE_YEAR, wrong),
            _repair("Hela Orsk built the mill.", "Orsk designed the mill"),
            _Reply(content="not json"),
        ]
    )
    ledger = RoleUsageLedger(role="judge")

    with pytest.raises(ClaimGradingFailed, match=r"could not repair 1 claim\(s\) in 2 rounds: claim 1: its excerpt"):
        await _extractor(client).extract(_CELL, ledger=ledger)

    assert len(client.calls) == 3
    assert "the last correction could not be used" not in client.calls[2]["user"]
    (row,) = ledger.rows()
    assert row.call_count == 3
    assert row.cost_usd == pytest.approx(0.006)


async def test_an_unreadable_repair_reply_is_named_in_the_next_round() -> None:
    client = _FakeScriptedClient(
        [
            _listing(_item(2, "Hela Orsk built the mill.", "Orsk designed the mill")),
            _Reply(content="not json"),
            _repair("Hela Orsk built the mill.", "Hela Orsk built this too"),
        ]
    )

    (claim,) = await _extractor(client).extract(_CELL, ledger=RoleUsageLedger(role="judge"))

    assert claim.excerpt == "Hela Orsk built this too"
    assert "the last correction could not be used" in client.calls[2]["user"]


async def test_an_unreadable_reply_is_asked_again_in_full_naming_why_then_refused() -> None:
    unreadable = [
        _Reply(content="not json"),
        _listing({"segment": 1, "claim": "The Varn Bridge was built in 1894."}),  # the tags are missing
        _listing(_item(5, "The Varn Bridge was built in 1894.", "built in 1894")),  # no such segment
    ]
    client = _FakeScriptedClient(unreadable)
    ledger = RoleUsageLedger(role="judge")

    with pytest.raises(ClaimGradingFailed, match="did not parse in 3 attempts"):
        await _extractor(client).extract(_CELL, ledger=ledger)

    first, second, third = client.calls
    assert "# Your last reply" not in first["user"]
    assert "It could not be used: No valid JSON object" in second["user"]
    assert "It could not be used: it is not the shape the response format states" in third["user"]
    assert all(call["system"] == first["system"] for call in client.calls)
    (row,) = ledger.rows()
    assert row.call_count == 3


async def test_a_reply_naming_a_segment_the_cell_lacks_is_asked_again_in_full() -> None:
    client = _FakeScriptedClient(
        [_listing(_item(3, "The Varn Bridge was built in 1894.", "built in 1894")), _listing(_BRIDGE_YEAR)]
    )

    (claim,) = await _extractor(client).extract(_CELL, ledger=RoleUsageLedger(role="judge"))

    assert claim.segment == 1
    assert "claim 0 names segment 3, and the segments are 0 to 2" in client.calls[1]["user"]


async def test_a_reply_cut_short_is_refused_at_once_and_still_billed() -> None:
    client = _FakeScriptedClient([_Reply(content='{"claims": [', stop_reason="max_tokens")])
    ledger = RoleUsageLedger(role="judge")

    with pytest.raises(ClaimGradingFailed, match="cut short"):
        await _extractor(client).extract(_CELL, ledger=ledger)

    assert len(client.calls) == 1
    (row,) = ledger.rows()
    assert row.call_count == 1


async def test_a_repair_reply_cut_short_is_refused_at_once_and_still_billed() -> None:
    client = _FakeScriptedClient(
        [
            _listing(_item(2, "Hela Orsk built the mill.", "Orsk designed the mill")),
            _Reply(content='{"claim": "Hela', stop_reason="max_tokens"),
        ]
    )
    ledger = RoleUsageLedger(role="judge")

    with pytest.raises(ClaimGradingFailed, match="cut short"):
        await _extractor(client).extract(_CELL, ledger=ledger)

    assert len(client.calls) == 2
    (row,) = ledger.rows()
    assert row.call_count == 2


@pytest.mark.parametrize(
    "units",
    [
        pytest.param((1,), id="not-from-0"),
        pytest.param((0, 0), id="repeated"),
        pytest.param((0, 2), id="a-gap"),
        pytest.param((1, 2, 3), id="shifted"),
    ],
)
async def test_units_not_numbered_0_to_n_minus_1_each_once_are_refused_before_any_call(units: tuple[int, ...]) -> None:
    client = _FakeScriptedClient([])
    cell = (Segment(text="Hello."), *(Segment(text=f"Unit {unit}.", unit=unit) for unit in units))

    with pytest.raises(ValueError, match="numbered 0..n-1, each once"):
        await _extractor(client).extract(cell, ledger=RoleUsageLedger(role="judge"))
    assert client.calls == []


async def test_units_in_any_order_with_other_segments_between_are_accepted() -> None:
    client = _FakeScriptedClient([_listing()])
    cell = (
        Segment(text="A preamble."),
        Segment(text="Later.", unit=1),
        Segment(text="An aside."),
        Segment(text="Earlier.", unit=0),
    )

    assert await _extractor(client).extract(cell, ledger=RoleUsageLedger(role="judge")) == ()
    assert len(client.calls) == 1


@pytest.mark.parametrize(("text", "unit"), [("", None), ("  \n", None), ("Fine.", -1)])
def test_a_blank_segment_or_a_negative_unit_is_refused(text: str, unit: int | None) -> None:
    with pytest.raises(ValidationError):
        Segment(text=text, unit=unit)


async def test_no_segments_make_no_call() -> None:
    client = _FakeScriptedClient([])
    ledger = RoleUsageLedger(role="judge")

    assert await _extractor(client).extract((), ledger=ledger) == ()
    assert client.calls == []
    assert ledger.rows() == []


async def test_a_ledger_of_another_role_is_refused() -> None:
    client = _FakeScriptedClient([])

    with pytest.raises(ValueError, match="'judge' role"):
        await _extractor(client).extract(_CELL, ledger=RoleUsageLedger(role="candidate"))
    assert client.calls == []


def test_a_client_that_names_no_model_is_refused() -> None:
    with pytest.raises(ValueError, match="names its model"):
        ClaimExtractor(client=_FakeUnnamedClient())


async def test_the_hosts_guidance_reaches_both_prompts() -> None:
    client = _FakeScriptedClient(
        [
            _listing(_item(2, "Hela Orsk built the mill.", "Orsk designed the mill")),
            _repair("Hela Orsk built the mill.", "Hela Orsk built this too"),
        ]
    )
    guidance = "A dated restoration counts as a claim."

    await _extractor(client, guidance=guidance).extract(_CELL, ledger=RoleUsageLedger(role="judge"))

    assert all(call["system"].endswith(guidance) for call in client.calls)


#: The identity of :func:`_extractor` over a client of :data:`_MODEL`, from the shipped prompts, schemas, fold and
#: budgets. Produced by ``ClaimExtractor.identity`` itself; it moves with any change to them, and with the fold's
#: tables or Unicode version, which is a new definition: write the new value here deliberately.
_PINNED_IDENTITY = "scripted/claims|claims:b326c9108c71"


def test_the_identity_names_the_model_and_moves_with_the_guidance() -> None:
    client = _FakeScriptedClient([])
    base = _extractor(client)

    assert base.identity == _PINNED_IDENTITY
    assert base.identity == _extractor(_FakeScriptedClient([])).identity
    others = {
        _extractor(client, guidance="A dated restoration counts as a claim.").identity,
        _extractor(client, guidance="A named architect counts as a claim.").identity,
        _extractor(_FakeScriptedClient([], model_name="scripted/other")).identity,
    }
    assert base.identity not in others
    assert len(others) == 3
    assert _extractor(client, guidance="  ").identity == base.identity, "blank guidance is no guidance"


def test_a_failed_grading_is_no_fault_of_the_rig() -> None:
    assert not issubclass(ClaimGradingFailed, ApparatusError)
