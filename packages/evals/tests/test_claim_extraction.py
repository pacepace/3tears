"""The claim extractor: the model step that lists the factual claims a cell's utterances assert, on a scripted client.

Every behaviour is driven through a client that answers from a script and records what it was sent, so nothing is
spent and each reply is chosen. The claims come back ordered by utterance, each excerpt the utterance's own slice at
offsets the verbatim check found, ``restates`` renumbered with the order. Repair, not refusal: a reply that will not
parse is asked again in full naming why; an item that fails a check (an excerpt not in its utterance or elided, a
``restates`` pointing forward or nowhere) is asked again alone with its failure named, for two rounds, then the
extraction raises; a cut-short reply raises at once. Every completion is billed to the ``judge`` ledger. The cell's
stops are numbered 0..n-1, each once, or the cell is refused before any call. The identity names the model and moves
with the host's segmentation and guidance, the prompts they enter, and is pinned for the shipped prompts, schemas and
fold. The utterances are invented: a walking-tour guide's lines about made-up places.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest
from pydantic import ValidationError

from threetears.evals.kernel import (
    ClaimExtractionFailed,
    ClaimExtractor,
    ExtractedClaim,
    RoleUsageLedger,
    Utterance,
)
from threetears.evals.kernel.host import ApparatusError

_MODEL = "scripted/claims"

_SEGMENTATION = "tour-stops-v1"

_CELL = (
    Utterance(text="Welcome aboard! Today we walk the old harbour."),
    Utterance(text="First stop: the Varn Bridge, built in 1894 by Hela Orsk. It’s a beauty.", stop_index=0),
    Utterance(text="Next, the mill. Hela Orsk built this too, I believe, around 1901.", stop_index=1),
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
    utterance: int | None,
    claim: str,
    excerpt: str,
    *,
    specific: bool = True,
    hedged: bool = False,
    restates: int | None = None,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "claim": claim,
        "excerpt": excerpt,
        "specific": specific,
        "hedged": hedged,
        "restates": restates,
    }
    if utterance is not None:
        item["utterance"] = utterance
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
    return ClaimExtractor(client=client, segmentation=_SEGMENTATION, **options)


async def test_the_claims_come_back_checked_with_each_excerpt_the_utterances_own_slice() -> None:
    client = _FakeScriptedClient(
        [
            _listing(
                _BRIDGE_YEAR,
                # The model straightens the apostrophe and changes case; the stored excerpt is the utterance's.
                _item(1, "The guide likes the bridge.", "it's a BEAUTY", specific=False),
                _MILL_BUILDER,
            )
        ]
    )

    claims = await _extractor(client).extract(_CELL, ledger=RoleUsageLedger(role="judge"))

    stop_text = _CELL[1].text
    start = stop_text.index("It’s")
    assert claims[1] == ExtractedClaim(
        utterance=1,
        stop_index=0,
        claim="The guide likes the bridge.",
        excerpt="It’s a beauty",
        start=start,
        end=start + len("It’s a beauty"),
        specific=False,
        hedged=False,
        restates=None,
    )
    assert [claim.stop_index for claim in claims] == [0, 0, 1]
    assert [claim.hedged for claim in claims] == [False, False, True]
    for claim in claims:
        text = _CELL[claim.utterance].text
        assert text[claim.start : claim.end] == claim.excerpt
    (call,) = client.calls
    for index, utterance in enumerate(_CELL):
        assert f"U{index}:\n{utterance.text}" in call["user"]
    schema = call["response_format"]["json_schema"]
    assert schema["strict"] is True
    item = schema["schema"]["properties"]["claims"]["items"]
    assert item["required"] == ["utterance", "claim", "excerpt", "specific", "hedged", "restates"]


async def test_the_claims_are_ordered_by_utterance_and_restates_is_renumbered_with_them() -> None:
    client = _FakeScriptedClient(
        [
            _listing(
                _MILL_BUILDER,
                _BRIDGE_BUILDER,
                # The mill line restates the builder, listed out of order: position 1 is heard first.
                _item(2, "Hela Orsk built the mill.", "Hela Orsk built this too", restates=1),
            )
        ]
    )

    claims = await _extractor(client).extract(_CELL, ledger=RoleUsageLedger(role="judge"))

    assert [claim.utterance for claim in claims] == [1, 2, 2]
    assert claims[0].claim == "Hela Orsk built the Varn Bridge."
    assert claims[2].restates == 0
    assert all(claim.restates is None or claim.restates < at for at, claim in enumerate(claims))
    assert len(client.calls) == 1, "an order the model chose is no failure: nothing is asked again"


async def test_an_excerpt_not_in_its_utterance_is_asked_again_alone_with_the_failure_named() -> None:
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
    assert "its excerpt is not in utterance U2" in repair["user"]
    assert _CELL[2].text in repair["user"]
    assert _CELL[1].text not in repair["user"], "a failing item is asked about alone, with its own utterance"
    assert "0: The Varn Bridge was built in 1894." in repair["user"], "the claims heard before it are listed"
    assert "utterance" not in repair["response_format"]["json_schema"]["schema"]["properties"]
    (row,) = ledger.rows()
    assert row.call_count == 2


async def test_an_elided_excerpt_is_asked_again_naming_the_ellipsis() -> None:
    client = _FakeScriptedClient(
        [
            _listing(_item(1, "Hela Orsk built the Varn Bridge.", "the Varn Bridge… by Hela Orsk")),
            _repair("Hela Orsk built the Varn Bridge.", "built in 1894 by Hela Orsk"),
        ]
    )

    (claim,) = await _extractor(client).extract(_CELL, ledger=RoleUsageLedger(role="judge"))

    assert claim.excerpt == "built in 1894 by Hela Orsk"
    assert "shortened with an ellipsis" in client.calls[1]["user"]


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
    assert "its restates names claim 1, which is heard after it" in repair["user"]
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

    with pytest.raises(ClaimExtractionFailed, match=r"could not repair 1 claim\(s\) in 2 rounds: claim 1: its excerpt"):
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
        _listing({"utterance": 1, "claim": "The Varn Bridge was built in 1894."}),  # the tags are missing
        _listing(_item(5, "The Varn Bridge was built in 1894.", "built in 1894")),  # no such utterance
    ]
    client = _FakeScriptedClient(unreadable)
    ledger = RoleUsageLedger(role="judge")

    with pytest.raises(ClaimExtractionFailed, match="did not parse in 3 attempts"):
        await _extractor(client).extract(_CELL, ledger=ledger)

    first, second, third = client.calls
    assert "# Your last reply" not in first["user"]
    assert "It could not be used: No valid JSON object" in second["user"]
    assert "It could not be used: it is not the shape the response format states" in third["user"]
    assert all(call["system"] == first["system"] for call in client.calls)
    (row,) = ledger.rows()
    assert row.call_count == 3


async def test_a_reply_naming_an_utterance_the_cell_lacks_is_asked_again_in_full() -> None:
    client = _FakeScriptedClient(
        [_listing(_item(3, "The Varn Bridge was built in 1894.", "built in 1894")), _listing(_BRIDGE_YEAR)]
    )

    (claim,) = await _extractor(client).extract(_CELL, ledger=RoleUsageLedger(role="judge"))

    assert claim.utterance == 1
    assert "claim 0 names utterance U3, and the utterances are U0 to U2" in client.calls[1]["user"]


async def test_a_reply_cut_short_is_refused_at_once_and_still_billed() -> None:
    client = _FakeScriptedClient([_Reply(content='{"claims": [', stop_reason="max_tokens")])
    ledger = RoleUsageLedger(role="judge")

    with pytest.raises(ClaimExtractionFailed, match="cut short"):
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

    with pytest.raises(ClaimExtractionFailed, match="cut short"):
        await _extractor(client).extract(_CELL, ledger=ledger)

    assert len(client.calls) == 2
    (row,) = ledger.rows()
    assert row.call_count == 2


@pytest.mark.parametrize(
    "stops",
    [
        pytest.param((1,), id="not-from-0"),
        pytest.param((0, 0), id="repeated"),
        pytest.param((0, 2), id="a-gap"),
        pytest.param((1, 2, 3), id="shifted"),
    ],
)
async def test_stops_not_numbered_0_to_n_minus_1_each_once_are_refused_before_any_call(stops: tuple[int, ...]) -> None:
    client = _FakeScriptedClient([])
    cell = (Utterance(text="Hello."), *(Utterance(text=f"Stop {stop}.", stop_index=stop) for stop in stops))

    with pytest.raises(ValueError, match="numbered 0..n-1, each once"):
        await _extractor(client).extract(cell, ledger=RoleUsageLedger(role="judge"))
    assert client.calls == []


async def test_stops_in_any_order_with_other_utterances_between_are_accepted() -> None:
    client = _FakeScriptedClient([_listing()])
    cell = (
        Utterance(text="Stop one, then."),
        Utterance(text="Later.", stop_index=1),
        Utterance(text="An aside."),
        Utterance(text="Earlier.", stop_index=0),
    )

    assert await _extractor(client).extract(cell, ledger=RoleUsageLedger(role="judge")) == ()
    assert len(client.calls) == 1


@pytest.mark.parametrize(("text", "stop_index"), [("", None), ("  \n", None), ("Fine.", -1)])
def test_a_blank_utterance_or_a_negative_stop_is_refused(text: str, stop_index: int | None) -> None:
    with pytest.raises(ValidationError):
        Utterance(text=text, stop_index=stop_index)


async def test_no_utterances_make_no_call() -> None:
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
        ClaimExtractor(client=_FakeUnnamedClient(), segmentation=_SEGMENTATION)


def test_a_blank_segmentation_is_refused() -> None:
    with pytest.raises(ValueError, match="segmentation"):
        ClaimExtractor(client=_FakeScriptedClient([]), segmentation="  ")


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
#: budgets. Produced by ``ClaimExtractor.identity`` itself; it moves with any change to them, which is a new
#: definition: write the new value here deliberately.
_PINNED_IDENTITY = "scripted/claims|claims:06f3e7aa9695"


def test_the_identity_names_the_model_and_moves_with_the_segmentation_and_the_prompts() -> None:
    client = _FakeScriptedClient([])
    base = _extractor(client)

    assert base.identity == _PINNED_IDENTITY
    assert base.identity == _extractor(_FakeScriptedClient([])).identity
    others = {
        ClaimExtractor(client=client, segmentation="tour-stops-v2").identity,
        _extractor(client, guidance="A dated restoration counts as a claim.").identity,
        _extractor(client, guidance="A named architect counts as a claim.").identity,
        _extractor(_FakeScriptedClient([], model_name="scripted/other")).identity,
    }
    assert base.identity not in others
    assert len(others) == 4
    assert _extractor(client, guidance="  ").identity == base.identity, "blank guidance is no guidance"


def test_a_failed_extraction_is_a_fault_of_the_rig() -> None:
    assert issubclass(ClaimExtractionFailed, ApparatusError)
