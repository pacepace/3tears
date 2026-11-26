"""The claim extractor: the step that lists the factual claims a cell's segments state, on a scripted client.

Every behaviour is driven through a client that answers from a script and records what it was sent, so nothing is
spent and each reply is chosen. The claims come back ordered by segment, each excerpt the segment's own slice at
offsets the verbatim check found, ``restates`` resolved from its quote of an earlier claim's excerpt to the first claim
of its chain and renumbered with the order. A quote names the claim whose excerpt holds it, never a counted place: a
line said word for word in two segments resolves to the first, and a position given in its place is unreadable.
Repair, not refusal: a reply that will not parse is asked again in full naming why; an item that fails a check (an
excerpt not in its segment, a ``restates`` quote no earlier excerpt holds, or that excerpts of different facts hold)
is asked again alone with its failure named,
for two rounds, then the extraction raises. Every claim carries a scope, ``world`` or ``session``: a back-announce
line splits into a session claim and a world claim, and a ``restates`` crossing scopes is repaired; a cut-short reply raises at once. The failure is ``ClaimGradingFailed``,
no ``ApparatusError``. Every completion is billed to the ``judge`` ledger. A cell's units are numbered 0..n-1 with no
gap, a unit spanning any number of segments, or the cell is refused before any call; a segment's tags never reach a prompt. The identity names the model,
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
from threetears.evals.schema.models import RepairBudget

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
    restates: str | None = None,
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
                _item(
                    1,
                    "The tour just visited the Varn Bridge.",
                    "as we just saw",
                    scope="session",
                    restates="That was the Varn Bridge",
                ),
                _item(
                    1, "The Varn Bridge spans the harbour.", "spans the harbour", restates="That was the Varn Bridge"
                ),
            ),
            _repair("The Varn Bridge spans the harbour.", "spans the harbour", restates=None),
        ]
    )

    claims = await _extractor(client).extract(cell, ledger=RoleUsageLedger(role="judge"))

    assert [(claim.scope, claim.restates) for claim in claims] == [("session", None), ("session", 0), ("world", None)]
    repair = client.calls[1]
    assert "position 2" in repair["user"]
    assert "no world claim listed before it has that excerpt" in repair["user"]


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
                _item(
                    1,
                    "The tour just visited the Varn Bridge.",
                    "the one we just crossed",
                    restates="That was the bridge",
                ),
                _item(1, "The Varn Bridge opened in 1894.", "opened in 1894"),
            ),
            # The repair fixes the excerpt and corrects the scope to session: claim 1 now restates across scopes.
            _repair("The tour just visited the Varn Bridge.", "That was the Varn Bridge", scope="session"),
            _repair(
                "The tour just crossed the Varn Bridge.",
                "the one we just crossed",
                scope="session",
                restates="That was the Varn Bridge",
            ),
        ]
    )

    claims = await _extractor(client).extract(cell, ledger=RoleUsageLedger(role="judge"))

    assert [(claim.scope, claim.restates) for claim in claims] == [("session", None), ("session", 0), ("world", None)]
    assert len(client.calls) == 3
    second_round = client.calls[2]["user"]
    assert "position 1" in second_round
    assert 'its restates quotes "That was the bridge", and no world claim listed before it' in second_round


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
                _item(
                    2, "Hela Orsk built the mill.", "Hela Orsk built this too", restates="built in 1894 by Hela Orsk"
                ),
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
                _item(2, "Hela Orsk built the Varn Bridge.", "Hela Orsk built this too", restates="by Hela Orsk"),
                # It quotes the restatement, whose own quote names the first claim: the chain resolves to that one.
                _item(2, "Hela Orsk built the Varn Bridge.", "Hela Orsk", restates="Hela Orsk built this too"),
            )
        ]
    )

    claims = await _extractor(client).extract(_CELL, ledger=RoleUsageLedger(role="judge"))

    assert [claim.restates for claim in claims] == [None, 0, 0]


#: A tour whose last stop repeats its first line word for word, and whose two first-stop facts are cut from one
#: sentence: the shapes a quote must tell apart.
_LOOP = (
    Segment(text="The Varn Bridge opened in 1894, built by Hela Orsk.", unit=0),
    Segment(text="The mill is older.", unit=1),
    Segment(text="Back where we began: the Varn Bridge opened in 1894.", unit=2),
    Segment(text="The Varn Bridge opened in 1894. Hela Orsk again.", unit=3),
)
_OPENED = _item(0, "The Varn Bridge opened in 1894.", "The Varn Bridge opened in 1894")
_BUILT_BY = _item(0, "Hela Orsk built the Varn Bridge.", "built by Hela Orsk")
_OLDER = _item(1, "The mill is older than the Varn Bridge.", "The mill is older")


async def test_a_text_identical_restatement_in_another_segment_resolves_to_the_first_saying() -> None:
    client = _FakeScriptedClient(
        [
            _listing(
                _OPENED,
                _BUILT_BY,
                _OLDER,
                # Said word for word again: only the first saying is listed before it, so the quote names that one.
                _item(
                    2,
                    "The Varn Bridge opened in 1894.",
                    "the Varn Bridge opened in 1894",
                    restates="The Varn Bridge opened in 1894",
                ),
                # A third saying: both earlier sayings hold the quote, and both stand for the first.
                _item(
                    3,
                    "The Varn Bridge opened in 1894.",
                    "The Varn Bridge opened in 1894",
                    restates="the Varn Bridge opened in 1894",
                ),
                # One of two facts cut from one sentence: its excerpt, not the sentence, is what the quote finds.
                _item(3, "Hela Orsk built the Varn Bridge.", "Hela Orsk again", restates="by Hela Orsk"),
            )
        ]
    )

    claims = await _extractor(client).extract(_LOOP, ledger=RoleUsageLedger(role="judge"))

    assert [claim.restates for claim in claims] == [None, None, None, 0, 0, 1]
    assert len(client.calls) == 1, "every quote named one claim: nothing is asked again"


async def test_a_quote_names_the_claim_whose_excerpt_holds_it_never_a_counted_place() -> None:
    client = _FakeScriptedClient(
        [
            _listing(
                _OPENED,
                _BUILT_BY,
                _OLDER,
                _item(3, "Hela Orsk built the Varn Bridge.", "Hela Orsk again", restates="built by Hela Orsk"),
            )
        ]
    )

    claims = await _extractor(client).extract(_LOOP, ledger=RoleUsageLedger(role="judge"))

    assert claims[3].restates == 1
    assert claims[claims[3].restates].excerpt == "built by Hela Orsk", "the target is the claim the quote names"


async def test_a_position_given_for_restates_is_unreadable_and_asked_again_in_full() -> None:
    """A counted place, the shape a miscounted pointer took, never passes as a quote: the whole reply is asked again."""
    by_place = {**_item(3, "Hela Orsk built the Varn Bridge.", "Hela Orsk again"), "restates": 0}
    client = _FakeScriptedClient(
        [
            _listing(_OPENED, _BUILT_BY, by_place),
            _listing(
                _OPENED,
                _BUILT_BY,
                _item(3, "Hela Orsk built the Varn Bridge.", "Hela Orsk again", restates="built by Hela Orsk"),
            ),
        ]
    )

    claims = await _extractor(client).extract(_LOOP, ledger=RoleUsageLedger(role="judge"))

    assert claims[2].restates == 1, "the corrected quote names the builder, which the counted place 0 did not"
    assert "It could not be used: it is not the shape the response format states" in client.calls[1]["user"]
    restates = client.calls[0]["response_format"]["json_schema"]["schema"]["properties"]["claims"]["items"]
    assert restates["properties"]["restates"] == {"anyOf": [{"type": "string"}, {"type": "null"}]}
    assert "Never a number: the excerpt is what names the claim." in client.calls[0]["system"]


#: A line whose two facts are cut so that one claim's excerpt sits inside the other's, said again in a later line.
_NESTED = (
    Segment(text="Nineteen eighty-three: Gareth Jones engineering People Are People.", unit=0),
    Segment(text="Jones again, on People Are People.", unit=1),
)
_YEAR = _item(
    0,
    "People Are People came out in 1983.",
    "Nineteen eighty-three: Gareth Jones engineering People Are People",
)
_ENGINEER = _item(0, "Gareth Jones engineered People Are People.", "Gareth Jones engineering People Are People")
_AGAIN = ("Gareth Jones engineered People Are People.", "Jones again, on People Are People")


async def test_a_quote_of_a_whole_excerpt_names_it_over_a_longer_excerpt_that_holds_it() -> None:
    client = _FakeScriptedClient(
        [_listing(_YEAR, _ENGINEER, _item(1, *_AGAIN, restates="Gareth Jones engineering People Are People"))]
    )

    claims = await _extractor(client).extract(_NESTED, ledger=RoleUsageLedger(role="judge"))

    assert claims[2].restates == 1, "the engineer's whole excerpt names the engineer, not the year it sits inside"
    assert len(client.calls) == 1


async def test_a_quote_adding_a_neighbours_words_never_names_the_neighbour() -> None:
    """The engineer's excerpt with the year line's words before it is the year's excerpt in part, and the engineer's
    whole: it could name either, so it is asked again, and the engineer's whole excerpt then names the engineer."""
    widened = "eighty-three: Gareth Jones engineering People Are People"
    client = _FakeScriptedClient(
        [
            _listing(_YEAR, _ENGINEER, _item(1, *_AGAIN, restates=widened)),
            _repair(*_AGAIN, restates="Gareth Jones engineering People Are People"),
        ]
    )

    claims = await _extractor(client).extract(_NESTED, ledger=RoleUsageLedger(role="judge"))

    assert claims[2].restates == 1
    failure = client.calls[1]["user"]
    assert f'its restates quotes "{widened}", which could name 2 earlier world claims stating different facts' in (
        failure
    )
    assert "copy the whole excerpt of the one this claim restates, exactly as listed and no more, or give null" in (
        failure
    )
    assert "copy more" not in failure


#: Two facts cut from one stretch, each with that stretch as its excerpt, said again later.
_SHARED = (
    Segment(text="Hela Orsk's Varn Bridge, 1894.", unit=0),
    Segment(text="The Varn Bridge once more.", unit=1),
)
_SHARED_BUILDER = _item(0, "Hela Orsk built the Varn Bridge.", "Hela Orsk's Varn Bridge, 1894")
_SHARED_YEAR = _item(0, "The Varn Bridge dates from 1894.", "Hela Orsk's Varn Bridge, 1894")
_ONCE_MORE = ("The Varn Bridge dates from 1894.", "The Varn Bridge once more")


async def test_an_excerpt_two_facts_share_is_asked_again_offering_null_as_the_answer() -> None:
    client = _FakeScriptedClient(
        [
            _listing(_SHARED_BUILDER, _SHARED_YEAR, _item(1, *_ONCE_MORE, restates="Hela Orsk's Varn Bridge, 1894")),
            _repair(*_ONCE_MORE, restates=None),
        ]
    )

    claims = await _extractor(client).extract(_SHARED, ledger=RoleUsageLedger(role="judge"))

    assert claims[2].restates is None, "null leaves the restatement to be checked on its own"
    failure = client.calls[1]["user"]
    assert "the excerpt of 2 earlier world claims stating different facts, so no quote can name one of them" in failure
    assert "give null, which is the right answer here" in failure
    assert (
        "When that excerpt is also the excerpt of another claim stating a different fact, give null"
        in (client.calls[0]["system"])
    )


async def test_an_excerpt_a_claim_and_its_restatement_share_names_the_first_of_them() -> None:
    shared = "Hela Orsk's Varn Bridge, 1894"
    client = _FakeScriptedClient(
        [
            _listing(
                _SHARED_BUILDER,
                _item(0, "Hela Orsk built the bridge.", shared, restates=shared),
                _item(1, "Hela Orsk built the Varn Bridge.", "The Varn Bridge once more", restates=shared),
            )
        ]
    )

    claims = await _extractor(client).extract(_SHARED, ledger=RoleUsageLedger(role="judge"))

    assert [claim.restates for claim in claims] == [None, 0, 0], "both holders state one fact: the first of them"
    assert len(client.calls) == 1


@pytest.mark.parametrize(
    "quote",
    [
        pytest.param("1", id="a-counted-place-as-a-string"),
        pytest.param("1894", id="a-year-one-excerpt-holds"),
        pytest.param("Var", id="a-fragment"),
        pytest.param(",;", id="punctuation"),
    ],
)
async def test_a_number_or_a_fragment_is_no_quote_even_where_one_excerpt_holds_it(quote: str) -> None:
    client = _FakeScriptedClient(
        [
            _listing(
                _OPENED,
                _BUILT_BY,
                _OLDER,
                _item(3, "The Varn Bridge opened in 1894.", "The Varn Bridge opened in 1894", restates=quote),
            ),
            _repair(
                "The Varn Bridge opened in 1894.",
                "The Varn Bridge opened in 1894",
                restates="The Varn Bridge opened in 1894",
            ),
        ]
    )

    claims = await _extractor(client).extract(_LOOP, ledger=RoleUsageLedger(role="judge"))

    assert claims[3].restates == 0
    assert (
        f"its restates quotes {json.dumps(quote)}, a number or a fragment, which names no claim"
        in (client.calls[1]["user"])
    )


async def test_a_short_excerpt_quoted_whole_names_its_claim() -> None:
    cell = (Segment(text="The Varn Bridge opened in 1894.", unit=0), Segment(text="Yes, 1894.", unit=1))
    client = _FakeScriptedClient(
        [
            _listing(
                _item(0, "The Varn Bridge opened in 1894.", "1894"),
                _item(1, "The Varn Bridge opened in 1894.", "Yes, 1894", restates="1894"),
            )
        ]
    )

    claims = await _extractor(client).extract(cell, ledger=RoleUsageLedger(role="judge"))

    assert [claim.restates for claim in claims] == [None, 0], "a number that is a claim's whole excerpt names it"
    assert len(client.calls) == 1


async def test_a_short_number_no_excerpt_equals_is_still_refused() -> None:
    cell = (Segment(text="Platform 2 opened in 1894.", unit=0), Segment(text="Yes, 1894.", unit=1))
    client = _FakeScriptedClient(
        [
            _listing(
                _item(0, "Platform 2 opened in 1894.", "Platform 2 opened in 1894"),
                _item(1, "Platform 2 opened in 1894.", "Yes, 1894", restates="2"),
            ),
            _repair("Platform 2 opened in 1894.", "Yes, 1894", restates="Platform 2 opened in 1894"),
        ]
    )

    claims = await _extractor(client).extract(cell, ledger=RoleUsageLedger(role="judge"))

    assert claims[1].restates == 0
    assert 'its restates quotes "2", a number or a fragment, which names no claim' in client.calls[1]["user"]


@pytest.mark.parametrize(
    ("cell", "listed", "quote", "named"),
    [
        pytest.param(
            _SHARED,
            (_SHARED_BUILDER, _SHARED_YEAR, _item(1, *_ONCE_MORE)),
            "Hela Orsk's Varn Bridge, 1894",
            "so no quote can name one of them",
            id="a-shared-excerpt",
        ),
        pytest.param(
            _NESTED,
            (_YEAR, _ENGINEER, _item(1, *_AGAIN)),
            "eighty-three: Gareth Jones engineering People Are People",
            "which could name 2 earlier world claims stating different facts",
            id="a-widened-quote",
        ),
        pytest.param(
            _LOOP,
            (_OPENED, _BUILT_BY, _OLDER, _item(3, "Hela Orsk built the Varn Bridge.", "Hela Orsk again")),
            "built by Ida Venn",
            "and no world claim listed before it has that excerpt",
            id="absent",
        ),
    ],
)
async def test_a_quote_naming_no_single_claim_after_the_repair_rounds_fails_the_grading(
    cell: tuple[Segment, ...], listed: tuple[dict[str, Any], ...], quote: str, named: str
) -> None:
    *earlier, restating = listed
    again = {key: value for key, value in restating.items() if key != "segment"}
    client = _FakeScriptedClient(
        [
            _listing(*earlier, {**restating, "restates": quote}),
            _repair(again["claim"], again["excerpt"], restates=quote),
            _repair(again["claim"], again["excerpt"], restates=quote),
        ]
    )

    with pytest.raises(
        ClaimGradingFailed, match=r"could not repair 1 claim\(s\) in 2 rounds: claim \d+: its restates"
    ) as raised:
        await _extractor(client).extract(cell, ledger=RoleUsageLedger(role="judge"))

    assert named in str(raised.value)
    assert raised.value.loss == "repair_budget_spent"
    assert len(client.calls) == 3


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
    assert (
        "- excerpt: the Varn Bridge, built in 1894\n  claim: The Varn Bridge was built in 1894.\n  scope: world"
        in repair["user"]
    ), "the claims listed before it are shown by their excerpts, the words a restates copies, never by a number"
    assert "0:" not in repair["user"]
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


async def test_a_restates_quoting_a_later_claim_is_asked_again_and_the_correction_is_kept() -> None:
    client = _FakeScriptedClient(
        [
            _listing(
                _item(
                    1,
                    "Hela Orsk built the Varn Bridge.",
                    "built in 1894 by Hela Orsk",
                    restates="Hela Orsk built this too",
                ),
                _MILL_BUILDER,
            ),
            _repair("Hela Orsk built the Varn Bridge.", "built in 1894 by Hela Orsk", restates=None),
        ]
    )

    claims = await _extractor(client).extract(_CELL, ledger=RoleUsageLedger(role="judge"))

    assert [claim.restates for claim in claims] == [None, None]
    repair = client.calls[1]
    assert 'its restates quotes "Hela Orsk built this too", and no world claim listed before it' in repair["user"]
    assert "position 0" in repair["user"]
    assert "# Claims listed before it\n\nnone" in repair["user"]


@pytest.mark.parametrize(
    "restates",
    [
        pytest.param("built in 1893 by Hela Orsk", id="held-by-no-excerpt"),
        pytest.param("built in 1894 by Hela Orsk", id="its-own-excerpt"),
    ],
)
async def test_a_restates_naming_no_other_claim_is_asked_again(restates: str) -> None:
    client = _FakeScriptedClient(
        [
            _listing(_item(1, "Hela Orsk built the Varn Bridge.", "built in 1894 by Hela Orsk", restates=restates)),
            _repair("Hela Orsk built the Varn Bridge.", "built in 1894 by Hela Orsk"),
        ]
    )

    (claim,) = await _extractor(client).extract(_CELL, ledger=RoleUsageLedger(role="judge"))

    assert claim.restates is None
    assert "and no world claim listed before it has that excerpt" in client.calls[1]["user"]


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
        pytest.param((0, 2), id="a-gap"),
        pytest.param((1, 2, 3), id="shifted"),
    ],
)
async def test_units_not_numbered_0_to_n_minus_1_with_no_gap_are_refused_before_any_call(
    units: tuple[int, ...],
) -> None:
    client = _FakeScriptedClient([])
    cell = (Segment(text="Hello."), *(Segment(text=f"Unit {unit}.", unit=unit) for unit in units))

    with pytest.raises(ValueError, match="numbered 0..n-1 with no gap"):
        await _extractor(client).extract(cell, ledger=RoleUsageLedger(role="judge"))
    assert client.calls == []


async def test_a_unit_spanning_several_segments_is_accepted() -> None:
    """An entry's description and its facts are one unit, each its own segment."""
    client = _FakeScriptedClient([_listing()])
    cell = (
        Segment(text="A preamble."),
        Segment(text="Unit 0.", unit=0),
        Segment(text="Unit 0 again.", unit=0),
        Segment(text="Unit 1.", unit=1),
        Segment(text="Unit 1 more.", unit=1),
    )

    await _extractor(client).extract(cell, ledger=RoleUsageLedger(role="judge"))
    assert len(client.calls) == 1


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


# parity-with: threetears.evals.schema.completion.BoundCompletionClient
@dataclass
class _FakeBytePricedClient:
    """Prices a call by the bytes it sends, and records every prompt it priced and every one it was sent."""

    replies: list[_Reply]
    model_name: str = _MODEL
    priced: list[str] = field(default_factory=list)
    sent: list[str] = field(default_factory=list)

    def price_ceiling(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> float | None:
        del response_format
        self.priced.append(user)
        return _bytes_price(system, user)

    async def generate(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> _Reply:
        del response_format
        self.sent.append(system + "\x00" + user)
        return self.replies[len(self.sent) - 1]

    async def aclose(self) -> None:
        """Nothing to release."""

    async def __aenter__(self) -> _FakeBytePricedClient:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self.aclose()


def _bytes_price(system: str, user: str) -> float:
    return (len(system.encode()) + len(user.encode())) * 1e-6


def _wide(base: int, length: int) -> str:
    """``length`` characters of four bytes each in UTF-8, letters a fold leaves alone, from ``base`` on."""
    return "".join(chr(0x20000 + base + at) for at in range(length))


#: The headings a repair prompt is cut into, in order: its segment, the claims before it, the item and its failure.
_REPAIR_PARTS = ("# Claims listed before it\n\n", "# The claim, at position ", "# What is wrong\n\n")


def _parts(user: str) -> list[int]:
    """The bytes of each part of a repair prompt, between its headings."""
    sizes: list[int] = []
    rest = user
    for heading in _REPAIR_PARTS:
        part, rest = rest.split(heading, 1)
        sizes.append(len(part.encode()))
    return [*sizes, len(rest.encode())]


async def test_a_repair_naming_every_claim_before_it_costs_no_more_than_the_price_it_was_admitted_at() -> None:
    """The worst repair the check can ask: many long claims before it, each shown with its excerpt, and a quote every
    one of their excerpts holds, which the failure lists in full. Every part of the prompt sent fits the part priced, so
    halving the allowance for the claims before it, or pricing the failure at the floor, is red here."""
    shared = _wide(30_000, 5)
    excerpts = [f"{_wide(at * 40, 40)} {shared}" for at in range(60)]
    cell = (Segment(text="; ".join(excerpts) + ".", unit=0), Segment(text=f"{shared} again", unit=1))
    listed = [_item(0, _wide(4_000 + at * 150, 150), excerpt) for at, excerpt in enumerate(excerpts)]
    restating = _item(1, "It is said again.", f"{shared} again", restates=shared)
    client = _FakeBytePricedClient(
        [_listing(*listed, restating), _repair("It is said again.", f"{shared} again", restates=None)]
    )
    budget = RepairBudget(rounds=1, max_items_per_cell=1)
    extractor = ClaimExtractor(client=client, repairs=budget)

    one = extractor.price_ceiling(cell)
    two = ClaimExtractor(client=client, repairs=RepairBudget(rounds=1, max_items_per_cell=2)).price_ceiling(cell)
    claims = await extractor.extract(cell, ledger=RoleUsageLedger(role="judge"))

    assert one is not None and two is not None
    assert claims[-1].restates is None
    system, user = client.sent[1].split("\x00")
    assert "which could name 60 earlier world claims stating different facts" in user
    assert _bytes_price(system, user) <= two - one, "the repair sent costs no more than its share of the ceiling"
    (priced,) = {prompt for prompt in client.priced if "# What is wrong" in prompt}
    for sent_bytes, priced_bytes, part in zip(
        _parts(user), _parts(priced), ("segment", "earlier", "item", "failure"), strict=True
    ):
        assert sent_bytes <= priced_bytes, f"the repair's {part} outgrew the part priced"


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
_PINNED_IDENTITY = "scripted/claims|claims:017fd2c3f122"

#: The identity the same extractor had while ``restates`` named its original by position in the list. Grades listed
#: under it never pool with grades whose restatements were named by quote.
_POSITION_POINTER_IDENTITY = "scripted/claims|claims:b326c9108c71"


def test_the_identity_names_the_model_and_moves_with_the_guidance() -> None:
    client = _FakeScriptedClient([])
    base = _extractor(client)

    assert base.identity == _PINNED_IDENTITY
    assert base.identity != _POSITION_POINTER_IDENTITY, "naming the original by quote is a new definition"
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
