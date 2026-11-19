"""The entity extractor: the model step that lists the entities prose names, on a scripted client.

Every behaviour is driven through a client that answers from a script and records what it was sent, so nothing is
spent and each reply is chosen: the list comes back in the reply's order, every completion is billed to the ``judge``
ledger, an unreadable reply is asked again twice and then refused, a cut-short one is refused at once, blank text
makes no call, and a client that names no model is refused before any call.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from threetears.evals.kernel import (
    EntityExtractionFailed,
    EntityExtractor,
    NamedEntity,
    RoleUsageLedger,
    grade_set,
)
from threetears.evals.schema.reference import ReferenceKey

_MODEL = "scripted/extractor"

_TEXT = "Tonight we open with Band A, then Band C from Leeds, and close on a deep cut by Band E."


@dataclass(frozen=True)
class _Reply:
    """One completion, shaped as the engine's completion port describes it."""

    content: str
    stop_reason: str = "end_turn"
    model: str = _MODEL
    served_model: str | None = _MODEL
    input_tokens: int | None = 200
    output_tokens: int | None = 40
    reasoning_tokens: int | None = None
    cost_usd: float | None = 0.001
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


def _listing(*entities: dict[str, Any]) -> _Reply:
    return _Reply(content=json.dumps({"entities": list(entities)}))


def _extractor(client: _FakeScriptedClient, *, fields: tuple[str, ...] = ("origin",)) -> EntityExtractor:
    return EntityExtractor(client=client, entity_kind="bands", fields=fields)


async def test_the_extractor_returns_the_entities_in_the_replys_order_with_the_fields_stated() -> None:
    client = _FakeScriptedClient(
        [
            _listing(
                {"name": "Band A", "fields": {"origin": None}},
                {"name": "Band C", "fields": {"origin": "Leeds"}},
                {"name": "Band E", "fields": {"origin": " "}},
            )
        ]
    )

    named = await _extractor(client).extract(_TEXT, ledger=RoleUsageLedger(role="judge"))

    assert named == (
        NamedEntity(name="Band A"),
        NamedEntity(name="Band C", fields={"origin": "Leeds"}),
        NamedEntity(name="Band E"),
    )
    (call,) = client.calls
    assert _TEXT in call["user"]
    assert "bands" in call["user"]
    schema = call["response_format"]["json_schema"]
    assert schema["strict"] is True
    entity = schema["schema"]["properties"]["entities"]["items"]
    assert entity["required"] == ["name", "fields"]
    assert entity["properties"]["fields"]["required"] == ["origin"]


async def test_the_extractor_never_sees_the_key_and_its_list_grades_against_it() -> None:
    key = ReferenceKey.model_validate({"core": [{"name": "Band A"}], "acceptable": [{"name": "Band C"}]})
    client = _FakeScriptedClient([_listing({"name": "Band A"}, {"name": "Band E"})])

    named = await _extractor(client, fields=()).extract(_TEXT, ledger=RoleUsageLedger(role="judge"))

    (call,) = client.calls
    assert "Band C" not in call["system"] + call["user"].replace(_TEXT, "")
    grade = grade_set(named, key)
    assert (grade.found, grade.unverified_names) == (("Band A",), ("Band E",))


async def test_every_completion_is_recorded_on_the_judge_ledger() -> None:
    client = _FakeScriptedClient([_Reply(content="not json"), _listing({"name": "Band A", "fields": {"origin": None}})])
    ledger = RoleUsageLedger(role="judge")

    await _extractor(client).extract(_TEXT, ledger=ledger)

    (row,) = ledger.rows()
    assert row.role == "judge"
    assert row.model == _MODEL
    assert row.call_count == 2
    assert row.cost_usd == pytest.approx(0.002)


async def test_an_unreadable_reply_is_asked_twice_more_then_refused() -> None:
    unreadable = [
        _Reply(content="not json"),
        _listing({"name": "Band A"}),  # the declared field is missing
        _listing({"name": "Band A", "fields": {"origin": None, "decade": "1970s"}}),  # a field nobody asked for
    ]
    client = _FakeScriptedClient(unreadable)
    ledger = RoleUsageLedger(role="judge")

    with pytest.raises(EntityExtractionFailed, match="did not parse in 3 attempts"):
        await _extractor(client).extract(_TEXT, ledger=ledger)

    assert len(client.calls) == 3
    (row,) = ledger.rows()
    assert row.call_count == 3


async def test_a_reply_cut_short_is_refused_at_once_and_still_billed() -> None:
    client = _FakeScriptedClient(
        [_Reply(content='{"entities": [{"name": "Band A", "fields": {"origin": null}}]}', stop_reason="max_tokens")]
    )
    ledger = RoleUsageLedger(role="judge")

    with pytest.raises(EntityExtractionFailed, match="cut short"):
        await _extractor(client).extract(_TEXT, ledger=ledger)

    assert len(client.calls) == 1
    (row,) = ledger.rows()
    assert row.call_count == 1


@pytest.mark.parametrize("text", ["", "   \n\t"])
async def test_blank_text_names_nothing_and_makes_no_call(text: str) -> None:
    client = _FakeScriptedClient([])
    ledger = RoleUsageLedger(role="judge")

    assert await _extractor(client).extract(text, ledger=ledger) == ()
    assert client.calls == []
    assert ledger.rows() == []


def test_a_client_that_names_no_model_is_refused() -> None:
    with pytest.raises(ValueError, match="names its model"):
        EntityExtractor(client=_FakeUnnamedClient(), entity_kind="bands")


@pytest.mark.parametrize(
    ("entity_kind", "fields"),
    [(" ", ()), ("bands", ("origin", "origin")), ("bands", ("",))],
)
def test_a_blank_kind_or_a_blank_or_repeated_field_is_refused(entity_kind: str, fields: tuple[str, ...]) -> None:
    with pytest.raises(ValueError, match="entity_kind|fields"):
        EntityExtractor(client=_FakeScriptedClient([]), entity_kind=entity_kind, fields=fields)


async def test_a_ledger_of_another_role_is_refused() -> None:
    client = _FakeScriptedClient([])

    with pytest.raises(ValueError, match="'judge' role"):
        await _extractor(client).extract(_TEXT, ledger=RoleUsageLedger(role="candidate"))
    assert client.calls == []


def test_the_identity_names_the_model_and_moves_with_how_the_list_is_made() -> None:
    client = _FakeScriptedClient([])
    base = _extractor(client)

    assert base.identity.startswith(f"{_MODEL}|extract:")
    assert len(base.identity.rsplit(":", 1)[1]) == 12
    assert base.identity == _extractor(_FakeScriptedClient([])).identity
    others = {
        EntityExtractor(client=client, entity_kind="cities", fields=("origin",)).identity,
        _extractor(client, fields=()).identity,
        _extractor(client, fields=("origin", "decade")).identity,
        _extractor(_FakeScriptedClient([], model_name="scripted/other")).identity,
    }
    assert base.identity not in others
    assert len(others) == 4
