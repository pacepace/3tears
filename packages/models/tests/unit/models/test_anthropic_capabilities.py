"""tests for the Anthropic Models API capability loader.

Every request goes through ``httpx.MockTransport``: no network, no key. The
recorded body is the shape the Models API returned for ``claude-haiku-5-5``
(verified live by the coordinator, 2026-10-13), with the id replaced by a
synthetic one so the test pins no model id: effort words low..max, thinking
with ``enabled`` false and ``adaptive`` / ``disabled`` true, structured
outputs, plus capability keys this build does not read.
"""

from __future__ import annotations

import copy
import json
import logging
from collections.abc import Callable, Iterator
from typing import Any
from urllib.parse import parse_qs
from uuid import uuid4

import httpx
import pytest

from threetears.models.anthropic_capabilities import (
    ANTHROPIC_MODELS_API_SOURCE,
    ANTHROPIC_MODELS_API_VERSION,
    ANTHROPIC_MODELS_URL,
    PublishedCapabilityShapeError,
    anthropic_models_api_headers,
    anthropic_published_facts,
    load_all_anthropic_capabilities,
    load_anthropic_capabilities,
)
from threetears.models.capabilities import (
    CapabilityOverride,
    ModelCapabilities,
    get_capabilities,
    get_published_capabilities,
    register_capabilities,
    register_capability_override,
    unregister_capability_override,
    unregister_published_capabilities,
)
from threetears.models.enums import ModelTier, ModelType

_KEY = "test-key-not-real"


def _recorded_body(model_id: str) -> dict[str, Any]:
    """the recorded Models API body, under ``model_id``.

    :param model_id: the id to answer with
    :ptype model_id: str
    :return: a fresh copy of the body
    :rtype: dict[str, Any]
    """
    return {
        "type": "model",
        "id": model_id,
        "display_name": "Recorded shape",
        "created_at": "2026-09-01T00:00:00Z",
        "max_input_tokens": 1_000_000,
        "max_tokens": 128_000,
        "capabilities": {
            "batch": {"supported": True},
            "citations": {"supported": True},
            "code_execution": {"supported": True},
            "context_management": {"compact_20260112": {"supported": True}, "supported": True},
            "effort": {
                "supported": True,
                "low": {"supported": True},
                "medium": {"supported": True},
                "high": {"supported": True},
                "xhigh": {"supported": True},
                "max": {"supported": True},
            },
            "image_input": {"supported": True},
            "pdf_input": {"supported": True},
            "structured_outputs": {"supported": True},
            "thinking": {
                "supported": True,
                "types": {
                    "enabled": {"supported": False},
                    "adaptive": {"supported": True},
                    "disabled": {"supported": True},
                },
            },
        },
    }


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.AsyncClient:
    """an AsyncClient over a mock transport, carrying the Models API headers.

    :param handler: answers each request
    :ptype handler: Callable[[httpx.Request], httpx.Response]
    :return: the client
    :rtype: httpx.AsyncClient
    """
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), headers=anthropic_models_api_headers(_KEY))


def _refuse_connection(request: httpx.Request) -> httpx.Response:
    """a transport that cannot connect.

    :param request: the request
    :ptype request: httpx.Request
    :return: never returns
    :rtype: httpx.Response
    :raises httpx.ConnectError: always
    """
    raise httpx.ConnectError("connection refused", request=request)


@pytest.fixture
def model_id() -> Iterator[str]:
    """a fresh registered model id with a static entry; every layer dropped afterwards.

    :return: the id
    :rtype: Iterator[str]
    """
    model_id = f"test-models-api-{uuid4().hex}"
    register_capabilities(
        model_id,
        ModelCapabilities(
            model_name=model_id,
            model_type=ModelType.CHAT,
            model_tier=ModelTier.SMALL,
            context_window=200_000,
            reasoning_efforts=[],
            reasoning_budget=True,
            adaptive_reasoning=False,
            accepts_temperature=False,
        ),
    )
    yield model_id
    unregister_published_capabilities(model_id)
    unregister_capability_override(model_id)


class TestTheMapping:
    """each published leaf lands on its field; nothing else is set."""

    def test_the_recorded_body_maps_onto_the_fields(self) -> None:
        """the recorded shape: effort words, adaptive only, can turn off, structured.

        :return: nothing
        :rtype: None
        """
        facts = anthropic_published_facts(_recorded_body("m"))
        assert facts.model_dump(exclude_unset=True) == {
            "reasoning_efforts": ["low", "medium", "high", "xhigh", "max"],
            "reasoning_budget": False,
            "adaptive_reasoning": True,
            "reasoning_disable": True,
            "structured_output": True,
        }

    def test_temperature_is_never_set(self) -> None:
        """the Models API does not publish temperature, so the static fact stands.

        :return: nothing
        :rtype: None
        """
        facts = anthropic_published_facts(_recorded_body("m"))
        assert "accepts_temperature" not in facts.model_dump(exclude_unset=True)

    def test_effort_unsupported_means_no_effort_parameter(self) -> None:
        """``effort.supported`` false empties the list whatever the word leaves say.

        :return: nothing
        :rtype: None
        """
        body = _recorded_body("m")
        body["capabilities"]["effort"]["supported"] = False
        assert anthropic_published_facts(body).reasoning_efforts == []

    def test_unsupported_words_are_left_out_in_ladder_order(self) -> None:
        """only the supported words, lowest first, whatever order the keys came in.

        :return: nothing
        :rtype: None
        """
        body = _recorded_body("m")
        effort = body["capabilities"]["effort"]
        effort["xhigh"]["supported"] = False
        body["capabilities"]["effort"] = dict(reversed(list(effort.items())))
        assert anthropic_published_facts(body).reasoning_efforts == ["low", "medium", "high", "max"]

    def test_a_budget_only_model(self) -> None:
        """the budget-only shape: enabled true, adaptive false, no effort parameter.

        :return: nothing
        :rtype: None
        """
        body = _recorded_body("m")
        body["capabilities"]["effort"] = {"supported": False}
        types = body["capabilities"]["thinking"]["types"]
        types["enabled"]["supported"] = True
        types["adaptive"]["supported"] = False
        facts = anthropic_published_facts(body)
        assert facts.reasoning_efforts == []
        assert facts.reasoning_budget is True
        assert facts.adaptive_reasoning is False

    def test_no_thinking_at_all_refuses_every_on_mode(self) -> None:
        """``thinking.supported`` false: neither a budget nor adaptive, even with no type leaves.

        :return: nothing
        :rtype: None
        """
        body = _recorded_body("m")
        body["capabilities"]["thinking"] = {"supported": False}
        facts = anthropic_published_facts(body)
        assert facts.reasoning_budget is False
        assert facts.adaptive_reasoning is False
        assert "reasoning_disable" not in facts.model_dump(exclude_unset=True)

    def test_an_unknown_capability_key_is_ignored(self) -> None:
        """a key this build does not read, in any shape, changes nothing.

        :return: nothing
        :rtype: None
        """
        body = _recorded_body("m")
        expected = anthropic_published_facts(body)
        body["capabilities"]["a_future_capability"] = "not even an object"
        body["capabilities"]["effort"]["a_future_word"] = 7
        body["capabilities"]["thinking"]["types"]["a_future_type"] = None
        assert anthropic_published_facts(body) == expected

    def test_a_model_object_without_capabilities_publishes_nothing(self) -> None:
        """no tree: an empty result, so every static fact stands.

        :return: nothing
        :rtype: None
        """
        body = _recorded_body("m")
        del body["capabilities"]
        assert anthropic_published_facts(body).model_dump(exclude_unset=True) == {}

    @pytest.mark.parametrize(
        ("mutate", "path"),
        [
            (lambda c: c.__setitem__("structured_outputs", {"supported": "yes"}), "capabilities.structured_outputs"),
            (lambda c: c.__setitem__("effort", ["low"]), "capabilities.effort"),
            (lambda c: c["effort"].__setitem__("high", True), "capabilities.effort.high"),
            (lambda c: c["effort"].pop("supported"), "capabilities.effort"),
            (lambda c: c["thinking"]["types"].__setitem__("adaptive", {}), "capabilities.thinking.types.adaptive"),
            (lambda c: c["thinking"].__setitem__("types", "adaptive"), "capabilities.thinking.types"),
        ],
    )
    def test_a_malformed_known_key_is_an_error(self, mutate: Callable[[dict[str, Any]], Any], path: str) -> None:
        """a key this build reads, in a shape it cannot read, raises and names the path.

        :param mutate: breaks one known node of the capabilities tree
        :ptype mutate: Callable[[dict[str, Any]], Any]
        :param path: the path the error must name
        :ptype path: str
        :return: nothing
        :rtype: None
        """
        body = _recorded_body("m")
        mutate(body["capabilities"])
        with pytest.raises(PublishedCapabilityShapeError) as raised:
            anthropic_published_facts(body)
        assert raised.value.path.startswith(path)


class TestLoadingOneModel:
    """``load_anthropic_capabilities`` reads, records, caches, and survives a failed fetch."""

    async def test_the_request_carries_the_auth_headers_and_the_model_path(self, model_id: str) -> None:
        """one GET to ``/v1/models/{id}`` with ``x-api-key`` and ``anthropic-version``.

        :param model_id: the registered id
        :ptype model_id: str
        :return: nothing
        :rtype: None
        """
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json=_recorded_body(model_id))

        async with _client(handler) as client:
            load = await load_anthropic_capabilities(client, model_id)

        assert load.ok and not load.from_cache
        assert load.source == ANTHROPIC_MODELS_API_SOURCE
        [request] = seen
        assert str(request.url) == f"{ANTHROPIC_MODELS_URL}/{model_id}"
        assert request.headers["x-api-key"] == _KEY
        assert request.headers["anthropic-version"] == ANTHROPIC_MODELS_API_VERSION

    async def test_fetched_facts_sit_above_static_and_below_an_override(self, model_id: str) -> None:
        """override > fetched > static, field by field.

        :param model_id: the registered id
        :ptype model_id: str
        :return: nothing
        :rtype: None
        """
        async with _client(lambda _: httpx.Response(200, json=_recorded_body(model_id))) as client:
            await load_anthropic_capabilities(client, model_id)

        caps = get_capabilities(model_id)
        assert caps is not None
        assert caps.reasoning_efforts == ["low", "medium", "high", "xhigh", "max"]  # fetched over static []
        assert caps.reasoning_budget is False  # fetched over static True
        assert caps.accepts_temperature is False  # not published: static stands
        assert caps.context_window == 200_000

        register_capability_override(model_id, CapabilityOverride(reasoning_efforts=["low"]))
        caps = get_capabilities(model_id)
        assert caps is not None
        assert caps.reasoning_efforts == ["low"]
        assert caps.adaptive_reasoning is True

    async def test_a_second_load_answers_from_the_cache(self, model_id: str) -> None:
        """within ``max_age_seconds`` nothing is fetched; ``max_age_seconds=0`` reads afresh.

        :param model_id: the registered id
        :ptype model_id: str
        :return: nothing
        :rtype: None
        """
        calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(str(request.url))
            return httpx.Response(200, json=_recorded_body(model_id))

        async with _client(handler) as client:
            first = await load_anthropic_capabilities(client, model_id)
            second = await load_anthropic_capabilities(client, model_id)
            third = await load_anthropic_capabilities(client, model_id, max_age_seconds=0)

        assert len(calls) == 2
        assert second.from_cache and second.facts == first.facts
        assert not third.from_cache

    @pytest.mark.parametrize(
        "respond",
        [
            lambda _: httpx.Response(500, json={"type": "error"}),
            lambda _: httpx.Response(404, json={"type": "error", "error": {"type": "not_found_error"}}),
            lambda _: httpx.Response(200, text="<html>a proxy page</html>"),
            _refuse_connection,
        ],
        ids=["5xx", "404", "not-json", "transport"],
    )
    async def test_a_failed_fetch_leaves_the_static_facts_and_logs(
        self,
        model_id: str,
        respond: Callable[[httpx.Request], httpx.Response],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """the load reports the failure, logs a WARNING, raises nothing, records nothing.

        :param model_id: the registered id
        :ptype model_id: str
        :param respond: the failing answer
        :ptype respond: Callable[[httpx.Request], httpx.Response]
        :param caplog: log capture
        :ptype caplog: pytest.LogCaptureFixture
        :return: nothing
        :rtype: None
        """
        before = get_capabilities(model_id)
        with caplog.at_level(logging.WARNING, logger="threetears.models.anthropic_capabilities"):
            async with _client(respond) as client:
                load = await load_anthropic_capabilities(client, model_id)

        assert not load.ok
        assert load.failure
        assert load.facts == {}
        assert get_capabilities(model_id) == before
        assert get_published_capabilities(model_id) is None
        assert any(r.levelno == logging.WARNING and "Models API read failed" in r.getMessage() for r in caplog.records)

    async def test_a_failed_refresh_keeps_the_earlier_published_facts(self, model_id: str) -> None:
        """a failure changes nothing, including facts an earlier load recorded.

        :param model_id: the registered id
        :ptype model_id: str
        :return: nothing
        :rtype: None
        """
        async with _client(lambda _: httpx.Response(200, json=_recorded_body(model_id))) as client:
            await load_anthropic_capabilities(client, model_id)
        recorded = get_published_capabilities(model_id)

        async with _client(lambda _: httpx.Response(503)) as client:
            load = await load_anthropic_capabilities(client, model_id, max_age_seconds=0)

        assert not load.ok
        assert get_published_capabilities(model_id) == recorded

    async def test_a_malformed_known_key_raises_and_records_nothing(self, model_id: str) -> None:
        """a contract change is an error to the loader's caller, never a quiet fact.

        :param model_id: the registered id
        :ptype model_id: str
        :return: nothing
        :rtype: None
        """
        body = _recorded_body(model_id)
        body["capabilities"]["thinking"]["supported"] = "true"
        async with _client(lambda _: httpx.Response(200, json=body)) as client:
            with pytest.raises(PublishedCapabilityShapeError):
                await load_anthropic_capabilities(client, model_id)
        assert get_published_capabilities(model_id) is None

    async def test_an_alias_is_recorded_under_both_ids(self, model_id: str) -> None:
        """the API answering with a different (dated) id records the facts under each.

        :param model_id: the registered id, used as the alias asked for
        :ptype model_id: str
        :return: nothing
        :rtype: None
        """
        answered = f"{model_id}-20260901"
        async with _client(lambda _: httpx.Response(200, json=_recorded_body(answered))) as client:
            load = await load_anthropic_capabilities(client, model_id)
        try:
            assert set(load.facts) == {model_id, answered}
            assert get_published_capabilities(answered) == get_published_capabilities(model_id)
        finally:
            unregister_published_capabilities(answered)


class TestLoadingTheListing:
    """``load_all_anthropic_capabilities`` pages through ``/v1/models``."""

    async def test_every_page_is_read_and_every_model_recorded(self) -> None:
        """two pages, joined by ``after_id``; each model's facts recorded.

        :return: nothing
        :rtype: None
        """
        ids = [f"test-listing-{uuid4().hex}" for _ in range(3)]
        budget_only = copy.deepcopy(_recorded_body(ids[2]))
        budget_only["capabilities"]["thinking"]["types"]["enabled"]["supported"] = True
        pages = {
            None: {"data": [_recorded_body(ids[0]), _recorded_body(ids[1])], "has_more": True, "last_id": ids[1]},
            ids[1]: {"data": [budget_only], "has_more": False, "last_id": ids[2]},
        }
        queries: list[dict[str, list[str]]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            query = parse_qs(request.url.query.decode())
            queries.append(query)
            return httpx.Response(200, content=json.dumps(pages[query.get("after_id", [None])[0]]))

        try:
            async with _client(handler) as client:
                load = await load_all_anthropic_capabilities(client, max_age_seconds=0)
                cached = await load_all_anthropic_capabilities(client)

            assert load.ok
            assert set(load.facts) == set(ids)
            assert [q.get("after_id") for q in queries] == [None, [ids[1]]]
            first = get_published_capabilities(ids[0])
            last = get_published_capabilities(ids[2])
            assert first is not None and first.reasoning_budget is False
            assert last is not None and last.reasoning_budget is True
            assert cached.from_cache and set(cached.facts) == set(ids)
            assert len(queries) == 2
        finally:
            for model_id in ids:
                unregister_published_capabilities(model_id)

    async def test_a_failure_on_a_later_page_records_nothing(self) -> None:
        """the first page's models are not recorded when the second page fails.

        :return: nothing
        :rtype: None
        """
        first_id = f"test-listing-{uuid4().hex}"

        def handler(request: httpx.Request) -> httpx.Response:
            if "after_id" in parse_qs(request.url.query.decode()):
                return httpx.Response(502)
            return httpx.Response(200, json={"data": [_recorded_body(first_id)], "has_more": True, "last_id": first_id})

        async with _client(handler) as client:
            load = await load_all_anthropic_capabilities(client, max_age_seconds=0)

        assert not load.ok
        assert get_published_capabilities(first_id) is None

    async def test_a_listing_entry_without_an_id_is_an_error(self) -> None:
        """the id keys the record; an entry without one is unreadable, not skipped.

        :return: nothing
        :rtype: None
        """
        body = _recorded_body("unused")
        del body["id"]
        async with _client(lambda _: httpx.Response(200, json={"data": [body], "has_more": False})) as client:
            with pytest.raises(PublishedCapabilityShapeError):
                await load_all_anthropic_capabilities(client, max_age_seconds=0)
