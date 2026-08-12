"""Check 8: the search-results metadata key survives the NATS hop intact.

The migration path this whole workstream rests on is "structure rides
``ToolResult.metadata`` under one named key" (D22). That claim is cheap to
make in-process and only true if it holds across the wire -- and the wire is
where it could quietly stop being true, because a dispatch serializes a
``CallResponse`` to JSON and a consumer rebuilds it on the far side. A field
the pod populates but the envelope drops, or a value that serializes to
something ``model_validate`` will not take back, would leave every
structure-reading consumer silently parsing prose again.

So this drives the real :class:`~threetears.agent.tools.server.ToolServer`
dispatch with the real :class:`~threetears.agent.tools.builtin.web_search.WebSearchTool`
over a stub transport, takes the bytes the pod actually published, and
reconstructs the projection from them with
:meth:`~threetears.search.contracts.SearchResultsMetadata.from_metadata` --
the same call a consumer makes. Nothing here asserts on an in-process object:
the payload is decoded from bytes first, every time.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any
from uuid import uuid4

import pytest
from pydantic import JsonValue

from threetears.agent.tools.builtin.web_search import WebSearchTool
from threetears.agent.tools.server import ToolServer
from threetears.nats import IncomingMessage
from threetears.search.contracts import (
    SEARCH_RESULTS_METADATA_KEY,
    EGRESS_DIRECT,
    SearchResultsMetadata,
    TransportResponse,
)

from unit.tools._pod_auth import StubReplayGuard as _PodReplayGuard
from unit.tools._pod_auth import jwks_provider as _pod_jwks_provider
from unit.tools._pod_auth import signed_call_payload as _signed_call_payload

_NS = "3tears"
_POD = "search-pod-1"
_BASE_URL = "http://searxng.internal:8080"

_PAYLOAD: dict[str, Any] = {
    "query": "otter husbandry",
    "results": [
        {
            "url": "https://example.test/otters",
            "title": "Otter husbandry",
            "content": "Everything about keeping otters.",
            "engine": "duckduckgo",
            "score": 1.5,
        }
    ],
}


# parity-with: threetears.search.contracts.transport.SearchTransport
class _StubSearchTransport:
    """one canned SearXNG payload, or a status that makes the call fail."""

    def __init__(self, *, status_code: int = 200) -> None:
        self._status_code = status_code

    @property
    def egress_name(self) -> str:
        return EGRESS_DIRECT

    async def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        params: Mapping[str, str] | None = None,
        json_body: Mapping[str, JsonValue] | None = None,
        timeout_seconds: float | None = None,
    ) -> TransportResponse:
        del method, headers, params, json_body, timeout_seconds
        return TransportResponse(
            status_code=self._status_code,
            body=json.dumps(_PAYLOAD).encode("utf-8"),
            final_url=url,
            egress=EGRESS_DIRECT,
            elapsed_seconds=0.01,
            headers={"content-type": "application/json"},
        )


# parity-exempt: stands in for the one NatsClient method a short-call dispatch answers through (publish_reply); the wider connection lifecycle has no branch in this test's path  # noqa: E501
class _FakeNats:
    """captures what the pod published, as bytes wherever the transport carries bytes."""

    def __init__(self) -> None:
        self.replies: list[tuple[str, Any]] = []

    async def publish(self, *, subject: Any, message: Any, reply_to: Any = None) -> None:
        del subject, message, reply_to

    async def jetstream_publish(self, *, subject: Any, payload: bytes) -> None:
        del subject, payload

    async def publish_reply(self, *, reply_subject: str, message: Any) -> None:
        self.replies.append((reply_subject, message))


def _server(nats: _FakeNats, *, status_code: int = 200) -> ToolServer:
    """a real pod serving the real web_search tool over a stub transport."""
    server = ToolServer(
        namespace=_NS,
        nats_client=nats,  # type: ignore[arg-type]
        pod_id=_POD,
        namespace_collection=None,
        jwks_provider=_pod_jwks_provider,
        assertion_replay_guard=_PodReplayGuard(),
    )
    server.register(WebSearchTool(base_url=_BASE_URL, transport=_StubSearchTransport(status_code=status_code)))
    return server


def _call() -> IncomingMessage:
    payload = _signed_call_payload(
        pod_id=_POD,
        tool_name="threetears.web_search",
        tool_version="1.0",
        arguments={"query": "otter husbandry"},
        conversation_id=uuid4(),
        user_id=uuid4(),
    )
    return IncomingMessage(
        data=json.dumps(payload).encode("utf-8"),
        reply_subject="_INBOX_registry_reg-1.abc",
        subject=f"{_NS}.tools.internal.{_POD}",
    )


async def _dispatched(*, status_code: int = 200) -> dict[str, Any]:
    """run one real dispatch and return the response as it left the pod, via JSON.

    The round-trip through ``model_dump_json`` is the point: it is what the
    NATS client does to a ``CallResponse`` on the way out, and it is where a
    field that cannot serialize would be lost.
    """
    nats = _FakeNats()
    server = _server(nats, status_code=status_code)

    await server.handle_call(_call())

    assert len(nats.replies) == 1
    _subject, response = nats.replies[0]
    decoded: dict[str, Any] = json.loads(response.model_dump_json())
    return decoded


@pytest.mark.asyncio
async def test_the_named_key_is_present_on_the_wire() -> None:
    """check 8, in its narrowest form: the key exists in the published payload."""
    decoded = await _dispatched()

    assert decoded["metadata"] is not None
    assert SEARCH_RESULTS_METADATA_KEY in decoded["metadata"]


@pytest.mark.asyncio
async def test_a_consumer_reconstructs_the_projection_from_the_wire_bytes() -> None:
    """the payload is not merely present but READABLE by the contract's own reader."""
    decoded = await _dispatched()

    projection = SearchResultsMetadata.from_metadata(decoded["metadata"][SEARCH_RESULTS_METADATA_KEY])

    assert projection.query == "otter husbandry"
    assert [candidate.title for candidate in projection.candidates] == ["Otter husbandry"]
    assert projection.candidates[0].locators[0].url == "https://example.test/otters"


@pytest.mark.asyncio
async def test_the_typed_details_survive_rather_than_flattening_to_strings() -> None:
    """the parts a prose-parsing consumer could never recover: scores, provenance, spend."""
    decoded = await _dispatched()

    projection = SearchResultsMetadata.from_metadata(decoded["metadata"][SEARCH_RESULTS_METADATA_KEY])
    candidate = projection.candidates[0]

    assert [entry.name for entry in candidate.scores] == ["engine-fusion-weight"]
    assert candidate.provenance.egress == EGRESS_DIRECT
    assert candidate.provenance.retrieved_at.tzinfo is not None
    assert projection.spend is not None


@pytest.mark.asyncio
async def test_the_schema_version_rides_along_so_a_reader_can_refuse() -> None:
    """D13: a reader meeting a newer payload must be able to say so before parsing it."""
    decoded = await _dispatched()

    assert decoded["metadata"][SEARCH_RESULTS_METADATA_KEY]["schema_version"] == 1


@pytest.mark.asyncio
async def test_a_failed_search_carries_its_typed_failure_across_too() -> None:
    """D10/SR-E3: the far side learns WHICH failure class refused, and what it cost.

    The non-vacuous half of check 8. A metadata carry that only worked on the
    success path would leave the exact case a consumer most needs structure
    for -- a failure it has to decide how to react to -- back on prose.
    """
    decoded = await _dispatched(status_code=500)

    assert decoded["success"] is False
    projection = SearchResultsMetadata.from_metadata(decoded["metadata"][SEARCH_RESULTS_METADATA_KEY])
    assert projection.failure is not None
    assert projection.failure.failure_class
    assert not decoded["content"].startswith("[TOOL ERROR]")
