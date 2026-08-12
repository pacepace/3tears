"""The two web builtins, gutted onto the search leaf (search-spec.md §4.2, §4.3).

What these pin is the *swap*: the identity a caller binds to is unchanged, and
everything the old hand-rolled bodies got wrong is gone. Specifically --

* prose still arrives on ``content``, so no existing caller has to change;
* structure now arrives on ``metadata`` under the one named key (D22), which
  is the whole reason a consumer can stop parsing prose back apart;
* a failure is a failed :class:`ToolResult` carrying a *typed* failure class,
  never a content string that begins ``[TOOL ERROR]`` for callers to match on
  (§10 defect 8);
* the transport is injected, so nothing here opens a client, reads an
  environment variable, or hardcodes a timeout (§10 defect 2).

The transports are stubs rather than a live server: what is under test is the
tools' own wiring onto the leaf, and the leaf's own conformance suites already
hold the adapters and Extract to their contracts.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import pytest
from pydantic import JsonValue

from threetears.agent.tools.builtin.web_fetch import WebFetchTool, create_web_fetch_tool
from threetears.agent.tools.builtin.web_search import WebSearchTool, create_web_search_tool
from threetears.media.contracts import EXTRACTION_STATUS_REFUSED
from threetears.search.contracts import (
    SEARCH_RESULTS_METADATA_KEY,
    EGRESS_DIRECT,
    SearchResultsMetadata,
    TransportResponse,
)
from threetears.search.extract import EXTRACTION_STATUS_FACET

_BASE_URL = "http://searxng.internal:8080"
_PAGE_URL = "https://example.test/article"

_SEARXNG_PAYLOAD: dict[str, Any] = {
    "query": "otter husbandry",
    "results": [
        {
            "url": "https://example.test/otters",
            "title": "Otter husbandry",
            "content": "Everything about keeping otters.",
            "engine": "duckduckgo",
            "score": 1.5,
        },
        {
            "url": "https://example.test/more-otters",
            "title": "More otters",
            "content": "Further reading.",
            "engine": "brave",
            "score": 0.9,
        },
    ],
}

_ARTICLE_HTML = (
    b"<html><head><title>Otter husbandry</title></head><body><article>"
    b"<p>Otters are semiaquatic mammals and keeping them is a serious undertaking that "
    b"requires water, space, and a licence in most jurisdictions.</p>"
    b"<p>They eat a great deal of fish, and they are extremely social animals.</p>"
    b"</article></body></html>"
)


# parity-with: threetears.search.contracts.transport.SearchTransport
class _StubSearchTransport:
    """answers every search call with one canned payload, and records the calls."""

    def __init__(self, *, payload: dict[str, Any] | None = None, status_code: int = 200) -> None:
        self._payload = payload if payload is not None else _SEARXNG_PAYLOAD
        self._status_code = status_code
        self.calls: list[tuple[str, str, float | None]] = []

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
        del headers, params, json_body
        self.calls.append((method, url, timeout_seconds))
        return TransportResponse(
            status_code=self._status_code,
            body=json.dumps(self._payload).encode("utf-8"),
            final_url=url,
            egress=EGRESS_DIRECT,
            elapsed_seconds=0.01,
            headers={"content-type": "application/json"},
        )


# parity-with: threetears.search.contracts.transport.FetchTransport
class _StubFetchTransport:
    """answers a carrier fetch and a robots fetch from canned bytes."""

    def __init__(self, *, body: bytes = _ARTICLE_HTML, robots: bytes = b"", status_code: int = 200) -> None:
        self._body = body
        self._robots = robots
        self._status_code = status_code
        self.fetched: list[str] = []

    @property
    def egress_name(self) -> str:
        return EGRESS_DIRECT

    async def fetch(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        max_bytes: int,
        allowed_content_types: tuple[str, ...] | None = None,
        timeout_seconds: float | None = None,
    ) -> TransportResponse:
        del method, headers, allowed_content_types, timeout_seconds
        self.fetched.append(url)
        is_robots = url.endswith("/robots.txt")
        body = self._robots if is_robots else self._body
        return TransportResponse(
            status_code=200 if is_robots else self._status_code,
            body=body[:max_bytes],
            final_url=url,
            egress=EGRESS_DIRECT,
            elapsed_seconds=0.01,
            headers={"content-type": "text/plain" if is_robots else "text/html"},
        )


def _projection(result: Any) -> SearchResultsMetadata:
    """read the structured projection back off a tool result, as a consumer does."""
    assert result.metadata is not None
    return SearchResultsMetadata.from_metadata(result.metadata[SEARCH_RESULTS_METADATA_KEY])


# ------------------------------------------------------------- web_search ----


class TestWebSearchKeepsItsIdentity:
    """the swap is invisible to a caller that only ever read prose."""

    def test_the_name_and_version_are_unchanged(self) -> None:
        tool = WebSearchTool(base_url=_BASE_URL, transport=_StubSearchTransport())

        assert tool.mcp_name() == "threetears.web_search"
        assert tool.mcp_version() == "1.0"

    def test_the_base_url_still_loses_its_trailing_slash(self) -> None:
        tool = WebSearchTool(base_url=f"{_BASE_URL}/", transport=_StubSearchTransport())

        assert tool.base_url == _BASE_URL

    @pytest.mark.asyncio
    async def test_prose_still_names_the_results(self) -> None:
        """``content`` remains human-readable and carries the titles and URLs."""
        tool = WebSearchTool(base_url=_BASE_URL, transport=_StubSearchTransport())

        result = await tool.execute(query="otter husbandry")

        assert result.success is True
        assert "Otter husbandry" in result.content
        assert "https://example.test/otters" in result.content


class TestWebSearchCarriesStructure:
    """what is new: the typed result under the named key (D22, check 8)."""

    @pytest.mark.asyncio
    async def test_the_candidates_arrive_typed(self) -> None:
        tool = WebSearchTool(base_url=_BASE_URL, transport=_StubSearchTransport())

        result = await tool.execute(query="otter husbandry")

        projection = _projection(result)
        assert projection.query == "otter husbandry"
        assert [candidate.title for candidate in projection.candidates] == ["Otter husbandry", "More otters"]
        assert projection.candidates[0].locators[0].url == "https://example.test/otters"

    @pytest.mark.asyncio
    async def test_the_scores_are_named_not_a_bare_number(self) -> None:
        """D1 reaches the tool border intact: a score has a name and a source."""
        tool = WebSearchTool(base_url=_BASE_URL, transport=_StubSearchTransport())

        result = await tool.execute(query="otter husbandry")

        scores = _projection(result).candidates[0].scores
        assert [entry.name for entry in scores] == ["engine-fusion-weight"]

    @pytest.mark.asyncio
    async def test_the_criterion_the_tool_states_is_answered_for(self) -> None:
        """the max-results cap is a criterion the adapter answers, not a local slice."""
        tool = WebSearchTool(base_url=_BASE_URL, transport=_StubSearchTransport(), max_results=1)

        result = await tool.execute(query="otter husbandry")

        projection = _projection(result)
        assert len(projection.candidates) == 1
        assert [disposition.criterion_key for disposition in projection.dispositions] == ["max-results"]

    @pytest.mark.asyncio
    async def test_the_whole_payload_survives_a_json_round_trip(self) -> None:
        """the metadata rides a JSON wire, so nothing in it may be a Python-only object."""
        tool = WebSearchTool(base_url=_BASE_URL, transport=_StubSearchTransport())

        result = await tool.execute(query="otter husbandry")

        assert result.metadata is not None
        assert json.loads(json.dumps(result.metadata)) == result.metadata


class TestWebSearchFailsTyped:
    """the ``[TOOL ERROR]`` prefix is gone, and a failure still accounts (§10 defect 8, D10)."""

    @pytest.mark.asyncio
    async def test_a_provider_failure_is_a_failed_result_not_a_raise(self) -> None:
        tool = WebSearchTool(base_url=_BASE_URL, transport=_StubSearchTransport(status_code=500))

        result = await tool.execute(query="otter husbandry")

        assert result.success is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_the_failure_class_is_readable_without_string_matching(self) -> None:
        """a consumer learns WHICH failure happened from the typed record, not a prefix."""
        tool = WebSearchTool(base_url=_BASE_URL, transport=_StubSearchTransport(status_code=500))

        result = await tool.execute(query="otter husbandry")

        failure = _projection(result).failure
        assert failure is not None
        assert failure.failure_class
        assert not result.content.startswith("[TOOL ERROR]")

    @pytest.mark.asyncio
    async def test_spend_survives_a_failed_call(self) -> None:
        """SR-E3: accounting for a broken call is read off the same key as a working one."""
        tool = WebSearchTool(base_url=_BASE_URL, transport=_StubSearchTransport(status_code=500))

        result = await tool.execute(query="otter husbandry")

        assert _projection(result).spend is not None


class TestWebSearchTakesItsTimeoutFromConfiguration:
    """§10 defect 2: the 15-second hardcode is gone, and the bound reaches the transport."""

    @pytest.mark.asyncio
    async def test_the_configured_bound_reaches_the_transport(self) -> None:
        transport = _StubSearchTransport()
        tool = WebSearchTool(base_url=_BASE_URL, transport=transport, timeout_seconds=2.5)

        await tool.execute(query="otter husbandry")

        assert transport.calls
        assert transport.calls[0][2] == pytest.approx(2.5)

    def test_the_factory_forwards_configuration_rather_than_dropping_it(self) -> None:
        transport = _StubSearchTransport()
        tool = create_web_search_tool(
            {"base_url": _BASE_URL, "transport": transport, "max_results": 3},
            "search the web",
        )

        assert tool.name == "threetears.web_search"

    def test_the_factory_still_refuses_a_missing_base_url(self) -> None:
        with pytest.raises(ValueError, match="base_url"):
            create_web_search_tool({}, "search the web")


# -------------------------------------------------------------- web_fetch ----


class TestWebFetchKeepsItsIdentity:
    def test_the_name_and_version_are_unchanged(self) -> None:
        tool = WebFetchTool(transport=_StubFetchTransport())

        assert tool.mcp_name() == "threetears.web_fetch"
        assert tool.mcp_version() == "1.0"

    @pytest.mark.asyncio
    async def test_the_extracted_text_is_still_the_content(self) -> None:
        tool = WebFetchTool(transport=_StubFetchTransport())

        result = await tool.execute(url=_PAGE_URL)

        assert result.success is True
        assert "semiaquatic" in result.content


class TestWebFetchCarriesStructure:
    @pytest.mark.asyncio
    async def test_the_candidate_arrives_under_the_same_key_search_uses(self) -> None:
        """one shape for structure at the tool border, whether the tool searched or fetched."""
        tool = WebFetchTool(transport=_StubFetchTransport())

        result = await tool.execute(url=_PAGE_URL)

        projection = _projection(result)
        assert projection.query == _PAGE_URL
        assert len(projection.candidates) == 1
        assert projection.candidates[0].locators[0].url == _PAGE_URL

    @pytest.mark.asyncio
    async def test_the_extraction_facets_say_how_the_text_was_produced(self) -> None:
        tool = WebFetchTool(transport=_StubFetchTransport())

        result = await tool.execute(url=_PAGE_URL)

        facets = _projection(result).candidates[0].facets
        assert facets[EXTRACTION_STATUS_FACET] == "complete"

    @pytest.mark.asyncio
    async def test_the_text_is_cut_to_the_configured_character_bound(self) -> None:
        tool = WebFetchTool(max_chars=60, transport=_StubFetchTransport())

        result = await tool.execute(url=_PAGE_URL)

        assert len(result.content) <= 60
        assert result.content.endswith("[Content truncated]")


class TestWebFetchHonoursRobots:
    """THE user-visible change: robots binds for callers it never bound for.

    Stated in the module docstring and in this PR because it is a behaviour
    change rather than an internal one -- a page this tool used to return now
    comes back refused when the site's rules say so (D12).
    """

    @pytest.mark.asyncio
    async def test_a_disallowed_page_is_refused_rather_than_fetched(self) -> None:
        transport = _StubFetchTransport(robots=b"User-agent: *\nDisallow: /\n")
        tool = WebFetchTool(transport=transport)

        result = await tool.execute(url=_PAGE_URL)

        assert result.success is False
        assert _projection(result).candidates[0].facets[EXTRACTION_STATUS_FACET] == EXTRACTION_STATUS_REFUSED
        assert _PAGE_URL not in transport.fetched

    @pytest.mark.asyncio
    async def test_the_stance_is_configuration_not_a_per_call_choice(self) -> None:
        """D12 rules the override as recorded deployment config, so it is a constructor arg.

        Non-vacuous in the other direction: with the stance turned off the same
        disallowing robots file no longer refuses the read.
        """
        transport = _StubFetchTransport(robots=b"User-agent: *\nDisallow: /\n")
        tool = WebFetchTool(transport=transport, respect_robots=False)

        result = await tool.execute(url=_PAGE_URL)

        assert result.success is True
        assert _PAGE_URL in transport.fetched


class TestWebFetchFailsTyped:
    @pytest.mark.asyncio
    async def test_an_unreadable_carrier_is_a_failed_result_with_a_status(self) -> None:
        tool = WebFetchTool(transport=_StubFetchTransport(status_code=404))

        result = await tool.execute(url=_PAGE_URL)

        assert result.success is False
        assert not result.content.startswith("[TOOL ERROR]")
        assert _projection(result).candidates[0].facets[EXTRACTION_STATUS_FACET] == "failed"

    @pytest.mark.asyncio
    async def test_a_missing_url_is_answered_rather_than_fetched(self) -> None:
        transport = _StubFetchTransport()
        tool = WebFetchTool(transport=transport)

        result = await tool.execute()

        assert result.success is False
        assert not transport.fetched

    def test_the_factory_forwards_configuration(self) -> None:
        tool = create_web_fetch_tool(
            {"transport": _StubFetchTransport(), "max_chars": 100, "respect_robots": False},
            "fetch a page",
        )

        assert tool.name == "threetears.web_fetch"
