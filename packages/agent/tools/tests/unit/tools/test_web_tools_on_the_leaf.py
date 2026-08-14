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

Most transports here are stubs rather than a live server: what is under test
is the tools' own wiring onto the leaf, and the leaf's own conformance suites
already hold the adapters and Extract to their contracts.

**But a stub cannot test the transport the tool builds when nobody injects
one**, and that is the shape production actually runs -- ``WebFetchTool()``
with ``transport=None``. Stubbing it everywhere is how this suite shipped a
tool whose default transport refused to follow redirects: the tool's tests
passed because the stub always answered 200, and the transport's tests passed
because they never served a response. ``TestTheDefaultWiringMeetsARealServer``
closes that seam by injecting nothing and answering over a socket.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import pytest
from pydantic import JsonValue

from threetears.agent.tools.builtin.web_fetch import WebFetchTool, create_web_fetch_tool
from threetears.agent.tools.builtin.web_search import WebSearchTool, create_web_search_tool
from threetears.media.contracts import EXTRACTION_STATUS_COMPLETE, EXTRACTION_STATUS_REFUSED
from threetears.search.contracts import (
    SEARCH_RESULTS_METADATA_KEY,
    EGRESS_DIRECT,
    LocalCapExceeded,
    SearchResultsMetadata,
    Spend,
    TransportResponse,
)
from threetears.search.extract import (
    EXTRACTION_STATUS_FACET,
    EXTRACTOR_UNAVAILABLE_SCOPE,
)
from threetears.search.testing import LocalHttpServer, Reply

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


def _no_extractor_installed() -> Any:
    """stand in for ``_load_extractor`` on a host without the ``[fetch]`` extra.

    Raises exactly what the real loader raises there, so the refusal under
    test is the shipped one rather than a test's idea of it.
    """
    raise LocalCapExceeded(
        "no HTML extractor is installed",
        spend=Spend(),
        remediation="install 3tears-agent-tools[fetch] to extract page content",
        scope=EXTRACTOR_UNAVAILABLE_SCOPE,
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

    @pytest.mark.asyncio
    async def test_a_whole_run_refusal_carries_the_typed_record_not_just_prose(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The refusal this tool's own docstring is about, read as a consumer reads it.

        ``extract`` raises for a refusal that applies to the whole run --
        today, a missing ``[fetch]`` extra. That is precisely the case a
        structure-reading consumer must not have to parse prose for, and it
        was the one path with no test: the projection was built from an empty
        candidate set, which never populates ``failure``.
        """
        monkeypatch.setattr(
            "threetears.search.extract._load_extractor",
            _no_extractor_installed,
        )
        tool = WebFetchTool(transport=_StubFetchTransport())

        result = await tool.execute(url=_PAGE_URL)

        assert result.success is False
        projection = _projection(result)
        assert projection.failure is not None, "a refusal must reach the border as a record"
        assert projection.failure.failure_class
        assert projection.failure.scope == EXTRACTOR_UNAVAILABLE_SCOPE
        assert "fetch" in (projection.failure.remediation or "")

    @pytest.mark.asyncio
    async def test_a_missing_url_also_answers_with_a_readable_projection(self) -> None:
        """The other prose-only path: an argument fault is still a shape a consumer can read."""
        tool = WebFetchTool(transport=_StubFetchTransport())

        result = await tool.execute()

        projection = _projection(result)
        assert projection.query == ""
        assert not projection.candidates


class TestTheCharacterBoundHoldsAtItsEdges:
    """``max_chars`` is deployment config, so it must hold for the values a deployment can set.

    The existing pin uses 60, comfortably past the truncation marker's own
    length. Under it the arithmetic goes negative and the slice runs from the
    *end* of the string, returning more than the bound rather than less.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize("max_chars", [1, 10, 21, 22, 23, 200])
    async def test_the_result_never_exceeds_the_bound(self, max_chars: int) -> None:
        tool = WebFetchTool(max_chars=max_chars, transport=_StubFetchTransport())

        result = await tool.execute(url=_PAGE_URL)

        assert len(result.content) <= max_chars, f"max_chars={max_chars} returned {len(result.content)} chars"

    @pytest.mark.asyncio
    async def test_a_bound_under_the_marker_does_not_return_the_tail_of_the_page(self) -> None:
        """The specific corruption: a negative slice keeps the END of the text."""
        tool = WebFetchTool(max_chars=10, transport=_StubFetchTransport())

        result = await tool.execute(url=_PAGE_URL)

        assert "social animals" not in result.content


class TestTheDefaultWiringMeetsARealServer:
    """``WebFetchTool()`` -- no injected transport, a real socket, real bytes.

    This is the configuration a pod runs and the one no other test in this
    file exercises. Every assertion here is about the *host's choice* of
    transport configuration rather than about the tool's branching, so a
    change to ``build_fetch_transport``'s defaults has to come through here.

    Loopback needs ``allow_private_addresses``, which is deployment config a
    test is entitled to set (D21) -- so the tool is constructed with a
    transport built the way the default builds it, plus that one flag. The
    redirect policy, the byte cap and the content-type gate all remain the
    builder's own.
    """

    @staticmethod
    def _tool(**kwargs: Any) -> WebFetchTool:
        """the production default, with only the loopback guard relaxed."""
        from threetears.agent.tools.search_transport import build_fetch_transport

        return WebFetchTool(transport=build_fetch_transport(allow_private_addresses=True), **kwargs)

    @pytest.mark.asyncio
    async def test_a_page_behind_a_canonicalising_redirect_still_extracts(self) -> None:
        """http->https, www, trailing slash: most of the real web arrives via a 301."""
        async with LocalHttpServer(
            (
                Reply(status=200, body=b"User-agent: *\nAllow: /\n", headers={"Content-Type": "text/plain"}),
                Reply(status=301, headers={"Location": "/article", "Content-Length": "0"}, body=b""),
                Reply(status=200, body=_ARTICLE_HTML, headers={"Content-Type": "text/html"}),
            )
        ) as server:
            result = await self._tool().execute(url=f"{server.base_url}/")

        assert result.success is True, result.content
        assert "semiaquatic" in result.content
        assert _projection(result).candidates[0].facets[EXTRACTION_STATUS_FACET] == EXTRACTION_STATUS_COMPLETE

    @pytest.mark.asyncio
    async def test_a_plain_page_extracts_without_any_redirect(self) -> None:
        """Non-vacuous baseline: the seam works at all, so the redirect pin means something."""
        async with LocalHttpServer(
            (
                Reply(status=200, body=b"User-agent: *\nAllow: /\n", headers={"Content-Type": "text/plain"}),
                Reply(status=200, body=_ARTICLE_HTML, headers={"Content-Type": "text/html"}),
            )
        ) as server:
            result = await self._tool().execute(url=f"{server.base_url}/article")

        assert result.success is True, result.content
        assert "semiaquatic" in result.content

    @pytest.mark.asyncio
    async def test_a_robots_file_that_merely_moved_does_not_refuse_the_page(self) -> None:
        """The second route to the same defect: robots is fetched through this transport too."""
        async with LocalHttpServer(
            (
                Reply(status=301, headers={"Location": "/robots.txt", "Content-Length": "0"}, body=b""),
                Reply(status=200, body=b"User-agent: *\nAllow: /\n", headers={"Content-Type": "text/plain"}),
                Reply(status=200, body=_ARTICLE_HTML, headers={"Content-Type": "text/html"}),
            )
        ) as server:
            result = await self._tool().execute(url=f"{server.base_url}/article")

        assert result.success is True, result.content

    @pytest.mark.asyncio
    async def test_a_robots_file_that_refuses_is_still_binding_over_a_real_socket(self) -> None:
        """D12 holds on the real transport, not only against the stub."""
        async with LocalHttpServer(
            (Reply(status=200, body=b"User-agent: *\nDisallow: /\n", headers={"Content-Type": "text/plain"}),)
        ) as server:
            result = await self._tool().execute(url=f"{server.base_url}/article")

        assert result.success is False
        assert _projection(result).candidates[0].facets[EXTRACTION_STATUS_FACET] == EXTRACTION_STATUS_REFUSED

    @pytest.mark.asyncio
    async def test_a_carrier_the_gate_refuses_never_reaches_the_extractor(self) -> None:
        """The content-type gate is the builder's obligation, and it fires before the body."""
        async with LocalHttpServer(
            (
                Reply(status=200, body=b"User-agent: *\nAllow: /\n", headers={"Content-Type": "text/plain"}),
                Reply(status=200, body=b"\x00\x01\x02binary", headers={"Content-Type": "application/octet-stream"}),
            )
        ) as server:
            result = await self._tool().execute(url=f"{server.base_url}/blob")

        assert result.success is False
        assert not result.content.startswith("[TOOL ERROR]")

    @pytest.mark.asyncio
    async def test_a_body_past_the_cap_is_refused_rather_than_held(self) -> None:
        """SR-G5 on the real path: the cap the host configured is the cap that fires."""
        async with LocalHttpServer(
            (
                Reply(status=200, body=b"User-agent: *\nAllow: /\n", headers={"Content-Type": "text/plain"}),
                Reply(status=200, body=b"x" * 40000, headers={"Content-Type": "text/html"}),
            )
        ) as server:
            result = await self._tool(max_bytes=1024).execute(url=f"{server.base_url}/huge")

        assert result.success is False

    @pytest.mark.asyncio
    async def test_a_loopback_url_is_refused_by_the_untouched_default(self) -> None:
        """The one flag these tests relax is genuinely load-bearing when it is not set."""
        tool = WebFetchTool()

        result = await tool.execute(url="http://127.0.0.1:9/article")

        assert result.success is False


class TestRegistrationSkipsAToolItCannotServe:
    """The skip-with-reason pattern, on the registry path as well as the pod path.

    ``serve.py`` probes for the extractor because Extract imports it lazily:
    without the extra the tool constructs fine and refuses every call, so an
    ``ImportError`` never arrives to be caught. ``register_builtins`` was left
    catching only that ``ImportError``, which means it registered a
    ``web_fetch`` that fails every request -- the exact condition the probe
    was added to prevent, reached by the other door.
    """

    @staticmethod
    def _registry_without_an_extractor(monkeypatch: pytest.MonkeyPatch) -> Any:
        """register onto a fresh registry with ``trafilatura`` reported absent."""
        import threetears.agent.tools.builtin as builtin_pkg
        from threetears.agent.tools.registry import ToolRegistry

        real_find_spec = builtin_pkg.find_spec

        def _absent(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "trafilatura":
                return None
            return real_find_spec(name, *args, **kwargs)

        monkeypatch.setattr(builtin_pkg, "find_spec", _absent)
        registry = ToolRegistry()
        builtin_pkg.register_builtins(registry)
        return registry

    def test_web_fetch_is_skipped_when_no_extractor_is_installed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        registry = self._registry_without_an_extractor(monkeypatch)

        assert "web_fetch" not in registry.list_types()

    def test_the_other_builtins_still_register(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Non-vacuous: the probe skips one tool, it does not break registration."""
        registry = self._registry_without_an_extractor(monkeypatch)

        assert "calculator" in registry.list_types()
        assert "web_search" in registry.list_types()

    def test_web_fetch_registers_when_the_extractor_is_present(self) -> None:
        """The other direction: with the extra installed the tool is served as before."""
        from importlib.util import find_spec

        from threetears.agent.tools.builtin import register_builtins
        from threetears.agent.tools.registry import ToolRegistry

        if find_spec("trafilatura") is None:
            pytest.skip("this direction needs 3tears-agent-tools[fetch] installed")
        registry = ToolRegistry()

        register_builtins(registry)

        assert "web_fetch" in registry.list_types()


def _socket_transport() -> Any:
    """A real transport pointed at the loopback test server.

    The conditional headers are the thing under test, so they must cross a real
    socket -- a stub that records its arguments would prove the parameter was
    passed, not that the request was conditional.
    """
    from threetears.search.standalone import StandaloneTransport

    return StandaloneTransport(allowed_hosts=("127.0.0.1",), allow_private_addresses=True)


class TestConditionalRevalidationAtTheToolBorder:
    """D30 at the tool face: validators in, ``unchanged`` out.

    The caller holds the bytes, so it sends the VALIDATORS without its copy of
    the text -- sending the body back to revalidate it would spend exactly the
    bytes the conditional request exists to save.
    """

    async def test_validators_make_the_fetch_conditional(self) -> None:
        from threetears.search.testing.http_server import LocalHttpServer, Reply

        async with LocalHttpServer((Reply(status=304, body=b""),)) as server:
            tool = WebFetchTool(transport=_socket_transport(), respect_robots=False)
            await tool.execute(url=f"{server.base_url}/page", etag='"v1"')
            requests = list(server.requests)

        assert any("if-none-match" in request.lower() for request in requests), requests

    async def test_unchanged_is_a_success_not_a_failure(self) -> None:
        """The reader fix: 'no content' is the one outcome that means good."""
        from threetears.search.testing.http_server import LocalHttpServer, Reply

        async with LocalHttpServer((Reply(status=304, body=b""),)) as server:
            tool = WebFetchTool(transport=_socket_transport(), respect_robots=False)
            result = await tool.execute(url=f"{server.base_url}/page", etag='"v1"')

        assert result.success is True
        assert "unchanged" in result.content.lower()

    async def test_the_typed_status_says_unchanged(self) -> None:
        from threetears.media.contracts import EXTRACTION_STATUS_UNCHANGED
        from threetears.search.contracts import SEARCH_RESULTS_METADATA_KEY
        from threetears.search.extract import EXTRACTION_STATUS_FACET
        from threetears.search.testing.http_server import LocalHttpServer, Reply

        async with LocalHttpServer((Reply(status=304, body=b""),)) as server:
            tool = WebFetchTool(transport=_socket_transport(), respect_robots=False)
            result = await tool.execute(url=f"{server.base_url}/page", etag='"v1"')

        assert result.metadata is not None
        payload = result.metadata[SEARCH_RESULTS_METADATA_KEY]
        facets = payload["candidates"][0]["facets"]
        assert facets[EXTRACTION_STATUS_FACET] == EXTRACTION_STATUS_UNCHANGED

    async def test_no_validators_means_no_conditional_headers(self) -> None:
        from threetears.search.testing.http_server import LocalHttpServer, Reply

        page = b"<html><body><article><p>" + b"Readable body text here. " * 20 + b"</p></article></body></html>"
        reply = Reply(status=200, body=page, headers={"Content-Type": "text/html"})
        async with LocalHttpServer((reply,)) as server:
            tool = WebFetchTool(transport=_socket_transport(), respect_robots=False)
            await tool.execute(url=f"{server.base_url}/page")
            requests = list(server.requests)

        joined = " ".join(requests).lower()
        assert "if-none-match" not in joined, requests
        assert "if-modified-since" not in joined, requests
