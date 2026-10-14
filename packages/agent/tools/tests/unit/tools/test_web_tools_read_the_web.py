"""``web_search`` and ``web_fetch`` on recorded pages: what the agent is told, and what each guard lets through.

The pages and the search answer are recordings under ``fixtures/web``, served
through :class:`threetears.search.testing.RecordedFetchTransport` (the real
transport's cap and gate) and
:class:`threetears.search.testing.ScriptedTransport`. Nothing here touches
the network.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from threetears.agent.tools.builtin import web_fetch as web_fetch_module
from threetears.agent.tools.builtin.web_fetch import WebFetchTool, create_web_fetch_tool, pdf_reader
from threetears.agent.tools.builtin.web_search import WebSearchTool, create_web_search_tool
from threetears.search.bind import DISPLAY_ZONE_METADATA_KEY
from threetears.search.contracts import SEARCH_RESULTS_METADATA_KEY, Spend, TimedOut
from threetears.search.testing import RecordedFetchTransport, RecordedPage, ScriptedTransport, TransportScript

_FIXTURES = Path(__file__).parent / "fixtures" / "web"
_SITE = "https://news.example.org"
_ROBOTS = f"{_SITE}/robots.txt"
_ALLOW = RecordedPage(body=b"User-agent: *\nAllow: /\n", content_type="text/plain")


def _fixture(name: str) -> bytes:
    return (_FIXTURES / name).read_bytes()


def _search(**kwargs: Any) -> tuple[WebSearchTool, ScriptedTransport]:
    transport = ScriptedTransport(
        [TransportScript(body=_fixture("searxng-otters.json"), headers={"content-type": "application/json"})]
    )
    return WebSearchTool(base_url="http://searxng.internal:8080", transport=transport, **kwargs), transport


def _params(transport: ScriptedTransport) -> dict[str, str]:
    return dict(transport.calls[0]["params"])  # type: ignore[arg-type]


# --- web_search -----------------------------------------------------------


class TestSearchTakesItsFilters:
    @pytest.mark.asyncio
    async def test_a_plain_query_sends_only_the_query(self) -> None:
        """Good traffic: no filter asked for, none sent."""
        tool, transport = _search()

        result = await tool.execute(query="otters river")

        assert result.success is True
        assert _params(transport) == {"q": "otters river", "format": "json"}

    @pytest.mark.asyncio
    async def test_category_time_range_page_and_site_reach_searxng(self) -> None:
        tool, transport = _search()

        result = await tool.execute(
            query="otters river", category="news", time_range="week", page=2, site="https://www.Example.org/news"
        )

        assert result.success is True
        params = _params(transport)
        assert params["categories"] == "news"
        assert params["time_range"] == "week"
        assert params["pageno"] == "2"
        assert params["q"] == "otters river site:example.org"

    @pytest.mark.asyncio
    async def test_site_keeps_only_that_site_s_results(self) -> None:
        """An engine that ignores ``site:`` still cannot hand back another site's page."""
        tool, _ = _search()

        result = await tool.execute(query="otters river", site="example.org")

        urls = [c["identity"] for c in result.metadata[SEARCH_RESULTS_METADATA_KEY]["candidates"]]
        assert urls == ["https://news.example.org/otters-return"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("arguments", "said"),
        [
            ({"category": "gossip"}, "web_search has no category 'gossip'. The categories: general"),
            ({"time_range": "decade"}, "web_search has no time_range 'decade'. It takes day, week, month, year"),
            ({"page": 0}, "web_search's page is a whole number from 1; 0 is not."),
            ({"site": "not a site"}, "web_search's site is a domain such as nature.com; 'not a site' is not one."),
        ],
    )
    async def test_an_argument_it_cannot_take_is_answered_before_anything_is_sent(
        self, arguments: dict[str, Any], said: str
    ) -> None:
        tool, transport = _search()

        result = await tool.execute(query="otters", **arguments)

        assert result.success is False
        assert result.content.startswith(said)
        assert not transport.calls

    @pytest.mark.asyncio
    async def test_the_schema_names_every_filter(self) -> None:
        tool, _ = _search()

        properties = tool.mcp_schema().input_schema["properties"]

        assert set(properties) == {"query", "category", "time_range", "page", "site"}
        assert properties["time_range"]["enum"] == ["day", "week", "month", "year"]


class TestSearchResultsSayTheirDateAndEngine:
    @pytest.mark.asyncio
    async def test_the_first_line_says_what_was_searched_and_how_many_came_back(self) -> None:
        tool, _ = _search()

        result = await tool.execute(query="otters river", category="news", page=1)

        assert result.content.splitlines()[0] == 'Searched the web for "otters river" in news: 2 result(s) on page 1.'

    @pytest.mark.asyncio
    async def test_a_full_page_names_the_next_one(self) -> None:
        tool, _ = _search(max_results=2)

        result = await tool.execute(query="otters river")

        assert result.content.splitlines()[0].endswith("More may be on page 2.")

    @pytest.mark.asyncio
    async def test_each_result_carries_its_date_in_the_person_s_zone_and_its_engines(self) -> None:
        tool, _ = _search(zone="America/New_York")

        result = await tool.execute(query="otters river")

        assert "   published 2026-09-14 04:30 EDT; found by duckduckgo, bing news" in result.content
        assert "   no date given; found by wikipedia" in result.content
        assert result.metadata[DISPLAY_ZONE_METADATA_KEY] == "America/New_York"

    def test_the_factory_takes_the_injected_zone(self) -> None:
        tool = create_web_search_tool(
            {"base_url": "http://searxng.internal:8080", "_user_timezone": "Europe/Dublin"}, "search"
        )

        assert tool.name == "threetears.web_search"


# --- web_fetch ------------------------------------------------------------


def _fetch_tool(pages: dict[str, RecordedPage], **kwargs: Any) -> tuple[WebFetchTool, RecordedFetchTransport]:
    transport = RecordedFetchTransport(pages)
    return WebFetchTool(transport=transport, **kwargs), transport


class TestAFetchOpensWithWhatItRead:
    @pytest.mark.asyncio
    async def test_title_date_final_address_and_the_characters_come_first(self) -> None:
        tool, _ = _fetch_tool(
            {
                _ROBOTS: _ALLOW,
                f"{_SITE}/otters": RecordedPage(body=_fixture("article.html"), final_url=f"{_SITE}/2026/otters"),
            }
        )

        result = await tool.execute(url=f"{_SITE}/otters")

        opening = result.content.splitlines()[0]
        total = result.metadata["window"]["total_chars"]
        assert opening == (
            f'Read "Otters return to the river" (published 2026-09-14) at {_SITE}/2026/otters: '
            f"characters 0-{total:,} of {total:,}."
        )
        assert result.content.split("\n\n", 1)[1].startswith("Otters return to the river\nOtters have returned")

    @pytest.mark.asyncio
    async def test_a_followed_frame_is_said_on_the_second_line(self) -> None:
        tool, _ = _fetch_tool(
            {
                _ROBOTS: _ALLOW,
                f"{_SITE}/issue/3": RecordedPage(body=_fixture("menu-with-frame.html")),
                f"{_SITE}/viewer/otters.html": RecordedPage(body=_fixture("article.html")),
            }
        )

        result = await tool.execute(url=f"{_SITE}/issue/3")

        lines = result.content.splitlines()
        assert lines[0].startswith(f'Read "Otters return to the river" (published 2026-09-14) at {_SITE}/viewer/')
        assert lines[1] == (
            f"The page at {_SITE}/issue/3 held only its menu; this is the article it frames, "
            f"from {_SITE}/viewer/otters.html."
        )

    @pytest.mark.asyncio
    async def test_frame_following_is_deployment_config(self) -> None:
        tool, transport = _fetch_tool(
            {_ROBOTS: _ALLOW, f"{_SITE}/issue/3": RecordedPage(body=_fixture("menu-with-frame.html"))},
            thin_page_chars=0,
        )

        await tool.execute(url=f"{_SITE}/issue/3")

        assert transport.fetched == [_ROBOTS, f"{_SITE}/issue/3"]


class TestAFetchThatReadNothingSaysWhyAndWhatNext:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("page", "said"),
        [
            (
                RecordedPage(status_code=403, body=b"no"),
                "the site refused the request (403 Forbidden); it may turn away automated readers; "
                "find the same facts on another site.",
            ),
            (
                RecordedPage(status_code=429, body=b"", headers={"retry-after": "60"}),
                "the site is turning requests away for now (429 Too Many Requests); it asks to wait 60 s; "
                "wait and try again later, or find the same facts on another site.",
            ),
        ],
    )
    async def test_a_refused_request(self, page: RecordedPage, said: str) -> None:
        tool, _ = _fetch_tool({_ROBOTS: _ALLOW, f"{_SITE}/x": page})

        result = await tool.execute(url=f"{_SITE}/x")

        assert result.success is False
        assert result.content == f"Did not read {_SITE}/x: {said}"

    @pytest.mark.asyncio
    async def test_a_robots_file_that_does_not_answer(self) -> None:
        tool, _ = _fetch_tool({_ROBOTS: RecordedPage(failure=TimedOut("slow", spend=Spend()))})

        result = await tool.execute(url=f"{_SITE}/x")

        assert (
            result.content == f"Did not read {_SITE}/x: robots.txt could not be read (timed out after 5 s); try again."
        )

    @pytest.mark.asyncio
    async def test_a_deployment_may_read_past_an_unreadable_robots_file(self) -> None:
        tool, _ = _fetch_tool(
            {
                _ROBOTS: RecordedPage(failure=TimedOut("slow", spend=Spend())),
                f"{_SITE}/otters": RecordedPage(body=_fixture("article.html")),
            },
            robots_unreadable="allow",
        )

        result = await tool.execute(url=f"{_SITE}/otters")

        assert result.success is True
        assert result.content.splitlines()[1].startswith("robots.txt could not be read (timed out after 5 s)")

    def test_the_unreadable_setting_is_checked_when_the_tool_is_built(self) -> None:
        with pytest.raises(ValueError, match="robots_unreadable"):
            WebFetchTool(transport=RecordedFetchTransport({}), robots_unreadable="sometimes")

    def test_it_is_not_an_argument_of_the_call(self) -> None:
        tool, _ = _fetch_tool({})

        assert "robots_unreadable" not in tool.mcp_schema().input_schema["properties"]


class TestTheUserAgentNamesTheDeployment:
    @pytest.mark.asyncio
    async def test_robots_rules_are_read_for_the_deployment_s_name(self) -> None:
        rules = RecordedPage(body=b"User-agent: metallm\nDisallow: /\n", content_type="text/plain")
        tool, _ = _fetch_tool({_ROBOTS: rules}, user_agent="metallm/0.71.0")

        result = await tool.execute(url=f"{_SITE}/otters")

        assert result.success is False
        assert "robots.txt does not allow it" in result.content

    def test_the_default_transport_sends_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: dict[str, Any] = {}

        def _build(**kwargs: Any) -> RecordedFetchTransport:
            seen.update(kwargs)
            return RecordedFetchTransport({})

        monkeypatch.setattr("threetears.agent.tools.search_transport.build_fetch_transport", _build)

        create_web_fetch_tool({"user_agent": "metallm/0.71.0", "pdf_max_bytes": 30_000_000}, "fetch")

        assert seen["user_agent"] == "metallm/0.71.0"
        assert seen["max_response_bytes"] == 30_000_000, "the transport's ceiling is the largest carrier read"

    @pytest.mark.asyncio
    async def test_every_fetch_sends_it_over_a_real_socket(self) -> None:
        """The robots read and the page read both say who is asking."""
        from threetears.agent.tools.search_transport import build_fetch_transport
        from threetears.search.testing import LocalHttpServer, Reply

        async with LocalHttpServer(
            (
                Reply(status=200, body=b"User-agent: *\nAllow: /\n", headers={"Content-Type": "text/plain"}),
                Reply(status=200, body=_fixture("article.html"), headers={"Content-Type": "text/html"}),
            )
        ) as server:
            tool = WebFetchTool(
                transport=build_fetch_transport(allow_private_addresses=True, user_agent="metallm/0.71.0"),
                user_agent="metallm/0.71.0",
            )
            result = await tool.execute(url=f"{server.base_url}/otters")

        assert result.success is True, result.content
        assert len(server.requests) == 2
        assert all("user-agent: metallm/0.71.0" in request.lower() for request in server.requests)


class TestPdfs:
    @pytest.mark.asyncio
    async def test_a_pdf_is_read_through_the_document_parser(self) -> None:
        tool, _ = _fetch_tool(
            {
                _ROBOTS: _ALLOW,
                f"{_SITE}/survey.pdf": RecordedPage(body=_fixture("otter-survey.pdf"), content_type="application/pdf"),
            }
        )

        result = await tool.execute(url=f"{_SITE}/survey.pdf")

        lines = result.content.splitlines()
        assert result.success is True
        assert lines[0].startswith(f'Read "Otter survey" (published 2025-03-01 12:00 UTC) at {_SITE}/survey.pdf:')
        assert lines[1] == "A PDF of 5 page(s), all read."
        assert "Page 5. Otter populations" in result.content

    @pytest.mark.asyncio
    async def test_the_page_cap_says_what_was_not_read(self) -> None:
        tool, _ = _fetch_tool(
            {
                _ROBOTS: _ALLOW,
                f"{_SITE}/survey.pdf": RecordedPage(body=_fixture("otter-survey.pdf"), content_type="application/pdf"),
            },
            pdf_max_pages=2,
        )

        result = await tool.execute(url=f"{_SITE}/survey.pdf")

        assert result.content.splitlines()[1] == (
            "A PDF of 5 pages: pages 1-2 were read, the most read from one PDF. The rest was not read."
        )
        assert "Page 2." in result.content
        assert "Page 3." not in result.content

    @pytest.mark.asyncio
    async def test_a_pdf_past_its_cap_says_its_own_cap(self) -> None:
        body = _fixture("otter-survey.pdf")
        tool, _ = _fetch_tool(
            {_ROBOTS: _ALLOW, f"{_SITE}/survey.pdf": RecordedPage(body=body, content_type="application/pdf")},
            pdf_max_bytes=1024,
            max_bytes=4096,
        )

        result = await tool.execute(url=f"{_SITE}/survey.pdf")

        assert result.success is False
        assert result.content.startswith(f"Did not read {_SITE}/survey.pdf: the PDF is ")
        assert "past the 1 KB a PDF may be" in result.content

    @pytest.mark.asyncio
    async def test_a_body_without_the_pdf_signature_is_not_parsed(self) -> None:
        tool, _ = _fetch_tool(
            {
                _ROBOTS: _ALLOW,
                f"{_SITE}/fake.pdf": RecordedPage(body=b"<html>hi</html>", content_type="application/pdf"),
            }
        )

        result = await tool.execute(url=f"{_SITE}/fake.pdf")

        assert result.content == (
            f"Did not read {_SITE}/fake.pdf: the file says it is a PDF but is not one; find the document another way."
        )

    @pytest.mark.asyncio
    async def test_a_parse_that_runs_long_is_stopped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from threetears.agent.tools.document import DocumentResult

        async def _slow(*args: Any, **kwargs: Any) -> Any:
            await asyncio.sleep(0.5)
            return DocumentResult(text="late", title=None, page_count=1, word_count=1, was_ocr=False)

        monkeypatch.setattr("threetears.agent.tools.document.parse_document", _slow)
        reader = pdf_reader(parse_timeout_seconds=0.05)

        with pytest.raises(web_fetch_module.CarrierUnreadable, match=r"took longer than 0\.05 s"):
            await reader.read(b"%PDF-1.7 body")

    @pytest.mark.asyncio
    async def test_no_ocr_is_asked_for(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: dict[str, Any] = {}
        from threetears.agent.tools import document

        real = document.parse_document

        async def _spy(*args: Any, **kwargs: Any) -> Any:
            seen.update(kwargs)
            return await real(*args, **kwargs)

        monkeypatch.setattr("threetears.agent.tools.document.parse_document", _spy)

        await pdf_reader().read(_fixture("otter-survey.pdf"))

        assert seen["ocr_config"].enabled is False
        assert seen["max_pages"] == 100

    def test_the_fixture_is_the_recording_it_claims_to_be(self) -> None:
        assert _fixture("otter-survey.pdf").startswith(b"%PDF-")
        assert json.loads(_fixture("searxng-otters.json"))["results"]
