"""Unit tests for MultiDocumentDriver -- listing -> N document fetches -> one combined page.

All tests are fully mocked -- no real network calls, no real document
parsing (the injected document_driver is a fake). The real, live proof
against Hawaii's/West Virginia's actual pages lives in
tests/e2e/test_warn_act_eval_loop_live.py (faidh repo).
"""

from __future__ import annotations

import json

import httpx
import pytest

from threetears.scrape.driver import RenderedPage, ScrapeDriver
from threetears.scrape.drivers.multi_document import MultiDocumentDriver, MultiDocumentDriverError


class _FakeDocumentDriver(ScrapeDriver):
    """A fake per-document fetcher -- returns canned pages, or raises for configured URLs."""

    def __init__(
        self,
        pages: dict[str, str] | None = None,
        fail_urls: set[str] | None = None,
        was_ocr_urls: set[str] | None = None,
    ) -> None:
        self._pages = pages or {}
        self._fail_urls = fail_urls or set()
        self._was_ocr_urls = was_ocr_urls or set()
        self.fetched_urls: list[str] = []

    @property
    def name(self) -> str:
        return "fake_document"

    async def render(self, url: str, **kwargs: object) -> RenderedPage:
        self.fetched_urls.append(url)
        if url in self._fail_urls:
            raise RuntimeError(f"simulated fetch failure for {url}")
        html = self._pages.get(url, f"<html><body><p>content for {url}</p></body></html>")
        return RenderedPage(html=html, status=200, final_url=url, timing_ms=1.0, was_ocr=url in self._was_ocr_urls)


def _listing_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _html_listing_handler(links: list[str]):
    body = "<html><body><ul>" + "".join(f'<li><a href="{link}">notice</a></li>' for link in links) + "</ul></body></html>"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body.encode())

    return handler


def _json_listing_handler(records: list[dict]):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=json.dumps(records).encode())

    return handler


class TestMultiDocumentDriver:
    async def test_neither_discovery_mode_configured_raises(self):
        client = _listing_client(lambda r: httpx.Response(200, content=b"<html></html>"))
        driver = MultiDocumentDriver(document_driver=_FakeDocumentDriver(), client=client)
        with pytest.raises(MultiDocumentDriverError) as exc_info:
            await driver.render("https://example.gov/listing")
        assert exc_info.value.code == "missing_config"
        await client.aclose()

    async def test_both_discovery_modes_configured_raises(self):
        driver = MultiDocumentDriver(
            document_driver=_FakeDocumentDriver(), client=_listing_client(lambda r: httpx.Response(200, content=b"[]"))
        )
        with pytest.raises(MultiDocumentDriverError) as exc_info:
            await driver.render(
                "https://example.gov/listing", link_selector="a", results_path="", fragment_field="url"
            )
        assert exc_info.value.code == "missing_config"

    async def test_html_mode_discovers_and_fetches_each_document(self):
        links = ["https://example.gov/notice1.pdf", "https://example.gov/notice2.pdf"]
        client = _listing_client(_html_listing_handler(links))
        fake_docs = _FakeDocumentDriver(
            {
                "https://example.gov/notice1.pdf": "<html><body><p>Acme Corp layoff</p></body></html>",
                "https://example.gov/notice2.pdf": "<html><body><p>Beta Inc layoff</p></body></html>",
            }
        )
        driver = MultiDocumentDriver(document_driver=fake_docs, client=client)

        page = await driver.render("https://example.gov/listing", link_selector="a")

        assert page.status == 200
        assert "Acme Corp layoff" in page.html
        assert "Beta Inc layoff" in page.html
        assert page.html.count('class="notice"') == 2
        assert fake_docs.fetched_urls == links

    async def test_was_ocr_is_propagated_onto_each_documents_own_wrapping_div(self):
        """scrape-task-06: eval_loop's vision-vs-text routing reads this
        attribute back out per document -- a document driver's own
        RenderedPage.was_ocr must actually reach the combined page, not just
        this driver's own status/html fields."""
        links = ["https://example.gov/scanned.pdf", "https://example.gov/born-digital.pdf"]
        client = _listing_client(_html_listing_handler(links))
        fake_docs = _FakeDocumentDriver(
            {
                "https://example.gov/scanned.pdf": "<html><body><p>Scanned notice</p></body></html>",
                "https://example.gov/born-digital.pdf": "<html><body><p>Digital notice</p></body></html>",
            },
            was_ocr_urls={"https://example.gov/scanned.pdf"},
        )
        driver = MultiDocumentDriver(document_driver=fake_docs, client=client)

        page = await driver.render("https://example.gov/listing", link_selector="a")

        from bs4 import BeautifulSoup

        soup = BeautifulSoup(page.html, "html.parser")
        notice_divs = soup.select("div.notice")
        assert len(notice_divs) == 2
        by_content = {div.get_text(): div.get("data-was-ocr") for div in notice_divs}
        assert by_content["Scanned notice"] == "true"
        assert by_content["Digital notice"] == "false"

    async def test_html_mode_resolves_relative_hrefs_to_absolute(self):
        client = _listing_client(_html_listing_handler(["/wp-content/notice1.pdf"]))
        fake_docs = _FakeDocumentDriver()
        driver = MultiDocumentDriver(document_driver=fake_docs, client=client)

        await driver.render("https://example.gov/listing/page/", link_selector="a")

        assert fake_docs.fetched_urls == ["https://example.gov/wp-content/notice1.pdf"]

    async def test_json_mode_discovers_and_fetches_each_document(self):
        records = [
            {"title": "Notice A", "source_url": "https://example.gov/a.pdf"},
            {"title": "Notice B", "source_url": "https://example.gov/b.pdf"},
        ]
        client = _listing_client(_json_listing_handler(records))
        fake_docs = _FakeDocumentDriver()
        driver = MultiDocumentDriver(document_driver=fake_docs, client=client)

        page = await driver.render("https://example.gov/wp-json/media", results_path="", fragment_field="source_url")

        assert page.status == 200
        assert fake_docs.fetched_urls == ["https://example.gov/a.pdf", "https://example.gov/b.pdf"]

    async def test_json_mode_with_nested_results_path(self):
        body = {"Results": [{"source_url": "https://example.gov/a.pdf"}]}
        client = _listing_client(_json_listing_handler(body))
        fake_docs = _FakeDocumentDriver()
        driver = MultiDocumentDriver(document_driver=fake_docs, client=client)

        await driver.render("https://example.gov/api", results_path="Results", fragment_field="source_url")

        assert fake_docs.fetched_urls == ["https://example.gov/a.pdf"]

    async def test_json_mode_bad_results_path_raises_multi_document_error(self):
        client = _listing_client(_json_listing_handler({"NotResults": []}))
        driver = MultiDocumentDriver(document_driver=_FakeDocumentDriver(), client=client)

        with pytest.raises(MultiDocumentDriverError) as exc_info:
            await driver.render("https://example.gov/api", results_path="Results", fragment_field="source_url")
        assert exc_info.value.code == "bad_results_path"

    async def test_json_mode_invalid_json_raises(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"not json at all")

        client = _listing_client(handler)
        driver = MultiDocumentDriver(document_driver=_FakeDocumentDriver(), client=client)

        with pytest.raises(MultiDocumentDriverError) as exc_info:
            await driver.render("https://example.gov/api", results_path="", fragment_field="source_url")
        assert exc_info.value.code == "invalid_json"

    async def test_one_document_fetch_failure_does_not_sink_the_others(self):
        links = ["https://example.gov/good1.pdf", "https://example.gov/bad.pdf", "https://example.gov/good2.pdf"]
        client = _listing_client(_html_listing_handler(links))
        fake_docs = _FakeDocumentDriver(
            {
                "https://example.gov/good1.pdf": "<html><body><p>First good notice</p></body></html>",
                "https://example.gov/good2.pdf": "<html><body><p>Second good notice</p></body></html>",
            },
            fail_urls={"https://example.gov/bad.pdf"},
        )
        driver = MultiDocumentDriver(document_driver=fake_docs, client=client)

        page = await driver.render("https://example.gov/listing", link_selector="a")

        assert "First good notice" in page.html
        assert "Second good notice" in page.html
        assert page.html.count('class="notice"') == 2  # the failed one is skipped, not a crash

    async def test_max_documents_bounds_the_fetch_count(self):
        links = [f"https://example.gov/notice{i}.pdf" for i in range(5)]
        client = _listing_client(_html_listing_handler(links))
        fake_docs = _FakeDocumentDriver()
        driver = MultiDocumentDriver(document_driver=fake_docs, client=client, max_documents=2)

        await driver.render("https://example.gov/listing", link_selector="a")

        assert fake_docs.fetched_urls == links[:2]

    async def test_seen_urls_none_fetches_every_document_unchanged(self):
        """The default (seen_urls omitted) must match this driver's
        pre-2026-07-16 behavior byte-for-byte -- every discovered document
        (re-)fetched, no dedup."""
        links = ["https://example.gov/a.pdf", "https://example.gov/b.pdf"]
        client = _listing_client(_html_listing_handler(links))
        fake_docs = _FakeDocumentDriver()
        driver = MultiDocumentDriver(document_driver=fake_docs, client=client)

        await driver.render("https://example.gov/listing", link_selector="a")

        assert fake_docs.fetched_urls == links

    async def test_seen_urls_skips_already_seen_documents(self):
        links = ["https://example.gov/a.pdf", "https://example.gov/b.pdf"]
        client = _listing_client(_html_listing_handler(links))
        fake_docs = _FakeDocumentDriver()
        driver = MultiDocumentDriver(document_driver=fake_docs, client=client)
        seen = {"https://example.gov/a.pdf"}

        page = await driver.render("https://example.gov/listing", link_selector="a", seen_urls=seen)

        assert fake_docs.fetched_urls == ["https://example.gov/b.pdf"]
        assert page.html.count('class="notice"') == 1

    async def test_seen_urls_marks_successfully_fetched_documents(self):
        links = ["https://example.gov/a.pdf", "https://example.gov/b.pdf"]
        client = _listing_client(_html_listing_handler(links))
        fake_docs = _FakeDocumentDriver()
        driver = MultiDocumentDriver(document_driver=fake_docs, client=client)
        seen: set[str] = set()

        await driver.render("https://example.gov/listing", link_selector="a", seen_urls=seen)

        assert seen == set(links)

    async def test_seen_urls_does_not_mark_a_failed_fetch(self):
        """A document that fails to fetch must stay eligible for retry on
        the next poll -- never silently skipped forever, same invariant
        every checkpoint cursor in this codebase already holds."""
        links = ["https://example.gov/good.pdf", "https://example.gov/bad.pdf"]
        client = _listing_client(_html_listing_handler(links))
        fake_docs = _FakeDocumentDriver(fail_urls={"https://example.gov/bad.pdf"})
        driver = MultiDocumentDriver(document_driver=fake_docs, client=client)
        seen: set[str] = set()

        await driver.render("https://example.gov/listing", link_selector="a", seen_urls=seen)

        assert seen == {"https://example.gov/good.pdf"}

    async def test_seen_urls_filtered_before_max_documents_cap(self):
        """A real bug shape this test guards against: if already-seen
        documents were capped BEFORE filtering, a raw listing with more
        already-seen entries than max_documents would starve genuinely new
        documents sitting just past the cap."""
        links = [f"https://example.gov/notice{i}.pdf" for i in range(3)]
        client = _listing_client(_html_listing_handler(links))
        fake_docs = _FakeDocumentDriver()
        driver = MultiDocumentDriver(document_driver=fake_docs, client=client, max_documents=1)
        # The first 2 (of 3) documents are already seen -- with a cap of 1,
        # naive "cap first" ordering would consider only notice0 (seen,
        # skipped) and fetch nothing; filter-first must still reach notice2.
        seen = {"https://example.gov/notice0.pdf", "https://example.gov/notice1.pdf"}

        await driver.render("https://example.gov/listing", link_selector="a", seen_urls=seen)

        assert fake_docs.fetched_urls == ["https://example.gov/notice2.pdf"]

    async def test_listing_fetch_transport_failure_raises(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused", request=request)

        client = _listing_client(handler)
        driver = MultiDocumentDriver(document_driver=_FakeDocumentDriver(), client=client)

        with pytest.raises(MultiDocumentDriverError) as exc_info:
            await driver.render("https://example.gov/listing", link_selector="a")
        assert exc_info.value.code == "transport"

    async def test_listing_fetch_http_error_status_raises(self):
        client = _listing_client(lambda r: httpx.Response(404, content=b"not found"))
        driver = MultiDocumentDriver(document_driver=_FakeDocumentDriver(), client=client)

        with pytest.raises(MultiDocumentDriverError) as exc_info:
            await driver.render("https://example.gov/listing", link_selector="a")
        assert exc_info.value.code == "fetch_failed"

    async def test_name(self):
        assert MultiDocumentDriver(document_driver=_FakeDocumentDriver()).name == "multi_document"

    async def test_no_documents_found_returns_an_empty_but_well_formed_page(self):
        client = _listing_client(_html_listing_handler([]))
        driver = MultiDocumentDriver(document_driver=_FakeDocumentDriver(), client=client)

        page = await driver.render("https://example.gov/listing", link_selector="a")

        assert page.status == 200
        assert page.html == "<html><body></body></html>"

    async def test_an_injected_client_is_used_as_given_no_default_user_agent_override(self):
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["user_agent"] = request.headers.get("user-agent")
            return httpx.Response(200, content=b"<html><body></body></html>")

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        driver = MultiDocumentDriver(document_driver=_FakeDocumentDriver(), client=client)

        await driver.render("https://example.gov/listing", link_selector="a")

        assert captured["user_agent"] != (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        )
        await client.aclose()


class TestDiscoverLinks:
    """The promoted public :func:`discover_links` — the pure listing-page link walk."""

    def test_resolves_relative_and_absolute_hrefs_in_document_order(self) -> None:
        from threetears.scrape.drivers.multi_document import discover_links

        html = (
            '<a class="doc" href="/files/a.pdf">A</a>'
            '<a class="nav" href="/home">skip</a>'
            '<a class="doc" href="https://cdn.example/b.pdf">B</a>'
        )
        urls = discover_links(html, base_url="https://gov.example/list", link_selector="a.doc")
        assert urls == ["https://gov.example/files/a.pdf", "https://cdn.example/b.pdf"]

    def test_selector_scopes_which_anchors_are_followed(self) -> None:
        from threetears.scrape.drivers.multi_document import discover_links

        html = '<a href="/x.pdf">no class</a><a class="doc" href="/y.pdf">yes</a>'
        assert discover_links(html, base_url="https://h", link_selector="a.doc") == ["https://h/y.pdf"]

    def test_anchor_without_href_is_skipped(self) -> None:
        from threetears.scrape.drivers.multi_document import discover_links

        html = '<a class="doc">no href</a><a class="doc" href="/z.pdf">z</a>'
        assert discover_links(html, base_url="https://h", link_selector="a.doc") == ["https://h/z.pdf"]

    def test_discover_links_is_the_url_projection_of_the_labeled_walk(self) -> None:
        # discover_links delegates to discover_links_labeled — same URLs, anchor text dropped. Pins the
        # promised backward-compatible behavior so the delegation can never drift from the URL-only set.
        from threetears.scrape.drivers.multi_document import discover_links, discover_links_labeled

        html = '<a class="doc" href="/a.pdf">A</a><a class="doc" href="https://cdn.example/b.pdf">B</a>'
        labeled = discover_links_labeled(html, base_url="https://h/list", link_selector="a.doc")
        assert discover_links(html, base_url="https://h/list", link_selector="a.doc") == [u for u, _ in labeled]


class TestDiscoverLinksLabeled:
    """:func:`discover_links_labeled` — the same walk, carrying each anchor's visible text."""

    def test_carries_anchor_text_alongside_the_resolved_url(self) -> None:
        from threetears.scrape.drivers.multi_document import discover_links_labeled

        html = (
            '<a class="doc" href="/nov.pdf">November 2024 schedule</a>'
            '<a class="nav" href="/home">skip</a>'
            '<a class="doc" href="https://cdn.example/dec.pdf">December 2024 schedule</a>'
        )
        assert discover_links_labeled(html, base_url="https://gov.example/list", link_selector="a.doc") == [
            ("https://gov.example/nov.pdf", "November 2024 schedule"),
            ("https://cdn.example/dec.pdf", "December 2024 schedule"),
        ]

    def test_anchor_text_is_whitespace_collapsed_and_gathers_nested_tags(self) -> None:
        # Real listings wrap the label in spans / add newlines & indentation; the label is normalized to
        # a single clean line, and text from nested elements is gathered (an icon <span> + the month name).
        from threetears.scrape.drivers.multi_document import discover_links_labeled

        html = '<a class="doc" href="/m.pdf">\n  <span>March</span>   2025\n  schedule\n</a>'
        assert discover_links_labeled(html, base_url="https://h", link_selector="a.doc") == [
            ("https://h/m.pdf", "March 2025 schedule")
        ]

    def test_anchor_with_no_text_yields_an_empty_label_not_a_skip(self) -> None:
        # An image-only link still resolves to a URL; its label is "" (the caller decides what a missing
        # label means), never a dropped entry.
        from threetears.scrape.drivers.multi_document import discover_links_labeled

        html = '<a class="doc" href="/scan.pdf"><img src="/thumb.png"></a>'
        assert discover_links_labeled(html, base_url="https://h", link_selector="a.doc") == [("https://h/scan.pdf", "")]

    def test_anchor_without_href_is_skipped(self) -> None:
        from threetears.scrape.drivers.multi_document import discover_links_labeled

        html = '<a class="doc">no href</a><a class="doc" href="/z.pdf">z</a>'
        assert discover_links_labeled(html, base_url="https://h", link_selector="a.doc") == [("https://h/z.pdf", "z")]
