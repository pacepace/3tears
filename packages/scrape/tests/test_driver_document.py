"""Unit tests for DocumentDriver and document_text_to_html.

All tests are fully mocked -- no real network calls, no real PDF/DOCX/XLSX
parsing (``parse_document`` is monkeypatched). The real, live proof against
a genuine New Jersey WARN Act Excel file lives in
tests/e2e/test_warn_act_eval_loop_live.py (target_id="warn_act_nj").

**Not added to tests/scrape/test_driver_contract.py's shared ``_BACKENDS``
list, on purpose:** that contract's ``test_render_returns_the_backend_
supplied_content`` asserts ``page.html`` equals the literal HTML the fake
backend was configured to return -- true for NodriverSidecarDriver/
CamoufoxDriver (both render already-HTML pages verbatim), structurally
false for DocumentDriver (it always transforms parsed document text into
synthetic HTML via ``document_text_to_html``, never a byte-identical
passthrough). The properties that contract actually verifies -- stable
``name``, correct ``RenderedPage`` field types, ``wait_for``/``nav_steps``/
``capture_network`` acceptance -- are covered here instead
(``test_name``, ``test_render_fetches_parses_and_returns_synthetic_html``,
``test_render_accepts_and_ignores_wait_for_and_nav_steps``).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
from threetears.agent.tools.document import DocumentResult, DocumentSection, OcrConfig

from threetears.scrape.driver import RenderedPage
from threetears.scrape.drivers.document import (
    OCR_PAGE_IMAGE_CLASS,
    DocumentDriver,
    DocumentDriverError,
    ParsedDocumentHtml,
    document_text_to_html,
    parse_document_bytes_to_html,
)

# ===========================================================================
# document_text_to_html
# ===========================================================================


class TestDocumentTextToHtml:
    def test_plain_paragraphs_become_p_tags(self) -> None:
        html = document_text_to_html("First line.\n\nSecond line.")
        assert "<p>First line.</p>" in html
        assert "<p>Second line.</p>" in html

    def test_headings_become_h_tags_at_the_right_level(self) -> None:
        html = document_text_to_html("# Title\n\n### Subsection")
        assert "<h1>Title</h1>" in html
        assert "<h3>Subsection</h3>" in html

    def test_blank_lines_produce_no_empty_tags(self) -> None:
        html = document_text_to_html("Line one.\n\n\n\nLine two.")
        assert "<p></p>" not in html

    def test_pipe_table_becomes_a_real_html_table(self) -> None:
        md = "| Employer | City |\n| --- | --- |\n| Acme Corp | Trenton |\n| Foo Inc | Newark |"
        html = document_text_to_html(md)
        assert "<table>" in html
        assert "<tr><th>Employer</th><th>City</th></tr>" in html
        assert "<tr><td>Acme Corp</td><td>Trenton</td></tr>" in html
        assert "<tr><td>Foo Inc</td><td>Newark</td></tr>" in html

    def test_table_separator_row_with_alignment_colons_is_recognized(self) -> None:
        md = "| A | B |\n| :--- | ---: |\n| x | y |"
        html = document_text_to_html(md)
        assert "<table>" in html
        assert "<tr><td>x</td><td>y</td></tr>" in html

    def test_text_around_a_table_is_preserved(self) -> None:
        md = "# Report\n\n| A | B |\n| --- | --- |\n| 1 | 2 |\n\nFooter note."
        html = document_text_to_html(md)
        assert "<h1>Report</h1>" in html
        assert "<table>" in html
        assert "<p>Footer note.</p>" in html

    def test_multiple_tables_each_become_their_own_table_element(self) -> None:
        md = "| A | B |\n| --- | --- |\n| 1 | 2 |\n\n| C | D |\n| --- | --- |\n| 3 | 4 |"
        html = document_text_to_html(md)
        assert html.count("<table>") == 2

    def test_cell_content_is_html_escaped(self) -> None:
        md = "| Employer |\n| --- |\n| Macy's & <Co> |"
        html = document_text_to_html(md)
        assert "Macy&#x27;s &amp; &lt;Co&gt;" in html

    def test_a_single_pipe_row_with_no_separator_is_not_treated_as_a_table(self) -> None:
        """A lone `| a | b |`-shaped line with no following separator row
        isn't a real markdown table -- must not be misparsed as one."""
        html = document_text_to_html("| just some text | with pipes |")
        assert "<table>" not in html

    def test_a_header_cell_with_an_embedded_newline_still_forms_a_real_table(self) -> None:
        """A word-wrapped table cell's text can contain a literal newline,
        splitting one logical pipe-table row across multiple physical
        lines -- none of which alone starts AND ends with `|`. Undetected,
        the whole row (and the table) silently degrades to plain paragraphs
        instead of a real <table>."""
        md = "| County/Parish | Notice\nDate | Company |\n| --- | --- | --- |\n| Los Angeles | 2026-06-30 | Acme Corp |"
        html = document_text_to_html(md)
        assert "<table>" in html
        assert "<tr><th>County/Parish</th><th>Notice Date</th><th>Company</th></tr>" in html
        assert "<tr><td>Los Angeles</td><td>2026-06-30</td><td>Acme Corp</td></tr>" in html

    def test_a_body_cell_with_an_embedded_newline_is_also_merged(self) -> None:
        md = "| A | B |\n| --- | --- |\n| 1 | wrapped\ntext |\n| 2 | plain |"
        html = document_text_to_html(md)
        assert "<tr><td>1</td><td>wrapped text</td></tr>" in html
        assert "<tr><td>2</td><td>plain</td></tr>" in html

    def test_an_unterminated_pipe_row_at_end_of_text_does_not_hang_or_crash(self) -> None:
        """A malformed/truncated document (the very last line starts with `|`
        but never closes) must degrade gracefully, not loop forever."""
        html = document_text_to_html("| never closes")
        assert "<table>" not in html

    def test_no_sections_arg_preserves_original_markdown_only_behavior(self) -> None:
        """Backward compatibility: omitting sections changes nothing -- a
        plain-text line matching no markdown # syntax stays a <p>, even if
        by coincidence its text would match some OTHER document's heading."""
        html = document_text_to_html("Not A Heading")
        assert "<p>Not A Heading</p>" in html
        assert "<h1>" not in html

    def test_sections_heading_not_expressed_as_markdown_still_becomes_an_h_tag(self) -> None:
        """The PDF path computes real headings via a font-size heuristic
        without ever emitting markdown '#' syntax in .text -- previously
        silently flattened to a <p>."""
        sections = [DocumentSection(heading="Hull War Committee", content="body", page_number=1, level=3)]
        html = document_text_to_html("Hull War Committee\nbody text here", sections)
        assert "<h3>Hull War Committee</h3>" in html
        assert "<p>Hull War Committee</p>" not in html

    def test_sections_heading_level_1_from_font_size_heuristic(self) -> None:
        sections = [DocumentSection(heading="Main Title", content="", page_number=1, level=1)]
        html = document_text_to_html("Main Title", sections)
        assert "<h1>Main Title</h1>" in html

    def test_markdown_heading_syntax_still_wins_when_both_would_match(self) -> None:
        """A line matching BOTH markdown '#' syntax and a known section heading
        text uses the markdown-derived level -- the pre-existing, more
        specific signal, checked first."""
        sections = [DocumentSection(heading="Title", content="", page_number=1, level=3)]
        html = document_text_to_html("# Title", sections)
        assert "<h1>Title</h1>" in html
        assert "<h3>Title</h3>" not in html

    def test_sections_heading_text_only_matches_exactly_not_substrings(self) -> None:
        sections = [DocumentSection(heading="Report", content="", page_number=1, level=2)]
        html = document_text_to_html("Annual Report Summary", sections)
        assert "<h2>" not in html
        assert "<p>Annual Report Summary</p>" in html

    def test_sections_with_no_heading_are_ignored_not_treated_as_headings(self) -> None:
        sections = [DocumentSection(heading=None, content="Some body text", page_number=1, level=0)]
        html = document_text_to_html("Some body text", sections)
        assert not any(f"<h{n}>" in html for n in range(1, 7))
        assert "<p>Some body text</p>" in html

    def test_none_sections_behaves_identically_to_omitted_sections(self) -> None:
        assert document_text_to_html("Plain text", None) == document_text_to_html("Plain text")

    def test_a_table_row_is_never_reinterpreted_as_a_heading_even_if_text_matches(self) -> None:
        """Table detection runs first in the line-processing loop -- a pipe-table
        row's raw text is never checked against known_headings at all, so this
        just confirms the table path still wins (no change from existing
        behavior, guards the new branch didn't get inserted in the wrong place)."""
        sections = [DocumentSection(heading="A", content="", page_number=1, level=1)]
        md = "| A | B |\n| --- | --- |\n| 1 | 2 |"
        html = document_text_to_html(md, sections)
        assert "<table>" in html
        assert "<h1>A</h1>" not in html


# ===========================================================================
# parse_document_bytes_to_html -- was_ocr / embedded page images (scrape-task-06)
# ===========================================================================


class TestParseDocumentBytesToHtmlOcrImages:
    async def test_was_ocr_false_embeds_no_images(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from unittest.mock import Mock

        fake_result = DocumentResult(text="Acme Corp", title=None, page_count=None, word_count=2, was_ocr=False)
        monkeypatch.setattr("threetears.scrape.drivers.document.parse_document", AsyncMock(return_value=fake_result))
        render_mock = Mock()
        monkeypatch.setattr("threetears.scrape.drivers.document.render_pdf_pages_to_images", render_mock)

        result = await parse_document_bytes_to_html(b"fake-bytes", content_type="application/pdf", filename="x.pdf")

        assert isinstance(result, ParsedDocumentHtml)
        assert result.was_ocr is False
        assert OCR_PAGE_IMAGE_CLASS not in result.html
        render_mock.assert_not_called()

    async def test_was_ocr_true_embeds_each_page_as_a_base64_img_tag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_result = DocumentResult(text="Scanned letter text", title=None, page_count=2, word_count=3, was_ocr=True)
        monkeypatch.setattr("threetears.scrape.drivers.document.parse_document", AsyncMock(return_value=fake_result))
        monkeypatch.setattr(
            "threetears.scrape.drivers.document.render_pdf_pages_to_images",
            lambda data: [b"page0-png-bytes", b"page1-png-bytes"],
        )

        result = await parse_document_bytes_to_html(b"fake-pdf-bytes", content_type="application/pdf", filename="x.pdf")

        assert result.was_ocr is True
        assert "Scanned letter text" in result.html
        assert f'class="{OCR_PAGE_IMAGE_CLASS}"' in result.html
        assert result.html.count(f'class="{OCR_PAGE_IMAGE_CLASS}"') == 2
        import base64

        assert f"data:image/png;base64,{base64.b64encode(b'page0-png-bytes').decode('ascii')}" in result.html
        assert f"data:image/png;base64,{base64.b64encode(b'page1-png-bytes').decode('ascii')}" in result.html
        assert result.html.strip().endswith("</html>")  # embedded before the closing tag, still well-formed

    async def test_was_ocr_true_but_image_rendering_fails_still_returns_the_text_html(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A rendering failure must never take down an otherwise-usable OCR'd
        text result -- render_pdf_pages_to_images's own honest-empty-list
        contract (see its docstring) means zero <img> tags get embedded,
        not a crash."""
        fake_result = DocumentResult(text="Scanned letter text", title=None, page_count=1, word_count=3, was_ocr=True)
        monkeypatch.setattr("threetears.scrape.drivers.document.parse_document", AsyncMock(return_value=fake_result))
        monkeypatch.setattr("threetears.scrape.drivers.document.render_pdf_pages_to_images", lambda data: [])

        result = await parse_document_bytes_to_html(b"fake-pdf-bytes", content_type="application/pdf", filename="x.pdf")

        assert result.was_ocr is True
        assert "Scanned letter text" in result.html
        assert OCR_PAGE_IMAGE_CLASS not in result.html

    async def test_force_images_embeds_pages_even_when_was_ocr_is_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """scrape-task-07: a born-digital PDF (Nevada's real master WARN table) can
        still need a vision read -- its own table STRUCTURE, not scan quality,
        defeats text-based extraction, so was_ocr alone can't gate this."""
        fake_result = DocumentResult(
            text="Born-digital table text", title=None, page_count=1, word_count=3, was_ocr=False
        )
        monkeypatch.setattr("threetears.scrape.drivers.document.parse_document", AsyncMock(return_value=fake_result))
        monkeypatch.setattr(
            "threetears.scrape.drivers.document.render_pdf_pages_to_images", lambda data: [b"page0-png-bytes"]
        )

        result = await parse_document_bytes_to_html(
            b"fake-pdf-bytes", content_type="application/pdf", filename="x.pdf", force_images=True
        )

        assert result.was_ocr is False  # unaffected -- force_images doesn't lie about what parse_document found
        assert f'class="{OCR_PAGE_IMAGE_CLASS}"' in result.html

    async def test_force_images_false_is_the_default_and_embeds_nothing_for_a_non_ocr_document(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_result = DocumentResult(
            text="Born-digital table text", title=None, page_count=1, word_count=3, was_ocr=False
        )
        monkeypatch.setattr("threetears.scrape.drivers.document.parse_document", AsyncMock(return_value=fake_result))
        render_mock = AsyncMock()
        monkeypatch.setattr("threetears.scrape.drivers.document.render_pdf_pages_to_images", render_mock)

        result = await parse_document_bytes_to_html(b"fake-pdf-bytes", content_type="application/pdf", filename="x.pdf")

        assert OCR_PAGE_IMAGE_CLASS not in result.html
        render_mock.assert_not_called()

    async def test_merge_wrapped_table_rows_is_forwarded_to_parse_document(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """scrape-task-07 follow-up: opt-in, not folded into a default-True
        change for every document-backed target -- forwarded explicitly."""
        fake_result = DocumentResult(text="Table text", title=None, page_count=1, word_count=2, was_ocr=False)
        parse_mock = AsyncMock(return_value=fake_result)
        monkeypatch.setattr("threetears.scrape.drivers.document.parse_document", parse_mock)

        await parse_document_bytes_to_html(
            b"fake-pdf-bytes", content_type="application/pdf", filename="x.pdf", merge_wrapped_table_rows=True
        )

        assert parse_mock.call_args.kwargs["merge_wrapped_table_rows"] is True

    async def test_merge_wrapped_table_rows_false_is_the_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_result = DocumentResult(text="Table text", title=None, page_count=1, word_count=2, was_ocr=False)
        parse_mock = AsyncMock(return_value=fake_result)
        monkeypatch.setattr("threetears.scrape.drivers.document.parse_document", parse_mock)

        await parse_document_bytes_to_html(b"fake-pdf-bytes", content_type="application/pdf", filename="x.pdf")

        assert parse_mock.call_args.kwargs["merge_wrapped_table_rows"] is False

    async def test_result_sections_flow_through_to_real_heading_tags(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A parse_document result with real heading structure in .sections
        but no markdown '#' syntax in .text must still come out as a real
        <h#> tag, not a flattened <p> -- confirms parse_document_bytes_to_html
        actually threads result.sections through, not just document_text_to_html
        in isolation."""
        fake_result = DocumentResult(
            text="Hull War Committee\nSome body text",
            title=None,
            page_count=1,
            word_count=5,
            was_ocr=False,
            sections=[DocumentSection(heading="Hull War Committee", content="Some body text", page_number=1, level=3)],
        )
        monkeypatch.setattr("threetears.scrape.drivers.document.parse_document", AsyncMock(return_value=fake_result))

        result = await parse_document_bytes_to_html(b"fake-pdf-bytes", content_type="application/pdf", filename="x.pdf")

        assert "<h3>Hull War Committee</h3>" in result.html
        assert "<p>Hull War Committee</p>" not in result.html


# ===========================================================================
# DocumentDriver
# ===========================================================================


def _xlsx_response_handler(
    *, status: int = 200, content_type: str = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, headers={"content-type": content_type}, content=b"fake-bytes")

    return handler


class TestDocumentDriver:
    def test_name(self) -> None:
        driver = DocumentDriver()
        assert driver.name == "document"

    async def test_a_default_constructed_client_gets_the_browser_user_agent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A plain httpx client's default User-Agent gets a flat 403/401 from the
        CDN/WAF in front of some document hosts -- same fix, same reason as
        ApiDriver's own."""
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["user_agent"] = request.headers.get("user-agent")
            return httpx.Response(200, headers={"content-type": "text/plain"}, content=b"fake-bytes")

        real_async_client = httpx.AsyncClient

        def spying_async_client(**kwargs: Any) -> httpx.AsyncClient:
            captured["constructor_headers"] = kwargs.get("headers")
            kwargs["transport"] = httpx.MockTransport(handler)
            return real_async_client(**kwargs)

        monkeypatch.setattr("threetears.scrape.drivers.document.httpx.AsyncClient", spying_async_client)
        driver = DocumentDriver()

        fake_result = DocumentResult(text="Acme Corp", title="notices", page_count=None, word_count=2, was_ocr=False)
        monkeypatch.setattr("threetears.scrape.drivers.document.parse_document", AsyncMock(return_value=fake_result))

        await driver.render("https://example.gov/warn.txt")

        expected_ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        assert captured["constructor_headers"] == {"User-Agent": expected_ua}
        assert captured["user_agent"] == expected_ua

    async def test_an_injected_client_is_used_as_given_no_default_user_agent_override(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The default browser User-Agent only applies to a client this driver constructs
        itself -- an injected client's own header policy (or lack of one) must be left alone."""
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["user_agent"] = request.headers.get("user-agent")
            return httpx.Response(200, headers={"content-type": "text/plain"}, content=b"fake-bytes")

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        driver = DocumentDriver(client=client)

        fake_result = DocumentResult(text="Acme Corp", title="notices", page_count=None, word_count=2, was_ocr=False)
        monkeypatch.setattr("threetears.scrape.drivers.document.parse_document", AsyncMock(return_value=fake_result))

        await driver.render("https://example.gov/warn.txt")

        assert captured["user_agent"] != (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        )
        await client.aclose()

    async def test_render_fetches_parses_and_returns_synthetic_html(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = httpx.AsyncClient(transport=httpx.MockTransport(_xlsx_response_handler()))
        driver = DocumentDriver(client=client)

        fake_result = DocumentResult(
            text="| Employer |\n| --- |\n| Acme Corp |",
            title="notices",
            page_count=None,
            word_count=2,
            was_ocr=False,
        )
        mock_parse = AsyncMock(return_value=fake_result)
        monkeypatch.setattr("threetears.scrape.drivers.document.parse_document", mock_parse)

        page = await driver.render("https://example.gov/warn.xlsx")

        assert isinstance(page, RenderedPage)
        assert page.status == 200
        assert "<table>" in page.html
        assert "<td>Acme Corp</td>" in page.html
        await client.aclose()

    async def test_render_derives_filename_and_mime_from_the_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                headers={"content-type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"},
                content=b"fake-bytes",
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        driver = DocumentDriver(client=client)

        async def fake_parse_document(
            data: bytes,
            mime_type: str,
            filename: str | None = None,
            ocr_config: OcrConfig | None = None,
            *,
            merge_wrapped_table_rows: bool = False,
        ) -> DocumentResult:
            captured["data"] = data
            captured["mime_type"] = mime_type
            captured["filename"] = filename
            return DocumentResult(text="no table here", title=None, page_count=None, word_count=3, was_ocr=False)

        monkeypatch.setattr("threetears.scrape.drivers.document.parse_document", fake_parse_document)

        await driver.render("https://example.gov/reports/warn_notices.xlsx")

        assert captured["filename"] == "warn_notices.xlsx"
        assert captured["mime_type"] == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        assert captured["data"] == b"fake-bytes"
        await client.aclose()

    async def test_render_raises_on_transport_failure(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused", request=request)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        driver = DocumentDriver(client=client)

        with pytest.raises(DocumentDriverError) as exc_info:
            await driver.render("https://example.gov/warn.xlsx")

        assert exc_info.value.code == "transport"
        await client.aclose()

    async def test_render_raises_on_http_error_status(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, content=b"not found")

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        driver = DocumentDriver(client=client)

        with pytest.raises(DocumentDriverError) as exc_info:
            await driver.render("https://example.gov/missing.xlsx")

        assert exc_info.value.code == "fetch_failed"
        await client.aclose()

    async def test_render_raises_when_parse_document_reports_unsupported_type(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(_xlsx_response_handler(content_type="application/octet-stream"))
        )
        driver = DocumentDriver(client=client)

        fake_result = DocumentResult(
            text="[Unsupported document type: application/octet-stream]",
            title=None,
            page_count=None,
            word_count=0,
            was_ocr=False,
        )
        monkeypatch.setattr("threetears.scrape.drivers.document.parse_document", AsyncMock(return_value=fake_result))

        with pytest.raises(DocumentDriverError) as exc_info:
            await driver.render("https://example.gov/mystery-file")

        assert exc_info.value.code == "parse_failed"
        await client.aclose()

    async def test_render_raises_when_parse_document_reports_a_parsing_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = httpx.AsyncClient(transport=httpx.MockTransport(_xlsx_response_handler()))
        driver = DocumentDriver(client=client)

        fake_result = DocumentResult(
            text="[Parsing failed: corrupt file]", title=None, page_count=None, word_count=0, was_ocr=False
        )
        monkeypatch.setattr("threetears.scrape.drivers.document.parse_document", AsyncMock(return_value=fake_result))

        with pytest.raises(DocumentDriverError) as exc_info:
            await driver.render("https://example.gov/warn.xlsx")

        assert exc_info.value.code == "parse_failed"
        await client.aclose()

    async def test_render_propagates_was_ocr_true_onto_rendered_page(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = httpx.AsyncClient(transport=httpx.MockTransport(_xlsx_response_handler()))
        driver = DocumentDriver(client=client)

        fake_result = DocumentResult(text="Scanned text", title=None, page_count=1, word_count=2, was_ocr=True)
        monkeypatch.setattr("threetears.scrape.drivers.document.parse_document", AsyncMock(return_value=fake_result))
        monkeypatch.setattr("threetears.scrape.drivers.document.render_pdf_pages_to_images", lambda data: [])

        page = await driver.render("https://example.gov/warn.pdf")

        assert page.was_ocr is True
        await client.aclose()

    async def test_force_images_constructor_flag_is_threaded_through_to_the_parse_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = httpx.AsyncClient(transport=httpx.MockTransport(_xlsx_response_handler()))
        driver = DocumentDriver(client=client, force_images=True)

        fake_result = DocumentResult(text="Table text", title=None, page_count=1, word_count=2, was_ocr=False)
        monkeypatch.setattr("threetears.scrape.drivers.document.parse_document", AsyncMock(return_value=fake_result))
        monkeypatch.setattr(
            "threetears.scrape.drivers.document.render_pdf_pages_to_images", lambda data: [b"page0-png-bytes"]
        )

        page = await driver.render("https://example.gov/warn.pdf")

        assert page.was_ocr is False
        assert f'class="{OCR_PAGE_IMAGE_CLASS}"' in page.html
        await client.aclose()

    async def test_merge_wrapped_table_rows_constructor_flag_is_threaded_through_to_the_parse_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client = httpx.AsyncClient(transport=httpx.MockTransport(_xlsx_response_handler()))
        driver = DocumentDriver(client=client, merge_wrapped_table_rows=True)

        fake_result = DocumentResult(text="Table text", title=None, page_count=1, word_count=2, was_ocr=False)
        parse_mock = AsyncMock(return_value=fake_result)
        monkeypatch.setattr("threetears.scrape.drivers.document.parse_document", parse_mock)

        await driver.render("https://example.gov/warn.pdf")

        assert parse_mock.call_args.kwargs["merge_wrapped_table_rows"] is True
        await client.aclose()

    async def test_render_accepts_and_ignores_wait_for_and_nav_steps(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Interface conformance only -- a document fetch has no browser to
        wait on or drive, but must still accept the full ScrapeDriver
        signature like every other backend."""
        from threetears.scrape.driver import NavStep

        client = httpx.AsyncClient(transport=httpx.MockTransport(_xlsx_response_handler()))
        driver = DocumentDriver(client=client)
        monkeypatch.setattr(
            "threetears.scrape.drivers.document.parse_document",
            AsyncMock(return_value=DocumentResult(text="hi", title=None, page_count=None, word_count=1, was_ocr=False)),
        )

        page = await driver.render(
            "https://example.gov/warn.xlsx",
            wait_for=".content",
            nav_steps=[NavStep(action="click", selector="#x")],
        )

        assert isinstance(page, RenderedPage)
        await client.aclose()
