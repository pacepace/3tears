"""Tests for document parsing module."""

from __future__ import annotations

import base64
import sys
from types import ModuleType
from typing import Any

import pytest

from threetears.agent.tools.document import (
    DocumentResult,
    OcrConfig,
    ParseDocumentInput,
    _extract_pdf_tables,
    _merge_wrapped_table_rows,
    _ocr_page,
    create_parse_document_tool,
    detect_mime_from_filename,
    parse_document,
    render_pdf_pages_to_images,
)


# -- OcrConfig ---------------------------------------------------------------


class TestOcrConfig:
    def test_defaults(self):
        cfg = OcrConfig()
        assert cfg.enabled is False
        assert cfg.language == "eng"
        assert cfg.psm == 4

    def test_custom(self):
        cfg = OcrConfig(enabled=True, language="deu", psm=3)
        assert cfg.enabled is True
        assert cfg.language == "deu"
        assert cfg.psm == 3


# -- _ocr_page -----------------------------------------------------------------
# pytesseract/pdf2image are lazily imported inside _ocr_page (an optional "ocr"
# extra, not installed in this package's own default test env) -- fake modules
# injected via sys.modules so this test runs regardless of whether the real
# packages happen to be installed, matching how the function itself resolves them.


class TestOcrPage:
    def test_passes_the_given_psm_to_pytesseract(self, monkeypatch):
        """scrape-task-06, 2026-07-16: psm is caller-configurable (OcrConfig.psm),
        not a hardcoded module constant -- one target's own PSM 4 evidence
        (scrape-task-05: PSM 3 can drop a narrow numeric table column entirely,
        a documented failure mode distinct from misrecognition) shouldn't become
        an unconditional default for every consumer of this shared tool. This
        test only proves the config string reaches pytesseract's own call, not
        OCR accuracy itself (that needs the real Tesseract binary, proven
        separately against real live documents)."""
        captured: dict[str, Any] = {}

        fake_pdf2image = ModuleType("pdf2image")
        fake_pdf2image.convert_from_bytes = lambda *a, **kw: ["fake-image"]  # type: ignore[attr-defined]

        fake_pytesseract = ModuleType("pytesseract")

        def fake_image_to_string(image, lang=None, config=None):
            captured["image"] = image
            captured["lang"] = lang
            captured["config"] = config
            return "extracted text"

        fake_pytesseract.image_to_string = fake_image_to_string  # type: ignore[attr-defined]

        monkeypatch.setitem(sys.modules, "pdf2image", fake_pdf2image)
        monkeypatch.setitem(sys.modules, "pytesseract", fake_pytesseract)

        result = _ocr_page(b"fake-pdf-bytes", page_num=0, language="eng", psm=4)

        assert result == "extracted text"
        assert captured["config"] == "--psm 4"
        assert captured["lang"] == "eng"

    def test_a_different_psm_is_passed_through_unchanged(self, monkeypatch):
        captured: dict[str, Any] = {}

        fake_pdf2image = ModuleType("pdf2image")
        fake_pdf2image.convert_from_bytes = lambda *a, **kw: ["fake-image"]  # type: ignore[attr-defined]

        fake_pytesseract = ModuleType("pytesseract")

        def fake_image_to_string(image, lang=None, config=None):
            captured["config"] = config
            return "extracted text"

        fake_pytesseract.image_to_string = fake_image_to_string  # type: ignore[attr-defined]

        monkeypatch.setitem(sys.modules, "pdf2image", fake_pdf2image)
        monkeypatch.setitem(sys.modules, "pytesseract", fake_pytesseract)

        _ocr_page(b"fake-pdf-bytes", page_num=0, language="eng", psm=3)

        assert captured["config"] == "--psm 3"


# -- render_pdf_pages_to_images -------------------------------------------------
# Same lazy-import-fake pattern as TestOcrPage -- pdf2image is an optional "ocr"
# extra, not installed in this package's own default test env.


# parity-exempt: hand-rolled subset stub of PIL's third-party Image (only save(), the only surface render_pdf_pages_to_images calls)
class _FakePILImage:
    def __init__(self, label: str) -> None:
        self.label = label

    def save(self, buf, format=None):  # noqa: A002 -- matches PIL.Image.save's own kwarg name
        buf.write(f"png-bytes-for-{self.label}".encode())


class TestRenderPdfPagesToImages:
    def test_encodes_each_page_as_png_bytes(self, monkeypatch):
        captured: dict[str, Any] = {}

        fake_pdf2image = ModuleType("pdf2image")

        def fake_convert_from_bytes(pdf_data, *, dpi=None, first_page=None, last_page=None):
            captured["dpi"] = dpi
            captured["first_page"] = first_page
            captured["last_page"] = last_page
            return [_FakePILImage("page0"), _FakePILImage("page1")]

        fake_pdf2image.convert_from_bytes = fake_convert_from_bytes  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "pdf2image", fake_pdf2image)

        result = render_pdf_pages_to_images(b"fake-pdf-bytes", dpi=150, max_pages=3)

        assert result == [b"png-bytes-for-page0", b"png-bytes-for-page1"]
        assert captured["dpi"] == 150
        assert captured["first_page"] == 1
        assert captured["last_page"] == 3

    def test_render_failure_returns_empty_list_not_a_crash(self, monkeypatch):
        fake_pdf2image = ModuleType("pdf2image")

        def fake_convert_from_bytes(*args, **kwargs):
            raise RuntimeError("boom")

        fake_pdf2image.convert_from_bytes = fake_convert_from_bytes  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "pdf2image", fake_pdf2image)

        result = render_pdf_pages_to_images(b"fake-pdf-bytes")

        assert result == []

    def test_default_max_pages_is_three(self, monkeypatch):
        captured: dict[str, Any] = {}

        fake_pdf2image = ModuleType("pdf2image")

        def fake_convert_from_bytes(pdf_data, *, dpi=None, first_page=None, last_page=None):
            captured["last_page"] = last_page
            return []

        fake_pdf2image.convert_from_bytes = fake_convert_from_bytes  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "pdf2image", fake_pdf2image)

        render_pdf_pages_to_images(b"fake-pdf-bytes")

        assert captured["last_page"] == 3


# -- detect_mime_from_filename ------------------------------------------------


class TestDetectMime:
    @pytest.mark.parametrize(
        "filename, expected",
        [
            ("report.pdf", "application/pdf"),
            ("doc.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
            ("data.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
            ("data.csv", "text/csv"),
            ("notes.txt", "text/plain"),
            ("README.md", "text/markdown"),
            ("paper.tex", "application/x-tex"),
            ("paper.latex", "application/x-latex"),
        ],
    )
    def test_known_extensions(self, filename: str, expected: str):
        assert detect_mime_from_filename(filename) == expected

    def test_no_extension(self):
        result = detect_mime_from_filename("noextension")
        assert result is None

    def test_unknown_extension(self):
        # Falls back to mimetypes.guess_type
        result = detect_mime_from_filename("file.xyz123nope")
        # Should be None or whatever mimetypes guesses
        assert result is None or isinstance(result, str)


# -- parse_document: text formats ---------------------------------------------


class TestParseText:
    async def test_plain_text(self):
        data = b"Hello world, this is a test document."
        result = await parse_document(data, "text/plain", "test.txt")
        assert isinstance(result, DocumentResult)
        assert "Hello world" in result.text
        assert result.word_count > 0
        assert result.was_ocr is False
        assert result.title == "test.txt"
        assert len(result.sections) == 1

    async def test_plain_text_latin1_fallback(self):
        data = "café résumé".encode("latin-1")
        result = await parse_document(data, "text/plain")
        assert "café" in result.text

    async def test_markdown_sections(self):
        md = b"# Title\n\nIntro text.\n\n## Section One\n\nBody here.\n"
        result = await parse_document(md, "text/markdown", "doc.md")
        assert result.title == "Title"
        assert len(result.sections) >= 2
        assert result.sections[0].heading == "Title"
        assert result.sections[0].level == 1
        assert result.sections[1].heading == "Section One"
        assert result.sections[1].level == 2

    async def test_markdown_passthrough(self):
        md = b"**bold** and *italic*"
        result = await parse_document(md, "text/markdown")
        assert result.text == "**bold** and *italic*"


class TestParseCsv:
    async def test_basic_table(self):
        data = b"Employer,County,Affected\nAcme Corp,Oakland,42\nWidgets Inc,Wayne,7\n"
        result = await parse_document(data, "text/csv", "warn.csv")
        assert isinstance(result, DocumentResult)
        assert result.title == "warn.csv"
        assert result.was_ocr is False
        assert result.text == (
            "| Employer | County | Affected |\n"
            "| --- | --- | --- |\n"
            "| Acme Corp | Oakland | 42 |\n"
            "| Widgets Inc | Wayne | 7 |"
        )
        assert result.word_count > 0
        assert len(result.sections) == 1

    async def test_embedded_comma_in_quoted_field_is_not_split(self):
        data = b'Employer,City\n"Acme, Corp",Oakland\n'
        result = await parse_document(data, "text/csv")
        assert "| Acme, Corp | Oakland |" in result.text

    async def test_embedded_newline_in_quoted_field_stays_one_row(self):
        data = b'Employer,Notes\n"Acme Corp","line one\nline two"\n'
        result = await parse_document(data, "text/csv")
        assert "line one\nline two" in result.text
        # exactly one data row -- the embedded newline must not have been
        # mistaken for a new CSV row
        assert result.text.count("| Acme Corp") == 1
        assert result.text.count("| --- | --- |") == 1

    async def test_ragged_short_row_is_padded(self):
        data = b"A,B,C\n1,2,3\n4,5\n"
        result = await parse_document(data, "text/csv")
        assert "| 4 | 5 |  |" in result.text

    async def test_ragged_long_row_is_trimmed(self):
        data = b"A,B\n1,2,3,4\n"
        result = await parse_document(data, "text/csv")
        assert "| 1 | 2 |" in result.text
        assert "3" not in result.text

    async def test_blank_rows_are_skipped(self):
        data = b"A,B\n1,2\n,\n3,4\n"
        result = await parse_document(data, "text/csv")
        assert result.text.count("\n") == 3  # header + sep + 2 real data rows

    async def test_empty_file(self):
        result = await parse_document(b"", "text/csv", "empty.csv")
        assert result.text == "(empty file)"
        assert result.word_count == 0

    async def test_latin1_fallback(self):
        data = "Employer,City\ncafé résumé,Oakland\n".encode("latin-1")
        result = await parse_document(data, "text/csv")
        assert "café résumé" in result.text

    async def test_no_filename_still_parses(self):
        # Google Sheets' CSV export URL has no .csv extension -- content-type
        # alone (already resolved to "text/csv" by the caller) must be enough.
        data = b"A,B\n1,2\n"
        result = await parse_document(data, "text/csv")
        assert "| 1 | 2 |" in result.text
        assert result.title is None


class TestParseLaTeX:
    async def test_basic_latex(self):
        tex = rb"""\documentclass{article}
\title{My Paper}
\begin{document}
\section{Introduction}
Hello world.
\textbf{Bold text} and \textit{italic text}.
\end{document}"""
        result = await parse_document(tex, "application/x-tex", "paper.tex")
        assert result.title == "My Paper"
        assert "Introduction" in result.text
        assert "**Bold text**" in result.text
        assert "*italic text*" in result.text
        assert result.was_ocr is False

    async def test_latex_lists(self):
        tex = rb"""\begin{document}
\begin{itemize}
\item First
\item Second
\end{itemize}
\end{document}"""
        result = await parse_document(tex, "application/x-tex")
        assert "- First" in result.text
        assert "- Second" in result.text


# -- parse_document: unsupported format ---------------------------------------


class TestUnsupported:
    async def test_unsupported_mime(self):
        result = await parse_document(b"data", "application/octet-stream")
        assert "Unsupported" in result.text
        assert result.word_count == 0

    async def test_unsupported_with_filename_fallback(self):
        result = await parse_document(b"hello", "application/octet-stream", "test.txt")
        # Should fall back to filename detection and parse as text
        assert "hello" in result.text


# -- create_parse_document_tool -----------------------------------------------


class TestParseDocumentTool:
    def _create(self, ocr: OcrConfig | None = None) -> Any:
        return create_parse_document_tool({}, "Parse documents", ocr_config=ocr)

    async def test_parse_txt_via_tool(self):
        tool = self._create()
        content = base64.b64encode(b"Hello from tool test.").decode()
        result = await tool.ainvoke({"content_base64": content, "filename": "test.txt"})
        assert "Hello from tool test" in result
        assert "Words:" in result

    async def test_parse_csv_via_tool(self):
        tool = self._create()
        content = base64.b64encode(b"Employer,Count\nAcme Corp,42\n").decode()
        result = await tool.ainvoke({"content_base64": content, "filename": "warn.csv"})
        assert "| Acme Corp | 42 |" in result
        assert "Words:" in result

    async def test_invalid_base64(self):
        tool = self._create()
        result = await tool.ainvoke({"content_base64": "!!!invalid!!!", "filename": "test.txt"})
        assert "[TOOL ERROR]" in result
        assert "decode" in result

    async def test_unknown_format(self):
        tool = self._create()
        content = base64.b64encode(b"data").decode()
        result = await tool.ainvoke({"content_base64": content, "filename": "file.xyz123nope"})
        assert "[TOOL ERROR]" in result
        assert "format" in result.lower()

    async def test_truncation(self):
        tool = self._create()
        # 20K chars of text
        big_text = ("word " * 4000).encode()
        content = base64.b64encode(big_text).decode()
        result = await tool.ainvoke({"content_base64": content, "filename": "big.txt"})
        assert "[Content truncated]" in result


# -- ParseDocumentInput schema ------------------------------------------------


class TestParseDocumentInput:
    def test_schema(self):
        inp = ParseDocumentInput(content_base64="abc", filename="test.pdf")
        assert inp.content_base64 == "abc"
        assert inp.filename == "test.pdf"


# -- _merge_wrapped_table_rows / _extract_pdf_tables --------------------------
# scrape-task-07 follow-up (2026-07-16): find_tables() gets column boundaries
# right, but a long-text cell that word-wraps inside one logical PDF table row
# becomes its OWN separate row in table.extract()'s output -- live-found against
# Mississippi's real quarterly WARN Act PDF, confirmed before/after against the
# real file (git-stashed the fix, re-ran, saw the fragmented rows this fixes).


class TestMergeWrappedTableRows:
    def test_a_table_with_no_wrapped_cells_round_trips_unchanged(self):
        rows = [
            ["Date", "Employer", "Count"],
            ["1/1/2026", "Acme Corp", "42"],
            ["2/2/2026", "Beta LLC", "7"],
        ]
        assert _merge_wrapped_table_rows(rows) == rows

    def test_a_continuation_row_with_empty_first_column_merges_into_the_row_above(self):
        rows = [
            ["Date of Notice", "Company", "NAICS Description"],
            ["4/14/2026", "Greenwood Leflore Hospital", "622110 – General"],
            ["", "", "Medical and"],
            ["", "", "Surgical Hospital"],
            ["2/20/2026", "Stanley Black & Decker", "333991 – Power Tools"],
        ]
        result = _merge_wrapped_table_rows(rows)
        assert result == [
            ["Date of Notice", "Company", "NAICS Description"],
            ["4/14/2026", "Greenwood Leflore Hospital", "622110 – General Medical and Surgical Hospital"],
            ["2/20/2026", "Stanley Black & Decker", "333991 – Power Tools"],
        ]

    def test_multiple_wrapped_columns_on_the_same_continuation_row_all_merge(self):
        rows = [
            ["Date of Notice", "Company", "Event Number", "Reason"],
            ["4/17/2026", "Aramark Services, Inc", "RR-MS-", "WARN – Due"],
            ["", "", "2025-0020", "Businesses Circumstances"],
        ]
        result = _merge_wrapped_table_rows(rows)
        assert result == [
            ["Date of Notice", "Company", "Event Number", "Reason"],
            ["4/17/2026", "Aramark Services, Inc", "RR-MS- 2025-0020", "WARN – Due Businesses Circumstances"],
        ]

    def test_none_cells_are_treated_the_same_as_empty_strings(self):
        rows = [
            ["Date", "Company", "Notes"],
            ["1/1/2026", "Acme Corp", "first part"],
            [None, None, "second part"],
        ]
        result = _merge_wrapped_table_rows(rows)
        assert result == [
            ["Date", "Company", "Notes"],
            ["1/1/2026", "Acme Corp", "first part second part"],
        ]

    def test_header_row_is_never_merged_even_if_its_own_first_cell_is_empty(self):
        rows = [["", "Company"], ["1/1/2026", "Acme Corp"]]
        assert _merge_wrapped_table_rows(rows) == rows

    def test_first_data_row_is_never_merged_even_if_its_own_first_cell_is_empty(self):
        """No parent row exists yet for the very first data row to merge into --
        treated as a normal (if malformed) row rather than silently dropped."""
        rows = [["Date", "Company"], ["", "Orphan Row"]]
        assert _merge_wrapped_table_rows(rows) == [["Date", "Company"], ["", "Orphan Row"]]

    def test_header_and_first_row_protection_holds_when_the_merge_loop_actually_runs(self):
        """The two tests above only exercise the len(rows)<=2 short-circuit --
        this one has a genuine continuation row present, forcing the merge loop
        itself to run, and still proves the header is untouched and row 1 (also
        empty-first-cell) is not treated as a continuation of the header."""
        rows = [
            ["", "Company"],  # header, deliberately blank first cell too
            ["", "Orphan Row"],  # first DATA row -- no parent to merge into
            ["1/1/2026", "Acme Corp", "fragment one"],
            ["", "", "fragment two"],
        ]
        result = _merge_wrapped_table_rows(rows)
        assert result[0] == ["", "Company"]  # header untouched
        assert result[1] == ["", "Orphan Row"]  # not silently merged into the header
        assert result[2] == ["1/1/2026", "Acme Corp", "fragment one fragment two"]

    def test_header_plus_one_data_row_is_a_no_op_short_circuit(self):
        rows = [["Date", "Company"], ["1/1/2026", "Acme Corp"]]
        assert _merge_wrapped_table_rows(rows) == rows

    def test_empty_rows_list_is_a_no_op(self):
        assert _merge_wrapped_table_rows([]) == []

    def test_does_not_mutate_the_caller_supplied_rows(self):
        rows = [
            ["Date", "Company"],
            ["1/1/2026", "Acme"],
            ["", "Corp"],
        ]
        original = [list(r) for r in rows]
        _merge_wrapped_table_rows(rows)
        assert rows == original

    def test_a_continuation_column_wider_than_the_parent_row_does_not_crash(self, caplog):
        """An independent review flagged the original version of this test as
        constructing exactly this scenario but never checking what happened to
        the out-of-bounds cell -- it was silently dropped with no trace. Fixed
        function now logs the drop; this test asserts BOTH that the in-bounds
        merge still succeeds and that the drop is observable, not silent."""
        rows = [
            ["Date", "Company"],
            ["1/1/2026", "Acme"],
            ["", "extra", "wildly out of bounds"],
        ]
        with caplog.at_level("DEBUG"):
            result = _merge_wrapped_table_rows(rows)
        assert result[1] == ["1/1/2026", "Acme extra"]
        assert not any("wildly out of bounds" in str(row) for row in result)
        assert "wildly out of bounds" in caplog.text


# parity-exempt: hand-rolled subset stub of pdfplumber's third-party Table (only extract(), the only method the table-merge path calls)
class _FakeExtractedTable:
    def __init__(self, rows: list[list[Any]]) -> None:
        self._rows = rows

    def extract(self) -> list[list[Any]]:
        return self._rows


# parity-exempt: hand-rolled subset stub of pdfplumber's third-party TableFinder (only the .tables attribute, the only surface read)
class _FakeTables:
    def __init__(self, tables: list[_FakeExtractedTable]) -> None:
        self.tables = tables


# parity-exempt: hand-rolled subset stub of pdfplumber's third-party Page (only find_tables(), the only method the table-merge path calls)
class _FakePage:
    def __init__(self, tables: list[_FakeExtractedTable]) -> None:
        self._tables = tables

    def find_tables(self) -> _FakeTables:
        return _FakeTables(self._tables)


class TestExtractPdfTables:
    def test_no_tables_found_returns_empty_string(self):
        assert _extract_pdf_tables(_FakePage([])) == ""

    def _wrapped_page(self) -> _FakePage:
        return _FakePage(
            [
                _FakeExtractedTable(
                    [
                        ["Date of Notice", "Company", "NAICS Description"],
                        ["4/14/2026", "Greenwood Leflore Hospital", "622110 – General"],
                        ["", "", "Medical and Surgical Hospital"],
                    ]
                )
            ]
        )

    def test_wrapped_rows_are_merged_when_opted_in(self):
        md = _extract_pdf_tables(self._wrapped_page(), merge_wrapped_rows=True)
        assert "| 4/14/2026 | Greenwood Leflore Hospital | 622110 – General Medical and Surgical Hospital |" in md
        # the continuation fragment must never appear as its own markdown row
        assert "|  |  | Medical and Surgical Hospital |" not in md

    def test_wrapped_rows_are_left_unmerged_by_default(self):
        """merge_wrapped_rows defaults False -- an independent review correctly
        flagged the merge heuristic as unsafe to apply unconditionally to every
        document-backed target sharing this general-purpose tool (a table with a
        legitimately blank first column on an independent row would get silently
        fused into its neighbor). Only a caller that already knows its own table
        needs this opts in."""
        md = _extract_pdf_tables(self._wrapped_page())
        assert "| 4/14/2026 | Greenwood Leflore Hospital | 622110 – General |" in md
        assert "|  |  | Medical and Surgical Hospital |" in md

    def test_a_table_extraction_exception_degrades_to_empty_string(self):
        class _RaisingPage:
            def find_tables(self):
                raise RuntimeError("boom")

        assert _extract_pdf_tables(_RaisingPage()) == ""
