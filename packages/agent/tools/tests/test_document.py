"""Tests for document parsing module."""

from __future__ import annotations

import base64
from typing import Any

import pytest

from threetears.agent.tools.document import (
    DocumentResult,
    OcrConfig,
    ParseDocumentInput,
    create_parse_document_tool,
    detect_mime_from_filename,
    parse_document,
)


# -- OcrConfig ---------------------------------------------------------------


class TestOcrConfig:
    def test_defaults(self):
        cfg = OcrConfig()
        assert cfg.enabled is False
        assert cfg.language == "eng"

    def test_custom(self):
        cfg = OcrConfig(enabled=True, language="deu")
        assert cfg.enabled is True
        assert cfg.language == "deu"


# -- detect_mime_from_filename ------------------------------------------------


class TestDetectMime:
    @pytest.mark.parametrize(
        "filename, expected",
        [
            ("report.pdf", "application/pdf"),
            ("doc.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
            ("data.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
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
