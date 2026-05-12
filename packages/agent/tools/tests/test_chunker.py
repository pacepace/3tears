"""Tests for the content chunker."""

from threetears.agent.tools.chunker import (
    ChunkResult,
    chunk_by_headers,
    chunk_by_lines,
    chunk_by_sections,
    chunk_content,
    register_chunk_strategy,
)


def test_short_content_returns_empty() -> None:
    """Content under threshold returns no chunks."""
    result = chunk_content("Short content", strategy_hint="anything")
    assert result == []


def test_sections_strategy_splits_on_double_hash() -> None:
    """Section-based chunking splits on ## headers."""
    content = (
        "Scan of 10.0.0.1 complete.\n"
        "Tools executed: nmap\n"
        "Total findings: 5\n"
        "Severity breakdown: 2 critical, 1 high, 2 medium\n\n"
        "## Critical\n"
        "- [CRITICAL] RCE via Java deserialization in Apache Struts — 10.0.0.1:8080\n"
        "  Unvalidated ObjectInputStream allows remote code execution via crafted serialized objects.\n"
        "- [CRITICAL] SQL Injection in /api/login endpoint — 10.0.0.1:443\n"
        "  Unparameterized query in the authentication handler allows full database access.\n\n"
        "## High\n"
        "- [HIGH] Reflected XSS in search parameter — 10.0.0.1:443\n"
        "  User input in the q parameter is echoed without encoding in the response body.\n\n"
        "## Medium\n"
        "- [MEDIUM] Missing Content-Security-Policy header — 10.0.0.1:443\n"
        "  No CSP header present, increasing risk of XSS and data injection attacks.\n"
        "- [MEDIUM] Server version disclosure via Server header — 10.0.0.1:8080\n"
        "  Apache/2.4.41 version string disclosed in HTTP response headers.\n"
    )
    chunks = chunk_by_sections(content)

    assert len(chunks) >= 2
    assert all(isinstance(c, ChunkResult) for c in chunks)
    assert all(c.short_desc for c in chunks)
    assert all(c.long_desc for c in chunks)
    assert all(c.content for c in chunks)


def test_lines_strategy_groups_by_line_count() -> None:
    """Line-based chunking groups into ~20 line blocks."""
    lines = [f"Line {i}: some content about finding #{i}" for i in range(60)]
    content = "\n".join(lines)

    chunks = chunk_by_lines(content)

    assert len(chunks) >= 2
    for chunk in chunks:
        assert len(chunk.content) > 0
        assert len(chunk.short_desc) <= 150


def test_headers_strategy_splits_on_hashes() -> None:
    """Header-based chunking splits on # or ## headers."""
    content = (
        "# CVE-2024-1234\n\n"
        "A critical remote code execution vulnerability in Apache HTTPd server.\n\n"
        "## Description\n\n"
        "The vulnerability allows unauthenticated attackers to execute\n"
        "arbitrary code via a crafted HTTP request to the /api endpoint.\n"
        "This affects versions 2.0 through 2.5 of the application framework.\n"
        "The root cause is improper input validation in the request parser\n"
        "which fails to sanitize user-controlled headers before processing.\n\n"
        "## Impact\n\n"
        "Full system compromise with CVSS score 9.8 (Critical).\n"
        "All deployments using the default configuration are affected.\n"
        "An attacker can gain remote code execution as the web server user.\n\n"
        "## Remediation\n\n"
        "Upgrade to version 2.6 or later which includes the security fix.\n"
        "Apply the security patch from the vendor advisory SA-2024-001.\n"
        "Restrict network access to the /api endpoint using firewall rules.\n"
    )
    chunks = chunk_by_headers(content)

    assert len(chunks) >= 2
    assert any("CVE" in c.short_desc or "Description" in c.short_desc for c in chunks)


def test_sequential_indices() -> None:
    """Chunks have sequential indices starting from 0."""
    lines = [f"Line {i}: content" for i in range(80)]
    content = "\n".join(lines)

    chunks = chunk_by_lines(content)

    for i, chunk in enumerate(chunks):
        assert chunk.chunk_index == i


def test_preserves_all_content() -> None:
    """All original content is represented across chunks."""
    lines = [f"Finding {i}: vulnerability details" for i in range(40)]
    content = "\n".join(lines)

    chunks = chunk_by_lines(content)

    all_chunk_content = "\n".join(c.content for c in chunks)
    for line in lines:
        assert line in all_chunk_content


def test_short_desc_within_limit() -> None:
    """Chunk short_desc is always ≤150 chars."""
    content = "\n".join([f"A very long finding title for item {i} with extra details" for i in range(50)])
    chunks = chunk_by_lines(content)

    for chunk in chunks:
        assert len(chunk.short_desc) <= 150


def test_long_desc_within_limit() -> None:
    """Chunk long_desc is always ≤500 chars."""
    content = "\n".join([f"Line {i}: detailed content" for i in range(50)])
    chunks = chunk_by_lines(content)

    for chunk in chunks:
        assert len(chunk.long_desc) <= 500


def test_registered_strategy_used() -> None:
    """A registered strategy is used when the hint matches."""
    called = []

    def custom_strategy(content: str) -> list[ChunkResult]:
        """Custom test strategy.

        :param content: content to chunk
        :ptype content: str
        :return: single chunk
        :rtype: list[ChunkResult]
        """
        called.append(True)
        return [ChunkResult(chunk_index=0, short_desc="custom", long_desc="", content=content)]

    register_chunk_strategy("my_tool", custom_strategy)
    result = chunk_content("x" * 600, strategy_hint="my_tool")

    assert len(called) == 1
    assert result[0].short_desc == "custom"


def test_default_web_fetch_uses_headers() -> None:
    """The default web_fetch strategy uses header-based chunking."""
    content = (
        "# Title\n\nIntro paragraph with enough content to exceed the threshold.\n"
        + "Extra lines.\n" * 30
        + "\n## Section Two\n\nMore content here.\n"
        + "Additional lines.\n" * 20
    )
    chunks = chunk_content(content, strategy_hint="web_fetch")

    assert len(chunks) >= 2


def test_unknown_hint_falls_back_to_lines() -> None:
    """Unknown hint falls back to line-based chunking."""
    lines = [f"Line {i}: some content about finding #{i}" for i in range(60)]
    content = "\n".join(lines)

    chunks = chunk_content(content, strategy_hint="unknown_tool")

    assert len(chunks) >= 2
