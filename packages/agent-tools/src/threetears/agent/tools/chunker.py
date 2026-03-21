"""Content chunker — splits large tool outputs into semantic chunks.

Each chunk gets a short_desc (≤150 chars) and long_desc (≤500 chars)
for hierarchical retrieval. Chunking strategies vary by hint:
section-based for structured output, header-based for web content,
line-group-based as fallback.

The strategy registry is configurable — callers can register custom
hint → strategy mappings for domain-specific tool outputs.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

_MIN_CHUNK_CONTENT = 500
_LINES_PER_CHUNK = 20


@dataclass
class ChunkResult:
    """Result of chunking a piece of content.

    :param chunk_index: ordering within parent (0-based)
    :ptype chunk_index: int
    :param short_desc: chunk summary (≤150 chars)
    :ptype short_desc: str
    :param long_desc: expanded description (≤500 chars)
    :ptype long_desc: str
    :param content: full chunk content
    :ptype content: str
    """

    chunk_index: int
    short_desc: str
    long_desc: str
    content: str


# Strategy type: takes content string, returns list of ChunkResult
ChunkStrategy = Callable[[str], list[ChunkResult]]

# Default strategy registry: hint string → chunking function
_STRATEGY_REGISTRY: dict[str, ChunkStrategy] = {}


def register_chunk_strategy(hint: str, strategy: ChunkStrategy) -> None:
    """Register a chunking strategy for a tool hint.

    :param hint: tool name or strategy hint (e.g. 'web_fetch', 'run_scan')
    :ptype hint: str
    :param strategy: function that takes content and returns chunks
    :ptype strategy: ChunkStrategy
    """
    _STRATEGY_REGISTRY[hint] = strategy


def chunk_content(content: str, strategy_hint: str = "") -> list[ChunkResult]:
    """Split content into semantic chunks based on strategy hint.

    Returns empty list if content is too short to chunk.
    Looks up the hint in the strategy registry first, then
    falls back to line-based chunking.

    :param content: raw content to split
    :ptype content: str
    :param strategy_hint: tool name or hint for strategy selection
    :ptype strategy_hint: str
    :return: list of chunk results
    :rtype: list[ChunkResult]
    """
    if len(content) < _MIN_CHUNK_CONTENT:
        return []

    strategy = _STRATEGY_REGISTRY.get(strategy_hint)
    if strategy is not None:
        return strategy(content)

    return chunk_by_lines(content)


def chunk_by_sections(content: str) -> list[ChunkResult]:
    """Chunk content by markdown-style ## section headers.

    Splits on ## headers. Falls back to line-based chunking
    if no headers found.

    :param content: content with markdown sections
    :ptype content: str
    :return: list of chunks
    :rtype: list[ChunkResult]
    """
    sections = re.split(r"\n(?=##\s)", content)
    if len(sections) < 2:
        return chunk_by_lines(content)

    chunks: list[ChunkResult] = []
    for i, section in enumerate(sections):
        section = section.strip()
        if not section:
            continue
        short_desc = _extract_short_desc(section)
        long_desc = section[:500]
        chunks.append(
            ChunkResult(
                chunk_index=i,
                short_desc=short_desc,
                long_desc=long_desc,
                content=section,
            )
        )

    return _reindex(chunks)


def chunk_by_headers(content: str) -> list[ChunkResult]:
    """Chunk content by markdown headers (# or ## or ###).

    :param content: content with markdown headers
    :ptype content: str
    :return: list of chunks
    :rtype: list[ChunkResult]
    """
    sections = re.split(r"\n(?=#{1,3}\s)", content)
    if len(sections) < 2:
        return chunk_by_lines(content)

    chunks: list[ChunkResult] = []
    for i, section in enumerate(sections):
        section = section.strip()
        if not section:
            continue
        short_desc = _extract_short_desc(section)
        long_desc = section[:500]
        chunks.append(
            ChunkResult(
                chunk_index=i,
                short_desc=short_desc,
                long_desc=long_desc,
                content=section,
            )
        )

    return _reindex(chunks)


def chunk_by_lines(content: str) -> list[ChunkResult]:
    """Chunk content into groups of ~20 lines.

    :param content: raw content
    :ptype content: str
    :return: list of chunks
    :rtype: list[ChunkResult]
    """
    lines = content.split("\n")
    if len(lines) <= _LINES_PER_CHUNK:
        return []

    chunks: list[ChunkResult] = []
    for i in range(0, len(lines), _LINES_PER_CHUNK):
        chunk_lines = lines[i : i + _LINES_PER_CHUNK]
        chunk_content = "\n".join(chunk_lines)
        first_line = chunk_lines[0].strip() if chunk_lines else ""
        short_desc = _make_short_desc(first_line, i, len(lines))
        long_desc = chunk_content[:500]
        chunks.append(
            ChunkResult(
                chunk_index=len(chunks),
                short_desc=short_desc,
                long_desc=long_desc,
                content=chunk_content,
            )
        )

    return chunks


def _extract_short_desc(section: str) -> str:
    """Extract a short description from a section.

    :param section: section content
    :ptype section: str
    :return: short description (≤150 chars)
    :rtype: str
    """
    first_line = section.split("\n")[0].strip()
    first_line = re.sub(r"^#+\s*", "", first_line)
    return first_line[:150]


def _make_short_desc(first_line: str, start_idx: int, total_lines: int) -> str:
    """Make a short description for a line-based chunk.

    :param first_line: first line of the chunk
    :ptype first_line: str
    :param start_idx: starting line index
    :ptype start_idx: int
    :param total_lines: total number of lines
    :ptype total_lines: int
    :return: short description (≤150 chars)
    :rtype: str
    """
    end_idx = min(start_idx + _LINES_PER_CHUNK, total_lines)
    prefix = f"Lines {start_idx + 1}-{end_idx}"
    desc = f"{prefix}: {first_line}" if first_line else prefix
    return desc[:150]


def _reindex(chunks: list[ChunkResult]) -> list[ChunkResult]:
    """Re-index chunks to have sequential indices from 0.

    :param chunks: list of chunks to reindex
    :ptype chunks: list[ChunkResult]
    :return: reindexed chunks
    :rtype: list[ChunkResult]
    """
    for i, chunk in enumerate(chunks):
        chunk.chunk_index = i
    return chunks


# Register default strategies for common tool types
register_chunk_strategy("web_fetch", chunk_by_headers)
