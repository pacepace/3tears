"""exact-output tests for the memory context block :class:`MemoryRetriever` hands the model.

every memory, media excerpt and chunk headline is free text that can span lines. each item is
its heading line followed by every line of its text indented under it, with no blank line
inside an item, so the model never reads one item's lines as the next item's. these tests
drive ``retrieve_with_candidates`` over stubbed collections and pin the fenced items exactly.
"""

from __future__ import annotations

import re
import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from threetears.agent.memory.retrieval import MemoryRetriever, RetrievalResult
from threetears.agent.memory.types import MemoryConfig

_MANY_LINES = "first line\n\n  indented line\nlast line"
_MANY_LINES_UNDER = ["  first line", "    indented line", "  last line"]
_MEMORY_ID = uuid.UUID("aaaaaaaa-0000-0000-0000-000000000001")
_MEDIA_ID = uuid.UUID("bbbbbbbb-0000-0000-0000-000000000002")
_CONTENT_ID = uuid.UUID("bbbbbbbb-0000-0000-0000-000000000003")
_CHUNK_ID = uuid.UUID("cccccccc-0000-0000-0000-000000000004")
_FENCE = re.compile(r"<untrusted nonce=(\w+)>\n(.*?)\n</untrusted nonce=\1>", re.DOTALL)


def _retriever(
    *,
    memories: list[dict[str, Any]] | None = None,
    media: list[dict[str, Any]] | None = None,
    chunks: list[dict[str, Any]] | None = None,
) -> MemoryRetriever:
    """a retriever whose three collections answer with the given rows."""
    embedder = MagicMock()
    embedder.aembed_query = AsyncMock(return_value=[1.0, 0.0])
    memories_collection = MagicMock()
    memories_collection.hybrid_search = AsyncMock(return_value=memories or [])
    memories_collection.bump_salience = AsyncMock()
    media_collection = MagicMock()
    media_collection.hybrid_search = AsyncMock(return_value=media or [])
    chunk_collection = MagicMock()
    chunk_collection.hybrid_search = AsyncMock(return_value=chunks or [])
    return MemoryRetriever(
        config=MemoryConfig(),
        embedding_provider=embedder,
        authorizer=MagicMock(),
        memories_collection=memories_collection,
        media_content_collection=media_collection,
        memory_chunk_collection=chunk_collection,
    )


async def _retrieve(retriever: MemoryRetriever) -> RetrievalResult:
    return await retriever.retrieve_with_candidates(
        uuid.uuid4(), "what do you remember", agent_id=uuid.uuid4(), customer_id=uuid.uuid4()
    )


def _memory(*, content: str, summary: str | None = None, score: float = 0.9) -> dict[str, Any]:
    return {
        "memory_id": _MEMORY_ID,
        "content": content,
        "summary": summary,
        "type_memory": "fact",
        "hybrid_score": score,
        "embedding": [1.0, 0.0],
    }


def _media(*, content: str, media_id: uuid.UUID | None = _MEDIA_ID) -> dict[str, Any]:
    return {
        "content_id": _CONTENT_ID,
        "media_id": media_id,
        "content": content,
        "summary": None,
        "hybrid_score": 0.9,
        "embedding": [1.0, 0.0],
    }


def _chunk(*, summary: str | None, memory_id: uuid.UUID | None = None, **extra: Any) -> dict[str, Any]:
    return {
        "chunk_id": _CHUNK_ID,
        "summary": summary,
        "content": "ignored",
        "memory_id": memory_id,
        "hybrid_score": 0.5,
        "embedding": [1.0, 0.0],
        **extra,
    }


def _fenced_items(block: str | None) -> list[list[str]]:
    """the lines of each fenced section of a rendered block, in order."""
    assert block is not None
    return [body.split("\n") for _nonce, body in _FENCE.findall(block)]


class TestMemoryItems:
    async def test_every_line_of_a_memory_sits_indented_under_its_heading(self) -> None:
        result = await _retrieve(_retriever(memories=[_memory(content=_MANY_LINES)]))
        assert _fenced_items(result.context) == [
            [f"- [memory:{_MEMORY_ID}] first line", "    indented line", "  last line (in full)"]
        ]

    async def test_a_partial_memory_shows_its_summary_lines_under_the_heading(self) -> None:
        result = await _retrieve(_retriever(memories=[_memory(content="x", summary=_MANY_LINES, score=0.5)]))
        assert _fenced_items(result.context) == [
            [f"- [memory:{_MEMORY_ID}] first line", "    indented line", "  last line"]
        ]

    async def test_two_memories_have_no_blank_line_between_or_inside_them(self) -> None:
        second = {**_memory(content="other\n\nthing"), "memory_id": uuid.UUID("aaaaaaaa-0000-0000-0000-000000000009")}
        result = await _retrieve(_retriever(memories=[_memory(content=_MANY_LINES), second]))
        [section] = _fenced_items(result.context)
        assert "" not in section
        assert section[3:] == ["- [memory:aaaaaaaa-0000-0000-0000-000000000009] other", "  thing (in full)"]

    async def test_a_memory_with_blank_text_still_shows_its_id_line(self) -> None:
        result = await _retrieve(_retriever(memories=[_memory(content="  \n\n ")]))
        assert _fenced_items(result.context) == [[f"- [memory:{_MEMORY_ID}] (in full)"]]


class TestMediaItems:
    async def test_every_line_of_a_media_excerpt_sits_indented_under_its_heading(self) -> None:
        result = await _retrieve(_retriever(media=[_media(content=_MANY_LINES)]))
        assert _fenced_items(result.context) == [
            [f"- [media:{_MEDIA_ID}] first line", "    indented line", "  last line (in full)"]
        ]

    async def test_a_media_excerpt_without_a_media_id_shows_its_content_id(self) -> None:
        result = await _retrieve(_retriever(media=[_media(content="a\nb", media_id=None)]))
        assert _fenced_items(result.context) == [[f"- [media:{_CONTENT_ID}] a", "  b (in full)"]]

    async def test_a_media_excerpt_with_blank_text_still_shows_its_id_line(self) -> None:
        result = await _retrieve(_retriever(media=[_media(content="\n \n")]))
        assert _fenced_items(result.context) == [[f"- [media:{_MEDIA_ID}] (in full)"]]


class TestChunkItems:
    async def test_every_line_of_a_chunk_headline_sits_indented_under_its_heading(self) -> None:
        result = await _retrieve(_retriever(chunks=[_chunk(summary=_MANY_LINES)]))
        assert _fenced_items(result.context) == [
            [f"- [chunk:{_CHUNK_ID}] first line", "    indented line", "  last line"]
        ]

    async def test_a_chunk_heading_carries_its_location_before_the_headline(self) -> None:
        chunk = _chunk(summary="a\nb", title="Doc", page_number=3, heading_context="Intro")
        result = await _retrieve(_retriever(chunks=[chunk]))
        assert _fenced_items(result.context) == [[f'- [chunk:{_CHUNK_ID} "Doc" p.3 "Intro"] a', "  b"]]

    async def test_a_parent_summary_is_joined_to_one_line_in_the_heading(self) -> None:
        memory = _memory(content="x", summary="parent one\n\n  parent two", score=0.5)
        chunk = _chunk(summary="a\nb", memory_id=_MEMORY_ID)
        result = await _retrieve(_retriever(memories=[memory], chunks=[chunk]))
        assert _fenced_items(result.material_context) == [
            [f'- [chunk:{_CHUNK_ID}] (from [memory:{_MEMORY_ID}]: "parent one parent two") a', "  b"]
        ]

    async def test_a_chunk_with_blank_text_still_shows_its_id_line(self) -> None:
        result = await _retrieve(_retriever(chunks=[_chunk(summary=None, content="")]))
        assert _fenced_items(result.context) == [[f"- [chunk:{_CHUNK_ID}]"]]
