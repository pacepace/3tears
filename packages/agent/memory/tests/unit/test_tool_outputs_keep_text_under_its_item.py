"""What each memory tool prints for an item's free text, pinned exactly.

An item in a tool's result is a heading and the text that belongs to it. Every
line of that text sits indented under the heading, none is blank, and an item
whose text is empty still shows its heading (and so its id). Items are parted
as the tool parts them. These tests drive each tool through its factory and
compare the whole rendered result.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

from langchain_core.embeddings import Embeddings

from threetears.agent.memory.authorize import MemoryAuthorizerDependencies
from threetears.agent.memory.tools import (
    load_chunk_search_tool,
    load_memory_recall_tool,
    load_memory_search_tool,
)

#: A many-line text: a blank line, an indented line, trailing spaces.
MANY_LINES = "first line\n\n  indented line  \nlast line"
#: What the text above becomes under a two-space indent.
MANY_LINES_UNDER = "  first line\n    indented line\n  last line"

MEMORY_ID = UUID("00000000-0000-4000-8000-000000000001")
MEDIA_ID = UUID("00000000-0000-4000-8000-000000000002")
CONTENT_ID = UUID("00000000-0000-4000-8000-000000000003")
CHUNK_ID = UUID("00000000-0000-4000-8000-000000000004")
CHUNK_ID_2 = UUID("00000000-0000-4000-8000-000000000005")
MEDIA_ID_2 = UUID("00000000-0000-4000-8000-000000000006")
CONTENT_ID_2 = UUID("00000000-0000-4000-8000-000000000007")

AGENT_ID = uuid4()
CUSTOMER_ID = uuid4()
USER_ID = uuid4()


class FakeEmbeddings(Embeddings):
    """Embeds every text as one fixed vector."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2]

    async def aembed_query(self, text: str) -> list[float]:
        return [0.1, 0.2]


# parity-exempt: answers only the memory reads memory_search and memory_recall make; the SQL is covered by collection tests
class FakeMemories:
    """A memories collection that holds a fixed list of rows."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    async def search_by_ids(self, ids: list[UUID], user_id: UUID, *, agent_id: UUID) -> list[dict[str, Any]]:
        return self.rows

    async def search_by_semantic(self, **kwargs: Any) -> list[dict[str, Any]]:
        return self.rows

    async def search_by_fts(self, **kwargs: Any) -> list[dict[str, Any]]:
        return []

    async def find_for_recall(self, *, memory_id: UUID, user_id: UUID, agent_id: UUID) -> dict[str, Any] | None:
        return self.rows[0] if self.rows else None


# parity-exempt: answers only the media reads memory_search makes; the SQL is covered by collection tests
class FakeMedia:
    """A media-content collection that holds a fixed list of rows."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    async def search_by_ids(self, ids: list[UUID], user_id: UUID, *, agent_id: UUID) -> list[dict[str, Any]]:
        return self.rows

    async def search_by_semantic(self, **kwargs: Any) -> list[dict[str, Any]]:
        return self.rows

    async def search_by_fts(self, **kwargs: Any) -> list[dict[str, Any]]:
        return []


# parity-exempt: answers the chunk reads memory_search, memory_recall and chunk_search make; the SQL is covered elsewhere
class FakeChunks:
    """A chunk collection that holds a fixed list of rows."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    async def search_by_ids(self, ids: list[UUID], user_id: UUID, *, agent_id: UUID) -> list[dict[str, Any]]:
        return self.rows

    async def search_by_semantic(self, **kwargs: Any) -> list[dict[str, Any]]:
        return self.rows

    async def hybrid_search(self, **kwargs: Any) -> list[dict[str, Any]]:
        return self.rows

    async def find_by_memory_id(self, memory_id: UUID, **kwargs: Any) -> list[dict[str, Any]]:
        return self.rows

    async def count_by_memory_id(self, memory_id: UUID, **kwargs: Any) -> int:
        return len(self.rows)


def memory_row(content: str) -> dict[str, Any]:
    return {
        "memory_id": MEMORY_ID,
        "type_memory": "fact",
        "content": content,
        "date_created": None,
        "similarity": 0.9,
    }


def media_row(
    content: str, *, media_id: UUID = MEDIA_ID, content_id: UUID = CONTENT_ID, **extra: Any
) -> dict[str, Any]:
    return {
        "content_id": content_id,
        "media_id": media_id,
        "content": content,
        "metadata_json": {"title": "Trip notes"},
        "media_category": "document",
        "content_type": None,
        "date_created": None,
        "similarity": 0.8,
        **extra,
    }


def passage_row(content: str, *, chunk_id: UUID = CHUNK_ID) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "content": content,
        "metadata_json": {"title": "Handbook"},
        "heading_context": None,
        "page_number": 3,
        "date_created": None,
        "similarity": 0.7,
    }


async def run_memory_search(
    authorizer: MemoryAuthorizerDependencies,
    *,
    memories: list[dict[str, Any]] | None = None,
    media: list[dict[str, Any]] | None = None,
    chunks: list[dict[str, Any]] | None = None,
    **call: Any,
) -> str:
    (search,) = await load_memory_search_tool(
        USER_ID,
        FakeEmbeddings(),
        AGENT_ID,
        CUSTOMER_ID,
        authorizer,
        FakeMemories(memories or []),  # type: ignore[arg-type]
        FakeMedia(media or []),  # type: ignore[arg-type]
        FakeChunks(chunks or []),  # type: ignore[arg-type]
    )
    result = await search.ainvoke(call)
    assert isinstance(result, str)
    return result


async def run_memory_recall(
    authorizer: MemoryAuthorizerDependencies, *, memory_content: str, chunks: list[dict[str, Any]]
) -> str:
    (recall,) = await load_memory_recall_tool(
        USER_ID,
        AGENT_ID,
        CUSTOMER_ID,
        authorizer,
        FakeMemories([memory_row(memory_content)]),  # type: ignore[arg-type]
        FakeMedia([]),  # type: ignore[arg-type]
        FakeChunks(chunks),  # type: ignore[arg-type]
    )
    result = await recall.ainvoke({"memory_id": str(MEMORY_ID)})
    assert isinstance(result, str)
    return result


async def run_chunk_search(authorizer: MemoryAuthorizerDependencies, chunks: list[dict[str, Any]]) -> str:
    (search,) = await load_chunk_search_tool(
        USER_ID,
        AGENT_ID,
        CUSTOMER_ID,
        FakeEmbeddings(),
        authorizer,
        FakeChunks(chunks),  # type: ignore[arg-type]
    )
    result = await search.ainvoke({"query": "needle"})
    assert isinstance(result, str)
    return result


# -- memory_search by query: the memory, file and passage listings ----------------------------------------------------


async def test_memory_search_lists_a_memorys_lines_all_under_its_heading(
    permissive_memory_authorizer: MemoryAuthorizerDependencies,
) -> None:
    result = await run_memory_search(
        permissive_memory_authorizer, memories=[memory_row(MANY_LINES)], query="needle", mode="fuzzy"
    )
    assert result == (
        "Found 1 relevant memories:\n"
        f"- [memory:{MEMORY_ID}] [fact] [match 0.90] first line\n"
        "    indented line\n"
        "  last line"
    )


async def test_memory_search_shows_the_heading_of_a_memory_with_no_text(
    permissive_memory_authorizer: MemoryAuthorizerDependencies,
) -> None:
    result = await run_memory_search(
        permissive_memory_authorizer, memories=[memory_row(" \n\n ")], query="needle", mode="fuzzy"
    )
    assert result == f"Found 1 relevant memories:\n- [memory:{MEMORY_ID}] [fact] [match 0.90]"


async def test_memory_search_lists_a_documents_lines_under_its_media_heading(
    permissive_memory_authorizer: MemoryAuthorizerDependencies,
) -> None:
    result = await run_memory_search(
        permissive_memory_authorizer, media=[media_row(MANY_LINES)], query="needle", mode="fuzzy"
    )
    assert result == (f"Found 1 files:\n- [media:{MEDIA_ID}] [Trip notes] first line\n    indented line\n  last line")


async def test_memory_search_lists_an_images_lines_under_its_media_heading(
    permissive_memory_authorizer: MemoryAuthorizerDependencies,
) -> None:
    result = await run_memory_search(
        permissive_memory_authorizer,
        media=[media_row(MANY_LINES, media_category="image")],
        query="needle",
        mode="fuzzy",
    )
    assert result == (f"Found 1 files:\n- [media:{MEDIA_ID}] first line\n    indented line\n  last line")


async def test_memory_search_shows_the_heading_of_a_document_and_an_image_with_no_text(
    permissive_memory_authorizer: MemoryAuthorizerDependencies,
) -> None:
    result = await run_memory_search(
        permissive_memory_authorizer,
        media=[
            media_row("  "),
            media_row("\n", media_id=MEDIA_ID_2, content_id=CONTENT_ID_2, media_category="image"),
        ],
        query="needle",
        mode="fuzzy",
    )
    assert result == f"Found 2 files:\n- [media:{MEDIA_ID}] [Trip notes]\n- [media:{MEDIA_ID_2}]"


async def test_memory_search_lists_a_passages_lines_under_its_chunk_heading(
    permissive_memory_authorizer: MemoryAuthorizerDependencies,
) -> None:
    result = await run_memory_search(
        permissive_memory_authorizer, chunks=[passage_row(MANY_LINES)], query="needle", mode="fuzzy"
    )
    assert result == (
        f'Found 1 passages:\n- [chunk:{CHUNK_ID}] [from "Handbook" p.3] first line\n    indented line\n  last line'
    )


async def test_memory_search_shows_the_heading_of_a_passage_with_no_text(
    permissive_memory_authorizer: MemoryAuthorizerDependencies,
) -> None:
    result = await run_memory_search(
        permissive_memory_authorizer, chunks=[passage_row("\n \n")], query="needle", mode="fuzzy"
    )
    assert result == f'Found 1 passages:\n- [chunk:{CHUNK_ID}] [from "Handbook" p.3]'


async def test_memory_search_parts_its_listings_by_one_blank_line_and_keeps_each_items_text_together(
    permissive_memory_authorizer: MemoryAuthorizerDependencies,
) -> None:
    result = await run_memory_search(
        permissive_memory_authorizer,
        memories=[memory_row(MANY_LINES)],
        media=[media_row(MANY_LINES)],
        chunks=[passage_row(MANY_LINES)],
        query="needle",
        mode="fuzzy",
    )
    assert result == (
        "Found 1 relevant memories:\n"
        f"- [memory:{MEMORY_ID}] [fact] [match 0.90] first line\n"
        "    indented line\n"
        "  last line\n"
        "\n"
        "Found 1 files:\n"
        f"- [media:{MEDIA_ID}] [Trip notes] first line\n"
        "    indented line\n"
        "  last line\n"
        "\n"
        "Found 1 passages:\n"
        f'- [chunk:{CHUNK_ID}] [from "Handbook" p.3] first line\n'
        "    indented line\n"
        "  last line"
    )


# -- memory_search by id ----------------------------------------------------------------------------------------------


async def test_memory_search_by_id_lists_a_memorys_lines_all_under_its_heading(
    permissive_memory_authorizer: MemoryAuthorizerDependencies,
) -> None:
    result = await run_memory_search(
        permissive_memory_authorizer, memories=[memory_row(MANY_LINES)], ids=[str(MEMORY_ID)]
    )
    assert result == (
        f"Found 1 memories:\n- [memory:{MEMORY_ID}] [fact] [match 0.90] first line\n    indented line\n  last line"
    )


async def test_memory_search_by_id_lists_a_files_lines_under_its_media_heading(
    permissive_memory_authorizer: MemoryAuthorizerDependencies,
) -> None:
    result = await run_memory_search(permissive_memory_authorizer, media=[media_row(MANY_LINES)], ids=[str(CONTENT_ID)])
    assert result == (f"Found 1 files:\n- [media:{MEDIA_ID}] [Trip notes] first line\n    indented line\n  last line")


async def test_memory_search_by_id_lists_a_passages_lines_under_its_chunk_heading(
    permissive_memory_authorizer: MemoryAuthorizerDependencies,
) -> None:
    result = await run_memory_search(
        permissive_memory_authorizer, chunks=[passage_row(MANY_LINES)], ids=[str(CHUNK_ID)]
    )
    assert result == (
        f'Found 1 passages:\n- [chunk:{CHUNK_ID}] [from "Handbook" p.3] first line\n    indented line\n  last line'
    )


async def test_memory_search_by_id_shows_the_heading_of_each_item_with_no_text(
    permissive_memory_authorizer: MemoryAuthorizerDependencies,
) -> None:
    result = await run_memory_search(
        permissive_memory_authorizer,
        memories=[memory_row("  ")],
        media=[media_row("\n\n")],
        chunks=[passage_row(" ")],
        ids=[str(MEMORY_ID), str(CONTENT_ID), str(CHUNK_ID)],
    )
    assert result == (
        "Found 1 memories:\n"
        f"- [memory:{MEMORY_ID}] [fact] [match 0.90]\n"
        "\n"
        "Found 1 files:\n"
        f"- [media:{MEDIA_ID}] [Trip notes]\n"
        "\n"
        "Found 1 passages:\n"
        f'- [chunk:{CHUNK_ID}] [from "Handbook" p.3]'
    )


# -- memory_recall: the part list -------------------------------------------------------------------------------------


async def test_memory_recall_lists_each_parts_lines_indented_under_its_part_heading(
    permissive_memory_authorizer: MemoryAuthorizerDependencies,
) -> None:
    result = await run_memory_recall(
        permissive_memory_authorizer,
        memory_content="the memory",
        chunks=[{"chunk_id": CHUNK_ID, "chunk_index": 0, "content": MANY_LINES}],
    )
    assert result == (
        f"[memory:{MEMORY_ID}] [fact]\n"
        "the memory\n"
        "\n"
        "Part 0 of 1.\n"
        f"  [chunk:{CHUNK_ID}] (part 0)\n"
        "    first line\n"
        "      indented line\n"
        "    last line"
    )


async def test_memory_recall_keeps_parts_together_with_no_blank_line_between_their_lines(
    permissive_memory_authorizer: MemoryAuthorizerDependencies,
) -> None:
    result = await run_memory_recall(
        permissive_memory_authorizer,
        memory_content="the memory",
        chunks=[
            {"chunk_id": CHUNK_ID, "chunk_index": 0, "content": MANY_LINES},
            {"chunk_id": CHUNK_ID_2, "chunk_index": 1, "content": "second\n\npart"},
        ],
    )
    assert result == (
        f"[memory:{MEMORY_ID}] [fact]\n"
        "the memory\n"
        "\n"
        "Parts 0-1 of 2.\n"
        f"  [chunk:{CHUNK_ID}] (part 0)\n"
        "    first line\n"
        "      indented line\n"
        "    last line\n"
        f"  [chunk:{CHUNK_ID_2}] (part 1)\n"
        "    second\n"
        "    part"
    )


async def test_memory_recall_shows_the_heading_of_a_part_with_no_text(
    permissive_memory_authorizer: MemoryAuthorizerDependencies,
) -> None:
    result = await run_memory_recall(
        permissive_memory_authorizer,
        memory_content="the memory",
        chunks=[{"chunk_id": CHUNK_ID, "chunk_index": 0, "content": " \n\n "}],
    )
    assert result == (f"[memory:{MEMORY_ID}] [fact]\nthe memory\n\nPart 0 of 1.\n  [chunk:{CHUNK_ID}] (part 0)")


# -- chunk_search -----------------------------------------------------------------------------------------------------


def hit(content: str, *, chunk_id: UUID = CHUNK_ID) -> dict[str, Any]:
    return {"chunk_id": chunk_id, "memory_id": MEMORY_ID, "content": content, "title": None, "page_number": None}


async def test_chunk_search_keeps_a_hits_heading_and_lines_in_one_block(
    permissive_memory_authorizer: MemoryAuthorizerDependencies,
) -> None:
    result = await run_chunk_search(permissive_memory_authorizer, [hit(MANY_LINES)])
    assert result == (f"[chunk:{CHUNK_ID}] (from [memory:{MEMORY_ID}])\n  first line\n    indented line\n  last line")


async def test_chunk_search_parts_hits_by_one_blank_line(
    permissive_memory_authorizer: MemoryAuthorizerDependencies,
) -> None:
    result = await run_chunk_search(
        permissive_memory_authorizer, [hit(MANY_LINES), hit("second\n\nhit", chunk_id=CHUNK_ID_2)]
    )
    assert result == (
        f"[chunk:{CHUNK_ID}] (from [memory:{MEMORY_ID}])\n"
        "  first line\n"
        "    indented line\n"
        "  last line\n"
        "\n"
        f"[chunk:{CHUNK_ID_2}] (from [memory:{MEMORY_ID}])\n"
        "  second\n"
        "  hit"
    )


async def test_chunk_search_shows_the_heading_of_a_hit_with_no_text(
    permissive_memory_authorizer: MemoryAuthorizerDependencies,
) -> None:
    result = await run_chunk_search(permissive_memory_authorizer, [hit("  \n ")])
    assert result == f"[chunk:{CHUNK_ID}] (from [memory:{MEMORY_ID}])"
