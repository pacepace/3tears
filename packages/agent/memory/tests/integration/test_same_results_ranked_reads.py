"""the memory package's owner-scoped search reads, moved onto the keyed helpers, give the rows the statements they
replaced gave.

Each search of ``memories``, ``media_content`` and ``memory_chunks`` once wrote its own statement: the owner pinned,
the rows ranked by vector distance or full-text rank under a ``LIMIT``, the chunk's or content's file joined in. They
now send one ranked read (``read_nearest`` with ``Nearest`` or ``TextRank``, the file joined after the limit by its
key through ``joins=``), or a keyed read, page or count. Each is compared, through
:func:`~threetears.core.testing.same_results.assert_same_results`, with the statement it replaced: empty, one row,
many, NULLs, ties in the rank, a tenant boundary, a page boundary and one past it. A ranked arm of a method that merges
two arms is compared as the rows that arm's read answered (the method run with ``read_nearest`` recorded); a ranked
read's ``LIMIT`` cut in a tie is compared as the harness compares a limited read (which tied rows were kept is the
engine's choice).

The seeds exercise each guard: other users and customers, a second agent, NULL embeddings, retracted, superseded
(by a live gist and by a gone one) and dormant memories, other users' content on the caller's file, two files on one
memory (a chunk under it answered once for each) and a memory with none.

Runs on PostgreSQL and YugabyteDB (the session's ``db_container``), a schema of its own.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

import asyncpg
import pytest

from threetears.agent.memory import collections as memory_collections
from threetears.agent.memory.collections import MediaContentCollection, MemoriesCollection, MemoryChunkCollection
from threetears.core.keyed import Nearest, TextRank, read_nearest
from threetears.core.testing.same_results import Seeded, assert_same_results

from .memory_same_results import (
    Serial,
    text_vectors,
    AGENTS,
    CUSTOMER,
    DIM,
    MEMORIES,
    PAGE,
    T0,
    USER,
    chunk_row,
    collection,
    conn,
    content_row,
    ident,
    memory_row,
    parents,
)

__all__ = ["conn"]

pytestmark = [pytest.mark.integration, pytest.mark.asyncio(loop_scope="module")]

#: the query every vector read is ranked against, and the text every keyword read matches
QUERY = [1.0] * (DIM - 1) + [0.0]
TEXT = "apple"
FLOOR = 0.5


def _vector(seed: Seeded) -> str | None:
    return None if seed.null else "[" + ",".join(["1"] * (DIM - 1) + [str(seed.tie)]) + "]"


def _text(kind: str, seed: Seeded) -> str:
    # the keyword's count, and so its rank, follows the tie: rows tied in the vector order are tied in the rank too
    return f"{kind} {'apple ' * (seed.tie % 3 + 1)}number {seed.index}"


def _memory(seed: Seeded) -> Mapping[str, Any]:
    """a memory (:func:`memory_row`), also: retracted every sixth, tagged ``t1`` every third, replaced by the memory
    before it every fourth (a live gist) or by one that is gone every eighth, dormant every fifth."""
    index = seed.index
    superseded = ident("memory", 9999) if index % 8 == 7 else ident("memory", index - 1) if index % 4 == 3 else None
    return {
        **memory_row(seed),
        "content": _text("memory", seed),
        "summary": None if seed.null else f"summary {index}",
        "salience": 0.2 if index % 5 == 1 else 0.9,
        "tags": '["retracted"]' if index % 6 == 5 else '["t1"]' if index % 3 == 0 else "[]",
        "superseded_by": superseded,
    }


def _content(seed: Seeded) -> Mapping[str, Any]:
    return {**content_row(seed), "content": _text("content", seed), "embedding": _vector(seed)}


def _chunk(seed: Seeded) -> Mapping[str, Any]:
    return {
        **chunk_row(seed),
        "content": _text("chunk", seed),
        "summary": None if seed.null else f"summary {seed.index}",
        "embedding": _vector(seed),
        "date_created": T0 + timedelta(minutes=seed.index),
    }


def _on(kind: type, c: Any) -> Any:
    """the collection of ``kind`` reading ``c`` (one connection), each statement in turn."""
    return collection(kind, Serial(c))


class _Ranked:
    """``read_nearest`` as the collections module calls it, each call's arguments and rows kept."""

    def __init__(self) -> None:
        self.calls: list[tuple[dict[str, Any], list[Any]]] = []

    async def __call__(self, *args: Any, **kwargs: Any) -> list[Any]:
        rows = await read_nearest(*args, **kwargs)
        self.calls.append((kwargs, rows))
        return rows

    def arm(self, kind: type, *, pinned: str | None = None) -> list[Any]:
        """the rows of the one call ranked by ``kind`` (pinning ``pinned`` too, when named)."""
        (rows,) = [
            rows
            for kwargs, rows in self.calls
            if isinstance(kwargs["nearest"], kind) and (pinned is None or pinned in kwargs["by"])
        ]
        return rows


@pytest.fixture
def ranked(monkeypatch: pytest.MonkeyPatch) -> Iterator[_Ranked]:
    spy = _Ranked()
    monkeypatch.setattr(memory_collections, "read_nearest", spy)
    yield spy


def _arm(
    spy: _Ranked, run: Callable[[Any], Awaitable[Any]], kind: type, pinned: str | None = None
) -> Callable[[Any], Awaitable[list[Any]]]:
    async def helper(c: Any) -> list[Any]:
        spy.calls.clear()
        await run(c)
        return spy.arm(kind, pinned=pinned)

    return helper


async def _same(
    conn: asyncpg.Connection,
    table: str,
    row: Callable[[Seeded], Mapping[str, Any]],
    old_sql: str,
    params: Sequence[Any],
    helper: Callable[[Any], Awaitable[Sequence[Mapping[str, Any]]]],
    columns: Sequence[str],
    order_by: Sequence[str],
    *,
    limited: bool = True,
) -> None:
    await text_vectors(conn)
    async with parents(conn) if table != "memories" else _nothing():
        await assert_same_results(
            conn,
            table=table,
            row=row,
            old_sql=old_sql,
            params=params,
            helper=helper,
            columns=columns,
            order_by=order_by,
            page_size=PAGE,
            limited=limited,
        )


class _nothing:  # noqa: N801 - an empty context, as ``parents`` is one
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, *_: Any) -> None:
        return None


# ---------------------------------------------------------------------------------------------------------- memories

_MEMORY_HYBRID = ("memory_id", "content", "summary", "type_memory", "date_created", "embedding")

#: MemoriesCollection.hybrid_search's vector arm as written by hand (date_after and the floor bound)
_MEMORY_VECTOR_SQL = """
    SELECT memory_id, content, summary, type_memory, date_created,
           embedding::text AS embedding,
           1 - (embedding OPERATOR(public.<=>) $1::text::public.vector) AS similarity
    FROM memories
    WHERE agent_id = $2 AND customer_id = $3 AND user_id = $4
      AND embedding IS NOT NULL AND date_created >= $6
      AND (superseded_by IS NULL OR NOT EXISTS (
            SELECT 1 FROM memories g
            WHERE g.agent_id = memories.agent_id AND g.memory_id = memories.superseded_by))
      AND NOT (COALESCE(tags, '[]'::jsonb) ? 'retracted')
      AND salience >= $7
    ORDER BY embedding OPERATOR(public.<=>) $1::text::public.vector
    LIMIT $5
"""

#: its keyword arm as written by hand
_MEMORY_KEYWORD_SQL = """
    SELECT memory_id, content, summary, type_memory, date_created,
           embedding::text AS embedding,
           ts_rank_cd(search_vector, websearch_to_tsquery('english', $1)) AS fts_rank
    FROM memories
    WHERE agent_id = $2 AND customer_id = $3 AND user_id = $4
      AND embedding IS NOT NULL
      AND (search_vector @@ websearch_to_tsquery('english', $1)) IS TRUE AND date_created >= $6
      AND (superseded_by IS NULL OR NOT EXISTS (
            SELECT 1 FROM memories g
            WHERE g.agent_id = memories.agent_id AND g.memory_id = memories.superseded_by))
      AND NOT (COALESCE(tags, '[]'::jsonb) ? 'retracted')
      AND salience >= $7
    ORDER BY fts_rank DESC
    LIMIT $5
"""

#: its history arm as written by hand: the owner triple and the date bound side by side
_MEMORY_HISTORY_SQL = """
    SELECT memory_id, content, summary, type_memory, date_created,
           embedding::text AS embedding,
           1 - (embedding OPERATOR(public.<=>) $1::text::public.vector) AS similarity
    FROM memories
    WHERE agent_id = $2 AND customer_id = $3 AND user_id = $4
      AND embedding IS NOT NULL AND date_created >= $6
      AND (superseded_by IS NULL OR NOT EXISTS (
            SELECT 1 FROM memories g
            WHERE g.agent_id = memories.agent_id AND g.memory_id = memories.superseded_by))
      AND NOT (COALESCE(tags, '[]'::jsonb) ? 'retracted')
      AND salience >= $7 AND date_created < $8
    ORDER BY embedding OPERATOR(public.<=>) $1::text::public.vector
    LIMIT $5
"""

#: memories older than this are the history pool's (the window, 30 days, is the hybrid search's below)
_EARLIEST = T0 - timedelta(days=400)


def _memory_hybrid(c: Any) -> Any:
    return _on(MemoriesCollection, c).hybrid_search(
        user_id=USER,
        agent_id=AGENTS[0],
        customer_id=CUSTOMER,
        embedding=QUERY,
        user_text=TEXT,
        top_k=PAGE,
        candidate_limit=PAGE,
        similarity_threshold=0.0,
        recency_half_life_hours=24.0,
        signal_weights={"semantic": 1.0, "keyword": 1.0, "recency": 1.0},
        salience_ambient_floor=FLOOR,
        date_after=_EARLIEST,
        recent_cap=2,
        recent_window_hours=24.0 * 30,
    )


@pytest.mark.parametrize(
    ("old_sql", "kind", "pinned", "order_by"),
    [
        (_MEMORY_VECTOR_SQL, Nearest, "customer_id", "similarity"),
        (_MEMORY_KEYWORD_SQL, TextRank, None, "fts_rank"),
    ],
    ids=["vector", "keyword"],
)
async def test_memory_hybrid_search_arms_give_the_hand_written_rows(
    conn: asyncpg.Connection, ranked: _Ranked, old_sql: str, kind: type, pinned: str | None, order_by: str
) -> None:
    """Flips if hybrid_search's vector or keyword read returns a memory its statement did not (another owner's, an
    unembedded, retracted, dormant or live-superseded one, one before the window), misses one (a memory whose gist is
    gone), or ranks them differently."""
    first = QUERY if kind is Nearest else TEXT
    await _same(
        conn,
        "memories",
        _memory,
        old_sql,
        [json.dumps(first) if kind is Nearest else first, AGENTS[0], CUSTOMER, USER, PAGE, _EARLIEST, FLOOR],
        _arm(ranked, _memory_hybrid, kind, pinned),
        [*_MEMORY_HYBRID, order_by],
        [order_by],
    )


async def test_memory_hybrid_search_history_gives_the_hand_written_rows(
    conn: asyncpg.Connection, ranked: _Ranked
) -> None:
    """Flips if hybrid_search's history pool (the user's memories older than the window, read by agent and user, the
    customer and the bound checked beside) returns a memory the owner-and-date statement did not, misses one, or ranks
    them differently."""
    # the window's bound is now() at each call: every seeded memory is months before it or decades after, so any
    # instant between binds the old statement to the same rows
    bound = datetime.now(UTC) - timedelta(days=30)

    async def same_instant(c: Any) -> list[Any]:
        ranked.calls.clear()
        await _memory_hybrid(c)
        (rows,) = [
            rows
            for kwargs, rows in ranked.calls
            if isinstance(kwargs["nearest"], Nearest) and "customer_id" not in kwargs["by"]
        ]
        return rows

    await _same(
        conn,
        "memories",
        _memory,
        _MEMORY_HISTORY_SQL,
        [json.dumps(QUERY), AGENTS[0], CUSTOMER, USER, PAGE, _EARLIEST, FLOOR, bound],
        same_instant,
        [*_MEMORY_HYBRID, "similarity"],
        ["similarity"],
    )


_MARKS_SQL = (
    "alias, evergreen, tags, superseded_by, "
    "(SELECT g.date_created FROM memories g "
    "WHERE g.agent_id = memories.agent_id AND g.memory_id = memories.superseded_by) AS superseded_at, "
    "(SELECT c.consolidated_memory_id FROM memory_consolidations c "
    "WHERE c.agent_id = memories.agent_id AND c.source_memory_id = memories.memory_id "
    "AND memories.superseded_by IS NULL ORDER BY c.date_created DESC LIMIT 1) AS corrected_by"
)
_MARKED = ("memory_id", "type_memory", "content", "date_created", "alias", "evergreen", "tags", "superseded_by")
_MARKED_ALL = (*_MARKED, "superseded_at", "corrected_by")

#: MemoriesCollection.search_by_semantic as written by hand (a type, the window's start, unretracted)
_MEMORY_SEMANTIC_SQL = f"""
    SELECT memory_id, type_memory, content, date_created, {_MARKS_SQL},
           1 - (embedding OPERATOR(public.<=>) $1::text::public.vector) AS similarity
    FROM memories
    WHERE agent_id = $2 AND user_id = $3 AND type_memory = $4 AND date_created >= $5
      AND embedding IS NOT NULL AND NOT (COALESCE(tags, '[]'::jsonb) ? 'retracted')
    ORDER BY embedding OPERATOR(public.<=>) $1::text::public.vector
    LIMIT $6
"""

#: MemoriesCollection.search_by_fts as written by hand
_MEMORY_FTS_SQL = f"""
    SELECT memory_id, type_memory, content, date_created, {_MARKS_SQL},
           ts_rank_cd(search_vector, websearch_to_tsquery('english', $1)) AS fts_rank
    FROM memories
    WHERE agent_id = $2 AND user_id = $3 AND (search_vector @@ websearch_to_tsquery('english', $1)) IS TRUE
      AND type_memory = $4 AND date_created >= $5 AND NOT (COALESCE(tags, '[]'::jsonb) ? 'retracted')
    ORDER BY fts_rank DESC
    LIMIT $6
"""


async def test_memory_search_by_semantic_gives_the_hand_written_rows(conn: asyncpg.Connection, ranked: _Ranked) -> None:
    """Flips if search_by_semantic's read returns a memory its statement did not (another user's or agent's, an
    unembedded or retracted one, one outside the window or the scope), misses one, ranks them differently, or marks
    them differently."""

    async def run(c: Any) -> Any:
        return await _on(MemoriesCollection, c).search_by_semantic(
            user_id=USER,
            agent_id=AGENTS[0],
            embedding=QUERY,
            max_results=PAGE,
            similarity_threshold=-1.0,
            type_filter="fact",
            date_after=_EARLIEST,
        )

    await _same(
        conn,
        "memories",
        _memory,
        _MEMORY_SEMANTIC_SQL,
        [json.dumps(QUERY), AGENTS[0], USER, "fact", _EARLIEST, PAGE],
        _arm(ranked, run, Nearest),
        [*_MARKED_ALL, "similarity"],
        ["similarity"],
    )


async def test_memory_search_by_fts_gives_the_hand_written_rows(conn: asyncpg.Connection, ranked: _Ranked) -> None:
    """Flips if search_by_fts's read returns a memory its statement did not (another user's, a retracted one, one not
    matching), misses one, ranks them differently, or marks them differently."""

    async def run(c: Any) -> Any:
        return await _on(MemoriesCollection, c).search_by_fts(
            user_id=USER, agent_id=AGENTS[0], fts_text=TEXT, max_results=PAGE, type_filter="fact", date_after=_EARLIEST
        )

    await _same(
        conn,
        "memories",
        _memory,
        _MEMORY_FTS_SQL,
        [TEXT, AGENTS[0], USER, "fact", _EARLIEST, PAGE],
        _arm(ranked, run, TextRank),
        [*_MARKED_ALL, "fts_rank"],
        ["fts_rank"],
    )


_STATES = {
    "current": (
        " AND NOT (COALESCE(tags, '[]'::jsonb) ? 'retracted') AND NOT (superseded_by IS NOT NULL AND EXISTS "
        "(SELECT 1 FROM memories g WHERE g.agent_id = memories.agent_id AND g.memory_id = memories.superseded_by))"
    ),
    "retired": " AND NOT (NOT (COALESCE(tags, '[]'::jsonb) ? 'retracted'))",
    "replaced": (
        " AND (superseded_by IS NOT NULL AND EXISTS (SELECT 1 FROM memories g "
        "WHERE g.agent_id = memories.agent_id AND g.memory_id = memories.superseded_by))"
    ),
    "all": "",
}


@pytest.mark.parametrize("state", sorted(_STATES))
async def test_memory_list_for_user_gives_the_hand_written_page_and_count(conn: asyncpg.Connection, state: str) -> None:
    """Flips if list_for_user's page (the first rows by the key's order, a page in) or its count, in any state,
    differs from the statements': another user's memory listed or counted, a state's memory missed, or the order or
    the offset moved."""
    narrowing = _STATES[state]

    async def page(c: Any) -> Any:
        rows, _total = await _on(MemoriesCollection, c).list_for_user(
            user_id=USER, agent_id=AGENTS[0], state=state, limit=3, offset=2
        )
        return rows

    async def count(c: Any) -> list[dict[str, Any]]:
        _rows, total = await _on(MemoriesCollection, c).list_for_user(
            user_id=USER, agent_id=AGENTS[0], state=state, limit=3, offset=2
        )
        return [{"count": total}]

    params = [AGENTS[0], USER]
    await _same(
        conn,
        "memories",
        _memory,
        f"SELECT memory_id, type_memory, content, date_created, {_MARKS_SQL} FROM memories "
        f"WHERE agent_id = $1 AND user_id = $2{narrowing} ORDER BY date_created DESC, memory_id DESC LIMIT $3 OFFSET $4",
        [*params, 3, 2],
        page,
        _MARKED_ALL,
        ["date_created", "memory_id"],
        limited=False,
    )
    await _same(
        conn,
        "memories",
        _memory,
        f"SELECT count(*) FROM memories WHERE agent_id = $1 AND user_id = $2{narrowing}",
        params,
        count,
        ["count"],
        [],
        limited=False,
    )


async def test_memory_ids_by_tags_gives_the_hand_written_ids(conn: asyncpg.Connection) -> None:
    """Flips if find_ids_by_tags returns an id its statement did not (another owner's, one without the tag), misses
    one, or orders them differently."""

    async def helper(c: Any) -> list[dict[str, Any]]:
        found = await _on(MemoriesCollection, c).find_ids_by_tags(
            agent_id=AGENTS[0], customer_id=CUSTOMER, user_id=USER, tags_any=("t1",)
        )
        return [{"memory_id": memory_id} for memory_id in found]

    await _same(
        conn,
        "memories",
        _memory,
        "SELECT memory_id, date_created FROM memories WHERE agent_id = $1 AND customer_id = $2 AND user_id = $3"
        " AND (tags ?| $4::text[]) IS TRUE ORDER BY date_created",
        [AGENTS[0], CUSTOMER, USER, ["t1"]],
        helper,
        ["memory_id"],
        ["memory_id"],
        limited=False,
    )


# ----------------------------------------------------------------------------------------------------- media_content

_CONTENT_HYBRID = (
    "content_id",
    "content",
    "summary",
    "content_type",
    "media_id",
    "date_created",
    "embedding",
    "media_category",
    "metadata_json",
)

_CONTENT_HYBRID_SQL = """
    SELECT mc.content_id, mc.content, mc.summary, mc.content_type,
           mc.media_id, mc.date_created, mc.embedding::text AS embedding,
           med.media_category, med.metadata_json,
           {rank}
    FROM media_content mc
    JOIN media med
      ON mc.agent_id = med.agent_id
     AND mc.media_id = med.media_id
    WHERE mc.agent_id = $2 AND mc.customer_id = $3 AND mc.user_id = $4 AND mc.embedding IS NOT NULL{match}
    ORDER BY {order}
    LIMIT $5
"""
_VECTOR_RANK = "1 - (mc.embedding OPERATOR(public.<=>) $1::text::public.vector) AS similarity"
_VECTOR_ORDER = "mc.embedding OPERATOR(public.<=>) $1::text::public.vector"
_TEXT_RANK = "ts_rank_cd(mc.search_vector, websearch_to_tsquery('english', $1)) AS fts_rank"
_TEXT_MATCH = " AND (mc.search_vector @@ websearch_to_tsquery('english', $1)) IS TRUE"


def _content_hybrid(c: Any) -> Any:
    return _on(MediaContentCollection, c).hybrid_search(
        user_id=USER,
        agent_id=AGENTS[0],
        customer_id=CUSTOMER,
        embedding=QUERY,
        user_text=TEXT,
        top_k=PAGE,
        candidate_limit=PAGE,
        similarity_threshold=0.0,
        recency_half_life_hours=24.0,
        signal_weights={"semantic": 1.0, "keyword": 1.0, "recency": 1.0},
    )


@pytest.mark.parametrize("kind", [Nearest, TextRank], ids=["vector", "keyword"])
async def test_content_hybrid_search_arms_give_the_hand_written_rows(
    conn: asyncpg.Connection, ranked: _Ranked, kind: type
) -> None:
    """Flips if media-content hybrid_search's vector or keyword read, its file joined after the limit, returns a row
    the join did not (another user's content on the caller's file, an unembedded one), misses one, ranks them
    differently, or carries another file's category or metadata."""
    vector = kind is Nearest
    sql = _CONTENT_HYBRID_SQL.format(
        rank=_VECTOR_RANK if vector else _TEXT_RANK,
        match="" if vector else _TEXT_MATCH,
        order=_VECTOR_ORDER if vector else "fts_rank DESC",
    )
    rank = "similarity" if vector else "fts_rank"
    await _same(
        conn,
        "media_content",
        _content,
        sql,
        [json.dumps(QUERY) if vector else TEXT, AGENTS[0], CUSTOMER, USER, PAGE],
        _arm(ranked, _content_hybrid, kind),
        [*_CONTENT_HYBRID, rank],
        [rank],
    )


_CONTENT_SEARCH_SQL = """
    SELECT mc.content_id, mc.content, mc.content_type,
           mc.media_id, med.media_category, med.metadata_json,
           med.date_created,
           {rank}
    FROM media_content mc
    JOIN media med
      ON mc.agent_id = med.agent_id
     AND mc.media_id = med.media_id
    WHERE mc.agent_id = $2
      AND mc.user_id = $3
      AND mc.embedding IS NOT NULL{match}
      AND ($5::timestamptz IS NULL OR med.date_created >= $5)
      AND ($6::timestamptz IS NULL OR med.date_created <= $6)
    ORDER BY {order}
    LIMIT $4
"""


#: the files' window: none, one holding the caller's file, one ending before it
_FILE_WINDOWS = {
    "any date": (None, None),
    "a window holding the file": (T0 - timedelta(days=1), T0 + timedelta(days=1)),
    "a window before the file": (None, T0 - timedelta(hours=1)),
}


@pytest.mark.parametrize("kind", [Nearest, TextRank], ids=["semantic", "fts"])
@pytest.mark.parametrize("window", sorted(_FILE_WINDOWS))
async def test_content_searches_give_the_hand_written_rows(
    conn: asyncpg.Connection, ranked: _Ranked, kind: type, window: str
) -> None:
    """Flips if media-content search_by_semantic or search_by_fts returns a row its join did not (another user's
    content, an unembedded one, one whose file is outside the window: the window narrows before the limit), misses
    one, ranks them differently, or carries another file's columns."""
    vector = kind is Nearest
    after, before = _FILE_WINDOWS[window]

    async def run(c: Any) -> Any:
        contents = _on(MediaContentCollection, c)
        if vector:
            return await contents.search_by_semantic(
                user_id=USER,
                agent_id=AGENTS[0],
                embedding=QUERY,
                max_results=PAGE,
                similarity_threshold=-1.0,
                date_after=after,
                date_before=before,
            )
        return await contents.search_by_fts(
            user_id=USER, agent_id=AGENTS[0], fts_text=TEXT, max_results=PAGE, date_after=after, date_before=before
        )

    sql = _CONTENT_SEARCH_SQL.format(
        rank=_VECTOR_RANK if vector else _TEXT_RANK,
        match="" if vector else _TEXT_MATCH,
        order=_VECTOR_ORDER if vector else "fts_rank DESC",
    )
    rank = "similarity" if vector else "fts_rank"
    columns = ["content_id", "content", "content_type", "media_id", "media_category", "metadata_json", "date_created"]
    await _same(
        conn,
        "media_content",
        _content,
        sql,
        [json.dumps(QUERY) if vector else TEXT, AGENTS[0], USER, PAGE, after, before],
        _arm(ranked, run, kind),
        [*columns, rank],
        [rank],
    )


# ----------------------------------------------------------------------------------------------------- memory_chunks

_CHUNK_HYBRID = (
    "chunk_id",
    "content",
    "summary",
    "heading_context",
    "page_number",
    "memory_id",
    "message_id_start",
    "message_id_end",
    "embedding",
    "metadata_json",
    "media_id",
)

_CHUNK_HYBRID_SQL = """
    SELECT mc.chunk_id, mc.content, mc.summary, mc.heading_context,
           mc.page_number, mc.memory_id,
           mc.message_id_start, mc.message_id_end,
           mc.embedding::text AS embedding, med.metadata_json, med.media_id,
           {rank}
    FROM memory_chunks mc
    LEFT JOIN media med
      ON mc.agent_id = med.agent_id
     AND mc.memory_id = med.memory_id
    WHERE mc.agent_id = $2 AND mc.customer_id = $3 AND mc.user_id = $4
      {within}{embedded}{match}{cursor}
    ORDER BY {order}
    LIMIT $5
"""


def _chunk_hybrid(within: bool, after: Any) -> Callable[[Any], Awaitable[Any]]:
    def run(c: Any) -> Any:
        chunks = _on(MemoryChunkCollection, c)
        if within:
            return chunks.hybrid_search_within_memory(
                memory_id=MEMORIES[(AGENTS[0], "mine")][0],
                user_id=USER,
                agent_id=AGENTS[0],
                customer_id=CUSTOMER,
                embedding=QUERY,
                user_text=TEXT,
                candidate_k=PAGE,
                similarity_threshold=0.0,
                chunk_signal_weights={"semantic": 1.0, "keyword": 1.0},
            )
        return chunks.hybrid_search(
            user_id=USER,
            agent_id=AGENTS[0],
            customer_id=CUSTOMER,
            embedding=QUERY,
            user_text=TEXT,
            candidate_k=PAGE,
            similarity_threshold=0.0,
            chunk_signal_weights={"semantic": 1.0, "keyword": 1.0},
            chunk_id_after=after,
        )

    return run


@pytest.mark.parametrize("kind", [Nearest, TextRank], ids=["vector", "keyword"])
@pytest.mark.parametrize("shape", ["owner", "past a cursor", "within a memory"])
async def test_chunk_hybrid_search_arms_give_the_hand_written_rows(
    conn: asyncpg.Connection, ranked: _Ranked, kind: type, shape: str
) -> None:
    """Flips if chunk hybrid_search's (or hybrid_search_within_memory's) vector or keyword read, the files joined
    after the limit, returns a chunk the LEFT JOIN did not (another user's, an unembedded one, one before the cursor or
    under another memory), misses one (a chunk under a memory with no file), answers a chunk under two files other
    than once for each, or ranks them differently."""
    vector = kind is Nearest
    within = shape == "within a memory"
    after = ident("chunk", 4) if shape == "past a cursor" else None
    params: list[Any] = [json.dumps(QUERY) if vector else TEXT, AGENTS[0], CUSTOMER, USER, PAGE]
    if within:
        params.append(MEMORIES[(AGENTS[0], "mine")][0])
    if after is not None:
        params.append(after)
    sql = _CHUNK_HYBRID_SQL.format(
        rank=_VECTOR_RANK if vector else _TEXT_RANK,
        within="AND mc.memory_id = $6" if within else "",
        # the keyword arm within a memory never asked for an embedding
        embedded="" if within and not vector else " AND mc.embedding IS NOT NULL",
        match="" if vector else _TEXT_MATCH,
        cursor=" AND mc.chunk_id > $6" if after is not None else "",
        order=_VECTOR_ORDER if vector else "fts_rank DESC",
    )
    rank = "similarity" if vector else "fts_rank"
    await _same(
        conn,
        "memory_chunks",
        _chunk,
        sql,
        params,
        _arm(ranked, _chunk_hybrid(within, after), kind),
        [*_CHUNK_HYBRID, rank],
        [rank],
    )


_CHUNK_SEMANTIC_SQL = """
    SELECT mc.chunk_id, mc.content, mc.heading_context, mc.page_number,
           mc.memory_id, mc.message_id_start, mc.message_id_end,
           mc.date_created, med.metadata_json,
           1 - (mc.embedding OPERATOR(public.<=>) $1::text::public.vector) AS similarity
    FROM memory_chunks mc
    LEFT JOIN LATERAL (
        SELECT media.metadata_json FROM media
        WHERE media.agent_id = mc.agent_id AND media.memory_id = mc.memory_id
    ) med ON TRUE
    WHERE mc.agent_id = $2
      AND mc.user_id = $3
      AND mc.embedding IS NOT NULL
      AND ($5::timestamptz IS NULL OR mc.date_created >= $5)
      AND ($6::timestamptz IS NULL OR mc.date_created <= $6)
    ORDER BY mc.embedding OPERATOR(public.<=>) $1::text::public.vector
    LIMIT $4
"""


@pytest.mark.parametrize("window", [False, True], ids=["any date", "a window"])
async def test_chunk_search_by_semantic_gives_the_hand_written_rows(
    conn: asyncpg.Connection, ranked: _Ranked, window: bool
) -> None:
    """Flips if chunk search_by_semantic returns a chunk its lateral join did not (another user's, an unembedded one,
    one outside the window), misses one, answers a chunk under two files other than once for each, or ranks them
    differently."""
    after, before = (T0 + timedelta(minutes=2), T0 + timedelta(minutes=9)) if window else (None, None)

    async def run(c: Any) -> Any:
        return await _on(MemoryChunkCollection, c).search_by_semantic(
            user_id=USER,
            agent_id=AGENTS[0],
            embedding=QUERY,
            max_results=PAGE,
            similarity_threshold=-1.0,
            date_after=after,
            date_before=before,
        )

    columns = [
        "chunk_id",
        "content",
        "heading_context",
        "page_number",
        "memory_id",
        "message_id_start",
        "message_id_end",
        "date_created",
        "metadata_json",
        "similarity",
    ]
    await _same(
        conn,
        "memory_chunks",
        _chunk,
        _CHUNK_SEMANTIC_SQL,
        [json.dumps(QUERY), AGENTS[0], USER, PAGE, after, before],
        _arm(ranked, run, Nearest),
        columns,
        ["similarity"],
    )


async def test_chunk_search_by_ids_gives_the_hand_written_rows(conn: asyncpg.Connection) -> None:
    """Flips if chunk search_by_ids returns a chunk its LEFT JOIN did not (another user's, another agent's), misses
    one (a chunk under a memory with no file), or answers a chunk under two files other than once for each."""
    ids = [ident("chunk", index) for index in (0, 1, 2, 4, 9, 13, 30)]

    async def helper(c: Any) -> Any:
        return await _on(MemoryChunkCollection, c).search_by_ids(ids, USER, agent_id=AGENTS[0])

    await _same(
        conn,
        "memory_chunks",
        _chunk,
        """
        SELECT mc.chunk_id, mc.content, mc.heading_context, mc.page_number,
               mc.memory_id, mc.message_id_start, mc.message_id_end,
               mc.chunk_index, mc.date_created,
               med.metadata_json, med.media_id
        FROM memory_chunks mc
        LEFT JOIN media med
          ON mc.agent_id = med.agent_id
         AND mc.memory_id = med.memory_id
        WHERE mc.agent_id = $1
          AND mc.chunk_id = ANY($2::uuid[])
          AND mc.user_id = $3
        """,
        [AGENTS[0], ids, USER],
        helper,
        [
            "chunk_id",
            "content",
            "heading_context",
            "page_number",
            "memory_id",
            "message_id_start",
            "message_id_end",
            "chunk_index",
            "date_created",
            "metadata_json",
            "media_id",
        ],
        [],
        limited=False,
    )


@pytest.mark.parametrize("cursor", ["first", "after", "before"])
async def test_chunk_find_by_memory_id_gives_the_hand_written_page(conn: asyncpg.Connection, cursor: str) -> None:
    """Flips if find_by_memory_id's page returns a chunk its statement did not (another owner's, one under another
    memory, one on the wrong side of the cursor), misses one, or orders them differently."""
    memory = MEMORIES[(AGENTS[0], "mine")][0]
    mark = ident("chunk", 9)
    past = {"first": "", "after": " AND mc.chunk_id > $5", "before": " AND mc.chunk_id < $5"}[cursor]
    direction = "DESC" if cursor == "before" else "ASC"
    params: list[Any] = [AGENTS[0], memory, USER, CUSTOMER, *([mark] if cursor != "first" else []), PAGE]

    async def helper(c: Any) -> Any:
        return await _on(MemoryChunkCollection, c).find_by_memory_id(
            memory,
            user_id=USER,
            agent_id=AGENTS[0],
            customer_id=CUSTOMER,
            limit=PAGE,
            chunk_id_after=mark if cursor == "after" else None,
            chunk_id_before=mark if cursor == "before" else None,
        )

    old = f"""
        SELECT * FROM (
            SELECT mc.chunk_id, mc.memory_id, mc.chunk_index, mc.content, mc.summary,
                   mc.heading_context, mc.page_number,
                   mc.message_id_start, mc.message_id_end,
                   mc.date_created
            FROM memory_chunks mc
            WHERE mc.agent_id = $1
              AND mc.memory_id = $2
              AND mc.user_id = $3
              AND mc.customer_id = $4
              {past}
            ORDER BY mc.chunk_id {direction}
            LIMIT ${len(params)}
        ) page ORDER BY chunk_id
    """
    await _same(
        conn,
        "memory_chunks",
        _chunk,
        old,
        params,
        helper,
        [
            "chunk_id",
            "memory_id",
            "chunk_index",
            "content",
            "summary",
            "heading_context",
            "page_number",
            "message_id_start",
            "message_id_end",
            "date_created",
        ],
        ["chunk_id"],
        limited=False,
    )
