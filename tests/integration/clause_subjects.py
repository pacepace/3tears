"""the proof subjects of the repo's clause builders (``clause_builders`` in ``tests/enforcement/no_whole_reads_support``).

The generated use-the-helper proof (``test_use_the_helper_rewrites.py``) binds a rewritten statement's clause to
``None`` and to each subject here, and compares the old statement, its builder's answer written in, with the helper
call spreading ``clause_guards`` of the same builder. Each subject must restrict the seeded rows (a seeded ``tags`` is
``{"n": <row>}``), and binds a value, so a clause numbered wrongly against the statement's own parameters shows. A
builder deciding on another table's rows (a chunk's or a media content's parent memory) has them seeded by its
``clause_parents`` seeder here, two rows of three given a parent, the third an orphan.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any
from uuid import NAMESPACE_OID, uuid5

from threetears.agent.memory.read_filter import ReadFilter
from threetears.core.testing.rewrites import seed_rows

__all__ = ["chunk_parents", "media_parents", "read_filters"]


def _even_rows(tags: str, bind: Callable[[Any], str]) -> str:
    """keep the rows whose seeded tag number is even (a NULL ``tags`` is kept out)."""
    return f"(({tags} ->> 'n')::int % 2) = {bind(0)}::int"


def _odd_rows(tags: str, bind: Callable[[Any], str]) -> str:
    """keep the rows whose seeded tag number is odd: the first row (a point read's) is dropped."""
    return f"(({tags} ->> 'n')::int % 2) = {bind(1)}::int"


def _all_but_one(tags: str, bind: Callable[[Any], str]) -> str:
    """keep every row but the second seeded one (a NULL ``tags`` is kept)."""
    return f"({tags} ->> 'n') IS DISTINCT FROM {bind('1')}::text"


def read_filters() -> list[ReadFilter]:
    """memory's host read filters the proof binds ``filter_clause`` and its parent-row builders to.

    :return: the filters
    :rtype: list[ReadFilter]
    """
    return [_even_rows, _odd_rows, _all_but_one]


def _parent_tags(index: int) -> str:
    """a seeded parent memory's tags: its row's number, as the seed writes a jsonb (its text)."""
    return json.dumps({"n": index})


def _kept(rows: list[Mapping[str, Any]]) -> list[tuple[int, Mapping[str, Any]]]:
    """the seeded rows given a parent: two of three (the third an orphan, which a filter's ``EXISTS`` never matches)."""
    return [(index, row) for index, row in enumerate(rows) if index % 3 != 2]


async def chunk_parents(conn: Any, rows: list[Mapping[str, Any]]) -> None:
    """seed each seeded ``memory_chunks`` row's parent memory (two rows of three; tags numbered by the chunk's row),
    which ``_chunk_parent_clause`` decides on.

    :param conn: the proof's connection, inside the case's transaction
    :ptype conn: Any
    :param rows: the seeded chunk rows
    :ptype rows: list[Mapping[str, Any]]
    :return: nothing
    :rtype: None
    """
    await seed_rows(
        conn,
        "memories",
        [
            {"agent_id": row["agent_id"], "memory_id": row["memory_id"], "tags": _parent_tags(index)}
            for index, row in _kept(rows)
        ],
    )


async def media_parents(conn: Any, rows: list[Mapping[str, Any]]) -> None:
    """seed each seeded ``media_content`` row's media row and its paired memory (two rows of three; tags numbered by
    the content's row), which ``_media_parent_clause`` decides on.

    :param conn: the proof's connection, inside the case's transaction
    :ptype conn: Any
    :param rows: the seeded content rows
    :ptype rows: list[Mapping[str, Any]]
    :return: nothing
    :rtype: None
    """
    kept = _kept(rows)
    paired = {index: uuid5(NAMESPACE_OID, f"media-parent-{index}") for index, _ in kept}
    await seed_rows(
        conn,
        "media",
        [{"agent_id": row["agent_id"], "media_id": row["media_id"], "memory_id": paired[index]} for index, row in kept],
    )
    await seed_rows(
        conn,
        "memories",
        [{"agent_id": row["agent_id"], "memory_id": paired[index], "tags": _parent_tags(index)} for index, row in kept],
    )
