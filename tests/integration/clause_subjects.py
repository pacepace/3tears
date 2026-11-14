"""the proof subjects of the repo's clause builders (``clause_builders`` in ``tests/enforcement/no_whole_reads_support``).

The generated use-the-helper proof (``test_use_the_helper_rewrites.py``) binds a rewritten statement's clause to
``None`` and to each subject here, and compares the old statement, its builder's answer written in, with the helper
call spreading ``clause_guards`` of the same builder. Each subject must restrict the seeded rows (a seeded ``tags`` is
``{"n": <row>}``), and binds a value, so a clause numbered wrongly against the statement's own parameters shows.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from threetears.agent.memory.read_filter import ReadFilter

__all__ = ["read_filters"]


def _even_rows(tags: str, bind: Callable[[Any], str]) -> str:
    """keep the rows whose seeded tag number is even (a NULL ``tags`` is kept out)."""
    return f"(({tags} ->> 'n')::int % 2) = {bind(0)}::int"


def _all_but_one(tags: str, bind: Callable[[Any], str]) -> str:
    """keep every row but the second seeded one (a NULL ``tags`` is kept)."""
    return f"({tags} ->> 'n') IS DISTINCT FROM {bind('1')}::text"


def read_filters() -> list[ReadFilter]:
    """memory's host read filters the proof binds ``filter_clause`` and its parent-row builders to.

    :return: the filters
    :rtype: list[ReadFilter]
    """
    return [_even_rows, _all_but_one]
