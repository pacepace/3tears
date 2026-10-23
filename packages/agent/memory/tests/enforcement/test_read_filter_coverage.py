"""every public async method of the three read-filtered collections has chosen: filtered, or not and why.

A host restricts memory reads through a :class:`ReadFilter`. A read method
that forgets to apply it hands that host's excluded rows to whoever calls it,
with every other test still green. So each public coroutine method, inherited
ones included, sits in exactly one of two lists here:

- ``FILTERED``: its SQL carries the filter, through the helper that decides it
  on the right row (the memory itself, a chunk's parent memory, or the memory a
  media row is paired with);
- ``UNFILTERED``: a write or bookkeeping method, with the reason it must reach
  rows the current task cannot read.

A new read method fails here until it is put in one list. The SQL each filtered
method builds is exercised on Postgres by ``tests/integration/test_read_filter.py``.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from threetears.agent.memory.collections import MediaContentCollection, MemoriesCollection, MemoryChunkCollection

_COLLECTIONS_SRC = (
    Path(__file__).resolve().parent.parent.parent / "src" / "threetears" / "agent" / "memory" / "collections.py"
)

#: the call a filtered method of each collection makes, so a chunk read cannot filter on its own tags
_APPLIES = {
    MemoriesCollection: ('filter_clause(self.read_filter, "memories.tags"', 'filter_clause(self.read_filter, "m.tags"'),
    MemoryChunkCollection: ("_chunk_parent_clause(self.read_filter",),
    MediaContentCollection: ("_media_parent_clause(self.read_filter",),
}

FILTERED: dict[type, frozenset[str]] = {
    MemoriesCollection: frozenset(
        {
            "fetch_content_for_recall",
            "find_active_for_consolidation",
            "find_by_alias",
            "find_by_scope",
            "find_by_user",
            "find_for_recall",
            "find_for_user_in_agents",
            "find_ids_by_tags",
            "find_same_event",
            "find_similar_for_dedup",
            "hybrid_search",
            "list_for_user",
            "search_by_fts",
            "search_by_ids",
            "search_by_semantic",
        }
    ),
    MemoryChunkCollection: frozenset(
        {
            "count_by_memory_id",
            "fetch_content_for_recall",
            "find_by_chunk_indexes",
            "find_by_conversation_id",
            "find_by_memory_id",
            "find_for_recall",
            "hybrid_search",
            "hybrid_search_within_memory",
            "search_by_ids",
            "search_by_semantic",
        }
    ),
    MediaContentCollection: frozenset(
        {"fetch_content_for_recall", "hybrid_search", "search_by_fts", "search_by_ids", "search_by_semantic"}
    ),
}

UNFILTERED: dict[type, dict[str, str]] = {
    MemoriesCollection: {
        "save_memory": "a write; the evaluator authorizes it, and a filter would only hide the row it writes",
        "delete_by_tags": "a host deleting a source's derived memories must reach every one, readable or not",
        "count_by_user": "an existence probe gating the first-write owner assignment; returns no row or content",
        "mark_retracted": "a write by id; its callers (the memory tools, extraction) find the memory through "
        "find_for_recall first",
        "clear_retracted": "a write by id; memory_restore finds the memory through find_for_recall first",
        "decay_salience": "a maintenance sweep over every row; returns only a count",
        "set_salience": "a write by id set; returns nothing",
        "bump_salience": "reinforces ids an already-filtered retrieval surfaced; returns nothing",
        "mark_superseded": "points sources at a gist; a write that returns nothing",
    },
    MemoryChunkCollection: {},
    MediaContentCollection: {},
}

#: the framework methods every collection inherits. none builds memory SQL a per-task predicate could join.
INHERITED_UNFILTERED: dict[str, str] = {
    "get": "a primary-key read through the L1/L2 cache tiers; a host gates by-id reads with find_for_recall",
    "ensure": "a primary-key get-or-create; the cache tiers serve it, as get",
    "reload_entity": "re-reads one cached entity by its key, for the writer that holds it",
    "fetch_from_store": "the L3 leg of get, by primary key",
    "read_rows_led_by": "the generic leading-key scan the framework's cascade and cache warmers use",
    "save_entity": "a write of an entity the caller already holds; it returns no other row",
    "save_to_store": "the L3 leg of save_entity; writes the caller's own entity",
    "save_ordered_to_store": "the ordered L3 write behind save_entity; writes the caller's own entities",
    "save_rows": "a bulk write of rows the caller supplies; it returns no stored row",
    "persist_to_store": "the write-buffer flush of entities callers already saved",
    "delete": "a delete by key must reach the row whether or not this task may read it",
    "delete_from_store": "the L3 leg of delete; must reach the row whatever the filter says",
    "delete_rows": "a bulk delete by key; must reach every named row whatever the filter says",
    "truncate": "empties the whole table, as an operator's clear; reads and returns no row",
    "load_files": "a bulk load of files the caller supplies into the table; it returns a count, no stored row",
    "delete_l2_entry": "drops one L2 cache entry by key; touches no stored row",
    "invalidate_cache": "evicts one key from the cache tiers; returns nothing",
    "invalidate_cache_many": "evicts many keys from the cache tiers; returns nothing",
    "l2_cas_mutate": "a compare-and-set on one L2 cache entry the writer holds; no SQL read",
    "announce_flushed": "publishes that a write reached L3; returns nothing",
    "aclose": "shuts the collection down; reads nothing",
}

_CLASSES = list(FILTERED)
_IDS = [cls.__name__ for cls in _CLASSES]


def _public_coroutines(cls: type) -> set[str]:
    """every public coroutine method, inherited ones too; a decorated one is seen through its wrapper."""
    found = {
        name
        for name, value in inspect.getmembers(cls)
        if not name.startswith("_") and callable(value) and inspect.iscoroutinefunction(inspect.unwrap(value))
    }
    return found | {name for name in _own_method_sources(cls) if not name.startswith("_")}


def _own_method_sources(cls: type) -> dict[str, str]:
    """each async method defined in the class body, by name, as source text."""
    text = _COLLECTIONS_SRC.read_text(encoding="utf-8")
    tree = ast.parse(text)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == cls.__name__:
            return {
                item.name: ast.get_source_segment(text, item) or ""
                for item in node.body
                if isinstance(item, ast.AsyncFunctionDef)
            }
    raise AssertionError(f"{cls.__name__} not found in collections.py")


@pytest.mark.parametrize("cls", _CLASSES, ids=_IDS)
def test_every_public_coroutine_has_chosen(cls: type) -> None:
    filtered, unfiltered = FILTERED[cls], set(UNFILTERED[cls])
    own = set(_own_method_sources(cls))
    inherited = set(INHERITED_UNFILTERED) - own
    assert not filtered & unfiltered, f"listed both ways: {sorted(filtered & unfiltered)}"
    unlisted = _public_coroutines(cls) - filtered - unfiltered - inherited
    assert not unlisted, (
        f"{cls.__name__}: {sorted(unlisted)} must be put in FILTERED (and apply the read filter) "
        "or in UNFILTERED with the reason it reaches rows the task cannot read"
    )


@pytest.mark.parametrize("cls", _CLASSES, ids=_IDS)
def test_no_listed_method_is_stale(cls: type) -> None:
    listed = FILTERED[cls] | set(UNFILTERED[cls])
    stale = listed - _public_coroutines(cls)
    assert not stale, f"{cls.__name__}: listed but not a public coroutine: {sorted(stale)}"


@pytest.mark.parametrize("cls", _CLASSES, ids=_IDS)
def test_an_inherited_method_overridden_here_must_choose_again(cls: type) -> None:
    overridden = set(INHERITED_UNFILTERED) & set(_own_method_sources(cls))
    unlisted = overridden - FILTERED[cls] - set(UNFILTERED[cls])
    assert not unlisted, f"{cls.__name__} overrides {sorted(unlisted)}: list it again"


@pytest.mark.parametrize("cls", _CLASSES, ids=_IDS)
def test_a_filtered_method_applies_the_filter_on_the_right_row(cls: type) -> None:
    sources = _own_method_sources(cls)
    for name in sorted(FILTERED[cls]):
        body = sources[name]
        applies = any(call in body for call in _APPLIES[cls]) or "filtered=True" in body
        assert applies, f"{cls.__name__}.{name} is listed FILTERED but never applies {_APPLIES[cls]}"


@pytest.mark.parametrize("cls", _CLASSES, ids=_IDS)
def test_an_unfiltered_method_does_not_claim_otherwise(cls: type) -> None:
    sources = _own_method_sources(cls)
    for name in sorted(UNFILTERED[cls]):
        assert "self.read_filter" not in sources[name], f"{cls.__name__}.{name} is listed UNFILTERED but reads it"


@pytest.mark.parametrize("cls", _CLASSES, ids=_IDS)
def test_every_unfiltered_entry_says_why(cls: type) -> None:
    reasons = {**UNFILTERED[cls], **INHERITED_UNFILTERED}
    # a reason is a sentence saying why this method may reach rows the task cannot read, not a category
    short = [name for name, reason in reasons.items() if len(reason.split()) < 6]
    assert not short, f"{cls.__name__}: say why, in a sentence, for {short}"
