"""collection stand-ins for entity unit tests.

An entity reads three things off its collection during construction and
attribute access: the declared ``primary_key_columns`` (which is where
:class:`~threetears.core.entities.base.BaseEntity` derives its
addressing ``_id`` from), the synchronous L1 cache surface
(``write_to_cache_sync`` / ``get_field_sync`` / ``set_field_sync`` /
``get_row_sync``), and the async persistence delegates (``save_entity``
/ ``reload_entity``).

Entity unit tests want that surface without a registry, an L1 backend,
a NATS client and a config object. Before this module every package
grew its own ``mock_collection`` fixture -- a bare :class:`MagicMock`
with hand-written ``side_effect`` closures over a dict. Those copies
shared one defect: none declared ``primary_key_columns``, so a
composite-pk entity built against them silently addressed rows by the
bare id. That went unnoticed while every composite entity carried its
own ``_id`` override, and surfaced the moment the override was deleted
in favour of the framework's derivation.

So the shape lives here once, per the rule in
:mod:`threetears.core.testing`'s own docstring: import the harness, do
not re-grow it per repo.

The stub is a real :class:`MagicMock` so ``assert_called_with`` on
``set_field_sync`` keeps working, but its ``primary_key_columns`` is a
genuine tuple rather than an auto-created child mock -- which is the
whole point.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

from threetears.core.cache import MISSING

__all__ = ["entity_collection_stub"]


def _normalize(entity_id: Any, primary_key_columns: tuple[str, ...]) -> tuple[Any, ...]:
    """coerce an entity id to the tuple form the cache is keyed by.

    mirrors :meth:`BaseCollection.normalize_pk`, arity validation
    INCLUDED. an earlier version of this stub skipped the validation on
    the grounds that raising would fail tests for reasons unrelated to
    what they assert. that was backwards: a scalar reaching a
    composite-pk collection is precisely the defect this whole area
    exists to prevent, and a stub that answers it with ``MISSING``
    instead of the real ``ValueError`` makes every test built on it
    blind to that defect. one such test shipped green against a
    collection whose declaration was wrong.

    :param entity_id: scalar pk value or tuple of pk values
    :ptype entity_id: Any
    :param primary_key_columns: declared pk column names
    :ptype primary_key_columns: tuple[str, ...]
    :return: tuple of pk values
    :rtype: tuple[Any, ...]
    :raises ValueError: if arity does not match the declared columns
    """
    values = entity_id if isinstance(entity_id, tuple) else (entity_id,)
    if len(values) != len(primary_key_columns):
        raise ValueError(
            f"primary key arity mismatch: got {len(values)} value(s) "
            f"for {len(primary_key_columns)} column(s) {primary_key_columns}"
        )
    return values


def entity_collection_stub(
    primary_key_columns: tuple[str, ...] = ("id",),
) -> tuple[MagicMock, dict[tuple[Any, ...], dict[str, Any]]]:
    """build a collection stand-in backed by an in-memory row dict.

    parity-with: threetears.core.collections.base.BaseCollection

    the returned cache is keyed by the normalized pk tuple, so a
    composite-pk stub keeps two tenants' rows apart exactly as L1 does.
    keying by ``str(pk)`` -- what the hand-rolled per-package fixtures
    did -- collapses ``("cust-a", "row-1")`` and ``("cust-b", "row-1")``
    only if the pk is stringified before the tuple is formed, so the
    tuple key is both simpler and stricter.

    :param primary_key_columns: declared pk column names in order; a
        1-tuple gives single-pk behaviour, longer gives composite
    :ptype primary_key_columns: tuple[str, ...]
    :return: the collection stub and the row dict backing it, so a test
        may assert on cache contents directly
    :rtype: tuple[MagicMock, dict[tuple[Any, ...], dict[str, Any]]]
    """
    cache: dict[tuple[Any, ...], dict[str, Any]] = {}

    def _key_from_row(data: dict[str, Any]) -> tuple[Any, ...]:
        return tuple(data.get(col) for col in primary_key_columns)

    def _write_to_cache(data: dict[str, Any]) -> bool:
        cache[_key_from_row(data)] = dict(data)
        return True

    def _get_field(entity_id: Any, field: str) -> Any:
        row = cache.get(_normalize(entity_id, primary_key_columns))
        if row is None:
            return MISSING
        return row.get(field, MISSING)

    def _set_field(entity_id: Any, field: str, value: Any) -> bool:
        row = cache.get(_normalize(entity_id, primary_key_columns))
        if row is None:
            return False
        row[field] = value
        return True

    def _get_row(entity_id: Any) -> dict[str, Any] | None:
        # a COPY, matching SQLiteBackend.select_by_id. returning the live
        # dict let a caller mutate the "cache" through a read result,
        # which no real backend permits.
        row = cache.get(_normalize(entity_id, primary_key_columns))
        return dict(row) if row is not None else None

    collection = MagicMock()
    # a genuine tuple, NOT an auto-created child mock: BaseEntity reads
    # this to decide whether the entity's addressing id is a scalar or a
    # composite tuple, and a mock answers that question wrongly.
    collection.primary_key_columns = primary_key_columns
    collection.write_to_cache_sync = MagicMock(side_effect=_write_to_cache)
    collection.get_field_sync = MagicMock(side_effect=_get_field)
    collection.set_field_sync = MagicMock(side_effect=_set_field)
    collection.get_row_sync = MagicMock(side_effect=_get_row)
    collection.save_entity = AsyncMock()
    collection.reload_entity = AsyncMock()
    return collection, cache
