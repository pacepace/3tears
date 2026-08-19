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

Where the stub answers differently from the collection it replaces, the
difference is untestable: every test built on the stub agrees with the
stub, and none of them notices the real behaviour. So each accessor
matches ``BaseCollection``'s observable contract, including the two
cases that are easy to get subtly wrong:

* ``has_l1=False`` models a collection whose ``_l1`` is ``None``. the
  four sync accessors the stub implements short-circuit, writes report
  failure, and the entity's ``_changes`` fallback / ``set_data``
  ``RuntimeError`` paths become reachable from a unit test. the
  short-circuit precedes pk normalization, as it does on the real
  collection, so a no-L1 stub skips the arity check exactly as the real
  one skips it.
* ``set_field_sync`` upserts a detached copy rather than mutating the
  cached row, so writing a PRIMARY KEY column leaves the old row behind
  and creates a second one -- what the real collection does, and what an
  in-place rename hid.

``packages/core/tests/test_entity_collection_stub.py`` runs the same
assertions against this stub and against a real ``BaseCollection`` so
the two cannot drift apart again unnoticed.
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
    *,
    has_l1: bool = True,
) -> tuple[MagicMock, dict[tuple[Any, ...], dict[str, Any]]]:
    """build a collection stand-in backed by an in-memory row dict.

    parity-with: threetears.core.collections.base.BaseCollection

    the returned cache is keyed by the normalized pk tuple, so a
    composite-pk stub keeps two tenants' rows apart exactly as L1 does.
    keying by ``str(pk)`` -- what the hand-rolled per-package fixtures
    did -- collapses ``("cust-a", "row-1")`` and ``("cust-b", "row-1")``
    only if the pk is stringified before the tuple is formed, so the
    tuple key is both simpler and stricter.

    ``has_l1=False`` models a collection constructed against a registry
    with no L1 backend, where :attr:`BaseCollection._l1` is ``None`` and
    every sync accessor short-circuits: reads answer ``MISSING`` /
    ``None`` and writes answer ``False``. that is not a cosmetic
    difference. :meth:`BaseEntity.__init__` reads the write's return
    value to decide whether the row lives in L1 or in the in-memory
    ``_changes`` buffer, and :meth:`BaseEntity.set_data` raises
    ``RuntimeError`` on a refused write. a stub hard-coded to report
    success cannot reach either branch, so both went unexercised by
    every test built on it.

    :param primary_key_columns: declared pk column names in order; a
        1-tuple gives single-pk behaviour, longer gives composite
    :ptype primary_key_columns: tuple[str, ...]
    :param has_l1: whether the stubbed collection has an L1 backend;
        ``False`` refuses every cache read and write, as a collection
        whose ``_l1`` is ``None`` does
    :ptype has_l1: bool
    :return: the collection stub and the row dict backing it, so a test
        may assert on cache contents directly. with ``has_l1=False``
        the dict stays empty, matching a collection that caches nothing
    :rtype: tuple[MagicMock, dict[tuple[Any, ...], dict[str, Any]]]
    """
    cache: dict[tuple[Any, ...], dict[str, Any]] = {}

    def _key_from_row(data: dict[str, Any]) -> tuple[Any, ...]:
        return tuple(data.get(col) for col in primary_key_columns)

    def _write_to_cache(data: dict[str, Any]) -> bool:
        if not has_l1:
            return False
        cache[_key_from_row(data)] = dict(data)
        return True

    def _get_field(entity_id: Any, field: str) -> Any:
        if not has_l1:
            return MISSING
        row = cache.get(_normalize(entity_id, primary_key_columns))
        if row is None:
            return MISSING
        return row.get(field, MISSING)

    def _set_field(entity_id: Any, field: str, value: Any) -> bool:
        # an upsert of a DETACHED copy, matching set_field_sync: the
        # real one reads the row via select_by_id (which hands back a
        # fresh dict), mutates that, and upserts it under whatever pk
        # the mutated row now carries. mutating the cached row in place
        # -- what this did before -- is indistinguishable for ordinary
        # columns and WRONG for a pk column: the real write lands at the
        # NEW key and leaves the old row sitting there, so the table
        # ends up with two rows and the caller still addresses the old
        # one. in-place mutation renamed the single row instead, which
        # no backend does, and made that divergence unreproducible.
        if not has_l1:
            return False
        row = cache.get(_normalize(entity_id, primary_key_columns))
        if row is None:
            return False
        updated = dict(row)
        updated[field] = value
        cache[_key_from_row(updated)] = updated
        return True

    def _get_row(entity_id: Any) -> dict[str, Any] | None:
        # a COPY, matching SQLiteBackend.select_by_id. returning the live
        # dict let a caller mutate the "cache" through a read result,
        # which no real backend permits.
        if not has_l1:
            return None
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
