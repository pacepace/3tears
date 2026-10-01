"""Base entity class — thin cache proxy with change tracking.

Entities hold _id + _collection reference. A NEW entity's field data lives in the
L1 cache, accessed via collection.get_field_sync() / set_field_sync(); the
_changes dict tracks individual field mutations (write path).

A LOADED entity (``is_new=False``) with a collection holds its own row and writes
no cache tier on construction; _changes still tracks only its edits. Whoever read
the row decides whether L1 takes it: a by-key read does, under the collection's
per-key fence, and a multi-row scan does not, since it read L3 outside that fence
and a write of the key may have landed meanwhile. A new entity holds its row from its
first save on, as a loaded one does: L1's copy of the key is a cache any eviction may
drop, and a handle reading through it would answer None. Entities without a collection
(factory-created) use _changes as transient storage until saved.
"""

from __future__ import annotations

from typing import Any

from threetears.core.cache import MISSING
from threetears.observe import get_logger

__all__ = ["BaseEntity", "derive_addressing_id"]

log = get_logger(__name__)

# Internal attribute names that bypass __setattr__ routing. The
# un-prefixed ``original_date_updated`` is public state (the entity's
# last-persisted timestamp, read by collections for optimistic-
# concurrency CAS) but is still routed through ``__setattr__`` direct
# to ``object.__setattr__`` so the change-tracking path does not treat
# it as a user-column edit.
_INTERNAL_ATTRS = frozenset(
    {
        "_id",
        "_row_id",
        "_collection",
        "_is_new",
        "_dirty",
        "_changes",
        "_held",
        "original_date_updated",
        "_column_names",
    }
)


def derive_addressing_id(row_id: Any, data: dict[str, Any], collection: Any, *, strict: bool = False) -> Any:
    """derive the tier-addressing key for one entity's row.

    mirrors the rule :meth:`BaseCollection.save_entity` already applies
    when it rebuilds a composite key from the on-the-wire payload, so
    construction and persistence address the same row. a single-pk
    collection -- and any entity constructed with no collection at all,
    which cannot know its table's key shape -- keeps the scalar
    ``row_id`` it always had.

    the composite branch requires every declared pk column to be present
    in ``data``. a payload missing one is left on the scalar form rather
    than silently addressing a ``None`` component: the miss then surfaces
    as :meth:`BaseCollection.normalize_pk`'s arity error at the first
    tier access, which names the table and the expected columns.

    :param row_id: scalar value of the entity's ``primary_key_field``
    :ptype row_id: Any
    :param data: row dict the entity was constructed from
    :ptype data: dict[str, Any]
    :param collection: owning collection, or ``None`` for a transient
        factory-created entity
    :ptype collection: Any
    :param strict: when true, a composite-pk payload missing a declared
        pk column raises instead of falling back to the scalar. the
        write path sets this: a save that silently addressed the wrong
        row would write L3 under one key and cache it under another
    :ptype strict: bool
    :return: scalar pk value for single-pk tables, declared-order tuple
        for composite-pk tables
    :rtype: Any
    :raises KeyError: when ``strict`` and a declared pk column is absent
    """
    if collection is None:
        return row_id
    pk_cols = getattr(collection, "primary_key_columns", ())
    # the tuple check is not defensive noise: ``BaseCollection`` types
    # this property as ``tuple[str, ...]``, and anything else reaching
    # here is a stand-in (a bare ``MagicMock`` in a unit test) that
    # cannot describe a key shape. those keep the scalar form.
    if not isinstance(pk_cols, tuple) or len(pk_cols) < 2:
        return row_id
    missing = [col for col in pk_cols if col not in data]
    if missing:
        if strict:
            table = getattr(collection, "table_name", "<unknown>")
            raise KeyError(f"{table}: cannot address row -- payload is missing pk column(s) {missing} of {pk_cols}")
        return row_id
    return tuple(data[col] for col in pk_cols)


class BaseEntity:
    """Thin cache proxy — holds _id + _collection reference, no data dict.

    Read path:
        _get_raw(field, default) checks _changes first, then the entity's
        held row when it holds one (:meth:`hold_row`), then reads from the
        L1 cache via collection.get_field_sync(). __getattr__ dispatches
        to _get_raw() for attributes not found via normal Python lookup.

    Write path:
        __setattr__ records the change in _changes for dirty tracking and,
        for an entity that reads through L1 (a new one), writes it to the
        L1 cache via collection.set_field_sync(). An entity holding its own
        row leaves L1 alone: the row L1 holds for the key may be another
        version.

    Serialization:
        to_dict() returns the held row with the entity's edits, or the full
        row from the L1 cache via collection.get_row_sync(), filtered to
        columns that belong to this entity.

    A loaded entity (``is_new=False``) with a collection holds its own row
    and writes no cache tier on construction: only the read that produced
    the row may decide whether L1 takes it, under the collection's per-key
    fence. A new entity reads through L1 only until it is saved: from then on
    the collection has it hold the row as stored (and a reload, the row it
    reloaded), since L1 may drop the key at any time.

    Entities created without a collection use _changes as temporary
    in-memory storage until they are attached to a collection via save().

    Subclasses set primary_key_field (class attribute) to their entity-
    specific PK name (e.g. "user_id", "provider_id"). collections read
    this as part of the cache-coherence contract; renaming it would
    break persistence in every subclass, so it is public API.

    Two identities, and they differ only on composite-pk tables:

    - ``_id`` is the **addressing** key the framework hands to
      :meth:`BaseCollection.normalize_pk` / :meth:`BaseCollection.l2_key`
      and every L1 / L2 / L3 path beneath them. It is derived from the
      owning collection's ``primary_key_columns``: the scalar value for
      a single-pk table, the declared-order tuple for a composite-pk
      one.
    - ``id`` is the entity's own **scalar** identity -- the value of the
      column named by :attr:`primary_key_field`. On a single-pk table
      the two coincide, which is why this distinction was invisible
      until composite keys arrived.

    Deriving ``_id`` here is what makes a composite-pk entity need no
    ``__init__`` override: the same rule
    :meth:`BaseCollection.save_entity` already applies on the write path
    now applies on construction, so the two cannot disagree.

    :cvar primary_key_field: name of the column whose value :attr:`id`
        returns; default ``"id"``, subclasses override. On a composite-pk
        table this names the bare row id, NOT the partition column --
        the partition column reaches ``_id`` through the collection's
        declared ``primary_key_columns``
    :ivar original_date_updated: timestamp stamped on the row when it
        was last loaded from L3. collections read this as the
        optimistic-concurrency CAS token on save. cleared to ``None``
        on new entities and refreshed after every successful persist
    """

    primary_key_field: str = "id"

    def __init__(
        self,
        data: dict[str, Any],
        is_new: bool = True,
        collection: Any = None,
    ) -> None:
        pk_field = type(self).primary_key_field
        row_id = data.get(pk_field, data.get("id", ""))
        object.__setattr__(self, "_row_id", row_id)
        object.__setattr__(self, "_id", derive_addressing_id(row_id, data, collection))
        object.__setattr__(self, "_collection", collection)
        object.__setattr__(self, "_is_new", is_new)
        object.__setattr__(self, "_dirty", is_new)
        object.__setattr__(
            self,
            "original_date_updated",
            None if is_new else data.get("date_updated"),
        )
        object.__setattr__(self, "_column_names", frozenset(data.keys()))
        # A loaded entity never writes L1 here. Its row came from a read, and only the read knows
        # whether L1 may take it: a by-key read caches under the collection's per-key fence
        # before building the entity, while a scan read L3 outside that fence, and a row it
        # cached could be older than a write that landed meanwhile, with nothing left to evict
        # it. Writing here would bypass the fence for every scan in every package.
        object.__setattr__(self, "_held", None)
        if collection is None:
            # No collection — transient dict storage for factory-created entities.
            object.__setattr__(self, "_changes", dict(data))
        elif not is_new:
            object.__setattr__(self, "_changes", {})
            self.hold_row(data)
        elif collection.write_to_cache_sync(data):
            object.__setattr__(self, "_changes", {})
        else:
            # No L1 backend — store data in _changes as fallback
            object.__setattr__(self, "_changes", dict(data))

    @property
    def id(self) -> Any:
        """Get entity primary key value.

        returns the **scalar** value of the column named by
        :attr:`primary_key_field`. on a single-pk table that is the
        whole key and equals ``_id``; on a composite-pk table it is the
        bare row id, while ``_id`` carries the full addressing tuple.

        :return: scalar primary-key value
        :rtype: Any
        """
        return self._row_id

    @property
    def addressing_id(self) -> Any:
        """Get the key every tier addresses this entity's row by.

        the shape :meth:`BaseCollection.normalize_pk` expects: the scalar
        pk value on a single-pk table, the declared-order tuple of pk
        values on a composite-pk one. use this, not :attr:`id`, whenever
        a row is being fetched, cached, invalidated or deleted -- on a
        composite-pk table :attr:`id` names only the bare row id and
        addresses nothing on its own.

        :return: addressing key matching the collection's pk arity
        :rtype: Any
        """
        return self._id

    @property
    def is_dirty(self) -> bool:
        """Check if entity has unsaved changes."""
        dirty: bool = self._dirty
        return dirty

    @property
    def is_new(self) -> bool:
        """Check if entity is newly created (not loaded from storage)."""
        is_new_flag: bool = self._is_new
        return is_new_flag

    @property
    def holds_row(self) -> bool:
        """whether the entity holds its own row rather than reading it through L1.

        :return: ``True`` when the entity answers from a row it holds
        :rtype: bool
        """
        return object.__getattribute__(self, "_held") is not None

    def hold_row(self, data: dict[str, Any]) -> None:
        """keep ``data`` as this entity's own row, detached from L1.

        Reads and :meth:`to_dict` answer from it with the entity's edits on top, and attribute
        writes change the entity without touching L1: the row L1 holds for the key, if any, may be
        another version of it. A loaded entity starts this way; a collection calls this once it has
        saved or reloaded the entity, and when a write of it did not land, so the handle reads what
        it saved or read whatever later drops L1's copy of the key. Unsaved edits stay tracked.

        :param data: the row, keyed by column name
        :ptype data: dict[str, Any]
        :return: nothing
        :rtype: None
        """
        object.__setattr__(self, "_held", dict(data))
        columns = object.__getattribute__(self, "_column_names")
        object.__setattr__(self, "_column_names", columns | frozenset(data.keys()))

    def _get_raw(self, field: str, default: Any = None) -> Any:
        """Read a single field: the entity's edits, then its held row, then L1 via the collection."""
        changes = object.__getattribute__(self, "_changes")
        if field in changes:
            return changes[field]
        held = object.__getattribute__(self, "_held")
        if held is not None:
            return held.get(field, default)
        collection = object.__getattribute__(self, "_collection")
        if collection is None:
            return default
        entity_id = object.__getattribute__(self, "_id")
        result = collection.get_field_sync(entity_id, field)
        return result if result is not MISSING else default

    def __getattr__(self, name: str) -> Any:
        """Get attribute value via cache proxy."""
        result = self._get_raw(name, MISSING)
        if result is not MISSING:
            return result
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        """Set attribute value via cache proxy with change tracking."""
        if name in _INTERNAL_ATTRS:
            object.__setattr__(self, name, value)
            return
        collection = object.__getattribute__(self, "_collection")
        if collection is not None and object.__getattribute__(self, "_held") is None:
            entity_id = object.__getattribute__(self, "_id")
            collection.set_field_sync(entity_id, name, value)
        changes = object.__getattribute__(self, "_changes")
        changes[name] = value
        # Expand column set when new fields are written
        columns = object.__getattribute__(self, "_column_names")
        if name not in columns:
            object.__setattr__(self, "_column_names", columns | {name})
        object.__setattr__(self, "_dirty", True)

    def get_changes(self) -> dict[str, Any]:
        """Get dictionary of modified fields."""
        if object.__getattribute__(self, "_is_new"):
            return self.to_dict()
        return dict(object.__getattribute__(self, "_changes"))

    def to_dict(self) -> dict[str, Any]:
        """Export entity data as dictionary: its held row with its edits, L1, or the _changes fallback.

        Only returns columns that belong to this entity (tracked via
        _column_names).
        """
        collection = object.__getattribute__(self, "_collection")
        changes = object.__getattribute__(self, "_changes")
        if collection is None:
            return dict(changes)
        held = object.__getattribute__(self, "_held")
        if held is not None:
            return {**held, **changes}
        entity_id = object.__getattribute__(self, "_id")
        row = collection.get_row_sync(entity_id)
        if row is None:
            if changes:
                return dict(changes)
            raise RuntimeError(
                f"L1 cache miss in to_dict() for {type(self).__name__} id={entity_id}; entity data must be in L1"
            )
        columns = object.__getattribute__(self, "_column_names")
        result: dict[str, Any] = {k: v for k, v in row.items() if k in columns}
        return result

    def mark_clean(self) -> None:
        """Reset dirty state and clear change tracking; a held row takes the edits into itself."""
        object.__setattr__(self, "_dirty", False)
        object.__setattr__(self, "_is_new", False)
        held = object.__getattribute__(self, "_held")
        if held is not None:
            # a held row stays held: the edits just persisted become part of it.
            object.__setattr__(self, "_held", {**held, **object.__getattribute__(self, "_changes")})
        object.__setattr__(self, "_changes", {})
        log.debug(
            "Entity marked clean",
            extra={"extra_data": {"id": str(self._id)}},
        )

    async def save(self) -> None:
        """Persist entity changes through parent collection."""
        collection = self._collection
        if collection is None:
            raise RuntimeError("Cannot save entity without collection reference")
        await collection.save_entity(self)

    async def reload(self) -> None:
        """Reload entity data from storage through parent collection."""
        collection = self._collection
        if collection is None:
            raise RuntimeError("Cannot reload entity without collection reference")
        await collection.reload_entity(self)

    def set_data(self, data: dict[str, Any]) -> None:
        """replace entity data with freshly-loaded row; called by collection reload.

        rewrites the L1 cache with the given row (when the entity is
        attached to a collection and reads through L1), or takes it as
        the entity's held row (when it holds one, :meth:`hold_row`:
        whether L1 takes the row is its reader's decision, under the
        collection's fence), resets the per-field change buffer,
        clears the dirty/new flags, and refreshes the optimistic-
        concurrency token so subsequent saves check against the new
        persisted timestamp. name is public because the collection
        invokes this across the class boundary during ``reload_entity``.

        :param data: row dict as returned by the backing store; must
            contain all columns the collection expects for this entity
        :ptype data: dict[str, Any]
        :return: None
        :rtype: None
        :raises RuntimeError: when an L1 backend is wired but the
            cache write returns false (indicates bad metadata or a
            backend that rejected the row)
        """
        collection = object.__getattribute__(self, "_collection")
        holding = object.__getattribute__(self, "_held") is not None
        if collection is not None and not holding:
            wrote = collection.write_to_cache_sync(data)
            if not wrote:
                raise RuntimeError(f"L1 cache write failed in set_data() for {type(self).__name__} id={self._id}")
        # Re-derive identity from the incoming row. Skipping this left
        # the entity addressing its PREVIOUS row after a reload that
        # returned a different one: the new row went into L1 under its
        # own key while every read still went to the old one, silently
        # and with no exception. Harmless while pk columns never change,
        # and this is the method that makes them able to.
        pk_field = type(self).primary_key_field
        row_id = data.get(pk_field, data.get("id", ""))
        collection = object.__getattribute__(self, "_collection")
        object.__setattr__(self, "_row_id", row_id)
        object.__setattr__(self, "_id", derive_addressing_id(row_id, data, collection))
        object.__setattr__(self, "_column_names", frozenset(data.keys()))
        object.__setattr__(self, "_changes", {})
        object.__setattr__(self, "_held", dict(data) if holding else None)
        object.__setattr__(self, "_dirty", False)
        object.__setattr__(self, "_is_new", False)
        object.__setattr__(self, "original_date_updated", data.get("date_updated"))

    def __repr__(self) -> str:
        entity_id = self._id
        return f"<{type(self).__name__} id={entity_id} dirty={self._dirty}>"
