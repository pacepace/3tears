"""Base entity class — thin cache proxy with change tracking.

Entities hold _id + _collection reference. All field data lives in the L1 cache,
accessed via collection._get_field_sync() / _set_field_sync(). No in-memory
data dicts.

When a collection is present, entity data MUST live in L1. The _changes dict
tracks individual field mutations (write path). Entities without a collection
(factory-created) use _changes as transient storage until saved.
"""

from __future__ import annotations

from typing import Any

from threetears.core.cache import MISSING
from threetears.core.logging import get_logger

log = get_logger(__name__)

# Internal attribute names that bypass __setattr__ routing
_INTERNAL_ATTRS = frozenset(
    {
        "_id",
        "_collection",
        "_is_new",
        "_dirty",
        "_changes",
        "_original_date_updated",
        "_column_names",
    }
)


class BaseEntity:
    """Thin cache proxy — holds _id + _collection reference, no data dict.

    Read path:
        _get_raw(field, default) checks _changes first, then reads from
        the L1 cache via collection._get_field_sync(). __getattr__ dispatches
        to _get_raw() for attributes not found via normal Python lookup.

    Write path:
        __setattr__ writes to the L1 cache via collection._set_field_sync()
        and records the change in _changes for dirty tracking.

    Serialization:
        to_dict() returns the full row from the L1 cache via
        collection._get_row_sync(), filtered to columns that belong to
        this entity.

    Entities created without a collection use _changes as temporary
    in-memory storage until they are attached to a collection via save().

    Subclasses set _primary_key_field to their entity-specific PK name
    (e.g. "user_id", "provider_id").
    """

    _primary_key_field: str = "id"

    def __init__(
        self,
        data: dict[str, Any],
        is_new: bool = True,
        collection: Any = None,
    ) -> None:
        pk_field = type(self)._primary_key_field
        entity_id = data.get(pk_field, data.get("id", ""))
        object.__setattr__(self, "_id", entity_id)
        object.__setattr__(self, "_collection", collection)
        object.__setattr__(self, "_is_new", is_new)
        object.__setattr__(self, "_dirty", is_new)
        object.__setattr__(
            self,
            "_original_date_updated",
            None if is_new else data.get("date_updated"),
        )
        object.__setattr__(self, "_column_names", frozenset(data.keys()))
        if collection is not None:
            wrote = collection._write_to_cache_sync(data)
            if wrote:
                object.__setattr__(self, "_changes", {})
            else:
                # No L1 backend — store data in _changes as fallback
                object.__setattr__(self, "_changes", dict(data))
        else:
            # No collection — transient dict storage for factory-created entities.
            object.__setattr__(self, "_changes", dict(data))

    @property
    def id(self) -> Any:
        """Get entity primary key value."""
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

    def _get_raw(self, field: str, default: Any = None) -> Any:
        """Read a single field. Checks _changes first, then L1 cache via collection."""
        changes = object.__getattribute__(self, "_changes")
        if field in changes:
            return changes[field]
        collection = object.__getattribute__(self, "_collection")
        if collection is None:
            return default
        entity_id = object.__getattribute__(self, "_id")
        result = collection._get_field_sync(entity_id, field)
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
        if collection is not None:
            entity_id = object.__getattribute__(self, "_id")
            collection._set_field_sync(entity_id, name, value)
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
        """Export entity data as dictionary from L1 cache or _changes fallback.

        Only returns columns that belong to this entity (tracked via
        _column_names).
        """
        collection = object.__getattribute__(self, "_collection")
        changes = object.__getattribute__(self, "_changes")
        if collection is None:
            return dict(changes)
        entity_id = object.__getattribute__(self, "_id")
        row = collection._get_row_sync(entity_id)
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
        """Reset dirty state and clear change tracking."""
        object.__setattr__(self, "_dirty", False)
        object.__setattr__(self, "_is_new", False)
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

    def _set_data(self, data: dict[str, Any]) -> None:
        """Replace entity data in the L1 cache (used by reload)."""
        collection = object.__getattribute__(self, "_collection")
        if collection is not None:
            wrote = collection._write_to_cache_sync(data)
            if not wrote:
                raise RuntimeError(f"L1 cache write failed in _set_data() for {type(self).__name__} id={self._id}")
        object.__setattr__(self, "_column_names", frozenset(data.keys()))
        object.__setattr__(self, "_changes", {})
        object.__setattr__(self, "_dirty", False)
        object.__setattr__(self, "_is_new", False)
        object.__setattr__(self, "_original_date_updated", data.get("date_updated"))

    def __repr__(self) -> str:
        entity_id = self._id
        return f"<{type(self).__name__} id={entity_id} dirty={self._dirty}>"
