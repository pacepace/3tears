"""A timestamptz key is named in one spelling: UTC, to the microsecond, in ISO form.

The contracts. ``key_text`` names an aware ``datetime`` in UTC to the microsecond whatever zone it came in, and
anything else as its ``str``; a collection's change record, its row message, its L2 key and its fence key are all
built from that spelling, so a message's text values (a row broadcast's, or a key read back from the change notes)
land on the very keys the typed values made.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

from threetears.core.collections.epoch_change import key_text
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.collections.schema_backed import (
    DATETIMETZ_TYPE,
    STRING_TYPE,
    Column,
    TableSchema,
    collection_for_schema,
)
from threetears.core.config import DefaultCoreConfig
from threetears.core.entities.base import BaseEntity

_PACIFIC = timezone(timedelta(hours=-8))
_AT = datetime(2026, 11, 3, 19, 0, 0, tzinfo=_PACIFIC)
_CANONICAL = "2026-11-04T03:00:00.000000+00:00"


class _Snapshot(BaseEntity):
    """a snapshot row: its own id is the instant, within the office its key leads with."""

    primary_key_field: str = "at"


def _collection() -> Any:
    registry = CollectionRegistry()
    registry.configure(kv_key_scope="test-principal")
    schema = TableSchema(
        name=f"snaps_{uuid.uuid4().hex[:8]}",
        primary_key=("office", "at"),
        columns=[Column("office", STRING_TYPE), Column("at", DATETIMETZ_TYPE), Column("seen", DATETIMETZ_TYPE)],
    )
    cls = collection_for_schema(schema, entity_class=_Snapshot)
    cls.invalidation_columns = ("seen",)
    return cls(registry, DefaultCoreConfig(), None)


class TestAnInstantHasOneSpelling:
    def test_an_aware_datetime_is_named_in_utc_to_the_microsecond(self) -> None:
        """Flips if a whole second drops its fraction or the zone it came in shows through."""
        assert key_text(_AT) == _CANONICAL
        assert key_text(_AT.astimezone(UTC)) == _CANONICAL
        assert key_text(datetime(2026, 11, 4, 3, 0, 0, 82658, tzinfo=UTC)) == "2026-11-04T03:00:00.082658+00:00"

    def test_anything_else_is_its_str(self) -> None:
        """Flips if a text, integer, uuid or naive value is respelled."""
        ident = uuid.uuid4()
        naive = datetime(2026, 11, 4, 3, 0, 0)
        assert [key_text(value) for value in ("pres", 7, ident, naive)] == ["pres", "7", str(ident), str(naive)]


class TestACollectionsKeysAreBuiltFromThatSpelling:
    def test_its_record_and_message_name_the_instant_canonically(self) -> None:
        """Flips if a change record or row message names a timestamptz key or column another way than its note."""
        collection = _collection()
        row = collection.changed_row(("pres", _AT), {"office": "pres", "at": _AT, "seen": _AT})
        assert row.ids == ["pres", _CANONICAL]
        assert row.columns == {"seen": _CANONICAL}
        message = collection.registry.row_message(collection.table_name, ("pres", _AT))
        assert message.ids == ["pres", _CANONICAL]

    def test_the_typed_key_and_its_text_land_on_one_l2_key(self) -> None:
        """Flips if a message's text key misses the L2 entry the typed key's write made."""
        collection = _collection()
        typed, text = ("pres", _AT), ("pres", _CANONICAL)
        assert collection.l2_key(typed) == collection.l2_key(text)
