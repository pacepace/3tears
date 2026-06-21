"""
unit tests for the Folder primitive: :class:`Folder` entity +
:class:`FolderCollection`.

mirrors the conversations entity / collection tests:
- entity property getters / setters + UUID coercion + the composite-pk
  ``_id`` tuple
- collection L3 CRUD round-trip via an asyncpg mock + the L2
  serialize / deserialize round-trip
- the ``find_by_user`` domain query SQL contract + metadata round-trip
- a cross-entity check that a conversation's mutable ``folder_id`` sets
  / reads and that :meth:`ConversationsCollection.find_by_folder`
  filters by the partition + folder columns
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

from uuid import uuid7

from threetears.conversations.collection import ConversationsCollection
from threetears.conversations.entity import Conversation
from threetears.conversations.folder_collection import FolderCollection
from threetears.conversations.folder_entity import Folder


class _BackendlessFolderCollection(FolderCollection):
    """FolderCollection with the L1/L2 cache plumbing stubbed out.

    ``find_by_user`` promotes each fetched row into L1 (via the entity's
    ``write_to_cache_sync``) and L2 (via ``_save_to_l2``). a bare
    ``__new__`` instance has neither backend wired, so this subclass
    overrides both to clean no-ops -- keeping the coercion test focused
    on the entity hydration without reaching into the collection's
    private cache handles.
    """

    def __init__(self, postgres_pool: Any) -> None:
        """wire only the L3 pool the coercion path needs.

        :param postgres_pool: mock asyncpg pool returning fetch rows
        :ptype postgres_pool: Any
        :return: nothing
        :rtype: None
        """
        self.l3_pool = postgres_pool

    def write_to_cache_sync(
        self,
        data: dict[str, Any],
        primary_key: str | tuple[str, ...] | None = None,
    ) -> bool:
        """no-op L1 write so entity construction falls back to _changes.

        :param data: row dict (ignored)
        :ptype data: dict[str, Any]
        :param primary_key: pk override (ignored)
        :ptype primary_key: str | tuple[str, ...] | None
        :return: always ``False`` (no L1 backend)
        :rtype: bool
        """
        return False

    async def _save_to_l2(self, entity_id: Any, data: dict[str, Any]) -> bool:
        """no-op L2 promotion so the coercion path needs no NATS client.

        :param entity_id: composite pk (ignored)
        :ptype entity_id: Any
        :param data: row dict (ignored)
        :ptype data: dict[str, Any]
        :return: always ``False`` (no L2 backend)
        :rtype: bool
        """
        return False


def _sample_folder_data() -> dict[str, Any]:
    """
    build a fully populated folder row dict.

    :return: sample folder row dict
    :rtype: dict[str, Any]
    """
    now = datetime.now(UTC)
    return {
        "agent_id": uuid7(),
        "folder_id": uuid7(),
        "customer_id": uuid7(),
        "user_id": uuid7(),
        "name": "Work",
        "metadata": {"color": "#ff8800", "sort_order": 3},
        "date_created": now,
        "date_updated": now,
    }


def _make_pg_mock(store: dict[str, dict[str, Any]] | None = None) -> AsyncMock:
    """
    build a mock asyncpg pool backed by an in-memory dict.

    composite-pk schema column order (matches SchemaBackedCollection
    generator): agent_id, folder_id, customer_id, user_id, name,
    metadata, date_created, date_updated.

    :param store: optional existing row dict keyed by folder_id
    :ptype store: dict[str, dict[str, Any]] | None
    :return: async-mock pool
    :rtype: AsyncMock
    """
    if store is None:
        store = {}
    pg = AsyncMock()

    async def _fetchrow(query: str, *args: object) -> dict[str, Any] | None:
        """
        simulate composite-pk fetch ``WHERE agent_id = $1 AND folder_id = $2``.

        :param query: SQL text (ignored by the mock)
        :ptype query: str
        :param args: positional params
        :ptype args: object
        :return: stored row dict or ``None``
        :rtype: dict[str, Any] | None
        """
        entity_id = args[1] if len(args) > 1 else None
        return store.get(str(entity_id))

    async def _execute(query: str, *args: object) -> str:
        """
        simulate INSERT / UPDATE / DELETE against the in-memory store.

        :param query: SQL text
        :ptype query: str
        :param args: positional params
        :ptype args: object
        :return: status string compatible with asyncpg
        :rtype: str
        """
        result: str
        if "INSERT" in query:
            keys = [
                "agent_id",
                "folder_id",
                "customer_id",
                "user_id",
                "name",
                "metadata",
                "date_created",
                "date_updated",
            ]
            row = dict(zip(keys, args, strict=False))
            store[str(row["folder_id"])] = row
            result = "INSERT 0 1"
            return result
        if "UPDATE" in query:
            # CAS UPDATE pk-first: $1=agent_id, $2=folder_id, then mutables
            # (name, metadata, date_updated), then the CAS fence.
            entity_id = str(args[1])
            existing = store.get(entity_id)
            if existing is None:
                result = "UPDATE 0"
                return result
            existing["name"] = args[2]
            existing["metadata"] = args[3]
            existing["date_updated"] = args[4]
            result = "UPDATE 1"
            return result
        if "DELETE" in query:
            entity_id = str(args[1])
            if entity_id in store:
                del store[entity_id]
            result = "DELETE 1"
            return result
        result = "OK"
        return result

    pg.fetchrow = AsyncMock(side_effect=_fetchrow)
    pg.execute = AsyncMock(side_effect=_execute)
    pg.fetch = AsyncMock(return_value=[])
    return pg


class TestFolderEntityIdentity:
    """verify UUID-typed identity properties + composite-pk ``_id``."""

    def test_agent_id_returns_uuid(self) -> None:
        """agent_id returns the row value as a UUID."""
        data = _sample_folder_data()
        entity = Folder(data)
        assert entity.agent_id == data["agent_id"]
        assert isinstance(entity.agent_id, UUID)

    def test_agent_id_coerces_string(self) -> None:
        """string-valued agent_id gets coerced back to UUID."""
        data = _sample_folder_data()
        agent_uuid = data["agent_id"]
        data["agent_id"] = str(agent_uuid)
        entity = Folder(data)
        assert entity.agent_id == agent_uuid
        assert isinstance(entity.agent_id, UUID)

    def test_folder_id_returns_uuid(self) -> None:
        """folder_id returns the row value as a UUID."""
        data = _sample_folder_data()
        entity = Folder(data)
        assert entity.folder_id == data["folder_id"]
        assert isinstance(entity.folder_id, UUID)

    def test_customer_and_user_ids_return_uuid(self) -> None:
        """customer_id / user_id surface as UUIDs."""
        data = _sample_folder_data()
        entity = Folder(data)
        assert entity.customer_id == data["customer_id"]
        assert entity.user_id == data["user_id"]
        assert isinstance(entity.customer_id, UUID)
        assert isinstance(entity.user_id, UUID)

    def test_id_returns_composite_pk_tuple(self) -> None:
        """id property surfaces the (agent_id, folder_id) tuple."""
        data = _sample_folder_data()
        entity = Folder(data)
        assert entity.id == (data["agent_id"], data["folder_id"])


class TestFolderEntityValueProperties:
    """verify name / metadata / timestamp round-trips."""

    def test_name_round_trip(self) -> None:
        """name getter returns the stored display label."""
        data = _sample_folder_data()
        entity = Folder(data)
        assert entity.name == "Work"

    def test_metadata_round_trip(self) -> None:
        """metadata getter returns the stored dict (app-specific bits)."""
        data = _sample_folder_data()
        entity = Folder(data)
        assert entity.metadata == {"color": "#ff8800", "sort_order": 3}

    def test_metadata_none_is_preserved(self) -> None:
        """None metadata surfaces as None."""
        data = _sample_folder_data()
        data["metadata"] = None
        entity = Folder(data)
        assert entity.metadata is None

    def test_date_created_returns_datetime(self) -> None:
        """date_created getter returns the stored datetime."""
        data = _sample_folder_data()
        entity = Folder(data)
        assert entity.date_created == data["date_created"]
        assert isinstance(entity.date_created, datetime)

    def test_name_setter_tracks_change(self) -> None:
        """name setter records the mutation in get_changes."""
        cache: dict[str, dict[str, object]] = {}
        coll = MagicMock()

        def get_field(entity_id: object, field: str) -> object:
            row = cache.get(str(entity_id))
            from threetears.core.cache import MISSING

            return MISSING if row is None else row.get(field, MISSING)

        def set_field(entity_id: object, field: str, value: object) -> None:
            row = cache.get(str(entity_id))
            if row is not None:
                row[field] = value

        coll.get_field_sync = MagicMock(side_effect=get_field)
        coll.set_field_sync = MagicMock(side_effect=set_field)
        coll.get_row_sync = MagicMock(side_effect=lambda eid: cache.get(str(eid)))

        data = _sample_folder_data()
        entity = Folder(data, is_new=False, collection=coll)
        entity.name = "Personal"
        changes = entity.get_changes()
        assert changes["name"] == "Personal"


class TestFolderCollectionCrud:
    """L3 CRUD round-trip + L2 serialize/deserialize for FolderCollection."""

    async def test_table_name_and_entity_class(self) -> None:
        """static class props used by BaseCollection."""
        collection = FolderCollection.__new__(FolderCollection)
        assert collection.table_name == "folders"
        assert collection.entity_class is Folder

    async def test_insert_then_fetch_round_trip(self) -> None:
        """save_to_store insert path stores a row fetch_from_store reads back."""
        data = _sample_folder_data()
        l3: dict[str, dict[str, Any]] = {}
        pg = _make_pg_mock(l3)
        collection = FolderCollection.__new__(FolderCollection)
        collection.l3_pool = pg

        affected = await collection.save_to_store(data)
        assert affected == 1
        assert str(data["folder_id"]) in l3

        fetched = await collection.fetch_from_store((data["agent_id"], data["folder_id"]))
        assert fetched is not None
        assert fetched["name"] == "Work"
        assert fetched["metadata"] == {"color": "#ff8800", "sort_order": 3}

    async def test_update_renames_folder(self) -> None:
        """CAS update path mutates the name and respects the fence timestamp."""
        data = _sample_folder_data()
        l3: dict[str, dict[str, Any]] = {str(data["folder_id"]): data}
        pg = _make_pg_mock(l3)
        collection = FolderCollection.__new__(FolderCollection)
        collection.l3_pool = pg

        updated = dict(data)
        updated["name"] = "Archive"
        affected = await collection.save_to_store(updated, original_timestamp=data["date_updated"])
        assert affected == 1
        assert l3[str(data["folder_id"])]["name"] == "Archive"

    async def test_delete_removes_row(self) -> None:
        """delete_from_store removes the row from the backing store."""
        data = _sample_folder_data()
        l3: dict[str, dict[str, Any]] = {str(data["folder_id"]): data}
        pg = _make_pg_mock(l3)
        collection = FolderCollection.__new__(FolderCollection)
        collection.l3_pool = pg

        await collection.delete_from_store((data["agent_id"], data["folder_id"]))
        assert str(data["folder_id"]) not in l3

    async def test_l2_round_trip_preserves_uuids_and_metadata(self) -> None:
        """serialize + deserialize reconstructs UUIDs, datetimes, metadata."""
        data = _sample_folder_data()
        collection = FolderCollection.__new__(FolderCollection)

        payload = collection.serialize(data)
        restored = collection.deserialize(payload)

        assert isinstance(restored["agent_id"], UUID)
        assert isinstance(restored["folder_id"], UUID)
        assert restored["agent_id"] == data["agent_id"]
        assert isinstance(restored["date_created"], datetime)
        assert restored["date_created"] == data["date_created"]
        assert restored["metadata"] == {"color": "#ff8800", "sort_order": 3}


class TestFolderFindByUser:
    """tests for :meth:`FolderCollection.find_by_user` SQL contract."""

    async def test_empty_result_returns_empty_list(self) -> None:
        """No folders for the user -> empty list (no entity construction)."""
        pg = _make_pg_mock()
        pg.fetch = AsyncMock(return_value=[])
        collection = FolderCollection.__new__(FolderCollection)
        collection.l3_pool = pg

        result = await collection.find_by_user(uuid7(), uuid7())
        assert result == []

    async def test_query_filters_by_agent_and_user_ordered_by_name(self) -> None:
        """SQL pins the partition (agent_id $1) + owner (user_id $2) and
        orders by name ascending -- a generic, stable default; consumers
        re-sort by metadata themselves."""
        pg = _make_pg_mock()
        pg.fetch = AsyncMock(return_value=[])
        collection = FolderCollection.__new__(FolderCollection)
        collection.l3_pool = pg

        agent_id = uuid7()
        user_id = uuid7()
        await collection.find_by_user(agent_id, user_id)

        pg.fetch.assert_called_once()
        sql, *params = pg.fetch.call_args.args
        assert "agent_id = $1" in sql
        assert "user_id = $2" in sql
        assert "ORDER BY name ASC" in sql
        assert params[0] == agent_id
        assert params[1] == user_id

    async def test_find_by_user_coerces_rows_to_entities(self) -> None:
        """returned rows are coerced into Folder entities (metadata intact).

        the L1/L2 promotion that find_by_user performs is exercised against
        a fake collection whose cache write + KV resolution short-circuit,
        so the assertion stays on the entity coercion (the load-bearing
        per-collection behaviour) rather than the shared cache plumbing
        (covered by the core suite).
        """
        data = _sample_folder_data()
        pg = _make_pg_mock()
        pg.fetch = AsyncMock(return_value=[data])

        collection = _BackendlessFolderCollection(pg)

        result = await collection.find_by_user(data["agent_id"], data["user_id"])
        assert len(result) == 1
        assert isinstance(result[0], Folder)
        assert result[0].name == "Work"
        assert result[0].metadata == {"color": "#ff8800", "sort_order": 3}


class TestConversationFolderId:
    """the mutable ``conversations.folder_id`` FK + find_by_folder."""

    def test_conversation_folder_id_defaults_none(self) -> None:
        """a conversation with no folder_id reads None (created unfiled)."""
        now = datetime.now(UTC)
        data = {
            "conversation_id": uuid7(),
            "agent_id": uuid7(),
            "customer_id": uuid7(),
            "user_id": uuid7(),
            "channel_type": "slack",
            "status": "active",
            "date_created": now,
            "date_updated": now,
        }
        entity = Conversation(data)
        assert entity.folder_id is None

    def test_conversation_folder_id_set_and_read(self) -> None:
        """folder_id setter files the conversation; getter coerces to UUID."""
        now = datetime.now(UTC)
        folder_uuid = uuid7()
        data = {
            "conversation_id": uuid7(),
            "agent_id": uuid7(),
            "customer_id": uuid7(),
            "user_id": uuid7(),
            "channel_type": "slack",
            "status": "active",
            "folder_id": str(folder_uuid),
            "date_created": now,
            "date_updated": now,
        }
        entity = Conversation(data)
        assert entity.folder_id == folder_uuid
        assert isinstance(entity.folder_id, UUID)

    async def test_find_by_folder_filters_by_agent_and_folder_newest_first(self) -> None:
        """ConversationsCollection.find_by_folder pins the partition
        (agent_id $1) + folder_id ($2) and orders newest-first."""
        pg = AsyncMock()
        pg.fetch = AsyncMock(return_value=[])
        collection = ConversationsCollection.__new__(ConversationsCollection)
        collection.l3_pool = pg

        agent_id = uuid7()
        folder_id = uuid7()
        result = await collection.find_by_folder(agent_id, folder_id)

        assert result == []
        pg.fetch.assert_called_once()
        sql, *params = pg.fetch.call_args.args
        assert "agent_id = $1" in sql
        assert "folder_id = $2" in sql
        assert "ORDER BY date_created DESC" in sql
        assert params[0] == agent_id
        assert params[1] == folder_id
