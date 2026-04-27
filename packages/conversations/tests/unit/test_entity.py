"""
unit tests for :class:`threetears.conversations.entity.Conversation`.

exercises the property getter / setter pairs, string-to-UUID coercion,
and the serialized shape that the collection relies on.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock
from uuid import UUID

import pytest
from uuid import uuid7

from threetears.conversations.entity import Conversation
from threetears.core.cache import MISSING


@pytest.fixture()
def mock_collection() -> tuple[MagicMock, dict[str, dict[str, object]]]:
    """
    provide a mock collection with an in-memory L1 cache simulation.

    :return: mock collection and the backing cache dict
    :rtype: tuple[MagicMock, dict[str, dict[str, object]]]
    """
    cache: dict[str, dict[str, object]] = {}
    coll = MagicMock()

    def write_to_cache(data: dict[str, object]) -> bool:
        """
        write a row into the cache keyed on primary-key id.

        :param data: row dict
        :ptype data: dict[str, object]
        :return: always True (cache always accepts)
        :rtype: bool
        """
        pk = data.get("id", "")
        cache[str(pk)] = dict(data)
        return True

    def get_field(entity_id: object, field: str) -> object:
        """
        read a field out of the cache, returning MISSING when absent.

        :param entity_id: entity primary key
        :ptype entity_id: object
        :param field: column name
        :ptype field: str
        :return: cached value or MISSING
        :rtype: object
        """
        row = cache.get(str(entity_id))
        result: object
        if row is None:
            result = MISSING
        else:
            result = row.get(field, MISSING)
        return result

    def set_field(entity_id: object, field: str, value: object) -> None:
        """
        write a field into the cache row.

        :param entity_id: entity primary key
        :ptype entity_id: object
        :param field: column name
        :ptype field: str
        :param value: new value
        :ptype value: object
        """
        row = cache.get(str(entity_id))
        if row is not None:
            row[field] = value

    def get_row(entity_id: object) -> dict[str, object] | None:
        """
        return the full cached row for entity_id.

        :param entity_id: entity primary key
        :ptype entity_id: object
        :return: row dict or ``None``
        :rtype: dict[str, object] | None
        """
        return cache.get(str(entity_id))

    coll.write_to_cache_sync = MagicMock(side_effect=write_to_cache)
    coll.get_field_sync = MagicMock(side_effect=get_field)
    coll.set_field_sync = MagicMock(side_effect=set_field)
    coll.get_row_sync = MagicMock(side_effect=get_row)

    return coll, cache


def _sample_data() -> dict[str, object]:
    """
    build a fully populated conversation row dict for testing.

    :return: sample conversation data
    :rtype: dict[str, object]
    """
    now = datetime.now(UTC)
    return {
        "id": uuid7(),
        "agent_id": uuid7(),
        "customer_id": uuid7(),
        "user_id": uuid7(),
        "channel_type": "slack",
        "conversation_ref": "C1234567890",
        "status": "active",
        "summary": "discussing quarterly metrics",
        "date_created": now,
        "date_updated": now,
        "date_last_message": now,
        "metadata": {"source": "test"},
    }


class TestConversationIdentityProperties:
    """verify UUID-typed properties on :class:`Conversation`."""

    def test_agent_id_returns_uuid(self) -> None:
        """agent_id returns the row value as a UUID."""
        data = _sample_data()
        entity = Conversation(data)

        result = entity.agent_id

        assert result == data["agent_id"]
        assert isinstance(result, UUID)

    def test_agent_id_coerces_string(self) -> None:
        """string-valued agent_id gets coerced back to UUID."""
        data = _sample_data()
        agent_uuid = data["agent_id"]
        data["agent_id"] = str(agent_uuid)
        entity = Conversation(data)

        result = entity.agent_id

        assert result == agent_uuid
        assert isinstance(result, UUID)

    def test_customer_id_returns_uuid(self) -> None:
        """customer_id returns the row value as a UUID."""
        data = _sample_data()
        entity = Conversation(data)

        result = entity.customer_id

        assert result == data["customer_id"]
        assert isinstance(result, UUID)

    def test_user_id_returns_uuid(self) -> None:
        """user_id returns the row value as a UUID."""
        data = _sample_data()
        entity = Conversation(data)

        result = entity.user_id

        assert result == data["user_id"]
        assert isinstance(result, UUID)

    def test_id_returns_underlying_pk(self) -> None:
        """id property surfaces the composite primary key tuple."""
        data = _sample_data()
        entity = Conversation(data)

        assert entity.id == (data["agent_id"], data["id"])


class TestConversationChannelProperties:
    """verify channel-type and channel-reference properties."""

    def test_channel_type_round_trip(self) -> None:
        """channel_type getter returns the stored string."""
        data = _sample_data()
        entity = Conversation(data)

        assert entity.channel_type == "slack"

    def test_conversation_ref_round_trip(self) -> None:
        """conversation_ref getter returns the stored string."""
        data = _sample_data()
        entity = Conversation(data)

        assert entity.conversation_ref == "C1234567890"

    def test_conversation_ref_none_is_preserved(self) -> None:
        """None conversation_ref surfaces as None."""
        data = _sample_data()
        data["conversation_ref"] = None
        entity = Conversation(data)

        assert entity.conversation_ref is None


class TestConversationLifecycleProperties:
    """verify status, summary, and timestamp properties."""

    def test_status_round_trip(self) -> None:
        """status getter surfaces the stored enum token."""
        data = _sample_data()
        entity = Conversation(data)

        assert entity.status == "active"

    def test_summary_round_trip(self) -> None:
        """summary getter returns the stored text."""
        data = _sample_data()
        entity = Conversation(data)

        assert entity.summary == "discussing quarterly metrics"

    def test_summary_none_is_preserved(self) -> None:
        """None summary surfaces as None."""
        data = _sample_data()
        data["summary"] = None
        entity = Conversation(data)

        assert entity.summary is None

    def test_date_created_returns_datetime(self) -> None:
        """date_created getter returns the stored datetime."""
        data = _sample_data()
        entity = Conversation(data)

        assert entity.date_created == data["date_created"]
        assert isinstance(entity.date_created, datetime)

    def test_date_updated_returns_datetime(self) -> None:
        """date_updated getter returns the stored datetime."""
        data = _sample_data()
        entity = Conversation(data)

        assert entity.date_updated == data["date_updated"]

    def test_date_last_message_round_trip(self) -> None:
        """date_last_message getter returns the stored datetime."""
        data = _sample_data()
        entity = Conversation(data)

        assert entity.date_last_message == data["date_last_message"]

    def test_date_last_message_none_is_preserved(self) -> None:
        """None date_last_message surfaces as None."""
        data = _sample_data()
        data["date_last_message"] = None
        entity = Conversation(data)

        assert entity.date_last_message is None


class TestConversationMetadata:
    """verify metadata JSONB property."""

    def test_metadata_round_trip(self) -> None:
        """metadata getter returns the stored dict."""
        data = _sample_data()
        entity = Conversation(data)

        assert entity.metadata == {"source": "test"}

    def test_metadata_none_is_preserved(self) -> None:
        """None metadata surfaces as None."""
        data = _sample_data()
        data["metadata"] = None
        entity = Conversation(data)

        assert entity.metadata is None


class TestConversationSetterWithCollection:
    """verify setters propagate into the collection L1 cache."""

    def test_status_setter_writes_to_cache(
        self,
        mock_collection: tuple[MagicMock, dict[str, dict[str, object]]],
    ) -> None:
        """setter routes the new value through the collection."""
        coll, _ = mock_collection
        data = _sample_data()
        entity = Conversation(data, is_new=False, collection=coll)

        entity.status = "closed"

        assert entity.status == "closed"
        # composite-pk entity addresses cache via the (agent_id, id) tuple.
        coll.set_field_sync.assert_called_with((data["agent_id"], data["id"]), "status", "closed")

    def test_summary_setter_tracks_change(
        self,
        mock_collection: tuple[MagicMock, dict[str, dict[str, object]]],
    ) -> None:
        """summary setter records the mutation in get_changes."""
        coll, _ = mock_collection
        data = _sample_data()
        entity = Conversation(data, is_new=False, collection=coll)

        entity.summary = "new distilled summary"

        changes = entity.get_changes()
        assert changes["summary"] == "new distilled summary"


class TestConversationToDict:
    """verify ``to_dict`` reflects the stored scoping and lifecycle data."""

    def test_to_dict_includes_all_scope_ids(self) -> None:
        """to_dict surfaces agent / customer / user identity fields."""
        data = _sample_data()
        entity = Conversation(data)

        out = entity.to_dict()

        assert out["agent_id"] == data["agent_id"]
        assert out["customer_id"] == data["customer_id"]
        assert out["user_id"] == data["user_id"]
