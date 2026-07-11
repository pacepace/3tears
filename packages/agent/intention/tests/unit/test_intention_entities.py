"""Unit tests: IntentionStatus value set + IntentionEntity accessors."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal

from threetears.agent.intention.entities import IntentionEntity
from threetears.agent.intention.types import INTENTION_STATUS_VALUES, IntentionStatus


class TestIntentionStatus:
    def test_status_values_and_order(self) -> None:
        """The enum carries exactly the four lifecycle values, in order."""
        assert INTENTION_STATUS_VALUES == ("open", "asked", "granted", "dropped")

    def test_members_compare_equal_to_wire_value(self) -> None:
        """Members are ``str`` subclasses that equal their wire value."""
        assert IntentionStatus.OPEN == "open"
        assert IntentionStatus.ASKED == "asked"
        assert IntentionStatus.GRANTED == "granted"
        assert IntentionStatus.DROPPED == "dropped"

    def test_values_derived_from_enum(self) -> None:
        """The tuple is derived from the enum -- no drift possible."""
        assert INTENTION_STATUS_VALUES == tuple(s.value for s in IntentionStatus)


class TestIntentionEntity:
    def _row(self, **overrides: object) -> dict[str, object]:
        now = datetime.now(UTC)
        row: dict[str, object] = {
            "intention_id": uuid.uuid4(),
            "agent_id": uuid.uuid4(),
            "customer_id": uuid.uuid4(),
            "user_id": uuid.uuid4(),
            "status": "open",
            "content": "ask Pace about the wake threads",
            "embedding": None,
            "salience": Decimal("0.5000"),
            "last_decayed_at": None,
            "last_surfaced_at": None,
            "source_memory_id": None,
            "source_conversation_id": None,
            "date_created": now,
            "date_updated": now,
        }
        row.update(overrides)
        return row

    def test_scalar_accessors_round_trip(self) -> None:
        row = self._row()
        entity = IntentionEntity(row, is_new=False)
        assert entity.intention_id == row["intention_id"]
        assert entity.agent_id == row["agent_id"]
        assert entity.customer_id == row["customer_id"]
        assert entity.user_id == row["user_id"]
        assert entity.status == "open"
        assert entity.content == "ask Pace about the wake threads"
        assert entity.date_created == row["date_created"]
        assert entity.date_updated == row["date_updated"]

    def test_salience_decimal_exposed_as_float(self) -> None:
        """NUMERIC(5,4) arrives as ``Decimal``; the accessor yields ``float``."""
        entity = IntentionEntity(self._row(salience=Decimal("0.9000")), is_new=False)
        assert entity.salience == 0.9
        assert isinstance(entity.salience, float)

    def test_null_scope_and_provenance_tolerated(self) -> None:
        """NULL user/customer/source columns hydrate to ``None`` (no UUID('None'))."""
        entity = IntentionEntity(
            self._row(
                customer_id=None,
                user_id=None,
                source_memory_id=None,
                source_conversation_id=None,
            ),
            is_new=False,
        )
        assert entity.customer_id is None
        assert entity.user_id is None
        assert entity.source_memory_id is None
        assert entity.source_conversation_id is None

    def test_provenance_soft_refs_round_trip_when_set(self) -> None:
        mem_id = uuid.uuid4()
        conv_id = uuid.uuid4()
        entity = IntentionEntity(
            self._row(source_memory_id=mem_id, source_conversation_id=conv_id),
            is_new=False,
        )
        assert entity.source_memory_id == mem_id
        assert entity.source_conversation_id == conv_id

    def test_string_ids_coerced_to_uuid(self) -> None:
        """Cache tiers hand back stringified UUIDs; accessors coerce them."""
        aid = uuid.uuid4()
        iid = uuid.uuid4()
        entity = IntentionEntity(
            self._row(agent_id=str(aid), intention_id=str(iid)),
            is_new=False,
        )
        assert entity.agent_id == aid
        assert entity.intention_id == iid
