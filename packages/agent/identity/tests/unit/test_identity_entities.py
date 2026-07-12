"""agent-identity entity tests: composite-pk _id, property coercion, nullables."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from threetears.agent.identity.entities import IdentityVersionEntity

_AGENT = UUID("00000000-0000-0000-0000-000000000001")
_VERSION = UUID("00000000-0000-0000-0000-0000000000a1")
_USER = UUID("00000000-0000-0000-0000-00000000000b")


def _entity(**overrides: object) -> IdentityVersionEntity:
    data: dict[str, object] = {
        "agent_id": _AGENT,
        "version_id": _VERSION,
        "customer_id": _USER,
        "user_id": _USER,
        "block_key": "personality",
        "content": "You are Saoirse.",
        "rationale": "sharpen the voice",
        "content_hash": "abc123",
        "parent_version_id": None,
        "status": "proposed",
        "proposer_agent_id": _AGENT,
        "consenter_user_id": None,
        "date_created": datetime.now(UTC),
        "date_updated": None,
    }
    data.update(overrides)
    return IdentityVersionEntity(data, is_new=True)


def test_properties_expose_columns() -> None:
    entity = _entity()
    # the composite (agent_id, version_id) pk components resolve; the
    # tuple-_id addressing they back is exercised by the collection
    # round-trip (integration).
    assert entity.version_id == _VERSION
    assert entity.agent_id == _AGENT
    assert entity.block_key == "personality"
    assert entity.content == "You are Saoirse."
    assert entity.rationale == "sharpen the voice"
    assert entity.content_hash == "abc123"
    assert entity.status == "proposed"
    assert entity.proposer_agent_id == _AGENT


def test_nullable_fields_tolerate_none() -> None:
    entity = _entity(rationale=None, parent_version_id=None, consenter_user_id=None)
    assert entity.rationale is None
    assert entity.parent_version_id is None
    assert entity.consenter_user_id is None


def test_uuid_columns_coerce_string_from_cache_tier() -> None:
    """L1/L2 tiers hand back str uuids; the accessors coerce them."""
    entity = IdentityVersionEntity(
        {
            "agent_id": str(_AGENT),
            "version_id": str(_VERSION),
            "parent_version_id": str(_VERSION),
            "block_key": "presence",
            "content": "x",
            "content_hash": "h",
            "status": "active",
        },
        is_new=False,
    )
    assert entity.agent_id == _AGENT
    assert entity.version_id == _VERSION
    assert entity.parent_version_id == _VERSION


def test_mutable_lifecycle_setters() -> None:
    entity = _entity()
    entity.status = "active"
    entity.consenter_user_id = _USER
    assert entity.status == "active"
    assert entity.consenter_user_id == _USER
