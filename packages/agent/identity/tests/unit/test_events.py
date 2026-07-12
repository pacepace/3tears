"""Unit coverage for the identity FrameworkEvents: discriminators + registration."""

from __future__ import annotations

from threetears.agent.identity.events import (
    IdentityAppliedEvent,
    IdentityConsentedEvent,
    IdentityProposedEvent,
    IdentityRolledBackEvent,
)
from threetears.langgraph.events import default_registry


def test_wire_discriminators() -> None:
    # the type literal is the wire contract; read the default without
    # constructing (agent_id / version_id / block_key are required).
    assert IdentityProposedEvent.model_fields["type"].default == "identity_proposed"
    assert IdentityConsentedEvent.model_fields["type"].default == "identity_consented"
    assert IdentityAppliedEvent.model_fields["type"].default == "identity_applied"
    assert IdentityRolledBackEvent.model_fields["type"].default == "identity_rolled_back"


def test_applied_event_carries_auto_applied_flag() -> None:
    common = {"agent_id": "a", "version_id": "v", "block_key": "personality"}
    assert IdentityAppliedEvent(**common, auto_applied=True).auto_applied is True
    assert IdentityAppliedEvent(**common).auto_applied is False


def test_events_registered_in_default_registry() -> None:
    names = default_registry.names()
    for wire in (
        "identity_proposed",
        "identity_consented",
        "identity_applied",
        "identity_rolled_back",
    ):
        assert wire in names
