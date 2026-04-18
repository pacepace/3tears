"""unit tests for ``threetears.agent.workspace.audit.publish_workspace_event``."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel, ValidationError

from threetears.agent.workspace.audit import (
    WorkspaceAuditEnvelope,
    publish_workspace_event,
)


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------


@dataclass
class _FakeNats:
    """records ``publish`` calls; optionally raises to simulate outage."""

    published: list[tuple[str, bytes]] = field(default_factory=list)
    raise_on_publish: BaseException | None = None

    async def publish(self, subject: str, payload: bytes) -> None:
        self.published.append((subject, payload))
        if self.raise_on_publish is not None:
            raise self.raise_on_publish


def _identity_kwargs(
    *,
    actor_user_id: UUID | None = None,
    calling_agent_id: UUID | None = None,
    owner_agent_id: UUID | None = None,
    customer_id: UUID | None = None,
    namespace_id: UUID | None = None,
) -> dict[str, UUID]:
    """build the WS-ACL-10 five-UUID identity tuple with defaults."""
    return {
        "actor_user_id": actor_user_id or uuid4(),
        "calling_agent_id": calling_agent_id or uuid4(),
        "owner_agent_id": owner_agent_id or uuid4(),
        "customer_id": customer_id or uuid4(),
        "namespace_id": namespace_id or uuid4(),
    }


# ---------------------------------------------------------------------------
# envelope shape
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_publish_builds_envelope_with_every_required_field() -> None:
    """envelope carries every canonical field with UUIDs coerced to strings."""
    nats = _FakeNats()
    ident = _identity_kwargs()
    agent_id = ident["calling_agent_id"]
    correlation_id = uuid4()
    await publish_workspace_event(
        nats_client=nats,
        namespace="3tears",
        event_type="workspace.fs_write",
        agent_id=agent_id,
        resource_type="workspace_file",
        resource_id="abc/notes.md",
        action="write",
        details={"bytes_after": 5, "sha256_after": "a" * 64, "version": 1},
        correlation_id=correlation_id,
        **ident,
    )
    assert len(nats.published) == 1
    subject, payload = nats.published[0]
    envelope = json.loads(payload.decode("utf-8"))
    assert envelope["event_type"] == "workspace.fs_write"
    assert envelope["actor_type"] == "agent"
    assert envelope["actor_user_id"] == str(ident["actor_user_id"])
    assert envelope["agent_id"] == str(agent_id)
    assert envelope["calling_agent_id"] == str(ident["calling_agent_id"])
    assert envelope["owner_agent_id"] == str(ident["owner_agent_id"])
    assert envelope["customer_id"] == str(ident["customer_id"])
    assert envelope["namespace_id"] == str(ident["namespace_id"])
    assert envelope["resource_type"] == "workspace_file"
    assert envelope["resource_id"] == "abc/notes.md"
    assert envelope["action"] == "write"
    assert envelope["details"] == {
        "bytes_after": 5,
        "sha256_after": "a" * 64,
        "version": 1,
    }
    assert envelope["correlation_id"] == str(correlation_id)
    assert isinstance(envelope["timestamp"], str)
    # timestamp parses as ISO-8601
    assert "T" in envelope["timestamp"]


# ---------------------------------------------------------------------------
# subject derivation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_publish_subject_uses_namespace_and_action() -> None:
    """subject is ``{namespace}.audit.workspace.{action}``, nothing more."""
    nats = _FakeNats()
    await publish_workspace_event(
        nats_client=nats,
        namespace="proj",
        event_type="workspace.doc_set",
        agent_id=uuid4(),
        resource_type="workspace_file",
        resource_id="ws/conf.yaml",
        action="set",
        details={},
        correlation_id=uuid4(),
        **_identity_kwargs(),
    )
    assert nats.published[0][0] == "proj.audit.workspace.set"


@pytest.mark.asyncio
async def test_publish_subject_reflects_each_action_verb() -> None:
    """every action verb routes to a distinct subject."""
    nats = _FakeNats()
    agent_id = uuid4()
    correlation_id = uuid4()
    for action in (
        "create",
        "reset",
        "delete",
        "write",
        "edit",
        "set",
        "merge",
        "rollback_to",
        "bind",
    ):
        await publish_workspace_event(
            nats_client=nats,
            namespace="ns",
            event_type=f"workspace.{action}",
            agent_id=agent_id,
            resource_type="workspace",
            resource_id="w-1",
            action=action,
            details={},
            correlation_id=correlation_id,
            **_identity_kwargs(calling_agent_id=agent_id, owner_agent_id=agent_id),
        )
    subjects = [s for s, _p in nats.published]
    assert subjects == [
        "ns.audit.workspace.create",
        "ns.audit.workspace.reset",
        "ns.audit.workspace.delete",
        "ns.audit.workspace.write",
        "ns.audit.workspace.edit",
        "ns.audit.workspace.set",
        "ns.audit.workspace.merge",
        "ns.audit.workspace.rollback_to",
        "ns.audit.workspace.bind",
    ]


# ---------------------------------------------------------------------------
# failure-non-blocking
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_publish_swallows_publish_exception() -> None:
    """a raising NATS client does not propagate -- audit is best-effort."""
    nats = _FakeNats(raise_on_publish=RuntimeError("nats offline"))
    # must NOT raise
    await publish_workspace_event(
        nats_client=nats,
        namespace="ns",
        event_type="workspace.fs_write",
        agent_id=uuid4(),
        resource_type="workspace_file",
        resource_id="ws/f.txt",
        action="write",
        details={},
        correlation_id=uuid4(),
        **_identity_kwargs(),
    )
    # publish was attempted once
    assert len(nats.published) == 1


@pytest.mark.asyncio
async def test_publish_is_noop_when_nats_client_none() -> None:
    """``nats_client=None`` short-circuits; no crash, nothing published."""
    # passing None must not raise -- covers bootstrap windows
    await publish_workspace_event(
        nats_client=None,
        namespace="ns",
        event_type="workspace.fs_write",
        agent_id=uuid4(),
        resource_type="workspace_file",
        resource_id="ws/f.txt",
        action="write",
        details={},
        correlation_id=uuid4(),
        **_identity_kwargs(),
    )


# ---------------------------------------------------------------------------
# serialize-to-json integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_publish_uses_core_json_encoder_for_uuid_and_datetime() -> None:
    """pydantic JSON encoder handles UUID-in-details naturally."""
    nats = _FakeNats()
    details: dict[str, Any] = {"trace_id": UUID("00000000-0000-0000-0000-00000000abcd")}
    await publish_workspace_event(
        nats_client=nats,
        namespace="ns",
        event_type="workspace.bind",
        agent_id=uuid4(),
        resource_type="workspace_file",
        resource_id="ws/file",
        action="bind",
        details=details,
        correlation_id=uuid4(),
        **_identity_kwargs(),
    )
    envelope = json.loads(nats.published[0][1].decode("utf-8"))
    # UUID in details survives as its string form via pydantic serialization
    assert envelope["details"]["trace_id"] == "00000000-0000-0000-0000-00000000abcd"


# ---------------------------------------------------------------------------
# typed envelope model
# ---------------------------------------------------------------------------


def test_workspace_audit_envelope_is_basemodel_instance() -> None:
    """envelope constructs cleanly as a pydantic BaseModel with typed fields."""
    ident = _identity_kwargs()
    agent_id = uuid4()
    correlation_id = uuid4()
    timestamp = datetime.now()
    envelope = WorkspaceAuditEnvelope(
        event_type="workspace.fs_write",
        actor_type="agent",
        agent_id=agent_id,
        resource_type="workspace_file",
        resource_id="abc/x.md",
        action="write",
        details={"bytes_after": 5},
        correlation_id=correlation_id,
        timestamp=timestamp,
        **ident,
    )
    assert isinstance(envelope, BaseModel)
    assert isinstance(envelope, WorkspaceAuditEnvelope)
    assert envelope.actor_user_id == ident["actor_user_id"]
    assert isinstance(envelope.actor_user_id, UUID)
    assert isinstance(envelope.correlation_id, UUID)
    assert envelope.timestamp == timestamp


def test_workspace_audit_envelope_json_roundtrip_matches_consumer_contract() -> None:
    """envelope.model_dump_json -> model_validate_json preserves every field.

    this is load-bearing: the Hub-side consumer parses exactly this JSON
    back through :meth:`WorkspaceAuditEnvelope.model_validate_json`. any
    drift in the serialization shape (UUID coercion, datetime format,
    dict passthrough) breaks the pipeline, so we validate the exact
    contract here.
    """
    ident = _identity_kwargs()
    agent_id = uuid4()
    correlation_id = uuid4()
    timestamp = datetime.now()
    original = WorkspaceAuditEnvelope(
        event_type="workspace.doc_merge",
        actor_type="agent",
        agent_id=agent_id,
        resource_type="workspace_file",
        resource_id="ws/conf.yaml",
        action="merge",
        details={"partial_keys": ["a", "b"]},
        correlation_id=correlation_id,
        timestamp=timestamp,
        **ident,
    )
    payload = original.model_dump_json().encode("utf-8")
    reconstructed = WorkspaceAuditEnvelope.model_validate_json(payload)
    assert reconstructed.event_type == original.event_type
    assert reconstructed.actor_user_id == ident["actor_user_id"]
    assert reconstructed.calling_agent_id == ident["calling_agent_id"]
    assert reconstructed.owner_agent_id == ident["owner_agent_id"]
    assert reconstructed.customer_id == ident["customer_id"]
    assert reconstructed.namespace_id == ident["namespace_id"]
    assert reconstructed.agent_id == agent_id
    assert reconstructed.resource_type == "workspace_file"
    assert reconstructed.resource_id == "ws/conf.yaml"
    assert reconstructed.action == "merge"
    assert reconstructed.details == {"partial_keys": ["a", "b"]}
    assert reconstructed.correlation_id == correlation_id
    assert reconstructed.timestamp == timestamp


def test_workspace_audit_envelope_rejects_extra_fields() -> None:
    """WS-ACL-10 uses ``extra='forbid'``; unknown fields fail the parse."""
    ident = _identity_kwargs()
    raw = {
        "event_type": "workspace.fs_write",
        "actor_type": "agent",
        "actor_user_id": str(ident["actor_user_id"]),
        "agent_id": str(uuid4()),
        "calling_agent_id": str(ident["calling_agent_id"]),
        "owner_agent_id": str(ident["owner_agent_id"]),
        "customer_id": str(ident["customer_id"]),
        "namespace_id": str(ident["namespace_id"]),
        "resource_type": "workspace_file",
        "resource_id": "ws/a.md",
        "action": "write",
        "details": {},
        "correlation_id": str(uuid4()),
        "timestamp": datetime.now().isoformat(),
        "future_field": "not-forward-compat",
    }
    with pytest.raises(ValidationError):
        WorkspaceAuditEnvelope.model_validate_json(json.dumps(raw))


def test_workspace_audit_envelope_rejects_legacy_actor_id_with_clear_message() -> None:
    """WS-ACL-10: ``actor_id=`` raises naming ``actor_user_id`` as the rename."""
    ident = _identity_kwargs()
    with pytest.raises(ValidationError) as excinfo:
        WorkspaceAuditEnvelope(
            event_type="workspace.fs_write",
            actor_type="agent",
            actor_id=ident["actor_user_id"],  # type: ignore[call-arg]
            agent_id=uuid4(),
            calling_agent_id=ident["calling_agent_id"],
            owner_agent_id=ident["owner_agent_id"],
            customer_id=ident["customer_id"],
            namespace_id=ident["namespace_id"],
            resource_type="workspace_file",
            resource_id="ws/a.md",
            action="write",
            details={},
            correlation_id=uuid4(),
            timestamp=datetime.now(),
        )
    assert "actor_user_id" in str(excinfo.value)


@pytest.mark.asyncio
async def test_publish_emits_wire_shape_the_consumer_can_parse() -> None:
    """publish_workspace_event emits bytes that parse as a WorkspaceAuditEnvelope.

    exact contract between agent (publisher) and Hub (consumer).
    """
    nats = _FakeNats()
    ident = _identity_kwargs()
    agent_id = uuid4()
    correlation_id = uuid4()
    await publish_workspace_event(
        nats_client=nats,
        namespace="ns",
        event_type="workspace.fs_edit",
        agent_id=agent_id,
        resource_type="workspace_file",
        resource_id="ws/f.md",
        action="edit",
        details={"occurrences": 1, "version": 2},
        correlation_id=correlation_id,
        **ident,
    )
    subject, payload = nats.published[0]
    assert subject == "ns.audit.workspace.edit"
    parsed = WorkspaceAuditEnvelope.model_validate_json(payload)
    assert parsed.event_type == "workspace.fs_edit"
    assert parsed.actor_user_id == ident["actor_user_id"]
    assert parsed.calling_agent_id == ident["calling_agent_id"]
    assert parsed.owner_agent_id == ident["owner_agent_id"]
    assert parsed.customer_id == ident["customer_id"]
    assert parsed.namespace_id == ident["namespace_id"]
    assert parsed.agent_id == agent_id
    assert parsed.correlation_id == correlation_id
    assert parsed.details == {"occurrences": 1, "version": 2}
