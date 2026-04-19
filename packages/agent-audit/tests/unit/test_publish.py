"""unit tests for :func:`threetears.agent.audit.publish_audit`.

covers subject-naming, payload-shape, and the fire-and-forget
invariant: every publish failure logs at WARN and returns cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid7

import pytest

from threetears.agent.audit import AuditEvent, publish_audit


@dataclass
class _FakeNats:
    """minimal fake NATS client recording every publish call."""

    publish_calls: list[tuple[str, bytes]] = field(default_factory=list)
    raise_on_publish: BaseException | None = None

    async def publish(self, subject: str, payload: bytes) -> None:
        """record the publish invocation or raise when configured to."""
        self.publish_calls.append((subject, payload))
        if self.raise_on_publish is not None:
            raise self.raise_on_publish


def _build_event(event_type: str = "workspace.fs_write") -> AuditEvent:
    """build a valid envelope for the publish-helper tests."""
    return AuditEvent(
        id=uuid7(),
        timestamp=datetime.now(UTC),
        event_type=event_type,
        actor_user_id=uuid7(),
        calling_agent_id=uuid7(),
        owner_agent_id=uuid7(),
        customer_id=uuid7(),
        resource_namespace_id=uuid7(),
        resource_namespace_type="workspace",
        action="write",
        correlation_id=uuid7(),
        details={"bytes": 42},
    )


async def test_publish_posts_to_namespace_dot_audit_dot_event_type() -> None:
    """subject is ``{namespace}.audit.{event_type}`` verbatim."""
    nats = _FakeNats()
    event = _build_event("workspace.fs_write")

    await publish_audit(event, nats_client=nats, namespace="dev")

    assert len(nats.publish_calls) == 1
    subject, _ = nats.publish_calls[0]
    assert subject == "dev.audit.workspace.fs_write"


async def test_publish_emits_json_payload_round_trippable() -> None:
    """payload bytes decode back to the same envelope."""
    nats = _FakeNats()
    event = _build_event("rbac.assignment.create")

    await publish_audit(event, nats_client=nats, namespace="staging")

    _, payload = nats.publish_calls[0]
    decoded = AuditEvent.model_validate_json(payload)
    assert decoded == event


async def test_publish_failure_is_swallowed_not_raised() -> None:
    """publish failure must NOT propagate -- audit is fire-and-forget."""
    nats = _FakeNats(raise_on_publish=RuntimeError("nats down"))
    event = _build_event()

    # must NOT raise; tool-call success never depends on audit health
    await publish_audit(event, nats_client=nats, namespace="dev")


async def test_publish_none_client_is_noop() -> None:
    """``nats_client=None`` is an explicit no-op (bootstrap / tests)."""
    event = _build_event()

    # must NOT raise; no-op
    await publish_audit(event, nats_client=None, namespace="dev")


@pytest.mark.parametrize(
    "event_type",
    [
        "workspace.fs_write",
        "workspace.fs_edit",
        "rbac.group.create",
        "rbac.assignment.delete",
        "rbac.introspect.explain",
        "memory.retrieve",
        "custom.tool.call",
    ],
)
async def test_publish_preserves_dotted_event_type_in_subject(
    event_type: str,
) -> None:
    """every dotted event_type appears verbatim in the subject."""
    nats = _FakeNats()
    event = _build_event(event_type)

    await publish_audit(event, nats_client=nats, namespace="prod")

    subject, _ = nats.publish_calls[0]
    assert subject == f"prod.audit.{event_type}"


async def test_publish_serialization_error_does_not_raise() -> None:
    """a mid-publish serialization exception is caught and logged."""

    class _BrokenNats:
        async def publish(self, subject: str, payload: bytes) -> None:
            del subject, payload
            raise TimeoutError("publish timeout")

    event = _build_event()

    # must NOT raise
    await publish_audit(event, nats_client=_BrokenNats(), namespace="dev")
