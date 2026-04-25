"""unit tests for :func:`threetears.agent.audit.publish_audit`.

covers subject-naming, payload-shape, and the fire-and-forget
invariant: every publish failure logs at WARN and returns cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import uuid7

import pytest
from pydantic import BaseModel

from threetears.agent.audit import AuditEvent, publish_audit
from threetears.nats import Subject, set_default_namespace


@dataclass
class _FakeWrapper:
    """minimal fake :class:`threetears.nats.NatsClient` recording each publish.

    matches the wrapper's :meth:`publish` shape (kw-only ``subject`` +
    ``message``) so tests exercise the same call surface production
    code uses. payload bytes are derived from the recorded
    :class:`BaseModel` to keep round-trip assertions intact.
    """

    publish_calls: list[tuple[Subject, BaseModel]] = field(default_factory=list)
    raise_on_publish: BaseException | None = None

    async def publish(
        self,
        *,
        subject: Subject,
        message: BaseModel,
        reply_to: Subject | None = None,
    ) -> None:
        """record the publish invocation or raise when configured to."""
        del reply_to  # unused in audit publish
        self.publish_calls.append((subject, message))
        if self.raise_on_publish is not None:
            raise self.raise_on_publish


@pytest.fixture(autouse=True)
def _bind_namespace(request: pytest.FixtureRequest) -> None:
    """bind a known default namespace for each test.

    individual tests override via the ``namespace`` parameter on
    :func:`publish_audit` (passed through for diagnostic logging) and
    by calling :func:`set_default_namespace` directly to control the
    subject that :class:`Subjects.audit_event` produces.
    """
    set_default_namespace("aibots")


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
    set_default_namespace("dev")
    nats = _FakeWrapper()
    event = _build_event("workspace.fs_write")

    await publish_audit(event, nats_client=nats, namespace="dev")

    assert len(nats.publish_calls) == 1
    subject, _ = nats.publish_calls[0]
    assert subject.path == "dev.audit.workspace.fs_write"


async def test_publish_emits_typed_audit_event() -> None:
    """recorded message is the same :class:`AuditEvent` instance round-trippable to JSON."""
    set_default_namespace("staging")
    nats = _FakeWrapper()
    event = _build_event("rbac.assignment.create")

    await publish_audit(event, nats_client=nats, namespace="staging")

    _, message = nats.publish_calls[0]
    assert isinstance(message, AuditEvent)
    assert message == event
    # round-trip the wire form to confirm the typed payload survives serialization
    decoded = AuditEvent.model_validate_json(message.model_dump_json())
    assert decoded == event


async def test_publish_failure_is_swallowed_not_raised() -> None:
    """publish failure must NOT propagate -- audit is fire-and-forget."""
    nats = _FakeWrapper(raise_on_publish=RuntimeError("nats down"))
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
    set_default_namespace("prod")
    nats = _FakeWrapper()
    event = _build_event(event_type)

    await publish_audit(event, nats_client=nats, namespace="prod")

    subject, _ = nats.publish_calls[0]
    assert subject.path == f"prod.audit.{event_type}"


async def test_publish_serialization_error_does_not_raise() -> None:
    """a mid-publish exception is caught and logged."""

    class _BrokenWrapper:
        async def publish(
            self,
            *,
            subject: Subject,
            message: BaseModel,
            reply_to: Subject | None = None,
        ) -> None:
            del subject, message, reply_to
            raise TimeoutError("publish timeout")

    event = _build_event()

    # must NOT raise
    await publish_audit(event, nats_client=_BrokenWrapper(), namespace="dev")
