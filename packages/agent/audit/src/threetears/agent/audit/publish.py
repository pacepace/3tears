"""durable-but-fire-and-forget audit publish helper.

:func:`publish_audit` is the single publish path for every domain.
serializes the :class:`AuditEvent` via pydantic and awaits one
:meth:`NatsClient.jetstream_publish` on ``{namespace}.audit.{event_type}``
-- a JetStream publish that PERSISTS the envelope to the durable
``{ns}-audit`` stream and awaits the broker ``PubAck``. an envelope
published while the hub consumer is restarting is retained and
redelivered when it reconnects (at-least-once), so the audit trail is
never silently dropped on a transient consumer outage.

it stays fire-and-forget at the PRODUCER: any exception (a JetStream
outage, no stream, a broker timeout) is caught and logged at WARN; no
exception propagates to the caller. tool-call success must never depend
on audit infrastructure availability; this invariant is load-bearing.
the common path is now durable, but a producer-side JetStream stall can
never block the tool call.

the hub-side ``unified_audit_consumer`` binds a durable push consumer on
``{namespace}.audit.>`` (manual ack after the L3 write, bounded
redelivery, dead-letter) so new event types route automatically without
a consumer-side change and redelivery collapses to a single row: a
redelivered envelope repeats its ``id``, which is the row's primary key.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from threetears.nats import Subject, Subjects
from threetears.observe import get_logger

from threetears.agent.audit.envelope import AuditEvent

if TYPE_CHECKING:
    # From the submodule, not the package: these are Protocols that `threetears.nats`
    # stopped re-exporting when its nats-py-backed surface went lazy. Annotation-only,
    # so the eager `kv` import here costs an L1 consumer nothing.
    from threetears.nats.kv import JetStreamPublisher

__all__ = ["publish_audit"]


log = get_logger(__name__)


async def publish_audit(
    event: AuditEvent,
    *,
    nats_client: JetStreamPublisher | None,
    namespace: str,
    tool_pod_id: UUID | None = None,
) -> None:
    """
    publish one audit envelope on ``{namespace}.audit.{event_type}``.

    a TOOL POD passes its ``tool_pods.id`` as ``tool_pod_id`` and the envelope rides
    :meth:`threetears.nats.Subjects.tool_pod_audit_event` instead --
    ``{namespace}.audit.tool_pod.{tool_pod_id}.{event_type}``, the one audit subtree a tool pod is
    granted for its own events. The event type and the envelope are unchanged; the hub's collector
    reads the actor off that subject.

    durable transport: the envelope is JetStream-published (persisted to
    the ``{ns}-audit`` stream + ``PubAck`` awaited), so it survives a
    consumer restart and is redelivered at-least-once. fire-and-forget at
    the producer: any exception during publish is caught and logged at
    WARN; no exception propagates to the caller. when ``nats_client`` is
    ``None`` the call is an explicit no-op (useful in tests and bootstrap
    windows before NATS wiring is complete).

    the subject is built with the explicit ``namespace`` argument
    rather than reading the :class:`Subjects` ContextVar so callers
    that route audit traffic on a per-call namespace (multi-tenant
    test fixtures, in-process audit consumers under explicit prefix
    control) get the namespace they passed regardless of which
    ContextVar value happens to be bound on the calling task.
    ``event.event_type`` carries dots verbatim (e.g.
    ``workspace.fs_write``); they form the subject hierarchy and are
    NOT sanitized.

    :param event: typed audit envelope to publish
    :ptype event: AuditEvent
    :param nats_client: connected canonical
        :class:`threetears.nats.NatsClient` wrapper; ``None`` is a no-op
    :ptype nats_client: JetStreamPublisher | None
    :param namespace: NATS subject namespace (environment-scoped
        prefix from ``THREETEARS_NATS_SUBJECT_NAMESPACE``)
    :ptype namespace: str
    :param tool_pod_id: the publishing tool pod's ``tool_pods.id``, or ``None`` for every
        principal that is not a tool pod
    :ptype tool_pod_id: UUID | None
    :return: nothing
    :rtype: None
    """
    if nats_client is None:
        # bootstrap / test scenario; explicit no-op
        return
    subject = (
        Subject.raw(f"{namespace}.audit.{event.event_type}")
        if tool_pod_id is None
        else Subjects.tool_pod_audit_event(tool_pod_id, event.event_type, namespace=namespace)
    )
    try:
        # serialize at the border and JetStream-publish for durability:
        # the envelope is persisted to the ``{ns}-audit`` stream and the
        # broker ``PubAck`` is awaited, so a consumer restart does not drop it.
        await nats_client.jetstream_publish(
            subject=subject,
            payload=event.model_dump_json().encode(),
        )
    # NOSILENT: audit publish is fire-and-forget at the producer; failures
    # (JetStream outage, no stream, broker timeout) log at WARN so the
    # producing call path is never blocked by audit health.
    except Exception as exc:
        log.warning(
            "audit publish failed",
            extra={
                "extra_data": {
                    "subject": subject.path,
                    "event_type": event.event_type,
                    "namespace": namespace,
                    "error": str(exc),
                },
            },
        )
