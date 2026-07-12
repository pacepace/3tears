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
a consumer-side change and redelivery collapses to a single row via the
idempotency indexes on ``audit_events``.
"""

from __future__ import annotations

from threetears.nats import NatsClient, Subject
from threetears.observe import get_logger

from threetears.agent.audit.envelope import AuditEvent

__all__ = ["publish_audit"]


log = get_logger(__name__)


async def publish_audit(
    event: AuditEvent,
    *,
    nats_client: NatsClient | None,
    namespace: str,
) -> None:
    """
    publish one audit envelope on ``{namespace}.audit.{event_type}``.

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
    :ptype nats_client: NatsClient | None
    :param namespace: NATS subject namespace (environment-scoped
        prefix from ``THREETEARS_NATS_SUBJECT_NAMESPACE``)
    :ptype namespace: str
    :return: nothing
    :rtype: None
    """
    if nats_client is None:
        # bootstrap / test scenario; explicit no-op
        return
    subject = Subject.raw(f"{namespace}.audit.{event.event_type}")
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
