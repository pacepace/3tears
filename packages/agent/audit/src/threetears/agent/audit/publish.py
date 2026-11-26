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

a failure is never only logged, though: :func:`publish_audit` answers
:attr:`AuditPublishOutcome.FAILED` to its caller, and every failure in the
process is counted -- :func:`audit_publish_failure_count`, for a server's own
status, and the Prometheus counter ``threetears_audit_publish_failures_total``
(by event type) where ``prometheus_client`` is installed.

the hub-side ``unified_audit_consumer`` binds a durable push consumer on
``{namespace}.audit.>`` (manual ack after the L3 write, bounded
redelivery, dead-letter) so new event types route automatically without
a consumer-side change and redelivery collapses to a single row: a
redelivered envelope repeats its ``id``, which is the row's primary key.
"""

from __future__ import annotations

import threading
from enum import StrEnum
from typing import TYPE_CHECKING
from uuid import UUID

from threetears.nats import Subjects
from threetears.observe import contain, counter, get_logger

from threetears.agent.audit.anonymize import refuse_identifying_details
from threetears.agent.audit.envelope import AuditEvent

if TYPE_CHECKING:
    # From the submodule, not the package: these are Protocols that `threetears.nats`
    # stopped re-exporting when its nats-py-backed surface went lazy. Annotation-only,
    # so the eager `kv` import here costs an L1 consumer nothing.
    from threetears.nats.kv import JetStreamPublisher

__all__ = ["AuditPublishOutcome", "audit_publish_failure_count", "publish_audit"]


log = get_logger(__name__)


class AuditPublishOutcome(StrEnum):
    """what one :func:`publish_audit` call did with its envelope.

    :cvar PUBLISHED: the broker acknowledged it; it is durable in the audit stream
    :cvar SKIPPED: no NATS client was given (a test, a bootstrap window): nothing was sent, by the caller's choice
    :cvar FAILED: the publish failed; the envelope is not in the audit stream, and the failure is counted
    """

    PUBLISHED = "published"
    SKIPPED = "skipped"
    FAILED = "failed"


#: the Prometheus counter of failed publishes, by event type; a no-op where prometheus_client is not installed
_FAILURES_METRIC = counter(
    "threetears_audit_publish_failures_total",
    description="audit envelopes whose publish failed: not in the audit stream",
    label_names=("event_type",),
)

#: failed publishes in this process, and the lock that makes each count whole across threads
_failures = 0
_failures_lock = threading.Lock()


def audit_publish_failure_count() -> int:
    """how many audit publishes have failed in this process, for a server's own status.

    :return: the count since the process started
    :rtype: int
    """
    with _failures_lock:
        return _failures


def _count_failure(event_type: str) -> None:
    """count one failed publish.

    :param event_type: the envelope's event type
    :ptype event_type: str
    :return: nothing
    :rtype: None
    """
    global _failures
    with _failures_lock:
        _failures += 1
    _FAILURES_METRIC.labels(event_type=event_type).inc()


async def publish_audit(
    event: AuditEvent,
    *,
    nats_client: JetStreamPublisher | None,
    namespace: str,
    tool_pod_id: UUID | None = None,
) -> AuditPublishOutcome:
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
    WARN, counted (:func:`audit_publish_failure_count`), and answered as
    :attr:`AuditPublishOutcome.FAILED`; no exception propagates to the
    caller. when ``nats_client`` is ``None`` the call is an explicit no-op
    answering :attr:`AuditPublishOutcome.SKIPPED` (useful in tests and
    bootstrap windows before NATS wiring is complete).

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
    :return: what was done with the envelope; ``FAILED`` is also counted
    :rtype: AuditPublishOutcome
    :raises IdentifyingAuditDetail: when ``event.details`` hold a key that names a person, at any depth (an audit
        row holds ids: refused before anything is sent, whether NATS is wired or not)
    """
    refuse_identifying_details(event.details, event_type=event.event_type)
    if nats_client is None:
        # bootstrap / test scenario; explicit no-op
        return AuditPublishOutcome.SKIPPED
    subject = (
        Subjects.audit_event(event.event_type, namespace=namespace)
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
        contain(
            exc,
            log=log,
            what="audit publish failed",
            extra={"subject": subject.path, "event_type": event.event_type, "namespace": namespace, "error": str(exc)},
        )
        _count_failure(event.event_type)
        return AuditPublishOutcome.FAILED
    return AuditPublishOutcome.PUBLISHED
