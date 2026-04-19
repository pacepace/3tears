"""fire-and-forget audit publish helper.

:func:`publish_audit` is the single publish path for every domain.
serializes the :class:`AuditEvent` via pydantic, awaits one
``nats_client.publish`` on ``{namespace}.audit.{event_type}``, and
swallows every exception at WARN. tool-call success must never depend
on audit infrastructure availability; this invariant is load-bearing.

the hub-side ``unified_audit_consumer`` subscribes to
``{namespace}.audit.>`` so new event types route automatically without
a consumer-side change.
"""

from __future__ import annotations

from typing import Any

from threetears.observe import get_logger

from threetears.agent.audit.envelope import AuditEvent

__all__ = ["publish_audit"]


log = get_logger(__name__)


async def publish_audit(
    event: AuditEvent,
    *,
    nats_client: Any,
    namespace: str,
) -> None:
    """
    publish one audit envelope on ``{namespace}.audit.{event_type}``.

    fire-and-forget: any exception during publish is caught and logged
    at WARN; no exception propagates to the caller. when
    ``nats_client`` is ``None`` the call is an explicit no-op (useful
    in tests and bootstrap windows before NATS wiring is complete).

    :param event: typed audit envelope to publish
    :ptype event: AuditEvent
    :param nats_client: connected NATS client exposing
        ``publish(subject, payload)``; ``None`` is a no-op
    :ptype nats_client: Any
    :param namespace: NATS subject namespace (environment-scoped
        prefix from ``FOURTEENAIBOTS_NATS_SUBJECT_NAMESPACE``)
    :ptype namespace: str
    :return: nothing
    :rtype: None
    """
    if nats_client is None:
        # bootstrap / test scenario; explicit no-op
        return
    subject = f"{namespace}.audit.{event.event_type}"
    try:
        payload = event.model_dump_json().encode("utf-8")
        await nats_client.publish(subject, payload)
    # NOSILENT: audit publish is fire-and-forget; failures log at WARN
    # so the producing call path is never blocked by audit health.
    except Exception as exc:
        log.warning(
            "audit publish failed",
            extra={
                "extra_data": {
                    "subject": subject,
                    "event_type": event.event_type,
                    "error": str(exc),
                },
            },
        )
