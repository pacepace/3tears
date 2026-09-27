"""Persist the audit trail where there is no hub to do it: table, consumer, prune and erasure.

The hub persists every ``{ns}.audit.>`` event into its own platform table. A deployment with NO hub --
one application that owns its own control plane -- had to write that persister itself (scriob did, and
dropped ``acting_as_principal_id`` on the way). This is it, once:

- :data:`AUDIT_EVENTS_DDL` / :func:`ensure_audit_events_table`: an ``audit_events`` table carrying every
  :class:`AuditEvent` field, ``ip_address`` for erasure, and both idempotency anchors the envelope
  documents -- the ``id`` primary key and the unique ``(correlation_id, event_type)`` index.
- :func:`persist_audit_event`: an insert that collapses an at-least-once redelivery, and the same logical
  event re-emitted under a new envelope id, to one row.
- :func:`start_audit_persister`: ensures the ``audit`` stream (file-backed by default, matching the
  platform's other declarers -- a storage mismatch on one stream name crashes the second declarer) with
  its sibling dead-letter subject, and runs a shared durable PULL consumer, so every replica may run it
  and each event is persisted once. A malformed event is acked and dropped; a database fault raises, so
  the consumer retries and finally dead-letters it rather than losing the record.
- :func:`prune_audit_events`: an age-based retention delete.
- :func:`anonymize_audit_rows`: erasure under the platform's rule. Every row and every id survives; only
  ``details`` (through ``anonymize_details``, under each row's own event type) and ``ip_address`` (through
  ``anonymize_ip``) change. It is this deployment's own erasure -- it never answers the hub's
  ``hub.audit.anonymize`` subject, which is for the hub's table.

The persister writes Postgres only (no cache tiers), so erasure has no cache to evict.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol
from uuid import UUID

from pydantic import ValidationError
from threetears.nats import Subjects
from threetears.observe import get_logger, spawn_background

from threetears.agent.audit.envelope import AuditEvent

__all__ = [
    "AUDIT_EVENTS_DDL",
    "AUDIT_MAX_DELIVER",
    "AUDIT_STREAM_NAME",
    "AuditAnonymizationResult",
    "AuditPersisterHandle",
    "AuditStore",
    "anonymize_audit_rows",
    "ensure_audit_events_table",
    "handle_audit_message",
    "persist_audit_event",
    "prune_audit_events",
    "start_audit_persister",
]

log = get_logger(__name__)

#: JetStream stream suffix (namespace-prefixed to ``{ns}-audit``) holding every audit subject.
AUDIT_STREAM_NAME = "audit"
#: delivery attempts before an event that cannot be persisted is dead-lettered.
AUDIT_MAX_DELIVER = 5

#: the table and its indexes; idempotent (``IF NOT EXISTS``).
AUDIT_EVENTS_DDL: tuple[str, ...] = (
    "CREATE TABLE IF NOT EXISTS audit_events ("
    "id UUID PRIMARY KEY, "
    "timestamp TIMESTAMPTZ NOT NULL, "
    "event_type TEXT NOT NULL, "
    "action TEXT NOT NULL, "
    "outcome TEXT NOT NULL DEFAULT 'success', "
    "actor_user_id UUID, "
    "acting_as_principal_id UUID, "
    "calling_agent_id UUID, "
    "owner_agent_id UUID, "
    "customer_id UUID, "
    "resource_namespace_id UUID, "
    "resource_namespace_type TEXT, "
    "correlation_id UUID NOT NULL, "
    "conversation_id UUID, "
    "details JSONB NOT NULL DEFAULT '{}', "
    "ip_address TEXT)",
    # the SECONDARY idempotency anchor (AuditEvent.correlation_id): one logical event re-emitted under a
    # new envelope id is one row
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_audit_events_correlation_event ON audit_events (correlation_id, event_type)",
    "CREATE INDEX IF NOT EXISTS idx_audit_events_customer_time ON audit_events (customer_id, timestamp DESC)",
    "CREATE INDEX IF NOT EXISTS idx_audit_events_type ON audit_events (event_type)",
    "CREATE INDEX IF NOT EXISTS idx_audit_events_actor ON audit_events (actor_user_id)",
)

_INSERT = (
    "INSERT INTO audit_events ("
    "id, timestamp, event_type, action, outcome, actor_user_id, acting_as_principal_id, calling_agent_id, "
    "owner_agent_id, customer_id, resource_namespace_id, resource_namespace_type, correlation_id, "
    "conversation_id, details, ip_address"
    ") VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15::text::jsonb, $16) "
    # no conflict target: the id key AND the (correlation_id, event_type) index both absorb a replay
    "ON CONFLICT DO NOTHING"
)


class AuditStore(Protocol):
    """The two operations the persister makes: an asyncpg pool (or connection) has both."""

    async def execute(self, sql: str, /, *args: Any) -> Any:  # noqa: ANN401
        """run a statement.

        :param sql: the statement
        :ptype sql: str
        :param args: positional parameters
        :ptype args: Any
        :return: the status
        :rtype: Any
        """
        ...

    async def fetch(self, sql: str, /, *args: Any) -> Sequence[Any]:  # noqa: ANN401
        """run a query.

        :param sql: the query
        :ptype sql: str
        :param args: positional parameters
        :ptype args: Any
        :return: the rows
        :rtype: Sequence[Any]
        """
        ...


async def ensure_audit_events_table(db: AuditStore) -> None:
    """create the ``audit_events`` table and its indexes if absent.

    :param db: the deployment's database
    :ptype db: AuditStore
    :return: None
    :rtype: None
    """
    for statement in AUDIT_EVENTS_DDL:
        await db.execute(statement)


async def persist_audit_event(db: AuditStore, event: AuditEvent, *, ip_address: str | None = None) -> None:
    """insert one event; a redelivery, or a re-emission under a new id, is a no-op.

    details are written as ``$n::text::jsonb`` from ``json.dumps``, so the value is parsed once whatever
    JSONB codec the pool carries.

    :param db: the deployment's database
    :ptype db: AuditStore
    :param event: the envelope
    :ptype event: AuditEvent
    :param ip_address: the caller's address, when the producer recorded one
    :ptype ip_address: str | None
    :return: None
    :rtype: None
    """
    await db.execute(
        _INSERT,
        event.id,
        event.timestamp,
        event.event_type,
        event.action,
        event.outcome,
        event.actor_user_id,
        event.acting_as_principal_id,
        event.calling_agent_id,
        event.owner_agent_id,
        event.customer_id,
        event.resource_namespace_id,
        event.resource_namespace_type,
        event.correlation_id,
        event.conversation_id,
        json.dumps(event.details),
        ip_address if ip_address is not None else getattr(event, "ip_address", None),
    )


async def handle_audit_message(db: AuditStore, msg: Any) -> None:  # noqa: ANN401 -- a JetStream message
    """parse, persist, ack. A malformed event is acked and dropped; a database fault raises.

    :param db: the deployment's database
    :ptype db: AuditStore
    :param msg: the JetStream message (``data``, ``ack``)
    :ptype msg: Any
    :return: None
    :rtype: None
    """
    try:
        event = AuditEvent.model_validate_json(bytes(msg.data))
    except (ValidationError, ValueError) as exc:
        # it will never parse: dropping it beats redelivering it forever
        log.warning(
            "audit persister: dropping an undecodable event", extra={"extra_data": {"error": type(exc).__name__}}
        )
        await msg.ack()
        return
    await persist_audit_event(db, event)  # a database fault raises: the consumer retries, then dead-letters
    await msg.ack()


@dataclass
class AuditPersisterHandle:
    """a running persister: its consumer and the task driving it."""

    consumer: Any
    task: asyncio.Task[Any]

    async def stop(self) -> None:
        """stop fetching, then end the task.

        :return: None
        :rtype: None
        """
        await self.consumer.stop()
        self.task.cancel()
        # NOSILENT: this IS the cancellation requested on the line above
        with contextlib.suppress(asyncio.CancelledError):
            await self.task


async def start_audit_persister(
    nats_client: Any,  # noqa: ANN401 -- the connected NatsClient
    db: AuditStore,
    *,
    durable: str,
    storage: str = "file",
    max_deliver: int = AUDIT_MAX_DELIVER,
) -> AuditPersisterHandle:
    """ensure the audit stream and run a shared durable pull consumer persisting every event.

    safe on every replica: a pull consumer hands each pending event to exactly one fetcher.

    :param nats_client: the connected NATS client
    :ptype nats_client: NatsClient
    :param db: the deployment's database (the table must exist: :func:`ensure_audit_events_table`)
    :ptype db: AuditStore
    :param durable: the consumer's durable name, stable across restarts and shared by replicas
    :ptype durable: str
    :param storage: the stream's storage; ``file`` matches the platform's other declarers
    :ptype storage: str
    :param max_deliver: attempts before an event is dead-lettered
    :ptype max_deliver: int
    :return: the running persister
    :rtype: AuditPersisterHandle
    """
    await nats_client.ensure_jetstream_stream(
        name=AUDIT_STREAM_NAME,
        subjects=[Subjects.audit_wildcard().path, Subjects.audit_deadletter().path],
        storage=storage,
    )

    async def _on_message(msg: Any) -> None:  # noqa: ANN401
        await handle_audit_message(db, msg)

    # the dead-letter subject is a SIBLING token ({ns}.audit-deadletter), so the wildcard never
    # re-consumes what it parked: an event that cannot be persisted is kept, not dropped
    consumer = await nats_client.jetstream_pull_subscribe(
        subject=Subjects.audit_wildcard(),
        durable=durable,
        cb=_on_message,
        max_deliver=max_deliver,
        dead_letter_subject=Subjects.audit_deadletter(),
    )
    task = spawn_background(consumer.run(), name=f"audit-persister:{durable}", logger=log)
    log.info("audit persister started", extra={"extra_data": {"durable": durable, "storage": storage}})
    return AuditPersisterHandle(consumer, task)


async def prune_audit_events(db: AuditStore, *, older_than: timedelta, now: datetime | None = None) -> int:
    """delete events older than the retention window.

    :param db: the deployment's database
    :ptype db: AuditStore
    :param older_than: the retention window
    :ptype older_than: timedelta
    :param now: the reference instant (default: now)
    :ptype now: datetime | None
    :return: rows deleted
    :rtype: int
    """
    cutoff = (now if now is not None else datetime.now(UTC)) - older_than
    status = await db.execute("DELETE FROM audit_events WHERE timestamp < $1", cutoff)
    return int(str(status).rsplit(" ", 1)[-1]) if str(status).startswith("DELETE") else 0


@dataclass(frozen=True)
class AuditAnonymizationResult:
    """what an erasure touched.

    :ivar rows_matched: rows naming one of the actors
    :ivar rows_changed: rows whose details or address actually changed (a second pass changes none)
    """

    rows_matched: int
    rows_changed: int


def _erasure() -> tuple[Callable[..., dict[str, Any]], Callable[[str | None], str | None], str]:
    """the platform's erasure rule: ``anonymize_details``, ``anonymize_ip`` and the marker.

    imported late: they live in ``threetears.agent.audit.anonymize`` and ``threetears.observe.erasure``.

    :return: the details anonymizer, the address anonymizer, the marker
    :rtype: tuple[Callable[..., dict[str, Any]], Callable[[str | None], str | None], str]
    """
    anonymize = importlib.import_module("threetears.agent.audit.anonymize")
    erasure = importlib.import_module("threetears.observe.erasure")
    return anonymize.anonymize_details, anonymize.anonymize_ip, erasure.ANONYMIZED_MARKER


def _anonymize_stored(
    raw: str | None, *, event_type: str, anonymize: Callable[..., dict[str, Any]], marker: str
) -> str | None:
    """rewrite one stored ``details`` value, keeping the shape it was stored in.

    an object is judged key by key; a JSON string holding an object is judged as that object and
    written back string-held; SQL NULL and JSON null stay; anything else becomes the marker.

    :param raw: ``details::text``
    :ptype raw: str | None
    :param event_type: the row's own event type
    :ptype event_type: str
    :param anonymize: the details anonymizer
    :ptype anonymize: Callable[..., dict[str, Any]]
    :param marker: the anonymized marker
    :ptype marker: str
    :return: the new ``details`` as JSON text, or ``None`` for SQL NULL
    :rtype: str | None
    """
    if raw is None:
        return None
    value = json.loads(raw)
    result: Any
    if value is None:
        result = None
    elif isinstance(value, Mapping):
        result = anonymize(value, event_type=event_type)
    elif isinstance(value, str):
        try:
            inner = json.loads(value)
        except ValueError:
            inner = None
        result = json.dumps(anonymize(inner, event_type=event_type)) if isinstance(inner, Mapping) else marker
    else:
        result = marker
    return json.dumps(result)


async def anonymize_audit_rows(
    db: AuditStore,
    *,
    actor_user_ids: Iterable[UUID],
    batch_size: int = 500,
    anonymize: Callable[..., dict[str, Any]] | None = None,
    anonymize_ip: Callable[[str | None], str | None] | None = None,
    marker: str | None = None,
) -> AuditAnonymizationResult:
    """erase the personal data in every event naming these actors; idempotent.

    every row and every id survives (row id, actor, entity ids in details, event type, action, outcome,
    correlation ids, timestamps); only ``details`` and ``ip_address`` change. rows are read in keyset
    batches of ``batch_size``. the anonymizers default to the platform's (``anonymize_details``,
    ``anonymize_ip``, ``ANONYMIZED_MARKER``); pass them to override.

    :param db: the deployment's database
    :ptype db: AuditStore
    :param actor_user_ids: the actors whose events to erase
    :ptype actor_user_ids: Iterable[UUID]
    :param batch_size: rows per batch
    :ptype batch_size: int
    :param anonymize: details anonymizer ``(details, *, event_type) -> dict``
    :ptype anonymize: Callable[..., dict[str, Any]] | None
    :param anonymize_ip: address anonymizer
    :ptype anonymize_ip: Callable[[str | None], str | None] | None
    :param marker: the value an unreadable ``details`` becomes
    :ptype marker: str | None
    :return: rows matched and rows changed
    :rtype: AuditAnonymizationResult
    """
    actors = list(actor_user_ids)
    if not actors:
        return AuditAnonymizationResult(0, 0)
    if anonymize is None or anonymize_ip is None or marker is None:
        default_details, default_ip, default_marker = _erasure()
        anonymize = anonymize or default_details
        anonymize_ip = anonymize_ip or default_ip
        marker = marker if marker is not None else default_marker
    matched = changed = 0
    after: UUID | None = None
    while True:
        rows = await db.fetch(
            "SELECT id, event_type, details::text AS details, ip_address FROM audit_events "
            "WHERE actor_user_id = ANY($1::uuid[]) AND ($2::uuid IS NULL OR id > $2) ORDER BY id LIMIT $3",
            actors,
            after,
            batch_size,
        )
        if not rows:
            break
        for row in rows:
            matched += 1
            details = _anonymize_stored(
                row["details"], event_type=row["event_type"], anonymize=anonymize, marker=marker
            )
            address = anonymize_ip(row["ip_address"])
            if details != row["details"] or address != row["ip_address"]:
                await db.execute(
                    "UPDATE audit_events SET details = $1::text::jsonb, ip_address = $2 WHERE id = $3",
                    details,
                    address,
                    row["id"],
                )
                changed += 1
        after = rows[-1]["id"]
    log.info("audit rows anonymized", extra={"extra_data": {"rows_matched": matched, "rows_changed": changed}})
    return AuditAnonymizationResult(matched, changed)
