"""Persist the audit trail where there is no hub to do it: table, consumer, prune and erasure.

The hub persists every ``{ns}.audit.>`` event into its own platform table. A deployment with NO hub --
one application that owns its own control plane -- had to write that persister itself (scriob did, and
dropped ``acting_as_principal_id`` on the way). This is it, once:

- :data:`AUDIT_EVENTS_DDL` / :func:`ensure_audit_events_table`: an ``audit_events`` table carrying every
  :class:`AuditEvent` field and ``ip_address`` for erasure. An EXISTING table is migrated: every column the
  insert names beyond the table's key and its four required fields (``id``, ``timestamp``, ``event_type``,
  ``action``, ``correlation_id``, which no audit table lacks) is added if missing, with the default the
  CREATE gives it, so a deployment that already persisted audit keeps working.
- :func:`persist_audit_event`: an insert idempotent on the envelope ``id``, so an at-least-once redelivery
  is one row. Deliberately NOT on ``(correlation_id, event_type)``: producers stamp every event of a
  request with the request's correlation id, and two writes of one type in one request are two records
  an audit trail must keep.
- :func:`start_audit_persister`: ensures the ``audit`` stream with its sibling dead-letter subject and runs
  a shared durable PULL consumer, so every replica may run it and each event is persisted once. The
  stream's storage must match every other declarer of that stream name in the deployment (a mismatch
  crashes the second declarer); it defaults to memory, as the NATS client does. The durable name must be
  unique per table: two apps sharing a namespace and a durable would split the events between them. A malformed event is acked and dropped; a database fault raises, so
  the consumer retries and finally dead-letters it rather than losing the record.
- :func:`prune_audit_events`: an age-based retention delete, in batches.
- :func:`anonymize_audit_rows`: erasure under THE platform rule and no other. Every row and every id
  survives; only ``details`` (through :func:`~threetears.agent.audit.anonymize_details`, under each row's
  own event type) and ``ip_address`` (through :func:`~threetears.agent.audit.anonymize_ip`) change. It
  takes no replacement for either: a deployment that could substitute its own scrub is the per-consumer
  divergence the one rule exists to end. It is this deployment's own erasure -- it never answers the
  hub's ``hub.audit.anonymize`` subject, which is for the hub's table -- and answers with the
  :class:`~threetears.agent.audit.AuditAnonymization` the hub path answers with.

The persister writes Postgres only (no cache tiers), so erasure has no cache to evict.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol
from uuid import UUID

from pydantic import ValidationError
from threetears.nats import Subjects
from threetears.observe import get_logger, spawn_background
from threetears.observe.erasure import ANONYMIZED_MARKER

from threetears.agent.audit.anonymize import anonymize_details, anonymize_ip
from threetears.agent.audit.envelope import AuditEvent
from threetears.agent.audit.erasure import AuditAnonymization

__all__ = [
    "AUDIT_EVENTS_DDL",
    "AUDIT_MAX_DELIVER",
    "AUDIT_STREAM_NAME",
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
    # an existing table (created before a column existed) gains every column the insert names beyond the
    # key and the four required fields, with the CREATE's own type and default
    *(
        f"ALTER TABLE audit_events ADD COLUMN IF NOT EXISTS {column}"
        for column in (
            "outcome TEXT NOT NULL DEFAULT 'success'",
            "actor_user_id UUID",
            "acting_as_principal_id UUID",
            "calling_agent_id UUID",
            "owner_agent_id UUID",
            "customer_id UUID",
            "resource_namespace_id UUID",
            "resource_namespace_type TEXT",
            "conversation_id UUID",
            "details JSONB NOT NULL DEFAULT '{}'",
            "ip_address TEXT",
        )
    ),
    "CREATE INDEX IF NOT EXISTS idx_audit_events_time ON audit_events (timestamp)",
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
    "ON CONFLICT (id) DO NOTHING"
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
    :param ip_address: the caller's address; the envelope carries none, so only a direct caller that has
        one passes it (the consumer path stores NULL)
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
        # JSON-mode dump: details built in-process may hold UUIDs and datetimes; ensure_ascii off so text
        # is stored as written
        json.dumps(event.model_dump(mode="json")["details"], ensure_ascii=False),
        ip_address,
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
        # it will never parse: dropping it beats redelivering it forever. the subject and stream sequence
        # locate the dropped event in the stream; the exception's text is left out because it can echo the
        # event's personal content.
        metadata = getattr(msg, "metadata", None)
        log.warning(
            "audit persister: dropping an undecodable event",
            extra={
                "extra_data": {
                    "error": type(exc).__name__,
                    "subject": getattr(msg, "subject", None),
                    "stream_sequence": getattr(getattr(metadata, "sequence", None), "stream", None),
                }
            },
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
        try:
            await self.consumer.stop()
        finally:
            # the task ends even when stopping the consumer fails (a connection already closed at shutdown)
            self.task.cancel()
            # NOSILENT: this IS the cancellation requested on the line above
            with contextlib.suppress(asyncio.CancelledError):
                await self.task


async def start_audit_persister(
    nats_client: Any,  # noqa: ANN401 -- the connected NatsClient
    db: AuditStore,
    *,
    durable: str,
    storage: str = "memory",
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
    :param storage: the stream's storage (``memory`` or ``file``); must match every other declarer of the
        ``audit`` stream in this deployment
    :ptype storage: str
    :param max_deliver: attempts before an event is dead-lettered
    :ptype max_deliver: int
    :return: the running persister
    :rtype: AuditPersisterHandle
    """
    stream = await nats_client.ensure_jetstream_stream(
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
        # named, so binding needs no stream-names lookup (not granted to least-privilege accounts)
        stream=stream,
    )
    task = spawn_background(consumer.run(), name=f"audit-persister:{durable}", logger=log)
    log.info("audit persister started", extra={"extra_data": {"durable": durable, "storage": storage}})
    return AuditPersisterHandle(consumer, task)


async def prune_audit_events(
    db: AuditStore, *, older_than: timedelta, now: datetime | None = None, batch_size: int = 5000
) -> int:
    """delete events older than the retention window, ``batch_size`` rows at a time.

    :param db: the deployment's database
    :ptype db: AuditStore
    :param older_than: the retention window
    :ptype older_than: timedelta
    :param now: the reference instant (default: now)
    :ptype now: datetime | None
    :param batch_size: rows per delete, so a large backlog never holds one long lock
    :ptype batch_size: int
    :return: rows deleted
    :rtype: int
    :raises ValueError: ``batch_size`` below one
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    cutoff = (now if now is not None else datetime.now(UTC)) - older_than
    deleted = 0
    while True:
        status = await db.execute(
            "DELETE FROM audit_events WHERE id IN (SELECT id FROM audit_events WHERE timestamp < $1 LIMIT $2)",
            cutoff,
            batch_size,
        )
        count = _rowcount(status)
        deleted += count
        if count < batch_size:
            break
    return deleted


def _rowcount(status: Any) -> int:  # noqa: ANN401 -- a driver status string
    """the row count from a command status such as ``"DELETE 3"`` or ``"UPDATE 1"``.

    :param status: the status
    :ptype status: Any
    :return: the count (0 when there is none)
    :rtype: int
    """
    tail = str(status).rsplit(" ", 1)[-1]
    return int(tail) if tail.isdigit() else 0


def _anonymize_stored(raw: str | None, *, event_type: str) -> str | None:
    """rewrite one stored ``details`` value by the platform rule, keeping the shape it was stored in.

    an object is judged key by key; a JSON string holding an object is judged as that object and
    written back string-held; SQL NULL and JSON null stay; anything else becomes the marker.

    :param raw: ``details::text``
    :ptype raw: str | None
    :param event_type: the row's own event type
    :ptype event_type: str
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
        result = anonymize_details(value, event_type=event_type)
    elif isinstance(value, str):
        try:
            inner = json.loads(value)
        except ValueError:
            inner = None
        result = (
            json.dumps(anonymize_details(inner, event_type=event_type))
            if isinstance(inner, Mapping)
            else ANONYMIZED_MARKER
        )
    else:
        result = ANONYMIZED_MARKER
    return json.dumps(result, ensure_ascii=False)


async def anonymize_audit_rows(
    db: AuditStore,
    *,
    actor_user_ids: Iterable[UUID],
    batch_size: int = 500,
) -> AuditAnonymization:
    """erase the personal data in every event naming these actors, by the platform rule; idempotent.

    every row and every id survives (row id, actor, entity ids in details, event type, action, outcome,
    correlation ids, timestamps); only ``details`` (through ``anonymize_details``, under the row's own
    event type) and ``ip_address`` (through ``anonymize_ip``) change. rows are read in keyset batches of
    ``batch_size``. a family whose details keys are safe for this deployment declares them with
    ``declare_safe_detail_keys``; there is no other way to change what the rule keeps.

    :param db: the deployment's database
    :ptype db: AuditStore
    :param actor_user_ids: the actors whose events to erase
    :ptype actor_user_ids: Iterable[UUID]
    :param batch_size: rows per batch
    :ptype batch_size: int
    :return: rows matched and rows changed
    :rtype: AuditAnonymization
    :raises ValueError: ``batch_size`` below one
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    actors = list(actor_user_ids)
    if not actors:
        return AuditAnonymization(rows_matched=0, rows_changed=0)
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
            details = _anonymize_stored(row["details"], event_type=row["event_type"])
            address = anonymize_ip(row["ip_address"])
            # the DATABASE decides "changed": jsonb renders non-ASCII and numbers its own way, so comparing
            # JSON text counted untouched rows as changed on every pass
            status = await db.execute(
                "UPDATE audit_events SET details = $1::text::jsonb, ip_address = $2 WHERE id = $3 "
                "AND (details IS DISTINCT FROM $1::text::jsonb OR ip_address IS DISTINCT FROM $2)",
                details,
                address,
                row["id"],
            )
            changed += _rowcount(status)
        after = rows[-1]["id"]
    log.info("audit rows anonymized", extra={"extra_data": {"rows_matched": matched, "rows_changed": changed}})
    return AuditAnonymization(rows_matched=matched, rows_changed=changed)
