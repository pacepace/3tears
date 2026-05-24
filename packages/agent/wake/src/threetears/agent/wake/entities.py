"""Agent-wake entities -- cache proxies for the three wake tables.

Three entity classes:

- :class:`WakeScheduleEntity` -- one row per active wake schedule for a
  conversation. Composite PK ``(conversation_id, schedule_id)``;
  standalone ``UNIQUE (schedule_id)`` so cross-package FKs can address
  the bare id (notably ``wake_fires.schedule_id``).
- :class:`WakeFireEntity` -- one row per wake fire. Composite PK
  ``(conversation_id, fire_id)``. Immutable post-finalize: ``status``
  moves once from ``'fired'`` / ``'fired_silent'`` / ``'yielded'`` /
  ``'skipped_*'`` / ``'failed'``.
- :class:`WebhookSubscriptionEntity` -- one row per inbound webhook.
  Composite PK ``(conversation_id, subscription_id)``; standalone
  ``UNIQUE (subscription_id)``. ``secret_ciphertext`` is bytes
  (Fernet-encrypted) and is NEVER decoded by the entity; the consumer
  supplies an encryption service that opens it on demand.

Each entity carries a tuple ``_id`` so ``BaseCollection.normalize_pk``
and ``BaseCollection.l2_key`` address the row uniformly across L1 / L2
/ L3. ``primary_key_field`` names the bare-id column so
``BaseEntity.id`` returns the singular UUID downstream callers need
(e.g. the wake-fires ``schedule_id`` FK target, tool input args).

Partition column for every wake table is ``conversation_id`` per
PLACEMENT §1.3 / §1.7 / §1.9 -- wake operations are
conversation-scoped, not agent-scoped. There is NO database-level FK
on ``conversation_id`` -> ``conversations(conversation_id)``: the
3tears ``conversations`` table has composite PK ``(agent_id,
conversation_id)`` and no standalone ``UNIQUE (conversation_id)``, so
a single-column FK is not legal. Same precedent as
``context_items.conversation_id`` (agent-tools v003) and
``agent_skill_invocations.conversation_id`` (agent-skills v002).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol, runtime_checkable
from uuid import UUID

from threetears.core.entities.base import BaseEntity

__all__ = [
    "EncryptionService",
    "WakeFireEntity",
    "WakeScheduleEntity",
    "WebhookSubscriptionEntity",
]


@runtime_checkable
class EncryptionService(Protocol):
    """Minimal Protocol for the encryption service the consumer supplies.

    The 3tears platform does NOT ship a canonical encryption service
    (key management is consumer-specific: metallm has
    ``src.services.encryption`` wrapping Fernet, aibots will have its
    own, etc.). The Protocol declares the two surfaces the agent-tools
    layer uses: :meth:`encrypt` (called by
    ``webhook_subscription_create`` / ``rotate_secret`` in shard 04 to
    encrypt newly-minted HMAC secrets) and :meth:`decrypt` (called by
    :meth:`WebhookSubscriptionEntity.decrypt_secret` at receive time).

    ``@runtime_checkable`` is set so an ``isinstance(obj,
    EncryptionService)`` check at the agent-tools layer can confirm
    structural conformance without a hard ABC dependency.
    """

    def encrypt(self, plaintext: bytes) -> bytes:
        """Encrypt ``plaintext`` and return ciphertext bytes.

        Used by the webhook-subscription tools (shard 04) to seal
        newly-minted HMAC secrets before persistence. The platform
        does NOT decrypt them again until the webhook receiver fires
        :meth:`WebhookSubscriptionEntity.decrypt_secret`.

        :param plaintext: secret bytes (typically a hex-encoded
            ``secrets.token_hex(N)`` payload)
        :ptype plaintext: bytes
        :return: ciphertext bytes (Fernet or equivalent)
        :rtype: bytes
        """
        ...

    def decrypt(self, ciphertext: bytes) -> str | bytes:
        """Decrypt ``ciphertext`` and return the plaintext.

        :param ciphertext: Fernet (or compatible) ciphertext bytes
        :ptype ciphertext: bytes
        :return: plaintext secret string (or bytes for the platform
            to decode)
        :rtype: str | bytes
        """
        ...


def _as_uuid(value: object) -> UUID:
    """Coerce a value to stdlib ``UUID``.

    Mirrors the agent-skills convention: cache + serialization layers
    may surface UUIDs as strings; this helper normalises every read
    path to a single shape so ``isinstance`` checks elsewhere stay
    reliable.

    :param value: UUID-shaped input from any tier
    :ptype value: object
    :return: stdlib UUID
    :rtype: UUID
    """
    if isinstance(value, UUID):
        return value
    return UUID(str(value))


class WakeScheduleEntity(BaseEntity):
    """Cache proxy entity for the ``agent_wake_schedules`` table.

    Composite primary key is ``(conversation_id, schedule_id)``; the
    entity sets ``_id`` to the tuple form so the framework's pk-aware
    paths (``BaseCollection.normalize_pk``, ``l2_key``,
    ``_publish_invalidation``) address rows uniformly across tiers.

    Field accessors mirror the column list. Change tracking inherits
    from ``BaseEntity.__setattr__`` -- mutating a field via attribute
    assignment records the change for the next ``save_entity`` flush.

    Spec ref: ``docs/agent-wake/shard-01-schema-and-collections.md``
    section "Schema specification / agent_wake_schedules" (with the
    2026-05-19 revision deltas applied: nullable ``skill_id`` FK,
    ``missed_fire_policy`` column, ``no_agent`` / ``pre_check_type`` /
    ``pre_check_config`` columns DROPPED).
    """

    primary_key_field: str = "schedule_id"

    def __init__(
        self,
        data: dict[str, Any],
        is_new: bool = True,
        collection: Any = None,
    ) -> None:
        """Initialise with composite ``_id`` for tuple-pk lookup.

        :param data: row dict; ``conversation_id`` and ``schedule_id``
            must be present so the tuple ``_id`` can be assembled
        :ptype data: dict[str, Any]
        :param is_new: whether the entity is unsaved
        :ptype is_new: bool
        :param collection: owning collection reference
        :ptype collection: Any
        :return: nothing
        :rtype: None
        """
        super().__init__(data, is_new=is_new, collection=collection)
        if "conversation_id" in data and "schedule_id" in data:
            object.__setattr__(
                self,
                "_id",
                (data["conversation_id"], data["schedule_id"]),
            )

    @property
    def schedule_id(self) -> UUID:
        """Return the bare ``schedule_id`` UUID."""
        return _as_uuid(self._get_raw("schedule_id"))

    @property
    def conversation_id(self) -> UUID:
        """Return the partition column ``conversation_id``."""
        return _as_uuid(self._get_raw("conversation_id"))

    @conversation_id.setter
    def conversation_id(self, value: UUID) -> None:
        """Set the partition column."""
        BaseEntity.__setattr__(self, "conversation_id", value)

    @property
    def user_id(self) -> UUID:
        """Return the owning user (opaque -- no platform FK; see module docstring)."""
        return _as_uuid(self._get_raw("user_id"))

    @user_id.setter
    def user_id(self, value: UUID) -> None:
        """Set the owning user."""
        BaseEntity.__setattr__(self, "user_id", value)

    @property
    def agent_id(self) -> UUID:
        """Return the agent owning the schedule."""
        return _as_uuid(self._get_raw("agent_id"))

    @agent_id.setter
    def agent_id(self, value: UUID) -> None:
        """Set the agent owner."""
        BaseEntity.__setattr__(self, "agent_id", value)

    @property
    def skill_id(self) -> UUID | None:
        """Return the attached ``skill_id`` (``None`` when no skill).

        FK to ``agent_skills.skill_id`` via the cross-package standalone
        ``UNIQUE (skill_id)`` constraint declared in agent-skills v001.
        Nullable: a wake may run without an attached skill (default
        scheduled-check-in behaviour). ``ON DELETE SET NULL`` so
        deleting the skill leaves the schedule active but unbound.
        """
        value = self._get_raw("skill_id")
        if value is None:
            return None
        return _as_uuid(value)

    @skill_id.setter
    def skill_id(self, value: UUID | None) -> None:
        """Set the attached skill_id (``None`` to detach)."""
        BaseEntity.__setattr__(self, "skill_id", value)

    @property
    def schedule_type(self) -> str:
        """Return the ``schedule_type`` enum value."""
        value: str = self._get_raw("schedule_type")
        return value

    @schedule_type.setter
    def schedule_type(self, value: str) -> None:
        """Set the schedule type (validated by DB CHECK)."""
        BaseEntity.__setattr__(self, "schedule_type", value)

    @property
    def schedule_config(self) -> dict[str, Any]:
        """Return the schedule-type-specific config JSON.

        Shape varies by ``schedule_type``; structural validation lives
        in the agent-tools shard. The platform stores opaque JSONB.
        Empty-dict fallback on NULL so callers can ``.get(...)``
        without a None-guard at every site.
        """
        value = self._get_raw("schedule_config")
        if value is None:
            return {}
        return dict(value)

    @schedule_config.setter
    def schedule_config(self, value: dict[str, Any]) -> None:
        """Set the schedule config."""
        BaseEntity.__setattr__(self, "schedule_config", dict(value))

    @property
    def task_prompt(self) -> str | None:
        """Return the optional override prompt (``None`` for the default)."""
        value: str | None = self._get_raw("task_prompt")
        return value

    @task_prompt.setter
    def task_prompt(self, value: str | None) -> None:
        """Set the task prompt override."""
        BaseEntity.__setattr__(self, "task_prompt", value)

    @property
    def execution_mode(self) -> str:
        """Return the ``execution_mode`` enum value (``'inline'`` | ``'spawn'``)."""
        value: str = self._get_raw("execution_mode")
        return value

    @execution_mode.setter
    def execution_mode(self, value: str) -> None:
        """Set the execution mode."""
        BaseEntity.__setattr__(self, "execution_mode", value)

    @property
    def status(self) -> str:
        """Return the ``status`` enum value (``'active'`` | ``'paused'`` | ``'expired'``)."""
        value: str = self._get_raw("status")
        return value

    @status.setter
    def status(self, value: str) -> None:
        """Set the status (validated by DB CHECK)."""
        BaseEntity.__setattr__(self, "status", value)

    @property
    def next_fire_at(self) -> datetime | None:
        """Return the next scheduled fire time (``None`` when paused / expired)."""
        value: datetime | None = self._get_raw("next_fire_at")
        return value

    @next_fire_at.setter
    def next_fire_at(self, value: datetime | None) -> None:
        """Set the next_fire_at timestamp."""
        BaseEntity.__setattr__(self, "next_fire_at", value)

    @property
    def last_fired_at(self) -> datetime | None:
        """Return the timestamp of the most recent fire (``None`` for unfired)."""
        value: datetime | None = self._get_raw("last_fired_at")
        return value

    @last_fired_at.setter
    def last_fired_at(self, value: datetime | None) -> None:
        """Set the last_fired_at timestamp."""
        BaseEntity.__setattr__(self, "last_fired_at", value)

    @property
    def name(self) -> str | None:
        """Return the optional human-readable schedule name."""
        value: str | None = self._get_raw("name")
        return value

    @name.setter
    def name(self, value: str | None) -> None:
        """Set the schedule name."""
        BaseEntity.__setattr__(self, "name", value)

    @property
    def missed_fire_policy(self) -> str:
        """Return the ``missed_fire_policy`` enum value.

        Added per PLACEMENT §1.7 in the 2026-05-19 revision. Default
        ``'coalesce'`` (fire once for a backlog) vs ``'catch_up'`` (fire
        once per missed tick).
        """
        value: str = self._get_raw("missed_fire_policy")
        return value

    @missed_fire_policy.setter
    def missed_fire_policy(self, value: str) -> None:
        """Set the missed-fire policy."""
        BaseEntity.__setattr__(self, "missed_fire_policy", value)

    @property
    def context_from_schedule_id(self) -> UUID | None:
        """Return the optional context-source schedule id.

        Single-hop, same-conversation only (PLACEMENT §1.6 lock). Cycle
        detection lives in shard 04 (agent-tools layer). ``ON DELETE
        SET NULL`` on the self-FK so deleting the context source
        leaves the dependent schedule active but unbound.
        """
        value = self._get_raw("context_from_schedule_id")
        if value is None:
            return None
        return _as_uuid(value)

    @context_from_schedule_id.setter
    def context_from_schedule_id(self, value: UUID | None) -> None:
        """Set the context-source schedule id."""
        BaseEntity.__setattr__(self, "context_from_schedule_id", value)

    @property
    def date_created(self) -> datetime:
        """Return the creation timestamp."""
        value: datetime = self._get_raw("date_created")
        return value

    @date_created.setter
    def date_created(self, value: datetime) -> None:
        """Set the creation timestamp."""
        BaseEntity.__setattr__(self, "date_created", value)

    @property
    def date_updated(self) -> datetime:
        """Return the last-modified timestamp."""
        value: datetime = self._get_raw("date_updated")
        return value

    @date_updated.setter
    def date_updated(self, value: datetime) -> None:
        """Set the last-modified timestamp."""
        BaseEntity.__setattr__(self, "date_updated", value)


class WakeFireEntity(BaseEntity):
    """Cache proxy entity for the ``wake_fires`` table.

    Composite primary key ``(conversation_id, fire_id)``; partition
    column ``conversation_id``. One row per wake fire (scheduled or
    webhook-driven).

    Lifecycle: a fire row starts in ``status='fired'`` /
    ``'fired_silent'`` / ``'yielded'`` / ``'skipped_*'`` / ``'failed'``
    -- the dispatcher computes the terminal status once and writes
    once. The schema does NOT model a separate ``'dispatching'``
    transient state because the row is only inserted post-decision.

    ``schedule_id`` and ``webhook_subscription_id`` are exclusive-OR
    (CHECK constraint): exactly one is non-null. ``conversation_id`` is
    denormalised onto the row so fires outlive deleted schedules (same
    pattern as agent-memory's ``conversation_memory_refs``).

    Spec ref: ``docs/agent-wake/shard-01-schema-and-collections.md``
    section "Schema specification / wake_fires" (with the wake-yield
    2026-05-19 revision applied: ``status`` enum gains ``'yielded'``;
    ``display_suppressed`` boolean).
    """

    primary_key_field: str = "fire_id"

    def __init__(
        self,
        data: dict[str, Any],
        is_new: bool = True,
        collection: Any = None,
    ) -> None:
        """Initialise with composite ``_id`` for tuple-pk lookup.

        :param data: row dict; ``conversation_id`` and ``fire_id`` must
            be present
        :ptype data: dict[str, Any]
        :param is_new: whether the entity is unsaved
        :ptype is_new: bool
        :param collection: owning collection reference
        :ptype collection: Any
        :return: nothing
        :rtype: None
        """
        super().__init__(data, is_new=is_new, collection=collection)
        if "conversation_id" in data and "fire_id" in data:
            object.__setattr__(
                self,
                "_id",
                (data["conversation_id"], data["fire_id"]),
            )

    @property
    def fire_id(self) -> UUID:
        """Return the bare ``fire_id`` UUID."""
        return _as_uuid(self._get_raw("fire_id"))

    @property
    def conversation_id(self) -> UUID:
        """Return the partition column ``conversation_id``."""
        return _as_uuid(self._get_raw("conversation_id"))

    @conversation_id.setter
    def conversation_id(self, value: UUID) -> None:
        """Set the partition column."""
        BaseEntity.__setattr__(self, "conversation_id", value)

    @property
    def schedule_id(self) -> UUID | None:
        """Return the source schedule_id (``None`` for webhook fires)."""
        value = self._get_raw("schedule_id")
        if value is None:
            return None
        return _as_uuid(value)

    @schedule_id.setter
    def schedule_id(self, value: UUID | None) -> None:
        """Set the source schedule_id."""
        BaseEntity.__setattr__(self, "schedule_id", value)

    @property
    def webhook_subscription_id(self) -> UUID | None:
        """Return the source webhook subscription id (``None`` for scheduled fires)."""
        value = self._get_raw("webhook_subscription_id")
        if value is None:
            return None
        return _as_uuid(value)

    @webhook_subscription_id.setter
    def webhook_subscription_id(self, value: UUID | None) -> None:
        """Set the source webhook_subscription_id."""
        BaseEntity.__setattr__(self, "webhook_subscription_id", value)

    @property
    def scheduled_fire_at(self) -> datetime | None:
        """Return the originally scheduled fire time (``None`` for webhook fires)."""
        value: datetime | None = self._get_raw("scheduled_fire_at")
        return value

    @scheduled_fire_at.setter
    def scheduled_fire_at(self, value: datetime | None) -> None:
        """Set the scheduled_fire_at timestamp."""
        BaseEntity.__setattr__(self, "scheduled_fire_at", value)

    @property
    def actual_fired_at(self) -> datetime:
        """Return the actual fire time (post-drift)."""
        value: datetime = self._get_raw("actual_fired_at")
        return value

    @actual_fired_at.setter
    def actual_fired_at(self, value: datetime) -> None:
        """Set the actual_fired_at timestamp."""
        BaseEntity.__setattr__(self, "actual_fired_at", value)

    @property
    def status(self) -> str:
        """Return the terminal status (see :data:`FireStatus`)."""
        value: str = self._get_raw("status")
        return value

    @status.setter
    def status(self, value: str) -> None:
        """Set the status (validated by DB CHECK)."""
        BaseEntity.__setattr__(self, "status", value)

    @property
    def display_suppressed(self) -> bool:
        """Return whether visible message display was suppressed.

        ``True`` when the agent emitted ``[SILENT]`` (status =
        ``'fired_silent'``). Mirrors the row-level discriminator so the
        product's messages table can render the wake notice without a
        join.
        """
        value: bool = self._get_raw("display_suppressed")
        return value

    @display_suppressed.setter
    def display_suppressed(self, value: bool) -> None:
        """Set the display_suppressed flag."""
        BaseEntity.__setattr__(self, "display_suppressed", value)

    @property
    def output_text(self) -> str | None:
        """Return the assistant's response text (``None`` when not yet captured)."""
        value: str | None = self._get_raw("output_text")
        return value

    @output_text.setter
    def output_text(self, value: str | None) -> None:
        """Set the captured output text."""
        BaseEntity.__setattr__(self, "output_text", value)

    @property
    def latency_ms(self) -> int | None:
        """Return the end-to-end fire latency in ms (``None`` on dispatch failure)."""
        value: int | None = self._get_raw("latency_ms")
        return value

    @latency_ms.setter
    def latency_ms(self, value: int | None) -> None:
        """Set the latency."""
        BaseEntity.__setattr__(self, "latency_ms", value)

    @property
    def error(self) -> str | None:
        """Return the captured error message (``None`` for non-failed fires)."""
        value: str | None = self._get_raw("error")
        return value

    @error.setter
    def error(self, value: str | None) -> None:
        """Set the error message."""
        BaseEntity.__setattr__(self, "error", value)

    @property
    def date_created(self) -> datetime:
        """Return the row creation timestamp."""
        value: datetime = self._get_raw("date_created")
        return value

    @date_created.setter
    def date_created(self, value: datetime) -> None:
        """Set the creation timestamp."""
        BaseEntity.__setattr__(self, "date_created", value)


class WebhookSubscriptionEntity(BaseEntity):
    """Cache proxy entity for the ``webhook_subscriptions`` table.

    Composite primary key ``(conversation_id, subscription_id)``;
    standalone ``UNIQUE (subscription_id)`` so cross-package FKs can
    reference the bare id. Partition column ``conversation_id``.

    ``secret_ciphertext`` is ``bytes`` and is NEVER decoded by the
    entity. The framework's encryption service (the consumer supplies
    one; 3tears does not own a canonical implementation) opens it via
    :meth:`decrypt_secret`. The plaintext is returned ONCE on create
    and ONCE on rotate at the API surface in shard 04 -- the entity
    itself simply holds the ciphertext.

    Spec ref: ``docs/agent-wake/shard-01-schema-and-collections.md``
    section "Schema specification / webhook_subscriptions" (with the
    2026-05-19 revision applied: nullable ``default_skill_id`` FK).
    """

    primary_key_field: str = "subscription_id"

    def __init__(
        self,
        data: dict[str, Any],
        is_new: bool = True,
        collection: Any = None,
    ) -> None:
        """Initialise with composite ``_id`` for tuple-pk lookup.

        :param data: row dict; ``conversation_id`` and
            ``subscription_id`` must be present
        :ptype data: dict[str, Any]
        :param is_new: whether the entity is unsaved
        :ptype is_new: bool
        :param collection: owning collection reference
        :ptype collection: Any
        :return: nothing
        :rtype: None
        """
        super().__init__(data, is_new=is_new, collection=collection)
        if "conversation_id" in data and "subscription_id" in data:
            object.__setattr__(
                self,
                "_id",
                (data["conversation_id"], data["subscription_id"]),
            )

    @property
    def subscription_id(self) -> UUID:
        """Return the bare ``subscription_id`` UUID."""
        return _as_uuid(self._get_raw("subscription_id"))

    @property
    def conversation_id(self) -> UUID:
        """Return the partition column ``conversation_id``."""
        return _as_uuid(self._get_raw("conversation_id"))

    @conversation_id.setter
    def conversation_id(self, value: UUID) -> None:
        """Set the partition column."""
        BaseEntity.__setattr__(self, "conversation_id", value)

    @property
    def user_id(self) -> UUID:
        """Return the owning user (opaque -- no platform FK)."""
        return _as_uuid(self._get_raw("user_id"))

    @user_id.setter
    def user_id(self, value: UUID) -> None:
        """Set the owning user."""
        BaseEntity.__setattr__(self, "user_id", value)

    @property
    def agent_id(self) -> UUID:
        """Return the owning agent."""
        return _as_uuid(self._get_raw("agent_id"))

    @agent_id.setter
    def agent_id(self, value: UUID) -> None:
        """Set the owning agent."""
        BaseEntity.__setattr__(self, "agent_id", value)

    @property
    def default_skill_id(self) -> UUID | None:
        """Return the default attached ``skill_id`` (``None`` for none).

        FK to ``agent_skills.skill_id`` via the cross-package standalone
        ``UNIQUE (skill_id)`` constraint. Used at fire time as the
        attached-skill default unless the webhook payload overrides
        (shard 04 logic). ``ON DELETE SET NULL``.
        """
        value = self._get_raw("default_skill_id")
        if value is None:
            return None
        return _as_uuid(value)

    @default_skill_id.setter
    def default_skill_id(self, value: UUID | None) -> None:
        """Set the default attached skill_id."""
        BaseEntity.__setattr__(self, "default_skill_id", value)

    @property
    def name(self) -> str | None:
        """Return the optional human-readable subscription name."""
        value: str | None = self._get_raw("name")
        return value

    @name.setter
    def name(self, value: str | None) -> None:
        """Set the subscription name."""
        BaseEntity.__setattr__(self, "name", value)

    @property
    def secret_ciphertext(self) -> bytes:
        """Return the Fernet-encrypted HMAC secret.

        The entity NEVER decrypts. The consumer's encryption service
        opens it via :meth:`decrypt_secret`. The plaintext is returned
        only on create + rotate at the API surface (shard 04).
        """
        value = self._get_raw("secret_ciphertext")
        if isinstance(value, memoryview):
            return bytes(value)
        if isinstance(value, bytes):
            return value
        # asyncpg surfaces BYTEA as ``bytes`` by default; the memoryview
        # branch + this last-resort coercion are defence-in-depth.
        return bytes(value)

    @secret_ciphertext.setter
    def secret_ciphertext(self, value: bytes) -> None:
        """Set the ciphertext."""
        BaseEntity.__setattr__(self, "secret_ciphertext", bytes(value))

    def decrypt_secret(self, encryption_service: EncryptionService) -> str:
        """Decrypt and return the raw HMAC secret.

        ``encryption_service`` is a consumer-supplied object satisfying
        the :class:`EncryptionService` Protocol (Fernet is the platform
        reference; the 3tears stack does not ship a canonical
        implementation because key management is consumer-specific).
        Returns the decoded UTF-8 secret string.

        :param encryption_service: object satisfying
            :class:`EncryptionService` (``decrypt(bytes) -> str | bytes``)
        :ptype encryption_service: EncryptionService
        :return: raw secret string
        :rtype: str
        """
        plaintext = encryption_service.decrypt(self.secret_ciphertext)
        if isinstance(plaintext, bytes):
            return plaintext.decode("utf-8")
        return str(plaintext)

    @property
    def allowed_source_pattern(self) -> str | None:
        """Return the optional source-IP / sender regex pattern (``None`` = any)."""
        value: str | None = self._get_raw("allowed_source_pattern")
        return value

    @allowed_source_pattern.setter
    def allowed_source_pattern(self, value: str | None) -> None:
        """Set the source pattern."""
        BaseEntity.__setattr__(self, "allowed_source_pattern", value)

    @property
    def execution_mode(self) -> str:
        """Return the ``execution_mode`` enum value (``'inline'`` | ``'spawn'``)."""
        value: str = self._get_raw("execution_mode")
        return value

    @execution_mode.setter
    def execution_mode(self, value: str) -> None:
        """Set the execution mode."""
        BaseEntity.__setattr__(self, "execution_mode", value)

    @property
    def task_prompt_template(self) -> str | None:
        """Return the optional Jinja2 sandboxed template.

        Variables available at render time: ``{{event}}`` (entire JSON
        payload) plus ``{{event.field.subfield}}`` access. Rendering
        happens at the webhook receiver (shard 06) using
        :class:`jinja2.sandbox.SandboxedEnvironment`; the platform
        stores the raw template text.
        """
        value: str | None = self._get_raw("task_prompt_template")
        return value

    @task_prompt_template.setter
    def task_prompt_template(self, value: str | None) -> None:
        """Set the template text."""
        BaseEntity.__setattr__(self, "task_prompt_template", value)

    @property
    def verification_scheme(self) -> str:
        """Return the ``verification_scheme`` enum value."""
        value: str = self._get_raw("verification_scheme")
        return value

    @verification_scheme.setter
    def verification_scheme(self, value: str) -> None:
        """Set the verification scheme."""
        BaseEntity.__setattr__(self, "verification_scheme", value)

    @property
    def status(self) -> str:
        """Return the ``status`` enum value (``'active'`` | ``'paused'``)."""
        value: str = self._get_raw("status")
        return value

    @status.setter
    def status(self, value: str) -> None:
        """Set the status."""
        BaseEntity.__setattr__(self, "status", value)

    @property
    def rate_limit_per_minute(self) -> int | None:
        """Return the per-subscription rate-limit cap (``None`` defers to product default)."""
        value: int | None = self._get_raw("rate_limit_per_minute")
        return value

    @rate_limit_per_minute.setter
    def rate_limit_per_minute(self, value: int | None) -> None:
        """Set the rate-limit cap."""
        BaseEntity.__setattr__(self, "rate_limit_per_minute", value)

    @property
    def last_fired_at(self) -> datetime | None:
        """Return the timestamp of the most recent fire."""
        value: datetime | None = self._get_raw("last_fired_at")
        return value

    @last_fired_at.setter
    def last_fired_at(self, value: datetime | None) -> None:
        """Set the last_fired_at timestamp."""
        BaseEntity.__setattr__(self, "last_fired_at", value)

    @property
    def date_created(self) -> datetime:
        """Return the creation timestamp."""
        value: datetime = self._get_raw("date_created")
        return value

    @date_created.setter
    def date_created(self, value: datetime) -> None:
        """Set the creation timestamp."""
        BaseEntity.__setattr__(self, "date_created", value)

    @property
    def date_updated(self) -> datetime:
        """Return the last-modified timestamp."""
        value: datetime = self._get_raw("date_updated")
        return value

    @date_updated.setter
    def date_updated(self, value: datetime) -> None:
        """Set the last-modified timestamp."""
        BaseEntity.__setattr__(self, "date_updated", value)
