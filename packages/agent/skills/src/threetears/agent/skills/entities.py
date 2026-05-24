"""Agent-skills entities -- cache proxies for ``agent_skills`` +
``agent_skill_invocations``.

Both tables are partitioned on ``agent_id``; the composite primary key
shapes are ``(agent_id, skill_id)`` and ``(agent_id, invocation_id)``
respectively. Per the collections-task-04 convention the composite key
is the structural defense against cross-partition data bleed.

Each entity carries a tuple ``_id`` so ``BaseCollection.normalize_pk``
and ``BaseCollection.l2_key`` address the row uniformly across L1 /
L2 / L3. ``primary_key_field`` names the bare-id column so
``BaseEntity.id`` returns the singular UUID that downstream callers
(wake-side ``skill_id`` FK, MCP tool args) need.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from threetears.core.entities.base import BaseEntity

__all__ = [
    "AgentSkillEntity",
    "AgentSkillInvocationEntity",
]


def _as_uuid(value: object) -> UUID:
    """Coerce a value to stdlib ``UUID``.

    Mirrors the agent-memory convention: cache + serialization layers
    may surface UUIDs as strings; this helper normalises every read
    path to a single shape so isinstance checks elsewhere stay reliable.

    :param value: UUID-shaped input from any tier
    :ptype value: object
    :return: stdlib UUID
    :rtype: UUID
    """
    if isinstance(value, UUID):
        return value
    return UUID(str(value))


class AgentSkillEntity(BaseEntity):
    """Cache proxy entity for the ``agent_skills`` table.

    Composite primary key is ``(agent_id, skill_id)``; the entity sets
    ``_id`` to the tuple form so the framework's pk-aware paths
    (``BaseCollection.normalize_pk``, ``l2_key``,
    ``_publish_invalidation``) address rows uniformly across tiers.

    Field accessors mirror the column list. Change tracking inherits
    from ``BaseEntity.__setattr__`` -- mutating a field via attribute
    assignment records the change for the next ``save_entity`` flush.

    Spec ref: ``docs/agent-skills/shard-01-schema-and-collection.md``
    section "Schema specification / agent_skills".
    """

    primary_key_field: str = "skill_id"

    def __init__(
        self,
        data: dict[str, Any],
        is_new: bool = True,
        collection: Any = None,
    ) -> None:
        """Initialise with composite ``_id`` for tuple-pk lookup.

        :param data: row dict; ``agent_id`` and ``skill_id`` must be
            present so the tuple ``_id`` can be assembled
        :ptype data: dict[str, Any]
        :param is_new: whether the entity is unsaved
        :ptype is_new: bool
        :param collection: owning collection reference
        :ptype collection: Any
        :return: nothing
        :rtype: None
        """
        super().__init__(data, is_new=is_new, collection=collection)
        if "agent_id" in data and "skill_id" in data:
            object.__setattr__(
                self,
                "_id",
                (data["agent_id"], data["skill_id"]),
            )

    @property
    def skill_id(self) -> UUID:
        """Return the bare ``skill_id`` UUID."""
        return _as_uuid(self._get_raw("skill_id"))

    @property
    def agent_id(self) -> UUID:
        """Return the partition column ``agent_id``."""
        return _as_uuid(self._get_raw("agent_id"))

    @agent_id.setter
    def agent_id(self, value: UUID) -> None:
        """Set the partition column."""
        BaseEntity.__setattr__(self, "agent_id", value)

    @property
    def user_id(self) -> UUID:
        """Return the owning ``user_id`` (denormalised onto the row)."""
        return _as_uuid(self._get_raw("user_id"))

    @user_id.setter
    def user_id(self, value: UUID) -> None:
        """Set the owning user."""
        BaseEntity.__setattr__(self, "user_id", value)

    @property
    def name(self) -> str:
        """Return the human-readable skill name (unique per ``(agent_id, user_id)``)."""
        value: str = self._get_raw("name")
        return value

    @name.setter
    def name(self, value: str) -> None:
        """Set the skill name."""
        BaseEntity.__setattr__(self, "name", value)

    @property
    def summary(self) -> str:
        """Return the one-line catalog entry (required; visible in ``skill_list``)."""
        value: str = self._get_raw("summary")
        return value

    @summary.setter
    def summary(self, value: str) -> None:
        """Set the catalog summary."""
        BaseEntity.__setattr__(self, "summary", value)

    @property
    def body(self) -> str | None:
        """Return the optional prose body (markdown).

        NULL for pure tool-composition skills (the DB-side CHECK
        constraint enforces at-least-one-payload across ``body``,
        ``tool_additions``, ``tool_restrictions``).
        """
        value: str | None = self._get_raw("body")
        return value

    @body.setter
    def body(self, value: str | None) -> None:
        """Set the prose body."""
        BaseEntity.__setattr__(self, "body", value)

    @property
    def prompt_mode(self) -> str:
        """Return the ``prompt_mode`` enum value (``'additive'`` | ``'replace'``)."""
        value: str = self._get_raw("prompt_mode")
        return value

    @prompt_mode.setter
    def prompt_mode(self, value: str) -> None:
        """Set the prompt mode (validated by DB CHECK constraint)."""
        BaseEntity.__setattr__(self, "prompt_mode", value)

    @property
    def tool_additions(self) -> list[str]:
        """Return the ``tool_additions`` list (mcp_name entries).

        Stored as a Postgres ``TEXT[]``; asyncpg surfaces it as a
        Python list. Empty list is the default; CHECK constraint
        applies the "at least one payload" rule across all three
        payload fields, not on this one in isolation.
        """
        value = self._get_raw("tool_additions")
        if value is None:
            return []
        return list(value)

    @tool_additions.setter
    def tool_additions(self, value: list[str]) -> None:
        """Set the tool_additions list."""
        BaseEntity.__setattr__(self, "tool_additions", list(value))

    @property
    def tool_restrictions(self) -> list[str]:
        """Return the ``tool_restrictions`` list (mcp_name entries)."""
        value = self._get_raw("tool_restrictions")
        if value is None:
            return []
        return list(value)

    @tool_restrictions.setter
    def tool_restrictions(self, value: list[str]) -> None:
        """Set the tool_restrictions list."""
        BaseEntity.__setattr__(self, "tool_restrictions", list(value))

    @property
    def trigger_keywords(self) -> str:
        """Return the keywords string (``skill_list`` filter only, never auto-load)."""
        value: str = self._get_raw("trigger_keywords")
        return value

    @trigger_keywords.setter
    def trigger_keywords(self, value: str) -> None:
        """Set the trigger keywords."""
        BaseEntity.__setattr__(self, "trigger_keywords", value)

    @property
    def tags(self) -> list[str]:
        """Return the tags list."""
        value = self._get_raw("tags")
        if value is None:
            return []
        return list(value)

    @tags.setter
    def tags(self, value: list[str]) -> None:
        """Set the tags list."""
        BaseEntity.__setattr__(self, "tags", list(value))

    @property
    def source(self) -> str:
        """Return the ``source`` enum value (``'manual'`` in v1)."""
        value: str = self._get_raw("source")
        return value

    @source.setter
    def source(self, value: str) -> None:
        """Set the source provenance."""
        BaseEntity.__setattr__(self, "source", value)

    @property
    def enabled(self) -> bool:
        """Return whether the skill is enabled (eligible for load)."""
        value: bool = self._get_raw("enabled")
        return value

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Set the enabled flag."""
        BaseEntity.__setattr__(self, "enabled", value)

    @property
    def use_count(self) -> int:
        """Return total number of loads ever (bumped by ``bump_use_count``)."""
        value: int = self._get_raw("use_count")
        return value

    @use_count.setter
    def use_count(self, value: int) -> None:
        """Set the use count."""
        BaseEntity.__setattr__(self, "use_count", value)

    @property
    def last_used_at(self) -> datetime | None:
        """Return last-load timestamp (``None`` if never loaded)."""
        value: datetime | None = self._get_raw("last_used_at")
        return value

    @last_used_at.setter
    def last_used_at(self, value: datetime | None) -> None:
        """Set the last-used timestamp."""
        BaseEntity.__setattr__(self, "last_used_at", value)

    @property
    def success_count(self) -> int:
        """Return the success-outcome count (populated by ``increment_outcome_counts``)."""
        value: int = self._get_raw("success_count")
        return value

    @success_count.setter
    def success_count(self, value: int) -> None:
        """Set the success count."""
        BaseEntity.__setattr__(self, "success_count", value)

    @property
    def failure_count(self) -> int:
        """Return the failure-outcome count."""
        value: int = self._get_raw("failure_count")
        return value

    @failure_count.setter
    def failure_count(self, value: int) -> None:
        """Set the failure count."""
        BaseEntity.__setattr__(self, "failure_count", value)

    @property
    def last_failure_at(self) -> datetime | None:
        """Return last-failure timestamp."""
        value: datetime | None = self._get_raw("last_failure_at")
        return value

    @last_failure_at.setter
    def last_failure_at(self, value: datetime | None) -> None:
        """Set the last-failure timestamp."""
        BaseEntity.__setattr__(self, "last_failure_at", value)

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


class AgentSkillInvocationEntity(BaseEntity):
    """Cache proxy entity for the ``agent_skill_invocations`` table.

    One row per skill load -- captures both wake-attached and explicit
    ``skill_invoke`` events. ``message_id`` is consumer-populated
    post-LLM (the consumer's loader calls ``set_message_id`` after the
    assistant response lands; the row is intentionally NOT FK'd to
    ``messages`` because ``messages`` is consumer-owned and may be
    hard-deleted).

    Composite primary key ``(agent_id, invocation_id)``; partition
    column ``agent_id``. Composite FK
    ``(agent_id, skill_id) REFERENCES agent_skills(agent_id, skill_id)``
    with ``ON DELETE CASCADE`` so deleting a skill discards its history.

    Spec ref: ``docs/agent-skills/shard-01-schema-and-collection.md``
    section "Schema specification / agent_skill_invocations".
    """

    primary_key_field: str = "invocation_id"

    def __init__(
        self,
        data: dict[str, Any],
        is_new: bool = True,
        collection: Any = None,
    ) -> None:
        """Initialise with composite ``_id`` for tuple-pk lookup.

        :param data: row dict; ``agent_id`` and ``invocation_id`` must
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
        if "agent_id" in data and "invocation_id" in data:
            object.__setattr__(
                self,
                "_id",
                (data["agent_id"], data["invocation_id"]),
            )

    @property
    def invocation_id(self) -> UUID:
        """Return the bare ``invocation_id`` UUID."""
        return _as_uuid(self._get_raw("invocation_id"))

    @property
    def agent_id(self) -> UUID:
        """Return the partition column ``agent_id``."""
        return _as_uuid(self._get_raw("agent_id"))

    @agent_id.setter
    def agent_id(self, value: UUID) -> None:
        """Set the partition column."""
        BaseEntity.__setattr__(self, "agent_id", value)

    @property
    def skill_id(self) -> UUID:
        """Return the referenced ``skill_id`` (FK target on ``agent_skills``)."""
        return _as_uuid(self._get_raw("skill_id"))

    @skill_id.setter
    def skill_id(self, value: UUID) -> None:
        """Set the referenced skill_id."""
        BaseEntity.__setattr__(self, "skill_id", value)

    @property
    def user_id(self) -> UUID:
        """Return the user the skill ran under (denormalised onto the row)."""
        return _as_uuid(self._get_raw("user_id"))

    @user_id.setter
    def user_id(self, value: UUID) -> None:
        """Set the user_id."""
        BaseEntity.__setattr__(self, "user_id", value)

    @property
    def conversation_id(self) -> UUID:
        """Return the conversation the skill loaded into."""
        return _as_uuid(self._get_raw("conversation_id"))

    @conversation_id.setter
    def conversation_id(self, value: UUID) -> None:
        """Set the conversation_id."""
        BaseEntity.__setattr__(self, "conversation_id", value)

    @property
    def message_id(self) -> UUID | None:
        """Return the assistant-response ``message_id`` (consumer-populated post-LLM).

        NULL until the consumer's loader calls ``set_message_id`` --
        intentionally NOT FK'd to ``messages`` because the messages
        table is consumer-owned and may be hard-deleted; the
        invocation history must survive a message deletion.
        """
        value = self._get_raw("message_id")
        if value is None:
            return None
        return _as_uuid(value)

    @message_id.setter
    def message_id(self, value: UUID | None) -> None:
        """Set the message_id (typically called by the consumer's post-LLM hook)."""
        BaseEntity.__setattr__(self, "message_id", value)

    @property
    def invocation_source(self) -> str:
        """Return the ``invocation_source`` enum value (``'wake'`` | ``'invoke'``)."""
        value: str = self._get_raw("invocation_source")
        return value

    @invocation_source.setter
    def invocation_source(self, value: str) -> None:
        """Set the invocation source."""
        BaseEntity.__setattr__(self, "invocation_source", value)

    @property
    def invoked_at(self) -> datetime:
        """Return the load timestamp."""
        value: datetime = self._get_raw("invoked_at")
        return value

    @invoked_at.setter
    def invoked_at(self, value: datetime) -> None:
        """Set the invoked_at timestamp."""
        BaseEntity.__setattr__(self, "invoked_at", value)

    @property
    def outcome(self) -> str | None:
        """Return the outcome (``'success'`` | ``'failure'`` | ``None``).

        ``None`` means no ``[SUCCESS]``/``[FAILED]`` marker was present
        in the assistant's response. Populated synchronously by the
        consumer's post-LLM hook (PLACEMENT §1.10); no background
        classifier tick.
        """
        value: str | None = self._get_raw("outcome")
        return value

    @outcome.setter
    def outcome(self, value: str | None) -> None:
        """Set the outcome (validated by DB CHECK)."""
        BaseEntity.__setattr__(self, "outcome", value)

    @property
    def outcome_source(self) -> str | None:
        """Return the outcome provenance (``'agent_marker'`` | ``'user_feedback'`` | ``None``)."""
        value: str | None = self._get_raw("outcome_source")
        return value

    @outcome_source.setter
    def outcome_source(self, value: str | None) -> None:
        """Set the outcome source."""
        BaseEntity.__setattr__(self, "outcome_source", value)

    @property
    def notes(self) -> str | None:
        """Return optional free-form notes."""
        value: str | None = self._get_raw("notes")
        return value

    @notes.setter
    def notes(self, value: str | None) -> None:
        """Set optional notes."""
        BaseEntity.__setattr__(self, "notes", value)
