"""Memory entity -- thin cache proxy for user memory records.

Also defines entities for orphan tables adopted under namespace-task-01
phase 8.5b: :class:`MediaEntity` (parent record from migration v006),
:class:`MediaContentEntity` (content rows from v006), and
:class:`MemoryChunkEntity` (document-style chunks from v007).

:class:`MemoryRefEntity` covers the ``conversation_memory_refs`` table
(migration v002) with its composite primary key
``(conversation_id, item_id)`` — adopted under namespace-task-01 phase
8.5l-2 on top of 8.5l-1's composite-pk BaseCollection support. It
retires the bespoke :class:`MemoryLedger` wrapper entirely.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from threetears.core.entities.base import BaseEntity

__all__ = [
    "MediaContentEntity",
    "MediaEntity",
    "MemoryChunkEntity",
    "MemoryEntity",
    "MemoryRefEntity",
]


def _as_uuid(value: object) -> UUID:
    """Coerce a value to UUID, handling strings from cache/data layers."""
    if isinstance(value, UUID):
        return value
    return UUID(str(value))


class MemoryEntity(BaseEntity):
    """Cache proxy entity for the ``memories`` table.

    collections-task-04 partitioned ``memories`` on ``agent_id``;
    the composite primary key is ``(agent_id, memory_id)``. the
    constructor sets ``_id`` to that tuple so
    :meth:`BaseCollection.normalize_pk` and :meth:`BaseCollection.l2_key`
    address the row uniformly across L1 / L2 / L3.

    Memory is the cognitive anchor under the unified data model:
    :class:`MediaEntity` rows hang off via ``media.memory_id`` and
    :class:`MemoryChunkEntity` rows hang off via
    ``memory_chunks.memory_id`` (both NOT NULL FKs with CASCADE on
    memory delete after v017). Deletion is hard-only — the legacy
    ``is_deleted`` / ``date_deleted`` columns were removed in v018.
    ``conversation_id`` is NOT NULL after v019.
    """

    primary_key_field: str = "memory_id"

    def __init__(
        self,
        data: dict[str, Any],
        is_new: bool = True,
        collection: Any = None,
    ) -> None:
        """initialize entity with composite ``_id`` for composite-pk lookup.

        :param data: row dict; must carry ``agent_id`` and ``memory_id``
        :ptype data: dict[str, Any]
        :param is_new: whether entity is unsaved
        :ptype is_new: bool
        :param collection: owning collection reference
        :ptype collection: Any
        :return: nothing
        :rtype: None
        """
        super().__init__(data, is_new=is_new, collection=collection)
        if "agent_id" in data and "memory_id" in data:
            object.__setattr__(
                self,
                "_id",
                (data["agent_id"], data["memory_id"]),
            )

    @property
    def memory_id(self) -> UUID:
        """Get the memory ID."""
        return _as_uuid(self._get_raw("memory_id"))

    @property
    def agent_id(self) -> UUID:
        """Get agent ID for memory scoping."""
        return _as_uuid(self._get_raw("agent_id"))

    @agent_id.setter
    def agent_id(self, value: UUID) -> None:
        """Set agent ID."""
        BaseEntity.__setattr__(self, "agent_id", value)

    @property
    def customer_id(self) -> UUID:
        """Get customer ID for memory scoping."""
        return _as_uuid(self._get_raw("customer_id"))

    @customer_id.setter
    def customer_id(self, value: UUID) -> None:
        """Set customer ID."""
        BaseEntity.__setattr__(self, "customer_id", value)

    @property
    def user_id(self) -> UUID:
        """Get the user ID that owns this memory."""
        return _as_uuid(self._get_raw("user_id"))

    @user_id.setter
    def user_id(self, value: UUID) -> None:
        """Set the user ID."""
        BaseEntity.__setattr__(self, "user_id", value)

    @property
    def conversation_id(self) -> UUID:
        """Get the conversation ID where this memory was extracted."""
        return _as_uuid(self._get_raw("conversation_id"))

    @conversation_id.setter
    def conversation_id(self, value: UUID) -> None:
        """Set the conversation ID."""
        BaseEntity.__setattr__(self, "conversation_id", value)

    @property
    def message_id_source(self) -> UUID:
        """Get the source message ID for this memory."""
        return _as_uuid(self._get_raw("message_id_source"))

    @message_id_source.setter
    def message_id_source(self, value: UUID) -> None:
        """Set the source message ID."""
        BaseEntity.__setattr__(self, "message_id_source", value)

    @property
    def type_memory(self) -> str:
        """Get the memory type classification."""
        value: str = self._get_raw("type_memory")
        return value

    @type_memory.setter
    def type_memory(self, value: str) -> None:
        """Set the memory type."""
        BaseEntity.__setattr__(self, "type_memory", value)

    @property
    def content(self) -> str:
        """Get the memory content text."""
        value: str = self._get_raw("content")
        return value

    @content.setter
    def content(self, value: str) -> None:
        """Set the memory content."""
        BaseEntity.__setattr__(self, "content", value)

    @property
    def embedding(self) -> list[float]:
        """Get the embedding vector."""
        value: list[float] = self._get_raw("embedding")
        return value

    @embedding.setter
    def embedding(self, value: list[float]) -> None:
        """Set the embedding vector."""
        BaseEntity.__setattr__(self, "embedding", value)

    @property
    def date_updated(self) -> datetime | None:
        """Get the last-updated timestamp."""
        value: datetime | None = self._get_raw("date_updated")
        return value

    @date_updated.setter
    def date_updated(self, value: datetime | None) -> None:
        """Set the last-updated timestamp."""
        BaseEntity.__setattr__(self, "date_updated", value)

    @property
    def alias(self) -> str | None:
        """Optional named anchor for direct-lookup retrieval (v0.7.4).

        Per-user unique on the metallm DB side via the partial unique
        index ``ix_memories_user_alias ON memories(agent_id, user_id,
        alias) WHERE alias IS NOT NULL`` (alembic 088). NULL on
        legacy rows.
        """
        value: str | None = self._get_raw("alias")
        return value

    @alias.setter
    def alias(self, value: str | None) -> None:
        """Set the named-anchor alias."""
        BaseEntity.__setattr__(self, "alias", value)


class MediaEntity(BaseEntity):
    """cache proxy entity for the ``media`` parent table (v006).

    Under the unified memory model media is an attachment under a
    memory: ``memory_id`` is a NOT NULL FK to :class:`MemoryEntity`
    with CASCADE on parent delete (v017). The memory wraps the
    cognitive description; the media row carries the raw artifact;
    :class:`MediaContentEntity` rows carry the extracted text.

    columns: ``media_id`` PK, ``memory_id`` parent (NOT NULL after
    v017), nullable ``agent_id`` / ``customer_id``, required
    ``user_id``, ``media_category`` discriminator, ``metadata_json``
    blob, plus ``date_created`` / ``date_updated``.
    """

    primary_key_field: str = "media_id"

    def __init__(
        self,
        data: dict[str, Any],
        is_new: bool = True,
        collection: Any = None,
    ) -> None:
        """initialize entity with composite ``_id`` for composite-pk lookup.

        collections-task-04 partitioned ``media`` on ``agent_id``;
        composite PK is ``(agent_id, media_id)``.

        :param data: row dict; must carry ``agent_id`` and ``media_id``
        :ptype data: dict[str, Any]
        :param is_new: whether entity is unsaved
        :ptype is_new: bool
        :param collection: owning collection reference
        :ptype collection: Any
        :return: nothing
        :rtype: None
        """
        super().__init__(data, is_new=is_new, collection=collection)
        if "agent_id" in data and "media_id" in data:
            object.__setattr__(
                self,
                "_id",
                (data["agent_id"], data["media_id"]),
            )

    @property
    def media_id(self) -> UUID:
        """get the media ID.

        :return: media UUID
        :rtype: UUID
        """
        return _as_uuid(self._get_raw("media_id"))

    @property
    def memory_id(self) -> UUID:
        """get the parent memory ID.

        Every media row is an attachment under a memory; the parent
        memory is the cognitive anchor and the media row carries
        the raw artifact. NOT NULL after v017 enforces the FK with
        CASCADE on memory delete.

        :return: parent memory UUID
        :rtype: UUID
        """
        return _as_uuid(self._get_raw("memory_id"))

    @memory_id.setter
    def memory_id(self, value: UUID) -> None:
        """set the parent memory ID.

        :param value: new parent memory UUID
        :ptype value: UUID
        """
        BaseEntity.__setattr__(self, "memory_id", value)

    @property
    def agent_id(self) -> UUID | None:
        """get optional agent ID for scoping.

        :return: agent UUID or ``None``
        :rtype: UUID | None
        """
        value = self._get_raw("agent_id")
        if value is None:
            return None
        return _as_uuid(value)

    @agent_id.setter
    def agent_id(self, value: UUID | None) -> None:
        """set the agent ID.

        :param value: new agent UUID or ``None``
        :ptype value: UUID | None
        """
        BaseEntity.__setattr__(self, "agent_id", value)

    @property
    def customer_id(self) -> UUID | None:
        """get optional customer ID for scoping.

        :return: customer UUID or ``None``
        :rtype: UUID | None
        """
        value = self._get_raw("customer_id")
        if value is None:
            return None
        return _as_uuid(value)

    @customer_id.setter
    def customer_id(self, value: UUID | None) -> None:
        """set the customer ID.

        :param value: new customer UUID or ``None``
        :ptype value: UUID | None
        """
        BaseEntity.__setattr__(self, "customer_id", value)

    @property
    def user_id(self) -> UUID:
        """get owning user ID.

        :return: user UUID
        :rtype: UUID
        """
        return _as_uuid(self._get_raw("user_id"))

    @user_id.setter
    def user_id(self, value: UUID) -> None:
        """set owning user ID.

        :param value: new user UUID
        :ptype value: UUID
        """
        BaseEntity.__setattr__(self, "user_id", value)

    @property
    def media_category(self) -> str:
        """get the media category discriminator.

        :return: category string
        :rtype: str
        """
        value: str = self._get_raw("media_category")
        return value

    @media_category.setter
    def media_category(self, value: str) -> None:
        """set the media category discriminator.

        :param value: new category
        :ptype value: str
        """
        BaseEntity.__setattr__(self, "media_category", value)

    @property
    def metadata_json(self) -> Any:
        """get the metadata JSONB blob.

        :return: metadata payload (dict / list / scalar / ``None``)
        :rtype: Any
        """
        return self._get_raw("metadata_json")

    @metadata_json.setter
    def metadata_json(self, value: Any) -> None:
        """set the metadata JSONB blob.

        :param value: new metadata payload
        :ptype value: Any
        """
        BaseEntity.__setattr__(self, "metadata_json", value)

    @property
    def date_created(self) -> datetime:
        """get the creation timestamp.

        :return: creation datetime
        :rtype: datetime
        """
        value: datetime = self._get_raw("date_created")
        return value

    @date_created.setter
    def date_created(self, value: datetime) -> None:
        """set the creation timestamp.

        :param value: new creation datetime
        :ptype value: datetime
        """
        BaseEntity.__setattr__(self, "date_created", value)

    @property
    def date_updated(self) -> datetime:
        """get the last-updated timestamp.

        :return: update datetime
        :rtype: datetime
        """
        value: datetime = self._get_raw("date_updated")
        return value

    @date_updated.setter
    def date_updated(self, value: datetime) -> None:
        """set the last-updated timestamp.

        :param value: new update datetime
        :ptype value: datetime
        """
        BaseEntity.__setattr__(self, "date_updated", value)


class MediaContentEntity(BaseEntity):
    """cache proxy entity for the ``media_content`` table (v006).

    content rows attached to a :class:`MediaEntity` parent through
    ``media_id``. carry extracted text / transcripts / OCR with their
    own embedding + FTS search_vector maintained by trigger.
    """

    primary_key_field: str = "content_id"

    def __init__(
        self,
        data: dict[str, Any],
        is_new: bool = True,
        collection: Any = None,
    ) -> None:
        """initialize entity with composite ``_id`` for composite-pk lookup.

        collections-task-04 partitioned ``media_content`` on
        ``agent_id``; composite PK is ``(agent_id, content_id)``.

        :param data: row dict; must carry ``agent_id`` and ``content_id``
        :ptype data: dict[str, Any]
        :param is_new: whether entity is unsaved
        :ptype is_new: bool
        :param collection: owning collection reference
        :ptype collection: Any
        :return: nothing
        :rtype: None
        """
        super().__init__(data, is_new=is_new, collection=collection)
        if "agent_id" in data and "content_id" in data:
            object.__setattr__(
                self,
                "_id",
                (data["agent_id"], data["content_id"]),
            )

    @property
    def content_id(self) -> UUID:
        """get the content ID.

        :return: content UUID
        :rtype: UUID
        """
        return _as_uuid(self._get_raw("content_id"))

    @property
    def media_id(self) -> UUID:
        """get the parent media ID.

        :return: media UUID
        :rtype: UUID
        """
        return _as_uuid(self._get_raw("media_id"))

    @media_id.setter
    def media_id(self, value: UUID) -> None:
        """set the parent media ID.

        :param value: new media UUID
        :ptype value: UUID
        """
        BaseEntity.__setattr__(self, "media_id", value)

    @property
    def agent_id(self) -> UUID | None:
        """get optional agent ID for scoping.

        :return: agent UUID or ``None``
        :rtype: UUID | None
        """
        value = self._get_raw("agent_id")
        if value is None:
            return None
        return _as_uuid(value)

    @agent_id.setter
    def agent_id(self, value: UUID | None) -> None:
        """set the agent ID.

        :param value: new agent UUID or ``None``
        :ptype value: UUID | None
        """
        BaseEntity.__setattr__(self, "agent_id", value)

    @property
    def customer_id(self) -> UUID | None:
        """get optional customer ID for scoping.

        :return: customer UUID or ``None``
        :rtype: UUID | None
        """
        value = self._get_raw("customer_id")
        if value is None:
            return None
        return _as_uuid(value)

    @customer_id.setter
    def customer_id(self, value: UUID | None) -> None:
        """set the customer ID.

        :param value: new customer UUID or ``None``
        :ptype value: UUID | None
        """
        BaseEntity.__setattr__(self, "customer_id", value)

    @property
    def user_id(self) -> UUID:
        """get owning user ID.

        :return: user UUID
        :rtype: UUID
        """
        return _as_uuid(self._get_raw("user_id"))

    @user_id.setter
    def user_id(self, value: UUID) -> None:
        """set owning user ID.

        :param value: new user UUID
        :ptype value: UUID
        """
        BaseEntity.__setattr__(self, "user_id", value)

    @property
    def content_type(self) -> str:
        """get the content-type discriminator.

        :return: content type string
        :rtype: str
        """
        value: str = self._get_raw("content_type")
        return value

    @content_type.setter
    def content_type(self, value: str) -> None:
        """set the content-type discriminator.

        :param value: new content type
        :ptype value: str
        """
        BaseEntity.__setattr__(self, "content_type", value)

    @property
    def content(self) -> str:
        """get the content text.

        :return: content text
        :rtype: str
        """
        value: str = self._get_raw("content")
        return value

    @content.setter
    def content(self, value: str) -> None:
        """set the content text.

        :param value: new content text
        :ptype value: str
        """
        BaseEntity.__setattr__(self, "content", value)

    @property
    def summary(self) -> str | None:
        """get optional summary text.

        :return: summary text or ``None``
        :rtype: str | None
        """
        value: str | None = self._get_raw("summary")
        return value

    @summary.setter
    def summary(self, value: str | None) -> None:
        """set optional summary text.

        :param value: new summary or ``None``
        :ptype value: str | None
        """
        BaseEntity.__setattr__(self, "summary", value)

    @property
    def embedding(self) -> list[float] | None:
        """get the embedding vector.

        :return: embedding vector or ``None``
        :rtype: list[float] | None
        """
        value: list[float] | None = self._get_raw("embedding")
        return value

    @embedding.setter
    def embedding(self, value: list[float] | None) -> None:
        """set the embedding vector.

        :param value: new embedding or ``None``
        :ptype value: list[float] | None
        """
        BaseEntity.__setattr__(self, "embedding", value)

    @property
    def date_created(self) -> datetime:
        """get the creation timestamp.

        :return: creation datetime
        :rtype: datetime
        """
        value: datetime = self._get_raw("date_created")
        return value

    @date_created.setter
    def date_created(self, value: datetime) -> None:
        """set the creation timestamp.

        :param value: new creation datetime
        :ptype value: datetime
        """
        BaseEntity.__setattr__(self, "date_created", value)


class MemoryChunkEntity(BaseEntity):
    """cache proxy entity for the ``memory_chunks`` table (v007).

    chunks are the verbatim source layer under the unified memory
    model. every chunk parents to exactly one :class:`MemoryEntity`
    via ``memory_id`` (NOT NULL FK with CASCADE after v017). there
    are two shapes:

    - document chunks: ``heading_context`` / ``page_number`` from
      the source artifact; ``message_id_start`` / ``message_id_end``
      are NULL. parent memory wraps a :class:`MediaEntity` carrying
      the original file.
    - transcript chunks: ``message_id_start`` / ``message_id_end``
      back-reference the message range the chunk summarizes;
      ``heading_context`` / ``page_number`` are NULL. parent memory
      is created by ``conversation_summarize``.

    same embedding / FTS shape as :class:`MediaContentEntity`. the
    cascading delete chain is memory -> chunk (this entity) and
    memory -> media -> media_content.
    """

    primary_key_field: str = "chunk_id"

    def __init__(
        self,
        data: dict[str, Any],
        is_new: bool = True,
        collection: Any = None,
    ) -> None:
        """initialize entity with composite ``_id`` for composite-pk lookup.

        collections-task-04 partitioned ``memory_chunks`` on
        ``agent_id``; composite PK is ``(agent_id, chunk_id)``.

        :param data: row dict; must carry ``agent_id`` and ``chunk_id``
        :ptype data: dict[str, Any]
        :param is_new: whether entity is unsaved
        :ptype is_new: bool
        :param collection: owning collection reference
        :ptype collection: Any
        :return: nothing
        :rtype: None
        """
        super().__init__(data, is_new=is_new, collection=collection)
        if "agent_id" in data and "chunk_id" in data:
            object.__setattr__(
                self,
                "_id",
                (data["agent_id"], data["chunk_id"]),
            )

    @property
    def chunk_id(self) -> UUID:
        """get the chunk ID.

        :return: chunk UUID
        :rtype: UUID
        """
        return _as_uuid(self._get_raw("chunk_id"))

    @property
    def memory_id(self) -> UUID:
        """get the parent memory ID.

        Every chunk parents to exactly one memory under the unified
        model. Document chunks parent to the memory that wraps the
        source media; transcript chunks parent to the memory created
        by ``conversation_summarize``. NOT NULL after v017 enforces
        the FK with CASCADE on memory delete.

        :return: parent memory UUID
        :rtype: UUID
        """
        return _as_uuid(self._get_raw("memory_id"))

    @memory_id.setter
    def memory_id(self, value: UUID) -> None:
        """set the parent memory ID.

        :param value: new parent memory UUID
        :ptype value: UUID
        """
        BaseEntity.__setattr__(self, "memory_id", value)

    @property
    def message_id_start(self) -> UUID | None:
        """get the first-message backlink for transcript chunks.

        NULL on document chunks (no source message range). On
        transcript chunks, points at the first message in the
        summarized range. No FK — the messages table is owned by a
        sibling system and may be hard-deleted; dangling refs are
        intentional.

        :return: first-message UUID or ``None``
        :rtype: UUID | None
        """
        value = self._get_raw("message_id_start")
        if value is None:
            return None
        return _as_uuid(value)

    @message_id_start.setter
    def message_id_start(self, value: UUID | None) -> None:
        """set the first-message backlink.

        :param value: new first-message UUID or ``None``
        :ptype value: UUID | None
        """
        BaseEntity.__setattr__(self, "message_id_start", value)

    @property
    def message_id_end(self) -> UUID | None:
        """get the last-message backlink for transcript chunks.

        NULL on document chunks; on transcript chunks, points at the
        last message in the summarized range. No FK (same reasoning
        as ``message_id_start``).

        :return: last-message UUID or ``None``
        :rtype: UUID | None
        """
        value = self._get_raw("message_id_end")
        if value is None:
            return None
        return _as_uuid(value)

    @message_id_end.setter
    def message_id_end(self, value: UUID | None) -> None:
        """set the last-message backlink.

        :param value: new last-message UUID or ``None``
        :ptype value: UUID | None
        """
        BaseEntity.__setattr__(self, "message_id_end", value)

    @property
    def agent_id(self) -> UUID | None:
        """get optional agent ID for scoping.

        :return: agent UUID or ``None``
        :rtype: UUID | None
        """
        value = self._get_raw("agent_id")
        if value is None:
            return None
        return _as_uuid(value)

    @agent_id.setter
    def agent_id(self, value: UUID | None) -> None:
        """set the agent ID.

        :param value: new agent UUID or ``None``
        :ptype value: UUID | None
        """
        BaseEntity.__setattr__(self, "agent_id", value)

    @property
    def customer_id(self) -> UUID | None:
        """get optional customer ID for scoping.

        :return: customer UUID or ``None``
        :rtype: UUID | None
        """
        value = self._get_raw("customer_id")
        if value is None:
            return None
        return _as_uuid(value)

    @customer_id.setter
    def customer_id(self, value: UUID | None) -> None:
        """set the customer ID.

        :param value: new customer UUID or ``None``
        :ptype value: UUID | None
        """
        BaseEntity.__setattr__(self, "customer_id", value)

    @property
    def user_id(self) -> UUID:
        """get owning user ID.

        :return: user UUID
        :rtype: UUID
        """
        return _as_uuid(self._get_raw("user_id"))

    @user_id.setter
    def user_id(self, value: UUID) -> None:
        """set owning user ID.

        :param value: new user UUID
        :ptype value: UUID
        """
        BaseEntity.__setattr__(self, "user_id", value)

    @property
    def content(self) -> str:
        """get the chunk content.

        :return: chunk text
        :rtype: str
        """
        value: str = self._get_raw("content")
        return value

    @content.setter
    def content(self, value: str) -> None:
        """set the chunk content.

        :param value: new chunk text
        :ptype value: str
        """
        BaseEntity.__setattr__(self, "content", value)

    @property
    def summary(self) -> str | None:
        """get optional summary text.

        :return: summary text or ``None``
        :rtype: str | None
        """
        value: str | None = self._get_raw("summary")
        return value

    @summary.setter
    def summary(self, value: str | None) -> None:
        """set optional summary text.

        :param value: new summary or ``None``
        :ptype value: str | None
        """
        BaseEntity.__setattr__(self, "summary", value)

    @property
    def heading_context(self) -> str | None:
        """get optional heading / section context.

        :return: heading context or ``None``
        :rtype: str | None
        """
        value: str | None = self._get_raw("heading_context")
        return value

    @heading_context.setter
    def heading_context(self, value: str | None) -> None:
        """set optional heading / section context.

        :param value: new heading context or ``None``
        :ptype value: str | None
        """
        BaseEntity.__setattr__(self, "heading_context", value)

    @property
    def page_number(self) -> int | None:
        """get optional page number.

        :return: page number or ``None``
        :rtype: int | None
        """
        value: int | None = self._get_raw("page_number")
        return value

    @page_number.setter
    def page_number(self, value: int | None) -> None:
        """set optional page number.

        :param value: new page number or ``None``
        :ptype value: int | None
        """
        BaseEntity.__setattr__(self, "page_number", value)

    @property
    def embedding(self) -> list[float] | None:
        """get the embedding vector.

        :return: embedding vector or ``None``
        :rtype: list[float] | None
        """
        value: list[float] | None = self._get_raw("embedding")
        return value

    @embedding.setter
    def embedding(self, value: list[float] | None) -> None:
        """set the embedding vector.

        :param value: new embedding or ``None``
        :ptype value: list[float] | None
        """
        BaseEntity.__setattr__(self, "embedding", value)

    @property
    def date_created(self) -> datetime:
        """get the creation timestamp.

        :return: creation datetime
        :rtype: datetime
        """
        value: datetime = self._get_raw("date_created")
        return value

    @date_created.setter
    def date_created(self, value: datetime) -> None:
        """set the creation timestamp.

        :param value: new creation datetime
        :ptype value: datetime
        """
        BaseEntity.__setattr__(self, "date_created", value)


class MemoryRefEntity(BaseEntity):
    """cache proxy entity for the ``conversation_memory_refs`` table (v002).

    tracks a single ``(conversation_id, item_id)`` reference recording
    that a memory / media-content / chunk row has been surfaced to the
    agent inside one conversation. composite primary key retires the
    bespoke :class:`MemoryLedger` wrapper on top of 8.5l-1's composite-
    pk support: ``_id`` holds the ``(conversation_id, item_id)`` tuple
    so :class:`BaseCollection`'s tuple-aware pk path addresses L1 / L2
    / L3 uniformly.

    columns mirror the migration-v002 DDL with the v014 date-column
    rename: ``conversation_id`` UUID, ``item_id`` UUID, ``item_type``
    VARCHAR(50), ``short_desc`` VARCHAR(150), ``date_created``
    TIMESTAMPTZ, ``date_updated`` TIMESTAMPTZ. The original column
    name was ``date_added`` (v002) which read as "imported from
    elsewhere" -- v014 renamed it to the standard ``date_created`` /
    ``date_updated`` pair to match every other 3tears table and to
    satisfy ``BaseCollection.save``'s unconditional ``date_created``
    write at the L1 boundary.
    """

    primary_key_field: str = "conversation_id"

    def __init__(
        self,
        data: dict[str, Any],
        is_new: bool = True,
        collection: Any = None,
    ) -> None:
        """initialize entity with tuple ``_id`` for composite-pk lookup.

        :class:`BaseEntity.__init__` captures the single-pk field by
        name; composite-pk entities overwrite ``_id`` with the
        declared-order tuple so :meth:`BaseCollection.normalize_pk`
        and :meth:`BaseCollection.l2_key` address the row uniformly
        across tiers.

        :param data: row dict; must carry ``conversation_id`` and
            ``item_id`` keys
        :ptype data: dict[str, Any]
        :param is_new: whether entity is unsaved
        :ptype is_new: bool
        :param collection: owning collection reference
        :ptype collection: Any
        :return: nothing
        :rtype: None
        """
        super().__init__(data, is_new=is_new, collection=collection)
        object.__setattr__(
            self,
            "_id",
            (data["conversation_id"], data["item_id"]),
        )

    @property
    def conversation_id(self) -> UUID:
        """get the conversation UUID (first pk column).

        :return: conversation UUID
        :rtype: UUID
        """
        value = self._get_raw("conversation_id")
        if isinstance(value, UUID):
            return value
        return UUID(str(value))

    @conversation_id.setter
    def conversation_id(self, value: UUID) -> None:
        """set the conversation UUID.

        :param value: new conversation UUID
        :ptype value: UUID
        """
        BaseEntity.__setattr__(self, "conversation_id", value)

    @property
    def item_id(self) -> UUID:
        """get the item UUID (second pk column).

        :return: item UUID
        :rtype: UUID
        """
        value = self._get_raw("item_id")
        if isinstance(value, UUID):
            return value
        return UUID(str(value))

    @item_id.setter
    def item_id(self, value: UUID) -> None:
        """set the item UUID.

        :param value: new item UUID
        :ptype value: UUID
        """
        BaseEntity.__setattr__(self, "item_id", value)

    @property
    def item_type(self) -> str:
        """get the item-type discriminator (``memory`` / ``media`` / ``chunk``).

        :return: item type string
        :rtype: str
        """
        value: str = self._get_raw("item_type")
        return value

    @item_type.setter
    def item_type(self, value: str) -> None:
        """set the item-type discriminator.

        :param value: new item type
        :ptype value: str
        """
        BaseEntity.__setattr__(self, "item_type", value)

    @property
    def short_desc(self) -> str:
        """get the short description (truncated to 150 chars on save).

        :return: description text
        :rtype: str
        """
        value: str = self._get_raw("short_desc")
        return value

    @short_desc.setter
    def short_desc(self, value: str) -> None:
        """set the short description.

        :param value: new description text
        :ptype value: str
        """
        BaseEntity.__setattr__(self, "short_desc", value)

    @property
    def date_created(self) -> datetime:
        """get the timestamp when the reference was created.

        Renamed from ``date_added`` in v014 to match the standard
        3tears ``date_created`` convention and to satisfy
        ``BaseCollection.save``'s L1-write contract (which
        unconditionally writes ``date_created`` for every new row).

        :return: created datetime
        :rtype: datetime
        """
        value: datetime = self._get_raw("date_created")
        return value

    @date_created.setter
    def date_created(self, value: datetime) -> None:
        """set the created timestamp.

        :param value: new created datetime
        :ptype value: datetime
        """
        BaseEntity.__setattr__(self, "date_created", value)

    @property
    def date_updated(self) -> datetime:
        """get the timestamp when the reference was last updated.

        Added in v014 alongside the ``date_added`` -> ``date_created``
        rename to match the standard 3tears
        ``(date_created, date_updated)`` convention. For an
        append-only ledger this typically equals ``date_created``;
        the framework still writes both on save.

        :return: updated datetime
        :rtype: datetime
        """
        value: datetime = self._get_raw("date_updated")
        return value

    @date_updated.setter
    def date_updated(self, value: datetime) -> None:
        """set the updated timestamp.

        :param value: new updated datetime
        :ptype value: datetime
        """
        BaseEntity.__setattr__(self, "date_updated", value)
