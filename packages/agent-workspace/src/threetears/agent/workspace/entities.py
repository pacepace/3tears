"""workspace entities -- thin cache-proxy classes for workspace tables."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from threetears.core.entities.base import BaseEntity

__all__ = [
    "Workspace",
    "WorkspaceFile",
    "WorkspaceFileVersion",
]


def _as_uuid(value: object) -> UUID:
    """
    coerces value to UUID, handling strings from cache or wire layers.

    :param value: raw value that may be UUID or str
    :ptype value: object
    :return: UUID instance
    :rtype: UUID
    :raises ValueError: if value cannot be parsed as UUID
    """
    if isinstance(value, UUID):
        result: UUID = value
        return result
    return UUID(str(value))


def _as_uuid_or_none(value: object) -> UUID | None:
    """
    coerces value to UUID when present, returning None otherwise.

    :param value: raw value that may be UUID, str, or None
    :ptype value: object
    :return: UUID instance or None
    :rtype: UUID | None
    """
    if value is None:
        return None
    return _as_uuid(value)


class Workspace(BaseEntity):
    """cache-proxy entity for workspaces table.

    workspace-task-19 (WS-ACL-03) makes every workspace a platform-level
    namespace: the row in ``agent_<owner>.workspaces`` shares its primary
    key with a row in ``platform.namespaces`` and the latter carries the
    ``customer_id`` the authorization helper gates on. the
    :attr:`customer_id`, :attr:`owner_agent_id`, :attr:`created_by_user_id`
    and :attr:`namespace_name` properties expose these authorization
    dimensions so the entity directly satisfies
    :class:`~threetears.agent.workspace.authorize.WorkspaceLike`.

    :attr:`owner_agent_id` aliases the physical :attr:`agent_id` (the
    workspace lives in that agent's schema). :attr:`created_by_user_id`
    aliases the existing :attr:`created_by` column. :attr:`namespace_name`
    is deterministically derived from :attr:`id` as
    ``f"workspace.{id}"`` (matches v003 migration + broker discovery
    subject convention). :attr:`customer_id` is the one field that must
    be loaded from :class:`platform.namespaces` at resolve-time; the
    ``_resolve_workspace`` helper stamps it onto the entity via the
    ``customer_id`` setter after a single platform lookup.
    """

    primary_key_field: str = "id"

    @property
    def agent_id(self) -> UUID:
        """
        returns owning agent identifier for workspace.

        :return: agent UUID
        :rtype: UUID
        """
        return _as_uuid(self._get_raw("agent_id"))

    @agent_id.setter
    def agent_id(self, value: UUID) -> None:
        """
        sets owning agent identifier.

        :param value: agent UUID
        :ptype value: UUID
        """
        BaseEntity.__setattr__(self, "agent_id", value)

    @property
    def name(self) -> str:
        """
        returns workspace human-readable name.

        :return: workspace name
        :rtype: str
        """
        value: str = self._get_raw("name")
        return value

    @name.setter
    def name(self, value: str) -> None:
        """
        sets workspace human-readable name.

        :param value: workspace name
        :ptype value: str
        """
        BaseEntity.__setattr__(self, "name", value)

    @property
    def description(self) -> str | None:
        """
        returns optional workspace description.

        :return: description text or None
        :rtype: str | None
        """
        value: str | None = self._get_raw("description")
        return value

    @description.setter
    def description(self, value: str | None) -> None:
        """
        sets optional workspace description.

        :param value: description text or None
        :ptype value: str | None
        """
        BaseEntity.__setattr__(self, "description", value)

    @property
    def template_name(self) -> str | None:
        """
        returns template used to seed workspace, if any.

        :return: template name or None when workspace not seeded from template
        :rtype: str | None
        """
        value: str | None = self._get_raw("template_name")
        return value

    @template_name.setter
    def template_name(self, value: str | None) -> None:
        """
        sets template used to seed workspace.

        :param value: template name or None
        :ptype value: str | None
        """
        BaseEntity.__setattr__(self, "template_name", value)

    @property
    def created_by(self) -> UUID:
        """
        returns identifier of actor that created workspace.

        :return: creating actor UUID
        :rtype: UUID
        """
        return _as_uuid(self._get_raw("created_by"))

    @created_by.setter
    def created_by(self, value: UUID) -> None:
        """
        sets identifier of creating actor.

        :param value: creating actor UUID
        :ptype value: UUID
        """
        BaseEntity.__setattr__(self, "created_by", value)

    @property
    def current_version(self) -> int:
        """
        returns head workspace version pointer.

        :return: integer version number, starts at 0
        :rtype: int
        """
        value: int = self._get_raw("current_version")
        return value

    @current_version.setter
    def current_version(self, value: int) -> None:
        """
        sets head workspace version pointer.

        :param value: integer version number
        :ptype value: int
        """
        BaseEntity.__setattr__(self, "current_version", value)

    @property
    def date_created(self) -> datetime:
        """
        returns creation timestamp.

        :return: timezone-aware UTC datetime of creation
        :rtype: datetime
        """
        value: datetime = self._get_raw("date_created")
        return value

    @date_created.setter
    def date_created(self, value: datetime) -> None:
        """
        sets creation timestamp.

        :param value: timezone-aware UTC datetime
        :ptype value: datetime
        """
        BaseEntity.__setattr__(self, "date_created", value)

    @property
    def date_updated(self) -> datetime:
        """
        returns last-update timestamp.

        :return: timezone-aware UTC datetime of most recent write
        :rtype: datetime
        """
        value: datetime = self._get_raw("date_updated")
        return value

    @date_updated.setter
    def date_updated(self, value: datetime) -> None:
        """
        sets last-update timestamp.

        :param value: timezone-aware UTC datetime
        :ptype value: datetime
        """
        BaseEntity.__setattr__(self, "date_updated", value)

    @property
    def date_deleted(self) -> datetime | None:
        """
        returns soft-delete timestamp when set, None for live workspaces.

        soft-delete preserves the journal so history queries can still
        traverse a deleted workspace; list queries default-filter rows
        with non-null date_deleted.

        :return: timezone-aware UTC instant of soft-delete or None
        :rtype: datetime | None
        """
        value: datetime | None = self._get_raw("date_deleted")
        return value

    @date_deleted.setter
    def date_deleted(self, value: datetime | None) -> None:
        """
        sets soft-delete timestamp; None marks workspace as live.

        :param value: timezone-aware UTC datetime or None
        :ptype value: datetime | None
        """
        BaseEntity.__setattr__(self, "date_deleted", value)

    @property
    def owner_agent_id(self) -> UUID:
        """
        returns owning-agent UUID (alias of :attr:`agent_id`).

        exposed so the entity satisfies
        :class:`~threetears.agent.workspace.authorize.WorkspaceLike`
        without requiring a separate adapter; the physical storage home
        for a workspace IS its owner agent's schema.

        :return: owning agent UUID
        :rtype: UUID
        """
        return self.agent_id

    @property
    def created_by_user_id(self) -> UUID:
        """
        returns creating-user UUID (alias of :attr:`created_by`).

        workspaces-task-19 standardizes on ``created_by_user_id`` as the
        authorization-visible identity of the workspace creator; the
        underlying column name remains ``created_by`` for backwards
        compatibility with the v001 schema. this alias is the attribute
        the authorize helper's :class:`WorkspaceLike` protocol requires.

        :return: creating user UUID
        :rtype: UUID
        """
        return self.created_by

    @property
    def namespace_name(self) -> str:
        """
        returns canonical namespace name for this workspace.

        derived deterministically from :attr:`id` as
        ``"workspace.{id}"``; this is the form the v003 backfill
        migration + broker discovery subject uses, so every consumer
        (authorize cache lookup, L3 proxy routing, discovery queries)
        agrees on the key without a network round trip.

        :return: namespace name in the canonical ``workspace.<uuid>`` form
        :rtype: str
        """
        return f"workspace.{self.id}"

    @property
    def customer_id(self) -> UUID | None:
        """
        returns owning-customer UUID once resolved from platform.namespaces.

        unlike :attr:`owner_agent_id` and :attr:`created_by_user_id`
        which alias columns already on the agent-schema row, the customer
        dimension lives on the paired :class:`platform.namespaces` row.
        ``_resolve_workspace`` stamps the value via the :attr:`customer_id`
        setter after a single platform lookup before returning the
        entity to a tool's ``execute``. returns ``None`` when the entity
        was hydrated without a platform lookup (tests, direct L1 reads);
        the authorize helper treats that as an unroutable call and rejects.

        :return: customer UUID when resolved, else None
        :rtype: UUID | None
        """
        value = self._get_raw("customer_id")
        return _as_uuid_or_none(value)

    @customer_id.setter
    def customer_id(self, value: UUID | None) -> None:
        """
        stamps the resolved customer UUID onto the entity.

        invoked by ``_resolve_workspace`` once the platform.namespaces
        row for this workspace id has been fetched. the value is NOT
        persisted back to the agent schema (the column does not exist
        there); it lives only on the in-memory entity for the duration
        of the current tool call.

        :param value: customer UUID or None
        :ptype value: UUID | None
        """
        BaseEntity.__setattr__(self, "customer_id", value)


class WorkspaceFile(BaseEntity):
    """cache-proxy entity for workspace_files head-state table."""

    primary_key_field: str = "id"

    @property
    def workspace_id(self) -> UUID:
        """
        returns parent workspace identifier.

        :return: workspace UUID
        :rtype: UUID
        """
        return _as_uuid(self._get_raw("workspace_id"))

    @workspace_id.setter
    def workspace_id(self, value: UUID) -> None:
        """
        sets parent workspace identifier.

        :param value: workspace UUID
        :ptype value: UUID
        """
        BaseEntity.__setattr__(self, "workspace_id", value)

    @property
    def relative_path(self) -> str:
        """
        returns workspace-relative path key.

        :return: relative path string validated by sandbox at write time
        :rtype: str
        """
        value: str = self._get_raw("relative_path")
        return value

    @relative_path.setter
    def relative_path(self, value: str) -> None:
        """
        sets workspace-relative path key.

        :param value: relative path string
        :ptype value: str
        """
        BaseEntity.__setattr__(self, "relative_path", value)

    @property
    def content(self) -> bytes:
        """
        returns raw file content bytes.

        :return: file content bytes
        :rtype: bytes
        """
        value: bytes = self._get_raw("content")
        return value

    @content.setter
    def content(self, value: bytes) -> None:
        """
        sets raw file content bytes.

        :param value: file content bytes
        :ptype value: bytes
        """
        BaseEntity.__setattr__(self, "content", value)

    @property
    def sha256(self) -> str:
        """
        returns hex sha256 digest of content for optimistic concurrency.

        :return: 64-character hex digest
        :rtype: str
        """
        value: str = self._get_raw("sha256")
        return value

    @sha256.setter
    def sha256(self, value: str) -> None:
        """
        sets hex sha256 digest of content.

        :param value: 64-character hex digest
        :ptype value: str
        """
        BaseEntity.__setattr__(self, "sha256", value)

    @property
    def version(self) -> int:
        """
        returns monotonic version for path matching latest journal row.

        :return: integer version number
        :rtype: int
        """
        value: int = self._get_raw("version")
        return value

    @version.setter
    def version(self, value: int) -> None:
        """
        sets monotonic version for path.

        :param value: integer version number
        :ptype value: int
        """
        BaseEntity.__setattr__(self, "version", value)

    @property
    def date_updated(self) -> datetime:
        """
        returns last-update timestamp.

        :return: timezone-aware UTC datetime of most recent write
        :rtype: datetime
        """
        value: datetime = self._get_raw("date_updated")
        return value

    @date_updated.setter
    def date_updated(self, value: datetime) -> None:
        """
        sets last-update timestamp.

        :param value: timezone-aware UTC datetime
        :ptype value: datetime
        """
        BaseEntity.__setattr__(self, "date_updated", value)


class WorkspaceFileVersion(BaseEntity):
    """cache-proxy entity for workspace_file_versions append-only journal table."""

    primary_key_field: str = "id"

    @property
    def workspace_id(self) -> UUID:
        """
        returns parent workspace identifier.

        :return: workspace UUID
        :rtype: UUID
        """
        return _as_uuid(self._get_raw("workspace_id"))

    @workspace_id.setter
    def workspace_id(self, value: UUID) -> None:
        """
        sets parent workspace identifier.

        :param value: workspace UUID
        :ptype value: UUID
        """
        BaseEntity.__setattr__(self, "workspace_id", value)

    @property
    def relative_path(self) -> str:
        """
        returns workspace-relative path key for journal entry.

        :return: relative path string
        :rtype: str
        """
        value: str = self._get_raw("relative_path")
        return value

    @relative_path.setter
    def relative_path(self, value: str) -> None:
        """
        sets workspace-relative path key.

        :param value: relative path string
        :ptype value: str
        """
        BaseEntity.__setattr__(self, "relative_path", value)

    @property
    def version(self) -> int:
        """
        returns version number this journal row records.

        :return: integer version number
        :rtype: int
        """
        value: int = self._get_raw("version")
        return value

    @version.setter
    def version(self, value: int) -> None:
        """
        sets version number for journal row.

        :param value: integer version number
        :ptype value: int
        """
        BaseEntity.__setattr__(self, "version", value)

    @property
    def content(self) -> bytes:
        """
        returns file content bytes captured at this version.

        :return: content bytes
        :rtype: bytes
        """
        value: bytes = self._get_raw("content")
        return value

    @content.setter
    def content(self, value: bytes) -> None:
        """
        sets file content bytes for this version.

        :param value: content bytes
        :ptype value: bytes
        """
        BaseEntity.__setattr__(self, "content", value)

    @property
    def sha256(self) -> str:
        """
        returns hex sha256 digest of content at this version.

        :return: 64-character hex digest
        :rtype: str
        """
        value: str = self._get_raw("sha256")
        return value

    @sha256.setter
    def sha256(self, value: str) -> None:
        """
        sets hex sha256 digest of content.

        :param value: 64-character hex digest
        :ptype value: str
        """
        BaseEntity.__setattr__(self, "sha256", value)

    @property
    def action(self) -> str:
        """
        returns action verb for journal row.

        :return: one of create, update, delete, revert, checkpoint
        :rtype: str
        """
        value: str = self._get_raw("action")
        return value

    @action.setter
    def action(self, value: str) -> None:
        """
        sets action verb for journal row.

        :param value: one of create, update, delete, revert, checkpoint
        :ptype value: str
        """
        BaseEntity.__setattr__(self, "action", value)

    @property
    def label(self) -> str | None:
        """
        returns optional label text set only when action is checkpoint.

        :return: label string or None
        :rtype: str | None
        """
        value: str | None = self._get_raw("label")
        return value

    @label.setter
    def label(self, value: str | None) -> None:
        """
        sets optional label text.

        :param value: label string or None
        :ptype value: str | None
        """
        BaseEntity.__setattr__(self, "label", value)

    @property
    def actor_id(self) -> UUID:
        """
        returns actor identifier for journal row.

        :return: actor UUID, agent_id at minimum or user_id when available
        :rtype: UUID
        """
        return _as_uuid(self._get_raw("actor_id"))

    @actor_id.setter
    def actor_id(self, value: UUID) -> None:
        """
        sets actor identifier for journal row.

        :param value: actor UUID
        :ptype value: UUID
        """
        BaseEntity.__setattr__(self, "actor_id", value)

    @property
    def correlation_id(self) -> UUID:
        """
        returns correlation identifier from originating tool-call envelope.

        :return: correlation UUID
        :rtype: UUID
        """
        return _as_uuid(self._get_raw("correlation_id"))

    @correlation_id.setter
    def correlation_id(self, value: UUID) -> None:
        """
        sets correlation identifier.

        :param value: correlation UUID
        :ptype value: UUID
        """
        BaseEntity.__setattr__(self, "correlation_id", value)

    @property
    def date_created(self) -> datetime:
        """
        returns creation timestamp for journal row.

        :return: timezone-aware UTC datetime of creation
        :rtype: datetime
        """
        value: datetime = self._get_raw("date_created")
        return value

    @date_created.setter
    def date_created(self, value: datetime) -> None:
        """
        sets creation timestamp for journal row.

        :param value: timezone-aware UTC datetime
        :ptype value: datetime
        """
        BaseEntity.__setattr__(self, "date_created", value)
