"""SQLAlchemy model mixins for common patterns.

These mixins are optional conveniences — consuming apps can use plain
SQLAlchemy models without them. They encode the conventions used in
upstream services and recommended for new services.

Usage::

    from sqlalchemy.orm import DeclarativeBase, Mapped
    from threetears.core.models import UUIDPrimaryKeyMixin, TimestampMixin, SoftDeleteMixin

    class Base(DeclarativeBase):
        pass

    class UserModel(UUIDPrimaryKeyMixin, TimestampMixin, Base):
        __tablename__ = "users"
        email: Mapped[str] = mapped_column(Text, nullable=False)

    class MemoryModel(UUIDPrimaryKeyMixin, TimestampMixin, SoftDeleteMixin, Base):
        __tablename__ = "memories"
        content: Mapped[str] = mapped_column(Text, nullable=False)
"""

from __future__ import annotations

from typing import Optional

from sqlalchemy import Boolean, DateTime, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

__all__ = [
    "SoftDeleteMixin",
    "TimestampMixin",
    "UUIDPrimaryKeyMixin",
]


class UUIDPrimaryKeyMixin:
    """Adds a UUID primary key column.

    By default the column is named ``id``. Subclasses can override
    ``_pk_column_name`` to use a different name (e.g. ``user_id``).

    The UUID is generated server-side via ``gen_random_uuid()`` on
    PostgreSQL. For SQLite (L1 cache), the application layer provides
    the UUID.
    """

    _pk_column_name: str = "id"

    # Using __init_subclass__ to allow column name customization is fragile
    # with SQLAlchemy's metaclass. Instead, override at the class level:
    #   _pk_column_name = "user_id"
    # and re-declare the mapped_column.

    id: Mapped[object] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=func.gen_random_uuid(),
        sort_order=-100,
    )


class TimestampMixin:
    """Adds ``date_created`` and ``date_updated`` timestamp columns.

    ``date_created`` defaults to the current time on INSERT.
    ``date_updated`` defaults to the current time on INSERT and is
    updated on every UPDATE via ``onupdate``.

    Both use timezone-aware ``DateTime(timezone=True)`` matching the
    convention of ``TIMESTAMPTZ``.
    """

    date_created: Mapped[object] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        sort_order=900,
    )
    date_updated: Mapped[object] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
        sort_order=901,
    )


class SoftDeleteMixin:
    """Adds soft-delete columns: ``is_deleted`` and ``date_deleted``.

    ``is_deleted`` defaults to ``False``. When an entity is soft-deleted,
    set ``is_deleted = True`` and ``date_deleted`` to the current time.

    This mixin does NOT enforce soft-delete filtering — that is the
    responsibility of the collection or query layer.
    """

    is_deleted: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default="false",
        sort_order=910,
    )
    date_deleted: Mapped[Optional[object]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        sort_order=911,
    )
