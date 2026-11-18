"""PostgreSQL's SQLAlchemy dialect, for a test compiling DDL as a migration emits it."""

from __future__ import annotations

from sqlalchemy.dialects import postgresql
from sqlalchemy.engine.interfaces import Dialect


def pg_dialect() -> Dialect:
    """PostgreSQL's dialect (``postgresql.dialect()``), typed: SQLAlchemy leaves the dialect classes' constructors
    unannotated, so the call goes through the typed :class:`Dialect` it is one of.

    :return: a PostgreSQL dialect
    :rtype: Dialect
    """
    dialect_class: type[Dialect] = postgresql.dialect
    return dialect_class()
