"""Unit + parity tests for :meth:`TableSchema.to_sqlalchemy_table`.

The shape conformance bar: feeding a hand-enriched :class:`TableSchema`
that mirrors one of the v0.7.5 hand-written SQLAlchemy table factories
through :meth:`TableSchema.to_sqlalchemy_table` MUST produce a Table
that is structurally identical (column types, PK, FKs, indexes,
server defaults) to what the factory produces. The parity tests are
the load-bearing regression net for the v0.8.0 release: if any of
them passes when the conversion regresses, the test isn't doing its
job.

Per shard 02 design, the equivalence helper reads Index
``postgresql_where`` via ``dialect_kwargs`` (public, stable) NOT
``dialect_options`` (internal-state, versioned shape) and avoids any
``__repr__``-driven equality (SQLAlchemy's repr drifts).
"""

from __future__ import annotations

from typing import Any

import pytest
import sqlalchemy as sa

from threetears.agent.memory.collections import (
    conversation_memory_refs_table,
    media_content_table,
    media_table,
    memories_table,
    memory_chunks_table,
)
from threetears.agent.tools.collections import context_items_table
from threetears.mcp.rbac import mcp_tool_grants_table
from threetears.core.collections.schema_backed import (
    BOOL_TYPE,
    BYTES_TYPE,
    DATETIMETZ_TYPE,
    ENUM_TYPE,
    INT_TYPE,
    JSONB_TYPE,
    NUMERIC_TYPE,
    STRING_TYPE,
    TSVECTOR_TYPE,
    UUID_TYPE,
    VECTOR_TYPE,
    Column,
    ForeignKey,
    Index,
    TableSchema,
)


# ---------------------------------------------------------------------------
# structural-equivalence helpers (per shard-02 §"Parity tests")
# ---------------------------------------------------------------------------


def _index_signature(
    idx: sa.Index,
) -> tuple[str, frozenset[str], bool, str | None]:
    """return a structural signature for a SA Index.

    reads ``postgresql_where`` via the public ``dialect_kwargs`` API
    (NOT ``dialect_options``, which is SQLAlchemy-internal state with
    a versioned shape).

    :param idx: SQLAlchemy ``Index``
    :ptype idx: sqlalchemy.Index
    :return: ``(name, column-name-set, unique, compiled-WHERE-or-None)``
    :rtype: tuple[str, frozenset[str], bool, str | None]
    """
    where = idx.dialect_kwargs.get("postgresql_where")
    where_str: str | None
    if where is None:
        where_str = None
    else:
        # ``sa.text(...)`` clauses round-trip cleanly via compile() so
        # the comparison is value-based not identity-based.
        where_str = str(
            where.compile(compile_kwargs={"literal_binds": True}),
        )
    return (
        idx.name,
        frozenset(c.name for c in idx.columns),
        bool(idx.unique),
        where_str,
    )


def _fk_constraint_signature(
    constraint: sa.ForeignKeyConstraint,
) -> tuple[tuple[str, ...], str, tuple[str, ...], str | None]:
    """return a structural signature for a table-level FK constraint.

    Reads the referenced ``table.col`` pair from each
    ``ForeignKey.target_fullname`` string rather than via ``.column``.
    The latter resolves the reference through the Table's MetaData
    and raises :class:`NoReferencedTableError` when the target table
    is not registered on the same MetaData -- which is normal in the
    parity tests because the v0.7.5 factories declare FKs to
    ``users`` / ``messages`` / etc. tables that the metallm app
    registers separately and 3tears tests never load.

    :param constraint: SQLAlchemy ``ForeignKeyConstraint``
    :ptype constraint: sqlalchemy.ForeignKeyConstraint
    :return: ``(local_cols, ref_table, ref_cols, ondelete)``
    :rtype: tuple[tuple[str, ...], str, tuple[str, ...], str | None]
    """
    local_cols = tuple(constraint.column_keys)
    # ``ForeignKey.target_fullname`` is the public string accessor for
    # the ``table.col`` reference; it does NOT resolve through
    # MetaData (so the referenced table need not be registered to
    # introspect the structural shape).
    ref_pairs = [el.target_fullname.split(".", 1) for el in constraint.elements]
    ref_table = ref_pairs[0][0]
    ref_cols = tuple(pair[1] for pair in ref_pairs)
    return (local_cols, ref_table, ref_cols, constraint.ondelete)


def _inline_fk_signatures(
    table: sa.Table,
) -> set[tuple[str, str, str | None]]:
    """collect ``(local_col, target_fullname, ondelete)`` for inline FKs.

    inline ``ForeignKey`` objects sit on the column, not on a
    table-level :class:`ForeignKeyConstraint`. SQLAlchemy materialises
    each inline FK into a single-column ``ForeignKeyConstraint`` AS
    WELL, which is what ``table.constraints`` exposes -- but the
    inline form retains an identity on the column and we capture both
    for byte-for-byte structural parity.

    :param table: SQLAlchemy ``Table``
    :ptype table: sqlalchemy.Table
    :return: set of inline FK signatures
    :rtype: set[tuple[str, str, str | None]]
    """
    sigs: set[tuple[str, str, str | None]] = set()
    for col in table.columns:
        for fk in col.foreign_keys:
            sigs.add((col.name, fk.target_fullname, fk.ondelete))
    return sigs


def _column_signature(
    col: sa.Column,
) -> tuple[
    str,
    str,
    bool,
    bool,
    str | None,
    int | None,
    tuple[str, ...],
    tuple[int | None, int | None],
]:
    """return a structural signature for a SA Column.

    Captures structural fields that drive Alembic auto-gen decisions
    AND the type-parameter axes (vector dim, enum values, numeric
    precision/scale) that ``type(col.type).__name__`` collapses --
    pgvector renders every ``Vector(N)`` as type-class ``VECTOR``,
    every ``Enum(*vals)`` as ``Enum``, and every ``Numeric(p, s)`` as
    ``Numeric``. Without the appended fields a shard-03 enrichment
    regression that wrote ``vector_dim=768`` instead of ``1024``, or
    dropped an enum value, or changed numeric scale, would pass the
    parity test silently and only surface at the shard-05 metallm
    Alembic auto-gen step (too late).

    :param col: SQLAlchemy ``Column``
    :ptype col: sqlalchemy.Column
    :return: ``(name, type-class-name, primary_key, nullable,
        server_default_text-or-None, vector_dim-or-None,
        enum_values_tuple, (numeric_precision, numeric_scale))``
    :rtype: tuple[str, str, bool, bool, str | None, int | None,
        tuple[str, ...], tuple[int | None, int | None]]
    """
    sd: str | None
    if col.server_default is not None:
        # ``server_default.arg`` is either a string (literal) or a
        # ``TextClause`` for ``sa.text(...)`` defaults; cast to str for
        # comparison either way.
        arg = col.server_default.arg
        sd = str(arg)
    else:
        sd = None
    # type-parameter axes -- ``getattr`` with default so non-vector /
    # non-enum / non-numeric columns yield None / () / (None, None)
    # rather than raising AttributeError.
    vector_dim: int | None = getattr(col.type, "dim", None)
    enum_values: tuple[str, ...] = tuple(getattr(col.type, "enums", ()))
    numeric_ps: tuple[int | None, int | None] = (
        getattr(col.type, "precision", None),
        getattr(col.type, "scale", None),
    )
    return (
        col.name,
        type(col.type).__name__,
        bool(col.primary_key),
        bool(col.nullable),
        sd,
        vector_dim,
        enum_values,
        numeric_ps,
    )


def _assert_tables_equivalent(
    a: sa.Table,
    b: sa.Table,
    *,
    extra_fks_in_b: frozenset[tuple[tuple[str, ...], str, tuple[str, ...], str | None]] = frozenset(),
) -> None:
    """assert two SQLAlchemy Tables are structurally identical.

    compares column names, types, primary-key columns, indexes,
    table-level FK constraints, and inline FKs. ignores Table
    metadata identity (the Tables are intentionally registered on
    different :class:`MetaData` instances).

    :param a: reference Table (typically the v0.7.5 hand-written
        factory output)
    :ptype a: sqlalchemy.Table
    :param b: candidate Table (typically ``to_sqlalchemy_table`` output)
    :ptype b: sqlalchemy.Table
    :param extra_fks_in_b: FK signatures that are EXPECTED to appear
        only on ``b`` because the v0.7.5 factory documented but did
        NOT emit them. Used by ``media_content`` / ``memory_chunks``
        whose composite FKs to ``media`` / ``memories`` live only in
        factory comments. Per shard-02 §"Critical correctness note
        for composite FKs": enrichment MUST surface these so prod
        Alembic auto-gen produces zero phantom migrations.
    :ptype extra_fks_in_b: frozenset[tuple]
    :return: nothing
    :rtype: None
    :raises AssertionError: when any structural field diverges
    """
    assert a.name == b.name, f"table name mismatch: {a.name!r} vs {b.name!r}"

    # column-name sets
    a_col_names = {c.name for c in a.columns}
    b_col_names = {c.name for c in b.columns}
    assert a_col_names == b_col_names, (
        f"column-name sets differ on {a.name}: "
        f"only-in-A={a_col_names - b_col_names!r}, "
        f"only-in-B={b_col_names - a_col_names!r}"
    )

    # per-column signature
    a_col_sigs = {c.name: _column_signature(c) for c in a.columns}
    b_col_sigs = {c.name: _column_signature(c) for c in b.columns}
    for name in a_col_names:
        assert a_col_sigs[name] == b_col_sigs[name], (
            f"column signature mismatch on {a.name}.{name}: "
            f"factory={a_col_sigs[name]!r} vs to_sqla={b_col_sigs[name]!r}"
        )

    # PK column sets
    a_pks = {c.name for c in a.primary_key.columns}
    b_pks = {c.name for c in b.primary_key.columns}
    assert a_pks == b_pks, f"primary-key column sets differ on {a.name}: factory={a_pks!r} vs to_sqla={b_pks!r}"

    # indexes (by signature)
    a_idx_sigs = {_index_signature(i) for i in a.indexes}
    b_idx_sigs = {_index_signature(i) for i in b.indexes}
    assert a_idx_sigs == b_idx_sigs, (
        f"index-signature sets differ on {a.name}: "
        f"only-in-factory={a_idx_sigs - b_idx_sigs!r}, "
        f"only-in-to_sqla={b_idx_sigs - a_idx_sigs!r}"
    )

    # table-level FK constraints
    a_fk_sigs = {_fk_constraint_signature(c) for c in a.constraints if isinstance(c, sa.ForeignKeyConstraint)}
    b_fk_sigs = {_fk_constraint_signature(c) for c in b.constraints if isinstance(c, sa.ForeignKeyConstraint)}
    # extra FKs in b are expected (factory comments → real DDL).
    # everything else must match.
    expected_a = a_fk_sigs
    expected_b = a_fk_sigs | set(extra_fks_in_b)
    assert expected_b == b_fk_sigs, (
        f"FK-constraint sets differ on {a.name}: "
        f"only-in-factory={expected_a - b_fk_sigs!r}, "
        f"only-in-to_sqla-unexpected={b_fk_sigs - expected_b!r}"
    )

    # inline FK signatures -- compare only the FKs that are NOT part
    # of an extra (composite-augmentation) FK. SQLAlchemy attaches an
    # inline ``ForeignKey`` to each column participating in a
    # table-level ``ForeignKeyConstraint``; those bleed into
    # _inline_fk_signatures(b) and would falsely diverge from a.
    a_inline = {sig for sig in _inline_fk_signatures(a) if not _inline_in_extra(sig, extra_fks_in_b)}
    b_inline = {sig for sig in _inline_fk_signatures(b) if not _inline_in_extra(sig, extra_fks_in_b)}
    assert a_inline == b_inline, (
        f"inline FK sets differ on {a.name}: "
        f"only-in-factory={a_inline - b_inline!r}, "
        f"only-in-to_sqla={b_inline - a_inline!r}"
    )


def _inline_in_extra(
    sig: tuple[str, str, str | None],
    extra: frozenset[tuple[tuple[str, ...], str, tuple[str, ...], str | None]],
) -> bool:
    """true iff inline-FK ``sig`` is part of one of the composite
    augmentation FKs in ``extra``.

    SQLAlchemy attaches an inline ``ForeignKey`` record to each local
    column participating in a composite ``ForeignKeyConstraint``; the
    parity helper must filter those out so the augmentation FKs
    don't bleed into the per-column inline-FK comparison.

    :param sig: ``(local_col, target_fullname, ondelete)``
    :param extra: known composite augmentation FK signatures
    :return: True when ``sig`` is part of any augmentation
    """
    local_col, target_fullname, ondelete = sig
    target_table, _, target_col = target_fullname.partition(".")
    for ex_locals, ex_table, ex_cols, ex_ondelete in extra:
        if ex_table != target_table:
            continue
        if ex_ondelete != ondelete:
            continue
        # local_col must be one of ex_locals at the same position as
        # target_col is in ex_cols.
        if local_col in ex_locals and target_col in ex_cols:
            return True
    return False


# ---------------------------------------------------------------------------
# enriched TableSchemas mirroring each v0.7.5 factory
# ---------------------------------------------------------------------------


def _memories_schema() -> TableSchema:
    """build the enriched TableSchema mirroring ``memories_table``.

    ``message_id_source`` FK is table-level with ``on_delete="SET NULL"``
    because the inline 2-tuple ``foreign_key=`` form carries no
    ``on_delete=``. Prod metallm (alembic
    ``memories_message_id_source_fkey``) declares SET NULL; v0.8.0 the
    factory was updated to match — the table-level form is the only
    way to express the prod shape in this declaration.
    """
    return TableSchema(
        name="memories",
        primary_key=("memory_id", "agent_id"),
        columns=[
            Column("memory_id", UUID_TYPE),
            Column("agent_id", UUID_TYPE, partition=True),
            Column("customer_id", UUID_TYPE),
            Column(
                "user_id",
                UUID_TYPE,
                foreign_key=("users", "user_id"),
            ),
            Column("conversation_id", UUID_TYPE),
            Column(
                "message_id_source",
                UUID_TYPE,
                nullable=True,
            ),
            Column(
                "type_memory",
                ENUM_TYPE,
                enum_type=(
                    "preference",
                    "fact",
                    "decision",
                    "topical_context",
                    "relational_context",
                ),
                enum_name="memory_type",
            ),
            Column("content", STRING_TYPE),
            Column("summary", STRING_TYPE, nullable=True),
            Column(
                "embedding",
                VECTOR_TYPE,
                vector_dim=1024,
                nullable=True,
            ),
            Column(
                "search_vector",
                TSVECTOR_TYPE,
                nullable=True,
                immutable=True,
            ),
            Column("alias", STRING_TYPE, nullable=True),
            Column("date_created", DATETIMETZ_TYPE, immutable=True),
            Column("date_updated", DATETIMETZ_TYPE, nullable=True),
        ],
        foreign_keys=(
            ForeignKey(
                "message_id_source",
                "messages",
                "message_id",
                on_delete="SET NULL",
            ),
        ),
        indexes=(Index("ix_memories_user_date", "user_id", "date_created"),),
    )


def _media_schema() -> TableSchema:
    """build the enriched TableSchema mirroring ``media_table``.

    The single-column FKs with non-default ``ondelete`` (``memory_id``
    → CASCADE, ``cloud_connection_id`` → SET NULL) are declared as
    table-level :class:`ForeignKey` factories because
    :attr:`Column.foreign_key` (the inline 2-tuple) carries no
    ``ondelete`` -- per the v0.8.0 locked decision the two FK shapes
    are intentional: terse inline for NO ACTION, table-level for
    everything else.
    """
    return TableSchema(
        name="media",
        primary_key=("agent_id", "media_id"),
        columns=[
            Column("agent_id", UUID_TYPE, partition=True),
            Column("media_id", UUID_TYPE),
            Column("customer_id", UUID_TYPE),
            Column(
                "user_id",
                UUID_TYPE,
                foreign_key=("users", "user_id"),
            ),
            Column("s3_key", STRING_TYPE, nullable=True),
            Column("mime_type", STRING_TYPE),
            Column("size_bytes", INT_TYPE),
            Column("source", STRING_TYPE),
            Column(
                "metadata_json",
                JSONB_TYPE,
                server_default="{}",
            ),
            Column("generation_prompt", STRING_TYPE, nullable=True),
            Column(
                "media_category",
                STRING_TYPE,
                server_default="image",
            ),
            Column(
                "extraction_status",
                STRING_TYPE,
                server_default="none",
            ),
            Column("thumbnail_s3_key", STRING_TYPE, nullable=True),
            Column(
                "cloud_connection_id",
                UUID_TYPE,
                nullable=True,
            ),
            Column("cloud_file_id", STRING_TYPE, nullable=True),
            Column("cloud_file_url", STRING_TYPE, nullable=True),
            Column("memory_id", UUID_TYPE),
            Column("date_created", DATETIMETZ_TYPE, immutable=True),
        ],
        foreign_keys=(
            ForeignKey(
                "cloud_connection_id",
                "cloud_connections",
                "cloud_connection_id",
                on_delete="SET NULL",
            ),
            ForeignKey(
                "memory_id",
                "memories",
                "memory_id",
                on_delete="CASCADE",
            ),
        ),
        indexes=(
            Index("ix_media_user_date", "user_id", "date_created"),
            Index("ix_media_mime_type", "mime_type"),
            Index("ix_media_memory_id", "memory_id"),
            Index(
                "uq_media_cloud_connection_file",
                "cloud_connection_id",
                "cloud_file_id",
                unique=True,
            ),
        ),
    )


def _media_content_schema() -> TableSchema:
    """build the enriched TableSchema mirroring ``media_content_table``.

    Carries the composite FK ``(agent_id, media_id) → media`` that the
    factory only documents in a comment but never emits at SQLAlchemy
    level. The parity test below augments the factory-comparison with
    a direct assertion on the composite FK so the prod-shape
    requirement does not slip through a trivially-passing comparison.
    """
    return TableSchema(
        name="media_content",
        primary_key=("agent_id", "content_id"),
        columns=[
            Column("agent_id", UUID_TYPE, partition=True),
            Column("content_id", UUID_TYPE),
            Column("customer_id", UUID_TYPE),
            Column("media_id", UUID_TYPE),
            Column(
                "user_id",
                UUID_TYPE,
                foreign_key=("users", "user_id"),
            ),
            Column("content_type", STRING_TYPE),
            Column("content", STRING_TYPE),
            Column("summary", STRING_TYPE, nullable=True),
            Column(
                "embedding",
                VECTOR_TYPE,
                vector_dim=1024,
                nullable=True,
            ),
            Column(
                "search_vector",
                TSVECTOR_TYPE,
                nullable=True,
                immutable=True,
            ),
            Column(
                "model_id",
                UUID_TYPE,
                nullable=True,
                foreign_key=("models", "model_id"),
            ),
            Column(
                "provider_id",
                UUID_TYPE,
                nullable=True,
                foreign_key=("providers", "provider_id"),
            ),
            Column("model_name", STRING_TYPE, nullable=True),
            Column("provider_name", STRING_TYPE, nullable=True),
            Column("token_count_prompt", INT_TYPE, nullable=True),
            Column("token_count_completion", INT_TYPE, nullable=True),
            Column(
                "cost",
                NUMERIC_TYPE,
                precision=12,
                scale=8,
                nullable=True,
            ),
            Column("metadata_json", JSONB_TYPE, nullable=True),
            Column("date_created", DATETIMETZ_TYPE, immutable=True),
        ],
        foreign_keys=(
            ForeignKey(
                ("agent_id", "media_id"),
                "media",
                ("agent_id", "media_id"),
                on_delete="CASCADE",
            ),
        ),
        indexes=(
            Index(
                "ix_media_content_media_type",
                "media_id",
                "content_type",
            ),
            Index("ix_media_content_user", "user_id"),
        ),
    )


def _memory_chunks_schema() -> TableSchema:
    """build the enriched TableSchema mirroring ``memory_chunks_table``.

    Carries the composite FK ``(agent_id, memory_id) → memories`` that
    the factory only documents in a comment but never emits at
    SQLAlchemy level.
    """
    return TableSchema(
        name="memory_chunks",
        primary_key=("agent_id", "chunk_id"),
        columns=[
            Column("agent_id", UUID_TYPE, partition=True),
            Column("chunk_id", UUID_TYPE),
            Column("customer_id", UUID_TYPE),
            Column("memory_id", UUID_TYPE),
            Column(
                "user_id",
                UUID_TYPE,
                foreign_key=("users", "user_id"),
            ),
            Column("chunk_index", INT_TYPE),
            Column("content", STRING_TYPE),
            Column("summary", STRING_TYPE, nullable=True),
            Column("heading_context", STRING_TYPE, nullable=True),
            Column("page_number", INT_TYPE, nullable=True),
            Column("token_count", INT_TYPE),
            Column(
                "embedding",
                VECTOR_TYPE,
                vector_dim=1024,
                nullable=True,
            ),
            Column(
                "search_vector",
                TSVECTOR_TYPE,
                nullable=True,
                immutable=True,
            ),
            Column("message_id_start", UUID_TYPE, nullable=True),
            Column("message_id_end", UUID_TYPE, nullable=True),
            Column("date_created", DATETIMETZ_TYPE, immutable=True),
        ],
        foreign_keys=(
            ForeignKey(
                ("agent_id", "memory_id"),
                "memories",
                ("agent_id", "memory_id"),
                on_delete="CASCADE",
            ),
        ),
        indexes=(
            Index(
                "ix_memory_chunks_memory",
                "memory_id",
                "chunk_index",
            ),
            Index("ix_memory_chunks_user", "user_id"),
        ),
    )


def _conversation_memory_refs_schema() -> TableSchema:
    """build the enriched TableSchema mirroring
    ``conversation_memory_refs_table``.

    Shape mirrors prod (FK on ``conversation_id`` with CASCADE, lookup
    index on ``conversation_id``, ``date_created`` immutable with
    ``server_default="now()"``). The hand-written factory does not
    emit these directly — they live in prod metallm Alembic.

    Production ``conversation_memory_refs.date_created`` is
    ``TIMESTAMPTZ NOT NULL DEFAULT now()`` and the FK has
    ``ON DELETE CASCADE`` (alembic
    ``conversation_memory_refs_conversation_id_fkey``); the v0.8.0
    ``MemoryRefsCollection.schema`` carries the same shape, and v021
    pins the ``date_created`` server default.
    """
    # TODO(v0.8.0 shard 04): replace with
    # _reference_<table>_table per shard 04 §"Concrete steps per
    # factory". Production schema now carries FK + indexes this
    # helper does not — keep this helper in sync until shard 04
    # collapses every factory onto :meth:`TableSchema.to_sqlalchemy_table`.
    return TableSchema(
        name="conversation_memory_refs",
        primary_key=("conversation_id", "item_id"),
        columns=[
            Column("conversation_id", UUID_TYPE),
            Column("item_id", UUID_TYPE),
            Column("item_type", STRING_TYPE),
            Column("short_desc", STRING_TYPE),
            Column(
                "date_created",
                DATETIMETZ_TYPE,
                immutable=True,
                server_default="now()",
            ),
            Column("date_updated", DATETIMETZ_TYPE),
        ],
        foreign_keys=(
            ForeignKey(
                "conversation_id",
                "conversations",
                "conversation_id",
                on_delete="CASCADE",
            ),
        ),
        indexes=(
            Index(
                "ix_conversation_memory_refs_cid",
                "conversation_id",
            ),
        ),
    )


def _mcp_tool_grants_schema() -> TableSchema:
    """build the enriched TableSchema mirroring ``mcp_tool_grants_table``.

    Shape mirrors prod (v001 migration creates the table with
    ``date_created TIMESTAMPTZ NOT NULL DEFAULT now()`` and two
    lookup indexes). The v0.8.0 ``McpToolGrantCollection.schema``
    carries the same shape.
    """
    return TableSchema(
        name="mcp_tool_grants",
        primary_key="grant_id",
        columns=[
            Column("grant_id", UUID_TYPE),
            Column("principal_type", STRING_TYPE),
            Column("principal_id", UUID_TYPE),
            Column("tool_name", STRING_TYPE),
            Column("permission", STRING_TYPE),
            Column(
                "date_created",
                DATETIMETZ_TYPE,
                immutable=True,
                server_default="now()",
            ),
        ],
        indexes=(
            Index(
                "idx_mcp_tool_grants_principal",
                "principal_id",
                "permission",
            ),
            Index("idx_mcp_tool_grants_tool", "tool_name"),
        ),
    )


def _context_items_schema() -> TableSchema:
    """build the enriched TableSchema mirroring ``context_items_table``."""
    # TODO(v0.8.0 shard 04): replace with
    # _reference_<table>_table per shard 04 §"Concrete steps per
    # factory". Production schema now carries FK + indexes this
    # helper does not.
    return TableSchema(
        name="context_items",
        primary_key=("conversation_id", "context_id"),
        columns=[
            Column("conversation_id", UUID_TYPE),
            Column("context_id", UUID_TYPE),
            Column("context_type", STRING_TYPE),
            Column("key", STRING_TYPE),
            Column("short_desc", STRING_TYPE),
            Column(
                "long_desc",
                STRING_TYPE,
                server_default="",
            ),
            Column("content", STRING_TYPE),
            Column("metadata", JSONB_TYPE, nullable=True),
            Column("date_accessed", DATETIMETZ_TYPE),
            Column("date_created", DATETIMETZ_TYPE, immutable=True),
            Column("date_updated", DATETIMETZ_TYPE),
        ],
    )


# ---------------------------------------------------------------------------
# parity tests (REGRESSION NET for the v0.8.0 release)
# ---------------------------------------------------------------------------


def test_parity_memories_table() -> None:
    """enriched TableSchema produces a Table structurally equal to
    ``memories_table``.
    """
    via_factory = memories_table(sa.MetaData())
    via_to_sqla = _memories_schema().to_sqlalchemy_table(sa.MetaData())
    _assert_tables_equivalent(via_factory, via_to_sqla)


def test_parity_media_table() -> None:
    """enriched TableSchema produces a Table structurally equal to
    ``media_table``.
    """
    via_factory = media_table(sa.MetaData())
    via_to_sqla = _media_schema().to_sqlalchemy_table(sa.MetaData())
    _assert_tables_equivalent(via_factory, via_to_sqla)


def test_parity_media_content_table() -> None:
    """enriched TableSchema produces a Table structurally equal to
    ``media_content_table`` AND carries the composite FK that the
    factory only documents in a comment.
    """
    via_factory = media_content_table(sa.MetaData())
    via_to_sqla = _media_content_schema().to_sqlalchemy_table(sa.MetaData())
    _assert_tables_equivalent(
        via_factory,
        via_to_sqla,
        extra_fks_in_b=frozenset(
            {
                (
                    ("agent_id", "media_id"),
                    "media",
                    ("agent_id", "media_id"),
                    "CASCADE",
                ),
            },
        ),
    )

    # ground-truth augmentation: the v0.7.5 factory documents the
    # composite FK in a comment but does NOT emit it. enrichment MUST
    # carry the composite FK so prod auto-gen produces zero phantom
    # migrations.
    composite_fks = [
        c for c in via_to_sqla.constraints if isinstance(c, sa.ForeignKeyConstraint) and len(c.column_keys) > 1
    ]
    assert any(
        tuple(c.column_keys) == ("agent_id", "media_id")
        and c.elements[0].target_fullname.split(".", 1)[0] == "media"
        and c.ondelete == "CASCADE"
        for c in composite_fks
    ), "composite FK (agent_id, media_id) → media missing from media_content"


def test_parity_memory_chunks_table() -> None:
    """enriched TableSchema produces a Table structurally equal to
    ``memory_chunks_table`` AND carries the composite FK that the
    factory only documents in a comment.
    """
    via_factory = memory_chunks_table(sa.MetaData())
    via_to_sqla = _memory_chunks_schema().to_sqlalchemy_table(sa.MetaData())
    _assert_tables_equivalent(
        via_factory,
        via_to_sqla,
        extra_fks_in_b=frozenset(
            {
                (
                    ("agent_id", "memory_id"),
                    "memories",
                    ("agent_id", "memory_id"),
                    "CASCADE",
                ),
            },
        ),
    )

    composite_fks = [
        c for c in via_to_sqla.constraints if isinstance(c, sa.ForeignKeyConstraint) and len(c.column_keys) > 1
    ]
    assert any(
        tuple(c.column_keys) == ("agent_id", "memory_id")
        and c.elements[0].target_fullname.split(".", 1)[0] == "memories"
        and c.ondelete == "CASCADE"
        for c in composite_fks
    ), "composite FK (agent_id, memory_id) → memories missing from memory_chunks"


def test_parity_conversation_memory_refs_table() -> None:
    """enriched TableSchema produces a Table structurally equal to
    ``conversation_memory_refs_table``.
    """
    via_factory = conversation_memory_refs_table(sa.MetaData())
    via_to_sqla = _conversation_memory_refs_schema().to_sqlalchemy_table(
        sa.MetaData(),
    )
    _assert_tables_equivalent(via_factory, via_to_sqla)


def test_parity_mcp_tool_grants_table() -> None:
    """enriched TableSchema produces a Table structurally equal to
    ``mcp_tool_grants_table``.
    """
    via_factory = mcp_tool_grants_table(sa.MetaData())
    via_to_sqla = _mcp_tool_grants_schema().to_sqlalchemy_table(sa.MetaData())
    _assert_tables_equivalent(via_factory, via_to_sqla)


def test_parity_context_items_table() -> None:
    """enriched TableSchema produces a Table structurally equal to
    ``context_items_table``.
    """
    via_factory = context_items_table(sa.MetaData())
    via_to_sqla = _context_items_schema().to_sqlalchemy_table(sa.MetaData())
    _assert_tables_equivalent(via_factory, via_to_sqla)


# ---------------------------------------------------------------------------
# unit tests (small isolated cases)
# ---------------------------------------------------------------------------


def test_to_sqla_idempotent() -> None:
    """calling :meth:`to_sqlalchemy_table` twice on the same metadata
    returns the existing Table on the second call (no duplicate-table
    error).
    """
    schema = TableSchema(
        name="t",
        primary_key="a",
        columns=[
            Column("a", UUID_TYPE),
            Column("b", STRING_TYPE, nullable=True),
        ],
    )
    md = sa.MetaData()
    first = schema.to_sqlalchemy_table(md)
    second = schema.to_sqlalchemy_table(md)
    assert first is second, "to_sqlalchemy_table is not idempotent: second call returned a different Table object"


def test_to_sqla_composite_pk() -> None:
    """schema with ``primary_key=("a", "b")`` produces a Table where
    both columns are flagged ``primary_key=True``.
    """
    schema = TableSchema(
        name="t",
        primary_key=("a", "b"),
        columns=[
            Column("a", UUID_TYPE),
            Column("b", UUID_TYPE),
            Column("c", STRING_TYPE, nullable=True),
        ],
    )
    t = schema.to_sqlalchemy_table(sa.MetaData())
    pks = {c.name for c in t.primary_key.columns}
    assert pks == {"a", "b"}
    assert t.c.a.primary_key is True
    assert t.c.b.primary_key is True
    assert t.c.c.primary_key is False


def test_to_sqla_single_pk() -> None:
    """schema with ``primary_key="a"`` produces a Table where only
    column ``a`` is flagged PK.
    """
    schema = TableSchema(
        name="t",
        primary_key="a",
        columns=[
            Column("a", UUID_TYPE),
            Column("b", STRING_TYPE, nullable=True),
        ],
    )
    t = schema.to_sqlalchemy_table(sa.MetaData())
    pks = {c.name for c in t.primary_key.columns}
    assert pks == {"a"}


def test_to_sqla_vector_dim_propagates() -> None:
    """schema with ``Column("emb", VECTOR_TYPE, vector_dim=768)``
    produces a SQLAlchemy Vector column with dim 768.
    """
    pytest.importorskip("pgvector")
    from pgvector.sqlalchemy import Vector

    schema = TableSchema(
        name="t",
        primary_key="a",
        columns=[
            Column("a", UUID_TYPE),
            Column(
                "emb",
                VECTOR_TYPE,
                vector_dim=768,
                nullable=True,
            ),
        ],
    )
    t = schema.to_sqlalchemy_table(sa.MetaData())
    emb_col = t.c.emb
    assert isinstance(emb_col.type, Vector)
    assert emb_col.type.dim == 768


def test_to_sqla_enum_creates_check_constraint() -> None:
    """schema with an ENUM_TYPE column produces a SQLAlchemy Enum
    with ``create_constraint=True``.
    """
    from sqlalchemy import Enum as SAEnum

    schema = TableSchema(
        name="t",
        primary_key="a",
        columns=[
            Column("a", UUID_TYPE),
            Column(
                "status",
                ENUM_TYPE,
                enum_type=("alpha", "beta", "gamma"),
                enum_name="t_status_enum",
            ),
        ],
    )
    t = schema.to_sqlalchemy_table(sa.MetaData())
    status_col = t.c.status
    assert isinstance(status_col.type, SAEnum)
    assert status_col.type.create_constraint is True
    assert status_col.type.name == "t_status_enum"
    assert tuple(status_col.type.enums) == ("alpha", "beta", "gamma")


def test_to_sqla_single_fk_inline() -> None:
    """schema with ``Column(..., foreign_key=("users", "user_id"))``
    produces a column whose inline FK targets ``users.user_id``.
    """
    schema = TableSchema(
        name="t",
        primary_key="a",
        columns=[
            Column("a", UUID_TYPE),
            Column(
                "user_id",
                UUID_TYPE,
                foreign_key=("users", "user_id"),
            ),
        ],
    )
    t = schema.to_sqlalchemy_table(sa.MetaData())
    fks = list(t.c.user_id.foreign_keys)
    assert len(fks) == 1
    assert fks[0].target_fullname == "users.user_id"


def test_to_sqla_composite_fk_table_level() -> None:
    """schema with a composite FK in :attr:`foreign_keys` produces a
    table-level :class:`ForeignKeyConstraint` with the requested
    ``ondelete``.
    """
    schema = TableSchema(
        name="t",
        primary_key=("a", "b"),
        columns=[
            Column("a", UUID_TYPE),
            Column("b", UUID_TYPE),
        ],
        foreign_keys=(
            ForeignKey(
                ("a", "b"),
                "parent",
                ("a", "b"),
                on_delete="CASCADE",
            ),
        ),
    )
    t = schema.to_sqlalchemy_table(sa.MetaData())
    composite_fks = [c for c in t.constraints if isinstance(c, sa.ForeignKeyConstraint) and len(c.column_keys) > 1]
    assert len(composite_fks) == 1
    fk = composite_fks[0]
    assert tuple(fk.column_keys) == ("a", "b")
    assert fk.ondelete == "CASCADE"
    # read referenced table via target_fullname (resolution via
    # .column would require ``parent`` to be registered on the same
    # MetaData).
    assert fk.elements[0].target_fullname.split(".", 1)[0] == "parent"


def test_to_sqla_composite_fk_no_action_omits_ondelete() -> None:
    """composite FK declared with the default ``on_delete="NO ACTION"``
    must NOT emit an explicit ``ondelete=`` argument on the SA
    constraint (the hand-written factories do not emit redundant
    ``ON DELETE NO ACTION`` clauses; matching that keeps Alembic
    auto-gen quiet).
    """
    schema = TableSchema(
        name="t",
        primary_key=("a", "b"),
        columns=[
            Column("a", UUID_TYPE),
            Column("b", UUID_TYPE),
        ],
        foreign_keys=(ForeignKey(("a", "b"), "parent", ("a", "b")),),
    )
    t = schema.to_sqlalchemy_table(sa.MetaData())
    composite_fks = [c for c in t.constraints if isinstance(c, sa.ForeignKeyConstraint) and len(c.column_keys) > 1]
    assert len(composite_fks) == 1
    assert composite_fks[0].ondelete is None


def test_to_sqla_partial_index_postgresql_where() -> None:
    """schema with an :class:`IndexDef` carrying a ``where=`` predicate
    produces a SA Index with ``postgresql_where`` set.
    """
    schema = TableSchema(
        name="t",
        primary_key="a",
        columns=[
            Column("a", UUID_TYPE),
            Column("b", STRING_TYPE, nullable=True),
        ],
        indexes=(
            Index(
                "ix_t_b_partial",
                "b",
                unique=True,
                where="b IS NOT NULL",
            ),
        ),
    )
    t = schema.to_sqlalchemy_table(sa.MetaData())
    by_name = {i.name: i for i in t.indexes}
    idx = by_name["ix_t_b_partial"]
    where = idx.dialect_kwargs.get("postgresql_where")
    assert where is not None
    compiled = str(where.compile(compile_kwargs={"literal_binds": True}))
    assert compiled == "b IS NOT NULL"
    assert idx.unique is True


def test_to_sqla_server_default_propagates() -> None:
    """schema with ``Column(..., server_default="image")`` produces a
    column whose ``server_default`` is set to the literal value.
    """
    schema = TableSchema(
        name="t",
        primary_key="a",
        columns=[
            Column("a", UUID_TYPE),
            Column(
                "category",
                STRING_TYPE,
                server_default="image",
            ),
        ],
    )
    t = schema.to_sqlalchemy_table(sa.MetaData())
    sd = t.c.category.server_default
    assert sd is not None
    assert str(sd.arg) == "image"


def test_to_sqla_pgvector_unavailable_raises_at_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """when ``pgvector`` cannot be imported and a schema declares a
    VECTOR_TYPE column, ``to_sqlalchemy_table`` raises ``ImportError``
    at call time (mirrors the v0.7.5 factory behaviour -- the failure
    is legible at registration time, not deep inside INSERT).
    """
    import builtins

    real_import = builtins.__import__

    def _failing_import(
        name: str,
        globals: Any = None,
        locals: Any = None,
        fromlist: Any = (),
        level: int = 0,
    ) -> Any:
        if name.startswith("pgvector"):
            raise ImportError(
                "simulated: pgvector not installed",
            )
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _failing_import)

    schema = TableSchema(
        name="t",
        primary_key="a",
        columns=[
            Column("a", UUID_TYPE),
            Column(
                "emb",
                VECTOR_TYPE,
                vector_dim=8,
                nullable=True,
            ),
        ],
    )
    with pytest.raises(ImportError, match="pgvector"):
        schema.to_sqlalchemy_table(sa.MetaData())


def test_to_sqla_numeric_precision_scale() -> None:
    """schema with ``Column("cost", NUMERIC_TYPE, precision=12, scale=8,
    nullable=True)`` produces a SQLAlchemy ``Numeric(12, 8)`` column.
    """
    from sqlalchemy import Numeric

    schema = TableSchema(
        name="t",
        primary_key="a",
        columns=[
            Column("a", UUID_TYPE),
            Column(
                "cost",
                NUMERIC_TYPE,
                precision=12,
                scale=8,
                nullable=True,
            ),
        ],
    )
    t = schema.to_sqlalchemy_table(sa.MetaData())
    cost_col = t.c.cost
    assert isinstance(cost_col.type, Numeric)
    assert cost_col.type.precision == 12
    assert cost_col.type.scale == 8


def test_to_sqla_tsvector_immutable() -> None:
    """schema with ``Column("search_vector", TSVECTOR_TYPE, nullable=True,
    immutable=True)`` produces a SQLAlchemy TSVECTOR column.

    immutability is a Collection-side concern (the UPDATE generator
    uses :attr:`Column.immutable` to exclude the column from ``SET``
    clauses); this test just confirms the SQLAlchemy type round-trips.
    """
    from sqlalchemy.dialects.postgresql import TSVECTOR

    schema = TableSchema(
        name="t",
        primary_key="a",
        columns=[
            Column("a", UUID_TYPE),
            Column(
                "search_vector",
                TSVECTOR_TYPE,
                nullable=True,
                immutable=True,
            ),
        ],
    )
    t = schema.to_sqlalchemy_table(sa.MetaData())
    assert isinstance(t.c.search_vector.type, TSVECTOR)
    assert t.c.search_vector.nullable is True


# ---------------------------------------------------------------------------
# additional unit tests: BOOL_TYPE / BYTES_TYPE / unknown-tag failure
# ---------------------------------------------------------------------------


def test_to_sqla_bool_and_bytes_types() -> None:
    """BOOL_TYPE → SQLAlchemy ``Boolean``; BYTES_TYPE → ``BYTEA``."""
    from sqlalchemy import Boolean
    from sqlalchemy.dialects.postgresql import BYTEA

    schema = TableSchema(
        name="t",
        primary_key="a",
        columns=[
            Column("a", UUID_TYPE),
            Column("flag", BOOL_TYPE),
            Column("blob", BYTES_TYPE, nullable=True),
        ],
    )
    t = schema.to_sqlalchemy_table(sa.MetaData())
    assert isinstance(t.c.flag.type, Boolean)
    assert isinstance(t.c.blob.type, BYTEA)


def test_to_sqla_unknown_column_type_raises_keyerror() -> None:
    """a column whose ``column_type`` tag is unrecognised raises
    :class:`KeyError` with the column name + tag in the message --
    unknown tags must fail loudly, never silently drop.
    """
    # bypass Column.__post_init__ validation by constructing through
    # object.__setattr__ on a frozen dataclass: simulates a future tag
    # added at module scope without a corresponding mapping entry.
    col = Column("a", UUID_TYPE)
    object.__setattr__(col, "column_type", "totally_unknown_tag")

    schema = TableSchema(
        name="t",
        primary_key="a",
        columns=[col],
    )

    with pytest.raises(KeyError, match="totally_unknown_tag"):
        schema.to_sqlalchemy_table(sa.MetaData())


# ---------------------------------------------------------------------------
# regression-net self-checks (prove the parity helper still catches drift)
# ---------------------------------------------------------------------------
#
# ``_column_signature`` collapses ``Vector(N)`` / ``Enum(*vals)`` /
# ``Numeric(p, s)`` to a single ``type-class-name`` field; without the
# appended ``vector_dim`` / ``enum_values`` / ``(precision, scale)``
# fields a shard-03 enrichment regression (e.g. ``vector_dim=768``
# instead of ``1024`` on ``memories.embedding``, or a dropped enum
# value, or a numeric scale change) would pass the parity test
# silently. These three tests confirm at CI time that the helper
# continues to catch each of those drift modes -- protecting against a
# future refactor accidentally weakening ``_column_signature``.


def test_parity_helper_catches_vector_dim_regression() -> None:
    """parity helper raises AssertionError when two schemas differ
    only in ``vector_dim`` (1024 vs 768).

    proves the appended ``vector_dim`` field is load-bearing -- if a
    future refactor drops it, this test breaks loudly.
    """
    pytest.importorskip("pgvector")

    def _build(dim: int) -> sa.Table:
        schema = TableSchema(
            name="t",
            primary_key="a",
            columns=[
                Column("a", UUID_TYPE),
                Column(
                    "emb",
                    VECTOR_TYPE,
                    vector_dim=dim,
                    nullable=True,
                ),
            ],
        )
        return schema.to_sqlalchemy_table(sa.MetaData())

    a = _build(1024)
    b = _build(768)
    with pytest.raises(AssertionError, match="1024"):
        _assert_tables_equivalent(a, b)


def test_parity_helper_catches_enum_value_regression() -> None:
    """parity helper raises AssertionError when two schemas differ
    only in their ENUM value tuple (one drops a value).

    proves the appended ``enum_values`` field is load-bearing.
    """

    def _build(values: tuple[str, ...]) -> sa.Table:
        schema = TableSchema(
            name="t",
            primary_key="a",
            columns=[
                Column("a", UUID_TYPE),
                Column(
                    "status",
                    ENUM_TYPE,
                    enum_type=values,
                    enum_name="t_status_enum",
                ),
            ],
        )
        return schema.to_sqlalchemy_table(sa.MetaData())

    a = _build(("alpha", "beta", "gamma"))
    b = _build(("alpha", "beta"))  # dropped "gamma"
    with pytest.raises(AssertionError, match="gamma"):
        _assert_tables_equivalent(a, b)


def test_parity_helper_catches_numeric_scale_regression() -> None:
    """parity helper raises AssertionError when two schemas differ
    only in numeric ``scale`` (8 vs 4).

    proves the appended ``(precision, scale)`` field is load-bearing.
    """

    def _build(scale: int) -> sa.Table:
        schema = TableSchema(
            name="t",
            primary_key="a",
            columns=[
                Column("a", UUID_TYPE),
                Column(
                    "cost",
                    NUMERIC_TYPE,
                    precision=12,
                    scale=scale,
                    nullable=True,
                ),
            ],
        )
        return schema.to_sqlalchemy_table(sa.MetaData())

    a = _build(8)
    b = _build(4)
    # match on the differing scale value to prove the helper caught
    # the (precision, scale) tuple drift specifically.
    with pytest.raises(AssertionError, match=r"12,\s*8"):
        _assert_tables_equivalent(a, b)
