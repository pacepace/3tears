"""structural-equality helpers for SQLAlchemy ``Table`` parity tests.

Used by per-package parity tests to compare a Table produced by
:meth:`threetears.core.collections.schema_backed.TableSchema.to_sqlalchemy_table`
against a hand-written reference fixture that encodes the v0.8.0
canonical shape.

The signature builders deliberately include the type-parameter axes
that ``type(col.type).__name__`` collapses:

- ``Vector(1024)`` and ``Vector(768)`` both report ``__name__ ==
  "VECTOR"`` — without ``getattr(col.type, "dim", None)`` in the
  signature, a vector-dimension regression would pass silently.
- ``SAEnum("a", "b")`` and ``SAEnum("a", "b", "c")`` both report
  ``__name__ == "Enum"`` — without ``tuple(getattr(col.type, "enums",
  ()))`` in the signature, a dropped enum value would pass silently.
- ``Numeric(12, 8)`` and ``Numeric(10, 4)`` both report ``__name__ ==
  "Numeric"`` — without the precision/scale pair in the signature, a
  scale or precision change would pass silently.

PK column ORDER is compared (not just sets). Postgres + SQLAlchemy
preserve column order in the ``PRIMARY KEY`` constraint definition;
Alembic auto-gen compares ordered. ``(memory_id, agent_id)`` vs
``(agent_id, memory_id)`` would slip past a set-only assertion.

Per-package parity tests typically wire up like:

.. code-block:: python

    import sqlalchemy as sa
    from threetears.agent.memory.collections import MemoriesCollection
    from threetears.core.testing.sqla_parity import assert_tables_equivalent

    def test_parity_memories_collection_schema() -> None:
        via_reference = _reference_memories_table(sa.MetaData())
        via_collection = MemoriesCollection.schema.to_sqlalchemy_table(
            sa.MetaData(),
        )
        assert_tables_equivalent(via_reference, via_collection)
"""

from __future__ import annotations

from typing import Any

import sqlalchemy as sa

__all__ = [
    "assert_tables_equivalent",
    "column_signature",
    "fk_constraint_signature",
    "index_signature",
    "inline_fk_signatures",
    "unique_constraint_signature",
]


def column_signature(
    col: sa.Column[Any],
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
    ``Numeric``. Without the appended fields a future enrichment
    regression that wrote ``vector_dim=768`` instead of ``1024`` on
    ``memories.embedding``, or dropped an enum value, or changed
    numeric scale, would pass the parity test silently and only
    surface at metallm Alembic auto-gen (too late).

    :param col: column to summarise
    :ptype col: sqlalchemy.Column
    :return: 8-tuple of (name, type-class-name, primary_key, nullable,
        server_default-arg, vector_dim, enum_values, numeric_ps)
    :rtype: tuple
    """
    sd: str | None
    if col.server_default is not None:
        # ``server_default.arg`` is the literal string / TextClause for
        # ``DefaultClause`` instances (the common case for declared
        # defaults like ``"now()"`` / ``"'{}'::jsonb"``). The
        # ``FetchedValue`` subclass (used when SQLAlchemy detects the
        # default server-side via reflection) lacks ``.arg``; we don't
        # construct those in the parity path, but guard with getattr
        # to keep mypy + runtime safe across both shapes.
        arg = getattr(col.server_default, "arg", None)
        sd = str(arg) if arg is not None else None
    else:
        sd = None
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


def index_signature(
    idx: sa.Index,
) -> tuple[
    str,
    frozenset[str],
    bool,
    str | None,
    str | None,
    tuple[tuple[str, str], ...],
    tuple[tuple[str, str], ...],
]:
    """return a structural signature for a SA Index.

    Reads the partial-index ``WHERE`` clause, access method,
    per-column operator classes, and access-method parameter
    storage via the public ``dialect_kwargs`` API (NOT
    ``dialect_options``, which is internal state and has versioned
    shape).

    v0.8.1 added the ``postgresql_using`` / ``postgresql_ops`` /
    ``postgresql_with`` axes so a future schema regression that
    flipped an HNSW index to btree, dropped a ``vector_cosine_ops``
    opclass binding, or perturbed the ``m`` / ``ef_construction``
    parameters does not pass parity silently.

    :param idx: index to summarise
    :ptype idx: sqlalchemy.Index
    :return: 7-tuple of (name, frozenset of column names, unique,
        compiled where-clause text or ``None``, access-method name or
        ``None``, tuple of sorted ``(column, opclass)`` pairs, tuple
        of sorted ``(key, value)`` pairs for the ``WITH`` clause)
    :rtype: tuple
    """
    where = idx.dialect_kwargs.get("postgresql_where")
    using = idx.dialect_kwargs.get("postgresql_using")
    ops_map = idx.dialect_kwargs.get("postgresql_ops") or {}
    with_map = idx.dialect_kwargs.get("postgresql_with") or {}
    # ``idx.name`` is typed as ``quoted_name | None`` by SQLAlchemy 2.x
    # but is always a real string in our declarations (the v0.8.0
    # spec disallows auto-named indexes). Coerce to ``str`` so the
    # signature tuple is hashable + comparable across runs.
    return (
        str(idx.name) if idx.name is not None else "",
        frozenset(c.name for c in idx.columns),
        bool(idx.unique),
        str(where.compile(compile_kwargs={"literal_binds": True})) if where is not None else None,
        str(using) if using is not None else None,
        tuple(sorted((str(k), str(v)) for k, v in ops_map.items())),
        tuple(sorted((str(k), str(v)) for k, v in with_map.items())),
    )


def unique_constraint_signature(
    constraint: sa.UniqueConstraint,
) -> tuple[str, tuple[str, ...]]:
    """return a structural signature for a SA UniqueConstraint.

    Distinct axis from :func:`index_signature` even though Postgres
    surfaces both as identical rows in ``pg_indexes``. Alembic
    auto-gen reads UNIQUE constraints out of
    ``information_schema.table_constraints``, so a schema that declares
    ``CREATE UNIQUE INDEX uq_foo`` will diff against a prod that
    declares ``ALTER TABLE foo ADD CONSTRAINT uq_foo UNIQUE`` and
    surface a ``drop_constraint`` / ``create_unique_constraint`` op.
    v0.8.1 added :class:`UniqueConstraintDef` so the two shapes are
    independently modellable.

    PK column order is NOT preserved on UNIQUE constraints (Postgres
    treats UNIQUE column lists as sets; Alembic auto-gen compares
    unordered), so the signature uses a tuple of sorted column names.

    :param constraint: SA ``UniqueConstraint``
    :ptype constraint: sqlalchemy.UniqueConstraint
    :return: 2-tuple of (name, sorted-tuple of column names)
    :rtype: tuple
    """
    return (
        str(constraint.name) if constraint.name is not None else "",
        tuple(sorted(c.name for c in constraint.columns)),
    )


def fk_constraint_signature(
    constraint: sa.ForeignKeyConstraint,
) -> tuple[tuple[str, ...], str, tuple[str, ...], str | None]:
    """return a structural signature for a table-level FK constraint.

    Reads the referenced ``table.col`` pair from each
    :attr:`ForeignKey.target_fullname` string rather than via
    :attr:`ForeignKey.column`. The latter resolves the reference
    through the Table's :class:`MetaData` and raises
    :class:`NoReferencedTableError` when the target table is not
    registered on the same MetaData -- which is normal in parity
    tests because the v0.8.0 schemas declare FKs to ``users`` /
    ``messages`` / etc. tables that the metallm app registers
    separately and 3tears tests never load.

    :param constraint: SA ``ForeignKeyConstraint``
    :ptype constraint: sqlalchemy.ForeignKeyConstraint
    :return: 4-tuple of (local_cols, ref_table, ref_cols, ondelete)
    :rtype: tuple
    """
    local_cols = tuple(constraint.column_keys)
    ref_pairs = [el.target_fullname.split(".", 1) for el in constraint.elements]
    ref_table = ref_pairs[0][0]
    ref_cols = tuple(pair[1] for pair in ref_pairs)
    return (local_cols, ref_table, ref_cols, constraint.ondelete)


def inline_fk_signatures(
    table: sa.Table,
) -> list[tuple[str, str, str | None]]:
    """return signatures for every inline (per-column) ForeignKey.

    Iterates ``table.columns`` looking for columns with attached
    ``foreign_keys``; emits one signature per FK. Useful for the
    parity helper to distinguish inline single-column FKs from
    table-level ``ForeignKeyConstraint`` objects.

    :param table: SA Table to inspect
    :ptype table: sqlalchemy.Table
    :return: list of (local_col_name, target_fullname, ondelete) for
        every inline FK on the table
    :rtype: list[tuple]
    """
    sigs: list[tuple[str, str, str | None]] = []
    for col in table.columns:
        for fk in col.foreign_keys:
            sigs.append((col.name, fk.target_fullname, fk.ondelete))
    return sigs


def assert_tables_equivalent(a: sa.Table, b: sa.Table) -> None:
    """assert two SQLAlchemy Tables are structurally identical.

    compares column names, types, primary-key columns + order,
    indexes, table-level FK constraints, and inline FKs. ignores
    Table metadata identity (the Tables are intentionally registered
    on different :class:`MetaData` instances).

    Reference fixtures encode the v0.8.0 canonical shape directly,
    so the assertion is a strict structural equality with no
    augmentation escape hatch.

    :param a: reference Table (typically the hand-written
        ``_reference_<table>_table`` fixture output)
    :ptype a: sqlalchemy.Table
    :param b: candidate Table (typically ``to_sqlalchemy_table`` output)
    :ptype b: sqlalchemy.Table
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
    a_col_sigs = {c.name: column_signature(c) for c in a.columns}
    b_col_sigs = {c.name: column_signature(c) for c in b.columns}
    for name in a_col_names:
        assert a_col_sigs[name] == b_col_sigs[name], (
            f"column signature mismatch on {a.name}.{name}: "
            f"reference={a_col_sigs[name]!r} vs to_sqla={b_col_sigs[name]!r}"
        )

    # PK column sets AND order. SQLAlchemy + Postgres preserve column
    # order in the ``PRIMARY KEY`` constraint definition; Alembic
    # auto-gen compares ordered. Comparing sets only would let an
    # ordering drift (e.g. ``(memory_id, agent_id)`` vs ``(agent_id,
    # memory_id)``) pass silently — caught here.
    a_pks_ordered = [c.name for c in a.primary_key.columns]
    b_pks_ordered = [c.name for c in b.primary_key.columns]
    assert a_pks_ordered == b_pks_ordered, (
        f"primary-key column ORDER differs on {a.name}: reference={a_pks_ordered!r} vs to_sqla={b_pks_ordered!r}"
    )

    # indexes (by signature)
    a_idx_sigs = {index_signature(i) for i in a.indexes}
    b_idx_sigs = {index_signature(i) for i in b.indexes}
    assert a_idx_sigs == b_idx_sigs, (
        f"index-signature sets differ on {a.name}: "
        f"only-in-reference={a_idx_sigs - b_idx_sigs!r}, "
        f"only-in-to_sqla={b_idx_sigs - a_idx_sigs!r}"
    )

    # table-level FK constraints
    a_fk_sigs = {fk_constraint_signature(c) for c in a.constraints if isinstance(c, sa.ForeignKeyConstraint)}
    b_fk_sigs = {fk_constraint_signature(c) for c in b.constraints if isinstance(c, sa.ForeignKeyConstraint)}
    assert a_fk_sigs == b_fk_sigs, (
        f"FK-constraint sets differ on {a.name}: "
        f"only-in-reference={a_fk_sigs - b_fk_sigs!r}, "
        f"only-in-to_sqla={b_fk_sigs - a_fk_sigs!r}"
    )

    # table-level UNIQUE constraints. v0.8.1: distinct axis from
    # indexes; see ``unique_constraint_signature`` docstring.
    a_uc_sigs = {unique_constraint_signature(c) for c in a.constraints if isinstance(c, sa.UniqueConstraint)}
    b_uc_sigs = {unique_constraint_signature(c) for c in b.constraints if isinstance(c, sa.UniqueConstraint)}
    assert a_uc_sigs == b_uc_sigs, (
        f"unique-constraint sets differ on {a.name}: "
        f"only-in-reference={a_uc_sigs - b_uc_sigs!r}, "
        f"only-in-to_sqla={b_uc_sigs - a_uc_sigs!r}"
    )

    # inline FK signatures
    a_inline = set(inline_fk_signatures(a))
    b_inline = set(inline_fk_signatures(b))
    assert a_inline == b_inline, (
        f"inline FK sets differ on {a.name}: "
        f"only-in-reference={a_inline - b_inline!r}, "
        f"only-in-to_sqla={b_inline - a_inline!r}"
    )
