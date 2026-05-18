"""Generic mechanics tests for :meth:`TableSchema.to_sqlalchemy_table`.

Framework-level unit tests that exercise the conversion in isolation
with minimal :class:`TableSchema` declarations -- composite-PK shape,
inline vs composite FK emission, partial-index ``postgresql_where``,
server-default propagation, NUMERIC precision/scale, TSVECTOR round-
trip, unknown-tag failure, etc.

Per-table parity tests (production ``<Collection>.schema`` vs
hand-written ``_reference_<table>_table`` fixture) live in each
owning package's test directory:

- ``packages/agent/memory/tests/unit/test_to_sqlalchemy_table_parity.py``
- ``packages/agent/tools/tests/unit/test_to_sqlalchemy_table_parity.py``
- ``packages/mcp/tests/unit/test_to_sqlalchemy_table_parity.py``

The shared structural-comparison helpers
(:func:`assert_tables_equivalent`, :func:`column_signature`,
:func:`index_signature`, :func:`fk_constraint_signature`,
:func:`inline_fk_signatures`) live in
:mod:`threetears.core.testing.sqla_parity` so the per-package parity
files can import them without re-implementing the structural-compare
logic and so this generic-mechanics file stays focused on the framework.

The three ``test_parity_helper_catches_*`` tests at the bottom of
this file exercise the helper directly -- they prove that the
``column_signature`` axes (``vector_dim`` / ``enum_values`` /
``(precision, scale)``) are load-bearing. Removing or weakening any
of those axes would let a future enrichment regression slip past
the parity tests silently; these guard against that.
"""

from __future__ import annotations

from typing import Any

import pytest
import sqlalchemy as sa
from threetears.core.collections.schema_backed import (
    BOOL_TYPE,
    BYTES_TYPE,
    ENUM_TYPE,
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
from threetears.core.testing.sqla_parity import assert_tables_equivalent


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
# parity-helper regression tests
# ---------------------------------------------------------------------------
#
# These three tests guard the load-bearing fields appended to
# ``column_signature`` in :mod:`threetears.core.testing.sqla_parity`
# beyond the basic ``(name, type-class-name, primary_key, nullable,
# server_default)`` tuple. Without the appended ``vector_dim`` /
# ``enum_values`` / ``(precision, scale)`` fields a future enrichment
# regression (e.g. ``vector_dim=768`` instead of ``1024`` on
# ``memories.embedding``, or a dropped enum value, or a numeric scale
# change) would pass the parity tests silently and only surface at
# metallm Alembic auto-gen (too late). These three tests confirm at CI
# time that the helper continues to catch each of those drift modes --
# protecting against a future refactor accidentally weakening
# ``column_signature``.


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
        assert_tables_equivalent(a, b)


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
        assert_tables_equivalent(a, b)


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
        assert_tables_equivalent(a, b)


# ---------------------------------------------------------------------------
# v0.8.1: HNSW / GIN / opclass / WITH-clause emission
# ---------------------------------------------------------------------------


def test_to_sqla_hnsw_index_using_ops() -> None:
    """schema with an HNSW :class:`IndexDef` carrying ``using="hnsw"``,
    ``ops={"embedding": "vector_cosine_ops"}``, and ``pg_with={"m": "16",
    "ef_construction": "64"}`` produces a SA Index with the correct
    dialect_kwargs propagated.
    """
    schema = TableSchema(
        name="t",
        primary_key="a",
        columns=[
            Column("a", UUID_TYPE),
            Column("embedding", VECTOR_TYPE, vector_dim=8, nullable=True),
        ],
        indexes=(
            Index(
                "ix_t_embedding_hnsw",
                "embedding",
                using="hnsw",
                ops={"embedding": "vector_cosine_ops"},
                pg_with={"m": "16", "ef_construction": "64"},
            ),
        ),
    )
    t = schema.to_sqlalchemy_table(sa.MetaData())
    by_name = {i.name: i for i in t.indexes}
    idx = by_name["ix_t_embedding_hnsw"]
    assert idx.dialect_kwargs.get("postgresql_using") == "hnsw"
    assert idx.dialect_kwargs.get("postgresql_ops") == {"embedding": "vector_cosine_ops"}
    assert idx.dialect_kwargs.get("postgresql_with") == {
        "m": "16",
        "ef_construction": "64",
    }
    # negative-control: a plain index in the same schema would not have
    # these kwargs set; the explicit None / absence assertion guards
    # against an accidental schema-wide leak of the dialect kwargs.
    assert idx.unique is False
    assert idx.dialect_kwargs.get("postgresql_where") is None


def test_to_sqla_gin_index_using() -> None:
    """schema with a GIN :class:`IndexDef` (``using="gin"``, no
    ``ops`` / ``pg_with``) produces a SA Index with
    ``postgresql_using="gin"`` and no ops / with kwargs.
    """
    schema = TableSchema(
        name="t",
        primary_key="a",
        columns=[
            Column("a", UUID_TYPE),
            Column("search_vector", TSVECTOR_TYPE, immutable=True, nullable=True),
        ],
        indexes=(
            Index(
                "ix_t_search_vector",
                "search_vector",
                using="gin",
            ),
        ),
    )
    t = schema.to_sqlalchemy_table(sa.MetaData())
    by_name = {i.name: i for i in t.indexes}
    idx = by_name["ix_t_search_vector"]
    assert idx.dialect_kwargs.get("postgresql_using") == "gin"
    # ops / pg_with are unset on this index; reading them back via
    # dialect_kwargs.get returns the empty-mapping default that
    # SQLAlchemy materialises for dict-typed dialect kwargs (so the
    # parity helper's ``or {}`` fallback resolves to the same empty
    # tuple on both reference and candidate sides).
    assert not idx.dialect_kwargs.get("postgresql_ops")
    assert not idx.dialect_kwargs.get("postgresql_with")


def test_parity_helper_catches_using_regression() -> None:
    """parity helper raises AssertionError when two schemas differ
    only in the access method (``using="hnsw"`` vs ``using="btree"``).

    proves the appended ``postgresql_using`` field on the
    :func:`index_signature` 7-tuple is load-bearing — a future
    regression that flipped HNSW back to btree on the ``embedding``
    column would otherwise pass parity silently and only surface at
    metallm Alembic auto-gen.
    """

    def _build(using: str) -> sa.Table:
        schema = TableSchema(
            name="t",
            primary_key="a",
            columns=[
                Column("a", UUID_TYPE),
                Column("embedding", VECTOR_TYPE, vector_dim=8, nullable=True),
            ],
            indexes=(
                Index(
                    "ix_t_embedding",
                    "embedding",
                    using=using,
                ),
            ),
        )
        return schema.to_sqlalchemy_table(sa.MetaData())

    a = _build("hnsw")
    b = _build("btree")
    with pytest.raises(AssertionError, match="index-signature"):
        assert_tables_equivalent(a, b)
