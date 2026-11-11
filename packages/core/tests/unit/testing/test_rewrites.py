"""the rewrite proof's source check and row factory (:mod:`threetears.core.testing.rewrites`); its database half runs
in each repo's generated equivalence test."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from threetears.core.testing.rewrites import _Column, _row_factory, _scalar, _Table, assert_source_says, entry_id
from threetears.core.testing.same_results import Seeded

_ENTRY: dict[str, Any] = {
    "path": "m.py",
    "function": "Reader.one",
    "table": "agents",
    "helper": "read_row",
    "columns": ["id", "name"],
    "pins": [
        {"column": "id", "kind": "param", "param": 1, "literal": None},
        {"column": "is_deleted", "kind": "literal", "param": None, "literal": False},
    ],
    "args": ["self.key(agent_id)"],
}


def _module(tmp_path: Path, body: str) -> Path:
    (tmp_path / "m.py").write_text(f"class Reader:\n    async def one(self, conn, agent_id):\n{body}", encoding="utf-8")
    return tmp_path


def test_the_recorded_call_is_found() -> None:
    """Flips if an entry's id stops naming its module, function, helper, table and pins."""
    assert entry_id(_ENTRY) == "m.py::Reader.one:read_row(agents:id,is_deleted)"


def test_the_source_check_holds_the_recorded_call(tmp_path: Path) -> None:
    """Flips if the source check misses the call the manifest records."""
    root = _module(
        tmp_path,
        "        return await read_row(\n            conn, 'agents', by={'id': self.key( agent_id ), 'is_deleted': False},"
        " columns=('id', 'name'), records=True\n        )\n",
    )
    assert_source_says(root, _ENTRY)


@pytest.mark.parametrize(
    "call",
    [
        "read_row(conn, 'agents', by={'id': agent_id, 'is_deleted': False}, columns=('id', 'name'), records=True)",
        "read_row(conn, 'agents', by={'id': self.key(agent_id)}, columns=('id', 'name'), records=True)",
        "read_row(conn, 'agents', by={'id': self.key(agent_id), 'is_deleted': True}, columns=('id', 'name'), records=True)",
        "read_row(conn, 'agent', by={'id': self.key(agent_id), 'is_deleted': False}, columns=('id', 'name'), records=True)",
        "read_row(conn, 'agents', by={'id': self.key(agent_id), 'is_deleted': False}, columns=('id',), records=True)",
        "read_row(conn, 'agents', by={'id': self.key(agent_id), 'is_deleted': False}, columns=('id', 'name'))",
        "read_by(conn, 'agents', by={'id': self.key(agent_id), 'is_deleted': False}, columns=('id', 'name'), records=True)",
    ],
)
def test_the_source_check_refuses_any_other_call(tmp_path: Path, call: str) -> None:
    """Flips if a call differing from the manifest in its argument, pins, table, columns or helper passes."""
    root = _module(tmp_path, f"        return await {call}\n")
    with pytest.raises(AssertionError, match="as the manifest records it"):
        assert_source_says(root, _ENTRY)


def _module_with_columns(tmp_path: Path, constant: str, call_columns: str) -> Path:
    call = (
        f"read_row(conn, 'agents', by={{'id': self.key(agent_id), 'is_deleted': False}}, columns={call_columns},"
        " records=True)"
    )
    (tmp_path / "m.py").write_text(
        f"{constant}\n\nclass Reader:\n    async def one(self, conn, agent_id):\n        return await {call}\n",
        encoding="utf-8",
    )
    return tmp_path


@pytest.mark.parametrize("constant", ["_COLUMNS = ('id', 'name')", "_COLUMNS: tuple[str, ...] = ('id', 'name')"])
def test_the_source_check_reads_a_module_level_column_tuple(tmp_path: Path, constant: str) -> None:
    """Flips if a call naming its columns by a module-level tuple of the recorded columns is not found."""
    assert_source_says(_module_with_columns(tmp_path, constant, "_COLUMNS"), _ENTRY)


@pytest.mark.parametrize(
    ("constant", "call_columns"),
    [
        ("_COLUMNS = ('id',)", "_COLUMNS"),
        ("_COLUMNS = ('name', 'id')", "_COLUMNS"),
        ("_COLUMNS = tuple(c for c in ('id', 'name'))", "_COLUMNS"),
        ("_COLUMNS = ('id', 'name')", "_OTHER"),
        ("_NAME = 'name'", "('id', _NAME)"),
    ],
)
def test_the_source_check_refuses_columns_it_cannot_read_as_recorded(
    tmp_path: Path, constant: str, call_columns: str
) -> None:
    """Flips if columns named by a constant that differs, is not a literal tuple, or is not there pass the check."""
    with pytest.raises(AssertionError, match="as the manifest records it"):
        assert_source_says(_module_with_columns(tmp_path, constant, call_columns), _ENTRY)


def _shape() -> _Table:
    def column(name: str, kind: str, category: str, *, nullable: bool = True) -> _Column:
        return _Column(name, kind, category, "", nullable, False, False)

    return _Table(
        columns={
            "id": column("id", "uuid", "U", nullable=False),
            "customer_id": column("customer_id", "uuid", "U"),
            "slug": column("slug", "character varying(8)", "S"),
            "is_deleted": column("is_deleted", "boolean", "B", nullable=False),
            "note": column("note", "text", "S"),
        },
        unique=[frozenset({"id"}), frozenset({"customer_id", "slug"})],
    )


def _seed(index: int, tenant: int = 0, null: bool = False) -> Seeded:
    return Seeded(index=index, case="c", tenant=tenant, null=null, tie=index)


def test_a_key_column_is_shared_by_a_tenant_s_rows_unless_a_unique_key_needs_it_apart() -> None:
    """Flips if a bound column stops naming many rows of a tenant, or a fully pinned unique key collides."""
    by_customer = {
        "table": "agents",
        "pins": [{"column": "customer_id", "kind": "param", "param": 1, "literal": None}],
    }
    row, params = _row_factory(_shape(), by_customer, {})
    assert row(_seed(0))["customer_id"] == row(_seed(5))["customer_id"] == params[0]
    assert row(_seed(5, tenant=1))["customer_id"] != params[0]
    assert len({row(_seed(index))["id"] for index in range(10)}) == 10
    assert all(len(row(_seed(index))["slug"]) <= 8 for index in range(30))

    by_slug = {
        "table": "agents",
        "pins": [
            {"column": "customer_id", "kind": "param", "param": 2, "literal": None},
            {"column": "slug", "kind": "param", "param": 1, "literal": None},
        ],
    }
    row, params = _row_factory(_shape(), by_slug, {})
    assert len({(row(_seed(index))["customer_id"], row(_seed(index))["slug"]) for index in range(10)}) == 10
    assert params == [row(_seed(0))["slug"], row(_seed(0))["customer_id"]]


def test_a_literal_pin_holds_its_value_on_two_rows_in_three_and_nulls_follow_the_case() -> None:
    """Flips if a literal-pinned column never holds another value, or a nullable column ignores the NULLs case."""
    entry = {
        "table": "agents",
        "pins": [
            {"column": "id", "kind": "param", "param": 1, "literal": None},
            {"column": "is_deleted", "kind": "literal", "param": None, "literal": False},
        ],
    }
    row, _ = _row_factory(_shape(), entry, {"note": lambda seed: "fixed"})
    assert [row(_seed(index))["is_deleted"] for index in range(6)] == [False, False, True, False, False, True]
    assert row(_seed(1, null=True))["customer_id"] is None
    assert row(_seed(1, null=True))["note"] == "fixed"


def test_a_json_column_is_seeded_as_its_text() -> None:
    """Flips if a json or jsonb column is seeded as a Python mapping: asyncpg with no json codec refuses it (the
    generated test then errors seeding every table holding one), and takes the value as text."""
    for kind in ("jsonb", "json"):
        value = _scalar(_Column("doc", kind, "U", "", True, False, False), "U", 7)
        assert isinstance(value, str)
        assert json.loads(value) == {"n": 7}
