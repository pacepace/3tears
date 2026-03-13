"""Enforcement test -- schema agreement for collections.

Verifies that collection _FIELD_TYPES dictionaries agree with the
collection's structure. Since 3tears is a library (no Alembic), this
tests:
1. _FIELD_TYPES keys must include the primary key
2. Primary key consistency between entity and collection
3. _FIELD_TYPES keys in MemoriesCollection match its _save_to_postgres SQL

Uses AST analysis for SQL inspection and runtime imports only for
collection metadata that is safe to import.
"""

from __future__ import annotations

import ast
import importlib
import re
from pathlib import Path
from typing import Any

import pytest


# Discovery: find all collection modules that define _FIELD_TYPES
# For now, we know about MemoriesCollection in agent-memory.
# This test lives in core but validates any package that registers collections.


def _try_import_field_types(module_path: str) -> dict[str, Any] | None:
    """Try to import a module and return its _FIELD_TYPES, or None."""
    try:
        mod = importlib.import_module(module_path)
        return getattr(mod, "_FIELD_TYPES", None)
    except ImportError:
        return None


def _try_get_collection_pk(module_path: str) -> str | None:
    """Try to find _primary_key_column from a collection class."""
    try:
        mod = importlib.import_module(module_path)
        for attr_name in dir(mod):
            obj = getattr(mod, attr_name)
            if isinstance(obj, type) and hasattr(obj, "_primary_key_column") and attr_name not in ("BaseCollection",):
                pk = getattr(obj, "_primary_key_column", None)
                if pk:
                    return pk
    except ImportError:
        pass
    return None


def _try_get_entity_pk(module_path: str) -> str | None:
    """Try to find _primary_key_field from an entity class in the same package."""
    try:
        mod = importlib.import_module(module_path)
        for attr_name in dir(mod):
            obj = getattr(mod, attr_name)
            if isinstance(obj, type) and hasattr(obj, "_primary_key_field") and attr_name not in ("BaseEntity",):
                pk = getattr(obj, "_primary_key_field", None)
                if pk:
                    return pk
    except ImportError:
        pass
    return None


# Known collections with _FIELD_TYPES
_COLLECTIONS: list[tuple[str, str, str]] = [
    # (label, collection_module, entity_module)
    ("memories", "threetears.agent.memory.collections", "threetears.agent.memory.entities"),
]

# Filter to only importable collections
_AVAILABLE_COLLECTIONS = [
    (label, col_mod, ent_mod)
    for label, col_mod, ent_mod in _COLLECTIONS
    if _try_import_field_types(col_mod) is not None
]

_COLLECTION_IDS = [c[0] for c in _AVAILABLE_COLLECTIONS]


class TestFieldTypesConsistency:
    """_FIELD_TYPES must be internally consistent."""

    @pytest.mark.parametrize(
        "label,col_module,ent_module",
        _AVAILABLE_COLLECTIONS,
        ids=_COLLECTION_IDS,
    )
    def test_primary_key_in_field_types(
        self,
        label: str,
        col_module: str,
        ent_module: str,
    ) -> None:
        """The primary key column must appear in _FIELD_TYPES."""
        field_types = _try_import_field_types(col_module)
        assert field_types is not None

        collection_pk = _try_get_collection_pk(col_module)
        assert collection_pk is not None, f"{label}: could not find _primary_key_column in {col_module}"

        assert collection_pk in field_types, (
            f"{label}: _primary_key_column {collection_pk!r} is not in _FIELD_TYPES. "
            f"Available keys: {sorted(field_types.keys())}"
        )


class TestPrimaryKeyConsistency:
    """Entity _primary_key_field must match collection _primary_key_column."""

    @pytest.mark.parametrize(
        "label,col_module,ent_module",
        _AVAILABLE_COLLECTIONS,
        ids=_COLLECTION_IDS,
    )
    def test_entity_pk_matches_collection(
        self,
        label: str,
        col_module: str,
        ent_module: str,
    ) -> None:
        """Entity PK field must match collection PK column."""
        collection_pk = _try_get_collection_pk(col_module)
        entity_pk = _try_get_entity_pk(ent_module)

        if collection_pk is None or entity_pk is None:
            pytest.skip(f"Could not resolve PKs for {label}")

        assert collection_pk == entity_pk, (
            f"{label}: collection _primary_key_column={collection_pk!r} but entity _primary_key_field={entity_pk!r}"
        )


class TestSaveToPostgresConsistency:
    """SQL in _save_to_postgres must reference columns consistent with _FIELD_TYPES."""

    @pytest.mark.parametrize(
        "label,col_module,ent_module",
        _AVAILABLE_COLLECTIONS,
        ids=_COLLECTION_IDS,
    )
    def test_insert_columns_in_field_types(
        self,
        label: str,
        col_module: str,
        ent_module: str,
    ) -> None:
        """INSERT column names in _save_to_postgres must all be in _FIELD_TYPES."""
        field_types = _try_import_field_types(col_module)
        assert field_types is not None

        # Find the source file for the collection module
        mod = importlib.import_module(col_module)
        source_file = Path(mod.__file__)

        sql = self._extract_save_to_postgres_sql(source_file)
        if not sql:
            pytest.skip(f"No _save_to_postgres found in {col_module}")

        insert_cols = self._parse_insert_columns(sql)
        if not insert_cols:
            pytest.skip(f"Could not parse INSERT columns from {col_module}")

        field_type_keys = set(field_types.keys())
        missing = insert_cols - field_type_keys
        assert not missing, (
            f"{label}: _save_to_postgres INSERT references "
            f"{len(missing)} column(s) not in _FIELD_TYPES: {sorted(missing)}"
        )

    @staticmethod
    def _extract_save_to_postgres_sql(collection_file: Path) -> str:
        """Extract SQL strings from _save_to_postgres via AST."""
        tree = ast.parse(collection_file.read_text(encoding="utf-8"))
        sql_parts: list[str] = []

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name != "_save_to_postgres":
                continue
            for child in ast.walk(node):
                if isinstance(child, ast.Constant) and isinstance(child.value, str):
                    sql_parts.append(child.value)

        return "\n".join(sql_parts)

    @staticmethod
    def _parse_insert_columns(sql: str) -> set[str]:
        """Extract column names from INSERT INTO ... (...) patterns."""
        columns: set[str] = set()
        pattern = r"INSERT\s+INTO\s+\w+\s*\(([^)]+)\)"
        for match in re.finditer(pattern, sql, re.IGNORECASE | re.DOTALL):
            col_list = match.group(1)
            for col in col_list.split(","):
                col_name = col.strip()
                if col_name:
                    columns.add(col_name)
        return columns
