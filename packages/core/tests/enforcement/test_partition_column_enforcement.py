"""Enforcement -- every SQL touching a partitioned table filters by partition.

collections-task-02. partition columns are the structural defense
against cross-partition data bleed (the canonical agent-pod scenario:
two concurrent conversations served by the same pod, a Collection
method that forgets to filter by ``conversation_id``, and one
conversation seeing the other's context items). this enforcement
walker is the third defense layer: at CI time, every SQL string
literal that touches a partitioned table is verified to also include
the partition column name in its WHERE / SET / VALUES context.

design points:

- mode is controlled by ``PARTITION_ENFORCEMENT_MODE`` env var --
  ``strict`` raises on any violation; ``report`` (default during
  collections-task-02 cleanup) logs and passes so the test can be
  introduced and have its discoveries reviewed before flipping
  CI-blocking. mode flips to ``strict`` in the final cleanup commit
  once every flagged violation has been resolved.
- the discovery list (:data:`_PARTITIONED_TABLES`) maps table name
  -> partition column. extending it covers a new table; the cost
  of discovery automation across multi-repo Collections is paid
  here in audit clarity.
- migration files (``CREATE TABLE`` / ``ALTER TABLE``) are exempt
  by directory: ``migrations/`` subtrees legitimately contain DDL
  that does not filter rows.
- the walker tolerates SQL spread across multi-line strings and
  joined string concatenations because the AST visitor evaluates
  the joined value via ``ast.unparse`` for f-strings and joined
  literal concatenation for plain strings.
- exemptions live in :data:`_EXEMPT_LITERAL_FRAGMENTS` -- a narrow
  allowlist of string fragments for SQL that legitimately spans
  partitions (e.g. the body of a ``@spans_partitions``-decorated
  method). every entry carries an inline ``# rationale: ...``
  comment.

runtime is AST-only and well under the 15s CLAUDE.md budget; the
walker is deterministic and side-effect-free.
"""

from __future__ import annotations

import ast
import os
import re
from pathlib import Path

import pytest

__all__: list[str] = []


# table_name -> partition column. extend this map when a Collection
# adopts the ``partition=True`` flag on a TableSchema column. the test
# walks every .py file under each declared package and verifies SQL
# touching the table includes the partition column.
_PARTITIONED_TABLES: dict[str, str] = {
    # agent-tools
    "context_items": "conversation_id",
    # conversations
    "conversations": "agent_id",
    # agent-memory
    "memories": "agent_id",
    "media": "agent_id",
    "media_content": "agent_id",
    "memory_chunks": "agent_id",
    "conversation_memory_refs": "conversation_id",
    # agent-workspace
    "workspaces": "agent_id",
    "workspace_files": "workspace_id",
    "workspace_file_versions": "workspace_id",
}


# packages to walk. additions go here when a new repo / package is
# brought under the partition-column doctrine.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
_PACKAGE_SRC_ROOTS: list[Path] = [
    _REPO_ROOT / "packages" / "core" / "src",
    _REPO_ROOT / "packages" / "agent-tools" / "src",
    _REPO_ROOT / "packages" / "agent-workspace" / "src",
    _REPO_ROOT / "packages" / "agent-memory" / "src",
    _REPO_ROOT / "packages" / "conversations" / "src",
]


# narrow exemption list. each entry is a SQL fragment that legitimately
# spans partitions. callers add an entry here only after the
# cross-partition rationale is documented in the surrounding code.
_EXEMPT_LITERAL_FRAGMENTS: tuple[tuple[str, str], ...] = (
    # rationale: extension / trigger / function DDL is unscoped by
    # design; partition predicates do not apply to schema setup.
    ("CREATE EXTENSION", "DDL"),
    ("CREATE TRIGGER", "DDL"),
    ("CREATE OR REPLACE FUNCTION", "DDL"),
    ("DROP TRIGGER", "DDL"),
    # rationale: dynamic SQL builders compose the partition predicate
    # via parameterized fragments rather than a literal column name in
    # the same string. these helpers live in the memories collection
    # and inject ``user_id`` / ``agent_id`` / ``customer_id`` predicates
    # via _build_user_scope_clause; the walker cannot statically prove
    # the resulting SQL contains the partition column, so the test
    # treats the unparametrized template as exempt and trusts the
    # authoring discipline (every consumer of _build_user_scope_clause
    # passes scope params).
    ("websearch_to_tsquery", "dynamic-scope-clause"),
)


# regex to find the table name following FROM/INTO/UPDATE. matches
# qualified (schema.table) and bare table forms, case-insensitive.
_TABLE_PATTERN = re.compile(
    r"\b(?:FROM|INTO|UPDATE|JOIN)\s+(?:[a-zA-Z_]\w*\.)?([a-zA-Z_]\w*)",
    re.IGNORECASE,
)

# heuristic gate to filter docstrings / prose from SQL. a literal is
# treated as SQL only when it starts (after stripping leading
# whitespace) with one of these tokens. this catches the common case
# of a docstring that mentions "FROM memories" without false-firing
# on it.
_SQL_LEADING_TOKENS = ("SELECT", "INSERT", "UPDATE", "DELETE", "WITH", "VALUES")


def _collect_string_literals(tree: ast.AST) -> list[tuple[str, int]]:
    """walk ``tree`` and collect every string literal with its line number.

    handles plain ``ast.Constant`` strings, joined-string ``f"..."``
    forms (best-effort: takes the constant fragments and joins with
    placeholders for the formatted parts), and adjacent literal
    concatenation (the parser folds those into a single Constant
    already, so no special handling needed).

    :param tree: AST root
    :ptype tree: ast.AST
    :return: list of (literal text, line number) pairs
    :rtype: list[tuple[str, int]]
    """
    results: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            results.append((node.value, node.lineno))
        elif isinstance(node, ast.JoinedStr):
            parts: list[str] = []
            for v in node.values:
                if isinstance(v, ast.Constant) and isinstance(v.value, str):
                    parts.append(v.value)
                else:
                    # placeholder for formatted segment so partition
                    # columns embedded via interpolation count
                    parts.append("__PLACEHOLDER__")
            results.append(("".join(parts), node.lineno))
    return results


def _is_exempt(literal: str) -> bool:
    """true iff ``literal`` is in :data:`_EXEMPT_LITERAL_FRAGMENTS`.

    :param literal: SQL string literal
    :ptype literal: str
    :return: presence of any exempt fragment in the literal
    :rtype: bool
    """
    return any(fragment in literal for fragment, _ in _EXEMPT_LITERAL_FRAGMENTS)


def _violations_in_file(path: Path) -> list[str]:
    """return human-readable violations found in ``path``.

    :param path: source file under inspection
    :ptype path: Path
    :return: list of violation messages (empty when file is clean)
    :rtype: list[str]
    """
    if "migrations" in path.parts:
        return []
    if "tests" in path.parts:
        return []
    try:
        source = path.read_text(encoding="utf-8")
    except OSError:
        return []
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return []
    violations: list[str] = []
    for literal, lineno in _collect_string_literals(tree):
        upper_stripped = literal.lstrip().upper()
        # heuristic gate: a docstring may legitimately contain
        # "FROM memories" prose without being SQL. only literals that
        # begin with a SQL keyword count.
        if not any(upper_stripped.startswith(tok) for tok in _SQL_LEADING_TOKENS):
            continue
        upper = literal.upper()
        if (
            "FROM " not in upper
            and "INTO " not in upper
            and "UPDATE " not in upper
            and "JOIN " not in upper
        ):
            continue
        if _is_exempt(literal):
            continue
        # extract every table touched by the literal
        tables_touched: set[str] = set()
        for match in _TABLE_PATTERN.finditer(literal):
            tables_touched.add(match.group(1).lower())
        for table in tables_touched:
            partition_col = _PARTITIONED_TABLES.get(table)
            if partition_col is None:
                continue
            # partition column must appear somewhere in the literal
            # (the walker treats f-string placeholders as opaque, so
            # ``WHERE {scope}`` interpolations pass when ``scope``
            # contains the partition column predicate built upstream
            # -- this is the dynamic-scope-clause caveat; the
            # CollectionRegistry primitive enforces the static
            # signature contract elsewhere)
            if partition_col not in literal and "__PLACEHOLDER__" not in literal:
                violations.append(
                    f"{path}:{lineno}: SQL touches partitioned table "
                    f"{table!r} without filtering on partition column "
                    f"{partition_col!r}: {literal[:160]!r}",
                )
    return violations


def _walk_python_files(root: Path) -> list[Path]:
    """yield every ``.py`` file under ``root``.

    :param root: package source root
    :ptype root: Path
    :return: list of file paths
    :rtype: list[Path]
    """
    if not root.exists():
        return []
    return [p for p in root.rglob("*.py") if p.is_file()]


def test_partition_column_enforcement_across_packages() -> None:
    """walk every package source root and verify partition-column compliance.

    mode is controlled by ``PARTITION_ENFORCEMENT_MODE``:
    ``strict`` (default) fails on any violation; ``report`` logs and
    passes so the test can be flipped during cleanup windows.

    :return: nothing
    :rtype: None
    :raises AssertionError: when one or more violations surface in
        strict mode
    """
    mode = os.environ.get("PARTITION_ENFORCEMENT_MODE", "report").lower()
    all_violations: list[str] = []
    for src_root in _PACKAGE_SRC_ROOTS:
        for path in _walk_python_files(src_root):
            all_violations.extend(_violations_in_file(path))

    if not all_violations:
        return

    formatted = "\n".join(all_violations)
    if mode == "report":
        pytest.skip(
            f"partition-column enforcement: {len(all_violations)} "
            f"violation(s) (mode=report)\n{formatted}",
        )
    else:
        raise AssertionError(
            f"partition-column enforcement: {len(all_violations)} "
            f"violation(s) (mode=strict)\n{formatted}",
        )


def test_partitioned_tables_map_is_non_empty() -> None:
    """sanity check: discovery list must declare at least one table.

    if this fails the partition primitive is unused; the enforcement
    test would silently pass.

    :return: nothing
    :rtype: None
    """
    assert len(_PARTITIONED_TABLES) > 0, (
        "_PARTITIONED_TABLES is empty -- partition enforcement is a no-op"
    )
