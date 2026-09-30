"""enforcement: a migration's catalog lookup names the schema it is looking in.

Package migrations run once per agent schema, several of which can share a
database, and ``public`` is on every search path. A guard that asks the catalog
"does constraint X exist?" without saying where finds another schema's X and
skips its own step. That is how memory v022 left a second agent schema without
its ``uq_*`` constraints: the lookup matched the first schema's.

So every SQL string in a migration module (and in the core migration helpers)
that reads ``pg_constraint``, ``pg_class``, ``pg_index``, ``pg_indexes``,
``pg_trigger`` or ``information_schema`` must also filter on the schema --
``current_schema()``, or a schema the caller passes and the SQL binds. The
scan is static; a string is SQL when it carries ``SELECT`` and ``FROM``.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

_CATALOG = re.compile(
    r"\b(pg_constraint|pg_class|pg_indexes|pg_index|pg_trigger|to_regclass|information_schema\.\w+)\b"
)

# The ways a lookup names its schema: the current one, or one it was handed.
_SCHEMA_FILTER = re.compile(
    r"current_schema\(\)|table_schema\s*=|schemaname\s*=|nspname\s*=|relnamespace\s*=|constraint_schema\s*=",
    re.IGNORECASE,
)


def _migration_sources() -> list[Path]:
    """Every migration module in the workspace, plus the core helpers they call.

    :return: source files to scan
    :rtype: list[Path]
    """
    found = [
        path
        for path in sorted((_REPO_ROOT / "packages").rglob("*.py"))
        if "migrations" in path.parts and "src" in path.parts and "tests" not in path.parts
    ]
    helpers = _REPO_ROOT / "packages" / "core" / "src" / "threetears" / "core" / "data" / "migrations" / "helpers.py"
    if helpers not in found:
        found.append(helpers)
    return found


def _sql_strings(path: Path) -> list[tuple[int, str]]:
    """Each string in ``path`` that is SQL reading the catalog, with its line.

    f-strings are joined with a placeholder for their interpolated parts, so a
    schema passed in by the caller still shows as ``= {}``-shaped text.

    :param path: a source file
    :ptype path: Path
    :return: ``(line, text)`` pairs
    :rtype: list[tuple[int, str]]
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    inner: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            inner.update(id(v) for v in node.values)
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in inner:
            text = node.value
        elif isinstance(node, ast.JoinedStr):
            text = "".join(v.value if isinstance(v, ast.Constant) else "{}" for v in node.values)
        else:
            continue
        upper = text.upper()
        if "SELECT" in upper and "FROM" in upper and _CATALOG.search(text):
            found.append((node.lineno, text))
    return found


def _offences(path: Path) -> list[str]:
    """Where ``path`` reads the catalog without naming a schema.

    :param path: a source file
    :ptype path: Path
    :return: ``file:line`` for each
    :rtype: list[str]
    """
    return [
        f"{path.relative_to(_REPO_ROOT)}:{line}" for line, text in _sql_strings(path) if not _SCHEMA_FILTER.search(text)
    ]


def test_every_catalog_lookup_in_a_migration_names_its_schema() -> None:
    offenders = [o for path in _migration_sources() for o in _offences(path)]
    assert not offenders, (
        "these catalog lookups do not filter on a schema, so they can find another agent schema's object "
        "and skip their own step. Add `AND n.nspname = current_schema()` (or table_schema = current_schema()):\n  "
        + "\n  ".join(offenders)
    )


def test_the_scan_finds_migrations() -> None:
    """A scan that reads nothing passes everything; make sure it reads the tree."""
    assert len(_migration_sources()) > 50


def test_an_unscoped_lookup_is_caught(tmp_path: Path) -> None:
    planted = tmp_path / "planted.py"
    planted.write_text(
        'SQL = """\nDO $$ BEGIN\n  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = \'x\') THEN\n'
        '    ALTER TABLE t ADD CONSTRAINT x CHECK (true);\n  END IF;\nEND $$\n"""\n'
    )
    assert len(_sql_strings(planted)) == 1
    planted_rel = planted
    assert [o.rsplit(":", 1)[1] for o in _offences_of(planted_rel)] == ["1"]


def test_a_scoped_lookup_passes(tmp_path: Path) -> None:
    planted = tmp_path / "planted.py"
    planted.write_text(
        'SQL = "SELECT 1 FROM pg_constraint c JOIN pg_class t ON t.oid = c.conrelid '
        "JOIN pg_namespace n ON n.oid = t.relnamespace WHERE c.conname = 'x' AND n.nspname = current_schema()\"\n"
    )
    assert _offences_of(planted) == []


def _offences_of(path: Path) -> list[str]:
    """:func:`_offences` for a file outside the repo, keyed by line only.

    :param path: a source file anywhere
    :ptype path: Path
    :return: ``name:line`` for each unscoped lookup
    :rtype: list[str]
    """
    return [f"{path.name}:{line}" for line, text in _sql_strings(path) if not _SCHEMA_FILTER.search(text)]
