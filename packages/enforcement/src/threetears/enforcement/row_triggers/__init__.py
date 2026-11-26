"""no per-row trigger on a versioned (page-written) table, found in source before anything runs.

Append-only L3 (``design-append-only-l3.md``, decision 4): a versioned table (``TableSchema(row_epoch=row_versions(
...))``) is its own change record, and a per-row trigger on it switches off YugabyteDB's write batching (1.58 ms a row
against 0.43 without, 2026-11-25). The runtime half refuses one found in the catalog
(``threetears.core.ddl.epoch_catalog.refuse_row_triggers``, at every migration run's end and the keeper's install); this
rule catches the source that would create one, so it never reaches a database.

What it reads, by AST and nothing else imported:

- the versioned tables: every ``TableSchema(...)`` call whose ``row_epoch=`` is a ``row_versions(...)`` call, by its
  ``name=`` string;
- the per-row triggers: every ``CREATE [OR REPLACE] [CONSTRAINT] TRIGGER ... ON <table> ... FOR EACH ROW`` in a string
  constant of a Python file (each f-string's literal parts joined, an interpolation read as ``{}``) or in a ``.sql``
  file. A target naming an interpolation cannot be resolved here; the runtime half judges it.

A finding is ``path:line: trigger on <table>``, the table a versioned one (its bare name, any schema or quoting
dropped).
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

__all__ = ["RowTriggerFinding", "row_triggers_in", "row_triggers_on_versioned", "versioned_tables_in"]

#: a per-row trigger statement and its target: up to the first ``;`` (one statement), case-blind
_ROW_TRIGGER: Final = re.compile(
    r"CREATE\s+(?:OR\s+REPLACE\s+)?(?:CONSTRAINT\s+)?TRIGGER\s+(?P<name>[\w\"{}]+)[^;]*?\bON\s+(?P<table>[\w.\"{}]+)"
    r"[^;]*?\bFOR\s+EACH\s+ROW\b",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True, slots=True)
class RowTriggerFinding:
    """one per-row trigger statement found in source.

    :ivar path: the file
    :ivar line: the line its string starts on (1 for a ``.sql`` file's statement, its own line)
    :ivar trigger: the trigger's name as written
    :ivar table: its target's bare name, unquoted and without its schema
    """

    path: Path
    line: int
    trigger: str
    table: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: trigger {self.trigger} FOR EACH ROW on {self.table}"


def _bare(target: str) -> str:
    """a trigger target's bare table name: its last dotted part, unquoted."""
    return target.rpartition(".")[2].strip('"')


def _texts(tree: ast.AST) -> Iterable[tuple[int, str]]:
    """every string constant of a module, an f-string's literal parts joined with ``{}`` for each interpolation."""
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            parts = [
                part.value if isinstance(part, ast.Constant) and isinstance(part.value, str) else "{}"
                for part in node.values
            ]
            yield node.lineno, "".join(parts)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.lineno, node.value


def _called(node: ast.expr, name: str) -> bool:
    """whether ``node`` is a call of ``name`` (bare or as an attribute)."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    return (isinstance(func, ast.Name) and func.id == name) or (isinstance(func, ast.Attribute) and func.attr == name)


def versioned_tables_in(paths: Iterable[Path]) -> frozenset[str]:
    """the versioned tables the Python files declare: each ``TableSchema(name=..., row_epoch=row_versions(...))``.

    :param paths: the files
    :ptype paths: Iterable[Path]
    :return: their names
    :rtype: frozenset[str]
    """
    found: set[str] = set()
    for path in paths:
        if path.suffix != ".py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not _called(node, "TableSchema"):
                continue
            keywords = {each.arg: each.value for each in node.keywords if each.arg}
            name = keywords.get("name")
            epoch = keywords.get("row_epoch")
            if epoch is not None and _called(epoch, "row_versions") and isinstance(name, ast.Constant):
                if isinstance(name.value, str):
                    found.add(name.value)
    return frozenset(found)


def row_triggers_in(paths: Iterable[Path]) -> tuple[RowTriggerFinding, ...]:
    """every per-row trigger statement in the Python and ``.sql`` files.

    :param paths: the files
    :ptype paths: Iterable[Path]
    :return: the statements found, in file and line order
    :rtype: tuple[RowTriggerFinding, ...]
    """
    found: list[RowTriggerFinding] = []
    for path in paths:
        if path.suffix == ".sql":
            text = path.read_text(encoding="utf-8")
            for match in _ROW_TRIGGER.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                found.append(RowTriggerFinding(path, line, match["name"], _bare(match["table"])))
        elif path.suffix == ".py":
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for line, text in _texts(tree):
                for match in _ROW_TRIGGER.finditer(text):
                    found.append(RowTriggerFinding(path, line, match["name"], _bare(match["table"])))
    return tuple(sorted(found, key=lambda each: (str(each.path), each.line, each.trigger)))


def row_triggers_on_versioned(paths: Iterable[Path]) -> tuple[RowTriggerFinding, ...]:
    """the per-row triggers whose target is a versioned table any of the files declares.

    :param paths: the files: every source and migration file of the tree judged
    :ptype paths: Iterable[Path]
    :return: the findings; none is the rule passing
    :rtype: tuple[RowTriggerFinding, ...]
    """
    files = tuple(paths)
    versioned = versioned_tables_in(files)
    return tuple(each for each in row_triggers_in(files) if each.table in versioned)
