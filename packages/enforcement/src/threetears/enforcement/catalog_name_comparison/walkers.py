"""the walkers: every SQL text in a module, and the uncast catalogue name comparisons in each.

A SQL text is a string literal, an f-string's literal parts, a ``+`` chain of those, or a ``str.join``
over a list or tuple of them -- the ways a query is written across lines.

A comparison is flagged when a catalogue name column is compared with an array (``=``, ``<>`` or ``!=``
against ``ANY``, ``ALL`` or ``SOME``) with no cast between the column and the operator:

- a column only the catalogue has (``relname``, ``nspname``, ``attname``, ...) wherever it appears, so
  an f-string placeholder naming the catalogue table does not hide it;
- a column an application table may share (``table_name``, ``table_schema``, ``column_name``) only in a
  text that reads the catalogue (``information_schema`` or a ``pg_`` relation).
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Final

__all__ = ["find_catalog_name_violations", "violations_in"]

#: ``name``-typed columns no application table is expected to have
_CATALOGUE_ONLY: Final[tuple[str, ...]] = (
    "relname",
    "nspname",
    "attname",
    "tgname",
    "proname",
    "typname",
    "conname",
    "collname",
    "rolname",
    "usename",
    "datname",
    "extname",
    "schemaname",
    "tablename",
    "indexname",
    "viewname",
    "sequencename",
)

#: ``information_schema`` columns (domains over ``name``) an application table may also have
_SHARED: Final[tuple[str, ...]] = ("table_name", "table_schema", "table_catalog", "column_name")

_ARRAY_COMPARISON: Final[str] = r"\s*(?:=|<>|!=)\s*(?:ANY|ALL|SOME)\s*\("

_CATALOGUE_ONLY_UNCAST: Final[re.Pattern[str]] = re.compile(
    rf"\b(?:\w+\.)?({'|'.join(_CATALOGUE_ONLY)}){_ARRAY_COMPARISON}", re.IGNORECASE
)
_SHARED_UNCAST: Final[re.Pattern[str]] = re.compile(
    rf"\b(?:\w+\.)?({'|'.join(_SHARED)}){_ARRAY_COMPARISON}", re.IGNORECASE
)

#: a text reads the catalogue when it names ``information_schema`` or a ``pg_`` relation
_READS_CATALOGUE: Final[re.Pattern[str]] = re.compile(r"\binformation_schema\b|\bpg_[a-z_]+\b", re.IGNORECASE)


def violations_in(sql: str) -> list[str]:
    """the uncast catalogue name columns compared with an array in one SQL text.

    :param sql: the SQL text
    :ptype sql: str
    :return: each offending column, in order
    :rtype: list[str]
    """
    found = [match.group(1) for match in _CATALOGUE_ONLY_UNCAST.finditer(sql)]
    if _READS_CATALOGUE.search(sql):
        found += [match.group(1) for match in _SHARED_UNCAST.finditer(sql)]
    return found


def _literal_text(node: ast.AST) -> str | None:
    """the SQL text one expression spells, when it is built only of string literals.

    :param node: the expression
    :ptype node: ast.AST
    :return: its text, or ``None``
    :rtype: str | None
    """
    text: str | None = None
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        text = node.value
    elif isinstance(node, ast.JoinedStr):
        text = "".join(
            part.value for part in node.values if isinstance(part, ast.Constant) and isinstance(part.value, str)
        )
    elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _literal_text(node.left), _literal_text(node.right)
        if left is not None or right is not None:
            text = (left or "") + (right or "")
    elif (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "join"
        and len(node.args) == 1
        and isinstance(node.args[0], ast.List | ast.Tuple)
    ):
        parts = [_literal_text(element) for element in node.args[0].elts]
        if any(part is not None for part in parts):
            text = " ".join(part or "" for part in parts)
    return text


def _texts(path: Path) -> Iterator[tuple[int, str]]:
    """every SQL text a module spells, with its line; a part of a larger one is read as that one, and a
    docstring is not one.

    :param path: the module
    :ptype path: Path
    :return: ``(line, text)`` pairs
    :rtype: Iterator[tuple[int, str]]
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    # a string standing alone as a statement is prose (a docstring), never a query
    inside: set[int] = {
        id(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
    }
    for node in ast.walk(tree):
        if id(node) in inside:
            continue
        text = _literal_text(node)
        if text is not None:
            for child in ast.walk(node):
                inside.add(id(child))
            yield getattr(node, "lineno", 0), text


def find_catalog_name_violations(roots: Iterable[Path], *, relative_to: Path) -> list[str]:
    """every uncast catalogue name comparison in the modules under ``roots``.

    :param roots: source directories to walk
    :ptype roots: Iterable[Path]
    :param relative_to: the path each finding is named relative to
    :ptype relative_to: Path
    :return: ``path:line column`` for each
    :rtype: list[str]
    """
    found: list[str] = []
    for root in roots:
        for path in sorted(root.rglob("*.py")):
            for line, text in _texts(path):
                found += [f"{path.relative_to(relative_to)}:{line} {column}" for column in violations_in(text)]
    return found
