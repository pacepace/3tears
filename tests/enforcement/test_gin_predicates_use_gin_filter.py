"""enforcement: a GIN predicate that can need several scan entries goes through ``gin_filter``.

YugabyteDB implements ``USING gin`` as ``ybgin``, which serves a scan with exactly one
required entry and REFUSES any other when the planner picks the index -- the query fails
with ``unsupported ybgin index scan``; it does not degrade. Four operators produce such
scans from ordinary input:

- ``@@ websearch_to_tsquery(...)`` / ``@@ to_tsquery(...)`` -- websearch syntax turns an
  "or" or a leading "-" in a person's or an agent's text into OR / NOT;
- ``?|`` and ``?&`` -- jsonb any-key / all-keys over a list;
- ``&&`` -- array overlap.

It shipped once in memory, conversations and skills search at the same time, and showed up
on cobalt-dev as a hub ERROR on every agent turn whose text contained "or" -- the agent
answered without its keyword memory, and nothing else said so. The
fix is :func:`threetears.core.data.gin.gin_filter`, which renders the predicate as a row
filter the planner cannot serve from the index. This guard makes it the only way to write
one: every string literal in package source that uses one of those operators must be the
argument of a ``gin_filter(...)`` call. ``plainto_tsquery`` (plain AND) and ``@>``
(containment) have one required entry and are not refused, so they are not flagged.

What it cannot see: a predicate assembled from fragments that are each innocent on their
own, and SQL that lives outside ``packages/*/src``. The common case -- a query written as a
literal -- is the one it closes.

Static parsing only -- no imports executed, no network -- consistent with the rest of
``tests/enforcement``.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PACKAGE_GLOBS = ("packages/*/src", "packages/agent/*/src")

#: the module that defines gin_filter (its docstring quotes the operators).
_DEFINING_MODULE = "packages/core/src/threetears/core/data/gin.py"

#: operators that can need more than one required ybgin scan entry. ``?|`` / ``?&`` must stand
#: between spaces, as SQL writes them: bare, they also match a regex alternation (``results?|``).
_MULTI_ENTRY = re.compile(r"@@\s*(?:websearch_to_tsquery|to_tsquery)\s*\(|\s\?\|\s|\s\?&\s|&&\s*\$")


def _docstring_nodes(tree: ast.Module) -> set[int]:
    """Return the ids of every module, class and function docstring node.

    :param tree: parsed module
    :ptype tree: ast.Module
    :return: ``id()`` of each docstring ``Constant`` node
    :rtype: set[int]
    """
    found: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                if isinstance(body[0].value.value, str):
                    found.add(id(body[0].value))
    return found


def _gin_filter_arguments(tree: ast.Module) -> set[int]:
    """Return the ids of every node passed as the argument of a ``gin_filter(...)`` call.

    :param tree: parsed module
    :ptype tree: ast.Module
    :return: ``id()`` of each argument node, and of every node inside it
    :rtype: set[int]
    """
    found: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", None)
            if name == "gin_filter":
                for arg in node.args:
                    found.update(id(inner) for inner in ast.walk(arg))
    return found


def _literal_text(node: ast.AST) -> str | None:
    """Return the literal text of a string constant or f-string, else ``None``.

    :param node: any AST node
    :ptype node: ast.AST
    :return: the literal characters, f-string placeholders omitted
    :rtype: str | None
    """
    text: str | None = None
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        text = node.value
    elif isinstance(node, ast.JoinedStr):
        text = "".join(v.value for v in node.values if isinstance(v, ast.Constant) and isinstance(v.value, str))
    return text


def _source_files() -> list[Path]:
    """Return every Python source file under the package ``src`` trees.

    :return: sorted source paths
    :rtype: list[Path]
    """
    files: list[Path] = []
    for pattern in _PACKAGE_GLOBS:
        for src in _REPO_ROOT.glob(pattern):
            files.extend(src.rglob("*.py"))
    return sorted(files)


def test_every_multi_entry_gin_predicate_goes_through_gin_filter() -> None:
    files = _source_files()
    assert files, "no package source found -- the scan would pass having read nothing"
    offenders: list[str] = []
    wrapped = 0
    for path in files:
        rel = path.relative_to(_REPO_ROOT).as_posix()
        if rel == _DEFINING_MODULE:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docstrings = _docstring_nodes(tree)
        allowed = _gin_filter_arguments(tree)
        for node in ast.walk(tree):
            if id(node) in docstrings:
                continue
            text = _literal_text(node)
            if text is None or not _MULTI_ENTRY.search(text):
                continue
            if id(node) in allowed:
                wrapped += 1
                continue
            if isinstance(node, ast.Constant) and any(
                isinstance(parent, ast.JoinedStr) and node in parent.values for parent in ast.walk(tree)
            ):
                continue  # a piece of an f-string: the f-string itself is judged
            offenders.append(f"{rel}:{getattr(node, 'lineno', '?')}: {text.strip()[:100]!r}")
    assert wrapped, "no gin_filter call found -- the guard's positive case is gone, so it proves nothing"
    assert not offenders, (
        "these SQL literals use a GIN operator that can need several scan entries, which "
        "YugabyteDB refuses outright; pass the predicate through "
        "threetears.core.data.gin.gin_filter:\n" + "\n".join(offenders)
    )
