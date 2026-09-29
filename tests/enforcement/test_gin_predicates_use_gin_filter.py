"""enforcement: a GIN predicate that can need several scan entries goes through ``gin_filter``.

YugabyteDB implements ``USING gin`` as ``ybgin``, which serves a scan with exactly one
required entry and REFUSES any other when the planner picks the index -- the query fails
with ``unsupported ybgin index scan``; it does not degrade. It shipped once in memory,
conversations and skills search together, and showed up on cobalt-dev as a hub ERROR on
every agent turn whose text contained "or": the agent answered without its keyword memory,
and nothing else said so. :func:`threetears.core.data.gin.gin_filter` renders such a
predicate as a row filter the planner cannot serve from the index; this guard makes it the
only way to write one.

**The rule is measured, not enumerated.** Each shape below was run on YugabyteDB with the
GIN index forced by plan hint; ``scripts/probe-ybgin-shapes.py`` re-runs that measurement:

- REFUSED: a full-text ``@@`` whose query can carry OR or NOT -- ``websearch_to_tsquery``
  (an "or" or a leading "-" in the text), ``to_tsquery``, a pre-built ``$n::tsquery``, and
  the reversed ``<tsquery> @@ <column>`` order alike; ``?|`` (jsonb any-key); ``&&`` (array
  overlap) with more than one element, which a literal cannot rule out.
- SERVED: ``@@ plainto_tsquery(...)`` and ``@@ phraseto_tsquery(...)`` (AND / phrase),
  ``?&`` (all-keys), ``?`` and ``@>``.
- NEVER INDEXED: array ``<@``. The planner will not read the GIN index for it, so it scans
  and cannot be refused.

So a string literal fails the guard when it contains ``?|`` or ``&&``, or an ``@@`` that is
not against ``plainto_tsquery`` / ``phraseto_tsquery``, and it is not the argument of a
``gin_filter(...)`` call. Docstrings and the arguments of ``re`` calls are not SQL and are
skipped (``results?|outputs?`` is a regex alternation, not an operator).

What it cannot see: a predicate assembled from fragments that are each innocent on their
own, SQL outside ``packages/*/src``, and ``pg_trgm`` similarity (``name % $n``), which
YugabyteDB also refuses but whose ``%`` cannot be told from a format specifier, a modulo or a
LIKE wildcard in a string literal. No package here uses trigram similarity. Static parsing only -- no imports executed, no
network -- consistent with the rest of ``tests/enforcement``.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PACKAGE_GLOBS = ("packages/*/src", "packages/agent/*/src")

#: the module that defines gin_filter (its docstring quotes the operators).
_DEFINING_MODULE = "packages/core/src/threetears/core/data/gin.py"

_ANY_FULL_TEXT_MATCH = re.compile(r"@@")
_INDEX_SERVED_FULL_TEXT = re.compile(
    r"@@\s*(?:plainto_tsquery|phraseto_tsquery)\s*\(|(?:plainto_tsquery|phraseto_tsquery)\s*\([^()]*\)\s*@@"
)
_ANY_OF = re.compile(r"\?\||&&")


def _refused_shape(text: str) -> bool:
    """Say whether SQL text uses a GIN operator YugabyteDB's index can refuse.

    :param text: the literal characters of one string
    :ptype text: str
    :return: whether ``?|`` or ``&&`` appears, or an ``@@`` not against plainto / phraseto
    :rtype: bool
    """
    unserved_full_text = len(_ANY_FULL_TEXT_MATCH.findall(text)) > len(_INDEX_SERVED_FULL_TEXT.findall(text))
    return unserved_full_text or bool(_ANY_OF.search(text))


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


def _call_arguments(tree: ast.Module, *, gin: bool) -> set[int]:
    """Return the ids of every node inside the arguments of ``gin_filter`` or of an ``re`` call.

    :param tree: parsed module
    :ptype tree: ast.Module
    :param gin: ``True`` for ``gin_filter(...)`` arguments, ``False`` for ``re.<fn>(...)`` ones
    :ptype gin: bool
    :return: ``id()`` of each argument node and every node inside it
    :rtype: set[int]
    """
    found: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if gin:
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            matched = name == "gin_filter"
        else:
            matched = isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "re"
        if matched:
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


def find_unfiltered_gin_predicates(source: str, rel: str) -> tuple[list[str], int]:
    """Scan one module's source for refused GIN shapes outside ``gin_filter``.

    :param source: the module's text
    :ptype source: str
    :param rel: repo-relative path, for the report
    :ptype rel: str
    :return: offender lines, and how many refused shapes were correctly wrapped
    :rtype: tuple[list[str], int]
    """
    tree = ast.parse(source)
    skipped = _docstring_nodes(tree) | _call_arguments(tree, gin=False)
    wrapped_ids = _call_arguments(tree, gin=True)
    f_string_parts = {id(value) for node in ast.walk(tree) if isinstance(node, ast.JoinedStr) for value in node.values}
    offenders: list[str] = []
    wrapped = 0
    for node in ast.walk(tree):
        if id(node) in skipped or id(node) in f_string_parts:
            continue
        text = _literal_text(node)
        if text is None or not _refused_shape(text):
            continue
        if id(node) in wrapped_ids:
            wrapped += 1
        else:
            offenders.append(f"{rel}:{getattr(node, 'lineno', '?')}: {text.strip()[:100]!r}")
    return offenders, wrapped


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


def test_every_refused_gin_shape_goes_through_gin_filter() -> None:
    files = _source_files()
    assert files, "no package source found -- the scan would pass having read nothing"
    offenders: list[str] = []
    wrapped = 0
    for path in files:
        rel = path.relative_to(_REPO_ROOT).as_posix()
        if rel == _DEFINING_MODULE:
            continue
        found, ok = find_unfiltered_gin_predicates(path.read_text(encoding="utf-8"), rel)
        offenders.extend(found)
        wrapped += ok
    assert wrapped, "no gin_filter call found -- the guard's positive case is gone, so it proves nothing"
    assert not offenders, (
        "these SQL literals use a GIN operator YugabyteDB's index refuses outright; pass the "
        "predicate through threetears.core.data.gin.gin_filter:\n" + "\n".join(offenders)
    )


@pytest.mark.parametrize(
    "sql",
    [
        "WHERE search_vector @@ websearch_to_tsquery('english', $1)",
        "WHERE search_vector @@ to_tsquery('english', $1)",
        "WHERE search_vector @@ $1::tsquery",
        "WHERE websearch_to_tsquery('english', $1) @@ search_vector",
        "WHERE tags ?| $2::text[]",
        "WHERE tags?|$2",
        "WHERE labels && $3",
        "WHERE labels && ARRAY['a', 'b']",
    ],
)
def test_each_refused_shape_is_caught_when_unwrapped(sql: str) -> None:
    offenders, wrapped = find_unfiltered_gin_predicates(f"QUERY = {sql!r}\n", "probe.py")
    assert offenders and not wrapped


@pytest.mark.parametrize(
    "sql",
    [
        "WHERE search_vector @@ plainto_tsquery('english', $1)",
        "WHERE search_vector @@ phraseto_tsquery('english', $1)",
        "WHERE plainto_tsquery('english', $1) @@ search_vector",
        "WHERE tags ?& $2::text[]",
        "WHERE tags @> $2::jsonb",
        "WHERE labels <@ $3",
    ],
)
def test_each_served_shape_is_left_alone(sql: str) -> None:
    offenders, wrapped = find_unfiltered_gin_predicates(f"QUERY = {sql!r}\n", "probe.py")
    assert not offenders and not wrapped


def test_a_wrapped_predicate_passes_and_counts() -> None:
    source = "QUERY = gin_filter(f\"search_vector @@ websearch_to_tsquery('english', ${n})\")\n"
    offenders, wrapped = find_unfiltered_gin_predicates(source, "probe.py")
    assert not offenders
    assert wrapped == 1


def test_docstrings_and_regex_patterns_are_not_sql() -> None:
    source = (
        '"""mentions search_vector @@ websearch_to_tsquery and tags ?| in prose."""\n'
        "import re\n"
        "PATTERN = re.compile(r'(?:results?|outputs?)')\n"
    )
    offenders, _ = find_unfiltered_gin_predicates(source, "probe.py")
    assert not offenders
