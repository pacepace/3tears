"""Enforcement -- ``namespace_contains`` is the ONLY containment rule.

"is this name under that one" is asked of ``namespaces.name``
values, and of the mcp names those are built from, in several places.
Asked with a raw ``name.startswith(node)`` it has a standing bug: the
node's characters can be shared by a sibling, so ``pentest`` reaches
``pentestimposter`` and ``threetears`` reaches ``threetearsimposter``.
That was worked around at the VALUE level for a while -- every
``allowed_namespaces`` entry carried a trailing dot, so the raw prefix
test happened to be segment-aware -- which left a namespace written
without the dot silently wider than it looks.

:func:`threetears.core.namespaces.namespace_contains` composes the
separator itself, so the awareness is structural. This test is what
stops a second, weaker copy appearing beside it: it walks the AST of
every package source file and fails a ``startswith`` call whose
RECEIVER is a namespace name.

The test is AST-only, deterministic and side-effect-free.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

__all__: list[str] = []


# resolves to the 3tears repo root: this file lives at
# packages/core/tests/enforcement/, four levels under the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent

#: the one module permitted to spell the containment rule out.
_CONTAINMENT_MODULE = _REPO_ROOT / "packages" / "core" / "src" / "threetears" / "core" / "namespaces.py"

#: every package source root. mirrors
#: ``test_no_datetime_type_columns.py``'s walk -- a partial list is how
#: the registry's own copy of a banned pattern survived a previous
#: sweep.
_PACKAGE_SRC_ROOTS: list[Path] = sorted(path for path in _REPO_ROOT.glob("packages/*/src") if path.is_dir()) + sorted(
    path for path in _REPO_ROOT.glob("packages/agent/*/src") if path.is_dir()
)

#: an expression whose rendered source matches this is holding a
#: namespace name (or the mcp name a namespace name is built from).
#: matched against the RECEIVER of the ``startswith`` call, because the
#: bug shape is ``<a name>.startswith(<a node>)``.
#:
#: the ``ns`` spellings are here because a guard must match the
#: DECLARATION rather than one rendering of it: ``ns``, ``ns_entity``
#: and ``target_ns`` all hold a namespace and none of them contains the
#: word. verified to produce no false positive across either repo when
#: it was widened.
_NAMESPACE_NAME_RECEIVER = re.compile(
    r"\b(namespace|namespaces|namespace_name|canonical_name|tool_namespace_name"
    r"|ns|ns_[a-z_]+|[a-z_]+_ns)\b|\btool\.name\b",
)


def _walk_source_files(root: Path) -> list[Path]:
    """return every ``.py`` file under ``root``.

    :param root: directory to walk
    :ptype root: Path
    :return: every python file beneath it
    :rtype: list[Path]
    """
    return [path for path in root.rglob("*.py") if path.is_file()]


def _startswith_calls(tree: ast.AST) -> list[ast.Call]:
    """return every ``X.startswith(...)`` call node in ``tree``.

    :param tree: parsed module
    :ptype tree: ast.AST
    :return: the call nodes
    :rtype: list[ast.Call]
    """
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "startswith"
    ]


def _violations_in(path: Path) -> list[tuple[Path, int, str]]:
    """return ``[(path, lineno, receiver_source)]`` for offending calls.

    :param path: source file to inspect
    :ptype path: Path
    :return: one entry per ``startswith`` on a namespace name
    :rtype: list[tuple[Path, int, str]]
    """
    try:
        source = path.read_text()
    except OSError, UnicodeDecodeError:
        return []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    found: list[tuple[Path, int, str]] = []
    for call in _startswith_calls(tree):
        assert isinstance(call.func, ast.Attribute)
        receiver = ast.unparse(call.func.value)
        if _NAMESPACE_NAME_RECEIVER.search(receiver):
            found.append((path, call.lineno, receiver))
    return found


class TestOneNamespaceContainment:
    """no second containment rule, and no second spelling of the one."""

    def test_no_startswith_on_a_namespace_name_outside_the_containment_module(self) -> None:
        violations: list[tuple[Path, int, str]] = []
        for root in _PACKAGE_SRC_ROOTS:
            for path in _walk_source_files(root):
                if path == _CONTAINMENT_MODULE:
                    continue
                violations.extend(_violations_in(path))
        if violations:
            rendered = "\n".join(
                f"  {path.relative_to(_REPO_ROOT)}:{lineno} -- {receiver}.startswith(...)"
                for path, lineno, receiver in violations
            )
            pytest.fail(
                "a namespace name is being tested with a raw prefix, which admits a "
                "prefix sibling (`pentest` reaching `pentestimposter`). call "
                "threetears.core.namespaces.namespace_contains instead:\n" + rendered,
            )

    def test_the_containment_module_spells_the_rule_exactly_once(self) -> None:
        tree = ast.parse(_CONTAINMENT_MODULE.read_text())
        calls = _startswith_calls(tree)
        assert len(calls) == 1, (
            "threetears.core.namespaces holds the ONE containment implementation; "
            f"found {len(calls)} startswith calls at lines {[call.lineno for call in calls]}"
        )

    def test_the_one_call_lives_inside_namespace_contains(self) -> None:
        tree = ast.parse(_CONTAINMENT_MODULE.read_text())
        containing = [
            node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and _startswith_calls(node)
        ]
        assert containing == ["namespace_contains"]

    def test_the_walk_actually_reaches_the_packages_that_matter(self) -> None:
        # a walk that silently resolves to nothing passes every
        # assertion above. name the two roots whose absence would make
        # this test decorative.
        roots = {path.relative_to(_REPO_ROOT).as_posix() for path in _PACKAGE_SRC_ROOTS}
        assert "packages/registry/src" in roots
        assert "packages/agent/acl/src" in roots
        assert len(roots) > 10
