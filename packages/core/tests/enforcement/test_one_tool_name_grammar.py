"""Enforcement -- the tool-name BUILDER and PARSER are pinned to each other.

``namespaces.name`` for a tool row is written by one process
and rebuilt for comparison by several others, in two repositories and
across a NATS boundary. Nothing raises when the two spellings disagree:
the row is simply not found, the tool becomes unaddressable and the
dispatch denies, with the row sitting there under a name one character
different. So the agreement is enforced rather than described.

Two rules, and they close different holes.

**The grammar is spelled ONCE.** ``build_namespace_name`` is the
generic builder and it SANITIZES every segment; the tool grammar
deliberately does not sanitize the mcp name, so a call site that
composes the tool name out of the generic builder plus the plural
prefix produces the OLD shape and no error. That was a live spelling in
the registry's dispatch authorizer, and it is exactly the shape this
rule refuses.

**Everything the builder emits, the parser accepts.** An AST rule
cannot see a grammar drift inside the two functions themselves, so the
round trip is exercised over a generated space rather than over one
example -- segment counts and version shapes crossed, including the
two version shapes production actually carries.
"""

from __future__ import annotations

import ast
import re
from itertools import product
from pathlib import Path

import pytest

from threetears.core.namespaces import (
    PLURAL_PREFIX_TOOL,
    build_tool_namespace_name,
    parse_tool_namespace_name,
    sanitize_segment,
)

__all__: list[str] = []


# resolves to the 3tears repo root: this file lives at
# packages/core/tests/enforcement/, four levels under the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent

#: the one module permitted to spell the tool-name grammar out.
_GRAMMAR_MODULE = _REPO_ROOT / "packages" / "core" / "src" / "threetears" / "core" / "namespaces.py"

#: every package source root, mirroring the containment enforcement's
#: walk -- a partial list is how a package's own copy of a banned
#: pattern survives a sweep.
_PACKAGE_SRC_ROOTS: list[Path] = sorted(path for path in _REPO_ROOT.glob("packages/*/src") if path.is_dir()) + sorted(
    path for path in _REPO_ROOT.glob("packages/agent/*/src") if path.is_dir()
)

#: a literal that roots a name at the tool prefix. matched against
#: rendered source so both the constant and the bare string are caught.
_TOOL_PREFIX_LITERAL = re.compile(rf"""(PLURAL_PREFIX_TOOL|["']{PLURAL_PREFIX_TOOL}\.?["'])""")


def _walk_source_files(root: Path) -> list[Path]:
    """return every ``.py`` file under ``root``.

    :param root: directory to walk
    :ptype root: Path
    :return: every python file beneath it
    :rtype: list[Path]
    """
    return [path for path in root.rglob("*.py") if path.is_file()]


def _generic_builder_calls_rooted_at_tools(tree: ast.AST) -> list[tuple[int, str]]:
    """return ``build_namespace_name`` calls whose first argument is the tool prefix.

    the generic builder sanitizes every segment, so rooting it at the
    tool prefix reproduces the pre-cutover flattened shape while
    reading like the current one.

    :param tree: parsed module
    :ptype tree: ast.AST
    :return: ``[(lineno, rendered_call)]`` for each offending call
    :rtype: list[tuple[int, str]]
    """
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
        if name == "build_namespace_name" and _TOOL_PREFIX_LITERAL.search(ast.unparse(node.args[0])):
            found.append((node.lineno, ast.unparse(node)))
    return found


def _violations_in(path: Path) -> list[tuple[Path, int, str]]:
    """return ``[(path, lineno, call)]`` for offending calls in one file.

    :param path: source file to inspect
    :ptype path: Path
    :return: one entry per second spelling of the tool grammar
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
    return [(path, lineno, call) for lineno, call in _generic_builder_calls_rooted_at_tools(tree)]


class TestOneToolNameGrammar:
    """no second spelling of the tool namespace name."""

    def test_no_package_builds_a_tool_name_from_the_generic_builder(self) -> None:
        violations: list[tuple[Path, int, str]] = []
        for root in _PACKAGE_SRC_ROOTS:
            for path in _walk_source_files(root):
                if path == _GRAMMAR_MODULE:
                    continue
                violations.extend(_violations_in(path))
        if violations:
            rendered = "\n".join(
                f"  {path.relative_to(_REPO_ROOT)}:{lineno} -- {call}" for path, lineno, call in violations
            )
            pytest.fail(
                "a tool namespace name is being composed from the GENERIC builder, which "
                "sanitizes every segment and so flattens the mcp name back to the "
                "pre-cutover shape. the row it looks up will not be found and the "
                "dispatch will deny. call "
                "threetears.core.namespaces.build_tool_namespace_name instead:\n" + rendered,
            )


class TestTheBuilderAndParserAreInverses:
    """the round trip, over a generated space rather than one example."""

    #: mcp-name segment counts worth crossing. one segment is the
    #: degenerate case the parser's two-component floor sits on; four
    #: is past anything production carries, so a grammar that quietly
    #: assumed a fixed arity fails here.
    _SEGMENT_COUNTS = (1, 2, 3, 4)

    #: production carries BOTH shapes -- the ``aibots`` / ``threetears``
    #: families a two-part ``1.0``, the ``addrnorm`` family a
    #: three-part ``1.0.0``. a corpus drawn from one stack has only the
    #: first, and a parser tuned to it recovers the wrong version for
    #: the other.
    _VERSIONS = ("1.0", "1.0.0", "2", "10.20.30")

    def test_every_name_the_builder_emits_parses_back_to_its_inputs(self) -> None:
        for count, version in product(self._SEGMENT_COUNTS, self._VERSIONS):
            mcp_name = ".".join(f"seg{index}" for index in range(count))
            built = build_tool_namespace_name(mcp_name, version)
            parsed = parse_tool_namespace_name(built)
            assert parsed.mcp_name == mcp_name, f"mcp name lost for {built!r}"
            assert parsed.version_segment == sanitize_segment(version), f"version lost for {built!r}"

    def test_rebuilding_from_a_parse_is_the_identity(self) -> None:
        for count, version in product(self._SEGMENT_COUNTS, self._VERSIONS):
            mcp_name = ".".join(f"seg{index}" for index in range(count))
            built = build_tool_namespace_name(mcp_name, version)
            parsed = parse_tool_namespace_name(built)
            assert build_tool_namespace_name(parsed.mcp_name, parsed.version_segment) == built

    def test_a_hyphen_and_a_dot_in_the_mcp_name_never_collide(self) -> None:
        # the collision the generic builder admits and the tool grammar
        # closes. asserted here, in the enforcement suite, because a
        # regression to the sanitized spelling would silently merge two
        # tools onto one row and pass every unit test about one of them.
        for version in self._VERSIONS:
            assert build_tool_namespace_name("a.b", version) != build_tool_namespace_name("a-b", version)
