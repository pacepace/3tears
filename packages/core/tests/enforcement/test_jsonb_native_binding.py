"""enforcement: jsonb columns are bound NATIVELY, never hand-rolled (collections-task-04).

The platform registers ONE text-format jsonb codec (``init_connection``) whose
encoder is ``json.dumps``. A collection write that ``json.dumps``-es a value AND
binds it ``$N::jsonb`` / ``$N::text::jsonb`` double-encodes the cell into a JSON
STRING scalar (the restart-time corruption that shipped green because no-codec
test pools never reproduced it). collections-task-04 removed that hand-rolled
duplication: jsonb columns are bound through ``encode_jsonb`` as native python
objects and the codec applies the single encode.

This gate keeps it from coming back: inside any ``save_to_postgres`` method
across the platform's collection modules, a ``::text::jsonb`` cast in a SQL
literal OR a ``json.dumps(`` call is a reintroduced hand-rolled jsonb encode and
FAILS THE BUILD. The canonical mechanism (``encode_jsonb`` + native ``$N`` bind)
needs neither.

Mode: ``JSONB_ENFORCEMENT_MODE`` (default ``strict``). AST-only, no dynamic
execution, well under 15s. Exemptions in ``_jsonb_native_binding_exemptions.txt``
with mandatory specific rationales.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest

# repo root is 3tears/ (this file is packages/core/tests/enforcement/...).
_REPO_ROOT = Path(__file__).resolve().parents[4]
_PACKAGES_ROOT = _REPO_ROOT / "packages"
_EXEMPTIONS_FILE = Path(__file__).resolve().parent / "_jsonb_native_binding_exemptions.txt"

#: the write method whose body must not hand-roll jsonb encoding.
_WRITE_METHODS = frozenset({"save_to_postgres"})
#: the cast workaround Option B removed (native bind needs no cast).
_BANNED_CAST = "::text::jsonb"

_BLANKET_RATIONALES = frozenset({"internal", "needed", "n/a", "legacy", "tests need this"})
_MIN_RATIONALE_LEN = 25


def _mode() -> str:
    """resolve the enforcement mode from the environment.

    :return: ``"strict"`` (default) or ``"report"``
    :rtype: str
    """
    return os.environ.get("JSONB_ENFORCEMENT_MODE", "strict").strip().lower()


def parse_jsonb_exemptions(text: str) -> set[str]:
    """parse the exemption text into a set of exempt ``path::method`` keys.

    every entry MUST be preceded by a ``# rationale: <specific reason>`` line;
    the rationale must be specific (not blanket, past the minimum length).

    :param text: raw exemptions-file contents
    :ptype text: str
    :return: the set of exempted ``relative/path.py::method`` keys
    :rtype: set[str]
    :raises AssertionError: when an entry has no rationale or it is blanket / short
    """
    exempt: set[str] = set()
    pending: str | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("# rationale:"):
            pending = line[len("# rationale:") :].strip()
            continue
        if line.startswith("#"):
            continue
        assert pending is not None, f"jsonb exemption {line!r} has no preceding '# rationale:' line"
        assert pending.lower() not in _BLANKET_RATIONALES, (
            f"jsonb exemption {line!r} has a blanket rationale {pending!r}; "
            "state why this write cannot bind the jsonb column natively via encode_jsonb"
        )
        assert len(pending) >= _MIN_RATIONALE_LEN, f"jsonb exemption {line!r} rationale {pending!r} is too vague"
        exempt.add(line)
        pending = None
    return exempt


def _collection_modules() -> list[Path]:
    """collect every ``collections.py`` module under ``packages/*/src``.

    :return: sorted module paths
    :rtype: list[Path]
    """
    return sorted(
        p for p in _PACKAGES_ROOT.rglob("src/**/collections.py") if p.stat().st_size > 0 and "/tests/" not in str(p)
    )


def _strip_sql_line_comments(text: str) -> str:
    """remove SQL ``-- ...`` line comments from a (possibly multi-line) string.

    the gate checks for a real ``::text::jsonb`` CAST, not the token appearing
    in an explanatory ``-- ...`` SQL comment inside the query string. stripping
    the comments first keeps the guard from firing on documentation.

    :param text: a SQL string literal
    :ptype text: str
    :return: the string with each line's ``--`` comment removed
    :rtype: str
    """
    cleaned: list[str] = []
    for line in text.splitlines():
        idx = line.find("--")
        cleaned.append(line if idx == -1 else line[:idx])
    return "\n".join(cleaned)


def _string_constants(node: ast.AST) -> list[str]:
    """collect every string constant under ``node`` with SQL comments stripped.

    :param node: AST subtree
    :ptype node: ast.AST
    :return: the string-constant values found (``-- ...`` comments removed)
    :rtype: list[str]
    """
    return [
        _strip_sql_line_comments(child.value)
        for child in ast.walk(node)
        if isinstance(child, ast.Constant) and isinstance(child.value, str)
    ]


def _calls_json_dumps(node: ast.AST) -> bool:
    """report whether ``node`` calls ``json.dumps`` (the hand-rolled encode).

    :param node: AST subtree
    :ptype node: ast.AST
    :return: ``True`` when a ``json.dumps`` call is present
    :rtype: bool
    """
    found = False
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            func = child.func
            if isinstance(func, ast.Attribute) and func.attr == "dumps":
                found = True
                break
    return found


def _violations_in_module(tree: ast.Module) -> list[str]:
    """return the hand-rolled-jsonb violations among a module's write methods.

    :param tree: parsed module
    :ptype tree: ast.Module
    :return: ``method: reason`` strings for each offending write method
    :rtype: list[str]
    """
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in _WRITE_METHODS:
            if any(_BANNED_CAST in s for s in _string_constants(node)):
                violations.append(f"{node.name}: binds a jsonb column with a {_BANNED_CAST} cast")
            if _calls_json_dumps(node):
                violations.append(f"{node.name}: calls json.dumps to hand-roll a jsonb write")
    return violations


_MODULES = _collection_modules()
_MODULE_IDS = [str(p.relative_to(_REPO_ROOT)) for p in _MODULES]


class TestJsonbNativeBinding:
    """no collection write hand-rolls jsonb encoding."""

    @pytest.mark.parametrize("module", _MODULES, ids=_MODULE_IDS)
    def test_no_hand_rolled_jsonb_write(self, module: Path) -> None:
        """a save_to_postgres must bind jsonb natively (encode_jsonb), no cast/dumps."""
        rel = str(module.relative_to(_REPO_ROOT))
        exempt = parse_jsonb_exemptions(_EXEMPTIONS_FILE.read_text())
        tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
        violations = [
            f"  {rel}::{v.split(':', 1)[0]} -> {v}"
            for v in _violations_in_module(tree)
            if f"{rel}::{v.split(':', 1)[0]}" not in exempt
        ]
        if violations and _mode() == "strict":
            pytest.fail(
                "hand-rolled jsonb write(s) found (collections-task-04 bound these natively via "
                "encode_jsonb + a plain $N param under the registered codec):\n"
                + "\n".join(violations)
                + "\n\nbind the jsonb column as encode_jsonb(value) with a plain $N (no ::text::jsonb), "
                "or exempt with a specific rationale in _jsonb_native_binding_exemptions.txt."
            )

    def test_scope_sanity_scans_the_datasource_collections(self) -> None:
        """the walker actually reaches the migrated datasource collections."""
        rels = [str(p.relative_to(_REPO_ROOT)) for p in _MODULES]
        assert any("datasources" in r for r in rels), (
            "the datasource collections module was not discovered -- the gate would pass "
            "vacuously; fix the discovery glob"
        )


class TestJsonbExemptionIntegrity:
    """the exemption parser enforces specific rationales (meta gate)."""

    def test_real_exemptions_file_parses(self) -> None:
        """the committed exemptions file parses without error."""
        parse_jsonb_exemptions(_EXEMPTIONS_FILE.read_text())

    def test_entry_without_rationale_is_rejected(self) -> None:
        """an entry with no preceding rationale is rejected."""
        with pytest.raises(AssertionError):
            parse_jsonb_exemptions("packages/x/src/x/collections.py::save_to_postgres\n")

    def test_blanket_rationale_is_rejected(self) -> None:
        """a blanket rationale ('legacy') is rejected."""
        with pytest.raises(AssertionError):
            parse_jsonb_exemptions("# rationale: legacy\npackages/x/src/x/collections.py::save_to_postgres\n")
