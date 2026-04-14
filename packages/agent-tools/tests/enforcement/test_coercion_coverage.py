"""enforcement: every TearsTool subclass must inherit coercion via ``execute``.

the input-coercion path (``TearsTool.execute`` -> ``normalize_kwargs`` ->
subclass ``_execute``) only runs if subclasses override ``_execute`` and
leave ``execute`` alone. a subclass that overrides ``execute`` directly
bypasses coercion silently and reintroduces the 422 bug this task fixed
(empty-string / JSON-encoded-string inputs for object/array fields).

this AST-based test scans every production Python source file under
``src/threetears/agent/tools`` and fails if any class whose base list
mentions ``TearsTool`` defines an ``execute`` method. subclasses must
override ``_execute`` instead. the test is intentionally narrow and
fast: no imports, no instantiation, pure static analysis.

behavioral coverage of the coercion itself lives in
``tests/unit/tools/test_coercion.py``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


_SRC_ROOT = (
    Path(__file__).resolve().parent.parent.parent
    / "src"
    / "threetears"
    / "agent"
    / "tools"
)


def _collect_src_files() -> list[Path]:
    """collect production Python source files to scan.

    :return: sorted list of source file paths under agent/tools src
    :rtype: list[Path]
    """
    return sorted(p for p in _SRC_ROOT.rglob("*.py") if p.name != "__init__.py")


def _base_mentions_tears_tool(bases: list[ast.expr]) -> bool:
    """return True when any base expression references ``TearsTool``.

    matches bare ``TearsTool`` and ``module.TearsTool`` attribute
    forms. structural check only -- does not resolve imports.

    :param bases: class ``bases`` field as parsed by ast
    :ptype bases: list[ast.expr]
    :return: whether any base names ``TearsTool``
    :rtype: bool
    """
    result = False
    for base in bases:
        if isinstance(base, ast.Name) and base.id == "TearsTool":
            result = True
            break
        if isinstance(base, ast.Attribute) and base.attr == "TearsTool":
            result = True
            break
    return result


_SRC_FILES = _collect_src_files()
_SRC_IDS = [str(p.relative_to(_SRC_ROOT)) for p in _SRC_FILES]


class TestTearsToolSubclassesUseExecuteHook:
    """TearsTool subclasses must override ``_execute``, never ``execute``."""

    @pytest.mark.parametrize("src_file", _SRC_FILES, ids=_SRC_IDS)
    def test_subclass_does_not_override_execute(self, src_file: Path) -> None:
        """verify no TearsTool subclass in this file defines ``execute``.

        overriding ``execute`` bypasses ``normalize_kwargs`` and silently
        skips input coercion. subclasses must override ``_execute``.

        :param src_file: source file under scan
        :ptype src_file: Path
        :raises AssertionError: when a TearsTool subclass overrides ``execute``
        """
        tree = ast.parse(src_file.read_text(encoding="utf-8"), filename=str(src_file))
        violations: list[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            if node.name == "TearsTool":
                continue
            if not _base_mentions_tears_tool(node.bases):
                continue
            for member in node.body:
                if isinstance(
                    member, (ast.FunctionDef, ast.AsyncFunctionDef),
                ) and member.name == "execute":
                    violations.append(
                        f"{src_file.relative_to(_SRC_ROOT)}:{member.lineno} -- "
                        f"class {node.name} overrides execute(); rename to "
                        f"_execute() so TearsTool.execute input coercion runs"
                    )
        assert violations == [], "\n".join(violations)


class TestTearsToolSubclassesDefineExecuteHook:
    """TearsTool subclasses must define ``_execute`` (the abstract hook)."""

    @pytest.mark.parametrize("src_file", _SRC_FILES, ids=_SRC_IDS)
    def test_subclass_defines_execute_hook(self, src_file: Path) -> None:
        """verify every TearsTool subclass in this file defines ``_execute``.

        the template method in ``TearsTool.execute`` delegates to
        subclass ``_execute``. a subclass missing ``_execute`` cannot
        be instantiated (abstract) and indicates an incomplete rename.

        :param src_file: source file under scan
        :ptype src_file: Path
        :raises AssertionError: when a TearsTool subclass omits ``_execute``
        """
        tree = ast.parse(src_file.read_text(encoding="utf-8"), filename=str(src_file))
        violations: list[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            if node.name == "TearsTool":
                continue
            if not _base_mentions_tears_tool(node.bases):
                continue
            has_hook = any(
                isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))
                and member.name == "_execute"
                for member in node.body
            )
            if not has_hook:
                violations.append(
                    f"{src_file.relative_to(_SRC_ROOT)}:{node.lineno} -- "
                    f"class {node.name} extends TearsTool but does not define "
                    f"_execute(); subclasses must implement the _execute hook"
                )
        assert violations == [], "\n".join(violations)
