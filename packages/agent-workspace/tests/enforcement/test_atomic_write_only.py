"""enforcement: no bare ``open(path, "w"...)`` in workspace write paths.

the workspace package routes every file content write through
:func:`threetears.core.utils.atomic_write.atomic_write` -- a helper that
writes to a temp sibling and renames into place so crash recovery never
sees a half-written file. this test walks the AST of every module under
``src/threetears/agent/workspace/tools/`` plus ``materialize.py`` and
flags any ``open(...)`` call (built-in) or ``.open(...)`` call whose
mode argument is writable (``"w"``, ``"wb"``, ``"a"``, ``"ab"``, ``"x"``,
``"xb"``, ``"w+"``, ``"rb+"``, ``"r+b"``, etc.). read-only ``open(path,
"r")`` usages would be ignored by this test; in practice the modules in
scope never call ``open`` at all.

allow-list: modules that by contract handle no file content writes
(``validators.py``, ``audit.py``, ``pin.py``, ``entities.py``,
``collections.py``, ``migrations.py``, ``config.py``, every package
``__init__.py``). the test does not scan those paths so they remain
free to ``open`` (or not) as they see fit.
"""

from __future__ import annotations

import ast
from pathlib import Path


_SRC_ROOT = (
    Path(__file__).resolve().parent.parent.parent
    / "src"
    / "threetears"
    / "agent"
    / "workspace"
)
_TOOLS_ROOT = _SRC_ROOT / "tools"
_MATERIALIZE_MODULE = _SRC_ROOT / "materialize.py"

# any mode containing a writable mode letter is forbidden. "rb" / "r"
# read-only modes are fine; write, append, exclusive-create, and
# read-update ("r+") modes mutate the file and must route through
# atomic_write. ordering is irrelevant: python accepts mode letters in
# any order.
_WRITE_MODE_LETTERS = frozenset({"w", "a", "x", "+"})


def _scan_modules() -> list[Path]:
    """
    collect the set of ``.py`` files subject to the atomic-write rule.

    every file under ``tools/`` plus ``materialize.py``. excludes
    ``__init__.py`` files because they are import re-exports only.

    :return: sorted list of module paths
    :rtype: list[Path]
    """
    modules: list[Path] = []
    for path in sorted(_TOOLS_ROOT.rglob("*.py")):
        if path.name == "__init__.py":
            continue
        modules.append(path)
    if _MATERIALIZE_MODULE.is_file():
        modules.append(_MATERIALIZE_MODULE)
    return modules


def _extract_mode_string(call: ast.Call) -> str | None:
    """
    return the literal string mode arg of an ``open`` / ``.open`` call, if present.

    handles positional (second arg) and keyword (``mode=...``) forms.
    a non-literal expression returns ``None`` so the scanner can flag
    the call conservatively (the test fails on non-literal modes too,
    since a write mode could be hidden behind a variable).

    :param call: AST Call node for the ``open`` invocation
    :ptype call: ast.Call
    :return: literal mode string, or None if mode is missing / non-literal
    :rtype: str | None
    """
    result: str | None = None
    # positional second arg
    if len(call.args) >= 2:
        arg = call.args[1]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            result = arg.value
    # keyword mode=...
    for kw in call.keywords:
        if kw.arg == "mode":
            if isinstance(kw.value, ast.Constant) and isinstance(
                kw.value.value, str
            ):
                result = kw.value.value
    return result


def _is_writable_mode(mode: str) -> bool:
    """
    true iff the mode string contains any writable mode letter.

    :param mode: python file mode string (e.g. ``"w"``, ``"rb"``, ``"a+"``)
    :ptype mode: str
    :return: True if mode mutates the file
    :rtype: bool
    """
    return any(ch in _WRITE_MODE_LETTERS for ch in mode)


def _is_builtin_open_call(call: ast.Call) -> bool:
    """
    true iff ``call.func`` is a bare ``Name("open")``.

    :param call: AST Call node
    :ptype call: ast.Call
    :return: True for builtin open
    :rtype: bool
    """
    return isinstance(call.func, ast.Name) and call.func.id == "open"


def _is_attribute_open_call(call: ast.Call) -> bool:
    """
    true iff ``call.func`` is ``<expr>.open`` (Path.open and siblings).

    :param call: AST Call node
    :ptype call: ast.Call
    :return: True for attribute-style ``.open`` calls
    :rtype: bool
    """
    return isinstance(call.func, ast.Attribute) and call.func.attr == "open"


class TestAtomicWriteOnly:
    """no write-mode open() in workspace tool / materialize modules."""

    def test_no_writable_open_in_scanned_modules(self) -> None:
        """
        AST-walk every scanned module; fail on any writable ``open``.

        :return: None
        :rtype: None
        """
        violations: list[str] = []
        for path in _scan_modules():
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                is_open = _is_builtin_open_call(node) or _is_attribute_open_call(
                    node
                )
                if not is_open:
                    continue
                mode = _extract_mode_string(node)
                if mode is None:
                    # no explicit mode arg == defaults to "r" (read).
                    # we still flag non-literal mode expressions so a
                    # callable variable cannot smuggle write intent past
                    # the scan.
                    if len(node.args) >= 2 or any(
                        kw.arg == "mode" for kw in node.keywords
                    ):
                        violations.append(
                            f"{path.relative_to(_SRC_ROOT)}:{node.lineno}: "
                            f"open(...) with non-literal mode; use atomic_write"
                        )
                    continue
                if _is_writable_mode(mode):
                    violations.append(
                        f"{path.relative_to(_SRC_ROOT)}:{node.lineno}: "
                        f"open(..., mode={mode!r}) -- route through atomic_write"
                    )
        assert not violations, (
            f"{len(violations)} writable open() call(s) in workspace modules:\n"
            + "\n".join(violations)
            + "\n\nroute all content writes through "
            "threetears.core.utils.atomic_write.atomic_write."
        )
