"""enforcement: no production module imports stdlib ``logging`` directly.

static AST check, <5s. asserts that every non-empty ``.py`` file under
``src/threetears/langgraph/`` uses ``threetears.observe`` as its sole
logging ingress.

rationale: langgraph-logging-task-01. stdlib ``logging`` imports bypass
the ``ContextFormatter`` that renders correlation tags, call-site info,
and ``extra_data`` JSON. a single stray ``import logging`` followed by
``logging.getLogger(...)`` silently downgrades a module's log output to
plain text and drops the correlation context. since
``threetears.langgraph.builders.build_tool_agent`` is what every agent
calls, regressions here would silently bypass platform-wide structured
logging across the whole agent fleet.

the ``threetears.observe`` package itself is the only legitimate user
of stdlib ``logging`` (it is the implementation of ``get_logger``); it
lives in a sibling package and is not covered by this test.

exemptions: a module that configures a third-party logger by name
(e.g., silencing a noisy library) may need a transient stdlib reference.
mark such lines with the per-line marker ``# stdlib-logging: ok`` on the
same line.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_SRC_ROOT = (
    Path(__file__).resolve().parent.parent.parent
    / "src"
    / "threetears"
    / "langgraph"
)


# --- exemption list --------------------------------------------------------

_EXEMPT: dict[str, str] = {}

# per-line exemption marker for third-party logger configuration
_LINE_MARKER = "# stdlib-logging: ok"


def _collect_src_files() -> list[Path]:
    """collect all non-empty Python source files under the package root.

    :return: sorted list of absolute Path objects
    :rtype: list[Path]
    """
    return sorted(
        p for p in _SRC_ROOT.rglob("*.py") if p.stat().st_size > 0
    )


def _rel_posix(path: Path) -> str:
    """return path relative to the src root as POSIX-separated string.

    :param path: absolute path under ``_SRC_ROOT``
    :ptype path: Path
    :return: POSIX-separated relative path
    :rtype: str
    """
    return str(path.relative_to(_SRC_ROOT)).replace("\\", "/")


def _find_stdlib_logging_import_lines(tree: ast.Module) -> list[tuple[int, str]]:
    """return ``(lineno, description)`` for every stdlib logging import.

    :param tree: parsed AST module
    :ptype tree: ast.Module
    :return: list of ``(lineno, description)`` tuples
    :rtype: list[tuple[int, str]]
    """
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "logging":
                    hits.append((node.lineno, "import logging"))
        elif isinstance(node, ast.ImportFrom):
            if node.module == "logging":
                hits.append((node.lineno, "from logging import ..."))
    return hits


def _find_logging_getlogger_calls(tree: ast.Module) -> list[tuple[int, str]]:
    """return ``(lineno, description)`` for every ``logging.getLogger`` call.

    catches the case where a module obtained ``logging`` indirectly (e.g.,
    via ``import logging as L``) and called ``L.getLogger(...)``. for the
    direct ``logging.getLogger(...)`` form the import-level walker already
    fires, but pairing the two checks gives a second AST-level signal that
    catches regressions even if a future refactor renames the binding.

    :param tree: parsed AST module
    :ptype tree: ast.Module
    :return: list of ``(lineno, description)`` tuples
    :rtype: list[tuple[int, str]]
    """
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "getLogger"
            and isinstance(func.value, ast.Name)
            and func.value.id == "logging"
        ):
            hits.append((node.lineno, "logging.getLogger(...)"))
    return hits


_SRC_FILES = _collect_src_files()
_SRC_IDS = [_rel_posix(p) for p in _SRC_FILES]


class TestNoStdlibLogging:
    """no production module may import stdlib ``logging`` directly."""

    @pytest.mark.parametrize("src_file", _SRC_FILES, ids=_SRC_IDS)
    def test_no_stdlib_logging_import(self, src_file: Path) -> None:
        """verify the file does not import stdlib ``logging`` unless exempt.

        :param src_file: parametrized path under the langgraph package root
        :ptype src_file: Path
        :return: nothing
        :rtype: None
        """
        rel = _rel_posix(src_file)
        if rel in _EXEMPT:
            return
        source = src_file.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(src_file))
        hits = _find_stdlib_logging_import_lines(tree)
        hits.extend(_find_logging_getlogger_calls(tree))
        if not hits:
            return

        lines = source.splitlines()
        uncovered: list[str] = []
        for lineno, desc in sorted(set(hits)):
            line_text = lines[lineno - 1] if 0 < lineno <= len(lines) else ""
            if _LINE_MARKER in line_text:
                continue
            uncovered.append(f"  line {lineno}: {desc}")

        if uncovered:
            detail = "\n".join(uncovered)
            pytest.fail(
                f"{rel} uses stdlib logging:\n{detail}\n"
                f"use threetears.observe.get_logger instead. if configuring a "
                f"third-party logger by name, add '{_LINE_MARKER}' on the line "
                f"with a justification, or add the module to the exemption "
                f"list in {Path(__file__).name}.",
            )

    def test_exemption_list_is_current(self) -> None:
        """exemption entries must still correspond to real files.

        :return: nothing
        :rtype: None
        """
        missing = [rel for rel in _EXEMPT if not (_SRC_ROOT / rel).is_file()]
        if missing:
            pytest.fail(
                "stale exemption entries (file no longer exists): "
                + ", ".join(missing),
            )
