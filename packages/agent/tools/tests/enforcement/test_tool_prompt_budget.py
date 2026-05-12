"""Enforcement: builtin + workspace tool DESCRIPTION + INPUT_SCHEMA stay under budget.

Catches the bug class round 6 fixed: LLM-visible tool-schema bloat
on the threetears built-in toolset (calculator, current_date,
dictionary, timezone_converter, unit_converter, image_prep,
web_fetch, web_search) and the workspace tools (fs_*, doc_*,
workspace_*).

The test is AST-based, no instantiation: walks every ``.py`` file
under the builtin + workspace tool directories and reads the
literal ``_INPUT_SCHEMA`` dict (either class-level on
``TearsTool`` subclasses or module-level on the workspace tools)
plus the ``description=`` kwarg passed to ``MCPToolDefinition()``
inside ``mcp_schema()``.

Two checks per tool:

1. **Total budget**: ``len(description) + json.dumps(input_schema)``
   stays below :data:`_PER_TOOL_BUDGET_CHARS`.

2. **Anti-bloat patterns**: descriptions and field-level descriptions
   are scanned for the well-known LLM-bloat shapes documented in
   the agents-repo sibling test.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest

_BUILTIN_TOOLS_DIR = (
    Path(__file__).resolve().parent.parent.parent / "src" / "threetears" / "agent" / "tools" / "builtin"
)

_WORKSPACE_TOOLS_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "workspace"
    / "src"
    / "threetears"
    / "agent"
    / "workspace"
    / "tools"
)

_PER_TOOL_BUDGET_CHARS = 900

_PROMPT_BUDGET_EXEMPTIONS: dict[str, tuple[int, str]] = {
    # tool_name: (new_budget, reason)
}

_BLOAT_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(
            r"\b(?:manage|query|list|update|describe)\b[^.]*?:\s*"
            r"(?:list|create|get|update|delete)\s*,",
            re.IGNORECASE,
        ),
        "description leads with an action-list that duplicates the enum",
    ),
    (
        re.compile(r"\buse\s+this\s+(?:when|before|to)\b", re.IGNORECASE),
        "description contains 'use this when/before/to' coaching prose",
    ),
    (
        re.compile(r"\(required\s+for\s+\w+", re.IGNORECASE),
        "field description has '(required for X)' couplings; use the top-level required[] array instead",
    ),
    (
        re.compile(r"\(optional\s+for\s+\w+", re.IGNORECASE),
        "field description has '(optional for X)' couplings; the required[] array already documents what is mandatory",
    ),
]


def _iter_tool_modules(directory: Path) -> list[Path]:
    """return every ``.py`` file under ``directory`` excluding __init__ and helpers."""
    if not directory.exists():
        return []
    skip = {"__init__", "helpers", "image_prep"}  # image_prep has no schema-only test target
    return sorted(p for p in directory.glob("*.py") if p.stem not in {"__init__", "helpers"})


def _extract_class_level_input_schema(class_node: ast.ClassDef) -> dict | None:
    """find a ``_INPUT_SCHEMA`` assignment on the class body and literal-eval it."""
    for stmt in class_node.body:
        target_name: str | None = None
        value_node: ast.AST | None = None
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value is not None:
            target_name = stmt.target.id
            value_node = stmt.value
        elif isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            target_name = stmt.targets[0].id
            value_node = stmt.value
        if target_name == "_INPUT_SCHEMA" and value_node is not None:
            try:
                return ast.literal_eval(value_node)  # type: ignore[arg-type]
            except ValueError, SyntaxError:
                return None
    return None


def _extract_module_level_input_schema(tree: ast.Module) -> dict | None:
    """find module-level ``_INPUT_SCHEMA`` (workspace-tool pattern) and literal-eval it."""
    for stmt in tree.body:
        target_name: str | None = None
        value_node: ast.AST | None = None
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value is not None:
            target_name = stmt.target.id
            value_node = stmt.value
        elif isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            target_name = stmt.targets[0].id
            value_node = stmt.value
        if target_name == "_INPUT_SCHEMA" and value_node is not None:
            try:
                return ast.literal_eval(value_node)  # type: ignore[arg-type]
            except ValueError, SyntaxError:
                return None
    return None


def _extract_mcp_schema_description(tree: ast.AST) -> str | None:
    """walk the tree for ``MCPToolDefinition(... description=<literal>, ...)`` calls.

    returns the literal description string from the first match found
    inside an ``mcp_schema`` method body. multiple MCPToolDefinition
    calls in one file (unusual) take the first.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        is_target = (isinstance(func, ast.Name) and func.id == "MCPToolDefinition") or (
            isinstance(func, ast.Attribute) and func.attr == "MCPToolDefinition"
        )
        if not is_target:
            continue
        for kw in node.keywords:
            if kw.arg != "description":
                continue
            try:
                value = ast.literal_eval(kw.value)
            except ValueError, SyntaxError:
                value = None
            if isinstance(value, str):
                return value
    return None


def _collect_tools(directory: Path) -> list[tuple[str, str, dict]]:
    """parse every tool module under ``directory`` and return (tool_id, description, input_schema).

    ``tool_id`` is the file stem (e.g. ``current_date``, ``fs_read``).
    ``input_schema`` is the literal dict; tools that don't define one
    are skipped.
    """
    results: list[tuple[str, str, dict]] = []
    for path in _iter_tool_modules(directory):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        input_schema: dict | None = _extract_module_level_input_schema(tree)
        if input_schema is None:
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    candidate = _extract_class_level_input_schema(node)
                    if candidate is not None:
                        input_schema = candidate
                        break
        if input_schema is None:
            continue
        description = _extract_mcp_schema_description(tree) or ""
        results.append((path.stem, description, input_schema))
    return results


class TestBuiltinToolPromptBudget:
    """Builtin tools (calculator/current_date/etc.) under budget + no bloat."""

    def test_builtin_tools_dir_present(self) -> None:
        """guard the test itself: if the directory moves, fail loudly."""
        assert _BUILTIN_TOOLS_DIR.exists(), f"builtin tools directory not found at {_BUILTIN_TOOLS_DIR}"

    def test_per_tool_under_budget(self) -> None:
        """each builtin tool fits inside the per-tool char budget."""
        violations: list[str] = []
        for tool_id, desc, schema in _collect_tools(_BUILTIN_TOOLS_DIR):
            input_json = json.dumps(schema, separators=(",", ":"))
            total = len(desc) + len(input_json)
            budget = _PROMPT_BUDGET_EXEMPTIONS.get(
                tool_id,
                (_PER_TOOL_BUDGET_CHARS, ""),
            )[0]
            if total > budget:
                violations.append(
                    f"  {tool_id}: {total} chars > {budget} budget (desc={len(desc)} input_schema={len(input_json)})",
                )
        if violations:
            pytest.fail(
                "Builtin tool prompt budget exceeded -- trim the schema, "
                "do not bump the budget:\n" + "\n".join(violations),
            )

    def test_no_bloat_patterns(self) -> None:
        """builtin tool descriptions + field descriptions match no bloat regex."""
        violations: list[str] = []
        for tool_id, desc, schema in _collect_tools(_BUILTIN_TOOLS_DIR):
            for pattern, label in _BLOAT_PATTERNS:
                if pattern.search(desc):
                    violations.append(
                        f"  {tool_id}.description: {label!r} -- matched: {pattern.search(desc).group(0)!r}",
                    )
            for field_name, field_def in (
                (schema or {})
                .get(
                    "properties",
                    {},
                )
                .items()
            ):
                if not isinstance(field_def, dict):
                    continue
                field_desc = field_def.get("description", "")
                if not isinstance(field_desc, str):
                    continue
                for pattern, label in _BLOAT_PATTERNS:
                    if pattern.search(field_desc):
                        violations.append(
                            f"  {tool_id}.{field_name}: {label!r} -- matched: {pattern.search(field_desc).group(0)!r}",
                        )
        if violations:
            pytest.fail("Bloat patterns:\n" + "\n".join(violations))


class TestWorkspaceToolPromptBudget:
    """Workspace tools (fs_*, doc_*, workspace_*) under budget + no bloat."""

    def test_workspace_tools_dir_present(self) -> None:
        """guard the test itself: if the directory moves, fail loudly."""
        assert _WORKSPACE_TOOLS_DIR.exists(), f"workspace tools directory not found at {_WORKSPACE_TOOLS_DIR}"

    def test_per_tool_under_budget(self) -> None:
        """each workspace tool fits inside the per-tool char budget."""
        violations: list[str] = []
        for tool_id, desc, schema in _collect_tools(_WORKSPACE_TOOLS_DIR):
            input_json = json.dumps(schema, separators=(",", ":"))
            total = len(desc) + len(input_json)
            budget = _PROMPT_BUDGET_EXEMPTIONS.get(
                tool_id,
                (_PER_TOOL_BUDGET_CHARS, ""),
            )[0]
            if total > budget:
                violations.append(
                    f"  {tool_id}: {total} chars > {budget} budget (desc={len(desc)} input_schema={len(input_json)})",
                )
        if violations:
            pytest.fail(
                "Workspace tool prompt budget exceeded -- trim the schema, "
                "do not bump the budget:\n" + "\n".join(violations),
            )

    def test_no_bloat_patterns(self) -> None:
        """workspace tool descriptions + field descriptions match no bloat regex."""
        violations: list[str] = []
        for tool_id, desc, schema in _collect_tools(_WORKSPACE_TOOLS_DIR):
            for pattern, label in _BLOAT_PATTERNS:
                if pattern.search(desc):
                    violations.append(
                        f"  {tool_id}.description: {label!r} -- matched: {pattern.search(desc).group(0)!r}",
                    )
            for field_name, field_def in (
                (schema or {})
                .get(
                    "properties",
                    {},
                )
                .items()
            ):
                if not isinstance(field_def, dict):
                    continue
                field_desc = field_def.get("description", "")
                if not isinstance(field_desc, str):
                    continue
                for pattern, label in _BLOAT_PATTERNS:
                    if pattern.search(field_desc):
                        violations.append(
                            f"  {tool_id}.{field_name}: {label!r} -- matched: {pattern.search(field_desc).group(0)!r}",
                        )
        if violations:
            pytest.fail("Bloat patterns:\n" + "\n".join(violations))
