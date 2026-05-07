"""enforcement: every write-class tool dispatches registered validators.

the single source of validator dispatch for per-file writes is
:func:`threetears.agent.workspace.tools.helpers._write_file_atomic`,
which calls :func:`dispatch_validators` inside its transaction before
any INSERT/UPSERT. a tool that writes file content without routing
through this helper would skip validator dispatch entirely.

lifecycle tools that seed files through a bulk-insert code path (at
this time, ``workspace_create`` and ``workspace_reset``) do their own
pre-insert validator sweep via a direct call to
:func:`dispatch_validators`. this test accepts either marker:

- a call to ``_write_file_atomic(...)`` somewhere in the module, OR
- a call to ``dispatch_validators(...)`` somewhere in the module.

workspace_delete is excluded because it writes no file content: it
soft-deletes the workspace row; per-file validators have nothing to
validate.

the scan walks the whole module (not just ``execute``) so validator
dispatches in private helpers -- e.g. ``_insert_all`` in
``workspace_create`` -- are accepted.
"""

from __future__ import annotations

import ast
from pathlib import Path


_SRC_ROOT = Path(__file__).resolve().parent.parent.parent / "src" / "threetears" / "agent" / "workspace"
_TOOLS_ROOT = _SRC_ROOT / "tools"


# every tool module that persists file content (either per-file via
# _write_file_atomic or bulk via an in-class helper + dispatch_validators).
_WRITE_FILE_TOOL_MODULES: tuple[str, ...] = (
    "fs_write",
    "fs_edit",
    "doc_set",
    "doc_merge",
    "workspace_rollback",
    "workspace_create",
    "workspace_reset",
)


def _module_path(module_name: str) -> Path:
    """
    resolve a tool module name to its source path.

    :param module_name: short tool module name (no ``.py`` suffix)
    :ptype module_name: str
    :return: absolute path to the module source
    :rtype: Path
    """
    return _TOOLS_ROOT / f"{module_name}.py"


def _attribute_chain(node: ast.AST) -> list[str]:
    """
    flatten an attribute chain into a list of segment names.

    mirrors the helper in ``test_sandbox_enforce_called.py``; kept
    private per test file so each test stays self-contained.

    :param node: AST node
    :ptype node: ast.AST
    :return: ordered list of segment names
    :rtype: list[str]
    """
    parts: list[str] = []
    current: ast.AST = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    parts.reverse()
    return parts


def _is_write_file_atomic_call(call: ast.Call) -> bool:
    """true for ``_write_file_atomic(...)`` or qualified equivalent."""
    chain = _attribute_chain(call.func)
    return chain[-1:] == ["_write_file_atomic"]


def _is_dispatch_validators_call(call: ast.Call) -> bool:
    """true for ``dispatch_validators(...)`` or qualified equivalent."""
    chain = _attribute_chain(call.func)
    return chain[-1:] == ["dispatch_validators"]


class TestValidatorsDispatched:
    """every write-class tool module dispatches validators at write time."""

    def test_every_write_tool_routes_through_validator_dispatch(self) -> None:
        """
        each module contains at least one validator dispatch site.

        scan the whole module AST: either ``_write_file_atomic`` (the
        helper internally calls ``dispatch_validators``) or a direct
        ``dispatch_validators`` call counts.

        :return: None
        :rtype: None
        """
        violations: list[str] = []
        for module_name in _WRITE_FILE_TOOL_MODULES:
            path = _module_path(module_name)
            assert path.is_file(), f"missing tool module: {path}"
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            has_atomic_write = False
            has_direct_dispatch = False
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                if _is_write_file_atomic_call(node):
                    has_atomic_write = True
                if _is_dispatch_validators_call(node):
                    has_direct_dispatch = True
            if not (has_atomic_write or has_direct_dispatch):
                violations.append(f"{module_name}: no _write_file_atomic or dispatch_validators call found in module")
        assert not violations, (
            f"{len(violations)} validator-dispatch violation(s):\n"
            + "\n".join(violations)
            + "\n\nwrite-class tools must route through _write_file_atomic "
            "(which runs dispatch_validators) or call dispatch_validators "
            "directly before persisting content."
        )
