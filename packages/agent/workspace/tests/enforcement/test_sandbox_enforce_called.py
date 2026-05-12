"""enforcement: every per-file write-class tool authorizes before mutating file content.

scope is the set of write-class tools that mutate file content through
:func:`_write_file_atomic`: ``fs_write``, ``fs_edit``, ``doc_set``,
``doc_merge``, and ``workspace_rollback``. lifecycle tools
(``workspace_create``, ``workspace_reset``, ``workspace_delete``) are
excluded deliberately -- lifecycle ops are NOT gated by per-file
access rules -- and ``workspace_delete`` soft-deletes rows without
writing file content at all.

the historical rule locked in ``sandbox.enforce("write", path)``
before any mutation. namespace-task-01 phase 7 retired that
enforcement surface: ``sandbox.enforce`` and ``sandbox.check_relative_key``
are gone, replaced by ``sandbox.validate_syntax(path)`` (syntactic-only
path check) followed by ``authorize_workspace_file(...)`` (unified
rbac evaluator with path-glob-bearing custom action types). this
enforcement test now pins the new ordering: each write-class tool's
``execute`` must call ``authorize_workspace_file(...)`` strictly
before any ``_write_file_atomic(...)``. validate_syntax is a
preceding syntactic guard that is paired with authorize_workspace_file
in every site; the authorize call is the gating one we enforce.

ordering is enforced per function via ``lineno``; the authorize
call must appear before every ``_write_file_atomic`` in the same
``execute``. for ``workspace_rollback`` the pre-sweep authorize
loop runs before the second-phase write loop, matching the
same-function ordering rule.
"""

from __future__ import annotations

import ast
from pathlib import Path


_SRC_ROOT = Path(__file__).resolve().parent.parent.parent / "src" / "threetears" / "agent" / "workspace"
_TOOLS_ROOT = _SRC_ROOT / "tools"


# per-file write-class tools. each of these tools runs
# ``authorize_workspace_file("write", ...)`` on exactly one path before
# calling ``_write_file_atomic`` on that same path.
_WRITE_CLASS_TOOL_MODULES: tuple[str, ...] = (
    "fs_write",
    "fs_edit",
    "doc_set",
    "doc_merge",
    "workspace_rollback",
)


def _module_path(module_name: str) -> Path:
    """
    resolve a write-class tool module name to its source file.

    :param module_name: short tool module name (no ``.py`` suffix)
    :ptype module_name: str
    :return: absolute path to the module source
    :rtype: Path
    """
    return _TOOLS_ROOT / f"{module_name}.py"


def _find_execute_function(
    tree: ast.Module,
) -> ast.AsyncFunctionDef | ast.FunctionDef | None:
    """
    locate the tool class's ``execute`` method (async or sync).

    we look inside every class defined at module level for a method
    named ``execute``; the workspace tool surface is strictly one tool
    class per module so the first match is authoritative.

    :param tree: parsed module AST
    :ptype tree: ast.Module
    :return: ``execute`` function node, or None if absent
    :rtype: ast.AsyncFunctionDef | ast.FunctionDef | None
    """
    result: ast.AsyncFunctionDef | ast.FunctionDef | None = None
    for cls in (n for n in tree.body if isinstance(n, ast.ClassDef)):
        for item in cls.body:
            if isinstance(item, (ast.AsyncFunctionDef, ast.FunctionDef)) and item.name == "execute":
                result = item
                break
        if result is not None:
            break
    return result


def _attribute_chain(node: ast.AST) -> list[str]:
    """
    flatten an attribute chain into a list of segment names.

    ``helpers.authorize_workspace_file`` -> ``["helpers", "authorize_workspace_file"]``;
    any other expression returns an empty list.

    :param node: AST node to flatten
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


def _is_authorize_call(call: ast.Call) -> bool:
    """
    true iff call is ``authorize_workspace_file(...)`` (bare or module-qualified).

    :param call: AST Call node
    :ptype call: ast.Call
    :return: True for an authorize_workspace_file call
    :rtype: bool
    """
    chain = _attribute_chain(call.func)
    return chain[-1:] == ["authorize_workspace_file"]


def _is_write_file_atomic_call(call: ast.Call) -> bool:
    """
    true iff call is ``_write_file_atomic(...)`` (bare or module-qualified).

    :param call: AST Call node
    :ptype call: ast.Call
    :return: True for a _write_file_atomic call
    :rtype: bool
    """
    chain = _attribute_chain(call.func)
    return chain[-1:] == ["_write_file_atomic"]


class TestAuthorizeCalledBeforeWrite:
    """per-file write-class tools authorize_workspace_file before any mutation."""

    def test_every_write_class_tool_authorizes_before_writing(self) -> None:
        """
        AST-walk each write-class tool's ``execute``; the
        ``authorize_workspace_file`` call's lineno must strictly
        precede every ``_write_file_atomic`` call.

        :return: None
        :rtype: None
        """
        violations: list[str] = []
        for module_name in _WRITE_CLASS_TOOL_MODULES:
            path = _module_path(module_name)
            assert path.is_file(), f"missing tool module: {path}"
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            execute_fn = _find_execute_function(tree)
            if execute_fn is None:
                violations.append(f"{module_name}: no execute() method found on tool class")
                continue

            authorize_lines: list[int] = []
            write_lines: list[int] = []
            for node in ast.walk(execute_fn):
                if not isinstance(node, ast.Call):
                    continue
                if _is_authorize_call(node):
                    authorize_lines.append(node.lineno)
                if _is_write_file_atomic_call(node):
                    write_lines.append(node.lineno)

            if not authorize_lines:
                violations.append(
                    f"{module_name}:{execute_fn.lineno}: execute() contains no authorize_workspace_file call",
                )
                continue
            # rule: every write must come after at least one authorize call.
            first_authorize = min(authorize_lines)
            offending = [ln for ln in write_lines if ln < first_authorize]
            if offending:
                violations.append(
                    f"{module_name}: _write_file_atomic at line(s) "
                    f"{offending} precedes first authorize_workspace_file "
                    f"at line {first_authorize}",
                )
        assert not violations, (
            f"{len(violations)} authorize-before-write ordering violation(s):\n"
            + "\n".join(violations)
            + "\n\nwrite-class tools must authorize_workspace_file(..., 'write', ...) "
            "before any _write_file_atomic(...)."
        )
