"""enforcement: every per-file write-class tool calls ``sandbox.enforce`` first.

scope is the set of write-class tools that mutate file content through
:func:`_write_file_atomic`: ``fs_write``, ``fs_edit``, ``doc_set``,
``doc_merge``, and ``workspace_rollback``. lifecycle tools
(``workspace_create``, ``workspace_reset``, ``workspace_delete``) are
excluded deliberately: shard 10 is explicit that lifecycle ops are NOT
gated by ``allow.write`` globs -- their creation + reset semantics live
outside the per-file write rails -- and ``workspace_delete`` soft-deletes
rows without writing file content at all. the sandbox-enforce rule binds
the per-file write surface, which is what this test locks in.

the rule: within the tool class's ``execute`` (or ``execute`` plus the
helper it delegates to for each file, in the rollback case), the first
call whose attribute chain matches ``self._sandbox.enforce`` /
``sandbox.enforce`` must appear strictly before any mutating call. a
mutating call is any ``await _write_file_atomic(...)``. ordering is
measured by ``lineno`` because AST walk order is document-order.

ordering is enforced per function: rollback's ``execute`` enforces on
every file in a pre-sweep then invokes ``_write_file_atomic`` in a
second phase (see shard docstring); that pattern satisfies the
enforce-before-write rule for the whole rollback set.
"""

from __future__ import annotations

import ast
from pathlib import Path


_SRC_ROOT = Path(__file__).resolve().parent.parent.parent / "src" / "threetears" / "agent" / "workspace"
_TOOLS_ROOT = _SRC_ROOT / "tools"


# per-file write-class tools. each of these tools runs
# ``sandbox.enforce("write", relative_path)`` on exactly one path before
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
    :return: the ``execute`` function node, or None if absent
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

    ``self._sandbox.enforce`` -> ``["self", "_sandbox", "enforce"]``;
    ``sandbox.enforce`` -> ``["sandbox", "enforce"]``; any other
    expression returns an empty list.

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


def _is_sandbox_enforce_call(call: ast.Call) -> bool:
    """
    true iff call is ``sandbox.enforce(...)`` or ``self._sandbox.enforce(...)``.

    :param call: AST Call node
    :ptype call: ast.Call
    :return: True for a sandbox enforce call
    :rtype: bool
    """
    chain = _attribute_chain(call.func)
    if chain[-1:] != ["enforce"]:
        return False
    # chain forms we accept:
    #   ["sandbox", "enforce"]              (local var)
    #   [<owner>, "_sandbox", "enforce"]    (attribute access, typically self)
    if len(chain) == 2 and chain[0] == "sandbox":
        return True
    if len(chain) >= 3 and chain[-2] == "_sandbox":
        return True
    return False


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


class TestSandboxEnforceCalledBeforeWrite:
    """per-file write-class tools enforce sandbox before any mutation."""

    def test_every_write_class_tool_enforces_before_writing(self) -> None:
        """
        AST-walk each write-class tool's ``execute``; sandbox.enforce
        lineno must strictly precede every ``_write_file_atomic`` call.

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

            enforce_lines: list[int] = []
            write_lines: list[int] = []
            for node in ast.walk(execute_fn):
                if not isinstance(node, ast.Call):
                    continue
                if _is_sandbox_enforce_call(node):
                    enforce_lines.append(node.lineno)
                if _is_write_file_atomic_call(node):
                    write_lines.append(node.lineno)

            if not enforce_lines:
                violations.append(f"{module_name}:{execute_fn.lineno}: execute() contains no sandbox.enforce call")
                continue
            # rule: every write must come after at least one enforce.
            first_enforce = min(enforce_lines)
            offending = [ln for ln in write_lines if ln < first_enforce]
            if offending:
                violations.append(
                    f"{module_name}: _write_file_atomic at line(s) "
                    f"{offending} precedes first sandbox.enforce at line "
                    f"{first_enforce}"
                )
        assert not violations, (
            f"{len(violations)} sandbox-enforce ordering violation(s):\n"
            + "\n".join(violations)
            + "\n\nwrite-class tools must sandbox.enforce('write', path) "
            "before any _write_file_atomic(...)."
        )
