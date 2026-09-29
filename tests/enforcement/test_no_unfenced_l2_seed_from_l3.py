"""
enforcement: a function that reads rows from L3 never puts them into L2 unconditionally.

A row read from L3 is only as new as the moment its query ran. A writer whose ``save_entity``
commits and puts its row into L2 after that moment holds a NEWER value, and an unconditional put
from the read lands the older row over it -- every reader on every replica is then served the older
value until the next write or the entry's lifetime. Even a create-if-absent is not enough: the
save's broadcast makes each peer in its scope delete the key it just wrote, so the read can find it
empty and recreate the older row.

``BaseCollection._seed_l2`` is the one sanctioned way a read moves an L3 row into L2: it writes at
the key's latest revision read BEFORE the query, so it lands only if nothing happened to the key
since. A list loader has no per-key revision from before its query, so it does not write L2 at
all; the next single-row read seeds it correctly.

The rule, checked here: a function that reads L3 -- ``fetch_from_store``, or ``fetch`` /
``fetchrow`` on ``l3_pool`` / ``required_l3_pool`` -- does not call ``_save_to_l2``. A writer is
untouched: ``save_entity`` reads nothing from L3 before its put, and a statement that WRITES and
returns its own row (``INSERT ... RETURNING``, ``UPDATE ... RETURNING``) is the write itself, not a
read that a write can overtake.

Static AST parsing only (no imports, no execution), consistent with the rest of
``tests/enforcement``.
"""

from __future__ import annotations

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PACKAGE_GLOBS = ("packages/*/src", "packages/agent/*/src")
_L3_HANDLES = frozenset({"l3_pool", "required_l3_pool"})
_L3_READS = frozenset({"fetch", "fetchrow"})


def _source_files() -> list[Path]:
    """return every shipped python module across the workspace packages.

    :return: module paths, sorted for stable reporting
    :rtype: list[Path]
    """
    found: list[Path] = []
    for glob in _PACKAGE_GLOBS:
        for src_dir in sorted(_REPO_ROOT.glob(glob)):
            found.extend(sorted(src_dir.rglob("*.py")))
    return found


def _is_self_attr(node: ast.expr, names: frozenset[str]) -> bool:
    """whether ``node`` is ``self.<one of names>``.

    :param node: expression under inspection
    :ptype node: ast.expr
    :param names: attribute names that match
    :ptype names: frozenset[str]
    :return: whether it matches
    :rtype: bool
    """
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
        and node.attr in names
    )


_WRITE_VERBS = ("INSERT", "UPDATE", "DELETE", "MERGE")


def _is_write_statement(call: ast.Call) -> bool:
    """whether a ``fetch`` / ``fetchrow`` call runs a statement that writes and returns its own row.

    :param call: the call node
    :ptype call: ast.Call
    :return: whether its SQL text statically begins with a write verb
    :rtype: bool
    """
    if not call.args:
        return False
    sql = call.args[0]
    head = ""
    if isinstance(sql, ast.Constant) and isinstance(sql.value, str):
        head = sql.value
    elif isinstance(sql, ast.JoinedStr) and sql.values and isinstance(sql.values[0], ast.Constant):
        head = str(sql.values[0].value)
    return head.lstrip().upper().startswith(_WRITE_VERBS)


def _calls(function: ast.AST) -> tuple[bool, bool]:
    """what one function body does: reads L3, and puts into L2 unconditionally.

    :param function: a function definition node
    :ptype function: ast.AST
    :return: ``(reads L3, calls _save_to_l2)``
    :rtype: tuple[bool, bool]
    """
    reads_l3 = False
    puts_l2 = False
    for node in ast.walk(function):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        func = node.func
        if _is_self_attr(func, frozenset({"fetch_from_store"})):
            reads_l3 = True
        if func.attr in _L3_READS and _is_self_attr(func.value, _L3_HANDLES) and not _is_write_statement(node):
            reads_l3 = True
        if _is_self_attr(func, frozenset({"_save_to_l2"})):
            puts_l2 = True
    return reads_l3, puts_l2


def _scan() -> tuple[list[str], int, int]:
    """walk every function in the workspace sources.

    :return: violations as ``path:line:function``, and how many functions read L3 and how many
        put into L2 -- so a walk that found nothing cannot pass for a clean one
    :rtype: tuple[list[str], int, int]
    """
    violations: list[str] = []
    readers = 0
    putters = 0
    for path in _source_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            reads_l3, puts_l2 = _calls(node)
            readers += reads_l3
            putters += puts_l2
            if reads_l3 and puts_l2:
                violations.append(f"{path.relative_to(_REPO_ROOT)}:{node.lineno}:{node.name}")
    return violations, readers, putters


def test_no_function_puts_a_row_it_read_from_l3_into_l2_unfenced() -> None:
    violations, readers, putters = _scan()
    # both halves must be seen, or an empty walk would pass for a clean one.
    assert readers > 0, "the walk found no function reading L3; the scan is not looking where it should"
    assert putters > 0, "the walk found no _save_to_l2 call; the scan is not looking where it should"
    assert not violations, (
        "these functions read rows from L3 and put them into L2 unconditionally, which can land an "
        "older row over a newer write for every reader. Seed through BaseCollection._seed_l2 with the "
        "revision read before the query, or leave L2 to the next single-row read:\n  " + "\n  ".join(violations)
    )
