"""
enforcement: an entity's UUID coercer says so when a field read empty.

Entity field reads come from L1 only. When the row has left L1 -- an
invalidation, a salience bump, an eviction -- a held entity reads every field
as ``None``, and a coercer written as ``UUID(str(value))`` turns that into
``ValueError: badly formed hexadecimal UUID string``. That message sends the
reader after a malformed id. In metallm 0.56.0 it hid a memory push lost to an
evicted row (the extractor handed ``on_memory_created`` a handle held across a
two-minute model call); the wake package had already been bitten the same way.

So every ``_as_uuid`` that promises a ``UUID`` (not ``UUID | None``) checks
``value is None`` first and raises a ``ValueError`` of its own. Static AST
parsing only (no imports, no execution), consistent with the rest of
``tests/enforcement``.
"""

from __future__ import annotations

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_GLOBS = ("packages/*/src", "packages/agent/*/src")
_COERCER = "_as_uuid"


def _guards_none(func: ast.FunctionDef) -> bool:
    """whether ``func`` raises ``ValueError`` under an ``if <param> is None`` test.

    :param func: the coercer's definition
    :ptype func: ast.FunctionDef
    :return: ``True`` when such a guard is in the function body
    :rtype: bool
    """
    param = func.args.args[0].arg if func.args.args else None
    for node in ast.walk(func):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name)
            and test.left.id == param
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Is)
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value is None
        ):
            continue
        for stmt in node.body:
            if isinstance(stmt, ast.Raise) and isinstance(stmt.exc, ast.Call):
                if isinstance(stmt.exc.func, ast.Name) and stmt.exc.func.id == "ValueError":
                    return True
    return False


def _coercers(tree: ast.AST) -> list[ast.FunctionDef]:
    """every module-level ``_as_uuid`` annotated to return a bare ``UUID``.

    :param tree: a parsed module
    :ptype tree: ast.AST
    :return: the matching definitions
    :rtype: list[ast.FunctionDef]
    """
    found: list[ast.FunctionDef] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == _COERCER and node.returns is not None:
            if ast.unparse(node.returns) == "UUID":
                found.append(node)
    return found


def _source_files() -> list[Path]:
    files: list[Path] = []
    for pattern in _SOURCE_GLOBS:
        for root in _REPO_ROOT.glob(pattern):
            files.extend(sorted(root.rglob("*.py")))
    return files


def test_every_uuid_coercer_names_an_empty_field() -> None:
    offenders: list[str] = []
    seen = 0
    for path in _source_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for func in _coercers(tree):
            seen += 1
            if not _guards_none(func):
                offenders.append(f"{path.relative_to(_REPO_ROOT)}:{func.lineno}")
    # nine entity modules carry one today; finding none means the walk broke.
    assert seen >= 9, f"found only {seen} {_COERCER} definitions; the source walk is not reaching the packages"
    assert not offenders, (
        f"{_COERCER} returning UUID must raise its own ValueError when the value is None, "
        f"or an evicted row reads as a malformed UUID: {offenders}"
    )


def test_the_check_flags_a_bare_coercer() -> None:
    bare = ast.parse("def _as_uuid(value: object) -> UUID:\n    return UUID(str(value))\n")
    guarded = ast.parse(
        "def _as_uuid(value: object) -> UUID:\n"
        "    if value is None:\n"
        "        raise ValueError('read empty')\n"
        "    return UUID(str(value))\n"
    )
    nullable = ast.parse("def _as_uuid(value: object) -> UUID | None:\n    return None\n")
    assert [_guards_none(f) for f in _coercers(bare)] == [False]
    assert [_guards_none(f) for f in _coercers(guarded)] == [True]
    assert _coercers(nullable) == []
