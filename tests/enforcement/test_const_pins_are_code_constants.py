"""enforcement: a :class:`~threetears.core.keyed.Const` is only ever made from text the code itself fixes.

A ``Const`` pin is written into a keyed statement as its SQL literal, not bound (Pace's ruling 2026-11-16: a bound
discriminator kept PostgreSQL from caching the visible-customer walk's generic plan, so every call was re-planned;
the why and numbers at ``threetears.interfaces.keyed.filters.Const``). The ruling allows that for fixed text from our
own code only, never anything a caller or an end user supplies. The type refuses all but a plain identifier-like
word; this gate keeps a run-time string from reaching it at all: every ``Const(...)`` in the workspace's source is
called with one string literal, or with a name the module binds once, at its top level, to a string literal; and
``Const`` is never handed on as a value (``map(Const, names)``, ``make = Const``), which would hide such a call.

Static parsing only, as the rest of ``tests/enforcement``.
"""

from __future__ import annotations

import ast
from pathlib import Path

from threetears.enforcement.common import parse_python_path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_GLOBS = ("packages/*/src", "packages/agent/*/src")
_NAME = "Const"


def _module_texts(tree: ast.Module) -> set[str]:
    """the names the module binds once, at its top level, to a string literal (``_USER: Final = "user"``).

    :param tree: the module
    :ptype tree: ast.Module
    :return: the names
    :rtype: set[str]
    """
    bound: dict[str, int] = {}
    texts: set[str] = set()
    for node in ast.walk(tree):
        for target in _targets(node):
            bound[target] = bound.get(target, 0) + 1
    for node in tree.body:
        value = node.value if isinstance(node, ast.Assign | ast.AnnAssign) else None
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            texts.update(_targets(node))
    return {name for name in texts if bound[name] == 1}


def _targets(node: ast.AST) -> list[str]:
    """the plain names ``node`` binds by assignment (``=``, ``: T =``, ``+=``, ``:=``, a ``for`` or ``with`` target).

    :param node: any node
    :ptype node: ast.AST
    :return: the names
    :rtype: list[str]
    """
    if isinstance(node, ast.Assign):
        targets: list[ast.expr] = list(node.targets)
    elif isinstance(node, ast.AnnAssign | ast.AugAssign | ast.NamedExpr | ast.For | ast.AsyncFor):
        targets = [node.target]
    elif isinstance(node, ast.With | ast.AsyncWith):
        targets = [item.optional_vars for item in node.items if item.optional_vars is not None]
    else:
        return []
    return [name.id for target in targets for name in ast.walk(target) if isinstance(name, ast.Name)]


def _is_const(node: ast.expr) -> bool:
    return (isinstance(node, ast.Name) and node.id == _NAME) or (isinstance(node, ast.Attribute) and node.attr == _NAME)


def violations(path: Path, *, relative_to: Path) -> list[str]:
    """every ``Const`` use in the file at ``path`` that is not a call on a code constant.

    :param path: a python file
    :ptype path: Path
    :param relative_to: the root its name is given from
    :ptype relative_to: Path
    :return: ``<file>:<line> <why>`` each
    :rtype: list[str]
    """
    tree = parse_python_path(path)
    texts = _module_texts(tree)
    where = path.relative_to(relative_to).as_posix()
    found: list[str] = []
    called: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom | ast.Import):
            found += [
                f"{where}:{node.lineno} imports Const as {alias.asname}"
                for alias in node.names
                if alias.name.rsplit(".", 1)[-1] == _NAME and alias.asname not in (None, _NAME)
            ]
        if isinstance(node, ast.Call) and _is_const(node.func):
            called.add(id(node.func))
            argument = node.args[0] if len(node.args) == 1 else None
            constant = isinstance(argument, ast.Constant) and isinstance(argument.value, str)
            named = isinstance(argument, ast.Name) and argument.id in texts
            if node.keywords or len(node.args) != 1 or not (constant or named):
                found.append(f"{where}:{node.lineno} Const({ast.unparse(node)[len(_NAME) + 1 : -1]}) is no constant")
        # an isinstance / issubclass check reads the type; any other load of it hands the constructor on
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in ("isinstance", "issubclass")
        ):
            called.update(
                id(argument) for argument in ast.walk(node) if isinstance(argument, ast.expr) and _is_const(argument)
            )
    annotations = {
        id(name) for node in ast.walk(tree) for annotation in _annotations(node) for name in ast.walk(annotation)
    }
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Name | ast.Attribute)
            and _is_const(node)
            and isinstance(node.ctx, ast.Load)
            and id(node) not in called
            and id(node) not in annotations
        ):
            found.append(f"{where}:{node.lineno} hands Const on as a value")
    return found


def _annotations(node: ast.AST) -> list[ast.expr]:
    """the annotations ``node`` carries (an argument's, a return's, an annotated assignment's).

    :param node: any node
    :ptype node: ast.AST
    :return: the annotations
    :rtype: list[ast.expr]
    """
    if isinstance(node, ast.AnnAssign):
        return [node.annotation]
    if isinstance(node, ast.arg) and node.annotation is not None:
        return [node.annotation]
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.returns is not None:
        return [node.returns]
    return []


def test_every_const_is_made_from_a_code_constant() -> None:
    files = sorted(
        path for pattern in _SOURCE_GLOBS for root in _REPO_ROOT.glob(pattern) for path in root.rglob("*.py")
    )
    offenders = [entry for path in files for entry in violations(path, relative_to=_REPO_ROOT)]
    assert offenders == [], (
        "a Const is written into SQL as a literal: make it only from a string literal or a module-level name bound "
        "to one, never from a value that arrives at run time (Pace's ruling 2026-11-16):\n" + "\n".join(offenders)
    )


def test_the_walker_reads_the_shapes_it_must(tmp_path: Path) -> None:
    module = tmp_path / "pins.py"
    module.write_text(
        "from threetears.core.keyed import Const\n"
        "from threetears.core import keyed\n"
        "from threetears.core.keyed import Const as C\n"
        "from typing import Final\n"
        "_USER: Final = 'user'\n"
        "_TWICE = 'a'\n"
        "_TWICE = 'b'\n"
        "_GROUP: Final[Const] = Const('group')\n"
        "OK = (Const(_USER), keyed.Const('x'))\n"
        "def f(kind: str, typed: Const) -> Const:\n"
        "    isinstance(kind, Const)\n"
        "    Const(kind)\n"
        "    Const(_TWICE)\n"
        "    Const(f'{kind}')\n"
        "    Const(text='x')\n"
        "    list(map(Const, [kind]))\n"
        "    make = keyed.Const\n"
        "    return Const('ok')\n",
        encoding="utf-8",
    )
    assert [entry.split(" ", 1)[1] for entry in violations(module, relative_to=tmp_path)] == [
        "imports Const as C",
        "Const(kind) is no constant",
        "Const(_TWICE) is no constant",
        "Const(f'{kind}') is no constant",
        "Const(text='x') is no constant",
        "hands Const on as a value",
        "hands Const on as a value",
    ]
