"""the public API surface of a Python source tree, read statically.

**What counts.** A module whose path has no ``_``-prefixed component offers:

- every ``__all__`` entry, or -- when it declares no ``__all__`` -- every public
  name it DEFINES at top level (Python's own rule for ``import *``, less what it
  merely imports);
- ``Class.method`` for every public method of a class it exports;
- whatever the caller's extractors add: :func:`http_routes` for a service whose
  API is its HTTP routes, or a repo's own extractor for a surface no name sweep
  can see (tool actions, input fields).

**What is never quietly empty.** A module that does not parse, an ``__all__``
that is not a literal list of strings (``[*base.__all__, "x"]``,
``sorted([...])``), an ``__all__`` augmented or changed with anything but literal
strings (``__all__ += other``, ``__all__.remove(...)``), assigned twice, assigned
inside a block, or changed without being assigned -- each raises
:class:`SurfaceReadError` naming the module. Literal additions
(``__all__.append("SlackAdapter")`` under an optional-extra ``try``) are read, and
count: the name is API wherever the extra is installed. Reading such a module as exporting
nothing would compare two empty sets and pass any growth inside it, which is the
one answer a gate must never give silently.

Static parsing of source text and git blobs: no import, no install, no network.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from threetears.enforcement.release.versions import run_git

__all__ = [
    "ROUTE_METHODS",
    "BaselineUnreadableError",
    "SurfaceExtractor",
    "SurfaceReadError",
    "TreeSurface",
    "http_routes",
    "is_private_module",
    "module_surface",
    "surface_at",
    "surface_now",
]

#: reads extra public surface out of one parsed module. Raises ``ValueError``,
#: with a sentence saying why, when it cannot read what it is looking for.
type SurfaceExtractor = Callable[[ast.Module], set[str]]

#: decorator attributes that declare an HTTP route on a FastAPI app or router.
ROUTE_METHODS = frozenset({"get", "post", "put", "patch", "delete", "head", "options", "api_route", "websocket"})


class SurfaceReadError(Exception):
    """a module's public surface cannot be read, so it must not be compared.

    :param module_path: repo-relative path of the module
    :ptype module_path: str
    :param reason: what made it unreadable
    :ptype reason: str
    """

    def __init__(self, module_path: str, reason: str) -> None:
        """records which module failed and why.

        :param module_path: repo-relative path of the module
        :ptype module_path: str
        :param reason: what made it unreadable
        :ptype reason: str
        :return: nothing
        :rtype: None
        """
        super().__init__(f"{module_path}: {reason}")
        self.module_path = module_path
        self.reason = reason


class BaselineUnreadableError(Exception):
    """a git ref's tree, or an object in it, cannot be read in this clone."""


@dataclass
class TreeSurface:
    """the public surface of one source root, module by module.

    :param present: whether the root exists at all (in the tree or at the ref)
    :ptype present: bool
    :param modules: repo-relative module path onto its surface
    :ptype modules: dict[str, set[str]]
    :param errors: one sentence per module whose surface could not be read
    :ptype errors: list[str]
    """

    present: bool
    modules: dict[str, set[str]] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


def is_private_module(path: str) -> bool:
    """whether *path* names an internal module a consumer may not import by name.

    :param path: repo-relative module path
    :ptype path: str
    :return: ``True`` when any path component is ``_``-prefixed
    :rtype: bool
    """
    return any(part.startswith("_") and part != "__init__.py" for part in PurePosixPath(path).parts)


def http_routes(tree: ast.Module) -> set[str]:
    """every HTTP route a decorator in *tree* declares, as ``route METHOD path``.

    A new route is a feature a caller can reach whether or not it adds a name.
    Routes added by ``add_api_route`` are not seen.

    :param tree: parsed module
    :ptype tree: ast.Module
    :return: route entries
    :rtype: set[str]
    """
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        for decorator in node.decorator_list:
            if (
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Attribute)
                and decorator.func.attr in ROUTE_METHODS
                and decorator.args
                and isinstance(decorator.args[0], ast.Constant)
                and isinstance(decorator.args[0].value, str)
                and (decorator.args[0].value == "" or decorator.args[0].value.startswith("/"))
            ):
                found.add(f"route {decorator.func.attr.upper()} {decorator.args[0].value}")
    return found


def _module_level(body: Sequence[ast.stmt], nested: bool = False) -> Iterator[tuple[ast.stmt, bool]]:
    """every statement that runs at module level, with whether it sits inside a block.

    Descends ``if``/``try``/``with``/``for``/``while`` bodies, never a function or class.

    :param body: statements to walk
    :ptype body: Sequence[ast.stmt]
    :param nested: whether *body* is itself inside a block
    :ptype nested: bool
    :return: each statement and its nesting
    :rtype: Iterator[tuple[ast.stmt, bool]]
    """
    for node in body:
        yield node, nested
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            continue
        for name in ("body", "orelse", "finalbody"):
            yield from _module_level(getattr(node, name, []), nested=True)
        for handler in getattr(node, "handlers", []):
            yield from _module_level(handler.body, nested=True)
        for case in getattr(node, "cases", []):
            yield from _module_level(case.body, nested=True)


def _names_all(target: ast.expr) -> bool:
    """whether an assignment target binds ``__all__``, directly or by unpacking.

    :param target: assignment target
    :ptype target: ast.expr
    :return: ``True`` when ``__all__`` is among the names bound
    :rtype: bool
    """
    return any(isinstance(node, ast.Name) and node.id == "__all__" for node in ast.walk(target))


def _literal_exports(value: ast.expr | None) -> set[str]:
    """the names a literal ``__all__`` value lists.

    :param value: the assigned expression
    :ptype value: ast.expr | None
    :return: exported names
    :rtype: set[str]
    :raises ValueError: if *value* is not a literal list, tuple or set of strings
    """
    if not isinstance(value, ast.List | ast.Tuple | ast.Set):
        raise ValueError("not a literal")
    names: set[str] = set()
    for element in value.elts:
        if not (isinstance(element, ast.Constant) and isinstance(element.value, str)):
            raise ValueError("not a literal")
        names.add(element.value)
    return names


def _added_exports(node: ast.stmt, module_path: str) -> set[str] | None:
    """the names a statement adds to an existing ``__all__``, or ``None`` when it touches none.

    ``__all__ += [...]``, ``__all__.append("x")``, ``__all__.extend([...])`` and
    ``__all__.insert(i, "x")`` with literal strings are read: the pattern that
    exports an optional extra's names only when it imports. Every name so added
    counts, because each is API wherever the extra is installed.

    :param node: one module-level statement
    :ptype node: ast.stmt
    :param module_path: repo-relative path, for the error
    :ptype module_path: str
    :return: added names, or ``None`` when *node* does not change ``__all__``
    :rtype: set[str] | None
    :raises SurfaceReadError: if it changes ``__all__`` in a way that cannot be read
    """
    added: set[str] | None = None
    shown = ast.unparse(node)[:80]
    if isinstance(node, ast.AugAssign) and _names_all(node.target):
        if not isinstance(node.op, ast.Add):
            raise SurfaceReadError(module_path, f"line {node.lineno}: `__all__` is changed by `{shown}`")
        try:
            added = _literal_exports(node.value)
        except ValueError:
            raise SurfaceReadError(
                module_path, f"line {node.lineno}: `__all__` is augmented with a non-literal (`{shown}`)"
            ) from None
    elif (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and isinstance(node.value.func.value, ast.Name)
        and node.value.func.value.id == "__all__"
    ):
        method, args = node.value.func.attr, node.value.args
        literal = args[-1] if args else None
        if method in ("append", "insert") and isinstance(literal, ast.Constant) and isinstance(literal.value, str):
            added = {literal.value}
        elif method == "extend" and literal is not None and not isinstance(literal, ast.Constant):
            try:
                added = _literal_exports(literal)
            except ValueError:
                added = None
        if added is None:
            raise SurfaceReadError(
                module_path, f"line {node.lineno}: `__all__` is changed in a way the gate cannot read (`{shown}`)"
            )
    return added


def _declared_exports(tree: ast.Module, module_path: str) -> set[str] | None:
    """the module's ``__all__``, or ``None`` when it declares none.

    One unconditional, top-level, literal assignment, plus any literal additions
    (:func:`_added_exports`).

    :param tree: parsed module
    :ptype tree: ast.Module
    :param module_path: repo-relative path, for the error
    :ptype module_path: str
    :return: exported names, or ``None`` for a module with no ``__all__``
    :rtype: set[str] | None
    :raises SurfaceReadError: if ``__all__`` is not one readable literal declaration
    """
    assignments: list[ast.Assign | ast.AnnAssign] = []
    additions: set[str] = set()
    changed_at: int | None = None
    for node, nested in _module_level(tree.body):
        added = _added_exports(node, module_path)
        if added is not None:
            additions |= added
            changed_at = changed_at or node.lineno
            continue
        binding: ast.Assign | ast.AnnAssign | None = None
        if isinstance(node, ast.Assign) and any(_names_all(target) for target in node.targets):
            binding = node
        elif isinstance(node, ast.AnnAssign) and _names_all(node.target):
            binding = node
        if binding is None:
            continue
        if nested:
            raise SurfaceReadError(module_path, f"line {node.lineno}: `__all__` is assigned inside a block")
        assignments.append(binding)
    if len(assignments) > 1:
        lines = ", ".join(str(node.lineno) for node in assignments)
        raise SurfaceReadError(module_path, f"`__all__` is assigned more than once (lines {lines})")
    if changed_at is not None and not assignments:
        raise SurfaceReadError(module_path, f"line {changed_at}: `__all__` is changed but never assigned")
    exports: set[str] | None = None
    if assignments:
        node = assignments[0]
        try:
            exports = _literal_exports(node.value) | additions
        except ValueError:
            shown = ast.unparse(node.value) if node.value is not None else "<no value>"
            raise SurfaceReadError(
                module_path,
                f"line {node.lineno}: `__all__` is not a literal list of strings ({shown[:80]})",
            ) from None
    return exports


def _defined_public_names(tree: ast.Module) -> set[str]:
    """every public name a module DEFINES at top level: functions, classes, assignments.

    :param tree: parsed module
    :ptype tree: ast.Module
    :return: defined names with no leading underscore
    :rtype: set[str]
    """
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names |= {target.id for target in node.targets if isinstance(target, ast.Name)}
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.TypeAlias) and isinstance(node.name, ast.Name):
            names.add(node.name.id)
    return {name for name in names if not name.startswith("_")}


def module_surface(source: str, module_path: str, extractors: Sequence[SurfaceExtractor] = ()) -> set[str]:
    """the names one module offers a consumer.

    :param source: module source
    :ptype source: str
    :param module_path: repo-relative path, named in any error
    :ptype module_path: str
    :param extractors: extra surface readers, applied to the parsed module
    :ptype extractors: Sequence[SurfaceExtractor]
    :return: exported names, ``Class.method`` for exported classes, and extractor entries
    :rtype: set[str]
    :raises SurfaceReadError: if the module does not parse, its ``__all__`` cannot
        be read as a literal, or an extractor cannot read it
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise SurfaceReadError(module_path, f"does not parse (line {exc.lineno}: {exc.msg})") from None
    declared = _declared_exports(tree, module_path)
    exported = declared if declared is not None else _defined_public_names(tree)
    surface = set(exported)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in exported:
            for item in node.body:
                if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef) and not item.name.startswith("_"):
                    surface.add(f"{node.name}.{item.name}")
    for extractor in extractors:
        try:
            surface |= extractor(tree)
        except ValueError as exc:
            raise SurfaceReadError(module_path, str(exc)) from None
    return surface


def _read_into(result: TreeSurface, module_path: str, source: str, extractors: Sequence[SurfaceExtractor]) -> None:
    """reads one module into *result*, recording an unreadable one as an error.

    :param result: the tree being built
    :ptype result: TreeSurface
    :param module_path: repo-relative path
    :ptype module_path: str
    :param source: module source
    :ptype source: str
    :param extractors: extra surface readers
    :ptype extractors: Sequence[SurfaceExtractor]
    :return: nothing
    :rtype: None
    """
    try:
        result.modules[module_path] = module_surface(source, module_path, extractors)
    except SurfaceReadError as exc:
        result.errors.append(str(exc))


def surface_now(root: Path, source_root: str, extractors: Sequence[SurfaceExtractor] = ()) -> TreeSurface:
    """every public module's surface under *source_root* in the WORKING TREE.

    The working tree, not ``HEAD``: a gate that ignores uncommitted work passes
    locally and fails in CI.

    :param root: repository root
    :ptype root: Path
    :param source_root: repo-relative source tree
    :ptype source_root: str
    :param extractors: extra surface readers
    :ptype extractors: Sequence[SurfaceExtractor]
    :return: the tree's surface; ``present`` is ``False`` when the directory is absent
    :rtype: TreeSurface
    """
    base = root / source_root
    result = TreeSurface(present=base.is_dir())
    for path in sorted(base.rglob("*.py")) if result.present else []:
        relative = path.relative_to(root).as_posix()
        if not is_private_module(relative):
            _read_into(result, relative, path.read_text(encoding="utf-8"), extractors)
    return result


def surface_at(root: Path, ref: str, source_root: str, extractors: Sequence[SurfaceExtractor] = ()) -> TreeSurface:
    """every public module's surface under *source_root* at git ref *ref*, in one git read.

    One ``cat-file --batch`` for the whole tree rather than a ``git show`` per
    file. Blobs are keyed back to EVERY path that holds them, so two modules with
    byte-identical content each keep their own baseline.

    :param root: repository root
    :ptype root: Path
    :param ref: tag or commit
    :ptype ref: str
    :param source_root: repo-relative source tree
    :ptype source_root: str
    :param extractors: extra surface readers
    :ptype extractors: Sequence[SurfaceExtractor]
    :return: the tree's surface; ``present`` is ``False`` when *ref* has no such tree
    :rtype: TreeSurface
    :raises BaselineUnreadableError: if *ref*'s tree or one of its blobs is not in this clone
    """
    listing = run_git(root, "ls-tree", "-r", ref, "--", source_root)
    if not listing.ok:
        raise BaselineUnreadableError(f"`git ls-tree {ref}` failed: {listing.stderr or 'no output'}")
    paths_by_blob: dict[str, list[str]] = {}
    any_entry = False
    for line in listing.text.splitlines():
        meta, _, path = line.partition("\t")
        fields = meta.split()
        any_entry = True
        if len(fields) == 3 and fields[1] == "blob" and path.endswith(".py") and not is_private_module(path):
            paths_by_blob.setdefault(fields[2], []).append(path)
    result = TreeSurface(present=any_entry)
    if not paths_by_blob:
        return result
    batch = run_git(root, "cat-file", "--batch", stdin="".join(f"{sha}\n" for sha in paths_by_blob).encode())
    if not batch.ok:
        raise BaselineUnreadableError(f"`git cat-file --batch` failed at {ref}: {batch.stderr or 'no output'}")
    output = batch.stdout
    cursor = 0
    while cursor < len(output):
        header_end = output.index(b"\n", cursor)
        header = output[cursor:header_end].decode().split(" ")
        if len(header) != 3:
            raise BaselineUnreadableError(f"object {' '.join(header)} at {ref}: the clone lacks it (shallow?)")
        sha, _, size = header
        body_start = header_end + 1
        body_end = body_start + int(size)
        source = output[body_start:body_end].decode("utf-8", errors="replace")
        for path in paths_by_blob[sha]:
            _read_into(result, path, source, extractors)
        cursor = body_end + 1
    return result
