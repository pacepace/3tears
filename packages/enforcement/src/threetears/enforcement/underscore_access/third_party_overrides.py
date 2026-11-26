"""shape J: a src class that overrides, or reads through ``super()``, a third-party base's private member.

Reaching into a library's private members is confined to one module per library and recorded in the
underscore exemptions ledger. Most of those reaches are attribute reads ruff's SLF001 reports, but two
spellings of one are invisible to it: a subclass that **overrides** a base's private member (defines
``_client_params`` on a ``ChatAnthropic`` subclass), and one that **reads it through super()**
(``super()._client_params``), which SLF001 skips as the owner's own access. Both bind to the library's
implementation exactly as an attribute read does, and break the same way on its next release. This
shape reports them, so each takes a ledger entry that names the library version it was verified
against.

A member is reported when:

- the class body defines it (a method, or a name it assigns), or a method of the class reads it as
  ``super()._x``; and
- it is single-underscore: not a dunder, not a sunder (``_missing_``, which :func:`is_private_name`
  already treats as public), and not name-mangled (``__x`` can't collide with a base's); and
- the first class in the class's ancestry that defines it is **from outside the repo's own namespace**
  -- the top-level packages under its src roots. A first definer the repo owns is an in-repo
  protected member, which shape D governs; and
- it is not a hook the library documents for subclasses to implement
  (the table below): LangChain's custom chat model guide asks for ``_generate`` and
  ``_llm_type``, and a custom tool implements ``_run``. Those underscores are the library's front
  door, not its implementation.

One violation per class and member, at its first definition; a ``super()`` read of a member the class
doesn't define is reported at the read.

**How a base is resolved.** Statically to a dotted name, through the module's imports (relative and
function-level ones included). A base in the repo's own namespace is followed statically, through the
module that defines it (and a package's re-export), so no repo code runs. A third-party base is
imported to read its MRO, since a library's ancestry is only knowable from the library: this shape,
unlike A to I, imports the third-party modules that define the bases it checks. One it can't import is
reported rather than passed, since nothing about its members is known.

**Not seen:** a base spelled as anything but a name or a dotted name (a call, a variable); a base's
private member set only on an instance (``self._x = ...`` in its ``__init__``); and a mixin with no
third-party base of its own, whose overrides reach a library only once a class composes it.
"""

from __future__ import annotations

import ast
import builtins
import importlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from threetears.enforcement.common import (
    Violation,
    find_local_src_roots,
    is_private_name,
    iter_python_files,
    nodes_by_type,
    parse_python_file,
)

__all__ = [
    "shape_j_file_violations",
    "shape_j_violations",
]

#: the category every shape-J violation carries.
_SHAPE_J: Final = "underscore_access.J"

#: underscore members a library documents as the ones a subclass implements, by the defining class's
#: ``module.qualname``. Overriding one is the library's extension contract, not a private reach.
#:
#: - ``BaseChatModel``: LangChain's "custom chat model" guide lists ``_generate`` and ``_llm_type`` as
#:   required, and ``_stream``, ``_agenerate``, ``_astream`` and ``_identifying_params`` as optional
#:   (verified against langchain-core 1.6.8, where ``_generate`` and ``_llm_type`` are abstract).
#: - ``BaseTool``: LangChain's "custom tools" guide subclasses ``BaseTool`` by implementing ``_run``
#:   and, optionally, ``_arun``.
_DOCUMENTED_SUBCLASS_HOOKS: Final[Mapping[str, frozenset[str]]] = {
    "langchain_core.language_models.chat_models.BaseChatModel": frozenset(
        {"_generate", "_agenerate", "_stream", "_astream", "_llm_type", "_identifying_params"}
    ),
    "langchain_core.tools.base.BaseTool": frozenset({"_run", "_arun"}),
}

#: how many re-exports a dotted name is followed through before it is given up as unresolved.
_MAX_REEXPORTS: Final = 8


def shape_j_violations(scan_roots: tuple[Path, ...], repo_root: Path) -> list[Violation]:
    """every class under ``scan_roots`` that overrides or ``super()``-reads a third-party private member.

    :param scan_roots: the src roots to scan; their top-level packages are the repo's own namespace
    :ptype scan_roots: tuple[Path, ...]
    :param repo_root: repo root (retained for parity with sibling walkers)
    :ptype repo_root: Path
    :return: shape-J violations, in file and source order
    :rtype: list[Violation]
    """
    del repo_root  # retained for parity with sibling walkers
    index = _Index(scan_roots)
    return [
        index.violation(file, finding)
        for root in scan_roots
        for file in iter_python_files(root)
        for finding in index.findings(file, root)
    ]


def shape_j_file_violations(path: Path, repo_root: Path) -> list[Violation]:
    """the shape-J violations in one file, as :func:`shape_j_violations` over the repo's src roots reports them.

    For the ledger, which resolves an entry against one file at a time. Empty for a file outside every
    src root of ``repo_root``.

    :param path: the source file
    :ptype path: Path
    :param repo_root: the repo's root, whose src roots are its own namespace
    :ptype repo_root: Path
    :return: the file's violations, in source order
    :rtype: list[Violation]
    """
    roots = find_local_src_roots(repo_root)
    root = next((r for r in roots if path.resolve().is_relative_to(r.resolve())), None)
    if root is None:
        return []
    index = _Index(roots)
    return [index.violation(path, finding) for finding in index.findings(path, root)]


@dataclass(frozen=True, slots=True)
class _Finding:
    """one reported member: where, and why."""

    line: int
    symbol: str
    reason: str


@dataclass(frozen=True, slots=True)
class _Module:
    """one of the repo's own modules, as far as resolving a base needs it."""

    name: str
    bindings: dict[str, str]
    classes: dict[str, ast.ClassDef]


@dataclass(frozen=True, slots=True)
class _OwnClass:
    """an ancestor the repo defines, with the module its own bases resolve in."""

    node: ast.ClassDef
    module: _Module


#: an ancestor that couldn't be imported, by its dotted name.
@dataclass(frozen=True, slots=True)
class _Unknown:
    dotted: str


_Ancestor = _OwnClass | type | _Unknown

#: what :func:`_import_object` answers for a dotted name it can't reach.
_UNIMPORTABLE: Final = object()


class _Index:
    """one run's view of the repo's own modules, parsed on demand and kept for the run."""

    def __init__(self, roots: Iterable[Path]) -> None:
        self._roots = tuple(root for root in roots if root.is_dir())
        self._own = _own_namespaces(self._roots)
        self._modules: dict[str, _Module | None] = {}

    def violation(self, file: Path, finding: _Finding) -> Violation:
        return Violation(category=_SHAPE_J, file=file, line=finding.line, symbol=finding.symbol, reason=finding.reason)

    def findings(self, file: Path, root: Path) -> list[_Finding]:
        """the findings in ``file``, which lives under ``root``."""
        tree = parse_python_file(file)
        if tree is None:
            return []
        module = self._module_from(tree, _module_name(file, root), file.name == "__init__.py")
        found: list[_Finding] = []
        for cls in nodes_by_type(file, ast.ClassDef):
            found.extend(self._class_findings(cls, module))
        return found

    # -- one class ---------------------------------------------------------------------------------

    def _class_findings(self, cls: ast.ClassDef, module: _Module) -> list[_Finding]:
        defined = _defined_members(cls)
        read = [(name, line) for name, line in _super_reads(cls) if name not in defined]
        if not defined and not read:
            return []
        ancestry = self._ancestry(cls, module, {id(cls)})
        found: list[_Finding] = []
        for member, line in [*defined.items(), *read]:
            finding = self._reach(cls.name, member, line, ancestry, through_super=member not in defined)
            if finding is not None:
                found.append(finding)
        return found

    def _reach(
        self, cls_name: str, member: str, line: int, ancestry: list[_Ancestor], *, through_super: bool
    ) -> _Finding | None:
        """the finding for ``member``, or ``None`` when its first definer is the repo's, or no one's."""
        how = "reads, through super()," if through_super else "overrides"
        for ancestor in ancestry:
            if isinstance(ancestor, _Unknown):
                return _Finding(
                    line,
                    member,
                    f"class '{cls_name}' {how} '{member}', and its base '{ancestor.dotted}' could not be "
                    f"imported to tell whether that is a third-party private member",
                )
            if isinstance(ancestor, _OwnClass):
                if member in _defined_members(ancestor.node):
                    return None  # the repo's own protected member: shape D's ground, not this one's
                continue
            if not _defines(ancestor, member):
                continue
            if ancestor.__module__.partition(".")[0] in self._own:
                return None
            if any(
                member in _DOCUMENTED_SUBCLASS_HOOKS.get(_qualname(other), ())
                for other in ancestry
                if isinstance(other, type)
            ):
                return None
            return _Finding(
                line,
                member,
                f"class '{cls_name}' {how} '{member}', a private member of the third-party base "
                f"'{_qualname(ancestor)}'; keep the reach in the one module that owns that library's "
                f"private members, and record it in the exemptions ledger with the version it was "
                f"verified against",
            )
        return None

    # -- ancestry ----------------------------------------------------------------------------------

    def _ancestry(self, cls: ast.ClassDef, module: _Module, seen: set[int]) -> list[_Ancestor]:
        """``cls``'s ancestors, nearest first, approximating its MRO: the repo's own classes are
        followed through their bases, statically; a third-party class contributes its ``__mro__``."""
        out: list[_Ancestor] = []
        for base in cls.bases:
            target = base.value if isinstance(base, ast.Subscript) else base
            dotted = _dotted(target)
            if dotted is None:
                continue
            for ancestor in self._resolve(dotted, module):
                if isinstance(ancestor, _OwnClass):
                    if id(ancestor.node) in seen:
                        continue
                    seen.add(id(ancestor.node))
                    out.append(ancestor)
                    out.extend(self._ancestry(ancestor.node, ancestor.module, seen))
                else:
                    out.append(ancestor)
        return out

    def _resolve(self, dotted: str, module: _Module) -> list[_Ancestor]:
        """what a base spelled ``dotted`` inside ``module`` is: one own class, a third-party MRO, an
        unknown, or nothing (a base this shape doesn't follow)."""
        head, _, rest = dotted.partition(".")
        if not rest and head in module.classes:
            return [_OwnClass(module.classes[head], module)]
        if head in module.bindings:
            qualified = f"{module.bindings[head]}.{rest}" if rest else module.bindings[head]
        elif not rest and hasattr(builtins, head):
            qualified = f"builtins.{head}"
        else:
            return []  # bound some other way, a variable or a parameter: not followed
        for _ in range(_MAX_REEXPORTS):
            if qualified.partition(".")[0] not in self._own:
                return self._third_party(qualified)
            found = self._own_target(qualified)
            if isinstance(found, _OwnClass):
                return [found]
            if found is None:
                return []  # the repo's own name, not a class this shape can see: not followed
            qualified = found  # a re-export: follow it to where it is defined
        return []

    def _own_target(self, qualified: str) -> _OwnClass | str | None:
        """the repo's class ``qualified`` names, the dotted name it re-exports, or ``None``."""
        parts = qualified.split(".")
        for split in range(len(parts) - 1, 0, -1):
            module = self._own_module(".".join(parts[:split]))
            if module is None:
                continue
            name, *deeper = parts[split:]
            if deeper:
                return None  # a class nested in a class, or a submodule's attribute: not followed
            if name in module.classes:
                return _OwnClass(module.classes[name], module)
            return module.bindings.get(name)
        return None

    def _third_party(self, qualified: str) -> list[_Ancestor]:
        found = _import_object(qualified)
        if isinstance(found, type):
            return list(found.__mro__)
        if found is _UNIMPORTABLE:
            return [_Unknown(qualified)]
        return []  # not a class -- ``typing.NamedTuple``, a factory -- so no members to override

    def _own_module(self, name: str) -> _Module | None:
        if name in self._modules:
            return self._modules[name]
        self._modules[name] = None
        relative = Path(*name.split("."))
        for root in self._roots:
            for candidate, is_package in (
                (root / relative.with_suffix(".py"), False),
                (root / relative / "__init__.py", True),
            ):
                if candidate.is_file():
                    tree = parse_python_file(candidate)
                    if tree is not None:
                        self._modules[name] = self._module_from(tree, name, is_package)
                        return self._modules[name]
        return None

    def _module_from(self, tree: ast.Module, name: str, is_package: bool) -> _Module:
        package = name if is_package else name.rpartition(".")[0]
        classes: dict[str, ast.ClassDef] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.setdefault(node.name, node)
        return _Module(name, _module_bindings(tree, package), classes)


# -- module-level helpers ----------------------------------------------------------------------------


def _own_namespaces(roots: Iterable[Path]) -> frozenset[str]:
    """the top-level package and module names under ``roots``: the repo's own import namespace."""
    names: set[str] = set()
    for root in roots:
        for child in root.iterdir():
            if child.is_dir() and child.name.isidentifier():
                names.add(child.name)
            elif child.suffix == ".py" and child.stem.isidentifier():
                names.add(child.stem)
    return frozenset(names)


def _module_name(file: Path, root: Path) -> str:
    """``file``'s dotted module name under ``root``; a package's ``__init__`` is the package."""
    parts = list(file.resolve().relative_to(root.resolve()).with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _module_bindings(tree: ast.Module, package: str) -> dict[str, str]:
    """every name the module's imports bind, function-level ones included, to the dotted name bound."""
    names: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname is not None:
                    names[alias.asname] = alias.name
                else:
                    head = alias.name.partition(".")[0]
                    names[head] = head
        elif isinstance(node, ast.ImportFrom):
            source = _absolute_module(node, package)
            if source is None:
                continue
            for alias in node.names:
                if alias.name != "*":
                    names[alias.asname or alias.name] = f"{source}.{alias.name}"
    return names


def _absolute_module(node: ast.ImportFrom, package: str) -> str | None:
    """the absolute module an ``ImportFrom`` names, a relative one resolved against ``package``."""
    if node.level == 0:
        return node.module
    parts = package.split(".") if package else []
    if node.level - 1 > len(parts):
        return None
    base = parts[: len(parts) - (node.level - 1)]
    if node.module:
        base.append(node.module)
    return ".".join(base) if base else None


def _defined_members(cls: ast.ClassDef) -> dict[str, int]:
    """the single-underscore names ``cls``'s body defines, each at its first definition."""
    found: dict[str, int] = {}
    for item in cls.body:
        targets: list[str] = []
        if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
            targets = [item.name]
        elif isinstance(item, ast.Assign):
            targets = [target.id for target in item.targets if isinstance(target, ast.Name)]
        elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            targets = [item.target.id]
        for name in targets:
            if _is_reachable_private(name):
                found.setdefault(name, item.lineno)
    return found


def _super_reads(cls: ast.ClassDef) -> list[tuple[str, int]]:
    """every ``super()._x`` in ``cls``'s own methods (not a nested class's), in source order."""
    found: list[tuple[str, int]] = []
    pending: list[ast.AST] = list(cls.body)
    while pending:
        node = pending.pop()
        if isinstance(node, ast.ClassDef):
            continue
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "super"
            and _is_reachable_private(node.attr)
        ):
            found.append((node.attr, node.lineno))
        pending.extend(ast.iter_child_nodes(node))
    return sorted(found, key=lambda item: item[1])


def _is_reachable_private(name: str) -> bool:
    """single-underscore and not name-mangled: a name a base of another class could also define."""
    return is_private_name(name) and not name.startswith("__")


def _dotted(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        inner = _dotted(node.value)
        return f"{inner}.{node.attr}" if inner is not None else None
    return None


def _import_object(qualified: str) -> object:
    """the object ``qualified`` names, importing its longest importable module prefix, or
    :data:`_UNIMPORTABLE` when no prefix imports or the module lacks the name."""
    parts = qualified.split(".")
    for split in range(len(parts) - 1, 0, -1):
        try:
            found: object = importlib.import_module(".".join(parts[:split]))
        except ImportError:  # NOSILENT: a shorter prefix may be the module; when none is, it is reported
            continue
        for attr in parts[split:]:
            found = getattr(found, attr, _UNIMPORTABLE)
            if found is _UNIMPORTABLE:
                break
        return found
    return _UNIMPORTABLE


def _defines(cls: type, member: str) -> bool:
    """whether ``cls`` itself defines ``member``: in its namespace, or as a pydantic private attribute."""
    namespace = vars(cls)
    private_attributes = namespace.get("__private_attributes__")
    return member in namespace or (isinstance(private_attributes, dict) and member in private_attributes)


def _qualname(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"
