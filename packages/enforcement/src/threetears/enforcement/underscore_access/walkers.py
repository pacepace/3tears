"""six-shape walkers for the underscore-access enforcement domain.

each walker takes a tuple of src roots plus a repo root and returns a
list of :class:`~threetears.enforcement.common.violations.Violation`
records keyed by ``category="underscore_access.<LETTER>"``. the shapes
are deliberately independent so a consumer can run any subset, and
the runner orchestrates a combined pass.

implementation notes:

- shape A walks ``ImportFrom`` nodes; a private import is a violation
  iff the importer and the imported module live in different
  top-level packages under the same src-root.
- shape B shells out to ``ruff check --select SLF001`` because ruff
  already does the scope analysis correctly. when ruff isn't
  available the walker returns empty (the runner controls whether to
  even try, via :attr:`UnderscoreAccessConfig.enable_shape_b_ruff`).
- shape C scans every module for public top-level names without a
  matching ``__all__`` declaration. ``__init__.py`` files are
  scanned same as any other module — an empty package legitimately
  has no public surface and so produces no violation.
- shape D collects every class's class-body private attrs/methods,
  then re-walks every class looking for subclasses that re-declare a
  parent's private name. textual base-name match (last segment) is
  used; aliased imports and metaclass tricks are accepted false
  negatives.
- shape E inspects literal-shaped ``__all__`` assignments for entries
  whose string value is a private name.
- shape F walks ``setattr`` / ``getattr`` / ``delattr`` / ``hasattr``
  calls whose name argument is a private string literal -- the spelling
  of a private access that every attribute-node check, SLF001
  included, cannot see. unlike A, C, D and E it is meant to scan the
  ``tests/`` trees too.
- shape I walks the classes a ``tests/`` tree defines for ``self._x`` / ``cls._x`` reaching a
  PRIVATE STATE attribute a production base class keeps (one its methods assign on ``self`` or
  ``cls``, or its class body binds to a non-function value), and that the test class does not
  define itself. a base's private METHODS stay callable: a protected hook is what a subclass is
  for. its state is the base's implementation, and a test bound to it passes vacuously once the
  base keeps that state another way.
"""

from __future__ import annotations

import ast
import re
import subprocess
from collections.abc import Iterable
from pathlib import Path

from threetears.enforcement.common import (
    Violation,
    is_private_name,
    iter_python_files,
    parse_python_file,
)

__all__ = [
    "package_id",
    "same_package",
    "shape_a_violations",
    "shape_b_violations",
    "shape_c_violations",
    "shape_d_violations",
    "shape_e_violations",
    "shape_f_violations",
    "shape_i_violations",
]


_RUFF_TIMEOUT_SECONDS = 60

# ruff's concise output format: ``path:line:col: CODE message``
_RUFF_LINE_RE = re.compile(r"^(?P<file>.+?):(?P<line>\d+):(?P<col>\d+):\s+SLF001\s+(?P<detail>.+)$")
# ruff's SLF001 message embeds the offending name as ``\`_name\```.
_RUFF_SYMBOL_RE = re.compile(r"`(_[A-Za-z_0-9]+)`")


def package_id(path: Path, src_roots: Iterable[Path]) -> tuple[str, ...]:
    """compute package identity for a source file.

    identity is ``(src_root_str, top_level_package_name)``: two files
    share a package iff they share an identity. files outside every
    provided src root return ``()`` (the empty tuple acts as "no
    identity"). we resolve paths so symlinked / relative arguments
    behave identically.

    :param path: source file path
    :ptype path: Path
    :param src_roots: candidate src-root directories
    :ptype src_roots: Iterable[Path]
    :return: identity tuple, or ``()`` if outside any src root
    :rtype: tuple[str, ...]
    """
    resolved = path.resolve()
    for root in src_roots:
        root_resolved = root.resolve()
        try:
            rel = resolved.relative_to(root_resolved)
        except ValueError:
            # NOSILENT: this IS the containment test -- the loop asks each root in turn whether it
            # holds the path, and "no" is the expected answer for every root but one.
            continue
        parts = rel.parts
        if not parts:
            continue
        return (str(root_resolved), parts[0])
    return ()


def same_package(a: Path, b: Path, src_roots: Iterable[Path]) -> bool:
    """true iff two source paths share a package identity under some src root.

    :param a: first source file
    :ptype a: Path
    :param b: second source file
    :ptype b: Path
    :param src_roots: candidate src-root directories
    :ptype src_roots: Iterable[Path]
    :return: whether both files share a non-empty package identity
    :rtype: bool
    """
    roots = tuple(src_roots)
    id_a = package_id(a, roots)
    id_b = package_id(b, roots)
    return id_a != () and id_a == id_b


def _resolve_module_to_file(
    module: str,
    src_roots: Iterable[Path],
) -> Path | None:
    """find the source file that defines a fully-qualified module name.

    tries the ``<root>/<a>/<b>/<c>.py`` shape first, then the
    ``<root>/<a>/<b>/<c>/__init__.py`` package shape. returns the
    first hit across the iterable in order; ``None`` when no root
    contains the module.

    :param module: dotted module path, e.g. ``threetears.core.x``
    :ptype module: str
    :param src_roots: candidate src roots
    :ptype src_roots: Iterable[Path]
    :return: path to the ``.py`` file, or ``None`` if unresolved
    :rtype: Path | None
    """
    parts = module.split(".")
    for root in src_roots:
        file_candidate = root.joinpath(*parts).with_suffix(".py")
        if file_candidate.is_file():
            return file_candidate
        pkg_candidate = root.joinpath(*parts, "__init__.py")
        if pkg_candidate.is_file():
            return pkg_candidate
    return None


def shape_a_violations(
    scan_roots: tuple[Path, ...],
    repo_root: Path,
    inheritance_roots: tuple[Path, ...],
) -> list[Violation]:
    """walk every source file for cross-module private imports (shape A).

    a violation is an ``ImportFrom`` whose module resolves to a file
    in a different top-level package than the importer, and at least
    one alias has a private name. relative imports (``from .x import
    _y``) are skipped because they are always intra-package; modules
    that don't resolve (third-party deps not in any inheritance root)
    are also skipped.

    note the asymmetry: importers come from ``scan_roots`` (the
    consumer's own code), but the module-to-file lookup and the
    same-package classification both use ``inheritance_roots`` (the
    union including path-deps). this makes a private import from a
    sibling package resolve correctly even when the consumer's
    ``scan_roots`` doesn't include that sibling's code.

    :param scan_roots: where to look for importers / violations
    :ptype scan_roots: tuple[Path, ...]
    :param repo_root: repo root (retained for parity with sibling walkers)
    :ptype repo_root: Path
    :param inheritance_roots: union of src roots used to resolve
        ``from <mod> import ...`` to a defining file and to compute
        same-package identity
    :ptype inheritance_roots: tuple[Path, ...]
    :return: shape-A violations
    :rtype: list[Violation]
    """
    _ = repo_root
    violations: list[Violation] = []
    for root in scan_roots:
        for importer in iter_python_files(root):
            tree = parse_python_file(importer)
            if tree is None:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom):
                    continue
                module = node.module
                if module is None:
                    continue
                if node.level and node.level > 0:
                    continue
                private_aliases = [alias for alias in node.names if is_private_name(alias.name)]
                if not private_aliases:
                    continue
                defining_file = _resolve_module_to_file(
                    module,
                    inheritance_roots,
                )
                if defining_file is None:
                    continue
                if same_package(importer, defining_file, inheritance_roots):
                    continue
                for alias in private_aliases:
                    reason = f"imports private name '{alias.name}' from '{module}' across package boundary"
                    violations.append(
                        Violation(
                            category="underscore_access.A",
                            file=importer,
                            line=node.lineno,
                            symbol=alias.name,
                            reason=reason,
                        )
                    )
    return violations


def shape_b_violations(
    repo_root: Path,
    scan_roots: tuple[Path, ...],
) -> list[Violation]:
    """delegate cross-class protected access to ruff's SLF001 (shape B).

    invokes ``ruff check --select SLF001`` over the provided scan
    roots only. ruff already does the scope analysis correctly
    (``obj._x`` is only flagged when ``obj`` is not ``self`` / ``cls``)
    and re-implementing it would be a tar pit. when ruff is missing
    the function returns an empty list rather than raising — the
    runner is the policy point that decides whether to even invoke
    this walker (via :attr:`UnderscoreAccessConfig.enable_shape_b_ruff`).

    :param repo_root: repo root (used as cwd for the subprocess)
    :ptype repo_root: Path
    :param scan_roots: src roots ruff should lint; only the consumer's
        own code is in scope
    :ptype scan_roots: tuple[Path, ...]
    :return: shape-B violations
    :rtype: list[Violation]
    :raises subprocess.TimeoutExpired: ruff did not complete within
        the timeout (60s); surfaces upward so the test fails loudly
    """
    if not scan_roots:
        return []
    args = [
        "ruff",
        "check",
        "--select",
        "SLF001",
        "--output-format",
        "concise",
        "--no-fix",
        "--force-exclude",
    ]
    args.extend(str(r) for r in scan_roots)
    try:
        completed = subprocess.run(
            args,
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=_RUFF_TIMEOUT_SECONDS,
            check=False,
        )
    except FileNotFoundError:
        # ruff not on PATH; runner caller decides via enable_shape_b_ruff
        # whether to call us at all. an empty result is the honest answer
        # when the tool is absent.
        return []
    return parse_ruff_slf001_output(completed.stdout, repo_root)


def parse_ruff_slf001_output(stdout: str, repo_root: Path) -> list[Violation]:
    """parse ruff's concise-format output into shape-B violations.

    extracted as a public helper so unit tests can exercise the
    parsing without spawning a subprocess. the canonical line shape
    is::

        path/to/file.py:LINE:COL: SLF001 Private member accessed: `_name`

    relative paths get re-anchored against ``repo_root`` so the
    resulting :class:`Violation` records carry absolute paths
    (matching the rest of the domain).

    :param stdout: ruff's stdout text
    :ptype stdout: str
    :param repo_root: repo root for anchoring relative paths
    :ptype repo_root: Path
    :return: parsed violations in source order
    :rtype: list[Violation]
    """
    violations: list[Violation] = []
    for raw in stdout.splitlines():
        match = _RUFF_LINE_RE.match(raw)
        if match is None:
            continue
        file_path = Path(match.group("file"))
        if not file_path.is_absolute():
            file_path = (repo_root / file_path).resolve()
        detail = match.group("detail")
        symbol_match = _RUFF_SYMBOL_RE.search(detail)
        symbol = symbol_match.group(1) if symbol_match else "_<unknown>"
        violations.append(
            Violation(
                category="underscore_access.B",
                file=file_path,
                line=int(match.group("line")),
                symbol=symbol,
                reason=detail,
            )
        )
    return violations


def shape_c_violations(
    scan_roots: tuple[Path, ...],
    repo_root: Path,
    skip_basenames: frozenset[str],
) -> list[Violation]:
    """walk every source file for missing ``__all__`` with public surface (shape C).

    a module is flagged iff it:

    - is not in ``skip_basenames`` (per-repo carve-out for conftests,
      ``__main__`` shims, version stubs)
    - defines at least one public top-level name (class / function /
      assignment whose target is a public ``ast.Name``)
    - does not declare ``__all__``

    a module with zero public top-level names is silently passed —
    legitimately empty packages (``__init__.py`` re-export shims) and
    private-only modules are not violations.

    :param scan_roots: where to look for violations; only iterates
        these roots
    :ptype scan_roots: tuple[Path, ...]
    :param repo_root: repo root (retained for parity with sibling walkers)
    :ptype repo_root: Path
    :param skip_basenames: file basenames excluded from the check
    :ptype skip_basenames: frozenset[str]
    :return: shape-C violations
    :rtype: list[Violation]
    """
    _ = repo_root
    violations: list[Violation] = []
    for root in scan_roots:
        for module_path in iter_python_files(root):
            if module_path.name in skip_basenames:
                continue
            tree = parse_python_file(module_path)
            if tree is None:
                continue
            has_all = False
            public_names: list[tuple[str, int]] = []
            for node in tree.body:
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if not isinstance(target, ast.Name):
                            continue
                        if target.id == "__all__":
                            has_all = True
                        elif not target.id.startswith("_"):
                            public_names.append((target.id, node.lineno))
                elif isinstance(node, ast.AnnAssign):
                    target = node.target
                    if not isinstance(target, ast.Name):
                        continue
                    if target.id == "__all__":
                        has_all = True
                    elif not target.id.startswith("_"):
                        public_names.append((target.id, node.lineno))
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if not node.name.startswith("_"):
                        public_names.append((node.name, node.lineno))
                elif isinstance(node, ast.ClassDef):
                    if not node.name.startswith("_"):
                        public_names.append((node.name, node.lineno))
            if has_all:
                continue
            if not public_names:
                continue
            first_name, first_line = public_names[0]
            reason = f"module defines {len(public_names)} public name(s) but no __all__; first is '{first_name}'"
            violations.append(
                Violation(
                    category="underscore_access.C",
                    file=module_path,
                    line=first_line,
                    symbol="__all__",
                    reason=reason,
                )
            )
    return violations


def _collect_class_private_attrs(
    src_roots: tuple[Path, ...],
) -> dict[str, dict[str, tuple[Path, int]]]:
    """build a map of ``class_name`` -> ``{_private_name -> (file, line)}``.

    keyed by bare class name (last textual segment). this is lossy
    for aliased imports (``from x import Base as B``) but captures
    the overwhelming common case. namespace collisions across modules
    (two distinct ``Base`` classes) merge into one entry — downstream
    shape-D may over-flag for that pattern, which is acceptable: name
    collisions across a codebase are themselves a smell worth surfacing.

    a private attr is detected from class-body :class:`ast.FunctionDef`,
    :class:`ast.AsyncFunctionDef`, :class:`ast.Assign`, and
    :class:`ast.AnnAssign` nodes whose name (or first ``ast.Name``
    target) passes :func:`is_private_name`.

    :param src_roots: every src root the scanner should consider
    :ptype src_roots: tuple[Path, ...]
    :return: nested map class -> private name -> defining (file, line)
    :rtype: dict[str, dict[str, tuple[Path, int]]]
    """
    result: dict[str, dict[str, tuple[Path, int]]] = {}
    for root in src_roots:
        for file in iter_python_files(root):
            tree = parse_python_file(file)
            if tree is None:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue
                entries = result.setdefault(node.name, {})
                for item in node.body:
                    _record_class_private(item, entries, file)
    return result


def _record_class_private(
    item: ast.stmt,
    entries: dict[str, tuple[Path, int]],
    file: Path,
) -> None:
    """record a single class-body statement's private attr / method (if any).

    extracted from :func:`_collect_class_private_attrs` to keep the
    nesting shallow. handles function defs, plain assignments (with
    multiple targets), and annotated assignments.

    :param item: class-body statement
    :ptype item: ast.stmt
    :param entries: accumulator for this class
    :ptype entries: dict[str, tuple[Path, int]]
    :param file: file containing the class
    :ptype file: Path
    """
    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
        if is_private_name(item.name):
            entries[item.name] = (file, item.lineno)
        return
    if isinstance(item, ast.Assign):
        for target in item.targets:
            if isinstance(target, ast.Name) and is_private_name(target.id):
                entries[target.id] = (file, item.lineno)
        return
    if isinstance(item, ast.AnnAssign):
        if isinstance(item.target, ast.Name) and is_private_name(item.target.id):
            entries[item.target.id] = (file, item.lineno)


def shape_d_violations(
    scan_roots: tuple[Path, ...],
    repo_root: Path,
    inheritance_roots: tuple[Path, ...],
) -> list[Violation]:
    """walk every class for subclass shadowing of a base private name (shape D).

    a class ``Sub`` with base ``Base`` violates shape D iff it
    declares any class-body ``_name`` (attribute assignment or method)
    that ``Base`` also declares. resolution is textual — base
    classes are last-segment names — so multiple-inheritance through
    aliased imports may be missed; the common case (direct subclass
    in the same repo) is what motivates the shard.

    a class is never flagged for shadowing its own definition. the
    detail message references the base's defining file and line so
    the operator can find it without grepping.

    note the asymmetry: the class-private attr inventory is built
    from ``inheritance_roots`` (so a sub in this repo correctly
    detects shadowing of a base declared in a sibling package), but
    the violation walk only iterates ``scan_roots`` (only the
    consumer's own subclasses are flagged).

    :param scan_roots: where to look for shadowing violations;
        iterates only these roots when collecting subclasses
    :ptype scan_roots: tuple[Path, ...]
    :param repo_root: repo root for the relative-path rendering in
        the violation reason text
    :ptype repo_root: Path
    :param inheritance_roots: src roots from which to build the
        class-private attr map used to look up base-declared names;
        path-dep aware so cross-package shadowing resolves
    :ptype inheritance_roots: tuple[Path, ...]
    :return: shape-D violations
    :rtype: list[Violation]
    """
    class_privates = _collect_class_private_attrs(inheritance_roots)
    violations: list[Violation] = []
    for root in scan_roots:
        for file in iter_python_files(root):
            tree = parse_python_file(file)
            if tree is None:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue
                base_names = _base_names_textual(node)
                if not base_names:
                    continue
                sub_privates: list[tuple[str, int]] = []
                for item in node.body:
                    _accumulate_sub_private(item, sub_privates)
                for base_name in base_names:
                    if base_name == node.name:
                        continue
                    base_entries = class_privates.get(base_name, {})
                    for name, line in sub_privates:
                        if name not in base_entries:
                            continue
                        defining_file, defining_line = base_entries[name]
                        if defining_file == file and defining_line == line:
                            continue
                        try:
                            rel_def: Path | str = defining_file.relative_to(repo_root)
                        except ValueError:
                            rel_def = defining_file
                        reason = (
                            f"class '{node.name}' shadows private '{name}' "
                            f"declared on base class '{base_name}' at "
                            f"{rel_def}:{defining_line}; underscore names are "
                            f"implementation detail of the defining class "
                            f"and must not be overridden"
                        )
                        violations.append(
                            Violation(
                                category="underscore_access.D",
                                file=file,
                                line=line,
                                symbol=name,
                                reason=reason,
                            )
                        )
    return violations


def _base_names_textual(cls: ast.ClassDef) -> list[str]:
    """return the textual last-segment name of every base in ``cls.bases``.

    only :class:`ast.Name` (``Base``) and :class:`ast.Attribute`
    (``pkg.Base``) bases are recognised — generic-subscript shapes
    (``Base[T]``) and call shapes (``make_base()``) are skipped
    because shape D needs an exact-name handle for the
    :data:`class_privates` lookup.

    :param cls: class definition to inspect
    :ptype cls: ast.ClassDef
    :return: list of last-segment base names in source order
    :rtype: list[str]
    """
    names: list[str] = []
    for base in cls.bases:
        if isinstance(base, ast.Name):
            names.append(base.id)
        elif isinstance(base, ast.Attribute):
            names.append(base.attr)
    return names


def _accumulate_sub_private(
    item: ast.stmt,
    sub_privates: list[tuple[str, int]],
) -> None:
    """append ``item``'s private name to ``sub_privates`` if applicable.

    mirror of :func:`_record_class_private` but flat-list shaped —
    shape D consumers want order of declaration, not a dict.

    :param item: class-body statement
    :ptype item: ast.stmt
    :param sub_privates: accumulator (mutated in place)
    :ptype sub_privates: list[tuple[str, int]]
    """
    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
        if is_private_name(item.name):
            sub_privates.append((item.name, item.lineno))
        return
    if isinstance(item, ast.Assign):
        for target in item.targets:
            if isinstance(target, ast.Name) and is_private_name(target.id):
                sub_privates.append((target.id, item.lineno))
        return
    if isinstance(item, ast.AnnAssign):
        if isinstance(item.target, ast.Name) and is_private_name(item.target.id):
            sub_privates.append((item.target.id, item.lineno))


def shape_e_violations(
    scan_roots: tuple[Path, ...],
    repo_root: Path,
) -> list[Violation]:
    """walk every ``__all__`` for underscore-prefixed entries (shape E).

    only literal :class:`ast.List` / :class:`ast.Tuple` /
    :class:`ast.Set` shapes are inspected. computed forms
    (``__all__ = list(generated())``, ``__all__ += [...]``) are
    skipped — static analysis cannot resolve them and cases in
    practice are rare. each private entry produces a separate
    violation pinned to the element's source line.

    :param scan_roots: where to look for violations; only iterates
        these roots
    :ptype scan_roots: tuple[Path, ...]
    :param repo_root: repo root (retained for parity with sibling walkers)
    :ptype repo_root: Path
    :return: shape-E violations
    :rtype: list[Violation]
    """
    _ = repo_root
    violations: list[Violation] = []
    for root in scan_roots:
        for file in iter_python_files(root):
            tree = parse_python_file(file)
            if tree is None:
                continue
            for node in tree.body:
                value, target_line = _extract_all_value(node)
                if value is None:
                    continue
                if not isinstance(value, (ast.List, ast.Tuple, ast.Set)):
                    continue
                for elt in value.elts:
                    if not isinstance(elt, ast.Constant):
                        continue
                    if not isinstance(elt.value, str):
                        continue
                    name = elt.value
                    if not is_private_name(name):
                        continue
                    reason = (
                        f"__all__ lists private name '{name}'; underscore "
                        f"prefix declares implementation detail, __all__ "
                        f"declares public api -- the two declarations "
                        f"contradict. either drop the underscore (name is "
                        f"public) or remove from __all__ (name is private)"
                    )
                    line = elt.lineno if hasattr(elt, "lineno") else target_line
                    violations.append(
                        Violation(
                            category="underscore_access.E",
                            file=file,
                            line=line,
                            symbol=name,
                            reason=reason,
                        )
                    )
    return violations


#: the builtins that reach an attribute through a NAME passed as data. ruff's SLF001 and every other
#: shape look at attribute nodes, imports and ``__all__``, so ``setattr(obj, "_x", v)`` passed them all
#: while binding to exactly the implementation detail they exist to protect.
_REFLECTIVE_ACCESSORS: frozenset[str] = frozenset({"setattr", "getattr", "delattr", "hasattr"})

#: the receivers that ARE the owner of a private name, the same test SLF001 applies.
_OWNER_RECEIVERS: frozenset[str] = frozenset({"self", "cls"})

#: a node that opens a name scope, for resolving what a receiver name is bound to.
_ScopeNode = ast.Module | ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda | ast.ClassDef

#: the node types whose body is a NEW scope, not part of the scope that contains them.
_NESTED_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)


def shape_f_violations(
    scan_roots: tuple[Path, ...],
    repo_root: Path,
) -> list[Violation]:
    """walk every call that reaches a private name through ``setattr``/``getattr``/``delattr``/``hasattr`` (shape F).

    a call is a violation when its name argument is a string constant naming a private
    (single-leading-underscore, not dunder) attribute and its receiver is none of:

    - ``self`` or ``cls``, the owner, exactly as SLF001 excludes them;
    - an object this module DEFINES -- a name whose nearest binding is a ``def`` or ``class``
      statement. constructing an instance of somebody else's class does not make its private
      slots yours; only defining the object does.

    and a name this module stamps onto an object it defines (``setattr(wrapper, "_marker", True)``
    on its own ``def wrapper``) is the module's own protocol: reading it back anywhere in the SAME
    module is allowed, off any object. that is the decorator-marker shape. any other module reading
    or writing it is still a violation.

    **there is no reflective escape hatch.** where a private genuinely must be reached -- a
    third-party object with no public accessor -- spell it as an attribute, where SLF001 sees it and
    the per-file exemption and its ledger entry record why. the string spelling is the one no tool
    reads back.

    :param scan_roots: where to look for violations; unlike shapes A, C, D and E this is meant to
        include ``tests/`` trees, where every instance that surfaced the gap lived
    :ptype scan_roots: tuple[Path, ...]
    :param repo_root: repo root (retained for parity with sibling walkers)
    :ptype repo_root: Path
    :return: shape-F violations
    :rtype: list[Violation]
    """
    _ = repo_root
    violations: list[Violation] = []
    for root in scan_roots:
        for file in iter_python_files(root):
            tree = parse_python_file(file)
            if tree is None:
                continue
            violations.extend(_reflective_private_violations(tree, file))
    return violations


def _reflective_private_violations(tree: ast.Module, file: Path) -> list[Violation]:
    """shape-F violations in one parsed module.

    two passes because a marker's ownership is a property of the whole module: the ``setattr``
    that stamps it may sit below the ``getattr`` that reads it.

    :param tree: the parsed module
    :ptype tree: ast.Module
    :param file: the module's path, for the violation records
    :ptype file: Path
    :return: the module's violations, in source order
    :rtype: list[Violation]
    """
    bindings_cache: dict[int, dict[str, set[str]]] = {}
    owned_markers: set[str] = set()
    candidates: list[tuple[ast.Call, str, str]] = []
    for call, scopes in _reflective_private_calls(tree):
        accessor = call.func.id if isinstance(call.func, ast.Name) else ""
        name = _private_name_argument(call, scopes)
        receiver = call.args[0]
        if isinstance(receiver, ast.Name) and receiver.id in _OWNER_RECEIVERS:
            continue
        if _is_module_own_object(receiver, scopes, bindings_cache):
            if accessor == "setattr":
                owned_markers.add(name)
            continue
        candidates.append((call, accessor, name))
    violations: list[Violation] = _namespace_subscript_violations(tree, file)
    for call, accessor, name in candidates:
        if name in owned_markers:
            continue
        violations.append(
            Violation(
                category="underscore_access.F",
                file=file,
                line=call.lineno,
                symbol=name,
                reason=(
                    f"reaches private '{name}' through {accessor}() on an object that is not self, cls or "
                    f"one this module defines. SLF001 cannot see a name passed as a string, so this binds to "
                    f"another module's implementation detail unseen. promote the name, pass the value through "
                    f"a public argument or accessor, or -- for a third-party object with no public accessor -- "
                    f"read it as an attribute under a per-file SLF001 exemption and its ledger entry"
                ),
            )
        )
    return violations


def _reflective_private_calls(tree: ast.Module) -> list[tuple[ast.Call, tuple[_ScopeNode, ...]]]:
    """every reflective call with a private-name literal, with its enclosing scopes outermost first.

    :param tree: the parsed module
    :ptype tree: ast.Module
    :return: ``(call, scopes)`` pairs in source order
    :rtype: list[tuple[ast.Call, tuple[_ScopeNode, ...]]]
    """
    found: list[tuple[ast.Call, tuple[_ScopeNode, ...]]] = []

    def _visit(node: ast.AST, scopes: tuple[_ScopeNode, ...]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, _NESTED_SCOPES):
                _visit(child, (*scopes, child))
                continue
            if isinstance(child, ast.Call) and _private_name_argument(child, scopes):
                found.append((child, scopes))
            _visit(child, scopes)

    _visit(tree, (tree,))
    return sorted(found, key=lambda pair: (pair[0].lineno, pair[0].col_offset))


def _private_name_argument(call: ast.Call, scopes: tuple[_ScopeNode, ...] = ()) -> str:
    """the private attribute name a reflective builtin call names, or ``""`` when it names none.

    The name is a string literal, or a variable whose values the enclosing code spells as literals:
    a ``for`` over a literal list or tuple, or a ``pytest.mark.parametrize`` of the function it sits
    in. ``getattr(module, name)`` over ``["_A", "_B"]`` binds the privates as surely as the literal.

    :param call: a call node
    :ptype call: ast.Call
    :param scopes: the call's enclosing scopes, outermost first
    :ptype scopes: tuple[_ScopeNode, ...]
    :return: the private name, or the empty string for any other call
    :rtype: str
    """
    result = ""
    if isinstance(call.func, ast.Name) and call.func.id in _REFLECTIVE_ACCESSORS and len(call.args) >= 2:
        argument = call.args[1]
        if isinstance(argument, ast.Constant) and isinstance(argument.value, str) and is_private_name(argument.value):
            result = argument.value
        elif isinstance(argument, ast.Name):
            result = next((v for v in _literal_values(argument.id, scopes) if is_private_name(v)), "")
    return result


def _literal_values(name: str, scopes: tuple[_ScopeNode, ...]) -> list[str]:
    """the string literals a variable is given by the code around it: the ``for`` loops over literal
    sequences that bind it, and a ``parametrize`` decorator of an enclosing function that names it.

    :param name: the variable
    :ptype name: str
    :param scopes: the enclosing scopes, outermost first
    :ptype scopes: tuple[_ScopeNode, ...]
    :return: the literal strings, in source order
    :rtype: list[str]
    """
    values: list[str] = []
    for scope in scopes:
        for node in ast.walk(scope):
            if (
                isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension))
                and isinstance(node.target, ast.Name)
                and node.target.id == name
            ):
                values += _strings_in(node.iter)
        if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for decorator in scope.decorator_list:
                values += _parametrized_strings(decorator, name)
    return values


def _parametrized_strings(decorator: ast.expr, name: str) -> list[str]:
    """the strings a ``parametrize(...)`` decorator gives ``name``, or none.

    :param decorator: a decorator expression
    :ptype decorator: ast.expr
    :param name: the parameter
    :ptype name: str
    :return: its literal strings
    :rtype: list[str]
    """
    found: list[str] = []
    if (
        isinstance(decorator, ast.Call)
        and isinstance(decorator.func, ast.Attribute)
        and decorator.func.attr == "parametrize"
        and len(decorator.args) >= 2
        and isinstance(decorator.args[0], ast.Constant)
        and isinstance(decorator.args[0].value, str)
    ):
        names = [n.strip() for n in decorator.args[0].value.split(",")]
        if name in names:
            position = names.index(name)
            for case in getattr(decorator.args[1], "elts", []):
                if len(names) == 1:
                    found += _strings_in(case)
                elif isinstance(case, (ast.Tuple, ast.List)) and position < len(case.elts):
                    found += _strings_in(case.elts[position])
    return found


def _strings_in(expr: ast.expr) -> list[str]:
    """the string constants an expression is, or holds as a literal list, tuple or set.

    :param expr: the expression
    :ptype expr: ast.expr
    :return: its strings
    :rtype: list[str]
    """
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        return [expr.value]
    if isinstance(expr, (ast.List, ast.Tuple, ast.Set)):
        return [e.value for e in expr.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]
    return []


def _namespace_subscript_violations(tree: ast.Module, file: Path) -> list[Violation]:
    """a private name read out of another object's namespace by subscript: ``vars(x)["_y"]`` or
    ``x.__dict__["_y"]`` -- the getattr of shape F, spelled as a dictionary lookup.

    :param tree: the parsed module
    :ptype tree: ast.Module
    :param file: the module's path
    :ptype file: Path
    :return: the violations, in source order
    :rtype: list[Violation]
    """
    violations: list[Violation] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Subscript)
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
            and is_private_name(node.slice.value)
        ):
            continue
        namespace = node.value
        receiver: ast.expr | None = None
        if (
            isinstance(namespace, ast.Call)
            and isinstance(namespace.func, ast.Name)
            and namespace.func.id == "vars"
            and namespace.args
        ):
            receiver = namespace.args[0]
        elif isinstance(namespace, ast.Attribute) and namespace.attr == "__dict__":
            receiver = namespace.value
        if receiver is None or (isinstance(receiver, ast.Name) and receiver.id in _OWNER_RECEIVERS):
            continue
        violations.append(
            Violation(
                category="underscore_access.F",
                file=file,
                line=node.lineno,
                symbol=node.slice.value,
                reason=(
                    f"reads private '{node.slice.value}' out of another object's namespace (vars() / "
                    f"__dict__), which SLF001 cannot see; reach it through public behaviour, or promote it"
                ),
            )
        )
    return violations


def _is_module_own_object(
    receiver: ast.expr,
    scopes: tuple[_ScopeNode, ...],
    bindings_cache: dict[int, dict[str, set[str]]],
) -> bool:
    """whether *receiver* names an object this module defines with ``def`` or ``class``.

    resolved the way python resolves a name: innermost scope first, class bodies visible only to a
    call sitting directly in them. the nearest scope that binds the name decides, and it must bind it
    ONLY by definition -- a parameter, an assignment or an import is somebody else's object, and a
    name that is both defined and rebound cannot be trusted to be the definition at the call.

    :param receiver: the reflective call's first argument
    :ptype receiver: ast.expr
    :param scopes: the enclosing scopes, outermost first
    :ptype scopes: tuple[_ScopeNode, ...]
    :param bindings_cache: per-scope binding maps, keyed by node identity, so a module is analysed once
    :ptype bindings_cache: dict[int, dict[str, set[str]]]
    :return: true when the nearest binding of the name is a definition and nothing else
    :rtype: bool
    """
    if not isinstance(receiver, ast.Name):
        return False
    for depth, scope in enumerate(reversed(scopes)):
        if isinstance(scope, ast.ClassDef) and depth > 0:
            continue
        key = id(scope)
        if key not in bindings_cache:
            bindings_cache[key] = _scope_bindings(scope)
        kinds = bindings_cache[key].get(receiver.id)
        if kinds:
            return kinds == {"def"}
    return False


def _scope_bindings(scope: _ScopeNode) -> dict[str, set[str]]:
    """every name *scope* binds, with how it binds it: ``def``, ``param``, ``assign`` or ``import``.

    nested functions, classes, lambdas and comprehensions are their own scopes: their bodies are not
    walked, though a nested ``def`` or ``class`` statement binds its own NAME here.

    :param scope: the scope node
    :ptype scope: _ScopeNode
    :return: name -> the kinds of binding it receives in this scope
    :rtype: dict[str, set[str]]
    """
    bindings: dict[str, set[str]] = {}

    def _bind(name: str, kind: str) -> None:
        bindings.setdefault(name, set()).add(kind)

    if isinstance(scope, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
        arguments = scope.args
        for arg in (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs):
            _bind(arg.arg, "param")
        for extra in (arguments.vararg, arguments.kwarg):
            if extra is not None:
                _bind(extra.arg, "param")
    stack: list[ast.AST] = list(scope.body) if isinstance(scope.body, list) else [scope.body]
    while stack:
        node = stack.pop()
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            _bind(node.name, "def")
            continue
        if isinstance(node, ast.Lambda | ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp):
            continue
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store | ast.Del):
            _bind(node.id, "assign")
        elif isinstance(node, ast.Import | ast.ImportFrom):
            for alias in node.names:
                _bind(alias.asname or alias.name.split(".")[0], "import")
        elif isinstance(node, ast.Global | ast.Nonlocal):
            for name in node.names:
                _bind(name, "assign")
        elif isinstance(node, ast.ExceptHandler | ast.MatchAs | ast.MatchStar) and node.name:
            _bind(node.name, "assign")
        elif isinstance(node, ast.MatchMapping) and node.rest:
            _bind(node.rest, "assign")
        stack.extend(ast.iter_child_nodes(node))
    return bindings


def _extract_all_value(node: ast.stmt) -> tuple[ast.expr | None, int]:
    """return ``(__all__'s rhs expr, defining lineno)`` or ``(None, 0)``.

    handles both plain assignment and annotated assignment shapes.
    matches only top-level ``__all__`` because shape E applies to
    module-level declarations.

    :param node: candidate top-level statement
    :ptype node: ast.stmt
    :return: rhs expression and lineno, or sentinel ``(None, 0)``
    :rtype: tuple[ast.expr | None, int]
    """
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "__all__":
                return node.value, node.lineno
        return None, 0
    if isinstance(node, ast.AnnAssign):
        if isinstance(node.target, ast.Name) and node.target.id == "__all__" and node.value is not None:
            return node.value, node.lineno
        return None, 0
    return None, 0


def _collect_class_private_state(
    src_roots: tuple[Path, ...],
) -> dict[str, dict[str, tuple[Path, int]]]:
    """build a map of ``class_name`` -> ``{_private state name -> (file, line)}``.

    state is a private name the class's own methods assign on ``self`` / ``cls``
    (``self._lock = ...``), or that its class body binds to anything but a function.
    keyed by bare class name, as :func:`_collect_class_private_attrs` is, with the
    same accepted imprecision.

    :param src_roots: every src root the scanner should consider
    :ptype src_roots: tuple[Path, ...]
    :return: nested map class -> private state name -> defining (file, line)
    :rtype: dict[str, dict[str, tuple[Path, int]]]
    """
    result: dict[str, dict[str, tuple[Path, int]]] = {}
    for root in src_roots:
        for file in iter_python_files(root):
            tree = parse_python_file(file)
            if tree is None:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    entries = result.setdefault(node.name, {})
                    for name, line in _class_state(node):
                        entries.setdefault(name, (file, line))
    return result


def _class_state(cls: ast.ClassDef) -> list[tuple[str, int]]:
    """every private state name ``cls`` binds: class-body values, and ``self``/``cls`` stores in its methods.

    :param cls: the class
    :ptype cls: ast.ClassDef
    :return: (name, line) pairs
    :rtype: list[tuple[str, int]]
    """
    found: list[tuple[str, int]] = []
    for item in cls.body:
        targets: list[ast.expr] = []
        if isinstance(item, ast.Assign):
            targets = list(item.targets)
        elif isinstance(item, ast.AnnAssign):
            targets = [item.target]
        found.extend(
            (target.id, item.lineno)
            for target in targets
            if isinstance(target, ast.Name) and is_private_name(target.id)
        )
    for node in _own_nodes(cls):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.ctx, ast.Store)
            and isinstance(node.value, ast.Name)
            and node.value.id in _OWNER_RECEIVERS
            and is_private_name(node.attr)
        ):
            found.append((node.attr, node.lineno))
    return found


def _own_nodes(cls: ast.ClassDef) -> list[ast.AST]:
    """every node in ``cls``'s body that belongs to it, not to a class nested in it.

    :param cls: the class
    :ptype cls: ast.ClassDef
    :return: the nodes
    :rtype: list[ast.AST]
    """
    out: list[ast.AST] = []
    stack: list[ast.AST] = list(cls.body)
    while stack:
        node = stack.pop()
        out.append(node)
        stack.extend(child for child in ast.iter_child_nodes(node) if not isinstance(child, ast.ClassDef))
    return out


def _class_defines(cls: ast.ClassDef) -> set[str]:
    """every private name ``cls`` itself declares: its methods and its class-body bindings.

    a ``self._x = ...`` in its methods is not a declaration of ``_x`` when a base keeps ``_x``:
    that is a write of the base's state.

    :param cls: the class
    :ptype cls: ast.ClassDef
    :return: the names
    :rtype: set[str]
    """
    names: set[str] = set()
    for item in cls.body:
        targets: list[ast.expr] = []
        if isinstance(item, ast.Assign):
            targets = list(item.targets)
        elif isinstance(item, ast.AnnAssign):
            targets = [item.target]
        names.update(target.id for target in targets if isinstance(target, ast.Name) and is_private_name(target.id))
    names.update(
        item.name
        for item in cls.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and is_private_name(item.name)
    )
    return names


def shape_i_violations(
    scan_roots: tuple[Path, ...],
    repo_root: Path,
    inheritance_roots: tuple[Path, ...],
) -> list[Violation]:
    """walk the classes ``scan_roots`` define for a reach into a production base's private state (shape I).

    a class violates shape I when one of its own methods reads or writes ``self._x`` /
    ``cls._x`` where ``_x`` is private state of a base class the inheritance roots define
    (see :func:`_collect_class_private_state`) and the class does not declare ``_x`` itself
    (as a method or a class-body binding; assigning ``self._x`` is writing the base's state).
    a base's private methods are not state, and calling one is not flagged. meant for the
    ``tests/`` trees: a test subclass reaches its base through the front door.

    :param scan_roots: where to look; the ``tests/`` trees
    :ptype scan_roots: tuple[Path, ...]
    :param repo_root: repo root for the relative-path rendering in the reason text
    :ptype repo_root: Path
    :param inheritance_roots: src roots whose classes' private state is protected
    :ptype inheritance_roots: tuple[Path, ...]
    :return: shape-I violations
    :rtype: list[Violation]
    """
    base_state = _collect_class_private_state(inheritance_roots)
    violations: list[Violation] = []
    for root in scan_roots:
        for file in iter_python_files(root):
            tree = parse_python_file(file)
            if tree is None:
                continue
            for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
                violations.extend(_base_state_reaches(cls, file, base_state, repo_root))
    return violations


def _base_state_reaches(
    cls: ast.ClassDef,
    file: Path,
    base_state: dict[str, dict[str, tuple[Path, int]]],
    repo_root: Path,
) -> list[Violation]:
    """the shape-I violations of one class.

    :return: the violations
    :rtype: list[Violation]
    """
    bases = [name for name in _base_names_textual(cls) if name != cls.name]
    inherited = {name: (base, where) for base in bases for name, where in base_state.get(base, {}).items()}
    if not inherited:
        return []
    own = _class_defines(cls)
    found: list[Violation] = []
    for node in _own_nodes(cls):
        if not (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in _OWNER_RECEIVERS
            and node.attr in inherited
            and node.attr not in own
        ):
            continue
        base, (defining_file, defining_line) = inherited[node.attr]
        try:
            rel_def: Path | str = defining_file.relative_to(repo_root)
        except ValueError:
            rel_def = defining_file
        found.append(
            Violation(
                category="underscore_access.I",
                file=file,
                line=node.lineno,
                symbol=node.attr,
                reason=(
                    f"class '{cls.name}' reaches '{node.attr}', private state of its base "
                    f"'{base}' ({rel_def}:{defining_line}); reach the base through its public "
                    f"surface, or a protected method it offers subclasses"
                ),
            )
        )
    return found
