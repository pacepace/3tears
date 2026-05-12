"""five-shape walkers for the underscore-access enforcement domain.

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
