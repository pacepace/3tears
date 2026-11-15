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
import hashlib
import io
import re
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from threetears.observe import BuildOnce
from threetears.enforcement.common import (
    Violation,
    clear_with_parse_cache,
    is_private_name,
    iter_python_files,
    note_unscanned,
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


@dataclass(frozen=True)
class _ClassFacts:
    """what the shapes read of one class definition, kept without its tree.

    :ivar name: the class's name
    :ivar textual_bases: :func:`_base_names_textual` -- shape D's bases
    :ivar generic_bases: :func:`_base_names_generic` -- shape I's bases
    :ivar private_members: class-body private methods and bindings, ``(name, line)`` in body order
        -- shape D's declarations, on both the base and the subclass side
    :ivar state: private state the class binds, its class-body values then its own ``self`` / ``cls`` stores
        -- shape I's protected state of a production class
    :ivar defines: :func:`_class_defines` -- what a test class declares itself, for shape I
    :ivar owner_reaches: every ``self._x`` / ``cls._x`` among the class's own nodes, ``(name, line)``
        in :func:`_own_nodes` order -- shape I's candidate reaches
    """

    name: str
    textual_bases: tuple[str, ...]
    generic_bases: tuple[str, ...]
    private_members: tuple[tuple[str, int], ...]
    state: tuple[tuple[str, int], ...]
    defines: frozenset[str]
    owner_reaches: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class _ModuleFacts:
    """everything every shape reads of one module, extracted from ONE parse and kept without the tree.

    a whole workspace's trees run to gigabytes, so no cache can hold them; these facts are a few
    kilobytes a module, so one cache holds the workspace's and each shape reads them instead of
    parsing the module again. a module that did not parse keeps only why, and every shape that
    reaches it says so through :func:`note_unscanned`, as each shape did when it parsed for itself.

    :ivar path: the module
    :ivar digest: the hash of the module's bytes these facts were read from
    :ivar unscanned: why the module could not be parsed, or ``""`` when it was
    :ivar private_imports: shape A's absolute ``from <module> import`` statements naming a private
        alias: ``(module, line, private aliases)`` in :func:`ast.walk` order
    :ivar has_all: shape C: the module body assigns ``__all__``
    :ivar public_names: shape C: the module body's public top-level names, ``(name, line)``
    :ivar all_privates: shape E: the private entries of a literal ``__all__``, ``(name, line)``
    :ivar classes: every class the module defines, nested ones included, in :func:`ast.walk` order
    :ivar reflective: shape F's violations in the module
    """

    path: Path
    digest: bytes
    unscanned: str
    private_imports: tuple[tuple[str, int, tuple[str, ...]], ...] = ()
    has_all: bool = False
    public_names: tuple[tuple[str, int], ...] = ()
    all_privates: tuple[tuple[str, int], ...] = ()
    classes: tuple[_ClassFacts, ...] = ()
    reflective: tuple[Violation, ...] = ()


def _content_digest(data: bytes) -> bytes:
    """the hash a module's facts are current against.

    :param data: the module's bytes
    :ptype data: bytes
    :return: the digest
    :rtype: bytes
    """
    return hashlib.blake2b(data, digest_size=16).digest()


def _same_content(facts: _ModuleFacts) -> bool:
    """whether the module still holds the bytes *facts* were read from.

    the content, not a modification stamp: a file rewritten inside one tick of a coarse timestamp
    clock, at the same size, keeps its stamp, and a stamp-keyed cache would answer from the old
    source. reading the bytes back costs a small fraction of parsing them.

    :param facts: cached facts
    :ptype facts: _ModuleFacts
    :return: ``True`` while the file's bytes hash to the digest the facts were read at
    :rtype: bool
    :raises OSError: when the file can no longer be read
    """
    return _content_digest(facts.path.read_bytes()) == facts.digest


def _one_module(_facts: _ModuleFacts) -> int:
    """what keeping one module's facts costs: one entry of :data:`_FACTS_MAX_MODULES`.

    :param _facts: the facts
    :ptype _facts: _ModuleFacts
    :return: 1
    :rtype: int
    """
    return 1


#: how many modules' facts :data:`_FACTS` keeps. a module's facts weigh a few kilobytes, so this is
#: tens of megabytes at most, and several times this repo's ~3,600 scanned modules; past it a module
#: is read and parsed again on every ask rather than evicting another.
_FACTS_MAX_MODULES = 20_000

#: every module's facts, by path, current while its bytes are unchanged. a ``BuildOnce`` because a
#: hand-filled module-level cache races between threads
#: (``tests/enforcement/test_build_once_is_the_only_lazy_fill.py``).
_FACTS: BuildOnce[Path, _ModuleFacts] = BuildOnce(
    is_current=_same_content, cost=_one_module, max_cost=_FACTS_MAX_MODULES
)
# a caller that clears the parse cache to read the source afresh gets fresh facts too
clear_with_parse_cache(_FACTS.clear)


def _read_module_facts(path: Path) -> _ModuleFacts:
    """read and parse *path* once, and extract what every shape reads of it.

    the source is decoded exactly as :func:`~threetears.enforcement.common.parse_python_path`
    decodes it (``Path.read_text(encoding="utf-8")``: strict utf-8, universal newlines), so the
    tree, its line numbers and its string constants are the ones that parse would produce.

    :param path: the module
    :ptype path: Path
    :return: its facts, or the reason it could not be parsed
    :rtype: _ModuleFacts
    :raises OSError: when the file cannot be read
    """
    data = path.read_bytes()
    digest = _content_digest(data)
    try:
        source = io.TextIOWrapper(io.BytesIO(data), encoding="utf-8").read()
        tree = ast.parse(source, filename=str(path))
    except UnicodeDecodeError:
        return _ModuleFacts(path=path, digest=digest, unscanned="not valid utf-8")
    except SyntaxError as exc:
        return _ModuleFacts(path=path, digest=digest, unscanned=f"syntax error at line {exc.lineno}")
    by_type: dict[type[ast.AST], list[ast.AST]] = {}
    for node in ast.walk(tree):
        by_type.setdefault(type(node), []).append(node)
    has_all, public_names = _module_surface(tree)
    return _ModuleFacts(
        path=path,
        digest=digest,
        unscanned="",
        private_imports=_private_imports(by_type.get(ast.ImportFrom, [])),
        has_all=has_all,
        public_names=public_names,
        all_privates=_all_privates(tree),
        classes=tuple(_class_facts(node) for node in by_type.get(ast.ClassDef, []) if isinstance(node, ast.ClassDef)),
        reflective=tuple(
            _reflective_private_violations(
                tree,
                path,
                [n for n in by_type.get(ast.Call, []) if isinstance(n, ast.Call)],
                [n for n in by_type.get(ast.Subscript, []) if isinstance(n, ast.Subscript)],
            )
        ),
    )


def _module_facts(path: Path) -> _ModuleFacts | None:
    """*path*'s facts, or ``None`` -- said through :func:`note_unscanned` -- when it does not parse.

    every shape asks here for every module it walks, so a module is parsed once however many
    shapes read it, and a module no shape could check is reported by each shape that skipped it.

    :param path: the module
    :ptype path: Path
    :return: its facts, or ``None`` for a module that is not valid utf-8 python
    :rtype: _ModuleFacts | None
    :raises OSError: when the file cannot be read
    """
    facts = _FACTS.get(path, lambda: _read_module_facts(path))
    if facts.unscanned:
        note_unscanned(path, facts.unscanned)
        return None
    return facts


def _private_imports(import_froms: list[ast.AST]) -> tuple[tuple[str, int, tuple[str, ...]], ...]:
    """shape A's candidates: absolute ``from`` imports that name a private alias.

    :param import_froms: the module's ``ImportFrom`` nodes, in :func:`ast.walk` order
    :ptype import_froms: list[ast.AST]
    :return: ``(module, line, private aliases)``, in the same order
    :rtype: tuple[tuple[str, int, tuple[str, ...]], ...]
    """
    found: list[tuple[str, int, tuple[str, ...]]] = []
    for node in import_froms:
        if not isinstance(node, ast.ImportFrom) or node.module is None or node.level:
            continue
        private_aliases = tuple(alias.name for alias in node.names if is_private_name(alias.name))
        if private_aliases:
            found.append((node.module, node.lineno, private_aliases))
    return tuple(found)


def _module_surface(tree: ast.Module) -> tuple[bool, tuple[tuple[str, int], ...]]:
    """shape C's reading of a module body: whether it assigns ``__all__``, and its public names.

    :param tree: the parsed module
    :ptype tree: ast.Module
    :return: ``(has __all__, public (name, line) in body order)``
    :rtype: tuple[bool, tuple[tuple[str, int], ...]]
    """
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
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not node.name.startswith("_"):
                public_names.append((node.name, node.lineno))
    return has_all, tuple(public_names)


def _all_privates(tree: ast.Module) -> tuple[tuple[str, int], ...]:
    """shape E's reading of a module body: the private string entries of a literal ``__all__``.

    :param tree: the parsed module
    :ptype tree: ast.Module
    :return: ``(name, line)`` per private entry, in source order
    :rtype: tuple[tuple[str, int], ...]
    """
    found: list[tuple[str, int]] = []
    for node in tree.body:
        value, target_line = _extract_all_value(node)
        if value is None or not isinstance(value, (ast.List, ast.Tuple, ast.Set)):
            continue
        for elt in value.elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str) and is_private_name(elt.value):
                found.append((elt.value, elt.lineno if hasattr(elt, "lineno") else target_line))
    return tuple(found)


def _class_facts(cls: ast.ClassDef) -> _ClassFacts:
    """what shapes D and I read of one class.

    :param cls: the class
    :ptype cls: ast.ClassDef
    :return: its facts
    :rtype: _ClassFacts
    """
    private_members: list[tuple[str, int]] = []
    for item in cls.body:
        _accumulate_sub_private(item, private_members)
    owner_reaches: list[tuple[str, int]] = []
    state = _class_body_state(cls)
    for node in _own_nodes(cls):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in _OWNER_RECEIVERS
            and is_private_name(node.attr)
        ):
            owner_reaches.append((node.attr, node.lineno))
            if isinstance(node.ctx, ast.Store):
                state.append((node.attr, node.lineno))
    return _ClassFacts(
        name=cls.name,
        textual_bases=tuple(_base_names_textual(cls)),
        generic_bases=tuple(_base_names_generic(cls)),
        private_members=tuple(private_members),
        state=tuple(state),
        defines=frozenset(_class_defines(cls)),
        owner_reaches=tuple(owner_reaches),
    )


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
            facts = _module_facts(importer)
            if facts is None:
                continue
            for module, line, private_aliases in facts.private_imports:
                defining_file = _resolve_module_to_file(
                    module,
                    inheritance_roots,
                )
                if defining_file is None:
                    continue
                if same_package(importer, defining_file, inheritance_roots):
                    continue
                for alias in private_aliases:
                    reason = f"imports private name '{alias}' from '{module}' across package boundary"
                    violations.append(
                        Violation(
                            category="underscore_access.A",
                            file=importer,
                            line=line,
                            symbol=alias,
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
            facts = _module_facts(module_path)
            if facts is None:
                continue
            if facts.has_all:
                continue
            if not facts.public_names:
                continue
            first_name, first_line = facts.public_names[0]
            reason = f"module defines {len(facts.public_names)} public name(s) but no __all__; first is '{first_name}'"
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
            facts = _module_facts(file)
            if facts is None:
                continue
            for cls in facts.classes:
                entries = result.setdefault(cls.name, {})
                for name, line in cls.private_members:
                    entries[name] = (file, line)
    return result


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
            facts = _module_facts(file)
            if facts is None:
                continue
            for node in facts.classes:
                base_names = node.textual_bases
                if not base_names:
                    continue
                sub_privates = node.private_members
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

    flat-list shaped, in declaration order: shape D reads it as the
    subclass's declarations, and folds it into a dict (last one wins)
    for the base side.

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
            facts = _module_facts(file)
            if facts is None:
                continue
            for name, line in facts.all_privates:
                reason = (
                    f"__all__ lists private name '{name}'; underscore "
                    f"prefix declares implementation detail, __all__ "
                    f"declares public api -- the two declarations "
                    f"contradict. either drop the underscore (name is "
                    f"public) or remove from __all__ (name is private)"
                )
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
            facts = _module_facts(file)
            if facts is None:
                continue
            violations.extend(facts.reflective)
    return violations


def _reflective_private_violations(
    tree: ast.Module, file: Path, calls: list[ast.Call], subscripts: list[ast.Subscript]
) -> list[Violation]:
    """shape-F violations in one parsed module.

    two passes because a marker's ownership is a property of the whole module: the ``setattr``
    that stamps it may sit below the ``getattr`` that reads it.

    :param tree: the parsed module
    :ptype tree: ast.Module
    :param file: the module's path, for the violation records
    :ptype file: Path
    :param calls: every ``Call`` node in *tree*, in :func:`ast.walk` order
    :ptype calls: list[ast.Call]
    :param subscripts: every ``Subscript`` node in *tree*, in :func:`ast.walk` order
    :ptype subscripts: list[ast.Subscript]
    :return: the module's violations, in source order
    :rtype: list[Violation]
    """
    bindings_cache: dict[int, dict[str, set[str]]] = {}
    owned_markers: set[str] = set()
    candidates: list[tuple[ast.Call, str, str]] = []
    reflective = _reflective_private_calls(tree) if _may_reflect_privately(calls) else []
    for call, scopes, name in reflective:
        accessor = call.func.id if isinstance(call.func, ast.Name) else ""
        receiver = call.args[0]
        if isinstance(receiver, ast.Name) and receiver.id in _OWNER_RECEIVERS:
            continue
        if _is_module_own_object(receiver, scopes, bindings_cache):
            if accessor == "setattr":
                owned_markers.add(name)
            continue
        candidates.append((call, accessor, name))
    violations: list[Violation] = _namespace_subscript_violations(file, subscripts)
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


def _may_reflect_privately(calls: list[ast.Call]) -> bool:
    """whether *calls* hold a call :func:`_private_name_argument` could name a private for.

    a necessary condition of every call :func:`_reflective_private_calls` keeps -- a reflective
    builtin by name, two or more arguments, the second a private string literal or a bare name --
    so a file without one has no reflective private call, and its scope walk is skipped.

    :param calls: every ``Call`` node of the module
    :ptype calls: list[ast.Call]
    :return: ``True`` when such a call is present
    :rtype: bool
    """
    return any(
        isinstance(call.func, ast.Name)
        and call.func.id in _REFLECTIVE_ACCESSORS
        and len(call.args) >= 2
        and (
            isinstance(call.args[1], ast.Name)
            or (
                isinstance(call.args[1], ast.Constant)
                and isinstance(call.args[1].value, str)
                and is_private_name(call.args[1].value)
            )
        )
        for call in calls
    )


def _reflective_private_calls(tree: ast.Module) -> list[tuple[ast.Call, tuple[_ScopeNode, ...], str]]:
    """every reflective call with a private-name literal, with its enclosing scopes outermost first.

    :param tree: the parsed module
    :ptype tree: ast.Module
    :return: ``(call, scopes, private name)`` in source order
    :rtype: list[tuple[ast.Call, tuple[_ScopeNode, ...], str]]
    """
    found: list[tuple[ast.Call, tuple[_ScopeNode, ...], str]] = []
    loops: dict[int, dict[str, list[str]]] = {}

    def _visit(node: ast.AST, scopes: tuple[_ScopeNode, ...]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, _NESTED_SCOPES):
                _visit(child, (*scopes, child))
                continue
            if isinstance(child, ast.Call):
                name = _private_name_argument(child, scopes, loops)
                if name:
                    found.append((child, scopes, name))
            _visit(child, scopes)

    _visit(tree, (tree,))
    return sorted(found, key=lambda pair: (pair[0].lineno, pair[0].col_offset))


def _private_name_argument(
    call: ast.Call, scopes: tuple[_ScopeNode, ...], loops: dict[int, dict[str, list[str]]]
) -> str:
    """the private attribute name a reflective builtin call names, or ``""`` when it names none.

    The name is a string literal, or a variable whose values the enclosing code spells as literals:
    a ``for`` over a literal list or tuple, or a ``pytest.mark.parametrize`` of the function it sits
    in. ``getattr(module, name)`` over ``["_A", "_B"]`` binds the privates as surely as the literal.

    :param call: a call node
    :ptype call: ast.Call
    :param scopes: the call's enclosing scopes, outermost first
    :ptype scopes: tuple[_ScopeNode, ...]
    :param loops: :func:`_loop_strings` per scope, filled as scopes are first asked about
    :ptype loops: dict[int, dict[str, list[str]]]
    :return: the private name, or the empty string for any other call
    :rtype: str
    """
    result = ""
    if isinstance(call.func, ast.Name) and call.func.id in _REFLECTIVE_ACCESSORS and len(call.args) >= 2:
        argument = call.args[1]
        if isinstance(argument, ast.Constant) and isinstance(argument.value, str) and is_private_name(argument.value):
            result = argument.value
        elif isinstance(argument, ast.Name):
            result = next((v for v in _literal_values(argument.id, scopes, loops) if is_private_name(v)), "")
    return result


def _literal_values(name: str, scopes: tuple[_ScopeNode, ...], loops: dict[int, dict[str, list[str]]]) -> list[str]:
    """the string literals a variable is given by the code around it: the ``for`` loops over literal
    sequences that bind it, and a ``parametrize`` decorator of an enclosing function that names it.

    :param name: the variable
    :ptype name: str
    :param scopes: the enclosing scopes, outermost first
    :ptype scopes: tuple[_ScopeNode, ...]
    :param loops: :func:`_loop_strings` per scope, keyed by node identity, so each scope is walked
        once per module rather than once per call inside it
    :ptype loops: dict[int, dict[str, list[str]]]
    :return: the literal strings, in source order
    :rtype: list[str]
    """
    values: list[str] = []
    for scope in scopes:
        key = id(scope)
        if key not in loops:
            loops[key] = _loop_strings(scope)
        values += loops[key].get(name, [])
        if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for decorator in scope.decorator_list:
                values += _parametrized_strings(decorator, name)
    return values


def _loop_strings(scope: _ScopeNode) -> dict[str, list[str]]:
    """the strings every ``for`` loop and comprehension under *scope* gives a plain-name target.

    :param scope: the scope node; nested scopes are walked too, as a variable's loops may sit in them
    :ptype scope: _ScopeNode
    :return: target name -> the literal strings its loops iterate, in :func:`ast.walk` order
    :rtype: dict[str, list[str]]
    """
    found: dict[str, list[str]] = {}
    for node in ast.walk(scope):
        if isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)) and isinstance(node.target, ast.Name):
            found.setdefault(node.target.id, []).extend(_strings_in(node.iter))
    return found


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


def _namespace_subscript_violations(file: Path, subscripts: list[ast.Subscript]) -> list[Violation]:
    """a private name read out of another object's namespace by subscript: ``vars(x)["_y"]`` or
    ``x.__dict__["_y"]`` -- the getattr of shape F, spelled as a dictionary lookup.

    :param file: the module's path
    :ptype file: Path
    :param subscripts: every ``Subscript`` node of the module, in :func:`ast.walk` order
    :ptype subscripts: list[ast.Subscript]
    :return: the violations, in source order
    :rtype: list[Violation]
    """
    violations: list[Violation] = []
    for node in subscripts:
        if not (
            isinstance(node.slice, ast.Constant)
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
            facts = _module_facts(file)
            if facts is None:
                continue
            for cls in facts.classes:
                entries = result.setdefault(cls.name, {})
                for name, line in cls.state:
                    entries.setdefault(name, (file, line))
    return result


def _class_body_state(cls: ast.ClassDef) -> list[tuple[str, int]]:
    """the private state ``cls``'s class body binds to anything but a function.

    the first part of a class's state; :func:`_class_facts` appends the second, the ``self._x`` /
    ``cls._x`` stores among its own nodes, in :func:`_own_nodes` order.

    :param cls: the class
    :ptype cls: ast.ClassDef
    :return: (name, line) pairs, in body order
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
    ``cls._x`` where ``_x`` is private state of any production ancestor (a base, its bases, and
    so on, followed through test-defined classes in between) the inheritance roots define
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
    production_bases = _collect_class_bases(inheritance_roots)
    scanned: list[tuple[_ClassFacts, Path]] = []
    for root in scan_roots:
        for file in iter_python_files(root):
            facts = _module_facts(file)
            if facts is not None:
                scanned.extend((cls, file) for cls in facts.classes)
    # a test class's ancestors run through test-defined classes too (a harness over a harness)
    test_bases = {cls.name: list(cls.generic_bases) for cls, _ in scanned}
    test_declares = {cls.name: cls.defines for cls, _ in scanned}
    violations: list[Violation] = []
    for cls, file in scanned:
        ancestry = _ancestors(cls, test_bases, production_bases)
        violations.extend(_base_state_reaches(cls, file, ancestry, base_state, test_declares, repo_root))
    return violations


def _collect_class_bases(src_roots: tuple[Path, ...]) -> dict[str, list[str]]:
    """``class_name`` -> its bases' last-segment names, over every class the roots define.

    :param src_roots: every src root the scanner should consider
    :ptype src_roots: tuple[Path, ...]
    :return: class -> base names (merged across same-named classes, as the other maps are)
    :rtype: dict[str, list[str]]
    """
    result: dict[str, list[str]] = {}
    for root in src_roots:
        for file in iter_python_files(root):
            facts = _module_facts(file)
            if facts is None:
                continue
            for cls in facts.classes:
                result.setdefault(cls.name, []).extend(cls.generic_bases)
    return result


def _ancestors(
    cls: _ClassFacts,
    test_bases: dict[str, list[str]],
    production_bases: dict[str, list[str]],
) -> tuple[list[str], list[str]]:
    """every ancestor of ``cls`` by name, nearest first: those the tests define, and production ones.

    a name the tests define is followed through the tests' classes; any other through the
    production classes. each name is visited once, so a cycle of same-named classes ends.

    :return: (test-defined ancestors, production ancestors)
    :rtype: tuple[list[str], list[str]]
    """
    test_side: list[str] = []
    production: list[str] = []
    seen = {cls.name}
    pending = [name for name in cls.generic_bases if name != cls.name]
    while pending:
        name = pending.pop(0)
        if name in seen:
            continue
        seen.add(name)
        if name in test_bases:
            test_side.append(name)
            pending.extend(test_bases[name])
        else:
            production.append(name)
            pending.extend(production_bases.get(name, []))
    return test_side, production


def _base_state_reaches(
    cls: _ClassFacts,
    file: Path,
    ancestry: tuple[list[str], list[str]],
    base_state: dict[str, dict[str, tuple[Path, int]]],
    test_declares: dict[str, frozenset[str]],
    repo_root: Path,
) -> list[Violation]:
    """the shape-I violations of one class, against the state every production ancestor keeps.

    :return: the violations
    :rtype: list[Violation]
    """
    test_side, production = ancestry
    inherited: dict[str, tuple[str, tuple[Path, int]]] = {}
    for base in production:
        for name, where in base_state.get(base, {}).items():
            inherited.setdefault(name, (base, where))
    if not inherited:
        return []
    own = cls.defines.union(*(test_declares.get(name, frozenset()) for name in test_side))
    found: list[Violation] = []
    # owner_reaches holds only private names, which every inherited state name is
    for attr, line in cls.owner_reaches:
        if attr not in inherited or attr in own:
            continue
        base, (defining_file, defining_line) = inherited[attr]
        try:
            rel_def: Path | str = defining_file.relative_to(repo_root)
        except ValueError:
            rel_def = defining_file
        found.append(
            Violation(
                category="underscore_access.I",
                file=file,
                line=line,
                symbol=attr,
                reason=(
                    f"class '{cls.name}' reaches '{attr}', private state of its base "
                    f"'{base}' ({rel_def}:{defining_line}); reach the base through its public "
                    f"surface, or a protected method it offers subclasses"
                ),
            )
        )
    return found


def _base_names_generic(cls: ast.ClassDef) -> list[str]:
    """every base's last-segment name, a generic one (``Base[T]``) included by its class.

    shape I follows ancestry through generic bases (``SchemaBackedCollection[Conversation]``),
    which :func:`_base_names_textual` skips for shape D.

    :param cls: class definition to inspect
    :ptype cls: ast.ClassDef
    :return: last-segment base names in source order
    :rtype: list[str]
    """
    names: list[str] = []
    for base in cls.bases:
        target = base.value if isinstance(base, ast.Subscript) else base
        if isinstance(target, ast.Name):
            names.append(target.id)
        elif isinstance(target, ast.Attribute):
            names.append(target.attr)
    return names
