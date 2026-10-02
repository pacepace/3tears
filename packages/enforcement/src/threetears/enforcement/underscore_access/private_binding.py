"""the private-binding gate: a private name bound by import or by string, in src, tests and scripts alike.

Owner ruling, 2026-10-01: a leading underscore is a stability contract -- nothing outside the
defining module (or, in ``src``, the defining package) binds to it, in ``src`` or in ``tests``. The
underscore walkers A-F, ruff's SLF001 and :mod:`.pragma_policy` held that line for attribute reads,
for ``__all__``, for subclass shadowing and for the reflective builtins, and two spellings still
passed every one of them:

- **shape G -- a private name or module bound by an import.** Shape A scans ``src`` only and only
  for first-party private NAMES, so a test importing ``_CANCEL_TIMEOUT_SECONDS`` from the module
  under test, a test importing ``_valid_auth`` from a sibling test module, and anything importing
  through a private module path (``from vendor._internal.query import Query``) all passed.
- **shape H -- a private name bound by a string.** ``monkeypatch.setattr(obj, "_name", ...)``,
  ``monkeypatch.setattr("pkg.mod._name", ...)``, ``patch("pkg.mod._name")``,
  ``patch.object(obj, "_name")``, ``mocker.patch(...)``, ``mocker.spy(obj, "_name")``,
  ``monkeypatch.delattr(...)``, ``patch.multiple(..., _name=...)``, ``patch.dict("pkg._REG")`` and
  ``importlib.import_module("pkg._mod")``. Shape F covers only the four builtins, and no attribute
  check can see a name passed as data.

**The rules, precisely.** "Private" is :func:`~threetears.enforcement.common.is_private_name`:
one leading underscore, not a dunder, not ``_`` and not a trailing-underscore keyword escape. A
binding names a private when any module-path segment or the bound name itself is private.

- **G.name** -- ``from M import _x``. Allowed only in a ``src`` module when ``M`` resolves to a
  module of the importer's own package (same ``src`` root, same top-level package), which is the
  package-level privacy shape A has always applied. Anywhere else -- every test, script and
  conftest -- it is a violation, including a private helper imported from ANOTHER test module:
  the test module that defines ``_valid_auth`` is its owner, and a sibling test is outside it.
- **G.module** -- an import whose module path has a private segment (``from pkg._internals import
  thing``, ``import pkg._internals``, ``from pkg import _internals``). Allowed when the module
  resolves inside the importer's own boundary: for a ``src`` module, its own package; for anything
  else, its own tests tree (the nearest ``tests``/``test`` ancestor directory, or the file's own
  directory when it has none) and never a ``src`` module. So a ``tests/support/_pod_auth.py``
  imported by tests in the same tree is allowed -- the underscore on a test-support module marks it
  as support pytest does not collect, and the tests package is its owner -- while a test importing
  through the private module of the code under test is not: the contract is about the stability
  boundary of the code under test, and a test is outside that boundary.
- **H.attribute** -- a binder call naming a private attribute of an object by string. Allowed only
  when the object is ``self``/``cls`` or a name the same file defines by ``def``/``class`` and binds
  no other way (a test's own fake), mirroring shape F.
- **H.path** -- a binder call naming a dotted target with a private segment. The longest prefix that
  resolves to a module of this repo splits it into module and attribute. A private module segment
  follows the G.module boundary; a private attribute segment is allowed only when the module is the
  calling file itself (or, in ``src``, a module of the caller's own package). A target that does
  not resolve to this repo is a library's, and patching a library's private by string is always a
  violation.

**Third-party confinement modules remain the only sanctioned private access.** A recorded
confinement module is a ``src`` module that carries a per-file SLF001 ignore and has entries in
the exemptions ledger (:mod:`.pragma_policy`). Its G bindings of a THIRD-PARTY private -- the
``from claude_agent_sdk._internal.query import Query`` that its ledgered attribute reads depend on
-- are sanctioned by that record, since the module is the library's one recorded owner. Its
bindings of a first-party private are not, and nothing sanctions an H binding anywhere: patching by
string is not how a confinement module reaches a library, and a test that patches a library's
private (``patch.object(ChatAnthropic, "_astream")``) is a violation whichever library it is. A
test that needs that seam calls a public function of the confinement module, or drives the library
through its own public surface.

**What is scanned** is every python file that is the repo's own, exactly what
:func:`~threetears.enforcement.underscore_access.pragma_policy.scanned_python_files` returns. Both
inputs can come back empty for a reason that is not "clean", so :class:`PrivateBindingScan` carries
the counts a consumer's floor asserts on, and :func:`undetected_planted_controls` proves the
recognisers still fire in the consumer's own environment.

Shape G is a superset of shape A for imports: a first-party private name imported across packages
in ``src`` is reported by both. Shape A stays because its consumers already run it.

**Enabling it in a consumer repo** is one thin test module under ``tests/enforcement/``::

    from pathlib import Path

    import pytest

    from threetears.enforcement.underscore_access import (
        PrivateBindingScan,
        private_binding_findings,
        scan_private_bindings,
        undetected_planted_controls,
    )

    _REPO_ROOT = Path(__file__).resolve().parents[2]
    _LEDGER = _REPO_ROOT / "tests" / "enforcement" / "_underscore_exemptions.txt"


    @pytest.fixture(scope="module")
    def scan() -> PrivateBindingScan:
        return scan_private_bindings(_REPO_ROOT, _LEDGER)


    def test_the_scan_covers_this_repo(scan: PrivateBindingScan) -> None:
        assert scan.files_scanned > 50  # floors sized to the repo
        assert scan.imports_examined > 500
        assert scan.binding_calls_examined > 20


    def test_every_planted_shape_is_detected(tmp_path: Path) -> None:
        assert not undetected_planted_controls(tmp_path)


    def test_no_private_binding_outside_its_owner(scan: PrivateBindingScan) -> None:
        findings = private_binding_findings(scan, _REPO_ROOT)
        assert not findings, "\\n".join(findings)
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path

from threetears.enforcement.common import Violation, is_private_name, parse_python_file
from threetears.enforcement.common.repo_layout import find_local_src_roots
from threetears.enforcement.underscore_access.ledger import ledger_paths
from threetears.enforcement.underscore_access.pragma_policy import (
    TEST_DIRECTORIES,
    is_src_module,
    scanned_python_files,
    slf001_ignored_files,
)

__all__ = [
    "PRIVATE_BINDING_CATEGORIES",
    "SHAPE_G_MODULE",
    "SHAPE_G_NAME",
    "SHAPE_H_ATTRIBUTE",
    "SHAPE_H_PATH",
    "PrivateBindingScan",
    "confinement_modules",
    "private_binding_findings",
    "scan_private_bindings",
    "undetected_planted_controls",
]

#: a private name bound by ``from M import _x``.
SHAPE_G_NAME = "underscore_access.G.name"
#: an import through a module path with a private segment.
SHAPE_G_MODULE = "underscore_access.G.module"
#: a binder call naming an object's private attribute by string.
SHAPE_H_ATTRIBUTE = "underscore_access.H.attribute"
#: a binder call naming a dotted target with a private segment.
SHAPE_H_PATH = "underscore_access.H.path"

#: every category this gate reports, in report order.
PRIVATE_BINDING_CATEGORIES: tuple[str, ...] = (SHAPE_G_NAME, SHAPE_G_MODULE, SHAPE_H_ATTRIBUTE, SHAPE_H_PATH)

#: a string that can only be a dotted python target: two or more identifiers joined by dots. a URL
#: (``client.patch("/api/v1/x")``) or a bare word never matches, which is what keeps an HTTP client's
#: ``patch`` method out of the patch recogniser.
_DOTTED_TARGET = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+$")

#: a string that can be a module name, for ``import_module``: a single identifier is a module too.
_MODULE_TARGET = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*$")

#: the modules ``patch`` is imported from.
_MOCK_MODULES = frozenset({"unittest.mock", "mock"})

#: ``patch.<helper>`` forms.
_PATCH_HELPERS = frozenset({"object", "multiple", "dict"})

#: attribute calls that bind ``(object, "name")`` or ``("dotted.path")``: pytest's ``monkeypatch``
#: (any receiver: the fixture, ``MonkeyPatch.context()``'s handle, an attribute holding one),
#: ``object.__setattr__`` spelled through the type, and pytest-mock's ``spy``. the builtins of the
#: same names are shape F's.
_ATTRIBUTE_BINDERS = frozenset({"setattr", "delattr", "__setattr__", "__delattr__", "spy"})

#: the binders through which a class may set its own declared private on an instance it built.
_OWNER_DUNDER_BINDERS = frozenset({"__setattr__", "__delattr__"})

#: calls that bind a module by its dotted name.
_MODULE_IMPORTERS = frozenset({"import_module", "__import__"})

#: the receivers that ARE the owner of a private name, the same test SLF001 applies.
_OWNER_RECEIVERS = frozenset({"self", "cls"})

_FIX_G_NAME = (
    "a private name is its module's implementation detail. Reach the behaviour through the front door: "
    "a public function, constructor argument or accessor of the code under test. A helper shared between "
    "test modules belongs in a support module under a public name"
)
_FIX_G_MODULE = (
    "a private module is its package's implementation detail. Import the public module that re-exports "
    "the name, or promote the module. Only a recorded third-party confinement module (a src module with a "
    "per-file SLF001 ignore and ledger entries) may import a library's private module"
)
_FIX_H = (
    "a string names the private where SLF001 cannot see it. Inject the dependency through a public "
    "constructor argument with a production default, or assert the behaviour at the front door; a "
    "library's private is reached only through its one recorded confinement module"
)


@dataclass(frozen=True)
class PrivateBindingScan:
    """the result of one scan: what was read, and what was found.

    The counts are the non-vacuity inputs: a scan that read no file, or whose recognisers matched no
    import and no binder call, reports exactly what a compliant repo reports.

    :ivar files_scanned: python files parsed
    :ivar imports_examined: ``import`` and ``from ... import`` statements read
    :ivar binding_calls_examined: patch, monkeypatch, spy and import-by-name calls recognised,
        whatever they named
    :ivar violations: every private binding outside its owner, in file and line order
    """

    files_scanned: int
    imports_examined: int
    binding_calls_examined: int
    violations: tuple[Violation, ...]


@dataclass
class _FileContext:
    """everything the rules need to know about the file being scanned.

    :ivar repo_root: the repo's root
    :ivar path: the file
    :ivar is_src: whether the file is production source (:func:`.pragma_policy.is_src_module`)
    :ivar src_root: the file's own ``src`` directory, for a src file
    :ivar home: the file's tests tree, or its own directory, for a file that is not src
    :ivar search_roots: directories an absolute module name resolves against, in order
    :ivar confinement: whether the file is a recorded third-party confinement module
    :ivar patch_names: local names bound to ``unittest.mock.patch`` / ``mock.patch``
    :ivar own_objects: names the file defines by ``def``/``class`` and binds no other way
    :ivar violations: accumulated findings
    :ivar imports_examined: import statements read
    :ivar binding_calls_examined: binder calls recognised
    """

    repo_root: Path
    path: Path
    is_src: bool
    src_root: Path | None
    home: Path | None
    search_roots: tuple[Path, ...]
    confinement: bool
    patch_names: frozenset[str] = frozenset()
    own_objects: frozenset[str] = frozenset()
    violations: list[Violation] = field(default_factory=list)
    imports_examined: int = 0
    binding_calls_examined: int = 0


def confinement_modules(repo_root: Path, exemptions_path: Path | None) -> frozenset[str]:
    """the recorded third-party confinement modules: src, SLF001-ignored, and in the ledger.

    :param repo_root: the repo's root
    :ptype repo_root: Path
    :param exemptions_path: the underscore-access exemptions ledger, or ``None`` for none
    :ptype exemptions_path: Path | None
    :return: repo-relative posix paths
    :rtype: frozenset[str]
    """
    confined: frozenset[str] = frozenset()
    # no ledger records nothing, so nothing is sanctioned and the ruff configs need not be read
    if exemptions_path is not None and exemptions_path.is_file():
        recorded = set(ledger_paths(exemptions_path))
        ignored = {path for _config, _key, path in slf001_ignored_files(repo_root)}
        confined = frozenset(path for path in ignored & recorded if is_src_module(path))
    return confined


def scan_private_bindings(repo_root: Path, exemptions_path: Path | None) -> PrivateBindingScan:
    """scan every python file of the repo for shapes G and H.

    :param repo_root: the repo's root
    :ptype repo_root: Path
    :param exemptions_path: the underscore-access exemptions ledger, which (with the ruff config)
        names the confinement modules; ``None`` when the repo has none
    :ptype exemptions_path: Path | None
    :return: the counts read and the violations found
    :rtype: PrivateBindingScan
    """
    repo_root = repo_root.resolve()
    src_roots = find_local_src_roots(repo_root)
    confined = confinement_modules(repo_root, exemptions_path)
    files = 0
    imports = 0
    calls = 0
    violations: list[Violation] = []
    for path in scanned_python_files(repo_root):
        tree = parse_python_file(path)
        if tree is None:
            continue
        files += 1
        context = _context_for(path.resolve(), repo_root, src_roots, confined, tree)
        _scan_tree(tree, context)
        imports += context.imports_examined
        calls += context.binding_calls_examined
        violations.extend(context.violations)
    ordered = sorted(violations, key=lambda v: (v.file.as_posix(), v.line, v.category, v.symbol))
    return PrivateBindingScan(
        files_scanned=files,
        imports_examined=imports,
        binding_calls_examined=calls,
        violations=tuple(ordered),
    )


def private_binding_findings(scan: PrivateBindingScan, repo_root: Path) -> list[str]:
    """one report line per violation, each naming the place, the binding and the fix.

    :param scan: a scan's result
    :ptype scan: PrivateBindingScan
    :param repo_root: the repo's root, for relative paths
    :ptype repo_root: Path
    :return: findings; empty when the repo complies
    :rtype: list[str]
    """
    return [violation.format(repo_root) for violation in scan.violations]


#: the planted repo: one instance of every category, and the category each line must report.
_PLANTED_FILES: dict[str, str] = {
    "pyproject.toml": "[project]\nname = 'planted'\n",
    "src/pkg/__init__.py": "",
    "src/pkg/_internals.py": "thing = 1\n",
    "src/pkg/mod.py": "_hidden = 1\n",
    "tests/test_planted.py": (
        "from pkg._internals import thing\n"
        "from pkg.mod import _hidden\n"
        "\n"
        "\n"
        "def test_planted(monkeypatch, obj):\n"
        '    monkeypatch.setattr(obj, "_state", thing)\n'
        '    monkeypatch.setattr("pkg.mod._hidden", _hidden)\n'
    ),
}


def undetected_planted_controls(workdir: Path) -> list[str]:
    """plant one instance of every category in a scratch repo and name any the scan misses.

    The positive control for a consumer's gate: an empty verdict over the real repo is only
    evidence when the same installed walker, in the same environment, still reports a planted
    instance of each shape.

    :param workdir: an empty scratch directory (pytest's ``tmp_path``)
    :ptype workdir: Path
    :return: the categories the scan did not report; empty when every one was detected
    :rtype: list[str]
    """
    repo = workdir / "planted-private-binding-repo"
    for relative, text in _PLANTED_FILES.items():
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    reported = {violation.category for violation in scan_private_bindings(repo, None).violations}
    return [category for category in PRIVATE_BINDING_CATEGORIES if category not in reported]


def _context_for(
    path: Path,
    repo_root: Path,
    src_roots: tuple[Path, ...],
    confined: frozenset[str],
    tree: ast.Module,
) -> _FileContext:
    """build the per-file context: where the file sits, what it may reach, what it defines.

    :param path: the file, resolved
    :ptype path: Path
    :param repo_root: the repo's root, resolved
    :ptype repo_root: Path
    :param src_roots: this repo's own ``src`` trees
    :ptype src_roots: tuple[Path, ...]
    :param confined: repo-relative paths of the recorded confinement modules
    :ptype confined: frozenset[str]
    :param tree: the file's parsed module
    :ptype tree: ast.Module
    :return: the context
    :rtype: _FileContext
    """
    relative = path.relative_to(repo_root).as_posix()
    is_src = is_src_module(relative)
    src_root: Path | None = None
    home: Path | None = None
    leading: tuple[Path, ...] = ()
    if is_src:
        src_root = _nearest_ancestor_named(path, repo_root, frozenset({"src"}))
        leading = (src_root,) if src_root is not None else ()
    else:
        tree_root = _nearest_ancestor_named(path, repo_root, TEST_DIRECTORIES) or path.parent
        home = tree_root
        leading = (path.parent, tree_root, tree_root.parent)
    search_roots = tuple(dict.fromkeys((*leading, *src_roots, repo_root)))
    return _FileContext(
        repo_root=repo_root,
        path=path,
        is_src=is_src,
        src_root=src_root,
        home=home,
        search_roots=search_roots,
        confinement=relative in confined,
        patch_names=_patch_names(tree),
        own_objects=_own_objects(tree),
    )


def _nearest_ancestor_named(path: Path, repo_root: Path, names: frozenset[str]) -> Path | None:
    """the nearest directory above *path*, inside *repo_root*, whose name is one of *names*.

    :param path: a file under *repo_root*
    :ptype path: Path
    :param repo_root: the repo's root
    :ptype repo_root: Path
    :param names: directory names to look for
    :ptype names: frozenset[str]
    :return: the directory, or ``None`` when no ancestor below the root carries one of the names
    :rtype: Path | None
    """
    found: Path | None = None
    for ancestor in path.parents:
        if ancestor == repo_root or repo_root not in ancestor.parents:
            break
        if ancestor.name in names:
            found = ancestor
            break
    return found


def _patch_names(tree: ast.Module) -> frozenset[str]:
    """local names bound to ``patch`` by an import from ``unittest.mock`` or ``mock``.

    A bare ``patch(...)`` is a patch only when the file imported it as one; any other function of
    that name is somebody else's.

    :param tree: the parsed module
    :ptype tree: ast.Module
    :return: the local names
    :rtype: frozenset[str]
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module in _MOCK_MODULES:
            for alias in node.names:
                if alias.name == "patch":
                    names.add(alias.asname or alias.name)
    return frozenset(names)


def _own_objects(tree: ast.Module) -> frozenset[str]:
    """names the file defines by ``def``/``class`` and binds in no other way anywhere in it.

    A name that is both defined and assigned, imported or taken as a parameter cannot be trusted to
    be the definition at the call, so it is not the file's own.

    :param tree: the parsed module
    :ptype tree: ast.Module
    :return: the names
    :rtype: frozenset[str]
    """
    defined: set[str] = set()
    other: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            defined.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store | ast.Del):
            other.add(node.id)
        elif isinstance(node, ast.Import | ast.ImportFrom):
            other.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.arg):
            other.add(node.arg)
    return frozenset(defined - other)


def _scan_tree(tree: ast.Module, context: _FileContext) -> None:
    """apply every rule to every node of one file, accumulating into *context*.

    :param tree: the parsed module
    :ptype tree: ast.Module
    :param context: the file's context
    :ptype context: _FileContext
    """
    _visit(tree, context, frozenset())


def _visit(node: ast.AST, context: _FileContext, class_privates: frozenset[str]) -> None:
    """walk *node*'s children, tracking the privates the enclosing classes declare.

    :param node: the node whose children to walk
    :ptype node: ast.AST
    :param context: the file's context
    :ptype context: _FileContext
    :param class_privates: private names declared by every class enclosing *node*
    :ptype class_privates: frozenset[str]
    """
    for child in ast.iter_child_nodes(node):
        inner = class_privates | _declared_privates(child) if isinstance(child, ast.ClassDef) else class_privates
        if isinstance(child, ast.Import):
            context.imports_examined += 1
            _check_import(child, context)
        elif isinstance(child, ast.ImportFrom):
            context.imports_examined += 1
            _check_import_from(child, context)
        elif isinstance(child, ast.Call):
            _check_call(child, context, class_privates)
        _visit(child, context, inner)


def _declared_privates(cls: ast.ClassDef) -> frozenset[str]:
    """the private names a class declares: its owner's names, which its own code may bind by string.

    Class-body assignments and annotations, methods, ``__slots__`` entries, and ``self._x`` /
    ``cls._x`` stores anywhere in the class. A class that builds an instance without ``__init__``
    and sets its slots through ``object.__setattr__(instance, "_x", v)`` is the owner binding its
    own private, the same access SLF001 allows it as ``self._x``.

    :param cls: the class
    :ptype cls: ast.ClassDef
    :return: the names
    :rtype: frozenset[str]
    """
    names: set[str] = set()
    for item in cls.body:
        if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(item.name)
        elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            names.add(item.target.id)
        elif isinstance(item, ast.Assign):
            for target in item.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
                    if target.id == "__slots__":
                        names.update(_string_entries(item.value))
    for node in ast.walk(cls):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.ctx, ast.Store)
            and isinstance(node.value, ast.Name)
            and node.value.id in _OWNER_RECEIVERS
        ):
            names.add(node.attr)
    return frozenset(name for name in names if is_private_name(name))


def _string_entries(value: ast.expr) -> set[str]:
    """the string constants of a literal tuple, list or set (or a lone string).

    :param value: the expression
    :ptype value: ast.expr
    :return: the strings
    :rtype: set[str]
    """
    found: set[str] = set()
    if isinstance(value, ast.Tuple | ast.List | ast.Set):
        found = {
            element.value
            for element in value.elts
            if isinstance(element, ast.Constant) and isinstance(element.value, str)
        }
    elif isinstance(value, ast.Constant) and isinstance(value.value, str):
        found = {value.value}
    return found


def _report(context: _FileContext, category: str, line: int, symbol: str, reason: str) -> None:
    """record one violation.

    :param context: the file's context
    :ptype context: _FileContext
    :param category: one of :data:`PRIVATE_BINDING_CATEGORIES`
    :ptype category: str
    :param line: 1-based line of the binding
    :ptype line: int
    :param symbol: the private segment or name bound
    :ptype symbol: str
    :param reason: what was bound and how to fix it
    :ptype reason: str
    """
    context.violations.append(Violation(category=category, file=context.path, line=line, symbol=symbol, reason=reason))


def _first_private(segments: list[str]) -> str:
    """the first private segment, or ``""`` when none is private.

    :param segments: dotted-name segments
    :ptype segments: list[str]
    :return: the segment or the empty string
    :rtype: str
    """
    return next((segment for segment in segments if is_private_name(segment)), "")


def _module_file(base: Path, segments: list[str]) -> Path | None:
    """the file defining the module *segments* under *base*: ``a/b.py`` or ``a/b/__init__.py``.

    :param base: a directory modules resolve against
    :ptype base: Path
    :param segments: the module's dotted segments; empty names *base*'s own ``__init__.py``
    :ptype segments: list[str]
    :return: the file, or ``None`` when *base* holds no such module
    :rtype: Path | None
    """
    result: Path | None = None
    package_init = base.joinpath(*segments, "__init__.py")
    if segments and base.joinpath(*segments).with_suffix(".py").is_file():
        result = base.joinpath(*segments).with_suffix(".py")
    elif package_init.is_file():
        result = package_init
    return result


def _resolve_absolute(dotted: str, context: _FileContext) -> Path | None:
    """the file of this repo an absolute module name resolves to, or ``None`` for a library's.

    A name whose top-level package is this repo's resolves first-party even when the submodule is
    not on disk, as a missing file under a first-party package: the binding is still to this repo's
    private, and treating it as a library's would hand it the confinement module's sanction.

    :param dotted: the module name
    :ptype dotted: str
    :param context: the file's context
    :ptype context: _FileContext
    :return: the file, or ``None``
    :rtype: Path | None
    """
    segments = dotted.split(".")
    result: Path | None = None
    for root in context.search_roots:
        found = _module_file(root, segments)
        if found is not None:
            result = found
            break
        if len(segments) > 1 and _module_file(root, segments[:1]) is not None:
            result = root.joinpath(*segments).with_suffix(".py")
            break
    return result


def _resolve_relative(level: int, module: str | None, context: _FileContext) -> Path | None:
    """the file a relative import resolves to.

    :param level: the number of leading dots
    :ptype level: int
    :param module: the dotted part after the dots, if any
    :ptype module: str | None
    :param context: the file's context
    :ptype context: _FileContext
    :return: the file (possibly not on disk, for a missing module), or ``None`` above the repo
    :rtype: Path | None
    """
    base = context.path.parent
    for _ in range(level - 1):
        base = base.parent
    segments = module.split(".") if module else []
    result: Path | None = None
    if context.repo_root == base or context.repo_root in base.parents:
        result = _module_file(base, segments) or (
            base.joinpath(*segments).with_suffix(".py") if segments else base / "__init__.py"
        )
    return result


def _submodule(package_file: Path | None, name: str) -> Path | None:
    """the submodule *name* of the package whose ``__init__.py`` is *package_file*, when it exists.

    :param package_file: the resolved module of a ``from M import name``
    :ptype package_file: Path | None
    :param name: the imported name
    :ptype name: str
    :return: the submodule's file, or ``None`` when *name* is not a submodule on disk
    :rtype: Path | None
    """
    result: Path | None = None
    if package_file is not None and package_file.name == "__init__.py":
        result = _module_file(package_file.parent, [name])
    return result


def _package_of(path: Path) -> tuple[Path, str] | None:
    """a src file's package identity: its ``src`` directory and its top-level package.

    :param path: a file
    :ptype path: Path
    :return: the identity, or ``None`` for a file under no ``src`` directory
    :rtype: tuple[Path, str] | None
    """
    result: tuple[Path, str] | None = None
    for ancestor in path.parents:
        if ancestor.name == "src":
            relative = path.relative_to(ancestor).parts
            result = (ancestor, relative[0]) if len(relative) > 1 else None
            break
    return result


def _inside_boundary(target: Path, context: _FileContext) -> bool:
    """whether *target* lies inside the binding file's own boundary.

    For a src file the boundary is its package; for anything else it is its tests tree (or its own
    directory), never reaching a src module.

    :param target: the resolved module file
    :ptype target: Path
    :param context: the file's context
    :ptype context: _FileContext
    :return: whether the private is the binding file's own to bind
    :rtype: bool
    """
    home = context.home
    if context.is_src:
        own = _package_of(context.path)
        inside = own is not None and _package_of(target) == own
    else:
        inside = (
            home is not None
            and context.repo_root in target.parents
            and home in target.parents
            and not is_src_module(target.relative_to(context.repo_root).as_posix())
        )
    return inside


def _module_allowed(target: Path | None, context: _FileContext) -> bool:
    """whether binding through a private module segment of *target* is allowed.

    :param target: the resolved module, or ``None`` for a library's
    :ptype target: Path | None
    :param context: the file's context
    :ptype context: _FileContext
    :return: whether it is allowed
    :rtype: bool
    """
    if target is None:
        return context.confinement
    return _inside_boundary(target, context)


def _name_allowed(target: Path | None, context: _FileContext) -> bool:
    """whether importing a private NAME from *target* is allowed.

    :param target: the resolved module, or ``None`` for a library's
    :ptype target: Path | None
    :param context: the file's context
    :ptype context: _FileContext
    :return: whether it is allowed
    :rtype: bool
    """
    if target is None:
        return context.confinement
    return context.is_src and _inside_boundary(target, context)


def _check_import(node: ast.Import, context: _FileContext) -> None:
    """shape G.module for ``import a._b.c``.

    :param node: the statement
    :ptype node: ast.Import
    :param context: the file's context
    :ptype context: _FileContext
    """
    for alias in node.names:
        private = _first_private(alias.name.split("."))
        if private and not _module_allowed(_resolve_absolute(alias.name, context), context):
            _report(
                context, SHAPE_G_MODULE, node.lineno, private, f"imports private module '{alias.name}'; {_FIX_G_MODULE}"
            )


def _check_import_from(node: ast.ImportFrom, context: _FileContext) -> None:
    """shapes G.module and G.name for ``from M import n``.

    :param node: the statement
    :ptype node: ast.ImportFrom
    :param context: the file's context
    :ptype context: _FileContext
    """
    module_text = node.module or ""
    label = "." * node.level + module_text
    if node.level:
        target = _resolve_relative(node.level, node.module, context)
    else:
        target = _resolve_absolute(module_text, context)
    private_segment = _first_private(module_text.split(".")) if module_text else ""
    if private_segment and not _module_allowed(target, context):
        _report(
            context,
            SHAPE_G_MODULE,
            node.lineno,
            private_segment,
            f"imports from private module '{label}'; {_FIX_G_MODULE}",
        )
    for alias in node.names:
        if not is_private_name(alias.name):
            continue
        submodule = _submodule(target, alias.name)
        if submodule is not None:
            if not _module_allowed(submodule, context):
                reason = f"imports private module '{alias.name}' from '{label}'; {_FIX_G_MODULE}"
                _report(context, SHAPE_G_MODULE, node.lineno, alias.name, reason)
        elif not _name_allowed(target, context):
            reason = f"imports private name '{alias.name}' from '{label}'; {_FIX_G_NAME}"
            _report(context, SHAPE_G_NAME, node.lineno, alias.name, reason)


def _argument(call: ast.Call, index: int, keyword: str) -> ast.expr | None:
    """the call's argument at *index*, or the keyword argument *keyword*.

    :param call: the call
    :ptype call: ast.Call
    :param index: positional index
    :ptype index: int
    :param keyword: keyword name
    :ptype keyword: str
    :return: the argument expression, or ``None`` when absent
    :rtype: ast.expr | None
    """
    result: ast.expr | None = call.args[index] if len(call.args) > index else None
    if result is None:
        result = next((kw.value for kw in call.keywords if kw.arg == keyword), None)
    return result


def _string(expr: ast.expr | None) -> str | None:
    """the value of a string constant, else ``None``.

    :param expr: an expression
    :ptype expr: ast.expr | None
    :return: the string, or ``None``
    :rtype: str | None
    """
    result: str | None = None
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        result = expr.value
    return result


def _is_patch_reference(expr: ast.expr, context: _FileContext) -> bool:
    """whether *expr* is ``patch``: an imported mock ``patch`` or any ``<x>.patch``.

    :param expr: the callee (or the value of ``patch.<helper>``)
    :ptype expr: ast.expr
    :param context: the file's context
    :ptype context: _FileContext
    :return: whether it is a patch reference
    :rtype: bool
    """
    return (isinstance(expr, ast.Name) and expr.id in context.patch_names) or (
        isinstance(expr, ast.Attribute) and expr.attr == "patch"
    )


def _check_call(call: ast.Call, context: _FileContext, class_privates: frozenset[str]) -> None:
    """shape H: recognise a binder call and check what it names.

    :param call: the call
    :ptype call: ast.Call
    :param context: the file's context
    :ptype context: _FileContext
    :param class_privates: private names declared by the classes enclosing the call
    :ptype class_privates: frozenset[str]
    """
    func = call.func
    if _is_patch_reference(func, context):
        target = _string(_argument(call, 0, "target"))
        if target is not None and _DOTTED_TARGET.match(target):
            context.binding_calls_examined += 1
            _check_dotted(target, call.lineno, f"patch('{target}')", context)
    elif isinstance(func, ast.Attribute) and func.attr in _PATCH_HELPERS and _is_patch_reference(func.value, context):
        context.binding_calls_examined += 1
        _check_patch_helper(call, func.attr, context)
    elif isinstance(func, ast.Attribute) and func.attr in _ATTRIBUTE_BINDERS:
        _check_attribute_binder(call, func.attr, context, class_privates)
    elif (isinstance(func, ast.Attribute) and func.attr in _MODULE_IMPORTERS) or (
        isinstance(func, ast.Name) and func.id in _MODULE_IMPORTERS
    ):
        module = _string(_argument(call, 0, "name"))
        if module is not None and _MODULE_TARGET.match(module):
            context.binding_calls_examined += 1
            _check_dotted(module, call.lineno, f"import_module('{module}')", context)


def _check_patch_helper(call: ast.Call, helper: str, context: _FileContext) -> None:
    """``patch.object``, ``patch.multiple`` and ``patch.dict``.

    :param call: the call
    :ptype call: ast.Call
    :param helper: ``object``, ``multiple`` or ``dict``
    :ptype helper: str
    :param context: the file's context
    :ptype context: _FileContext
    """
    if helper == "object":
        receiver = _argument(call, 0, "target")
        name = _string(_argument(call, 1, "attribute"))
        if receiver is not None and name is not None:
            _check_attribute(receiver, name, call.lineno, "patch.object", context)
        return
    first = _argument(call, 0, "in_dict" if helper == "dict" else "target")
    dotted = _string(first)
    if dotted is not None and _DOTTED_TARGET.match(dotted):
        _check_dotted(dotted, call.lineno, f"patch.{helper}('{dotted}')", context)
    if helper != "multiple":
        return
    for keyword in call.keywords:
        if keyword.arg is None or not is_private_name(keyword.arg):
            continue
        if dotted is not None and _DOTTED_TARGET.match(dotted):
            _check_dotted(
                f"{dotted}.{keyword.arg}", call.lineno, f"patch.multiple('{dotted}', {keyword.arg}=)", context
            )
        elif first is not None and dotted is None:
            _check_attribute(first, keyword.arg, call.lineno, "patch.multiple", context)


def _check_attribute_binder(call: ast.Call, binder: str, context: _FileContext, class_privates: frozenset[str]) -> None:
    """``monkeypatch.setattr``/``delattr`` (object or dotted form), ``__setattr__``, ``spy``.

    ``object.__setattr__(instance, "_x", v)`` inside a class that declares ``_x`` is that class
    setting its own slot on an instance it built -- the frozen-slots constructor -- and is the
    owner's access. Only the dunder spelling earns that: a ``monkeypatch.setattr`` or ``spy`` in a
    test class that happens to declare the same name is still binding somebody else's private.

    :param call: the call
    :ptype call: ast.Call
    :param binder: the method name
    :ptype binder: str
    :param context: the file's context
    :ptype context: _FileContext
    :param class_privates: private names declared by the classes enclosing the call
    :ptype class_privates: frozenset[str]
    """
    first = _argument(call, 0, "target")
    dotted = _string(first)
    if dotted is not None:
        if _DOTTED_TARGET.match(dotted):
            context.binding_calls_examined += 1
            _check_dotted(dotted, call.lineno, f"{binder}('{dotted}')", context)
        return
    name = _string(_argument(call, 1, "name"))
    if first is None or name is None:
        return
    context.binding_calls_examined += 1
    if binder in _OWNER_DUNDER_BINDERS and name in class_privates:
        return
    _check_attribute(first, name, call.lineno, binder, context)


def _check_attribute(receiver: ast.expr, name: str, line: int, spelling: str, context: _FileContext) -> None:
    """shape H.attribute: a private attribute named by string on *receiver*.

    :param receiver: the object expression
    :ptype receiver: ast.expr
    :param name: the attribute name
    :ptype name: str
    :param line: the call's line
    :ptype line: int
    :param spelling: the binder, for the report
    :ptype spelling: str
    :param context: the file's context
    :ptype context: _FileContext
    """
    if not is_private_name(name):
        return
    owned = isinstance(receiver, ast.Name) and (receiver.id in _OWNER_RECEIVERS or receiver.id in context.own_objects)
    if not owned:
        reason = f"{spelling} names private attribute '{name}' of {ast.unparse(receiver)} by string; {_FIX_H}"
        _report(context, SHAPE_H_ATTRIBUTE, line, name, reason)


def _check_dotted(dotted: str, line: int, spelling: str, context: _FileContext) -> None:
    """shape H.path: a dotted target with a private segment.

    :param dotted: the target
    :ptype dotted: str
    :param line: the call's line
    :ptype line: int
    :param spelling: the binder and target, for the report
    :ptype spelling: str
    :param context: the file's context
    :ptype context: _FileContext
    """
    segments = dotted.split(".")
    private = _first_private(segments)
    if not private:
        return
    target: Path | None = None
    split = 0
    for end in range(len(segments), 0, -1):
        found = _resolve_absolute(".".join(segments[:end]), context)
        if found is not None and found.is_file():
            target, split = found, end
            break
    # a target with no module of this repo on its path is a library's: patching a library's private
    # by string is never sanctioned, so it stays disallowed.
    allowed = False
    if target is not None:
        module_private = _first_private(segments[:split])
        attribute_private = _first_private(segments[split:])
        module_ok = not module_private or _inside_boundary(target, context)
        attribute_ok = (
            not attribute_private or target == context.path or (context.is_src and _inside_boundary(target, context))
        )
        allowed = module_ok and attribute_ok
        private = module_private if not module_ok else attribute_private
    if not allowed:
        _report(context, SHAPE_H_PATH, line, private, f"{spelling} binds private '{private}' by string; {_FIX_H}")
