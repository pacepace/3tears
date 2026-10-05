"""Structural gate: the engine's packages import only what the allowed-dependency matrix permits.

``threetears.evals`` is eleven physical subpackages, and one allowed-dependency matrix says which may
import which:

============  =========  =====  ========  ===  =======  =======  =====  ====  ============================================
from          contracts  run    analysis  gen  storage  testing  quick  vega  third-party
============  =========  =====  ========  ===  =======  =======  =====  ====  ============================================
contracts     yes        --     --        --   --       --       --     --    pydantic, threetears.observe
run           yes        yes    --        --   --       --       --     --    pydantic, threetears.observe
analysis      yes        --     yes       --   --       --       --     --    pydantic, threetears.observe
gen           yes        --     --        yes  --       --       --     --    pydantic, threetears.observe
storage       yes        --     --        --   yes      --       --     --    (none)
testing       yes        --     --        --   --       yes      --     --    (none)
quick         yes        yes    yes       --   yes      --       yes    --    pydantic
vega          yes        --     yes       --   --       --       --     yes   threetears.observe; ``vl_convert`` only in vega.render
============  =========  =====  ========  ===  =======  =======  =====  ====  ============================================

``vega`` is the optional Vega-Lite chart renderer (the ``[vega]`` extra): an adapter over the chart
intent ``analysis`` decides, so it reaches ``analysis`` and nothing reaches it. Its column is empty
but its own row, which is what keeps the core free of a charting library: no core package may import
the renderer, and the rasteriser it takes (``vl_convert``) is the extra's dependency, never the core's.

Three more sit above the engine, as the surfaces an agent drives it through, and the table does not
name them:

* ``ops`` -- typed operations and the job contract -- may import contracts, run, analysis and itself;
  pydantic and threetears.observe.
* ``actions`` -- the action catalogue -- may import contracts, run, ops and itself; pydantic and
  threetears.observe. It reaches analysis only through ``ops``.
* ``transports`` -- one adapter per server -- may import contracts, ops, actions and itself; pydantic, and
  each adapter its own server alone (``transports.fastmcp``: ``fastmcp``, the package's ``fastmcp`` extra).

``quick`` may import ``ops`` too, where the run summary it prints lives. Nothing in the engine imports
any of the three, and none of the three imports ``vega``.

``storage`` holds adapters behind the one port and ``testing`` the conformance kits an adopter runs
against its own adapter; each needs nothing but the port it implements or checks, so neither may
reach the engine's machinery, and nothing in the engine may reach either.

``quick`` is the batteries: ``run_eval`` and the command line. It is the one package that COMPOSES
the others -- a launch from ``run``, a report from ``analysis``, the reference store from ``storage``
-- which is why it is a package of its own rather than a module of ``run``: ``run`` building the
in-memory store would be the engine reaching an adapter. Nothing imports ``quick``, so composing
sits above everything it composes. ``python -m threetears.evals`` (``__main__``) is held to its row.

``threetears.observe`` is in every row because the repository's logging convention routes every
module's logger through it (``get_logger``); it is a declared dependency of the package and itself
has none.

**Placement is the path, not a list.** A module is in a package because it lives in that package's
directory: ``threetears/evals/contracts/``, ``run/``, ``analysis/``, ``gen/``, ``storage/``,
``testing/`` or ``quick/``. The two markers the path cannot place -- the ``threetears.evals`` root and its
``__main__`` -- are placed by name in :data:`~packages.evals.tests.package_placement.TREE_MARKERS`, the root
held to contracts' row and ``__main__`` to quick's. The rule
itself lives in :mod:`packages.evals.tests.package_placement`.

**What is checked:**

* Every import edge out of a placed module: runtime, ``TYPE_CHECKING`` and function-level imports all
  count, relative imports resolved and ``from pkg import submodule`` expanded to the submodule.
* **Every module under ``threetears/evals/`` is placed**, and :func:`test_every_eval_module_is_placed`
  keeps it that way, so a module born outside every package fails rather than going unchecked.
* A consumer reaches a package only from one of the public roots (:data:`PUBLIC_ROOTS`), and only a
  name that root's ``__all__`` declares. The consumers walked here are the toy host under
  ``tests/fixtures/toyhost/`` -- the second consumer the engine is proven against, which implements
  the host contract from outside as any other host will -- and the minimal courier host under
  ``tests/fixtures/courierhost/``, the third. A host's own tree is the host's to walk on
  the same terms. String couplings count too: a string literal that is wholly a dotted path below a
  package root is refused like the import it stands for.
* Filesystem couplings: a placed module may resolve data beside itself from ``__file__``, never a
  directory above its own (:func:`test_no_placed_module_climbs_out_of_its_directory_by_path`). It
  reads ``pathlib`` chains (``.parents[...]``, ``.parent.parent``); an ``os.path.dirname`` nest or a
  path assembled from strings is not seen.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

import pytest

from packages.evals.tests.import_resolution import absolute_module
from packages.evals.tests.package_placement import discover, eval_modules, join, placement, unplaced_modules

#: The package's ``src``: the directory holding the ``threetears`` namespace.
SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"

#: The package's tests directory, under which the toy host lives.
TESTS_ROOT = Path(__file__).resolve().parent

#: The repository root, which the probes run from.
REPO_ROOT = TESTS_ROOT.parents[2]

#: The matrix, for the eleven packages.
ALLOWED_PACKAGES: dict[str, frozenset[str]] = {
    "contracts": frozenset({"contracts"}),
    "run": frozenset({"contracts", "run"}),
    "analysis": frozenset({"contracts", "analysis"}),
    "gen": frozenset({"contracts", "gen"}),
    "storage": frozenset({"contracts", "storage"}),
    "testing": frozenset({"contracts", "testing"}),
    "quick": frozenset({"contracts", "run", "analysis", "storage", "quick", "ops"}),
    "vega": frozenset({"contracts", "analysis", "vega"}),
    "ops": frozenset({"contracts", "run", "analysis", "ops"}),
    "actions": frozenset({"contracts", "run", "ops", "actions"}),
    "transports": frozenset({"contracts", "ops", "actions", "transports"}),
}

#: The matrix's third-party column, by import root (a ``threetears`` namespace package by its two
#: leading segments, since the namespace itself is shared by the whole family).
ALLOWED_THIRD_PARTY: dict[str, frozenset[str]] = {
    "contracts": frozenset({"pydantic", "threetears.observe"}),
    "run": frozenset({"pydantic", "threetears.observe"}),
    "analysis": frozenset({"pydantic", "threetears.observe"}),
    "gen": frozenset({"pydantic", "threetears.observe"}),
    "storage": frozenset(),
    "testing": frozenset(),
    "quick": frozenset({"pydantic"}),
    "vega": frozenset({"threetears.observe"}),
    "ops": frozenset({"pydantic", "threetears.observe"}),
    "actions": frozenset({"pydantic", "threetears.observe"}),
    "transports": frozenset({"pydantic"}),
}

#: The module-level third-party exceptions, one per extra: the Vega renderer's rasteriser and nothing
#: else may import ``vl_convert``, the ``[vega]`` extra's dependency, and the FastMCP transport and
#: nothing else may import ``fastmcp``, the ``[fastmcp]`` extra's.
THIRD_PARTY_EXCEPTIONS: dict[str, frozenset[str]] = {
    "vega.render": frozenset({"vl_convert"}),
    "transports.fastmcp": frozenset({"fastmcp"}),
}

#: The public roots, relative to ``threetears.evals``. A consumer reaches a package only through one
#: of these. ``contracts.host`` and ``analysis.viz`` are roots of their own inside a package: the first
#: is the contract a host implements, the second the chart intent and the renderer seam. ``vega`` is the
#: optional Vega-Lite renderer.
PUBLIC_ROOTS: tuple[str, ...] = (
    "contracts",
    "contracts.host",
    "run",
    "analysis",
    "analysis.viz",
    "gen",
    "storage",
    "testing",
    "quick",
    "ops",
    "actions",
    "transports.fastmcp",
    "vega",
)

#: The consumers walked, as globs under the tests directory.
SECOND_CONSUMERS: tuple[str, ...] = ("fixtures/toyhost/**/*.py", "fixtures/courierhost/**/*.py")


# --- the walk --------------------------------------------------------------------------------------


class Edge(NamedTuple):
    """One import edge out of an engine module, resolved to the module that owns the target."""

    source: str
    line: int
    target: str


class Violation(NamedTuple):
    """An edge the matrix forbids, with the placements that make it forbidden."""

    source: str
    line: int
    source_package: str
    target: str
    target_package: str

    def render(self) -> str:
        """Describe the violation on one line, with the line number to go and look at."""
        return f"{self.source}:{self.line} ({self.source_package}) -> {self.target} ({self.target_package})"


def _owner(target: str, modules: dict[str, Path]) -> str:
    """The longest prefix of ``target`` that is a real module (an attribute path resolves upward)."""
    parts = target.split(".")
    while parts and ".".join(parts) not in modules:
        parts.pop()
    return ".".join(parts) or target


def _import_edges(module: str, path: Path, modules: dict[str, Path], root: Path) -> Iterator[Edge]:
    """Every import in ``path`` at any depth -- runtime, TYPE_CHECKING and function-level alike."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield Edge(module, node.lineno, alias.name)
        elif isinstance(node, ast.ImportFrom):
            base = absolute_module(path, node, root=root)
            for alias in node.names:
                candidate = f"{base}.{alias.name}"
                yield Edge(module, node.lineno, candidate if candidate in modules else base)


def _in_the_engine(target: str) -> bool:
    return target == "threetears.evals" or target.startswith("threetears.evals.")


def third_party_root(target: str) -> str | None:
    """The import root of ``target`` when it is neither the stdlib nor the engine; None otherwise."""
    parts = target.split(".")
    if not parts[0] or _in_the_engine(target) or parts[0] in sys.stdlib_module_names or parts[0] == "__future__":
        return None
    return ".".join(parts[:2]) if parts[0] == "threetears" else parts[0]


def matrix_violations(root: Path) -> tuple[list[Violation], list[Violation]]:
    """Package-matrix violations and third-party violations over every placed module."""
    modules = discover(root)
    package_violations: list[Violation] = []
    third_party_violations: list[Violation] = []
    for relative, path in sorted(eval_modules(root).items()):
        source = join(relative)
        source_package = placement(source)
        if source_package not in ALLOWED_PACKAGES:
            continue  # a module in no package fails test_every_eval_module_is_placed instead
        for edge in _import_edges(source, path, modules, root):
            if _in_the_engine(edge.target):
                target = _owner(edge.target, modules)
                target_package = placement(target)
                if target_package is None:
                    continue  # a module in no package fails test_every_eval_module_is_placed instead
                if target_package not in ALLOWED_PACKAGES[source_package]:
                    package_violations.append(Violation(source, edge.line, source_package, target, target_package))
                continue
            third_party = third_party_root(edge.target)
            if third_party is None:
                continue
            allowed = ALLOWED_THIRD_PARTY[source_package] | THIRD_PARTY_EXCEPTIONS.get(relative, frozenset())
            if third_party not in allowed:
                third_party_violations.append(Violation(source, edge.line, source_package, third_party, "third-party"))
    return package_violations, third_party_violations


# --- the gate over today's tree --------------------------------------------------------------------


def test_placed_modules_obey_the_package_matrix() -> None:
    """No placed module imports a package its row forbids."""
    package_violations, _ = matrix_violations(SOURCE_ROOT)
    assert not package_violations, (
        "These imports break the allowed-dependency matrix. Fix the import -- move the symbol to a "
        "package the importer may reach, or pass it in as a value; there is no allowlist to add it to:\n  "
        + "\n  ".join(v.render() for v in package_violations)
    )


def test_placed_modules_import_only_their_third_party_column() -> None:
    """Past the stdlib, the packages reach pydantic and threetears.observe; vl_convert only in vega.render."""
    _, third_party_violations = matrix_violations(SOURCE_ROOT)
    assert not third_party_violations, (
        "These third-party imports are outside the matrix's third-party column:\n  "
        + "\n  ".join(v.render() for v in third_party_violations)
    )


def test_every_eval_module_is_placed() -> None:
    """A module under threetears/evals/ must be born in the package directory that owns it."""
    unplaced = sorted(unplaced_modules(SOURCE_ROOT))
    assert not unplaced, (
        f"Engine modules outside every package directory: {unplaced}. Create them under "
        "threetears/evals/{contracts,run,analysis,gen}/ (the package that owns the data it acts on, among "
        "those the matrix lets it import from). The matrix holds no row for a module in no package, so it "
        "would import anything unchecked."
    )


def test_the_placed_population_is_not_empty() -> None:
    """The gate means nothing over an empty population, so assert the walk found placed modules."""
    placed = {placement(join(r)) for r in eval_modules(SOURCE_ROOT)} - {None}
    assert {"analysis", "contracts", "gen", "run"} <= placed, placed


def _refers_to_file(node: ast.AST) -> bool:
    return any(isinstance(n, ast.Name) and n.id == "__file__" for n in ast.walk(node))


def path_climbs(root: Path) -> list[str]:
    """``module:line`` for every placed module that walks above its own directory by path.

    A package is installed as a directory, so a module finding a file by ``__file__`` may look
    beside itself (``Path(__file__).parent / "x.json"`` is package data that travels with it) and
    nowhere higher: a ``.parents[...]`` or ``.parent.parent`` over ``__file__`` reaches a directory
    the package does not ship -- and the import walk above cannot see it, because nothing is imported.

    Args:
        root: The directory holding the ``threetears`` namespace.

    Returns:
        The offending sites, sorted.
    """
    climbs: set[str] = set()
    for relative, path in eval_modules(root).items():
        if placement(join(relative)) not in ALLOWED_PACKAGES:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute) or not _refers_to_file(node.value):
                continue
            nested_parent = (
                node.attr == "parent" and isinstance(node.value, ast.Attribute) and node.value.attr == "parent"
            )
            if node.attr == "parents" or nested_parent:
                climbs.add(f"{relative}:{node.lineno}")
    return sorted(climbs)


def test_no_placed_module_climbs_out_of_its_directory_by_path() -> None:
    """The packages find no file above their own directory through ``__file__``."""
    climbs = path_climbs(SOURCE_ROOT)
    assert not climbs, (
        "These modules locate a file above their own directory by path, which an installed package does "
        "not have. Take the location as a value from the host instead:\n  " + "\n  ".join(climbs)
    )


# --- the consumer column: public roots only --------------------------------------------------------


class RootViolation(NamedTuple):
    """A consumer import that reaches a package other than through a public root's ``__all__``."""

    source: str
    line: int
    imported: str
    why: str

    def render(self) -> str:
        """Describe the violation on one line, with the line number to go and look at."""
        return f"{self.source}:{self.line} {self.imported} — {self.why}"


def public_api(root: Path) -> dict[str, frozenset[str]]:
    """Each public root's ``__all__``, read from its ``__init__.py`` without importing it.

    Read statically so the check works on a synthetic tree and on a root whose import would fail,
    which is the state a broken root is in. A root with no literal ``__all__`` declares nothing.

    Args:
        root: The directory holding the ``threetears`` namespace.

    Returns:
        Absolute root module name -> the names it declares public.
    """
    declared: dict[str, frozenset[str]] = {}
    for relative in PUBLIC_ROOTS:
        init = root.joinpath("threetears", "evals", *relative.split("."), "__init__.py")
        names: frozenset[str] = frozenset()
        if init.is_file():
            for node in ast.parse(init.read_text(encoding="utf-8")).body:
                if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
                ):
                    names = frozenset(ast.literal_eval(node.value))
        declared[join(relative)] = names
    return declared


def consumer_files(tests_root: Path) -> list[tuple[str, Path]]:
    """Every consumer source file the public-root rule binds, with a label for messages.

    Args:
        tests_root: The directory the consumer globs hang off.

    Returns:
        ``(label, file)`` pairs, sorted.
    """
    found: dict[str, Path] = {}
    for pattern in SECOND_CONSUMERS:
        for path in tests_root.glob(pattern):
            found[path.relative_to(tests_root).as_posix()] = path
    return sorted(found.items())


def _in_a_package(module: str) -> bool:
    """Whether ``module`` is one of the eleven packages or below one."""
    return placement(module) in ALLOWED_PACKAGES


#: A string literal that IS a dotted path into one of the engine packages, below the package name.
#: Whole-string and dotted only: a docstring or a sentence naming a module never matches, and a
#: file path in slash form is a file to edit rather than a module to import, so it is left alone.
_DOTTED_PACKAGE_PATH = re.compile(r"threetears\.evals\.(?:contracts|run|analysis|gen|vega)(?:\.\w+)+")


def _string_addressed_imports(tree: ast.AST, declared: dict[str, frozenset[str]]) -> Iterator[tuple[int, str]]:
    """``(line, path)`` for every string literal naming a module below a package root.

    An ``importlib`` string, a registry's ``(module, attribute)`` reference or a module list a walk
    imports by name all reach a module the way an ``import`` statement does, and the import walk
    cannot see any of them. A public root named by string is admitted, exactly as a ``from`` import
    of it is. A literal naming a root's attribute (``threetears.evals.contracts.EvalRun``) is refused
    too, because nothing tells it apart from a submodule by its spelling.
    """
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and _DOTTED_PACKAGE_PATH.fullmatch(node.value)
            and node.value not in declared
        ):
            yield node.lineno, node.value


def public_root_violations(
    root: Path, consumers: list[tuple[str, Path]], *, consumer_root: Path
) -> list[RootViolation]:
    """Consumer imports of a package that do not go through a public root's ``__all__``.

    Runtime, ``TYPE_CHECKING`` and function-level imports all count, and relative imports are
    resolved. Four shapes are refused: a whole-module import of anything in a package (it grants
    every name the module grows), a ``from`` import out of a module that is not a public root (a
    submodule reached past its root, or the ``threetears.evals`` marker handing out a package as a
    module), a name the root does not declare in ``__all__``, and a string literal that is a dotted
    path below a root (:func:`_string_addressed_imports`).

    Args:
        root: The directory holding the ``threetears`` namespace, for the roots' ``__all__``.
        consumers: The ``(label, file)`` pairs to walk.
        consumer_root: The directory the consumers' relative imports resolve against.

    Returns:
        The violations, in file order.
    """
    declared = public_api(root)
    violations: list[RootViolation] = []
    for label, path in consumers:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        violations.extend(
            RootViolation(label, line, f'"{dotted}"', "a string naming a module below a public root; name the root")
            for line, dotted in _string_addressed_imports(tree, declared)
        )
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                violations.extend(
                    RootViolation(
                        label, node.lineno, f"import {alias.name}", "a whole-module import; import names from its root"
                    )
                    for alias in node.names
                    if _in_a_package(alias.name)
                )
                continue
            if not isinstance(node, ast.ImportFrom):
                continue
            base = absolute_module(path, node, root=consumer_root)
            reaches_a_package = _in_a_package(base) or (
                base == "threetears.evals" and any(_in_a_package(join(alias.name)) for alias in node.names)
            )
            if not reaches_a_package:
                continue
            if base not in declared:
                violations.append(RootViolation(label, node.lineno, f"from {base} import …", "not a public root"))
                continue
            violations.extend(
                RootViolation(label, node.lineno, f"from {base} import {alias.name}", f"not in {base}.__all__")
                for alias in node.names
                if alias.name not in declared[base]
            )
    return violations


def test_the_toy_host_imports_only_public_names() -> None:
    """The second consumer reaches the packages the way any client will: through the roots' ``__all__``."""
    violations = public_root_violations(SOURCE_ROOT, consumer_files(TESTS_ROOT), consumer_root=REPO_ROOT)
    assert not violations, (
        "These imports reach an engine package other than through a public root's __all__. Import the "
        "name from its package root; if the root does not export it, decide whether it is public API "
        "(add it to that root's imports and __all__) or something the caller should not be reaching for. "
        "There is no allowlist:\n  " + "\n  ".join(v.render() for v in violations)
    )


def test_the_example_hosts_are_held_to_the_rule_as_consumers() -> None:
    """The walk reaches the two example hosts and nothing else under tests: white-box tests reach what they test."""
    labels = {label for label, _path in consumer_files(TESTS_ROOT)}
    assert {"fixtures/toyhost/kind.py", "fixtures/toyhost/run.py", "fixtures/courierhost/__init__.py"} <= labels
    assert all(label.startswith(("fixtures/toyhost/", "fixtures/courierhost/")) for label in labels)


def _example_host_reaches(path: Path) -> Iterator[tuple[int, str]]:
    """``(line, module)`` for every import a host file makes into this repository's own tree."""
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
        if isinstance(node, ast.ImportFrom):
            module = absolute_module(path, node, root=REPO_ROOT)
            if module.startswith("packages."):
                yield node.lineno, module
        elif isinstance(node, ast.Import):
            yield from ((node.lineno, alias.name) for alias in node.names if alias.name.startswith("packages."))


def test_an_example_host_imports_only_itself() -> None:
    """A host is reference code an adopter copies, so it reaches nothing of this repository's tests.

    Its own package and the public roots (held above) are the whole of what it imports — its store
    included, which is the engine's shipped ``InMemoryDocumentStore``. A host that borrowed a test
    helper, a factory or another host's module would not run once copied out.
    """
    offenders = [
        f"{label}:{line}: {module}"
        for label, path in consumer_files(TESTS_ROOT)
        for own in ["packages.evals.tests." + ".".join(Path(label).parts[:2])]
        for line, module in _example_host_reaches(path)
        if module != own and not module.startswith(own + ".")
    ]
    assert not offenders, "an example host imports from outside itself:\n  " + "\n  ".join(offenders)


def test_every_public_root_declares_what_it_binds() -> None:
    """Each root's ``__all__`` names something the root actually binds, so a declared name imports."""
    import importlib

    missing = []
    for module, names in public_api(SOURCE_ROOT).items():
        assert names, f"{module} declares no __all__, so nothing in it is public and every consumer import is refused"
        loaded = importlib.import_module(module)
        missing.extend(f"{module}.{name}" for name in sorted(names) if not hasattr(loaded, name))
    assert not missing, "Declared in __all__ but not bound by the root:\n  " + "\n  ".join(missing)


#: Each public root, imported first in a fresh interpreter, must load and must not load
#: ``vl_convert``; each root but the renderer's own must not load the renderer either. Eager roots turn
#: a submodule import into a whole-package import, so an import cycle surfaces only when a root is the
#: FIRST thing a process imports; and the rasteriser is the ``[vega]`` extra's one dependency, which
#: ``vega.render`` takes at call time so that compiling a spec does not need it.
_ROOT_PROBE = (
    "import importlib, sys\n"
    "importlib.import_module({module!r})\n"
    "assert 'vl_convert' not in sys.modules, 'importing {module} loaded vl_convert'\n"
    "assert {module!r} == 'threetears.evals.vega' or 'threetears.evals.vega' not in sys.modules, "
    "'importing {module} loaded the Vega renderer'\n"
)


@pytest.mark.parametrize("relative", PUBLIC_ROOTS)
def test_each_public_root_imports_first_and_alone(relative: str) -> None:
    """A root that only imports after something else has warmed its package is a cycle waiting."""
    module = join(relative)
    proc = subprocess.run(
        [sys.executable, "-c", _ROOT_PROBE.format(module=module)],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=REPO_ROOT,
        check=False,
    )
    assert proc.returncode == 0, f"importing {module} first, in a fresh interpreter, failed:\n{proc.stderr}"


#: ``vl_convert`` made unimportable (a ``None`` entry in ``sys.modules`` is what Python reads as
#: "this module cannot be imported"), then every root imported, then a render attempted: the
#: rasteriser is optional for everything but drawing, and drawing says so by name.
_WITHOUT_VL_CONVERT_PROBE = (
    "import sys\n"
    "sys.modules['vl_convert'] = None\n"
    "import threetears.evals.contracts, threetears.evals.contracts.host, threetears.evals.run, threetears.evals.gen\n"
    "import threetears.evals.analysis, threetears.evals.analysis.viz\n"
    "from threetears.evals.vega import compile_chart, render_png\n"
    "try:\n"
    "    render_png({})\n"
    "except ImportError as exc:\n"
    "    assert 'vl_convert' in str(exc), str(exc)\n"
    "else:\n"
    "    raise AssertionError('render_png drew without vl_convert')\n"
)


def test_every_root_imports_without_the_rasteriser_installed() -> None:
    """A host with no ``vl_convert`` can import every root, ``vega`` included, and only rasterising fails."""
    proc = subprocess.run(
        [sys.executable, "-c", _WITHOUT_VL_CONVERT_PROBE],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=REPO_ROOT,
        check=False,
    )
    assert proc.returncode == 0, f"a root needed vl_convert to import, or drawing did not:\n{proc.stderr}"


#: The core with neither the renderer nor its rasteriser importable: the toy host's analysis generated,
#: read as its report, serialized three ways, and one finding's chart decided — everything a host does
#: with charts short of drawing one. A host that installs ``3tears-evals`` without the ``[vega]`` extra
#: and brings its own renderer (or none) is in exactly this state.
_WITHOUT_THE_RENDERER_PROBE = (
    "import asyncio, sys\n"
    "sys.modules['vl_convert'] = None\n"
    "sys.modules['threetears.evals.vega'] = None\n"
    "from threetears.evals.analysis import finding_chart_intent, report_html, report_markdown\n"
    "from packages.evals.tests.report_support import toy_report\n"
    "host, analysis, report = asyncio.run(toy_report())\n"
    "assert report.to_canonical_json() and report_markdown(report) and '<table' in report_html(report)\n"
    "intent = finding_chart_intent(host.storage, analysis.id, analysis.scope_id, '0')\n"
    "assert intent.type == 'delta_table' and intent.rows, intent\n"
    "assert not [n for n in sys.modules if n.startswith('threetears.evals.vega.')]\n"
)


def test_the_core_runs_the_toy_analysis_without_the_renderer() -> None:
    """Generating, reporting and deciding a chart need neither ``threetears.evals.vega`` nor ``vl_convert``."""
    proc = subprocess.run(
        [sys.executable, "-c", _WITHOUT_THE_RENDERER_PROBE],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=REPO_ROOT,
        check=False,
    )
    assert proc.returncode == 0, f"the core needed the Vega renderer or its rasteriser:\n{proc.stderr}"


# --- the checker fires: synthetic trees, one per forbidden and permitted shape ----------------------


def _tree(tmp_path: Path, files: dict[str, str]) -> Path:
    """Build a ``threetears/evals/`` tree under ``tmp_path`` with the given sources (and package inits)."""
    for relative, source in files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    engine = tmp_path / "threetears" / "evals"
    for directory in {p for p in engine.rglob("*") if p.is_dir()} | {engine}:
        init = directory / "__init__.py"
        if not init.exists():
            init.write_text("", encoding="utf-8")
    return tmp_path


def _consumers(root: Path) -> list[tuple[str, Path]]:
    """Every file of a synthetic tree outside the engine, as the consumers it stands for."""
    return sorted(
        (path.relative_to(root).as_posix(), path)
        for path in root.rglob("*.py")
        if not path.relative_to(root).as_posix().startswith("threetears/evals/")
    )


#: Synthetic names (``legacy``, ``widgets``) for the unplaced modules: a real module here would be
#: rewritten by the package move that places it, silently changing the case.
_BASE_FILES = {
    "threetears/evals/legacy.py": "",
    "threetears/evals/contracts/models.py": "",
    "threetears/evals/run/jobs.py": "",
    "threetears/evals/analysis/stats.py": "",
    "threetears/evals/gen/proposers.py": "",
    "threetears/evals/vega/compiler.py": "",
}


@pytest.mark.parametrize(
    ("relative", "source", "target", "target_package"),
    [
        (
            "threetears/evals/run/loop.py",
            "from ..analysis.stats import mean\n",
            "threetears.evals.analysis.stats",
            "analysis",
        ),
        (
            "threetears/evals/run/loop.py",
            "from threetears.evals.analysis import stats\n",
            "threetears.evals.analysis.stats",
            "analysis",
        ),
        (
            "threetears/evals/run/loop.py",
            "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from threetears.evals.analysis.stats import mean\n",
            "threetears.evals.analysis.stats",
            "analysis",
        ),
        (
            "threetears/evals/run/loop.py",
            "def f():\n    import threetears.evals.analysis.stats\n",
            "threetears.evals.analysis.stats",
            "analysis",
        ),
        (
            "threetears/evals/contracts/leaf.py",
            "from threetears.evals.run.jobs import J\n",
            "threetears.evals.run.jobs",
            "run",
        ),
        (
            "threetears/evals/analysis/lens.py",
            "from threetears.evals.run import jobs\n",
            "threetears.evals.run.jobs",
            "run",
        ),
        (
            "threetears/evals/gen/expand.py",
            "import threetears.evals.analysis.stats\n",
            "threetears.evals.analysis.stats",
            "analysis",
        ),
        ("threetears/evals/__init__.py", "from threetears.evals.run import jobs\n", "threetears.evals.run.jobs", "run"),
        # The core never reaches the renderer: not from analysis, which it draws for, nor from quick.
        (
            "threetears/evals/analysis/lens.py",
            "from threetears.evals.vega.compiler import draw_intent\n",
            "threetears.evals.vega.compiler",
            "vega",
        ),
        (
            "threetears/evals/quick/report.py",
            "from threetears.evals.vega import compiler\n",
            "threetears.evals.vega.compiler",
            "vega",
        ),
        # Nor the renderer the engine machinery it has no business with.
        (
            "threetears/evals/vega/compiler.py",
            "from threetears.evals.run.jobs import J\n",
            "threetears.evals.run.jobs",
            "run",
        ),
    ],
)
def test_the_matrix_refuses_each_forbidden_edge(
    tmp_path: Path, relative: str, source: str, target: str, target_package: str
) -> None:
    """Each forbidden shape is reported, naming the owning target module and its package."""
    violations, _ = matrix_violations(_tree(tmp_path, {**_BASE_FILES, relative: source}))
    assert [(v.target, v.target_package) for v in violations] == [(target, target_package)]


@pytest.mark.parametrize(
    ("relative", "source"),
    [
        ("threetears/evals/run/loop.py", "from threetears.evals.contracts.models import M\nfrom . import jobs\n"),
        (
            "threetears/evals/analysis/lens.py",
            "from threetears.evals.contracts import models\nfrom .stats import mean\n",
        ),
        ("threetears/evals/gen/expand.py", "from ..contracts.models import M\n"),
        # An edge into a module in no package is the placement test's to refuse, not the walk's.
        ("threetears/evals/run/loop.py", "from threetears.evals.legacy import execute_run\n"),
        # An unplaced module is held to no row at all.
        ("threetears/evals/legacy.py", "import acme.config\n"),
        # The renderer reads the intent it draws.
        ("threetears/evals/vega/compiler.py", "from threetears.evals.analysis.stats import mean\nfrom . import arms\n"),
    ],
)
def test_the_matrix_admits_each_permitted_edge(tmp_path: Path, relative: str, source: str) -> None:
    """The permitted shapes pass, so a checker that refuses everything cannot pass the refusals above."""
    violations, third_party = matrix_violations(_tree(tmp_path, {**_BASE_FILES, relative: source}))
    assert violations == [] and third_party == []


@pytest.mark.parametrize(
    ("relative", "source", "refused"),
    [
        ("threetears/evals/contracts/leaf.py", "import httpx\nimport pydantic\nimport json\n", ["httpx"]),
        ("threetears/evals/run/loop.py", "from langchain_openai import ChatOpenAI\n", ["langchain_openai"]),
        ("threetears/evals/analysis/lens.py", "import vl_convert\n", ["vl_convert"]),
        # The rasteriser left the core with the renderer: where it used to be admitted, it is refused.
        ("threetears/evals/analysis/viz/render.py", "import vl_convert\n", ["vl_convert"]),
        ("threetears/evals/vega/render.py", "import vl_convert\nfrom threetears.observe import get_logger\n", []),
        ("threetears/evals/vega/compiler.py", "import vl_convert\n", ["vl_convert"]),
        ("threetears/evals/run/loop.py", "from threetears.observe import get_logger\n", []),
        # A sibling family package the manifest does not declare is a dependency like any other.
        ("threetears/evals/run/loop.py", "from threetears.core import thing\n", ["threetears.core"]),
        # A host package, at runtime, under TYPE_CHECKING, deferred, and reached by a relative climb.
        ("threetears/evals/run/loop.py", "import acme.config\n", ["acme"]),
        (
            "threetears/evals/run/loop.py",
            "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from acme.config import X\n",
            ["acme"],
        ),
        ("threetears/evals/run/loop.py", "def f():\n    import acme.config\n", ["acme"]),
        ("threetears/evals/run/loop.py", "from ...config import X\n", ["threetears.config"]),
    ],
)
def test_the_third_party_column(tmp_path: Path, relative: str, source: str, refused: list[str]) -> None:
    """pydantic, observe and the stdlib pass where their row says; vl_convert only in vega.render; the rest is refused."""
    _, third_party = matrix_violations(_tree(tmp_path, {**_BASE_FILES, relative: source}))
    assert [v.target for v in third_party] == refused


def test_a_new_module_outside_every_package_is_seen(tmp_path: Path) -> None:
    """The new-module check's input: an unlisted module at the top of the tree reads as unplaced."""
    root = _tree(
        tmp_path, {**_BASE_FILES, "threetears/evals/brand_new.py": "", "threetears/evals/widgets/chart.py": ""}
    )
    assert unplaced_modules(root) == {"legacy", "brand_new", "widgets", "widgets.chart"}


@pytest.mark.parametrize(
    ("source", "climbs"),
    [
        ('from pathlib import Path\nSEED = Path(__file__).resolve().parents[1] / "seed"\n', True),
        ('from pathlib import Path\nSEED = Path(__file__).parent.parent / "seed"\n', True),
        ('from pathlib import Path\nDATA = Path(__file__).resolve().parent / "palette.json"\n', False),
        ('from pathlib import Path\nDATA = Path("x").parents[1]\n', False),
    ],
)
def test_the_path_climb_rule(tmp_path: Path, source: str, climbs: bool) -> None:
    """A climb over ``__file__`` is refused in a placed module; package data beside it is not.

    The fourth case is a ``.parents`` that never touches ``__file__``, so a checker that refused
    every ``.parents`` would fail it.
    """
    root = _tree(tmp_path, {**_BASE_FILES, "threetears/evals/run/seed.py": source})
    assert path_climbs(root) == (["run.seed:2"] if climbs else [])


#: A tree whose run root declares one public name and binds a second it does not declare, so the
#: refusal of an undeclared name is not a refusal of an unbound one.
_ROOTS_FILES = {
    **_BASE_FILES,
    "threetears/evals/run/__init__.py": (
        "from threetears.evals.run.jobs import EvalJobManager, adaptive_job_timeout_s\n__all__ = ['EvalJobManager']\n"
    ),
    "threetears/evals/contracts/__init__.py": "from threetears.evals.contracts.models import EvalRun\n__all__ = ['EvalRun']\n",
}


@pytest.mark.parametrize(
    ("relative", "source", "imported"),
    [
        # A submodule reached past its root.
        ("host/routes.py", "from threetears.evals.run.jobs import EvalJobManager\n", "from threetears.evals.run.jobs"),
        (
            "tools/probe.py",
            "from threetears.evals.contracts.models import EvalRun\n",
            "from threetears.evals.contracts.models",
        ),
        # Function-level and TYPE_CHECKING imports count, as the matrix's own rows count them.
        (
            "host/routes.py",
            "def f():\n    from threetears.evals.run.jobs import EvalJobManager\n",
            "from threetears.evals.run.jobs",
        ),
        (
            "host/routes.py",
            "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from threetears.evals.contracts.models import EvalRun\n",
            "from threetears.evals.contracts.models",
        ),
        # A name the root binds but does not declare.
        (
            "host/routes.py",
            "from threetears.evals.run import adaptive_job_timeout_s\n",
            "import adaptive_job_timeout_s",
        ),
        # A whole module, by either spelling.
        ("host/routes.py", "import threetears.evals.run.jobs\n", "import threetears.evals.run.jobs"),
        ("host/routes.py", "from threetears.evals import run\n", "from threetears.evals import"),
        ("host/routes.py", "from threetears.evals.run import jobs\n", "import jobs"),
        # A module reached by string: an importlib call, a reference held as data, a list a walk imports.
        (
            "host/routes.py",
            'import importlib\nimportlib.import_module("threetears.evals.run.jobs")\n',
            '"threetears.evals.run.jobs"',
        ),
        (
            "host/app.py",
            'SEED = ("threetears.evals.contracts.models", "EvalRun")\n',
            '"threetears.evals.contracts.models"',
        ),
        ("tools/probe.py", 'MODULES = ["threetears.evals.contracts.models"]\n', '"threetears.evals.contracts.models"'),
        # A root's attribute by string reads exactly like a submodule, so it is refused with them.
        ("host/routes.py", 'TARGET = "threetears.evals.run.EvalJobManager"\n', '"threetears.evals.run.EvalJobManager"'),
    ],
)
def test_the_public_root_rule_refuses_each_forbidden_import(
    tmp_path: Path, relative: str, source: str, imported: str
) -> None:
    """Each shape that bypasses a root's ``__all__`` is reported once, naming what it imported."""
    root = _tree(tmp_path, {**_ROOTS_FILES, relative: source})
    violations = public_root_violations(root, _consumers(root), consumer_root=root)
    assert len(violations) == 1, violations
    assert imported in violations[0].imported


@pytest.mark.parametrize(
    ("relative", "source"),
    [
        ("host/app.py", "from threetears.evals.run import EvalJobManager\n"),
        (
            "host/routes.py",
            "from threetears.evals.contracts import EvalRun\nfrom threetears.evals.run import EvalJobManager\n",
        ),
        ("tools/probe.py", "def f():\n    from threetears.evals.contracts import EvalRun\n"),
        # A consumer's own modules import each other freely; the rule binds what it takes from the engine.
        ("host/app.py", "from .routes import X\nimport acme.config\n"),
        # A package's own modules import each other directly; the rule binds consumers, not the package.
        ("threetears/evals/run/loop.py", "from threetears.evals.run.jobs import EvalJobManager\n"),
        # By string: a root, a file path (a file to edit, not a module to import), a sentence that
        # mentions a module, and a package's own string.
        ("host/routes.py", 'SEED = ("threetears.evals.run", "EvalJobManager")\n'),
        ("host/routes.py", 'PATH = "threetears/evals/run/jobs.py"\n'),
        ("host/routes.py", '"""See threetears.evals.run.jobs for the admission rules."""\n'),
        ("threetears/evals/run/loop.py", 'LOGGER = "threetears.evals.run.jobs"\n'),
    ],
)
def test_the_public_root_rule_admits_each_permitted_import(tmp_path: Path, relative: str, source: str) -> None:
    """The permitted shapes pass, so a checker that refuses everything cannot pass the refusals above."""
    root = _tree(tmp_path, {**_ROOTS_FILES, relative: source})
    assert public_root_violations(root, _consumers(root), consumer_root=root) == []


def test_the_package_publishes_the_roots_this_gate_enforces() -> None:
    """``threetears.evals.PUBLIC_ROOTS`` is the matrix's roots, in order, so a consumer reads the real set.

    A host checks its own imports against the installed package's tuple rather than keeping a copy,
    so the published tuple has to be exactly the set this gate holds consumers to: a root missing
    from it refuses a legal import, and an extra one admits an import nothing here checks.
    """
    import threetears.evals

    assert threetears.evals.PUBLIC_ROOTS == tuple(join(relative) for relative in PUBLIC_ROOTS)
