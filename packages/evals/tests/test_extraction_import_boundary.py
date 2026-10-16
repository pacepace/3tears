"""Structural gate: the engine names, and loads, nothing a second host does not have.

The engine was cut out of its first host. The cut is only worth having if it holds, and it is the
easiest property here to undo by accident: one import of a host module reads as nothing in review
and arrives at every other consumer as an ``ImportError`` naming a package it has never had. So the
boundary is held from three sides, each with the mechanism that answers its own question:

**What a module NAMES** -- an AST walk over every module under ``threetears/evals``, which sees
through ``TYPE_CHECKING``, dead branches and imports deferred into function bodies, and resolves
relative imports rather than skipping them (:func:`~packages.evals.tests.import_resolution.absolute_module`).
Every module it names is the standard library, the engine itself, or a dependency the package's own
manifest declares (:data:`IMPORT_ROOTS_BY_DEPENDENCY`). The first host's package name is refused on
its own as well, in a message of its own, because that is the specific regression an extraction
invites.

**What importing it LOADS** -- a probe in a fresh interpreter that imports every module of the
package and then inspects ``sys.modules``. That is the transitive closure, which no per-file walk
reports, and it has to be a fresh process: the test process's ``sys.modules`` is shared with every
other test.

**What importing it INSTALLS** -- nothing can be: the engine holds no host state at module level for
an import to set (``test_no_process_global_state.py`` walks for one), and every entrypoint takes its
host as an argument, so there is no default for one host's vocabulary to reach another host's data
through.

The portable test support (the host-neutral factories, the shared import resolver, the host-noun
list and the toy host) is held to the same rule, because the toy host is the second consumer the
engine is proven against and a toy host that reached the first host would prove nothing.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import tomllib
from collections.abc import Iterator
from pathlib import Path

import pytest

from packages.evals.tests.import_resolution import absolute_module

#: The package directory: ``pyproject.toml``, ``src`` and ``tests`` hang off it.
PACKAGE_ROOT = Path(__file__).resolve().parents[1]

#: The repository root, which the probes run from so ``packages.evals.tests`` resolves.
REPO_ROOT = PACKAGE_ROOT.parents[1]

#: The source root the engine's dotted names are rooted at.
SOURCE_ROOT = PACKAGE_ROOT / "src"

#: The first host's top-level package. Named on its own so a regression to it gets the message
#: that says what it is, rather than the general "undeclared dependency" one.
FIRST_HOST = "discodon"

#: Each runtime dependency the manifest declares, mapped to the import roots it provides. Keyed by
#: distribution name exactly as ``pyproject.toml`` spells it, and checked against the manifest by
#: :func:`test_the_dependency_map_is_the_manifests`, so a dependency added to one and not the other
#: fails rather than widening or narrowing the walk below in silence.
IMPORT_ROOTS_BY_DEPENDENCY: dict[str, tuple[str, ...]] = {
    "httpx": ("httpx",),
    "pydantic": ("pydantic",),
}

#: Each optional extra the manifest declares: the subpackage that is the extra (and so the only
#: modules that may name its dependencies), and each of its dependencies' import roots. Checked against
#: the manifest by :func:`test_the_extras_map_is_the_manifests`. An extra is the dependency of the
#: subpackage it serves and of nothing else: the engine imported without the extra must still import,
#: so a core module naming an extra's dependency would be an ``ImportError`` for every host that
#: installed without it, and one extra's subpackage naming ANOTHER extra's dependency would tie the two
#: extras together.
IMPORT_ROOTS_BY_EXTRA: dict[str, tuple[str, dict[str, tuple[str, ...]]]] = {
    "fastmcp": ("threetears.evals.transports.fastmcp", {"fastmcp": ("fastmcp",)}),
}

#: Packages the portable test support may neither name nor load: the first host, and the web and
#: telemetry stacks a host wraps the engine in, which a second consumer installs neither of.
BARRED_FROM_TEST_SUPPORT: tuple[str, ...] = (FIRST_HOST, "fastapi", "opentelemetry")

#: The test-support modules that travel with the engine's suite, as paths under ``tests``.
PORTABLE_TEST_SUPPORT: tuple[str, ...] = (
    "factories.py",
    "import_resolution.py",
    "host_vocabulary.py",
    "host_vocabulary_register.py",
    "fixtures/toyhost",
)


def _matches(module: str, target: str) -> bool:
    """True if ``module`` is ``target`` or a submodule of it."""
    return module == target or module.startswith(f"{target}.")


def _distribution(requirement: str) -> str:
    """A requirement string's distribution name."""
    for separator in ("<", ">", "=", "!", "~", ";", "[", " "):
        requirement = requirement.split(separator, 1)[0]
    return requirement.strip()


def _manifest() -> dict:
    return tomllib.loads((PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _declared_dependencies() -> list[str]:
    """The runtime dependencies ``pyproject.toml`` declares, by distribution name."""
    return [_distribution(requirement) for requirement in _manifest()["project"]["dependencies"]]


def _declared_extras() -> dict[str, list[str]]:
    """The optional extras ``pyproject.toml`` declares, each with its dependencies by distribution name."""
    extras = _manifest()["project"].get("optional-dependencies", {})
    return {
        extra: [_distribution(requirement) for requirement in requirements] for extra, requirements in extras.items()
    }


def _allowed(module: str, *, importer: str = "threetears.evals") -> bool:
    """Whether ``importer`` may name ``module``: the stdlib, the engine, a declared dependency's root, or —
    from inside an extra's own subpackage only — that extra's dependencies' roots."""
    if module.split(".")[0] in sys.stdlib_module_names:
        return True
    roots = ["threetears.evals", *(root for roots in IMPORT_ROOTS_BY_DEPENDENCY.values() for root in roots)]
    for subpackage, dependencies in IMPORT_ROOTS_BY_EXTRA.values():
        if _matches(importer, subpackage):
            roots.extend(root for extra_roots in dependencies.values() for root in extra_roots)
    return any(_matches(module, root) for root in roots)


def _module_of(path: Path) -> str:
    """The dotted name of an engine source file."""
    parts = path.relative_to(SOURCE_ROOT).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def named_modules(path: Path, *, root: Path) -> Iterator[tuple[int, str]]:
    """Every ``(line, absolute module)`` a file names, at any depth, relative imports resolved.

    :param path: The file to read.
    :ptype path: Path
    :param root: The directory its dotted names are rooted at.
    :ptype root: Path
    :yield: One pair per module named.
    :rtype: Iterator[tuple[int, str]]
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            yield node.lineno, absolute_module(path, node, root=root)


def _engine_files() -> list[Path]:
    return sorted((SOURCE_ROOT / "threetears" / "evals").rglob("*.py"))


def _support_files() -> list[Path]:
    files: list[Path] = []
    for entry in PORTABLE_TEST_SUPPORT:
        path = PACKAGE_ROOT / "tests" / entry
        assert path.exists(), f"{entry} is on PORTABLE_TEST_SUPPORT and does not exist"
        files.extend(sorted(path.rglob("*.py")) if path.is_dir() else [path])
    return files


def _probe(source: str) -> subprocess.CompletedProcess[str]:
    """Run one probe in a fresh interpreter rooted at the repository.

    :param source: The probe program; a literal from this file, never anything assembled from input.
    :ptype source: str
    :return: The finished process, for the caller to assert on.
    :rtype: subprocess.CompletedProcess[str]
    """
    return subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        # A guard against a hung interpreter, not a speed bound: importing every root loads the FastMCP stack
        # too, and on a loaded machine importing every root ran past a minute.
        timeout=300,
        cwd=REPO_ROOT,
        check=False,
    )


# ---------------------------------------------------------------------------
# What a module names
# ---------------------------------------------------------------------------


def test_the_walk_reads_the_engine():
    """Non-vacuity: a walk that read no module would pass every assertion below."""
    files = _engine_files()

    assert any(path.name == "runner.py" for path in files), "the walk found no engine module"
    assert sum(1 for path in files for _ in named_modules(path, root=SOURCE_ROOT)) > 100


def test_the_dependency_map_is_the_manifests():
    """The import roots the walk admits are exactly those of the dependencies the manifest declares."""
    assert sorted(IMPORT_ROOTS_BY_DEPENDENCY) == sorted(_declared_dependencies())


def test_the_extras_map_is_the_manifests():
    """Each extra the walk admits is one the manifest declares, with exactly its dependencies."""
    assert {extra: sorted(deps) for extra, (_, deps) in IMPORT_ROOTS_BY_EXTRA.items()} == {
        extra: sorted(deps) for extra, deps in _declared_extras().items()
    }
    for subpackage, _ in IMPORT_ROOTS_BY_EXTRA.values():
        path = SOURCE_ROOT.joinpath(*subpackage.split("."))
        assert path.is_dir(), f"an extra serves {subpackage}, which does not exist"


def test_root_extras_names_only_public_roots_each_under_the_extra_that_serves_it():
    """``threetears.evals.ROOT_EXTRAS`` maps a public root to the extra whose subpackage it is, and to no other."""
    from threetears.evals import PUBLIC_ROOTS, ROOT_EXTRAS

    served = {subpackage: extra for extra, (subpackage, _) in IMPORT_ROOTS_BY_EXTRA.items()}
    listed = {root: needs.extra for root, needs in ROOT_EXTRAS.items()}
    assert listed.items() <= served.items(), "a root is listed under an extra that does not serve it"
    assert set(ROOT_EXTRAS) <= set(PUBLIC_ROOTS), "an extra's root that is not public belongs in no consumer's list"


def _names_outside_the_base_install(subpackage: str) -> set[str]:
    """Every module the files under ``subpackage`` name that the base install does not provide, by the name the
    import gives it: the stdlib, the engine and the base dependencies' roots left out, and a ``threetears`` namespace
    package kept to its two leading segments, as the extras map names it."""
    base = ["threetears.evals", *(root for roots in IMPORT_ROOTS_BY_DEPENDENCY.values() for root in roots)]
    named: set[str] = set()
    for path in sorted(SOURCE_ROOT.joinpath(*subpackage.split(".")).rglob("*.py")):
        for _, module in named_modules(path, root=SOURCE_ROOT):
            if module.split(".")[0] in sys.stdlib_module_names or any(_matches(module, root) for root in base):
                continue
            parts = module.split(".")
            named.add(".".join(parts[:2]) if parts[0] == "threetears" else parts[0])
    return named


def _optional_roots() -> list[str]:
    from threetears.evals import ROOT_EXTRAS

    return sorted(ROOT_EXTRAS)


@pytest.mark.parametrize("root", _optional_roots())
def test_root_extras_lists_every_module_its_root_names_past_the_base_install(root):
    """The modules a root's entry lists are exactly the ones its source names that the base install lacks, sorted:
    a gate that checks them checks everything the root may fail on, and nothing it does not import."""
    from threetears.evals import ROOT_EXTRAS

    needs = ROOT_EXTRAS[root]

    assert set(needs.modules) == _names_outside_the_base_install(root)
    assert list(needs.modules) == sorted(set(needs.modules)), "listed out of order, or twice"
    assert set(needs.modules) <= set(_extra_roots(needs.extra)), "a module its extra does not bring in"


def test_the_root_extras_walk_finds_a_module_named_past_the_base_install():
    """Non-vacuity: the walk behind the equality above sees the FastMCP transport's own import, and nothing the base
    install provides (the client's ``httpx`` is a base dependency)."""
    assert _names_outside_the_base_install("threetears.evals.transports.fastmcp") == {"fastmcp"}
    assert _names_outside_the_base_install("threetears.evals.providers") == set()


#: A probe's preamble: makes every import root named in ``sys.argv[1]`` (a JSON list) unimportable, as an install
#: without the extras that provide them is, and defines ``importable(root)``: whether a public root imports, and, when
#: it does not, the root it could not import, which must be one of the blocked ones.
_WITHOUT_ROOTS = """
import importlib, json, sys

BLOCKED = tuple(json.loads(sys.argv[1]))


def _blocked(name):
    return any(name == root or name.startswith(root + ".") for root in BLOCKED)


class _Absent:
    def find_spec(self, name, path=None, target=None):
        if _blocked(name):
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return None


assert not [name for name in sys.modules if _blocked(name)], "a blocked root was loaded before the probe began"
sys.meta_path.insert(0, _Absent())


def importable(root):
    try:
        importlib.import_module(root)
    except ModuleNotFoundError as missing:
        assert missing.name is not None and _blocked(missing.name), f"{root} failed on {missing.name}, not a blocked root"
        return False
    return True
"""

#: Imports every public root with the blocked roots unimportable and prints the roots that failed, as JSON.
_FAILING_ROOTS = (
    _WITHOUT_ROOTS
    + """
from threetears.evals import PUBLIC_ROOTS

print(json.dumps(sorted(root for root in PUBLIC_ROOTS if not importable(root))))
"""
)


#: Each extra dependency's own import root: what is absent when that distribution alone is not installed. Narrower
#: than :data:`IMPORT_ROOTS_BY_EXTRA`'s roots, which may also name what a distribution brings in itself, so an install
#: with one extra and not another still has those.
_DISTRIBUTION_ROOT: dict[str, str] = {
    "fastmcp": "fastmcp",
}


def _extra_roots(*extras: str) -> list[str]:
    """Every import root the named extras' dependencies provide, what they bring in included: the base install's
    absences."""
    return sorted({root for extra in extras for roots in IMPORT_ROOTS_BY_EXTRA[extra][1].values() for root in roots})


def test_each_extra_dependency_names_its_own_root():
    """The per-extra probe blocks each distribution's own root; one the map did not name would block nothing."""
    declared = {dist: roots for _, deps in IMPORT_ROOTS_BY_EXTRA.values() for dist, roots in deps.items()}

    assert sorted(_DISTRIBUTION_ROOT) == sorted(declared)
    assert all(_DISTRIBUTION_ROOT[dist] in roots for dist, roots in declared.items())


def _failing_roots(blocked: list[str]) -> set[str]:
    """The public roots that fail to import in a fresh interpreter where ``blocked`` cannot be imported."""
    proc = subprocess.run(
        [sys.executable, "-c", _FAILING_ROOTS, json.dumps(blocked)],
        capture_output=True,
        text=True,
        # A guard against a hung interpreter, not a speed bound: see ``_probe``.
        timeout=300,
        cwd=REPO_ROOT,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return set(json.loads(proc.stdout))


def test_on_the_base_install_exactly_the_listed_roots_fail_to_import():
    """With every extra's dependencies unimportable, each root ``ROOT_EXTRAS`` lists fails to import and every other
    public root imports: the map is what an install without the extras meets."""
    from threetears.evals import ROOT_EXTRAS

    assert _failing_roots(_extra_roots(*IMPORT_ROOTS_BY_EXTRA)) == set(ROOT_EXTRAS)


@pytest.mark.parametrize("extra", sorted(IMPORT_ROOTS_BY_EXTRA))
def test_each_extra_alone_missing_fails_exactly_the_roots_listed_under_it(extra):
    """With one extra's dependencies unimportable and the rest installed, the roots that fail are the ones the map lists
    under that extra: each listed root needs the extra it is listed under, and no other."""
    from threetears.evals import ROOT_EXTRAS

    blocked = sorted(_DISTRIBUTION_ROOT[dist] for dist in IMPORT_ROOTS_BY_EXTRA[extra][1])

    assert _failing_roots(blocked) == {root for root, needs in ROOT_EXTRAS.items() if needs.extra == extra}


#: Imports one root with the blocked roots unimportable and prints whether it imported, as JSON.
_ROOT_IMPORTS = (
    _WITHOUT_ROOTS
    + """
print(json.dumps(importable(sys.argv[2])))
"""
)


def _listed_modules() -> list[tuple[str, str]]:
    from threetears.evals import ROOT_EXTRAS

    return [(root, module) for root, needs in ROOT_EXTRAS.items() for module in needs.modules]


@pytest.mark.parametrize(("root", "module"), _listed_modules())
def test_each_module_root_extras_lists_is_one_its_root_cannot_import_without(root, module):
    """With any one listed module unimportable and everything else installed, its root fails to import: every module
    a gate is told to check is one the root needs at import, not only somewhere in a call."""
    proc = subprocess.run(
        [sys.executable, "-c", _ROOT_IMPORTS, json.dumps([module]), root],
        capture_output=True,
        text=True,
        # A guard against a hung interpreter, not a speed bound: see ``_probe``.
        timeout=300,
        cwd=REPO_ROOT,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout) is False, f"{root} imported with {module} unimportable"


def test_the_probe_reports_a_root_that_fails_without_a_blocked_root():
    """Non-vacuity: blocking pydantic, the base install's one dependency, fails every public root."""
    from threetears.evals import PUBLIC_ROOTS

    assert _failing_roots(["pydantic"]) == set(PUBLIC_ROOTS)


def test_no_engine_module_names_the_first_host():
    """No module under ``threetears/evals`` names any module of the host it was cut from."""
    offenders = [
        f"{path.relative_to(SOURCE_ROOT)}:{line} imports {module}"
        for path in _engine_files()
        for line, module in named_modules(path, root=SOURCE_ROOT)
        if _matches(module, FIRST_HOST)
    ]

    assert not offenders, (
        "the engine names its first host's modules, which no other consumer has:\n  "
        + "\n  ".join(offenders)
        + "\n\nMove what is needed into threetears.evals, or have the host inject it through a contract."
    )


def test_every_module_the_engine_names_is_declared():
    """The stdlib, the engine, or a declared dependency -- an undeclared import is an ``ImportError`` elsewhere."""
    offenders = [
        f"{path.relative_to(SOURCE_ROOT)}:{line} imports {module}"
        for path in _engine_files()
        for line, module in named_modules(path, root=SOURCE_ROOT)
        if not _allowed(module, importer=_module_of(path))
    ]

    assert not offenders, (
        "the engine names modules its manifest does not declare a dependency on:\n  "
        + "\n  ".join(offenders)
        + "\n\nDeclare the dependency in packages/evals/pyproject.toml AND in IMPORT_ROOTS_BY_DEPENDENCY, "
        "or remove the import."
    )


@pytest.mark.parametrize(
    ("source", "allowed"),
    [
        ("import json\n", True),
        ("from threetears.evals.schema import models\n", True),
        ("from pydantic import BaseModel\n", True),
        # The client's HTTP is a base dependency; a provider SDK is not one.
        ("import httpx\n", True),
        ("import anthropic\n", False),
        # No 3tears package is a dependency, observe included.
        ("from threetears.observe import get_logger\n", False),
        ("import discodon.config\n", False),
        ("from threetears.core import thing\n", False),
        ("import fastapi\n", False),
    ],
)
def test_the_rule_admits_what_is_declared_and_refuses_the_rest(tmp_path, source, allowed):
    """Both directions on one fixture, so an inverted rule cannot pass."""
    planted = tmp_path / "threetears" / "evals" / "planted.py"
    planted.parent.mkdir(parents=True)
    planted.write_text(source, encoding="utf-8")

    [(_, module)] = list(named_modules(planted, root=tmp_path))

    assert _allowed(module) is allowed


@pytest.mark.parametrize(
    ("importer", "allowed"),
    [
        ("threetears.evals.transports.fastmcp", True),
        ("threetears.evals.transports", False),
        ("threetears.evals.actions.catalogue", False),
        ("threetears.evals.providers.model", False),
        ("threetears.evals", False),
    ],
)
def test_each_extra_is_admitted_only_inside_its_own_subpackage(importer, allowed):
    """``fastmcp`` is the ``[fastmcp]`` extra's: its transport may name it; the core and the other extra may not."""
    assert _allowed("fastmcp.tools", importer=importer) is allowed


def test_an_extras_subpackage_may_not_name_an_undeclared_dependency():
    """An extra admits its own dependencies and nothing more: the transport may not name another 3tears package."""
    assert _allowed("threetears.models", importer="threetears.evals.transports.fastmcp") is False


def test_a_relative_import_is_judged_where_it_lands(tmp_path):
    """A relative import walking out of the package is a crossing, not an import that names nothing."""
    planted = tmp_path / "threetears" / "evals" / "planted.py"
    planted.parent.mkdir(parents=True)
    planted.write_text("from ..core import thing\nfrom .sibling import other\n", encoding="utf-8")

    named = [module for _, module in named_modules(planted, root=tmp_path)]

    assert named == ["threetears.core", "threetears.evals.sibling"]
    assert [_allowed(module) for module in named] == [False, True]


# ---------------------------------------------------------------------------
# What importing it loads
# ---------------------------------------------------------------------------

#: Imports every engine module in a fresh interpreter, then reports any first-host module loaded.
_LOADS_EVERY_MODULE = (
    "import importlib, pkgutil, sys, threetears.evals as e\n"
    "for m in pkgutil.walk_packages(e.__path__, prefix='threetears.evals.'):\n"
    "    importlib.import_module(m.name)\n"
    "loaded = sorted(n for n in sys.modules if n == 'discodon' or n.startswith('discodon.'))\n"
    "assert len([n for n in sys.modules if n.startswith('threetears.evals.')]) > 50, 'the probe imported nothing'\n"
    "assert not loaded, loaded\n"
)


def test_importing_every_engine_module_loads_no_first_host_module():
    """The transitive closure, which a per-file walk cannot report."""
    proc = _probe(_LOADS_EVERY_MODULE)

    assert proc.returncode == 0, f"importing the engine loaded first-host modules:\n{proc.stderr}"


def test_the_probe_reports_a_load_when_there_is_one():
    """Positive control: the probe program fails when a first-host module is in ``sys.modules``."""
    planted = "import sys, types\nsys.modules['discodon.planted'] = types.ModuleType('discodon.planted')\n"

    proc = _probe(planted + _LOADS_EVERY_MODULE)

    assert proc.returncode != 0 and "discodon.planted" in proc.stderr


# ---------------------------------------------------------------------------
# The host-contract types live in the engine
# ---------------------------------------------------------------------------


def test_the_engine_owns_the_apparatus_error_and_the_external_spend_type():
    """A host raises ``ApparatusError`` and reports ``ExternalSpend``; both are DEFINED here.

    A host re-exports them rather than declaring twins: two classes with one name would let a
    raiser and a catcher disagree with no import error anywhere. Pinned to the stdlib-only contract
    leaves both sides may name without acquiring a dependency on each other's internals.
    """
    from threetears.evals.kernel.host.apparatus import ApparatusError
    from threetears.evals.schema.external_spend import ExternalSpend

    assert ApparatusError.__module__ == "threetears.evals.kernel.host.apparatus"
    assert ExternalSpend.__module__ == "threetears.evals.schema.external_spend"


# ---------------------------------------------------------------------------
# The portable test support
# ---------------------------------------------------------------------------


def test_the_portable_test_support_names_no_barred_package():
    """The factories, the resolver and the toy host name no host package, at any depth.

    An AST walk, because an import deferred into a function body never executes on import and a
    re-weld in that style is the likely one, not the exotic one.
    """
    tests_root = PACKAGE_ROOT / "tests"
    offenders = [
        f"{path.relative_to(tests_root)}:{line} imports {module}"
        for path in _support_files()
        for line, module in named_modules(path, root=REPO_ROOT)
        if any(_matches(module, barred) for barred in BARRED_FROM_TEST_SUPPORT)
    ]

    assert len(_support_files()) > len(PORTABLE_TEST_SUPPORT), "the toy host directory read no module"
    assert not offenders, "portable test support names a host package:\n  " + "\n  ".join(offenders)


def test_importing_the_portable_test_support_loads_no_barred_package():
    """The transitive half: importing the support modules loads no host package in a fresh interpreter."""
    probe = (
        "import sys\n"
        "import packages.evals.tests.factories, packages.evals.tests.import_resolution\n"
        "import packages.evals.tests.host_vocabulary\n"
        "import packages.evals.tests.fixtures.toyhost.run, packages.evals.tests.fixtures.toyhost.campaign\n"
        "import packages.evals.tests.fixtures.toyhost.judge\n"
        f"barred = {BARRED_FROM_TEST_SUPPORT!r}\n"
        "loaded = sorted(n for n in sys.modules for b in barred if n == b or n.startswith(b + '.'))\n"
        "assert not loaded, loaded\n"
    )

    proc = _probe(probe)

    assert proc.returncode == 0, f"importing the portable test support loaded a host package:\n{proc.stderr}"
