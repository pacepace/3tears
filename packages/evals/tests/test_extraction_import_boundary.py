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
    "3tears-observe": ("threetears.observe",),
    "pydantic": ("pydantic",),
}

#: Each optional extra the manifest declares: the subpackage that is the extra (and so the only
#: modules that may name its dependencies), and each of its dependencies' import roots. Checked against
#: the manifest by :func:`test_the_extras_map_is_the_manifests`. A core module naming an extra's
#: dependency would be an ``ImportError`` for every host that installed without the extra.
IMPORT_ROOTS_BY_EXTRA: dict[str, tuple[str, dict[str, tuple[str, ...]]]] = {
    "vega": ("threetears.evals.vega", {"vl-convert-python": ("vl_convert",)}),
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

    Args:
        path: The file to read.
        root: The directory its dotted names are rooted at.

    Yields:
        One pair per module named.
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

    Args:
        source: The probe program; a literal from this file, never anything assembled from input.

    Returns:
        The finished process, for the caller to assert on.
    """
    return subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        timeout=60,
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
        ("from threetears.evals.contracts import models\n", True),
        ("from pydantic import BaseModel\n", True),
        ("from threetears.observe import get_logger\n", True),
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
        ("threetears.evals.vega.render", True),
        ("threetears.evals.vega", True),
        ("threetears.evals.analysis.viz.intent", False),
        ("threetears.evals.vegan", False),
    ],
)
def test_an_extras_dependency_is_admitted_only_inside_its_subpackage(importer, allowed):
    """``vl_convert`` is the ``[vega]`` extra's: the renderer may name it and the core may not."""
    assert _allowed("vl_convert", importer=importer) is allowed


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
    from threetears.evals.contracts.host.apparatus import ApparatusError
    from threetears.evals.contracts.host.spend import ExternalSpend

    assert ApparatusError.__module__ == "threetears.evals.contracts.host.apparatus"
    assert ExternalSpend.__module__ == "threetears.evals.contracts.host.spend"


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
