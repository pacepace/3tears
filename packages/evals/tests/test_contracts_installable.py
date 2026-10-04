"""The contracts package is closed under import, and installs with nothing but its declared dependencies.

The contracts package is the part of the engine a host implements against -- the models, the host
contract, the scoring rules -- and the one most likely to be consumed alone. Its members are the
modules the package matrix places in contracts -- everything under ``threetears/evals/contracts/``
plus the ``threetears.evals`` root marker
(:func:`~packages.evals.tests.package_placement.modules_placed_in`) -- so membership is the path and
nothing here lists it. The two assertions below are the ones the matrix cannot make.

**Closure.** No member may import a module under ``threetears.evals`` outside the package. The
matrix judges an edge only once its target is placed, so an import of a module still at the
``threetears.evals`` root passes it; this walk refuses that edge too. A member reaching
``threetears/evals/analysis/bundle.py`` names nothing foreign and is clean by every import canary,
and it is still a module that arrives broken wherever contracts is consumed alone.

**Installability.** Every member imports in an interpreter whose importable world is the standard
library, pydantic (with pydantic's own dependencies), ``threetears.observe`` (which has none), and
the staged package -- with this repository off ``sys.path`` entirely. That is the claim an AST walk
is structurally unable to check: a walk sees the imports a file *names*, never whether the thing
they name can be loaded. A member whose module-level code reads host configuration, or whose
pydantic model resolves an annotation only a host defines, names nothing and still fails to import.

**Nothing is downloaded.** The isolated environment is built from what is already present:
pydantic's requirement closure is resolved from installed distribution metadata and symlinked into
one directory, ``threetears.observe`` is copied from its workspace source tree, the members are
copied into another, and the probe runs with ``-S`` (no site-packages) and ``-P`` (no implicit
``cwd``) so those directories plus the standard library are the whole of ``sys.path``.
"""

from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterable
from importlib import metadata
from pathlib import Path


from packages.evals.tests.import_resolution import absolute_module
from packages.evals.tests.package_placement import modules_placed_in

#: The package's ``src``: the directory holding the ``threetears`` namespace. Member paths are relative to it.
SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"

#: ``threetears.observe``'s source, in the workspace: the one family dependency contracts takes.
OBSERVE_SOURCE = Path(__file__).resolve().parents[2] / "observe" / "src" / "threetears" / "observe"


def _members() -> list[str]:
    """The contracts package, as source-relative paths: every module its path places in contracts.

    Returns:
        The member files, sorted.
    """
    return sorted(str(path.relative_to(SOURCE_ROOT)) for path in modules_placed_in(SOURCE_ROOT, "contracts").values())


#: Requirement names, for walking pydantic's dependency closure out of installed metadata.
#: A requirement line is ``name[extras] specifier ; marker``; the name is everything up to the
#: first character that cannot be in one.
_REQUIREMENT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _dotted(rel: str) -> str:
    """Source-relative module path to the dotted name it is imported under.

    Args:
        rel: A path like ``threetears/evals/contracts/models.py`` or ``threetears/evals/__init__.py``.

    Returns:
        The dotted module name; a package's ``__init__`` resolves to the package itself.
    """
    return rel.removesuffix(".py").replace("/", ".").removesuffix(".__init__")


def _module_file(module: str) -> str | None:
    """The source-relative file a dotted name resolves to, or ``None`` if it names no module.

    An ``ImportFrom`` contributes both the module it names and one candidate per imported
    alias, because ``from threetears.evals.run import ceilings`` and
    ``from threetears.evals.contracts.models import CostCapOrigin`` are written identically and only the
    tree says which is a module. A name that resolves to no file is a symbol, and symbols are
    carried by the module that defines them.

    Args:
        module: A dotted name.

    Returns:
        The source-relative path of the ``.py`` file it names, or ``None``.
    """
    parts = module.split(".")
    root = SOURCE_ROOT
    leaf = root.joinpath(*parts).with_suffix(".py")
    if leaf.is_file():
        return str(leaf.relative_to(root))
    package = root.joinpath(*parts, "__init__.py")
    if package.is_file():
        return str(package.relative_to(root))
    return None


def _eval_files_named(rel: str) -> list[str]:
    """Every file under ``threetears/evals/`` that ``rel`` names in an import.

    By AST rather than by importing, for the reason the seam canaries give: the question is
    what the file *names*, and a walk sees through ``TYPE_CHECKING`` guards, deferred imports
    inside functions, and branches this interpreter happens not to take. All three still have
    to resolve in the cut package — a ``TYPE_CHECKING`` import is a module the package's type
    checkers need, and a function-local one is a module its callers reach.

    Args:
        rel: Source-relative path of the member to walk.

    Returns:
        Source-relative paths under ``threetears/evals/``, deduplicated, in sorted order.
    """
    path = SOURCE_ROOT / rel
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    named: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            candidates = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            module = absolute_module(path, node, root=SOURCE_ROOT)
            candidates = [module, *(f"{module}.{alias.name}" for alias in node.names)]
        else:
            continue
        for candidate in candidates:
            if not candidate.startswith("threetears.evals"):
                continue
            resolved = _module_file(candidate)
            if resolved is not None and resolved != rel:
                named.add(resolved)
    return sorted(named)


def _closure_breaches(members: Iterable[str], reaches: Callable[[str], Iterable[str]]) -> list[tuple[str, str]]:
    """Every ``(member, imported non-member)`` pair in the set.

    A pure function over a membership set and a reach function, so the assertion that matters
    can be exercised against a set that IS broken — see
    :func:`test_the_closure_assertion_names_the_pair_that_breaks_it`. A check written inline
    against the real tree can only ever be observed passing.

    Args:
        members: The admitted set, as source-relative paths.
        reaches: What one member imports, as source-relative paths under ``threetears/evals/``.

    Returns:
        The offending pairs, sorted; empty when the set is closed.
    """
    admitted = set(members)
    return sorted((member, reached) for member in admitted for reached in reaches(member) if reached not in admitted)


def test_the_contracts_package_is_closed_under_import():
    """No member of the contracts package imports an eval module outside it.

    The property that decides whether the package can be consumed alone. A member importing a
    non-member is a module that arrives broken wherever contracts travels without the rest: the
    import is written, the target was left behind, and the failure surfaces in the consumer as an
    ``ImportError``.

    The pair is what the failure prints, because the pair is what the reader has to act on.
    Which half moves is a judgement — move the imported module into contracts, or move the
    import — and only one of the two is usually right: ``analysis/bundle.py`` is a pipeline, so a
    member reaching it is an import to move rather than a module to bring in.
    """
    breaches = _closure_breaches(_members(), _eval_files_named)

    assert not breaches, (
        "The contracts package is not closed under import. These members name an engine module outside "
        "threetears/evals/contracts/, so contracts consumed alone would ship them broken:\n"
        + "\n".join(f"  {member} -> {reached}" for member, reached in breaches)
        + "\nEither the imported module belongs in contracts and moves there, or the import is what moves."
    )


def test_the_closure_assertion_names_the_pair_that_breaks_it():
    """A member reaching a non-member is caught, and reported as that exact pair.

    The negative half, and it is a test rather than a note because the positive one above is
    green today and stays green if :func:`_closure_breaches` stops finding anything at all — a
    resolver that returned no modules, an ``ast.walk`` narrowed to the wrong node type, a
    membership check comparing paths to dotted names. Each of those reads as "closed".

    Built from the real package and the real walk with exactly one edge injected, so what is
    exercised is the assertion as it runs, not a model of it. The intruder is
    ``analysis/bundle.py`` on purpose: a pipeline module that belongs to a different package, and
    its absence from the package is asserted here rather than assumed, since an intruder that had
    quietly become a member would make this test vacuous.
    """
    offender = "threetears/evals/contracts/models.py"
    intruder = "threetears/evals/analysis/bundle.py"
    members = _members()
    assert offender in members, "the injected edge has to start from a real member"
    assert intruder not in members, f"{intruder} is now a member, so it can no longer stand in for a non-member"

    def reaches(rel: str) -> list[str]:
        named = _eval_files_named(rel)
        return [*named, intruder] if rel == offender else named

    breaches = _closure_breaches(members, reaches)

    assert breaches == [(offender, intruder)], (
        f"A deliberate import of a non-member must redden the closure check as exactly that pair. It reported {breaches!r}."
    )


def _pydantic_distributions() -> list[metadata.Distribution]:
    """Pydantic and its own runtime dependencies, from installed metadata.

    Resolved rather than listed, so the probe's environment tracks the pinned pydantic instead
    of a hand-kept copy of its 2.13 dependency list. Extras are skipped: a requirement guarded
    by an ``extra ==`` marker is not installed by ``pip install pydantic`` and must not be in an
    environment claiming to hold nothing but pydantic.

    Returns:
        The distributions, pydantic first.
    """
    seen: set[str] = set()
    order: list[metadata.Distribution] = []
    queue = ["pydantic"]
    while queue:
        name = queue.pop(0)
        key = name.lower().replace("_", "-")
        if key in seen:
            continue
        seen.add(key)
        dist = metadata.distribution(name)
        order.append(dist)
        for requirement in dist.requires or ():
            specifier, _, marker = requirement.partition(";")
            if "extra" in marker:
                continue
            matched = _REQUIREMENT_NAME.match(specifier.strip())
            if matched:
                queue.append(matched.group(0))
    return order


def _stage_the_package(stage: Path) -> None:
    """Copy the contracts package, and the one family package it depends on, into ``stage``.

    The members keep their own import paths under the ``threetears`` namespace, which carries no
    ``__init__.py`` of its own, exactly as an installed family package does not. ``threetears.observe``
    is copied whole from its workspace source: it declares no dependencies, so it brings nothing else.

    Args:
        stage: Directory to build the importable tree in.
    """
    for rel in _members():
        destination = stage / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SOURCE_ROOT / rel, destination)
    shutil.copytree(OBSERVE_SOURCE, stage / "threetears" / "observe", ignore=shutil.ignore_patterns("__pycache__"))


def _stage_pydantic(deps: Path) -> None:
    """Symlink pydantic and its dependencies into ``deps`` as the only third party on the path.

    Symlinked rather than copied because a compiled extension (``pydantic_core``) is the point:
    the probe must load the real wheel, not a reconstruction of it. Nothing is downloaded — every
    distribution here is one this environment already has.

    Args:
        deps: Directory to build the dependency path entry in.
    """
    for dist in _pydantic_distributions():
        site = Path(dist.locate_file(""))
        tops = {file.parts[0] for file in (dist.files or ()) if not file.parts[0].startswith("..")}
        for top in sorted(tops):
            link = deps / top
            if not link.exists():
                link.symlink_to(site / top)


#: What the isolated interpreter is asked to do: import every member, then report what it
#: loaded and from where. It judges nothing — the test does — because a probe that asserts is a
#: probe whose failure arrives as a return code with no way to say which module and why.
_INSTALL_PROBE = """
import importlib, json, sys, sysconfig

stage, deps, payload = sys.argv[1], sys.argv[2], sys.argv[3]
homes = (stage, deps, sysconfig.get_paths()["stdlib"], sysconfig.get_paths()["platstdlib"])

failed = {}
for name in json.loads(payload):
    try:
        importlib.import_module(name)
    except Exception as exc:
        failed[name] = f"{type(exc).__name__}: {exc}"

outside = sorted(
    f"{name} <- {module.__file__}"
    for name, module in list(sys.modules.items())
    if getattr(module, "__file__", None) and not module.__file__.startswith(homes)
)
print(json.dumps({"failed": failed, "outside": outside, "path": sys.path}))
"""


def test_every_contracts_member_imports_with_nothing_but_its_declared_dependencies(tmp_path: Path) -> None:
    """The contracts package loads in an interpreter that has the standard library, pydantic and observe.

    The claim the AST canaries cannot reach. They see the imports a file *names*; this sees whether
    the named thing loads when nothing else is there. Those come apart in both directions:
    module-level code that reads host configuration names nothing foreign and still fails here, and
    a pydantic model whose annotation resolves against a host type is a clean walk and a red probe.

    **The isolation is asserted, not assumed.** ``-S`` drops site-packages, ``-P`` drops the
    implicit ``cwd`` entry, and the probe reports the ``sys.path`` it actually ran with plus every
    loaded module whose file is outside the staged package, the dependency directory and the
    standard library. A probe that silently kept this repository on its path would import
    everything and prove nothing.

    Args:
        tmp_path: pytest's per-test directory, which is where the environment is built.
    """
    stage = tmp_path / "package"
    deps = tmp_path / "deps"
    stage.mkdir()
    deps.mkdir()
    _stage_the_package(stage)
    _stage_pydantic(deps)

    modules = sorted(_dotted(rel) for rel in _members())
    proc = subprocess.run(
        [sys.executable, "-S", "-P", "-c", _INSTALL_PROBE, str(stage), str(deps), json.dumps(modules)],
        capture_output=True,
        text=True,
        # No inherited environment at all: PYTHONPATH is the whole of what this interpreter may
        # import beyond its standard library, and an inherited PYTHONHOME or PYTHONSTARTUP would
        # make the isolation a property of the developer's shell.
        env={"PYTHONPATH": f"{stage}{os.pathsep}{deps}"},
        cwd=tmp_path,
        # Under pytest's own 30s ceiling, so an interpreter that wedges fails HERE naming this
        # test rather than taking the xdist worker down with no traceback.
        timeout=20,
    )

    assert proc.returncode == 0, f"the isolated probe did not finish:\n{proc.stdout}\n{proc.stderr}"
    report = json.loads(proc.stdout.splitlines()[-1])

    assert not report["failed"], (
        "These contracts modules do not import with only their declared dependencies available, so "
        "the contracts package does not load standalone:\n  "
        + "\n  ".join(f"{name}: {error}" for name, error in sorted(report["failed"].items()))
    )
    assert not report["outside"], (
        "The probe loaded modules from outside the staged package, its dependencies and the standard "
        "library, so it proved less than it claims:\n  " + "\n  ".join(report["outside"])
    )
    assert str(SOURCE_ROOT.parents[2]) not in " ".join(report["path"]), (
        f"the repository was on the probe's sys.path, which makes the result meaningless: {report['path']}"
    )
