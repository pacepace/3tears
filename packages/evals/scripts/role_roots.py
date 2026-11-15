"""The rule that gives every public name of ``3tears-evals`` its root after the re-root by role (issue #1199).

Dev tooling for ``3tears-evals``: it lives beside the package, not in it, and is not installed. ``reroot_map.py``
renders what :func:`plan` decides as the old-to-new map the rewrite script reads, and the list of names that leave the
public roots. Nothing here moves a name: the cut-over (chunk X3) does, from the map.

**The five roots.** Fifteen layer roots become five role roots:

=============================  ==========================================================================
``threetears.evals``           package metadata, and every ``typed`` name (what a newcomer writes)
``threetears.evals.host``      what a host binds, calls, raises, subclasses or passes to the engine
``threetears.evals.shapes``    what the engine hands a host to read
``threetears.evals.testing``   what a host runs in its own tests
``threetears.evals.actions``   the catalogue, its parts, its transport and its outcome types
=============================  ==========================================================================

**The rule** (design-reroot-cutover.md §1.1). The first rule that holds places a name. Kinds are the ones the snapshot
generator writes into the tier map (``surface_kinds.py``); observed use is read from the criterion-2 sources only (a
host's production code, the shipped examples and the other packages' source), never from tests.

1. Package metadata (:data:`METADATA`): the package root.
2. Kind ``typed``: the package root.
3. Kind ``tests``: ``testing``.
4. Defined under ``actions/`` or ``transports/``: ``actions``.
5. Kind ``bound``, ``invoked`` or ``action``, or a ``received`` name that observed use shows a host raising,
   subclassing or passing to the engine: ``host``.
6. Kind ``received`` otherwise: ``shapes``.

**Reached by definition.** A stored document class (a store receives it) and an action's result type (a caller
receives it) count as reached, whatever imports them, and are placed on ``shapes``. An action's parameter model does
not count: the catalogue serves its JSON schema.

**Provisional names** (owner ruling on Q1, 2026-11-15: they leave). A provisional name that a declared name reaches,
transitively, through a signature, a field, a base, a raised error or a docstring role (``surface_kinds.edges`` and
``surface_kinds.doc_edges``, the walk the admission criteria use), that a host's production code imports (it is
reached itself, and what it reaches is), or that is reached by definition, is placed by
``surface_kinds.kind_by_rule`` where that decides, and as ``received`` otherwise (a function, which nothing reads,
as ``invoked``). It keeps its provisional tier. Any other provisional name leaves the public roots, to be imported
from its home module. :data:`UNREACHED` = ``"stay"`` keeps them instead, placed the same way; it is the only switch.

**The debt registers.** Of ``AWAITING_DEMOTION``, the three in :data:`KEPT_AS_TESTS` stay public as the ``tests``
kind (§4.2); every other is demoted, with the replacement :data:`DEMOTED` records, and so is a declared name that only
demoted names reach (§4.1). ``AWAITING_SHIPPED_USE`` names stay and are placed by the rule.

    uv run python packages/evals/scripts/role_roots.py            # print the counts per root
    uv run python packages/evals/scripts/role_roots.py --record-host discodon /path/to/checkout
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import importlib
import inspect
import json
import subprocess
import typing
from collections.abc import Callable, Iterable, Iterator, Mapping
from functools import cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from threetears.evals.actions import definition_drift_actions, eval_catalogue

from public_surface import PACKAGE, PACKAGE_ROOT, TIER_MAP_PATH, SurfaceEntry, declared, defining_module, load_snapshot
from surface_kinds import (
    AWAITING_DEMOTION,
    TYPED,
    Kind,
    constant_edges,
    doc_edges,
    edges,
    kind_by_rule,
    objects_of,
)

# --- the roots -----------------------------------------------------------------------------------------------------

#: The package root: metadata and every ``typed`` name.
ROOT = PACKAGE
HOST = f"{PACKAGE}.host"
SHAPES = f"{PACKAGE}.shapes"
TESTING = f"{PACKAGE}.testing"
ACTIONS = f"{PACKAGE}.actions"

#: The five role roots, in the order a reader meets them.
ROLE_ROOTS: tuple[str, ...] = (ROOT, HOST, SHAPES, TESTING, ACTIONS)

#: The rule that placed a name (§1.1), by number.
Rule = Literal[1, 2, 3, 4, 5, 6]

#: Each rule's root.
RULE_ROOTS: Mapping[Rule, str] = MappingProxyType({1: ROOT, 2: ROOT, 3: TESTING, 4: ACTIONS, 5: HOST, 6: SHAPES})

#: Rule 1: package metadata, by its spelling today.
METADATA: frozenset[str] = frozenset(
    {
        f"{PACKAGE}.__version__",
        f"{PACKAGE}.PUBLIC_ROOTS",
        f"{PACKAGE}.ROOT_EXTRAS",
        f"{PACKAGE}.RootExtra",
        f"{PACKAGE}.surface_tier",
        f"{PACKAGE}.declared_names",
    }
)

#: The two names the cut-over renames (§1.2): no root fails to import without an extra once the fastmcp names are
#: lazy, so the map is keyed by qualified name, and a root-keyed name would be false.
RENAMED: Mapping[str, str] = MappingProxyType(
    {f"{PACKAGE}.ROOT_EXTRAS": f"{PACKAGE}.EXTRAS", f"{PACKAGE}.RootExtra": f"{PACKAGE}.Extra"}
)

#: How each renamed name's shape changed, which a host checks by hand at each use.
RENAMED_SHAPES: Mapping[str, str] = MappingProxyType(
    {
        f"{PACKAGE}.ROOT_EXTRAS": "keyed by each lazy name's qualified spelling, not by root",
        f"{PACKAGE}.RootExtra": "names the lazy names an extra serves, not a root",
    }
)

#: Provisional names given a kind by the design, which no rule of ``surface_kinds`` gives them, so they are placed
#: whether or not a declared name reaches them (§1.1, §4.4). ``Log`` is what an eval file writes to keep a line beside
#: its scores: typed, at the root, still provisional. The seven kits are admitted to ``tests`` under ruling R-a: a
#: host's own tests run them (its reader-conformance and keyless-matrix tests).
KIND_EXCEPTIONS: Mapping[str, Kind] = MappingProxyType(
    {
        f"{PACKAGE}.Log": "typed",
        f"{PACKAGE}.testing.READER_CONFORMANCE_CASES": "tests",
        f"{PACKAGE}.testing.ReaderConformanceCase": "tests",
        f"{PACKAGE}.testing.ReaderSample": "tests",
        f"{PACKAGE}.testing.RESULT_READINGS_CONFORMANCE_CASES": "tests",
        f"{PACKAGE}.testing.ResultReadingsConformanceCase": "tests",
        f"{PACKAGE}.run.resolve_constructor": "tests",
        f"{PACKAGE}.run.callers_missing_the_constructor": "tests",
    }
)

#: The three names of ``AWAITING_DEMOTION`` that stay public as the ``tests`` kind (§4.2): the only door a host's own
#: gate and conformance tests have to what they do (criteria 1 to 3 under ruling R-a), until the host proof (X4,
#: part 1) shows an action does it.
KEPT_AS_TESTS: frozenset[str] = frozenset(
    {
        f"{PACKAGE}.run.start_run",
        f"{PACKAGE}.run.launch_as_group",
        f"{PACKAGE}.analysis.assemble_context_bundle",
    }
)

#: Every other name of ``AWAITING_DEMOTION``, with what a host does instead (§4.1). Demoted, it stays a non-underscore
#: name in its home module, unpublic by its module. An action is called from Python through the catalogue.
DEMOTED: Mapping[str, str] = MappingProxyType(
    {
        f"{PACKAGE}.analysis.report_html": 'host.report_read(..., format="html"), or the report_read action',
        f"{PACKAGE}.ops.await_job": "host.job_manager.wait_for(ids), or the job_poll action",
        f"{PACKAGE}.quick.run_cli": "the command python -m threetears.evals",
        f"{PACKAGE}.run.quote_launch": "the launch_estimate action",
        f"{PACKAGE}.ops.judge_case_author": "the judge_case_author action",
        f"{PACKAGE}.run.mint_case_set": "the case_set_mint action",
        f"{PACKAGE}.run.regrade": "the result_regrade action",
        f"{PACKAGE}.negotiation.closest_settings": "host.negotiate",
        f"{PACKAGE}.ops.case_rename": "the cases_rename action",
        f"{PACKAGE}.analysis.gate_verdicts": "Result.gate()",
        f"{PACKAGE}.ops.campaign_export": "the campaign_export action, or python -m threetears.evals.harness export",
        f"{PACKAGE}.run.admit_template": "host.create_template or host.update_template",
        f"{PACKAGE}.kernel.CappedCompletion": "none needed: the engine applies the cap",
        f"{PACKAGE}.run.load_seed_corpus": "host.seed_eval_definitions; the definitions_drift and definitions_reset "
        "actions",
        f"{PACKAGE}.actions.render_jobs_started": "none needed: the catalogue renders results",
        f"{PACKAGE}.ops.case_set_mint": "the case_set_mint action",
        f"{PACKAGE}.ops.case_sets_list": "the case_sets_list action",
        f"{PACKAGE}.kernel.host.load_bar_registry": "build host.BarRegistry",
        f"{PACKAGE}.transports.fastmcp.CatalogueTool": "FastMCP's own tool listing on the mounted server",
    }
)

#: The six submodules the cut-over renames (§1.5): each shares its name with a public function of the root that holds
#: it, so a dotted spelling of the function would also be a prefix of the module's, and a rewrite by prefix would never
#: converge. The new names are proposed; ``tests/test_no_name_shadows_a_module.py`` decides them.
MODULE_RENAMES: Mapping[str, str] = MappingProxyType(
    {
        f"{PACKAGE}.run.regrade": f"{PACKAGE}.run.regrading",
        f"{PACKAGE}.run.definition_drift": f"{PACKAGE}.run.drift",
        f"{PACKAGE}.run.judge_kind": f"{PACKAGE}.run.judge_as_kind",
        f"{PACKAGE}.analysis.campaign_readings": f"{PACKAGE}.analysis.campaign_read",
        f"{PACKAGE}.analysis.judge_drift": f"{PACKAGE}.analysis.judge_drift_read",
        f"{PACKAGE}.analysis.judge_kind_readings": f"{PACKAGE}.analysis.judge_kind_read",
    }
)

#: Names a ruling places on another root than the rule would (design-reroot-cutover.md §9, Q2, answered 2026-11-15).
#: ``ProviderError`` is what a host catches around a model call and what the provider seam (#1013) keeps in evals: a
#: host binds it, so it is on ``host``, though no recorded host raises or subclasses it yet.
PLACED_BY_RULING: Mapping[str, Rule] = MappingProxyType({f"{PACKAGE}.providers.ProviderError": 5})

#: Former roots whose own ``__init__`` defines public names. The cut-over empties a former root's ``__init__`` (it
#: serves nothing and raises naming the new spelling), and a module ``__getattr__`` never fires for a name the module
#: defines, so what it defines moves to a submodule first; the name is proposed, and the cut-over decides it. Unlike
#: :data:`MODULE_RENAMES` this moves no module path: the package keeps its path and its definitions leave it.
INIT_DEFINITIONS_MOVE: Mapping[str, str] = MappingProxyType(
    {f"{PACKAGE}.transports.fastmcp": f"{PACKAGE}.transports.fastmcp.server"}
)

#: What becomes of a provisional name nothing declared reaches. Owner ruling on Q1 (2026-11-15): it leaves the public
#: roots. ``"stay"`` places it by the rule like a reached one.
Unreached = Literal["leave", "stay"]
UNREACHED: Unreached = "leave"

#: The modules whose kind decides rule 4: the catalogue, its parts and its transport.
ACTIONS_HOMES: tuple[str, ...] = (f"{PACKAGE}.actions", f"{PACKAGE}.transports")

#: What a host does with a name that rule 5 reads as binding: raises it, subclasses it, or passes it to the engine.
Use = Literal["raised", "subclassed", "passed"]

#: Where each host's observed use of the package is recorded (:func:`record_host`), beside its import list.
HOST_OBSERVED = PACKAGE_ROOT / "tests" / "fixtures" / "host_observed_use"

#: The host import lists ``record_host_imports.py`` writes: production, and tests.
HOST_IMPORTS = PACKAGE_ROOT / "tests" / "fixtures" / "host_imports"
HOST_TEST_IMPORTS = PACKAGE_ROOT / "tests" / "fixtures" / "host_test_imports"

#: The repository, whose other packages' sources are a criterion-2 source.
REPO = PACKAGE_ROOT.parents[1]

#: The package source, read for stored document classes and action jobs.
SOURCE = PACKAGE_ROOT / "src" / "threetears" / "evals"


# --- observed use ----------------------------------------------------------------------------------------------------


def _dotted(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute) and (base := _dotted(node.value)) is not None:
        return f"{base}.{node.attr}"
    return None


def observed_uses(source: str, filename: str, is_module: Callable[[str], bool]) -> set[tuple[str, Use]]:
    """Each package name one file raises, subclasses, or passes as an argument to a package callable, as
    ``(root.name as imported, use)``. A name is the package's when the file binds it by ``from threetears.evals… import``,
    or reads it off a package module it binds.

    :param source: The file's text.
    :param filename: Its path, for a parse error.
    :param is_module: Whether a dotted path is a module of the package, not a name in one.
    """
    tree = ast.parse(source, filename=filename)
    names: dict[str, str] = {}
    modules: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and (node.module or "").startswith(PACKAGE):
            for alias in node.names:
                qualified = f"{node.module}.{alias.name}"
                (modules if is_module(qualified) else names)[alias.asname or alias.name] = qualified
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(PACKAGE):
                    if alias.asname:
                        modules[alias.asname] = alias.name
                    else:
                        modules[alias.name] = alias.name

    def resolve(node: ast.expr) -> str | None:
        text = _dotted(node.value if isinstance(node, ast.Subscript) else node)
        if text is None:
            return None
        if text in names:
            return names[text]
        head, _, attribute = text.rpartition(".")
        for bound, module in modules.items():
            if head == bound or head.startswith(f"{bound}."):
                path = module + head[len(bound) :]
                if is_module(path):
                    return f"{path}.{attribute}"
        return None

    found: set[tuple[str, Use]] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Raise) and node.exc is not None:
            raised = resolve(node.exc.func if isinstance(node.exc, ast.Call) else node.exc)
            if raised is not None:
                found.add((raised, "raised"))
        elif isinstance(node, ast.ClassDef):
            for base in node.bases:
                if (subclassed := resolve(base)) is not None:
                    found.add((subclassed, "subclassed"))
        elif isinstance(node, ast.Call) and resolve(node.func) is not None:
            for value in (*node.args, *(keyword.value for keyword in node.keywords)):
                if isinstance(value, ast.Starred):
                    value = value.value
                if (passed := resolve(value)) is not None:
                    found.add((passed, "passed"))
    return found


@cache
def _is_module(path: str) -> bool:
    try:
        importlib.import_module(path)
    except ImportError:
        return False
    return True


def repo_sources() -> list[Path]:
    """The criterion-2 sources in this repository: the shipped examples and every other package's source."""
    found = sorted((PACKAGE_ROOT / "examples").rglob("*.py"))
    for package in sorted((REPO / "packages").rglob("pyproject.toml")):
        source = package.parent / "src"
        if package.parent != PACKAGE_ROOT and source.is_dir():
            found += sorted(source.rglob("*.py"))
    return found


def repo_observed_uses() -> set[tuple[str, Use]]:
    """What the shipped examples and the other packages' source raise, subclass or pass of the package."""
    found: set[tuple[str, Use]] = set()
    for path in repo_sources():
        text = path.read_text(encoding="utf-8")
        if PACKAGE in text:
            found |= observed_uses(text, str(path), _is_module)
    return found


def recorded_observed_uses() -> set[tuple[str, Use]]:
    """What each host's production code raises, subclasses or passes, as :func:`record_host` recorded it."""
    found: set[tuple[str, Use]] = set()
    for path in sorted(HOST_OBSERVED.glob("*.txt")):
        for raw in path.read_text(encoding="utf-8").splitlines():
            if raw.strip() and not raw.startswith("#"):
                name, use = raw.split("\t")
                if use not in typing.get_args(Use):
                    raise ValueError(f"{path.name}: {use!r} is not a use ({', '.join(typing.get_args(Use))})")
                found.add((name, typing.cast("Use", use)))
    return found


def _recorded_commit(path: Path, host: str) -> str:
    for line in path.read_text(encoding="utf-8").splitlines()[:6]:
        if line.startswith(f"# From {host} "):
            return line.split(" at ", 1)[1].split()[0]
    raise ValueError(f"{path} names no {host} commit")


def record_host(host: str, checkout: Path) -> Path:
    """Write what ``host``'s production code raises, subclasses or passes of the package, read at the commit its
    import list was recorded at (so the two agree), and return the file written. Reads the checkout's committed tree
    only (``git ls-files``, ``git show``), never its working tree, and changes nothing there.

    :raises ValueError: The checkout's ``HEAD`` is not the commit the host's import list names: re-record that list
        first (``record_host_imports.py``).
    """
    import record_host_imports  # noqa: PLC0415 - dev tooling beside this script, needed only to record

    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(checkout), *args], check=True, capture_output=True, text=True).stdout

    commit = _recorded_commit(HOST_IMPORTS / f"{host}.txt", host)
    head = git("rev-parse", "--short", "HEAD").strip()
    if not head.startswith(commit) and not commit.startswith(head):
        raise ValueError(f"{checkout} is at {head}, but {host}'s import list was recorded at {commit}: re-record it")
    found: set[tuple[str, Use]] = set()
    for path in record_host_imports.production_files(checkout):
        source = git("show", f"HEAD:{path}")
        if PACKAGE in source:
            found |= observed_uses(source, path, record_host_imports.is_module)
    target = HOST_OBSERVED / f"{host}.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    header = [
        f"# {host}'s production code raising, subclassing or passing a 3tears-evals name, generated by",
        "# packages/evals/scripts/role_roots.py --record-host: do not edit by hand. Production is what",
        "# record_host_imports.py counts; each line is the name as imported, a tab, and the use (rule 5 of the re-root).",
        f"# From {host} at {commit}.",
    ]
    target.write_text("\n".join([*header, *(f"{name}\t{use}" for name, use in sorted(found))]) + "\n", encoding="utf-8")
    return target


def host_import_names(directory: Path = HOST_IMPORTS) -> dict[str, frozenset[str]]:
    """Each host's recorded import list, by host: every line, modules included."""
    return {
        path.stem: frozenset(
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        )
        for path in sorted(directory.glob("*.txt"))
    }


# --- reached by definition -------------------------------------------------------------------------------------------


def _literal_doc_type(node: ast.ClassDef) -> str | None:
    for statement in node.body:
        if (
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.target.id == "doc_type"
            and isinstance(statement.annotation, ast.Subscript)
            and _dotted(statement.annotation.value) in ("Literal", "typing.Literal")
            and isinstance(statement.annotation.slice, ast.Constant)
            and isinstance(statement.annotation.slice.value, str)
        ):
            return statement.annotation.slice.value
    return None


def _module_of_path(path: Path) -> str:
    parts = path.relative_to(SOURCE.parents[1]).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


@cache
def stored_classes() -> Mapping[str, type]:
    """Every stored document class, by its ``doc_type``: each class in the package source that declares
    ``doc_type: Literal["…"]``.

    :raises ValueError: Two classes declare one ``doc_type``.
    """
    found: dict[str, type] = {}
    for path in sorted(SOURCE.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
            if isinstance(node, ast.ClassDef) and (doc_type := _literal_doc_type(node)) is not None:
                klass = getattr(importlib.import_module(_module_of_path(path)), node.name)
                if doc_type in found:
                    raise ValueError(f"doc_type {doc_type!r} is declared by {found[doc_type]} and {klass}")
                found[doc_type] = klass
    return MappingProxyType(dict(sorted(found.items())))


def _job_results() -> Iterator[type]:
    """The result type of each action job the package builds: the return annotation of the ``run`` an
    ``action_job_work(..., run=...)`` call is handed, which ``result_model`` stores by class name."""
    for path in sorted(SOURCE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        defs = {node.name: node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and _dotted(node.func) == "action_job_work"):
                continue
            run = next((keyword.value for keyword in node.keywords if keyword.arg == "run"), None)
            target = defs.get(run.id) if isinstance(run, ast.Name) else None
            if target is None or target.returns is None:
                continue
            module = importlib.import_module(_module_of_path(path))
            result = getattr(module, ast.unparse(target.returns), None)
            if inspect.isclass(result):
                yield result


@cache
def action_results() -> Mapping[str, type]:
    """Every type an action of the engine hands its caller, by ``module.qualname``: each catalogue action's result
    (the standard catalogue and the definition-drift pair), and each action job's result."""
    actions = [*eval_catalogue().actions, *definition_drift_actions(lambda _key: _never())]
    found = {f"{action.result.__module__}.{action.result.__qualname__}": action.result for action in actions}
    for result in _job_results():
        found[f"{result.__module__}.{result.__qualname__}"] = result
    return MappingProxyType(dict(sorted(found.items())))


def _never() -> Any:
    raise AssertionError("the definition-drift actions are built here only to read their result types")


@cache
def by_definition() -> Mapping[int, str]:
    """Every class reached by definition, by ``id``, with why: a stored document class, or an action's result."""
    found: dict[int, str] = {}
    for doc_type, klass in stored_classes().items():
        found[id(klass)] = f"stored document class ({doc_type})"
    for name, klass in action_results().items():
        found.setdefault(id(klass), f"action result ({name.rsplit('.', 1)[1]})")
    return MappingProxyType(found)


# --- the plan --------------------------------------------------------------------------------------------------------

#: What becomes of an old spelling: ``kept`` (its spelling does not change), ``name`` (it moves to a role root),
#: ``renamed`` (it moves and changes name), ``demoted`` (it leaves the roots, with a replacement), ``left`` (an
#: unreached provisional name, leaving the roots).
Record = Literal["kept", "name", "renamed", "demoted", "left"]


@dataclasses.dataclass(frozen=True)
class Placement:
    """One public spelling of today and what the cut-over makes of it.

    :ivar old: Its spelling today, ``root.name``.
    :ivar new: Its public spelling after the cut, or, for a name that leaves, its home spelling (the module that
        defines it, after :data:`MODULE_RENAMES`).
    :ivar record: What becomes of it (:data:`Record`).
    :ivar rule: The rule that placed it, or ``None`` for a name that leaves.
    :ivar kind: The kind it is placed as, or ``None`` for a name that leaves.
    :ivar tier: Its tier today.
    :ivar home: Its spelling in the module that defines it, after the cut's module moves.
    :ivar note: For a demoted name, its replacement; for a name that leaves, why it is unreached; for a renamed
        name, how its shape changed; else empty.
    """

    old: str
    new: str
    record: Record
    rule: Rule | None
    kind: Kind | None
    tier: str
    home: str
    note: str = ""


@dataclasses.dataclass(frozen=True)
class Added:
    """A name reached by definition that no root exports today, public after the cut.

    :ivar new: Its public spelling after the cut.
    :ivar home: The module that defines it.
    :ivar why: Why it is reached.
    """

    new: str
    home: str
    why: str


@dataclasses.dataclass(frozen=True)
class Plan:
    """Where every public name goes.

    :ivar placements: One per public spelling of today, in the snapshot's order.
    :ivar added: The names reached by definition that no root exports today.
    :ivar roots: Each former root (a layer root that is no role root) → the role roots its names went to.
    :ivar modules: The submodules renamed (:data:`MODULE_RENAMES`).
    :ivar unreached: The setting it was planned under.
    """

    placements: tuple[Placement, ...]
    added: tuple[Added, ...]
    roots: Mapping[str, tuple[str, ...]]
    modules: Mapping[str, str]
    unreached: Unreached

    def public(self) -> dict[str, list[Placement]]:
        """Each public spelling after the cut → the old spellings it serves."""
        found: dict[str, list[Placement]] = {}
        for placement in self.placements:
            if placement.record not in ("demoted", "left"):
                found.setdefault(placement.new, []).append(placement)
        return found

    def counts(self) -> dict[str, int]:
        """How many distinct public names each role root holds after the cut, the added ones included."""
        roots = {root: 0 for root in ROLE_ROOTS}
        for new in [*self.public(), *(added.new for added in self.added)]:
            roots[new.rsplit(".", 1)[0]] += 1
        return roots

    def declared_counts(self) -> dict[str, int]:
        """How many distinct names declared today each role root holds after the cut."""
        roots = {root: 0 for root in ROLE_ROOTS}
        for new, olds in self.public().items():
            if any(placement.tier != "provisional" for placement in olds):
                roots[new.rsplit(".", 1)[0]] += 1
        return roots

    def leaving(self) -> list[Placement]:
        """Every spelling that leaves the roots, demoted or unreached."""
        return [placement for placement in self.placements if placement.record in ("demoted", "left")]


def _tier_kinds() -> Mapping[str, Kind]:
    data = json.loads(TIER_MAP_PATH.read_text(encoding="utf-8"))
    return MappingProxyType(data["kinds"])


def home_of(qualified: str, obj: object) -> str:
    """The spelling of a public name in the module that defines it, after :data:`MODULE_RENAMES`.

    :raises LookupError: Neither its source nor the object says where it is defined.
    """
    root, name = qualified.rsplit(".", 1)
    found = defining_module(root, name)
    if found is None and (inspect.isclass(obj) or inspect.isroutine(obj)):
        module = getattr(obj, "__module__", "")
        if getattr(importlib.import_module(module), obj.__name__, None) is obj:  # type: ignore[attr-defined]
            found = (module, obj.__name__)  # type: ignore[attr-defined]
    if found is None:
        raise LookupError(f"{qualified}: no module assigns it")
    module, defined = found
    return f"{renamed_module(INIT_DEFINITIONS_MOVE.get(module, module))}.{defined}"


def renamed_module(module: str) -> str:
    """A module path after :data:`MODULE_RENAMES`."""
    for old, new in MODULE_RENAMES.items():
        if module == old or module.startswith(f"{old}."):
            return new + module[len(old) :]
    return module


def _holders(objects: Mapping[str, Any]) -> dict[int, set[tuple[Any, str]]]:
    found: dict[int, set[tuple[Any, str]]] = {}
    for name, obj in objects.items():
        for position, each in edges(obj):
            if each is not obj:
                found.setdefault(id(each), set()).add((position, name))
    return found


def _walk(starts: Iterable[object]) -> dict[int, object]:
    """Every engine type a start reaches, transitively, through ``edges`` and ``doc_edges``, by ``id``; the starts
    included."""
    reached: dict[int, object] = {}
    frontier = []
    for obj in starts:
        if id(obj) not in reached:
            reached[id(obj)] = obj
            frontier.append(obj)
    while frontier:
        following = []
        for obj in frontier:
            for found in (*(each for _, each in edges(obj)), *doc_edges(obj)):
                if id(found) not in reached:
                    reached[id(found)] = found
                    following.append(found)
        frontier = following
    return reached


def _rule(kind: Kind, home: str, observed: bool, metadata: bool) -> Rule:
    if metadata:
        return 1
    if kind == "typed":
        return 2
    if kind == "tests":
        return 3
    if any(home == prefix or home.startswith(f"{prefix}.") for prefix in ACTIONS_HOMES):
        return 4
    if kind in ("bound", "invoked", "action") or observed:
        return 5
    return 6


#: The reference kinds of a name that is a type: what a signature, a field or a docstring can name.
_TYPE_KINDS = ("model", "dataclass", "typed dict", "class", "protocol", "exception", "enum", "literal", "type alias")


def plan(
    unreached: Unreached = UNREACHED,
    *,
    observed: Iterable[tuple[str, Use]] | None = None,
    host_production: Iterable[str] | None = None,
) -> Plan:
    """Place every public name of today's snapshot by the rule.

    :param unreached: What becomes of a provisional name nothing declared reaches (:data:`UNREACHED`).
    :param observed: What criterion-2 sources raise, subclass or pass; by default the recorded hosts' and this
        repository's (:func:`recorded_observed_uses`, :func:`repo_observed_uses`).
    :param host_production: The names hosts' production code imports, which keep a declared name only demoted names
        reach; by default the recorded lists.
    :raises ValueError: The registers and :data:`DEMOTED` disagree, the snapshot names something the package does not
        export, one name is two objects, or two names would share one spelling.
    """
    surface = load_snapshot()
    kinds = _tier_kinds()
    awaiting = set(AWAITING_DEMOTION)
    if awaiting != set(DEMOTED) | KEPT_AS_TESTS or set(DEMOTED) & KEPT_AS_TESTS:
        raise ValueError(
            "AWAITING_DEMOTION is no longer DEMOTED and KEPT_AS_TESTS: "
            f"{sorted(awaiting ^ (set(DEMOTED) | KEPT_AS_TESTS))}; update scripts/role_roots.py"
        )
    observed = set(observed) if observed is not None else recorded_observed_uses() | repo_observed_uses()
    production = set(host_production) if host_production is not None else set().union(*host_import_names().values())
    names = list(surface)
    objects = objects_of(names)
    if missing := [name for name in names if name not in objects]:
        raise ValueError(f"the snapshot names what no root exports (regenerate it): {missing[:5]}")

    # One name, one object: a name exported from several roots is one re-export (api/reexports.toml).
    by_bare: dict[str, list[str]] = {}
    for name in names:
        by_bare.setdefault(name.rsplit(".", 1)[1], []).append(name)
    for bare, spellings in by_bare.items():
        if len({id(objects[spelling]) for spelling in spellings}) > 1 and not all(
            surface[spelling].kind.startswith("constant") for spelling in spellings
        ):
            raise ValueError(f"{bare} names two objects: {spellings}")

    is_declared = {name: declared(surface[name].tier) for name in names}
    kind_of: dict[str, Kind] = {name: kinds[name] for name in names if is_declared[name]}
    for name in KEPT_AS_TESTS:
        kind_of[name] = "tests"
    kind_of.update(KIND_EXCEPTIONS)
    demoted = set(DEMOTED)

    # A declared type that only demoted names name leaves with them (§4.1), unless a host's production code imports
    # it or it is reached by definition.
    holders = _holders({name: objects[name] for name in names if is_declared[name]})
    orphans: dict[str, str] = {}
    for name in names:
        obj = objects[name]
        if (
            not is_declared[name]
            or name in demoted
            or name in production
            or id(obj) in by_definition()
            or kind_of[name] == "typed"
            or name in METADATA
            or surface[name].kind not in _TYPE_KINDS
        ):
            continue
        named = {holder for _, holder in holders.get(id(obj), set())} - {name}
        if named and named <= demoted:
            orphans[name] = "only demoted names name it: " + ", ".join(sorted(named))
    demoted |= set(orphans)

    # The provisional names a kept declared name, an admitted kit or a host's production import reaches, or that are
    # reached by definition. A name a host's production code imports is reached itself (owner ruling on Q1).
    after = [
        objects[name]
        for name in names
        if (is_declared[name] or name in KIND_EXCEPTIONS or name in production) and name not in demoted
    ]
    after_constants = [
        each
        for name in names
        if is_declared[name] and name not in demoted and surface[name].kind.startswith("constant")
        for _, each in constant_edges(name)
    ]
    reached = _walk([*after, *after_constants, *_by_definition_objects()])

    observed_bare = {name.rsplit(".", 1)[1] for name, _ in observed}
    placements: dict[str, Placement] = {}
    all_holders = _all_holders(objects)
    leaving_bare: set[str] = set()
    for bare, spellings in by_bare.items():
        obj = objects[spellings[0]]
        home = home_of(spellings[0], obj)
        if any(s in demoted for s in spellings):
            note = DEMOTED.get(next((s for s in spellings if s in DEMOTED), ""), "") or orphans.get(
                next((s for s in spellings if s in orphans), ""), ""
            )
            for spelling in spellings:
                placements[spelling] = Placement(
                    spelling, home, "demoted", None, None, surface[spelling].tier, home, note
                )
            continue
        group_declared = any(is_declared[s] for s in spellings)
        metadata = any(s in METADATA for s in spellings)
        if not group_declared and not metadata and not any(s in KIND_EXCEPTIONS for s in spellings):
            is_reached = id(obj) in reached or id(obj) in by_definition()
            if not is_reached and unreached == "leave":
                leaving_bare.add(bare)
                continue
        candidates: list[tuple[Rule, Kind]] = []
        for spelling in spellings:
            kind = kind_of.get(spelling) or _provisional_kind(spelling, obj, surface, holders)
            candidates.append((_rule(kind, home, bare in observed_bare and kind == "received", metadata), kind))
        rule, kind = min(candidates)
        ruled = [PLACED_BY_RULING[s] for s in spellings if s in PLACED_BY_RULING]
        if ruled:
            rule = ruled[0]
        new_root = RULE_ROOTS[rule]
        for spelling in spellings:
            new = RENAMED.get(spelling) or f"{new_root}.{bare}"
            record: Record = "kept" if new == spelling else "renamed" if spelling in RENAMED else "name"
            placements[spelling] = Placement(
                spelling, new, record, rule, kind, surface[spelling].tier, home, RENAMED_SHAPES.get(spelling, "")
            )

    for bare in leaving_bare:
        for spelling in by_bare[bare]:
            obj = objects[spelling]
            home = home_of(spelling, obj)
            why = _why_unreached(obj, all_holders, leaving_bare, by_bare, demoted)
            placements[spelling] = Placement(spelling, home, "left", None, None, surface[spelling].tier, home, why)

    ordered = tuple(placements[name] for name in names)
    exported = {id(objects[name]) for name in names}
    added = tuple(
        Added(
            f"{SHAPES}.{klass.__name__}",
            f"{renamed_module(INIT_DEFINITIONS_MOVE.get(klass.__module__, klass.__module__))}.{klass.__qualname__}",
            why,
        )
        for klass, why in _by_definition_items()
        if id(klass) not in exported
    )
    roots: dict[str, set[str]] = {}
    for placement in ordered:
        old_root = placement.old.rsplit(".", 1)[0]
        if old_root not in ROLE_ROOTS and placement.record in ("name", "renamed"):
            roots.setdefault(old_root, set()).add(placement.new.rsplit(".", 1)[0])
    result = Plan(
        ordered,
        added,
        MappingProxyType({root: tuple(r for r in ROLE_ROOTS if r in went) for root, went in sorted(roots.items())}),
        MODULE_RENAMES,
        unreached,
    )
    if clashes := spelling_clashes(result):
        raise ValueError("two names would share one spelling: " + "; ".join(clashes))
    return result


def _is_shape(entry: SurfaceEntry) -> bool:
    return not entry.kind.startswith("constant")


def _by_definition_items() -> list[tuple[type, str]]:
    found: dict[int, tuple[type, str]] = {}
    for doc_type, klass in stored_classes().items():
        found.setdefault(id(klass), (klass, f"stored document class ({doc_type})"))
    for name, klass in action_results().items():
        found.setdefault(id(klass), (klass, f"action result ({name.rsplit('.', 1)[1]})"))
    return sorted(found.values(), key=lambda item: (item[0].__name__, item[0].__module__))


def _by_definition_objects() -> list[type]:
    return [klass for klass, _ in _by_definition_items()]


def _provisional_kind(qualified: str, obj: object, surface: Mapping[str, SurfaceEntry], holders: Any) -> Kind:
    """A provisional name's kind: by ``kind_by_rule`` where that decides; a function, which nothing reads, as
    ``invoked``; anything else as ``received``."""
    ruled = kind_by_rule(qualified, obj, surface[qualified], holders, surface)
    if ruled is not None:
        return ruled
    return "invoked" if surface[qualified].kind in ("function", "async function") else "received"


def _all_holders(objects: Mapping[str, Any]) -> dict[int, set[str]]:
    """Every public name that names each engine type, by the type's ``id``."""
    found: dict[int, set[str]] = {}
    for name, obj in objects.items():
        for _, each in (*edges(obj), *((None, doc) for doc in doc_edges(obj))):
            if each is not obj:
                found.setdefault(id(each), set()).add(name)
    return found


def _why_unreached(
    obj: object,
    holders: Mapping[int, set[str]],
    leaving: set[str],
    by_bare: Mapping[str, list[str]],
    demoted: set[str],
) -> str:
    """Why a provisional name is unreached: who among today's public names names it, and what becomes of them."""
    named = holders.get(id(obj), set())
    if not named:
        what = "a function" if inspect.isroutine(obj) else "a constant" if not inspect.isclass(obj) else "a type"
        return f"{what} no public name's signature, field, base, raised error or docstring names"
    gone = {name for name in named if name.rsplit(".", 1)[1] in leaving}
    dropped = {name for name in named if name in demoted}
    parts = []
    if gone:
        parts.append("names that leave too: " + ", ".join(sorted(gone)[:3]) + (" ..." if len(gone) > 3 else ""))
    if dropped:
        parts.append("demoted names: " + ", ".join(sorted(dropped)[:3]) + (" ..." if len(dropped) > 3 else ""))
    other = named - gone - dropped
    if other:
        # Public names that stay, yet reach it only through a provisional name that is itself unreached.
        parts.append("names no declared name reaches: " + ", ".join(sorted(other)[:3]))
    return "named only by " + "; ".join(parts)


def spelling_clashes(planned: Plan) -> list[str]:
    """Each new spelling two different names would share."""
    seen: dict[str, str] = {}
    clashes = []
    for new, olds in planned.public().items():
        bare = {old.old.rsplit(".", 1)[1] for old in olds} - {"ROOT_EXTRAS", "RootExtra"}
        if len(bare) > 1:
            clashes.append(f"{new} <- {sorted(bare)}")
        seen[new] = olds[0].old
    for added in planned.added:
        if added.new in seen:
            clashes.append(f"{added.new} <- {seen[added.new]} and {added.home}")
    return clashes


# --- the adopter's goals ---------------------------------------------------------------------------------------------


def taught_through(rung: int, planned: Plan) -> dict[str, str]:
    """Each name rungs 0 to ``rung`` teach, with its spelling after the cut."""
    after = {placement.old: placement.new for placement in planned.placements}
    return {
        name: after[name] for name, first in sorted(TYPED.items(), key=lambda item: (item[1], item[0])) if first <= rung
    }


# --- the command -----------------------------------------------------------------------------------------------------


def describe(planned: Plan) -> list[str]:
    """The plan's counts and the adopter's names, as lines to print."""
    leaving = planned.leaving()
    lines = [f"unreached provisional names: {planned.unreached}"]
    for root, count in planned.counts().items():
        lines.append(f"{root}\t{count} public\t{planned.declared_counts()[root]} declared")
    lines.append(f"demoted\t{sum(1 for each in leaving if each.record == 'demoted')} spellings")
    lines.append(f"left\t{sum(1 for each in leaving if each.record == 'left')} spellings")
    lines.append(f"added (reached by definition, exported by no root today)\t{len(planned.added)}")
    for rung in (1, 3):
        lines.append(f"through rung {rung}: " + ", ".join(taught_through(rung, planned).values()))
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--unreached", choices=typing.get_args(Unreached), default=UNREACHED)
    parser.add_argument(
        "--record-host",
        nargs=2,
        metavar=("HOST", "CHECKOUT"),
        help="record what a host's production code raises, subclasses or passes, read from its checkout's HEAD",
    )
    arguments = parser.parse_args(argv)
    if arguments.record_host:
        host, checkout = arguments.record_host
        print(record_host(host, Path(checkout)))
        return 0
    for line in describe(plan(arguments.unreached)):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
