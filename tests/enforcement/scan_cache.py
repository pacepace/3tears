"""the gates' side of the enforcement scan cache: the code its whole-workspace scans run, and what each reads.

The cache itself, :class:`ScanCache`, and the per-file walk most gates make, ``file_facts``, live in
:mod:`threetears.enforcement.common.scan_cache`, whose docstring says what a value is keyed by,
what is never stored and how concurrent processes share a directory. Here: :func:`gate_scan_cache`,
the cache the gates' fixture hands out, keyed by the code the scans below run, and the key builders
for those scans (the private-binding scan, the public API's surfaces and names), public so they are
tested as the gates use them.
"""

from __future__ import annotations

import ast
import os
import subprocess
from collections.abc import Iterable
from pathlib import Path

from threetears.enforcement.common import clear_parse_cache, find_local_src_roots
from threetears.enforcement.common.scan_cache import (
    CACHE_DIRECTORY,
    ImportIndex,
    ScanCache,
    ScanInputs,
    code_key,
    installed_module_file,
    module_closure,
)
from threetears.enforcement.release import surface_at, surface_now
from threetears.enforcement.underscore_access import PrivateBindingScan, scan_private_bindings
from threetears.observe import BuildOnce

#: the modules whose functions the scans here call; their import closure is the code a value comes from
SCANNER_MODULES = tuple(
    sorted(
        {
            function.__module__
            for function in (
                clear_parse_cache,
                find_local_src_roots,
                surface_at,
                surface_now,
                scan_private_bindings,
                BuildOnce,
                ScanCache,
            )
        }
    )
)


def gate_scan_cache(repo_root: Path) -> ScanCache:
    """the cache a gate in this directory uses: under the repo root, keyed by the code its scans run.

    The code key hashes the import closure of the modules the scans below call into
    (:data:`SCANNER_MODULES`), as installed, so it is the same in every process, with this
    directory's gate shells and helpers, the lockfile and the interpreter's version. The closure's
    imports come from an :class:`ImportIndex` beside the stores, so a warm process reads each
    module of it and parses none.

    :param repo_root: the repo's root
    :ptype repo_root: Path
    :return: the cache
    :rtype: ScanCache
    :raises OSError: when a file of the code cannot be read
    """
    directory = repo_root / CACHE_DIRECTORY
    gates = sorted((repo_root / "tests" / "enforcement").glob("*.py"))
    index = ImportIndex(directory)
    modules = module_closure(SCANNER_MODULES, installed_module_file, index)
    index.save()
    return ScanCache(directory=directory, code=code_key(modules, [*gates, repo_root / "uv.lock"], repo_root))


#: directories a workspace walk never enters, as the scans' own file discovery skips them
_UNSCANNED_DIRECTORIES = frozenset({".venv", "node_modules", ".git", "__pycache__"})

#: config files the private-binding scan reads ruff's per-file ignores from
_RUFF_CONFIG_NAMES = frozenset({"pyproject.toml", "ruff.toml", ".ruff.toml"})


def workspace_files(repo_root: Path) -> tuple[list[Path], list[Path]]:
    """every python file and every ruff config candidate of the repo's own tree, in one walk.

    The walk skips what the scans' discovery skips: ``.venv``, ``node_modules``, ``.git``,
    ``__pycache__`` and nested checkouts (a directory below the root holding its own ``.git``).

    :param repo_root: the repo's root, resolved
    :ptype repo_root: Path
    :return: the python files, and the ``pyproject.toml`` / ``ruff.toml`` / ``.ruff.toml`` files
    :rtype: tuple[list[Path], list[Path]]
    """
    python: list[Path] = []
    configs: list[Path] = []
    for directory, subdirectories, names in os.walk(repo_root):
        base = Path(directory)
        subdirectories[:] = [
            sub
            for sub in subdirectories
            if sub not in _UNSCANNED_DIRECTORIES and not os.path.lexists(os.path.join(directory, sub, ".git"))
        ]
        python.extend(base / name for name in names if name.endswith(".py"))
        configs.extend(base / name for name in names if name in _RUFF_CONFIG_NAMES)
    return python, configs


def private_binding_inputs(repo_root: Path, ledger: Path | None) -> ScanInputs:
    """what :func:`scan_private_bindings` reads: every python file and ruff config of the repo, the
    ledger, the ``src`` trees imports resolve against, and where the repo is (its findings name
    absolute paths).

    :param repo_root: the repo's root
    :ptype repo_root: Path
    :param ledger: the underscore-access exemptions ledger, or ``None``
    :ptype ledger: Path | None
    :return: the inputs
    :rtype: ScanInputs
    """
    root = repo_root.resolve()
    python, configs = workspace_files(root)
    ledgers = [ledger.resolve()] if ledger is not None and ledger.is_file() else []
    roots = "\n".join(source.relative_to(root).as_posix() for source in find_local_src_roots(root))
    return ScanInputs(
        root=root,
        files=tuple([*python, *configs, *ledgers]),
        listed=(root,),
        facts=("private_binding", root.as_posix(), roots, "ledger" if ledgers else "no ledger"),
    )


def private_binding_scan(cache: ScanCache, repo_root: Path, ledger: Path | None) -> PrivateBindingScan:
    """:func:`scan_private_bindings` of the repo, from *cache* when nothing it reads has changed.

    :param cache: the cache
    :ptype cache: ScanCache
    :param repo_root: the repo's root
    :ptype repo_root: Path
    :param ledger: the underscore-access exemptions ledger, or ``None``
    :ptype ledger: Path | None
    :return: the scan
    :rtype: PrivateBindingScan
    """

    def scan(_: str) -> PrivateBindingScan:
        # the process-wide parse cache trusts a file's stamp, not its bytes: read them afresh
        clear_parse_cache()
        return scan_private_bindings(repo_root, ledger)

    found = cache.memoised("private_binding", ["workspace"], lambda _: private_binding_inputs(repo_root, ledger), scan)
    return found["workspace"]


def tree_id(repo_root: Path, ref: str) -> str:
    """the id of *ref*'s tree: what every file read at *ref* is.

    :param repo_root: the repo's root
    :ptype repo_root: Path
    :param ref: a tag or commit
    :ptype ref: str
    :return: the tree's object id
    :rtype: str
    :raises subprocess.CalledProcessError: when this clone cannot name the ref's tree
    """
    found = subprocess.run(
        ["git", "rev-parse", "--verify", f"{ref}^{{tree}}"], cwd=repo_root, capture_output=True, text=True, check=True
    )
    return found.stdout.strip()


def surface_at_inputs(repo_root: Path, ref: str, source_root: str) -> ScanInputs:
    """what :func:`surface_at` reads: git's objects for one source root of *ref*'s tree, pinned by its id.

    :param repo_root: the repo's root
    :ptype repo_root: Path
    :param ref: a tag or commit
    :ptype ref: str
    :param source_root: repo-relative source tree
    :ptype source_root: str
    :return: the inputs
    :rtype: ScanInputs
    """
    return ScanInputs(
        root=repo_root, facts=("surface_at", tree_id(repo_root, ref), source_root), commands=frozenset({"git"})
    )


def surfaces_at(
    cache: ScanCache, repo_root: Path, ref: str, source_roots: Iterable[str]
) -> dict[str, dict[str, set[str]]]:
    """each source root's public surface at *ref*, module by module.

    :param cache: the cache
    :ptype cache: ScanCache
    :param repo_root: the repo's root
    :ptype repo_root: Path
    :param ref: a tag or commit
    :ptype ref: str
    :param source_roots: repo-relative source trees
    :ptype source_roots: Iterable[str]
    :return: source root -> module path -> its surface, in the order given
    :rtype: dict[str, dict[str, set[str]]]
    """
    return cache.memoised(
        "surface_at",
        source_roots,
        lambda source_root: surface_at_inputs(repo_root, ref, source_root),
        lambda source_root: surface_at(repo_root, ref, source_root).modules,
    )


def surface_now_inputs(repo_root: Path, source_root: str) -> ScanInputs:
    """what :func:`surface_now` reads: every python file under one source root of the working tree.

    :param repo_root: the repo's root
    :ptype repo_root: Path
    :param source_root: repo-relative source tree
    :ptype source_root: str
    :return: the inputs
    :rtype: ScanInputs
    """
    base = repo_root / source_root
    return ScanInputs(
        root=repo_root,
        files=tuple(sorted(base.rglob("*.py"))),
        listed=(base,) if base.is_dir() else (),
        facts=("surface_now", source_root),
    )


def surfaces_now(cache: ScanCache, repo_root: Path, source_roots: Iterable[str]) -> dict[str, dict[str, set[str]]]:
    """each source root's public surface in the working tree, module by module.

    :param cache: the cache
    :ptype cache: ScanCache
    :param repo_root: the repo's root
    :ptype repo_root: Path
    :param source_roots: repo-relative source trees
    :ptype source_roots: Iterable[str]
    :return: source root -> module path -> its surface, in the order given
    :rtype: dict[str, dict[str, set[str]]]
    """
    return cache.memoised(
        "surface_now",
        source_roots,
        lambda source_root: surface_now_inputs(repo_root, source_root),
        lambda source_root: surface_now(repo_root, source_root).modules,
    )


def referenced_and_defined(tree: ast.Module, *, re_exports: bool) -> tuple[frozenset[str], frozenset[str]]:
    """the names a module references, and the names it defines at top level.

    :param tree: the module
    :ptype tree: ast.Module
    :param re_exports: whether a ``from ... import`` counts as a reference (not in a package ``__init__``)
    :ptype re_exports: bool
    :return: references, definitions
    :rtype: tuple[frozenset[str], frozenset[str]]
    """
    used: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            used.add(node.id)
        elif isinstance(node, ast.Attribute):
            used.add(node.attr)
        elif isinstance(node, ast.ImportFrom) and re_exports:
            used.update(alias.name for alias in node.names)
    defined: set[str] = set()
    for statement in tree.body:
        if isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            defined.add(statement.name)
        elif isinstance(statement, ast.Assign | ast.AnnAssign):
            targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
            defined.update(target.id for target in targets if isinstance(target, ast.Name))
    return frozenset(used), frozenset(defined)


def module_names_inputs(repo_root: Path, module: tuple[str, bool]) -> ScanInputs:
    """what one module's names are read from: its bytes, and whether its imports count.

    :param repo_root: the repo's root
    :ptype repo_root: Path
    :param module: (repo-relative module path, whether a ``from ... import`` counts as a reference)
    :ptype module: tuple[str, bool]
    :return: the inputs
    :rtype: ScanInputs
    """
    relative, re_exports = module
    return ScanInputs(root=repo_root, files=(repo_root / relative,), facts=("names", str(re_exports)))


def module_names(
    cache: ScanCache, repo_root: Path, modules: Iterable[tuple[str, bool]]
) -> dict[tuple[str, bool], tuple[frozenset[str], frozenset[str]]]:
    """:func:`referenced_and_defined` of each module, parsed from its bytes on disk.

    :param cache: the cache
    :ptype cache: ScanCache
    :param repo_root: the repo's root
    :ptype repo_root: Path
    :param modules: (repo-relative module path, whether a ``from ... import`` counts as a reference)
    :ptype modules: Iterable[tuple[str, bool]]
    :return: module -> (references, definitions)
    :rtype: dict[tuple[str, bool], tuple[frozenset[str], frozenset[str]]]
    """

    def names(module: tuple[str, bool]) -> tuple[frozenset[str], frozenset[str]]:
        relative, re_exports = module
        path = repo_root / relative
        return referenced_and_defined(ast.parse(path.read_bytes(), filename=str(path)), re_exports=re_exports)

    return cache.memoised("names", modules, lambda module: module_names_inputs(repo_root, module), names)
