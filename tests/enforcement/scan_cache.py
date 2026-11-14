"""an on-disk memo for the workspace scans behind the enforcement gates, keyed by what they read.

The scans read every python file in the workspace, and a gate's module-scoped fixture is rebuilt
in every process that runs one of its tests: each xdist worker, and each batch a test pool cuts
from the file. A :class:`ScanCache` keeps what a scan computed, so the next process that reads the
same bytes takes the answer instead of walking the trees again.

**What a value is keyed by.** Each computation declares its inputs as :class:`ScanInputs`: the files
it reads (hashed by content, never by modification time, which several filesystems cannot tell
apart across quick rewrites), the directories it may list, the programs it may start and any other
fact its answer depends on (a git tree id). The key builders for the gates are the public
functions below, so they are tested as the gates use them. The store's name carries a key of the
code that computes: the source of every loaded ``threetears`` module, this directory's gate shells
and helpers, the lockfile and the interpreter's version.

**What is never stored.** A computed value is stored only when, while it ran, the computation
opened no file, listed no directory and started no program outside its declared inputs, as the
interpreter's audit events report them (``open``, ``os.scandir``, ``os.listdir``,
``subprocess.Popen`` and the ``os`` spawn calls), and when its inputs hash the same afterwards as
before. A scan that starts reading something its key does not cover is answered and not stored,
with a :class:`ScanCacheWarning`, so a key list that drifts from its scan costs speed, never a
stale verdict. What the audit cannot see is a bare ``stat``: a scan that decides by whether a path
exists, outside the files and directories it declares, is not caught by it.

**Concurrency.** Any number of processes may read and write one directory at once: up to four xdist
workers in one checkout, a pool's batches in one workspace. A store is written whole to a temporary
file beside it and renamed over it, so a reader sees an old store or a new one, never a partial
one. A store carries a checksum of its payload, and one that is truncated, corrupt or of another
format reads as empty. Writers never wait for each other: the last rename wins, and every store
holds a complete answer for the keys its writer looked up.

**Nothing here is required.** A missing directory is created; one that cannot be created, read or
written (a read-only checkout) leaves every lookup a miss, the scan is computed as it would be
without the cache, and the first failed write warns. The default directory,
``.cache/enforcement-scans`` under the repo root, is gitignored.
"""

from __future__ import annotations

import ast
import hashlib
import os
import pickle
import subprocess
import sys
import tempfile
import threading
import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

from threetears.enforcement.common import clear_parse_cache, find_local_src_roots
from threetears.enforcement.release import surface_at, surface_now
from threetears.enforcement.underscore_access import PrivateBindingScan, scan_private_bindings

#: where :meth:`ScanCache.for_repo` keeps its stores, under the repo root (gitignored)
CACHE_DIRECTORY = Path(".cache") / "enforcement-scans"

#: the first line of every store; a file without it is not one, and reads as empty
_MAGIC = b"3tears-enforcement-scan-cache 2\n"

#: a temporary file older than this was left by a writer that died before its rename
_ORPHAN_AGE_S = 3600.0

#: a store written under other code is removed once nothing has rewritten it for this long
_OTHER_CODE_AGE_S = 86400.0

#: directories a workspace walk never enters, as the scans' own file discovery skips them
_UNSCANNED_DIRECTORIES = frozenset({".venv", "node_modules", ".git", "__pycache__"})

#: config files the private-binding scan reads ruff's per-file ignores from
_RUFF_CONFIG_NAMES = frozenset({"pyproject.toml", "ruff.toml", ".ruff.toml"})

#: what a computation may open outside its inputs: the interpreter's own code, never data
_CODE_SUFFIXES = frozenset({".py", ".pyc", ".so"})
_CODE_ROOTS = tuple(sorted({os.path.realpath(prefix) for prefix in (sys.prefix, sys.base_prefix, sys.exec_prefix)}))

#: the audit events a computation's reads and program starts arrive as
_PROCESS_EVENTS = frozenset({"subprocess.Popen", "os.system", "os.exec", "os.spawn", "os.posix_spawn"})
_WATCHED_EVENTS = frozenset({"open", "os.scandir", "os.listdir"}) | _PROCESS_EVENTS

#: per thread, the audit events seen while a computation runs; absent while none does
_RECORDING = threading.local()


def _audit(event: str, args: tuple[Any, ...]) -> None:
    """keep a watched event for the computation running on this thread, if one is."""
    log = getattr(_RECORDING, "log", None)
    if log is not None and event in _WATCHED_EVENTS:
        log.append((event, args))


sys.addaudithook(_audit)


class ScanCacheWarning(UserWarning):
    """a value was computed and not stored: its scan read outside its key, or the store is unwritable."""


def content_key(*parts: str | bytes) -> str:
    """one key for an ordered sequence of parts, each length-prefixed so no two sequences collide.

    :param parts: the parts, text or bytes
    :ptype parts: str | bytes
    :return: a hex digest
    :rtype: str
    """
    digest = hashlib.blake2b(digest_size=32)
    for part in parts:
        data = part.encode("utf-8") if isinstance(part, str) else part
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def files_key(paths: Iterable[Path], root: Path) -> str:
    """one key for a set of files: each one's path relative to *root*, and its content.

    :param paths: the files
    :ptype paths: Iterable[Path]
    :param root: what each path is named relative to
    :ptype root: Path
    :return: a hex digest that changes when a file is added, removed, renamed or edited
    :rtype: str
    :raises OSError: when a file cannot be read
    """
    parts: list[str | bytes] = []
    for path in sorted(set(paths)):
        parts.append(path.relative_to(root).as_posix())
        parts.append(path.read_bytes())
    return content_key(*parts)


def code_key(modules: Iterable[ModuleType], files: Iterable[Path], root: Path) -> str:
    """one key for the code that computes: each module's name and source, and each file's content.

    :param modules: modules whose source is hashed; one without a source file adds its name only
    :ptype modules: Iterable[ModuleType]
    :param files: further files, named relative to *root*
    :ptype files: Iterable[Path]
    :param root: what each file is named relative to
    :ptype root: Path
    :return: a hex digest
    :rtype: str
    :raises OSError: when a source file cannot be read
    """
    parts: list[str | bytes] = [sys.version]
    for module in sorted(modules, key=lambda module: module.__name__):
        source = getattr(module, "__file__", None)
        parts.append(module.__name__)
        parts.append(Path(source).read_bytes() if source else b"")
    parts.append(files_key(files, root))
    return content_key(*parts)


@dataclass(frozen=True)
class ScanInputs:
    """everything one computation reads, declared before it runs.

    :ivar root: what files and directories are named relative to; every one sits under it
    :ivar files: files it reads, hashed by content
    :ivar listed: directories it may list, at any depth; whatever it reads from what it finds there
        must be in ``files``
    :ivar facts: anything else its value depends on, as text (a git tree id, the root's location)
    :ivar commands: programs it may start, by name; ``facts`` must pin what they answer
    """

    root: Path
    files: tuple[Path, ...] = ()
    listed: tuple[Path, ...] = ()
    facts: tuple[str, ...] = ()
    commands: frozenset[str] = frozenset()

    def key(self) -> str:
        """the content key: the facts, the files' names and bytes, the directories and the programs.

        :return: a hex digest
        :rtype: str
        :raises OSError: when a file cannot be read
        """
        listed = [directory.relative_to(self.root).as_posix() for directory in sorted(self.listed)]
        return content_key(
            *self.facts, files_key(self.files, self.root), "\n".join(listed), "\n".join(sorted(self.commands))
        )

    def outside(self, events: Iterable[tuple[str, tuple[Any, ...]]]) -> list[str]:
        """each recorded read or program start these inputs do not declare.

        :param events: audit events, as ``(event, args)``
        :ptype events: Iterable[tuple[str, tuple[Any, ...]]]
        :return: one line per undeclared access, empty when every one was declared
        :rtype: list[str]
        """
        files = {os.path.abspath(path) for path in self.files} | {os.path.realpath(path) for path in self.files}
        listed = [os.path.abspath(directory) for directory in self.listed]
        listed += [os.path.realpath(directory) for directory in self.listed]

        def declared(event: str, path: str) -> bool:
            if event == "open":
                return path in files or path == os.devnull or _is_code(path)
            return any(path == directory or path.startswith(directory + os.sep) for directory in listed)

        found: list[str] = []
        for event, args in events:
            if event in _PROCESS_EVENTS:
                program = _program(event, args)
                if program not in self.commands:
                    found.append(f"{event} {program}")
                continue
            target = args[0] if args else None
            if isinstance(target, int):
                continue  # a descriptor already opened, whose open was itself recorded
            named = os.path.abspath(os.fsdecode(target) if target is not None else os.getcwd())
            # the name as given first: resolving every path a whole-repo walk lists costs a stat per part
            if not declared(event, named) and not declared(event, os.path.realpath(named)):
                found.append(f"{event} {named}")
        return found


def _program(event: str, args: tuple[Any, ...]) -> str:
    """the name of the program a process event starts."""
    if event == "subprocess.Popen":
        executable, argv = args[0], args[1]
        named = executable or (argv if isinstance(argv, str | bytes | os.PathLike) else argv[0])
    elif event == "os.system":
        named = os.fsdecode(args[0]).split()[0]
    else:
        named = args[1] if event == "os.spawn" else args[0]
    return Path(os.fsdecode(named)).name


def _is_code(path: str) -> bool:
    """whether *path* is the interpreter's own code: a module, or its bytecode, under its installation prefix.

    an import's bytecode write goes through a temporary name (``x.cpython-314.pyc.<id>``), so anything
    in a ``__pycache__`` there counts.
    """
    parts = Path(path)
    in_code = parts.suffix in _CODE_SUFFIXES or "__pycache__" in parts.parts
    return in_code and any(path.startswith(root + os.sep) for root in _CODE_ROOTS)


class _Recorded:
    """record this thread's watched audit events while the block runs."""

    def __enter__(self) -> list[tuple[str, tuple[Any, ...]]]:
        self.log: list[tuple[str, tuple[Any, ...]]] = []
        _RECORDING.log = self.log
        return self.log

    def __exit__(self, *_: object) -> None:
        _RECORDING.log = None


@dataclass(frozen=True)
class ScanCache:
    """a directory of stores, each a mapping of content key to computed value.

    :ivar directory: where the stores live; need not exist, nor be writable
    :ivar code: a key for the code that computes every value; a store written under another is
        never read
    :ivar unstored: why each value this cache computed and did not store was not stored
    """

    directory: Path
    code: str
    unstored: list[str] = field(default_factory=list, compare=False)

    @classmethod
    def for_repo(cls, repo_root: Path) -> ScanCache:
        """the cache a gate in this directory uses: under the repo root, keyed by the code loaded now.

        Build it after the gates are imported (a session fixture is): the code key hashes every
        ``threetears`` module loaded at that moment, from wherever it was loaded.

        :param repo_root: the repo's root
        :ptype repo_root: Path
        :return: the cache
        :rtype: ScanCache
        :raises OSError: when a file of the code cannot be read
        """
        loaded = [
            module
            for name, module in list(sys.modules.items())
            if name == "threetears" or name.startswith("threetears.")
        ]
        gates = sorted((repo_root / "tests" / "enforcement").glob("*.py"))
        code = code_key(loaded, [*gates, repo_root / "uv.lock"], repo_root)
        return cls(directory=repo_root / CACHE_DIRECTORY, code=code)

    def memoised[K, V](
        self, name: str, items: Iterable[K], inputs: Callable[[K], ScanInputs], compute: Callable[[K], V]
    ) -> dict[K, V]:
        """each item's value: the stored one when its inputs were computed from before, else ``compute(item)``.

        A computed value is stored only when the computation read nothing its inputs do not
        declare and its inputs key the same afterwards: a file edited while it was being read is
        answered for this call and never stored under the content it no longer has. The store is
        rewritten when anything was computed, holding exactly the keys this call looked up, so an
        entry no current input needs is dropped, not kept forever.

        :param name: the store, one per kind of value
        :ptype name: str
        :param items: what to look up
        :ptype items: Iterable[K]
        :param inputs: an item's declared inputs
        :ptype inputs: Callable[[K], ScanInputs]
        :param compute: computes one item's value; it must be picklable
        :ptype compute: Callable[[K], V]
        :return: item -> value, for every item
        :rtype: dict[K, V]
        """
        stored = self._load(name)
        values: dict[K, V] = {}
        kept: dict[str, Any] = {}
        computed = False
        for item in items:
            declared = inputs(item)
            before = declared.key()
            if before in kept:
                values[item] = kept[before]
            elif before in stored:
                values[item] = kept[before] = stored[before]
            else:
                computed = True
                with _Recorded() as events:
                    values[item] = compute(item)
                undeclared = declared.outside(events)
                if undeclared:
                    self._refuse(f"{name} {item!r} read outside its key: " + "; ".join(sorted(set(undeclared))[:5]))
                elif inputs(item).key() != before:
                    self._refuse(f"{name} {item!r}: its inputs changed while it was computed")
                else:
                    kept[before] = values[item]
        if computed or kept.keys() != stored.keys():
            self._store(name, kept)
        return values

    def _refuse(self, reason: str) -> None:
        """record why a value was not stored, and warn on the first."""
        self.unstored.append(reason)
        if len(self.unstored) == 1:
            warnings.warn(f"enforcement scan cache: {reason}", ScanCacheWarning, stacklevel=3)

    def _path(self, name: str) -> Path:
        return self.directory / f"{name}-{self.code[:24]}.pickle"

    def _load(self, name: str) -> dict[str, Any]:
        """the store's entries, or none when it is absent, unreadable, partial or corrupt."""
        try:
            blob = self._path(name).read_bytes()
        except OSError:
            return {}
        header_end = len(_MAGIC) + 64
        if not blob.startswith(_MAGIC) or blob[header_end : header_end + 1] != b"\n":
            return {}
        checksum, payload = blob[len(_MAGIC) : header_end], blob[header_end + 1 :]
        if hashlib.blake2b(payload, digest_size=32).hexdigest().encode() != checksum:
            return {}
        entries: dict[str, Any] = pickle.loads(payload)  # noqa: S301 -- this checkout's own store, checksummed
        return entries

    def _store(self, name: str, entries: dict[str, Any]) -> None:
        """write the store whole, then rename it into place; a directory that refuses it is recorded."""
        payload = pickle.dumps(entries, protocol=pickle.HIGHEST_PROTOCOL)
        blob = _MAGIC + hashlib.blake2b(payload, digest_size=32).hexdigest().encode() + b"\n" + payload
        target = self._path(name)
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            handle, temporary = tempfile.mkstemp(dir=self.directory, prefix=f".{name}-", suffix=".tmp")
        except OSError as exc:
            self._refuse(f"{name}: {self.directory} cannot be written ({exc.strerror or exc})")
            return
        try:
            with os.fdopen(handle, "wb") as out:
                out.write(blob)
            os.replace(temporary, target)
        except OSError as exc:
            Path(temporary).unlink(missing_ok=True)
            self._refuse(f"{name}: {target} cannot be written ({exc.strerror or exc})")
            return
        self._prune(name, target)

    def _prune(self, name: str, kept: Path) -> None:
        """drop this store's files from other code no run has used for a day, and dead writers' temporaries.

        ages are measured against the store just written, not the clock: a test runner that caches
        results treats a clock read as an input it cannot replay.
        """
        try:
            now = kept.stat().st_mtime
            for path in self.directory.glob(f"{name}-*.pickle"):
                if path != kept and path.stat().st_mtime < now - _OTHER_CODE_AGE_S:
                    path.unlink(missing_ok=True)
            for path in self.directory.glob(f".{name}-*.tmp"):
                if path.stat().st_mtime < now - _ORPHAN_AGE_S:
                    path.unlink(missing_ok=True)
        except OSError:
            return  # pruning is housekeeping: a file another writer removed first is already gone


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
