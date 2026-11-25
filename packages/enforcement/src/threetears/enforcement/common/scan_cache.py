"""an on-disk memo for the enforcement scans, keyed by what they read and the code that computed them.

The gates read every python file in the workspace, and each process that runs one of their tests
pays for it again: each xdist worker, each batch a test pool cuts from a gate's file, and each test
of a gate whose sibling walked the same files a moment before (the parse cache keeps only part of
the workspace's trees, so a second walk parses most of it again). A :class:`ScanCache` keeps what a
scan computed, so the next process that reads the same bytes takes the answer instead.

**Two ways in.** :meth:`ScanCache.memoised` keeps one value per item a caller names, with the
inputs the caller declares for it (:class:`ScanInputs`): a whole-workspace scan, a surface at a git
tree. :func:`file_facts` keeps what one extractor makes of each python file, keyed by that file's
path and bytes alone, so an edit recomputes the edited file and nothing else; most gates are a walk
that turns each file into a small fact and a check over the facts, and this is the walk.

**What a value is keyed by.** Its inputs, hashed by content, never by modification time, which
several filesystems cannot tell apart across quick rewrites; and a key of the code that computes
it, carried in the store's name. For :meth:`ScanCache.memoised` the caller gives that code key
(:func:`code_key` over a :func:`module_closure`). For :func:`file_facts` it is
:func:`_extractor_code`: the extractor's module, every local module it imports or refers to, the
``threetears`` modules all of them import at any depth, the lockfile and the interpreter's version.
Each closure is read from the source, not from what the session happens to have imported, so every
process computes the same key whatever else it ran; the imports each module names are themselves
kept on disk by its bytes (:class:`ImportIndex`), so a key costs a read of each module, not a parse.

**What is never stored.** A computed value is stored only when, while it ran, the computation
opened no file, listed no directory and started no program outside its declared inputs, as the
interpreter's audit events report them (``open``, ``os.scandir``, ``os.listdir``,
``subprocess.Popen`` and the ``os`` spawn calls), and when its inputs hash the same afterwards as
before. A scan that starts reading something its key does not cover is answered and not stored,
with a :class:`_ScanCacheWarning`, so a key list that drifts from its scan costs speed, never a
stale verdict. What the audit cannot see: a bare ``stat`` (a scan that decides by whether a path
exists, outside the files and directories it declares); a read on another thread than the one
computing (recording is per thread, and the scans start none); sockets and other non-file reads; a
name learned from listing a declared directory, which is not keyed unless the file it names is
declared; and a C extension reading through ``fopen``, which raises no event. What it leaves out on
purpose: the import system's own listings and opens, while a module the scan imports is loaded.
What no audit can see is code: an extractor that calls into a module :func:`_extractor_code` cannot
find (one it reaches only through an attribute of an object, never by name in its own module) is
keyed without it. Name the module, or the function, in the extractor's module.

**The hook costs nothing until a scan computes.** A Python audit hook runs a Python call for every
audit event in the process (``id``, ``compile``, ``import``, ...) and cannot be removed once added,
so it is added on the first computation that records, never at import: a process whose scans
all come from the store, and every test that never asks for a scan, runs without it.

**Concurrency.** Any number of processes may read and write one directory at once: up to four xdist
workers in one checkout, a pool's batches in one workspace. A store is written whole to a temporary
file beside it and renamed over it, so a reader sees an old store or a new one, never a partial
one. A store carries a checksum of its payload, and one that is truncated, corrupt or of another
format reads as empty. Writers never wait for each other: the last rename wins, and every store
holds a complete answer for the keys its writer looked up.

**Nothing here is required.** A missing directory is created; one that cannot be created, read or
written (a read-only checkout) leaves every lookup a miss, the scan is computed as it would be
without the cache, and the first failed write is logged, never warned: a cache miss must not fail
a run that treats warnings as errors. The default directory is
``.cache/enforcement-scans`` under the repo root, and it carries its own ``.gitignore`` (as
``.pytest_cache`` does), so a consumer repo's checkout stays clean without naming it.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import os
import pickle
import re
import sys
import tempfile
import threading
import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from types import FrameType, FunctionType, ModuleType
from typing import Any, Literal

from threetears.enforcement.common.ast_helpers import note_unscanned
from threetears.observe import BuildOnce, get_logger

log = get_logger(__name__)

__all__ = [
    "CACHE_DIRECTORY",
    "ImportIndex",
    "ScanCache",
    "ScanInputs",
    "code_key",
    "file_facts",
    "installed_module_file",
    "module_closure",
]


#: where the stores live by default, under the repo root (gitignored)
CACHE_DIRECTORY = Path(".cache") / "enforcement-scans"

#: the first line of every store; a file without it is not one, and reads as empty
_MAGIC = b"3tears-enforcement-scan-cache 2\n"

#: a temporary file older than this was left by a writer that died before its rename
_ORPHAN_AGE_S = 3600.0

#: a store written under other code is removed once nothing has rewritten it for this long
_OTHER_CODE_AGE_S = 86400.0

#: what a store directory's own ``.gitignore`` says: everything here, itself included, is git's to ignore
_SELF_IGNORE = "# the enforcement scan cache (threetears.enforcement.common.scan_cache); never tracked\n*\n"

#: what a computation may open outside its inputs: the interpreter's own code, never data
_CODE_SUFFIXES = frozenset({".py", ".pyc", ".so"})
_CODE_ROOTS = tuple(sorted({os.path.realpath(prefix) for prefix in (sys.prefix, sys.base_prefix, sys.exec_prefix)}))

#: the audit events a computation's reads and program starts arrive as
_PROCESS_EVENTS = frozenset({"subprocess.Popen", "os.system", "os.exec", "os.spawn", "os.posix_spawn"})
_WATCHED_EVENTS = frozenset({"open", "os.scandir", "os.listdir"}) | _PROCESS_EVENTS

#: per thread, the audit events seen while a computation runs; absent while none does
_RECORDING = threading.local()


def _audit(event: str, args: tuple[Any, ...]) -> None:
    """keep a watched event for the computation running on this thread, if one is.

    an event the import system raises is never kept: loading a module the computation imports for the
    first time lists each ``sys.path`` directory and opens the module's code, which is the interpreter
    loading code, not the scan reading an input. Whether it happens depends only on what the process had
    imported before, so keeping it would refuse a scan in a fresh process and store it in a warm one.
    """
    if event in _WATCHED_EVENTS:
        log = getattr(_RECORDING, "log", None)
        if log is not None and not _importing():
            log.append((event, args))


def _importing() -> bool:
    """whether the import system raised the event being audited, so it is the interpreter loading code.

    only the frames between the event and the recording's own start count: a computation that runs
    while some module is being imported (a gate's table built at its import) has the import system
    below it on the stack, and its reads are still its own.
    """
    anchor = getattr(_RECORDING, "anchor", None)
    frame = inspect.currentframe()
    while frame is not None and frame is not anchor:
        if frame.f_code.co_filename.startswith("<frozen importlib"):
            return True
        frame = frame.f_back
    return False


def _install_audit() -> bool:
    """add :func:`_audit` to this process's audit hooks; :data:`_AUDIT` makes it once."""
    sys.addaudithook(_audit)
    return True


#: the hook, added once per process on the first computation that records
_AUDIT: BuildOnce[str, bool] = BuildOnce()


def _ignore_itself(directory: Path) -> None:
    """give *directory* a ``.gitignore`` that ignores everything in it, so any checkout's git leaves the stores alone.

    A consumer repo's gates write their stores under its root without its ``.gitignore`` naming them,
    as ``.pytest_cache`` and ``.mypy_cache`` do. Written whole and renamed into place, once.

    :raises OSError: when the directory cannot be written
    """
    target = directory / ".gitignore"
    if target.is_file():
        return
    handle, temporary = tempfile.mkstemp(dir=directory, prefix=".gitignore-", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as out:
            out.write(_SELF_IGNORE)
        os.replace(temporary, target)
    except OSError:
        Path(temporary).unlink(missing_ok=True)
        raise


class _ScanCacheWarning(UserWarning):
    """a value was computed and not stored: its scan read outside its key, or the store is unwritable."""


def _content_key(*parts: str | bytes) -> str:
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


def _files_key(paths: Iterable[Path], root: Path) -> str:
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
    return _content_key(*parts)


def installed_module_file(name: str) -> Path | None:
    """the source file of ``threetears`` module *name* as installed, found without importing it.

    The leaf package (``threetears.core``, ``threetears.agent.memory``) is located through the import
    system, so an editable install pointing at another checkout answers with that checkout's file;
    the rest of the dotted name is a path below it.

    :param name: a dotted module name under ``threetears``
    :ptype name: str
    :return: its ``.py`` file (a package's ``__init__.py``), or ``None`` when no such module is installed
    :rtype: Path | None
    """
    parts = name.split(".")
    depth = 3 if len(parts) > 2 and parts[1] == "agent" else 2
    if parts[0] != "threetears" or len(parts) < depth:
        return None
    try:
        spec = importlib.util.find_spec(".".join(parts[:depth]))
    except ModuleNotFoundError:  # NOSILENT: a leaf package that is not installed runs nowhere -- None is this function's documented "no such module", and the closure leaves it out
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    rest = parts[depth:]
    for location in spec.submodule_search_locations:
        base = Path(location).joinpath(*rest)
        for candidate in (base / "__init__.py", base.with_suffix(".py") if rest else None):
            if candidate is not None and candidate.is_file():
                return candidate
    return None


def _direct_imports(name: str, path: Path, source: bytes) -> tuple[str, ...]:
    """every absolute module name module *name* imports, each imported name tried as a submodule too.

    Relative imports are resolved against *name*'s package: *name* itself when *path* is a package's
    ``__init__.py``, else its parent.

    :param name: the module's dotted name
    :ptype name: str
    :param path: its source file
    :ptype path: Path
    :param source: the file's bytes
    :ptype source: bytes
    :return: the names, in the order the module's nodes name them
    :rtype: tuple[str, ...]
    :raises SyntaxError: when the source is not valid python
    """
    package = name if path.name == "__init__.py" else name.rpartition(".")[0]
    found: list[str] = []
    for node in ast.walk(ast.parse(source, filename=str(path))):
        if isinstance(node, ast.Import):
            found.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                anchor = package.split(".")[: len(package.split(".")) - node.level + 1]
                base = ".".join([*anchor, *([node.module] if node.module else [])])
            if base:
                found.append(base)
                found.extend(f"{base}.{alias.name}" for alias in node.names)
    return tuple(found)


def _read_imports(name: str, path: Path) -> tuple[str, ...]:
    """:func:`_direct_imports` of the module, read from disk now."""
    return _direct_imports(name, path, path.read_bytes())


def module_closure(
    entries: Iterable[str],
    locate: Callable[[str], Path | None],
    imports: Callable[[str, Path], tuple[str, ...]] = _read_imports,
) -> dict[str, Path]:
    """every ``threetears`` module the *entries* import at any depth, with each one's source file.

    Read statically from each module's imports, so the answer is the same whatever the process has
    imported: every ``import`` and ``from ... import`` of a ``threetears`` module (relative ones
    resolved), each imported name tried as a submodule too, and every package above a module, whose
    ``__init__`` runs on import. A name *locate* cannot find is not installed, so runs nowhere.

    :param entries: dotted module names to start from
    :ptype entries: Iterable[str]
    :param locate: a module's source file, or ``None`` (:func:`installed_module_file`)
    :ptype locate: Callable[[str], Path | None]
    :param imports: a module's :func:`_direct_imports`, by name and file; an :class:`ImportIndex` keeps them
    :ptype imports: Callable[[str, Path], tuple[str, ...]]
    :return: module name -> its source file
    :rtype: dict[str, Path]
    :raises OSError: when a source file cannot be read
    :raises SyntaxError: when a source file is not valid python
    """
    found: dict[str, Path] = {}
    pending = list(entries)
    while pending:
        name = pending.pop()
        if name in found:
            continue
        path = locate(name)
        if path is None:
            continue
        found[name] = path
        pending.extend(".".join(name.split(".")[:depth]) for depth in range(2, name.count(".") + 1))
        pending.extend(target for target in imports(name, path) if target.startswith("threetears."))
    return found


def code_key(modules: dict[str, Path], files: Iterable[Path], root: Path) -> str:
    """one key for the code that computes: each module's name and source, and each file's content.

    :param modules: module name -> source file (:func:`module_closure`)
    :ptype modules: dict[str, Path]
    :param files: further files, named relative to *root*
    :ptype files: Iterable[Path]
    :param root: what each file is named relative to
    :ptype root: Path
    :return: a hex digest
    :rtype: str
    :raises OSError: when a source file cannot be read
    """
    parts: list[str | bytes] = [sys.version]
    for name in sorted(modules):
        parts.append(name)
        parts.append(modules[name].read_bytes())
    parts.append(_files_key(files, root))
    return _content_key(*parts)


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
        return _content_key(
            *self.facts, _files_key(self.files, self.root), "\n".join(listed), "\n".join(sorted(self.commands))
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
    """record this thread's watched audit events while the block runs.

    a recording inside another (a per-file walk inside a whole-workspace scan) hands what it saw
    to the outer one when it ends: the outer scan read those files too.
    """

    def __enter__(self) -> list[tuple[str, tuple[Any, ...]]]:
        _AUDIT.get("audit", _install_audit)
        self.outer: list[tuple[str, tuple[Any, ...]]] | None = getattr(_RECORDING, "log", None)
        self.outer_anchor: FrameType | None = getattr(_RECORDING, "anchor", None)
        self.log: list[tuple[str, tuple[Any, ...]]] = []
        here = inspect.currentframe()
        _RECORDING.anchor = here.f_back if here is not None else None
        _RECORDING.log = self.log
        return self.log

    def __exit__(self, *_: object) -> None:
        _RECORDING.log = self.outer
        _RECORDING.anchor = self.outer_anchor
        if self.outer is not None:
            self.outer.extend(self.log)


class _Unrecorded:
    """leave the block out of any recording on this thread: the cache's own stores and code keys, never a scan's input.

    a scan that calls into a cached walk would otherwise read the walk's store, the lockfile and the
    walk's code as inputs it never declared, and never be stored itself.
    """

    def __enter__(self) -> None:
        self.outer: list[tuple[str, tuple[Any, ...]]] | None = getattr(_RECORDING, "log", None)
        _RECORDING.log = None

    def __exit__(self, *_: object) -> None:
        _RECORDING.log = self.outer


@dataclass(frozen=True)
class ScanCache:
    """a directory of stores, each a mapping of content key to computed value.

    :ivar directory: where the stores live; need not exist, nor be writable
    :ivar code: a key for the code that computes every value; a store written under another is
        never read
    :ivar unstored: why each value this cache computed and did not store was not stored
    :ivar defects: those of them that read outside their key, a defect in the scan (each also in ``unstored``)
    """

    directory: Path
    code: str
    unstored: list[str] = field(default_factory=list, compare=False)
    defects: list[str] = field(default_factory=list, compare=False)

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
        stored = self.load(name)
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
                    self._refuse(
                        f"{name} {item!r} read outside its key: " + "; ".join(sorted(set(undeclared))[:5]), defect=True
                    )
                elif inputs(item).key() != before:
                    self._refuse(f"{name} {item!r}: its inputs changed while it was computed", defect=False)
                else:
                    kept[before] = values[item]
        if computed or kept.keys() != stored.keys():
            self.store(name, kept)
        return values

    def per_file[V](
        self,
        name: str,
        extract: Callable[[Path, ast.Module], V],
        paths: Iterable[Path],
        *,
        root: Path,
        unparsable: Literal["raise", "note"] = "raise",
    ) -> dict[Path, V]:
        """*extract* of each python file's tree: the stored fact when the file's bytes were seen before.

        A file is keyed by its path relative to *root*, its bytes and where *root* is (a fact may
        name absolute paths), so an edit, a rename or a move recomputes that file alone. Its tree is
        parsed from the bytes the key was taken from, so a file rewritten mid-walk is answered for
        the bytes it was keyed by, never stored under bytes it no longer has. A fact is stored only
        when *extract* opened nothing but that file, listed no directory and started no program. A
        file that is not valid utf-8 or not valid python is kept as such and, on every call, raises
        an equal error (``unparsable="raise"``, as :func:`parse_python_path`) or goes through
        :func:`note_unscanned` and is left out (``"note"``, as :func:`parse_python_file`).

        The store keeps every fact this call looked up, and the facts of files it did not ask about,
        so callers walking different sets of files share one store; an older fact of a file it
        looked up is dropped.

        :param name: the store, one per extractor
        :ptype name: str
        :param extract: makes one file's fact from its path and its tree; the fact must be picklable,
            and is shared with every caller, so treat it as read-only
        :ptype extract: Callable[[Path, ast.Module], V]
        :param paths: the files, each under *root*; a repeat is looked up once
        :ptype paths: Iterable[Path]
        :param root: what each path is named relative to
        :ptype root: Path
        :param unparsable: what a file that does not parse does
        :ptype unparsable: Literal["raise", "note"]
        :return: path -> fact, in the order given, without the files ``"note"`` left out
        :rtype: dict[Path, V]
        :raises ValueError: when a path is not under *root*
        :raises SyntaxError: when a file is not valid python, under ``"raise"``
        :raises UnicodeDecodeError: when a file is not valid utf-8, under ``"raise"``
        :raises OSError: when a file cannot be read
        """
        stored = self.load(name)
        where = os.path.abspath(root)
        kept: dict[str, tuple[str, _Outcome]] = {}
        outcomes: dict[Path, _Outcome] = {}
        computed = False
        for path in dict.fromkeys(paths):
            relative = path.relative_to(root).as_posix()
            source = path.read_bytes()
            key = _content_key(name, where, relative, source)
            entry = kept.get(key) or stored.get(key)
            if entry is None:
                computed = True
                outcome, undeclared = _extract_recorded(extract, path, source, root)
                if undeclared:
                    self._refuse(
                        f"{name} {relative} read outside its key: " + "; ".join(sorted(set(undeclared))[:5]),
                        defect=True,
                    )
                else:
                    kept[key] = (relative, outcome)
            else:
                kept[key] = entry
                outcome = entry[1]
            outcomes[path] = outcome
        looked_up = {relative for relative, _ in kept.values()}
        others = {key: entry for key, entry in stored.items() if entry[0] not in looked_up}
        if computed or kept.keys() | others.keys() != stored.keys():
            self.store(name, others | kept)
        return _facts(outcomes, unparsable)

    def _refuse(self, reason: str, *, defect: bool) -> None:
        """record why a value was not stored, and say so on the first of its kind.

        a computation that read outside its key is a defect in the scan, and warns. anything the
        environment did (a directory that cannot be written, a file edited mid-scan) goes to the log
        instead: a cache miss must never fail a run that treats warnings as errors.
        """
        self.unstored.append(reason)
        if defect:
            self.defects.append(reason)
            if len(self.defects) == 1:
                warnings.warn(f"enforcement scan cache: {reason}", _ScanCacheWarning, stacklevel=3)
        elif len(self.unstored) - len(self.defects) == 1:
            log.warning(
                "enforcement scan cache: a value was computed and not stored", extra={"extra_data": {"reason": reason}}
            )

    def _path(self, name: str) -> Path:
        return self.directory / f"{name}-{self.code[:24]}.pickle"

    def load(self, name: str) -> dict[str, Any]:
        """one store's entries: content key -> value, as its last writer left them.

        A store absent, unreadable, partial, corrupt or of another format reads as empty, so the
        caller computes what it would have read.

        :param name: the store
        :ptype name: str
        :return: its entries, or none
        :rtype: dict[str, Any]
        """
        with _Unrecorded():
            return self._read(name)

    def _read(self, name: str) -> dict[str, Any]:
        """the store as it is on disk."""
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

    def store(self, name: str, entries: dict[str, Any]) -> None:
        """replace one store with *entries*, written whole and renamed into place.

        A reader sees the old store or the new one, never a partial one. A directory that refuses
        the write is recorded in :attr:`unstored` (the first such warns), and nothing is raised.

        :param name: the store
        :ptype name: str
        :param entries: content key -> value, each value picklable
        :ptype entries: dict[str, Any]
        :return: nothing
        :rtype: None
        """
        with _Unrecorded():
            self._write(name, entries)

    def _write(self, name: str, entries: dict[str, Any]) -> None:
        """write the store, as store() says."""
        payload = pickle.dumps(entries, protocol=pickle.HIGHEST_PROTOCOL)
        blob = _MAGIC + hashlib.blake2b(payload, digest_size=32).hexdigest().encode() + b"\n" + payload
        target = self._path(name)
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            _ignore_itself(self.directory)
            handle, temporary = tempfile.mkstemp(dir=self.directory, prefix=f".{name}-", suffix=".tmp")
        except OSError as exc:
            self._refuse(f"{name}: {self.directory} cannot be written ({exc.strerror or exc})", defect=False)
            return
        try:
            with os.fdopen(handle, "wb") as out:
                out.write(blob)
            os.replace(temporary, target)
        except OSError as exc:
            Path(temporary).unlink(missing_ok=True)
            self._refuse(f"{name}: {target} cannot be written ({exc.strerror or exc})", defect=False)
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
        except OSError:  # NOSILENT: pruning is housekeeping -- a file another writer removed first is already gone, and a store left behind is removed by a later write
            return


#: one file's outcome: ``(True, fact)``, or ``(False, the error parsing it raised)``
type _Outcome = tuple[Literal[True], Any] | tuple[Literal[False], SyntaxError | UnicodeDecodeError]


def _extract_recorded[V](
    extract: Callable[[Path, ast.Module], V], path: Path, source: bytes, root: Path
) -> tuple[_Outcome, list[str]]:
    """parse *source* as *path*'s and extract its fact, recording what the extraction reads.

    :return: the outcome, and each access outside *path* itself (empty when there was none)
    """
    outcome: _Outcome
    with _Recorded() as events:
        try:
            # decoded as parse_python_path reads it (read_text: strict utf-8, universal newlines), so the
            # tree, its line numbers and its string constants are the ones that parse would give
            text = source.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
            tree = ast.parse(text, filename=str(path))
        except (SyntaxError, UnicodeDecodeError) as exc:
            outcome = (False, exc)
        else:
            outcome = (True, extract(path, tree))
    return outcome, ScanInputs(root=root, files=(path,)).outside(events)


def _facts[V](outcomes: dict[Path, _Outcome], unparsable: Literal["raise", "note"]) -> dict[Path, V]:
    """each parsed file's fact; each unparsable one raised or noted, as *unparsable* says."""
    facts: dict[Path, V] = {}
    for path, outcome in outcomes.items():
        if outcome[0]:
            facts[path] = outcome[1]
            continue
        error = outcome[1]
        if unparsable == "raise":
            # a fresh error per caller, as parse_python_path raises one
            raise type(error)(*error.args) from None
        reason = "not valid utf-8" if isinstance(error, UnicodeDecodeError) else f"syntax error at line {error.lineno}"
        note_unscanned(path, reason)
    return facts


#: an :class:`ImportIndex` store holding more modules than this is cut back to what its writer looked up
_IMPORT_INDEX_MAX = 20_000


class ImportIndex:
    """each module's :func:`_direct_imports`, kept on disk by its name and bytes, so a closure reads and never parses.

    Computing a code key walks a few hundred modules; parsing them costs seconds in every process,
    reading them milliseconds. The store's code key is this module's own source, which holds
    :func:`_direct_imports`. Unlike a :meth:`ScanCache.memoised` store it keeps what other writers
    looked up too (each extractor's closure is a different set of modules), up to
    :data:`_IMPORT_INDEX_MAX` entries.

    :param directory: where the store lives; need not exist, nor be writable
    :ptype directory: Path
    """

    def __init__(self, directory: Path) -> None:
        self._cache = ScanCache(directory, _content_key(sys.version, Path(__file__).read_bytes()))
        self._stored: dict[str, tuple[str, ...]] = self._cache.load("imports")
        self._seen: dict[str, tuple[str, ...]] = {}

    def __call__(self, name: str, path: Path) -> tuple[str, ...]:
        """:func:`_direct_imports` of the module, from the store when its bytes were indexed before.

        :param name: the module's dotted name
        :ptype name: str
        :param path: its source file
        :ptype path: Path
        :return: the names it imports
        :rtype: tuple[str, ...]
        :raises OSError: when the file cannot be read
        :raises SyntaxError: when it is not valid python
        """
        with _Unrecorded():
            source = path.read_bytes()
        key = _content_key(name, path.name, source)
        found = self._seen.get(key) or self._stored.get(key)
        if found is None:
            found = _direct_imports(name, path, source)
        self._seen[key] = found
        return found

    def save(self) -> None:
        """write what was indexed since the store was read, when anything was.

        :return: nothing
        :rtype: None
        """
        if self._seen.keys() <= self._stored.keys():
            return
        merged = self._stored | self._seen
        self._cache.store("imports", merged if len(merged) <= _IMPORT_INDEX_MAX else dict(self._seen))
        self._stored = merged


#: directories whose modules are installed code, keyed by the lockfile, never a local module of the checkout
_INSTALLED_PARTS = frozenset({".venv", "site-packages", "node_modules"})


def _local_file(path: Path, code_root: Path) -> bool:
    """whether *path* is a module of the checkout itself rather than installed code."""
    return path.suffix == ".py" and path.is_relative_to(code_root) and not _INSTALLED_PARTS & set(path.parts)


def _local_module(name: str, code_root: Path) -> Path | None:
    """the checkout's own source file of absolute module *name*, when it has one (``tests.enforcement.x``)."""
    base = code_root.joinpath(*name.split("."))
    for candidate in (base / "__init__.py", base.with_suffix(".py")):
        if candidate.is_file():
            return candidate
    return None


def _referenced_modules(module: ModuleType) -> list[ModuleType]:
    """the modules *module*'s globals name: modules themselves, and the modules classes and functions come from."""
    found: list[ModuleType] = []
    for value in list(vars(module).values()):
        if isinstance(value, ModuleType):
            found.append(value)
        elif isinstance(value, type | FunctionType):
            owner = sys.modules.get(value.__module__)
            if owner is not None:
                found.append(owner)
    return found


def _extractor_code(extract: Callable[..., object]) -> str:
    """a key for the code *extract* runs, the same in every process.

    It hashes the interpreter's version; the lockfile above *extract*'s module (third-party code);
    *extract*'s module when it is the checkout's own (a gate under ``tests/``) and every local module
    that one imports by name, imports at any depth, or holds in a global (a helper loaded from a
    path, as ``scripts/integration_marks.py`` is); and the source of every ``threetears`` module all
    of them import at any depth, as installed (:func:`module_closure`). A module the extractor
    reaches only through an attribute of an object is not found: name it in the extractor's module.

    :param extract: the extractor; its module must be imported
    :ptype extract: Callable[..., object]
    :return: a hex digest
    :rtype: str
    :raises OSError: when a source file cannot be read
    :raises SyntaxError: when a source file is not valid python
    """
    module = sys.modules[extract.__module__]
    own = Path(inspect.getfile(module)).resolve()
    lock = next((directory / "uv.lock" for directory in own.parents if (directory / "uv.lock").is_file()), None)
    code_root = lock.parent if lock is not None else own.parent
    index = ImportIndex(code_root / CACHE_DIRECTORY)
    entries: set[str] = set()
    pending: list[tuple[str, Path]] = []
    for referenced in [module, *_referenced_modules(module)]:
        if referenced.__name__.startswith("threetears."):
            entries.add(referenced.__name__)
            continue
        source = getattr(referenced, "__file__", None)
        if source is not None and _local_file(Path(source).resolve(), code_root):
            pending.append((referenced.__name__, Path(source).resolve()))
    local: dict[Path, None] = {}
    while pending:
        name, path = pending.pop()
        if path in local:
            continue
        local[path] = None
        for target in index(name, path):
            if target.startswith("threetears."):
                entries.add(target)
            elif (found := _local_module(target, code_root)) is not None:
                pending.append((target, found))
    modules = module_closure(sorted(entries), installed_module_file, index)
    index.save()
    return code_key(modules, [*local, *([lock] if lock is not None else [])], code_root)


def _qualified(extract: Callable[..., object]) -> str:
    """*extract*'s module and qualified name, which name its store."""
    return f"{extract.__module__}.{extract.__qualname__}"


#: each extractor's :func:`_extractor_code`, once per process: the code a process runs does not change under it
_EXTRACTOR_CODES: BuildOnce[str, str] = BuildOnce()


def file_facts[V](
    extract: Callable[[Path, ast.Module], V],
    paths: Iterable[Path],
    *,
    root: Path,
    directory: Path | None = None,
    unparsable: Literal["raise", "note"] = "raise",
) -> dict[Path, V]:
    """*extract* of each python file's tree, from the on-disk store when the file's bytes were seen before.

    The walk most gates make, kept per file (:meth:`ScanCache.per_file`) under *extract*'s code
    (:func:`_extractor_code`): a warm process reads and hashes each file and parses none it has
    seen, and an edit recomputes the edited file alone. *extract* takes a file's path and its
    tree, reads nothing else, and returns a small picklable fact; keep it a module-level function,
    whose name keys its store.

    :param extract: makes one file's fact from its path and tree
    :ptype extract: Callable[[Path, ast.Module], V]
    :param paths: the files, each under *root*
    :ptype paths: Iterable[Path]
    :param root: what each path is named relative to: the repo's root
    :ptype root: Path
    :param directory: where the store lives; ``.cache/enforcement-scans`` under *root* when omitted
    :ptype directory: Path | None
    :param unparsable: what a file that does not parse does: ``"raise"`` an equal error, or
        ``"note"`` it through :func:`note_unscanned` and leave it out
    :ptype unparsable: Literal["raise", "note"]
    :return: path -> fact, in the order given
    :rtype: dict[Path, V]
    :raises ValueError: when a path is not under *root*
    :raises SyntaxError: when a file is not valid python, under ``"raise"``
    :raises UnicodeDecodeError: when a file is not valid utf-8, under ``"raise"``
    :raises OSError: when a file cannot be read
    """
    qualified = _qualified(extract)
    with _Unrecorded():
        code = _EXTRACTOR_CODES.get(qualified, lambda: _extractor_code(extract))
    cache = ScanCache(directory if directory is not None else root / CACHE_DIRECTORY, code)
    store = "facts-" + re.sub(r"[^\w.]+", "_", qualified)
    return cache.per_file(store, extract, paths, root=root, unparsable=unparsable)
