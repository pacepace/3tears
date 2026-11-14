"""an on-disk memo for the workspace scans behind the enforcement gates, keyed by content.

The scans read every python file in the workspace, and a gate's module-scoped fixture is rebuilt
in every process that runs one of its tests: each xdist worker, and each batch a test pool cuts
from the file. A :class:`ScanCache` keeps what a scan computed, so the next process that reads the
same bytes takes the answer instead of walking the trees again.

**A stale answer is impossible by construction.** An entry is found only by a key that hashes
everything the computation reads: the content of each input file (never its modification time,
which several filesystems cannot tell apart across quick rewrites), plus :attr:`ScanCache.code`,
which hashes the code that computes it -- the enforcement package, this directory's gate shells and
helpers, the lockfile and the interpreter's version. A store whose code key differs is another
file, and is never read.

**Concurrency.** Any number of processes may read and write one directory at once: up to four xdist
workers in one checkout, a pool's batches in one workspace. A store is written whole to a temporary
file beside it and renamed over it, so a reader sees an old store or a new one, never a partial
one. A store carries a checksum of its payload, and one that is truncated, corrupt or of another
format reads as empty. Writers never wait for each other: the last rename wins, and every store
holds a complete answer for the keys its writer looked up.

**Nothing here is required.** A missing directory is created; one that cannot be created, read or
written (a read-only checkout) leaves every lookup a miss, and the scan is computed as it would be
without the cache. The default directory, ``.cache/enforcement-scans`` under the repo root, is
gitignored.
"""

from __future__ import annotations

import hashlib
import os
import pickle
import sys
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: where :meth:`ScanCache.for_repo` keeps its stores, under the repo root (gitignored)
CACHE_DIRECTORY = Path(".cache") / "enforcement-scans"

#: the first line of every store; a file without it is not one, and reads as empty
_MAGIC = b"3tears-enforcement-scan-cache 1\n"

#: a temporary file older than this was left by a writer that died before its rename
_ORPHAN_AGE_S = 3600.0


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
    for path in sorted(paths):
        parts.append(path.relative_to(root).as_posix())
        parts.append(path.read_bytes())
    return content_key(*parts)


@dataclass(frozen=True)
class ScanCache:
    """a directory of stores, each a mapping of content key to computed value.

    :ivar directory: where the stores live; need not exist, nor be writable
    :ivar code: a key for the code that computes every value; a store written under another is
        never read
    """

    directory: Path
    code: str

    @classmethod
    def for_repo(cls, repo_root: Path) -> ScanCache:
        """the cache a gate in this directory uses: under the repo root, keyed by the gates' code.

        :param repo_root: the repo's root
        :ptype repo_root: Path
        :return: the cache
        :rtype: ScanCache
        :raises OSError: when a file of the gates' code cannot be read
        """
        package = repo_root / "packages" / "enforcement" / "src"
        gates = repo_root / "tests" / "enforcement"
        code_files = [*package.rglob("*.py"), *gates.glob("*.py"), repo_root / "uv.lock"]
        return cls(
            directory=repo_root / CACHE_DIRECTORY, code=content_key(sys.version, files_key(code_files, repo_root))
        )

    def memoised[K, V](
        self, name: str, items: Iterable[K], key: Callable[[K], str], compute: Callable[[K], V]
    ) -> dict[K, V]:
        """each item's value: the stored one when its key was computed before, else ``compute(item)``.

        An item computed here is keyed again afterwards, and kept only when its key has not moved: a
        file edited while it was being read is answered for this call and never stored under the
        content it no longer has. The store is rewritten when anything was computed, holding exactly
        the keys this call looked up, so an entry no current input needs is dropped, not kept forever.

        :param name: the store, one per kind of value
        :ptype name: str
        :param items: what to look up
        :ptype items: Iterable[K]
        :param key: an item's content key: a digest of everything its value is computed from
        :ptype key: Callable[[K], str]
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
            before = key(item)
            if before in kept:
                values[item] = kept[before]
            elif before in stored:
                values[item] = kept[before] = stored[before]
            else:
                computed = True
                values[item] = compute(item)
                if key(item) == before:
                    kept[before] = values[item]
        if computed or kept.keys() != stored.keys():
            self._store(name, kept)
        return values

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
        """write the store whole, then rename it into place; a directory that refuses it is skipped."""
        payload = pickle.dumps(entries, protocol=pickle.HIGHEST_PROTOCOL)
        blob = _MAGIC + hashlib.blake2b(payload, digest_size=32).hexdigest().encode() + b"\n" + payload
        target = self._path(name)
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            handle, temporary = tempfile.mkstemp(dir=self.directory, prefix=f".{name}-", suffix=".tmp")
        except OSError:
            return
        try:
            with os.fdopen(handle, "wb") as out:
                out.write(blob)
            os.replace(temporary, target)
        except OSError:
            Path(temporary).unlink(missing_ok=True)
            return
        self._prune(name, target)

    def _prune(self, name: str, kept: Path) -> None:
        """drop this store's files written under other code, and temporary files a dead writer left."""
        try:
            for path in self.directory.glob(f"{name}-*.pickle"):
                if path != kept:
                    path.unlink(missing_ok=True)
            for path in self.directory.glob(f".{name}-*.tmp"):
                if path.stat().st_mtime < kept.stat().st_mtime - _ORPHAN_AGE_S:
                    path.unlink(missing_ok=True)
        except OSError:
            return
