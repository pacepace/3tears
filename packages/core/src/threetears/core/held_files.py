"""the shared Arrow store's tally of the files its directories hold (:class:`HeldFiles`), and the reads of a
directory's files it is built from.

Moved out of :mod:`threetears.core.shared_arrow` (2026-11-23), whose store holds one tally, so its own tests reach it
by its public names. Every directory's files are one immutable Arrow batch; the bytes held are a file's once however
many directories link it, and no file is a Python object in the tally's state.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from threetears.core.frozen_state import EMPTY, with_entries, without_entries

try:
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc

    from threetears.core.arrow_keys import key_codes

    _ARROW_EXTRA: ImportError | None = None
except ImportError as exc:  # NOSILENT: the snapshot extra is optional; the tally says so when built without it
    _ARROW_EXTRA = exc

__all__ = ["HeldFiles", "directory_bytes", "directory_files", "files_batch", "files_bytes"]


#: a directory's files as the tally records them (:func:`files_batch`): each file's device, inode, bytes, and the
#: names the file system gave it (``st_nlink``) when it was read
_FILE_COLUMNS = ("device", "inode", "bytes", "links")


def files_batch(devices: Sequence[int], inodes: Sequence[int], sizes: Sequence[int], links: Sequence[int]) -> Any:
    """a directory's files as one immutable Arrow batch (:data:`_FILE_COLUMNS`), each file once however many names it
    has there (the first name's row kept).

    :param devices: each file's device
    :ptype devices: Sequence[int]
    :param inodes: each file's inode
    :ptype inodes: Sequence[int]
    :param sizes: each file's bytes
    :ptype sizes: Sequence[int]
    :param links: each file's link count
    :ptype links: Sequence[int]
    :return: the batch
    :rtype: pa.RecordBatch
    """
    files = pa.record_batch(
        [
            pa.array(devices, pa.uint64()),
            pa.array(inodes, pa.uint64()),
            pa.array(sizes, pa.int64()),
            pa.array(links, pa.int64()),
        ],
        names=list(_FILE_COLUMNS),
    )
    # one directory's files are on one device: a repeated inode is a file under two names
    if files.num_rows < 2 or pc.count_distinct(files["inode"]).as_py() == files.num_rows:
        return files
    # a file under two names in one directory: rare, so the coding is paid only then. Each (device, inode) once, in
    # the order first listed: its first row's bytes and its most links, by Arrow's hash kernels (never Acero's
    # group_by: :mod:`threetears.core.arrow_keys`)
    (codes,) = key_codes([pa.table(files)], ["device", "inode"])
    _, first, each = np.unique(codes, return_index=True, return_inverse=True)
    most = np.zeros(len(first), dtype=np.int64)
    np.maximum.at(most, each, files["links"].to_numpy(zero_copy_only=False))
    order = np.argsort(first, kind="stable")
    kept = files.take(pa.array(first[order]))
    return pa.record_batch(
        [kept["device"], kept["inode"], kept["bytes"], pa.array(most[order], pa.int64())], names=list(_FILE_COLUMNS)
    )


def _bytes_by_key(held: Any) -> tuple[Any, Any, Any]:
    """each key ``held`` names once, with the bytes its rows hold together and how many rows name it: by Arrow's
    dictionary encoding (a hash kernel) and NumPy sums over its codes, never Acero's ``group_by``
    (:mod:`threetears.core.arrow_keys`).

    :param held: rows of ``key`` (text) and ``bytes`` (int64)
    :ptype held: pa.Table
    :return: the keys, each one's bytes summed, and each one's rows counted (int64 arrays, aligned)
    :rtype: tuple[pa.Array, pa.Array, pa.Array]
    """
    encoded = pc.dictionary_encode(held["key"].combine_chunks())
    codes = encoded.indices.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    size = len(encoded.dictionary)
    sums = np.zeros(size, dtype=np.int64)
    np.add.at(sums, codes, held["bytes"].to_numpy().astype(np.int64, copy=False))
    counts = np.bincount(codes, minlength=size).astype(np.int64, copy=False)
    return encoded.dictionary, pa.array(sums, pa.int64()), pa.array(counts, pa.int64())


def _file_keys(files: Any) -> Any:
    """each file's (device, inode) as one string column, for matching files across batches (Arrow has no is_in over
    a struct). Built only for the few rows a match needs.

    :param files: a batch or table with ``device`` and ``inode``
    :ptype files: pa.RecordBatch | pa.Table
    :return: ``"{device}:{inode}"`` per row
    :rtype: pa.Array | pa.ChunkedArray
    """
    return pc.binary_join_element_wise(pc.cast(files["device"], pa.string()), pc.cast(files["inode"], pa.string()), ":")


def files_bytes(files: Any) -> int:
    """the bytes a batch of files holds (:func:`files_batch`).

    :param files: the batch
    :ptype files: pa.RecordBatch
    :return: bytes
    :rtype: int
    """
    return _sum(files["bytes"])


def directory_files(path: Path) -> Any:
    """the files directly in a directory, by (device, inode), with their bytes and link counts: each file once however
    many names it has there.

    :param path: the directory
    :ptype path: Path
    :return: the files (:func:`files_batch`); empty when it is gone
    :rtype: pa.RecordBatch
    """
    devices: list[int] = []
    inodes: list[int] = []
    sizes: list[int] = []
    links: list[int] = []
    with contextlib.suppress(FileNotFoundError):  # NOSILENT: a directory removed meanwhile holds nothing
        for entry in path.iterdir():
            with contextlib.suppress(FileNotFoundError):  # NOSILENT: likewise a file
                stat = entry.stat()
                devices.append(stat.st_dev)
                inodes.append(stat.st_ino)
                sizes.append(stat.st_size)
                links.append(stat.st_nlink)
    return files_batch(devices, inodes, sizes, links)


def directory_bytes(path: Path, seen: set[tuple[int, int]] | None = None) -> int:
    """the bytes the files directly in a directory hold, each file once however many names it has.

    A claimed result linked into a version as a part is one file under two names, whose pages exist once: ``seen``,
    shared across directories, counts each file (by device and inode) the first time it is met.

    :param path: the directory
    :ptype path: Path
    :param seen: the files counted already, by (device, inode); None to count this directory alone
    :ptype seen: set[tuple[int, int]] | None
    :return: bytes; 0 when it is gone
    :rtype: int
    """
    counted = set() if seen is None else seen
    total = 0
    with contextlib.suppress(FileNotFoundError):  # NOSILENT: a directory removed meanwhile holds nothing
        for entry in path.iterdir():
            with contextlib.suppress(FileNotFoundError):  # NOSILENT: likewise a file
                stat = entry.stat()
                key = (stat.st_dev, stat.st_ino)
                if key not in counted:
                    counted.add(key)
                    total += stat.st_size
    return total


class HeldFiles:
    """the store's own tally of the files its directories hold, so :meth:`SharedArrowStore.status` reads numbers and
    never the file system: each directory's files recorded as it takes its final name (a version published, the part
    files a publish wrote, a hand-off written, a result claimed, a directory found at the start) and forgotten once it
    is removed; a part file is dropped from its directory's record as it is removed (:meth:`drop`).

    Exact as :meth:`SharedArrowStore.held_bytes` counts: a file is counted once however many directories hold a link
    to it (a claimed result linked as a part), by (device, inode), and its bytes go only when the last directory holding
    it goes. Files never change after their directory takes its name, so what a directory held then is what it holds
    until it is removed, or a part file of it is. An inode met again at another size is a file the system reused the
    number for (the old one removed while its directory's removal was still being recorded): the newer size is the true
    one. Mutated under the store's ``_state``; :attr:`view` is replaced whole on each change, so a reader takes it
    without the lock.

    The shape (Pace, 2026-11-23): each directory's files are one immutable Arrow batch (:func:`files_batch`), held in
    a read-only mapping replaced whole, so a change costs references per directory, never a Python object per file.
    The bytes are a running total kept per change, not a group_by over every batch when read: the held-bytes gauge
    and a publish's room read it at every change, and the group_by took 3.8 ms at 50k files over 500 directories
    against the running total's tens of microseconds (2026-11-23). A file only one directory holds is counted as it
    comes and goes. A file more directories hold is met with a link count above one (``st_nlink`` when its second
    directory was read): only then are the other directories' batches searched for it (Arrow, by inode), and it goes
    into :attr:`_shared`, a small Arrow table of such files with how many directories hold each and its bytes, which
    every add and removal consults by inode first.

    :ivar view: (bytes held, the versions' names, the hand-offs held), each as of the last change
    :raises ImportError: when built where pyarrow does not import
    """

    def __init__(self, versions: Path, handoffs: Path) -> None:
        if _ARROW_EXTRA is not None:
            raise ImportError(f"the held-files tally needs pyarrow: install 3tears[snapshot] ({_ARROW_EXTRA})")
        self._versions = versions
        self._handoffs = handoffs
        #: directory (its path as a string: a str caches its hash, a Path does not, and the mapping is copied whole at
        #: each change) -> its files (:func:`files_batch`) with the lowest and highest inode among them (a directory's
        #: files are made together, so their numbers sit close: what lets a linked file's search skip every directory
        #: whose span misses it); each batch immutable and the mapping replaced whole
        self._dirs: Mapping[str, tuple[Any, int, int]] = EMPTY
        #: the files more than one recorded directory has held: key (``"{device}:{inode}"``), inode, the directories
        #: holding it now, its bytes; replaced whole, and only as large as the hard-linked files
        self._shared: Any = pa.table(
            {
                "key": pa.array([], pa.string()),
                "inode": pa.array([], pa.uint64()),
                "dirs": pa.array([], pa.int64()),
                "bytes": pa.array([], pa.int64()),
            }
        )
        #: the versions and hand-offs recorded under their own names: what :attr:`view` lists, kept as directories are
        #: recorded, renamed and forgotten, so a change never walks every directory recorded (a part directory each
        #: publish whose files a version still names)
        self._listed: frozenset[Path] = frozenset()
        self._bytes = 0
        self.view: tuple[int, tuple[str, ...], int] = (0, (), 0)
        #: told the bytes held at each change (the store's held-bytes gauge); None for nobody
        self.on_view: Callable[[int], object] | None = None

    @property
    def bytes(self) -> int:
        """the bytes recorded. Read under the store's ``_state``.

        :return: bytes
        :rtype: int
        """
        return self._bytes

    def __contains__(self, path: Path) -> bool:
        """whether a directory is recorded.

        :param path: the directory
        :ptype path: Path
        :return: whether it is
        :rtype: bool
        """
        return str(path) in self._dirs

    def add(self, path: Path, files: Any) -> None:
        """record a directory's files (replacing what it was recorded with).

        :param path: the directory
        :ptype path: Path
        :param files: its files (:func:`directory_files`, :func:`files_batch`)
        :ptype files: pa.RecordBatch
        :return: nothing
        :rtype: None
        """
        self._forget(path)
        self._bytes += self._linked(files)
        span = pc.min_max(files["inode"]).as_py()
        low, high = (1, 0) if span["min"] is None else (span["min"], span["max"])
        self._dirs = with_entries(self._dirs, {str(path): (files, low, high)})
        if self._lists(path):
            self._listed = self._listed | {path}
        self._viewed()

    def drop(self, path: Path, inodes: Any) -> int | None:
        """files of a recorded directory were removed: each goes from its record, its bytes with it when no other
        directory holds it.

        :param path: the directory
        :ptype path: Path
        :param inodes: the files removed, by inode (one directory's files are on one device)
        :ptype inodes: pa.Array
        :return: the files its record holds now; None when it is not recorded
        :rtype: int | None
        """
        recorded = self._dirs.get(str(path))
        if recorded is None:
            return None
        files, low, high = recorded
        gone = pc.is_in(files["inode"], value_set=inodes)
        if pc.any(gone).as_py():
            self._unlinked(files.filter(gone))
            files = files.filter(pc.invert(gone))
            # the span kept: still covering what is left
            self._dirs = with_entries(self._dirs, {str(path): (files, low, high)})
        self._viewed()
        return int(files.num_rows)

    def moved(self, old: Path, new: Path) -> None:
        """a directory was renamed (retired out of its name): its files are recorded under its new name.

        :param old: its name before
        :ptype old: Path
        :param new: its name now
        :ptype new: Path
        :return: nothing
        :rtype: None
        """
        files = self._dirs.get(str(old))
        if files is not None:
            self._dirs = with_entries(without_entries(self._dirs, (str(old),)), {str(new): files})
            listed = self._listed - {old}
            self._listed = listed | {new} if self._lists(new) else listed
            self._viewed()

    def forget(self, path: Path) -> None:
        """a directory was removed: its files go, each whose last directory it was.

        :param path: the directory
        :ptype path: Path
        :return: nothing
        :rtype: None
        """
        if self._forget(path):
            self._viewed()

    def _forget(self, path: Path) -> bool:
        """drop a directory's record, and each file's bytes whose last directory it was; :attr:`view` unchanged.

        :param path: the directory
        :ptype path: Path
        :return: whether it was recorded
        :rtype: bool
        """
        recorded = self._dirs.get(str(path))
        if recorded is None:
            return False
        files = recorded[0]
        self._dirs = without_entries(self._dirs, (str(path),))
        self._listed = self._listed - {path}
        self._unlinked(files)
        return True

    def _linked(self, files: Any) -> int:
        """take a directory's files into :attr:`_shared` where another directory holds them too; called before the
        directory is in :attr:`_dirs`.

        A file is looked for in the other directories only when it has more than one name (``links``) or is in
        :attr:`_shared` already: one name means no other directory links it, except a number reused while an old
        directory's removal is still being recorded, which is counted on its own and goes with its own directory.

        :param files: the directory's files
        :ptype files: pa.RecordBatch
        :return: the bytes held grow by
        :rtype: int
        """
        total = files_bytes(files)
        if not files.num_rows:
            return 0
        shared = self._shared
        candidate = pc.greater(files["links"], 1)
        if shared.num_rows:
            candidate = pc.or_(candidate, pc.is_in(files["inode"], value_set=shared["inode"]))
        if not pc.any(candidate).as_py():
            return total
        linked = files.filter(candidate)
        keys = _file_keys(linked)
        sizes = linked["bytes"]
        delta = total - files_bytes(linked)
        if shared.num_rows:
            # files _shared knows: one directory more, at the newer size
            where = pc.index_in(shared["key"], value_set=keys)
            known = pc.is_valid(where)
            newer = pc.take(sizes, where)
            delta += _sum(pc.subtract(newer, shared["bytes"]))
            shared = pa.table(
                {
                    "key": shared["key"],
                    "inode": shared["inode"],
                    "dirs": pc.add(shared["dirs"], pc.cast(known, pa.int64())),
                    "bytes": pc.coalesce(newer, shared["bytes"]),
                }
            )
            fresh = pc.invert(pc.is_in(keys, value_set=shared["key"]))
        else:
            fresh = pc.is_valid(keys)
        if pc.any(fresh).as_py():
            # files _shared does not know: the other directories holding each (one, but for a reused number), whose
            # bytes the total counts once each
            fresh_files = linked.filter(fresh)
            fresh_keys = keys.filter(fresh)
            span = pc.min_max(fresh_files["inode"]).as_py()
            others = [
                batch
                for batch, low, high in self._dirs.values()
                if batch.num_rows and low <= span["max"] and high >= span["min"]
            ]
            held = pa.table(
                {
                    "key": pa.array([], pa.string()),
                    "bytes": pa.array([], pa.int64()),
                }
            )
            if others:
                every = pa.Table.from_batches(others)
                near = every.filter(pc.is_in(every["inode"], value_set=fresh_files["inode"]))
                near_keys = _file_keys(near)
                exact = pc.is_in(near_keys, value_set=fresh_keys)
                held = pa.table({"key": near_keys.filter(exact), "bytes": near["bytes"].filter(exact)})
            counted_keys, counted_bytes, counted_holders = _bytes_by_key(held)
            at = pc.index_in(fresh_keys, value_set=counted_keys)
            before = pc.take(counted_bytes, at)
            holders = pc.take(counted_holders, at)
            fresh_sizes = fresh_files["bytes"]
            # held elsewhere: replaces what was counted for it; held nowhere else: counted as it comes
            delta += _sum(pc.subtract(fresh_sizes, pc.coalesce(before, 0)))
            elsewhere = pc.is_valid(at)
            if pc.any(elsewhere).as_py():
                shared = pa.concat_tables(
                    [
                        shared,
                        pa.table(
                            {
                                "key": fresh_keys.filter(elsewhere),
                                "inode": fresh_files["inode"].filter(elsewhere),
                                "dirs": pc.add(holders.filter(elsewhere), 1),
                                "bytes": fresh_sizes.filter(elsewhere),
                            }
                        ),
                    ]
                ).combine_chunks()
        self._shared = shared
        return delta

    def _unlinked(self, files: Any) -> None:
        """one directory's links to files are gone: a file's bytes go when it was the last directory holding it.

        :param files: the files, of one directory
        :ptype files: pa.RecordBatch
        :return: nothing
        :rtype: None
        """
        gone = files_bytes(files)
        shared = self._shared
        if shared.num_rows:
            maybe = pc.is_in(files["inode"], value_set=shared["inode"])
            if pc.any(maybe).as_py():
                keys = _file_keys(files.filter(maybe))
                # a file _shared counts goes with its last directory, at the bytes _shared holds
                gone -= _sum(files.filter(maybe)["bytes"].filter(pc.is_in(keys, value_set=shared["key"])))
                dirs = pc.subtract(shared["dirs"], pc.cast(pc.is_in(shared["key"], value_set=keys), pa.int64()))
                last = pc.equal(dirs, 0)
                gone += _sum(shared["bytes"].filter(last))
                shared = shared.set_column(2, "dirs", dirs).filter(pc.invert(last))
                self._shared = shared
        self._bytes -= gone

    def _lists(self, path: Path) -> bool:
        """whether :attr:`view` lists a directory: a version or a hand-off under its own name.

        :param path: the directory
        :ptype path: Path
        :return: whether it does
        :rtype: bool
        """
        return path.parent in (self._versions, self._handoffs) and not path.name.startswith(".")

    def _viewed(self) -> None:
        """replace :attr:`view` with what is recorded now.

        :return: nothing
        :rtype: None
        """
        versions = tuple(sorted(path.name for path in self._listed if path.parent == self._versions))
        handoffs = sum(1 for path in self._listed if path.parent == self._handoffs)
        self.view = (self._bytes, versions, handoffs)
        if self.on_view is not None:
            self.on_view(self._bytes)


def _sum(values: Any) -> int:
    """the sum of an Arrow integer column, nulls skipped; 0 when empty.

    :param values: the column
    :ptype values: pa.Array | pa.ChunkedArray
    :return: the sum
    :rtype: int
    """
    total = pc.sum(values).as_py()
    return 0 if total is None else int(total)
