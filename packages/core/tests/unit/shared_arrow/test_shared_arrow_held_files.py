"""The store's tally of the files its directories hold: exact, and a change costs what it changes.

The contracts. Every directory's files are one immutable Arrow batch; the bytes held are a file's once however many
directories link it, at the newest size met, and go with the last directory holding it (checked against a plain
recount after every step of a seeded run). A publish's changes to the tally (its part directory, its version, a
worker's result linked in as a part, a sweep's removals) cost the same with 50k files recorded as with 5k over the
same 500 directories (within 2x, medians): the tally never walks every file, only references per directory.

Why: Pace's ruling (2026-11-23): no per-file Python objects in the store's state. Measured on this test's shapes
(Apple M-series, 2026-11-23): the per-inode dict tally took 42 us per publish, 67 us with a linked result; the Arrow
tally takes about 180 us, 740 us with a linked result, the same at 5k and 50k files.
"""

from __future__ import annotations

import random
import statistics
import time
from pathlib import Path

import pyarrow as pa

from threetears.core.shared_arrow import _files_batch, _HeldFiles

VERSIONS = Path("/store/versions")
HANDOFFS = Path("/store/handoffs")
PARTS = Path("/store/parts")
SEGMENTS = Path("/store/segments")

#: the directories recorded in the cost check, and a publish's part files
DIRECTORIES = 500
PUBLISHED = 100


def _batch(inodes: list[int], sizes: list[int], links: list[int]) -> pa.RecordBatch:
    return _files_batch([7] * len(inodes), inodes, sizes, links)


def test_the_tally_matches_a_recount_at_every_step() -> None:
    for seed in range(200):
        chance = random.Random(seed)
        held = _HeldFiles(VERSIONS, HANDOFFS)
        recorded: dict[Path, dict[int, int]] = {}
        newest: dict[int, int] = {}
        for _ in range(60):
            step = chance.random()
            if step < 0.5 or not recorded:
                path = chance.choice([VERSIONS, HANDOFFS, PARTS, SEGMENTS]) / f"d{chance.randrange(10)}"
                recorded.pop(path, None)
                files = {chance.randrange(1, 50): chance.choice([100, 200, 300]) for _ in range(chance.randrange(8))}
                links = [1 + sum(1 for other in recorded.values() if inode in other) for inode in files]
                held.add(path, _batch(list(files), list(files.values()), links))
                recorded[path] = files
                newest.update(files)
            elif step < 0.7:
                path = chance.choice(list(recorded))
                held.forget(path)
                del recorded[path]
            elif step < 0.85:
                path = chance.choice(list(recorded))
                gone = chance.sample(sorted(recorded[path]), min(len(recorded[path]), chance.randrange(3)))
                left = held.drop(path, pa.array(gone, pa.uint64()))
                for inode in gone:
                    del recorded[path][inode]
                assert left == len(recorded[path])
            else:
                path = chance.choice(list(recorded))
                renamed = path.parent / f".retired-{path.name}"
                if renamed in recorded:
                    continue
                held.moved(path, renamed)
                recorded[renamed] = recorded.pop(path)
            inodes = {inode for files in recorded.values() for inode in files}
            versions = tuple(sorted(p.name for p in recorded if p.parent == VERSIONS and not p.name.startswith(".")))
            handoffs = sum(1 for p in recorded if p.parent == HANDOFFS and not p.name.startswith("."))
            assert held.view == (sum(newest[inode] for inode in inodes), versions, handoffs), seed
            assert all(path in held for path in recorded)


def test_a_reused_number_is_counted_with_its_own_directory() -> None:
    held = _HeldFiles(VERSIONS, HANDOFFS)
    held.add(PARTS / "old", _batch([5, 6], [100, 50], [1, 1]))
    # the old file is gone and the system gave its number to a new one before the old directory was forgotten
    held.add(PARTS / "new", _batch([5], [300], [1]))
    held.forget(PARTS / "old")
    assert held.bytes == 300
    held.forget(PARTS / "new")
    assert held.bytes == 0


def test_a_linked_file_goes_with_its_last_directory() -> None:
    held = _HeldFiles(VERSIONS, HANDOFFS)
    held.add(SEGMENTS / "s-1", _batch([10, 11], [400, 40], [1, 1]))
    held.add(PARTS / "p", _batch([10, 12], [400, 20], [2, 1]))
    assert held.bytes == 460
    held.forget(SEGMENTS / "s-1")
    assert held.bytes == 420
    assert held.drop(PARTS / "p", pa.array([10], pa.uint64())) == 1
    assert held.bytes == 20


def _publish_cost(files_per_directory: int) -> tuple[float, float]:
    """median seconds of a publish's changes to the tally, without and with a linked result, over 500 directories.

    :return: (plain, linked)
    """
    held = _HeldFiles(VERSIONS, HANDOFFS)
    for each in range(DIRECTORIES):
        low = each * files_per_directory
        inodes = list(range(low, low + files_per_directory))
        held.add(PARTS / f"d{each}", _batch(inodes, [1000] * files_per_directory, [1] * files_per_directory))
    plain: list[float] = []
    linked: list[float] = []
    for each in range(150):
        low = 10**8 + each * 1000
        result = list(range(low, low + PUBLISHED))
        start = time.perf_counter()
        if each % 2:
            held.add(SEGMENTS / f"s-{each}", _batch(result, [1000] * PUBLISHED, [1] * PUBLISHED))
            start = time.perf_counter()
            held.add(PARTS / f"new-{each}", _batch(result, [1000] * PUBLISHED, [2] * PUBLISHED))
        else:
            held.add(PARTS / f"new-{each}", _batch(result, [1000] * PUBLISHED, [1] * PUBLISHED))
        version = list(range(low + 500, low + 505))
        held.add(VERSIONS / f"v-{each}", _batch(version, [10] * 5, [1] * 5))
        held.drop(PARTS / f"new-{each}", pa.array(result[:3], pa.uint64()))
        held.forget(VERSIONS / f"v-{each}")
        held.forget(PARTS / f"new-{each}")
        (linked if each % 2 else plain).append(time.perf_counter() - start)
        held.forget(SEGMENTS / f"s-{each}")
        assert held.bytes == DIRECTORIES * files_per_directory * 1000
    return statistics.median(plain), statistics.median(linked)


def test_a_publish_costs_the_same_at_ten_times_the_files() -> None:
    small_plain, small_linked = _publish_cost(10)
    large_plain, large_linked = _publish_cost(100)
    assert large_plain < 2 * small_plain, (small_plain, large_plain)
    assert large_linked < 2 * small_linked, (small_linked, large_linked)
