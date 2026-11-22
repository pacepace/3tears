"""the write-mark queue's contract lands by its release: the empty ``collection_write_marks`` is dropped in 0.96.0.

0.95 took the mark triggers down (coordination v006) and kept the table, empty, for older replicas mid-roll
(:mod:`threetears.core.ddl.write_marks`). From :data:`WRITE_MARKS_CONTRACT_RELEASE` on this fails until a coordination
migration after v006 drops the table and the leftover-trigger sweep is gone.
"""

from __future__ import annotations

import inspect
from importlib.metadata import version

from threetears.core.collections import lost_writes
from threetears.core.coordination.migrations import register
from threetears.core.data.migrations import MigrationRunner
from threetears.core.ddl.write_marks import WRITE_MARKS_CONTRACT_RELEASE, WRITE_MARKS_TABLE


def _release(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(".")[:3])


def test_the_marks_table_is_dropped_by_its_contract_release() -> None:
    if _release(version("3tears")) < _release(WRITE_MARKS_CONTRACT_RELEASE):
        return
    later = [func for number, func in register(MigrationRunner()).versions.items() if number > 6]
    assert any(
        "DROP TABLE IF EXISTS" in inspect.getsource(func) and WRITE_MARKS_TABLE in inspect.getsource(func)
        for func in later
    ), (
        f"{WRITE_MARKS_CONTRACT_RELEASE} is the write-mark contract release: drop {WRITE_MARKS_TABLE} in a coordination "
        "migration after v006"
    )
    assert not hasattr(lost_writes.WriteBackstop, "drop_leftover_write_mark_triggers"), (
        "the leftover write-mark sweep is gone by now"
    )
