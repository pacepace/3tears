"""Shared Arrow's store and pool need pyarrow alone: a process without duckdb (the hub, offloading its broker's
work) hands, reserves, claims and runs jobs; only the calls that open DuckDB say the extra is missing.

The check runs in a fresh interpreter whose ``import duckdb`` fails (a ``sys.meta_path`` blocker installed at the top
of its main script, which every spawned worker runs again), so neither the serving process nor a worker can reach
duckdb however it is installed here.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

_SCRIPT = textwrap.dedent(
    """
    import importlib.abc
    import sys


    class _NoDuckdb(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name == "duckdb" or name.startswith("duckdb."):
                raise ImportError("duckdb is blocked in this process")
            return None


    sys.meta_path.insert(0, _NoDuckdb())

    import asyncio
    import json

    import pyarrow as pa

    from threetears.core.shared_arrow import SharedArrowPool, SharedArrowStore


    def handed_rows(context, column):
        return sum(context.tables.tables["t"].column(column).to_pylist())


    def write_rows(context, rows):
        context.result.write({"r": pa.table({"a": pa.array(range(rows), pa.int64())})})
        return "duckdb" in sys.modules


    def _raises(call):
        try:
            call()
        except ImportError as exc:
            return str(exc)
        return None


    async def main(root):
        said = {}
        store = SharedArrowStore(root, max_bytes=64 << 20)
        table = pa.table({"a": pa.array([1, 2, 3], pa.int64())})

        handed = store.hand({"t": table})
        with handed.mapped() as mapped:
            said["handed"] = mapped.tables["t"].column("a").to_pylist()
        store.release(handed)

        slot = store.reserve(1 << 20)
        slot.write({"r": table})
        claimed = store.claim(slot)
        said["claimed"] = claimed.tables["r"].num_rows
        claimed.close()
        dropped = store.reserve(1 << 20)
        dropped.write({"r": table})
        store.discard(dropped)
        said["left"] = sorted(p.name for p in (store.root / "segments").iterdir() if not p.name.startswith("."))

        pool = SharedArrowPool(store, workers=1, private_bytes=1 << 40, name="no-duckdb")
        try:
            said["pool_handed"] = (await pool.run(handed_rows, "a", read=False, tables={"t": table})).value
            ran = await pool.run(write_rows, 5, read=False, result_bytes=1 << 20)
            said["worker_saw_duckdb"] = ran.value
            said["pool_result"] = ran.segment.tables["r"].column("a").to_pylist()
            ran.segment.close()
        finally:
            await pool.close()

        store.publish("v1", {"t": table})
        said["connect"] = _raises(lambda: claimed_connect(store))
        said["reader"] = _raises(lambda: store.reader())
        said["publish_query"] = _raises(lambda: store.publish_query("v2", object(), {"t": "SELECT 1"}))
        said["duckdb_loaded"] = "duckdb" in sys.modules
        print(json.dumps(said))


    def claimed_connect(store):
        slot = store.reserve(1 << 20)
        slot.write({"r": pa.table({"a": [1]})})
        claimed = store.claim(slot)
        try:
            claimed.connect()
        finally:
            claimed.close()


    if __name__ == "__main__":
        asyncio.run(main(sys.argv[1]))
    """
)


def test_the_store_and_pool_run_without_duckdb(tmp_path: Path) -> None:
    script = tmp_path / "no_duckdb_main.py"
    script.write_text(_SCRIPT, encoding="utf-8")
    answer = subprocess.run(
        [sys.executable, str(script), str(tmp_path / "shm")], capture_output=True, text=True, timeout=120
    )
    assert answer.returncode == 0, answer.stderr
    said = json.loads(answer.stdout.strip().splitlines()[-1])
    assert said["handed"] == [1, 2, 3]
    assert said["claimed"] == 3
    assert said["left"] == []
    assert said["pool_handed"] == 6
    assert said["worker_saw_duckdb"] is False
    assert said["pool_result"] == [0, 1, 2, 3, 4]
    for call in ("connect", "reader", "publish_query"):
        assert said[call] is not None, f"{call} ran without duckdb"
        assert "3tears[snapshot]" in said[call] and "duckdb" in said[call], said[call]
    assert said["duckdb_loaded"] is False
