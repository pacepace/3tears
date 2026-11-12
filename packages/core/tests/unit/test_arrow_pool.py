"""Arrow's memory pool in a serving process (``threetears.core.arrow_pool``): the system allocator for that process
only, and a release of what the pool holds unused.

Each case runs in a fresh interpreter: the default pool is the process's, and switching it here would switch it for
every test this worker runs after.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import pyarrow as pa  # type: ignore[import-untyped]

from threetears.core.arrow_pool import release_unused


def _run(body: str) -> dict[str, object]:
    script = textwrap.dedent(body)
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=120, check=False)
    assert done.returncode == 0, done.stderr[-3000:]
    answer: dict[str, object] = json.loads(done.stdout.strip().splitlines()[-1])
    return answer


def test_the_serving_process_allocates_from_the_system_pool_after_the_call() -> None:
    """Flips if after the call the default pool is not the system allocator, or a new array is not allocated from
    it, or the call does not say which pool it replaced."""
    answer = _run(
        """
        import json
        import pyarrow as pa
        from threetears.core.arrow_pool import use_system_pool_in_serving_process
        default = pa.default_memory_pool().backend_name
        replaced = use_system_pool_in_serving_process()
        before = pa.system_memory_pool().bytes_allocated()
        held = pa.array(range(1_000_000))
        grew = pa.system_memory_pool().bytes_allocated() - before
        print(json.dumps({"default": default, "replaced": replaced,
                          "backend": pa.default_memory_pool().backend_name, "grew": grew, "len": len(held)}))
        """
    )
    assert answer["replaced"] == answer["default"]
    assert answer["backend"] == "system"
    assert isinstance(answer["grew"], int) and answer["grew"] >= 8_000_000, answer


def test_a_second_call_changes_nothing_and_says_system() -> None:
    """Flips if a second call switches again or answers another backend than the system pool it already is."""
    answer = _run(
        """
        import json
        import pyarrow as pa
        from threetears.core.arrow_pool import use_system_pool_in_serving_process
        use_system_pool_in_serving_process()
        again = use_system_pool_in_serving_process()
        print(json.dumps({"again": again, "backend": pa.default_memory_pool().backend_name}))
        """
    )
    assert answer == {"again": "system", "backend": "system"}


def test_a_spawned_worker_keeps_the_default_pool() -> None:
    """Flips if a worker process started by spawn after the switch allocates from the system pool: the switch is the
    serving process's alone, and a worker keeps pyarrow's default (mimalloc)."""
    answer = _run(
        """
        import json
        import multiprocessing
        import pyarrow as pa
        from threetears.core.arrow_pool import use_system_pool_in_serving_process
        default = pa.default_memory_pool().backend_name
        use_system_pool_in_serving_process()
        if __name__ == "__main__":
            with multiprocessing.get_context("spawn").Pool(1) as workers:
                child = workers.apply(eval, ("__import__('pyarrow').default_memory_pool().backend_name",))
            print(json.dumps({"default": default, "child": child, "parent": pa.default_memory_pool().backend_name}))
        """
    )
    assert answer["parent"] == "system"
    assert answer["child"] == answer["default"], answer


def test_release_unused_answers_the_bytes_still_allocated() -> None:
    """Flips if the release does not answer the default pool's bytes still allocated after it."""
    held = pa.array(range(100_000))
    allocated = release_unused()
    assert allocated == pa.default_memory_pool().bytes_allocated()
    assert allocated >= held.nbytes
