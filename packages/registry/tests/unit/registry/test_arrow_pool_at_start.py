"""the registry's serving process puts Arrow on the system allocator as it starts, before its first dependency.

The default pool is the process's, so the start runs in a fresh interpreter, as its Deployment runs it
(``python -m threetears.registry``): switching it here would switch it for every test this worker runs after.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pytest

#: the registry started up to its authorizer, the first thing it builds after logging (allow-all mode, so no hub
#: is needed), which answers Arrow's default pool there
_START = textwrap.dedent(
    """
    import json
    import runpy

    import pyarrow as pa

    import threetears.registry.server as server

    class FirstDependency(Exception):
        pass

    def first_dependency(*args, **kwargs):
        raise FirstDependency(pa.default_memory_pool().backend_name)

    server.AllowAllAuthorizer = first_dependency
    default = pa.default_memory_pool().backend_name
    try:
        runpy.run_module("threetears.registry", run_name="__main__", alter_sys=True)
    except FirstDependency as reached:
        print(json.dumps({"default": default, "at_first_dependency": str(reached)}))
    else:
        raise AssertionError("the start never reached its first dependency")
    """
)


def test_the_registry_allocates_arrow_from_the_system_pool_before_its_first_dependency() -> None:
    """Flips if the registry reaches its authorizer with Arrow's default pool still pyarrow's own (mimalloc): it loads
    pyarrow through core's keyed reads, and mimalloc held the serving thread 5-15 ms in about a quarter of Arrow builds
    (``arrow_copy``, 2026-11-15)."""
    pytest.importorskip("pyarrow")
    env = {name: value for name, value in os.environ.items() if name != "ARROW_DEFAULT_MEMORY_POOL"}
    env["THREETEARS_REGISTRY_ALLOW_ALL_TOOLS"] = "true"
    done = subprocess.run(  # noqa: S603
        [sys.executable, "-c", _START], capture_output=True, text=True, timeout=120, check=False, env=env
    )
    assert done.returncode == 0, done.stderr[-3000:]
    answer = json.loads(done.stdout.strip().splitlines()[-1])
    assert answer["default"] != "system", f"the test needs pyarrow's own default pool to start from: {answer}"
    assert answer["at_first_dependency"] == "system", answer
