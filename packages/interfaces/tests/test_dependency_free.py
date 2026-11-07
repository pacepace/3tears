"""The interfaces package's promise is an empty closure: importing it loads only the stdlib."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "module",
    ["threetears.interfaces", "threetears.interfaces.access", "threetears.interfaces.media"],
)
def test_importing_a_domain_loads_nothing_outside_the_stdlib(module: str) -> None:
    """Checked in a fresh interpreter, which nothing preloaded."""
    probe = (
        "import json, sys\n"
        "before = set(sys.modules)\n"
        f"import {module}\n"
        "print(json.dumps(sorted(set(sys.modules) - before)))\n"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=False, timeout=60)
    assert result.returncode == 0, result.stderr
    loaded: list[str] = json.loads(result.stdout)
    assert module in loaded
    offenders = [
        name
        for name in loaded
        if name.split(".")[0] not in sys.stdlib_module_names
        and not name.startswith(("_sysconfigdata", "threetears.interfaces"))
        and name != "threetears"
    ]
    assert offenders == []
