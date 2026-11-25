"""import-cost regression test: 3tears-models stays light.

guards the separate-concerns Phase 1a win: ``import threetears.models``
must not pull the agent stack, the core data layer, or any of the
heavyweight backend libraries onto the import path. a fresh interpreter
is used so this test is immune to whatever the surrounding pytest
process already imported.
"""

from __future__ import annotations

import json
import subprocess
import sys

# module prefixes that must NOT appear in sys.modules after importing
# threetears.models. each entry is a prefix match against module names.
_FORBIDDEN_PREFIXES = (
    "threetears.agent",
    "threetears.core",
    "threetears.nats",
    "sqlalchemy",
    "asyncpg",
    "pgvector",
    "nats",
)

_PROBE = """
import json
import sys

import threetears.models  # noqa: F401

prefixes = {prefixes!r}
loaded = sorted(
    name for name in sys.modules
    if any(name == p or name.startswith(p + ".") for p in prefixes)
)
print(json.dumps(loaded))
"""


class TestImportCost:
    def test_importing_models_does_not_load_agent_or_data_stack(self) -> None:
        probe = _PROBE.format(prefixes=_FORBIDDEN_PREFIXES)
        result = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, f"probe failed:\n{result.stderr}"
        loaded = json.loads(result.stdout.strip())
        assert loaded == [], f"importing threetears.models loaded forbidden modules: {loaded}"
