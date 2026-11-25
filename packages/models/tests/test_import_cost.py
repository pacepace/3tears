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

import pytest

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


#: imports the OpenAI provider, then builds the first chat and embedding clients, as a server does on its event
#: loop, and lists the OpenAI SDK modules that build imported.
_OPENAI_PROBE = """
import json
import sys

import threetears.models.providers.openai as provider

resources_loaded = "openai.resources" in sys.modules
before = set(sys.modules)
provider.create_openai_chat("gpt-4o", api_key="probe-not-a-key")
provider.create_openai_embedding("text-embedding-3-small", api_key="probe-not-a-key")
built = sorted(name for name in set(sys.modules) - before if name == "openai" or name.startswith("openai."))
print(json.dumps({"resources_loaded": resources_loaded, "imported_at_first_build": built}))
"""


class TestOpenAIResourcesLoadAtImport:
    def test_the_first_client_build_imports_no_openai_module(self) -> None:
        """openai.resources is loaded by the provider's import, not on the loop at the first client (0.30 s on 2.x).

        :return: nothing
        :rtype: None
        """
        pytest.importorskip("openai")
        result = subprocess.run([sys.executable, "-c", _OPENAI_PROBE], capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, f"probe failed:\n{result.stderr}"
        probe = json.loads(result.stdout.strip().splitlines()[-1])
        assert probe == {"resources_loaded": True, "imported_at_first_build": []}
