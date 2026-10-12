"""Core imports without the optional NATS client.

The client is core's ``nats`` extra. Using core must not require it: importing a
module must not load ``nats-py`` or ``nkeys``, or every consumer that builds on
core (``threetears.geo`` on ``DerivedCollection`` among them) fails to import
wherever the extra is not installed. That holds for the cross-pod lock too, which
lives in core (``threetears.core.coordination.distributed_lock``): it reaches the
broker only through the ``KvCapable`` its caller hands it, so it imports -- and
holds a key in a bucket that is not NATS -- with the client absent.

Each check runs in a fresh interpreter with ``nats`` and ``nkeys`` refused at
import, as an install without the extra refuses them, and then also asserts that
neither they nor the modules of ``threetears.nats`` that wrap them were loaded.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

#: refuses the client's distributions, as an install without the extra would
BLOCK_CLIENT = """
import importlib.abc, sys
class _NoClient(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("nats", "nkeys"):
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return None
sys.meta_path.insert(0, _NoClient())
"""


def test_derived_collection_imports_without_the_nats_client() -> None:
    code = BLOCK_CLIENT + "import threetears.core.collections.derived\nprint('imported')"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "imported"


def test_versioned_answers_import_without_the_nats_client() -> None:
    code = BLOCK_CLIENT + "import threetears.core.collections.versioned_answers\nprint('imported')"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "imported"


#: after the imports: nothing of the client, and none of the ``threetears.nats`` modules that wrap it
ASSERT_NO_CLIENT = """
loaded = sorted(
    name for name in sys.modules
    if name.split(".")[0] in ("nats", "nkeys")
    or name in ("threetears.nats.client", "threetears.nats.kv", "threetears.nats.object_store", "threetears.nats.oplog")
)
assert not loaded, f"importing core loaded the NATS client: {loaded}"
print("imported")
"""


def _run_without_the_client(body: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - this interpreter, a fixed script
        [sys.executable, "-c", BLOCK_CLIENT + body + ASSERT_NO_CLIENT], capture_output=True, text=True, check=False
    )


@pytest.mark.parametrize(
    "module",
    [
        "threetears.core",
        "threetears.core.coordination",
        "threetears.core.coordination.lease",
        "threetears.core.coordination.distributed_lock",
    ],
)
def test_core_and_its_lock_import_without_loading_the_nats_client(module: str) -> None:
    result = _run_without_the_client(f"import {module}\n")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "imported"


def test_every_public_name_of_core_coordination_resolves_without_the_nats_client() -> None:
    """the lock and the lease are reached by name, not only by module: resolving them loads no client either."""
    result = _run_without_the_client(
        "import threetears.core, threetears.core.coordination as coordination\n"
        "for name in coordination.__all__:\n"
        "    getattr(coordination, name)\n"
        "from threetears.core import KVLease\n"
        "from threetears.core.coordination import LockHeld, LockHold, LockLost, LockLossReason, nats_distributed_lock\n"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "imported"


def test_the_lock_holds_a_key_through_the_client_it_is_handed_with_no_nats_client_installed() -> None:
    """the lock's only way to a broker is its ``client`` argument: over a ``KvCapable`` that is not
    NATS it acquires, refuses a second holder and releases, with ``nats-py`` and ``nkeys`` refused."""
    result = _run_without_the_client(
        "import asyncio\n"
        "from threetears.core.coordination import LockHeld, nats_distributed_lock\n"
        "from threetears.core.testing.kv import FakeNatsClient\n"
        "async def main():\n"
        "    client = FakeNatsClient()\n"
        "    async with nats_distributed_lock(client, 'job'):\n"
        "        try:\n"
        "            async with nats_distributed_lock(client, 'job'):\n"
        "                raise AssertionError('two holders')\n"
        "        except LockHeld:\n"
        "            pass\n"
        "    async with nats_distributed_lock(client, 'job'):\n"
        "        pass\n"
        "    async with nats_distributed_lock(None, 'job'):\n"
        "        pass\n"
        "asyncio.run(main())\n"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "imported"
