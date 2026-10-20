"""Core imports without its ``security`` and ``http`` extras.

``cryptography`` and ``pyjwt`` are the ``security`` extra, ``httpx`` the ``http`` extra. The tiers,
the L3 SQL backend, migrations, coordination, the utilities and the light security modules (the
sandbox, secret references, proof freshness) must import without them -- most of the family uses
core for those and nothing else, and paid for the signed-token stack anyway while a subpackage
``__init__`` loaded every module beside the one asked for.

Each check runs in a fresh interpreter with the extras' modules refused at import, as an install
without them refuses them, then asserts none of them was loaded.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

#: refuses the extras' distributions, as an install without them would
BLOCK_EXTRAS = """
import importlib.abc, sys
_REFUSED = ("cryptography", "jwt", "httpx")
class _NoExtras(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in _REFUSED:
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return None
sys.meta_path.insert(0, _NoExtras())
"""

#: core's path for a host that keeps state and signs nothing
CORE_PATH = (
    "threetears.core.entities.base",
    "threetears.core.cache.sqlite",
    "threetears.core.collections",
    "threetears.core.collections.base",
    "threetears.core.collections.registry",
    "threetears.core.data.store",
    "threetears.core.data.migrations",
    "threetears.core.coordination",
    "threetears.core.backends",
    "threetears.core.backends.protocol",
    "threetears.core.backends.sql",
    "threetears.core.utils",
    "threetears.core.utils.atomic_write",
    "threetears.core.security",
    "threetears.core.security.sandbox",
    "threetears.core.security.secret_refs",
    "threetears.core.security.freshness",
    "threetears.core.serialization",
    "threetears.core.authored_yaml",
)


def _run(code: str) -> subprocess.CompletedProcess[str]:
    """run *code* in a fresh interpreter.

    :param code: the program
    :ptype code: str
    :return: the finished process
    :rtype: subprocess.CompletedProcess[str]
    """
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False, timeout=120)


@pytest.mark.parametrize("module", CORE_PATH)
def test_a_core_path_module_imports_without_the_extras(module: str) -> None:
    code = BLOCK_EXTRAS + (
        f"import importlib\nimportlib.import_module({module!r})\n"
        "loaded = sorted(n for n in sys.modules if n.split('.')[0] in _REFUSED)\n"
        "assert not loaded, loaded\n"
        "print('imported')"
    )
    result = _run(code)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "imported"


@pytest.mark.parametrize(
    ("package", "name", "extra"),
    [
        ("threetears.core.security", "seal", "security"),
        ("threetears.core.security", "verify_identity_token", "security"),
        ("threetears.core.backends", "NatsProxyL3Backend", "security"),
    ],
)
def test_an_extras_backed_name_without_its_extra_names_the_extra(package: str, name: str, extra: str) -> None:
    code = BLOCK_EXTRAS + (
        f"import importlib\npackage = importlib.import_module({package!r})\n"
        "try:\n"
        f"    getattr(package, {name!r})\n"
        "except ImportError as exc:\n"
        f"    assert '3tears[{extra}]' in str(exc), str(exc)\n"
        "else:\n"
        f"    raise AssertionError('{name} resolved without the {extra} extra')\n"
        "print('refused')"
    )
    result = _run(code)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "refused"
