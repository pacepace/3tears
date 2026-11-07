"""the probe: what a fresh interpreter loads to import one module, and what of it the floor permits.

Measured in a fresh interpreter, because whatever the test process already imported would
otherwise hide the cost entirely.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Iterable

from threetears.enforcement.import_floor.config import ImportFloorConfig

__all__ = ["ProbeError", "modules_loaded_by_importing", "outside_floor"]

#: CPython's build configuration module embeds the ABI flags and platform in its name, so it is
#: the interpreter rather than a dependency, yet it is absent from ``sys.stdlib_module_names``.
_INTERPRETER_ARTIFACT_PREFIXES = ("_sysconfigdata",)

#: the probe a fresh interpreter runs: refuse each root in ``refused`` as an install without it
#: would, import the module, and print every module the import loaded.
_PROBE = """
import importlib.abc
import json
import sys

refused = {refused!r}


class _Refuse(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if any(name == root or name.startswith(root + ".") for root in refused):
            raise ModuleNotFoundError(f"No module named {{name!r}}", name=name)
        return None


sys.meta_path.insert(0, _Refuse())
before = set(sys.modules)
import {module}  # noqa: F401
print(json.dumps(sorted(set(sys.modules) - before)))
"""


class ProbeError(RuntimeError):
    """the probe's interpreter could not import the module: it needs what the floor refuses, or it
    is broken. Either way there is no cost to measure, and the floor is not met."""


def modules_loaded_by_importing(module: str, *, refused: Iterable[str] = ()) -> tuple[str, ...]:
    """every module a fresh interpreter loads to import ``module``, with ``refused`` roots absent.

    :param module: the dotted module name to import
    :ptype module: str
    :param refused: import roots refused as an install without them refuses them
    :ptype refused: Iterable[str]
    :return: the module names the import loaded, sorted; ``module`` among them
    :rtype: tuple[str, ...]
    :raises ProbeError: when the import fails in the probe
    """
    result = subprocess.run(
        [sys.executable, "-c", _PROBE.format(module=module, refused=tuple(refused))],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if result.returncode != 0:
        raise ProbeError(f"importing {module} failed in a fresh interpreter:\n{result.stderr}")
    loaded: list[str] = json.loads(result.stdout.strip())
    return tuple(loaded)


def outside_floor(config: ImportFloorConfig, loaded: Iterable[str]) -> list[str]:
    """the modules in ``loaded`` the floor does not pay for.

    Permitted: the standard library, the interpreter's own build artifacts, the declared
    third-party roots, the declared family prefixes, and the namespace modules by exact name.

    :param config: the package's floor
    :ptype config: ImportFloorConfig
    :param loaded: dotted module names, from :func:`modules_loaded_by_importing`
    :ptype loaded: Iterable[str]
    :return: those outside the floor, sorted
    :rtype: list[str]
    """
    return sorted(name for name in loaded if not _permitted(config, name))


def _permitted(config: ImportFloorConfig, name: str) -> bool:
    """whether the floor pays for ``name``.

    :param config: the package's floor
    :ptype config: ImportFloorConfig
    :param name: a dotted module name
    :ptype name: str
    :return: whether it is stdlib, the interpreter's, a declared root's, or permitted family
    :rtype: bool
    """
    root = name.split(".")[0]
    return (
        root in sys.stdlib_module_names
        or name.startswith(_INTERPRETER_ARTIFACT_PREFIXES)
        or root in config.third_party_roots
        or any(name == prefix or name.startswith(f"{prefix}.") for prefix in config.family_prefixes)
        or name in config.namespace_modules
    )
