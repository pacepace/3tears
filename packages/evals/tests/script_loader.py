"""Load a file under ``scripts/`` as a module, the way ``python scripts/<file>.py`` would find its imports.

Run as a script, a dev tool's own directory is first on ``sys.path``, so ``from public_surface import ...`` finds the
walk over the public roots the tools share. Loaded from a test it is not, and a test tree never changes
``sys.path``, so this registers that one shared module under its own name instead.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

#: The dev tooling directory beside the package.
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"

#: The scripts' one shared sibling module, which they import as ``from public_surface import ...``.
SHARED = "public_surface"


def load_script(name: str) -> ModuleType:
    """Execute ``scripts/<name>`` as a module registered under its stem, its ``main`` not run. The shared module is
    executed once, so every script loaded reads the same one.

    :param name: The file's name, ``"generate_reference.py"``.
    :ptype name: str
    :return: The module.
    :rtype: ModuleType
    """
    if SHARED not in sys.modules:
        _load(SCRIPTS / f"{SHARED}.py", SHARED)
    path = SCRIPTS / name
    if path.stem == SHARED:
        return sys.modules[SHARED]
    return _load(path, path.stem)


def _load(path: Path, module_name: str) -> ModuleType:
    """Execute one file as a module named ``module_name``, registered before it runs, as an import would be: the
    scripts declare dataclasses, which read their module."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
