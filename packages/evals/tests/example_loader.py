"""Load a file under ``examples/`` as a module, the way ``python examples/<file>.py`` would find its imports.

Run as a script, an example's own directory is first on ``sys.path``, so ``from _live import ...`` finds the
examples' shared live client. Loaded from a test it is not, and a test tree never changes ``sys.path``, so this
registers that one shared module under its own name instead.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

#: The examples directory.
EXAMPLES = Path(__file__).resolve().parents[1] / "examples"

#: The examples' one shared sibling module, which they import as ``from _live import ...``.
SHARED = "_live"


def load_example(name: str) -> ModuleType:
    """Execute ``examples/<name>`` as a fresh module.

    Args:
        name: The file's name, ``"llm_judge.py"``.

    Returns:
        The module, its ``main`` not yet run.
    """
    if SHARED not in sys.modules:
        sys.modules[SHARED] = _load(EXAMPLES / f"{SHARED}.py", SHARED)
    path = EXAMPLES / name
    return _load(path, f"{path.stem}_example")


def _load(path: Path, module_name: str) -> ModuleType:
    """Execute one file as a fresh module named ``module_name``."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
