"""Load a file under ``examples/`` as a module, the way ``python examples/<file>.py`` would find its imports.

Run as a script, an example's own directory is first on ``sys.path``, so ``from _live import ...`` finds the
examples' shared live client. Loaded from a test it is not, so this puts it there for the load.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

#: The examples directory.
EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def load_example(name: str) -> ModuleType:
    """Execute ``examples/<name>`` as a fresh module.

    Args:
        name: The file's name, ``"llm_judge.py"``.

    Returns:
        The module, its ``main`` not yet run.
    """
    path = EXAMPLES / name
    spec = importlib.util.spec_from_file_location(f"{path.stem}_example", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(EXAMPLES))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(EXAMPLES))
    return module
