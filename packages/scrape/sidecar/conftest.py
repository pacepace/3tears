"""Make ``main`` importable from tests/ without installing this as a package.

Separate deployable, separate CI concern -- its own standalone
``pyproject.toml``/venv, not part of the 3tears workspace's own
``uv run pytest``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
