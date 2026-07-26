"""Make this directory's shared helpers importable by the suites beside them.

Same pattern the workspace and scrape suites already use: put the test directory on
``sys.path`` so a sibling module resolves by name. ``from conftest import ...`` is deliberately
not the mechanism -- a root-level ``conftest.py`` shadows a nested one, so that import silently
resolves to the wrong file.

Import sites read::

    from _ruff_config_discovery import ruff_configs
"""

from __future__ import annotations

import sys
from pathlib import Path

_ENFORCEMENT_TESTS_ROOT = Path(__file__).resolve().parent
if str(_ENFORCEMENT_TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_ENFORCEMENT_TESTS_ROOT))
