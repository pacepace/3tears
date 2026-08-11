"""shared test setup for the 3tears-search package test suite.

exposes ``_search_instances`` (the fully-populated contract instances the
round-trip and canonical suites share) to every test module by adding this
``tests`` directory to ``sys.path`` -- the same pattern the agent-workspace
suite uses for its ``_helpers``. a package-relative ``from tests.x import``
cannot work here: several workspace packages own a ``tests`` package, and
whichever one the workspace-wide run imports first would shadow this one.
"""

from __future__ import annotations

import sys
from pathlib import Path

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)
