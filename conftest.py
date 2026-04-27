"""Top-level conftest for the 3tears workspace.

pytest 8+ requires ``pytest_plugins`` declarations to live in the
top-level conftest at the rootdir. cross-package + per-package runs
both resolve their rootdir to the workspace root (the parent
``pyproject.toml`` at this directory), so this file is the canonical
home for plugin registration.

re-exports the testcontainer + nats fixtures from
:mod:`threetears.core.testing.fixtures` (test-harness-task-01) so
every package's integration suite picks them up without redeclaring
``pytest_plugins`` in a nested conftest.
"""

from __future__ import annotations

pytest_plugins = ["threetears.core.testing.fixtures"]
