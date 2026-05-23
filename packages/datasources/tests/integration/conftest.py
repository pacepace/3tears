"""integration-test scope marker for 3tears-datasources.

the canonical session-scoped ``db_container`` fixture comes from the
workspace root conftest's
``pytest_plugins = ["threetears.core.testing.fixtures"]`` declaration
(pytest 8+ requires that registration at the rootdir, not in nested
conftests). per-test fixtures unique to this package live in the test
modules.
"""

from __future__ import annotations
