"""integration-test scope marker for 3tears-mcp.

the canonical session-scoped containers (``db_container``,
``nats_container``) come from the workspace root conftest's
``pytest_plugins = ["threetears.core.testing.fixtures"]`` declaration
(pytest 8+ requires that registration at the rootdir, not in nested
conftests). per-test fixtures unique to this package live in the test
modules.
"""

from __future__ import annotations
