"""shared test setup for the agent-workspace package test suite.

the fake NATS KV these tests use is published as
:mod:`threetears.core.testing.kv` and imported normally. it previously lived in core's
test tree and was reached by inserting that directory onto ``sys.path``, which meant a
double every consumer needed could only be had by a path hack.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _bind_workspace_subject_namespace(monkeypatch: pytest.MonkeyPatch) -> None:
    """bind the ``3tears`` subject namespace for every workspace test.

    workspace tools build subjects (e.g. ``Subjects.workspaces_create``) deep in their execute
    path, and these tests assert the ``3tears`` prefix. the root conftest's
    ``_bind_test_subject_namespace`` runs first and clears the process-wide namespace global; this
    then overrides the environment variable it set. the name must differ from the root fixture's:
    sharing it replaced that fixture, so the global was never cleared, and a namespace another
    module set at import time leaked into whichever workspace test ran first in the process.
    """
    monkeypatch.setenv("THREETEARS_NATS_SUBJECT_NAMESPACE", "3tears")


# shared test-infra lives under ``tests/helpers/`` and is imported by its repo-root name::
#
#     from packages.agent.workspace.tests.helpers.asyncpg_shims import FakeAsyncpgConnection
#
# so the fake-protocol-parity walker has a single canonical class per shell type to subclass
# against. this directory is deliberately NOT put on ``sys.path``: its ``unit`` /
# ``integration`` / ``enforcement`` packages would then be importable as top-level names that
# the agent-tools suite also owns, and a combined run would resolve one suite's imports into
# the other's directory.
