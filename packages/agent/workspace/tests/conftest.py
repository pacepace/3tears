"""shared test setup for the agent-workspace package test suite.

the fake NATS KV these tests use is published as
:mod:`threetears.core.testing.kv` and imported normally. it previously lived in core's
test tree and was reached by inserting that directory onto ``sys.path``, which meant a
double every consumer needed could only be had by a path hack.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _bind_test_subject_namespace(monkeypatch: pytest.MonkeyPatch) -> None:
    """bind a subject namespace for every workspace test.

    the production subject namespace has no default and must be
    configured explicitly (see
    :func:`threetears.nats.get_default_namespace`). workspace tools build
    subjects (e.g. ``Subjects.workspaces_create``) deep in their execute
    path, so bind a value via the environment variable here. this mirrors
    the root conftest fixture and additionally covers isolated per-package
    runs (``pytest packages/agent/workspace/tests/``), which do not load
    the workspace-root conftest.
    """
    monkeypatch.setenv("THREETEARS_NATS_SUBJECT_NAMESPACE", "3tears")


# shared test-infra lives under ``tests/_helpers/`` and is imported by its repo-root name::
#
#     from packages.agent.workspace.tests._helpers.asyncpg_shims import FakeAsyncpgConnection
#
# so the fake-protocol-parity walker has a single canonical class per shell type to subclass
# against. this directory is deliberately NOT put on ``sys.path``: its ``unit`` /
# ``integration`` / ``enforcement`` packages would then be importable as top-level names that
# the agent-tools suite also owns, and a combined run would resolve one suite's imports into
# the other's directory.
