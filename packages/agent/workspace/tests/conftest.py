"""shared test setup for the agent-workspace package test suite.

exposes the core coordination fake NATS KV helpers (defined in the core
package's test tree) to workspace tests by adding the core tests root to
``sys.path``. the fake NATS KV is a test-only helper exercising the
semantics of ``nats-py`` :class:`KeyValue` that :class:`KVLease` depends
on; reusing it here rather than duplicating the implementation keeps the
workspace lease-wrapper tests in lockstep with core's lease tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

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


_CORE_COORDINATION_TESTS = (
    Path(__file__).resolve().parent.parent.parent.parent / "core" / "tests" / "unit" / "coordination"
)
if str(_CORE_COORDINATION_TESTS) not in sys.path:
    sys.path.insert(0, str(_CORE_COORDINATION_TESTS))

# expose ``_helpers.asyncpg_shims`` (and any future shared test-infra
# packages) to every workspace test by adding this ``tests`` directory
# to ``sys.path``. import sites use::
#
#     from _helpers.asyncpg_shims import FakeAsyncpgConnection
#
# centralised test fakes live under ``tests/_helpers/`` so the
# fake-protocol-parity walker has a single canonical class per shell
# type to subclass against (each per-test ``_FakePool`` /
# ``_FakeConnection`` etc. inherits the matching shell to declare
# parity).
_WORKSPACE_TESTS_ROOT = Path(__file__).resolve().parent
if str(_WORKSPACE_TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_TESTS_ROOT))
