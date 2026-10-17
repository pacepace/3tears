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

also installs a warnings filter for the upstream
``LangChainPendingDeprecationWarning`` raised by
``langgraph.checkpoint.serde.jsonplus``'s module-level
``LC_REVIVER = Reviver()``: that ``Reviver`` is constructed without
``allowed_objects`` because the module is library code we do not own,
and the warning is fired at import time -- before the
``[tool.pytest.ini_options].filterwarnings`` setting is applied. our
own ``threetears.langgraph.serde.UUIDSafeSerializer`` constructs a
``JsonPlusSerializer()`` (no ``Reviver`` kwargs are exposed on that
surface), so there is nothing for us to fix at the call site; revisit
when langchain-core flips the default in a future release.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator

import pytest

# Importing langchain_core re-enables LangChain's own deprecation warnings: its
# ``__init__`` calls ``surface_langchain_deprecation_warnings()``, which prepends a
# ``default``-action filter for its ``LangChainPendingDeprecationWarning`` and
# ``LangChainDeprecationWarning``. A filter installed BEFORE that import sits behind those
# and loses. So langchain_core is imported first, through its public package, and the
# ignores below are prepended after it, in front of its own. They match the stdlib bases
# the LangChain classes derive from (a filter's category matches subclasses), narrowed by
# the one message, so nothing here binds to ``langchain_core._api``, which is private.
import langchain_core  # noqa: E402, F401

warnings.filterwarnings(
    "ignore",
    message=r"The default value of .allowed_objects. will change in a future version",
    category=PendingDeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"The default value of .allowed_objects. will change in a future version",
    category=DeprecationWarning,
)

# ``tests.contain_tripwire`` fails a test whose code under test contained a programming error
# (see its docstring); ``pytester`` lets its own test run a session with it installed.
pytest_plugins = ["threetears.core.testing.fixtures", "tests.contain_tripwire", "pytester"]


@pytest.fixture(autouse=True)
def _bind_test_subject_namespace(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """bind a subject namespace for every test, isolated from its neighbours.

    the production subject namespace has no default and must be configured
    explicitly (see :func:`threetears.nats.get_default_namespace`). tests get a
    convenience value here so subject-building code paths resolve without each
    test wiring it up. it is provided via the environment variable so it is
    visible across every event-loop task and thread.

    :func:`threetears.nats.set_default_namespace` now writes a process-wide
    module global (it used to write a ContextVar that did not propagate into the
    sibling tasks pytest-asyncio and uvicorn spawn -- the same defect that broke
    the hub's chat path in production). that global takes precedence over the env
    var and does NOT reset itself between tests, so a test (or a NatsClient it
    connects) that sets it would otherwise shadow this env value for every later
    test. clear it before and after each test so isolation holds. tests that
    assert the unconfigured behavior clear the env var + call
    :func:`threetears.nats.testing.reset_default_namespace` locally within the test body.
    """
    from threetears.nats.testing import reset_default_namespace

    reset_default_namespace()
    monkeypatch.setenv("THREETEARS_NATS_SUBJECT_NAMESPACE", "aibots")
    yield
    reset_default_namespace()


def _package_of(nodeid: str) -> str:
    """the workspace package a test belongs to, from its rootdir-relative node id.

    ``packages/core/tests/...`` is ``packages/core``, ``packages/agent/memory/tests/...`` is
    ``packages/agent/memory``, and the repo-level ``tests/...`` is ``tests``.

    :param nodeid: the test's node id
    :ptype nodeid: str
    :return: the package's directory, relative to the repo root
    :rtype: str
    """
    parts = nodeid.split("::", 1)[0].split("/")
    if parts[0] == "packages" and len(parts) > 2:
        return "/".join(parts[:3] if parts[1] == "agent" else parts[:2])
    return parts[0]


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """put every test of one package on one xdist worker, in its usual order.

    under ``--dist loadgroup`` a package's tests run in one process exactly as they do serially, so
    a test meets no neighbour from its own package it would not meet in a serial run, and two tests
    of one package never hold a shared resource at once. packages share nothing at runtime, so
    splitting the suite at that boundary is the conservative cut. it is also faster than spreading
    single tests, because each worker imports fewer packages. without xdist the marker is inert.

    :param items: the collected tests
    :ptype items: list[pytest.Item]
    :return: nothing
    :rtype: None
    """
    for item in items:
        item.add_marker(pytest.mark.xdist_group(_package_of(item.nodeid)))
