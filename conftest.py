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

import os
import warnings
from collections.abc import Iterator

import pytest
from hypothesis import HealthCheck, settings

# No test reaches the Hugging Face Hub. ``huggingface_hub`` (loaded through ``voyageai`` and ``tokenizers``) fetches
# its agent-harness registry from huggingface.co the first time ``huggingface_hub.cli`` is imported, unless a copy
# under a day old is cached in HF_HOME. Nothing here imports that module, but freezegun's ``start()`` reads every
# attribute of every loaded module, and reading the package's lazy ``cli`` attribute imports it. So a test that froze
# the clock behind the offline guard (``packages/evals/tests/offline_support.py``) failed on a connect to huggingface.co
# whenever its process was the first to walk that package on a runner whose cache had aged past a day. The package
# reads this variable once, at its own import, so it is set here, before anything below can import it; it is set
# outright rather than defaulted, since no test downloads from the Hub.
os.environ["HF_HUB_OFFLINE"] = "1"

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

# Hypothesis bounds two things by wall clock: each example's run (``deadline``, 200 ms) and the time it takes to
# draw inputs (the ``too_slow`` health check). On the testament pool a test shares its runner with other batches, so
# both become bounds on the runner's load rather than the code: a property test failed "Input generation is slow"
# beside other work while its property held (#1278). Neither says anything about the behaviour under test, so the
# workspace's profile turns both off; every property and example a test states still runs.
# tests/test_hypothesis_profile.py holds every Hypothesis test to it.
settings.register_profile("workspace", deadline=None, suppress_health_check=[HealthCheck.too_slow])
settings.load_profile("workspace")

# ``tests.contain_tripwire`` fails a test whose code under test contained a programming error
# (see its docstring); ``tests.clock_isolation`` keeps a freeze from importing under the fake
# clock and fails a test that leaves the clock frozen; ``tests.require_services`` fails an integration
# test that skips at runtime under THREETEARS_REQUIRE_SERVICES=1 (the pool's integration suite);
# ``pytester`` lets their own tests run a session with them installed.
pytest_plugins = [
    "threetears.core.testing.fixtures",
    "tests.contain_tripwire",
    "tests.clock_isolation",
    "tests.require_services",
    "pytester",
]


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


#: the longest test id the suite accepts (the longest in it today is about 650)
_MAX_TEST_ID_LENGTH = 1024


@pytest.hookimpl(tryfirst=True)
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
    too_long = [item.nodeid[:120] for item in items if len(item.nodeid) > _MAX_TEST_ID_LENGTH]
    if too_long:
        # a test id is printed whole on every `-v` line, and a 130,000-character one (a bytes
        # parameter named after its contents) stalled GitHub's log reader: the CI step froze
        # mid-run with no output and was killed minutes later, on whichever branch hit it.
        raise pytest.UsageError(
            f"test ids longer than {_MAX_TEST_ID_LENGTH} characters; give the parametrize `ids=`:\n  "
            + "\n  ".join(too_long)
        )
    for item in items:
        item.add_marker(pytest.mark.xdist_group(_package_of(item.nodeid)))
