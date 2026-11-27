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
import sys
import warnings
from collections.abc import Iterator
from datetime import UTC, datetime

import pytest

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

# ``tests.contain_tripwire`` fails a test whose code under test contained a programming error
# (see its docstring); ``tests.clock_isolation`` keeps a freeze from importing under the fake
# clock and fails a test that leaves the clock frozen; ``tests.require_services`` fails an integration
# test that skips at runtime under THREETEARS_REQUIRE_SERVICES=1 (the pool's integration suite);
# ``tests.timing_suite`` fails a test that times against the host's quiet without the ``loop_timing``
# marker that runs it apart; ``threetears.core.testing.loop_stall`` fails a test whose event loop ran one turn past its
# limit, naming what held it; ``pytester`` lets their own tests run a session with them installed.
pytest_plugins = [
    "threetears.core.testing.fixtures",
    "threetears.core.testing.loop_stall",
    "tests.contain_tripwire",
    "tests.clock_isolation",
    "tests.require_services",
    "tests.timing_suite",
    "pytester",
]

# What the session's start pays once (pytest_collection_finish, below), each imported only where installed: the
# workspace installs them all, a narrower environment skips what it lacks
import freezegun  # noqa: E402

from tests.clock_isolation import resolve_lazy_names  # noqa: E402

try:
    import duckdb
except ImportError:  # NOSILENT: an extra this environment lacks; its first connect is then no test's to pay
    duckdb = None  # type: ignore[assignment]
try:
    import pyarrow
    import pyarrow.compute
except ImportError:  # NOSILENT: an extra this environment lacks; its first kernel call is then no test's to pay
    pyarrow = None  # type: ignore[assignment]
try:
    import PIL.Image
except ImportError:  # NOSILENT: Pillow absent; its plugin registry is then no test's to fill
    PIL = None  # type: ignore[assignment]


#: the threetears packages whose lazy attributes are already resolved in this process
_RESOLVED_LAZY: set[str] = set()


def _resolve_lazy_attributes() -> None:
    """resolve every lazy attribute of each threetears package loaded so far, once a package.

    A package whose ``__init__`` serves its names lazily (a module ``__getattr__``, with ``__dir__`` listing them)
    imports a submodule on the first read of a name. ``freezegun.freeze_time`` reads every attribute of every
    loaded module when it starts, after it has put ``FakeDatetime`` in place of ``datetime.datetime``: a lazy name
    read then imports its module under the freeze, and a pydantic model there with a ``datetime`` field is built on
    ``FakeDatetime`` and fails (``threetears.core.collections.epoch_change``). The failed start leaves the freeze
    half applied, so the tests after it in the same process see a fake clock. Which test met it first depended on
    whether anything earlier in the process had already imported the package. Resolving the names before each test,
    as the scan would, takes the import out of the freeze.
    """
    for name, module in list(sys.modules.items()):
        if name in _RESOLVED_LAZY or not name.startswith("threetears") or module is None:
            continue
        _RESOLVED_LAZY.add(name)
        if "__getattr__" not in vars(module) or "__dir__" not in vars(module):
            continue
        for attribute in dir(module):
            try:
                getattr(module, attribute)
            except ImportError, AttributeError:
                # an optional extra not installed: freezegun's own scan skips the name the same way
                continue


@pytest.fixture(autouse=True)
def _lazy_attributes_resolved_outside_any_freeze() -> None:
    """every loaded threetears package's lazy names resolved before the test (:func:`_resolve_lazy_attributes`)."""
    _resolve_lazy_attributes()


def pytest_collection_finish(session: pytest.Session) -> None:
    """pay the process's one-time costs before its first test, so no test's loop turn or time bound carries them.

    A cost paid once a process lands on whichever test reaches it first: in its event loop's turn (the loop-stall guard,
    ``threetears.core.testing.loop_stall``, fails a turn past 250 ms of CPU), inside a time bound, or not at all when an
    earlier test happened to pay it. Which test that is depends on the batch a runner was handed, so the same test was
    red on one runner and green on another. Paid here, after every test module is imported and before any test runs,
    in order:

    - the loaded packages' lazy names resolved (:func:`_resolve_lazy_attributes`, ``resolve_lazy_names``), so the
      freeze below imports nothing under its fake clock (``tests/clock_isolation.py``);
    - one ``freeze_time`` start and stop: freezegun's first start reads every attribute of every loaded module and
      caches what it found a module, and a later start reads only modules new since. The first start measured ~115 ms on
      a laptop and stalled a pool runner's loop 1.9 s (2026-11-26); a later one, ~14 ms;
    - ``PIL.Image.init()``: Pillow imports every image plugin on the first open or save that needs one, ~29 ms
      (2026-11-26; 17 ms on a laptop's second run, 575 ms cold off disk);
    - one in-memory DuckDB connection: its first open, ~5 ms, ~1 ms over a later one (2026-11-26, a laptop);
    - one pyarrow compute call: its kernel registry, 0.1-0.4 ms the first call, nothing after (2026-11-26, a laptop).

    Nothing is paid on a collect-only run, which runs no test.

    :param session: the session, its tests collected
    :ptype session: pytest.Session
    :return: nothing
    :rtype: None
    """
    if session.config.option.collectonly:
        return
    _resolve_lazy_attributes()
    resolve_lazy_names()
    # a datetime, not a string: freezegun parses a string by reading the clock, which testament counts against every
    # test of the process as a clock read in a hook
    freezer = freezegun.freeze_time(datetime(2026, 1, 1, tzinfo=UTC))
    freezer.start()
    freezer.stop()
    if PIL is not None:
        PIL.Image.init()
    if duckdb is not None:
        duckdb.connect(":memory:").close()
    if pyarrow is not None:
        pyarrow.compute.sum(pyarrow.array([1]))


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
