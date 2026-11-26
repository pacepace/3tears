"""The package root stays cheap: ``import threetears.evals`` loads none of the engine, and the quick path's names load
on first use.

A host imports ``threetears.evals`` for ``PUBLIC_ROOTS`` or to reach a root (``from threetears.evals import ops``); the
quick path's ladder (``Eval``, ``Judge``, ``Model``, ...) is served by the root's module ``__getattr__`` from the
module defining each, so none of it is imported until a name is asked for. The job lease's numbers, which a host's
config reads to size its database pool, load nothing beyond ``schema``. A host that imports
:class:`~threetears.evals.providers.ProviderError` only to catch it loads the failure's module and no engine: the
completion client's root serves each name from its defining module on first read; nor does it load ``httpx`` or
pydantic. A host that imports :class:`~threetears.evals.providers.Model` loads the client and the recorded replays it
takes, and no action or operation: the testing root, like the client's, serves each name from its defining module, so
the jobs kit's catalogue loads only for a host that asks for the kit. Each probe runs in a fresh
interpreter, since a module another test imported first would read as loaded.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Mapping
from types import MappingProxyType

import pytest

import threetears.evals
import threetears.evals.providers as providers

#: The quick path's names at the package root, the ladder a newcomer climbs.
LADDER = ("Answer", "Eval", "Example", "Judge", "Log", "Model", "Result", "World", "grader")

#: The names the root holds beside the ladder: the package's own metadata (its roots, extras, version and tiers), not
#: rungs of it.
PACKAGE_METADATA = ("PUBLIC_ROOTS", "ROOT_EXTRAS", "RootExtra", "__version__", "declared_names", "surface_tier")

_PROBE = """\
import sys
import threetears.evals
loaded = sorted(m for m in sys.modules if m.startswith("threetears.evals."))
print(",".join(loaded))
"""


def _loaded_after(probe: str) -> list[str]:
    done = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=120, check=False)
    assert done.returncode == 0, done.stderr
    return [name for name in done.stdout.strip().split(",") if name]


def test_importing_the_package_loads_none_of_the_engine() -> None:
    assert _loaded_after(_PROBE) == []


def test_a_root_reached_through_the_package_loads_no_quick_path() -> None:
    loaded = _loaded_after(_PROBE.replace("import threetears.evals\n", "from threetears.evals import schema\n"))

    assert not [name for name in loaded if name.startswith("threetears.evals.quick")]


def test_the_job_lease_numbers_load_no_engine_module_beyond_schema() -> None:
    """A host sizing its database pool reads the lease's numbers as its config loads, so they pull in no kernel."""
    loaded = _loaded_after(
        _PROBE.replace(
            "import threetears.evals\n",
            "from threetears.evals.schema import DEFAULT_JOB_LEASE_S, JOB_LEASE_RENEWAL_FRACTION, LEASE_POOL_WORKERS\n",
        )
    )

    assert "threetears.evals.schema.job_lease" in loaded, "the probe imported nothing, so this says nothing"
    assert [name for name in loaded if not name.startswith("threetears.evals.schema")] == []


#: Every engine module catching a provider's failure may load, each for a reason: ``providers`` is the failure
#: module's package, which Python runs first, and ``providers.failures`` defines :class:`ProviderError`. The package
#: root ``threetears.evals`` loads too, as every probe's does, and the probe leaves it out. Nothing else: a top-level
#: ``schema`` or ``negotiation`` import in ``failures.py`` costs ~0.3s, and the root's model ~1.5s.
CATCHING_A_FAILURE = ["threetears.evals.providers", "threetears.evals.providers.failures"]

#: A probe that loads ``failures.py`` from its source with one import planted after its ``__future__`` line, as if
#: the module had it at its top, then imports :class:`ProviderError` as a host does.
_PLANTED = """\
import importlib.util
import pathlib
import sys
import threetears.evals.providers as providers
path = pathlib.Path(providers.__file__).with_name("failures.py")
future = "from __future__ import annotations\\n"
source = path.read_text().replace(future, future + "import {planted}\\n", 1)
spec = importlib.util.spec_from_loader("threetears.evals.providers.failures", loader=None, origin=str(path))
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
exec(compile(source, str(path), "exec"), module.__dict__)
from threetears.evals.providers import ProviderError
loaded = sorted(m for m in sys.modules if m.startswith("threetears.evals."))
print(",".join(loaded))
"""


#: The third-party modules catching a provider's failure must not load, each for its reason. ``failures.py`` reads a
#: provider's error body as JSON and names no transport, so none is needed to catch what it defines.
THIRD_PARTY_A_FAILURE_LOADS_NONE_OF: Mapping[str, str] = MappingProxyType(
    {
        "httpx": "the client's transport (~0.13s); a host catching the failure brings its own",
        "pydantic": "the schema's models; the failure carries plain values, never a model",
        "pydantic_core": "pydantic's compiled core, which comes with it",
        "click": "httpx's command line, which importing httpx loads",
        "rich": "httpx's command line output, which importing httpx loads",
        "pygments": "httpx's command line highlighting, which importing httpx loads",
        "zstandard": "httpx's zstd decoder, which importing httpx loads",
    }
)

#: Reports which of :data:`THIRD_PARTY_A_FAILURE_LOADS_NONE_OF` are loaded after importing :class:`ProviderError`.
_THIRD_PARTY_PROBE = """\
import sys
{statement}
names = {names!r}
print(",".join(sorted(m for m in sys.modules if m.split(".")[0] in names)))
"""


def test_importing_the_client_root_loads_nothing_below_it() -> None:
    loaded = _loaded_after(_PROBE.replace("import threetears.evals\n", "import threetears.evals.providers\n"))

    assert loaded == ["threetears.evals.providers"]


@pytest.mark.parametrize(
    "statement",
    ["import threetears.evals.providers.failures\n", "from threetears.evals.providers import ProviderError\n"],
)
def test_catching_a_provider_failure_loads_its_module_and_nothing_else(statement: str) -> None:
    """A host imports ``ProviderError`` in its own client layer, so every process of its pays for what that loads."""
    loaded = _loaded_after(_PROBE.replace("import threetears.evals\n", statement))

    assert loaded == CATCHING_A_FAILURE


@pytest.mark.parametrize(
    "statement",
    ["import threetears.evals.providers.failures", "from threetears.evals.providers import ProviderError"],
)
def test_catching_a_provider_failure_loads_no_heavy_third_party_module(statement: str) -> None:
    """The engine's own modules are not the only cost: a transport or a model library at ``failures.py``'s top is."""
    loaded = _loaded_after(
        _THIRD_PARTY_PROBE.format(statement=statement, names=sorted(THIRD_PARTY_A_FAILURE_LOADS_NONE_OF))
    )

    assert loaded == []


def test_a_planted_third_party_import_in_the_failure_module_is_seen() -> None:
    """The check above turns red when ``failures.py`` imports the client's transport at its top."""
    probe = _PLANTED.format(planted="httpx").replace(
        'loaded = sorted(m for m in sys.modules if m.startswith("threetears.evals."))',
        f'loaded = sorted(m for m in sys.modules if m.split(".")[0] in {sorted(THIRD_PARTY_A_FAILURE_LOADS_NONE_OF)!r})',
    )

    assert "httpx" in _loaded_after(probe)


@pytest.mark.parametrize("planted", ["threetears.evals.schema", "threetears.evals.negotiation"])
def test_a_planted_engine_import_in_the_failure_module_is_seen(planted: str) -> None:
    """The check above turns red when ``failures.py`` imports what it used to at its top."""
    loaded = _loaded_after(_PLANTED.format(planted=planted))

    assert planted in loaded
    assert loaded != CATCHING_A_FAILURE


def test_the_client_root_serves_each_name_from_its_defining_module() -> None:
    """The lazy root holds every name of ``__all__``, as an eager one did, and refuses any other by name."""
    assert set(providers.__all__) <= set(dir(providers))
    for name in providers.__all__:
        assert getattr(providers, name) is not None
    assert providers.ProviderError.__module__ == "threetears.evals.providers.failures"
    with pytest.raises(AttributeError, match="has no attribute 'error_from_response'"):
        _ = providers.error_from_response  # type: ignore[attr-defined]


#: The engine packages building a client must not load: the action catalogue and the operations behind it, which the
#: testing root's jobs kit names, cost most of a second and a host making one model call uses neither.
NOT_LOADED_BY_A_MODEL = ("threetears.evals.actions", "threetears.evals.ops")

#: Every module below :data:`NOT_LOADED_BY_A_MODEL`.
NOT_LOADED_BY_A_MODEL_BELOW = tuple(f"{package}." for package in NOT_LOADED_BY_A_MODEL)


@pytest.mark.parametrize(
    "statement",
    ["from threetears.evals.providers import Model\n", "from threetears.evals import Model\nModel.__name__\n"],
)
def test_importing_the_client_loads_no_action_or_operation(statement: str) -> None:
    """A host that builds a model to call it pays for the client and what it records with, not for the job runner."""
    loaded = _loaded_after(_PROBE.replace("import threetears.evals\n", statement))

    assert "threetears.evals.providers.model" in loaded, "the probe imported no client, so this says nothing"
    assert [
        name for name in loaded if name in NOT_LOADED_BY_A_MODEL or name.startswith(NOT_LOADED_BY_A_MODEL_BELOW)
    ] == []


def test_asking_for_a_ladder_name_loads_the_quick_path() -> None:
    loaded = _loaded_after(_PROBE.replace("import threetears.evals\n", "from threetears.evals import Eval\n"))

    assert "threetears.evals.quick.contract" in loaded


@pytest.mark.parametrize("name", LADDER)
def test_each_ladder_name_is_served_from_the_module_defining_it(name: str) -> None:
    # The one Model is the completion client's (threetears.evals.providers); every other rung is the quick path's.
    home = "threetears.evals.providers." if name == "Model" else "threetears.evals.quick."
    assert getattr(threetears.evals, name).__module__.startswith(home)


def test_the_root_declares_the_ladder_and_no_more_than_ten_names_of_it() -> None:
    ladder = [name for name in threetears.evals.__all__ if name not in PACKAGE_METADATA]

    assert sorted(ladder) == sorted(LADDER)
    assert len(ladder) <= 10
    assert set(LADDER) <= set(dir(threetears.evals))


def test_a_name_the_root_does_not_serve_is_refused_as_any_module_refuses_it() -> None:
    with pytest.raises(AttributeError, match="has no attribute 'run_eval'"):
        _ = threetears.evals.run_eval  # type: ignore[attr-defined]
