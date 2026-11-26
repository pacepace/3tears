"""The package root stays cheap: ``import threetears.evals`` loads none of the engine, and the quick path's names load
on first use.

A host imports ``threetears.evals`` for ``PUBLIC_ROOTS`` or to reach a root (``from threetears.evals import ops``); the
quick path's ladder (``Eval``, ``Judge``, ``Model``, ...) is served by the root's module ``__getattr__`` from the
module defining each, so none of it is imported until a name is asked for. The job lease's numbers, which a host's
config reads to size its database pool, load nothing beyond ``schema``. A host that imports
:class:`~threetears.evals.providers.ProviderError` only to catch it loads the failure's module and no engine: the
completion client's root serves each name from its defining module on first read. Each probe runs in a fresh
interpreter, since a module another test imported first would read as loaded.
"""

from __future__ import annotations

import subprocess
import sys

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


#: The engine packages a host catching a provider's failure must not pay for: each pulls in the analysis and report
#: stack.
ENGINE_PACKAGES = ("threetears.evals.actions", "threetears.evals.ops", "threetears.evals.testing")


def _engine_loaded(loaded: list[str]) -> list[str]:
    return [
        name for name in loaded if name in ENGINE_PACKAGES or name.startswith(tuple(f"{p}." for p in ENGINE_PACKAGES))
    ]


@pytest.mark.parametrize(
    "statement",
    [
        "import threetears.evals.providers\n",
        "import threetears.evals.providers.failures\n",
        "from threetears.evals.providers import ProviderError\n",
    ],
)
def test_catching_a_provider_failure_loads_no_engine(statement: str) -> None:
    """A host imports ``ProviderError`` in its own client layer, so every process of its pays for what that loads."""
    loaded = _loaded_after(_PROBE.replace("import threetears.evals\n", statement))

    assert "threetears.evals.providers" in loaded, "the probe imported nothing, so this says nothing"
    assert _engine_loaded(loaded) == []
    assert "threetears.evals.providers.model" not in loaded


@pytest.mark.parametrize("package", ENGINE_PACKAGES)
def test_a_planted_engine_import_is_seen(package: str) -> None:
    """The check above reads each engine package it names as loaded when a probe does load it."""
    loaded = _loaded_after(
        _PROBE.replace("import threetears.evals\n", f"import threetears.evals.providers\nimport {package}\n")
    )

    assert package in _engine_loaded(loaded)


def test_the_client_root_serves_each_name_from_its_defining_module() -> None:
    """The lazy root holds every name of ``__all__``, as an eager one did, and refuses any other by name."""
    assert set(providers.__all__) <= set(dir(providers))
    for name in providers.__all__:
        assert getattr(providers, name) is not None
    assert providers.ProviderError.__module__ == "threetears.evals.providers.failures"
    with pytest.raises(AttributeError, match="has no attribute 'error_from_response'"):
        _ = providers.error_from_response  # type: ignore[attr-defined]


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
