"""The package root stays cheap: ``import threetears.evals`` loads none of the engine, and the quick path's names load
on first use.

A host imports ``threetears.evals`` for ``PUBLIC_ROOTS`` or to reach a root (``from threetears.evals import ops``); the
quick path's ladder (``Eval``, ``Judge``, ``Model``, ...) is served by the root's module ``__getattr__`` from the
module defining each, so none of it is imported until a name is asked for. Each probe runs in a fresh interpreter,
since a module another test imported first would read as loaded.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

import threetears.evals

#: The quick path's names at the package root, the ladder a newcomer climbs.
LADDER = ("Answer", "Eval", "Example", "Judge", "Log", "Model", "Result", "World", "grader")

#: The names the root held before the ladder: the package's own metadata, not rungs of it.
PACKAGE_METADATA = ("PUBLIC_ROOTS", "ROOT_EXTRAS", "RootExtra", "__version__")

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
