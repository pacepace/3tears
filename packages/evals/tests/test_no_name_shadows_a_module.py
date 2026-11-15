"""No public name shares its spelling with a module of the package that exports it, or exported it.

When ``threetears.evals.run`` exports a function ``definition_drift`` and also holds a module ``run/definition_drift.py``,
one dotted spelling names two things: ``from threetears.evals.run import definition_drift`` binds whichever the import
system finds first, a ``mock.patch`` target is ambiguous, and the re-root's rewrite by dotted prefix would append a
segment on every run, so it would never converge (design-reroot-cutover.md §1.5).

This walks every public root's ``__all__``, and both columns of the re-root map (``api/reroot-map.txt``: the old
spellings, and the planned new ones), against the modules of the package each spelling sits in.

Six names shadow a module today. They are a ratchet: the cut-over (chunk X3) renames the six modules and empties
:data:`SHADOWING_TODAY`, and no seventh may appear.
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

import threetears.evals

#: The public names that shadow a module of their root today, each renamed by the cut-over (``module`` lines of the
#: re-root map). Shrink only.
SHADOWING_TODAY: frozenset[str] = frozenset(
    {
        "threetears.evals.run.regrade",
        "threetears.evals.run.definition_drift",
        "threetears.evals.run.judge_kind",
        "threetears.evals.analysis.campaign_readings",
        "threetears.evals.analysis.judge_drift",
        "threetears.evals.analysis.judge_kind_readings",
    }
)

MAP_PATH = Path(threetears.evals.__file__).resolve().parents[3] / "api" / "reroot-map.txt"


def _submodules(package: str) -> frozenset[str]:
    """The modules directly inside ``package``, or nothing when it is no package (or does not exist yet)."""
    try:
        module = importlib.import_module(package)
    except ModuleNotFoundError:
        return frozenset()
    path = getattr(module, "__path__", None)
    return frozenset(info.name for info in pkgutil.iter_modules(path)) if path is not None else frozenset()


def _shadowing(spellings: set[str]) -> set[str]:
    return {spelling for spelling in spellings if spelling.rsplit(".", 1)[1] in _submodules(spelling.rsplit(".", 1)[0])}


def _map_columns() -> tuple[set[str], set[str]]:
    """The map's old and new public spellings: every name record (``module`` and ``root`` lines name modules)."""
    olds: set[str] = set()
    news: set[str] = set()
    for row in MAP_PATH.read_text(encoding="utf-8").splitlines():
        if row.startswith("#"):
            continue
        _epoch, record, old, new, _note = row.split("\t")
        if record in ("module", "root"):
            continue
        olds.add(old)
        if record in ("name", "renamed"):
            news.add(new)
    return olds, news


def test_no_public_name_of_today_shadows_a_module_beyond_the_six() -> None:
    exported = set()
    for root in ("threetears.evals", *threetears.evals.PUBLIC_ROOTS):
        module = importlib.import_module(root)
        exported |= {f"{root}.{name}" for name in getattr(module, "__all__", ())}
    olds, _news = _map_columns()
    found = _shadowing(exported | olds)
    assert not found - SHADOWING_TODAY, f"a public name shadows a module of its root: {sorted(found - SHADOWING_TODAY)}"
    assert not SHADOWING_TODAY - found, (
        f"no longer shadowing, so remove from SHADOWING_TODAY (shrink only): {sorted(SHADOWING_TODAY - found)}"
    )


def test_no_planned_spelling_shadows_a_module() -> None:
    _olds, news = _map_columns()
    assert news, f"{MAP_PATH.name} plans no new spelling"
    assert not _shadowing(news), f"a planned spelling shadows a module of its root: {sorted(_shadowing(news))}"
