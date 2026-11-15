"""Every reader of an arm's pooled counts, and every caller of a multi-run loader, asks the cells first.

An arm whose trials the cells keep apart (answered by more than one model, or read under more than one definition of
a host measure) has no one pass count, failure count or cost per reply: its ``ArmScoreboard`` counts pool what its
cells hold apart. Each miss so far (#1187 F-1, N-1, N-4; #1188 item 7) was a function reading those counts, or folding
a multi-run read, without asking. This test finds both from the source, not from a hand list:

- a function reading ``.passed``, ``.failed`` or ``.reply_cost_usd`` where it names a scoreboard (``.scoreboard``, a
  parameter typed ``ArmScoreboard``, or a call to a function of its module returning one);
- a function calling ``iter_result_readings`` or ``query_eval_results_by_runs``.

Each must ask the shared coordinate check (:data:`_CHECKS`) itself, or call a function of its module that asks it and
reads no count (a wrapper: ``floors._answer_split``, ``verdict._not_read``), or be private with every caller in its
module covered (a helper handed an arm already asked about); or be listed in :data:`_LISTED` with a one-line reason.
A listed entry the scan no longer finds, or that now asks the check, fails too, so the list never outlives its cause.

**Mutations these catch:** a new reader of an arm's counts that never asks whether the arm is split; a check deleted
from a reader that had it; a reason left behind after its reader changed.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Final

_SOURCE: Final = Path(__file__).resolve().parents[1] / "src" / "threetears" / "evals"

#: The shared coordinate check, in :mod:`threetears.evals.analysis.cells`.
_CHECKS: Final = frozenset({"unread_arms", "split_arms", "split_coordinates"})
#: An arm's pooled counts on its scoreboard.
_COUNTS: Final = frozenset({"passed", "failed", "reply_cost_usd"})
#: The loaders that read many runs' results at once.
_LOADERS: Final = frozenset({"iter_result_readings", "query_eval_results_by_runs"})

#: The readers that do not ask, each with why that is right, or the issue that will make it ask.
_LISTED: Final[dict[str, str]] = {
    "analysis/bundle/assemble.py::assemble_context_bundle": "builds the cells, where a split is decided; reads no count",
    "analysis/campaign_readings.py::_cost_on": "takes exactly one cell's cost error, so an arm of two cells has none",
    "analysis/costs.py::cost_source": "reads the report input's rows, which withhold a split arm's cost",
    "analysis/reads.py::_readings_of": "the read pivot, frontier, history and comparison sets fold; each asks the "
    "check, save frontier dominance and the cheapest pick, which say they pool definitions: issue draft 6",
    "analysis/report/aspect/common.py::_cost_ruler": "chooses the scale every cost prints on; prints no arm's figure",
    "analysis/report/aspect/trials.py::_census_line": "the census of every trial the aspect ran; no arm's figure",
    "analysis/report/campaign/provenance.py::priced_aspects": "whether any arm on an aspect was priced; no figure",
    "analysis/report/campaign/scorecard.py::_choice_matrix": "prints a split arm's pooled rate: issue draft 8 (S5a-2)",
    "analysis/report/keys/reads.py::_passing": "reads the report input's rows, which withhold a split arm's passes",
    "analysis/report/keys/reads.py::_counts_passes": "reads the report input's rows, which withhold a split arm's passes",
    "ops/lenses.py::detectable_effects": "pools variance and says so where definitions differ: issue draft 7",
    "ops/regressions.py::_judged_results": "judge agreement for a history step; history marks a split as a boundary",
}


def _functions(tree: ast.Module) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    """A module's functions and its classes' methods, by name; a nested function is read as part of its parent."""
    found: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    for node in tree.body:
        held = node.body if isinstance(node, ast.ClassDef) else [node]
        for each in held:
            if isinstance(each, (ast.FunctionDef, ast.AsyncFunctionDef)):
                found[each.name] = each
    return found


def _attributes(function: ast.AST) -> set[str]:
    return {node.attr for node in ast.walk(function) if isinstance(node, ast.Attribute)}


def _named(function: ast.AST) -> set[str]:
    return {node.id for node in ast.walk(function) if isinstance(node, ast.Name)}


def _called(function: ast.AST) -> set[str]:
    return {
        node.func.id for node in ast.walk(function) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }


def _unasked() -> set[str]:
    """Every reader the scan finds that does not ask the check, as ``path::name`` under the package's source."""
    found: set[str] = set()
    for path in sorted(_SOURCE.rglob("*.py")):
        functions = _functions(ast.parse(path.read_text()))
        getters = {
            name
            for name, function in functions.items()
            if function.returns is not None and "ArmScoreboard" in ast.unparse(function.returns)
        }
        calls = {name: _called(function) & set(functions) for name, function in functions.items()}
        direct = {name for name, function in functions.items() if _named(function) & _CHECKS}
        wrappers = {name for name in direct if not _attributes(functions[name]) & _COUNTS}
        covered = direct | {name for name in functions if calls[name] & wrappers}
        while more := {
            name
            for name in set(functions) - covered
            if name.startswith("_")
            and (callers := {caller for caller, called in calls.items() if name in called and caller != name})
            and callers <= covered
        }:
            covered |= more
        for name, function in functions.items():
            attributes = _attributes(function)
            board = bool(attributes & _COUNTS) and (
                "scoreboard" in attributes
                or "ArmScoreboard" in ast.unparse(function.args)
                or bool(calls[name] & getters)
            )
            if (board or attributes & _LOADERS) and name not in covered:
                found.add(f"{path.relative_to(_SOURCE)}::{name}")
    return found


def test_every_reader_of_an_arms_counts_or_of_many_runs_asks_the_cells_or_says_why_not() -> None:
    unasked = _unasked()

    missing = sorted(unasked - set(_LISTED))
    assert not missing, (
        "these read an arm's pooled counts, or many runs' results, and never ask whether the arm is split "
        f"(unread_arms, split_arms or split_coordinates); ask, or list each with why: {missing}"
    )


def test_every_listed_reader_is_still_one_that_does_not_ask() -> None:
    stale = sorted(set(_LISTED) - _unasked())

    assert not stale, f"no longer found, or now asks the check: drop them from the list: {stale}"


def test_every_listed_reason_is_one_line() -> None:
    assert all(reason.strip() and "\n" not in reason for reason in _LISTED.values())
