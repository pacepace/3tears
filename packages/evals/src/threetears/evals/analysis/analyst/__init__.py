"""The analyst's computations: what each analyst tool reads off a frozen report input.

A campaign report is written by an analyst that investigates the campaign through tools (``docs/eval-report-kit/
ANALYST.md``). Each tool is three layers: an action in :mod:`threetears.evals.actions`, an operation in
:mod:`threetears.evals.ops`, and its computation here. Every tool reads one frozen report input, the snapshot's
(:mod:`~threetears.evals.analysis.analyst.snapshot`), and returns every figure as the figure key that names it
(:mod:`~threetears.evals.analysis.analyst.cited`), so the prose an analyst writes cites what a tool returned and the
renderer prints the same number.

**No branch on the kind of eval.** Nothing here reads what kind of eval a campaign is, its design's shape, or the
planner's signals: a tool's conditions are on data (a control exists, tasks exist, an id repeats), so an eval nobody
anticipated is read with the same tools. ``tests/test_analyst_contract.py`` walks these modules to hold that.

The submodules are imported directly; this package re-exports nothing.
"""

from __future__ import annotations

__all__: list[str] = []
