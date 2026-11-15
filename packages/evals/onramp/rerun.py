"""Re-run an agent's eval file in this process and print what it read, as JSON: ``python rerun.py EVAL_FILE [--nothing]``.

Run by the docs build's own interpreter in a grading copy of the app, so it imports only what that build ships, and
only from the package's public root. The file is loaded as the eval door loads it (its directory first on
``sys.path``, its relative paths read beside it), and its first ``Eval`` is run into memory. With ``--nothing`` the
same eval is run again with every arm swapped for a runner that returns ``None``: the share of cases it then fails.

Prints ``{"refused": why}`` and exits 2 when the file does not load or the eval refuses to run.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from threetears.evals import Eval, Result


def nothing(case: dict[str, Any]) -> None:
    """Does nothing."""
    return None


def unpassed(result: Result) -> float:
    """The share of the control arm's cases that some repeat failed: lost, or graded false or zero by a grader.

    Read off each trial's grades rather than its state, since a trial's state counts only checks, and a function
    grader that returns False is a measure, not a check: the first line counts its case passed.
    """
    readings = ["match" if name == "accuracy" else name for name in result.grader_readings]
    failed: dict[str, bool] = defaultdict(bool)
    for trial in result.results(result.control):
        failed[trial.case] |= trial.state == "lost" or not all(trial.scores.get(name) for name in readings)
    return sum(failed.values()) / len(failed) if failed else 1.0


def directions(ev: Eval, module: Any) -> dict[str, bool | None]:  # Any: the eval file's module
    """Which way is better on each grader's measure, as the build declares it: a measure declared with ``@measure``
    states its own (``higher_is_better``); a guardrail states its own direction; every other grader is a quality score,
    higher is better (``scorer_measure`` in ``quick/one_call.py``, in every build this harness measures). A measure
    whose direction the build does not state here is ``None``, and counts for neither way."""
    declared: dict[str, bool | None] = {}
    for value in vars(module).values():
        descriptor = getattr(value, "descriptor", None)
        if isinstance(getattr(descriptor, "higher_is_better", None), bool):
            declared[descriptor.name] = descriptor.higher_is_better
    for name, guardrail in getattr(getattr(ev, "plan", None), "guardrails", {}).items():
        declared[name] = getattr(guardrail, "higher_is_better", None)
    return declared


def read(result: Result, declared: Mapping[str, bool | None]) -> dict[str, Any]:
    """The result's readings on the control arm: each measure's mean and which way is better on it (``declared``, else
    higher), and whether every arm's run finished."""
    means = {measure.name: measure.mean for measure in result.arms[result.control].measures}
    if "accuracy" in result.grader_readings and "match" in means:
        means.setdefault("accuracy", means["match"])
    return {
        "name": result.name,
        "control": result.control,
        "readings": list(result.grader_readings),
        "finished": all(summary.status == "completed" for summary in result.arms.values()),
        # Results a grader scored: not failed by the candidate (its runner raised) and not excluded.
        "scored": sum(1 for trial in result.results(result.control) if getattr(trial, "outcome", None) == "scored"),
        "means": {name: means.get(name) for name in result.grader_readings},
        "higher_is_better": {name: declared.get(name, True) for name in result.grader_readings},
    }


async def main(path: Path, *, nothing_too: bool) -> dict[str, Any]:
    os.chdir(path.parent)
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(f"onramp_rerun_{path.stem}", path)
    if spec is None or spec.loader is None:
        return {"refused": f"{path.name} cannot be imported"}
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        evals = [value for value in vars(module).values() if isinstance(value, Eval)]
        if not evals:
            return {"refused": f"{path.name} defines no Eval at module level"}
        ev = evals[0]
        read_out = read(await ev.run(store=":memory:"), directions(ev, module))
        if nothing_too:
            ev.arms = {arm: nothing for arm in ev.arms}
            read_out["nothing_unpassed"] = unpassed(await ev.run(store=":memory:"))
    except ValueError as refusal:
        return {"refused": str(refusal)}
    return read_out


if __name__ == "__main__":
    out = asyncio.run(main(Path(sys.argv[1]).resolve(), nothing_too="--nothing" in sys.argv[2:]))
    print(json.dumps(out))
    sys.exit(2 if "refused" in out else 0)
