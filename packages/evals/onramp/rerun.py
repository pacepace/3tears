"""Re-run an agent's eval file in this process and print what it read, as JSON: ``python rerun.py EVAL_FILE [--nothing]``.

Run by the docs build's own interpreter in a grading copy of the app, so it imports only what that build ships, and
only from the package's public root. The file is loaded as the eval door loads it (its directory first on
``sys.path``, its relative paths read beside it), and its first ``Eval`` is run into memory. With ``--nothing`` the
same eval is run again with every arm swapped for a runner that returns ``None``: how far the eval passes doing
nothing.

Prints ``{"refused": why}`` and exits 2 when the file does not load or the eval refuses to run.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from threetears.evals import Eval, Result


def nothing(case: dict[str, Any]) -> None:
    """Does nothing."""
    return None


def unpassed(result: Result) -> float:
    """The share of the control arm's cases with no passing repeat, as the first line counts a case passed."""
    states: dict[str, list[str]] = defaultdict(list)
    for trial in result.results(result.control):
        states[trial.case].append(trial.state)
    passed = sum(1 for repeats in states.values() if "failed" not in repeats and "passed" in repeats)
    return 1 - passed / len(states) if states else 1.0


def read(result: Result) -> dict[str, Any]:
    """The result's readings on the control arm: each measure's mean, and whether every arm's run finished."""
    means = {measure.name: measure.mean for measure in result.arms[result.control].measures}
    if "accuracy" in result.grader_readings and "match" in means:
        means.setdefault("accuracy", means["match"])
    return {
        "name": result.name,
        "control": result.control,
        "readings": list(result.grader_readings),
        "finished": all(summary.status == "completed" for summary in result.arms.values()),
        "means": {name: means.get(name) for name in result.grader_readings},
        "unpassed": unpassed(result),
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
        read_out = read(await ev.run(store=":memory:"))
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
