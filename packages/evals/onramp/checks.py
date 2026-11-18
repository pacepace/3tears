"""Success, checked by code: what the agent wrote is re-run against the app, its fixed twin and a do-nothing runner.

:func:`regrade` does the work once per trial, inside the harness's runner, and its facts are stored with the trial;
the three graders read them. The work runs there rather than in the graders because a function grader is called on
the event loop (``quick/one_call.py``), where three re-runs of an eval would stall every other trial.

- :func:`eval_runs`: the agent's eval file, run by the eval door in a fresh copy of the app, exits 0 or 4 and stores a
  run, that run finished, and a grader scored at least one of its results.
- :func:`measures_the_app`: on the app's fixed twin its reading moves the better way on at least one measure, the way
  the package states is better for it (``higher_is_better``), and a runner that does nothing passes at most half its
  cases: the eval beats doing nothing.
- :func:`read_right`: the agent's report matches the run: the first line's lead sentence (the verdict and its counts),
  given as the run printed it; the pass rate within half a point of the measure it names (of the eval's one measure,
  when the name is none the run printed), compared exactly; and ``finished`` true exactly when the run finished.

Every re-run is a command in the trial's container (:class:`~.isolation.Container`), as the agent's own were: the
regrade runs what the agent wrote, so it runs it where the agent ran it, with no network, and it makes no model call.
The whole regrade is held to :data:`REGRADE_BUDGET_S`.

The graders return ``bool`` and say so in their annotation, which makes each a check: a trial one fails is failed, and
the package's first line counts it (``scorer_measure`` in ``quick/one_call.py``).
"""

from __future__ import annotations

import asyncio
import json
import re
import shlex
import time
from collections.abc import Mapping, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any

from .sandbox import DOOR_EXITS_OK, Observation, Sandbox, is_door

#: The most the whole regrade of one trial may take, in seconds: with the agent's wall cap it fits the cell budget. The
#: baseline's regrades took 9 to 15 s each (run 20261115T224108Z, regraded again 2026-11-16).
REGRADE_BUDGET_S = 90.0
#: How far the reported pass rate may be from the stored one: half a point, inclusive, compared in exact decimals (a
#: rate of 0.72 against a mean of 0.725 is within it, though the floats' difference is a hair above 0.005).
PASS_RATE_TOLERANCE = Decimal("0.005")


def _eval_file(report: Mapping[str, Any] | None, commands: Sequence[Mapping[str, Any]]) -> str | None:
    """The eval file to re-run: the one the report names, else the last the agent ran through the door."""
    if report is not None and isinstance(report.get("eval_file"), str) and report["eval_file"].strip():
        return str(report["eval_file"]).strip()
    doors = [shlex.split(c["command"]) for c in commands if c["door"]]
    targets = [argv[3] for argv in doors if len(argv) > 3 and argv[3].endswith((".py", ".toml"))]
    return targets[-1] if targets else None


def _reading(observation: Observation) -> dict[str, Any]:  # Any: rerun.py's JSON
    """What ``rerun.py`` printed, or why it printed nothing readable."""
    lines = observation.text.splitlines()
    payload = next((line for line in reversed(lines) if line.startswith("{")), None)
    if payload is None:
        return {"refused": observation.first_error or "rerun.py printed no reading"}
    try:
        reading: dict[str, Any] = json.loads(payload)
    except json.JSONDecodeError as unreadable:
        return {"refused": f"rerun.py printed no reading: {unreadable.msg}"}
    return reading


async def regrade(
    sandbox: Sandbox, trial: Path, report: Mapping[str, Any] | None, commands: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:  # Any: the facts the graders read
    """Re-run the agent's eval three ways, in copies under ``trial``; the facts every grader reads.

    A regrade past :data:`REGRADE_BUDGET_S` is cut off, and says so as its problem: every grader then fails it.
    """
    eval_file = _eval_file(report, commands)
    if eval_file is None:
        return {"eval_file": None, "problem": "the agent named no eval file and ran none through the door"}
    if not (sandbox.root / eval_file).is_file():
        return {"eval_file": eval_file, "problem": f"{eval_file} is not in the app's directory"}
    try:
        async with asyncio.timeout(REGRADE_BUDGET_S):
            return await _regrade(sandbox, trial, eval_file, commands, time.monotonic() + REGRADE_BUDGET_S)
    except TimeoutError:
        return {"eval_file": eval_file, "problem": f"the regrade ran past its {REGRADE_BUDGET_S:.0f}s budget"}


async def _regrade(
    sandbox: Sandbox, trial: Path, eval_file: str, commands: Sequence[Mapping[str, Any]], deadline: float
) -> dict[str, Any]:  # Any: the facts the graders read
    python, rerun = sandbox.runner.python, sandbox.runner.rerun

    async def execute(argv: list[str], cwd: Path, *, door: bool = False) -> Observation:
        # Only the regrade's own reading (rerun.py) is handed rerun.py: a replay of the agent's command never sees it.
        left = max(1.0, deadline - time.monotonic())
        return await sandbox.execute(argv, cwd=cwd, door=door, timeout_s=left, grading=rerun in argv)

    left = sandbox.made()
    shipped = sandbox.grading_copy(trial / "regrade", twin=False, overlay=left)
    door = await execute([python, "-m", "threetears.evals", eval_file], shipped, door=True)
    printed = door.text.splitlines()[1:2]
    buggy = _reading(await execute([python, rerun, eval_file, "--nothing"], shipped))
    # The fixed twin, with what the agent wrote laid over it and every command it ran that made something run again,
    # so an output the app wrote (a log an eval reads) is the fixed app's.
    fixed_copy = sandbox.grading_copy(trial / "regrade-twin", twin=True, overlay=sandbox.written)
    for command in commands:
        argv = shlex.split(command["command"])
        if command["exit_code"] == 0 and argv[0] in ("python", "python3") and not is_door(argv):
            await execute([python, *argv[1:]], fixed_copy)
    fixed = _reading(await execute([python, rerun, eval_file], fixed_copy))
    return {
        "eval_file": eval_file,
        "door_exit": door.exit_code,
        "door_first_line": printed[0] if printed else "",
        "stored": (shipped / ".evals" / "evals.sqlite").is_file(),
        "app": buggy,
        "twin": fixed,
        "problem": "",
    }


def eval_runs(case: dict[str, Any], out: dict[str, Any]) -> bool:
    """The agent's eval, re-run by the eval door in a fresh copy of the app, exits 0 or 4 and stores a finished run in
    which a grader scored at least one result.

    An eval whose runner fails every case (it raised on each) finishes and stores a run, and measures nothing: it does
    not run as an eval. One that scores any case does, whatever the scores are.
    """
    checks = out["checks"]
    return (
        not checks["problem"]
        and checks["door_exit"] in DOOR_EXITS_OK
        and checks["stored"]
        and checks["app"].get("finished") is True
        and (checks["app"].get("scored") or 0) > 0
    )


def measures_the_app(case: dict[str, Any], out: dict[str, Any]) -> bool:
    """On the app's fixed twin the eval reads better on some measure, and doing nothing passes at most half the cases.

    Better is the way the package states for the measure (``higher_is_better``): up for a score, down for a count of
    misses. A measure with no stated direction counts for neither.
    """
    checks = out["checks"]
    if checks["problem"] or "refused" in checks["app"] or "refused" in checks["twin"]:
        return False
    app, twin = checks["app"]["means"], checks["twin"]["means"]
    higher = checks["app"].get("higher_is_better", {})
    moved = any(_better(higher.get(name), app.get(name), twin.get(name)) for name in app)
    return moved and checks["app"].get("nothing_unpassed", 0.0) >= 0.5


def _better(higher_is_better: bool | None, app: float | None, twin: float | None) -> bool:
    """Whether ``twin`` reads better than ``app`` on a measure whose better way is ``higher_is_better``."""
    if higher_is_better is None or app is None or twin is None:
        return False
    return twin > app if higher_is_better else twin < app


def _plain(name: str) -> str:
    """A measure's name as compared: ``Slug score``, ``slug`` and ``matches_expected`` as ``slug`` and ``matches
    expected``, whichever way the run printed it or the eval declared it."""
    words = re.sub(r"[^a-z0-9]+", " ", name.lower()).split()
    return " ".join(words[:-1] if len(words) > 1 and words[-1] == "score" else words)


def reported_measures(report: Mapping[str, Any], app: Mapping[str, Any]) -> list[float]:  # Any: JSON
    """The mean the report's ``pass_rate`` must be: the measure it names, when the run has one of that name (named as
    the console printed it or as the eval declared it); else the eval's one measure, when it has exactly one. An eval
    of several measures whose name matches none has no mean to read against, so nothing passes: a value that happens to
    equal another measure is not a reading of it."""
    means = {name: mean for name, mean in app["means"].items() if mean is not None}
    named = _plain(str(report.get("measure") or ""))
    chosen = [mean for name, mean in means.items() if named and _plain(name) == named]
    if chosen:
        return chosen
    return list(means.values()) if len(app["readings"]) == 1 else []


def within_tolerance(rate: float, mean: float) -> bool:
    """Whether a reported ``rate`` is within :data:`PASS_RATE_TOLERANCE` of ``mean``, each read as the decimal its
    shortest repr spells, so the bound is the one the docs state and not a float's rounding of it."""
    return abs(Decimal(repr(rate)) - Decimal(repr(mean))) <= PASS_RATE_TOLERANCE


#: Where a first line's lead sentence ends: a full stop followed by a space, or the line's end.
_SENTENCE_END = re.compile(r"\.(?=\s|$)")


def lead_sentence(line: str) -> str:
    """A first line's lead sentence: the verdict and its counts, through the first full stop that ends a sentence (the
    whole line when none does). What follows it (``Cases that failed a trial: ...``) names cases, and is printed only
    when an eval fails some; an agent that gives the lead and leaves the rest has read the line right."""
    stripped = line.strip()
    end = _SENTENCE_END.search(stripped)
    return stripped if end is None else stripped[: end.end()]


def first_line_read(reported: str, printed: str) -> bool:
    """Whether the report's first line is the run's: the line as printed, or a part of it from its start that holds at
    least its whole lead sentence (:func:`lead_sentence`)."""
    given, line = reported.strip(), printed.strip()
    return bool(line) and line.startswith(given) and given.startswith(lead_sentence(line))


def read_right(case: dict[str, Any], out: dict[str, Any]) -> bool:
    """The agent's report matches the run: its first line's lead sentence as printed (:func:`first_line_read`), its pass
    rate within half a point of the measure it names (or of the eval's one measure, when the name is none the run
    printed), its ending.

    A competent reading names the measure it reports, as the console printed it or as the eval declared it, and gives
    its value; of an eval with one measure, the name may be left out. A pass rate that is not the named measure's (a
    share read off the case count when a grader says otherwise, another measure's value, a figure made up) fails, and
    so does an unnamed one when the eval has several measures: a coincidence with one of them is not a reading.
    """
    report, checks = out["report"], out["checks"]
    if report is None or checks["problem"] or "refused" in checks["app"]:
        return False
    app = checks["app"]
    rate = report.get("pass_rate")
    return (
        first_line_read(str(report.get("first_line", "")), checks["door_first_line"])
        and isinstance(rate, int | float)
        and not isinstance(rate, bool)
        and any(within_tolerance(rate, mean) for mean in reported_measures(report, app))
        and report.get("finished") is app["finished"]
    )


GRADERS = [eval_runs, measures_the_app, read_right]
