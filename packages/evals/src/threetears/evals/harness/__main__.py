"""``python -m threetears.evals.harness FILE --write-report MODEL [--budget-usd N] [--verbose]``.

The file's evals are run, printed and reported as ``python -m threetears.evals FILE`` runs them
(:func:`~threetears.evals.quick.files.run_file`, which finds them through its ``Runnable`` protocol), then the report
writer analyses each one (:func:`~threetears.evals.harness.analyse.analyse`), its recordings beside the file
(``recordings/report/``). Each run files a new campaign, whose snapshot no earlier recording was made over, so this
door writes live, on ``ANTHROPIC_API_KEY``. A writer's report replays offline only over the snapshot its recording
was made on: a :class:`~threetears.evals.harness.WriterEval` over frozen case files does. The line after each eval's
report names the analysis page and the recording it was written from, or says why there is none.

Exit codes are the file door's, each eval's analysis folded in: 1 when the writer stored nothing, 2 when the
analysis was refused (no recording and no key, a campaign the freeze refuses, a result that files no campaign).
"""

from __future__ import annotations

import asyncio
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from threetears.evals.actions.report_writing import ReportNotStored
from threetears.evals.harness.analyse import DEFAULT_BUDGET_USD, Filed, analyse, analysis_page, recording_path
from threetears.evals.kernel.errors import EvalServiceError
from threetears.evals.kernel.eval_file import loading_eval_file
from threetears.evals.quick.cli import EXIT_OK, EXIT_REFUSED, EXIT_RUN_DID_NOT_COMPLETE
from threetears.evals.quick.files import EVAL_FILES, AfterRun, RunResult, Runnable, run_file
from threetears.evals.testing import RecordingRefused

#: The program name the harness's lines are printed under.
PROG: Final = "python -m threetears.evals.harness"

#: How the command is called.
USAGE: Final = f"{PROG} FILE --write-report MODEL [--budget-usd N] [--verbose]"


def _say(line: str) -> None:
    """One line of the command's output: its product, so a stream write rather than a log record."""
    sys.stdout.write(f"{line}\n")


def _refuse(why: str) -> int:
    sys.stderr.write(f"{PROG}: {why}\nusage: {USAGE}\n")
    return EXIT_REFUSED


def _analysed(path: Path, model: str, budget_usd: float) -> AfterRun:
    """What is done with each eval's result: the writer's report on it, and the exit code that comes to."""

    def after(ev: Runnable, result: RunResult) -> int:
        if not isinstance(result, Filed):
            sys.stderr.write(f"{PROG} {path}: {ev.name}: its result files no campaign for the writer to read\n")
            return EXIT_REFUSED
        try:
            with loading_eval_file(path):  # the recordings are read beside the file, as its own paths are
                written = asyncio.run(analyse(result, model, budget_usd=budget_usd))
                replayed = recording_path(result.name, model, written.snapshot_id)
        except (ValueError, RecordingRefused, EvalServiceError) as refused:
            sys.stderr.write(f"{PROG} {path}: {ev.name}: {refused}\n")
            return EXIT_REFUSED
        except ReportNotStored as not_stored:
            sys.stderr.write(f"{PROG} {path}: {ev.name}: the writer's report was not stored: {not_stored}\n")
            return EXIT_RUN_DID_NOT_COMPLETE
        if written.page is None:
            _say(f"analysis: none; the writer ended {written.ending}: {written.outcome.reason}")
            return EXIT_RUN_DID_NOT_COMPLETE
        _say(f"analysis: {analysis_page(result.name)}, written from {replayed}")
        return EXIT_OK

    return after


def main(argv: Sequence[str] | None = None) -> int:
    """Run an eval file's evals, then analyse each with the report writer.

    :param argv: The arguments after the program name; ``None`` reads ``sys.argv``.
    :ptype argv: Sequence[str] | None
    :return: The exit code.
    :rtype: int
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or Path(args[0]).suffix not in EVAL_FILES:
        return _refuse(f"the first argument is an eval file ({' or '.join(EVAL_FILES)})")
    path, rest = Path(args[0]), args[1:]
    model: str | None = None
    budget_usd = DEFAULT_BUDGET_USD
    verbose = False
    while rest:
        flag = rest.pop(0)
        if flag == "--verbose":
            verbose = True
        elif flag in ("--write-report", "--budget-usd") and rest:
            value = rest.pop(0)
            if flag == "--write-report":
                model = value
            else:
                try:
                    budget_usd = float(value)
                except ValueError:
                    return _refuse(f"--budget-usd takes an amount in US dollars, not {value!r}")
                if not math.isfinite(budget_usd) or budget_usd <= 0:
                    return _refuse(f"--budget-usd takes an amount above 0, not {value!r}")
        else:
            return _refuse(f"{flag} is not an option here, or names no value")
    if model is None:
        return _refuse("--write-report MODEL names the writer's model")
    return run_file(path, verbose=verbose, after=_analysed(path, model, budget_usd))


#: Nothing here is called from outside: the module is run.
__all__: list[str] = []


if __name__ == "__main__":
    sys.exit(main())
