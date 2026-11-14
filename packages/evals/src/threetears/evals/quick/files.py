"""``python -m threetears.evals FILE``: run every eval a ``.py`` or ``.toml`` file declares, print it, write its report.

The command line's first argument decides what it does. A ``.py`` or ``.toml`` file is run here:

- **A TOML file** declares one eval (:func:`~threetears.evals.quick.toml_file.load_toml`).
- **A Python file** is imported as a module, and every runnable value it defines at module level is run, in the order
  it defines them. No ``asyncio.run`` is needed in it.

**What runs is read through one protocol, never by kind.** A value is runnable when it satisfies :class:`Runnable`: a
``name`` and an ``async run()`` whose result satisfies :class:`RunResult` (``render``, ``write``, ``exit_code`` and
``why_failed``). :class:`~threetears.evals.quick.contract.Eval` and its
:class:`~threetears.evals.quick.result.Result` are one pair; any eval a file imports from elsewhere and builds is run
and printed the same way, with nothing here naming it.

Either file names its cases, recordings and labels relative to itself, so
``python -m threetears.evals examples/1-triage/triage.py`` runs the same from any directory: while a Python file is
imported, every relative path ``Eval``, ``Model``, ``World``, ``Judge`` and ``Log`` are handed is read beside it,
never by changing directory. A path the file's own code opens is its own to place
(``Path(__file__).with_name("policy.md")``). The store and the reports stay under the working directory.

**A Python file imports the modules beside it**, as ``python FILE`` lets it: its directory goes first on
``sys.path`` before it loads and stays there for the process, so ``from router_adapter import ...`` works at module
level and inside a runner that imports at call time alike. The door is a command, so the process is the file's;
:func:`load_evals` called as a library function does the same to its caller's process. A directory already on
``sys.path`` is not added again, and nothing is ever taken off it. A module is imported once per process, so two
files in two directories that each hold a ``helper.py`` both get the first one loaded; give siblings distinct names.

Each eval runs into its default store, prints its result quietly (``--verbose`` prints it whole), and writes its
report to ``.evals/reports/<name>/``. A run that failed (a replay miss fails one) prints its failure first, naming
what failed, and no verdict. The exit code is the worst of the evals': 0 done, 1 a run did not complete,
2 refused (the file, or what it declares), 3 an error nothing anticipated (its traceback printed), 4 a comparison's
gate failed, with a line after its report's saying why (:meth:`RunResult.why_failed`), in the report's words. Anything else is
the host command line (:func:`~threetears.evals.quick.cli.run_cli`), unchanged.
"""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
import re
import sys
import traceback
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Protocol, TypeGuard, runtime_checkable

from threetears.evals.kernel.eval_file import loading_eval_file
from threetears.evals.schema.logs import contain, get_logger
from threetears.evals.quick.cli import (
    DEFAULT_PROG,
    EXIT_FAILED,
    EXIT_OK,
    EXIT_REFUSED,
    run_cli,
)
from threetears.evals.quick.toml_file import load_toml

log = get_logger(__name__)


#: The file suffixes the command line runs as eval files rather than host commands.
EVAL_FILES: tuple[str, ...] = (".py", ".toml")

#: Where each eval's report is written, under the working directory, in a directory named after the eval.
REPORTS_DIR = Path(".evals") / "reports"


class RunResult(Protocol):
    """What a run comes to, as the file door prints it, writes it and exits on it.

    :class:`~threetears.evals.quick.result.Result` is one.
    """

    def render(self, *, full: bool = False) -> str:
        """The result as text: quiet, or whole with ``full=True``."""
        ...

    def write(self, out_dir: Path | str) -> Mapping[str, Path]:
        """Write the report under ``out_dir``; each file written, by role, the page under ``html``."""
        ...

    def exit_code(self) -> int:
        """What the command line exits with for it: 0 done, 1 a run did not complete, 4 a gate failed."""
        ...

    def why_failed(self) -> str | None:
        """Why its exit code is not 0, in one line and the report's words, where the report says; else ``None``."""
        ...


@runtime_checkable
class Runnable(Protocol):
    """A value an eval file defines that the file door runs: a name, and a run.

    :class:`~threetears.evals.quick.contract.Eval` is one. A file's module-level value is run when it satisfies this
    protocol and its ``run`` is a coroutine function (:func:`runnable`).
    """

    @property
    def name(self) -> str:
        """What titles its campaign and report, and names its report's directory (:func:`report_dir`)."""
        ...

    async def run(self) -> RunResult:
        """Run it, into its default store."""
        ...


#: What the file door does with each result after printing and writing it, handed the runnable and its result: the
#: exit code it adds, which the run's own is raised to.
AfterRun = Callable[[Runnable, RunResult], int]


def runnable(value: object) -> TypeGuard[Runnable]:
    """Whether a module-level value is one the file door runs: an instance satisfying :class:`Runnable` whose ``run``
    is a coroutine function. A class is never run, though it names both."""
    return (
        not isinstance(value, type)
        and isinstance(value, Runnable)
        and inspect.iscoroutinefunction(getattr(value, "run", None))
    )


def _say(line: str) -> None:
    """One line of the command's output: its product, so a stream write rather than a log record."""
    sys.stdout.write(f"{line}\n")


def _importable_beside(path: Path) -> None:
    """Put ``path``'s directory first on ``sys.path``, as ``python FILE`` does, unless it is already there."""
    directory = str(path.resolve().parent)
    if directory not in sys.path:
        sys.path.insert(0, directory)


def load_evals(path: Path) -> list[Runnable]:
    """Every eval a file declares: a TOML file's one, or a Python module's module-level runnable values, in order.

    A Python file's directory is put first on ``sys.path`` for the rest of the process, so the file and its runners
    import the modules beside it (see the module docstring).

    :param path: The ``.py`` or ``.toml`` file.
    :ptype path: Path
    :return: The evals, not yet run.
    :rtype: list[Runnable]
    :raises ValueError: A file that is not there, declares no eval, or that the eval refuses.
    """
    if not path.is_file():
        raise ValueError(f"{path} is not there")
    if path.suffix == ".toml":
        toml: list[Runnable] = [load_toml(path)]
        return toml
    spec = importlib.util.spec_from_file_location(f"threetears_evals_file_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"{path} cannot be imported as a module")
    module = importlib.util.module_from_spec(spec)
    _importable_beside(path)
    with loading_eval_file(path):  # the paths it names are read beside it, as a TOML file's are
        spec.loader.exec_module(module)
    found = [value for value in vars(module).values() if runnable(value)]
    if not found:
        raise ValueError(f"{path} defines no Eval at module level; assign one (my_eval = Eval(...)) to run it")
    return found


def report_dir(name: str) -> Path:
    """Where an eval's report is written: :data:`REPORTS_DIR` / its name, made safe for a directory."""
    return REPORTS_DIR / (re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-") or "eval")


def run_file(path: Path, *, verbose: bool = False, after: AfterRun | None = None) -> int:
    """Run every eval ``path`` declares, print each result, write each report, and return the exit code.

    An error nothing anticipated is printed whole to stderr with its traceback, and exits 3.

    :param path: The ``.py`` or ``.toml`` file.
    :ptype path: Path
    :param verbose: Print each result whole rather than quietly.
    :ptype verbose: bool
    :param after: What is done with each result once it is printed and its report written, whose exit code the
        file's is raised to; nothing when ``None``.
    :ptype after: AfterRun | None
    :return: The worst exit code of the evals' (see the module docstring).
    :rtype: int
    """
    try:
        return _run_file(path, verbose=verbose, after=after)
    except Exception as unanticipated:
        # NOSILENT: printed whole to stderr; the distinct exit code is the point — uncaught, Python exits 1, which
        # is the code for a run that did not complete.
        contain(unanticipated, log=log, what=f"{DEFAULT_PROG} {path}: failed on an unanticipated error", level="error")
        sys.stderr.write(f"{DEFAULT_PROG} {path}: failed on an unanticipated error\n{traceback.format_exc()}")
        return EXIT_FAILED


def _run_file(path: Path, *, verbose: bool, after: AfterRun | None) -> int:
    """:func:`run_file`, an unanticipated error raised."""
    try:
        evals = load_evals(path)
    except ValueError as refused:
        sys.stderr.write(f"{DEFAULT_PROG} {path}: {refused}\n")
        return EXIT_REFUSED
    code = EXIT_OK
    for ev in evals:
        try:
            result = asyncio.run(ev.run())
        except ValueError as refused:
            sys.stderr.write(f"{DEFAULT_PROG} {path}: {ev.name}: {refused}\n")
            code = max(code, EXIT_REFUSED)
            continue
        _say(result.render(full=verbose))
        written = result.write(report_dir(ev.name))
        _say(f"report: {written['html']}")
        exit_code = result.exit_code()
        if (why := result.why_failed()) is not None:
            # Why it exits as it does, in one line and the report's words: the guardrail or contrast a gate failed on.
            _say(f"exit {exit_code}: {why}")
        code = max(code, exit_code)
        if after is not None:
            code = max(code, after(ev, result))
    return code


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m threetears.evals``: an eval file, or the host command line.

    :param argv: The arguments after the program name; ``None`` reads ``sys.argv``.
    :ptype argv: Sequence[str] | None
    :return: The exit code.
    :rtype: int
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if args and Path(args[0]).suffix in EVAL_FILES:
        rest = args[1:]
        if unknown := [arg for arg in rest if arg != "--verbose"]:
            # The harness door takes the writer's flag; this door names it rather than leave the author guessing.
            elsewhere = (
                " (--write-report is python -m threetears.evals.harness's)" if "--write-report" in unknown else ""
            )
            sys.stderr.write(f"{DEFAULT_PROG} FILE takes only --verbose; not {' '.join(unknown)}{elsewhere}\n")
            return EXIT_REFUSED
        return run_file(Path(args[0]), verbose="--verbose" in rest)
    return run_cli(args)


__all__ = [
    "EVAL_FILES",
    "REPORTS_DIR",
    "AfterRun",
    "RunResult",
    "Runnable",
    "load_evals",
    "main",
    "report_dir",
    "run_file",
]
