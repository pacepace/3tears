"""``python -m threetears.evals FILE``: run every eval a ``.py`` or ``.toml`` file declares, print it, write its report.

The command line's first argument decides what it does. A ``.py`` or ``.toml`` file is run here:

- **A TOML file** declares one eval (:func:`~threetears.evals.quick.toml_file.load_toml`).
- **A Python file** is imported as a module, and every :class:`~threetears.evals.quick.contract.Eval` it defines at
  module level is run, in the order it defines them. No ``asyncio.run`` is needed in it.

Each eval runs into its default store, prints its result quietly (``--verbose`` prints it whole), and writes its
report to ``.evals/reports/<name>/``. The exit code is the worst of the evals': 0 done, 1 a run did not complete,
2 refused (the file, or what it declares), 3 an error nothing anticipated (its traceback printed), 4 a comparison's
gate failed. Anything else is the host command line (:func:`~threetears.evals.quick.cli.run_cli`), unchanged.
"""

from __future__ import annotations

import asyncio
import importlib.util
import re
import sys
import traceback
from collections.abc import Sequence
from pathlib import Path

from threetears.evals.quick.cli import (
    DEFAULT_PROG,
    EXIT_FAILED,
    EXIT_OK,
    EXIT_REFUSED,
    run_cli,
)
from threetears.evals.quick.contract import Eval
from threetears.evals.quick.toml_file import load_toml

#: The file suffixes the command line runs as eval files rather than host commands.
EVAL_FILES: tuple[str, ...] = (".py", ".toml")

#: Where each eval's report is written, under the working directory, in a directory named after the eval.
REPORTS_DIR = Path(".evals") / "reports"


def _say(line: str) -> None:
    """One line of the command's output: its product, so a stream write rather than a log record."""
    sys.stdout.write(f"{line}\n")


def load_evals(path: Path) -> list[Eval]:
    """Every eval a file declares: a TOML file's one, or a Python module's module-level ones, in order.

    Args:
        path: The ``.py`` or ``.toml`` file.

    Returns:
        The evals, not yet run.

    Raises:
        ValueError: A file that is not there, declares no eval, or that the eval refuses.
    """
    if not path.is_file():
        raise ValueError(f"{path} is not there")
    if path.suffix == ".toml":
        return [load_toml(path)]
    spec = importlib.util.spec_from_file_location(f"threetears_evals_file_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"{path} cannot be imported as a module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    found = [value for value in vars(module).values() if isinstance(value, Eval)]
    if not found:
        raise ValueError(f"{path} defines no Eval at module level; assign one (my_eval = Eval(...)) to run it")
    return found


def report_dir(name: str) -> Path:
    """Where an eval's report is written: :data:`REPORTS_DIR` / its name, made safe for a directory."""
    return REPORTS_DIR / (re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-") or "eval")


def run_file(path: Path, *, verbose: bool = False) -> int:
    """Run every eval ``path`` declares, print each result, write each report, and return the exit code.

    Args:
        path: The ``.py`` or ``.toml`` file.
        verbose: Print each result whole rather than quietly.

    Returns:
        The worst exit code of the evals' (see the module docstring).
    """
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
        code = max(code, result.exit_code())
    return code


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m threetears.evals``: an eval file, or the host command line.

    Args:
        argv: The arguments after the program name; ``None`` reads ``sys.argv``.

    Returns:
        The exit code.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if args and Path(args[0]).suffix in EVAL_FILES:
        rest = args[1:]
        if unknown := [arg for arg in rest if arg != "--verbose"]:
            sys.stderr.write(f"{DEFAULT_PROG} FILE takes only --verbose; not {' '.join(unknown)}\n")
            return EXIT_REFUSED
        try:
            return run_file(Path(args[0]), verbose="--verbose" in rest)
        except Exception:
            # NOSILENT: printed whole to stderr; the distinct exit code is the point — uncaught, Python exits 1, which
            # is the code for a run that did not complete.
            sys.stderr.write(f"{DEFAULT_PROG} {args[0]}: failed on an unanticipated error\n{traceback.format_exc()}")
            return EXIT_FAILED
    return run_cli(args)


__all__ = ["EVAL_FILES", "REPORTS_DIR", "load_evals", "main", "report_dir", "run_file"]
