"""``python -m threetears.evals.harness FILE --write-report MODEL [--budget-usd N] [--verbose]``, and ``... mint``.

The file's evals are run, printed and reported as ``python -m threetears.evals FILE`` runs them
(:func:`~threetears.evals.quick.files.run_file`, which finds them through its ``Runnable`` protocol), then the report
writer analyses each one (:func:`~threetears.evals.harness.analyse.analyse`), its recordings beside the file
(``recordings/report/``). Each run files a new campaign, whose snapshot no earlier recording was made over, so this
door writes live, on ``ANTHROPIC_API_KEY``. A writer's report replays offline only over the snapshot its recording
was made on: a :class:`~threetears.evals.harness.WriterEval` over frozen case files does. The line after each eval's
report names the analysis page and the recording it was written from, or says why there is none.

Exit codes are the file door's, each eval's analysis folded in: 1 when the writer stored nothing, 2 when the
analysis was refused (no recording and no key, a campaign the freeze refuses, a result that files no campaign).

``python -m threetears.evals.harness mint --documents FILE --profile FILE [--host module:factory] --campaign ID
--name NAME --out FILE.json.gz [...]`` mints one writer case from a host's export of a campaign
(:func:`~threetears.evals.harness.writer_case_from_export`), writing nothing to the host's store. Its exit codes are
the host commands': 0 when the case is written, 2 when the mint is refused (an export missing a document the freeze
reads, a profile factory that does not load, a profile whose declarations are not the ones declared, a manifest of
another campaign, a note the skill does not hold, a malformed command line), 3 when the
profile factory or the freeze raises an error nothing anticipated, its traceback on stderr.

``python -m threetears.evals.harness export --host module:factory --scope SCOPE --campaign ID --out DIR
[--built-by module:callable]`` writes the three files ``mint`` reads (``documents.json``, ``profile.json``,
``manifest.json``) from the host's own store (:func:`~threetears.evals.ops.campaign_export`), read-only. The factory
returns the host's :class:`~threetears.evals.kernel.host.EvalHost`; the profile declares it as the factory the mint
rebuilds the profile from, unless ``--built-by`` names another. Its exit codes are the mint's: 0 when the files are
written, 2 when the export is refused (a campaign the scope does not hold, a ``DIR`` that exists and holds files, a
factory that does not load or returns no host, a malformed command line), 3 when the factory or the export
raises an error nothing anticipated, its traceback on stderr. ``DIR`` must not exist yet, or be empty: the three
files are written into a new sibling directory renamed to ``DIR``, so either all three are there or none is.
"""

from __future__ import annotations

import argparse
import asyncio
import math
import shutil
import sys
import tempfile
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

from threetears.evals.actions.report_writing import ReportNotStored
from threetears.evals.harness.analyse import DEFAULT_BUDGET_USD, Filed, analyse, analysis_page, recording_path
from threetears.evals.analysis.analyst.skill import AnalystSkill
from threetears.evals.harness.writer_cases import WRITER_CASE_SUFFIX
from threetears.evals.harness.minting import (
    MintManifest,
    ProfileFile,
    eval_host_from,
    export_checked,
    host_profile_from,
    read_documents_file,
    read_profile_file,
    writer_case_from_export,
)
from threetears.evals.kernel.errors import EvalServiceError, RefusedInput
from threetears.evals.kernel.eval_file import loading_eval_file
from threetears.evals.ops import campaign_export
from threetears.evals.quick.cli import EXIT_FAILED, EXIT_OK, EXIT_REFUSED, EXIT_RUN_DID_NOT_COMPLETE
from threetears.evals.quick.files import EVAL_FILES, AfterRun, RunResult, Runnable, run_file
from threetears.evals.testing import RecordingRefused
from threetears.evals.schema.logs import contain, get_logger

log = get_logger(__name__)

#: The program name the harness's lines are printed under.
PROG: Final = "python -m threetears.evals.harness"

#: How the command is called.
USAGE: Final = f"{PROG} FILE --write-report MODEL [--budget-usd N] [--verbose]"

#: The subcommand that mints a writer case from a host's export.
MINT: Final = "mint"

#: Who a minted snapshot names as its freezer when the command line names nobody.
MINTED_BY: Final = "harness mint"

#: The subcommand that exports a campaign's closure as the files ``mint`` reads.
EXPORT: Final = "export"

#: The files an export writes, in the order the mint's command line names them.
_EXPORT_FILES: Final = ("documents.json", "profile.json", "manifest.json")


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


def _mint_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=f"{PROG} {MINT}",
        description="Mint one writer case from a host's export of a campaign, writing nothing to the host's store.",
    )
    parser.add_argument("--documents", required=True, help="the export: one JSON list of the campaign's documents")
    parser.add_argument(
        "--profile", help="the host's profile declaration: built_by (module:callable) and host_declarations_digest"
    )
    parser.add_argument(
        "--host",
        help="the profile's factory, module:callable, returning a HostProfile or an EvalHost; "
        "omitted, the declaration's built_by",
    )
    parser.add_argument("--manifest", help="a JSON file naming the documents that reasoned on the campaign")
    parser.add_argument(
        "--skill", help="the analyst skill directory whose notes are withheld; omitted, the shipped one"
    )
    parser.add_argument("--campaign", required=True, help="the campaign to freeze")
    parser.add_argument("--scope", help="the scope it lives in; omitted, the campaign document's own")
    parser.add_argument("--name", required=True, help="the case's name")
    parser.add_argument("--group", help="its split group; omitted, its name")
    parser.add_argument("--stratum", help="the kind of case it is (the domain it comes from)")
    parser.add_argument(
        "--withhold", action="append", default=[], metavar="NOTE", help="a note its analyst is never given (repeat)"
    )
    parser.add_argument(
        "--as-of", help="the instant it is frozen at (ISO-8601 with offset); omitted, when its evidence was complete"
    )
    parser.add_argument("--created-by", default=MINTED_BY, help=f"who mints it (default: {MINTED_BY!r})")
    parser.add_argument("--out", required=True, help="the case file to write, ending in .json.gz")
    return parser


def _skill_at(directory: str) -> AnalystSkill:
    """The analyst skill a ``--skill`` directory holds, or a refusal naming why it is not one."""
    try:
        return AnalystSkill.at(directory)
    except ValueError as unread:
        raise RefusedInput(
            f"--skill {directory}: {unread}", remedy="name a skill directory laid out as the shipped one"
        ) from unread


def _mint(argv: Sequence[str]) -> int:
    """Mint one writer case from a host's export, and say where it was written."""
    try:
        args = _mint_parser().parse_args(list(argv))
    except SystemExit as stopped:  # argparse's own exit: 0 after --help, 2 on a malformed command line
        return stopped.code if isinstance(stopped.code, int) else EXIT_REFUSED
    try:
        out = Path(args.out)
        if not out.name.endswith(WRITER_CASE_SUFFIX):
            raise RefusedInput(
                f"--out {args.out}: a writer case file's name ends in {WRITER_CASE_SUFFIX}",
                remedy=f"name the file <case>{WRITER_CASE_SUFFIX}",
            )
        documents = read_documents_file(args.documents)
        if args.scope is not None:  # whether the flag was given: the scope itself is only ever compared for identity
            documents = [document for document in documents if document.get("scope_id") == args.scope]
            if not documents:
                raise RefusedInput(
                    f"--scope {args.scope}: the export holds no document in it",
                    remedy="name a scope the export's documents carry, or none",
                )
        manifest = None if args.manifest is None else MintManifest.read(args.manifest)
        if manifest is not None and manifest.campaign_id != args.campaign:
            raise RefusedInput(
                f"the manifest {args.manifest} is campaign {manifest.campaign_id!r}'s, not {args.campaign!r}'s",
                remedy="name the manifest exported with the campaign: one campaign is one case",
            )
        declared: ProfileFile | None = None if args.profile is None else read_profile_file(args.profile)
        # The export is checked whole before the factory is imported, so an export can be dry-checked anywhere.
        asyncio.run(export_checked(documents, campaign_id=args.campaign, as_of=args.as_of))
        factory = args.host if args.host is not None else None if declared is None else declared.factory
        if factory is None:
            raise RefusedInput(
                "no profile factory is named: the freeze calls the profile's readers, which no file carries",
                remedy="set profile.json's 'built_by' to a module:callable importable here, or pass --host module:callable",
            )
        profile = host_profile_from(factory)
        skill = None if args.skill is None else _skill_at(args.skill)
        case = asyncio.run(
            writer_case_from_export(
                documents,
                profile,
                campaign_id=args.campaign,
                name=args.name,
                created_by=args.created_by,
                group=args.group,
                stratum=args.stratum,
                reasoned_on=[] if manifest is None else manifest.reasoned_on,
                withhold=args.withhold,
                skill=skill,
                declarations_digest=None if declared is None else declared.digest,
                as_of=args.as_of,
            )
        )
        try:
            written = case.write(out)
        except OSError as unwritten:
            raise RefusedInput(
                f"--out {out} cannot be written: {unwritten}", remedy="name a writable path"
            ) from unwritten
    except (RefusedInput, EvalServiceError) as refused:
        sys.stderr.write(f"{PROG} {MINT}: {refused}\n")
        return EXIT_REFUSED
    except Exception as unanticipated:
        # The distinct exit code is the point: uncaught, Python exits 1, which is a run that did not complete.
        contain(unanticipated, log=log, what=f"{PROG} {MINT}: failed on an unanticipated error", level="error")
        sys.stderr.write(f"{PROG} {MINT}: failed on an unanticipated error\n{traceback.format_exc()}")
        return EXIT_FAILED
    withheld = ", ".join(case.withhold) if case.withhold else "nothing"
    _say(f"{written}: case {case.name} over {case.snapshot.id}, withholding {withheld}")
    return EXIT_OK


def _export_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=f"{PROG} {EXPORT}",
        description="Export a campaign's closure from the host's store as the three files the mint reads, read-only.",
    )
    parser.add_argument(
        "--host", required=True, help="the host's factory, module:callable, returning the EvalHost whose store is read"
    )
    parser.add_argument("--scope", required=True, help="the scope the campaign lives in")
    parser.add_argument("--campaign", required=True, help="the campaign to export")
    parser.add_argument("--out", required=True, help="the directory the three files are written to")
    parser.add_argument(
        "--built-by",
        help="the module:callable the mint rebuilds the host's profile from; omitted, --host, which returns the host",
    )
    return parser


def _write_whole(out: Path, files: Mapping[str, bytes]) -> None:
    """Write ``files`` as the directory ``out``, all of them or none.

    They are written into a new sibling directory, which is then renamed to ``out``: a write failing part-way leaves
    no ``out`` behind, so the same command can run again. ``out`` does not exist yet, or is empty, and is replaced.

    :raises RefusedInput: A file or the directory cannot be written; nothing is left at ``out``.
    """
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        partial = Path(tempfile.mkdtemp(prefix=f".{out.name}.", suffix=".partial", dir=out.parent))
    except OSError as unwritten:
        raise RefusedInput(
            f"--out {out} cannot be written: {unwritten}", remedy="name a writable directory"
        ) from unwritten
    try:
        for name, data in files.items():
            (partial / name).write_bytes(data)
        if out.is_dir():
            out.rmdir()  # empty, as checked before the export: the rename puts the whole export in its place
        partial.rename(out)
    except OSError as unwritten:
        shutil.rmtree(partial)
        raise RefusedInput(
            f"--out {out} cannot be written: {unwritten}", remedy="name a writable directory"
        ) from unwritten


def _export(argv: Sequence[str]) -> int:
    """Write a campaign's closure as ``documents.json``, ``profile.json`` and ``manifest.json``, and say what."""
    try:
        args = _export_parser().parse_args(list(argv))
    except SystemExit as stopped:  # argparse's own exit: 0 after --help, 2 on a malformed command line
        return stopped.code if isinstance(stopped.code, int) else EXIT_REFUSED
    out = Path(args.out)
    try:
        # Checked before the host's module is imported, so a refused directory costs nothing and loses nothing.
        if held := [name for name in _EXPORT_FILES if (out / name).exists()]:
            raise RefusedInput(
                f"{', '.join(held)} already in {out}: an export never writes over one",
                remedy="name a directory that holds none of documents.json, profile.json and manifest.json",
            )
        if out.exists() and (not out.is_dir() or any(out.iterdir())):
            raise RefusedInput(
                f"--out {out} already holds other files: an export is written as one new directory",
                remedy="name a directory that does not exist yet, or an empty one",
            )
        host = eval_host_from(args.host)
        exported = campaign_export(
            host, args.scope, args.campaign, built_by=args.built_by if args.built_by is not None else args.host
        )
        _write_whole(out, exported.files())
    except (RefusedInput, EvalServiceError) as refused:
        sys.stderr.write(f"{PROG} {EXPORT}: {refused}\n")
        return EXIT_REFUSED
    except Exception as unanticipated:
        # The distinct exit code is the point: uncaught, Python exits 1, which is a run that did not complete.
        contain(unanticipated, log=log, what=f"{PROG} {EXPORT}: failed on an unanticipated error", level="error")
        sys.stderr.write(f"{PROG} {EXPORT}: failed on an unanticipated error\n{traceback.format_exc()}")
        return EXIT_FAILED
    counts = ", ".join(
        f"{doc_type} {count}" for doc_type, count in sorted(exported.manifest.doc_type_counts.items()) if count
    )
    unresolved = len(exported.manifest.unresolved)
    links = f"{unresolved} links unresolved, listed in manifest.json" if unresolved else "every link resolved"
    _say(f"{out}: {len(exported.documents)} documents of campaign {args.campaign} ({counts}); {links}")
    return EXIT_OK


def main(argv: Sequence[str] | None = None) -> int:
    """Run an eval file's evals, then analyse each with the report writer; or, after ``mint``, mint a writer case;
    or, after ``export``, export a campaign as the files a mint reads.

    :param argv: The arguments after the program name; ``None`` reads ``sys.argv``.
    :ptype argv: Sequence[str] | None
    :return: The exit code.
    :rtype: int
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if args[:1] == [MINT]:
        return _mint(args[1:])
    if args[:1] == [EXPORT]:
        return _export(args[1:])
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
