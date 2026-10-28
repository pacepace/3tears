"""The analyst's tool over the fixture snapshots, called from a shell: what an MCP client reads, one call at a time.

A prototype analyst run as an agent that reaches tools through a shell, not MCP, calls the analyst tool here. Each
call builds the toy operations host over an in-memory store, seeds it with every fixture snapshot (r1 to r5, the
misfit inputs, Kairo round 2 and the shape campaigns s1 to s3, :mod:`packages.evals.tests.analyst_support`), mounts ``evals_analyst`` on a FastMCP
server as a host does (:func:`~threetears.evals.transports.fastmcp.mount_fastmcp`, its primer as the server's
instructions), calls it once through an in-memory MCP client, and prints the text the client got. Run from the
repository root, as a file or as ``python -m packages.evals.tests.fixtures.report_plans.analyst_cli``::

    uv run python packages/evals/tests/fixtures/report_plans/analyst_cli.py --list-snapshots
    uv run python packages/evals/tests/fixtures/report_plans/analyst_cli.py --primer
    uv run python packages/evals/tests/fixtures/report_plans/analyst_cli.py '{"action": "help"}'
    uv run python packages/evals/tests/fixtures/report_plans/analyst_cli.py --snapshot-of r4 \\
        '{"action": "campaign_overview"}'
    uv run python packages/evals/tests/fixtures/report_plans/analyst_cli.py --role editor --json --snapshot-of kairo \\
        '{"action": "options_dominance"}'

- ``--role analyst|editor``: the analyst's cut (every workflow) or the report editor's (no blocks). Either is mounted
  read-only (``writes=False``), as a loop that froze the snapshot itself mounts it: the snapshots are frozen here.
- ``--snapshot-of <name>``: put that fixture's ``snapshot_id`` into the arguments (``kairo`` for Kairo round 2, a
  misfit by its name, quoted).
- ``--json``: print ``{"text", "structured_content", "is_error"}`` instead of the text alone.
- ``--primer``: print the server's instructions; ``--list-snapshots``: each fixture's name and ``snapshot_id``.

**Writing a report** (``write``): the report writer's entry point (``write_report``) run live over one fixture
snapshot, recorded, the one route here that reads a key::

    ANTHROPIC_API_KEY=... uv run python packages/evals/tests/fixtures/report_plans/analyst_cli.py write \\
        --snapshot-of kairo --model <claude model id> --budget-usd 20 --out report_docs/prototype4/kairo \\
        [--withhold kairo-round2.md] [--effort high]

It reads ``ANTHROPIC_API_KEY`` and nothing else, records every turn of every session into ``<out>/turns.jsonl``, and
writes beside it, whether the report was stored or the store refused it, ``replay.json`` (what a replay must reach:
the ending and why, whether it was stored, the rounds, the tally's sums; ``test_report_writer_replay.py`` replays
every recording that has one), ``outcome.json`` (the ending, why, a refusal's faults, the reviews), ``generation.json``
(the tally, the reviews and every session, the analyst's and each review round's) and, for a stored report,
``doc.json``, ``record.json``, ``page.html`` and ``page.md``. A recording is made once: an ``--out`` that holds one
is refused, as is a run with no key, each saying what to do instead. A paid run is never made from a test.

The exit status is 0 for a call that succeeded, 1 for one the tool refused (the MCP error flag), 2 for a usage error.
Every call is deterministic: one scope, one caller and one freeze instant, so a snapshot's id and every figure are the
same on every invocation.
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

#: This file as a module, named from the repository root as the test suite names it.
_MODULE = "packages.evals.tests.fixtures.report_plans.analyst_cli"

if __name__ == "__main__" and not __package__:
    # Run as a file, its imports could not be named from the repository root: run it again as the module, from the
    # root, which names them so without putting any directory on the path.
    os.chdir(Path(__file__).resolve().parents[5])
    os.execv(sys.executable, [sys.executable, "-m", _MODULE, *sys.argv[1:]])

from fastmcp import Client, FastMCP

from packages.evals.tests.analyst_support import KAIRO, fixture_snapshots, shape_snapshots
from packages.evals.tests.ops_support import CALLER, ops_fixture
from threetears.evals.actions import analyst_tools
from threetears.evals.actions.report_writing import ReportNotStored, ReportWritten, write_report
from threetears.evals.analysis.report.doc_page.html import doc_page_html
from threetears.evals.analysis.report.doc_page.markdown import doc_page_markdown
from threetears.evals.kernel import EvalReportSnapshot
from threetears.evals.ops import OpsHost
from threetears.evals.ops.report_writer.loop import TurnModel
from threetears.evals.ops.report_writer.tally import ReportBudget
from threetears.evals.providers import Model
from threetears.evals.transports.fastmcp import mount_fastmcp

__all__ = ["main", "replay_expectation", "seeded_host", "snapshot_ids", "written_files"]

#: When every fixture snapshot is recorded as frozen: one instant, so a snapshot reads the same on every invocation.
_FROZEN_AT = "2026-10-26T00:00:00+00:00"

#: The tool every call is made to.
_TOOL = "evals_analyst"

#: The name Kairo round 2 is asked for by on the command line.
_KAIRO_NAME = "kairo"


@functools.cache
def _snapshots() -> dict[str, EvalReportSnapshot]:
    """Every fixture snapshot in the caller's scope, then the shape campaigns s1 to s3, frozen at :data:`_FROZEN_AT`,
    by its command-line name."""
    frozen = fixture_snapshots(CALLER.scope_id) | shape_snapshots(CALLER.scope_id)
    return {
        _KAIRO_NAME if name == KAIRO else name: snapshot.model_copy(update={"created_at": _FROZEN_AT})
        for name, snapshot in frozen.items()
    }


def snapshot_ids() -> dict[str, str]:
    """Each fixture's ``snapshot_id``, by the name ``--snapshot-of`` takes: r1 to r5, the misfits, ``kairo``, then s1 to
    s3.

    :return: Name -> snapshot id.
    :rtype: dict[str, str]
    """
    return {name: snapshot.id for name, snapshot in _snapshots().items()}


def seeded_host() -> OpsHost:
    """The toy host as an operations host, its store seeded with every fixture snapshot, as every call here reads it.

    :return: The host.
    :rtype: OpsHost
    """
    fixture = ops_fixture(generation=False)
    for snapshot in _snapshots().values():
        fixture.host.eval_host.storage.save_report_snapshot(snapshot)
    return fixture.host


def _server(role: str) -> FastMCP:
    """The toy host's server: the store seeded with every fixture snapshot, the analyst tool mounted for ``role``."""
    tools = analyst_tools(writes=False, role="editor" if role == "editor" else "analyst")
    return mount_fastmcp(seeded_host(), scope=CALLER.scope_id, identity=CALLER.identity, tools=tools)


async def _called(server: FastMCP, arguments: Mapping[str, Any]) -> tuple[str, Any, bool]:
    """One call to the analyst tool through an MCP client: its text blocks joined, its structured content, its flag."""
    async with Client(server) as client:
        result = await client.call_tool(_TOOL, dict(arguments), raise_on_error=False)
    text = "\n".join(block.text for block in result.content if hasattr(block, "text"))
    return text, result.structured_content, result.is_error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="analyst_cli.py", description="Call the analyst tool over the fixture snapshots, as an MCP client would."
    )
    parser.add_argument("--role", choices=("analyst", "editor"), default="analyst")
    parser.add_argument("--snapshot-of", metavar="NAME", help="put this fixture's snapshot_id into the arguments")
    parser.add_argument("--json", action="store_true", help="print the text, the structured content and the flag")
    parser.add_argument("--primer", action="store_true", help="print the server's instructions")
    parser.add_argument("--list-snapshots", action="store_true", help="print each fixture's name and snapshot_id")
    parser.add_argument("arguments", nargs="?", help="the call's arguments, a JSON object naming its action")
    return parser


def _arguments(given: str | None, snapshot_of: str | None) -> dict[str, Any] | str:
    """The call's arguments, the fixture's snapshot id put in; or what is wrong with them, as a usage error."""
    if given is None:
        return "name the call's arguments: a JSON object, its action among them"
    try:
        arguments = json.loads(given)
    except json.JSONDecodeError as invalid:
        return f"the arguments are not JSON: {invalid}"
    if not isinstance(arguments, dict):
        return "the arguments are a JSON object, its action among them"
    if snapshot_of is None:
        return arguments
    ids = snapshot_ids()
    if snapshot_of not in ids:
        return f"no fixture snapshot is named {snapshot_of!r}; --list-snapshots names them: {', '.join(ids)}"
    if arguments.get("snapshot_id", ids[snapshot_of]) != ids[snapshot_of]:
        return f"--snapshot-of {snapshot_of} names {ids[snapshot_of]}, and the arguments name another snapshot_id"
    return {**arguments, "snapshot_id": ids[snapshot_of]}


#: The one variable ``write`` reads: the key of a live run it records.
_KEY = "ANTHROPIC_API_KEY"

#: Where ``write`` records every turn, inside its ``--out``.
RECORDING = "turns.jsonl"

#: What a replay of a recording must reach, beside it.
REPLAY = "replay.json"


def _write_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="analyst_cli.py write",
        description="Write a report live over one fixture snapshot with write_report, recording every turn.",
    )
    parser.add_argument("--snapshot-of", metavar="NAME", required=True, help="the fixture snapshot to write on")
    parser.add_argument("--model", metavar="ID", required=True, help="the Anthropic model id every session runs on")
    parser.add_argument("--budget-usd", metavar="USD", type=float, required=True, help="the report's dollar cap")
    parser.add_argument("--out", metavar="DIR", required=True, help="where the recording and the report are written")
    parser.add_argument("--withhold", metavar="NOTE", action="append", default=[], help="an exemplar note to withhold")
    parser.add_argument("--effort", default="high", help="the thinking effort word every turn is sent with")
    return parser


def replay_expectation(
    written: ReportWritten, *, snapshot_of: str, model: str, budget_usd: float, withhold: Sequence[str]
) -> dict[str, Any]:
    """What a replay of a recording must reach, as ``replay.json`` holds it: how it was run, and how it ended.

    :param written: What the recorded run came to.
    :ptype written: ReportWritten
    :param snapshot_of: The fixture snapshot, by its command-line name.
    :ptype snapshot_of: str
    :param model: The model id it ran on.
    :ptype model: str
    :param budget_usd: Its dollar cap.
    :ptype budget_usd: float
    :param withhold: The exemplar notes withheld.
    :ptype withhold: Sequence[str]
    :return: ``{"snapshot_of", "model", "budget_usd", "withhold", "ending", "reason", "stored", "analyst_rounds",
        "editor_rounds", "tally"}``; ``reason`` why it ended other than ``stored``, ``stored`` whether a document was
        stored, ``tally`` the total's turns, tool calls, bounces and cost.
    :rtype: dict[str, Any]
    """
    total = written.generation["tally"]["total"]
    return {
        "snapshot_of": snapshot_of,
        "model": model,
        "budget_usd": budget_usd,
        "withhold": list(withhold),
        "ending": written.ending,
        "reason": written.outcome.reason,
        "stored": written.stored is not None,
        "analyst_rounds": written.outcome.analyst_rounds,
        "editor_rounds": written.outcome.editor_rounds,
        "tally": {name: total[name] for name in ("turns", "tool_calls", "bounces", "cost_usd")},
    }


def written_files(written: ReportWritten, expectation: Mapping[str, Any]) -> dict[str, str]:
    """The files a run writes beside its recording, by name.

    :param written: What the run came to.
    :ptype written: ReportWritten
    :param expectation: Its ``replay.json`` (:func:`replay_expectation`).
    :ptype expectation: Mapping[str, Any]
    :return: File name -> its text.
    :rtype: dict[str, str]
    """
    outcome = written.outcome
    files = {
        REPLAY: json.dumps(expectation, indent=2) + "\n",
        "outcome.json": json.dumps(
            {
                "ending": written.ending,
                "reason": outcome.reason,
                "faults": [fault.line() for fault in outcome.faults],
                "reviews": [review.to_json() for review in outcome.reviews],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        "generation.json": json.dumps(written.generation, indent=2, ensure_ascii=False) + "\n",
    }
    if written.stored is not None and written.page is not None:
        files["doc.json"] = json.dumps(written.stored.doc, indent=2, ensure_ascii=False) + "\n"
        files["record.json"] = json.dumps(written.stored.to_dict(), indent=2, ensure_ascii=False) + "\n"
        files["page.html"] = doc_page_html(written.page)
        files["page.md"] = doc_page_markdown(written.page)
    return files


async def _written(
    host: OpsHost, model: Model, snapshot_id: str, budget_usd: float, withhold: Sequence[str]
) -> ReportWritten:
    turns: TurnModel = model
    try:
        return await write_report(
            host,
            CALLER,
            model=turns,
            budget=ReportBudget(usd=budget_usd),
            snapshot_id=snapshot_id,
            withhold=withhold,
        )
    finally:
        await model.aclose()


def _write(argv: Sequence[str]) -> int:
    """Write a report live and record it (the module docstring's ``write``); 2 for a route it refuses."""
    options = _write_parser().parse_args(list(argv))
    out = Path(options.out).resolve()
    recording = out / RECORDING
    if recording.exists():
        print(
            f"analyst_cli.py write: {recording} is recorded already, and a recording is made once: replay it with "
            f"test_report_writer_replay.py (it reads {REPLAY} beside it), or name a new --out to write again",
            file=sys.stderr,
        )
        return 2
    key = os.environ.get(_KEY)
    if not key:
        print(
            f"analyst_cli.py write records a live, paid run and reads its key from {_KEY} alone: set it to write one. "
            "A recorded run is replayed with test_report_writer_replay.py, which needs no key",
            file=sys.stderr,
        )
        return 2
    ids = snapshot_ids()
    if options.snapshot_of not in ids:
        print(
            f"analyst_cli.py write: no fixture snapshot is named {options.snapshot_of!r}; --list-snapshots names them",
            file=sys.stderr,
        )
        return 2
    out.mkdir(parents=True, exist_ok=True)
    model = Model(options.model, provider="anthropic", api_key=key, effort=options.effort, turn_recording=recording)
    refused: ReportNotStored | None = None
    try:
        written = asyncio.run(
            _written(seeded_host(), model, ids[options.snapshot_of], options.budget_usd, options.withhold)
        )
    except ReportNotStored as not_stored:
        # The paid run's record rides on the refusal: write it beside the recording, then raise it.
        written, refused = not_stored.written, not_stored
    expectation = replay_expectation(
        written,
        snapshot_of=options.snapshot_of,
        model=options.model,
        budget_usd=options.budget_usd,
        withhold=options.withhold,
    )
    for name, text in written_files(written, expectation).items():
        (out / name).write_text(text, encoding="utf-8")
    if refused is not None:
        raise refused
    print(f"{written.ending}: {written.outcome.reason or 'stored'}; written to {out}")
    return 0 if written.stored is not None else 1


def main(argv: Sequence[str]) -> int:
    """Print the primer, the fixtures' snapshot ids, or one call's text (with ``--json``, its structured content too).

    :param argv: The command line, without the program's name.
    :ptype argv: Sequence[str]
    :return: 0 for a call that succeeded, 1 for one the tool refused, 2 for a usage error; for ``write``, 0 for a
        report stored, 1 for one that was not, 2 for a route it refuses.
    :rtype: int
    """
    if argv and argv[0] == "write":
        return _write(argv[1:])
    options = _parser().parse_args(list(argv))
    if options.list_snapshots:
        print("\n".join(f"{name}\t{snapshot_id}" for name, snapshot_id in snapshot_ids().items()))
        return 0
    if options.primer:
        print(_server(options.role).instructions)
        return 0
    arguments = _arguments(options.arguments, options.snapshot_of)
    if isinstance(arguments, str):
        print(f"analyst_cli.py: {arguments}", file=sys.stderr)
        return 2
    text, structured, is_error = asyncio.run(_called(_server(options.role), arguments))
    if options.json:
        print(json.dumps({"text": text, "structured_content": structured, "is_error": is_error}, indent=2))
    else:
        print(text)
    return 1 if is_error else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
