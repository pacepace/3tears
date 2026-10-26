"""The analyst's tool over the fixture snapshots, called from a shell: what an MCP client reads, one call at a time.

A prototype analyst run as an agent that reaches tools through a shell, not MCP, calls the analyst tool here. Each
call builds the toy operations host over an in-memory store, seeds it with every fixture snapshot (r1 to r5, the
misfit inputs and Kairo round 2, :mod:`packages.evals.tests.analyst_support`), mounts ``evals_analyst`` on a FastMCP
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

from packages.evals.tests.analyst_support import KAIRO, fixture_snapshots
from packages.evals.tests.ops_support import CALLER, ops_fixture
from threetears.evals.actions import analyst_tools
from threetears.evals.kernel import EvalReportSnapshot
from threetears.evals.transports.fastmcp import mount_fastmcp

__all__ = ["main"]

#: When every fixture snapshot is recorded as frozen: one instant, so a snapshot reads the same on every invocation.
_FROZEN_AT = "2026-10-26T00:00:00+00:00"

#: The tool every call is made to.
_TOOL = "evals_analyst"

#: The name Kairo round 2 is asked for by on the command line.
_KAIRO_NAME = "kairo"


@functools.cache
def _snapshots() -> dict[str, EvalReportSnapshot]:
    """Every fixture snapshot in the caller's scope, frozen at :data:`_FROZEN_AT`, by its command-line name."""
    frozen = fixture_snapshots(CALLER.scope_id)
    return {
        _KAIRO_NAME if name == KAIRO else name: snapshot.model_copy(update={"created_at": _FROZEN_AT})
        for name, snapshot in frozen.items()
    }


def _snapshot_ids() -> dict[str, str]:
    """Each fixture's ``snapshot_id``, by the name ``--snapshot-of`` takes: r1 to r5, the misfits, then ``kairo``."""
    return {name: snapshot.id for name, snapshot in _snapshots().items()}


def _server(role: str) -> FastMCP:
    """The toy host's server: the store seeded with every fixture snapshot, the analyst tool mounted for ``role``."""
    fixture = ops_fixture(generation=False)
    for snapshot in _snapshots().values():
        fixture.host.eval_host.storage.save_report_snapshot(snapshot)
    tools = analyst_tools(writes=False, role="editor" if role == "editor" else "analyst")
    return mount_fastmcp(fixture.host, scope=CALLER.scope_id, identity=CALLER.identity, tools=tools)


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
    ids = _snapshot_ids()
    if snapshot_of not in ids:
        return f"no fixture snapshot is named {snapshot_of!r}; --list-snapshots names them: {', '.join(ids)}"
    if arguments.get("snapshot_id", ids[snapshot_of]) != ids[snapshot_of]:
        return f"--snapshot-of {snapshot_of} names {ids[snapshot_of]}, and the arguments name another snapshot_id"
    return {**arguments, "snapshot_id": ids[snapshot_of]}


def main(argv: Sequence[str]) -> int:
    """Print the primer, the fixtures' snapshot ids, or one call's text (with ``--json``, its structured content too).

    :param argv: The command line, without the program's name.
    :ptype argv: Sequence[str]
    :return: 0 for a call that succeeded, 1 for one the tool refused, 2 for a usage error.
    :rtype: int
    """
    options = _parser().parse_args(list(argv))
    if options.list_snapshots:
        print("\n".join(f"{name}\t{snapshot_id}" for name, snapshot_id in _snapshot_ids().items()))
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
