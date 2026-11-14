"""``python -m threetears.langgraph.checkpoint_maintenance``: report on, prune, or clean checkpoint tables.

::

    python -m threetears.langgraph.checkpoint_maintenance report --dsn DSN [--schema S] [--owned-sql SQL] [--json]
    python -m threetears.langgraph.checkpoint_maintenance prune --dsn DSN [--batch N] [--limit N] (--dry-run | --yes)
    python -m threetears.langgraph.checkpoint_maintenance delete-unowned --dsn DSN --owned-sql SQL (--dry-run | --yes)

``report`` is read-only. ``prune`` and ``delete-unowned`` delete rows, and run only with ``--yes``
or as ``--dry-run``. ``--vacuum`` follows a real run with a plain ``VACUUM (ANALYZE)`` of the two
tables, which makes the freed space reusable by them; nothing here runs ``VACUUM FULL``, which
returns space to the operating system but locks the table for its whole run.

The tool writes through an L3-only saver: it reaches the database and no app cache. What that
leaves in a running app's L2 (NATS KV) cache, read from the shipped adapter
(:class:`~threetears.langgraph.nats_l2.NatsCheckpointL2Cache`):

- after ``prune``: nothing stale. Each L2 key holds one bundle, the checkpoint that was newest when
  it was written, and ``keep_latest`` keeps exactly that one.
- after ``delete-unowned``: the deleted threads' bundles stay in the bucket. The adapter opens the
  bucket with no TTL, so they expire only if the app declared the bucket with one (a ``max_age``)
  before anything else opened it; otherwise they stay until overwritten, deleted, or the NATS server
  restarts (the bucket is in memory by default). An app never reads a thread it no longer owns, so
  this is space, not a wrong answer; an erasure must also delete the keys ``<stored thread id>`` and
  ``<stored thread id>.*`` from the bucket, or run the delete through the app's own saver with its
  caches wired (``aprune(..., strategy="delete")`` evicts them).

A pod's L1 is local to it and keeps a deleted thread until that pod evicts it or restarts.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from typing import TextIO

import asyncpg

from threetears.langgraph.checkpoint import ThreeTierCheckpointSaver
from threetears.langgraph.checkpoint_maintenance import (
    DEFAULT_BATCH,
    CheckpointSweep,
    CheckpointUsage,
    TableUsage,
    UnownedUsage,
    checkpoint_usage,
    delete_unowned_threads,
    prune_threads,
)
from threetears.langgraph.checkpoint_scope import CheckpointScope
from threetears.langgraph.protocols import AsyncpgPoolAdapter

__all__: list[str] = []

_EPILOG = """\
L2 caches: prune leaves nothing stale in an app's L2 (the newest checkpoint, the one kept, is
the only one an L2 key holds). delete-unowned leaves the deleted threads' bundles in the app's
NATS KV bucket: they expire only if the app declared the bucket with a TTL, else they stay until
overwritten, deleted or the server restarts. For an erasure, delete the keys too.

Space: deleted rows free space inside the tables (after VACUUM); only VACUUM FULL or pg_repack
returns it to the operating system. This tool never runs VACUUM FULL.
"""


def _parser() -> argparse.ArgumentParser:
    """the command line.

    :return: the parser
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog="python -m threetears.langgraph.checkpoint_maintenance",
        description="Report on, prune, or delete unowned threads from LangGraph checkpoint tables.",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    actions = parser.add_subparsers(dest="action", required=True)
    for name, summary in (
        ("report", "what the tables hold and what pruning would remove (read-only)"),
        ("prune", "keep only the newest checkpoint per thread and namespace"),
        ("delete-unowned", "delete every thread the --owned-sql query does not name"),
    ):
        action = actions.add_parser(name, help=summary, description=summary, epilog=_EPILOG)
        action.formatter_class = argparse.RawDescriptionHelpFormatter
        action.add_argument("--dsn", required=True, help="PostgreSQL DSN")
        action.add_argument("--schema", help="schema holding the checkpoint tables (sets search_path)")
        action.add_argument(
            "--owned-sql",
            required=name == "delete-unowned",
            help="a SELECT whose first column is the stored thread ids the app still owns",
        )
        action.add_argument("--json", action="store_true", help="print the result as JSON")
        if name != "report":
            action.add_argument("--batch", type=int, default=DEFAULT_BATCH, help="threads per batch")
            action.add_argument("--limit", type=int, help="most threads to handle")
            action.add_argument("--dry-run", action="store_true", help="count, delete nothing")
            action.add_argument("--yes", action="store_true", help="delete; required unless --dry-run")
            action.add_argument("--vacuum", action="store_true", help="run VACUUM (ANALYZE) on both tables after")
    return parser


def _mib(size: int) -> str:
    """a byte count for people.

    :param size: bytes
    :ptype size: int
    :return: the count in MiB, one decimal
    :rtype: str
    """
    return f"{size / 1_048_576:.1f} MiB"


def _table_lines(usage: TableUsage) -> list[str]:
    """one table's lines of the text report.

    :param usage: the table's usage
    :ptype usage: TableUsage
    :return: the lines
    :rtype: list[str]
    """
    return [
        f"{usage.table}: {_mib(usage.total_bytes)} total, {_mib(usage.heap_bytes)} heap",
        f"  about {usage.rows} rows and {usage.dead_tuples} dead tuples (statistics estimates, not counts)",
    ]


def _unowned_line(unowned: UnownedUsage) -> str:
    """the text report's line on threads the app no longer owns.

    :param unowned: what :func:`checkpoint_usage` measured outside the owned set
    :ptype unowned: UnownedUsage
    :return: the line
    :rtype: str
    """
    return (
        f"not owned (owned query returned {unowned.owned_threads} rows): {unowned.threads} threads, "
        f"{unowned.checkpoint_rows} checkpoint rows, {unowned.write_rows} write rows, {_mib(unowned.bytes)} of blobs"
    )


def _usage_text(usage: CheckpointUsage) -> str:
    """the text report.

    :param usage: what :func:`checkpoint_usage` measured
    :ptype usage: CheckpointUsage
    :return: the report
    :rtype: str
    """
    lines = [
        *_table_lines(usage.checkpoints),
        *_table_lines(usage.writes),
        f"threads: {usage.threads}, of which {usage.threads_with_history} hold more than one checkpoint "
        "in some namespace",
        f"prune (keep_latest) would remove {usage.prunable_checkpoint_rows} checkpoint rows and "
        f"{usage.prunable_write_rows} write rows, {_mib(usage.prunable_bytes)} of blobs",
    ]
    if usage.held_namespaces:
        lines.append(f"  {usage.held_namespaces} namespaces kept whole: their newest checkpoint needs its ancestors")
    if usage.unowned is not None:
        lines.append(_unowned_line(usage.unowned))
    lines.append("deleted rows become reusable space after VACUUM; only VACUUM FULL or pg_repack returns it to the OS")
    return "\n".join(lines)


def _sweep_text(sweep: CheckpointSweep) -> str:
    """the text result of a prune or delete-unowned run.

    :param sweep: the run's totals
    :ptype sweep: CheckpointSweep
    :return: the result
    :rtype: str
    """
    verb = "would delete" if sweep.dry_run else "deleted"
    text = (
        f"{sweep.strategy}: {sweep.threads} threads in {sweep.batches} batches; {verb} {sweep.checkpoint_rows} "
        f"checkpoint rows and {sweep.write_rows} write rows, {_mib(sweep.bytes)} of blobs"
    )
    if sweep.namespaces_held:
        text += f"; {sweep.namespaces_held} namespaces kept whole"
    return text


async def _run(args: argparse.Namespace, out: TextIO) -> int:
    """run one action against the database.

    :param args: the parsed command line
    :ptype args: argparse.Namespace
    :param out: where the result is written
    :ptype out: TextIO
    :return: the exit status
    :rtype: int
    """
    server_settings = {"search_path": args.schema} if args.schema else None
    pool = await asyncpg.create_pool(args.dsn, min_size=1, max_size=2, server_settings=server_settings)
    try:
        executor = AsyncpgPoolAdapter(pool)
        if args.action == "report":
            usage = await checkpoint_usage(executor, owned_threads_sql=args.owned_sql)
            out.write((json.dumps(asdict(usage)) if args.json else _usage_text(usage)) + "\n")
            return 0
        saver = ThreeTierCheckpointSaver(
            executor,
            scope=CheckpointScope.unscoped(reason="checkpoint maintenance works on stored thread ids"),
        )
        if args.action == "prune":
            sweep = await prune_threads(saver, executor, batch=args.batch, limit=args.limit, dry_run=args.dry_run)
        else:
            sweep = await delete_unowned_threads(
                saver,
                executor,
                owned_threads_sql=args.owned_sql,
                batch=args.batch,
                limit=args.limit,
                dry_run=args.dry_run,
            )
        if args.vacuum and not args.dry_run:
            # VACUUM cannot run in a transaction; the adapter's execute runs it on its own
            await executor.execute("VACUUM (ANALYZE) checkpoints")
            await executor.execute("VACUUM (ANALYZE) checkpoint_writes")
        out.write((json.dumps(asdict(sweep)) if args.json else _sweep_text(sweep)) + "\n")
        return 0
    finally:
        await pool.close()


def main(argv: Sequence[str] | None = None) -> int:
    """parse the command line and run the action.

    :param argv: the arguments, or None for ``sys.argv[1:]``
    :ptype argv: Sequence[str] | None
    :return: the exit status: 0 done, 2 a refused command line
    :rtype: int
    """
    parser = _parser()
    args = parser.parse_args(argv)
    if args.action != "report" and not (args.dry_run or args.yes):
        parser.error(f"{args.action} deletes rows: pass --yes to run it, or --dry-run to count only")
    return asyncio.run(_run(args, sys.stdout))


if __name__ == "__main__":
    raise SystemExit(main())
