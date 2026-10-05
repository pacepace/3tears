"""The engine's command line: launch a template's runs, list what a scope holds, and read a campaign's report.

``python -m threetears.evals`` runs it with ``--host module:factory`` naming the host to work in: a
zero-argument callable returning an :class:`~threetears.evals.contracts.host.EvalHost`, or a
:class:`~threetears.evals.run.LaunchHost` for ``run``, which launches. A product mounts the same
commands under its own CLI by calling :func:`run_cli` with its own ``host_factory``, and its users
then never name the host::

    python -m threetears.evals run --host myapp.evals:build_host --scope dev --template T --subject S --model M
    python -m threetears.evals ls --host myapp.evals:build_host --scope dev
    python -m threetears.evals report CAMPAIGN --host myapp.evals:build_host --scope dev

- ``run`` launches through :func:`~threetears.evals.run.start_run`, waits for the runs' jobs, and
  prints each run's summary. It exits 0 when every run completed and 1 when any did not.
- ``ls`` prints the scope's templates, runs and campaigns.
- ``report`` prints the campaign's analysis bundle as JSON
  (:func:`~threetears.evals.analysis.inspect_campaign_bundle`): what a generation would read, assembled
  without calling any model.

A refusal — a host that cannot be loaded, a template or campaign that is not there, a launch the
engine refuses — prints its reason to stderr and exits 2, which is also argparse's code for a
malformed command line.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import sys
from collections.abc import Callable, Sequence

from threetears.evals.analysis import inspect_campaign_bundle, list_campaigns
from threetears.evals.contracts import EvalServiceError
from threetears.evals.contracts.host import EvalHost
from threetears.evals.quick.summary import summarize_run
from threetears.evals.run import LaunchHost, list_runs, list_templates, start_run

#: What names the host the commands work in: called once, with no arguments, per invocation.
HostFactory = Callable[[], EvalHost | LaunchHost]

#: The program name the command line prints when it is run as ``python -m threetears.evals``.
DEFAULT_PROG = "python -m threetears.evals"

#: Exit codes: every run completed (or the command read what it was asked for); a run did not
#: complete; the command was refused before it could do anything.
EXIT_OK = 0
EXIT_RUN_DID_NOT_COMPLETE = 1
EXIT_REFUSED = 2


class _Refused(Exception):
    """A command the CLI cannot carry out, with the reason to print."""


def _load_host_factory(spec: str) -> HostFactory:
    """The host factory a ``module:attribute`` spec names.

    Args:
        spec: ``package.module:callable``.

    Returns:
        The callable.

    Raises:
        _Refused: The spec is not ``module:attribute``, the module cannot be imported, it has no such
            attribute, or the attribute is not callable.
    """
    module_name, colon, attribute = spec.partition(":")
    if not colon or not module_name or not attribute:
        raise _Refused(f"--host takes module:factory, a module path and a callable in it; got {spec!r}")
    try:
        module = importlib.import_module(module_name)
    except ImportError as missing:
        raise _Refused(f"--host {spec}: cannot import {module_name!r}: {missing}") from missing
    factory = getattr(module, attribute, None)
    if factory is None:
        raise _Refused(f"--host {spec}: module {module_name!r} has no attribute {attribute!r}")
    if not callable(factory):
        raise _Refused(f"--host {spec}: {attribute!r} is a {type(factory).__name__}, not a callable returning a host")
    return factory  # type: ignore[no-any-return]


def build_parser(prog: str = DEFAULT_PROG, *, takes_host: bool = True) -> argparse.ArgumentParser:
    """The command line's parser.

    Args:
        prog: The program name usage lines print.
        takes_host: Whether each command takes ``--host``. A product that mounts the commands with its
            own host factory passes ``False``, and the option does not exist.

    Returns:
        The parser; ``command`` on the parsed namespace names the subcommand.
    """
    parser = argparse.ArgumentParser(prog=prog, description="Run, list and report evals.")
    commands = parser.add_subparsers(dest="command", required=True, metavar="{run,ls,report}")

    def command(name: str, help_text: str) -> argparse.ArgumentParser:
        sub = commands.add_parser(name, help=help_text, description=help_text)
        if takes_host:
            sub.add_argument("--host", required=True, metavar="MODULE:FACTORY", help="the host to work in")
        sub.add_argument("--scope", required=True, help="the scope to read and write in")
        return sub

    run = command("run", "Launch a template's runs, wait for them, and print each run's summary.")
    run.add_argument("--template", required=True, help="the template to run, by id")
    run.add_argument("--subject", required=True, help="the subject the runs measure, as the host names it")
    run.add_argument("--model", action="append", default=[], help="a candidate model; repeat for one arm each")
    run.add_argument("--k", type=int, default=1, help="repeats per case (default 1)")
    command("ls", "List the scope's templates, runs and campaigns.")
    report = command("report", "Print a campaign's analysis bundle as JSON, without calling a model.")
    report.add_argument("campaign", help="the campaign, by id")
    return parser


def run_cli(
    argv: Sequence[str] | None = None, *, host_factory: HostFactory | None = None, prog: str = DEFAULT_PROG
) -> int:
    """Parse ``argv`` and carry out the command, printing to stdout and refusals to stderr.

    Args:
        argv: The arguments after the program name; ``None`` reads ``sys.argv``.
        host_factory: The host to work in, for a product mounting these commands; ``None`` takes it
            from ``--host``.
        prog: The program name usage lines print.

    Returns:
        The exit code: 0, 1 when a launched run did not complete, 2 when the command was refused.
    """
    args = build_parser(prog, takes_host=host_factory is None).parse_args(argv)
    try:
        factory = host_factory if host_factory is not None else _load_host_factory(args.host)
        host = factory()
        if not isinstance(host, EvalHost | LaunchHost):
            raise _Refused(f"the host factory returned a {type(host).__name__}, not an EvalHost or a LaunchHost")
        if args.command == "run":
            if not isinstance(host, LaunchHost):
                raise _Refused(
                    "run launches, so its host factory must return a LaunchHost — the EvalHost with the kinds "
                    "it can launch; this one returned an EvalHost, which ls and report can read but nothing can launch"
                )
            return asyncio.run(_launch(host, args))
        eval_host = host.eval_host if isinstance(host, LaunchHost) else host
        if args.command == "ls":
            _list(eval_host, args.scope)
        else:
            print(inspect_campaign_bundle(eval_host, args.campaign, args.scope).model_dump_json(indent=2))
        return EXIT_OK
    except (_Refused, EvalServiceError) as refused:
        print(f"{prog} {args.command}: {refused}", file=sys.stderr)
        return EXIT_REFUSED


async def _launch(host: LaunchHost, args: argparse.Namespace) -> int:
    """Launch the runs, wait for every job, and print each run's summary."""
    runs = await start_run(
        host, template_id=args.template, subject_id=args.subject, models=args.model, k_runs=args.k, scope_id=args.scope
    )
    try:
        await host.job_manager.wait_for([run.id for run in runs])
    except asyncio.CancelledError:
        # An interrupted command settles the runs it started rather than leaving them pending.
        await host.job_manager.shutdown()
        raise
    summaries = [summarize_run(host.eval_host, run.id, args.scope) for run in runs]
    for summary in summaries:
        print(summary.render())
    return EXIT_OK if all(summary.status == "completed" for summary in summaries) else EXIT_RUN_DID_NOT_COMPLETE


def _list(host: EvalHost, scope_id: str) -> None:
    """Print the scope's templates, runs and campaigns, one per line under a counted heading."""
    templates = list_templates(host.storage, scope_id)
    print(f"templates ({len(templates)})")
    for template in templates:
        print(f"  {template.id}  {template.candidate_kind}  {template.name}")
    runs = list_runs(host, scope_id)
    print(f"runs ({len(runs)})")
    for run in runs:
        print(f"  {run.id}  {run.status}  {run.candidate_model}  template {run.template_id}")
    campaigns = list_campaigns(host.storage, scope_id)
    print(f"campaigns ({len(campaigns)})")
    for campaign in campaigns:
        print(f"  {campaign.id}  {campaign.name}  {len(campaign.run_ids)} run(s)")


__all__ = ["DEFAULT_PROG", "HostFactory", "build_parser", "run_cli"]
