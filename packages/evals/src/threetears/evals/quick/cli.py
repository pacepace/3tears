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

A product mounting the commands may add its own beside them — ``run_cli(..., commands=[HostCommand(...)])``
— each parsed like the engine's (``--scope``, and ``--host`` when the host is named on the command line)
and handed the host the factory built. A name the engine already uses is refused.

A refusal — a host that cannot be loaded, a template or campaign that is not there, a launch the
engine refuses — prints its reason to stderr and exits 2, which is also argparse's code for a
malformed command line.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import sys
from collections.abc import Callable, Coroutine, Sequence
from dataclasses import dataclass
from typing import Any

from threetears.evals.analysis import inspect_campaign_bundle, list_campaigns
from threetears.evals.contracts import EvalServiceError
from threetears.evals.contracts.host import EvalHost
from threetears.evals.quick.summary import summarize_run
from threetears.evals.run import LaunchHost, list_runs, list_templates, start_run

#: What names the host the commands work in: called once, with no arguments, per invocation.
HostFactory = Callable[[], EvalHost | LaunchHost]

#: The commands the engine itself carries; a host command may take none of these names.
ENGINE_COMMANDS: tuple[str, ...] = ("run", "ls", "report")


@dataclass(frozen=True, kw_only=True)
class HostCommand:
    """A subcommand a host adds beside the engine's own, mounted by :func:`run_cli` under the same program.

    It is parsed like the engine's commands — with ``--scope`` always, and ``--host`` when the command
    line names the host rather than a product mounting it with its own factory — and its handler is
    handed the very host object the factory built for this invocation, with the parsed namespace.

    Attributes:
        name: The subcommand's name. None of :data:`ENGINE_COMMANDS`, and unique among the host's.
        help: One line, shown in the program's usage and as the command's description.
        configure: Adds the command's own arguments to its parser; ``--scope`` (and ``--host``) are
            already there.
        handler: Carries the command out and returns its exit code. It may be a coroutine function,
            which the command line runs to completion. A refusal it raises as
            :class:`~threetears.evals.contracts.EvalServiceError` is printed to stderr and exits 2, as
            the engine's commands' refusals are.
    """

    name: str
    help: str
    configure: Callable[[argparse.ArgumentParser], None]
    handler: Callable[[EvalHost | LaunchHost, argparse.Namespace], int | Coroutine[Any, Any, int]]


def _refuse_colliding_commands(commands: Sequence[HostCommand]) -> None:
    """Refuse host commands whose names would shadow an engine command or each other.

    Args:
        commands: The host's commands.

    Raises:
        ValueError: A name is an engine command's, or two host commands share one.
    """
    seen: set[str] = set()
    for host_command in commands:
        if host_command.name in ENGINE_COMMANDS:
            raise ValueError(
                f"host command {host_command.name!r} collides with the engine's own {host_command.name!r}; "
                f"a host command takes a name none of {', '.join(ENGINE_COMMANDS)} has"
            )
        if host_command.name in seen:
            raise ValueError(f"two host commands are both named {host_command.name!r}")
        seen.add(host_command.name)


#: The program name the command line prints when it is run as ``python -m threetears.evals``.
DEFAULT_PROG = "python -m threetears.evals"

#: Exit codes: every run completed (or the command read what it was asked for); a run did not
#: complete; the command was refused before it could do anything.
EXIT_OK = 0
EXIT_RUN_DID_NOT_COMPLETE = 1
EXIT_REFUSED = 2


def _say(line: str) -> None:
    """Write one line of the command's output to stdout — the CLI's product, so a stream write, not a log record."""
    sys.stdout.write(f"{line}\n")


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


def build_parser(
    prog: str = DEFAULT_PROG, *, takes_host: bool = True, commands: Sequence[HostCommand] = ()
) -> argparse.ArgumentParser:
    """The command line's parser.

    Args:
        prog: The program name usage lines print.
        takes_host: Whether each command takes ``--host``. A product that mounts the commands with its
            own host factory passes ``False``, and the option does not exist.
        commands: The host's own subcommands, added after the engine's.

    Returns:
        The parser; ``command`` on the parsed namespace names the subcommand.

    Raises:
        ValueError: A host command's name is an engine command's, or two host commands share one.
    """
    _refuse_colliding_commands(commands)
    names = [*ENGINE_COMMANDS, *(host_command.name for host_command in commands)]
    parser = argparse.ArgumentParser(prog=prog, description="Run, list and report evals.")
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="{" + ",".join(names) + "}")

    def command(name: str, help_text: str) -> argparse.ArgumentParser:
        sub = subparsers.add_parser(name, help=help_text, description=help_text)
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
    for host_command in commands:
        host_command.configure(command(host_command.name, host_command.help))
    return parser


def run_cli(
    argv: Sequence[str] | None = None,
    *,
    host_factory: HostFactory | None = None,
    prog: str = DEFAULT_PROG,
    commands: Sequence[HostCommand] = (),
) -> int:
    """Parse ``argv`` and carry out the command, printing to stdout and refusals to stderr.

    Args:
        argv: The arguments after the program name; ``None`` reads ``sys.argv``.
        host_factory: The host to work in, for a product mounting these commands; ``None`` takes it
            from ``--host``.
        prog: The program name usage lines print.
        commands: The host's own subcommands, mounted beside ``run``, ``ls`` and ``report``; each
            one's handler is handed the host the factory built.

    Returns:
        The exit code: 0, 1 when a launched run did not complete, 2 when the command was refused, or
        whatever a host command's handler returned.

    Raises:
        ValueError: A host command's name is an engine command's, or two host commands share one —
            refused before ``argv`` is parsed, since it is the mounting product's code and not the
            user's command line.
    """
    args = build_parser(prog, takes_host=host_factory is None, commands=commands).parse_args(argv)
    handlers = {host_command.name: host_command.handler for host_command in commands}
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
        if (handler := handlers.get(args.command)) is not None:
            outcome = handler(host, args)
            return asyncio.run(outcome) if isinstance(outcome, Coroutine) else outcome
        eval_host = host.eval_host if isinstance(host, LaunchHost) else host
        if args.command == "ls":
            _list(eval_host, args.scope)
        else:
            _say(inspect_campaign_bundle(eval_host, args.campaign, args.scope).model_dump_json(indent=2))
        return EXIT_OK
    except (_Refused, EvalServiceError) as refused:
        sys.stderr.write(f"{prog} {args.command}: {refused}\n")
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
        _say(summary.render())
    return EXIT_OK if all(summary.status == "completed" for summary in summaries) else EXIT_RUN_DID_NOT_COMPLETE


def _list(host: EvalHost, scope_id: str) -> None:
    """Print the scope's templates, runs and campaigns, one per line under a counted heading."""
    templates = list_templates(host.storage, scope_id)
    _say(f"templates ({len(templates)})")
    for template in templates:
        _say(f"  {template.id}  {template.candidate_kind}  {template.name}")
    runs = list_runs(host, scope_id)
    _say(f"runs ({len(runs)})")
    for run in runs:
        _say(f"  {run.id}  {run.status}  {run.candidate_model}  template {run.template_id}")
    campaigns = list_campaigns(host.storage, scope_id)
    _say(f"campaigns ({len(campaigns)})")
    for campaign in campaigns:
        _say(f"  {campaign.id}  {campaign.name}  {len(campaign.run_ids)} run(s)")


__all__ = ["DEFAULT_PROG", "ENGINE_COMMANDS", "HostCommand", "HostFactory", "build_parser", "run_cli"]
