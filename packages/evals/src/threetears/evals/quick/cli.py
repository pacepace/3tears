"""The engine's command line: launch a template's runs, list what a scope holds, and read a campaign's report.

``python -m threetears.evals`` runs it with ``--host module:factory`` naming the host to work in: a
zero-argument callable returning an :class:`~threetears.evals.kernel.host.EvalHost`, or a
:class:`~threetears.evals.run.LaunchHost` for ``run``, which launches. A product mounts the same
commands under its own CLI by calling :func:`run_cli` with its own ``host_factory``, and its users
then never name the host::

    python -m threetears.evals run --host myapp.evals:build_host --scope dev --template T --subject S --model M
    python -m threetears.evals ls --host myapp.evals:build_host --scope dev
    python -m threetears.evals report CAMPAIGN --host myapp.evals:build_host --scope dev --format html --out r.html
    python -m threetears.evals bundle CAMPAIGN --host myapp.evals:build_host --scope dev
    python -m threetears.evals spend --host myapp.evals:build_host --scope dev --purpose variation
    python -m threetears.evals gate CAMPAIGN --host myapp.evals:build_host --scope dev --fail-on regressed,breached
    python -m threetears.evals frontier --host myapp.evals:build_host --scope dev --bar 0.8

- ``run`` launches through :func:`~threetears.evals.run.start_run`, waits for the runs' jobs, and
  prints each run's summary. It exits 0 when every run completed and 1 when any did not. Each
  ``--model`` is one arm and one run; with none, the kind runs one arm on its own default model, and a
  kind with no default refuses the launch. ``--k`` is the repeats per case (the launch default when
  omitted).
  ``--max-cost-usd`` caps each run at or below the host's ceiling, as ``run_launch``'s ``max_cost_usd``
  does — it may only lower the host's ceiling, never raise it, and naming it is how a launch chooses
  the cap an unpriceable arm runs under rather than inheriting one nobody chose. ``--judge-model`` and
  ``--simulator-model`` pin the judge and the simulated user, as ``run_launch``'s ``judge_model`` and
  ``simulator_model`` do; omitted, the kind's own defaults apply. ``--n-variations`` and
  ``--variation-model`` generate the cases first, as ``run_launch``'s ``n_variations`` and
  ``variation_model`` do; the generation calls run before the runs and are outside their cost cap,
  so they are priced against the host's out-of-run cap before they are made. ``--apparatus-settings``
  sets host-declared apparatus values as a JSON object, as ``run_launch``'s ``apparatus_settings`` does.
- ``ls`` prints the scope's templates, runs and campaigns.
- ``report`` prints the campaign's report (:func:`~threetears.evals.analysis.campaign_report`, the same
  report the ``report_read`` action reads): its newest analysis that is not archived, else a code-only report of its
  evidence — which says, in its first lines, that no analysis was generated. ``--format`` is
  ``markdown`` (the default), ``html`` (needs no script) or ``json`` (what the published schema
  validates); ``--out PATH`` writes it to a file instead of stdout. No model is called.
- ``bundle`` prints the campaign's analysis bundle as JSON
  (:func:`~threetears.evals.analysis.inspect_campaign_bundle`): what a generation would read, assembled
  without calling any model.
- ``gate`` is the CI gate: it reads the campaign's typed verdicts as code reaches them on its evidence now
  (:func:`~threetears.evals.analysis.campaign_verdicts`, never an analysis's words), prints what it came to and
  every verdict that failed it or was not decided, and exits 4 when a verdict ``--fail-on`` names occurred.
  ``--fail-on`` takes a comma-separated list of ``regressed``, ``not-separated``, ``untested``, ``breached``,
  ``undecided-guardrail``, ``missed`` and ``undecided-bar``; the default is ``regressed,breached,undecided-guardrail``
  (:data:`~threetears.evals.analysis.DEFAULT_FAIL_ON`): a regression, a breached guardrail, and a guardrail not shown
  held, since one not known to be safe must not ship as if it were. A verdict left undecided that ``--fail-on``
  does not name exits 0 and is printed as undecided, never as passed. ``--reading`` (repeatable) gates only those
  readings. A campaign with no verdict at all is refused (exit 2): a gate that read nothing passed nothing.
- ``spend`` prints what the engine spent outside any run in the scope — case generations, rubric
  proposals and analysis generations, call by call, with totals overall, per purpose and per launch
  (:func:`~threetears.evals.analysis.scope_out_of_run_spend`, the read the ``scope_out_of_run_spend`` action
  makes). ``--purpose``, ``--launch-group`` and ``--template`` narrow it.
- ``frontier`` prints each subject's variants ranked on quality, cost and latency, and the cheapest that clears
  ``--bar`` (:func:`~threetears.evals.analysis.frontier` over the host's run listing, the read the
  ``scope_frontier`` action makes).
  ``--subject`` narrows it to one subject, and ``--json`` prints the frontier as JSON.

A product mounting the commands may add its own beside them — ``run_cli(..., commands=[HostCommand(...)])``
— each parsed like the engine's (``--scope``, and ``--host`` when the host is named on the command line)
and handed the host the factory built. A name the engine already uses is refused.

A refusal — a host that cannot be loaded, a template or campaign that is not there, a launch the
engine refuses — prints its reason to stderr and exits 2, which is also argparse's code for a
malformed command line.

Any other exception — a host factory, a kind's launcher or a host command's handler raising something
the engine did not anticipate, or the engine failing itself — prints its traceback to stderr and exits
3, never 1: a script branching on the exit code must not read a broken host as runs that finished and
did not complete.

Exit codes, in full: 0 done; 1 a launched run did not complete; 2 refused; 3 failed with an
unanticipated error; 4 the gate failed.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import sys
import traceback
from functools import partial
from collections.abc import Callable, Coroutine, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, get_args

from threetears.evals.analysis import (
    DEFAULT_FAIL_ON,
    GATE_TOKENS,
    FrontierResult,
    ReportFormat,
    campaign_report,
    campaign_verdicts,
    frontier,
    frontier_text,
    gate_verdicts,
    inspect_campaign_bundle,
    list_campaigns,
    out_of_run_spend_text,
    parse_fail_on,
    scope_out_of_run_spend,
    serialize_report,
)
from threetears.evals.schema import DEFAULT_LAUNCH_K_RUNS, OutOfRunPurpose
from threetears.evals.kernel import EvalServiceError
from threetears.evals.kernel.host import EvalHost
from threetears.evals.analysis.summary import summarize_run
from threetears.evals.run import LaunchHost, list_runs, list_templates, start_run

#: What names the host the commands work in: called once, with no arguments, per invocation.
HostFactory = Callable[[], EvalHost | LaunchHost]

#: The commands the engine itself carries; a host command may take none of these names.
ENGINE_COMMANDS: tuple[str, ...] = ("run", "ls", "report", "bundle", "spend", "gate", "frontier")


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
            :class:`~threetears.evals.kernel.EvalServiceError` is printed to stderr and exits 2, as
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

#: Exit code: every run completed, or the command read what it was asked for.
EXIT_OK = 0
#: Exit code: a launched run did not complete.
EXIT_RUN_DID_NOT_COMPLETE = 1
#: Exit code: the command was refused before it could do anything.
EXIT_REFUSED = 2
#: Exit code: the command failed on an error nothing anticipated — a host factory, a launcher or a handler
#: raising, or the engine's own fault. Never 1, so a broken host cannot read as runs that finished.
EXIT_FAILED = 3
#: Exit code: ``gate`` read a verdict its ``--fail-on`` names. Never 1 or 3, so a CI step tells a failed gate from
#: runs that did not complete and from a broken host.
EXIT_GATE_FAILED = 4


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
    run.add_argument(
        "--model",
        action="append",
        default=[],
        help=(
            "a candidate model; repeat for one arm each. Omitted, the kind runs one arm on its own default "
            "model, and a kind with no default refuses the launch"
        ),
    )
    run.add_argument(
        "--k",
        type=int,
        default=DEFAULT_LAUNCH_K_RUNS,
        help=f"repeats per case (default {DEFAULT_LAUNCH_K_RUNS})",
    )
    run.add_argument(
        "--max-cost-usd",
        type=float,
        default=None,
        metavar="DOLLARS",
        help="a per-run cost cap in dollars, at or below the host's ceiling (as run_launch's max_cost_usd)",
    )
    run.add_argument(
        "--judge-model",
        default=None,
        metavar="MODEL",
        help="the judge model, where the kind is model-judged (as run_launch's judge_model)",
    )
    run.add_argument(
        "--simulator-model",
        default=None,
        metavar="MODEL",
        help="the simulated user's model, where the kind has one (as run_launch's simulator_model)",
    )
    run.add_argument(
        "--n-variations",
        type=int,
        default=0,
        metavar="N",
        help="generate N new cases from the template's variation axes first (as run_launch's n_variations)",
    )
    run.add_argument(
        "--variation-model",
        default=None,
        metavar="MODEL",
        help="the model that writes the template's llm axes' values when generating (as run_launch's variation_model)",
    )
    run.add_argument(
        "--apparatus-settings",
        type=_json_object,
        default=None,
        metavar="JSON",
        help=(
            "host-declared apparatus values to set the runs' rig up with, as a JSON object keyed by apparatus "
            'dimension, e.g. \'{"adjudicator_seat": "model:m"}\' (as run_launch\'s apparatus_settings)'
        ),
    )
    run.add_argument(
        "--measure-latency",
        action="store_true",
        help="latency is under test: run each run's cells one at a time (as run_launch's measure_latency)",
    )
    command("ls", "List the scope's templates, runs and campaigns.")
    report = command(
        "report", "Print a campaign's report — its analysis, else its evidence alone — without calling a model."
    )
    report.add_argument("campaign", help="the campaign, by id")
    report.add_argument(
        "--format",
        choices=get_args(ReportFormat),
        default="markdown",
        help="markdown (default), html (needs no script) or json (the published schema's form)",
    )
    report.add_argument("--out", type=Path, metavar="PATH", help="write the report to PATH instead of stdout")
    bundle = command("bundle", "Print a campaign's analysis bundle as JSON — what a generation would read.")
    bundle.add_argument("campaign", help="the campaign, by id")
    spend = command(
        "spend", "Print what the engine spent outside any run — case generations, rubric proposals, analyses."
    )
    spend.add_argument("--purpose", choices=get_args(OutOfRunPurpose), default=None, help="only this purpose's calls")
    spend.add_argument("--launch-group", default=None, metavar="ID", help="only one launch's case generation")
    spend.add_argument("--template", default=None, metavar="ID", help="only calls made for this template")
    gate = command(
        "gate",
        "Exit 4 when the campaign's verdicts include an outcome --fail-on names — the CI gate. No model is called.",
    )
    gate.add_argument("campaign", help="the campaign, by id")
    gate.add_argument(
        "--fail-on",
        type=_fail_on,
        default=DEFAULT_FAIL_ON,
        metavar="OUTCOMES",
        help=(
            f"comma-separated outcomes that fail the gate, of {', '.join(GATE_TOKENS)} "
            f"(default {','.join(DEFAULT_FAIL_ON)})"
        ),
    )
    gate.add_argument(
        "--reading",
        action="append",
        default=None,
        metavar="MEASURE",
        help="gate only this reading, by key or as the report heads it; repeat for more (default: every reading)",
    )
    ranked = command(
        "frontier", "Rank each subject's variants on quality, cost and latency, and the cheapest that clears a bar."
    )
    ranked.add_argument("--bar", type=float, default=None, help="the pass^k a variant must clear, from 0 to 1")
    ranked.add_argument("--subject", default=None, help="only this subject's variants")
    ranked.add_argument("--json", action="store_true", help="print the frontier as JSON")
    for host_command in commands:
        host_command.configure(command(host_command.name, host_command.help))
    return parser


def _fail_on(text: str) -> tuple[str, ...]:
    """Parse ``--fail-on``, refusing an outcome no gate fails on so argparse names the argument.

    Raises:
        argparse.ArgumentTypeError: A name that is no outcome, or none at all.
    """
    try:
        return parse_fail_on(text)
    except ValueError as unknown:
        raise argparse.ArgumentTypeError(str(unknown)) from unknown


def _json_object(text: str) -> dict[str, Any]:
    """Parse a command-line JSON object, refusing anything else so argparse names the argument.

    Raises:
        argparse.ArgumentTypeError: ``text`` is not JSON, or is JSON but not an object.
    """
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"not JSON: {e}") from e
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError(f"a JSON object is expected, got a {type(parsed).__name__}")
    return parsed


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
        commands: The host's own subcommands, mounted beside the engine's (:data:`ENGINE_COMMANDS`); each
            one's handler is handed the host the factory built.

    Returns:
        The exit code: 0, 1 when a launched run did not complete, 2 when the command was refused, 3 when
        it failed on an error nothing anticipated (its traceback is printed to stderr), 4 when ``gate`` read an
        outcome its ``--fail-on`` names, or whatever a host command's handler returned.

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
                    "it can launch; this one returned an EvalHost, which ls, report, bundle, spend and gate can read but nothing can launch"
                )
            return asyncio.run(_launch(host, args))
        if (handler := handlers.get(args.command)) is not None:
            outcome = handler(host, args)
            return asyncio.run(outcome) if isinstance(outcome, Coroutine) else outcome
        eval_host = host.eval_host if isinstance(host, LaunchHost) else host
        if args.command == "gate":
            return _gate(eval_host, args)
        if args.command == "ls":
            _list(eval_host, args.scope)
        elif args.command == "report":
            _report(eval_host, args)
        elif args.command == "spend":
            _say(
                out_of_run_spend_text(
                    scope_out_of_run_spend(
                        eval_host,
                        args.scope,
                        purpose=args.purpose,
                        launch_group_id=args.launch_group,
                        template_id=args.template,
                    )
                )
            )
        elif args.command == "frontier":
            ranked = FrontierResult.model_validate(
                frontier(
                    eval_host.storage,
                    args.scope,
                    list_runs=partial(list_runs, eval_host),
                    bar=args.bar,
                    subject_id=args.subject,
                    profile=eval_host.profile,
                )
            )
            _say(ranked.model_dump_json(indent=2) if args.json else frontier_text(ranked))
        else:
            _say(inspect_campaign_bundle(eval_host, args.campaign, args.scope).model_dump_json(indent=2))
        return EXIT_OK
    except (_Refused, EvalServiceError) as refused:
        sys.stderr.write(f"{prog} {args.command}: {refused}\n")
        return EXIT_REFUSED
    except Exception:
        # NOSILENT: printed whole to stderr; the distinct exit code is the point — uncaught, Python exits 1,
        # which is the code for runs that finished and did not complete.
        sys.stderr.write(f"{prog} {args.command}: failed on an unanticipated error\n{traceback.format_exc()}")
        return EXIT_FAILED


async def _launch(host: LaunchHost, args: argparse.Namespace) -> int:
    """Launch the runs, wait for every job, and print each run's summary."""
    runs = await start_run(
        host,
        template_id=args.template,
        subject_id=args.subject,
        models=args.model,
        k_runs=args.k,
        scope_id=args.scope,
        max_cost_usd=args.max_cost_usd,
        judge_model=args.judge_model,
        simulator_model=args.simulator_model,
        n_variations=args.n_variations,
        variation_model=args.variation_model,
        apparatus_settings=args.apparatus_settings,
        measure_latency=args.measure_latency,
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


def _gate(host: EvalHost, args: argparse.Namespace) -> int:
    """Gate on the campaign's typed verdicts: print the result, and exit 4 when a named outcome occurred.

    Raises:
        _Refused: The campaign has no verdict to gate on, or a ``--reading`` no verdict is on.
    """
    verdicts = campaign_verdicts(host, args.campaign, args.scope)
    if not verdicts:
        raise _Refused(
            f"campaign {args.campaign} has no verdict to gate on: no contrast against a control, no guardrail and no "
            "bar was read, so a gate would pass nothing it checked. Designate a control (set_campaign_control) "
            "or declare a bar or a guardrail"
        )
    try:
        result = gate_verdicts(verdicts, fail_on=args.fail_on, readings=args.reading)
    except ValueError as unknown:
        raise _Refused(str(unknown)) from unknown
    _say(result.render())
    return EXIT_GATE_FAILED if result.failed else EXIT_OK


def _report(host: EvalHost, args: argparse.Namespace) -> None:
    """Print the campaign's report in the asked-for form, or write it to ``--out``.

    Raises:
        _Refused: ``--out`` cannot be written.
    """
    report = campaign_report(host, args.campaign, args.scope)
    # Markdown and HTML end in a newline already; canonical JSON is the model's own dump, which does not,
    # and a terminal or a file wants one.
    body = serialize_report(report, args.format) + ("\n" if args.format == "json" else "")
    if args.out is None:
        sys.stdout.write(body)
        return
    try:
        args.out.write_text(body, encoding="utf-8")
    except OSError as unwritable:
        raise _Refused(f"--out {args.out}: cannot write the report there: {unwritable}") from unwritable
    _say(f"wrote the {report.basis.replace('_', '-')} report of campaign {args.campaign} ({args.format}) to {args.out}")


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


__all__ = [
    "DEFAULT_PROG",
    "ENGINE_COMMANDS",
    "EXIT_FAILED",
    "EXIT_GATE_FAILED",
    "EXIT_OK",
    "EXIT_REFUSED",
    "EXIT_RUN_DID_NOT_COMPLETE",
    "HostCommand",
    "HostFactory",
    "build_parser",
    "run_cli",
]
