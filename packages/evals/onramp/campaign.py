"""The agent test as an eval on the package: cases are apps, arms are docs builds, the runner is an adopting agent.

``python -m packages.evals.onramp`` (``scripts/onramp-agent.sh`` builds today's wheel and runs it) measures one or
more docs builds. ``--scripted`` runs the harness offline with the fixed transcripts and spends nothing. A live run
spends money, so it is refused unless ``ONRAMP_LIVE=1`` is set and ``--cap`` states the most it may spend; it quotes
its cost before anything runs, and holds each run to its share of the cap.

Every command an agent's run causes, and every re-run of its work, runs in a trial container with no network
(:mod:`.isolation`). A run that cannot have one (no Docker, no image) is refused before anything is sent: there is no
fallback to this machine. Only ``--scripted --host`` runs commands here, and only a fixed transcript ever does.

Models follow the project's rules: the cheap tier first, through OpenRouter (``OPENROUTER_API_KEY``); Claude Haiku 5.5
as the ceiling, only ever direct to Anthropic on ``ANTHROPIC_API_KEY``, never through OpenRouter, and no model is
sent a temperature.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import math
import os
import re
import sys
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from threetears.evals import Eval, Model, Result

from .agent import (
    AGENT,
    CANNOT_TELL,
    CAUSES,
    HARNESS,
    REPLY_TOKENS,
    Brain,
    Caps,
    ModelBrain,
    ScriptedBrain,
    attempt,
    commands,
    errors_hit,
    format_error_turns,
    parse_report,
    scored_door_turns,
    stuck_point,
)
from .checks import GRADERS, regrade
from .isolation import IsolationUnavailable, unavailable
from .plan import Plan, build_wheel, built_wheel, commit_of, committed_at, load_plan, plan_refusal
from .reading import PRIMARY, Unreadable, power_lines, reading_lines
from .sandbox import HERE, DocsBuild, Sandbox

CASES = HERE / "apps.jsonl"
TRANSCRIPTS = HERE / "transcripts"
#: The commit today's wheel, the control arm, is built from: the docs and messages before any on-ramp fix.
CONTROL_REF = "f9ab6f74d"
#: Set to 1 to let a run spend money.
LIVE_ENV = "ONRAMP_LIVE"
#: The turn profile an expected cost assumes: turns a run, prompt tokens a turn, output tokens a turn.
PROFILE = (25, 15_000, 1_500)
#: The engine's budget for one cell (one agent run and its regrade), in seconds: ``DEFAULT_CELL_TIMEOUT_S`` in
#: ``threetears.evals.run.runner``. ``Eval`` takes no other, so a run's wall cap and its regrade's fit inside it.
CELL_BUDGET_S = 600.0
#: Key variables whose fallbacks would let one call be resent on a second key after a drop, past the bound a run holds.
FALLBACK_VARIABLES = ("OPENROUTER_API_KEY_FALLBACKS", "ANTHROPIC_API_KEY_FALLBACKS")

Runner = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]  # Any: a case, and a trial's record


@dataclass(frozen=True)
class Tier:
    """A model tier: its model, where it goes, the most one run may spend, and the rates a quote assumes.

    :raises ValueError: A Claude model routed anywhere but direct to Anthropic.
    """

    name: str
    model_id: str
    provider: str | None
    per_run_usd: float
    input_per_m: float
    output_per_m: float
    rates_from: str

    def __post_init__(self) -> None:
        claude = re.search(r"claude|anthropic", self.model_id, re.IGNORECASE) is not None
        if claude and (self.provider != "anthropic" or "/" in self.model_id):
            raise ValueError(f"{self.model_id} is a Claude model: it goes direct to Anthropic, never via OpenRouter")

    def model(self, recording: Path | None, **client: Any) -> Model:  # Any: Model's client options (http=, ...)
        """The tier's model, recording every call to ``recording``; never handed a temperature."""
        if self.provider == "anthropic":
            return Model(self.model_id, provider="anthropic", max_tokens=REPLY_TOKENS, recording=recording, **client)
        return Model(self.model_id, max_tokens=REPLY_TOKENS, effort="low", recording=recording, **client)


TIERS: Mapping[str, Tier] = {
    tier.name: tier
    for tier in (
        Tier(
            "cheap",
            "deepseek/deepseek-v4-flash",
            None,
            0.50,
            0.12,
            0.18,
            "fitted to the OpenRouter-reported costs of the repo's deepseek-v4-flash recordings",
        ),
        Tier(
            "cheap-alt",
            "google/gemini-2.5-flash-lite",
            None,
            0.50,
            0.10,
            0.40,
            "fitted to the OpenRouter-reported costs of examples/2-replies/recordings/judge.jsonl",
        ),
        Tier("ceiling", "claude-haiku-5-5", "anthropic", 1.50, 0.10, 0.50, "Anthropic's published rates"),
    )
}


@dataclass(frozen=True)
class Quote:
    """What a campaign may spend, said before it runs.

    :ivar runs: Agent runs: arms x apps x repeats.
    :ivar per_run_usd: The most one run may spend: the tier's cap, or its share of the campaign's if smaller.
    :ivar bound_usd: The most the campaign can spend: every run at its cap.
    :ivar expected_usd: What it is expected to spend at :data:`PROFILE` and the tier's rates.
    """

    tier: Tier
    runs: int
    per_run_usd: float
    bound_usd: float
    expected_usd: float

    def lines(self) -> list[str]:
        turns, prompt_tokens, output_tokens = PROFILE
        return [
            f"model: {self.tier.model_id} ({self.tier.name} tier)",
            f"runs: {self.runs}, each capped at ${self.per_run_usd:.2f}",
            f"at most: ${self.bound_usd:.2f}",
            f"expected: ${self.expected_usd:.2f} ({turns} turns a run, {prompt_tokens} prompt and {output_tokens} "
            f"output tokens a turn, at ${self.tier.input_per_m}/${self.tier.output_per_m} per million: "
            f"{self.tier.rates_from})",
        ]


def quote(tier: Tier, *, runs: int, cap_usd: float | None) -> Quote:
    """What ``runs`` runs on ``tier`` may spend, held to ``cap_usd`` when one is stated."""
    per_run = tier.per_run_usd if cap_usd is None else min(tier.per_run_usd, cap_usd / runs)
    turns, prompt_tokens, output_tokens = PROFILE
    each = turns * (prompt_tokens * tier.input_per_m + output_tokens * tier.output_per_m) / 1_000_000
    return Quote(tier, runs, per_run, per_run * runs, min(each, per_run) * runs)


def live_refusal(environment: Mapping[str, str], cap_usd: float | None) -> str | None:
    """Why a live run may not start, or ``None`` when it may. Reads which variables are set, never a value."""
    if environment.get(LIVE_ENV) != "1":
        return f"a live run spends money: set {LIVE_ENV}=1 to say it may"
    if cap_usd is None or cap_usd <= 0:
        return "a live run states the most it may spend: pass --cap DOLLARS"
    if fallbacks := [name for name in FALLBACK_VARIABLES if name in environment]:
        return f"a live run sends on one key, so its cap bounds what it is billed: unset {', '.join(fallbacks)}"
    return None


def isolation_refusal() -> str | None:
    """Why no agent run may start here, or ``None``: every agent-driven command runs in a trial container."""
    if (why := unavailable()) is not None:
        return f"an agent's commands run only in a trial container, and none can run here: {why}"
    return None


def slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-")


def adopter(build: DocsBuild, brains: Callable[[Sandbox], Brain], *, workdir: Path, caps: Caps) -> Runner:
    """The runner of one arm: an agent adds an eval to the case's app against ``build``, and the result is checked."""
    numbers = itertools.count(1)

    async def adopt(case: dict[str, Any]) -> dict[str, Any]:  # Any: a case, and the trial's record
        """An agent adds an eval to an app it has never seen, from the docs and the error messages alone."""
        app = case["app"]
        trial = workdir / slug(build.name) / f"{app}-{next(numbers)}"
        sandbox = Sandbox.create(trial / "app", app, build)
        brain = brains(sandbox)
        # Each step lands in steps.jsonl as it happens: a run the cell's budget cuts off keeps what it did there.
        run = await attempt(sandbox, brain, caps, log=trial / "steps.jsonl")
        report, problem = parse_report(run.report)
        ran = commands(run.steps)
        guide = build.llms_txt.read_text(encoding="utf-8") if build.llms_txt.is_file() else ""
        record = {
            "app": app,
            "arm": build.name,
            "build": build.as_record(),
            "model": brain.model,
            "stopped": run.stopped,
            "wall_s": run.wall_s,
            "turns": len(run.usages),
            "input_tokens": run.tokens("input_tokens"),
            "output_tokens": run.tokens("output_tokens"),
            "cost_usd": run.spent(),
            "charged_usd": run.charged(),
            "maybe_billed_sends": run.maybe_billed_sends(),
            "commands": ran,
            "errors_hit": errors_hit(run.steps),
            "pages_read": sandbox.pages,
            "written": sandbox.written,
            "stuck": stuck_point(run.steps, guide),
            "format_error_turns": format_error_turns(run.steps),
            "scored_door_turns": scored_door_turns(run.steps),
            "report": report,
            "report_from": run.report_from,
            "report_problem": problem,
            "sandbox": str(sandbox.root),
            "checks": await regrade(sandbox, trial, report, ran),
        }
        (trial / "record.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
        return record

    return adopt


def scripted(transcript: str | None = None) -> Callable[[Sandbox], Brain]:
    """Brains that follow ``transcripts/<transcript>.jsonl``, or each app's own (``transcripts/<app>.jsonl``)."""
    return lambda sandbox: ScriptedBrain(TRANSCRIPTS / f"{transcript or sandbox.app}.jsonl", sandbox)


def onramp_eval(arms: Mapping[str, Runner], *, k: int, max_cost_usd: float | None, apps: Sequence[str] = ()) -> Eval:
    """The agent test: every app, every arm, ``k`` times, graded by :data:`~.checks.GRADERS`."""
    cases = [json.loads(line) for line in CASES.read_text(encoding="utf-8").splitlines() if line.strip()]
    return Eval(
        [case for case in cases if not apps or case["app"] in apps],
        runner=dict(arms),
        grader=GRADERS,
        k=k,
        name="onramp",
        max_cost_usd=max_cost_usd,
        question=(
            "Does the candidate docs build let an agent add a working eval more often?" if len(arms) > 1 else None
        ),
    )


def trials(result: Result) -> list[dict[str, Any]]:  # Any: each trial's stored record and grades
    """Every trial as the eval's store holds it: the arm, the app it ran on (from its case, so a trial the run lost
    still names it), the agent's record, each grade, and each failed check's :func:`failure_causes`."""
    rows = [
        {
            "arm": arm,
            "app": trial.input.get("app") if isinstance(trial.input, dict) else None,
            "repeat": trial.repeat,
            "state": trial.state,
            "scores": trial.scores,
            "record": trial.answer,
        }
        for arm in result.arms
        for trial in result.results(arm)
    ]
    return [{**row, "causes": failure_causes(row)} for row in rows]


def failure_causes(row: Mapping[str, Any]) -> dict[str, str]:  # Any: a trial's stored row
    """What caused each check a trial failed, as far as the harness can tell mechanically (:data:`~.agent.CAUSES`).

    The harness's: every check of a run the wall stopped after a door run of it had scored (it had a working eval and
    no time to finish), or the dollar cap or a failed model call stopped; every check of a regrade that crashed or ran
    past its budget on an eval the door ran; and the reading of a run that gave no report while a reply of its, after
    its last door run, was refused as no tool call. The agent's: every check of a run that sent done with no eval file
    in its directory (a made-up report, or none). Anything else cannot be told without reading the run.
    """
    record = row.get("record")
    record = record if isinstance(record, Mapping) else {}
    failed = [grader.__name__ for grader in GRADERS if row["scores"].get(grader.__name__, 1.0) != 1.0]
    checks = record.get("checks") if isinstance(record.get("checks"), Mapping) else {}
    stopped = record.get("stopped")
    if stopped in ("cap", "error") or (stopped == "wall" and record.get("scored_door_turns")):
        return dict.fromkeys(failed, HARNESS)
    if stopped == "done" and str(checks.get("problem", "")).startswith(
        ("the agent named no eval", f"{checks.get('eval_file')} is not")
    ):
        return dict.fromkeys(failed, AGENT)
    crashed = any(
        str(reading.get("refused", "")).startswith("Traceback")
        for reading in (checks.get("app"), checks.get("twin"))
        if isinstance(reading, Mapping)
    )
    if "ran past its" in str(checks.get("problem", "")) or (crashed and checks.get("door_exit") in (0, 4)):
        return dict.fromkeys(failed, HARNESS)
    doors = [command["turn"] for command in record.get("commands", []) if command["door"]]
    late_format = [turn for turn in record.get("format_error_turns", []) if turn > max(doors, default=0)]
    lost_report = record.get("report") is None and bool(late_format)
    return {name: HARNESS if name == "read_right" and lost_report else CANNOT_TELL for name in failed}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="onramp-agent", description=__doc__.splitlines()[0])
    parser.add_argument(
        "--arm", action="append", default=[], metavar="NAME=WHEEL", help="a docs build; the first is the control"
    )
    parser.add_argument(
        "--scripted",
        choices=("succeeds", "stuck"),
        help="follow the fixed transcripts instead of asking a model: offline, and free",
    )
    parser.add_argument(
        "--host",
        action="store_true",
        help="with --scripted only: run the fixed transcripts' commands on this machine, not in a trial container",
    )
    parser.add_argument("--tier", choices=sorted(TIERS), default=None, help="the model tier (default: cheap)")
    parser.add_argument("--k", type=int, default=None, help=f"trials per arm on each app (default: {DEFAULT_K})")
    parser.add_argument("--cap", type=float, default=None, metavar="DOLLARS", help="the most a live run may spend")
    parser.add_argument("--apps", default="", help="comma-separated; every app when empty")
    parser.add_argument("--workdir", type=Path, default=HERE / "runs")
    parser.add_argument(
        "--recordings",
        type=Path,
        default=None,
        metavar="DIR",
        help="where each arm's model calls are recorded (<arm>.jsonl); when every arm's is there, they are replayed",
    )
    parser.add_argument("--dry-run", action="store_true", help="say what would run and what it may cost; run nothing")
    parser.add_argument(
        "--regrade",
        type=Path,
        default=None,
        metavar="RUN_DIR",
        help="grade a stored run again with today's harness, in trial containers: no model call, no agent run",
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=None,
        metavar="PLAN",
        help="run a committed plan (plans/*.toml): its arms, apps, k and tiers; refused if anything here disagrees",
    )
    parser.add_argument(
        "--read",
        type=Path,
        default=None,
        metavar="RUN_DIR",
        help="read a stored run by the package, apps as cases, tier by tier: no model call, nothing run",
    )
    parser.add_argument(
        "--power",
        nargs=3,
        type=float,
        default=None,
        metavar=("APPS", "K", "EFFECT"),
        help="how often the package's reading of a plan this size separates EFFECT, simulated: nothing run",
    )
    parser.add_argument("--spread", type=float, default=0.1, help="with --power: the arm-by-app noise, as an sd")
    parser.add_argument("--reps", type=int, default=200, help="with --power: simulated runs")
    parser.add_argument("--seed", type=int, default=1, help="with --power: the simulation's seed")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """The agent test's command line; the exit code is the eval door's, and 2 for a run refused before it began."""
    parser = _parser()
    args = parser.parse_args(argv)
    if args.regrade is not None:
        if (refusal := isolation_refusal()) is not None:
            print(f"onramp-agent: {refusal}", file=sys.stderr)
            return 2
        from .rescore import rescore  # it reads this module's records; imported here, so neither imports the other

        print("\n".join(rescore(args.regrade)))
        return 0
    if args.read is not None:
        try:
            print("\n".join(read_stored(args.read)))
        except Unreadable as unreadable:
            print(f"onramp-agent: {unreadable}", file=sys.stderr)
            return 2
        return 0
    if args.power is not None:
        apps_n, k_n, effect = args.power
        lines = power_lines(
            apps=int(apps_n), k=int(k_n), effect=effect, spread=args.spread, reps=args.reps, seed=args.seed
        )
        print("\n".join(lines))
        return 0
    if args.host and not args.scripted:
        parser.error("--host runs commands on this machine, which only the scripted agent may do: add --scripted")
    if args.plan is not None and args.scripted:
        parser.error("--plan states a live comparison; --scripted runs the fixed transcripts, which no plan names")
    plan: Plan | None = None
    if args.plan is not None:
        try:
            plan = load_plan(args.plan)
        except (OSError, ValueError) as unread:
            print(f"onramp-agent: {unread}", file=sys.stderr)
            return 2
    apps = [app for app in args.apps.split(",") if app]
    if plan is not None:
        apps, k, tiers = list(plan.apps), plan.k, list(plan.tiers)
    else:
        k, tiers = args.k or DEFAULT_K, [args.tier or "cheap"]
    cases = len(apps) if apps else len(_apps())
    run_dir = args.workdir / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    given = dict(arm.split("=", 1) for arm in args.arm)
    wheels = _planned_wheels(plan, given, args.workdir) if plan is not None else given
    print("\n".join(f"arm {name}: {wheel}" for name, wheel in wheels.items()) or "arm docs@here: this interpreter")
    if args.scripted:
        if args.dry_run:
            return 0
        brains = scripted(None if args.scripted == "succeeds" else "stuck")
        built: list[DocsBuild] | str
        if args.host:
            built = _host_builds(wheels, args.workdir) or [DocsBuild.here("docs@here")]
        elif not wheels:
            built = "a run in trial containers names its wheel (--arm NAME=WHEEL); --host runs this interpreter's"
        else:
            built = _isolated_builds(wheels, args.workdir)
        if isinstance(built, str):
            print(f"onramp-agent: {built}", file=sys.stderr)
            return 2
        arms = {build.name: adopter(build, brains, workdir=run_dir, caps=Caps()) for build in built}
        return _finish(onramp_eval(arms, k=k, max_cost_usd=None, apps=apps), run_dir)
    if not wheels:
        print("onramp-agent: name a docs build (--arm NAME=WHEEL), or run --scripted", file=sys.stderr)
        return 2
    priced = {
        name: quote(TIERS[name], runs=len(wheels) * cases * k, cap_usd=plan.tiers[name] if plan else args.cap)
        for name in tiers
    }
    for name, each in priced.items():
        print(
            "\n".join(each.lines() if len(tiers) == 1 else [f"tier {name}:", *(f"  {line}" for line in each.lines())])
        )
    if plan is not None:
        print("\n".join(_plan_lines(plan, priced)))
    base = args.recordings or run_dir / "recordings"
    recordings = {name: base / name if len(tiers) > 1 else base for name in tiers}
    # A replay serves only calls it holds and refuses any other by name: it never sends one, so it spends nothing.
    # RECORD=1 records the files again, live, so it is no replay.
    replay = os.environ.get("RECORD") != "1" and all(
        (recordings[name] / f"{slug(arm)}.jsonl").is_file() for name in tiers for arm in wheels
    )
    if replay:
        print(f"replays {base}: nothing is sent, and nothing spent")
    if args.dry_run:
        return 0
    if not replay and (refusal := live_refusal(os.environ, args.cap)) is not None:
        print(f"onramp-agent: {refusal}", file=sys.stderr)
        return 2
    if plan is not None:
        cap = plan.cap_usd if replay else args.cap
        refusal = plan_refusal(
            plan,
            commit=committed_at(plan.path),
            wheels=given,
            apps=[app for app in args.apps.split(",") if app],
            k=args.k,
            tier=args.tier,
            cap_usd=cap,
            known_apps=_apps(),
            workdir=args.workdir,
        )
        if refusal is not None:
            print(f"onramp-agent: {refusal}", file=sys.stderr)
            return 2
    # A replayed run's commands are a model's too: they run in trial containers like a live run's, or not at all.
    if (refusal := isolation_refusal()) is not None:
        print(f"onramp-agent: {refusal}", file=sys.stderr)
        return 2
    if plan is not None:
        wheels = {arm.name: str(build_wheel(args.workdir, arm.ref)) for arm in plan.arms}
    built = _isolated_builds(wheels, args.workdir)
    if isinstance(built, str):
        print(f"onramp-agent: {built}", file=sys.stderr)
        return 2
    if (refusal := preflight(built, run_dir / "preflight")) is not None:
        print(f"onramp-agent: {refusal}", file=sys.stderr)
        return 2
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(
        json.dumps(_run_record(plan, wheels, apps, k, priced), indent=1), encoding="utf-8"
    )
    exit_codes = []
    for name in tiers:
        tier = TIERS[name]
        arms: dict[str, Runner] = {}
        for build in built:
            try:
                recording = recordings[name] / f"{slug(build.name)}.jsonl"
                brain = ModelBrain(lambda meter, recording=recording, tier=tier: tier.model(recording, transport=meter))
            except ValueError as unbuilt:  # a missing key, named
                print(f"onramp-agent: {unbuilt}", file=sys.stderr)
                return 2
            caps = Caps(cost_usd=priced[name].per_run_usd)
            tier_dir = run_dir / name if len(tiers) > 1 else run_dir
            arms[build.name] = adopter(build, lambda _sandbox, brain=brain: brain, workdir=tier_dir, caps=caps)
        ev = onramp_eval(arms, k=k, max_cost_usd=plan.tiers[name] if plan else args.cap, apps=apps)
        exit_codes.append(_finish(ev, run_dir / name if len(tiers) > 1 else run_dir, tier=name, plan=plan))
    return next((code for code in exit_codes if code), 0)


#: Trials per arm on each app when neither the command line nor a plan says.
DEFAULT_K = 3


def _apps() -> list[str]:
    """Every app the harness has, in ``apps.jsonl``'s order."""
    return [json.loads(line)["app"] for line in CASES.read_text(encoding="utf-8").splitlines() if line.strip()]


def _planned_wheels(plan: Plan, given: Mapping[str, str], workdir: Path) -> dict[str, str]:
    """Each of the plan's arms, in its order, with the wheel it will run: one given for it on the command line (checked
    against the plan before the run), else the one already built from its commit, else where that will be built.
    An arm given on the command line that the plan does not name is kept, so the refusal can name it."""
    wheels = {
        arm.name: given.get(arm.name) or str(built_wheel(workdir, arm.ref) or f"<built from {arm.ref} at the run>")
        for arm in plan.arms
    }
    return {**wheels, **{name: wheel for name, wheel in given.items() if name not in wheels}}


def _plan_lines(plan: Plan, priced: Mapping[str, Quote]) -> list[str]:
    """What a plan's dry run adds: its total bound, whether it is committed, and what a live run still needs."""
    commit = committed_at(plan.path)
    lines = [
        f"plan {plan.name} ({plan.path.name}): {len(plan.arms)} arms x {len(plan.apps)} apps x k={plan.k} x "
        f"{len(plan.tiers)} tier(s); at most ${sum(q.bound_usd for q in priced.values()):.2f} in all, expected "
        f"${sum(q.expected_usd for q in priced.values()):.2f}",
        f"plan committed at: {commit or 'not committed, or changed since: a live run on it is refused'}",
    ]
    if placeholders := plan.placeholders():
        lines.append(f"plan apps still placeholders: {len(placeholders)}: a live run on it is refused until named")
    return lines


def _run_record(
    plan: Plan | None, wheels: Mapping[str, str], apps: Sequence[str], k: int, priced: Mapping[str, Quote]
) -> dict[str, Any]:  # Any: JSON
    """``run.json``: what a run ran, recorded before its first trial: the plan and the commit it was last changed in,
    each arm's wheel and commit, the apps, ``k``, and each tier's model and caps."""
    refs = {arm.name: arm.ref for arm in plan.arms} if plan is not None else {}
    return {
        "plan": None if plan is None else str(plan.path.resolve().relative_to(HERE.parents[2])),
        "plan_commit": None if plan is None else committed_at(plan.path),
        "question": None if plan is None else plan.question,
        "primary": PRIMARY if plan is None else plan.primary,
        "arms": {
            name: {"wheel": wheel, "ref": refs.get(name), "commit": commit_of(refs[name]) if name in refs else None}
            for name, wheel in wheels.items()
        },
        "apps": list(apps) or _apps(),
        "k": k,
        "tiers": {
            name: {"model": q.tier.model_id, "per_run_usd": q.per_run_usd, "cap_usd": q.bound_usd}
            for name, q in priced.items()
        },
    }


def read_stored(run_dir: Path) -> list[str]:
    """``--read``: the package's reading of a stored run, apps as cases, tier by tier (each tier's ``trials.jsonl``,
    in its own directory when the run had more than one), with the plan's primary check and question when it ran one.

    :raises Unreadable: No ``trials.jsonl`` under ``run_dir``, or rows the package cannot read with apps as cases.
    """
    files = (
        [run_dir / "trials.jsonl"] if (run_dir / "trials.jsonl").is_file() else sorted(run_dir.glob("*/trials.jsonl"))
    )
    if not files:
        raise Unreadable(f"{run_dir} holds no trials.jsonl, nor one per tier")
    rows = []
    for file in files:
        tier = None if file.parent == run_dir else file.parent.name
        for line in file.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            rows.append({**row, "tier": row.get("tier") or tier})
    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8")) if (run_dir / "run.json").is_file() else {}
    head = [f"plan {meta['plan']} at {meta.get('plan_commit')}"] if meta.get("plan") else []
    return head + reading_lines(rows, primary=meta.get("primary") or PRIMARY, question=meta.get("question"))


def preflight(builds: Sequence[DocsBuild], workdir: Path) -> str | None:
    """Why an agent run on ``builds`` may not start, or ``None``: the fixed transcripts that succeed are run against
    each build first, in its trial containers, offline and free, and every check must pass on every app. A build the
    harness cannot read right (a field renamed, a door that moved) would otherwise fail its arm's trials quietly."""
    for build in builds:
        arm = {build.name: adopter(build, scripted(), workdir=workdir, caps=Caps())}
        result = asyncio.run(onramp_eval(arm, k=1, max_cost_usd=None).run(store=workdir / f"{slug(build.name)}.sqlite"))
        if (refusal := preflight_refusal(build.name, trials(result))) is not None:
            return refusal
    return None


def preflight_refusal(arm: str, rows: Sequence[Mapping[str, Any]]) -> str | None:  # Any: a trial's stored row
    """Why ``arm``'s scripted pre-flight refuses an agent run, or ``None`` when every check passed on every app."""
    apps = [json.loads(line)["app"] for line in CASES.read_text(encoding="utf-8").splitlines() if line.strip()]
    passed = {
        row["record"]["app"]
        for row in rows
        if isinstance(row["record"], Mapping) and all(row["scores"].get(g.__name__) == 1.0 for g in GRADERS)
    }
    if len(passed) == len(apps) == len(rows):
        return None
    failed = [app for app in apps if app not in passed]
    said = []
    for row in rows:
        record = row["record"] if isinstance(row["record"], Mapping) else {}
        app_reading = record.get("checks", {}).get("app", {})
        if isinstance(app_reading, Mapping) and "refused" in app_reading:
            said.append(f"{record.get('app')}: {app_reading['refused']}")
    why = f"; {'; '.join(said)}" if said else ""
    return (
        f"the scripted pre-flight passed {len(passed)} of {len(apps)} apps on arm {arm} (failed: {', '.join(failed)})"
        f"{why}: the harness cannot read this build right, so an agent run on it would not measure the docs"
    )


def _host_builds(wheels: Mapping[str, str], workdir: Path) -> list[DocsBuild]:
    """Each wheel installed in an environment of its own under ``workdir``, on this machine: for ``--host`` only."""
    return [DocsBuild.from_wheel(name, Path(wheel), workdir / "venvs") for name, wheel in wheels.items()]


def _isolated_builds(wheels: Mapping[str, str], workdir: Path) -> list[DocsBuild] | str:
    """Each wheel installed in a trial image, as the arm it names; or why that cannot be done here, which refuses
    the run before anything is built or sent."""
    if (refusal := isolation_refusal()) is not None:
        return refusal
    try:
        return [DocsBuild.in_image(name, Path(wheel), workdir / "images") for name, wheel in wheels.items()]
    except IsolationUnavailable as why:
        return f"an agent's commands run only in a trial container, and none can run here: {why}"


def _finish(ev: Eval, run_dir: Path, *, tier: str | None = None, plan: Plan | None = None) -> int:
    """Run the eval into ``run_dir``'s store, print it, write its report and every trial (each stamped with its
    ``tier``, when the run names one), and, with two arms or more, the package's reading of the run with apps as cases
    (``reading.txt``); exit as the door does."""
    run_dir.mkdir(parents=True, exist_ok=True)
    result = asyncio.run(ev.run(store=run_dir / "onramp.sqlite"))
    print(result.render())
    written = result.write(run_dir / "report")
    rows = [{**row, "tier": tier} if tier is not None else row for row in trials(result)]
    print(spend_line(rows))
    (run_dir / "trials.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    for row in rows:
        record = row["record"] if isinstance(row["record"], dict) else {}
        if record.get("stuck"):
            stuck = record["stuck"]
            print(
                f"stuck ({stuck.get('cause', CANNOT_TELL)}): {row['arm']} {record['app']} turns "
                f"{stuck['from_turn']}-{stuck['to_turn']}: {stuck['opened_by']}"
            )
    said = census(rows)
    (run_dir / "census.txt").write_text("\n".join(said) + "\n", encoding="utf-8")
    print("\n".join(said))
    if len(result.arms) > 1:
        read = reading_lines(rows, **({"primary": plan.primary, "question": plan.question} if plan else {}))
        (run_dir / "reading.txt").write_text("\n".join(read) + "\n", encoding="utf-8")
        print("\n".join(read))
    for unfinished in sorted(path.parent for path in run_dir.glob("*/*/steps.jsonl")):
        if not (unfinished / "record.json").is_file():
            print(f"unfinished: {unfinished} (its steps.jsonl holds every step it took)")
    print(f"report: {written['html']}\ntrials: {run_dir / 'trials.jsonl'}")
    return result.exit_code()


def spend_line(rows: Sequence[Mapping[str, Any]]) -> str:  # Any: a trial's stored row
    """The most the run can have been billed, beside what the provider reported: the package's spend line sums priced
    calls only, and a call the wall cut off, or a send that broke, may have been billed at up to its ceiling."""
    records = [row["record"] for row in rows if isinstance(row["record"], Mapping)]
    reported = [record.get("cost_usd") for record in records]
    charged = [record.get("charged_usd") for record in records]
    said = "unpriced" if any(c is None for c in reported) else f"${sum(c for c in reported if c is not None):.3f}"
    bound = "unbounded" if any(c is None for c in charged) else f"${sum(c for c in charged if c is not None):.3f}"
    return f"spend: {said} reported by the provider; at most {bound} billed (each cut-off call at its ceiling)"


def clopper_pearson(passed: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """The exact (Clopper-Pearson) interval on a pass share from ``passed`` of ``n``: it covers the true share at
    least ``level`` of the time, at every share and every ``n``."""
    tail = (1 - level) / 2

    def at_least(p: float) -> float:  # P(X >= passed) at share p
        return 1 - sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(passed))

    def at_most(p: float) -> float:  # P(X <= passed) at share p
        return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(passed + 1))

    def solve(f: Callable[[float], float], rising: bool) -> float:
        low, high = 0.0, 1.0
        for _ in range(60):
            middle = (low + high) / 2
            if (f(middle) < tail) == rising:
                low = middle
            else:
                high = middle
        return (low + high) / 2

    lower = 0.0 if passed == 0 else solve(at_least, rising=True)
    upper = 1.0 if passed == n else solve(at_most, rising=False)
    return lower, upper


def census(rows: Sequence[Mapping[str, Any]]) -> list[str]:  # Any: a trial's stored row
    """What this run can and cannot say: each check's tally (with its exact 95% range, for one arm), and its limits.

    A handful of runs is a census of where agents get stuck and what they cost, not a rate: trials on one app share
    the app, so the ranges are if anything too narrow, and a failure does not say whether the docs or the model failed.
    Arms are never pooled: with more than one, each check is tallied per arm, its line headed by the arm and carrying
    no range, since an arm's trials pooled over apps, set beside a range over trials, would read as the test between
    arms that the design cannot support. With more than one app, each line ends with the tally app by app, since the
    app, not the trial, is the unit a claim beyond these apps rests on, and the footer sends the reader to it.
    """
    arms = list(dict.fromkeys(row.get("arm") for row in rows))
    lines = [f"what {len(rows)} run(s) can say:"]
    for arm in arms:
        own = [row for row in rows if row.get("arm") == arm]
        head = f"{arm} " if len(arms) > 1 else ""
        for grader in GRADERS:
            name = grader.__name__
            n = sum(1 for row in own if name in row["scores"])
            passed = sum(1 for row in own if row["scores"].get(name) == 1.0)
            if n:
                shown = ""
                if len(arms) == 1:
                    low, high = clopper_pearson(passed, n)
                    shown = f", 95% range {low:.2f} to {high:.2f}"
                causes = [row.get("causes", failure_causes(row)).get(name) for row in own if name in row["scores"]]
                counted = ", ".join(f"{causes.count(each)} {each}" for each in CAUSES if causes.count(each))
                harness = causes.count(HARNESS)
                apart = f"; {n - passed} failed: {counted}" if n > passed else ""
                apart += f"; {passed} of {n - harness} the harness failed none of" if harness else ""
                lines.append(f"  {head}{name}: {passed} of {n} trials{shown}{apart}{_by_app(own, name)}")
    lines += [
        "  it shows: whether the tier ever succeeds, where runs get stuck and on which message, and turns and cost",
        "  per run, on these apps.",
        "  it cannot show: a success rate to rely on, a difference between apps, whether a failure belongs to the",
        "  docs or to the model, or anything beyond these apps: trials on one app are repeats of that app.",
        "  between arms, compare each check app by app, on its tally by app: never an arm's k of n over all apps,",
        "  nor any range over trials, as a test between arms; never errors_hit, nor the count of trials the harness",
        "  failed none of, which differ with how each arm's agents behave, not with what the harness is.",
    ]
    return lines


def _app_of(row: Mapping[str, Any]) -> str | None:  # Any: a trial's stored row
    """The app a trial ran on, from its record; ``None`` when the record holds none."""
    record = row.get("record")
    app = record.get("app") if isinstance(record, Mapping) else None
    return app if isinstance(app, str) else None


def _by_app(rows: Sequence[Mapping[str, Any]], name: str) -> str:  # Any: a trial's stored row
    """``; by app: holds 1 of 7, recipes 3 of 7``: one check's tally app by app, when the rows span more than one
    app; empty otherwise."""
    apps = sorted({app for row in rows if (app := _app_of(row)) is not None and name in row["scores"]})
    if len(apps) < 2:
        return ""
    tallies = []
    for app in apps:
        on = [row for row in rows if _app_of(row) == app and name in row["scores"]]
        tallies.append(f"{app} {sum(1 for row in on if row['scores'][name] == 1.0)} of {len(on)}")
    return f"; by app: {', '.join(tallies)}"
