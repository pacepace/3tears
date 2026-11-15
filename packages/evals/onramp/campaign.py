"""The agent test as an eval on the package: cases are apps, arms are docs builds, the runner is an adopting agent.

``python -m packages.evals.onramp`` (``scripts/onramp-agent.sh`` builds today's wheel and runs it) measures one or
more docs builds. ``--scripted`` runs the harness offline with the fixed transcripts and spends nothing. A live run
spends money, so it is refused unless ``ONRAMP_LIVE=1`` is set and ``--cap`` states the most it may spend; it quotes
its cost before anything runs, and holds each run to its share of the cap.

Models follow the project's rules: the cheap tier first, through OpenRouter (``OPENROUTER_API_KEY``); Claude Haiku 5.5
as the ceiling, only ever direct to Anthropic on ``ANTHROPIC_API_KEY``, never through OpenRouter, and no model is
sent a temperature.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
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
    Brain,
    Caps,
    ModelBrain,
    ScriptedBrain,
    attempt,
    commands,
    errors_hit,
    parse_report,
    stuck_point,
)
from .checks import GRADERS, regrade
from .sandbox import HERE, DocsBuild, Sandbox

CASES = HERE / "apps.jsonl"
TRANSCRIPTS = HERE / "transcripts"
#: The commit today's wheel, the control arm, is built from: the docs and messages before any on-ramp fix.
CONTROL_REF = "f9ab6f74d"
#: Set to 1 to let a run spend money.
LIVE_ENV = "ONRAMP_LIVE"
#: The turn profile an expected cost assumes: turns a run, prompt tokens a turn, output tokens a turn.
PROFILE = (25, 15_000, 1_500)

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
            return Model(self.model_id, provider="anthropic", max_tokens=4096, recording=recording, **client)
        return Model(self.model_id, max_tokens=4096, effort="low", recording=recording, **client)


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
    """Why a live run may not start, or ``None`` when it may."""
    if environment.get(LIVE_ENV) != "1":
        return f"a live run spends money: set {LIVE_ENV}=1 to say it may"
    if cap_usd is None or cap_usd <= 0:
        return "a live run states the most it may spend: pass --cap DOLLARS"
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
        run = await attempt(sandbox, brain, caps)
        report, problem = parse_report(run.report)
        ran = commands(run.steps)
        return {
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
            "commands": ran,
            "errors_hit": errors_hit(run.steps),
            "pages_read": sandbox.pages,
            "written": sandbox.written,
            "stuck": stuck_point(run.steps),
            "report": report,
            "report_problem": problem,
            "sandbox": str(sandbox.root),
            "checks": await regrade(sandbox, trial, report, ran),
        }

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
    """Every trial as the eval's store holds it: the arm, the agent's record, and each grade."""
    return [
        {"arm": arm, "repeat": trial.repeat, "state": trial.state, "scores": trial.scores, "record": trial.answer}
        for arm in result.arms
        for trial in result.results(arm)
    ]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="onramp-agent", description=__doc__.splitlines()[0])
    parser.add_argument(
        "--arm", action="append", default=[], metavar="NAME=WHEEL", help="a docs build; first is control"
    )
    parser.add_argument("--scripted", action="append", default=[], choices=("succeeds", "stuck"), help="offline arms")
    parser.add_argument("--tier", choices=sorted(TIERS), default="cheap")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--cap", type=float, default=None, metavar="DOLLARS", help="the most a live run may spend")
    parser.add_argument("--apps", default="", help="comma-separated; every app when empty")
    parser.add_argument("--workdir", type=Path, default=HERE / "runs")
    parser.add_argument("--dry-run", action="store_true", help="say what would run and what it may cost; run nothing")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    apps = [app for app in args.apps.split(",") if app]
    cases = len(apps) if apps else sum(1 for line in CASES.read_text(encoding="utf-8").splitlines() if line.strip())
    run_dir = args.workdir / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    if args.scripted:
        arms = {
            f"scripted@{kind}": adopter(
                DocsBuild.here(f"scripted@{kind}"),
                scripted(None if kind == "succeeds" else "stuck"),
                workdir=run_dir,
                caps=Caps(),
            )
            for kind in args.scripted
        }
        return _finish(onramp_eval(arms, k=args.k, max_cost_usd=None, apps=apps), run_dir)
    wheels = dict(arm.split("=", 1) for arm in args.arm)
    if not wheels:
        print("onramp-agent: name a docs build (--arm NAME=WHEEL) or run --scripted", file=sys.stderr)
        return 2
    tier = TIERS[args.tier]
    priced = quote(tier, runs=len(wheels) * cases * args.k, cap_usd=args.cap)
    print("\n".join([*(f"arm {name}: {wheel}" for name, wheel in wheels.items()), *priced.lines()]))
    if args.dry_run:
        return 0
    if (refusal := live_refusal(os.environ, args.cap)) is not None:
        print(f"onramp-agent: {refusal}", file=sys.stderr)
        return 2
    arms: dict[str, Runner] = {}
    for name, wheel in wheels.items():
        build = DocsBuild.from_wheel(name, Path(wheel), args.workdir / "venvs")
        try:
            brain = ModelBrain(tier.model(run_dir / "recordings" / f"{slug(name)}.jsonl"))
        except ValueError as unbuilt:  # a missing key, named
            print(f"onramp-agent: {unbuilt}", file=sys.stderr)
            return 2
        caps = Caps(cost_usd=priced.per_run_usd)
        arms[name] = adopter(build, lambda _sandbox, brain=brain: brain, workdir=run_dir, caps=caps)
    return _finish(onramp_eval(arms, k=args.k, max_cost_usd=args.cap, apps=apps), run_dir)


def _finish(ev: Eval, run_dir: Path) -> int:
    """Run the eval into ``run_dir``'s store, print it, write its report and every trial, and exit as the door does."""
    run_dir.mkdir(parents=True, exist_ok=True)
    result = asyncio.run(ev.run(store=run_dir / "onramp.sqlite"))
    print(result.render())
    written = result.write(run_dir / "report")
    rows = trials(result)
    (run_dir / "trials.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    for row in rows:
        record = row["record"] if isinstance(row["record"], dict) else {}
        if record.get("stuck"):
            stuck = record["stuck"]
            print(
                f"stuck: {row['arm']} {record['app']} turns {stuck['from_turn']}-{stuck['to_turn']}: {stuck['opened_by']}"
            )
    print(f"report: {written['html']}\ntrials: {run_dir / 'trials.jsonl'}")
    return result.exit_code()
