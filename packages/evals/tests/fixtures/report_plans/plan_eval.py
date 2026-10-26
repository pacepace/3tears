"""Can a cheap model plan the five reference reports? The planner eval's recordings, and the plans they replay to.

The report planner (``threetears.evals.analysis.planner.generate_report_plan``) turns one campaign's report input into
its report plan in two model calls, a structure and its prose, each repaired at most once. This fixture runs it over
the five report inputs beside it (``r1/input.json`` to ``r5/input.json``) with one cheap model, and keeps what the model
wrote:

- ``recordings/r1.jsonl`` to ``recordings/r5.jsonl``: every call of each generation, repairs included, as
  :class:`~threetears.evals.testing.RecordedCompletions` writes them;
- ``recordings/prompt.txt``: the planner prompt the recordings were made under, which a replay hands the planner, as
  the shipped one moves on;
- ``r1/runs/<run_id>.plan.json`` to ``r5/runs/<run_id>.plan.json``: the plan each recording replays to, the planner
  eval's dataset (scored against ``rubric/`` by a model judge; ``dataset.py brief`` prints the brief for one run),
  each beside ``<run_id>.planner.json``, what wrote it;
- ``recordings/runs.json``: each input's current run, the one its recording replays to.

**A re-record adds runs.** The recordings and the prompt are replaced, since a replay answers only the code it was made
under, but every run plan, its provenance and its scores stay in the dataset (owner, 2026-10-25): a new prompt version
writes a new run id beside the old ones, and ``runs.json`` moves to it.

Without ``RECORD=1`` it replays the recordings, calls nothing and needs no key, which is how
``tests/test_plan_eval_fixture.py`` runs it. ``RECORD=1``, with ``OPENROUTER_API_KEY`` set, records all five again and
rewrites the prompt and the run plans; a recording answers only the prompt, report input and planner code it was made
under, so any of those changing makes a replay refuse by name, and the remedy is to record again::

    RECORD=1 uv run python -m packages.evals.tests.fixtures.report_plans.plan_eval

A generation the planner refuses twice raises, and that run gets no plan: every run must end in a plan that passes the
plan checks (``validate_plan``), or it is not a dataset entry.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from packages.evals.tests.fixtures.report_plans import dataset
from threetears.evals.analysis.plan_prompt import EVAL_REPORT_PLAN_DEFAULT
from threetears.evals.analysis.planner import PlanGeneration, generate_report_plan
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.analysis.report_plan_checks import validate_plan
from threetears.evals.kernel.host.measures import MeasureRegistry
from threetears.evals.kernel.host.profile import HostProfile
from threetears.evals.kernel.host.sweepables import SHARED_CORE
from threetears.evals.providers import Model
from threetears.evals.schema import ClientRequestSettings
from threetears.evals.testing import RecordedCompletions

__all__ = [
    "MODEL",
    "PROFILE",
    "RECORDED_PROMPT",
    "RUNS",
    "SETTINGS",
    "client",
    "keep",
    "main",
    "plan",
    "recording",
    "report_input",
    "run_id",
    "run_plan_path",
]

#: The planner: a flash-class model through OpenRouter. Cheaper flash models tried on r5 (deepseek-v4-flash and v4.1-flash,
#: gemini-3.1 and 3.5 flash-lite, qwen3.8-flash) were refused twice or ran out of output; this one planned all five.
MODEL = "google/gemini-3.8-flash"
#: How the planner is asked: room for the larger structure reply, and low effort.
SETTINGS = ClientRequestSettings(max_tokens=16_000, reasoning_effort="low")
#: The five report inputs, by directory.
RUNS = ("r1", "r2", "r3", "r4", "r5")

HERE = Path(__file__).resolve().parent
RECORDINGS = HERE / "recordings"
#: The planner prompt the recordings were made under.
RECORDED_PROMPT = RECORDINGS / "prompt.txt"
RERECORD = "RECORD=1 uv run python -m packages.evals.tests.fixtures.report_plans.plan_eval"

#: The host the plans are written for: a bare one, so the system prompt carries the default register and nothing else.
PROFILE = HostProfile(host_id="report-plan-eval", host_sweepables=SHARED_CORE, measures=MeasureRegistry(()))


def recording(run: str) -> Path:
    """The recording of one run's generation."""
    return RECORDINGS / f"{run}.jsonl"


def report_input(run: str) -> ReportInput:
    """One run's report input."""
    return ReportInput.model_validate_json((HERE / run / "input.json").read_text(encoding="utf-8"))


def recording_live() -> bool:
    """Whether this process records (``RECORD=1``) rather than replays."""
    return os.environ.get("RECORD") == "1"


def client(run: str, model: str, settings: ClientRequestSettings) -> Any:
    """``model`` for one run: live and recorded with ``RECORD=1``, else that run's recording, replayed."""
    if not recording_live():
        return RecordedCompletions.replay(recording(run), settings=settings, rerecord=RERECORD)
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("RECORD=1 records a live run, which needs OPENROUTER_API_KEY")
    live = Model(model, api_key=key, max_tokens=settings.max_tokens, effort=settings.reasoning_effort)
    return RecordedCompletions.record(live, recording(run), settings=settings, replace=True)


def prompt() -> str:
    """The planner prompt: the shipped one while recording, else the one the recordings were made under."""
    if recording_live():
        return EVAL_REPORT_PLAN_DEFAULT
    if not RECORDED_PROMPT.is_file():
        raise SystemExit(f"{RECORDED_PROMPT} is missing, and the recordings answer only it; {RERECORD} records it")
    return RECORDED_PROMPT.read_text(encoding="utf-8")


async def plan(run: str) -> PlanGeneration:
    """One run's plan, generated live (``RECORD=1``) or from its recording, and checked against its input."""
    llm = client(run, MODEL, SETTINGS)
    try:
        generated = await generate_report_plan(
            report_input(run),
            prompt=prompt(),
            model=MODEL,
            client=llm,
            profile=PROFILE,
            measuring_writers=True,  # an eval of planner models, so not held to a host's list of writers
        )
    finally:
        await llm.aclose()
    validate_plan(generated.plan, report_input(run))
    return generated


def run_id(run: str, generation: PlanGeneration) -> str:
    """The run's id: the model, the day its recording was made, and the planner's prompt version."""
    first = json.loads(recording(run).read_text(encoding="utf-8").splitlines()[0])
    made = str(first["recorded_at"])[:10].replace("-", "")
    model = re.sub(r"[^a-z0-9.]+", "-", MODEL.rsplit("/", 1)[-1].lower())
    return f"{model}-{made}-{generation.prompt_version}"


def run_plan_path(run: str, generation: PlanGeneration) -> Path:
    """Where one run's plan is kept in the dataset."""
    return HERE / run / "runs" / f"{run_id(run, generation)}.plan.json"


def plan_json(generation: PlanGeneration) -> str:
    """A plan as the dataset keeps it: by its JSON field names, indented, ending in a newline."""
    return generation.plan.model_dump_json(by_alias=True, indent=2) + "\n"


def keep(run: str, generation: PlanGeneration) -> None:
    """Add a freshly recorded run to the dataset: its plan, what wrote it, and its place as the input's current run.

    :raises SystemExit: A run under this id is kept already and differs: a run is never overwritten.
    """
    path = run_plan_path(run, generation)
    if path.is_file() and path.read_text(encoding="utf-8") != plan_json(generation):
        raise SystemExit(f"{path.name} is kept already and this recording differs; a kept run is never overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(plan_json(generation), encoding="utf-8")
    dataset.planner_path(path).write_text(dataset.planner_json(dataset.recorded_planner(run)), encoding="utf-8")
    current = dataset.current_runs() if dataset.CURRENT_RUNS.is_file() else {}
    current[run] = run_id(run, generation)
    dataset.CURRENT_RUNS.write_text(json.dumps(dict(sorted(current.items())), indent=2) + "\n", encoding="utf-8")


async def main(runs: tuple[str, ...] = RUNS) -> dict[str, PlanGeneration]:
    """Plan every run, print what each cost and how often it was repaired, and, recording, keep each plan."""
    how = f"recording into {RECORDINGS.name}/" if recording_live() else f"replaying {RECORDINGS.name}/"
    print(f"Planning {len(runs)} report inputs with {MODEL}, {how}.")
    if recording_live():
        RECORDED_PROMPT.parent.mkdir(parents=True, exist_ok=True)
        RECORDED_PROMPT.write_text(EVAL_REPORT_PLAN_DEFAULT, encoding="utf-8")
    generations: dict[str, PlanGeneration] = {}
    for run in runs:
        generated = await plan(run)
        generations[run] = generated
        repaired = [
            stage
            for stage, refusal in (("structure", generated.structure_refusal), ("prose", generated.prose_refusal))
            if refusal
        ]
        print(
            f"{run}: {generated.plan.mode}, {generated.repair_attempts} repair(s) {repaired}, "
            f"${generated.token_cost:.4f}, prompt {generated.prompt_version}"
        )
        if recording_live():
            keep(run, generated)
    total = sum(generated.token_cost for generated in generations.values())
    print(f"Total, as the recordings report it: ${total:.4f} ({datetime.now(UTC).date().isoformat()}).")
    return generations


if __name__ == "__main__":
    asyncio.run(main())
