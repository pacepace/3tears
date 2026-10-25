"""The planner eval's recordings (``fixtures/report_plans/plan_eval.py``) replay, with no key, to the dataset's plans.

Contracts held here:

- with no key and no ``RECORD``, the fixture builds no live client and serves every call of every run from its
  recording, each call recorded once, and every run ends in a plan;
- each replayed plan passes the plan checks against its report input, and is byte for byte the run plan the dataset
  keeps (``rN/runs/<run_id>.plan.json``), so a scored plan is the plan its recording makes;
- the run id names the model, the day the recording was made and the planner's prompt version, and every run plan the
  dataset keeps has its recording.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from packages.evals.tests.fixtures.report_plans import plan_eval
from threetears.evals.analysis.planner import PlanGeneration
from threetears.evals.analysis.report_plan_checks import validate_plan
from threetears.evals.schema import ClientRequestSettings
from threetears.evals.testing import RecordedCompletions


@pytest.fixture
def replayed(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Run the fixture against its recordings only, counting the calls each run's client serves."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("RECORD", raising=False)

    def no_live_client(model: str, **_: Any) -> Any:
        pytest.fail(f"a replay built a live client for {model}")

    monkeypatch.setattr(plan_eval, "Model", no_live_client)
    served: dict[str, int] = {}
    build = plan_eval.client

    def counted(run: str, model: str, settings: ClientRequestSettings) -> Any:
        llm = build(run, model, settings)
        assert isinstance(llm, RecordedCompletions) and llm.mode == "replay"
        generate = llm.generate

        async def counting(*, system: str, user: str, response_format: dict[str, Any] | None = None) -> Any:
            completion = await generate(system=system, user=user, response_format=response_format)
            served[run] = served.get(run, 0) + 1
            return completion

        llm.generate = counting  # type: ignore[method-assign]
        return llm

    monkeypatch.setattr(plan_eval, "client", counted)
    return served


@pytest.fixture
async def generations(replayed: dict[str, int], capsys: pytest.CaptureFixture[str]) -> dict[str, PlanGeneration]:
    made = await plan_eval.main()
    out = capsys.readouterr().out
    assert out.startswith(f"Planning 5 report inputs with {plan_eval.MODEL}, replaying recordings/.")
    return made


def _recorded_calls(run: str) -> int:
    return len(plan_eval.recording(run).read_text(encoding="utf-8").splitlines())


async def test_every_run_replays_every_recorded_call_to_a_plan(
    replayed: dict[str, int], generations: dict[str, PlanGeneration]
) -> None:
    assert list(generations) == list(plan_eval.RUNS)
    for run, generated in generations.items():
        assert replayed[run] == _recorded_calls(run), run
        assert replayed[run] == 2 + generated.repair_attempts, run
        assert generated.generator_model == plan_eval.MODEL


async def test_each_replayed_plan_passes_the_checks_and_is_the_kept_run_plan(
    generations: dict[str, PlanGeneration],
) -> None:
    for run, generated in generations.items():
        validate_plan(generated.plan, plan_eval.report_input(run))
        kept = plan_eval.run_plan_path(run, generated)
        assert kept.is_file(), f"{run}: the dataset keeps no plan at {kept.name}"
        assert kept.read_text(encoding="utf-8") == plan_eval.plan_json(generated), run


async def test_the_run_id_names_model_day_and_prompt_version(generations: dict[str, PlanGeneration]) -> None:
    for run, generated in generations.items():
        made = json.loads(plan_eval.recording(run).read_text(encoding="utf-8").splitlines()[0])["recorded_at"]
        assert plan_eval.run_id(run, generated) == (
            f"gemini-3.8-flash-{made[:10].replace('-', '')}-{generated.prompt_version}"
        )


def test_every_kept_run_plan_has_its_recording_and_the_prompt_it_was_made_under() -> None:
    assert plan_eval.RECORDED_PROMPT.is_file()
    kept = sorted(Path(plan_eval.HERE).glob("r*/runs/*.plan.json"))
    assert {path.parts[-3] for path in kept} == set(plan_eval.RUNS)
    for path in kept:
        assert plan_eval.recording(path.parts[-3]).is_file(), path


def test_recording_needs_its_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RECORD", "1")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="needs OPENROUTER_API_KEY"):
        plan_eval.client("r1", plan_eval.MODEL, plan_eval.SETTINGS)
