"""``examples/llm_judge.py`` runs as a newcomer would run it with no API key: against its recording, and reads its misses.

With no key the example replays ``examples/recordings/llm_judge.jsonl``, what Claude answered and judged when the
run was recorded, so no test calls the API. Its answers are checked against the recording before the run, so a
prompt that changed under it is named rather than met as a miss partway through; a judge prompt depends on the
answer it judges, so a miss there fails the test with the refusal that names it. Nothing here assumes what Claude
said: the run is held to its shape, the counts its docstring states, and the misses it prints.
"""

from __future__ import annotations

import pytest

from threetears.evals.analysis import EvalSummary
from threetears.evals.run import JUDGE_REQUEST_SETTINGS
from threetears.evals.testing import RecordedPrompt
from packages.evals.tests.example_loader import EXAMPLES, check_covers, load_example, replaying
from packages.evals.tests.test_package_matrix import REPO_ROOT, SOURCE_ROOT, public_root_violations


async def test_with_no_api_key_the_example_replays_its_recording_and_prints_each_miss_with_the_judge_s_reason(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = load_example("llm_judge.py")
    answers = [RecordedPrompt(module.SYSTEM, case["question"]) for case in module.CASES for _ in range(2)]
    check_covers(module, answers, settings=module.SETTINGS)
    replay = replaying(module, monkeypatch)

    summary = await module.main()

    assert replay.served() == 30, "5 questions x 2 repeats, each answered once and judged on 2 dimensions"
    asked = [settings for settings, _, _ in replay.calls]
    assert (asked.count(module.SETTINGS), asked.count(module.JUDGE_SETTINGS)) == (10, 20)
    assert isinstance(summary, EvalSummary)
    assert summary.status == "completed" and summary.errors == []
    assert (summary.n_cases, summary.k_runs, summary.n_results, summary.n_scored) == (5, 2, 10, 10)
    assert summary.measures == [], "the judge is the only grade"
    judged = {dimension.name: dimension for dimension in summary.judged}
    assert list(judged) == ["answer.helpful", "answer.grounded"]
    assert all(dimension.scale == "pass_fail" and dimension.n == 10 for dimension in judged.values())
    assert summary.role_spend("judge").calls == 20
    assert summary.judge_shares_candidate_model == [module.MODEL], "a model judging its own answers is disclosed"

    out = capsys.readouterr().out
    assert out.startswith("ANTHROPIC_API_KEY is not set: replaying recordings/llm_judge.jsonl, Claude (")
    assert summary.render() in out
    for miss in summary.misses():  # every miss is printed with the judge's reason
        assert f"  {miss.input['question']!r}, repeat {miss.repeat}, answered {miss.answer!r}\n" in out
        assert f"    {'; '.join(miss.missed_because)}\n" in out


def test_the_judge_is_asked_as_a_judged_run_records_it() -> None:
    module = load_example("llm_judge.py")
    assert module.JUDGE_SETTINGS == JUDGE_REQUEST_SETTINGS, "what a judged run records is what the judge is sent"
    assert module.SETTINGS.reasoning_effort == "low"


def test_the_example_reaches_the_engine_only_through_public_roots() -> None:
    path = EXAMPLES / "llm_judge.py"
    assert public_root_violations(SOURCE_ROOT, [("llm_judge.py", path)], consumer_root=REPO_ROOT) == []
