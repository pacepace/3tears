"""``examples/llm_judge.py`` runs as a newcomer would run it, with no API key, and reads its judge's misses.

Offline, the candidate and the judge are its stand-ins, so the run is deterministic and its output must say it is
not a model's. The judge is pass/fail, so a failed dimension is a miss the example prints with the judge's reason.
The live client is ``examples/_live.py``'s, tested in ``test_live_example.py``; here the live path is driven over a
faked SDK only to hold the call count the docstring states.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from threetears.evals.quick import EvalSummary
from packages.evals.tests.example_loader import EXAMPLES, load_example
from packages.evals.tests.test_package_matrix import REPO_ROOT, SOURCE_ROOT, public_root_violations


async def test_with_no_api_key_the_example_runs_offline_and_prints_each_miss_with_the_judge_s_reason(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    summary = await load_example("llm_judge.py").main()
    assert isinstance(summary, EvalSummary)
    assert summary.status == "completed"
    assert (summary.n_cases, summary.k_runs, summary.n_results, summary.n_scored) == (5, 2, 10, 10)
    assert summary.measures == [], "the judge is the only grade"
    judged = {dimension.name: dimension for dimension in summary.judged}
    assert list(judged) == ["answer.helpful", "answer.grounded"]
    assert all(dimension.scale == "pass_fail" and dimension.n == 10 for dimension in judged.values())
    assert (judged["answer.helpful"].mean, judged["answer.grounded"].mean) == (1.0, 0.8)
    assert summary.judge_calls == 20 and summary.errors == []
    assert summary.judge_shares_candidate_model == [], "the stand-in judge is not the stand-in candidate"

    misses = summary.misses()
    assert [(miss.input["question"], miss.repeat) for miss in misses] == [
        ("Do you offer gift wrapping?", 1),
        ("Do you offer gift wrapping?", 2),
    ]
    out = capsys.readouterr().out
    assert out.startswith("ANTHROPIC_API_KEY is not set: running OFFLINE")
    assert summary.render() in out
    assert "the judge failed it on answer.grounded: offline stand-in, word overlap: 2 of the answer's 8 words" in out


def test_the_example_reaches_the_engine_only_through_public_roots() -> None:
    path = EXAMPLES / "llm_judge.py"
    assert public_root_violations(SOURCE_ROOT, [("llm_judge.py", path)], consumer_root=REPO_ROOT) == []


async def test_live_it_makes_the_thirty_calls_its_docstring_states(monkeypatch: pytest.MonkeyPatch) -> None:
    anthropic = pytest.importorskip("anthropic")
    module = load_example("llm_judge.py")
    sent: list[dict[str, Any]] = []

    async def create(**request: Any) -> Any:
        sent.append(request)
        judging = "criteria_scores" in request["system"]
        dimension = request["system"].split('single key "', 1)[1].split('"', 1)[0] if judging else ""
        text = f'{{"reasoning": "ok", "criteria_scores": {{"{dimension}": "pass"}}}}' if judging else "30 days."
        return anthropic.types.Message.model_validate(
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": module.MODEL,
                "content": [{"type": "text", "text": text}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 400, "output_tokens": 60},
            }
        )

    async def close() -> None:
        return None

    monkeypatch.setattr(
        anthropic, "AsyncAnthropic", lambda: SimpleNamespace(messages=SimpleNamespace(create=create), close=close)
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    summary = await module.main()
    assert len(sent) == 30 and summary.judge_calls == 20
    assert summary.judge_shares_candidate_model == [module.MODEL], "a model judging its own answers is disclosed"
