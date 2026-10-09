"""``examples/llm_judge.py`` runs as a newcomer would run it, with no API key, and its adapter reads the real SDK's reply.

Offline, the example's candidate and judge are its stand-ins, so the run is deterministic and its
output must say it is not a model's. The adapter from the ``anthropic`` SDK is exercised against a
reply built from the SDK's own types, so a field the SDK renames turns this red rather than the first
live run.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from threetears.evals.quick import EvalSummary
from packages.evals.tests.test_package_matrix import REPO_ROOT, SOURCE_ROOT, public_root_violations

#: The example under test.
LLM_JUDGE = Path(__file__).resolve().parents[1] / "examples" / "llm_judge.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("llm_judge_example", LLM_JUDGE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def test_with_no_api_key_the_example_runs_offline_and_says_so(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    summary = await _load().main()
    assert isinstance(summary, EvalSummary)
    assert summary.status == "completed"
    assert (summary.n_cases, summary.k_runs, summary.n_results, summary.n_scored) == (5, 2, 10, 10)
    assert summary.candidate_model == "offline"
    assert [measure.name for measure in summary.measures] == ["concise"]
    assert [dimension.name for dimension in summary.judged] == ["answer.helpful", "answer.grounded"]
    assert all(dimension.n == 10 for dimension in summary.judged)
    assert summary.judge_calls == 20 and summary.judge_cost_usd == 0.0
    assert summary.errors == []
    out = capsys.readouterr().out
    assert out.startswith("ANTHROPIC_API_KEY is not set: running OFFLINE")
    assert summary.render() in out


def test_the_example_reaches_the_engine_only_through_public_roots() -> None:
    assert public_root_violations(SOURCE_ROOT, [("llm_judge.py", LLM_JUDGE)], consumer_root=REPO_ROOT) == []


async def test_the_claude_client_prices_a_reply_and_the_candidate_reports_it(monkeypatch: pytest.MonkeyPatch) -> None:
    anthropic = pytest.importorskip("anthropic")
    module = _load()
    # The id is the example's own, read from it, so a model rev in the example is the only edit.
    model, served = module.MODEL, f"{module.MODEL}-served"
    reply = anthropic.types.Message.model_validate(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": served,
            "content": [{"type": "text", "text": '{"ok": true}'}],
            "stop_reason": "max_tokens",
            "stop_sequence": None,
            "usage": {
                "input_tokens": 1_000,
                "output_tokens": 200,
                "output_tokens_details": {"thinking_tokens": 50},
            },
        }
    )
    sent: list[dict[str, Any]] = []

    async def create(**request: Any) -> Any:
        sent.append(request)
        return reply

    async def close() -> None:
        return None

    monkeypatch.setattr(
        anthropic, "AsyncAnthropic", lambda: SimpleNamespace(messages=SimpleNamespace(create=create), close=close)
    )
    client = module.claude_client()
    completion = await client.generate(system="be brief", user="hello", response_format={"type": "json_object"})

    assert sent[0]["model"] == model and sent[0]["system"] == "be brief"
    assert "response_format" not in sent[0]
    assert completion.cost_usd == pytest.approx(1_000 * 0.10 / 1e6 + 200 * 0.50 / 1e6)
    assert (completion.input_tokens, completion.output_tokens, completion.reasoning_tokens) == (1_000, 200, 50)
    assert (completion.content, completion.stop_reason) == ('{"ok": true}', "max_tokens")
    assert (completion.model, completion.served_model) == (model, served)

    # The candidate asks through the same client and returns the call's spend beside its answer.
    answer = await module.claude_answerer(client)(module.CASES[0])
    assert (sent[1]["system"], sent[1]["messages"][0]["content"]) == (module.SYSTEM, module.CASES[0]["question"])
    assert (answer.value, answer.model, answer.cost_usd) == ('{"ok": true}', model, completion.cost_usd)
    await client.aclose()
