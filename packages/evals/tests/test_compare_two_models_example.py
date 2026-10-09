"""``examples/compare_two_models.py`` runs as a newcomer would run it, and its live path prices the real SDK's reply.

Offline, each model is a keyword stand-in reporting made-up usage, so the run is deterministic: both
arms complete, each reports a non-zero spend, and the campaign report carries a verdict on accuracy and
on cost. The live path is exercised against replies built from the SDK's own types, so a field the SDK
renames turns this red rather than the first live run. Model ids are read off the example, never
written here.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from threetears.evals.analysis import TableBlock
from threetears.evals.quick import Answer, Comparison
from packages.evals.tests.test_package_matrix import REPO_ROOT, SOURCE_ROOT, public_root_violations

#: The example under test.
COMPARE_TWO_MODELS = Path(__file__).resolve().parents[1] / "examples" / "compare_two_models.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("compare_two_models_example", COMPARE_TWO_MODELS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _contrasts(comparison: Comparison) -> dict[str, Mapping[str, Any]]:
    """The report's contrasts against the control, by reading."""
    (table,) = [
        block for block in comparison.report.blocks if isinstance(block, TableBlock) and block.name == "comparisons"
    ]
    return {row["reading"]: row for row in table.rows}


async def test_with_no_api_key_the_example_runs_offline_and_weighs_accuracy_against_cost(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    module = _load()
    comparison = await module.main()

    assert comparison.control == module.OLDER
    assert list(comparison.arms) == [module.OLDER, module.NEWER]
    for summary in comparison.arms.values():
        assert summary.status == "completed" and (summary.n_cases, summary.k_runs, summary.n_scored) == (12, 2, 24)
        assert summary.candidate_calls == 24
        assert summary.candidate_cost_usd is not None and summary.candidate_cost_usd > 0
        assert summary.errors == []
    older, newer = comparison.arms[module.OLDER], comparison.arms[module.NEWER]
    assert newer.candidate_cost_usd < older.candidate_cost_usd  # type: ignore[operator]

    contrasts = _contrasts(comparison)
    assert {"accuracy", "cost_usd"} <= set(contrasts)
    assert all(row["verdict"] for row in contrasts.values())
    assert contrasts["cost_usd"]["delta"] < 0 and contrasts["cost_usd"]["verdict"] == "improved on the control"

    out = capsys.readouterr().out
    assert out.startswith("ANTHROPIC_API_KEY is not set: running OFFLINE")
    assert "say nothing about Claude" in out
    assert "candidate spend: $" in out
    assert comparison.render() in out


def test_the_example_reaches_the_engine_only_through_public_roots() -> None:
    assert (
        public_root_violations(SOURCE_ROOT, [("compare_two_models.py", COMPARE_TWO_MODELS)], consumer_root=REPO_ROOT)
        == []
    )


def _reply(anthropic: ModuleType, model: str, text: str) -> Any:
    """A reply as the SDK types it: a thinking block, then the label."""
    return anthropic.types.Message.model_validate(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [{"type": "thinking", "thinking": "", "signature": "sig"}, {"type": "text", "text": text}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 120, "output_tokens": 40},
        }
    )


def _fake_sdk(monkeypatch: pytest.MonkeyPatch, anthropic: ModuleType, reply: Any) -> list[dict[str, Any]]:
    """Point ``anthropic.AsyncAnthropic`` at a client whose ``create`` records each request and returns ``reply``."""
    sent: list[dict[str, Any]] = []

    async def create(**request: Any) -> Any:
        sent.append(request)
        return reply(request) if callable(reply) else reply

    monkeypatch.setattr(anthropic, "AsyncAnthropic", lambda: SimpleNamespace(messages=SimpleNamespace(create=create)))
    return sent


async def test_the_live_candidate_returns_the_label_with_its_priced_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    anthropic = pytest.importorskip("anthropic")
    module = _load()
    sent = _fake_sdk(monkeypatch, anthropic, _reply(anthropic, module.NEWER, " Phishing\n"))

    answer = await module.claude_classifier(module.NEWER)(module.CASES[0])

    assert isinstance(answer, Answer)
    assert sent[0]["model"] == module.NEWER and sent[0]["system"] == module.PROMPT
    assert sent[0]["messages"] == [{"role": "user", "content": module.CASES[0]["email"]}]
    assert sent[0]["output_config"] == {"effort": "low"}
    assert answer.value == "phishing"
    assert (answer.model, answer.input_tokens, answer.output_tokens) == (module.NEWER, 120, 40)
    input_rate, output_rate = module.RATES_PER_MILLION[module.NEWER]
    assert answer.cost_usd == pytest.approx((120 * input_rate + 40 * output_rate) / 1e6)


async def test_the_older_model_is_sent_no_effort(monkeypatch: pytest.MonkeyPatch) -> None:
    anthropic = pytest.importorskip("anthropic")
    module = _load()
    sent = _fake_sdk(monkeypatch, anthropic, _reply(anthropic, module.OLDER, "legit"))

    await module.claude_classifier(module.OLDER)(module.CASES[0])

    assert sent[0]["model"] == module.OLDER and "output_config" not in sent[0]


async def test_with_an_api_key_every_call_s_spend_reaches_its_arm(monkeypatch: pytest.MonkeyPatch) -> None:
    anthropic = pytest.importorskip("anthropic")
    module = _load()
    labels = {case["email"]: case["label"] for case in module.CASES}
    sent = _fake_sdk(
        monkeypatch,
        anthropic,
        lambda request: _reply(anthropic, request["model"], labels[request["messages"][0]["content"]]),
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    comparison = await module.main()

    assert len(sent) == 2 * 24
    for model, summary in comparison.arms.items():
        (accuracy,) = [measure.mean for measure in summary.measures if measure.name == "match"]
        assert accuracy == 1.0
        input_rate, output_rate = module.RATES_PER_MILLION[model]
        assert summary.candidate_cost_usd == pytest.approx(24 * (120 * input_rate + 40 * output_rate) / 1e6)
