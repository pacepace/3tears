"""``examples/compare_two_models.py`` answers "is the cheaper model good enough?" only as the evidence allows.

The package's headline question, so the example is held to the decision rule: "good enough" only on
``equivalent`` against the margin it declares; ``not separated`` means the cases could not tell the models
apart and never licenses the switch. Offline, the arms are keyword stand-ins that carry no real model's
name. Live, the SDK is faked, so no test calls the API; model ids are read off the example, never written here.
"""

from __future__ import annotations

from types import ModuleType, SimpleNamespace
from typing import Any, get_args

import pytest

from threetears.evals.analysis import ContrastOutcome
from threetears.evals.quick import Comparison
from packages.evals.tests.example_loader import load_example

NAME = "compare_two_models.py"


def _load() -> ModuleType:
    return load_example(NAME)


async def test_offline_the_stand_ins_name_no_model_and_not_separated_is_no_reason_to_switch(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    module = _load()
    comparison = await module.main()
    current, cheaper = module.STAND_INS

    assert isinstance(comparison, Comparison) and comparison.control == current
    assert list(comparison.arms) == [current, cheaper]
    for summary in comparison.arms.values():
        assert summary.status == "completed" and (summary.n_cases, summary.k_runs, summary.n_scored) == (12, 2, 24)
        assert summary.candidate_calls == 24 and summary.candidate_cost_usd is not None
    assert comparison.arms[cheaper].candidate_cost_usd < comparison.arms[current].candidate_cost_usd  # type: ignore[operator]
    # The margin is declared on accuracy, on both runs, so equivalence could be tested; no duplicate scorer is needed.
    assert "correct" not in comparison.host.profile.measures.names
    (verdict,) = comparison.verdicts("accuracy")
    assert (verdict.outcome, verdict.margin, verdict.margin_source) == ("not_separated", module.MARGIN, "run")

    (accuracy,) = comparison.contrasts("accuracy")
    assert accuracy["arm"] == cheaper and accuracy["delta"] == pytest.approx(-1 / 6)
    assert accuracy["interval"] is not None and accuracy["outcome"] == "not_separated"
    (spend,) = comparison.contrasts("production_replicating_cost")
    assert spend["outcome"] == "improved" and spend["verdict"] == "improved on the control"

    out = capsys.readouterr().out
    assert out.startswith("ANTHROPIC_API_KEY is not set: running OFFLINE")
    assert "claude" not in out.lower() and "haiku" not in out.lower(), "a stand-in's output reads as a model's"
    assert f"{cheaper} vs {current} on Accuracy: 1 -> 0.833, delta -0.167, interval [" in out
    assert "\nOn accuracy, not separated: these cases could not tell the models apart" in out
    assert "Keep the current model" in out
    assert f"    {current} said phishing, phishing; {cheaper} said legit, legit\n" in out


def test_each_verdict_has_its_decision_and_only_improved_or_equivalent_switch() -> None:
    module = _load()
    assert set(module.WHAT_TO_DO) == set(get_args(ContrastOutcome)), "one decision per typed contrast outcome"
    assert "good enough" in module.WHAT_TO_DO["equivalent"]
    assert "does not show the cheaper one is good enough" in module.WHAT_TO_DO["not_separated"]
    assert "switch," not in module.WHAT_TO_DO["not_separated"]


def _fake_sdk(monkeypatch: pytest.MonkeyPatch, anthropic: ModuleType, labels: dict[str, str]) -> list[dict[str, Any]]:
    """Point ``anthropic.AsyncAnthropic`` at a client answering every email with its right label."""
    sent: list[dict[str, Any]] = []

    async def create(**request: Any) -> Any:
        sent.append(request)
        return anthropic.types.Message.model_validate(
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": request["model"],
                "content": [{"type": "text", "text": labels[request["messages"][0]["content"]]}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 120, "output_tokens": 40},
            }
        )

    async def close() -> None:
        return None

    monkeypatch.setattr(
        anthropic, "AsyncAnthropic", lambda: SimpleNamespace(messages=SimpleNamespace(create=create), close=close)
    )
    return sent


async def test_live_two_models_alike_on_twelve_emails_are_still_not_shown_good_enough(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    anthropic = pytest.importorskip("anthropic")
    module = _load()
    live = load_example("_live.py")
    sent = _fake_sdk(monkeypatch, anthropic, {case["email"]: case["label"] for case in module.CASES})
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    comparison = await module.main()

    assert len(sent) == 48, "12 emails x 2 repeats x 2 models, as the docstring says"
    assert {request["model"] for request in sent} == set(module.LIVE)
    for model, summary in comparison.arms.items():
        input_rate, output_rate = live.PRICES[model]
        assert summary.candidate_cost_usd == pytest.approx(24 * (120 * input_rate + 40 * output_rate) / 1e6)
    (accuracy,) = comparison.contrasts("accuracy")
    assert accuracy["delta"] == 0 and accuracy["outcome"] == "not_separated"
    assert "\nOn accuracy, not separated" in capsys.readouterr().out
