"""``examples/guardrails.py`` declares a guardrail, reads held and breached per arm, and never ships a breach.

The example is the quick path's guardrail rung: a prompt that answers more questions and repeats the card's digits
is shown improved on the capability and breached on the guardrail, and its decision is "do not ship". Offline, the
stand-ins carry no real model's name; live, the SDK is faked, so no test calls the API, and the model id is read
off the example, never written here.
"""

from __future__ import annotations

from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from packages.evals.tests.example_loader import load_example
from threetears.evals.quick import Comparison

NAME = "guardrails.py"


def _load() -> ModuleType:
    return load_example(NAME)


async def test_offline_the_arm_that_gained_and_breached_is_never_the_one_to_ship(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    module = _load()
    comparison = await module.main()

    assert isinstance(comparison, Comparison) and list(comparison.arms) == ["current", "friendlier", "careful"]
    for summary in comparison.arms.values():
        assert summary.status == "completed" and (summary.n_cases, summary.k_runs, summary.n_scored) == (50, 1, 50)
    gains = {row["arm"]: row["verdict"] for row in comparison.contrasts("gives_status")}
    assert gains == {"friendlier": "improved on the control", "careful": "improved on the control"}
    assert comparison.contrasts("keeps_card_private") == [], "the guardrail is in no contrast"
    outcomes = {row["arm"]: row["outcome"] for row in comparison.guardrails()}
    assert outcomes == {"friendlier": "breached", "careful": "held"}
    assert comparison.guardrail_standing("friendlier").breached == ["keeps_card_private"]

    out = capsys.readouterr().out
    assert out.startswith("ANTHROPIC_API_KEY is not set: running OFFLINE")
    assert "claude" not in out.lower() and "haiku" not in out.lower(), "a stand-in's output reads as a model's"
    assert "  friendlier: 1.00 -> 0.66, interval [-0.5153, -0.2238] at 95%: breached\n" in out
    assert "  careful: 1.00 -> 1.00, interval [-0.08106, 0.08106] at 95%: held\n" in out
    assert "  friendlier: improved on the control; guardrail breached: do not ship it, whatever it gained.\n" in out
    assert "  careful: improved on the control; guardrail held: a candidate to ship.\n" in out
    assert "card ending" in out.split("Where the guardrail broke")[1], "the example ends by reading a breach"
    assert "Arm friendlier breached the guardrail Keeps card private score" in comparison.render()


def test_only_a_held_guardrail_with_a_gain_is_a_candidate_to_ship() -> None:
    module = _load()
    for outcome in ("improved", "not_separated", "regressed"):  # typed outcomes, never the printed words
        assert module.what_to_do(outcome, "breached") == "do not ship it, whatever it gained."
        assert "not known to be safe" in module.what_to_do(outcome, "undecided")
    assert module.what_to_do("improved", "held") == "a candidate to ship."
    assert "keep the current prompt" in module.what_to_do("not_separated", "held")
    assert "keep the current prompt" in module.what_to_do("improved on the control", "held"), "words are no outcome"


def _fake_sdk(monkeypatch: pytest.MonkeyPatch, anthropic: ModuleType) -> list[dict[str, Any]]:
    """Point ``anthropic.AsyncAnthropic`` at a client that states the record's status and nothing else."""
    sent: list[dict[str, Any]] = []

    async def create(**request: Any) -> Any:
        sent.append(request)
        status = request["messages"][0]["content"].split(": ", 1)[1].split(".", 1)[0]
        return anthropic.types.Message.model_validate(
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": request["model"],
                "content": [{"type": "text", "text": f"Your order is {status}."}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 60, "output_tokens": 12},
            }
        )

    async def close() -> None:
        return None

    monkeypatch.setattr(
        anthropic, "AsyncAnthropic", lambda: SimpleNamespace(messages=SimpleNamespace(create=create), close=close)
    )
    return sent


async def test_live_three_prompts_alike_hold_the_guardrail_and_ship_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    anthropic = pytest.importorskip("anthropic")
    module = _load()
    sent = _fake_sdk(monkeypatch, anthropic)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    comparison = await module.main()

    assert len(sent) == 150, "50 questions x 3 prompts, as the docstring says"
    assert {request["model"] for request in sent} == {module.MODEL}
    assert {row["arm"]: row["outcome"] for row in comparison.guardrails()} == {"friendlier": "held", "careful": "held"}
    assert all(row["verdict"].startswith("not separated") for row in comparison.contrasts("gives_status"))
