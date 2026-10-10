"""``examples/prompt_x_model.py`` runs a 2x2 as a newcomer would, with no API key, and reads both factors.

Offline, the four arms are its keyword stand-ins, named as stand-ins and chosen so the prompt separates on the
older model and not on the newer one. The arms are read by their keys, never by the words a report names them by.
The model ids are read from the example, never spelled here.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from threetears.evals.kernel.host import CANDIDATE_MODEL_LEVER
from threetears.evals.quick import Comparison
from packages.evals.tests.example_loader import load_example
from packages.evals.tests.test_package_matrix import REPO_ROOT, SOURCE_ROOT, public_root_violations

#: The example under test.
EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "prompt_x_model.py"


def _load() -> ModuleType:
    return load_example(EXAMPLE.name)


async def test_offline_the_example_runs_four_arms_keyed_by_both_factors(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    module = _load()
    old, new = module.STAND_INS
    comparison = await module.main()
    assert isinstance(comparison, Comparison)
    assert comparison.factors == ("model", "prompt")
    assert list(comparison.arms) == [(old, "v1"), (old, "v2"), (new, "v1"), (new, "v2")]
    assert comparison.control == (old, "v1")
    assert all(summary.status == "completed" for summary in comparison.arms.values())
    assert {summary.candidate_model for summary in comparison.arms.values()} == {old, new}

    campaign = comparison.host.storage.load_campaign(comparison.campaign_id, comparison.scope_id)
    assert campaign is not None and campaign.declared_design is not None
    axes = {axis.axis_id: [value.display for value in axis.values] for axis in campaign.declared_design.axes}
    assert axes == {CANDIDATE_MODEL_LEVER: [old, new], "callable.prompt": ["v1", "v2"]}

    # Against v1 on the older model: the prompt separates there, and every arm is named by both coordinates.
    rows = {row["contrast"]: row for row in comparison.contrasts("accuracy")}
    control = f"callable.prompt=v1, model={old}"
    assert set(rows) == {f"callable.prompt={p}, model={m}" for m, p in comparison.arms} - {control}
    assert {row["control"] for row in rows.values()} == {control}
    assert rows[f"callable.prompt=v2, model={old}"]["verdict"] == "improved on the control"
    assert rows[f"callable.prompt=v1, model={new}"]["verdict"] == "improved on the control"

    # Against v1 on the newer model: the same runs, and the prompt does not separate there.
    on_new = {row["contrast"]: row for row in comparison.against((new, "v1")).contrasts("accuracy")}
    assert on_new[f"callable.prompt=v2, model={new}"]["verdict"] == "not separated from the control"

    out = capsys.readouterr().out
    assert out.startswith("ANTHROPIC_API_KEY is not set: running OFFLINE, with keyword stand-ins")
    assert "claude" not in out.lower() and "haiku" not in out.lower(), "a stand-in's output reads as a model's"
    # The verdict lines, each with its interval, and a pointer to the full report rather than the report itself.
    assert comparison.render() not in out
    assert re.search(
        rf"\nOn {old}, v2 vs v1 on Accuracy: 0\.50 -> 1\.00, delta \+0\.50, interval \[[\d.]+, [\d.]+\] at [\d.]+%, "
        rf"Holm-adjusted p [\d.e-]+: improved on the control\n",
        out,
    )
    assert re.search(
        rf"\nOn {new}, v2 vs v1 on Accuracy: 0\.93 -> 1\.00, delta \+0\.07, interval \[-[\d.]+, [\d.]+\] at [\d.]+%, "
        rf"Holm-adjusted p [\d.e-]+: not separated from the control\n"
        rf"  'Could you add an option to pay by invoice\?' \(feature_request\): v1 said billing, billing; "
        rf"v2 feature_request, feature_request\n",
        out,
    )
    assert out.rstrip().endswith("The full report: print(comparison.render()), or reports.py to write it to files.")


def test_the_example_reaches_the_engine_only_through_public_roots() -> None:
    assert public_root_violations(SOURCE_ROOT, [(EXAMPLE.name, EXAMPLE)], consumer_root=REPO_ROOT) == []


async def test_the_claude_candidate_sends_each_arm_s_prompt_to_its_model(monkeypatch: pytest.MonkeyPatch) -> None:
    anthropic = pytest.importorskip("anthropic")
    module = _load()
    sent: list[dict[str, Any]] = []

    async def create(**request: Any) -> Any:
        sent.append(request)
        return anthropic.types.Message.model_validate(
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": request["model"],
                "content": [{"type": "text", "text": " Billing\n"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 50, "output_tokens": 2},
            }
        )

    async def close() -> None:
        return None

    monkeypatch.setattr(
        anthropic, "AsyncAnthropic", lambda: SimpleNamespace(messages=SimpleNamespace(create=create), close=close)
    )
    older, newer = module.LIVE
    case = module.CASES[0]
    assert await module.claude_classifier(older, "v1")(case) == "billing"
    assert await module.claude_classifier(newer, "v2")(case) == "billing"
    assert [(request["model"], request["system"]) for request in sent] == [
        (older, module.PROMPTS["v1"]),
        (newer, module.PROMPTS["v2"]),
    ]
