"""``examples/prompt_x_model.py`` runs a 2x2 as a newcomer would, with no API key, and reads both factors.

Offline, the four arms are its keyword stand-ins, chosen so the prompt separates on the older model and
not on the newer one. The model ids are read from the example, never spelled here.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from threetears.evals.contracts.host import CANDIDATE_MODEL_LEVER
from threetears.evals.quick import Comparison
from packages.evals.tests.test_package_matrix import REPO_ROOT, SOURCE_ROOT, public_root_violations

#: The example under test.
EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "prompt_x_model.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("prompt_x_model_example", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def test_offline_the_example_runs_four_arms_keyed_by_both_factors(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    module = _load()
    old, new = module.OLD, module.NEW
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
    assert comparison.render() in out
    assert out.rstrip().endswith(
        f"Does v2 beat v1?\n  on {old}: accuracy +0.50, improved on the control\n"
        f"  on {new}: accuracy +0.07, not separated from the control"
    )


def test_the_example_reaches_the_engine_only_through_public_roots() -> None:
    assert public_root_violations(SOURCE_ROOT, [(EXAMPLE.name, EXAMPLE)], consumer_root=REPO_ROOT) == []


async def test_the_claude_candidate_asks_each_model_as_it_accepts(monkeypatch: pytest.MonkeyPatch) -> None:
    """The older Haiku takes no effort setting; the newer one is asked at low effort. Both get the arm's prompt."""
    anthropic = pytest.importorskip("anthropic")
    module = _load()
    sent: list[dict[str, Any]] = []

    async def create(**request: Any) -> Any:
        sent.append(request)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=" Billing\n")])

    monkeypatch.setattr(anthropic, "AsyncAnthropic", lambda: SimpleNamespace(messages=SimpleNamespace(create=create)))
    case = module.CASES[0]
    assert await module.claude_classifier(module.OLD, "v1")(case) == "billing"
    assert await module.claude_classifier(module.NEW, "v2")(case) == "billing"
    old, new = sent
    assert (old["model"], old["system"], "output_config" in old) == (module.OLD, module.PROMPTS["v1"], False)
    assert (new["model"], new["system"], new["output_config"]) == (module.NEW, module.PROMPTS["v2"], {"effort": "low"})
