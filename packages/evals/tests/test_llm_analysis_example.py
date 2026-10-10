"""``examples/llm_analysis.py`` freezes a campaign's bundle to a file a reload reproduces, and its writer reads the real SDK.

Offline, the example's writer is its scripted stand-in, so the run is deterministic: the saved bundle must
reload to the fingerprint the run printed and both analyses cite, and the stand-in's refused first draft
must have gone through the generator's own check. The live writer is exercised against replies built from
the SDK's own types, so a field the SDK renames turns this red rather than the first live run.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from threetears.evals.analysis import AnalysisContextBundle, inspect_campaign_bundle
from threetears.evals.quick import compare
from packages.evals.tests.test_package_matrix import REPO_ROOT, SOURCE_ROOT, public_root_violations

#: The example under test.
LLM_ANALYSIS = Path(__file__).resolve().parents[1] / "examples" / "llm_analysis.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("llm_analysis_example", LLM_ANALYSIS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def test_offline_the_saved_bundle_reloads_to_the_fingerprint_both_analyses_cite(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    first, second = await _load().main(tmp_path)
    out = capsys.readouterr().out

    assert out.startswith("ANTHROPIC_API_KEY is not set: running OFFLINE")
    # The code's verdict comes once, before either analysis, for a reader to hold each conclusion against.
    verdict = re.search(r"\nCode's verdict: candidate vs baseline on Accuracy: \+0\.33 \(p=[\d.e-]+\): (.+)\n", out)
    assert verdict is not None and verdict.group(1) == "not separated from the control"
    assert out.index("Code's verdict") < out.index("\n# ")
    saved = tmp_path / "bundle.json"
    assert saved.is_file()
    printed = re.search(r"fingerprint ([0-9a-f]{64})", out)
    assert printed is not None
    reloaded = AnalysisContextBundle.from_json(saved.read_text()).fingerprint()
    assert reloaded == printed.group(1)
    assert first.generation.bundle_fingerprint == second.generation.bundle_fingerprint == reloaded
    assert "fingerprint matches" in out
    # Only the prompt moved between the two analyses.
    assert first.generation.prompt_version != second.generation.prompt_version
    # The stand-in's first draft cited a reading the bundle lacks; the generator refused it and repaired once.
    for analysis in (first, second):
        assert analysis.generation.repair_attempts == 1
        assert "precision" in (analysis.generation.repaired_refusal or "")
        assert analysis.document.headline.startswith("Offline stand-in")
        assert analysis.generation.token_cost == 0.0


async def test_assembling_the_same_campaign_twice_gives_the_same_fingerprint() -> None:
    module = _load()
    arms = {"baseline": module.keyword_classifier(module.BASELINE_RULES)}
    arms["candidate"] = module.keyword_classifier(module.CANDIDATE_RULES)
    comparison = await compare(
        module.CASES, arms, expected=lambda case: case["queue"], control="baseline", scope_id="determinism", k=2
    )
    once, twice = (
        inspect_campaign_bundle(comparison.host, comparison.campaign_id, comparison.scope_id).bundle for _ in range(2)
    )
    assert once.fingerprint() == twice.fingerprint()
    assert AnalysisContextBundle.from_json(once.to_json(indent=2)).fingerprint() == once.fingerprint()


def test_the_example_reaches_the_engine_only_through_public_roots() -> None:
    assert public_root_violations(SOURCE_ROOT, [("llm_analysis.py", LLM_ANALYSIS)], consumer_root=REPO_ROOT) == []


def _reply(anthropic: Any, model: str, text: str, *, stop_reason: str = "end_turn") -> Any:
    return anthropic.types.Message.model_validate(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [{"type": "text", "text": text}],
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {"input_tokens": 30_000, "output_tokens": 2_000},
        }
    )


async def test_the_claude_writer_sends_the_contract_as_a_structured_output_and_prices_the_reply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    anthropic = pytest.importorskip("anthropic")
    module = _load()
    # The id is the example's own, read from it, so a model rev in the example is the only edit.
    served = f"{module.MODEL}-served"
    sent: list[dict[str, Any]] = []

    async def create(**request: Any) -> Any:
        sent.append(request)
        return _reply(anthropic, served, '{"ok": true}', stop_reason="max_tokens")

    monkeypatch.setattr(anthropic, "AsyncAnthropic", lambda: SimpleNamespace(messages=SimpleNamespace(create=create)))
    schema = {"type": "object", "properties": {}, "additionalProperties": False}
    contract = {"type": "json_schema", "json_schema": {"name": "analysis", "strict": True, "schema": schema}}
    completion = await module.claude_writer().generate(system="be brief", user="hello", response_format=contract)

    (request,) = sent
    assert (request["model"], request["max_tokens"], request["system"]) == (module.MODEL, module.MAX_TOKENS, "be brief")
    assert request["output_config"] == {"effort": "low", "format": {"type": "json_schema", "schema": schema}}
    assert completion.cost_usd == pytest.approx(30_000 * 0.10 / 1e6 + 2_000 * 0.50 / 1e6)
    assert (completion.input_tokens, completion.output_tokens) == (30_000, 2_000)
    assert (completion.content, completion.stop_reason) == ('{"ok": true}', "max_tokens")
    assert (completion.model, completion.served_model) == (module.MODEL, served)


async def test_live_the_example_generates_through_the_claude_writer(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """The live path end to end, with the SDK answering what the stand-in would write."""
    anthropic = pytest.importorskip("anthropic")
    module = _load()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-a-real-key")

    async def create(**request: Any) -> Any:
        stand_in, prompt = (
            module.offline_writer(),
            {"system": request["system"], "user": request["messages"][0]["content"]},
        )
        await stand_in.generate(**prompt)  # its scripted first draft is the one the check refuses; skip it
        return _reply(anthropic, module.MODEL, (await stand_in.generate(**prompt)).content)

    monkeypatch.setattr(anthropic, "AsyncAnthropic", lambda: SimpleNamespace(messages=SimpleNamespace(create=create)))
    first, second = await module.main(tmp_path)

    assert capsys.readouterr().out.startswith(f"Running against Claude ({module.MODEL}).")
    for analysis in (first, second):
        assert analysis.generation.generator_model == module.MODEL
        assert analysis.generation.repair_attempts == 0
        assert analysis.generation.token_cost == pytest.approx(30_000 * 0.10 / 1e6 + 2_000 * 0.50 / 1e6)
