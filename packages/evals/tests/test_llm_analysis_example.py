"""``examples/llm_analysis.py`` freezes a campaign's bundle to a file a reload reproduces, and a writer reads only it.

Offline, the example's writer is its scripted stand-in, so the run is deterministic: the saved bundle must
reload to the fingerprint the run printed and the analysis cites, and the stand-in's refused first draft must
have gone through the generator's own check. The live client is ``examples/_live.py``'s, tested in
``test_live_example.py``; here the live path runs end to end over a faked SDK.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from threetears.evals.analysis import AnalysisContextBundle, inspect_campaign_bundle
from threetears.evals.quick import compare
from packages.evals.tests.example_loader import EXAMPLES, load_example
from packages.evals.tests.test_package_matrix import REPO_ROOT, SOURCE_ROOT, public_root_violations


def _load() -> ModuleType:
    return load_example("llm_analysis.py")


async def test_offline_the_saved_bundle_reloads_to_the_fingerprint_the_analysis_cites(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    analysis = await _load().main(tmp_path)
    out = capsys.readouterr().out

    assert out.startswith("ANTHROPIC_API_KEY is not set: running OFFLINE")
    # The code's verdict comes first, with its interval, for a reader to hold the analysis's conclusion against.
    verdict = re.search(
        r"\nCode's verdict: candidate vs baseline on Accuracy: delta \+0\.33, interval \[-[\d.]+, [\d.]+\] at 95%, "
        r"Holm-adjusted p [\d.e-]+: (.+)\n",
        out,
    )
    assert verdict is not None and verdict.group(1) == "not separated from the control"
    assert out.index("Code's verdict") < out.index("\n# ")
    saved = tmp_path / "bundle.json"
    printed = re.search(r"fingerprint ([0-9a-f]{64})", out)
    assert printed is not None and saved.is_file()
    reloaded = AnalysisContextBundle.from_json(saved.read_text()).fingerprint()
    assert reloaded == printed.group(1) == analysis.generation.bundle_fingerprint
    # The stand-in's first draft cited a reading the bundle lacks; the generator refused it and repaired once.
    assert analysis.generation.repair_attempts == 1
    assert "precision" in (analysis.generation.repaired_refusal or "")
    assert analysis.document.headline.startswith("Offline stand-in")
    # The stand-in claims no difference the code did not find.
    assert [finding.title for finding in analysis.document.findings] == ["Accuracy by arm"]
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
    path = EXAMPLES / "llm_analysis.py"
    assert public_root_violations(SOURCE_ROOT, [("llm_analysis.py", path)], consumer_root=REPO_ROOT) == []


async def test_live_the_example_generates_through_the_claude_writer(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    """The live path end to end, with the SDK answering what the stand-in would write on its second draft."""
    anthropic = pytest.importorskip("anthropic")
    module = _load()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-a-real-key")
    sent: list[dict[str, Any]] = []

    async def create(**request: Any) -> Any:
        sent.append(request)
        stand_in = module.offline_writer()
        prompt = {"system": request["system"], "user": request["messages"][0]["content"]}
        await stand_in.generate(**prompt)  # its scripted first draft is the one the check refuses; skip it
        return anthropic.types.Message.model_validate(
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": module.MODEL,
                "content": [{"type": "text", "text": (await stand_in.generate(**prompt)).content}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 30_000, "output_tokens": 2_000},
            }
        )

    async def close() -> None:
        return None

    monkeypatch.setattr(
        anthropic, "AsyncAnthropic", lambda: SimpleNamespace(messages=SimpleNamespace(create=create), close=close)
    )
    analysis = await module.main(tmp_path)

    assert capsys.readouterr().out.startswith(f"Running against Claude ({module.MODEL}).")
    (request,) = sent
    assert request["max_tokens"] == 12_000 and request["output_config"]["format"]["type"] == "json_schema"
    assert analysis.generation.generator_model == module.MODEL
    assert analysis.generation.repair_attempts == 0
    assert analysis.generation.token_cost == pytest.approx(30_000 * 0.10 / 1e6 + 2_000 * 0.50 / 1e6)
