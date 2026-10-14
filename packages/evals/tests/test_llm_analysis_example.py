"""``examples/llm_analysis.py`` freezes a campaign's bundle to a file a reload reproduces, and a writer reads only one.

The campaign's classifiers are keyword rules, so it is deterministic in everything but its ids and times. The
writer's first draft is canned, and cites a reading the bundle lacks, so the generator's own check refuses it; the
model writes the repair. With no API key the example replays its recording, and a recording answers only the
prompt it was made for, which is the bundle it was made from: so a replay hands the writer that bundle, kept beside
the recording, and the analysis cites its fingerprint, not this run's.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import ModuleType

import pytest

from threetears.evals.analysis import AnalysisContextBundle, inspect_campaign_bundle
from threetears.evals.quick import compare
from packages.evals.tests.example_loader import EXAMPLES, load_example, replaying
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.test_package_matrix import REPO_ROOT, SOURCE_ROOT, public_root_violations


def _load() -> ModuleType:
    return load_example("llm_analysis.py")


async def test_with_no_api_key_the_writer_repairs_the_canned_draft_from_the_recorded_bundle(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    module = _load()
    replay = replaying(module, monkeypatch)
    analysis = await module.main(tmp_path)
    out = capsys.readouterr().out

    assert replay.served() == 1, "the repair: the first draft is canned"
    assert out.startswith("ANTHROPIC_API_KEY is not set: replaying recordings/llm_analysis.jsonl, Claude (")
    # The code's verdict comes first, with its interval, for a reader to hold the analysis's conclusion against.
    verdict = re.search(
        r"\nCode's verdict: candidate vs baseline on Accuracy: delta \+0\.33, interval \[-[\d.]+, [\d.]+\] at 95%, "
        r"Holm-adjusted p [\d.e-]+: (.+)\n",
        out,
    )
    assert verdict is not None and verdict.group(1) == "not separated from the control"
    assert out.index("Code's verdict") < out.index("\n- ")  # before the first finding the writer wrote
    # This run's bundle is frozen and reloads to the fingerprint printed; the writer read the recorded one.
    saved = tmp_path / "bundle.json"
    printed = re.search(r"fingerprint ([0-9a-f]{64})", out)
    assert printed is not None and saved.is_file()
    assert AnalysisContextBundle.from_json(saved.read_text()).fingerprint() == printed.group(1)
    recorded = AnalysisContextBundle.from_json(module.RECORDED_BUNDLE.read_text()).fingerprint()
    assert analysis.generation.bundle_fingerprint == recorded
    assert f"Replaying: the writer reads {module.RECORDED_BUNDLE.name}" in out
    # The canned first draft cited a reading the bundle lacks; the generator refused it, and the model repaired it.
    assert analysis.generation.repair_attempts == 1
    assert "precision" in (analysis.generation.repaired_refusal or "")
    assert analysis.generation.generator_model == module.MODEL
    assert f"\nRead bundle {recorded[:12]} under prompt " in out


def test_the_canned_first_draft_cites_a_reading_the_campaign_never_measured() -> None:
    module = _load()
    draft = module.FIRST_DRAFT.read_text(encoding="utf-8")
    assert "|precision|measure|mean}}" in draft and "classifier:precision" not in draft


async def test_assembling_the_same_campaign_twice_gives_the_same_fingerprint() -> None:
    module = _load()
    arms = {"baseline": module.keyword_classifier(module.BASELINE_RULES)}
    arms["candidate"] = module.keyword_classifier(module.CANDIDATE_RULES)
    comparison = await compare(
        module.CASES, arms, expected=lambda case: case["queue"], control="baseline", scope_id="determinism", k=2
    )
    once, twice = (
        off_loop(inspect_campaign_bundle, comparison.host, comparison.campaign_id, comparison.scope_id).bundle
        for _ in range(2)
    )
    assert once.fingerprint() == twice.fingerprint()
    assert AnalysisContextBundle.from_json(once.to_json(indent=2)).fingerprint() == once.fingerprint()


def test_the_example_reaches_the_engine_only_through_public_roots() -> None:
    path = EXAMPLES / "llm_analysis.py"
    assert public_root_violations(SOURCE_ROOT, [("llm_analysis.py", path)], consumer_root=REPO_ROOT) == []
