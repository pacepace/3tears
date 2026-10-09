"""``examples/cassettes.py`` runs as a newcomer would run it, and shows what it says it shows.

It captures the search once, live, then replays that recording for two arms: neither arm calls the
search, and both are served the same results for every case and repeat. It needs no API key and calls
no model, so it runs here exactly as it runs for a reader.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from threetears.evals.quick import Comparison
from threetears.evals.run import get_run
from packages.evals.tests.test_package_matrix import REPO_ROOT, SOURCE_ROOT, public_root_violations

#: The example under test.
CASSETTES = Path(__file__).resolve().parents[1] / "examples" / "cassettes.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("cassettes_example", CASSETTES)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def test_the_example_captures_once_and_replays_one_recording_to_both_arms(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    module = _load()
    comparison = await module.main()
    assert isinstance(comparison, Comparison)
    out = capsys.readouterr().out
    assert "OFFLINE stand-in" in out.splitlines()[0]

    # The capture called the search once per case, and the replay never called it.
    n_cases = len(module.CASES)
    assert f"{n_cases} live search(es) recorded" in out
    assert "replayed both arms: 0 live search(es)" in out
    assert module.live_searches.total() == 0

    # Every arm's run replayed the capture's corpus, and every cell was scored.
    runs = [
        get_run(comparison.host.storage, summary.run_id, comparison.scope_id) for summary in comparison.arms.values()
    ]
    (capture_id,) = {run.cassette_corpus_id for run in runs}
    assert all(run.cassette_mode == "replay" for run in runs)
    assert get_run(comparison.host.storage, capture_id, comparison.scope_id).cassette_mode == "capture"
    for summary in comparison.arms.values():
        assert summary.status == "completed"
        assert (summary.n_scored, summary.n_excluded) == (n_cases * 2, 0)

    # Both arms were served identical results, case by case and repeat by repeat.
    served = module.served
    assert set(served) == {"top_hit", "newest_hit"}
    assert served["top_hit"] == served["newest_hit"]
    assert all(len(repeats) == 2 and repeats[0] == repeats[1] for repeats in served["top_hit"].values())
    assert "identical for every case and repeat: True" in out

    # Reading the newest page never does worse than trusting the top one on the same results.
    means = {arm: summary.measures[0].mean for arm, summary in comparison.arms.items()}
    assert means["newest_hit"] >= means["top_hit"]
    assert comparison.render() in out


def test_the_offline_search_drifts_between_calls_and_repeats_across_runs() -> None:
    first, second = _load(), _load()
    query = first.CASES[0]["query"]
    asked = [first.offline_search(query) for _ in range(6)]
    assert len({str(hits) for hits in asked}) > 1, "the stand-in must drift, or replay would show nothing"
    assert [second.offline_search(query) for _ in range(6)] == asked
    assert first.live_searches.total() == 6


def test_the_example_reaches_the_engine_only_through_public_roots() -> None:
    assert public_root_violations(SOURCE_ROOT, [("cassettes.py", CASSETTES)], consumer_root=REPO_ROOT) == []
