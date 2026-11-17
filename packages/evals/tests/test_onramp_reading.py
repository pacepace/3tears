"""The on-ramp's comparison as the package reads it, with the app as the case, and the plan a live comparison runs.

Offline and free: every reading replays stored-shaped trial rows through ``threetears.evals``'s front door, so no
agent runs and no model is called. ``test_onramp_end_to_end.py`` reads a real scripted run the same way.
"""

from __future__ import annotations

import asyncio
import json
import random
from pathlib import Path
from typing import Any

import pytest

from packages.evals.onramp.campaign import CASES, TRANSCRIPTS, clopper_pearson, main
from packages.evals.onramp.plan import PLANS, Plan, PlannedArm, load_plan, plan_refusal, wheel_dir
from packages.evals.onramp.reading import (
    CHECKS,
    PRIMARY,
    SEPARATED,
    Unreadable,
    design_of,
    read,
    reading_lines,
    simulated_rows,
)
from packages.evals.onramp.sandbox import APPS, TWINS

#: The level every comparison the package reads holds its error to, over its family.
ALPHA = 0.05


def _rows(tallies: dict[str, dict[str, tuple[int, int]]], *, tier: str | None = None) -> list[dict[str, Any]]:
    """Stored-shaped rows: ``{arm: {app: (passed, of)}}`` on the primary check, every other check passed."""
    rows: list[dict[str, Any]] = []
    for arm, apps in tallies.items():
        for app, (passed, of) in apps.items():
            for repeat in range(1, of + 1):
                grade = 1.0 if repeat <= passed else 0.0
                scores = {name: grade if name == PRIMARY else 1.0 for name in CHECKS}
                row = {"arm": arm, "app": app, "repeat": repeat, "state": "passed" if grade else "failed"}
                rows.append({**row, "scores": scores, "record": {"app": app}, **({"tier": tier} if tier else {})})
    return rows


async def test_a_stored_run_is_read_by_the_package_with_each_app_one_case_and_its_trials_averaged() -> None:
    tallies = {
        "docs@control": {"recipes": (1, 4), "holds": (0, 4), "glossator": (2, 4)},
        "docs@candidate": {"recipes": (3, 4), "holds": (2, 4), "glossator": (2, 4)},
    }
    result = await read(_rows(tallies))

    (row,) = result.contrasts(PRIMARY)
    assert row["arm"] == "docs@candidate" and row["cases"] == "3 paired", "three apps, never twelve trials"
    assert row["control_mean"] == pytest.approx(100 * (1 / 4 + 0 + 2 / 4) / 3)
    assert row["arm_mean"] == pytest.approx(100 * (3 / 4 + 2 / 4 + 2 / 4) / 3)
    assert row["outcome"] not in SEPARATED
    assert result.render().splitlines()[0].endswith("too few cases to call a difference."), "the package's verdict"
    assert {arm: len(result.results(arm)) for arm in result.arms} == {"docs@control": 12, "docs@candidate": 12}


async def test_a_trial_is_replayed_as_the_run_ended_it_lost_or_failed() -> None:
    rows = _rows({"a": {"x": (1, 2), "y": (1, 2)}, "b": {"x": (2, 2), "y": (0, 2)}})
    lost, raised = [row for row in rows if row["arm"] == "b" and row["app"] == "y"]
    lost["scores"], lost["state"] = {}, "lost"
    raised["scores"] = {}  # its runner raised: the run counted it failed, against its arm

    result = await read(rows)

    states = sorted(trial.state for trial in result.results("b"))
    assert states == ["failed", "lost", "passed", "passed"], "a lost trial is in no app's share; a failed one is"
    (row,) = result.contrasts(PRIMARY)
    assert row["arm_mean"] == pytest.approx(100 * (2 / 2 + 0 / 1) / 2), "y's share is over its one usable trial"


def test_trials_of_two_tiers_are_never_pooled_and_each_tier_is_read_on_its_own() -> None:
    each = {"control": {"x": (1, 2), "y": (0, 2)}, "candidate": {"x": (2, 2), "y": (1, 2)}}
    rows = _rows(each, tier="cheap") + _rows(each, tier="ceiling")
    with pytest.raises(Unreadable, match="never pooled"):
        design_of(rows)

    said = reading_lines(rows)

    heads = [line for line in said if line.startswith("tier ")]
    assert [line.split(":")[0] for line in heads] == ["tier cheap", "tier ceiling"]
    assert all("2 app(s), 2 trial(s) per arm on each" in line for line in heads)
    assert sum(1 for line in said if line.startswith(f"  {PRIMARY} (primary)")) == 2


def test_a_run_whose_arms_ran_different_counts_of_trials_or_different_apps_is_refused_by_name() -> None:
    with pytest.raises(Unreadable, match="k is one number"):
        design_of(_rows({"a": {"x": (1, 2), "y": (1, 3)}, "b": {"x": (1, 2), "y": (1, 2)}}))
    with pytest.raises(Unreadable, match="b ran no trial on y"):
        design_of(_rows({"a": {"x": (1, 2), "y": (1, 2)}, "b": {"x": (1, 2)}}))


def test_the_reading_states_every_check_the_primary_first_and_each_apps_tally_per_arm() -> None:
    rows = _rows({"docs@control": {"x": (0, 3), "y": (1, 3)}, "docs@candidate": {"x": (3, 3), "y": (2, 3)}})

    said = reading_lines(rows, primary="eval_runs")

    checks = [line.split(" (")[0].strip() for line in said if line.startswith("  ") and not line.startswith("    ")]
    assert checks == ["eval_runs", PRIMARY, "read_right"]
    assert "    by app (docs@control, docs@candidate): x 0 of 3, 3 of 3; y 1 of 3, 2 of 3" in said


@pytest.mark.timeout(900)
def test_with_apps_as_cases_arms_that_differ_only_by_noise_read_separated_at_most_alpha_and_pooled_trials_do_not() -> (
    None
):
    """The error rate the reading holds, simulated over the real path (stored rows replayed through the package).

    The arms differ only by noise: each app's base share is drawn once, and each arm's share on each app moves from it
    by an arm-by-app draw (sd 0.35) a binomial over trials has no term for, as two runs of one design on one app
    disagreed in the run this harness's history records. Read with the app as the case, the arms separate at most
    alpha of the time: the upper end of the exact 95% range on that share is at most alpha. Read with each trial as
    its own case (the arm-by-app draw then counted k times over), the same rows separate more than alpha of the time:
    the lower end of that range is above alpha, so this test fails a reading that pools trials.
    """
    apps, k, spread = 8, 8, 0.35

    def separated(rows: list[dict[str, Any]]) -> bool:
        result = asyncio.run(read(rows))
        return any(row["outcome"] in SEPARATED for row in result.contrasts(PRIMARY))

    rng = random.Random(20261117)
    as_cases = [separated(simulated_rows(rng, apps=apps, k=k, effect=0.0, spread=spread)) for _ in range(150)]
    low, high = clopper_pearson(sum(as_cases), len(as_cases))
    assert high <= ALPHA, (sum(as_cases), len(as_cases), low, high)

    rng = random.Random(20261118)
    pooled = []
    for _ in range(60):
        rows = simulated_rows(rng, apps=apps, k=k, effect=0.0, spread=spread)
        pooled.append(separated([{**row, "app": f"{row['app']}#{row['repeat']}"} for row in rows]))
    low, high = clopper_pearson(sum(pooled), len(pooled))
    assert low > ALPHA, (sum(pooled), len(pooled), low, high)


def _plan(**changed: Any) -> Plan:
    fields: dict[str, Any] = {
        "path": PLANS / "held-out-1.toml",
        "name": "test",
        "question": "Does it?",
        "arms": (PlannedArm("docs@control", "f9ab6f74d"), PlannedArm("docs@candidate", "8d433f44e")),
        "apps": ("recipes", "holds"),
        "k": 2,
        "tiers": {"cheap": 1.0, "ceiling": 2.0},
        "primary": PRIMARY,
        "secondary": ("eval_runs", "read_right"),
    }
    return Plan(**{**fields, **changed})


def test_a_live_run_on_a_plan_is_refused_when_the_plan_or_the_command_line_differs(tmp_path: Path) -> None:
    known = ["recipes", "holds", "glossator"]
    control = wheel_dir(tmp_path, "f9ab6f74d") / "3tears_evals-0.0.0-py3-none-any.whl"
    agreed: dict[str, Any] = {
        "commit": "0" * 40,
        "wheels": {"docs@control": str(control)},
        "apps": [],
        "k": None,
        "tier": None,
        "cap_usd": 3.0,
        "known_apps": known,
        "workdir": tmp_path,
    }
    assert plan_refusal(_plan(), **agreed) is None

    def refused(plan: Plan, **changed: Any) -> str:
        why = plan_refusal(plan, **{**agreed, **changed})
        assert why is not None, changed
        return why

    assert "not committed" in refused(_plan(), commit=None)
    assert "placeholders" in refused(_plan(apps=("recipes", "PLACEHOLDER-2")))
    assert "does not have (atlas)" in refused(_plan(apps=("recipes", "atlas")))
    assert "run the plan's apps" in refused(_plan(), apps=["recipes"])
    assert "the plan's k is 2" in refused(_plan(), k=3)
    assert "every tier it names" in refused(_plan(), tier="cheap")
    assert "--cap 3.00" in refused(_plan(), cap_usd=4.5)
    assert "--cap 3.00" in refused(_plan(), cap_usd=None)
    assert "not an arm of the plan" in refused(_plan(), wheels={"docs@other": str(control)})
    elsewhere = tmp_path / "mine" / "3tears_evals-0.0.0-py3-none-any.whl"
    assert "not the one built from the plan's f9ab6f74d" in refused(_plan(), wheels={"docs@control": str(elsewhere)})
    other_ref = wheel_dir(tmp_path, "8d433f44e") / "3tears_evals-0.0.0-py3-none-any.whl"
    assert "not the one built" in refused(_plan(), wheels={"docs@control": str(other_ref)}), "another arm's wheel"


def test_the_committed_plan_states_its_arms_tiers_check_and_k_and_its_apps_are_held_out() -> None:
    plan = load_plan(PLANS / "held-out-1.toml")
    assert [arm.ref for arm in plan.arms] == ["f9ab6f74d", "8d433f44e"] and plan.arms[0].name == "docs@control"
    assert list(plan.tiers) == ["cheap", "ceiling"] and plan.primary == PRIMARY
    assert sorted((plan.primary, *plan.secondary)) == sorted(CHECKS)
    development = {"recipes", "holds", "glossator"}
    assert not development & set(plan.apps), "the development set's apps are never the held-out set's"


def test_a_plan_with_placeholders_quotes_but_a_live_run_on_it_is_refused_before_anything_is_built(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    plan = load_plan(PLANS / "held-out-1.toml")
    argv = ["--plan", str(plan.path), "--workdir", str(tmp_path)]
    assert main([*argv, "--dry-run"]) == 0
    said = capsys.readouterr().out
    assert said.count("tier ") == len(plan.tiers) and f"at most ${plan.cap_usd:.2f} in all" in said
    monkeypatch.setenv("ONRAMP_LIVE", "1")
    monkeypatch.delenv("RECORD", raising=False)
    assert main([*argv, "--cap", f"{plan.cap_usd:.2f}"]) == 2
    assert "the plan" in capsys.readouterr().err
    assert not (tmp_path / "wheels").exists() and not (tmp_path / "images").exists(), "nothing built"


def test_a_stored_run_of_two_tiers_is_read_from_its_directories_tier_by_tier(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    each = {"docs@control": {"x": (1, 2), "y": (0, 2)}, "docs@candidate": {"x": (2, 2), "y": (1, 2)}}
    for tier in ("cheap", "ceiling"):
        (tmp_path / tier).mkdir()
        lines = "".join(json.dumps(row) + "\n" for row in _rows(each))  # rows before tiers were stamped
        (tmp_path / tier / "trials.jsonl").write_text(lines, encoding="utf-8")

    assert main(["--read", str(tmp_path)]) == 0

    said = capsys.readouterr().out.splitlines()
    assert [line.split(":")[0] for line in said if line.startswith("tier ")] == ["tier ceiling", "tier cheap"]
    assert main(["--read", str(tmp_path / "nothing")]) == 2


def test_every_app_has_its_twin_and_a_scripted_transcript_the_pre_flight_runs() -> None:
    """Adding an app is a line in apps.jsonl, the app and its twin, and a transcript that succeeds on it (README)."""
    apps = [json.loads(line)["app"] for line in CASES.read_text(encoding="utf-8").splitlines() if line.strip()]
    for app in apps:
        assert (APPS / app / "README.md").is_file(), app
        assert any((TWINS / app).iterdir()), f"{app} has no fixed twin"
        transcript = TRANSCRIPTS / f"{app}.jsonl"
        assert transcript.is_file(), f"{app} has no transcripts/{app}.jsonl"
        acts = [json.loads(line) for line in transcript.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert acts[-1]["tool"] == "done" and "{first_line}" in acts[-1]["text"], app
        assert any(act["tool"] == "run" and "threetears.evals" in act["command"] for act in acts), app
