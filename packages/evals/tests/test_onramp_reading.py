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

from packages.evals.onramp.campaign import CASES, HELD_OUT, TIERS, TRANSCRIPTS, main, quote, registry, run_record
from packages.evals.onramp.plan import (
    PLANS,
    continuation_refusal,
    Plan,
    PlannedArm,
    committed_at,
    head_commit,
    load_plan,
    plan_refusal,
    wheel_dir,
)
from packages.evals.onramp.reading import (
    CHECKS,
    Stage,
    read_continued,
    PRIMARY,
    SEPARATED,
    Unreadable,
    design_of,
    read,
    reading_lines,
    simulated_rows,
)
from packages.evals.onramp.sandbox import APPS, TWINS
from packages.evals.tests.stats_certificate import read as read_certificate

#: The level every comparison the package reads holds its error to.
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
    # As the live run read it: the failed trial has no grade, so the check's reading has no value for y on that arm,
    # and the pass rate counts it failed.
    (check,) = result.contrasts(PRIMARY)
    assert check["cases"] == "1 vs 2" and check["arm_mean"] == pytest.approx(100.0)
    (passed,) = result.contrasts("passed")
    assert passed["cases"] == "2 paired" and passed["arm_mean"] == pytest.approx(100 * (2 / 2 + 0 / 1) / 2)


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


#: The package's certified claims under the on-ramp's null (apps as cases, three trials each, base shares from 0.1 to
#: 0.6, an arm-by-app spread of sd 0.1): a question's primary read once, or continued to five reads, separates at most
#: alpha. Counted through the package's reading by ``packages/evals/scripts/certify_stats.py``.
ONRAMP_CLAIMS = ("small-n.once.onramp", "small-n.continued.onramp.separates", "small-n.planned.onramp.separates")


def test_the_error_rate_the_reading_holds_is_the_packages_certified_claim_under_the_onramp_null() -> None:
    """The reading is the package's own (an ``Eval`` over the apps), so the error rate it holds is the package's: its
    certificate holds both on-ramp claims at alpha. ``test_stats_certificate.py`` holds the certificate current."""
    certified = {claim["id"]: claim for claim in read_certificate()["claims"]}
    for claim in ONRAMP_CLAIMS:
        assert certified[claim]["holds"] and certified[claim]["test"] == "at_most", claim
        assert certified[claim]["target"] == ALPHA and certified[claim]["figure"] <= ALPHA, claim


@pytest.mark.timeout(300)
def test_seeded_noise_only_runs_separate_with_apps_as_cases_less_often_than_with_trials_pooled() -> None:
    """A fast guard on the replay, which the certificate does not cover: it must hand the package apps as cases.

    Fifteen seeded runs whose arms differ only by noise (12 apps, 10 trials each, an arm-by-app draw of sd 0.4 that a
    binomial over trials has no term for). Read with the app as the case, none separates; read with each trial as its
    own case, the arm-by-app draw counted ten times over, three do. The counts are the ones this code gave on this seed
    under the package's reading of 2026-11-18; a replay that pooled trials reads the second count in the first place.
    """

    def separated(rows: list[dict[str, Any]]) -> bool:
        result = asyncio.run(read(rows))
        return any(row["outcome"] in SEPARATED for row in result.contrasts(PRIMARY))

    rng = random.Random(20261118)
    as_cases = pooled = 0
    for _ in range(15):
        rows = simulated_rows(rng, apps=12, k=10, effect=0.0, spread=0.4)
        as_cases += separated(rows)
        pooled += separated([{**row, "app": f"{row['app']}#{row['repeat']}"} for row in rows])
    assert (as_cases, pooled) == (0, 3)


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
        "changed": [],
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
    edited = refused(_plan(), changed=[" M packages/evals/onramp/apps/x/x.py"])
    assert "changes not committed" in edited and "apps/x/x.py" in edited, "the head commit would not be what ran"
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
    assert [arm.ref for arm in plan.arms] == ["f9ab6f74d", "1f0ec4c68"] and plan.arms[0].name == "docs@control"
    assert list(plan.tiers) == ["cheap"] and plan.primary == PRIMARY and plan.k == 3 and len(plan.apps) == 24
    assert sorted((plan.primary, *plan.secondary)) == sorted(CHECKS)
    assert plan.registries == (HELD_OUT.name,) and not plan.placeholders()
    held_out = [case["app"] for case in registry([HELD_OUT])]
    second = load_plan(PLANS / "held-out-2.toml")
    assert [*plan.apps, *second.apps] == held_out, "held-out-1 then held-out-2: every held-out app, in its order"
    assert not {case["app"] for case in registry()} & set(plan.apps), "no held-out app is in the development set"


def test_a_run_without_a_plan_never_covers_a_held_out_app(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    held_out = registry([HELD_OUT])[0]["app"]
    assert held_out not in {case["app"] for case in registry()}, "apps.jsonl, what a default run reads, lacks it"
    argv = ["--scripted", "succeeds", "--host", "--k", "1", "--apps", held_out, "--workdir", str(tmp_path)]
    assert main(argv) == 2
    assert "runs only on a plan" in capsys.readouterr().err
    assert not any(tmp_path.iterdir()), "nothing ran"


def test_a_plan_quotes_but_a_live_run_on_one_not_committed_is_refused_before_anything_is_built(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    written = (PLANS / "held-out-1.toml").read_text(encoding="utf-8")
    (tmp_path / "copy.toml").write_text(written, encoding="utf-8")  # outside the repository: never committed
    plan = load_plan(tmp_path / "copy.toml")
    argv = ["--plan", str(plan.path), "--workdir", str(tmp_path)]
    assert main([*argv, "--dry-run"]) == 0
    said = capsys.readouterr().out
    assert said.count("model: ") == len(plan.tiers) and f"at most ${plan.cap_usd:.2f} in all" in said
    monkeypatch.setenv("ONRAMP_LIVE", "1")
    monkeypatch.delenv("RECORD", raising=False)
    assert main([*argv, "--cap", f"{plan.cap_usd:.2f}"]) == 2
    assert "is not committed" in capsys.readouterr().err
    assert not (tmp_path / "wheels").exists() and not (tmp_path / "images").exists(), "nothing built"


def test_the_rows_are_read_in_the_runs_own_arm_and_app_order_whatever_order_they_come_in() -> None:
    rows = _rows({"docs@control": {"x": (0, 3), "y": (1, 3)}, "docs@candidate": {"x": (3, 3), "y": (2, 3)}})
    reordered = [row for row in rows if row["arm"] == "docs@candidate"] + [
        row for row in rows if row["arm"] == "docs@control"
    ]
    assert design_of(reordered).arms == ("docs@candidate", "docs@control"), "unguided, the rows' order decides"
    assert design_of(reordered, ["docs@control", "docs@candidate"]).arms == ("docs@control", "docs@candidate")
    with pytest.raises(Unreadable, match="not the run's"):
        design_of(rows, ["docs@control", "docs@other"])
    assert design_of(list(reversed(rows)), apps=["x", "y"]).apps == ("x", "y"), "the run's case order, not the file's"
    with pytest.raises(Unreadable, match="not the run's"):
        design_of(rows, apps=["x", "z"])


def test_run_json_records_the_plan_and_head_commits_and_the_reading_takes_its_control_from_it(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    plan = load_plan(PLANS / "held-out-1.toml")
    wheels = {arm.name: str(wheel_dir(tmp_path, arm.ref) / "w.whl") for arm in plan.arms}
    priced = {name: quote(TIERS[name], runs=8, cap_usd=cap) for name, cap in plan.tiers.items()}
    record = run_record(plan, wheels, ["x", "y"], 2, priced)
    assert record["plan"] == "packages/evals/onramp/plans/held-out-1.toml"
    assert (record["plan_commit"], record["head_commit"]) == (committed_at(plan.path), head_commit())
    assert list(record["arms"]) == ["docs@control", "docs@candidate"] and record["apps"] == ["x", "y"]
    assert [arm["ref"] for arm in record["arms"].values()] == ["f9ab6f74d", "1f0ec4c68"] and record["k"] == 2
    assert set(record) >= {"uncommitted", "question", "primary", "tiers"}
    (tmp_path / "run.json").write_text(json.dumps(record), encoding="utf-8")
    rows = _rows({"docs@control": {"x": (0, 2), "y": (0, 2)}, "docs@candidate": {"x": (2, 2), "y": (2, 2)}})
    rows.reverse()  # the candidate's rows first: the run's own order, not the file's, names the control
    (tmp_path / "trials.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    assert main(["--read", str(tmp_path)]) == 0

    said = capsys.readouterr().out.splitlines()
    assert said[0] == f"plan {record['plan']} at {record['plan_commit']}, run at {record['head_commit']}"
    assert "    by app (docs@control, docs@candidate): x 0 of 2, 2 of 2; y 0 of 2, 2 of 2" in said, said


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
    """Adding an app is a line in apps.jsonl or held-out.jsonl, the app and its twin, and a transcript that succeeds on
    it (README)."""
    apps = [case["app"] for case in registry([CASES, HELD_OUT])]
    assert len(apps) == len(set(apps)), "an app is in one registry, once"
    for app in apps:
        assert (APPS / app / "README.md").is_file(), app
        assert any((TWINS / app).iterdir()), f"{app} has no fixed twin"
        transcript = TRANSCRIPTS / f"{app}.jsonl"
        assert transcript.is_file(), f"{app} has no transcripts/{app}.jsonl"
        acts = [json.loads(line) for line in transcript.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert acts[-1]["tool"] == "done" and "{first_line}" in acts[-1]["text"], app
        assert any(act["tool"] == "run" and "threetears.evals" in act["command"] for act in acts), app


def _stage(tallies: dict[str, dict[str, tuple[int, int]]]) -> Stage:
    rows = _rows(tallies, tier="cheap")
    return Stage(tuple(rows), tuple(tallies), tuple(next(iter(tallies.values()))))


async def test_two_runs_are_read_as_one_comparison_continued_and_the_first_alone_reads_as_it_did() -> None:
    first = _stage(
        {
            "docs@control": {"a": (1, 3), "b": (0, 3), "c": (2, 3)},
            "docs@candidate": {"a": (3, 3), "b": (2, 3), "c": (2, 3)},
        }
    )
    later = _stage({"docs@control": {"d": (0, 3), "e": (1, 3)}, "docs@candidate": {"d": (3, 3), "e": (3, 3)}})
    alone = await read(list(first.rows), arms=first.arms, apps=first.apps)
    (once,) = alone.contrasts(PRIMARY)
    (again,) = (await read_continued([first])).contrasts(PRIMARY)
    assert (once["p_adjusted"], once["cases"]) == (again["p_adjusted"], again["cases"]), "one stage reads as read does"

    continued = await read_continued([first, later])

    (row,) = continued.contrasts(PRIMARY)
    assert row["cases"] == "5 paired"
    assert row["control_mean"] == pytest.approx(100 * (1 / 3 + 0 + 2 / 3 + 0 + 1 / 3) / 5)
    assert any(line.startswith("Continues the comparison run before") for line in continued.render().splitlines())
    with pytest.raises(Unreadable, match="a, b, c is in two runs"):
        await read_continued([first, first])
    with pytest.raises(Unreadable, match="arms differ"):
        await read_continued([first, Stage(later.rows, ("docs@candidate", "docs@control"), later.apps)])


def test_a_run_may_continue_another_only_on_the_same_builds_tier_k_check_and_question() -> None:
    earlier = {
        "arms": {"docs@control": {"commit": "f9"}, "docs@candidate": {"commit": "1f"}},
        "tiers": {"cheap": {"model": "m"}},
        "k": 3,
        "primary": PRIMARY,
        "question": "Q?",
        "apps": ["a", "b"],
    }
    later = {**earlier, "apps": ["c"]}
    assert continuation_refusal(earlier, later) is None
    for changed, said in (
        ({"arms": {"docs@control": {"commit": "f9"}, "docs@candidate": {"commit": "zz"}}}, "arms differ"),
        ({"arms": {"docs@candidate": {"commit": "1f"}, "docs@control": {"commit": "f9"}}}, "arms differ"),
        ({"tiers": {"cheap": {"model": "other"}}}, "tiers or their models differ"),
        ({"k": 2}, "k differs"),
        ({"primary": "eval_runs"}, "primary differs"),
        ({"question": "Other?"}, "question differs"),
        ({"apps": ["c", "a"]}, "a is in both runs"),
    ):
        refused = continuation_refusal(earlier, {**later, **changed})
        assert refused is not None and said in refused, (changed, refused)


def test_held_out_2_continues_held_out_1_on_every_term_of_the_comparison_with_eight_new_apps() -> None:
    first, second = load_plan(PLANS / "held-out-1.toml"), load_plan(PLANS / "held-out-2.toml")
    assert second.continues is not None and first.continues is None
    assert (second.continues.run, second.continues.plan_commit) == ("20261118T062749Z", "0941140d8")

    def described(plan: Plan, apps: list[str]) -> dict[str, Any]:
        return {
            "arms": {arm.name: {"commit": arm.ref} for arm in plan.arms},
            "tiers": {name: {"model": TIERS[name].model_id} for name in plan.tiers},
            "k": plan.k,
            "primary": plan.primary,
            "question": plan.question,
            "apps": apps,
        }

    assert continuation_refusal(described(first, list(first.apps)), described(second, list(second.apps))) is None
    held_out = [case["app"] for case in registry([HELD_OUT])]
    assert list(second.apps) == held_out[24:32] and len(second.apps) == 8 and not second.placeholders()
    assert second.registries == (HELD_OUT.name,) and list(second.tiers) == ["cheap"]


def test_the_primary_opens_on_the_package_s_advice_and_a_secondary_on_its_proof_alone() -> None:
    """The primary's line is the package's first screen on one line: what to do, then what is proven and what proving
    it needs. A secondary check is a lead only, never advice on which build to run, so it reads its proof alone."""
    rows = _rows(
        {
            "docs@control": {app: (1, 3) for app in "abcdefgh"},
            "docs@candidate": {**{app: (3, 3) for app in "abcdef"}, "g": (1, 3), "h": (2, 3)},
        }
    )

    said = reading_lines(rows)

    (primary,) = [line for line in said if line.startswith(f"  {PRIMARY} (primary): ")]
    assert primary.startswith(
        f"  {PRIMARY} (primary): onramp: Advice: switch to docs@candidate. At worst it is 8 points behind docs@control. "
        "Measures the app score: docs@candidate ahead of docs@control, not yet proven. Over 8 paired cases"
    ), primary
    assert "; proving it needs 22." in primary
    secondaries = [line for line in said if "(secondary, a lead only): " in line]
    assert secondaries and not any("Advice" in line for line in secondaries), secondaries
