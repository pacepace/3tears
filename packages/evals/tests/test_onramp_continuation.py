"""A held-out comparison continued over two runs, through the path a real run takes: a run's ``run.json`` and
``trials.jsonl`` stored under ``runs/``, the earlier run found and held to the comparison (``earlier_stages``), the
two read as one (``--read``, ``read_lines``), what a run of two arms writes (``report_comparison``), and the power a
continuation is sized by (``continued_power``). Offline and free: every reading replays stored-shaped rows through the
package, and no agent runs.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from packages.evals.onramp.campaign import (
    EARLIER_REGRADED,
    earlier_stages,
    main,
    plan_rule,
    report_comparison,
)
from packages.evals.onramp.plan import PLANS, load_plan
from packages.evals.onramp.reading import (
    PRIMARY,
    Stage,
    Unreadable,
    continued_power,
    read,
    read_continued,
    read_lines,
)

ARMS = {"docs@control": {"wheel": "w0", "ref": "f9ab6f74d", "commit": "f" * 40},
        "docs@candidate": {"wheel": "w1", "ref": "1f0ec4c68", "commit": "1" * 40}}  # fmt: skip
QUESTION = "Does the candidate pass more often?"


def _facts(read_right_mean: float) -> dict[str, Any]:  # Any: a regrade's facts
    """Stored regrade facts: an eval that ran and stored, whose one measure read ``read_right_mean`` on the app and 1.0
    on its twin (so today's measures_the_app passes it), with doing nothing failing every case."""
    app = {"readings": ["m"], "finished": True, "scored": 3, "means": {"m": read_right_mean},
           "higher_is_better": {"m": True}, "nothing_unpassed": 1.0}  # fmt: skip
    return {"problem": "", "door_exit": 0, "stored": True, "door_first_line": "a: r passed 2 of 3 cases.", "app": app,
            "twin": {"means": {"m": 1.0}}}  # fmt: skip


def _rows(tallies: dict[str, dict[str, tuple[int, int]]], **stored: float) -> list[dict[str, Any]]:  # Any: rows
    """Stored rows of one tier: ``{arm: {app: (primary passed, of)}}``. Each trial's record holds facts on which
    today's read_right passes (a report of 0.72 against a 0.725 mean, which the old float rule failed); its stored
    grades are the primary's as given, and ``stored`` for the others (read_right 0.0 unless said)."""
    rows = []
    for arm, apps in tallies.items():
        for app, (passed, of) in apps.items():
            for repeat in range(1, of + 1):
                grade = 1.0 if repeat <= passed else 0.0
                scores = {"eval_runs": 1.0, "read_right": 0.0, **stored, PRIMARY: grade}
                report = {"first_line": "a: r passed 2 of 3 cases.", "finished": True, "pass_rate": 0.72}
                record = {"app": app, "checks": _facts(0.725), "report": report, "stopped": "done"}
                row = {"arm": arm, "app": app, "repeat": repeat, "tier": "cheap", "scores": scores, "record": record}
                rows.append({**row, "state": "passed" if grade else "failed"})
    return rows


def _meta(apps: list[str], **more: Any) -> dict[str, Any]:  # Any: run.json
    return {
        "plan": "packages/evals/onramp/plans/x.toml",
        "plan_commit": "a" * 40,
        "head_commit": "a" * 40,
        "question": QUESTION,
        "primary": PRIMARY,
        "arms": ARMS,
        "apps": apps,
        "k": 3,
        "tiers": {"cheap": {"model": "m", "per_run_usd": 0.2, "cap_usd": 1.0}},
        **more,
    }


def _store(run: Path, meta: dict[str, Any], rows: list[dict[str, Any]]) -> Path:  # Any: run.json, rows
    run.mkdir(parents=True)
    (run / "run.json").write_text(json.dumps(meta), encoding="utf-8")
    (run / "trials.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return run


EARLIER = {"docs@control": {"a": (0, 3), "b": (1, 3), "c": (2, 3), "d": (1, 3)},
           "docs@candidate": {"a": (3, 3), "b": (3, 3), "c": (2, 3), "d": (2, 3)}}  # fmt: skip
LATER = {"docs@control": {"e": (1, 3), "f": (0, 3)}, "docs@candidate": {"e": (3, 3), "f": (2, 3)}}


def _runs(tmp_path: Path) -> tuple[Path, list[dict[str, Any]]]:  # Any: rows
    """An earlier run (4 apps) and a later one continuing it (2 apps), stored under ``tmp_path/runs``; the later's one
    control trial on ``e`` ungraded (lost), as run 20261118T153025Z's tipsplit control trial was."""
    earlier = _rows(EARLIER)
    _store(tmp_path / "runs" / "R1", _meta(list(EARLIER["docs@control"])), earlier)
    later = _rows(LATER, read_right=1.0)
    lost = next(row for row in later if row["arm"] == "docs@control" and row["app"] == "e" and row["repeat"] == 3)
    lost["scores"], lost["state"] = {}, "lost"
    continues = {"run": "R1", "plan_commit": "a" * 9}
    return _store(tmp_path / "runs" / "R2", _meta(["e", "f"], continues=continues), later), later


def test_a_continuing_run_is_read_with_the_run_it_continues_as_one_comparison(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    later_dir, later = _runs(tmp_path)

    assert main(["--read", str(later_dir)]) == 0

    said = capsys.readouterr().out.splitlines()
    assert said[1] == EARLIER_REGRADED
    assert said[2].endswith("6 app(s), 3 trial(s) per arm on each, continued: 4 then 2 app(s) as one comparison")
    primary = said.index(next(line for line in said if line.startswith(f"  {PRIMARY} (primary)")))
    assert said[primary + 1].startswith("    docs@candidate: ") and "; 6 paired apps; " in said[primary + 1]
    tallies = said[primary + 2]
    assert "a 0 of 3, 3 of 3; b 1 of 3, 3 of 3" in tallies and "e 1 of 2, 3 of 3; f 0 of 3, 2 of 3" in tallies, (
        "every app of both runs, the lost trial out of e's share"
    )
    right = said.index(next(line for line in said if line.startswith("  read_right (")))
    assert "a 3 of 3, 3 of 3" in said[right + 2], "the earlier run's read_right graded by today's rule, not stored 0s"


async def test_the_continued_reading_has_a_fresh_read_of_every_app_as_its_cases_and_keeps_the_primary_as_it_ran(
    tmp_path: Path,
) -> None:
    later_dir, later = _runs(tmp_path)
    meta = json.loads((later_dir / "run.json").read_text(encoding="utf-8"))
    stages = [*earlier_stages(meta, tmp_path / "runs", later), Stage(tuple(later), tuple(ARMS), ("e", "f"))]

    continued = await read_continued(stages, PRIMARY, question=QUESTION)
    fresh = await read([*_rows(EARLIER), *later], PRIMARY, question=QUESTION, arms=list(ARMS), apps=list("abcdef"))

    (one,), (other,) = continued.contrasts(PRIMARY), fresh.contrasts(PRIMARY)
    assert one["cases"] == other["cases"] == "6 paired"
    assert (one["control_mean"], one["arm_mean"]) == pytest.approx((other["control_mean"], other["arm_mean"]))
    assert one["control_mean"] == pytest.approx(100 * (0 + 1 / 3 + 2 / 3 + 1 / 3 + 1 / 2 + 0) / 6), "e over 2 trials"
    earlier_primary = [row["scores"][PRIMARY] for row in stages[0].rows]
    assert earlier_primary == [row["scores"][PRIMARY] for row in _rows(EARLIER)], "the primary as it ran"
    again = await read_continued(stages, PRIMARY, question=QUESTION)
    assert again.contrasts(PRIMARY)[0]["p_adjusted"] == one["p_adjusted"], "one committed order, read the same twice"


def test_a_continuation_is_refused_by_name_on_any_term_of_the_comparison(tmp_path: Path) -> None:
    later_dir, later = _runs(tmp_path)
    meta = json.loads((later_dir / "run.json").read_text(encoding="utf-8"))
    runs = tmp_path / "runs"
    assert len(earlier_stages(meta, runs, later)) == 1 and earlier_stages({**meta, "continues": None}, runs) == []
    with pytest.raises(Unreadable, match="not the bbbbbbbbb the plan names"):
        earlier_stages({**meta, "continues": {"run": "R1", "plan_commit": "b" * 9}}, runs)
    swapped = {"docs@control": ARMS["docs@candidate"], "docs@candidate": ARMS["docs@control"]}
    with pytest.raises(Unreadable, match="the arms differ"):
        earlier_stages({**meta, "arms": swapped}, runs)
    with pytest.raises(Unreadable, match="a is in both runs"):
        earlier_stages({**meta, "apps": ["a", "e"]}, runs)
    lower = {"cheap": {"model": "m", "per_run_usd": 0.1, "cap_usd": 0.5}}
    assert len(earlier_stages({**meta, "tiers": lower}, runs, later)) == 1, "caps differ, but none bound"
    capped = [{**row, "record": {**row["record"], "stopped": "cap"}} if row["app"] == "f" else row for row in later]
    with pytest.raises(Unreadable, match="per-run cap differs"):
        earlier_stages({**meta, "tiers": lower}, runs, capped)
    with pytest.raises(Unreadable, match="holds no trials.jsonl"):
        earlier_stages({**meta, "continues": {"run": "R9", "plan_commit": "a" * 9}}, runs)


async def test_a_continued_reading_is_one_tiers() -> None:
    first = Stage(tuple(_rows(EARLIER)), tuple(ARMS), tuple("abcd"))
    two_tiers = [*_rows(LATER), *({**row, "tier": "ceiling"} for row in _rows(LATER))]
    with pytest.raises(Unreadable, match="one tier's"):
        await read_lines(two_tiers, earlier=[first])


def test_a_run_of_two_arms_reports_the_continued_primary_as_its_headline_and_states_its_plans_rule(
    tmp_path: Path,
) -> None:
    later_dir, later = _runs(tmp_path)
    meta = json.loads((later_dir / "run.json").read_text(encoding="utf-8"))
    earlier = earlier_stages(meta, tmp_path / "runs", later)
    campaign = asyncio.run(read(later, "read_right", lead=True, arms=list(ARMS), apps=["e", "f"]))
    plan = load_plan(PLANS / "held-out-2.toml")
    out = tmp_path / "out"
    out.mkdir()

    said, written = report_comparison(campaign, later, out, apps=["e", "f"], plan=plan, earlier=earlier)

    assert said[0] == EARLIER_REGRADED and said[-1] == plan_rule(plan)
    assert "one continuation" in said[-1] and not any("add cases to the end" in line for line in said)
    assert (out / "reading.txt").read_text(encoding="utf-8").splitlines() == said[:-2]
    headline = (out / "report" / "report.md").read_text(encoding="utf-8")
    assert written["html"] == out / "report" / "report.html"
    assert "Measures the app" in headline and "Read right" not in headline, "the primary heads the report"
    assert "6 paired" in headline or "Over 6 paired cases" in headline, "read over both runs' apps"
    assert "Read right" in (out / "campaign-report" / "report.md").read_text(encoding="utf-8"), "the run's own campaign"


def test_a_continuations_power_is_counted_through_the_continued_reading() -> None:
    """Pinned at what this code counts on its seed: a draw that read only the new apps, or never the earlier run, or
    resampled the wrong arm's share, reads otherwise."""
    apps = [f"app-{index}" for index in range(16)]
    # Every app's control passes 1 of 3; every third app's candidate passes 3: alone it reads not separated (p 0.12).
    tallies = {"docs@control": {app: (1, 3) for app in apps},
               "docs@candidate": {app: (3 if index % 3 == 0 else 1, 3) for index, app in enumerate(apps)}}  # fmt: skip
    earlier = Stage(tuple(_rows(tallies)), tuple(ARMS), tuple(apps))
    alone = asyncio.run(read(list(earlier.rows), PRIMARY, question=QUESTION, arms=list(ARMS), apps=apps))
    assert alone.contrasts(PRIMARY)[0]["outcome"] == "not_separated"
    full = continued_power(earlier, add=8, scale=1.0, reps=6, seed=3, question=QUESTION)
    none = continued_power(earlier, add=8, scale=0.0, reps=6, seed=3, question=QUESTION)
    assert (full, none) == ((2, 6), (0, 6))
