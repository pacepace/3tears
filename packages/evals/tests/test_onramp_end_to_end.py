"""The on-ramp's agent test, end to end and offline: the harness eval runs a scripted agent over every app.

Two arms, each a fixed transcript on this interpreter's build: one that adds a working eval to each app and reports
it right, and one that gets stuck on the newcomer's errors and gives up. Everything runs as it will live (the sandbox,
the eval door, the re-runs against the app, its fixed twin and a do-nothing runner, the package's store and report),
except that no model is asked and nothing is spent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from packages.evals.onramp.agent import Caps
from packages.evals.onramp.campaign import adopter, main, onramp_eval, scripted, trials
from packages.evals.onramp.sandbox import DocsBuild

#: Every field a trial's record carries.
RECORDED = {
    "app", "arm", "build", "model", "stopped", "wall_s", "turns", "input_tokens", "output_tokens", "cost_usd",
    "charged_usd", "maybe_billed_sends", "commands", "errors_hit", "pages_read", "written", "stuck", "format_error_turns",
    "report", "report_from", "report_problem", "sandbox", "checks",
}  # fmt: skip


@pytest.mark.timeout(1800)
async def test_a_scripted_agent_runs_the_whole_harness_and_every_field_is_recorded(tmp_path: Path) -> None:
    arms = {
        "scripted@succeeds": adopter(
            DocsBuild.here("scripted@succeeds"), scripted(), workdir=tmp_path, caps=Caps(wall_s=900.0)
        ),
        "scripted@stuck": adopter(
            DocsBuild.here("scripted@stuck"), scripted("stuck"), workdir=tmp_path, caps=Caps(wall_s=900.0)
        ),
    }
    result = await onramp_eval(arms, k=1, max_cost_usd=None).run(store=tmp_path / "onramp.sqlite")

    rows = trials(result)  # read back from the eval's own store
    assert {(row["arm"], row["record"]["app"]) for row in rows} == {
        (arm, app) for arm in arms for app in ("recipes", "holds", "glossator")
    }
    for row in rows:
        record: dict[str, Any] = row["record"]  # Any: a trial's record
        assert set(record) == RECORDED, row["arm"]
        assert record["stopped"] == "done" and record["turns"] >= 6 and record["wall_s"] > 0
        assert (record["input_tokens"], record["output_tokens"], record["cost_usd"]) == (0, 0, 0.0)
        assert (record["charged_usd"], record["maybe_billed_sends"]) == (0.0, 0)
        trial = Path(record["sandbox"]).parent
        assert json.loads((trial / "record.json").read_text(encoding="utf-8")) == record, "kept beside the trial"
        assert (trial / "steps.jsonl").read_text(encoding="utf-8").count("\n") == record["turns"] + 1
        assert record["commands"] and all({"exit_code", "first_error"} <= set(c) for c in record["commands"])
        assert {"page": "llms.txt", "lines": record["pages_read"][0]["lines"], "turn": 2} in record["pages_read"]
        succeeded = row["arm"] == "scripted@succeeds"
        grades = {name: row["scores"][name] for name in ("eval_runs", "measures_the_app", "read_right")}
        checks = record["checks"]
        if succeeded:
            assert grades == {"eval_runs": 1.0, "measures_the_app": 1.0, "read_right": 1.0}, (record["app"], checks)
            assert record["stuck"] is None and record["errors_hit"] == {}
            assert checks["door_exit"] == 0 and checks["twin"]["means"] != checks["app"]["means"]
            assert checks["app"]["nothing_unpassed"] == 1.0
        else:
            assert grades == {"eval_runs": 0.0, "measures_the_app": 0.0, "read_right": 0.0}, record["app"]
            assert set(record["errors_hit"]) >= {
                "help_first",
                "cases_missing",
                "grader_unnamed",
                "eval_not_at_module_level",
            }
            stuck = record["stuck"]
            assert stuck["from_turn"] == 4 and stuck["failed_turns"] == 4, stuck
            assert "cases.jsonl is not there" in stuck["opened_by"]
            assert checks["door_exit"] == 2
    assert result.render().strip()
    assert set(result.write(tmp_path / "report")) >= {"html", "json"}


@pytest.mark.timeout(900)
def test_the_command_runs_the_harness_offline_and_keeps_every_trial(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--scripted", "succeeds", "--host", "--k", "1", "--apps", "holds", "--workdir", str(tmp_path)]) == 0
    said = capsys.readouterr().out
    (run_dir,) = [path for path in tmp_path.iterdir() if path.is_dir()]
    (row,) = [json.loads(line) for line in (run_dir / "trials.jsonl").read_text(encoding="utf-8").splitlines()]
    assert row["arm"] == "docs@here" and row["record"]["app"] == "holds"
    assert {name: row["scores"][name] for name in ("eval_runs", "measures_the_app", "read_right")} == {
        "eval_runs": 1.0,
        "measures_the_app": 1.0,
        "read_right": 1.0,
    }
    assert (run_dir / "onramp.sqlite").is_file() and (run_dir / "report" / "report.html").is_file()
    assert f"trials: {run_dir / 'trials.jsonl'}" in said


@pytest.mark.timeout(900)
def test_a_run_where_every_agent_fails_says_so_on_its_first_line_and_in_its_census(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The graders are checks (each returns ``bool``, annotated so), so a trial one fails is failed and the package's
    first line counts it; and the harness's own census states each check's tally and what so few runs cannot show."""
    main(["--scripted", "stuck", "--host", "--k", "1", "--apps", "holds", "--workdir", str(tmp_path)])
    said = capsys.readouterr().out.splitlines()
    assert said[0] == "arm docs@here: this interpreter"
    assert said[1].startswith("onramp: docs@here passed 0 of 1 cases."), said[1]
    for name in ("eval_runs", "measures_the_app", "read_right"):
        assert any(line.startswith(f"  {name}: 0 of 1 trials, 95% range 0.00 to ") for line in said), said
    assert any("cannot show" in line for line in said)
    assert any("compare each check's k of n only: never errors_hit" in line for line in said), "the comparison rule"
    (run_dir,) = [path for path in tmp_path.iterdir() if path.is_dir()]
    written = (run_dir / "census.txt").read_text(encoding="utf-8").splitlines()
    start = said.index("what 1 run(s) can say:")
    assert written == said[start : start + len(written)] and len(written) == 10, "the census, as printed"
