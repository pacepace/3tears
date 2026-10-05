"""``python -m threetears.evals``: the courier end to end in a subprocess, each command, and each refusal.

The subprocess test is the one that proves the command line is real: a fresh interpreter, the host
named by ``module:factory`` alone, a launch through the engine's own path, and the runs' summaries on
stdout. The rest drive :func:`~threetears.evals.quick.run_cli` in process, which is also how a product
mounts the commands under its own CLI.
"""

from __future__ import annotations

import asyncio
import json
import re
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from threetears.evals.analysis import NO_ANALYSIS, published_report_schema
from threetears.evals.contracts import CandidateOutput, EvalRun, EvalTestCase, JudgedArtifact
from threetears.evals.ops import report_read
from threetears.evals.quick import callable_host, run_cli, run_eval
from threetears.evals.run import KindWiring, LaunchableKind, LaunchHost, LaunchRequest, default_job_timeout, launch_run
from packages.evals.tests.fixtures.courierhost import (
    COURIER_KIND,
    COURIER_LAUNCH_SETTINGS,
    COURIER_MODELS,
    COURIER_SCOPE,
    COURIER_SUBJECT,
    COURIER_TEMPLATE_ID,
    courier_cases,
    courier_host,
    courier_launch_host,
    courier_template,
    run_courier_campaign,
)
from packages.evals.tests.test_package_matrix import REPO_ROOT

COURIER_FACTORY = "packages.evals.tests.fixtures.courierhost:courier_launch_host"


def _courier_run_args(*models: str) -> list[str]:
    args = ["run", "--scope", COURIER_SCOPE, "--template", COURIER_TEMPLATE_ID, "--subject", COURIER_SUBJECT.subject_id]
    for model in models:
        args += ["--model", model]
    return args


def test_the_cli_runs_the_courier_end_to_end_from_a_subprocess() -> None:
    """Both planner arms launch, finish and are summarised, from nothing but the command line."""
    completed = subprocess.run(
        [sys.executable, "-m", "threetears.evals", *_courier_run_args(*COURIER_MODELS), "--k", "2", "--host"]
        + [COURIER_FACTORY],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    out = completed.stdout
    for model in COURIER_MODELS:
        assert f"completed: {model} over 3 case(s) x k=2" in out
    assert out.count("6 result(s): 6 scored, 0 failed by the candidate, 0 excluded") == 2
    # The pro planner is never late; the lite one misses stops behind the two closed roads.
    assert "on_time_rate: mean 1 (n=6" in out
    assert "on_time_rate: mean 0.639 (n=6" in out


def test_ls_lists_the_scopes_templates_runs_and_campaigns(capsys: pytest.CaptureFixture[str]) -> None:
    host = courier_host()
    campaign = asyncio.run(run_courier_campaign(host))
    assert run_cli(["ls", "--scope", COURIER_SCOPE], host_factory=lambda: host) == 0
    out = capsys.readouterr().out
    assert "templates (1)" in out and COURIER_TEMPLATE_ID in out
    assert "runs (2)" in out and out.count("  completed  planner-") == 2
    assert "campaigns (1)" in out and f"{campaign.id}  planner model bake-off  2 run(s)" in out


def test_ls_over_an_empty_store_says_so(capsys: pytest.CaptureFixture[str]) -> None:
    assert run_cli(["ls", "--scope", "empty"], host_factory=lambda: callable_host([_graded])) == 0
    assert capsys.readouterr().out.splitlines() == ["templates (0)", "runs (0)", "campaigns (0)"]


def test_bundle_prints_the_campaigns_bundle_as_json(capsys: pytest.CaptureFixture[str]) -> None:
    """The bundle inspection ``report`` used to print, under its own name."""
    host = courier_host()
    campaign = asyncio.run(run_courier_campaign(host))
    assert run_cli(["bundle", campaign.id, "--scope", COURIER_SCOPE], host_factory=lambda: host) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["campaign_id"] == report["bundle"]["campaign_id"] == campaign.id
    assert sorted(report["bundle"]["run_ids"]) == sorted(campaign.run_ids)


def test_report_of_a_campaign_with_no_analysis_is_the_code_only_report(capsys: pytest.CaptureFixture[str]) -> None:
    """No analysis, no keyless analyst: the report is the evidence as code computed it, and says so first."""
    host = courier_host()
    campaign = asyncio.run(run_courier_campaign(host))
    assert run_cli(["report", campaign.id, "--scope", COURIER_SCOPE], host_factory=lambda: host) == 0
    out = capsys.readouterr().out
    assert out.startswith(f"# Campaign {campaign.id}: its evidence, with no analysis\n")
    assert "No analysis was generated." in out.splitlines()[2]
    assert f"> {NO_ANALYSIS}" in out
    assert "## Arms" in out and "## Decision surface" in out


def test_report_is_the_report_read_actions_report(capsys: pytest.CaptureFixture[str]) -> None:
    """One resolver, both callers: the command line prints exactly the body the action returns."""
    host = courier_host()
    campaign = asyncio.run(run_courier_campaign(host))
    for form in ("markdown", "html", "json"):
        assert (
            run_cli(["report", campaign.id, "--scope", COURIER_SCOPE, "--format", form], host_factory=lambda: host) == 0
        )
        body = report_read(host, campaign.id, COURIER_SCOPE, format=form).body
        printed = body + "\n" if form == "json" else body
        assert _without_assembly_time(capsys.readouterr().out) == _without_assembly_time(printed)


def _without_assembly_time(text: str) -> str:
    """A code-only report states when it was assembled; two reads a moment apart differ there and only there."""
    return re.sub(r"\d{4}-\d{2}-\d{2}T[0-9:.+]+(Z|[+-]\d{2}:\d{2})?", "<at>", text)


def test_report_as_json_is_a_code_only_report_the_published_schema_validates(
    capsys: pytest.CaptureFixture[str],
) -> None:
    host = courier_host()
    campaign = asyncio.run(run_courier_campaign(host))
    assert (
        run_cli(["report", campaign.id, "--scope", COURIER_SCOPE, "--format", "json"], host_factory=lambda: host) == 0
    )
    document = json.loads(capsys.readouterr().out)

    jsonschema.Draft202012Validator(published_report_schema()).validate(document)
    assert (document["basis"], document["source"]["analysis_id"], document["source"]["generator_model"]) == (
        "code_only",
        None,
        None,
    )


def test_report_writes_html_to_out(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    host = courier_host()
    campaign = asyncio.run(run_courier_campaign(host))
    out = tmp_path / "report.html"
    args = ["report", campaign.id, "--scope", COURIER_SCOPE, "--format", "html", "--out", str(out)]
    assert run_cli(args, host_factory=lambda: host) == 0

    page = out.read_text(encoding="utf-8")
    assert page.startswith("<!doctype html>") and 'data-basis="code_only"' in page and "<script" not in page.lower()
    assert capsys.readouterr().out == f"wrote the code-only report of campaign {campaign.id} (html) to {out}\n"


def test_report_refuses_an_out_it_cannot_write(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    host = courier_host()
    campaign = asyncio.run(run_courier_campaign(host))
    args = ["report", campaign.id, "--scope", COURIER_SCOPE, "--out", str(tmp_path)]
    assert run_cli(args, host_factory=lambda: host) == 2
    assert f"--out {tmp_path}: cannot write the report there" in capsys.readouterr().err


def test_report_refuses_a_format_it_does_not_serialize() -> None:
    with pytest.raises(SystemExit) as exited:
        run_cli(["report", "c", "--scope", COURIER_SCOPE, "--format", "pdf"], host_factory=courier_host)
    assert exited.value.code == 2


def test_a_mounted_cli_takes_no_host_option(capsys: pytest.CaptureFixture[str]) -> None:
    """A product mounting the commands names the host itself, so its users cannot name another."""
    with pytest.raises(SystemExit) as exited:
        run_cli(["ls", "--scope", "s", "--host", COURIER_FACTORY], host_factory=courier_launch_host, prog="app evals")
    assert exited.value.code == 2
    assert "unrecognized arguments: --host" in capsys.readouterr().err


def test_run_exits_one_when_a_run_does_not_complete(capsys: pytest.CaptureFixture[str]) -> None:
    """A launch whose kind breaks every cell ends its run ``failed``, and the command says so by its code."""
    eval_host = courier_host()
    world = eval_host.profile.world
    assert world is not None
    eval_host.storage.save_template(courier_template())

    class _Broken:
        judged_artifact = JudgedArtifact.UNJUDGED

        async def prepare(self, **_: Any) -> None:
            raise RuntimeError("the planner would not start")

        async def invoke(self, instance: None, test_case: EvalTestCase, sink: Any) -> CandidateOutput:
            raise AssertionError("never prepared")

    async def launch(request: LaunchRequest) -> EvalRun:
        cases = courier_cases()
        for case in cases:
            eval_host.storage.save_test_case(case)
        wiring = KindWiring(
            kind_factory=lambda _cell: _Broken(),
            subject=COURIER_SUBJECT,
            test_cases=cases,
            payload={"courier": {"search_depth": 3, "traffic_feed": "feed-v2"}},
        )
        return await launch_run(launch_host, request, wiring)

    launch_host = LaunchHost(
        eval_host=eval_host,
        kinds={COURIER_KIND: LaunchableKind(launch=launch)},
        settings=lambda: COURIER_LAUNCH_SETTINGS,
        job_timeout_factory=default_job_timeout,
        world_placements=lambda _run: world.place(seeded=(), carriers=()),
    )
    assert run_cli(_courier_run_args("planner-lite"), host_factory=lambda: launch_host) == 1
    assert "failed: planner-lite" in capsys.readouterr().out


def _graded(case: Mapping[str, Any], answer: Any) -> float:
    return 1.0


async def _unused(case: Mapping[str, Any]) -> float:
    return 1.0


# --- refusals: each prints its reason to stderr and exits 2 ----------------------------------------


@pytest.mark.parametrize(
    ("spec", "said"),
    [
        ("packages.evals.tests.fixtures.courierhost", "--host takes module:factory"),
        (":courier_launch_host", "--host takes module:factory"),
        ("no.such.module:factory", "cannot import 'no.such.module'"),
        ("packages.evals.tests.fixtures.courierhost:no_such_factory", "has no attribute 'no_such_factory'"),
        ("packages.evals.tests.fixtures.courierhost:COURIER_SCOPE", "is a str, not a callable"),
        ("packages.evals.tests.fixtures.courierhost:courier_cases", "returned a list, not an EvalHost"),
    ],
    ids=["no colon", "no module", "unimportable", "no attribute", "not callable", "not a host"],
)
def test_a_host_spec_that_names_no_host_is_refused(spec: str, said: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert run_cli(["ls", "--scope", COURIER_SCOPE, "--host", spec]) == 2
    assert said in capsys.readouterr().err


def test_run_refuses_a_host_that_cannot_launch(capsys: pytest.CaptureFixture[str]) -> None:
    assert run_cli(_courier_run_args("planner-lite"), host_factory=courier_host) == 2
    assert "must return a LaunchHost" in capsys.readouterr().err


def test_run_refuses_a_template_the_scope_does_not_hold(capsys: pytest.CaptureFixture[str]) -> None:
    args = ["run", "--scope", COURIER_SCOPE, "--template", "nope", "--subject", "x", "--model", "planner-lite"]
    assert run_cli(args, host_factory=courier_launch_host) == 2
    assert "nope" in capsys.readouterr().err


def test_run_relays_the_launchers_own_refusal(capsys: pytest.CaptureFixture[str]) -> None:
    assert run_cli(_courier_run_args("planner-max"), host_factory=courier_launch_host, prog="app evals") == 2
    err = capsys.readouterr().err
    assert err.startswith("app evals run: ") and "'planner-max' is none of them" in err


@pytest.mark.parametrize("command", ["report", "bundle"])
def test_report_and_bundle_refuse_a_campaign_the_scope_does_not_hold(
    command: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run_cli([command, "nope", "--scope", COURIER_SCOPE], host_factory=courier_host) == 2
    assert "nope" in capsys.readouterr().err


async def test_a_run_eval_host_is_one_the_cli_can_read(capsys: pytest.CaptureFixture[str]) -> None:
    """``run_eval`` over a host the caller keeps leaves runs ``ls`` lists, so the two batteries meet."""
    host = callable_host([_graded])
    summary = await run_eval([{"q": "a"}], _unused, [_graded], scope_id="batteries", host=host)
    capsys.readouterr()
    assert run_cli(["ls", "--scope", "batteries"], host_factory=lambda: host) == 0
    out = capsys.readouterr().out
    assert f"  {summary.run_id}  completed  _unused  template {summary.template_id}" in out
