"""Rung 3, ``examples/3-invoices/``: two prompts compared on 30 invoices, replayed offline, with a report to hand over.

The rung runs here as a reader runs it, from a copy of its directory with no key set, so its model can only replay
``recordings/invoices.jsonl``. Nothing here assumes what the model answered beyond what the recording holds: every
call is served from it, every cell is graded (none excluded as a rig fault), the guardrail is decided apart from
the contrasts, and the exit code is the one the typed gate maps to.
"""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

from threetears.evals import Result
from threetears.evals.analysis import AnalysisContextBundle
from threetears.evals.kernel import strict_response_format
from threetears.evals.schema import ClientRequestSettings
from threetears.evals.testing import RecordedCompletions, RecordedPrompt
from packages.evals.tests.ladder_support import (
    EXAMPLES,
    count_calls,
    load,
    non_blank_lines,
    offline_copy,
    recorded_spend,
    secret_markers,
    stated_spend,
)
from packages.evals.tests.test_package_matrix import REPO_ROOT, SOURCE_ROOT, public_root_violations

RUNG = EXAMPLES / "3-invoices"
EXTRACT = RUNG / "extract.py"
RECORDING = RUNG / "recordings" / "invoices.jsonl"
CASES = [json.loads(line) for line in (RUNG / "invoices.jsonl").read_text(encoding="utf-8").splitlines()]


def test_the_cases_are_thirty_invoices_in_three_strata_with_hard_negatives() -> None:
    assert len(CASES) == 30 and len({case["id"] for case in CASES}) == 30
    strata = [case["stratum"] for case in CASES]
    assert {stratum: strata.count(stratum) for stratum in set(strata)} == {
        "clean": 10,
        "multi-currency": 10,
        "noisy-OCR": 10,
    }
    # A hard negative prints no amount due, so the only right total is none; every stratum holds some.
    negatives = {case["stratum"] for case in CASES if case["total"] is None}
    assert negatives == set(strata) and sum(case["total"] is None for case in CASES) >= 6


def test_the_recording_covers_every_prompt_both_arms_send(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    offline_copy("3-invoices", tmp_path, monkeypatch)
    module = load(Path("extract.py"), "extract", monkeypatch)
    directive = strict_response_format("Invoice", module.Invoice.model_json_schema())
    asked = [
        RecordedPrompt("", template.format(text=case["text"]), directive)
        for template in (module.V1, module.V2)
        for case in CASES
        for _ in range(module.invoices.k)
    ]
    RecordedCompletions.check_covers(
        RECORDING, asked, settings=ClientRequestSettings(max_tokens=8192, reasoning_effort="low")
    )


async def test_offline_it_replays_both_prompts_and_decides_by_the_typed_verdicts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    offline_copy("3-invoices", tmp_path, monkeypatch)
    module = load(Path("extract.py"), "extract", monkeypatch)
    served = count_calls(module.model, monkeypatch)

    result = await module.invoices.run(store=":memory:")

    assert len(served.calls) == recorded_spend(RECORDING)[0] == 180, "30 invoices x 2 prompts x k=3"
    assert isinstance(result, Result) and result.control == "v1" and list(result.arms) == ["v1", "v2"]
    for summary in result.arms.values():
        assert summary.status == "completed" and (summary.n_results, summary.n_excluded) == (90, 0)
    assert result.fields == ("currency", "number", "due", "total")
    # The guardrail is decided per arm against the control, never in the contrasts.
    assert result.contrasts("no_invented_total") == []
    assert {row["arm"] for row in result.guardrails()} == {"v2"}
    assert {row["outcome"] for row in result.guardrails()} <= {"held", "breached", "undecided"}
    assert {row["arm"] for row in result.contrasts("total")} == {"v2"}
    # Every reading is read again per kind of invoice.
    assert {stratum for row in result.strata("v2") for stratum in row["strata"]} == {
        "clean",
        "multi-currency",
        "noisy-OCR",
    }
    # The closing comment's claim: the same runs read against another control, with no run repeated.
    assert result.exit_code() == (4 if result.gate().failed else 0)
    again = await result.against("v2")
    assert again.control == "v2" and again.run_ids == result.run_ids


def test_run_as_a_script_it_writes_the_report_and_exits_with_the_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    copy = offline_copy("3-invoices", tmp_path, monkeypatch)
    with pytest.raises(SystemExit) as exited:
        runpy.run_path(str(copy / "extract.py"), run_name="__main__")

    out = capsys.readouterr().out
    assert out.startswith("invoices: ") and "v1 (control): 90 result(s)" in out
    html = (copy / "report" / "report.html").read_text(encoding="utf-8")
    assert "<script" not in html
    bundle = AnalysisContextBundle.from_json((copy / "report" / "bundle.json").read_text(encoding="utf-8"))
    assert bundle.fingerprint() in html
    assert (copy / "report" / "report.md").is_file()
    assert exited.value.code in (0, 4), "0 when the gate passed, 4 when it failed; never a crash"


def test_the_docstring_states_what_the_recording_cost() -> None:
    calls, dollars = recorded_spend(RECORDING)
    assert stated_spend(EXTRACT) == (calls, round(dollars, 4))


def test_the_recording_holds_no_key_material() -> None:
    assert secret_markers(RECORDING) == []


def test_the_rung_stays_about_45_lines() -> None:
    assert non_blank_lines(EXTRACT) <= 45


def test_the_rung_reaches_the_engine_only_through_public_roots() -> None:
    assert public_root_violations(SOURCE_ROOT, [(EXTRACT.name, EXTRACT)], consumer_root=REPO_ROOT) == []
