"""Rung 3, ``examples/3-invoices/``: two prompts compared on 30 invoices, one concept a file, replayed offline.

- 3a ``compare.py``: two arms, a control and a verdict;
- 3b ``margin.py``: a margin on total, which reads equivalent;
- 3c ``guardrail.py``: a guardrail decided apart from the contrasts, undecided, so the door exits 4;
- 3d ``ci.py``: the gate in CI, which writes the report and exits by the gate;
- 3e ``grid.py``: two levers at once, four arms named as ``k=v`` factors, read again against another control.

Each runs here as a reader runs it, from a copy of the directory with no key set, so its models can only replay
``recordings/invoices.jsonl`` (and, for the grid's high-effort cells, ``recordings/invoices-high.jsonl``). Nothing
here assumes what the model answered beyond what the recordings hold: every call is served from them, every cell is
graded (none excluded as a rig fault), and the exit code is the one the typed gate maps to.
"""

from __future__ import annotations

import json
import runpy
from pathlib import Path
from types import ModuleType

import pytest

from threetears.evals import Eval, Result
from threetears.evals.analysis.report.campaign.provenance import COST_ROUTING_CAVEAT
from threetears.evals.analysis import (
    AnalysisContextBundle,
    campaign_report_html,
    campaign_report_markdown,
    report_markdown,
)
from threetears.evals.kernel import strict_response_format
from threetears.evals.quick.files import main
from threetears.evals.schema import ClientRequestSettings
from threetears.evals.testing import RecordedCompletions, RecordedPrompt
from packages.evals.tests.style_support import unlinked
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
from packages.evals.tests.report_support import page_violations
from packages.evals.tests.rung_support import recorded
from packages.evals.tests.test_package_matrix import REPO_ROOT, SOURCE_ROOT, public_root_violations
from packages.evals.tests.page_contract_support import what_runs

RUNG = EXAMPLES / "3-invoices"
#: One file a concept: 3a to 3e.
COMPARE, MARGIN, GUARDRAIL, CI, GRID = (
    RUNG / f"{stem}.py" for stem in ("compare", "margin", "guardrail", "ci", "grid")
)
#: Every file of the rung, the shared module with no eval in it among them.
FILES = (RUNG / "invoice.py", COMPARE, MARGIN, GUARDRAIL, CI, GRID)
RECORDING = RUNG / "recordings" / "invoices.jsonl"
#: The grid's high-effort cells, recorded apart from the low-effort ones every other file replays.
RECORDING_HIGH = RUNG / "recordings" / "invoices-high.jsonl"
#: The question the rung declares, word for word.
QUESTION = "Does v2 read the currency an invoice charges in right more often than v1?"
CASES = [json.loads(line) for line in (RUNG / "invoices.jsonl").read_text(encoding="utf-8").splitlines()]


#: The result's sentences on every surface: the decision, then the figures it rests on, as the contrasts table
#: prints them.
_LEAD = (
    "Accuracy: the evidence cannot tell v2 and v1 apart. Of 90 trials each, Accuracy was right on 90 for v1 and 89 "
    "for v2. Over 30 paired cases, the interval allows v2 to be as much as 20 points worse or 18 better."
)


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


def _asked(module: ModuleType, k: int) -> list[RecordedPrompt]:
    """Every prompt two arms send, v1's then v2's, ``k`` times a case."""
    directive = strict_response_format("Invoice", module.Invoice.model_json_schema())
    return [
        RecordedPrompt("", template.format(text=case["text"]), directive)
        for template in (module.V1, module.V2)
        for case in CASES
        for _ in range(k)
    ]


@pytest.mark.parametrize("file", ["compare.py", "margin.py", "guardrail.py"])
def test_the_recording_covers_every_prompt_both_arms_send(
    file: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    offline_copy("3-invoices", tmp_path, monkeypatch)
    module = load(Path(file), Path(file).stem, monkeypatch)
    (runnable,) = (value for value in vars(module).values() if isinstance(value, Eval))
    RecordedCompletions.check_covers(
        RECORDING, _asked(module, runnable.k), settings=ClientRequestSettings(max_tokens=8192, reasoning_effort="low")
    )


def test_the_grid_s_two_recordings_cover_its_cells_at_each_effort(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The low-effort cells send what every other file sends; the high-effort cells the same prompts at high effort."""
    offline_copy("3-invoices", tmp_path, monkeypatch)
    module = load(Path("grid.py"), "grid", monkeypatch)
    for recording, effort in ((RECORDING, "low"), (RECORDING_HIGH, "high")):
        settings = ClientRequestSettings(max_tokens=8192, reasoning_effort=effort)
        RecordedCompletions.check_covers(recording, _asked(module, module.grid.k), settings=settings)
    assert {line["sent_settings"]["reasoning_effort"] for line in recorded(RECORDING_HIGH)} == {"high"}


async def test_3a_offline_it_replays_two_arms_against_the_control_and_leads_on_the_question_s_measure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    offline_copy("3-invoices", tmp_path, monkeypatch)
    module = load(Path("compare.py"), "compare", monkeypatch)
    served = count_calls(module.model, monkeypatch)

    result = await module.compare.run(store=":memory:")
    lines = report_markdown(result.report).splitlines()
    # Its own name, the declared question, the arms with the control marked, and the lead: the verdict alone, with no
    # guardrail to warn of.
    assert (lines[0], lines[1]) == ("# compare", f'"{QUESTION}"')
    assert lines[3].startswith("v1 (control) vs v2 · ")
    assert lines[5] == f"**{_LEAD}**"
    assert page_violations(result.report) == []

    assert len(served.calls) == 180
    assert result.control == "v1" and list(result.arms) == ["v1", "v2"]
    assert result.grader_readings == ("accuracy", "number", "due", "total")
    # Every other arm is read against the control on every reading; with no margin, none can read equivalent.
    assert {row["outcome"] for grader in result.grader_readings for row in result.contrasts(grader)} == {
        "not_separated"
    }
    assert result.guardrails() == [] and not result.gate().failed and result.exit_code() == 0


async def test_3b_a_margin_lets_total_read_equivalent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """compare.py's runs with a 20-point margin on total: v2's interval there (19 points either side) is inside it."""
    offline_copy("3-invoices", tmp_path, monkeypatch)
    module = load(Path("margin.py"), "margin", monkeypatch)

    result = await module.within_margin.run(store=":memory:")
    (total,) = result.contrasts("total")
    assert (total["outcome"], total["interval"]) == ("equivalent", "-19 to +19 pts at 99.2%")
    # Only the reading with a margin can: the others still cannot tell the arms apart.
    assert {row["outcome"] for grader in ("accuracy", "number", "due") for row in result.contrasts(grader)} == {
        "not_separated"
    }
    assert (
        "| Total score | 99% (88 to 99.9) | 99% (88 to 99.9) | 0 pts | -19 to +19 pts | v2 and v1 equivalent, within "
        "the measure's margin (margin ±20 pts) | 30 paired |"
    ) in campaign_report_markdown(result.page).splitlines()


async def test_3c_offline_it_replays_both_prompts_and_decides_the_guardrail_apart(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    offline_copy("3-invoices", tmp_path, monkeypatch)
    module = load(Path("guardrail.py"), "guardrail", monkeypatch)
    served = count_calls(module.model, monkeypatch)

    result = await module.invoices.run(store=":memory:")
    # The first screen, golden: the question, the arms with the control marked, the result, a bare scoreboard.
    lines = report_markdown(result.report).splitlines()
    # The heading is the campaign's name, its arms the line under the question; the declared question is quoted under
    # it, and the result reads as asked, never as exploratory.
    assert lines[0] == "# invoices"
    assert lines[1] == f'"{QUESTION}"'
    assert lines[3].startswith("v1 (control) vs v2 · ")
    # Never "no difference", never "not separated": the campaign's lead sentence, then what the guardrail means.
    assert lines[5] == f"**{_LEAD} v2 may do worse than v1 on the No invented total score guardrail.**"
    # The one failure cause beside checks and the judge heads its own column; no legend line is needed under it. Both
    # prompts cost well under a cent per invoice, so the cost column reads per 1,000 replies. The page prints a cost,
    # and its provider routing is said in the methods, not between the result and the scoreboard (owner reading,
    # 2026-11-06): the scoreboard follows the result.
    assert lines[7] == "**Scoreboard**" and COST_ROUTING_CAVEAT in report_markdown(result.report)
    assert lines[9] == (
        "| Arm | Ran | Usable | Passed | Failed by check | Failed by judge | Model failed | Cost per 1,000 replies |"
    )
    assert page_violations(result.report) == []

    assert len(served.calls) == recorded_spend(RECORDING)[0] == 180, "30 invoices x 2 prompts x k=3"
    assert isinstance(result, Result) and result.control == "v1" and list(result.arms) == ["v1", "v2"]
    # The question round-trips to the result and to its campaign, the graders' order with it.
    assert result.question == module.invoices.question == QUESTION
    assert result.grader_readings == ("accuracy", "number", "due", "total", "no_invented_total")
    for summary in result.arms.values():
        assert summary.status == "completed" and (summary.n_results, summary.n_excluded) == (90, 0)
    assert result.fields == ("currency", "number", "due", "total")
    # The guardrail is decided per arm against the control, never in the contrasts.
    assert result.contrasts("no_invented_total") == []
    assert {row["arm"] for row in result.guardrails()} == {"v2"}
    assert {row["outcome"] for row in result.guardrails()} <= {"held", "breached", "undecided"}
    # The guardrail's Decision cell is on its row's ruler, points for a rate, beside the interval and margin columns
    # that state it so; the interval is its own column's, the level in that column's header.
    (guarded,) = result.guardrails()
    assert (guarded["interval"], guarded["margin"]) == ("-13 to +13 pts", "2 pts")
    assert guarded["decision"] == (
        "undecided: v2 may be worse than v1 by more than the margin (the interval reaches both sides of -2 pts, the "
        "declared margin, so the arm is shown neither within it nor beyond it)"
    )
    # In a comparison each arm's guardrail line says what the comparison decided, never that it takes one.
    printed = str(result)
    assert "decided only against a control" not in printed
    assert "the guardrail the other arms are decided against" in printed
    # Named as the page names the guardrail, never by its key.
    assert "No invented total score: 99%, 88 of 89 trials; guardrail undecided against v1" in printed
    assert {row["arm"] for row in result.contrasts("total")} == {"v2"}
    # Every reading is read again per kind of invoice.
    assert {stratum for row in result.strata("v2") for stratum in row["strata"]} == {
        "clean",
        "multi-currency",
        "noisy-OCR",
    }
    assert result.exit_code() == (4 if result.gate().failed else 0) == 4


async def test_3c_its_page_leads_on_its_question_s_measure_with_the_guardrail_v2_is_not_shown_safe_on_beside_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    offline_copy("3-invoices", tmp_path, monkeypatch)
    module = load(Path("guardrail.py"), "guardrail", monkeypatch)
    result = await module.invoices.run(store=":memory:")
    lines = campaign_report_markdown(result.page).splitlines()
    first_screen = lines[: lines.index(next(line for line in lines if line.startswith("## ")))]
    # The page quotes the declared question and leads on its measure, the first grader (the classifier's accuracy), as
    # the printed line does, decided rather than a lead; the guardrail warns as loudly as a broken floor, beside the
    # lead and never in its place, and picks no arm.
    assert lines[0] == "# invoices"
    assert lines[1] == f'"{QUESTION}"'
    assert "Exploratory comparison." not in lines
    bold = [line for line in first_screen if line.startswith("**")]
    # The guardrail says what it means for the decision, then what leaves it undecided: of the 30 cases that could
    # fail it, one noisy invoice each, too few to settle it.
    warning = (
        "v2 may do worse than v1 on the No invented total score guardrail on invoices, by more than its 2-point "
        "margin. Of the 30 cases that could fail it, v2 failed 1 and v1 failed 1, too few to settle it."
    )
    assert bold == [f"**{_LEAD}**", f"**{warning}**"]
    html = unlinked(campaign_report_html(result.page))
    assert (
        f'<p class="flag warning"><b aria-hidden="true">!</b><strong>{warning}' in html
        and 'role="alert"' not in html[: html.index("</header>")]
    )
    assert "not adopted" not in "\n".join(lines)
    # The question asks whether v2 does better and writes no margin (a superiority question): the pass rate is read
    # against zero both ways, so no line says it is never shown worse, no margin is filed, and the row reads not
    # separated, its interval reaching below zero.
    assert not [line for line in lines if line.startswith("On classify, the evidence cannot tell a drop")]
    assert module.invoices.margin is None and result.margin is None
    assert (
        "| invoices | v2 | 30 | 73 of 90 trials | 88 of 90 trials | +17 pts | -1 to +26 | — | cannot tell apart |"
    ) in lines
    assert "no worse than the margin" not in "\n".join(lines)


def test_3d_run_as_a_script_in_ci_it_writes_the_report_and_exits_4_by_the_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    copy = offline_copy("3-invoices", tmp_path, monkeypatch)
    with pytest.raises(SystemExit) as exited:
        runpy.run_path(str(copy / "ci.py"), run_name="__main__")

    out = capsys.readouterr().out
    assert out.startswith("invoices: ") and "v1 (control): 90 result(s)" in out
    html = (copy / "report" / "report.html").read_text(encoding="utf-8")
    assert what_runs(html) == [], "the page runs the kit's script and nothing else"
    (written,) = (copy / "report").glob("bundle-*.json")
    bundle = AnalysisContextBundle.from_json(written.read_text(encoding="utf-8"))
    assert bundle.fingerprint() in html
    assert (copy / "report" / "report.md").is_file()
    assert exited.value.code == 4, "guardrail.py's guardrail is undecided, so the gate fails the step"


def test_3c_run_from_the_command_line_it_says_why_it_exits_4_in_one_line(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    offline_copy("3-invoices", tmp_path, monkeypatch)
    assert main(["guardrail.py"]) == 4
    out = capsys.readouterr().out.splitlines()
    # The last line opens with the result as the page leads with it, then names what failed the gate in the page's
    # words: the guardrail, the arm and its Decision cell.
    assert out[-1] == (
        f"exit 4: {_LEAD} The gate failed on guardrail No invented total score, v2 vs v1: undecided: v2 may be worse "
        "than v1 by more than the margin (the interval reaches both sides of -2 pts, the declared margin, so the arm is "
        "shown neither within it nor beyond it)."
    )


#: The grid's four arms, each a cell named by its level of both factors; the first is the control.
_CELLS = ["prompt=v1, effort=low", "prompt=v2, effort=low", "prompt=v1, effort=high", "prompt=v2, effort=high"]


async def test_3e_the_grid_crosses_two_factors_and_reads_one_lever_against_another_control(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    offline_copy("3-invoices", tmp_path, monkeypatch)
    module = load(Path("grid.py"), "grid", monkeypatch)
    low, high = count_calls(module.low, monkeypatch), count_calls(module.high, monkeypatch)

    result = await module.grid.run(store=":memory:")
    # Arms named as k=v pairs are a factorial design: prompt and effort, two levels each, beside the run's model.
    assert module.grid.factors == ("prompt", "effort") and result.factors == ("model", "prompt", "effort")
    assert list(result.arms) == _CELLS and result.control == _CELLS[0]
    # Each effort's cells replay their own recording, every recorded call once.
    assert len(low.calls) == recorded_spend(RECORDING)[0] == 180
    assert len(high.calls) == recorded_spend(RECORDING_HIGH)[0] == 180
    for summary in result.arms.values():
        assert summary.status == "completed" and (summary.n_results, summary.n_excluded) == (90, 0)
    # High effort costs more per reply on either prompt, and a cost regression fails the gate like any other.
    regressed = [verdict for verdict in result.gate().gated if verdict.outcome == "regressed"]
    assert [verdict.heading for verdict in regressed] == ["Production cost", "Production cost"]
    assert result.exit_code() == 4
    # The same runs read against another control, no run repeated: the prompt's effect at high effort.
    again = await result.against(_CELLS[2])
    assert again.control == _CELLS[2] and again.run_ids == result.run_ids
    assert {row["arm"] for row in again.contrasts("accuracy")} == set(_CELLS) - {_CELLS[2]}


def test_3e_run_as_a_script_it_prints_the_grid_then_reads_it_against_the_high_effort_control(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    copy = offline_copy("3-invoices", tmp_path, monkeypatch)
    runpy.run_path(str(copy / "grid.py"), run_name="__main__")
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("grid: ") and "prompt=v1, effort=low (control): 90 result(s): 90 scored" in out
    assert [line for line in out if line.startswith("grid, against prompt=v1, effort=high: ")]
    assert "prompt=v1, effort=high (control): 90 result(s): 90 scored" in out


@pytest.mark.parametrize("example", [COMPARE, MARGIN, GUARDRAIL], ids=lambda path: path.name)
def test_the_docstring_states_what_the_recording_cost(example: Path) -> None:
    calls, dollars = recorded_spend(RECORDING)
    assert stated_spend(example) == (calls, round(dollars, 4))


def test_the_grid_s_docstring_states_what_its_high_effort_recording_cost() -> None:
    calls, dollars = recorded_spend(RECORDING_HIGH)
    assert stated_spend(GRID) == (calls, round(dollars, 4))


@pytest.mark.parametrize("recording", [RECORDING, RECORDING_HIGH], ids=lambda path: path.name)
def test_the_recording_holds_no_key_material(recording: Path) -> None:
    assert secret_markers(recording) == []


#: Each file's ceiling in non-blank lines, its docstring included: one concept a file keeps each one short.
_LINE_CAPS = {"invoice.py": 15, "compare.py": 20, "margin.py": 20, "guardrail.py": 22, "ci.py": 10, "grid.py": 28}


@pytest.mark.parametrize("example", FILES, ids=lambda path: path.name)
def test_each_file_stays_short(example: Path) -> None:
    assert non_blank_lines(example) <= _LINE_CAPS[example.name]


def test_only_the_shared_module_declares_no_eval_and_every_other_file_builds_its_own_model() -> None:
    """A replay serves each recorded call once per model, so a model shared between two evals refuses the second in a
    process: each file builds its own, and the shared module holds no model and no eval."""
    shared = (RUNG / "invoice.py").read_text(encoding="utf-8")
    assert "Model(" not in shared and "Eval(" not in shared
    for example in (COMPARE, MARGIN, GUARDRAIL, GRID):
        assert "Model(" in example.read_text(encoding="utf-8"), example.name
    assert "Model(" not in CI.read_text(encoding="utf-8"), "ci.py runs guardrail.py's eval, on guardrail.py's model"


def test_the_rung_reaches_the_engine_only_through_public_roots() -> None:
    files = [(path.name, path) for path in FILES]
    assert public_root_violations(SOURCE_ROOT, files, consumer_root=REPO_ROOT) == []
