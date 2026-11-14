"""The three-slot contract: ``Eval(cases, runner, grader)``, what each slot accepts, and the ``Result`` a run returns.

Each slot's forms are pinned here from outside, through the package root's names alone: the cases (a list, or a
file, with three reserved keys), the runner (a function, a mapping of arms whose first key is the control, ``k=v``
cells, a world, a log) and the graders (functions, field names, expressions, a judge, ``grader(...)``). A field
grader is the classifier the old ``expected=`` declared, and its figures are the engine's classifier figures, as
the test reading both paths side by side shows.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Self

import pytest
from pydantic import BaseModel, Field

from threetears.evals import Eval, Example, Judge, Log, Model, Result, World, grader
from threetears.evals.analysis import campaign_report_markdown, get_campaign
from threetears.evals.quick import as_host
from threetears.evals.quick.graders import UNUSABLE_ANSWER
from threetears.evals.quick.logged import NOT_LOGGED
from threetears.evals.quick.one_call import CallableArm, callable_host, run_arms
from threetears.evals.quick.blind import SHEET_COLUMNS
from threetears.evals.quick.world import ToolRefused
from threetears.evals.schema import StopReason
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.recorded_model_support import CALL_COST_USD, record_model

#: Tickets, each with its expected label, how urgent it is, and the kind of case it is.
TICKETS = [
    {"id": "refund", "text": "please refund me", "label": "billing", "urgent": False, "stratum": "plain"},
    {"id": "crash", "text": "the app crashes", "label": "bug", "urgent": True, "stratum": "plain"},
    {"id": "both", "text": "refund the crash", "label": "billing", "urgent": True, "stratum": "lookalike"},
    {"id": "login", "text": "login fails", "label": "account", "urgent": False, "stratum": "plain"},
]


class Ticket(BaseModel):
    """A ticket's triage."""

    label: str
    urgent: bool


async def keywords(case: Mapping[str, Any]) -> Ticket:
    """Label the ticket by its keywords."""
    assert set(case) == {"text", "label", "urgent"}, "the reserved keys never reach the runner"
    text = str(case["text"])
    label = "billing" if "refund" in text else "bug" if "crash" in text else "account"
    return Ticket(label=label, urgent="crash" in text)


async def crash_first(case: Mapping[str, Any]) -> Ticket:
    """Label the ticket by its keywords."""
    text = str(case["text"])
    label = "bug" if "crash" in text else "billing" if "refund" in text else "account"
    return Ticket(label=label, urgent=False)


def short_label(case: Mapping[str, Any], out: Ticket) -> bool:
    """The label is a single short word."""
    return len(out.label) <= 7


# --- the cases slot -------------------------------------------------------------------------------------------------


async def test_cases_are_read_from_a_jsonl_csv_json_or_toml_file_and_name_the_eval(tmp_path: Path) -> None:
    rows = [{key: value for key, value in ticket.items() if key != "urgent"} for ticket in TICKETS]
    (tmp_path / "tickets.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    (tmp_path / "tickets.json").write_text(json.dumps({"cases": rows}), encoding="utf-8")
    with (tmp_path / "tickets.csv").open("w", encoding="utf-8", newline="") as written:
        writer = csv.DictWriter(written, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (tmp_path / "tickets.toml").write_text(
        "".join(f"[[cases]]\nid = {row['id']!r}\ntext = {row['text']!r}\nlabel = {row['label']!r}\n" for row in rows),
        encoding="utf-8",
    )

    for suffix in (".jsonl", ".json", ".csv", ".toml"):
        ev = Eval(tmp_path / f"tickets{suffix}", keywords, ["label"], k=1)
        assert ev.name == "tickets" and [case["id"] for case in ev.cases] == [row["id"] for row in rows]


@pytest.mark.parametrize(
    ("cases", "refusal"),
    [
        ([], "at least one case"),
        (["not an object"], "are not JSON objects"),
        ("tickets.yaml", "file, not tickets.yaml"),
        ("missing.jsonl", "is not there"),
    ],
)
def test_unusable_cases_are_refused_where_the_eval_is_written(cases: Any, refusal: str) -> None:
    with pytest.raises(ValueError, match=refusal):
        Eval(cases, keywords, ["label"])


async def test_the_reserved_keys_name_and_sort_each_case_and_reach_no_runner_or_grader() -> None:
    seen: list[Mapping[str, Any]] = []

    def records(case: Mapping[str, Any], out: Ticket) -> bool:
        """Records what a grader is handed."""
        seen.append(case)
        return True

    result = await Eval(TICKETS, keywords, ["label", records], k=1).run(store=":memory:")

    assert all(set(case) == {"text", "label", "urgent"} for case in seen)
    assert {result.case for result in result.results()} == {"refund", "crash", "both", "login"}
    assert {stratum.stratum for stratum in result.summary.strata} >= {"plain", "lookalike"}


# --- the grader slot ------------------------------------------------------------------------------------------------


async def test_a_field_grader_is_the_classifier_expected_was_with_the_same_figures() -> None:
    by_field = (await Eval(TICKETS, keywords, ["label", short_label], k=2).run(store=":memory:")).summary
    host = callable_host([short_label])
    (by_expected,) = await run_arms(
        TICKETS,
        [CallableArm(keywords)],
        [short_label],
        scope_id="expected",
        host=host,
        template_name="expected",
        expected=lambda case: str(case["label"]),
        label_field="label",
        k=2,
    )

    assert by_field.confusion == by_expected.confusion
    assert by_field.labels == by_expected.labels
    assert [(m.name, m.mean, m.n) for m in by_field.measures] == [(m.name, m.mean, m.n) for m in by_expected.measures]


async def test_every_field_after_the_first_is_a_pass_rate_of_its_own_with_its_own_confusion_matrix() -> None:
    result = await Eval(TICKETS, keywords, ["label", "urgent"], k=1).run(store=":memory:")

    urgent = next(measure for measure in result.summary.measures if measure.name == "urgent")
    assert urgent.mean == 1.0 and urgent.n == 4
    confusion, labels = result.labels("urgent")
    assert {(cell.expected, cell.predicted, cell.count) for cell in confusion} == {
        ("false", "false", 2),
        ("true", "true", 2),
    }
    assert {label.label: label.f1 for label in labels} == {"false": 1.0, "true": 1.0}
    assert result.labels() == (list(result.summary.confusion), list(result.summary.labels))
    with pytest.raises(ValueError, match="no field grader"):
        result.labels("text")


async def test_an_answer_with_no_usable_label_is_counted_as_its_own_label() -> None:
    async def says_nothing(case: Mapping[str, Any]) -> dict[str, Any]:
        """Answers with no label."""
        return {"label": "  "}

    result = await Eval(TICKETS[:1], says_nothing, ["label"], k=1).run(store=":memory:")

    assert [(cell.expected, cell.predicted) for cell in result.summary.confusion] == [("billing", UNUSABLE_ANSWER)]


async def test_grader_declares_a_margin_a_range_or_a_guard_on_its_reading() -> None:
    def rating(case: Mapping[str, Any], out: Ticket) -> float:
        """A one-to-five rating."""
        return 4.0

    ev = Eval(
        TICKETS,
        {"keywords": keywords, "crash-first": crash_first},
        [grader("label", margin=0.25), grader(rating, range=(1, 5)), grader(short_label, guard=0.1)],
        k=1,
        name="keywords vs crash-first",
    )
    host = ev.host(":memory:")

    assert host.profile.measures.get("rating").value_range == (1.0, 5.0)
    guarded = host.profile.measures.get("short_label")
    assert guarded.guardrail is True and guarded.materiality_threshold == 0.1
    assert ev.plan.run_margins == {"accuracy": 0.25}


@pytest.mark.parametrize(
    ("graders", "refusal"),
    [
        ([], "at least one grader"),
        ([42], "a grader is a function"),
        (["label", "label"], "graded more than once"),
        (["score"], "engine core measure"),
        (["state.light == 'on'"], "read a world's state"),
    ],
)
def test_unusable_graders_are_refused_where_the_eval_is_written(graders: Any, refusal: str) -> None:
    with pytest.raises(ValueError, match=refusal):
        Eval(TICKETS, keywords, graders)


@pytest.mark.parametrize(
    ("declared", "refusal"),
    [
        ({"margin": 0.1, "guard": 0.1}, "pass margin= or guard="),
        ({"margin": -1}, "positive number"),
        ({"range": (5, 1)}, "low first"),
    ],
)
def test_grader_refuses_a_declaration_no_reading_could_carry(declared: dict[str, Any], refusal: str) -> None:
    with pytest.raises(ValueError, match=refusal):
        grader(short_label, **declared)


# --- the runner slot: a comparison ----------------------------------------------------------------------------------


async def test_a_mapping_is_a_comparison_whose_first_key_is_the_control() -> None:
    result = await Eval(
        TICKETS, {"keywords": keywords, "crash-first": crash_first}, ["label"], k=2, name="keywords vs crash-first"
    ).run(store=":memory:")

    assert isinstance(result, Result)
    assert result.control == "keywords" and list(result.arms) == ["keywords", "crash-first"]
    assert result.factors == ("candidate",)
    assert {row["arm"] for row in result.contrasts()} == {"crash-first"}
    assert result.exit_code() in (0, 4)
    with pytest.raises(ValueError, match="compares 2 arms"):
        _ = result.summary
    assert "(control)" in result.render() and "stored in memory" in result.render()


async def test_arms_named_as_pairs_are_the_cells_of_a_factorial() -> None:
    ev = Eval(
        TICKETS,
        {"prompt=v1, effort=low": keywords, "prompt=v2, effort=low": crash_first},
        ["label"],
        k=1,
        name="factorial",
    )
    result = await ev.run(store=":memory:")

    assert ev.levers == ("prompt", "effort")
    assert result.factors == ("model", "prompt", "effort")
    assert set(result.arms) == {"prompt=v1, effort=low", "prompt=v2, effort=low"}


@pytest.mark.parametrize(
    ("runner", "refusal"),
    [
        ({}, "names no arm"),
        ({" ": keywords}, "is blank"),
        ({"prompt=v1": keywords, "baseline": crash_first}, "not both"),
        ({"prompt=v1": keywords, "effort=low": crash_first}, "same factors"),
        (42, "a function"),
    ],
)
def test_unusable_runners_are_refused_where_the_eval_is_written(runner: Any, refusal: str) -> None:
    with pytest.raises(ValueError, match=refusal):
        Eval(TICKETS, runner, ["label"])


def test_several_arms_with_no_name_and_no_cases_file_are_refused(tmp_path: Path) -> None:
    """An eval's name is where its runs are kept. One built from its arms (``keywords vs crash-first``) would move
    them, and split the comparison's history, each time an arm is added, renamed or dropped, so it is never built."""
    arms = {"keywords": keywords, "crash-first": crash_first}
    with pytest.raises(ValueError, match=r'^name this eval \(name="invoices"\): its name is where its runs are kept'):
        Eval(TICKETS, arms, ["label"])
    with pytest.raises(ValueError, match="'prompt=v1', 'prompt=v2'"):
        Eval(TICKETS, {"prompt=v1": keywords, "prompt=v2": crash_first}, ["label"])  # factors are arms too
    # Named by name=, by the cases file's stem, or by its one arm: never refused.
    assert Eval(TICKETS, arms, ["label"], name="triage").name == "triage"
    (tmp_path / "tickets.jsonl").write_text("".join(json.dumps(row) + "\n" for row in TICKETS), encoding="utf-8")
    assert Eval(tmp_path / "tickets.jsonl", arms, ["label"]).name == "tickets"
    assert Eval(TICKETS, {"keywords": keywords}, ["label"]).name == "keywords"
    assert Eval(TICKETS, keywords, ["label"]).name == "keywords"


# --- the runner slot: a world ---------------------------------------------------------------------------------------


class Room(BaseModel):
    light: Literal["on", "off"] = Field(description="what the agent must set")
    daylight: bool = Field(default=False, description="whether the room is lit with the light off")


def switch(state: Room, to: Literal["on", "off"]) -> str:
    """Switch the room's light on or off."""
    state.light = to
    return f"the light is {to}"


async def test_a_world_seeds_each_case_hands_the_agent_its_tools_and_grades_the_state_left() -> None:
    room = World(Room, tools=[switch])

    async def agent(case: Mapping[str, Any], tools: Any) -> str:
        """Set the light the room needs."""
        with pytest.raises(ToolRefused):
            await tools["switch"](to="dim")
        return str(await tools["switch"](to="on"))

    def lit(case: Mapping[str, Any], out: Any) -> bool:
        """The agent left the light on."""
        return bool(out.state.light == "on" and [call.action for call in out.calls] == ["switch"])

    cases = [{"id": "dark", "ask": "it is dark", "world": {"light": "off"}}, {"id": "lit", "world": {"light": "on"}}]
    result = await Eval(cases, room(agent), ["state.light == 'on'", 'call_count("room.switch") <= 1', lit], k=1).run(
        store=":memory:"
    )

    assert result.summary.n_scored == 2
    assert {goal.check: goal.passed for goal in result.summary.goal_checks} == {
        "state.light == 'on'": 2,
        'call_count("room.switch") <= 1': 2,
    }
    assert next(m for m in result.summary.measures if m.name == "lit").mean == 1.0


def test_a_world_refuses_a_state_field_with_no_reason_or_a_tool_with_no_types() -> None:
    class Bare(BaseModel):
        light: str

    def untyped(state: Room, to) -> str:  # type: ignore[no-untyped-def]
        """No type on its parameter."""
        return "x"

    with pytest.raises(ValueError, match="has no description"):
        World(Bare, tools=[switch])
    with pytest.raises(ValueError, match="has no type"):
        World(Room, tools=[untyped])
    with pytest.raises(ValueError, match="takes no recording"):
        World(Room, tools=[switch], recording="tools.jsonl")


async def test_a_world_of_lookups_records_its_answers_once_and_replays_them_after(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("RECORD", raising=False)
    asked: list[str] = []

    def search(query: str) -> dict[str, int]:
        """Search the help centre."""
        asked.append(query)
        return {"hits": len(query)}

    lookups = World(tools=[search], recording=tmp_path / "search.jsonl")

    async def agent(case: Mapping[str, Any], tools: Any) -> int:
        """Look the ticket up."""
        found = await tools["search"](query=case["text"])
        return int(found["hits"])

    def found(case: Mapping[str, Any], out: Any) -> bool:
        """Something was found."""
        return bool(out.answer > 0)

    first = await Eval(TICKETS, lookups(agent), [found], k=2).run(store=":memory:")
    calls_recorded = len(asked)
    second = await Eval(TICKETS, lookups(agent), [found], k=2).run(store=":memory:")

    assert calls_recorded == 8 and len(asked) == 8, "the replay calls no tool"
    assert first.summary.n_scored == second.summary.n_scored == 8
    assert any("recorded to" in line for line in first.disclosures)
    assert any("replayed from" in line for line in second.disclosures)

    changed = [{**ticket, "text": ticket["text"] + "!"} for ticket in TICKETS[:1]]
    missed = await Eval(changed, lookups(agent), [found], k=1).run(store=":memory:")
    assert missed.summary.status == "failed", "a lookup its recording does not hold fails the run (#1162)"
    assert missed.summary.n_excluded == 1 and "holds no answer" in missed.render()


# --- the runner slot: a log -----------------------------------------------------------------------------------------


async def test_a_log_is_graded_as_logged_and_a_repeat_never_logged_is_excluded_not_logged(tmp_path: Path) -> None:
    log = Log(tmp_path / "outputs.jsonl")
    log(TICKETS[0], {"label": "billing"}, cost_usd=0.01)
    log(TICKETS[0], {"label": "bug"}, cost_usd=0.01)
    log(TICKETS[1], {"label": "bug"})

    result = await Eval(TICKETS[:2], log, ["label"], k=2).run(store=":memory:")

    summary = result.summary
    assert (summary.n_scored, summary.n_excluded) == (3, 1)
    assert [error for error in summary.errors if NOT_LOGGED in error]
    assert f"excluded: 1 x apparatus: NotLogged: {NOT_LOGGED}" in result.render()
    with pytest.raises(ValueError, match="carries none"):
        log({"text": "no id"}, "x")


async def test_a_single_arm_prints_first_how_many_cases_passed_and_which_it_missed() -> None:
    result = await Eval(TICKETS, crash_first, ["label", "urgent"]).run(store=":memory:")

    # The aspect page's verdict line, as the campaign page leads with its first sentence: the count, then the misses.
    first = result.render().splitlines()[0]
    assert (
        first == f"{result.name}: crash_first passed 6 of 12 trials on 4 cases. Cases that failed a trial: crash, both."
    )


async def test_a_single_arm_s_first_line_says_what_its_page_says_of_the_trials_lost_and_its_lines_say_the_cause(
    tmp_path: Path,
) -> None:
    log = Log(tmp_path / "outputs.jsonl")
    log(TICKETS[0], {"label": "billing"})
    log(TICKETS[0], {"label": "billing"})

    result = await Eval(TICKETS[:2], log, ["label"], k=2).run(store=":memory:")

    (cause, *_) = next(repeat for repeat in result.results() if repeat.case == "crash").errors
    assert cause.startswith(f"apparatus: NotLogged: {NOT_LOGGED}: case crash ")
    # Half its trials were lost, so its page's verdict line is the setup gate's, and so is the print's first line;
    # the cause each lost trial records is the arm's excluded line, below it.
    rendered = result.render()
    (verdict,) = [block.text for block in result.report.blocks if getattr(block, "source", None) == "verdict"]
    assert rendered.splitlines()[0] == f"{result.name}: {verdict}"
    assert verdict.startswith("The test setup failed: it measured the setup instead of the models.")
    assert [line for line in rendered.splitlines() if line.strip().startswith("excluded: ") and NOT_LOGGED in line]


# --- spend from Model -----------------------------------------------------------------------------------------------


async def test_a_runner_calling_model_reports_its_spend_with_no_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("RECORD", raising=False)
    path = tmp_path / "recordings" / "label.jsonl"
    await record_model(path, "scripted/labeller", [str(t["text"]) for t in TICKETS], lambda _: "billing", repeats=2)
    model = Model("scripted/labeller", recording=path)

    async def labeller(case: Mapping[str, Any]) -> str:
        """Label the ticket."""
        return await model.text(str(case["text"]))

    async def plain(case: Mapping[str, Any]) -> dict[str, str]:
        """Label the ticket."""
        return {"label": await model.text(str(case["text"]))}

    result = await Eval(TICKETS, labeller, ["label"], k=2).run(store=":memory:")

    assert result.summary.total_cost_usd == pytest.approx(8 * CALL_COST_USD)
    assert result.summary.role_spend("candidate").calls == 8
    # With no recording and no key, the one Model is refused when it is built, naming the variable.
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="no key for openrouter: pass api_key= or set OPENROUTER_API_KEY"):
        Model("scripted/labeller")
    assert plain.__doc__


# --- the judge ------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Verdict:
    content: str
    model: str = "scripted/judge"
    served_model: str | None = "scripted/judge"
    input_tokens: int | None = 10
    output_tokens: int | None = 5
    cost_usd: float | None = 0.0001
    price_source: str | None = "scripted rate"
    reasoning_tokens: int | None = None
    stop_reason: StopReason = "end_turn"
    temperature: float | None = None


# parity-with: threetears.evals.schema.CompletionClient
class FakeJudge:
    """A judge that passes every answer naming billing, and fails the rest."""

    model_name = "scripted/judge"

    async def generate(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> _Verdict:
        dimension = system.split('single key "', 1)[1].split('"', 1)[0]
        passed = "billing" in user.split("# Output under review\n", 1)[1]
        verdict = {"reasoning": "read it", "criteria_scores": {dimension: "pass" if passed else "fail"}}
        return _Verdict(json.dumps(verdict))

    async def aclose(self) -> None:
        return None

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self.aclose()


def _sheet_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as read:
        return list(csv.DictReader(read))


def _save(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as filled:
        writer = csv.DictWriter(filled, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


async def test_a_judge_is_a_grader_its_answers_written_to_read_blind_and_read_back_as_its_calibration(
    tmp_path: Path,
) -> None:
    labels = tmp_path / "labels.csv"
    judge = Judge(FakeJudge(), {"billing": "The answer says the ticket is about billing."}, labels=labels)
    ev = Eval(TICKETS, keywords, ["label", judge], k=1)

    first = await ev.run(store=tmp_path / "evals.sqlite")
    assert first.calibration_lines() == [
        f"judge: 0 of 4 items read blind: judge uncalibrated ({labels.name} is not there yet); next step: open the "
        f"blind_read.html the report writes, answer its items from the top, and save its labels as {labels.name}; "
        "the calibrated tier's floor is 20 judged answers per dimension and the read holds 4, so it needs 16 more"
    ]
    written = first.write(tmp_path / "report")
    assert (written["blind_read"].name, written["blind_read_csv"].name) == ("blind_read.html", "blind_read.csv")
    assert "to_label" not in written and not (tmp_path / "report" / "to_label.csv").exists()
    rows = _sheet_rows(written["blind_read_csv"])
    assert len(rows) == 4 and list(rows[0]) == list(SHEET_COLUMNS)
    assert all(row["label"] == "" for row in rows), "no verdict is on the sheet"
    # No run or result of the store is named on either file, and the judge's model is on neither.
    assert first.blind_read is not None
    drawn = [output.result_id for item in first.blind_read.items for output in item.outputs]
    ids = [first.run_id, *drawn, "scripted/judge"]
    for path in (written["blind_read"], written["blind_read_csv"]):
        data = path.read_text(encoding="utf-8")
        assert not [mark for mark in ids if mark in data], path.name

    # The person agrees with the judge on every answer, reading from the top.
    for row in rows:
        row["label"] = "pass" if "billing" in row["answer"] else "fail"
        row["rater"] = "owner"
    _save(labels, rows)

    again = await ev.run(store=tmp_path / "evals.sqlite")
    (calibration,) = again.calibrations
    # Each label pairs once with the judge's score of the byte-identical answer in the replayed run.
    assert (calibration.dimension, calibration.labels) == ("answer.billing", 4)
    assert calibration.kappa == pytest.approx(1.0)
    read = again.blind_read
    assert read is not None and read.report is not None
    assert [(s.rater, s.rater_kind, s.standing, s.answered) for s in read.report.standings] == [
        ("owner", "person", "random", 4)
    ]
    lines = again.calibration_lines()
    assert lines[0] == "judge: 4 of 4 items read blind, in the drawn order (a random sample)"
    assert lines[1].startswith("  answer.billing  4 results from 4 cases  kappa 1.00")
    assert "undetermined: needs " in lines[1] and lines[1].endswith("the judge failed 2 of the 2 answers you failed")
    assert len(lines) == 2

    # Answered out of order, the same labels read as selected.
    _save(labels, [row if n in (1, 3) else {**row, "label": ""} for n, row in enumerate(rows)])
    hunted = await ev.run(store=tmp_path / "evals.sqlite")
    assert hunted.calibration_lines()[0] == (
        "judge: 2 of 4 items read blind, out of the drawn order (selected, not a random sample)"
    )


async def test_an_older_to_label_sheet_still_reads_and_counts_as_selected(tmp_path: Path) -> None:
    labels = tmp_path / "labels.csv"
    judge = Judge(FakeJudge(), {"billing": "The answer says the ticket is about billing."}, labels=labels)
    ev = Eval(TICKETS, keywords, ["label", judge], k=1)
    first = await ev.run(store=":memory:")
    # The sheet an earlier version wrote: store order, a row per judged answer, the run's and the result's ids on it.
    sheet = _sheet_rows(first.write(tmp_path / "report")["blind_read_csv"])
    old = [
        {
            "case": str(n),
            "repeat": "1",
            "dimension": row["dimension"],
            "case_material": row["case_material"],
            "answer": row["answer"],
            "label": "pass" if "billing" in row["answer"] else "fail",
            "reason": "",
            "rater": "owner",
            "run_id": "a-run-since-gone",
            "result_id": f"a-result-since-gone-{n}",
            "output_fingerprint": row["output_fingerprint"],
            "criterion_fingerprint": row["criterion_fingerprint"],
        }
        for n, row in enumerate(sheet)
    ]
    _save(labels, old)

    again = await ev.run(store=":memory:")

    (calibration,) = again.calibrations
    assert (calibration.labels, calibration.kappa) == (4, pytest.approx(1.0))
    assert len(again.loose_labels) == 4 and again.blind_read is not None and again.blind_read.report is not None
    assert again.blind_read.report.standings == []
    (reading,) = again.blind_read.report.agreement
    assert (reading.pairs, reading.blind_pairs, reading.random_pairs) == (4, 0, 0)
    assert again.calibration_lines()[0] == (
        f"judge: 4 label(s) in {labels.name} are not on this run's draw (an older sheet, or another run's): counted "
        "toward agreement as selected, not a random sample"
    )


#: Eight tickets two prompts answer differently, so every item of their blind read is a pair.
PAIRED = [{"id": f"t{n}", "text": f"ticket {n} about a refund"} for n in range(8)]


async def terse(case: Mapping[str, Any]) -> str:
    return f"billing: {case['text']}"


async def wordy(case: Mapping[str, Any]) -> str:
    return f"This one is a billing matter, I believe: {case['text']}"


async def test_two_arms_are_read_as_pairs_and_the_preference_is_printed_once_the_read_is_finished(
    tmp_path: Path,
) -> None:
    labels = tmp_path / "labels.csv"
    judge = Judge(FakeJudge(), {"billing": "The answer says the ticket is about billing."}, labels=labels)
    ev = Eval(PAIRED, {"terse": terse, "wordy": wordy}, [judge], k=1, name="tickets")

    first = await ev.run(store=":memory:")
    read = first.blind_read
    assert read is not None and [(arm.key, arm.role) for arm in read.arms] == [
        ("terse", "control"),
        ("wordy", "candidate"),
    ]
    assert len(read.items) == 8 and all(len(item.outputs) == 2 for item in read.items)
    written = first.write(tmp_path / "report")
    for path in (written["blind_read"], written["blind_read_csv"]):
        data = path.read_text(encoding="utf-8")
        assert "terse" not in data and "wordy" not in data, f"{path.name} names an arm"
    rows = _sheet_rows(written["blind_read_csv"])
    wordy_slot = {item.item: next(o.slot for o in item.outputs if o.arm == "wordy") for item in read.items}

    def answered(upto: int) -> list[dict[str, str]]:
        filled, seen = [], set()
        for row in rows:
            if int(row["position"]) <= upto:
                row = {**row, "label": "pass", "prefer": "" if row["item"] in seen else wordy_slot[row["item"]]}
                seen.add(row["item"])
            filled.append(row)
        return filled

    # Part way through, the line says how far the read is and nothing about which arm is ahead.
    _save(labels, answered(5))
    partial = (await ev.run(store=":memory:")).calibration_lines()
    assert "preference: shown when the first 8 items are answered (5 of 8)" in partial
    assert partial[-1] == "next: keep going from item 6 in blind_read.html"
    assert not any("wordy" in line or "sign test" in line for line in partial)

    _save(labels, answered(8))
    done = await ev.run(store=":memory:")
    assert "preference: wordy over terse on 8 of 8, sign test p 0.008: wordy preferred" in done.calibration_lines()
    assert done.blind_read is not None and done.blind_read.report is not None
    (preference,) = done.blind_read.report.preferences
    assert (preference.anchor, preference.random, preference.verdict) == ("person", True, "improved")
    assert "judge: 8 of 8 items read blind, in the drawn order (a random sample)" in str(done)


# parity-with: threetears.evals.schema.CompletionClient
class FakeReviewer:
    """A model that answers a blind read: it passes every output and prefers A."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.calls = 0

    async def generate(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> _Verdict:
        self.calls += 1
        slots = [line.removeprefix("# Output ") for line in user.splitlines() if line.startswith("# Output ")]
        verdicts = {slot: {"answer.billing": "pass"} for slot in slots}
        reply = {"verdicts": verdicts, "prefer": "A" if len(slots) == 2 else None, "reason": "read it"}
        return _Verdict(json.dumps(reply), model=self.model_name, served_model=self.model_name)

    async def aclose(self) -> None:
        return None

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self.aclose()


async def test_a_reviewer_answers_the_blind_read_as_a_reference_never_as_a_person() -> None:
    reviewer = FakeReviewer("scripted/reviewer")
    judge = Judge(FakeJudge(), {"billing": "The answer says the ticket is about billing."}, reviewer=reviewer)
    result = await Eval(PAIRED, {"terse": terse, "wordy": wordy}, [judge], k=1, name="tickets").run(store=":memory:")

    assert reviewer.calls == 8, "one call per item"
    assert result.calibrations == (), "agreement with a model is never agreement with people"
    read = result.blind_read
    assert read is not None and read.report is not None
    assert {(a.rater, a.rater_kind) for a in read.answers} == {("scripted/reviewer", "reference")}
    lines = result.calibration_lines()
    assert lines[0] == (
        "judge: 8 of 8 items read blind by scripted/reviewer, a reference reviewer, in the drawn order (a random "
        "sample)"
    )
    assert lines[1] == "  checked against scripted/reviewer, not yet against people:"
    assert "scripted/reviewer and the judge are one provider's models" in "\n".join(lines)
    assert any(
        line.startswith("preference of scripted/reviewer (a reference reviewer, not a person): ") for line in lines
    )
    assert not any(line.startswith("preference: ") or line.startswith("next:") for line in lines)
    # It chose slot A every time, whatever was in it: the slots are balanced, so that reads as no preference, said.
    assert (
        "preference of scripted/reviewer (a reference reviewer, not a person): wordy over terse on 4 of 8, slot A "
        "chosen on 8 of 8, sign test p 1.00: cannot tell apart"
    ) in lines

    # The judge's own model answers nothing a repeat would not: it is refused as calibration, and said.
    same = Judge(
        FakeJudge(),
        {"billing": "The answer says the ticket is about billing."},
        reviewer=FakeReviewer("scripted/judge"),
    )
    refused = await Eval(PAIRED, {"terse": terse, "wordy": wordy}, [same], k=1, name="tickets").run(store=":memory:")
    assert refused.blind_read is not None and refused.blind_read.report is not None
    assert refused.blind_read.report.agreement == []
    assert (
        "  scripted/judge is the judge's own model: its answers repeat the judge and are not read as calibration"
        in refused.calibration_lines()
    )


def test_a_judge_takes_a_client_that_names_its_model_and_guards_only_its_own_dimensions() -> None:
    class Nameless:
        async def generate(self, *, system: str, user: str, response_format: Any = None) -> Any:
            raise AssertionError("never called")

    with pytest.raises(ValueError, match="names none"):
        Judge(Nameless(), {"billing": "about billing"})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="no dimension of the rubric"):
        Judge(FakeJudge(), {"billing": "about billing"}, guard={"tone": 0.1})
    judge = Judge(
        FakeJudge(),
        {"billing": "about billing"},
        guard={"billing": 0.2},
        examples=[Example(dimension="billing", answer="a refund", expected="pass", note="a refund is billing")],
    )
    assert judge.dims[0].axis == "boundary" and judge.guard_margins == {"answer.billing": 0.2}


# --- where it runs --------------------------------------------------------------------------------------------------


async def test_with_no_store_the_runs_are_kept_under_the_working_directory_in_a_scope_named_after_the_eval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    result = await Eval(TICKETS, keywords, ["label"], k=1, name="tickets").run()

    assert (tmp_path / ".evals" / "evals.sqlite").is_file()
    assert result.scope_id == "tickets"
    assert result.render().endswith("stored in .evals/evals.sqlite, scope 'tickets'")


async def test_a_host_of_your_own_holds_the_runs_and_a_store_beside_it_is_refused() -> None:
    ev = Eval(TICKETS, {"arm-a": keywords}, ["label", short_label], k=1)
    host = as_host([ev], store=":memory:")

    result = await ev.run(host=host, scope="mine")

    assert (result.scope_id, result.summary.candidate_model) == ("mine", "arm-a")
    assert result.store == f"host {host.profile.host_id!r}"
    with pytest.raises(ValueError, match="brings its own storage"):
        await ev.run(host=host, store=":memory:")
    with pytest.raises(ValueError, match="declares no measure named short_label"):
        await ev.run(host=as_host([Eval(TICKETS, keywords, ["label"])], store=":memory:"), scope="other")


# --- the question ---------------------------------------------------------------------------------------------------


def test_a_blank_question_is_refused_where_the_eval_is_written() -> None:
    with pytest.raises(ValueError, match="question is filed on its campaign, and this one is blank"):
        Eval(
            TICKETS,
            {"keywords": keywords, "crash_first": crash_first},
            ["label"],
            question="  ",
            name="keywords vs crash_first",
        )


async def test_a_declared_question_reaches_the_result_and_the_page_and_its_campaign_asks_it() -> None:
    """The question is filed on the campaign whole: the result carries it, the page quotes it in place of the line
    saying none was asked, the aspect reads it as asked (never "names no axis"), and a question with no margin files
    none: it is a superiority question, its pass rate read against zero. The same eval with none keeps that line."""
    question = "Does crash_first label tickets right more often than keywords?"
    arms = {"keywords": keywords, "crash_first": crash_first}
    asked = await Eval(TICKETS, arms, ["label", short_label], k=1, question=question, name=" vs ".join(arms)).run(
        store=":memory:"
    )
    unasked = await Eval(TICKETS, arms, ["label", short_label], k=1, name=" vs ".join(arms)).run(store=":memory:")

    assert asked.question == question and unasked.question is None
    asked_page = campaign_report_markdown(asked.page).splitlines()
    assert asked_page[1] == f'"{question}"'
    assert "Exploratory comparison." not in asked_page
    assert "Exploratory comparison." in campaign_report_markdown(unasked.page).splitlines()
    (bundle,) = [fold.report.bundle for fold in asked.page.aspects]
    assert bundle.question_declared and bundle.reading_scope.disclosure is None
    (row,) = asked.page.readings.readings
    assert row.margin is None and row.confirmatory
    assert [
        aspect.margin
        for aspect in off_loop(get_campaign, asked.host.storage, asked.campaign_id, asked.scope_id).aspects
    ] == [None]
    # The graders' order is filed on the aspect's design, so the page can lead on the first, as the printed line does.
    assert asked.grader_readings == ("accuracy", "short_label")
    campaign = off_loop(get_campaign, asked.host.storage, asked.campaign_id, asked.scope_id)
    assert [aspect.design.measure_order for aspect in campaign.aspects if aspect.design] == [
        ["accuracy", "short_label"]
    ]


#: The one line a page says where criteria on the eval's aspect were set once its results existed: what was set late,
#: on the aspect the quick path labelled ``classify``, which keeps its label beside a campaign name cut at its cap.
_SET_LATE = (
    "The aspect and control on classify were set after the results there existed, so they were not fixed in advance."
)


async def test_a_question_declared_over_finished_runs_reads_as_asked_with_the_late_disclosure_once() -> None:
    """Owner, 2026-11-06: a question may sharpen once its runs are read, never refused. Declared over finished runs
    (an exploratory run read again asking it), it reads as asked, not as a lead, and the first screen says once that
    its criteria were set after the results existed; declared before its runs, as ``Eval(question=)`` is, it says
    nothing of the kind."""
    question = "Does crash_first label tickets right more often than keywords?"
    arms = {"keywords": keywords, "crash_first": crash_first}
    explored = await Eval(TICKETS, arms, ["label"], k=1, name=" vs ".join(arms)).run(store=":memory:")
    sharpened = await explored.against("keywords", question=question)
    first = campaign_report_markdown(sharpened.page).split("\n## ", 1)[0]
    assert first.splitlines()[1] == f'"{question}"' and sharpened.question == question
    assert first.count(_SET_LATE) == 1
    assert "a lead" not in first and "Exploratory comparison." not in first
    (bundle,) = [fold.report.bundle for fold in sharpened.page.aspects]
    assert bundle.question_declared
    assert "aspect" in [late.criterion for late in bundle.reading_scope.late_criteria]
    assert all(row.confirmatory for row in sharpened.page.readings.readings)

    asked = await Eval(TICKETS, arms, ["label"], k=1, question=question, name=" vs ".join(arms)).run(store=":memory:")
    assert _SET_LATE not in campaign_report_markdown(asked.page)
    (timely,) = [fold.report.bundle for fold in asked.page.aspects]
    assert timely.reading_scope.late_criteria == []
