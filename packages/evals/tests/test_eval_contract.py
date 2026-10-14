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
from threetears.evals.quick import as_host
from threetears.evals.quick.graders import UNUSABLE_ANSWER
from threetears.evals.quick.logged import NOT_LOGGED
from threetears.evals.quick.one_call import CallableArm, callable_host, run_arms
from threetears.evals.quick.result import LABEL_COLUMNS, TO_LABEL
from threetears.evals.quick.world import ToolRefused
from threetears.evals.schema import StopReason
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
    result = await Eval(TICKETS, {"keywords": keywords, "crash-first": crash_first}, ["label"], k=2).run(
        store=":memory:"
    )

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


async def test_a_judge_is_a_grader_its_answers_written_to_label_and_read_back_as_its_calibration(
    tmp_path: Path,
) -> None:
    labels = tmp_path / "labels.csv"
    judge = Judge(FakeJudge(), {"billing": "The answer says the ticket is about billing."}, labels=labels)
    ev = Eval(TICKETS, keywords, ["label", judge], k=1)

    first = await ev.run(store=tmp_path / "evals.sqlite")
    assert first.calibration_lines() == [
        f"judge: 0 labels: judge uncalibrated ({labels} is not there yet); next step: rate about 20 outputs in the "
        "to_label.csv the report writes, without looking at the judge's verdict"
    ]
    written = first.write(tmp_path / "report")
    assert written["to_label"].name == TO_LABEL
    with written["to_label"].open(encoding="utf-8", newline="") as read:
        rows = list(csv.DictReader(read))
    assert len(rows) == 4 and list(rows[0]) == list(LABEL_COLUMNS)
    assert all(row["label"] == "" for row in rows), "the judge's verdict is withheld from the person"

    # The person agrees with the judge on every answer.
    for row in rows:
        row["label"] = "pass" if "billing" in row["answer"] else "fail"
        row["rater"] = "owner"
    with labels.open("w", encoding="utf-8", newline="") as filled:
        writer = csv.DictWriter(filled, fieldnames=LABEL_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    again = await ev.run(store=tmp_path / "evals.sqlite")
    (calibration,) = again.calibrations
    # Each label pairs once with the judge's score of the byte-identical answer in the replayed run.
    assert (calibration.dimension, calibration.labels) == ("answer.billing", 4)
    assert calibration.kappa == pytest.approx(1.0)
    assert again.calibration_lines()[0].startswith(f"judge against {labels}: answer.billing: 4 label(s), kappa 1.00")


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
