"""A newcomer's mistakes are refused with the fix named, never turned into a score.

The errors a newcomer hits on the file door (design-on-ramp §3.4): asking the command line for help first, running a
model with no key, comparing two arms with no name, grading a field the answers do not carry, and a runner (or a
grader) that cannot take what it is handed. The last two once scored 0% and exited 0, a wiring bug read as the product
failing; each is now refused (exit 2 at the file door), and each refusal names the change to make, in the names the
author typed.
"""

from __future__ import annotations

import functools
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from threetears.evals import Eval, Model, World
from threetears.evals.quick import EXIT_OK, EXIT_REFUSED
from threetears.evals.quick.files import main

CASES = [
    {"id": "1", "text": "hello", "label": "greet"},
    {"id": "2", "text": "bye", "label": "leave"},
    {"id": "3", "text": "hi", "label": "greet"},
]


def _right(case: Mapping[str, Any]) -> str:
    return "leave" if case["text"] == "bye" else "greet"


@pytest.fixture
def in_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    # Loading a Python file puts its directory on sys.path for the process; the test's process gets its own back.
    monkeypatch.setattr(sys, "path", [*sys.path])
    return tmp_path


def _eval_file(directory: Path, body: str) -> str:
    """An eval file over three cases, with ``body`` defining its runner and its eval."""
    (directory / "cases.jsonl").write_text(
        "".join(f'{{"id": "{case["id"]}", "text": "{case["text"]}", "label": "{case["label"]}"}}\n' for case in CASES),
        encoding="utf-8",
    )
    (directory / "probe.py").write_text("from threetears.evals import Eval\n\n" + body, encoding="utf-8")
    return "probe.py"


# --- row 1: the door is the first thing the command line names ------------------------------------------------------


def test_help_opens_with_the_file_door_and_rung_one(in_tmp: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exited:
        main(["--help"])
    assert exited.value.code == EXIT_OK
    printed = capsys.readouterr().out
    first, second, *_ = printed.splitlines()
    assert first == "usage: python -m threetears.evals FILE.py [--verbose]"
    assert "rung 1 of docs/ladder.md" in second
    assert "python -m threetears.evals examples/1-triage/triage.py" in printed
    # The host commands' own help follows, after the door: --host is theirs, and said to be.
    assert printed.index("FILE.py") < printed.index("{run,ls,report")
    assert "a first eval does not need" in printed


def test_nothing_at_all_prints_the_door_and_exits_2(in_tmp: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == EXIT_REFUSED
    said = capsys.readouterr().err
    assert said.splitlines()[0] == "usage: python -m threetears.evals FILE.py [--verbose]"
    assert "examples/1-triage/triage.py" in said


@pytest.mark.parametrize("given", ["cases.jsonl", "examples/1-triage"])
def test_a_first_argument_neither_an_eval_file_nor_a_command_names_both(
    given: str, in_tmp: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([given]) == EXIT_REFUSED
    said = capsys.readouterr().err
    assert said.startswith(f"python -m threetears.evals {given}: an eval file ends in .py or .toml")
    assert "a command is one of run, ls," in said and f"{given} is neither" in said


# --- row 2: no key names the recording that would make one unneeded ------------------------------------------------


def test_no_key_and_no_recording_names_recording_as_the_way_to_run_without_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("RECORD", raising=False)
    with pytest.raises(ValueError) as refused:
        Model("deepseek/deepseek-v4-flash")
    said = str(refused.value)
    assert said.startswith("no key for openrouter: pass api_key= or set OPENROUTER_API_KEY; ")
    assert 'pass recording="recordings/deepseek-v4-flash.jsonl" and run once with it set' in said


def test_no_key_with_a_recording_not_yet_made_names_that_recording(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("RECORD", raising=False)
    with pytest.raises(ValueError, match=r"; or record .*calls\.jsonl once with it set$"):
        Model("deepseek/deepseek-v4-flash", recording=tmp_path / "calls.jsonl")


# --- row 9: two arms and no name ------------------------------------------------------------------------------------


def test_two_arms_with_no_name_name_both_fixes() -> None:
    with pytest.raises(ValueError) as refused:
        Eval(CASES, {"keywords": _right, "crash-first": _right}, ["label"])
    said = str(refused.value)
    assert said.startswith('name this eval: pass name="..."')
    assert "or give its cases as a file, whose stem names it" in said
    assert "('keywords', 'crash-first')" in said
    assert "invoices" not in said, "no example name from another app"


# --- row 6: a field grader no answer carries ------------------------------------------------------------------------


def test_a_field_no_answer_carries_exits_2_at_the_door_naming_the_fields_they_carry(
    in_tmp: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    body = 'def classify(case):\n    return {"category": "greet"}\n\nrow6 = Eval("cases.jsonl", classify, ["label"], k=1)\n'
    assert main([_eval_file(in_tmp, body)]) == EXIT_REFUSED
    captured = capsys.readouterr()
    assert captured.err == (
        "python -m threetears.evals probe.py: cases: no answer has a field 'label', so its grader could score none of "
        "them; the answers carry category: grade the field they carry, or have the runner return 'label'\n"
    )
    assert "Accuracy" not in captured.out, "no reading is printed for an eval refused as miswired"


async def test_answers_that_are_all_none_say_to_return_the_answer() -> None:
    def forgets_to_return(case: Mapping[str, Any]) -> None:
        case.get("text")

    with pytest.raises(ValueError, match="every answer was None: return the answer from the runner"):
        await Eval(CASES, forgets_to_return, ["label"], k=1).run(store=":memory:")


async def test_answers_with_no_fields_name_the_shapes_that_have_them() -> None:
    def as_list(case: Mapping[str, Any]) -> list[str]:
        return [_right(case)]

    with pytest.raises(
        ValueError, match=r"the answers are lists, which hold no fields: return a dict, a pydantic model or a dataclass"
    ):
        await Eval(CASES, as_list, ["label"], k=1).run(store=":memory:")


@dataclass(frozen=True)
class Labelled:
    label: str


@dataclass(frozen=True)
class Categorised:
    category: str


async def test_a_dataclass_answer_is_read_by_its_field_and_one_without_it_is_refused() -> None:
    """``out.label`` is read off any object, as it is off a pydantic model: a dataclass answer once scored 0%."""

    def labelled(case: Mapping[str, Any]) -> Labelled:
        return Labelled(_right(case))

    def categorised(case: Mapping[str, Any]) -> Categorised:
        return Categorised(_right(case))

    result = await Eval(CASES, labelled, ["label"], k=1).run(store=":memory:")
    assert all(cell.expected == cell.predicted for cell in result.summary.confusion)
    with pytest.raises(ValueError, match=r"the answers are objects without it, such as Categorised\(category="):
        await Eval(CASES, categorised, ["label"], k=1).run(store=":memory:")


async def test_what_is_the_product_s_failure_is_still_scored() -> None:
    """An arm lacking a field the other arm's answers carry is the product failing; a field every case expects absent
    is answered right by its absence; a field present but blank is an answer with no label. None is refused."""

    def carries(case: Mapping[str, Any]) -> dict[str, str]:
        return {"label": _right(case)}

    def drops(case: Mapping[str, Any]) -> dict[str, str]:
        return {"category": _right(case)}

    compared = await Eval(CASES, {"carries": carries, "drops": drops}, ["label"], k=1, name="drops").run(
        store=":memory:"
    )
    assert compared.arms["drops"].confusion and compared.arms["carries"].confusion

    def none_expected(case: Mapping[str, Any]) -> dict[str, str]:
        return {"label": _right(case)}

    absent = [{**case, "refund": None} for case in CASES]
    scored = await Eval(absent, none_expected, ["label", "refund"], k=1).run(store=":memory:")
    assert scored.summary.n_results == 3

    def blank(case: Mapping[str, Any]) -> dict[str, str]:
        return {"label": " "}

    await Eval(CASES, blank, ["label"], k=1).run(store=":memory:")


async def test_a_world_runner_s_answer_is_read_inside_its_outcome() -> None:
    """A world runner's stored answer wraps the agent's (with the calls it made); the field is read inside it."""

    def look(query: str) -> str:
        """Look something up."""
        return query

    async def agent(case: Mapping[str, Any], tools: Any) -> dict[str, str]:
        return {"label": _right(case)}

    async def misnamed(case: Mapping[str, Any], tools: Any) -> dict[str, str]:
        return {"category": _right(case)}

    world = World(tools=[look])
    result = await Eval(CASES, world(agent), ["label"], k=1).run(store=":memory:")
    assert all(cell.expected == cell.predicted for cell in result.summary.confusion)
    with pytest.raises(ValueError, match="the answers carry category"):
        await Eval(CASES, world(misnamed), ["label"], k=1).run(store=":memory:")


def test_a_case_a_second_field_grader_cannot_read_is_refused_before_anything_runs() -> None:
    ran: list[Mapping[str, Any]] = []

    def classify(case: Mapping[str, Any]) -> dict[str, str]:
        ran.append(case)
        return {"label": _right(case)}

    with pytest.raises(ValueError) as refused:
        Eval(CASES, classify, ["label", "urgent"], k=1)
    assert str(refused.value) == (
        "case 1: the field grader 'urgent' reads the case's 'urgent', which it lacks; it carries id, label, text: grade "
        "one of those, or give every case its 'urgent' (null where the answer should carry none)"
    )
    assert ran == []


# --- row 7: a runner, or a grader, that cannot take what it is handed -----------------------------------------------


def test_a_runner_that_cannot_take_a_case_exits_2_at_the_door_naming_its_signature(
    in_tmp: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    body = 'def classify():\n    return "greet"\n\nrow7 = Eval("cases.jsonl", classify, ["label"], k=1)\n'
    assert main([_eval_file(in_tmp, body)]) == EXIT_REFUSED
    assert capsys.readouterr().err == (
        "python -m threetears.evals probe.py: the runner classify() cannot be called as classify(case), which is how "
        "each case is handed to it; declare it to take exactly that: def classify(case): ...\n"
    )


def test_an_arm_and_a_world_agent_are_refused_in_their_own_names() -> None:
    def two(case: Mapping[str, Any], extra: str) -> str:
        return extra

    async def alone(case: Mapping[str, Any]) -> str:
        return "greet"

    def look(query: str) -> str:
        """Look something up."""
        return query

    with pytest.raises(ValueError, match=r"^the arm 'v2' two\(case, extra\) cannot be called as two\(case\)"):
        Eval(CASES, {"v1": _right, "v2": two}, ["label"], name="pair")
    with pytest.raises(ValueError, match=r"declare it to take exactly that: async def alone\(case, tools\): \.\.\.$"):
        Eval(CASES, World(tools=[look])(alone), ["label"])


def test_runners_that_can_take_a_case_are_accepted() -> None:
    def defaulted(case: Mapping[str, Any], effort: str = "low") -> str:
        return effort

    def anything(*args: Any, **kwargs: Any) -> str:
        return "greet"

    class Classifier:
        def __call__(self, case: Mapping[str, Any]) -> str:
            return "greet"

        def classify(self, case: Mapping[str, Any]) -> str:
            return "greet"

    def two(case: Mapping[str, Any], extra: str) -> str:
        return extra

    for runner in (defaulted, anything, Classifier(), Classifier().classify, functools.partial(two, extra="x")):
        Eval(CASES, runner, ["label"], name="accepted")


def test_a_grader_that_cannot_take_case_and_answer_is_refused() -> None:
    def right(out: Any) -> bool:
        return bool(out)

    with pytest.raises(ValueError) as refused:
        Eval(CASES, _right, [right])
    assert str(refused.value) == (
        "the grader right(out) cannot be called as right(case, out), which is how each answer is handed "
        "to it; declare it to take exactly that: def right(case, out): ..."
    )
