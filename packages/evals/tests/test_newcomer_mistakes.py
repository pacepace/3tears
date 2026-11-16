"""A newcomer's mistakes are refused with the fix named, never turned into a score.

The errors a newcomer hits on the file door (design-on-ramp §3.4): asking the command line for help first, running a
model with no key, comparing two arms with no name, grading a field the answers do not carry, and a runner (or a
grader) that cannot take what it is handed. The last two once scored 0% and exited 0, a wiring bug read as the product
failing; each is now refused (exit 2 at the file door), and each refusal names the change to make, in the names the
author typed.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import pytest

from threetears.evals import Eval, Model, World
from threetears.evals.kernel import MATCH_MEASURE
from threetears.evals.ops import scope_frontier, scope_history
from threetears.evals.quick import EXIT_OK, EXIT_REFUSED, as_host
from threetears.evals.run import list_runs
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


def test_a_runner_that_returns_nothing_is_refused_before_anything_runs() -> None:
    """A runner whose code has no return of a value answers None on every case, whatever the product does: wiring."""

    def forgets_to_return(case: Mapping[str, Any]) -> None:
        case.get("text")

    with pytest.raises(ValueError) as refused:
        Eval(CASES, forgets_to_return, ["label"], k=1)
    said = str(refused.value)
    assert said.startswith("the runner forgets_to_return returns nothing (it has no return of a value)")
    assert "its grader could read 'label' off none of them: return the answer from it" in said
    with pytest.raises(ValueError, match=r"^the arm 'v2' forgets_to_return returns nothing"):
        Eval(CASES, {"v1": _right, "v2": forgets_to_return}, ["label"], name="pair")


def test_a_runner_that_returns_nothing_is_run_where_no_field_is_read_off_its_answer() -> None:
    """A function grader may grade what the runner did rather than what it returned, a field every case expects absent
    is answered by None, and a runner that raises stands in for a product that fails: none is refused."""

    def acts(case: Mapping[str, Any]) -> None:
        case.get("text")

    def did(case: Mapping[str, Any], out: Any) -> bool:
        return out is None

    def refuses(case: Mapping[str, Any]) -> str:
        raise RuntimeError("the provider refused the request")

    Eval(CASES, acts, [did], k=1)
    Eval(CASES, refuses, ["label"], k=1)
    Eval([{**case, "refund": None} for case in CASES], acts, ["refund"], k=1)


async def test_a_runner_whose_answers_are_all_none_is_scored_as_answering_nothing() -> None:
    """A runner that returns None where the model's output never parsed is a product that answered nothing, not a
    mis-wired eval: scored (0%), each case's answer stored as None, which is how the result says every answer was
    absent."""

    def unparsed(case: Mapping[str, Any]) -> dict[str, str] | None:
        text = str(case["text"])
        if text.startswith("{"):
            return {"label": text}
        return None

    result = await Eval(CASES, unparsed, ["label"], k=1).run(store=":memory:")
    results = result.results()
    assert len(results) == 3
    assert all(r.answer is None and r.outcome == "scored" and r.state == "failed" for r in results)
    assert _accuracy(result) == 0.0


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


class LabelledTuple(NamedTuple):
    label: str
    urgent: str


async def test_a_named_tuple_answer_is_graded_by_its_fields_though_stored_as_a_list() -> None:
    """A named tuple is stored as a JSON list, a shape with no fields: what its grading read decides, so a correct
    eval returning one is scored, never refused."""

    def as_tuple(case: Mapping[str, Any]) -> LabelledTuple:
        return LabelledTuple(_right(case), "no")

    cases = [{**case, "urgent": "no"} for case in CASES]
    result = await Eval(cases, as_tuple, ["label", "urgent"], k=1).run(store=":memory:")
    assert _accuracy(result) == 1.0
    assert all(r.scores["urgent"] == 1.0 for r in result.results())


async def test_what_is_the_product_s_failure_is_still_scored() -> None:
    """An arm lacking a field the other arm's answers carry is the product failing; a field every case expects absent
    is answered right by its absence; a field present but blank is an answer with no label, scored 0%. None is
    refused."""

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

    blanks = await Eval(CASES, blank, ["label"], k=1).run(store=":memory:")
    assert blanks.summary.n_scored == 3
    assert all(cell.predicted != cell.expected for cell in blanks.summary.confusion)
    assert _accuracy(blanks) == 0.0


def _accuracy(result: Any) -> float:
    """The share of trials whose label matched, read off each result's own ``match``."""
    matched = [bool(r.scores[MATCH_MEASURE]) for r in result.results()]
    return sum(matched) / len(matched)


async def test_one_arm_whose_answers_carry_the_field_and_are_all_wrong_scores_0_percent() -> None:
    """The product's failure is scored as before: a single arm, every answer carrying the field and every one wrong,
    reads 0%; so does a runner answering free text as a plain string, its own label."""

    def wrong(case: Mapping[str, Any]) -> dict[str, str]:
        return {"label": "nonsense"}

    def chatty(case: Mapping[str, Any]) -> str:
        return "I think this message is a greeting of some kind."

    for runner in (wrong, chatty):
        result = await Eval(CASES, runner, ["label"], k=1).run(store=":memory:")
        assert result.summary.n_scored == 3
        assert _accuracy(result) == 0.0


@pytest.mark.parametrize(
    "answer", ['{"label": "nonsense"}', '"I think this message is a greeting of some kind."'], ids=["field", "text"]
)
def test_one_arm_all_wrong_exits_0_at_the_door(answer: str, in_tmp: Path, capsys: pytest.CaptureFixture[str]) -> None:
    body = f'def classify(case):\n    return {answer}\n\nwrong = Eval("cases.jsonl", classify, ["label"], k=1)\n'
    assert main([_eval_file(in_tmp, body)]) == EXIT_OK
    captured = capsys.readouterr()
    assert captured.err == ""
    assert "Accuracy: 0%, 0 of 3 trials" in captured.out


# --- a refused run is kept out of every reading ----------------------------------------------------------------------


def test_a_refused_run_is_archived_so_no_listing_history_or_frontier_reads_it(tmp_path: Path) -> None:
    """The runs a refusal after the arms ran leaves behind are archived: kept and restorable, but ls, the history and
    the frontier read none of them, and the first fixed run is a baseline, not a step up from 0%."""
    store = tmp_path / "evals.sqlite"

    def classify(case: Mapping[str, Any]) -> dict[str, str]:
        return {"category": _right(case)}

    refused = Eval(CASES, classify, ["label"], k=1, name="triage")
    with pytest.raises(ValueError, match="no answer has a field 'label'"):
        asyncio.run(refused.run(store=store))
    host = as_host([refused], store=store)
    assert list_runs(host, "triage") == []
    (kept,) = list_runs(host, "triage", include_archived=True)
    assert kept.archived
    history = scope_history(host, "triage")
    assert history.series == [] and history.exclusions.results_from_archived_runs == 3
    frontier = scope_frontier(host, "triage")
    assert frontier.subjects == [] and frontier.exclusions.results_from_archived_runs == 3

    def classify_fixed(case: Mapping[str, Any]) -> dict[str, str]:
        return {"label": _right(case)}

    classify_fixed.__name__ = "classify"
    fixed = asyncio.run(Eval(CASES, classify_fixed, ["label"], k=1, name="triage").run(store=store))
    assert _accuracy(fixed) == 1.0
    (series,) = scope_history(host, "triage").series
    (point,) = series.points
    assert point.is_baseline and point.run_id == fixed.summary.run_id


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


def _with_client(function: Callable[..., Any]) -> Callable[..., Any]:
    """A decorator that hands the function a client: what is called takes the case alone."""

    @functools.wraps(function)
    def run(case: Mapping[str, Any]) -> Any:
        return function("client", case)

    return run


def _passing_through(function: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(function)
    def run(*args: Any, **kwargs: Any) -> Any:
        return function(*args, **kwargs)

    return run


def _answering(function: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(function)
    def run(case: Mapping[str, Any]) -> Any:
        function(case)
        return _right(case)

    return run


def _unwrapped(function: Callable[..., Any]) -> Callable[..., Any]:
    def run(*args: Any, **kwargs: Any) -> Any:
        return function(*args, **kwargs)

    return run


def _no_arguments(function: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(function)
    def run() -> Any:
        return function("case")

    return run


class _Classifier:
    def __call__(self, case: Mapping[str, Any]) -> str:
        return _right(case)

    def classify(self, case: Mapping[str, Any]) -> str:
        return _right(case)

    @staticmethod
    def static(case: Mapping[str, Any]) -> str:
        return _right(case)


class _Bare:
    def __call__(self) -> str:
        return "greet"


def _plain(case: Mapping[str, Any]) -> str:
    return _right(case)


async def _async(case: Mapping[str, Any]) -> str:
    return _right(case)


def _two(case: Mapping[str, Any], extra: str) -> str:
    return _right(case)


def _keyworded(case: Mapping[str, Any], **settings: Any) -> str:
    return _right(case)


def _star(*args: Any) -> str:
    return _right(args[0])


@_with_client
def _injected(client: str, case: Mapping[str, Any]) -> str:
    return _right(case)


@_passing_through
def _wrapped(case: Mapping[str, Any]) -> str:
    return _right(case)


@_unwrapped
def _decorated(case: Mapping[str, Any]) -> str:
    return _right(case)


@_answering
def _answered_by_its_decorator(case: Mapping[str, Any]) -> None:
    # No return of its own: inspect.getsource follows __wrapped__ to this source, so only the decorator returns.
    print(case)


@functools.cache
def _cached(case: str) -> str:
    return case


@_no_arguments
def _taken_away(case: Mapping[str, Any]) -> str:
    return "greet"


#: Every shape of callable an author can pass as a runner, and whether it can take a case.
ACCEPTED_RUNNERS: dict[str, Any] = {
    "plain function": _plain,
    "async function": _async,
    "bound method": _Classifier().classify,
    "static method": _Classifier.static,
    "callable object": _Classifier(),
    "functools.partial": functools.partial(_two, extra="x"),
    "lambda with defaults": lambda case, effort="low": _right(case),
    "(case, **kw)": _keyworded,
    "(*args)": _star,
    "decorated with wraps, injecting an argument": _injected,
    "decorated with wraps, passing through": _wrapped,
    "decorated without wraps": _decorated,
    "decorated with wraps, the decorator returns the answer the function does not": _answered_by_its_decorator,
}


@pytest.mark.parametrize("runner", ACCEPTED_RUNNERS.values(), ids=ACCEPTED_RUNNERS.keys())
async def test_every_shape_of_runner_that_can_take_a_case_is_accepted_and_runs(runner: Any) -> None:
    # Named as an arm: a callable object or a partial has no __name__ to label an unnamed arm by.
    result = await Eval(CASES, {"arm": runner}, ["label"], k=1, name="accepted").run(store=":memory:")
    assert _accuracy(result) == 1.0


def test_a_runner_whose_signature_cannot_be_read_is_not_refused() -> None:
    """A C wrapper (``functools.cache``) shows no signature of its own: the check cannot tell, so it does not refuse."""
    Eval(CASES, _cached, ["label"], name="cached")


@pytest.mark.parametrize(
    ("runner", "said"),
    [
        (_two, r"^the runner _two\(case, extra\) cannot be called as _two\(case\)"),
        (lambda: "greet", r"^the runner <lambda>\(\) cannot be called as <lambda>\(case\)"),
        (
            _Bare(),
            r"^the runner _Bare\(\) cannot be called as _Bare\(case\), .*declare its __call__ to take exactly that: "
            r"def __call__\(self, case\): \.\.\.$",
        ),
        (
            functools.partial(_two, {}, "x"),
            r"^the runner partial\(_two, \.\.\.\)\(\) cannot be called as partial\(_two, \.\.\.\)\(case\), .*"
            r"declare its function so that what the partial leaves free is exactly \(case\)$",
        ),
        (_taken_away, r"^the runner _taken_away\(\) cannot be called as _taken_away\(case\)"),
    ],
    ids=["two parameters", "lambda of none", "callable object of none", "partial leaving none", "wrapper of none"],
)
def test_every_shape_of_runner_that_cannot_take_a_case_is_refused_in_its_own_terms(runner: Any, said: str) -> None:
    with pytest.raises(ValueError, match=said):
        Eval(CASES, runner, ["label"], name="refused")


def test_a_runner_and_grader_annotated_with_names_imported_for_type_checking_alone_run(
    in_tmp: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Annotations evaluated lazily (PEP 649) name what a module imports under ``if TYPE_CHECKING:`` alone; reading the
    signature never evaluates them, so such an eval runs as any other."""
    body = (
        "from typing import TYPE_CHECKING\n\n"
        "if TYPE_CHECKING:\n    from tickets import Ticket, Verdict\n\n\n"
        "def classify(case: Ticket) -> Verdict:\n"
        '    return {"label": "leave" if case["text"] == "bye" else "greet"}\n\n\n'
        "def right(case: Ticket, out: Verdict) -> bool:\n"
        '    return out["label"] == case["label"]\n\n\n'
        'annotated = Eval("cases.jsonl", classify, ["label", right], k=1)\n'
    )
    assert main([_eval_file(in_tmp, body)]) == EXIT_OK
    assert capsys.readouterr().err == ""


def test_a_decorated_grader_is_read_by_what_is_called() -> None:
    def judge_with(function: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(function)
        def grade(case: Mapping[str, Any], out: Any) -> bool:
            return bool(function("client", case, out))

        return grade

    @judge_with
    def right(client: str, case: Mapping[str, Any], out: Any) -> bool:
        return bool(out == case["label"])

    Eval(CASES, _right, [right])


def test_a_grader_that_cannot_take_case_and_answer_is_refused() -> None:
    def right(out: Any) -> bool:
        return bool(out)

    with pytest.raises(ValueError) as refused:
        Eval(CASES, _right, [right])
    assert str(refused.value) == (
        "the grader right(out) cannot be called as right(case, out), which is how each answer is handed "
        "to it; declare it to take exactly that: def right(case, out): ..."
    )


# --- what cannot be read is run, and only that: any other failure of reading reaches the author --------------------


class _SignatureThatFails:
    """A callable whose signature raises something other than what marks a callable that has none."""

    __name__ = "right"

    @property
    def __signature__(self) -> inspect.Signature:
        raise RuntimeError("reading this signature failed")

    def __call__(self, case: Mapping[str, Any], out: Any) -> bool:
        return bool(out == case["label"])


def test_a_runner_whose_signature_fails_to_read_raises_that_failure() -> None:
    with pytest.raises(RuntimeError, match="reading this signature failed"):
        Eval(CASES, _SignatureThatFails(), ["label"], name="unreadable")


def _defined_where_no_file_holds_it() -> Callable[..., Any]:
    """A runner with no return of a value, defined by ``exec``: ``inspect.getsource`` finds no source for it."""
    namespace: dict[str, Any] = {}
    exec("def classify(case):\n    case.get('text')\n", namespace)
    runner: Callable[..., Any] = namespace["classify"]
    return runner


def test_a_runner_whose_source_cannot_be_read_is_not_refused() -> None:
    """Its source would show it returns nothing; with none to read the check cannot tell, so it does not refuse."""
    Eval(CASES, _defined_where_no_file_holds_it(), ["label"], name="sourceless")


def test_a_runner_whose_source_fails_to_read_otherwise_raises_that_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def failing(_: object) -> str:
        raise RuntimeError("reading this source failed")

    monkeypatch.setattr(inspect, "getsource", failing)
    with pytest.raises(RuntimeError, match="reading this source failed"):
        Eval(CASES, _right, ["label"], name="unreadable")


def test_a_grader_whose_signature_cannot_be_read_is_not_refused() -> None:
    """A C wrapper (``functools.cache``) shows no signature of its own: the check cannot tell, so it does not refuse."""

    @functools.cache
    def right(case: Mapping[str, Any], out: Any) -> bool:
        return bool(out == case["label"])

    Eval(CASES, _right, [right])


def test_a_grader_whose_signature_fails_to_read_raises_that_failure() -> None:
    with pytest.raises(RuntimeError, match="reading this signature failed"):
        Eval(CASES, _right, [_SignatureThatFails()])
