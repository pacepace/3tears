"""Three lines an agent misread in the on-ramp runs (#1209 items 1, 2 and 5), each held to one reading.

- **One name per case, on every surface of one result.** A case is called by its own ``id`` where it has one, else by
  its place in the list given, from 1: the line of the cases file it is on. The first line, the top misses, the
  ``error:`` lines, ``results()`` and the report page's tables all call it that; the first line once numbered the
  failed cases from 1 and the misses under it from 0, and the page numbered them by the order of their labels.
- **No rate over nothing graded.** An arm whose runner raised on every case had no answer graded, so every measure of
  it reads ``none (no answer was graded)``, a classifier's accuracy as much as a function grader's, and no label's
  precision or recall is printed: never ``Accuracy: 0%, 0 of 3 trials`` beside another measure left unshown. The
  stored results are as before (a raise is landed as the classifier's miss, which a partly answering arm is counted
  against); only the reading changes, so a result stored before reads the same way.
- **Exit 1 when nothing was measured.** An eval in which no arm had an answer graded exits 1 from the file door, its
  first line the reason; one with some answers graded, and a comparison with one arm that answered nothing, keep
  their codes, and a refused build still exits 2.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest

from threetears.evals import Eval
from threetears.evals.analysis import campaign_report_markdown
from threetears.evals.quick import EXIT_OK, EXIT_REFUSED, EXIT_RUN_DID_NOT_COMPLETE
from threetears.evals.quick.files import main
from packages.evals.tests.clock_support import frozen_clock


@pytest.fixture(autouse=True)
def _frozen_clock() -> Iterator[None]:
    """Nothing here asserts on the time."""
    with frozen_clock():
        yield


#: Twelve tickets, the eleventh named by its author; ``hasty`` files every one wanting ``bug`` under ``billing``, so
#: it misses the cases at places 2, 4, 6, 8 and 10 and the named one: past place 9, so a place sorts as a number.
CASES = [
    *({"text": f"ticket {place}", "queue": "bug" if place % 2 == 0 else "billing"} for place in range(1, 11)),
    {"id": "refund", "text": "refund please", "queue": "bug"},
    {"text": "ticket 12", "queue": "billing"},
]


async def hasty(case: Mapping[str, Any]) -> str:
    """File every ticket under billing."""
    return "billing"


async def test_one_result_calls_each_case_one_name_on_every_surface() -> None:
    result = await Eval(CASES, hasty, ["queue"], k=1, name="names").run(store=":memory:")
    lines = result.render().splitlines()

    missed = ["case 2", "case 4", "case 6", "case 8", "case 10", "refund"]
    assert (
        lines[0]
        == f"names: hasty passed 6 of 12 cases. Cases that failed a trial: {', '.join(missed[:5])}, and 1 more."
    )
    misses = [line.split(" (repeat")[0].strip() for line in lines if re.match(r"  case .* \(repeat 1\): ", line)]
    assert misses == ["case 2", "case 4", "case 6", "case 8", "case 10"], (
        "the first five misses, as the page lists them"
    )
    assert [miss.case for miss in result.summary.misses()] == ["2", "4", "6", "8", "10", "refund"]
    # The listing names the first case as the first line of a cases file is numbered: 1.
    assert [case.case for case in result.summary.results()][:2] == ["1", "2"]
    page = campaign_report_markdown(result.page)
    cases_table = page.split("## Cases", 1)[1].split("\n## ", 1)[0]
    rows = [row.split("|")[1].strip() for row in cases_table.splitlines() if row.startswith("| ") and "✗" in row]
    assert rows == missed, "the page's Cases table calls each failed case as the console does"


async def test_an_error_line_calls_the_case_by_its_place_from_1() -> None:
    async def refuses_the_second(case: Mapping[str, Any]) -> str:
        if case["text"] == "ticket 2":
            raise RuntimeError("no")
        return "billing"

    result = await Eval(CASES, refuses_the_second, ["queue"], k=1).run(store=":memory:")

    assert result.summary.errors == ["case 2: the candidate raised RuntimeError: no"]
    assert "  error: case 2: the candidate raised RuntimeError: no" in result.render().splitlines()


async def test_a_case_refused_before_the_run_is_called_by_its_place_from_1() -> None:
    with pytest.raises(ValueError, match=r"^case 2's id is ''"):
        await Eval([{"text": "a", "queue": "x"}, {"id": "", "text": "b", "queue": "x"}], hasty, ["queue"]).run(
            store=":memory:"
        )


THREE = [{"id": name, "text": name, "want": "p"} for name in ("a", "b", "c")]


def raises_on(*texts: str) -> Callable[[Mapping[str, Any]], Any]:
    """A runner that raises on the cases named, as an app's function does on a key it lacks, and answers ``p``."""

    async def gloss(case: Mapping[str, Any]) -> str:
        if case["text"] in texts:
            raise KeyError("word")
        return "p"

    return gloss


def right(case: Mapping[str, Any], out: Any) -> bool:
    return bool(out == case["want"])


@pytest.mark.parametrize(
    ("graders", "printed"),
    [
        (["want"], ["  Accuracy: none (no answer was graded)"]),
        ([right], ["  Right score: none (no answer was graded)"]),
        # Accuracy, which the raised trials carry as misses, first; the grader's measure, which none carries, after.
        (["want", right], ["  Accuracy: none (no answer was graded)", "  Right score: none (no answer was graded)"]),
    ],
    ids=["a classifier", "a function grader", "both"],
)
async def test_every_measure_of_an_arm_that_graded_no_answer_reads_none_never_0_percent(
    graders: list[Any], printed: list[str]
) -> None:
    result = await Eval(THREE, raises_on("a", "b", "c"), graders, k=1).run(store=":memory:")
    lines = result.render().splitlines()

    assert lines[0].startswith("gloss: gloss failed on all 3 cases, so no answer was graded: ")
    assert [line for line in lines if "no answer was graded)" in line] == printed
    assert not [line for line in lines if "%" in line], "no rate, and no label's precision or recall, over nothing"
    assert "carried by no result" not in result.render()
    # The full print's arm summary, before the page it ends with, says the same.
    full = result.render(full=True).split("\n\n# ", 1)[0].splitlines()
    assert [line for line in full if "no answer was graded)" in line] == printed
    assert "per label:" not in [line.strip() for line in full]
    assert not [line for line in full if re.search(r"\d%", line)]


async def test_what_is_stored_for_an_arm_that_graded_no_answer_is_unchanged() -> None:
    result = await Eval(THREE, raises_on("a", "b", "c"), ["want"], k=1).run(store=":memory:")

    # A raise is the classifier's miss, as stored and summarised (an arm that refused some cases is counted against);
    # only the console reads "none" where the arm answered nothing, so a result stored earlier reads the same way.
    (match,) = [measure for measure in result.summary.measures if measure.name == "match"]
    assert (match.n, match.mean) == (3, 0.0)
    assert all(case.scores["match"] is False for case in result.summary.results())


async def test_an_arm_that_answered_some_cases_keeps_its_rate() -> None:
    result = await Eval(THREE, raises_on("b"), ["want"], k=1).run(store=":memory:")
    lines = result.render().splitlines()

    assert "  Accuracy: 67%, 2 of 3 trials" in lines
    assert "  per label:" in lines
    assert result.exit_code() == EXIT_OK


@pytest.fixture
def _in_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", [*sys.path])


_FILE = '''\
from threetears.evals import Eval

CASES = [{{"id": name, "text": name, "want": "p"}} for name in ("a", "b", "c")]

async def gloss(case):
    """Raise on the cases named."""
    if case["text"] in {raising!r}:
        raise KeyError("word")
    return "p"

async def steady(case):
    """Answer p."""
    return "p"

def right(case, out) -> bool:
    return out == case["want"]

evaluated = Eval(CASES, {arms}, [right], k=1, name="evaluated")
'''


@pytest.mark.usefixtures("_in_tmp")
@pytest.mark.parametrize(
    ("raising", "arms", "code"),
    [
        (("a", "b", "c"), "gloss", EXIT_RUN_DID_NOT_COMPLETE),
        (("a", "b", "c"), '{"one": gloss, "two": gloss}', EXIT_RUN_DID_NOT_COMPLETE),
        (("b",), "gloss", EXIT_OK),
        (("a", "b", "c"), '{"steady": steady, "gloss": gloss}', EXIT_OK),
    ],
    ids=["one arm raised on every case", "both arms did", "some answers graded", "one arm of two answered nothing"],
)
def test_the_file_door_exits_1_only_where_no_arm_had_an_answer_graded(
    raising: tuple[str, ...], arms: str, code: int, capsys: pytest.CaptureFixture[str]
) -> None:
    Path("evaluated.py").write_text(_FILE.format(raising=raising, arms=arms), encoding="utf-8")

    assert main(["evaluated.py"]) == code
    if code == EXIT_RUN_DID_NOT_COMPLETE:
        # The first line is the reason.
        assert "so no answer was graded: the runner raised KeyError: 'word'" in capsys.readouterr().out.splitlines()[0]


@pytest.mark.usefixtures("_in_tmp")
def test_a_refused_build_still_exits_2(capsys: pytest.CaptureFixture[str]) -> None:
    Path("refused.py").write_text("from threetears.evals import Eval\nbad = Eval([], print, ['x'])\n", encoding="utf-8")

    assert main(["refused.py"]) == EXIT_REFUSED
    assert "at least one case" in capsys.readouterr().err
