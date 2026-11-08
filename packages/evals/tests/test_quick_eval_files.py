"""``python -m threetears.evals FILE``: every eval a Python file defines is run, printed, reported, and exits with a code.

The exit codes are the host command line's: 0 done, 1 a run did not complete, 2 refused, 3 an error nothing
anticipated, 4 a comparison's gate failed. Anything that is not an eval file is the host command line, unchanged.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from threetears.evals.quick import EXIT_FAILED, EXIT_GATE_FAILED, EXIT_OK, EXIT_REFUSED, EXIT_RUN_DID_NOT_COMPLETE
from threetears.evals import Log
from threetears.evals.quick.files import load_evals, main, report_dir, run_file

#: Two evals in one file, run in the order the file defines them.
TWO_EVALS = '''\
from threetears.evals import Eval

CASES = [{"id": str(n), "n": n, "parity": "even" if n % 2 == 0 else "odd"} for n in range(6)]

async def parity(case):
    """Say whether the number is even."""
    return "even" if case["n"] % 2 == 0 else "odd"

def small(case, out) -> bool:
    """The number is small."""
    return case["n"] < 3

first = Eval(CASES, parity, ["parity"], k=1, name="first")
second = Eval(CASES, parity, ["parity", small], k=1, name="second eval")
'''

#: A comparison whose challenger gets every case wrong: the gate fails on the regression.
REGRESSION = '''\
from threetears.evals import Eval

CASES = [{"id": str(n), "label": "yes"} for n in range(20)]

async def right(case):
    """Answer yes."""
    return "yes"

async def wrong(case):
    """Answer yes."""
    return "no"

regression = Eval(CASES, {"right": right, "wrong": wrong}, ["label"], k=1, name="regression")
'''

#: One arm whose every answer costs more than the eval may spend: the run stops short.
OVER_BUDGET = '''\
from threetears.evals import Answer, Eval

async def dear(case):
    """Answer at a price."""
    return Answer("yes", model="m", cost_usd=1.0)

over = Eval([{"label": "yes"}] * 4, dear, ["label"], k=1, max_cost_usd=0.5, measure_latency=True, name="over")
'''


@pytest.fixture(autouse=True)
def _working_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    # Loading a Python file puts its directory on sys.path for the process; the test's process gets its own back.
    monkeypatch.setattr(sys, "path", [*sys.path])


def _write(name: str, text: str) -> str:
    Path(name).write_text(text, encoding="utf-8")
    return name


def test_every_eval_a_file_defines_runs_in_order_into_the_default_store_and_writes_its_report(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main([_write("evals.py", TWO_EVALS)]) == EXIT_OK

    printed = capsys.readouterr().out
    assert printed.index("first") < printed.index("second eval")
    assert "stored in .evals/evals.sqlite, scope 'first'" in printed
    assert "stored in .evals/evals.sqlite, scope 'second eval'" in printed
    assert report_dir("second eval") == Path(".evals/reports/second-eval")
    assert (report_dir("first") / "report.html").is_file() and (report_dir("second eval") / "report.md").is_file()


def test_verbose_prints_each_result_whole(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([_write("evals.py", TWO_EVALS), "--verbose"]) == EXIT_OK

    assert "## Cases" in capsys.readouterr().out


def test_a_comparison_whose_gate_fails_exits_4(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([_write("regression.py", REGRESSION)]) == EXIT_GATE_FAILED

    # The exit line opens with the result as the page leads with it, which is what the gate failed on. An interval's
    # two ends read at one precision, so the ceiling of the change reads "100.0" beside "63.3".
    exit_line = [line for line in capsys.readouterr().out.splitlines() if line.startswith("exit 4: ")]
    assert exit_line == [
        "exit 4: Accuracy: wrong worse than right. Of 20 trials each, Accuracy was right on 20 for right and 0 for "
        "wrong. Over 20 paired cases, the interval puts wrong between 63.3 and 100.0 points worse. The gate failed on it."
    ]


def test_a_run_stopped_short_exits_1(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([_write("over.py", OVER_BUDGET)]) == EXIT_RUN_DID_NOT_COMPLETE

    assert "stopped by its cost cap: $1.00 of $0.50 at the stop" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("text", "args", "refusal"),
    [
        ("x = 1\n", (), "defines no Eval at module level"),
        (TWO_EVALS, ("--fast",), "takes only --verbose"),
        # A module-level Eval that refuses is a refusal of the file, by its own words.
        ("from threetears.evals import Eval\nbad = Eval([], print, ['x'])\n", (), "at least one case"),
    ],
)
def test_what_is_refused_exits_2_and_says_why(
    text: str, args: tuple[str, ...], refusal: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([_write("bad.py", text), *args]) == EXIT_REFUSED
    assert refusal in capsys.readouterr().err


def test_a_file_that_is_not_there_is_refused(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["nowhere.py"]) == EXIT_REFUSED
    assert "nowhere.py is not there" in capsys.readouterr().err


def test_an_error_nothing_anticipated_exits_3_with_its_traceback(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([_write("boom.py", "raise RuntimeError('boom')\n")]) == EXIT_FAILED

    error = capsys.readouterr().err
    assert "failed on an unanticipated error" in error and "RuntimeError: boom" in error


#: A file naming its cases and its log by paths relative to itself, and one absolute path, which is taken as given.
BESIDE = """\
from threetears.evals import Eval, Log

log = Log("outputs.jsonl")
beside = Eval("cases.jsonl", log, ["label"], k=1, name="beside")
"""


def test_a_python_file_s_relative_paths_are_read_beside_it_and_an_absolute_one_as_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    rung = tmp_path / "rung"
    rung.mkdir()
    (rung / "cases.jsonl").write_text('{"id": "a", "label": "yes"}\n', encoding="utf-8")
    (rung / "outputs.jsonl").write_text('{"id": "a", "output": "yes"}\n', encoding="utf-8")
    (rung / "evals.py").write_text(BESIDE, encoding="utf-8")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    (loaded,) = load_evals(Path("../rung/evals.py"))
    assert len(loaded.cases) == 1 and loaded.arms[None].path == Path("../rung/outputs.jsonl")
    assert main(["../rung/evals.py"]) == EXIT_OK
    assert "stored in .evals/evals.sqlite, scope 'beside'" in capsys.readouterr().out
    assert (elsewhere / ".evals" / "reports" / "beside" / "report.html").is_file(), "the report stays where it is run"

    (rung / "evals.py").write_text(BESIDE.replace('"cases.jsonl"', repr(str(rung / "cases.jsonl"))), encoding="utf-8")
    (absolute,) = load_evals(rung / "evals.py")
    assert len(absolute.cases) == 1
    # Outside a load, a relative path is the working directory's, as anywhere in Python.
    assert Log("outputs.jsonl").path == Path("outputs.jsonl")


#: The modules beside the rung files below, each imported once per process by its own name.
SIBLINGS = ("ladder_lazy_helper", "ladder_eager_helper", "ladder_other_helper")

#: A helper module beside a rung file: the app code three rung files would otherwise each paste.
HELPER = """\
def answer(case):
    return "yes" if case["n"] % 2 == 0 else "no"
"""

#: A rung file importing its sibling at module level.
EAGER = """\
from threetears.evals import Eval
from {helper} import answer

CASES = [{{"id": str(n), "n": n, "label": "yes" if n % 2 == 0 else "no"}} for n in range(4)]

async def run(case):
    return answer(case)

{name} = Eval(CASES, run, ["label"], k=1, name="{name}")
"""

#: A rung file whose runner imports its sibling only when it is called, after the file has loaded.
LAZY = """\
from threetears.evals import Eval

CASES = [{{"id": str(n), "n": n, "label": "yes" if n % 2 == 0 else "no"}} for n in range(4)]

async def run(case):
    from {helper} import answer

    return answer(case)

{name} = Eval(CASES, run, ["label"], k=1, name="{name}")
"""


@pytest.fixture
def _siblings_forgotten() -> Iterator[None]:
    yield
    for name in SIBLINGS:
        sys.modules.pop(name, None)


@pytest.mark.usefixtures("_siblings_forgotten")
def test_a_python_file_imports_the_modules_beside_it_while_it_loads_and_while_it_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    rung, other, elsewhere = tmp_path / "rung", tmp_path / "other", tmp_path / "elsewhere"
    for directory in (rung, other, elsewhere):
        directory.mkdir()
    (rung / "ladder_lazy_helper.py").write_text(HELPER, encoding="utf-8")
    (rung / "ladder_eager_helper.py").write_text(HELPER, encoding="utf-8")
    (rung / "lazy.py").write_text(LAZY.format(helper="ladder_lazy_helper", name="lazy"), encoding="utf-8")
    (rung / "eager.py").write_text(EAGER.format(helper="ladder_eager_helper", name="eager"), encoding="utf-8")
    (other / "ladder_other_helper.py").write_text(HELPER, encoding="utf-8")
    (other / "other.py").write_text(EAGER.format(helper="ladder_other_helper", name="other"), encoding="utf-8")
    monkeypatch.chdir(elsewhere)  # neither directory is the working directory
    assert not any(name in sys.modules for name in SIBLINGS)

    # The lazy file loads with its sibling never imported; the runner imports it when the run calls it.
    assert run_file(Path("../rung/lazy.py")) == EXIT_OK
    assert run_file(Path("../rung/eager.py")) == EXIT_OK
    assert run_file(other / "other.py") == EXIT_OK

    assert all(name in sys.modules for name in SIBLINGS)
    printed = capsys.readouterr().out
    for name in ("lazy", "eager", "other"):  # every case answered by the sibling's code
        assert f"{name}: run passed 4 of 4 cases." in printed
    # Each directory first, in the order loaded, and once: the second file beside the first adds nothing.
    assert sys.path[:2] == [str(other), str(rung)]
    assert sys.path.count(str(rung)) == 1 and sys.path.count(str(other)) == 1


def test_anything_but_an_eval_file_is_the_host_command_line() -> None:
    with pytest.raises(SystemExit) as exited:
        main(["ls"])
    assert exited.value.code == EXIT_REFUSED, "argparse refuses ls with no --host, as the host command line does"
