"""``store=``: the quick path keeps its runs in a store of the caller's choosing, with nothing else to build.

The step after a first eval is "the same thing, but keep the results". Before ``store=``, keeping them meant
replacing the host ``run_eval`` builds with one built by hand — a profile, a measure per scorer, a kind contract
— and losing ``margins=`` and ``ranges=`` with it, since ``compare`` refuses those beside a host of the caller's
own. ``store=`` keeps the built host and changes only where it stores.

Mutations that turn this file red: ``callable_host`` ignoring ``store=``; ``run_eval`` or ``compare`` not passing it
on; accepting ``store=`` beside ``host=``; the SQLite store keeping anything in the connection rather than the
file.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import textwrap
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from threetears.evals.analysis import list_campaigns
from threetears.evals.contracts.errors import ValidationFailedError
from threetears.evals.ops import scope_history, summarize_run
from threetears.evals.quick import Judge, callable_host, compare, run_eval
from threetears.evals.run import list_runs
from threetears.evals.storage import SqliteDocumentStore

#: The tutorial, whose last step keeps its runs in a file.
TUTORIAL = Path(__file__).resolve().parents[1] / "docs" / "tutorial.md"

#: Enough cases for two arms that agree on every one to be shown within 0.1 of each other.
CASES = [{"n": index} for index in range(48)]


def correct(case: Mapping[str, Any], answer: str) -> bool:
    """Whether the answer is right."""
    return answer == "right"


async def always_right(case: Mapping[str, Any]) -> str:
    return "right"


async def misses_one(case: Mapping[str, Any]) -> str:
    return "wrong" if case["n"] == 0 else "right"


async def test_a_run_eval_run_is_read_back_by_a_new_process(tmp_path: Path) -> None:
    path = tmp_path / "evals.sqlite"
    with SqliteDocumentStore(path) as store:
        first = await run_eval(CASES[:6], always_right, [correct], scope_id="kept", k=2, store=store)
        second = await run_eval(CASES[:6], misses_one, [correct], scope_id="kept", k=2, store=store)
    # A process that never saw the runs, holding nothing but the file's path and the scorer.
    script = textwrap.dedent(
        f"""
        from threetears.evals.ops import summarize_run
        from threetears.evals.quick import callable_host
        from threetears.evals.run import list_runs
        from threetears.evals.storage import SqliteDocumentStore

        def correct(case, answer):
            return answer == "right"

        host = callable_host([correct], store=SqliteDocumentStore({str(path)!r}))
        for run in sorted(list_runs(host, "kept"), key=lambda run: run.created_at):
            summary = summarize_run(host, run.id, "kept")
            (measure,) = summary.measures
            print(run.id, run.candidate_model, summary.n_results, measure.name, measure.mean)
        """
    )
    read = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True, timeout=120)
    assert read.stdout.splitlines() == [
        f"{first.run_id} always_right 12 correct 1.0",
        f"{second.run_id} misses_one 12 correct {10 / 12}",
    ]


async def misses_half(case: Mapping[str, Any]) -> str:
    return "wrong" if case["n"] % 2 else "right"


async def test_a_kept_scorer_has_a_history_read_back_from_the_file(tmp_path: Path) -> None:
    """``scope_history(metric=<scorer>)`` series a quick-path scorer, read through a host over the reopened file."""
    path = tmp_path / "evals.sqlite"
    with SqliteDocumentStore(path) as store:
        first = await run_eval(CASES[:12], always_right, [correct], scope_id="kept", k=2, store=store, model="router")
        second = await run_eval(CASES[:12], misses_half, [correct], scope_id="kept", k=2, store=store, model="router")
    with SqliteDocumentStore(path) as reopened:
        host = callable_host([correct], store=reopened)
        history = scope_history(host, "kept", metric="correct")
        with pytest.raises(ValidationFailedError, match="a numeric measure the host declares"):
            scope_history(host, "kept", metric="corect")
    (series,) = history.series
    assert series.model == "router" and history.higher_is_better is True
    assert [(point.run_id, point.value, point.n, point.n_cases) for point in series.points] == [
        (first.run_id, 1.0, 24, 12),
        (second.run_id, 0.5, 24, 12),
    ]
    assert series.points[0].regression is None
    flag = series.points[1].regression
    assert flag is not None and flag.label == "regressed" and flag.n_pairs == 12


async def test_compare_keeps_its_margins_with_a_store(tmp_path: Path) -> None:
    path = tmp_path / "evals.sqlite"
    with SqliteDocumentStore(path) as store:
        comparison = await compare(
            CASES,
            # Alike on every case: 48 agreeing pairs show a 0.1 margin held whatever order the bounded test bets in.
            # One miss among them is shown equivalent in 18 of the 48 places it can fall, so it would test luck.
            {"current": always_right, "cheaper": always_right},
            [correct],
            control="current",
            scope_id="kept",
            k=2,
            store=store,
            margins={"correct": 0.1},
        )
    (row,) = comparison.contrasts("correct")
    assert row["verdict"].startswith("equivalent to the control") and "(margin ±0.1)" in row["verdict"]
    with SqliteDocumentStore(path) as reopened:
        host = callable_host([correct], arms=True, margins={"correct": 0.1}, store=reopened)
        assert [campaign.id for campaign in list_campaigns(host.storage, "kept")] == [comparison.campaign_id]
        assert {run.id for run in list_runs(host, "kept")} == {
            comparison.arms["current"].run_id,
            comparison.arms["cheaper"].run_id,
        }


async def _stand_in_judge(*, system: str, user: str, response_format: dict[str, Any] | None = None) -> SimpleNamespace:
    """An offline judge that gives every answer a 5."""
    reply = {"reasoning": "stand-in", "criteria_scores": {"answer.acknowledges": 5}}
    return SimpleNamespace(
        content=json.dumps(reply),
        input_tokens=None,
        output_tokens=None,
        reasoning_tokens=None,
        cost_usd=0.0,
        price_source="stand-in",
        model="stand-in",
        served_model=None,
        stop_reason="end_turn",
        temperature=0.0,
    )


async def test_rerunning_unchanged_cases_never_makes_an_earlier_run_read_as_edited(tmp_path: Path) -> None:
    """A second call over the same cases leaves the stored template alone, so the first run keeps its intent.

    The template is addressed by its content, so the second call finds it stored. Restamping it on every call made
    every earlier judged run's summary read its template as edited since, and withhold the intent its judge read.
    """
    judge = Judge(
        client=SimpleNamespace(generate=_stand_in_judge),
        model="stand-in",
        rubric={"acknowledges": "The reply acknowledges the problem."},
        case_material=lambda case: str(case["n"]),
    )
    with SqliteDocumentStore(tmp_path / "evals.sqlite") as store:
        first = await run_eval(
            CASES[:3], always_right, judge=judge, scope_id="kept", k=1, store=store, intent="Say it."
        )
        await run_eval(CASES[:3], always_right, judge=judge, scope_id="kept", k=1, store=store, intent="Say it.")
        again = summarize_run(callable_host([], store=store), first.run_id, "kept")
    assert first.intent == again.intent == "Say it."


async def test_a_store_beside_a_host_is_refused() -> None:
    host = callable_host([correct])
    with SqliteDocumentStore(":memory:") as store:
        with pytest.raises(ValueError, match="brings its own storage"):
            await run_eval(CASES[:2], always_right, [correct], scope_id="kept", host=host, store=store)
        with pytest.raises(ValueError, match="brings its own storage"):
            await compare(
                CASES[:2],
                {"a": always_right, "b": misses_one},
                [correct],
                control="a",
                scope_id="kept",
                host=host,
                store=store,
            )


def test_the_tutorial_s_keep_your_runs_step_runs_as_written(tmp_path: Path) -> None:
    """The tutorial's file as a reader builds it, run twice: the second run lists both, as the step prints."""
    page = TUTORIAL.read_text(encoding="utf-8")
    # Every block the reader adds, in order; each step's ``main()`` replaces the last, so only the final one runs.
    blocks = re.findall(r"```python\n(.*?)```", page, re.S)
    script = "\n\n".join(block.replace("asyncio.run(main())", "") for block in blocks) + "\nasyncio.run(main())\n"
    (tmp_path / "triage.py").write_text(script, encoding="utf-8")
    step = page[page.index("## 8. Keep your runs") : page.index("## 9. Where next")]
    # The printed lines, each a run's timestamp and what it read; a timestamp is when the reader ran it.
    printed = re.findall(r"^\d{4}-\d\d-\d\dT\S+ (.*)$", step, re.M)

    for _ in range(2):
        ran = subprocess.run(
            [sys.executable, "triage.py"], cwd=tmp_path, capture_output=True, text=True, check=True, timeout=300
        )
    assert [line.split(" ", 1)[1] for line in ran.stdout.splitlines()] == printed
