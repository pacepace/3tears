"""The guides' code blocks run as their pages say: in order, in one async context, each page in a namespace of its own.

A guide past the tutorial is a page of fenced blocks that use top-level ``await`` and lean on the names the blocks
above them defined, and each page says so at its top. A reader runs them in ``python -m asyncio`` or inside
``async def main()``, which puts every block on a running event loop. That is where a synchronous store call is
refused (``StoreCalledOnEventLoop``), so a block that makes one bare raises for the reader exactly as it raises
here. The tutorial, a file a reader builds step by step, is held by ``test_quick_store.py`` instead.

Each block is compiled as a reader's REPL compiles it (``PyCF_ALLOW_TOP_LEVEL_AWAIT``) and awaited on the test's own
loop. A block the page presents as a fragment, or one that needs more than the page builds, is named below with
the reason, and each name must still match exactly one block, so a skip cannot outlive the block it skipped.

Mutations that turn this file red: any doc block calling ``create_campaign``, ``frontier``, ``create_template``,
``list_results``, ``get_result_trace``, ``get_run``, ``rate_result`` or a ``host.storage`` read without
``run_blocking``; a block that stops printing what its page says it prints.
"""

from __future__ import annotations

import ast
import json
import re
import types
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

#: The guides directory.
DOCS = Path(__file__).resolve().parents[1] / "docs"

#: A fenced Python block on a page.
_BLOCK = re.compile(r"```python\n(.*?)```", re.S)


def page_blocks(page: str) -> list[str]:
    """Every fenced Python block on ``docs/<page>``, in page order."""
    return _BLOCK.findall((DOCS / page).read_text(encoding="utf-8"))


def without(blocks: list[str], skipped: Mapping[str, str]) -> list[str]:
    """``blocks`` less each one holding a key of ``skipped``, every key matching exactly one block."""
    for marker, reason in skipped.items():
        holding = [block for block in blocks if marker in block]
        assert len(holding) == 1, f"the skip for {marker!r} ({reason}) matches {len(holding)} blocks, not one"
    return [block for block in blocks if not any(marker in block for marker in skipped)]


async def run_in_order(blocks: list[str], namespace: dict[str, Any], *, page: str) -> None:
    """Run ``blocks`` one after another in ``namespace``, awaiting each that awaits, as ``python -m asyncio`` would."""
    for index, block in enumerate(blocks):
        code = compile(block, f"{page}[{index}]", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
        ran = eval(code, namespace)
        if isinstance(ran, types.CoroutineType):
            await ran


def printed_lines(capsys: pytest.CaptureFixture[str]) -> list[str]:
    return capsys.readouterr().out.splitlines()


async def test_evaluating_agents_runs_top_to_bottom(capsys: pytest.CaptureFixture[str]) -> None:
    """Every block, the template written in step 3 and the results read back in steps 6 and 7 included."""
    await run_in_order(page_blocks("evaluating-agents.md"), {}, page="evaluating-agents.md")
    page = (DOCS / "evaluating-agents.md").read_text(encoding="utf-8")
    placed = "{'light': 'representable', 'daylight': 'witnessed'}"
    assert placed in page and printed_lines(capsys)[-1] == placed


async def test_choosing_a_design_runs_after_the_world_it_names(capsys: pytest.CaptureFixture[str]) -> None:
    """The page's three blocks after the one block of another page it names (``ROOM``), printing what it shows."""
    namespace: dict[str, Any] = {}
    await run_in_order(page_blocks("evaluating-agents.md")[:1], namespace, page="evaluating-agents.md")
    assert "ROOM" in namespace, "the first block of evaluating-agents.md no longer declares ROOM"
    await run_in_order(page_blocks("choosing-a-design.md"), namespace, page="choosing-a-design.md")
    page = (DOCS / "choosing-a-design.md").read_text(encoding="utf-8")
    # What the page's comments say the blocks print: the design's shape, each frontier point, the cost decision.
    shown = [
        "one_factor_at_a_time [{'candidate': 'v2'}]",
        "0.002 1.0 0.82 cleared",
        "0.0005 0.9 0.67 undecided",
        "only_cleared",
    ]
    assert all(line in page for line in shown)
    assert printed_lines(capsys) == shown


async def _stand_in_judge(*, system: str, user: str, response_format: dict[str, Any] | None = None) -> Any:
    """The page's ``client``: passes the one dimension the engine's prompt asks for, as a model would answer."""
    asked = re.search(r'single key "([^"]+)"', system)
    assert asked is not None, "the judge prompt no longer names the one key its reply carries"
    reply = {"reasoning": "stand-in", "criteria_scores": {asked.group(1): "pass"}}
    return SimpleNamespace(
        content=json.dumps(reply),
        input_tokens=None,
        output_tokens=None,
        reasoning_tokens=None,
        cost_usd=0.0,
        price_source="stand-in",
        model="judge-model",
        served_model="judge-model",
        stop_reason="end_turn",
        temperature=0.0,
    )


async def _close() -> None:
    return None


#: The judges page's blocks this test does not run, each by a string only it holds, with why.
JUDGES_SKIPPED = {
    "label_key_of(evidence": "a fragment over an output the reader holds (`evidence`, `rubric_dim`, `scope_id`)",
    "SecondJudge(model=": "asks a second model, which the quick path's one lent judge client does not call",
    "judge_temperature_work(": "compares temperatures on borderline scores, and the stand-in's are all at an end",
    "JudgeProfilesRecord(judge_run_ids=": "records a judge campaign's arms (`arm_run_ids`), never launched on the page",
}


async def test_judges_and_calibration_runs_top_to_bottom_on_a_stand_in_client() -> None:
    """The rubric, the judged run, the ratings, agreement, tiers, repeats, frozen judge cases and their readings."""
    namespace: dict[str, Any] = {"client": SimpleNamespace(generate=_stand_in_judge, aclose=_close)}
    blocks = without(page_blocks("judges-and-calibration.md"), JUDGES_SKIPPED)
    await run_in_order(blocks, namespace, page="judges-and-calibration.md")
    assert namespace["frozen"].cases, "the freeze froze nothing, so the blocks after it read nothing"
