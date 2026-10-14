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
import sys
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
    """Run ``blocks`` one after another in ``namespace``, awaiting each that awaits, as ``python -m asyncio`` would.

    The blocks run in a module of their own, as a reader's ``__main__`` is one: a pydantic model a block declares
    resolves its annotations in the module it was defined in, which must be a module the interpreter knows.
    """
    # One module per namespace, made on its first page, so no test reads another's names.
    stem = page.removesuffix(".md").replace("-", "_")
    name = namespace.setdefault("__name__", f"doc_blocks_{stem}_{id(namespace)}")
    module = sys.modules.setdefault(name, types.ModuleType(name))
    module.__dict__.update(namespace)
    try:
        for index, block in enumerate(blocks):
            code = compile(block, f"{page}[{index}]", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
            ran = eval(code, module.__dict__)
            if isinstance(ran, types.CoroutineType):
                await ran
    finally:
        namespace.update(module.__dict__)


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
    # The page's client names the model it calls, as a Judge's client does.
    client = SimpleNamespace(generate=_stand_in_judge, aclose=_close, model_name="judge-model")
    namespace: dict[str, Any] = {"client": client}
    blocks = without(page_blocks("judges-and-calibration.md"), JUDGES_SKIPPED)
    await run_in_order(blocks, namespace, page="judges-and-calibration.md")
    assert namespace["frozen"].cases, "the freeze froze nothing, so the blocks after it read nothing"


async def test_the_host_guide_s_own_client_never_fails_a_request_and_passes_the_negotiation_check() -> None:
    """The guide's minimal router client over raw HTTP, and the check it shows, run as written.

    The second block is ``check_refusal_negotiation`` over the first block's client, so the canonical client a host
    copies is held to the same check the host runs; a completion it returns also conforms, ``sent_settings`` and all,
    and its stop reasons map the router's ``error`` finish reason, and a 200 with no choices, onto ``error``. A
    catalogue row the client is given is followed: a parameter no endpoint lists is never sent.
    """
    from threetears.evals.testing import check_completion_conformance

    blocks = [block for block in page_blocks("adopting-a-host.md") if "RouterClient" in block]
    assert len(blocks) == 2, "the guide's own-client section is one client block and one check block"
    namespace: dict[str, Any] = {}
    await run_in_order(blocks, namespace, page="adopting-a-host.md")

    finish: list[str | None] = ["length", "error", None]
    bodies: list[Mapping[str, object]] = []

    async def served(body: Mapping[str, object]) -> tuple[int, Mapping[str, object]]:
        bodies.append(body)
        reason = finish.pop(0)
        choices = [] if reason is None else [{"message": {"content": "{}"}, "finish_reason": reason}]
        return 200, {"model": "openai/gpt-6-luna-0901", "choices": choices, "usage": {"prompt_tokens": 3}}

    client = namespace["RouterClient"](
        "openai/gpt-6-luna",
        asked=namespace["WireSettings"](max_tokens=512, temperature=0.0),
        post=served,
        learned=namespace["LearnedSettings"](),
        supported_parameters=["max_tokens", "response_format", "structured_outputs"],
    )
    completion = await client.generate(system="s", user="u", response_format={"type": "json_object"})
    check_completion_conformance(completion, require_sent_settings=True)
    assert completion.stop_reason == "max_tokens" and completion.served_model == "openai/gpt-6-luna-0901"
    assert "temperature" not in bodies[0] and completion.temperature is None, "the catalogue lists no temperature"
    stops = [(await client.generate(system="s", user="u")).stop_reason for _ in range(2)]
    assert stops == ["error", "error"], "the router's error finish reason, then a 200 with no choices"
