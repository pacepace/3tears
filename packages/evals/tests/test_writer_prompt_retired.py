"""The memo writer's prompt is retired as the default: the planner's is what a host seeds and a generation runs.

A new analysis is written by the report planner under ``eval_report_plan``. The writer's ``eval_analysis_gen`` stays
registered only for the memo reporter cases (kind version 1) that generate a memo candidate, and for reading old memos.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from threetears.evals import analysis
from threetears.evals.analysis import plan_prompt
from threetears.evals.analysis.gen_prompt import EVAL_ANALYSIS_GEN_DEFAULT
from threetears.evals.analysis.reporter_kind import MEMO_WRITER_CALLS
from threetears.evals.kernel import ANALYSIS_GEN_PROMPT, SEED_PROMPTS
from threetears.evals.kernel.prompts.catalogue import REPORT_PLAN_PROMPT
from threetears.evals.ops import AnalysisGeneration

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src" / "threetears" / "evals"

#: The modules a generation runs through: from the ``analysis_generate`` action to the stored document's analysis, and
#: the planner's path (which the reporter kind runs) to its two calls.
GENERATION_PATH = (
    "ops/analysis.py",
    "ops/host.py",
    "analysis/doc_analysis.py",
    "analysis/service.py",
    "analysis/planned_analysis.py",
    "analysis/planner.py",
)

#: What only the memo writer's path names.
WRITER_NAMES = frozenset({"ANALYSIS_GEN_PROMPT", "EVAL_ANALYSIS_GEN_DEFAULT", "generate_analysis", "gen_prompt"})


def test_the_planners_default_is_exported_beside_the_writers():
    assert analysis.EVAL_REPORT_PLAN_DEFAULT is plan_prompt.EVAL_REPORT_PLAN_DEFAULT
    assert "EVAL_REPORT_PLAN_DEFAULT" in analysis.__all__
    assert analysis.EVAL_ANALYSIS_GEN_DEFAULT is EVAL_ANALYSIS_GEN_DEFAULT


def test_the_planners_seed_ref_is_the_exported_constant():
    (entry,) = [entry for entry in SEED_PROMPTS if entry.prompt_type == REPORT_PLAN_PROMPT]
    module, attribute = entry.seed_ref
    assert getattr(__import__(module, fromlist=[attribute]), attribute) is analysis.EVAL_REPORT_PLAN_DEFAULT


def test_a_generation_names_no_prompt_of_its_own():
    """``analysis_generate`` writes with the analyst skill the package ships, so its settings name no prompt at all:
    neither the writer's nor the planner's."""
    generation = AnalysisGeneration(budget_s=1.0)

    assert not any(hasattr(generation, name) for name in ("prompt_id", "resolve_prompt", "max_output_tokens", "prompt"))


@pytest.mark.parametrize("relative", GENERATION_PATH)
def test_no_module_on_the_generation_path_names_the_writers_prompt_or_entry_point(relative: str):
    tree = ast.parse((SOURCE_ROOT / relative).read_text(encoding="utf-8"))
    named = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    named |= {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    named |= {
        alias.name.rpartition(".")[2]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    named |= {
        node.module.rpartition(".")[2] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
    }

    assert not named & WRITER_NAMES, f"{relative} reaches the memo writer: {sorted(named & WRITER_NAMES)}"


def test_the_writers_prompt_stays_registered_for_memo_reporter_cases():
    assert ANALYSIS_GEN_PROMPT in {entry.prompt_type for entry in SEED_PROMPTS}
    assert MEMO_WRITER_CALLS == 2
