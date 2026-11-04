"""No generation runs the memo writer: ``analysis_generate`` writes with the report writer and names no prompt.

The memo writer's ``eval_analysis_gen`` stays registered until the memo writer itself leaves the package. The report
planner's slot, ``eval_report_plan``, is retired already (``test_retired_names``).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from threetears.evals.kernel import ANALYSIS_GEN_PROMPT, SEED_PROMPTS
from threetears.evals.ops import AnalysisGeneration

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src" / "threetears" / "evals"

#: The modules a generation runs through: from the ``analysis_generate`` action to the stored document's analysis.
GENERATION_PATH = (
    "ops/analysis.py",
    "ops/host.py",
    "analysis/doc_analysis.py",
    "analysis/service.py",
)

#: What only the memo writer's path names.
WRITER_NAMES = frozenset({"ANALYSIS_GEN_PROMPT", "EVAL_ANALYSIS_GEN_DEFAULT", "generate_analysis", "gen_prompt"})


def test_a_generation_names_no_prompt_of_its_own():
    """``analysis_generate`` writes with the analyst skill the package ships, so its settings name no prompt at all."""
    generation = AnalysisGeneration(budget_s=1.0)

    assert not any(hasattr(generation, name) for name in ("prompt_id", "resolve_prompt", "max_output_tokens", "prompt"))
    # A host still passing the planner's prompt settings is refused by name, never read.
    for retired in ("prompt_id", "resolve_prompt"):
        with pytest.raises(TypeError, match=retired):
            AnalysisGeneration(budget_s=1.0, **{retired: None})


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


def test_the_writers_prompt_stays_registered_until_the_memo_writer_leaves():
    assert ANALYSIS_GEN_PROMPT in {entry.prompt_type for entry in SEED_PROMPTS}
