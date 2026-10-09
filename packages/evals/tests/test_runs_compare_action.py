"""Two runs compared through the catalogue: ``runs_compare`` is the two-run lens with every comparison's disclosures.

Driven over the toy host through :meth:`MountedTool.call`, the path every transport takes. Pinned here:

- **The action is a thin binding of its lens.** ``comparison`` is ``compare_two_runs``'s answer as the lens
  returns it, value for value, and the structured result validates as :class:`RunsCompared`.
- **A short arm's comparison says the arm is short.** A run that delivered less than its matrix carries its
  completeness sentence beside the numbers, in the structured result and the text, and a whole one carries none.
- **A run outside the caller's scope is not found**, as on every read.
"""

from __future__ import annotations

from typing import Any

import pytest

from threetears.evals.actions import Caller, MountedTool, eval_catalogue, standard_tools
from threetears.evals.analysis.reads import compare_two_runs
from threetears.evals.analysis.reporting import completeness_disclosure
from threetears.evals.contracts.errors import NotFoundError
from threetears.evals.contracts.models import RubricScore, RunCompleteness
from threetears.evals.ops import RunsCompared, runs_compare, runs_compared_text
from packages.evals.tests.factories import make_eval_result, make_eval_run
from packages.evals.tests.ops_support import CALLER, TOYHOST_SCOPE, OpsFixture, ops_fixture

#: Two of five cells delivered: a run that came up short.
SHORT = RunCompleteness(
    expected_cells=5, produced_cells=2, persisted_cells=2, infra_excluded_cells=0, counted_from="run_loop"
)
WHOLE = RunCompleteness(
    expected_cells=2, produced_cells=2, persisted_cells=2, infra_excluded_cells=0, counted_from="run_loop"
)


@pytest.fixture
def evals() -> MountedTool:
    return eval_catalogue().mount_all(standard_tools())[0]


def _two_runs(fixture: OpsFixture, *, baseline: RunCompleteness, candidate: RunCompleteness) -> tuple[str, str]:
    """Store two runs of one template, over the same two cases, each with a scored result per case."""
    storage = fixture.host.eval_host.storage
    ids = []
    for name, model, completeness, composite in (
        ("run-a", "model-a", baseline, 3),
        ("run-b", "model-b", candidate, 4),
    ):
        run = make_eval_run(
            id=name,
            scope_id=TOYHOST_SCOPE,
            template_id="tpl-1",
            candidate_model=model,
            status="completed",
            test_case_ids=["case-0", "case-1"],
            completeness=completeness,
        )
        storage.save_eval_run(run)
        for index in range(2):
            storage.save_eval_result(
                make_eval_result(
                    id=f"{name}-{index}",
                    eval_run_id=run.id,
                    scope_id=TOYHOST_SCOPE,
                    model=model,
                    test_case_id=f"case-{index}",
                    rubric_scores=[RubricScore(dim="conversation.tone", score=composite + index, scale="ordinal")],
                )
            )
        ids.append(run.id)
    return ids[0], ids[1]


def _no_template(template_id: str) -> Any:
    raise NotFoundError("template", template_id)


async def _call(tool: MountedTool, fixture: OpsFixture, arguments: dict[str, Any], caller: Caller = CALLER) -> Any:
    return await tool.call(arguments, host=fixture.host, caller=caller)


async def test_the_action_returns_the_lens_answer_with_a_short_arm_disclosed(evals: MountedTool) -> None:
    fixture = ops_fixture()
    baseline, candidate = _two_runs(fixture, baseline=SHORT, candidate=WHOLE)

    outcome = await _call(
        evals, fixture, {"action": "runs_compare", "baseline_run_id": baseline, "candidate_run_id": candidate}
    )

    assert not outcome.is_error, outcome.text
    compared = RunsCompared.model_validate(outcome.structured)
    assert compared == runs_compare(fixture.host.eval_host, baseline, candidate, TOYHOST_SCOPE)
    lens = compare_two_runs(
        fixture.host.eval_host.storage,
        baseline,
        candidate,
        TOYHOST_SCOPE,
        load_template=_no_template,
        subject_detail=lambda run: {},
    )
    assert compared.comparison["comparison"] == lens["comparison"], "the numbers are the lens's, not re-derived"
    # The short arm is named, with the sentence every comparison surface carries for it; the whole one is not.
    sentence = completeness_disclosure(SHORT)
    assert sentence is not None
    assert compared.completeness_disclosures == {baseline: sentence}
    assert sentence in outcome.text and baseline in outcome.text


def test_two_whole_runs_carry_no_completeness_sentence() -> None:
    fixture = ops_fixture()
    baseline, candidate = _two_runs(fixture, baseline=WHOLE, candidate=WHOLE)

    compared = runs_compare(fixture.host.eval_host, baseline, candidate, TOYHOST_SCOPE)

    assert compared.completeness_disclosures == {}
    assert "incomplete runs" not in runs_compared_text(compared)
    assert compared.comparison["comparison"]["arm"]["paired"] is True, "the two runs scored the same two cases"


async def test_a_run_outside_the_callers_scope_is_not_found(evals: MountedTool) -> None:
    fixture = ops_fixture()
    baseline, _ = _two_runs(fixture, baseline=WHOLE, candidate=WHOLE)

    outcome = await _call(
        evals, fixture, {"action": "runs_compare", "baseline_run_id": baseline, "candidate_run_id": "no-such-run"}
    )

    assert outcome.is_error and "no-such-run" in outcome.text
