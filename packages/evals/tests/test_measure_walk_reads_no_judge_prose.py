"""The measure walk never reads the judge's prose, and a host measure named like it pools the host's own values.

The judge's prose is a score's ``reasoning`` and the result's ``judge_reasoning``: text the judge wrote, never a
measure. A host may still declare a measure of either name (the registry refuses only the engine's core names), and
report it on ``host_measures``. Pinned here, for a host declaring ``reasoning`` and ``judge_reasoning`` as text:

- **with only the judge's prose present**, neither name is pooled and neither is listed unreported. Until the walk
  read declared fields, it walked the prose under those names: ``judge_reasoning`` pooled the judge's narration, and
  ``reasoning`` pooled a score's reasoning (or was listed unreported when two score carriers both held one);
- **with the host's own values on** ``host_measures``, each name pools exactly the host's values. Until then
  ``judge_reasoning`` pooled the judge's narration beside the host's value.
"""

from __future__ import annotations

from typing import Any

import pytest

from threetears.evals.kernel import MetricDescriptor
from threetears.evals.kernel.analysis_measures import MeasureCollection
from threetears.evals.kernel.host import SHARED_CORE, HostProfile, MeasureRegistry
from threetears.evals.schema.models import TRANSCRIPT_DIM_ID, EvalResult, RubricScore
from packages.evals.tests.bundle_support import one_batch_bundle
from packages.evals.tests.factories import make_eval_result

#: The names of the judge's prose, each declared by the host as a text measure of its own.
PROSE_NAMES = ("judge_reasoning", "reasoning")


def _text_measure(name: str) -> MetricDescriptor:
    return MetricDescriptor(
        name=name,
        reader_name=f"{name} reading",
        data_type="text",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="end_to_end",
        description=f"The host's own {name}, for the test.",
    )


_PROFILE = HostProfile(
    host_id="prose-named",
    host_sweepables=SHARED_CORE,
    measures=MeasureRegistry(tuple(_text_measure(name) for name in PROSE_NAMES)),
)


def _result(case: str, *, transcript: bool = False, **host_measures: Any) -> EvalResult:
    """A judged result carrying the judge's prose on every place it can: its narration and each score's reasoning."""
    return make_eval_result(
        test_case_id=case,
        judge_reasoning="the judge narrates the whole exchange",
        rubric_scores=[
            RubricScore(
                dim="conversation.tone", score=4, scale="ordinal", axis="capability", reasoning="the judge on tone"
            )
        ],
        transcript_score=(
            RubricScore(
                dim=TRANSCRIPT_DIM_ID, score=4, scale="ordinal", axis="capability", reasoning="the judge on the whole"
            )
            if transcript
            else None
        ),
        goal_state_outcomes=[],
        host_measures=host_measures,
    )


def _cell(results: list[EvalResult]) -> MeasureCollection:
    (cell,) = one_batch_bundle(results, profile=_PROFILE).cell_measures
    return cell.measures


def _pooled(collection: MeasureCollection) -> dict[str, list[str]]:
    return {summary.name: list(summary.texts or []) for summary in collection.measures if summary.name in PROSE_NAMES}


@pytest.mark.parametrize("transcript", [False, True], ids=["one-score-carrier", "two-score-carriers"])
def test_the_judges_prose_is_neither_pooled_nor_listed_unreported(transcript: bool) -> None:
    collection = _cell([_result("c1", transcript=transcript), _result("c2", transcript=transcript)])

    assert _pooled(collection) == {}
    assert not [name for name in collection.unreported_observations if name.split(" ")[0] in PROSE_NAMES]


def test_a_host_measure_named_like_the_prose_pools_the_hosts_values_alone() -> None:
    collection = _cell(
        [
            _result("c1", transcript=True, reasoning="the host's first reason", judge_reasoning="the host's ruling"),
            _result("c2", transcript=True, reasoning="the host's second reason"),
        ]
    )

    assert _pooled(collection) == {
        "judge_reasoning": ["the host's ruling"],
        "reasoning": ["the host's first reason", "the host's second reason"],
    }
    assert not [name for name in collection.unreported_observations if name.split(" ")[0] in PROSE_NAMES]
