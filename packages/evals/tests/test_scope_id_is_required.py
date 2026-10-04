"""Every stored document names a non-blank scope, and nothing supplies one for it.

``scope_id`` is the one opaque partition the engine carries and never interprets. A blank
scope is not a smaller scope: it is a document no scoped read can ever return, which an
operator would meet as data that was written and then vanished. So the field has no
default and a minimum length of one, and storage stamps nothing in its place.

Each model is built valid from its factory, then re-validated with the scope missing,
empty, and whitespace-only. Whitespace is a case of its own because the eval base strips
strings before ``min_length`` runs; a scope of spaces must fail the same way an empty one
does, not survive as a distinct partition.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from threetears.evals.analysis.cells import Observation
from threetears.evals.contracts.models import CassetteKey, EvalCassette

from packages.evals.tests.factories import (
    make_analysis,
    make_analysis_attempt,
    make_campaign,
    make_eval_result,
    make_eval_run,
    make_eval_trace,
    make_insight,
    make_judge_config,
    make_rubric_dim,
    make_template,
    make_test_case,
)


def _cassette() -> EvalCassette:
    return EvalCassette.build(
        CassetteKey(
            corpus_id="run-capture",
            template_id="tpl-1",
            test_case_id="tc-1",
            tool="lookup",
            action="find",
            params_hash="0" * 64,
            occurrence=0,
        ),
        scope_id="scope-a",
        seam="action",
        response={"ok": True},
        captured_model="writer-a",
    )


def _observation() -> Observation:
    return Observation(
        id="obs-1",
        scope_id="scope-a",
        variant_key="v" * 64,
        apparatus_class_id="class-1",
        provenance="declared",
    )


_BUILDERS: dict[str, Callable[[], BaseModel]] = {
    "EvalTestCase": make_test_case,
    "EvalRun": make_eval_run,
    "EvalResult": make_eval_result,
    "EvalTrace": make_eval_trace,
    "EvalCassette": _cassette,
    "Observation": _observation,
    "EvalTemplate": make_template,
    "JudgeConfig": make_judge_config,
    "CatalogRubricDim": make_rubric_dim,
    "EvalCampaign": make_campaign,
    "EvalAnalysis": make_analysis,
    "EvalAnalysisAttempt": make_analysis_attempt,
    "EvalInsight": make_insight,
}


def _document(name: str) -> tuple[type[BaseModel], dict[str, Any]]:
    model = _BUILDERS[name]()
    return type(model), model.model_dump()


@pytest.mark.parametrize("name", sorted(_BUILDERS))
def test_the_valid_document_round_trips(name: str) -> None:
    """The positive control: the document the refusals mutate is itself accepted."""
    cls, document = _document(name)
    assert cls.model_validate(document).scope_id == document["scope_id"]


@pytest.mark.parametrize("name", sorted(_BUILDERS))
def test_a_missing_scope_is_refused(name: str) -> None:
    cls, document = _document(name)
    del document["scope_id"]
    with pytest.raises(ValidationError, match="scope_id"):
        cls.model_validate(document)


@pytest.mark.parametrize("blank", ["", "   "], ids=["empty", "whitespace"])
@pytest.mark.parametrize("name", sorted(_BUILDERS))
def test_a_blank_scope_is_refused(name: str, blank: str) -> None:
    cls, document = _document(name)
    document["scope_id"] = blank
    with pytest.raises(ValidationError, match="scope_id"):
        cls.model_validate(document)
