"""The stored eval models, derived from the model graph rather than listed.

It also holds one valid sample of each (:func:`stored_sample`), which the strictness tests tamper
with and the core's frozen fixture corpus was generated from.

A stored model is any :class:`~threetears.evals.schema.base.EvalBaseModel` subclass defined in the
package that declares a ``doc_type`` — the field a document store discriminates on. Deriving the set
is the point: a list kept by hand fails by omission, and a model added without a row would then be
exempt from every rule the set is used to enforce, with nothing going red.
"""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable
from typing import Any

import threetears.evals
from packages.evals.tests.factories import (
    make_analysis,
    make_analysis_attempt,
    make_calibration_rating,
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
from threetears.evals.kernel.campaign import EvalSweep, SweepArmRecord
from threetears.evals.schema.base import EvalBaseModel
from threetears.evals.schema.models import (
    CaseSet,
    CassetteKey,
    EvalCassette,
    JudgeConfigTombstone,
    JudgeEvidence,
    RubricDim,
    RubricDimTombstone,
    RubricScore,
    label_key_of,
)
from threetears.evals.schema.out_of_run_spend import OutOfRunSpend

__all__ = ["doc_type_of", "sampled_model_names", "stored_models", "stored_sample"]


def stored_models() -> list[type[EvalBaseModel]]:
    """Every package model that declares a ``doc_type``, after importing all of ``threetears.evals``.

    Returns:
        The models, sorted by name so a parametrised test has stable ids.
    """
    for module in pkgutil.walk_packages(threetears.evals.__path__, prefix="threetears.evals."):
        importlib.import_module(module.name)
    seen: set[type[EvalBaseModel]] = set()
    stack: list[type[EvalBaseModel]] = [EvalBaseModel]
    while stack:
        for sub in stack.pop().__subclasses__():
            if sub not in seen:
                seen.add(sub)
                stack.append(sub)
    return sorted(
        (
            model
            for model in seen
            if model.__module__.startswith("threetears.evals.") and "doc_type" in model.model_fields
        ),
        key=lambda model: model.__name__,
    )


def doc_type_of(model: type[EvalBaseModel]) -> str:
    """The ``doc_type`` value a stored model writes.

    Args:
        model: A stored model.

    Returns:
        Its ``doc_type`` default — the discriminator every document of it carries.
    """
    default = model.model_fields["doc_type"].default
    assert isinstance(default, str), f"{model.__name__}.doc_type has no string default"
    return default


def _cassette() -> EvalCassette:
    """One action-seam recording, built the way the cassette proxy builds one."""
    key = CassetteKey(
        corpus_id="run-capture",
        template_id="tpl-1",
        test_case_id="tc-1",
        tool="search",
        action="query",
        params_hash="0" * 16,
        occurrence=0,
    )
    return EvalCassette.build(key, scope_id="uni-1", seam="action", captured_model="m", response={"hits": []})


def _out_of_run_spend() -> OutOfRunSpend:
    """One ledgered proposer call, as an out-of-run budget writes one."""
    return OutOfRunSpend(
        scope_id="uni-1",
        purpose="proposer",
        model="m",
        outcome="completed",
        stop_reason="end_turn",
        prompt_tokens=120,
        completion_tokens=40,
        cost_usd=0.002,
        price_source="provider",
        priced_ceiling_usd=0.01,
        cap_usd=1.0,
        subject_id="subj-1",
    )


#: The label key a sampled judge score and the rating of it carry (#628): the sample result's tone score, read
#: off one rendered output on the tone dimension's definition.
_LABEL_KEY = label_key_of(
    JudgeEvidence(case_material="a customer asks about the queue", artifact="assistant: two tickets are waiting"),
    RubricDim(name="conversation.tone", description="Warm and on topic.", scale="ordinal"),
)


#: A valid instance of every stored model, by name. Keyed by the derived population below, so a
#: stored model added without a row here fails :func:`test_every_stored_model_has_a_sample` rather
#: than slipping past both refusals and the core's frozen fixtures.
_SAMPLES: dict[str, Callable[[], EvalBaseModel]] = {
    "CalibrationRating": lambda: make_calibration_rating(
        output_fingerprint=_LABEL_KEY.output_fingerprint, criterion_fingerprint=_LABEL_KEY.criterion_fingerprint
    ),
    "EvalSweep": lambda: EvalSweep(
        scope_id="uni-1",
        campaign_id="c-1",
        template_id="t-1",
        subject_id="s-1",
        arms=[SweepArmRecord(label="a", model="m")],
        max_concurrent_arms=1,
    ),
    "CaseSet": lambda: CaseSet(scope_id="uni-1", name="smoke", version=1, template_id="t-1", test_case_ids=["c-1"]),
    "CatalogRubricDim": make_rubric_dim,
    "EvalAnalysis": make_analysis,
    "EvalAnalysisAttempt": make_analysis_attempt,
    "EvalCampaign": make_campaign,
    "EvalCassette": _cassette,
    "EvalInsight": make_insight,
    "EvalResult": lambda: make_eval_result(
        rubric_scores=[
            RubricScore(
                dim="conversation.tone",
                score=4,
                scale="ordinal",
                output_fingerprint=_LABEL_KEY.output_fingerprint,
                criterion_fingerprint=_LABEL_KEY.criterion_fingerprint,
            )
        ]
    ),
    "EvalRun": make_eval_run,
    "EvalTemplate": make_template,
    "EvalTestCase": make_test_case,
    "EvalTrace": make_eval_trace,
    "JudgeConfig": make_judge_config,
    "OutOfRunSpend": _out_of_run_spend,
    "RubricDimTombstone": lambda: RubricDimTombstone(scope_id="uni-1", key="conversation.tone", deleted_dim_id="d-1"),
    "JudgeConfigTombstone": lambda: JudgeConfigTombstone(
        scope_id="uni-1", rubric_dim_id="conversation.tone", name="tone-strict", deleted_config_id="c-1"
    ),
}


def stored_sample(model: type[EvalBaseModel]) -> dict[str, Any]:
    """A stored document of ``model``, as the store hands it back, after checking it loads untampered.

    Args:
        model: A stored model (one :func:`stored_models` returns).

    Returns:
        A valid document of it, as ``to_dict`` writes one.
    """
    sample = _SAMPLES[model.__name__]()
    assert type(sample) is model, f"the sample for {model.__name__} is a {type(sample).__name__}"
    document = sample.to_dict()
    assert model.from_dict(document) == sample, "the control: an untampered document reads back whole"
    return document


def sampled_model_names() -> list[str]:
    """The names of the stored models :func:`stored_sample` has a sample for, sorted.

    Returns:
        The model names.
    """
    return sorted(_SAMPLES)
