"""A lever the launch named as ``null`` is a level, or a recovery, and never a vanished lever (#574).

The defect: the effective-config resolution skipped a launch-named member whose value was ``None``, in
its first pass and again in its third. The lever then reached none of ``RunSummary.config``,
``config_provenance`` and the coverage map, so a sweep between a value and ``null`` read ``unswept`` at
the non-null level — a swept axis reported as unswept.

Two classes of lever, two meanings of ``null``:

- **No recovery rule.** The operator set the lever to nothing, which is a level: ``NULL_LEVEL``,
  stamped ``overridden``.
- **A recovery rule** (``observed_model_levers``). There ``null`` means "not stated, read it off what
  ran", so it resolves through the rule — ``inherited``, or ``unknown`` where the role ran on more than
  one model — and never becomes a level of its own.

Every campaign here is the toy host's with retrieval tuning registered, its two batches overlaying the
``retrieval.rerank_depth`` member — one to ``8``, one to ``null`` — read through
:func:`~threetears.evals.analysis.assemble_context_bundle`.
"""

from __future__ import annotations

from dataclasses import replace

from threetears.evals.analysis import AnalysisContextBundle, RunSummary, assemble_context_bundle
from threetears.evals.analysis.reporting import NULL_LEVEL, lever_level
from threetears.evals.kernel import EvalCampaign
from threetears.evals.schema import EvalResult, EvalRun
from threetears.evals.kernel.host import HostProfile
from threetears.evals.schema.models import RoleUsage
from packages.evals.tests.fixtures.toyhost.corpus import (
    TOYHOST_SCOPE,
    TOYHOST_SUBJECT,
    ToyhostStorage,
    toyhost_batch,
    toyhost_measurements,
)
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile

#: The swept knob, as its lever name.
_KNOB = "retrieval.rerank_depth"


def _batch(depth: int | None) -> EvalRun:
    """A toy batch that overlaid the knob to ``depth`` — ``None`` is an explicit ``null`` overlay."""
    return toyhost_batch(
        chunk_tokens=512,
        retriever_top_k=3,
        extraction_schema="v1",
        ocr_engine_version="tess-5.3.1",
        reviewer_pool="pool-a",
        retrieval_overrides={"rerank_depth": depth},
        retrieval_config={"rerank_depth": depth, "dedupe_threshold": 0.9},
    )


def _with_inner_agent(results: list[EvalResult], models: tuple[str, ...]) -> list[EvalResult]:
    """The results with an ``inner_agent`` usage row each, cycling through ``models``."""
    return [
        result.model_copy(
            update={
                "usage": [
                    *result.usage,
                    RoleUsage(role="inner_agent", model=models[i % len(models)], cost_usd=0.0, call_count=1),
                ]
            }
        )
        for i, result in enumerate(results)
    ]


def _bundle(profile: HostProfile, *, inner_models: tuple[str, ...] = ()) -> tuple[AnalysisContextBundle, str, str]:
    """Assemble the two-batch null-vs-value sweep; returns the bundle and the (valued, null) run ids."""
    valued, nulled = _batch(8), _batch(None)
    results = {}
    for batch, accuracy in ((valued, 0.9), (nulled, 0.8)):
        measured = toyhost_measurements(batch, profile=profile, cost_usd=0.002, total_ms=900.0, field_accuracy=accuracy)
        results[batch.id] = _with_inner_agent(measured, inner_models) if inner_models else measured
    campaign = EvalCampaign(
        id="5f1c7a92-3e84-4b0d-a6c2-9d17e8b40f55",
        scope_id=TOYHOST_SCOPE,
        name="rerank depth: a value against null",
        subject_id=TOYHOST_SUBJECT.subject_id,
        subject_kind="extractor_config",
        behavior="extract_invoice_fields",
        template_id="",
        run_ids=[valued.id, nulled.id],
        created_by="test:fixture",
    )
    storage = ToyhostStorage([valued, nulled], results)
    return assemble_context_bundle(campaign, storage=storage, profile=profile), valued.id, nulled.id


def _summary(bundle: AnalysisContextBundle, run_id: str) -> RunSummary:
    (summary,) = [s for s in bundle.run_summaries if s.run_id == run_id]
    return summary


def test_a_null_level_has_one_deliberate_spelling() -> None:
    """``null``, JSON's own — not ``"None"`` keyed by accident, and not the inherited ``'—'``."""
    assert lever_level(None) == NULL_LEVEL == "null"


class TestANullOnALeverWithNoRecoveryRuleIsALevel:
    """The operator set the knob to nothing: a level, ``overridden``, on all three surfaces."""

    def test_the_null_run_carries_the_null_level_as_overridden(self) -> None:
        bundle, _, nulled = _bundle(toyhost_profile(tunable_retrieval=True))

        summary = _summary(bundle, nulled)

        assert summary.config[_KNOB] == NULL_LEVEL
        assert summary.config_provenance[_KNOB] == "overridden"

    def test_the_sweep_between_a_value_and_null_is_on_the_coverage_map_as_swept(self) -> None:
        bundle, _, _ = _bundle(toyhost_profile(tunable_retrieval=True))

        (row,) = [row for row in bundle.coverage if row.name == _KNOB]

        assert sorted(row.levels) == sorted(["8", NULL_LEVEL])
        assert row.cells == 2
        assert row.status != "unswept", "a sweep between a value and null was reported as unswept"


class TestANullOnALeverWithARecoveryRuleResolvesThroughIt:
    """``null`` there means "not stated": recovered from what ran, never a fabricated level."""

    @staticmethod
    def _profile() -> HostProfile:
        return replace(toyhost_profile(tunable_retrieval=True), observed_model_levers={_KNOB: "inner_agent"})

    def test_one_observed_model_reads_inherited_on_all_three_surfaces(self) -> None:
        bundle, valued, nulled = _bundle(self._profile(), inner_models=("inner-m1",))

        summary = _summary(bundle, nulled)

        assert summary.config[_KNOB] == "inner-m1"
        assert summary.config_provenance[_KNOB] == "inherited"
        assert _summary(bundle, valued).config_provenance[_KNOB] == "overridden"
        (row,) = [row for row in bundle.coverage if row.name == _KNOB]
        assert NULL_LEVEL not in row.levels, "a recoverable null became a level of its own"
        assert sorted(row.levels) == sorted(["8", "inner-m1"])

    def test_an_ambiguous_recovery_reads_unknown_and_claims_no_level(self) -> None:
        bundle, _, nulled = _bundle(self._profile(), inner_models=("inner-m1", "inner-m2"))

        summary = _summary(bundle, nulled)

        assert summary.config_provenance[_KNOB] == "unknown"
        assert _KNOB not in summary.config, "'we could not establish this' must never read as a level"
        (row,) = [row for row in bundle.coverage if row.name == _KNOB]
        assert NULL_LEVEL not in row.levels
