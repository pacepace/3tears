"""The mechanism check's false "moved" rate under an inert lever, computed exactly through the bundle (#601).

A swept lever that declares what it acts on is checked with the engine's separation test across its levels:
``moved`` when some pair of levels separates (Holm-corrected), and a gap with no spread at all — every case
moved by the same nonzero amount, or two different constants — counts as separated outright. A reader takes
``moved`` as evidence the lever took effect, so under a lever that did nothing it should read ``moved`` at
most α of the time.

For a mechanism that takes few values per observation (a count, a flag) the false-positive rate can be
computed EXACTLY rather than simulated: every assignment of values to the observations is enumerated, each
is assembled into a real bundle, and the rate is the probability mass of the assignments the bundle calls
``moved``. No Monte-Carlo error, and the decision is the bundle's own, private rule included.

The inert lever here: the toy host's ``chunk_tokens`` at 256 and 1024, which declares it acts on
``context_tokens_in``; each observation records that covariate as 0 or 1 with probability one half at
BOTH levels, independently, one repeat per case.
"""

from __future__ import annotations

import itertools

import pytest

from threetears.evals.analysis import assemble_context_bundle
from threetears.evals.analysis.stats import SIGNIFICANCE_ALPHA
from threetears.evals.contracts import EvalCampaign, EvalResult, EvalRun
from threetears.evals.contracts.host import HostProfile
from packages.evals.tests.fixtures.toyhost.corpus import (
    TOYHOST_DOCUMENTS,
    TOYHOST_SCOPE,
    TOYHOST_SUBJECT,
    ToyhostStorage,
    toyhost_batch,
    toyhost_measurements,
)
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile

_MECHANISM = "context_tokens_in"


def _batch(chunk_tokens: int) -> EvalRun:
    return toyhost_batch(
        chunk_tokens=chunk_tokens,
        retriever_top_k=3,
        extraction_schema="v1",
        ocr_engine_version="tess-5.3.1",
        reviewer_pool="pool-a",
    )


def _sweep(n_cases: int) -> tuple[HostProfile, tuple[EvalRun, EvalRun], dict[str, list[EvalResult]]]:
    """The two-level sweep's batches and their first-repeat observations of the first ``n_cases`` documents."""
    profile = toyhost_profile()
    batches = (_batch(256), _batch(1024))
    documents = TOYHOST_DOCUMENTS[:n_cases]
    observed = {
        batch.id: [
            result
            for result in toyhost_measurements(
                batch, profile=profile, cost_usd=0.02, total_ms=900.0, field_accuracy=0.8
            )
            if result.test_case_id in documents and result.k_iteration == 1
        ]
        for batch in batches
    }
    return profile, batches, observed


def _mechanism_state(
    sweep: tuple[HostProfile, tuple[EvalRun, EvalRun], dict[str, list[EvalResult]]], values: tuple[int, ...]
) -> str:
    """Assemble the sweep with ``values`` as the observations' covariate, and read the check.

    ``values`` holds the narrow level's observations, then the wide level's, one per document in order.
    """
    profile, batches, observed = sweep
    n_cases = len(values) // 2
    results = {
        batch.id: [
            result.model_copy(
                update={
                    "covariates": {
                        _MECHANISM: float(values[level * n_cases + TOYHOST_DOCUMENTS.index(result.test_case_id)])
                    }
                }
            )
            for result in observed[batch.id]
        ]
        for level, batch in enumerate(batches)
    }
    campaign = EvalCampaign(
        id="0b6f1c2e-4a7d-4e59-8c31-2d9e7f5a1b46",
        scope_id=TOYHOST_SCOPE,
        name="inert lever",
        subject_id=TOYHOST_SUBJECT.subject_id,
        subject_kind="extractor_config",
        behavior="extract_invoice_fields",
        template_id="",
        run_ids=[batch.id for batch in batches],
        created_by="test:fixture",
    )
    bundle = assemble_context_bundle(campaign, storage=ToyhostStorage(list(batches), results), profile=profile)
    (row,) = [entry for entry in bundle.coverage if entry.name == "chunk_tokens"]
    assert row.mechanism.measure == _MECHANISM, "the check must be reading the covariate this file writes"
    return row.mechanism.state


def _exact_moved_rate(n_cases: int) -> float:
    """The probability the check reads ``moved`` when every observation is a fair 0/1 at both levels."""
    sweep = _sweep(n_cases)
    assignments = list(itertools.product((0, 1), repeat=2 * n_cases))
    moved = sum(1 for values in assignments if _mechanism_state(sweep, values) == "moved")
    return moved / len(assignments)


def test_three_cases_a_level_read_moved_at_most_alpha() -> None:
    """At 3 cases the only false ``moved`` is every case shifting by the same ±1: 2 of 64 assignments, 0.031."""
    rate = _exact_moved_rate(3)
    assert rate <= SIGNIFICANCE_ALPHA, f"exact false 'moved' rate {rate:.4f} against α={SIGNIFICANCE_ALPHA}"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "#601 finding: the mechanism check counts a zero-spread gap as separated at any case count. With 2 cases a "
        "level and a 0/1 mechanism under an inert lever, both cases shifting by the same ±1 is 2 of 16 equally "
        "likely assignments: exact false 'moved' rate 0.125 against alpha 0.05. paired_change floors the same "
        "reasoning at the 5 pairs an exact sign-flip test needs; the mechanism check has no floor."
    ),
)
def test_two_cases_a_level_read_moved_at_most_alpha() -> None:
    rate = _exact_moved_rate(2)
    assert rate <= SIGNIFICANCE_ALPHA, f"exact false 'moved' rate {rate:.4f} against α={SIGNIFICANCE_ALPHA}"
