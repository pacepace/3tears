"""The mechanism check's false "moved" rate under an inert lever, computed exactly through the bundle (#601).

A swept lever that declares what it acts on is checked with the engine's separation test across its levels:
``moved`` when some pair of levels separates (Holm-corrected). A gap with no spread at all — every case moved
by the same nonzero amount, or two different constants — is read by the exact permutation test, so it
separates only over enough cases for that pattern to be rarer than α by chance, and below that the check is
``unchecked`` (``too_few_observations``), never ``moved`` or ``inert``. A reader takes ``moved`` as evidence
the lever took effect, so under a lever that did nothing it should read ``moved`` at most α of the time; and
it should still read ``inert`` wherever the data can say so.

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
from collections import Counter

import pytest

from threetears.evals.analysis import assemble_context_bundle
from threetears.evals.analysis.stats import SIGNIFICANCE_ALPHA
from threetears.evals.kernel import EvalCampaign
from threetears.evals.schema import EvalResult, EvalRun
from threetears.evals.kernel.host import HostProfile
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
) -> tuple[str, str | None]:
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
    return row.mechanism.state, row.mechanism.reason


def _exact_readings(n_cases: int) -> Counter[tuple[str, str | None]]:
    """The probability of each reading when every observation is a fair 0/1 at both levels."""
    sweep = _sweep(n_cases)
    assignments = list(itertools.product((0, 1), repeat=2 * n_cases))
    readings = Counter(_mechanism_state(sweep, values) for values in assignments)
    return Counter({reading: count / len(assignments) for reading, count in readings.items()})


@pytest.mark.parametrize("n_cases", [2, 3])
def test_an_inert_lever_reads_moved_at_most_alpha(n_cases: int) -> None:
    """Before the exact test, every case shifting by the same ±1 read ``moved``: 2 of 16 at 2 cases (0.125)."""
    rate = _exact_readings(n_cases)[("moved", None)]
    assert rate <= SIGNIFICANCE_ALPHA, f"exact false 'moved' rate {rate:.4f} against α={SIGNIFICANCE_ALPHA}"


@pytest.mark.parametrize("n_cases", [2, 3])
def test_an_inert_lever_is_still_caught_where_the_data_can_say_so(n_cases: int) -> None:
    """Only the alike-shifted assignments go uncalled; every other one still reads ``inert``.

    Both cases (or all three) shifting by the same ±1 is ``2 / 4 ** n`` of the mass. The covariate declares no
    range, so no test of the mean can call that shift at any n (#597), and reading it ``inert`` would hide a
    pattern the data cannot rule out: it is ``unchecked`` for ``uniform_move_needs_range``, naming the remedy.
    """
    readings = _exact_readings(n_cases)
    alike = 2 / 4**n_cases
    assert readings == pytest.approx({("inert", None): 1 - alike, ("unchecked", "uniform_move_needs_range"): alike}), (
        dict(readings)
    )
