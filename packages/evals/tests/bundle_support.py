"""Test support for suites that read a summary off the analysis bundle: one batch, assembled.

The bundle's summaries are built by the assembler and nowhere else, so a suite asserting how a
measure or a token count is summarised reaches it the way every consumer does — through
:func:`~threetears.evals.analysis.assemble_context_bundle` — over a store holding one batch of the
observations under test. The two surfaces a measure is summarised on take different populations by
default, and both are on the result: the single cell's facts (``scored``) and the batch's run
summary (``all_observed``).
"""

from __future__ import annotations

from collections.abc import Sequence

from threetears.evals.analysis import AnalysisContextBundle, assemble_context_bundle
from threetears.evals.contracts import EvalResult
from threetears.evals.contracts.campaign import EvalCampaign
from threetears.evals.contracts.host import HostProfile
from packages.evals.tests.factories import make_eval_run
from packages.evals.tests.fixtures.toyhost.corpus import ToyhostStorage

__all__ = ["one_batch_bundle"]


def one_batch_bundle(results: Sequence[EvalResult], *, profile: HostProfile) -> AnalysisContextBundle:
    """Assemble a one-batch campaign over ``results``, in ``profile``'s vocabulary.

    Args:
        results: The observations, rebound to the one completed batch the campaign holds.
        profile: The host whose measure registry the assembler reads.

    Returns:
        The assembled bundle.
    """
    run = make_eval_run(status="completed")
    campaign = EvalCampaign(
        scope_id=run.scope_id,
        name="one batch",
        subject_id=run.subject_snapshot.subject_id,
        subject_kind="s",
        behavior="b",
        run_ids=[run.id],
        created_by="test:fixture",
    )
    members = [result.model_copy(update={"eval_run_id": run.id, "scope_id": run.scope_id}) for result in results]
    return assemble_context_bundle(campaign, storage=ToyhostStorage([run], {run.id: members}), profile=profile)
