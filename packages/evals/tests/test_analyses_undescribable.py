"""Which stored analyses hold an arm this build cannot describe, across a scope (#640).

The per-arm disclosure (``VariantIndexEntry.levels_unavailable``) reaches one analysis's arm table, surface
table and reporter bank; nothing answered the fleet question without opening each analysis. The
``analyses_undescribable`` operation and action read every stored analysis's frozen variant index on each
call — nothing is stored beside it — and name those holding such an arm, with the count and the reasons.
"""

from __future__ import annotations

from typing import Any

from threetears.evals.actions import eval_catalogue, standard_tools
from threetears.evals.contracts.campaign import EvalAnalysis
from threetears.evals.ops import UndescribableArmsListing, analyses_undescribable, analysis_generate
from packages.evals.tests.ops_support import CALLER, TOYHOST_SCOPE, OpsFixture, ops_fixture, settled

REASON = "its key was minted by variant predicate v1, which this build does not reproduce"


async def _two_analyses() -> tuple[OpsFixture, EvalAnalysis, EvalAnalysis]:
    """The toy campaign's generated analysis, and a copy of it whose first arm cannot be described."""
    fixture = ops_fixture()
    (job,) = (await analysis_generate(fixture.host, fixture.campaign.id, TOYHOST_SCOPE)).jobs
    assert (await settled(fixture.host, job.job_id)).state == "completed"
    storage = fixture.host.eval_host.storage
    (described,) = storage.list_analyses_by_campaign(fixture.campaign.id, TOYHOST_SCOPE)
    stored: dict[str, Any] = described.model_dump(mode="json")
    stored["id"] = "analysis-undescribable"
    first = stored["variant_index"][0]
    first.update(levers={}, swept={}, folded=[], levels_unavailable=REASON)
    undescribable = EvalAnalysis.model_validate(stored)
    storage.save_analysis(undescribable)
    return fixture, described, undescribable


async def test_the_operation_names_the_analysis_with_an_undescribable_arm_and_not_the_other() -> None:
    fixture, described, undescribable = await _two_analyses()

    listing = analyses_undescribable(fixture.host.eval_host, TOYHOST_SCOPE)

    assert listing.analyses_read == 2
    (line,) = listing.analyses
    assert line.analysis_id == undescribable.id != described.id
    assert (line.campaign_id, line.undescribable_arms, line.arms) == (
        fixture.campaign.id,
        1,
        len(undescribable.variant_index),
    )
    assert line.reasons == [REASON]


async def test_the_action_reads_the_same_and_says_it() -> None:
    fixture, described, undescribable = await _two_analyses()
    (evals, *_rest) = eval_catalogue().mount_all(standard_tools())

    outcome = await evals.call({"action": "analyses_undescribable"}, host=fixture.host, caller=CALLER)

    assert not outcome.is_error, outcome.text
    listing = UndescribableArmsListing.model_validate(outcome.structured)
    assert [line.analysis_id for line in listing.analyses] == [undescribable.id]
    assert f"{undescribable.id} (campaign {fixture.campaign.id}): 1 of" in outcome.text
    assert REASON in outcome.text and described.id not in outcome.text


async def test_a_scope_with_none_says_none() -> None:
    fixture = ops_fixture()

    listing = analyses_undescribable(fixture.host.eval_host, TOYHOST_SCOPE)

    assert (listing.analyses_read, listing.analyses) == (0, [])
