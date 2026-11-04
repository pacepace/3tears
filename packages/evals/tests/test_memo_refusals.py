"""What survives of a stored memo's structure checks, and what an analysis keeps of each aspect's bundle.

The memo writer that checked a memo against the campaign before storing it is retired; what still holds a memo is the
stored models: :class:`~threetears.evals.kernel.authored.AuthoredCampaignAnalysis` refuses a field off its contract
or one it requires left out, and :class:`~threetears.evals.kernel.campaign.EvalAnalysis` refuses a recommendation
resting on a finding that is not there and an aspect summarised twice or not in the analysis. Each is driven over the
memo the memo writer stored over the toy corpus campaign (``fixtures/retired_writers/memo_analysis.json``).

What an analysis freezes of each aspect's bundle (its bundle, its fingerprint, the campaign frame) is code's
(:func:`~threetears.evals.analysis.doc_analysis.aspect_analysis`), held over the toy host's campaign, which declares a
question and the narrow chunk width as its control.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Final

import pytest
from pydantic import ValidationError

from packages.evals.tests.aspect_support import with_control
from packages.evals.tests.campaign_bundle_support import campaign_bundle
from packages.evals.tests.fixtures.toyhost.campaign import TOYHOST_NARROW, toyhost_campaign
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.toyhost_cells import cell_at
from packages.evals.tests.writer_host_support import doc_analysis_over
from threetears.evals.analysis import AnalysisContextBundle, CampaignBundle, assemble_context_bundle
from threetears.evals.analysis.cells import variant_of_cell_ref
from threetears.evals.kernel.authored import AuthoredCampaignAnalysis
from threetears.evals.kernel.campaign import EvalAnalysis

#: The analysis the memo writer stored over the toy corpus campaign, as its record holds it.
_STORED_MEMO: Final = Path(__file__).parent / "fixtures" / "retired_writers" / "memo_analysis.json"


def _stored_memo_analysis() -> dict[str, Any]:
    """The stored memo analysis, as a dict a test may change before validating it."""
    record: dict[str, Any] = json.loads(_STORED_MEMO.read_text(encoding="utf-8"))
    analysis: dict[str, Any] = copy.deepcopy(record["analysis"])
    return analysis


def _aspect() -> AnalysisContextBundle:
    """The toy campaign's bundle, its narrow width declared the control."""
    profile = toyhost_profile()
    campaign, storage = toyhost_campaign(profile=profile)
    bundle = assemble_context_bundle(campaign, storage=storage, profile=profile)
    control = variant_of_cell_ref(cell_at(bundle, TOYHOST_NARROW))
    # Validated, not copied: a control arm is stamped with the identity version it was addressed under.
    campaign = with_control(campaign, control)
    return assemble_context_bundle(campaign, storage=storage, profile=profile)


def _bundle() -> CampaignBundle:
    """The toy campaign as its writer reads it."""
    return campaign_bundle(_aspect())


class TestTheStoredMemoIsHeldToItsContract:
    """A field off the memo's contract, or one it requires left out, does not validate."""

    def test_the_memo_carries_no_headline_summary_or_chart(self) -> None:
        document = _stored_memo_analysis()["document"]
        assert AuthoredCampaignAnalysis.model_validate(document).summary, "the stored memo validates as stored"
        with pytest.raises(ValidationError, match="headline"):
            AuthoredCampaignAnalysis.model_validate({**document, "headline": "keep the narrow width"})

    def test_a_memo_with_no_summary_is_refused_as_off_the_contract(self) -> None:
        document = _stored_memo_analysis()["document"]
        del document["summary"]
        with pytest.raises(ValidationError, match="summary"):
            AuthoredCampaignAnalysis.model_validate(document)


class TestTheStoredAnalysisHoldsItsPositionsAndAspects:
    """The stored analysis refuses a recommendation resting on no finding there, and an aspect summarised wrongly."""

    def test_a_recommendation_resting_on_a_finding_that_is_not_there_is_refused(self) -> None:
        analysis = _stored_memo_analysis()
        analysis["document"]["recommendation"][0]["rests_on"] = [3]
        with pytest.raises(ValidationError, match=r"recommendation\.rests_on names finding position\(s\) \[3\]"):
            EvalAnalysis.model_validate(analysis)

    def test_a_summary_of_an_aspect_the_campaign_lacks_is_refused(self) -> None:
        analysis = _stored_memo_analysis()
        summaries = analysis["document"]["aspect_summaries"]
        summaries.append({**summaries[0], "aspect": "routing"})
        with pytest.raises(ValidationError, match=r"aspect_summaries name \['routing'\]"):
            EvalAnalysis.model_validate(analysis)

    def test_an_aspect_summarised_twice_is_refused(self) -> None:
        analysis = _stored_memo_analysis()
        summaries = analysis["document"]["aspect_summaries"]
        summaries.append(summaries[0])
        with pytest.raises(ValidationError, match="more than once"):
            EvalAnalysis.model_validate(analysis)


def test_a_trial_the_campaign_holds_is_found_and_each_aspect_s_bundle_is_stored_with_the_analysis() -> None:
    bundle = _bundle()
    analysis, _ = doc_analysis_over(bundle)
    assert bundle.trial("T1") == (bundle.labels[0], bundle.aspects[0].trials[0].trial_id)
    (aspect,) = analysis.aspects
    assert AnalysisContextBundle.from_dict(dict(aspect.bundle)).fingerprint() == bundle.aspects[0].fingerprint()
    assert aspect.bundle_fingerprint == bundle.aspects[0].fingerprint()
    assert (
        CampaignBundle.from_frame(analysis.campaign_bundle, [bundle.aspects[0]]).fingerprint() == bundle.fingerprint()
    )
    assert analysis.generation.bundle_fingerprint == bundle.fingerprint()
