"""What an analysis keeps of each aspect's bundle, and a trial the campaign holds found by its campaign id.

What an analysis freezes of each aspect's bundle (its bundle, its fingerprint, the campaign frame) is code's
(:func:`~threetears.evals.analysis.doc_analysis.aspect_analysis`), held over the toy host's campaign, which declares a
question and the narrow chunk width as its control.
"""

from __future__ import annotations

from packages.evals.tests.aspect_support import with_control
from packages.evals.tests.campaign_bundle_support import campaign_bundle
from packages.evals.tests.fixtures.toyhost.campaign import TOYHOST_NARROW, toyhost_campaign
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.toyhost_cells import cell_at
from packages.evals.tests.writer_host_support import doc_analysis_over
from threetears.evals.analysis import AnalysisContextBundle, CampaignBundle, assemble_context_bundle
from threetears.evals.analysis.cells import variant_of_cell_ref


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
    """The toy campaign's bundle of one aspect."""
    return campaign_bundle(_aspect())


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
