"""The campaign bundle: one naming across aspects, and its round trip through its JSON and its frame.

A campaign's report reads every aspect at once, so a cell, a trial and an arm must each be named one way across them:
a cell alias (``c1`` … ``cN``) and a trial id (``T1`` … ``TN``) are unique across the campaign and resolve back to
the aspect they came from, since one arm under one rig is the same ``cell_ref`` in every aspect.
"""

from __future__ import annotations

import pytest

from packages.evals.tests.aspect_support import with_control
from packages.evals.tests.campaign_bundle_support import campaign_bundle
from packages.evals.tests.fixtures.toyhost.campaign import TOYHOST_NARROW, toyhost_campaign
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.toyhost_cells import cell_at
from threetears.evals.analysis import AnalysisContextBundle, CampaignBundle, assemble_context_bundle
from threetears.evals.analysis.cells import variant_of_cell_ref
from threetears.evals.analysis.errors import UnresolvableReference
from threetears.evals.analysis.references import cell_aliases


def _aspect() -> AnalysisContextBundle:
    profile = toyhost_profile()
    campaign, storage = toyhost_campaign(profile=profile)
    bundle = assemble_context_bundle(campaign, storage=storage, profile=profile)
    campaign = with_control(campaign, variant_of_cell_ref(cell_at(bundle, TOYHOST_NARROW)))
    return assemble_context_bundle(campaign, storage=storage, profile=profile)


def _two_aspects() -> CampaignBundle:
    first = _aspect()
    second = first.model_copy(update={"aspect": "second"})
    return campaign_bundle(first, second)


class TestOneNamingAcrossAspects:
    def test_a_campaign_of_one_aspect_names_cells_and_trials_as_its_bundle_does(self) -> None:
        aspect = _aspect()
        bundle = campaign_bundle(aspect)
        assert {alias: address.cell_ref for alias, address in bundle.cell_aliases().items()} == cell_aliases(
            aspect.cell_measures
        )
        assert [where for where in bundle.trial_ids().values()] == [
            (aspect.aspect, trial.trial_id) for trial in aspect.trials
        ]
        assert list(bundle.trial_ids()) == [trial.trial_id for trial in aspect.trials]

    def test_cell_aliases_are_unique_across_aspects_and_resolve_to_their_own_aspect(self) -> None:
        bundle = _two_aspects()
        aliases = bundle.cell_aliases()
        per_aspect = len(bundle.aspects[0].cell_measures)
        assert list(aliases) == [f"c{n}" for n in range(1, 2 * per_aspect + 1)]
        # The same arm under the same rig is one cell_ref in both aspects, so only the alias tells them apart.
        first, second = list(aliases.values())[:per_aspect], list(aliases.values())[per_aspect:]
        assert [address.cell_ref for address in first] == [address.cell_ref for address in second]
        assert {address.aspect for address in first} == {bundle.labels[0]}
        assert {address.aspect for address in second} == {"second"}
        assert bundle.cell(f"c{per_aspect + 1}") == second[0]

    def test_an_alias_no_cell_carries_is_refused_naming_the_cells_there_are(self) -> None:
        bundle = _two_aspects()
        with pytest.raises(UnresolvableReference, match="the cells are: c1, c2"):
            bundle.cell("c99")

    def test_trial_ids_are_unique_across_aspects_and_resolve_to_their_own_aspect(self) -> None:
        bundle = _two_aspects()
        count = len(bundle.aspects[0].trials)
        ids = bundle.trial_ids()
        assert list(ids) == [f"T{n}" for n in range(1, 2 * count + 1)]
        assert bundle.trial(f"T{count + 1}") == ("second", bundle.aspects[1].trials[0].trial_id)
        assert bundle.trial("T999") is None


class TestTheBundleRoundTrips:
    def test_the_whole_bundle_and_its_frame_read_back_to_the_same_fingerprint(self) -> None:
        bundle = _two_aspects()
        assert CampaignBundle.from_json(bundle.to_json()).fingerprint() == bundle.fingerprint()
        assert CampaignBundle.from_frame(bundle.frame(), bundle.aspects).fingerprint() == bundle.fingerprint()

    def test_a_frame_over_other_aspects_is_refused(self) -> None:
        bundle = _two_aspects()
        with pytest.raises(ValueError, match="its aspects hold"):
            CampaignBundle.from_frame(bundle.frame(), bundle.aspects[:1])

    def test_one_aspect_s_bundle_frozen_as_a_campaign_s_is_refused_with_a_remedy(self) -> None:
        aspect = _aspect()
        with pytest.raises(ValueError, match="not a campaign bundle of this build's shape") as refused:
            CampaignBundle.from_dict(aspect.to_dict(), remedy="freeze it again")
        assert getattr(refused.value, "remedy", None) == "freeze it again"
