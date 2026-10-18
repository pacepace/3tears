"""The campaign bundle: one naming across aspects, and each aspect's part of the writer's message held to a budget.

The writer of a campaign's one memo reads every aspect at once, so a cell, a trial and an arm must each be named one
way across them: a cell alias (``c1`` … ``cN``) and a trial id (``T1`` … ``TN``) are unique across the campaign and
resolve back to the aspect they came from, since one arm under one rig is the same ``cell_ref`` in every aspect.

Each aspect's part of the message is held to :data:`CAMPAIGN_MESSAGE_CHARS_PER_ASPECT` characters of compact JSON
(the system prompt is budgeted apart). Over it, code trims in a fixed order and says what it left out: the flagged
trials' excerpts first, then the flagged trials, then the contrasts least separated first, then the cells' readings
last. Code's own readings and the floors are never trimmed.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from packages.evals.tests.aspect_support import with_control
from packages.evals.tests.campaign_bundle_support import campaign_bundle
from packages.evals.tests.fixtures.toyhost.campaign import TOYHOST_NARROW, toyhost_campaign
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.toyhost_memo import MODEL, PROMPT, PROMPT_ID, FixturedClient, cell_at, memo_payload
from threetears.evals.analysis import (
    CAMPAIGN_MESSAGE_CHARS_PER_ASPECT,
    AnalysisContextBundle,
    CampaignBundle,
    assemble_context_bundle,
    build_campaign_user_message,
    generate_analysis,
)
from threetears.evals.analysis.campaign_bundle import aspect_view
from threetears.evals.analysis.cells import variant_of_cell_ref
from threetears.evals.analysis.errors import UnresolvableReference
from threetears.evals.analysis.references import cell_aliases
from threetears.evals.schema.models import utc_now_iso


def _aspect() -> AnalysisContextBundle:
    profile = toyhost_profile()
    campaign, storage = toyhost_campaign(profile=profile)
    bundle = assemble_context_bundle(campaign, storage=storage, profile=profile)
    campaign = with_control(campaign, variant_of_cell_ref(cell_at(bundle, TOYHOST_NARROW)))
    return assemble_context_bundle(campaign, storage=storage, profile=profile)


def _flagged(aspect: AnalysisContextBundle, excerpt: int = 400) -> AnalysisContextBundle:
    """``aspect`` with every trial failed and carrying an excerpt, so the trim has excerpts and trials to drop."""
    trials = [
        trial.model_copy(update={"state": "failed", "cause": "check", "excerpt": f"{trial.trial_id} " + "x" * excerpt})
        for trial in aspect.trials
    ]
    return aspect.model_copy(update={"trials": trials})


def _two_aspects() -> CampaignBundle:
    first = _aspect()
    second = first.model_copy(update={"aspect": "second"})
    return campaign_bundle(first, second)


#: Room for the note an aspect's part carries when it was trimmed, which the budget counts.
_NOTE = 600


def _size(view: dict[str, Any]) -> int:
    return len(json.dumps(view, separators=(",", ":"), ensure_ascii=False))


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

    def test_the_writer_s_view_names_each_aspect_s_cells_by_their_campaign_aliases(self) -> None:
        bundle = _two_aspects()
        view = json.loads(build_campaign_user_message(bundle).split("\n", 1)[1])
        named = [[cell["cell"] for cell in aspect["cells"]] for aspect in view["aspects"]]
        assert named == [["c1", "c2"], ["c3", "c4"]]
        assert "variant_key" not in build_campaign_user_message(bundle)

    async def test_a_memo_citing_the_second_aspect_resolves_its_reading_there(self) -> None:
        bundle = _two_aspects()
        payload = memo_payload(campaign_bundle(bundle.aspects[0]))
        for row in payload["findings"][0]["evidence"]:
            row["cell"] = f"c{int(row['cell'][1:]) + 2}"
        payload["aspect_summaries"].append({**payload["aspect_summaries"][0], "aspect": "second"})
        analysis, _ = await generate_analysis(
            bundle,
            prompt=PROMPT,
            model=MODEL,
            client=FixturedClient(json.dumps(payload)),
            prompt_id=PROMPT_ID,
            bundle_assembled_at=utc_now_iso(),
            profile=toyhost_profile(),
        )
        assert [row.aspect for row in analysis.resolutions[0].evidence] == ["second", "second"]
        assert [aspect.label for aspect in analysis.aspects] == bundle.labels
        assert analysis.generation.bundle_fingerprint == bundle.fingerprint()


class TestEachAspectIsHeldToItsBudget:
    def test_the_budget_is_per_aspect(self) -> None:
        assert CAMPAIGN_MESSAGE_CHARS_PER_ASPECT == 24_500

    def test_an_aspect_within_budget_is_sent_whole(self) -> None:
        bundle = campaign_bundle(_aspect())
        view = aspect_view(bundle, 0)
        assert _size(view) <= CAMPAIGN_MESSAGE_CHARS_PER_ASPECT
        assert "trimmed" not in view
        assert len(view["trials"]) == sum(trial.state != "passed" for trial in bundle.aspects[0].trials)

    def test_a_trimmed_aspect_fits_its_budget_with_the_note_saying_so(self) -> None:
        bundle = campaign_bundle(_flagged(_aspect()))
        view = aspect_view(bundle, 0)
        assert _size(view) <= CAMPAIGN_MESSAGE_CHARS_PER_ASPECT
        assert "The page shows all of them." in view["trimmed"]

    def test_the_excerpts_go_first(self) -> None:
        bundle = campaign_bundle(_flagged(_aspect()))
        whole = aspect_view(bundle, 0, budget=10**9)
        budget = _size(whole) - 1
        view = aspect_view(bundle, 0, budget=budget)
        assert _size(view) <= budget
        assert len(view["trials"]) == len(whole["trials"])
        assert view["contrasts"] == whole["contrasts"] and view["cells"] == whole["cells"]
        # Excerpts went, the last trial's first, and nothing else.
        excerpts = [bool(trial["excerpt"]) for trial in view["trials"]]
        kept = excerpts.index(False)
        assert not excerpts[-1] and excerpts == [True] * kept + [False] * (len(excerpts) - kept)
        said = view["trimmed"].split("(")[0]
        assert "excerpts" in said and "trials," not in said and "contrasts" not in said
        assert "The page shows all of them" in view["trimmed"]

    def test_then_the_trials_then_the_contrasts_then_the_cells_readings(self) -> None:
        bundle = campaign_bundle(_flagged(_aspect()))
        whole = aspect_view(bundle, 0, budget=10**9)
        no_trials = {**whole, "trials": []}
        no_contrasts = {**no_trials, "contrasts": []}
        # Room for everything but the trials: every excerpt and trial goes, and nothing else.
        view = aspect_view(bundle, 0, budget=_size(no_trials) + _NOTE)
        assert view["trials"] == [] and view["contrasts"] == whole["contrasts"] and view["cells"] == whole["cells"]
        # Room for less than that: the contrasts go too, least separated first, the cells' readings kept.
        view = aspect_view(bundle, 0, budget=_size(no_contrasts) + _NOTE)
        kept = view["contrasts"]
        assert view["trials"] == [] and len(kept) < len(whole["contrasts"])
        assert kept == whole["contrasts"][: len(kept)] and view["cells"] == whole["cells"]
        assert [row["verdict"] for row in whole["contrasts"]][: len(kept)] == [row["verdict"] for row in kept]
        # Less still: the cells' readings go last, the last cell's first.
        view = aspect_view(bundle, 0, budget=_size(no_contrasts) - 50)
        assert view["contrasts"] == [] and view["trials"] == []
        assert view["cells"][0] == whole["cells"][0]

        def readings(cell: dict[str, Any]) -> int:
            return len(cell["measures"]) + len(cell["judged"])

        assert readings(view["cells"][-1]) < readings(whole["cells"][-1])
        assert "cell readings" in view["trimmed"]

    def test_code_s_readings_and_the_scoreboard_are_never_trimmed(self) -> None:
        bundle = campaign_bundle(_flagged(_aspect()))
        whole = aspect_view(bundle, 0, budget=10**9)
        view = aspect_view(bundle, 0, budget=10)
        assert view["readings"] == whole["readings"] and view["scoreboard"] == whole["scoreboard"]
        assert view["disclosures"] == whole["disclosures"]
        assert "It is still over budget: what is left is never trimmed." in view["trimmed"]

    def test_the_message_holds_each_aspect_within_its_budget(self) -> None:
        bundle = _two_aspects()
        view = json.loads(build_campaign_user_message(bundle, budget=4_000).split("\n", 1)[1])
        assert [_size(aspect) <= 4_000 or "still over budget" in aspect["trimmed"] for aspect in view["aspects"]] == [
            True,
            True,
        ]


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
