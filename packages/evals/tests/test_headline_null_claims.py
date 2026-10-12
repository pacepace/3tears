"""A headline or summary bullet that says arms are alike needs a finding under it that established a null.

The instance, from a production memo (2026-10): "Halving the token budget from 16384 to 12288 does not change this
research agent's latency, cost, or delivery", over a contrast that did not separate at five cases and whose own caveat said it "has not confirmed a null
so much as failed to separate two arms". A null is established by the bundle alone — a comparison reading
``equivalent``, or a guardrail reading ``held`` — and a ``not_separated`` contrast is unresolved.

The headline and each summary bullet name the findings they rest on. The generator matches each sentence against a
curated list of null phrasings, and refuses one when none of ITS findings cites a reading the bundle established
alike — a null another sentence's finding established licenses nothing here. The phrase tests pin both directions for every
pattern: the claim it catches, and the recommendation it must not. The generation tests drive
:func:`~threetears.evals.analysis.generator.generate_analysis` over the toy campaign with its narrow width declared the
control, its verdicts replaced the way the adoption tests replace them.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from packages.evals.tests.fixtures.toyhost.campaign import TOYHOST_NARROW, TOYHOST_WIDE, toyhost_campaign
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.toyhost_memo import (
    MODEL,
    PROMPT,
    PROMPT_ID,
    FixturedClient,
    FixturedCompletion,
    alias_at,
    cell_at,
    memo_payload,
)
from threetears.evals.analysis.bundle.assemble import assemble_context_bundle
from threetears.evals.analysis.bundle.schema import AnalysisContextBundle, ComparisonVerdict
from threetears.evals.analysis.cells import variant_of_cell_ref
from threetears.evals.analysis.errors import SoundnessRefusal
from threetears.evals.analysis.generator import generate_analysis
from threetears.evals.analysis.null_claims import NULL_CLAIM_PATTERNS, null_claims
from threetears.evals.kernel.authored import summary_bullets
from threetears.evals.kernel.campaign import EvalAnalysis
from threetears.evals.kernel.surface import GuardrailCell, GuardrailCheck, GuardrailReadings
from threetears.evals.schema.models import utc_now_iso

#: The #2083 headline, verbatim.
_HEADLINE_2083 = (
    "Halving the token budget from 16384 to 12288 does not change this research agent's latency, cost, or delivery."
)

# --- the phrases --------------------------------------------------------------------------------------------------

#: One null claim per pattern, with the phrase it is quoted by.
_CLAIMS = [
    ("Halving the budget does not change latency.", "does not change"),
    ("The bigger model doesn't measurably hurt delivery.", "doesn't measurably hurt"),
    ("Caching won't affect cost.", "won't affect"),
    ("The retry didn't make any real difference.", "didn't make any real difference"),
    ("Quality isn't affected by the shorter prompt.", "isn't affected"),
    ("The budget made no measurable difference to cost.", "no measurable difference"),
    ("Halving it had no effect on delivery.", "no effect"),
    ("The cheaper model is no worse at finishing the job.", "no worse"),
    ("Latency was essentially unchanged.", "was essentially unchanged"),
    ("Cost stayed flat across budgets.", "stayed flat"),
    ("Both settings cost about the same.", "about the same"),
    ("The small model is just as accurate.", "just as accurate"),
    ("Nothing changed when the budget was halved.", "nothing changed"),
    ("There is no clear trend across budgets.", "no clear trend"),
    ("On speed the two are a wash.", "are a wash"),
]

#: Sentences a sound analysis writes, several a near miss of a pattern: a recommendation, or an unresolved contrast
#: stated as unresolved.
_NOT_CLAIMS = [
    "Do not change the token budget until a larger run settles it.",
    "Keep the budget unchanged for now.",
    "I would not change the chunk width yet.",
    "Make no change to the pipeline before the confirming run.",
    "Whether halving the budget slows the search is too close to call on this evidence.",
    "Halving the budget is probably cheaper, but the gap is too close to call.",
    "Both arms ran on the same cases.",
    "Run twenty more cases per arm to settle it.",
    "The wider chunk is much slower on every document.",
]


@pytest.mark.parametrize(("sentence", "phrase"), _CLAIMS, ids=[phrase for _, phrase in _CLAIMS])
def test_each_null_phrasing_is_caught_and_quoted_as_written(sentence: str, phrase: str) -> None:
    assert null_claims(sentence) == [phrase]


def test_every_pattern_has_a_claim_it_catches() -> None:
    """A pattern no listed claim exercises is one whose behaviour nothing pins."""
    exercised = {name for name, pattern in NULL_CLAIM_PATTERNS for sentence, _ in _CLAIMS if pattern.search(sentence)}
    assert exercised == {name for name, _ in NULL_CLAIM_PATTERNS}


@pytest.mark.parametrize("sentence", _NOT_CLAIMS)
def test_a_recommendation_or_an_unresolved_contrast_is_not_a_null_claim(sentence: str) -> None:
    assert null_claims(sentence) == []


def test_the_2083_headline_is_a_null_claim() -> None:
    assert null_claims(_HEADLINE_2083) == ["does not change"]


def test_emphasis_and_a_curly_apostrophe_do_not_hide_a_claim() -> None:
    assert null_claims("Halving it **does not** change cost, and it doesn’t hurt delivery.") == [
        "does not change",
        "doesn't hurt",
    ]


def test_a_summary_splits_into_its_bullets_with_a_wrapped_line_kept_in_its_own() -> None:
    summary = "- Wider chunks are slower.\n- Cost is too close\n  to call.\n\n1. Run more cases."
    assert summary_bullets(summary) == ["Wider chunks are slower.", "Cost is too close to call.", "Run more cases."]


def test_a_summary_with_no_bullet_is_one_bullet() -> None:
    assert summary_bullets("Wider chunks are slower.\nNothing else was learned.") == [
        "Wider chunks are slower. Nothing else was learned."
    ]


# --- through the generator ----------------------------------------------------------------------------------------


class _MemoSequence(FixturedClient):
    """Hands back one prepared memo per call, in order: the first output, then its repair."""

    def __init__(self, *bodies: str) -> None:
        super().__init__(bodies[0])
        self.bodies = list(bodies)

    async def generate(
        self, *, system: str, user: str, response_format: Any = None, tools: Any = None
    ) -> FixturedCompletion:
        self.completion = FixturedCompletion(content=self.bodies[len(self.calls)])
        return await super().generate(system=system, user=user, response_format=response_format, tools=tools)


def _bundle(*, total_ms: ComparisonVerdict = "not_separated") -> AnalysisContextBundle:
    """The toy bundle, narrow width the control, every comparison ``not_separated`` but latency's, set to ``total_ms``."""
    profile = toyhost_profile()
    campaign, storage = toyhost_campaign(profile=profile)
    bundle = assemble_context_bundle(campaign, storage=storage, profile=profile)
    assert campaign.declared_design is not None
    design = campaign.declared_design.model_copy(
        update={"control": variant_of_cell_ref(cell_at(bundle, TOYHOST_NARROW))}
    )
    bundle = assemble_context_bundle(
        campaign.model_copy(update={"declared_design": design}), storage=storage, profile=profile
    )
    families = [
        family.model_copy(
            update={
                "comparisons": [
                    comparison.model_copy(
                        update={"verdict": total_ms if comparison.name == "total_ms" else "not_separated"}
                    )
                    for comparison in family.comparisons
                ]
            }
        )
        for family in bundle.multiple_comparisons.families
    ]
    assert any(c.name == "total_ms" for family in families for c in family.comparisons)
    return bundle.model_copy(
        update={"multiple_comparisons": bundle.multiple_comparisons.model_copy(update={"families": families})}
    )


def _memo(
    bundle: AnalysisContextBundle,
    *,
    headline: str = "Keep the narrow chunk width; the wide one is slower.",
    headline_rests_on: list[int] | None = None,
    summary: str = "- The wide width is slower.\n- Whether it is cheaper is too close to call.",
    summary_rests_on: list[list[int]] | None = None,
) -> str:
    """The toy memo with two findings — 0 citing latency at both widths, 1 citing cost — and its links replaced.

    By default the headline rests on finding 1 (cost) and every bullet on finding 0 (latency).
    """
    payload = memo_payload(bundle)
    latency = payload["findings"][0]
    cost = latency | {
        "title": "the wider chunk costs more per document",
        "evidence": [
            {"cell": alias_at(bundle, level), "measure_id": "production_replicating_cost", "reading": "measure"}
            for level in (TOYHOST_NARROW, TOYHOST_WIDE)
        ],
        "confidence": "medium",
    }
    payload["findings"] = [latency, cost]
    payload["headline"], payload["summary"] = headline, summary
    payload["headline_rests_on"] = [1] if headline_rests_on is None else headline_rests_on
    payload["summary_rests_on"] = (
        [[0] for _ in summary_bullets(summary)] if summary_rests_on is None else summary_rests_on
    )
    return json.dumps(payload)


async def _generate(bundle: AnalysisContextBundle, client: FixturedClient) -> EvalAnalysis:
    analysis, _insights = await generate_analysis(
        bundle,
        prompt=PROMPT,
        model=MODEL,
        client=client,
        prompt_id=PROMPT_ID,
        bundle_assembled_at=utc_now_iso(),
        profile=toyhost_profile(),
    )
    return analysis


async def _accepted(bundle: AnalysisContextBundle, memo: str) -> EvalAnalysis:
    client = FixturedClient(memo)
    analysis = await _generate(bundle, client)
    assert len(client.calls) == 1, "accepted first time"
    return analysis


class TestANullNoFindingUnderItEstablishedIsRefused:
    async def test_the_2083_headline_over_an_unresolved_contrast_is_refused_and_told_what_to_do(self):
        bundle = _bundle()
        with pytest.raises(SoundnessRefusal) as refused:
            await _generate(bundle, FixturedClient(_memo(bundle, headline=_HEADLINE_2083)))
        message = str(refused.value)
        assert 'the headline says "does not change" and rests on finding position(s) [1]' in message
        assert "none of those findings cites a reading the bundle established alike" in message
        assert "`equivalent` in `multiple_comparisons`, or a guardrail reading `held`" in message
        assert "is unresolved, not unchanged (rule 4)" in message
        assert "a reworded null is the same claim" in message
        assert "The bundle does establish" not in message, "it establishes nothing, so nothing is offered"

    async def test_a_summary_bullet_is_checked_against_its_own_findings_and_named_by_its_position(self):
        bundle = _bundle()
        summary = "- The wide width is slower.\n- Cost is too close to call.\n- Accuracy was unchanged between widths."
        with pytest.raises(SoundnessRefusal, match=r'summary bullet 3 says "was unchanged" and rests on finding'):
            await _generate(bundle, FixturedClient(_memo(bundle, summary=summary)))

    async def test_every_null_in_the_headline_and_summary_is_named_in_one_refusal(self):
        bundle = _bundle()
        memo = _memo(bundle, headline=_HEADLINE_2083, summary="- Cost made no difference.\n- Run more cases.")
        with pytest.raises(SoundnessRefusal) as refused:
            await _generate(bundle, FixturedClient(memo))
        message = str(refused.value)
        assert 'the headline says "does not change"' in message
        assert 'summary bullet 1 says "no difference"' in message

    async def test_a_null_another_sentences_finding_established_does_not_license_this_one(self):
        """Latency is equivalent and finding 0 cites it; the headline rests on finding 1 (cost), so it is refused."""
        bundle = _bundle(total_ms="equivalent")
        wide = alias_at(bundle, TOYHOST_WIDE)
        client = _MemoSequence(
            _memo(bundle, headline=_HEADLINE_2083, headline_rests_on=[1]),
            _memo(bundle, headline=_HEADLINE_2083, headline_rests_on=[0]),
        )

        analysis = await _generate(bundle, client)

        refusal = analysis.generation.repaired_refusal
        assert refusal is not None
        assert f"The bundle does establish total_ms at cell {wide} (equivalent to the control; cited by finding " in (
            refusal
        ), "named in the writer's own alias"
        assert "position(s) [0])" in refusal and "rest it on a finding that cites that reading" in refusal
        assert len(client.calls) == 2 and refusal in client.calls[1]["user"]
        assert analysis.document.headline_rests_on == [0]

    async def test_the_repair_round_states_it_as_unresolved_and_the_analysis_is_stored(self):
        bundle = _bundle()
        honest = "Too close to call: keep the current budget and run more cases."
        client = _MemoSequence(_memo(bundle, headline=_HEADLINE_2083), _memo(bundle, headline=honest))

        analysis = await _generate(bundle, client)

        assert len(client.calls) == 2
        assert analysis.generation.repair_attempts == 1
        assert analysis.document.headline == honest


class TestASoundHeadlineIsAccepted:
    async def test_a_null_resting_on_a_finding_that_cites_an_established_equivalence_passes(self):
        bundle = _bundle(total_ms="equivalent")
        await _accepted(
            bundle, _memo(bundle, headline="The wider chunk is just as fast; pick on cost.", headline_rests_on=[0])
        )

    async def test_a_null_resting_on_a_finding_that_cites_a_held_guardrail_passes(self):
        bundle = _bundle()
        comparison = next(
            c for family in bundle.multiple_comparisons.families for c in family.comparisons if c.name == "total_ms"
        )
        held = GuardrailCheck(
            reading="measure",
            name="total_ms",
            higher_is_better=False,
            control=GuardrailCell(
                variant_key=comparison.control.variant_key,
                apparatus_class_id=comparison.control.apparatus_class_id,
                n_cases=4,
                mean=920.0,
            ),
            contrast=GuardrailCell(
                variant_key=comparison.contrast.variant_key,
                apparatus_class_id=comparison.contrast.apparatus_class_id,
                n_cases=4,
                mean=930.0,
            ),
            test="paired",
            delta=10.0,
            interval=(-20.0, 40.0),
            interval_basis="t",
            margin=50.0,
            margin_declared=True,
            decision="held",
        )
        bundle = bundle.model_copy(update={"guardrails": GuardrailReadings(measures=["total_ms"], checks=[held])})
        await _accepted(
            bundle,
            _memo(bundle, headline="The wider chunk does not hurt speed; pick on cost.", headline_rests_on=[0]),
        )

    async def test_an_unresolved_contrast_stated_as_unresolved_passes(self):
        bundle = _bundle()
        await _accepted(
            bundle,
            _memo(
                bundle,
                headline="Halving the budget is probably cheaper, but too close to call.",
                summary="- Whether the wide width is cheaper is too close to call.\n- Do not change the width yet.",
            ),
        )

    async def test_a_recommendation_to_keep_a_setting_unchanged_passes(self):
        bundle = _bundle()
        await _accepted(bundle, _memo(bundle, headline="Keep the budget unchanged until a larger run settles it."))


class TestTheHeadlineAndEveryBulletNameTheirFindings:
    """The links are what the standing and the null check read, so a missing or stray one is refused and repaired."""

    async def test_an_empty_headline_link_over_findings_is_refused(self):
        bundle = _bundle()
        with pytest.raises(SoundnessRefusal, match=r"`headline_rests_on` is empty; name the finding positions"):
            await _generate(bundle, FixturedClient(_memo(bundle, headline_rests_on=[])))

    async def test_a_link_count_that_is_not_one_per_bullet_is_refused_with_the_bullets_read(self):
        bundle = _bundle()
        with pytest.raises(SoundnessRefusal) as refused:
            await _generate(bundle, FixturedClient(_memo(bundle, summary_rests_on=[[0]])))
        message = str(refused.value)
        assert "`summary_rests_on` has 1 entries and the summary has 2 bullet(s)" in message
        assert "0: 'The wide width is slower.'" in message

    async def test_an_empty_bullet_link_over_findings_is_refused(self):
        bundle = _bundle()
        with pytest.raises(SoundnessRefusal, match=r"`summary_rests_on\[1\]` is empty"):
            await _generate(bundle, FixturedClient(_memo(bundle, summary_rests_on=[[0], []])))

    async def test_a_position_past_the_findings_is_refused_naming_the_sentence(self):
        bundle = _bundle()
        with pytest.raises(
            SoundnessRefusal, match=r"`headline_rests_on` names finding position\(s\) \[5\], and there are 2"
        ):
            await _generate(bundle, FixturedClient(_memo(bundle, headline_rests_on=[5])))

    async def test_the_repair_round_supplies_the_link_and_the_analysis_is_stored(self):
        bundle = _bundle()
        client = _MemoSequence(_memo(bundle, headline_rests_on=[]), _memo(bundle, headline_rests_on=[0, 1]))
        analysis = await _generate(bundle, client)
        assert len(client.calls) == 2 and analysis.document.headline_rests_on == [0, 1]
