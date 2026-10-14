"""The memo writer authors findings and recommendations, each pinned, and an exploratory campaign adopts nothing.

The report's first screen is code's, so what the generator still checks of a memo is structure: every recommendation
names the findings it rests on, a finding names only trials the bundle holds, and in a campaign that declared no
question before its first run no recommendation is ``adopted`` — keeping the control included, which is what printed
"Keep the current model — adopted" over the G2 campaign. Each refusal buys the one repair round like any other.

Driven through :func:`~threetears.evals.analysis.generator.generate_analysis` over the toy host's campaign, which
declares a question; the exploratory case is the same bundle with no question declared before its run.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from packages.evals.tests.fixtures.toyhost.campaign import TOYHOST_NARROW, toyhost_campaign
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.toyhost_memo import MODEL, PROMPT, PROMPT_ID, FixturedClient, alias_at, cell_at, memo_payload
from threetears.evals.analysis import AnalysisContextBundle, assemble_context_bundle, generate_analysis
from threetears.evals.analysis.cells import variant_of_cell_ref
from threetears.evals.analysis.errors import SoundnessRefusal
from threetears.evals.kernel.campaign import EvalAnalysis
from threetears.evals.schema.models import utc_now_iso


def _bundle() -> AnalysisContextBundle:
    """The toy campaign's bundle, its narrow width declared the control."""
    profile = toyhost_profile()
    campaign, storage = toyhost_campaign(profile=profile)
    bundle = assemble_context_bundle(campaign, storage=storage, profile=profile)
    assert campaign.declared_design is not None
    control = variant_of_cell_ref(cell_at(bundle, TOYHOST_NARROW))
    # Validated, not copied: a control is stamped with the identity version it was addressed under.
    design = type(campaign.declared_design).model_validate(
        {**campaign.declared_design.model_dump(), "control": control}
    )
    campaign = campaign.model_copy(update={"declared_design": design})
    return assemble_context_bundle(campaign, storage=storage, profile=profile)


def _exploratory(bundle: AnalysisContextBundle) -> AnalysisContextBundle:
    """The bundle as a campaign that declared no question before its first run reads."""
    scope = bundle.reading_scope.model_copy(update={"questions_declared": False})
    return bundle.model_copy(update={"reading_scope": scope})


async def _generate(bundle: AnalysisContextBundle, payload: dict[str, Any]) -> EvalAnalysis:
    analysis, _ = await generate_analysis(
        bundle,
        prompt=PROMPT,
        model=MODEL,
        client=FixturedClient(json.dumps(payload)),
        prompt_id=PROMPT_ID,
        bundle_assembled_at=utc_now_iso(),
        profile=toyhost_profile(),
    )
    return analysis


def _keep_the_control(bundle: AnalysisContextBundle, disposition: str) -> dict[str, Any]:
    payload = memo_payload(bundle)
    payload["decisions"][0].update(
        {
            "proposal": "Keep the current chunk width.",
            "disposition": disposition,
            "cells": [alias_at(bundle, TOYHOST_NARROW)],
        }
    )
    if disposition == "adopted":
        payload["decisions"][0]["revisit_when"] = ""
    return payload


async def test_an_adopted_recommendation_in_an_exploratory_campaign_is_refused_even_keeping_the_control() -> None:
    bundle = _exploratory(_bundle())
    payload = _keep_the_control(bundle, "adopted")
    with pytest.raises(SoundnessRefusal) as refused:
        await _generate(bundle, payload)
    message = str(refused.value)
    assert "decisions [0] are `adopted`" in message
    assert "declared no question before its first run, so it is exploratory" in message
    assert "keeping the current arm included" in message


async def test_a_deferred_recommendation_in_an_exploratory_campaign_is_kept() -> None:
    bundle = _exploratory(_bundle())
    payload = _keep_the_control(bundle, "deferred")
    analysis = await _generate(bundle, payload)
    assert analysis.document.decisions[0].disposition == "deferred"


async def test_with_a_question_declared_before_the_run_the_control_may_be_kept() -> None:
    bundle = _bundle()
    assert bundle.reading_scope.questions_declared
    analysis = await _generate(bundle, _keep_the_control(bundle, "adopted"))
    assert analysis.document.decisions[0].disposition == "adopted"


async def test_a_recommendation_resting_on_no_finding_is_refused() -> None:
    bundle = _bundle()
    payload = memo_payload(bundle)
    payload["decisions"][0]["rests_on"] = []
    with pytest.raises(SoundnessRefusal, match=r"every recommendation names the findings it rests on.*decisions\[0\]"):
        await _generate(bundle, payload)


async def test_a_recommendation_resting_on_a_finding_that_is_not_there_is_refused() -> None:
    bundle = _bundle()
    payload = memo_payload(bundle)
    payload["decisions"][0]["rests_on"] = [3]
    with pytest.raises(SoundnessRefusal, match=r"decisions\[0\]\.rests_on names finding position\(s\) \[3\]"):
        await _generate(bundle, payload)


async def test_a_finding_naming_a_trial_the_bundle_does_not_hold_is_refused() -> None:
    bundle = _bundle()
    payload = memo_payload(bundle)
    payload["findings"][0]["trials"] = ["T999"]
    with pytest.raises(SoundnessRefusal, match=r"findings\[0\]\.trials names 'T999'"):
        await _generate(bundle, payload)


async def test_a_finding_naming_trials_the_bundle_holds_keeps_them_and_the_bundle_is_stored_with_it() -> None:
    bundle = _bundle()
    payload = memo_payload(bundle)
    payload["findings"][0]["trials"] = [bundle.trials[0].trial_id]
    analysis = await _generate(bundle, payload)
    assert analysis.document.findings[0].trials == [bundle.trials[0].trial_id]
    assert AnalysisContextBundle.from_dict(dict(analysis.bundle)).fingerprint() == bundle.fingerprint()


async def test_the_memo_carries_no_headline_summary_or_chart() -> None:
    bundle = _bundle()
    payload = memo_payload(bundle)
    payload["headline"] = "keep the narrow width"
    with pytest.raises(SoundnessRefusal, match="headline"):
        await _generate(bundle, payload)
