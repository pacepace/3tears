"""A host's registered bar may stand on a judged dimension, resolved per campaign as a campaign's own bar is.

No registry enumerates judged dimensions (templates mint them), so registration holds a judged bar to what it can
know without a template: a dimension's name, no measure's, clearing upward. Each campaign of the behavior resolves
the name against its own template through ``resolve_bar_name`` and is refused in the words a campaign's own bar is
refused in when the template does not declare it; the bundle adjudicates it with the judges' evidence tier, and the
ratchet holds a campaign's bar on the same dimension to it.

Mutations that turn this file red: refusing every non-registry name again; admitting a judged bar whose name a
measure shares, which resolution would read as the measure; skipping the per-template resolution, so a standing
bar on a dimension the template lacks passes authoring and is adjudicated as carried by nobody; resolving it for
another behavior's campaigns.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from threetears.evals.analysis import assemble_context_bundle
from threetears.evals.kernel.declaration import (
    BarOverride,
    CampaignDesign,
    DesignRefused,
    refuse_an_undeclarable_design,
)
from threetears.evals.kernel.host.bars import Bar, BarRegistrationError, BarRegistry, load_bar_registry
from threetears.evals.kernel.host.profile import HostProfile
from threetears.evals.schema import EvalTemplate
from packages.evals.tests.fixtures.toyhost.campaign import toyhost_campaign
from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_JUDGED_DIMENSION
from packages.evals.tests.fixtures.toyhost.judge import FAITHFULNESS_DIM, toyhost_judged_template
from packages.evals.tests.fixtures.toyhost.profile import TOYHOST_BARS, toyhost_profile
from packages.evals.tests.fixtures.toyhost.run import toyhost_template

BEHAVIOR = "extract_invoice_fields"


def _judged(
    measure: str = FAITHFULNESS_DIM, *, threshold: float = 3.0, behavior: str = BEHAVIOR, higher_is_better: bool = True
) -> Bar:
    return Bar(
        behavior=behavior,
        measure=measure,
        threshold=threshold,
        higher_is_better=higher_is_better,
        rationale="a faithfulness below acceptable sends every invoice to review",
        judged=True,
    )


def _profile(*bars: Bar) -> HostProfile:
    return replace(toyhost_profile(), bars=BarRegistry((*TOYHOST_BARS, *bars)))


# =============================================================================
# Registration
# =============================================================================


@pytest.mark.parametrize("dimension", [FAITHFULNESS_DIM, "__outcome__"])
def test_a_judged_bar_on_a_dimension_s_name_registers(dimension: str) -> None:
    (bar,) = [bar for bar in _profile(_judged(dimension)).bars.bars if bar.judged]
    assert bar.measure == dimension


def test_the_same_name_without_judged_is_still_refused_as_no_measure_this_host_declares() -> None:
    with pytest.raises(BarRegistrationError, match="names a measure this host does not declare"):
        _profile(replace(_judged(), judged=False))


@pytest.mark.parametrize(
    ("bar", "says"),
    [
        (_judged("faithfulness"), "no judged dimension can carry its name"),
        (_judged("field_accuracy", behavior="another_behavior"), "this host declares a measure by that name"),
        (_judged(higher_is_better=False), "which the engine declares higher-is-better"),
    ],
    ids=["not namespaced", "a measure's name", "lower is better"],
)
def test_a_judged_bar_no_judged_dimension_could_be_read_by_is_refused_at_registration(bar: Bar, says: str) -> None:
    with pytest.raises(BarRegistrationError, match=says):
        _profile(bar)


def test_a_declaration_kept_as_data_registers_a_judged_bar() -> None:
    registry = load_bar_registry(
        {
            "bars": [
                {
                    "behavior": BEHAVIOR,
                    "measure": FAITHFULNESS_DIM,
                    "threshold": 3.5,
                    "higher_is_better": True,
                    "judged": True,
                    "rationale": "only a faithful extraction is usable",
                }
            ]
        }
    )
    assert registry.bars == (replace(_judged(threshold=3.5), rationale="only a faithful extraction is usable"),)


# =============================================================================
# Each campaign resolves it against its template
# =============================================================================


def _gate(
    template: EvalTemplate | None, *declared: BarOverride, profile: HostProfile, behavior: str = BEHAVIOR
) -> None:
    campaign, _storage = toyhost_campaign()
    assert campaign.declared_design is not None
    design = campaign.declared_design.model_copy(update={"bars": list(declared)})
    refuse_an_undeclarable_design(design, behavior=behavior, template=template, profile=profile)


def test_a_template_declaring_the_dimension_admits_the_campaign() -> None:
    _gate(toyhost_judged_template(), profile=_profile(_judged()))


@pytest.mark.parametrize("template", [toyhost_template(), None], ids=["template without it", "no template"])
def test_a_template_lacking_the_dimension_is_refused_in_the_words_a_campaign_bar_is(
    template: EvalTemplate | None,
) -> None:
    carried = "a bar must name something this campaign's results will carry"
    with pytest.raises(DesignRefused, match=carried) as refused:
        _gate(template, profile=_profile(_judged()))
    assert f"registered bar at 3.0 names '{FAITHFULNESS_DIM}'" in str(refused.value)
    assert "neither a judged dimension nor a goal-state check here" in str(refused.value)


def test_another_behavior_s_campaign_is_not_held_to_it() -> None:
    _gate(toyhost_template(), profile=_profile(_judged(behavior="another_behavior")))


def test_a_campaign_bar_on_the_dimension_is_ratcheted_against_the_registered_one() -> None:
    profile = _profile(_judged(threshold=3.0))
    tighter = BarOverride(measure_id=FAITHFULNESS_DIM, threshold=4.0, direction="higher_is_better")
    looser = BarOverride(measure_id=FAITHFULNESS_DIM, threshold=2.0, direction="higher_is_better")

    _gate(toyhost_judged_template(), tighter, profile=profile)
    with pytest.raises(DesignRefused, match="registered at 3.0 and this campaign proposes 2.0, which is looser"):
        _gate(toyhost_judged_template(), looser, profile=profile)


# =============================================================================
# The bundle adjudicates it as a judged bar
# =============================================================================


def test_the_bundle_adjudicates_a_registered_judged_bar_with_the_judges_tier() -> None:
    campaign, storage = toyhost_campaign()
    assert campaign.declared_design is not None
    design = CampaignDesign.model_construct(**{**dict(campaign.declared_design), "bars": []})
    bundle = assemble_context_bundle(
        campaign.model_copy(update={"declared_design": design}),
        storage=storage,
        profile=_profile(_judged(TOYHOST_JUDGED_DIMENSION)),
    )

    (bar,) = [bar for bar in bundle.bar_adjudications if bar.measure_id == TOYHOST_JUDGED_DIMENSION]
    assert (bar.source, bar.direction, bar.state) == ("registered", "higher_is_better", "adjudicated")
    assert bar.verdicts and all(verdict.judge_evidence_tier is not None for verdict in bar.verdicts)
