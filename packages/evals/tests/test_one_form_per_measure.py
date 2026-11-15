"""One measure reads one way on every surface of its campaign (number policy, rule 1; owner ruling on set scope).

A measure's digits are decided once for its campaign: the written page reads them off its report input, the code page
and every aspect page off the bundles and the code page's readings, and the quick line, a program's rows and the
listing's verdict off the aspect page's forms. Held here, so that no surface reads its own set of a measure:

- **one text, four surfaces.** A quick comparison's change and interval print alike on its aspect page, on its
  written page's forms, in the code page's lead and in the quick line (and in :meth:`Result.contrasts`' rows);
- **one set, every page.** On every fixture campaign with contrasts, and on one with guardrails, the written page
  reads each contrast's and each guardrail check's measure at the places its aspect page does: a judged level counts
  as derived wherever it is gathered, and a guardrail's interval joins its measure's set on the written page too.
  One fixture's input states each judged level's interval (``Stat.half_width``, as a host's own input may), a pass/fail
  dimension's narrower than its contrasts': a written page that counted that level as compared, as its aspect pages
  do not, would read the dimension at one decimal where they read whole points.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator

import pytest

from packages.evals.tests.fixtures.report_inputs.dataset import kairo_bundle
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.report_shapes_support import s1_change, s4_change
from packages.evals.tests.test_quick_comparison import CASES, careful, hasty
from packages.evals.tests.test_quick_guardrails import ARMS, NO_LEAK, correct, no_leak
from packages.evals.tests.test_quick_guardrails import CASES as GUARDED_CASES
from threetears.evals import Eval, grader
from threetears.evals.analysis import TableBlock, assemble_campaign_bundle, campaign_report_markdown
from threetears.evals.analysis.bundle.schema import TRIAL_PASS_NAME
from threetears.evals.analysis.campaign_bundle import CampaignBundle
from threetears.evals.analysis.report.forms import PageForms
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.analysis.report_input_build import report_input

SCOPE = "one-form"


async def test_one_change_prints_one_text_on_the_written_page_the_code_page_an_aspect_page_and_the_quick_line() -> None:
    result = await Eval(CASES, {"hasty": hasty, "careful": careful}, ["label"], k=2, name="hasty vs careful").run(
        store=":memory:", scope=SCOPE
    )
    (fold,) = result.page.aspects
    (row,) = [
        row
        for block in fold.report.blocks
        if isinstance(block, TableBlock) and block.name == "comparisons"
        for row in block.rows
        if row["reading"].text == "Accuracy"
    ]
    change, interval = row["change"].text, row["interval"].text
    # careful against hasty, 100% against 50% of 16 paired cases: whole points, which the interval's width sets.
    assert (change, interval) == ("+50 pts", "+11 to +83 pts")

    held = report_input(off_loop(assemble_campaign_bundle, result.host, result.campaign_id, SCOPE))
    (contrast,) = held.contrasts
    assert contrast.delta is not None and contrast.interval is not None
    written = PageForms.of(held)
    assert written.change(contrast.measure, float(contrast.delta)).text == change
    low, high = (float(end) for end in contrast.interval)
    assert written.interval(contrast.measure, low, high, change=True).text == interval

    # The quick line and the code page's lead say the interval's ends in words, at the page's digits.
    said = "the interval puts careful between 11 and 83 points better"
    assert said in result.render().splitlines()[0]
    assert said in campaign_report_markdown(result.page)
    (program,) = result.contrasts()
    assert program["change"].startswith(f"{change} ({interval} at ")


def _guarded() -> CampaignBundle:
    """A quick comparison graded by a guardrail too: every arm's check on ``no_leak`` against the control."""
    ev = Eval(GUARDED_CASES, ARMS, [correct, grader(no_leak, guard=NO_LEAK)], k=1, name=" vs ".join(ARMS))
    result = asyncio.run(ev.run(store=":memory:", scope=SCOPE))
    return assemble_campaign_bundle(result.host, result.campaign_id, SCOPE)


def _shape(make: Callable[[], object]) -> Callable[[], CampaignBundle]:
    def built() -> CampaignBundle:
        shape = make()
        return assemble_campaign_bundle(shape.host(), shape.campaign.id, shape.campaign.scope_id)  # type: ignore[attr-defined]

    return built


#: How far either side of its mean a level's 95% interval reaches, in standard errors.
_Z95 = 1.96


def _levels_stated(held: ReportInput) -> ReportInput:
    """The input with each judged level's interval stated beside its mean, as a host's own input may state it: on
    Kairo a pass/fail dimension's level reads within about 2.5 points, its contrasts within about 4."""
    judged = {
        key: dimension.model_copy(
            update={
                "per_arm": {
                    arm: stat if stat.sem is None else stat.model_copy(update={"half_width": _Z95 * float(stat.sem)})
                    for arm, stat in dimension.per_arm.items()
                }
            }
        )
        for key, dimension in held.judged.items()
    }
    return held.model_copy(update={"judged": judged})


def _as_built(held: ReportInput) -> ReportInput:
    return held


_CAMPAIGNS: dict[str, Callable[[], CampaignBundle]] = {
    "kairo": kairo_bundle,
    "kairo, judged levels stated": kairo_bundle,
    "s1": _shape(s1_change),
    "s4": _shape(s4_change),
    "guarded": _guarded,
}

#: The written page's input over each campaign's bundle: as the package builds it, or as a host's may state it.
_WRITTEN: dict[str, Callable[[ReportInput], ReportInput]] = {"kairo, judged levels stated": _levels_stated}


def _stated(bundle: CampaignBundle) -> Iterator[tuple[str, str, str]]:
    """Every reading an aspect page states a decision on, a contrast's or a guardrail check's, by aspect."""
    for aspect in bundle.aspects:
        for family in aspect.multiple_comparisons.families:
            for comparison in family.comparisons:
                yield aspect.aspect, comparison.name, comparison.reading
        if aspect.guardrails is not None:
            for check in aspect.guardrails.checks:
                yield aspect.aspect, check.name, check.reading


@pytest.mark.parametrize("name", sorted(_CAMPAIGNS))
def test_the_written_page_reads_every_decided_measure_at_the_places_its_aspect_page_does(name: str) -> None:
    bundle = _CAMPAIGNS[name]()
    written = PageForms.of(_WRITTEN.get(name, _as_built)(report_input(bundle)))
    aspects = PageForms.of_aspects(bundle.aspects, bundle.readings)
    checked, apart = 0, []
    for aspect, measure, reading in dict.fromkeys(_stated(bundle)):
        places = aspects[aspect].contrast_places(measure, reading)
        form = written.measure(measure, kind="trial" if measure == TRIAL_PASS_NAME else reading)
        if places is None or form.places is None:
            continue
        checked += 1
        if places.change != form.places:
            apart.append(f"{aspect} {measure} ({reading}): aspect {places.change}, written {form.places}")
    assert checked
    assert apart == []
