"""What warns on a campaign's page, for every kind, under its question and answer: one plain line per arm for the bars
it misses, then one per arm for the floors it breaks, then one per arm and decision for the guardrails it is not
shown safe on. Each line says what the warning means for the decision first, names the aspects it holds on, and
leaves the figures to the tables below, but for one: an undecided guardrail on a pass/fail check then says how many
cases could fail it and how many each arm failed it on, too few to settle it, so a reader never reads the sample's
size as a defect. Each guardrail is named as the page names its measure everywhere
(:func:`~threetears.evals.analysis.report.verdict.guardrail_name`). Each only warns, and this is where it
warns."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from threetears.evals.analysis.arms import short_digest
from threetears.evals.analysis.bundle.schema import AnalysisContextBundle
from threetears.evals.analysis.campaign_readings import AnswerSplit, BarStatus, CampaignReadings, FloorStatus
from threetears.evals.analysis.report.campaign.common import named_arms
from threetears.evals.analysis.report.campaign.names import AspectNames
from threetears.evals.analysis.report.forms import PageForms
from threetears.evals.analysis.report.verdict import GUARDRAIL_SAID, guardrail_name
from threetears.evals.analysis.report.model import DisclosureBlock, Report, ReportBlock
from threetears.evals.kernel.campaign import EvalCampaign
from threetears.evals.kernel.surface import GuardrailCheck

__all__ = [
    "answer_split_warnings",
    "bar_warnings",
    "campaign_warnings",
    "floor_line",
    "floor_warnings",
    "guardrail_warnings",
    "listed",
    "where",
]


def bar_warnings(readings: CampaignReadings, *, names: AspectNames) -> list[ReportBlock]:
    """One line per arm missing a bar, for every kind: ``Adopting alpha overrides the 2 bars it misses on answers.``,
    or for a measurement, whose bars are its standard, ``solo falls short of the bar on clarity.``

    :param readings: What code read off the campaign.
    :ptype readings: CampaignReadings
    :param names: What the page calls each aspect.
    :ptype names: AspectNames
    :return: The lines.
    :rtype: list[ReportBlock]
    """
    return [DisclosureBlock(section="warnings", source="bar", text=text) for *_, text in _bar_lines(readings, names)]


def guardrail_warnings(
    campaign: EvalCampaign,
    readings: CampaignReadings,
    reports: Mapping[str, Report],
    *,
    names: AspectNames,
    forms: PageForms,
) -> list[ReportBlock]:
    """One line per arm and decision naming each guardrail it is not shown safe on with the aspects that guardrail
    holds on, after the broken floors, what it means for the decision first: ``v2 may do worse than v1 on the No
    invented total score guardrail on classify and the Polite guardrail on tone.`` Each guardrail is paired with its
    own aspects, never with another's, and named as the page names its measure. A guardrail only warns: no line says
    an arm is not adopted. A line naming one undecided pass/fail check then says what leaves it undecided
    (:func:`_undecided_bound`): ``v2 may do worse than v1 on the No invented total score guardrail on classify, by
    more than its 2-point margin. Of the 30 cases that could fail it, v2 failed 1 and v1 failed 1, too few to settle
    it.``

    :param campaign: The campaign.
    :ptype campaign: EvalCampaign
    :param readings: What code read off it.
    :ptype readings: CampaignReadings
    :param reports: Each aspect's own page, by its label.
    :ptype reports: Mapping[str, Report]
    :param names: What the page calls each aspect.
    :ptype names: AspectNames
    :param forms: The page's forms.
    :ptype forms: PageForms
    :return: The lines, the breached before the undecided, each arm in the order first met.
    :rtype: list[ReportBlock]
    """
    labels = readings.arm_labels()
    blocks: list[ReportBlock] = []
    for decision, words in GUARDRAIL_SAID.items():
        # Each arm's guardrails, each with the aspects it holds on, in the order first met; and each arm's checks.
        by_arm: dict[str, dict[str, list[str]]] = {}
        checks_of: dict[str, list[tuple[GuardrailCheck, AnalysisContextBundle, str]]] = {}
        for aspect in campaign.aspects:
            bundle = reports[aspect.label].bundle
            arms_named = named_arms(bundle, labels)

            def label(key: str) -> str:
                return arms_named[key].label if key in arms_named else short_digest(key)

            for check in bundle.guardrails.checks:
                if check.decision != decision:
                    continue
                arm = label(check.contrast.variant_key)
                named = guardrail_name(bundle, check.name, check.reading)
                aspects = by_arm.setdefault(arm, {}).setdefault(named, [])
                if aspect.label not in aspects:
                    aspects.append(aspect.label)
                checks_of.setdefault(arm, []).append((check, bundle, label(check.control.variant_key)))
        total = len(campaign.aspects)
        for arm, by_dimension in by_arm.items():
            each = [
                f"the {dimension} guardrail on {where([names(label) for label in aspects], total)}"
                for dimension, aspects in by_dimension.items()
            ]
            # One undecided check on the line says what bounds it; a line naming several leaves each count to its
            # aspect's guardrails table, since one sentence carries one count.
            bound = None
            if decision == "undecided" and len(checks_of[arm]) == 1:
                ((check, bundle, control),) = checks_of[arm]
                bound = _undecided_bound(check, bundle, arm=arm, control=control, forms=forms)
            said = words.format(arm=arm, control=readings.control_label, named=listed(each))
            text = f"{said}{bound}." if bound else f"{said}."
            blocks.append(DisclosureBlock(section="warnings", source="guardrails", text=text, term="guardrail"))
    return blocks


def _undecided_bound(
    check: GuardrailCheck, bundle: AnalysisContextBundle, *, arm: str, control: str, forms: PageForms
) -> str | None:
    """What leaves one undecided guardrail undecided, where it is a pass/fail check read on a declared margin: the
    margin the arm may be worse than the control by, then how many cases could fail it and how many each arm failed
    it on, too few to settle it: ``, by more than its 5-point margin. Of the 30 cases that could fail it, neither arm
    failed any, too few to settle it``.

    The counts are read off the readings: a case an arm failed it on is one where a trial of that arm failed the
    check (``TrialRow.failed_checks``), and an arm whose every value is the check's good end failed it on none. None
    where those cannot be read: a judged dimension or a measure that is no pass/fail check, no bounded interval (on a
    reading with no declared range a hold is never read, whatever the cases), no declared margin, or failed trials
    beyond the bundle's cap.

    :param check: The undecided guardrail check.
    :ptype check: GuardrailCheck
    :param bundle: The aspect's bundle it was read on.
    :ptype bundle: AnalysisContextBundle
    :param arm: The arm checked, by name.
    :ptype arm: str
    :param control: The control, by name.
    :ptype control: str
    :param forms: The page's forms, which state the margin as declared.
    :ptype forms: PageForms
    :return: The words, opening with the comma that follows the guardrail's name, or None.
    :rtype: str | None
    """
    descriptor = bundle.measure_catalog.get(check.name)
    if (
        check.reading != "measure"
        or descriptor is None
        or not descriptor.pass_fail
        or check.interval_basis != "bounded"
        or not check.margin_declared
        or check.margin <= 0
    ):
        return None
    good = 1.0 if check.higher_is_better else 0.0
    cases = check.contrast.n_cases
    head = (
        f", by more than its {forms.declared(check.margin * 100).text}-point margin. Of the {cases} cases that could "
        "fail it"
    )
    tail = "too few to settle it"
    if check.control.mean == good and check.contrast.mean == good:
        return f"{head}, neither arm failed any, {tail}"
    if bundle.trials_omitted:
        return None

    def failed(key: str) -> int:
        # A trial holds each check it failed by its measure's key, never by the name a page prints for it.
        return len(
            {
                trial.case_id
                for trial in bundle.trials
                if trial.arm == key and any(failure.name == check.name for failure in trial.failed_checks)
            }
        )

    # A count of cases, never of trials: with repeats it can run against the per-trial rates the tables show.
    return (
        f"{head}, {arm} failed {failed(check.contrast.variant_key)} and {control} failed "
        f"{failed(check.control.variant_key)}, {tail}"
    )


def floor_warnings(readings: CampaignReadings, *, names: AspectNames, forms: PageForms) -> list[ReportBlock]:
    """One line per candidate breaking a floor, naming the aspects it breaks one on: ``Adopting v2 overrides the
    10-point floor it breaks on billing.`` Which rate broke it, and the interval it broke on, are the aspect's floors
    table's.

    :param readings: What code read off the campaign.
    :ptype readings: CampaignReadings
    :param names: What the page calls each aspect.
    :ptype names: AspectNames
    :param forms: The page's forms, which state each margin.
    :ptype forms: PageForms
    :return: The lines.
    :rtype: list[ReportBlock]
    """
    return [
        DisclosureBlock(section="warnings", source="floor", text=text)
        for *_, text in _floor_lines(readings, names, forms)
    ]


def _answer_split_lines(readings: CampaignReadings, names: AspectNames) -> list[str]:
    """One line per arm whose results were answered by more than one candidate model, naming the models and the
    aspects: ``latest's results on billing were answered by 2 models (m-03 and m-06), so they are read per model: no
    change, interval or verdict is given for latest there.`` A split control's line says no candidate is read against
    it there. The per-model readings are each aspect's own contrasts.

    :param readings: What code read off the campaign.
    :ptype readings: CampaignReadings
    :param names: What the page calls each aspect.
    :ptype names: AspectNames
    :return: The lines, in the arms' order as first met; none where no arm is split.
    :rtype: list[str]
    """
    by_arm: dict[str, list[AnswerSplit]] = {}
    for split in readings.answer_splits:
        by_arm.setdefault(split.arm, []).append(split)
    lines: list[str] = []
    for arm, splits in by_arm.items():
        label = splits[0].arm_label
        models = sorted({model for split in splits for model in split.served_models})
        unnamed = any(split.unrecorded for split in splits)
        if models and unnamed:
            answered = f"{listed(models)}, and a model some responses did not name"
        elif len(models) > 1:
            answered = f"{len(models)} models ({listed(models)})"
        else:
            answered = "more than one model"
        on = where([names(split.aspect) for split in splits], len(readings.aspects))
        if arm == readings.control:
            # Every contrast is read against the control, so a control read per model leaves none to read: all of
            # the campaign where it is split on every aspect.
            scope = "the whole campaign" if len(splits) == len(readings.aspects) else on
            lines.append(
                f"{label} is the control, and its results on {on} were answered by {answered}, so they are read per "
                f"model: {scope} is not compared, since no candidate is read against {label} there."
            )
            continue
        lines.append(
            f"{label}'s results on {on} were answered by {answered}, so they are read per model: no change, interval "
            f"or verdict is given for {label} there."
        )
    return lines


def answer_split_warnings(readings: CampaignReadings, *, names: AspectNames) -> list[ReportBlock]:
    """:func:`_answer_split_lines` as the page's warnings, after the guardrails'."""
    return [
        DisclosureBlock(section="warnings", source="runs", text=text) for text in _answer_split_lines(readings, names)
    ]


def campaign_warnings(readings: CampaignReadings, names: AspectNames, *, forms: PageForms) -> list[str]:
    """What warns on a campaign's page of its bars and floors, as code words it: each arm missing a bar, then each
    candidate breaking a floor — never the guardrail lines after — then each arm read per answering model
    (:func:`_answer_split_lines`), which has no row for a listing to quote.

    The page's own warnings are these lines, so whatever repeats them (an analysis's listing line) says what its page
    says, word for word.

    :param readings: What code read off the campaign, over the floors frozen with it.
    :ptype readings: CampaignReadings
    :param names: What the page calls each aspect.
    :ptype names: AspectNames
    :param forms: The forms of the surface the lines are printed on, which state each margin: the page's, or an
        off-page surface's (:meth:`~threetears.evals.analysis.report.forms.PageForms.off_page`) for a listing.
    :ptype forms: PageForms
    :return: The lines, in the page's order; none where no bar is missed and no floor broken.
    :rtype: list[str]
    """
    return [
        *(text for *_, text in _bar_lines(readings, names)),
        *(text for *_, text in _floor_lines(readings, names, forms)),
        *_answer_split_lines(readings, names),
    ]


def _bar_lines(readings: CampaignReadings, names: AspectNames) -> list[tuple[str, list[BarStatus], str]]:
    """Each arm missing a bar, with the bars it misses and its line."""
    total = len(readings.aspects)
    lines = []
    for arm, statuses in _by_arm(readings.missed_bars()).items():
        on = where([names(label) for label in dict.fromkeys(status.aspect for status in statuses)], total)
        form = "one" if len(statuses) == 1 else "other"
        template = (_SHORT_LINE if readings.kind == "measurement" else _MISSED_LINE)[form]
        lines.append((arm, statuses, template.format(arm=arm, count=len(statuses), where=on)))
    return lines


#: A measurement's line for an arm missing bars, whose bars are its standard, by count form.
_SHORT_LINE = {
    "one": "{arm} falls short of the bar on {where}.",
    "other": "{arm} falls short of {count} bars on {where}.",
}

#: A change's or a choice's line for an arm missing bars, by count form.
_MISSED_LINE = {
    "one": "Adopting {arm} overrides the bar it misses on {where}.",
    "other": "Adopting {arm} overrides the {count} bars it misses on {where}.",
}

#: The line for a candidate breaking floors, by count form and by whether the floors share one margin.
_FLOOR_LINE = {
    ("one", True): "Adopting {arm} overrides the {margin}-point floor it breaks on {where}.",
    ("other", True): "Adopting {arm} overrides the {margin}-point floors it breaks on {where}.",
    ("one", False): "Adopting {arm} overrides the floor it breaks on {where}.",
    ("other", False): "Adopting {arm} overrides the floors it breaks on {where}.",
}


def floor_line(arm: str, margins: Sequence[float | None], where: str, *, forms: PageForms) -> str:
    """The line a page warns of the floors one candidate breaks in, on a campaign's page and an aspect's own alike:
    ``Adopting v2 overrides the 10-point floor it breaks on billing.``, the floors' margin said where they share one
    (a cost floor has none). Which rate broke it, and the interval it broke on, are the floors table's.

    :param arm: The candidate, by name.
    :ptype arm: str
    :param margins: Each broken floor's margin in points, None for a cost floor; one per floor.
    :ptype margins: Sequence[float | None]
    :param where: Where the floors are broken, as :func:`where` says it.
    :ptype where: str
    :param forms: The page's forms, which state the margin as declared.
    :ptype forms: PageForms
    :return: The line.
    :rtype: str
    """
    shared = set(margins)
    margin = next(iter(shared)) if len(shared) == 1 else None
    form = "one" if len(margins) == 1 else "other"
    return _FLOOR_LINE[(form, margin is not None)].format(
        arm=arm, margin=forms.declared(margin).text if margin is not None else "", where=where
    )


def _floor_lines(
    readings: CampaignReadings, names: AspectNames, forms: PageForms
) -> list[tuple[str, list[FloorStatus], str]]:
    """Each candidate breaking a floor, with the floors it breaks and its line: the floors' margin where they share
    one, as a cost floor has none."""
    total = len(readings.aspects)
    lines = []
    for arm, statuses in _by_arm(readings.broken_floors()).items():
        on = where([names(label) for label in dict.fromkeys(status.aspect for status in statuses)], total)
        lines.append((arm, statuses, floor_line(arm, [status.margin for status in statuses], on, forms=forms)))
    return lines


def _by_arm[Status: (BarStatus, FloorStatus)](statuses: Sequence[Status]) -> dict[str, list[Status]]:
    """``statuses`` by the arm each is of, by its label, each arm in the order first met."""
    found: dict[str, list[Status]] = {}
    for status in statuses:
        found.setdefault(status.arm_label, []).append(status)
    return found


def where(names: Sequence[str], total: int) -> str:
    """The aspects a line on the first screen holds on, as every warning and caveat names them: one by its name,
    ``every aspect`` where it holds on all of several, else how many of the page's and which
    (``2 of 3 aspects (billing and tone)``).

    :param names: The aspects it holds on, as the page calls them, in the page's order.
    :ptype names: Sequence[str]
    :param total: How many aspects the page holds.
    :ptype total: int
    :return: Where, as a line says it.
    :rtype: str
    """
    if len(names) == 1:
        return names[0]
    if len(names) == total:
        return "every aspect"
    return f"{len(names)} of {total} aspects ({listed(names)})"


def listed(items: Sequence[str]) -> str:
    """Items as a sentence or a cell lists them: ``billing``, ``billing and tone``, ``billing, tone and refunds``.

    :param items: The items, in order; at least one.
    :ptype items: Sequence[str]
    :return: The list.
    :rtype: str
    """
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} and {items[-1]}"
