"""The contrasts against the control: each decided (:func:`contrast_verdicts`), its evidence as the verdict line
states it (:func:`contrast_evidence`), the aspect's verdict over them and the guardrails (:func:`aspect_verdict`), and
the contrasts table and its rows (:func:`contrast_rows`).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from threetears.evals.analysis.bundle.schema import (
    AnalysisContextBundle,
    ComparedCell,
    ComparisonFamily,
    FamilyComparison,
)
from threetears.evals.analysis.bundle.surface import bundle_decision_surface
from threetears.evals.analysis.cells import cell_ref
from threetears.evals.analysis.errors import UnresolvableReference
from threetears.evals.analysis.quantities import PERCENT, delta_unit
from threetears.evals.analysis.references import resolve_reading
from threetears.evals.analysis.report.aspect.common import (
    _arm_names,
    _at,
    _case_namer,
    _cell_namer,
    _ChartLedger,
    _control_key,
    _interval,
    _interval_on_delta,
    _level_text,
    _listed,
    _namer,
    _note,
    _unit_on,
)
from threetears.evals.analysis.report.aspect.gates import guardrail_verdicts
from threetears.evals.analysis.report.figures import ContrastPlaces, MeasureRuler, Rulers, report_rulers
from threetears.evals.analysis.report.forms import PageForms
from threetears.evals.analysis.report.keys.vocabulary import joint_guarantee
from threetears.evals.analysis.report.model import (
    CellValue,
    DisclosureBlock,
    ReportBlock,
    TableBlock,
    TableCell,
    TableColumn,
    Verdict,
    VerdictReason,
)
from threetears.evals.analysis.report.number_forms.printed import Printed
from threetears.evals.analysis.report.parts.words import FAMILY_LEVEL_TEMPLATE
from threetears.evals.analysis.report.verdict import MIN_CASES, VerdictLine, verdict_line
from threetears.evals.analysis.report.words import (
    comparison_words,
    disclosure_words,
    reading_heading,
    reading_verb,
    reason_words,
)
from threetears.evals.analysis.stats import EQUIVALENCE_NEEDS_RANGE, INTERVAL_LEVEL
from threetears.evals.analysis.style import CANNOT_TELL_APART
from threetears.evals.kernel.metrics import ACCURACY_MEASURE, METRIC_DESCRIPTORS
from threetears.evals.kernel.surface import DecisionSurface


def aspect_verdict(
    bundle: AnalysisContextBundle,
    *,
    measure_order: Sequence[str] = (),
    arm_labels: Mapping[str, str] | None = None,
    forms: PageForms | None = None,
) -> VerdictLine:
    """The verdict line an aspect's own page opens with, each arm by its variant's name, read without laying the page
    out: what a stored analysis of one aspect is listed by, and what the quick print's first line and the exit line
    print (:func:`~threetears.evals.analysis.report.verdict.verdict_line`).

    :param bundle: The aspect's evidence.
    :ptype bundle: AnalysisContextBundle
    :param measure_order: The aspect's graders' measures, in the order its design lists them.
    :ptype measure_order: Sequence[str]
    :param arm_labels: The names a surface prints the arms by, by variant key, which replace their variants' names
        (:func:`~threetears.evals.analysis.report.words.reader_arm_labels`, as a listing reads them); None names each
        arm by its variant.
    :ptype arm_labels: Mapping[str, str] | None
    :param forms: The aspect page's forms, over its campaign's set of each measure
        (:meth:`~threetears.evals.analysis.report.forms.PageForms.of_aspects`), so the line reads every figure as the
        page does; None for an aspect that is its own campaign (its bundle's forms alone).
    :ptype forms: PageForms | None
    :return: The verdict line.
    :rtype: VerdictLine
    """
    # The line's verdicts and the figures they rest on are the aspect page's, decided once in the page's forms.
    if forms is None:
        forms = PageForms.of_bundle(bundle)
    return verdict_line(
        bundle,
        _namer(_arm_names(bundle, labels=arm_labels)),
        case_name=_case_namer(bundle),
        verdicts=_typed_verdicts(bundle, forms, arm_labels=arm_labels),
        order=measure_order,
        evidence=contrast_evidence(bundle, forms=forms),
    )


def contrast_evidence(
    bundle: AnalysisContextBundle, *, forms: PageForms | None = None
) -> Callable[[Verdict], str | None]:
    """The figures each contrast verdict over ``bundle`` rests on, as the sentences its verdict line says after the
    decision, each arm the subject: what each arm passed of its trials (or its mean, for a reading that is no rate),
    then what the interval allows the arm to be against the control, in points for a rate, over the cases the test
    read. Each figure is the contrasts table's, at its precision; the interval's level is the table's header's. A
    sentence carries at most three figures (:data:`~threetears.evals.analysis.style.MAX_NUMERALS`).

    :param bundle: The aspect's evidence.
    :ptype bundle: AnalysisContextBundle
    :param forms: The aspect page's forms (:meth:`~threetears.evals.analysis.report.forms.PageForms.of_bundle`), built
        once for the page; None for a verdict line read without its page (a listing's, the quick print's), which
        builds the bundle's own.
    :ptype forms: PageForms | None
    :return: A verdict's figures, or None where it has no change or interval to state (it is untested, or no
        contrast of the bundle is its).
    :rtype: Callable[[Verdict], str | None]
    :raises ValueError: ``forms`` were built from another bundle.
    """
    page = forms if forms is not None else PageForms.of_bundle(bundle)
    if page.bundle is not bundle:
        raise ValueError("an aspect page's forms were handed in beside a bundle they were not built from")
    surface = bundle_decision_surface(bundle)
    rulers = report_rulers(surface, bundle.measure_catalog)
    tested = {
        (comparison.contrast.variant_key, comparison.name, comparison.reading, family.aspect): comparison
        for family in bundle.multiple_comparisons.families
        for comparison in family.comparisons
    }

    def figures(verdict: Verdict) -> str | None:
        comparison = tested.get((verdict.variant_key, verdict.name, verdict.reading, verdict.aspect))
        if comparison is None or verdict.control is None or comparison.delta is None or comparison.interval is None:
            return None
        if comparison.control.mean is None or comparison.contrast.mean is None:
            return None
        ruler = rulers.of(comparison.name, comparison.reading)
        # The figures as the contrasts table prints them: each reading's one set, whichever surface states it.
        form = page.change_money(comparison.name, comparison.reading)
        places = page.contrast_places(comparison.name, comparison.reading)
        sides = [
            (verdict.control, *_side_reading(bundle, surface, comparison, comparison.control, ruler, page, places)),
            (verdict.arm, *_side_reading(bundle, surface, comparison, comparison.contrast, ruler, page, places)),
        ]
        said = [_sides_sentence(sides, verdict.heading, comparison.reading)]
        paired = (
            comparison.test == "paired" and not comparison.control.n_left_out and not comparison.contrast.n_left_out
        )
        allows = page.ruled_allows(
            verdict.arm,
            ruler,
            comparison.interval,
            form=form,
            places=places,
            higher_is_better=comparison.higher_is_better,
            measure=comparison.name,
            kind=comparison.reading,
        )
        if _unvaried_means(surface, comparison) is not None:
            # Neither arm varied, so the table draws no interval, and the sentence claims none either.
            allows = "Neither arm varied over these cases, so no interval is drawn on the change."
        if paired:
            said.append(f"Over {comparison.control.n_cases} paired cases, {allows[:1].lower()}{allows[1:]}")
        else:
            said += [f"The test read {_comparison_cases(comparison)} cases.", allows]
        return " ".join(said)

    return figures


def _sides_sentence(sides: Sequence[tuple[str, str, str | None, int | None]], heading: str, reading: str) -> str:
    """Both arms' figures in the reading's own words: the pass rate as what each arm passed (``Of 90 trials each, v1
    passed 73 and v2 88.``); a counted measure as what it was on each arm's trials, in its own verb
    (``Of 90 trials each, Accuracy was right on 90 for v1 and 89 for v2.``), never "passed", which is the pass rate's;
    one sentence per arm where they ran different trials; a reading that is no rate by its means."""
    (control, control_text, control_passed, control_trials), (arm, arm_text, arm_passed, arm_trials) = sides
    if control_passed is None or control_trials is None or arm_passed is None or arm_trials is None:
        return f"The mean is {control_text} on {control} and {arm_text} on {arm}."
    if reading == "trial":
        if control_trials == arm_trials:
            return f"Of {control_trials} trials each, {control} passed {control_passed} and {arm} {arm_passed}."
        return f"Of {control_trials} trials, {control} passed {control_passed}. Of {arm_trials}, {arm} passed {arm_passed}."
    verb = reading_verb(heading)
    if control_trials == arm_trials:
        return f"Of {control_trials} trials each, {heading} {verb} on {control_passed} for {control} and {arm_passed} for {arm}."
    return (
        f"{heading} {verb} on {control_passed} of {control_trials} trials for {control}. {heading} {verb} on "
        f"{arm_passed} of {arm_trials} for {arm}."
    )


def _side_reading(
    bundle: AnalysisContextBundle,
    surface: DecisionSurface,
    comparison: FamilyComparison,
    side: ComparedCell,
    ruler: MeasureRuler,
    forms: PageForms,
    places: ContrastPlaces | None = None,
) -> tuple[str, str | None, int | None]:
    """One arm's figure on a contrast's reading as the contrasts table prints it, without its own interval, and for a
    rate the trials it passed and the trials it is over: the pass rate's from the scoreboard, a pass/fail measure's
    from the arm's cell; None for each where no count is read."""
    figure, _ = _side_figure(surface, comparison, side)
    text = forms.ruled_level(
        ruler,
        figure if figure is not None else side.mean or 0.0,
        places,
        measure=comparison.name,
        kind=comparison.reading,
    ).text
    if ruler.unit != PERCENT:
        return text, None, None
    if comparison.reading == "trial":
        board = next((row for row in bundle.scoreboard if row.variant_key == side.variant_key), None)
        return (text, None, None) if board is None else (text, str(board.passed), board.usable)
    described = bundle.measure_catalog.get(comparison.name) or METRIC_DESCRIPTORS.get(comparison.name)
    # A share of trials: a pass/fail check, a boolean measure, or a classifier's accuracy (its trials' label match).
    counted = described is not None and (
        described.pass_fail or described.data_type == "boolean" or comparison.name == ACCURACY_MEASURE
    )
    if comparison.reading != "measure" or not counted:
        return text, None, None
    try:
        resolved = resolve_reading(
            surface, cell_ref(side.variant_key, side.apparatus_class_id), comparison.name, comparison.reading
        )
    except UnresolvableReference:
        return text, None, None
    if resolved.n <= 0 or figure != resolved.mean:
        return text, None, None
    passed = forms.hits(resolved.mean, resolved.n, measure=comparison.name, kind=comparison.reading)
    return text, passed.text, resolved.n


def _typed_verdicts(
    bundle: AnalysisContextBundle, forms: PageForms, *, arm_labels: Mapping[str, str] | None = None
) -> list[Verdict]:
    """The guardrails' and the contrasts' typed verdicts over one bundle, each arm by its variant's name (or by
    ``arm_labels``, where a surface names it otherwise), as its page words them, in the page's forms: what its verdict
    line prints, read without laying the page out."""
    surface = bundle_decision_surface(bundle)
    rulers = report_rulers(surface, bundle.measure_catalog)
    cell_name = _cell_namer(surface, _arm_names(bundle, labels=arm_labels))
    guardrails = guardrail_verdicts(
        surface, bundle.guardrails, cell_name, rulers, check_names=bundle.named_checks(), forms=forms
    )
    return [*guardrails, *contrast_verdicts(bundle, surface, arm_labels=arm_labels, forms=forms)]


def _with_scale(heading: str, places: ContrastPlaces | None) -> str:
    """A reading's heading with its score's scale, where it is a score: ``outcome (judged, 1–5)``."""
    if places is None or not places.scale:
        return heading
    if heading.endswith(")"):
        return f"{heading[:-1]}, {places.scale})"
    return f"{heading} ({places.scale})"


def _unvaried_means(surface: DecisionSurface, comparison: FamilyComparison) -> tuple[float, float] | None:
    """Both arms' means on a contrast's reading where neither varied, control first: each arm's own interval is a
    point, so no interval on their change says anything; None where an arm varied, or a side holds no interval of its
    own to tell."""
    means: list[float] = []
    for side in (comparison.control, comparison.contrast):
        if side.n_left_out or comparison.reading not in ("measure", "judged"):
            return None
        try:
            resolved = resolve_reading(
                surface, cell_ref(side.variant_key, side.apparatus_class_id), comparison.name, comparison.reading
            )
        except UnresolvableReference:
            # NOSILENT: a side the surface holds no reading for has no interval of its own to tell; the row keeps its own.
            return None
        if resolved.ci_low is None or resolved.ci_low != resolved.ci_high or resolved.ci_low != resolved.mean:
            return None
        means.append(resolved.mean)
    return means[0], means[1]


def _no_variation(
    surface: DecisionSurface,
    comparison: FamilyComparison,
    ruler: MeasureRuler,
    forms: PageForms,
    places: ContrastPlaces | None,
) -> str | None:
    """The words a contrast's interval reads as where neither arm varied (:func:`_unvaried_means`): ``no variation
    (all 5)``, where both read one figure, each arm's figure at the set's places for levels; None where an arm varied,
    or a side holds no interval of its own to tell."""
    if (means := _unvaried_means(surface, comparison)) is None:
        return None
    said = [
        forms.ruled_level(ruler, mean, places, measure=comparison.name, kind=comparison.reading).text for mean in means
    ]
    texts = list(dict.fromkeys(said))
    return f"no variation (all {texts[0]})" if len(texts) == 1 else f"no variation ({' vs '.join(said)})"


def _comparison_cases(comparison: FamilyComparison) -> str:
    """The cases a contrast's test read, and how many each side ran that it left out."""
    control, contrast = comparison.control, comparison.contrast
    if comparison.test == "paired":
        text = f"{control.n_cases} paired"
    else:
        text = f"{contrast.n_cases} vs {control.n_cases}" + (" unpaired" if comparison.test == "unpaired" else "")
    left_out = [
        f"{count} of the {side}'s"
        for side, count in (("arm", contrast.n_left_out), ("control", control.n_left_out))
        if count
    ]
    if left_out:
        text += f"; {' and '.join(left_out)} left out, not run by the other side"
    return text


def _side_cell(
    surface: DecisionSurface,
    comparison: FamilyComparison,
    side: ComparedCell,
    ruler: MeasureRuler,
    forms: PageForms,
    places: ContrastPlaces | None = None,
) -> TableCell:
    """One arm's figure on a contrast's reading (:func:`_side_text`), carrying the id a finding's figure lands on where
    it is the cell's own."""
    text, figure = _side_text(surface, comparison, side, ruler, forms, places)
    cell = _at(TableCell(value=text), surface, side, comparison.name, comparison.reading, figure)
    return forms.of_measure(cell, measure=comparison.name, kind=comparison.reading)


def _side_figure(
    surface: DecisionSurface, comparison: FamilyComparison, side: ComparedCell
) -> tuple[float | None, tuple[float, float] | None]:
    """One arm's figure on a contrast's reading, unscaled, with its own interval where the cell has one of some width.

    The figure and its interval are the arm's cell's (:func:`~threetears.evals.analysis.references.resolve_reading`)
    where the test read every case the cell holds, so the pair is one estimate; where cases were left out, or the
    reading is the pass rate, which no cell holds as a measure, the figure is the mean the test read, alone. An
    interval with no width (every case read one figure) is no interval. None for both where the side has no mean.
    """
    if side.mean is None:
        return None, None
    if comparison.reading in ("measure", "judged") and side.n_left_out == 0:
        try:
            resolved = resolve_reading(
                surface, cell_ref(side.variant_key, side.apparatus_class_id), comparison.name, comparison.reading
            )
        except UnresolvableReference:
            resolved = None
        if resolved is not None and resolved.ci_low is not None and resolved.ci_high is not None:
            if resolved.ci_low == resolved.ci_high:
                return resolved.mean, None
            return resolved.mean, (resolved.ci_low, resolved.ci_high)
    return side.mean, None


def _side_text(
    surface: DecisionSurface,
    comparison: FamilyComparison,
    side: ComparedCell,
    ruler: MeasureRuler,
    forms: PageForms,
    places: ContrastPlaces | None = None,
) -> tuple[str | None, float | None]:
    """One arm's figure on a contrast's reading (:func:`_side_figure`), its unit on it, with its own interval where the
    cell has one: ``98.9% (88.1 to 99.7)``, ``2.42 s (2.35 to 2.49)``, at the reading's one precision for levels
    (``places``); the figure alone where the interval has no width: ``5.00``, never ``5 (5 to 5)``.

    :return: The text, and the figure it prints, unscaled; None for both where the side has no mean.
    """
    figure, interval = _side_figure(surface, comparison, side)
    if figure is None:
        return None, None
    text = _unit_on(
        forms.ruled_level(ruler, figure, places, measure=comparison.name, kind=comparison.reading).text, ruler
    )
    if interval is not None:
        text += _interval(
            interval[0], interval[1], ruler, forms, places, measure=comparison.name, kind=comparison.reading
        )
    return text, figure


def _family_guarantee(family: ComparisonFamily, forms: PageForms, *, noun: str = "") -> str:
    """What a family's intervals guarantee, as a heading or a cell says it (owner ruling, level in headings): the
    joint level its correction holds them at, over its size (``95% (adjusted for 11 comparisons)``), never its
    per-interval level, which is the family's disclosure's and the glossary's; a family of one at its own level. A
    ``noun`` follows the level, as a column's head names what it holds (``95% interval (adjusted for 11
    comparisons)``)."""
    after = f" {noun}" if noun else ""
    guarantee = joint_guarantee(family.alpha, family.family_size, corrected=family.interval_level is not None)
    if guarantee is None:
        level = family.interval_level if family.interval_level is not None else INTERVAL_LEVEL
        return f"{_level_text(level, forms)}{after}"
    joint, size = guarantee
    return FAMILY_LEVEL_TEMPLATE.format(joint=f"{_level_text(joint, forms)}{after}", size=size)


def _contrast_margin(bundle: AnalysisContextBundle, comparison: FamilyComparison) -> float | None:
    """The margin a contrast's reading has — its equivalence margin, or the margin no equivalence test could read."""
    if comparison.equivalence_margin is not None:
        return comparison.equivalence_margin
    if comparison.reading != "measure":
        return None
    descriptor = bundle.measure_catalog.get(comparison.name)
    if descriptor is not None and descriptor.materiality_threshold is not None:
        return descriptor.materiality_threshold
    return bundle.run_margins.get(comparison.name)


def _margin_text(margin: float, ruler: MeasureRuler, forms: PageForms, comparison: FamilyComparison) -> str:
    """A margin as the row's reading is stated: on its measure's one ruler, in the unit a change in it is stated in."""
    declared = forms.in_unit(
        margin * ruler.factor,
        delta_unit(ruler.unit),
        role="declared",
        measure=comparison.name,
        kind=comparison.reading,
    )
    return f"±{declared.text}"


def _immaterial_words(
    margin: float | None, heading: str, ruler: MeasureRuler, forms: PageForms, comparison: FamilyComparison
) -> str:
    """The caveat an immaterial row carries, naming the margin it was read against and the reading it is on."""
    where = (
        "the margin declared on it"
        if margin is None
        else f"the margin of {_margin_text(margin, ruler, forms, comparison)} on {heading}"
    )
    return f"; immaterial: the observed delta is inside {where}, which does not show the true difference is that small"


def _contrast_reason(comparison: FamilyComparison) -> tuple[VerdictReason, str | None]:
    """Why a contrast's verdict is what it is, as a code and, where there is more to say, in the page's words (a
    stored reason reworded by :func:`~threetears.evals.analysis.report.words.reason_words`)."""
    if comparison.verdict == "untested":
        return "untestable", reason_words(comparison.untested_reason)
    if comparison.verdict in ("improved", "regressed"):
        return "separated", None
    if comparison.verdict == "equivalent":
        return "inside_margin", None
    if comparison.margin_source is None:
        return "no_margin", reason_words(comparison.not_separated_reason)
    if comparison.equivalence_p_adjusted is None:
        return "margin_untested", reason_words(
            comparison.not_separated_reason or comparison.equivalence_untested_reason
        ) or "the test was unpaired, and the equivalence test reads only cases both arms ran"
    return "not_inside_margin", reason_words(comparison.not_separated_reason)


def contrast_verdicts(
    bundle: AnalysisContextBundle,
    surface: DecisionSurface,
    *,
    arm_labels: Mapping[str, str] | None = None,
    forms: PageForms,
) -> list[Verdict]:
    """Each contrast the bundle tested against the control as a typed verdict, in the contrasts table's row order.

    A contrast with fewer than :data:`MIN_CASES` cases on either side is printed ``not tested (n=N)``: its typed
    outcome is the test's, and the words state no difference, as a rate on so few cases states none. ``arm_labels``
    name the arms as :func:`build_code_only_report` does; ``forms`` are the aspect page's
    (:meth:`~threetears.evals.analysis.report.forms.PageForms.of_bundle`), which state each margin the words name.
    """
    arm = _cell_namer(surface, _arm_names(bundle, labels=arm_labels))
    rulers = report_rulers(surface, bundle.measure_catalog)
    verdicts = []
    for family in bundle.multiple_comparisons.families:
        for comparison in family.comparisons:
            heading = reading_heading(
                surface, comparison.name, comparison.reading, declared=bundle.check_name(comparison.name)
            )
            margin = _contrast_margin(bundle, comparison)
            ruler = rulers.of(comparison.name, comparison.reading)
            reason, detail = _contrast_reason(comparison)
            named = arm(comparison.contrast.variant_key, comparison.contrast.apparatus_class_id)
            control = arm(comparison.control.variant_key, comparison.control.apparatus_class_id)
            if (small := PageForms.small_n(comparison)) is not None:
                reason = "too_few_cases"
                detail = f"{small} case{'' if small == 1 else 's'}, under the {MIN_CASES} a rate needs"
                words = f"not tested (n={small})"
            else:
                words = _contrast_words(comparison, (named, control), margin, heading, ruler, forms)
            verdicts.append(
                Verdict(
                    kind="contrast",
                    outcome=comparison.verdict,
                    reason=reason,
                    reason_detail=detail,
                    reading=comparison.reading,
                    name=comparison.name,
                    heading=heading,
                    arm=named,
                    variant_key=comparison.contrast.variant_key,
                    apparatus_class_id=comparison.contrast.apparatus_class_id,
                    control=control,
                    aspect=family.aspect,
                    delta=comparison.delta,
                    interval=comparison.interval,
                    interval_level=family.interval_level if comparison.interval is not None else None,
                    p_adjusted=comparison.p_adjusted,
                    margin=margin,
                    margin_source=comparison.margin_source,
                    materiality=comparison.materiality,
                    words=words,
                )
            )
    return verdicts


def _verdict_cell(verdict: Verdict) -> str:
    """A contrast's verdict as the contrasts table's cell says it: its words, the undecided reading in its short cell
    form (:data:`~threetears.evals.analysis.style.CANNOT_TELL_APART`), since the row's columns name both arms."""
    if verdict.outcome != "not_separated" or verdict.control is None:
        return verdict.words
    sentence = comparison_words("not_separated", verdict.arm, verdict.control)
    # prose-canary: allow — Verdict.words is the sentence comparison_words rendered, not anything a model wrote
    return CANNOT_TELL_APART + verdict.words.removeprefix(sentence)


def _contrast_words(
    comparison: FamilyComparison,
    arms: tuple[str, str],
    margin: float | None,
    heading: str,
    ruler: MeasureRuler,
    forms: PageForms,
) -> str:
    """A tested contrast's verdict in words: the verdict naming the arm and the control (``arms``) by label, why where
    there is more to say, its margin, materiality."""
    on_runs = ", declared on the runs" if comparison.margin_source == "run" else ""
    return (
        comparison_words(comparison.verdict, *arms, higher_is_better=comparison.higher_is_better)
        + (f" ({reason_words(comparison.untested_reason)})" if comparison.untested_reason else "")
        + (f" ({reason_words(comparison.not_separated_reason)})" if comparison.not_separated_reason else "")
        + (
            f" (margin {_margin_text(margin, ruler, forms, comparison)}{on_runs})"
            if comparison.verdict == "equivalent" and margin is not None
            else ""
        )
        + (
            _immaterial_words(margin, heading, ruler, forms, comparison)
            if comparison.materiality == "immaterial" and comparison.verdict != "equivalent"
            else ""
        )
    )


def _comparison_blocks(
    bundle: AnalysisContextBundle,
    surface: DecisionSurface,
    verdicts: Sequence[Verdict],
    charts: _ChartLedger,
    arm: Callable[[str], str],
    rulers: Rulers,
    forms: PageForms,
) -> list[ReportBlock]:
    """The contrasts that can resolve, as one table of each arm's value beside the control's, with their statistics
    folded away.

    Each row is a reading: the control's figure with its interval, then the arm's, then the change with its interval
    drawn in the row (the ``change`` cell carries it, and HTML draws it as a glyph on one scale per unit), the
    interval as text, the verdict and the cases. Two arms, and each arm's figure heads its own column, so a row reads
    across; more, and each row is one reading on one arm, the control's figure repeated, since a column per arm and a
    change per arm would not stay legible at three or four.

    A contrast with a side under the case floor is never a row: when none can resolve, the section is one line, and
    otherwise one line counts the ones left out. A reading one side carries on no trial is no contrast at all, and the
    methods count it. The level the intervals hold at is said once, in the column's header, where every shown row
    holds at one level; where families differ, each cell says its own.
    """
    comparisons = bundle.multiple_comparisons
    if not comparisons.families:
        if comparisons.withheld:
            return [DisclosureBlock(section="contrasts", source="comparisons", text=comparisons.withheld)]
        return []
    typed = iter(verdicts)
    shown: list[dict[str, TableCell]] = []
    statistics: list[dict[str, TableCell]] = []
    under: list[int] = []
    # What every shown interval guarantees: one guarantee, and the header says it; more, and each cell says its own.
    levels = {
        _family_guarantee(family, forms)
        for family in comparisons.families
        for comparison in family.comparisons
        if family.interval_level is not None and comparison.interval is not None and forms.small_n(comparison) is None
    }
    for family in comparisons.families:
        for comparison in family.comparisons:
            verdict = next(typed)
            if min(comparison.control.n_cases, comparison.contrast.n_cases) == 0:
                continue
            ruler = rulers.of(comparison.name, comparison.reading)
            if (small := forms.small_n(comparison)) is not None:
                under.append(small)
                continue
            heading = ruler.scaled(verdict.heading)
            form = forms.change_money(comparison.name, comparison.reading)
            places = forms.contrast_places(comparison.name, comparison.reading)
            # Where neither arm varied, an interval on their change says nothing, and the cell says so instead.
            unvaried = _no_variation(surface, comparison, ruler, forms, places)
            interval = unvaried or _interval_on_delta(
                comparison.interval,
                ruler,
                forms,
                form=form,
                places=places,
                said_level=None if len(levels) == 1 else _family_guarantee(family, forms),
                measure=comparison.name,
                kind=comparison.reading,
            )
            # The interval is drawn in the row only where both arms have the cases and one was computed.
            arm_ok = charts.drawable(
                comparison.contrast.variant_key,
                min(comparison.contrast.n_cases, comparison.control.n_cases),
                comparison.interval if comparison.delta is not None else None,
                heading,
            )
            control_ok = charts.drawable(
                comparison.control.variant_key, comparison.control.n_cases, (0.0, 0.0), heading
            )
            shown.append(
                {
                    "reading": TableCell(value=_with_scale(heading, places)),
                    "contrast": TableCell(value=verdict.arm),
                    "control_label": TableCell(value=verdict.control),
                    "control": _side_cell(surface, comparison, comparison.control, ruler, forms, places),
                    "arm": _side_cell(surface, comparison, comparison.contrast, ruler, forms, places),
                    "change": forms.ruled_change_cell(
                        ruler,
                        comparison.delta,
                        interval=comparison.interval if arm_ok and control_ok and unvaried is None else None,
                        margin=_contrast_margin(bundle, comparison),
                        form=form,
                        places=places.change if places is not None else None,
                        measure=comparison.name,
                        kind=comparison.reading,
                    ),
                    "interval": forms.of_measure(
                        TableCell(value=interval), measure=comparison.name, kind=comparison.reading
                    ),
                    "verdict": TableCell(value=_verdict_cell(verdict)),
                    "cases": TableCell(value=_comparison_cases(comparison)),
                }
            )
            statistics.append(
                {
                    "question": TableCell(value="Q1" if family.aspect is not None else "campaign-wide"),
                    "reading": TableCell(value=heading),
                    "contrast": TableCell(value=verdict.arm),
                    "control_mean": forms.ruled_level_cell(
                        ruler, comparison.control.mean, places, measure=comparison.name, kind=comparison.reading
                    ),
                    "arm_mean": forms.ruled_level_cell(
                        ruler, comparison.contrast.mean, places, measure=comparison.name, kind=comparison.reading
                    ),
                    # Each statistic in its fixed form: an effect and a correlation at two decimals, p as APA writes it.
                    "hedges_g": _fixed_cell(forms.effect(comparison.hedges_g)),
                    "pairing_r": _fixed_cell(
                        forms.effect(
                            comparison.pairing_correlation.r if comparison.pairing_correlation is not None else None
                        )
                    ),
                    "p_adjusted": _fixed_cell(forms.p(comparison.p_adjusted)),
                }
            )
    blocks: list[ReportBlock] = []
    if under and not shown:
        blocks.append(
            DisclosureBlock(
                section="contrasts",
                source="comparisons",
                text=(
                    f"No contrast can resolve: a side has {min(under)} usable case{'' if min(under) == 1 else 's'}; "
                    f"a difference is tested only from {MIN_CASES} independent cases."
                ),
                term="case floor",
            )
        )
    elif under:
        blocks.append(
            DisclosureBlock(
                section="contrasts",
                source="comparisons",
                text=f"{len(under)} more contrasts are not shown: a side has under {MIN_CASES} usable cases.",
                term="case floor",
            )
        )
    if shown:
        control = _control_key(bundle)
        guaranteed = [
            family
            for family in comparisons.families
            if family.interval_level is not None and any(c.interval is not None for c in family.comparisons)
        ]
        interval_head = (
            _family_guarantee(guaranteed[0], forms, noun="interval") if len(levels) == 1 and guaranteed else "Interval"
        )
        candidates = list(dict.fromkeys(str(row["contrast"].value) for row in shown))
        controls = list(dict.fromkeys(str(row["control_label"].value) for row in shown))
        # Two arms: each arm's figure heads its own column, and an arm column would only repeat the header.
        two_arms = len(candidates) == 1 and len(controls) == 1
        columns = [
            TableColumn(key="reading", header="Reading", term="reading"),
            *([] if two_arms else [TableColumn(key="contrast", header="Arm")]),
            TableColumn(key="control", header=f"▸ {controls[0]}" if len(controls) == 1 else "Control", term="control"),
            TableColumn(key="arm", header=candidates[0] if two_arms else "Value", term="arm mean"),
            TableColumn(key="change", header="Change", term="change"),
            TableColumn(key="interval", header=interval_head, term="interval"),
            TableColumn(key="verdict", header="Verdict", term="verdict"),
            TableColumn(key="cases", header="Cases", term="cases"),
        ]
        blocks += [
            TableBlock(
                section="contrasts",
                name="comparisons",
                title=f"Against {arm(control)}" if control is not None else "Against the control",
                columns=columns,
                rows=[{column.key: row[column.key] for column in columns} for row in shown],
                order="by declared question, then as the bundle tested them"
                + ("" if two_arms else "; one row per reading on each arm, the control's figure repeated"),
                total_rows=len(shown),
                key_columns=1 if two_arms else 2,
            ),
            TableBlock(
                section="contrasts",
                name="comparison_statistics",
                title="Statistics",
                columns=[
                    TableColumn(key="question", header="Question"),
                    TableColumn(key="reading", header="Reading", term="reading"),
                    TableColumn(key="contrast", header="Arm"),
                    TableColumn(key="control_mean", header="Control mean", term="control mean"),
                    TableColumn(key="arm_mean", header="Arm mean", term="arm mean"),
                    TableColumn(key="hedges_g", header="Hedges' g", term="Hedges' g"),
                    TableColumn(key="pairing_r", header="Pairing r", term="pairing r"),
                    TableColumn(key="p_adjusted", header="Adjusted p", term="adjusted p"),
                ],
                rows=statistics,
                order="row for row as the contrasts above",
                total_rows=len(statistics),
                details=True,
            ),
        ]
    blocks.extend(_note(disclosure_words(family.disclosure)) for family in comparisons.families)
    unranged = sorted(
        {
            reading_heading(surface, comparison.name, comparison.reading, declared=bundle.check_name(comparison.name))
            for family in comparisons.families
            for comparison in family.comparisons
            if comparison.equivalence_untested_reason is not None
        }
    )
    if unranged:
        blocks.append(_note(f"Equivalence untested on {_listed(unranged)}: {EQUIVALENCE_NEEDS_RANGE}."))
    if bundle.run_margins_withheld is not None:
        blocks.append(_note(bundle.run_margins_withheld))
    blocks += [_note(text) for text in bundle.launch_declarations]
    return blocks


def contrast_rows(
    bundle: AnalysisContextBundle,
    *,
    forms: PageForms | None = None,
    arm_labels: Mapping[str, str] | None = None,
) -> list[dict[str, CellValue]]:
    """Every contrast the bundle tested against the control, one row each, in :func:`contrast_verdicts`' order.

    The page shows only the contrasts that can resolve; a program reads them all here. A contrast with a side under
    the case floor states no change, interval, effect size or p: on so few cases a rate states none. Every figure
    is the page's, on the reading's one ruler for the report, and ``unit`` names it: a share in percent (its change
    and interval in points), a cost on the scale its ``reading`` names, a time in the unit the page states it in. A
    program that wants the measure's own units reads :attr:`Report.verdicts` or the bundle, never these rows.

    :param bundle: The campaign's evidence.
    :ptype bundle: AnalysisContextBundle
    :param forms: The aspect page's forms, over its campaign's set of each measure
        (:meth:`~threetears.evals.analysis.report.forms.PageForms.of_aspects`), so every row reads as the page does;
        None for an aspect that is its own campaign (its bundle's forms alone).
    :ptype forms: PageForms | None
    :param arm_labels: The names the page prints the arms by, by variant key, which replace their variants' names, as
        :func:`aspect_verdict` reads them; None names each arm by its variant.
    :ptype arm_labels: Mapping[str, str] | None
    :return: One row per contrast, keyed ``question`` (in the words it was asked, or ``(campaign-wide)``), ``reading``,
        ``unit`` (the unit ``control_mean``, ``arm_mean`` and ``delta`` are in; a percent's delta is in points),
        ``contrast`` and ``control`` (each arm as the page names it), ``control_mean``, ``arm_mean``, ``delta`` and
        ``change`` (the delta with its interval, in words), ``interval`` (with the level it holds at), ``hedges_g``,
        ``pairing_r``, ``p_adjusted``, ``verdict`` (in words) and ``cases``.
    :rtype: list[dict[str, CellValue]]
    """
    surface = bundle_decision_surface(bundle)
    # A program's rows read as the aspect's page does, its forms built once for every row.
    if forms is None:
        forms = PageForms.of_bundle(bundle)
    verdicts = iter(contrast_verdicts(bundle, surface, arm_labels=arm_labels, forms=forms))
    rulers = report_rulers(surface, bundle.measure_catalog)
    rows: list[dict[str, CellValue]] = []
    for family in bundle.multiple_comparisons.families:
        for comparison in family.comparisons:
            verdict = next(verdicts)
            ruler = rulers.of(comparison.name, comparison.reading)
            small = forms.small_n(comparison)
            form = forms.change_money(comparison.name, comparison.reading)
            places = forms.contrast_places(comparison.name, comparison.reading)
            # A program's row has no header to carry the interval's level, so the cell says it; where neither arm
            # varied, it says so, as the page's row does. A contrast under the case floor states neither.
            level = family.interval_level
            interval = (
                None
                if small is not None
                else _no_variation(surface, comparison, ruler, forms, places)
                or _interval_on_delta(
                    comparison.interval,
                    ruler,
                    forms,
                    level,
                    form=form,
                    places=places,
                    measure=comparison.name,
                    kind=comparison.reading,
                )
            )
            rows.append(
                {
                    "question": (
                        bundle.declared.question
                        if family.aspect is not None and bundle.declared is not None
                        else "(campaign-wide)"
                    ),
                    "reading": ruler.scaled(verdict.heading),
                    "unit": ruler.unit,
                    "contrast": verdict.arm,
                    "control": verdict.control,
                    "control_mean": _on_ruler(comparison.control.mean, ruler),
                    "arm_mean": _on_ruler(comparison.contrast.mean, ruler),
                    "delta": None if small is not None else _on_ruler(comparison.delta, ruler),
                    "change": None
                    if small is not None or comparison.delta is None
                    else forms.ruled_change_cell(
                        ruler,
                        comparison.delta,
                        form=form,
                        places=places.change if places is not None else None,
                        measure=comparison.name,
                        kind=comparison.reading,
                    ).text
                    # The interval beside the change it is on, in its one form, in parentheses as an arm's is.
                    + (f" ({interval})" if interval else ""),
                    "interval": interval,
                    "hedges_g": None if small is not None else comparison.hedges_g,
                    "pairing_r": comparison.pairing_correlation.r
                    if comparison.pairing_correlation is not None and small is None
                    else None,
                    "p_adjusted": None if small is not None else comparison.p_adjusted,
                    "verdict": verdict.words,
                    "cases": _comparison_cases(comparison),
                }
            )
    return rows


def _fixed_cell(printed: Printed) -> TableCell:
    """A figure that is no measurement as a table's cell: its number, which a program sorts by, printed as decided."""
    if printed.value is None:
        return TableCell(value=None)
    return TableCell(value=printed.value, value_text=printed.text)


def _on_ruler(value: float | None, ruler: MeasureRuler) -> float | None:
    """A figure restated onto its measure's ruler, as a number, for a program; None for none."""
    return None if value is None else value * ruler.factor


def report_cell_labels(bundle: AnalysisContextBundle, *, arm_labels: Mapping[str, str] | None = None) -> dict[str, str]:
    """Each cell's arm as the page names it, by cell reference: its label, with its test setup where it ran under two.

    :param bundle: The campaign's evidence.
    :ptype bundle: AnalysisContextBundle
    :param arm_labels: The names the page prints the arms by, by variant key, which replace their variants' names, as
        :func:`aspect_verdict` reads them; None names each arm by its variant.
    :ptype arm_labels: Mapping[str, str] | None
    :return: ``cell_ref -> name`` for every cell on the bundle's surface.
    :rtype: dict[str, str]
    """
    surface = bundle_decision_surface(bundle)
    name = _cell_namer(surface, _arm_names(bundle, labels=arm_labels))
    return {
        cell_ref(cell.variant_key, cell.apparatus_class_id): name(cell.variant_key, cell.apparatus_class_id)
        for cell in surface.cells
    }


__all__ = [
    "aspect_verdict",
    "contrast_evidence",
    "contrast_rows",
    "contrast_verdicts",
    "report_cell_labels",
]
