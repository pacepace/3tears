"""The last screen: what the campaign could not speak to (**Not tested**), then the **Methods** table and the
engine's disclosures.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from itertools import chain

from threetears.evals.analysis.lens_text import tier_sentence_in
from threetears.evals.analysis.bundle.schema import (
    AliasedFactors,
    AnalysisContextBundle,
    CellCoordinate,
    FactorPairScan,
    GoalCheckProofReading,
    UnexercisedFactor,
)
from threetears.evals.analysis.lenses.frontier import PASS_HAT_K_CELL_READING, pass_hat_k_cell_lines
from threetears.evals.analysis.report.aspect.common import (
    _case_namer,
    _control_key,
    _date,
    _listed,
    _namer,
    _readings_table,
)
from threetears.evals.analysis.report.campaign.names import measure_name_of
from threetears.evals.analysis.report.campaign.provenance import calibration_blocks, setup_disclosure
from threetears.evals.analysis.report.cost_unit import PRICED_TRIALS_TEMPLATE, money_ruler
from threetears.evals.analysis.report.forms import PageForms
from threetears.evals.analysis.report.model import (
    CellValue,
    DisclosureBlock,
    DisclosureSource,
    ReportBlock,
    ReportSource,
    TableBlock,
    TableCell,
    TableColumn,
)
from threetears.evals.analysis.report.verdict import arm_exclusions, cause_short, loss_words
from threetears.evals.analysis.style import judged_name
from threetears.evals.analysis.style import (
    CANNOT_TELL_APART,
    GOAL_CHECK,
    ArmName,
    code,
    lever_reader_name,
    measure_name,
)
from threetears.evals.kernel.metrics import METRIC_DESCRIPTORS, RUN_MARGIN_MEASURES, goal_check_of
from threetears.evals.kernel.surface import DecisionSurface, TimeAxis
from threetears.evals.schema.models import (
    CHECK_REFUSED_UNDER_CURRENT_GRAMMAR,
    ClientRequestSettings,
    sent_settings_phrase,
)

#: What the methods say of a report no analysis was written for.
NO_ANALYSIS = "none written yet; generating an analysis writes one from this same evidence"


#: What the methods say of the analysis inside a written report's evidence record: the report around it is the
#: analysis, and its last line says how it was written.
IN_THE_WRITTEN_REPORT = "this written report, from this same evidence"


def _unpriced_cost_line(bundle: AnalysisContextBundle, arm: Callable[[str], str]) -> str | None:
    """Where a trial's spend went unpriced, the arm's cost is a floor; None where every trial was priced."""
    unpriced = [board for board in bundle.scoreboard if board.n_cost_unpriced]
    if not unpriced:
        return None
    return (
        "The cost is a floor where a trial's spend went unpriced: "
        + "; ".join(f"{arm(board.variant_key)}, {board.n_cost_unpriced} of {board.ran}" for board in unpriced)
        + "."
    )


def _question_words(bundle: AnalysisContextBundle) -> str:
    """What the methods call the aspect's design: a question declared, or exploratory. When a criterion was set is
    said once, on the first screen."""
    return "question declared" if bundle.question_declared else "exploratory"


def _not_tested_blocks(
    bundle: AnalysisContextBundle,
    surface: DecisionSurface,
    arm: Callable[[str], str],
    *,
    also: Sequence[ReportBlock] = (),
) -> list[ReportBlock]:
    """Everything the campaign could not speak to, one line each — a list, never a score.

    ``also`` is the page's unasked readings (:func:`_question_blocks`), laid out after code's own lines and before
    the folded disclosures.
    """
    said: list[str] = []
    for row in bundle.coverage:
        lever = lever_reader_name(row.name, bundle.lever_names)
        if row.status == "unswept":
            # Said as what it means, never the word: one level ran, so nothing is known of changing it.
            said.append(f"One {lever} ran, so this page says nothing of changing the {lever}.")
        elif row.status == "thin":
            said.append(f"Each {lever} ran too few repeats to read what changing the {lever} does.")
    said += [_unexercised_sentence(factor, bundle, arm) for factor in bundle.unexercised_factors]
    usable = {(trial.arm, trial.case_id) for trial in bundle.trials if trial.state != "lost"}
    case_name = _case_namer(bundle)
    for board in bundle.scoreboard:
        unusable = sorted(
            {
                trial.case_id
                for trial in bundle.trials
                if trial.arm == board.variant_key and (trial.arm, trial.case_id) not in usable
            },
            key=case_name,
        )
        if unusable:
            named = _listed([case_name(case) for case in unusable])
            said.append(f"{arm(board.variant_key)} has no usable trial of {named}.")
    for proof in bundle.goal_check_proofs:
        if proof.proof != "proven":
            said.append(_unproven_check_sentence(proof, bundle.named_checks().get(proof.check)))
    blocks: list[ReportBlock] = [DisclosureBlock(section="not_tested", source="not_tested", text=text) for text in said]
    blocks += _no_margin_blocks(bundle)
    blocks += calibration_blocks(bundle)
    blocks += also
    quoted = [bundle.held_fixed_reading.disclosure] if bundle.held_fixed_reading.disclosure else []
    if bundle.cost_unmeasured:
        quoted.append(bundle.cost_unmeasured + _some_arms(bundle, bundle.cost_unmeasured_cells, "Unmeasured", arm))
    blocks += [
        DisclosureBlock(section="not_tested", source="not_tested", text=text, fold="Disclosures") for text in quoted
    ]
    if not blocks:
        return [DisclosureBlock(section="not_tested", source="not_tested", text="Nothing declared went untested.")]
    return blocks


def _unexercised_sentence(factor: UnexercisedFactor, bundle: AnalysisContextBundle, arm: Callable[[str], str]) -> str:
    """A lever no comparison exercised, in one line: ``tool model: no tool call was made on either arm.``

    Calls in one arm only say so: ``observed in one arm only``.
    """
    # The one rule a lever is named by: its stored name, else its id read as words (``lever_id_words``).
    name = lever_reader_name(factor.factor, bundle.lever_names)
    call = f"{factor.role.replace('_', ' ')} call"
    arms = len(bundle.scoreboard)
    if factor.results == 0 or not factor.arms:
        where = "the arm" if arms <= 1 else "either arm" if arms == 2 else "any arm"
        return f"{name}: no {call} was made on {where}."
    others = "the other arm" if arms == 2 else "the other arms"
    if factor.results > 1:
        return (
            f"{name}: {factor.results} results made {call}s, all on {arm(factor.arms[0])}, and none on {others}, "
            "so it was observed in one arm only and is not a level."
        )
    if arms <= 1:
        return f"{name}: one {call} was made in all its trials, too few to read a level from."
    return f"{name}: one {call} was made, on {arm(factor.arms[0])}, and none on {others}, so it is not a level."


#: What a goal check not shown to beat doing nothing is, by its proof, as the page and the print both say it.
CHECK_PROOF_WORDS: dict[str, str] = {
    "refuted": "refuted: its control shows it cannot tell acting from doing nothing",
    "unproven": "unproven: no control shows it can tell acting from doing nothing",
}


#: What follows from a goal check not shown to beat doing nothing, as the page and the print both say it.
NOT_SHOWN_TO_MEASURE = "a candidate that did nothing might pass it, so it is not shown to measure the behaviour"


def _unproven_check_sentence(proof: GoalCheckProofReading, declared: str | None) -> str:
    """A goal check not shown to beat doing nothing: what it is, and what its pass rate is not. Named by its name on the
    aspect (``Refund amount check``; a declared name that does not say it is a check is said as one, ``Goal check
    Total right``), never by its expression, which the glossary defines it by."""
    named = declared if declared is not None else GOAL_CHECK
    check = named if re.search(r"\bcheck\b", named, re.IGNORECASE) else f"{GOAL_CHECK} {named}"
    if proof.refused is not None:
        return (
            f"{check} is {CHECK_REFUSED_UNDER_CURRENT_GRAMMAR} ({proof.refused}), so no arm was graded on "
            "it. Any pass rate it has predates the rule."
        )
    notes = []
    if proof.unrecorded:
        notes.append(f"{proof.unrecorded} of {proof.runs} runs recorded no proof")
    if proof.stale:
        notes.append(f"{proof.stale} of {proof.runs} runs proved it under an earlier rule; relaunch to re-prove")
    tail = f" ({'; '.join(notes)})" if notes else ""
    words = CHECK_PROOF_WORDS["refuted" if proof.proof == "refuted" else "unproven"]
    return f"{check} is {words}{tail}. {NOT_SHOWN_TO_MEASURE[:1].upper()}{NOT_SHOWN_TO_MEASURE[1:]}."


def _no_margin_blocks(bundle: AnalysisContextBundle) -> list[ReportBlock]:
    """The measures tested with no margin to be equivalent on: one sentence, then each once with how it takes one."""
    catalog = bundle.measure_catalog
    names = list(
        dict.fromkeys(
            tested.name
            for family in bundle.multiple_comparisons.families
            for tested in family.comparisons
            if tested.name in catalog and tested.reading == "measure" and tested.margin_source is None
        )
    )
    if not names:
        return []

    def declaring(name: str) -> str:
        if name not in METRIC_DESCRIPTORS:
            ranged = catalog[name].family == "mechanical" and catalog[name].value_range is None
            # A check is graded by its scenario, not a grader the caller names: it takes its margin where declared.
            where = "on its check" if goal_check_of(name) is not None else "on its grader"
            return f"a margin {where}" + (", with the range its values take" if ranged else "")
        if name in RUN_MARGIN_MEASURES:
            return "a margin on the classifier's field's grader, on every arm's run"
        return "cannot: the engine owns its description"

    rows: list[dict[str, CellValue]] = [
        {
            "reading": measure_name(name, catalog[name].reader_name, declared=bundle.check_name(name)),
            "declare": declaring(name),
        }
        for name in names
    ]
    return [
        DisclosureBlock(
            section="not_tested",
            source="not_tested",
            text=(
                f"No margin is declared on these, so none can read equivalent: {CANNOT_TELL_APART} never means the "
                "arms are alike."
            ),
        ),
        _readings_table("no_margin", "No margin", rows),
    ]


def _method_blocks(
    bundle: AnalysisContextBundle,
    surface: DecisionSurface,
    arm: Callable[[str], str],
    names: dict[str, ArmName],
    forms: PageForms,
    *,
    source: ReportSource,
    writer: str | None,
    left_disclosures: Sequence[ReportBlock] = (),
) -> list[ReportBlock]:
    """How everything on the page was measured and made: a table, then the engine's disclosures and those the
    contrasts and charts left for the methods (``left_disclosures``) in one closed fold."""
    blocks: list[ReportBlock] = [_methods_table(bundle, names, forms, source=source, writer=writer)]

    def say(source_of: DisclosureSource, text: str | None) -> None:
        if text and text.strip():
            blocks.append(DisclosureBlock(section="methods", source=source_of, text=text.strip(), fold="Disclosures"))

    for index, sentence in enumerate(_judge_sentences(bundle, forms)):
        # The first judge's line links the judged score, so the glossary states its scale once.
        blocks.append(
            DisclosureBlock(
                section="methods",
                source="measurement",
                text=sentence,
                fold="Disclosures",
                term="judged score" if index == 0 else None,
            )
        )
    if (settings := setup_disclosure("settings", bundle, arm, forms=forms)) is not None:
        blocks.append(settings)
    for variant_key, cells in arm_exclusions(bundle).items():
        if truncated := sum(cell.by_cause.get("truncated", 0) for cell in cells):
            say(
                "measurement",
                f"{arm(variant_key)}: {truncated} of {sum(cell.n_trials for cell in cells)} trials lost because the "
                f"judge's reply was cut off at its output cap ({loss_words('truncated', exclusions=cells).words}).",
            )
    for flagged in bundle.differential_exclusions:
        others = ", ".join(
            f"{forms.loss_share(share).text} of {arm(key)}'s" for key, share in sorted(flagged.other_arms.items())
        )
        say(
            "measurement",
            f"Differential loss: {forms.loss_share(flagged.share).text} of {arm(flagged.variant_key)}'s trials were "
            f"lost to {cause_short(flagged.cause)}"
            + (f", against {others}" if others else "")
            + f"; above {forms.loss_share(flagged.threshold).text}, the arms are not compared on the same trials.",
        )
    say("measurement", forms.stored_counts(bundle.model_identity_disclosure or ""))
    for reading in bundle.arm_served_models:
        if reading.state == "pooled":
            say(
                "measurement",
                f"Arm {arm(reading.variant_key)} asked for one model and its candidate was answered by "
                f"{_listed(reading.served_models)}, so its numbers mix those models.",
            )
    say("measurement", bundle.launch_disclosure)
    say("cost", _unpriced_cost_line(bundle, arm))
    if (routed := setup_disclosure("cost", bundle, arm, forms=forms)) is not None:
        blocks.append(routed)
    if (latency := setup_disclosure("latency", bundle, arm, forms=forms)) is not None:
        blocks.append(latency)
    for _, sentence in sorted(bundle.short_runs.items()):
        say("runs", sentence)
    for negative in bundle.negative_values:
        say("measurement", negative.sentence(measure_name_of(bundle, negative.measure), arm(negative.variant_key)))
    _run_disclosures(bundle, say, arm, forms)
    if (footing := _production_footing_sentence(bundle, arm)) is not None:
        say("measurement", footing)
    _design_disclosures(bundle, say, arm)
    _time_axis_disclosures(surface.time_axis, bundle.time_axis_withheld, say)
    blocks.extend(left_disclosures)
    return blocks


def _methods_table(
    bundle: AnalysisContextBundle,
    names: dict[str, ArmName],
    forms: PageForms,
    *,
    source: ReportSource,
    writer: str | None,
) -> TableBlock:
    """Models, judge, settings sent, cost, the analysis's writer and the bundle, one row each."""
    rows: list[tuple[str, str]] = []
    control = _control_key(bundle)
    keys = [board.variant_key for board in bundle.scoreboard]
    for key in ([control] if control is not None else []) + [key for key in keys if key != control]:
        rows.append(("Arm", names[key].first() if key in names else _namer(names)(key)))
    for level in bundle.judge_change.levels:
        rows.append(("Judge", level.judge_model))
        rows.append(
            (
                "Judge temperature",
                "not sent" if level.judge_temperature is None else forms.declared(level.judge_temperature).text,
            )
        )
        if level.judge_sent_settings:
            rows.append(("Judge settings sent", "; ".join(_settings_words(sent) for sent in level.judge_sent_settings)))
    # What each arm's trials cost: totals, which count nothing and are never scaled, in their one form (#993).
    totals = money_ruler((board.test_cost_usd for board in bundle.scoreboard), per=0)
    # A total that counts only part of its trials' spend is a floor, and says so beside itself (#993).
    costs = [
        f"{_namer(names)(board.variant_key)} {forms.money_on(totals, board.test_cost_usd).text}"
        + (
            f" ({PRICED_TRIALS_TEMPLATE.format(priced=board.ran - board.n_cost_unpriced, trials=board.ran)})"
            if board.n_cost_unpriced
            else ""
        )
        for board in bundle.scoreboard
        if board.test_cost_usd is not None
    ]
    if costs:
        rows.append(("Test cost", "; ".join(costs)))
    rows.append(("Dataset", ", ".join(ref.label for ref in bundle.case_sets) or "each run's own cases"))
    rows.append(
        (
            "Design",
            _question_words(bundle)
            + (f", frozen {_date(bundle.design_frozen_at)}" if bundle.design_frozen_at else ", not frozen"),
        )
    )
    rows.append(("Analysis", writer or NO_ANALYSIS))
    rows.append(("Date", _date(source.generated_at)))
    if unshown := _uncarried(bundle):
        rows.append(("Readings not shown", f"{len(unshown)} carried by no trial: {_listed(unshown)}"))
    rows.append(
        (
            "Evidence",
            # The build as one token (``3tears-evals <version>,``), which a reader and the oracle read it by; an HTML
            # page holds the package's hyphenated name whole in a span, as every prose cell holds such a name.
            f"3tears-evals {bundle.package_version or 'version not recorded'}, {len(bundle.run_ids)} runs; its JSON "
            "beside this page",
        )
    )
    # The digest a reader checks the JSON against: an id, so a cell of its own, as code, never in a sentence.
    rows.append(("Fingerprint", code(source.bundle_fingerprint)))
    return TableBlock(
        section="methods",
        name="methods",
        title="Methods",
        columns=[TableColumn(key="item", header="Item"), TableColumn(key="value", header="Value")],
        rows=[{"item": TableCell(value=item), "value": TableCell(value=value)} for item, value in rows],
        order="the arms, the judge, the cost, then how the page was made",
        total_rows=len(rows),
    )


#: Where a judge sentence names its dimensions, while its judges are grouped by everything else they say.
_DIMS = "\x00"


def _judge_sentences(bundle: AnalysisContextBundle, forms: PageForms) -> list[str]:
    """Each judge's tier, one sentence per judge that reads alike: the dimensions it holds for named together, its
    settings as words, never ``key=value``, and its prompt's id left to the bundle's JSON, which names it by dimension
    (``tier_sentence_in`` states one dimension in full)."""
    alike: dict[str, list[str]] = {}
    for tier in bundle.judge_evidence_tiers:
        said = tier_sentence_in(tier.model_copy(update={"rubric_dim": _DIMS, "judge_config_id": None}), forms)
        said = said.replace(sent_settings_phrase(tier.judge_sent_settings), _sent_words(tier.judge_sent_settings))
        # The dimension by the name every other surface gives it, never its stored id.
        if (dim := judged_name(tier.rubric_dim)) not in alike.setdefault(said, []):
            alike[said].append(dim)
    return ["Judge: " + said.replace(_DIMS, _listed_words(dims), 1) for said, dims in alike.items()]


def _sent_words(sent: ClientRequestSettings | None) -> str:
    """A judge's sent settings as the clause its sentence carries, in words."""
    return "request settings unconfirmed" if sent is None else f"sent {_settings_words(sent)}"


def _settings_words(sent: ClientRequestSettings) -> str:
    """Request settings as words, never as ``key=value``."""
    parts = [f"max tokens {sent.max_tokens}"]
    if sent.reasoning_max_tokens is not None:
        parts.append(f"reasoning max tokens {sent.reasoning_max_tokens}")
    if sent.reasoning_effort is not None:
        parts.append(f"reasoning effort {sent.reasoning_effort}")
    if sent.strict_output:
        parts.append("strict output")
    return ", ".join(parts)


def _uncarried(bundle: AnalysisContextBundle) -> list[str]:
    """Each reading the campaign names that no arm carries on any trial, or that one side of a contrast never carried."""
    carried = {summary.name for cell in bundle.cell_measures for summary in cell.measures.measures if summary.n}
    names = [name for name in bundle.measure_catalog if name not in carried]
    names += [
        comparison.name
        for family in bundle.multiple_comparisons.families
        for comparison in family.comparisons
        if min(comparison.control.n_cases, comparison.contrast.n_cases) == 0
    ]
    catalog = bundle.measure_catalog
    return list(
        dict.fromkeys(
            measure_name(name, catalog[name].reader_name if name in catalog else None, declared=bundle.check_name(name))
            for name in names
        )
    )


def _run_disclosures(
    bundle: AnalysisContextBundle,
    say: Callable[[DisclosureSource, str | None], None],
    arm: Callable[[str], str],
    forms: PageForms,
) -> None:
    """The member runs left out, unfinished, short, or of unknown completeness, and every run an arm's pass^k kept out
    of its headline cell (:func:`~threetears.evals.analysis.lenses.frontier.pass_hat_k_cell_lines`),
    each line led by the arm, as the frontier's text leads it, its figures in the page's ``forms``."""
    if not bundle.run_ids:
        say("runs", "The campaign resolved no runs in this scope, so no table or chart here holds anything.")
    if bundle.unresolved_run_ids:
        say(
            "runs",
            f"{len(bundle.unresolved_run_ids)} run(s) the campaign names were not found in this scope and are in no "
            f"table here: {_listed(bundle.unresolved_run_ids)}.",
        )
    if bundle.archived_run_ids:
        say(
            "runs",
            f"{len(bundle.archived_run_ids)} member run(s) were archived and are held out of everything here: "
            f"{_listed(bundle.archived_run_ids)}.",
        )
    if bundle.incomplete_runs:
        say(
            "runs",
            f"{len(bundle.incomplete_runs)} run(s) did not complete and are pooled as far as they got: "
            + "; ".join(f"{run_id} ({status})" for run_id, status in sorted(bundle.incomplete_runs.items()))
            + ".",
        )
    if bundle.completeness_unknown_run_ids:
        say(
            "runs",
            f"Whether {len(bundle.completeness_unknown_run_ids)} run(s) came up short is unknown — they carry no "
            f"completeness record: {_listed(bundle.completeness_unknown_run_ids)}.",
        )
    for cell in bundle.short_cells:
        say("runs", cell.sentence)
    cell_lines = [
        f"{arm(point.variant_key)}: {forms.said(line)}."
        for subject in bundle.frontier.subjects
        for point in subject.points
        for reading in point.pass_hat_k_cells
        for line in pass_hat_k_cell_lines(reading)
    ]
    if cell_lines:
        # What "kept out" covers, once, before the first: the arm's tables above still count those runs.
        say("runs", PASS_HAT_K_CELL_READING)
    for line in cell_lines:
        say("runs", line)


def _design_disclosures(
    bundle: AnalysisContextBundle, say: Callable[[DisclosureSource, str | None], None], arm: Callable[[str], str]
) -> None:
    """How the arms were set up and compared: the control, the confounds, the factors, the declared crossing."""
    say("comparisons", bundle.design.stale_control_disclosure)
    say("comparisons", bundle.design.earlier_control.disclosure if bundle.design.earlier_control else None)
    say("comparisons", _verdict_order_sentence(bundle))
    for instability in bundle.subject_key_instabilities:
        if instability.kind == "one_key_many_labels":
            text = f"Subject key {instability.key} is labelled {len(instability.counterparts)} ways: "
        else:
            text = f"Subject label {instability.label} names {len(instability.counterparts)} keys: "
        say("measurement", text + _listed(instability.counterparts) + ".")
    if bundle.apparatus_confounds:
        say(
            "apparatus",
            "The test setup did not hold still across the campaign: "
            + "; ".join(f"{confound.dimension} ({confound.status})" for confound in bundle.apparatus_confounds)
            + ".",
        )
    unverified = sorted(
        {
            confound.dimension
            for confounds in chain(
                (row.confounded_by for row in bundle.coverage),
                (divergence.confounded_by for divergence in bundle.scope_divergences),
            )
            for confound in confounds
            if confound.kind == "unverified_fold"
        }
    )
    for dimension in unverified:
        reason = bundle.confound_catalog[dimension]
        say("comparisons", reason[:1].upper() + reason[1:] + ".")
    # The bundle holds the factor reads as facts, by id; each sentence is worded here, every factor by its reader name.
    for group in bundle.aliased_factors:
        say("comparisons", _aliased_sentence(group, bundle))
    if bundle.factor_pairs is not None and bundle.factor_pairs.factors:
        say("comparisons", f"{_pair_scan_completeness(bundle.factor_pairs)} {bundle.factor_pairs.interaction_aliasing}")
        for pivot in bundle.factor_pairs.pivots:
            holes = [f"{cell.row_level} with {cell.column_level}" for cell in pivot.cells if cell.status == "not_run"]
            if holes:
                pair = _listed_words(
                    [
                        lever_reader_name(factor, bundle.lever_names)
                        for factor in (pivot.row_factor, pivot.column_factor)
                    ]
                )
                say(
                    "comparisons",
                    f"{pair[:1].upper()}{pair[1:]} co-vary; {len(holes)} of their {len(pivot.cells)} combinations "
                    f"never ran: {_listed(holes)}.",
                )
    if bundle.declared_crossing is not None:
        say("comparisons", bundle.declared_crossing.sentence)
        for state, lead in (("not_run", "Declared arms never run"), ("skipped_by_design", "Arms skipped by design")):
            # Each level by the name the declaration gives it, read here: the bundle keeps each by its content hash.
            cells = [
                " × ".join(bundle.level_name(axis, level) for axis, level in cell.levels.items())
                for cell in bundle.declared_crossing.cells
                if cell.state == state
            ]
            if cells:
                say("comparisons", f"{lead}: {'; '.join(cells)}.")
    for row in bundle.coverage:
        lever = lever_reader_name(row.name, bundle.lever_names)
        not_run = [level.display for level in row.declared_levels if level.state == "not_run"]
        undetermined = [level.display for level in row.declared_levels if level.state == "undetermined"]
        if not_run:
            say("comparisons", f"Declared on axis {lever} and never run: {_listed(not_run)}.")
        if undetermined:
            say(
                "comparisons",
                f"Declared on axis {lever}, and whether any run sat at it cannot be established: "
                f"{_listed(undetermined)}.",
            )
        if row.cannot_be_an_arm:
            why = row.cannot_be_an_arm.rstrip(".")
            say("comparisons", f"Declared axis {lever} reads unswept whatever the runs did: {why}.")
    for merge in bundle.refused_merges:
        dimensions = f" on {_listed(merge.dimensions)}" if merge.dimensions else ""
        say(
            "apparatus",
            f"Arm {arm(merge.variant_key)} was measured under {len(merge.apparatus_class_ids)} test setups that did "
            f"not pool ({merge.reason.replace('_', ' ')}{dimensions}).",
        )
    if bundle.refused_merges_omitted:
        say(
            "apparatus",
            f"{bundle.refused_merges_omitted} more pair(s) of one arm's test setups did not pool; they are left out of "
            "this list, the smallest first.",
        )


def _time_axis_disclosures(
    axis: TimeAxis | None, withheld: str | None, say: Callable[[DisclosureSource, str | None], None]
) -> None:
    """What the time axis's positions are, when the runs span two builds or two days."""
    say("measurement", withheld)
    if axis is None:
        return
    if axis.basis == "release":
        say("measurement", f"The runs span builds of {axis.release_label}, in the order each first ran.")
    else:
        say("measurement", f"The runs span UTC days, not builds: {axis.basis_reason}.")
    for position in axis.positions:
        say("measurement", position.control_disclosure)


def _verdict_order_sentence(bundle: AnalysisContextBundle) -> str | None:
    """The tie-break order the campaign declared over its merit axes, with the bars each tier holds — or nothing."""
    order = bundle.verdict_order
    if not order.merit_priority:
        return None
    tiers = "; ".join(
        f"{tier.axis} ({_listed(tier.bar_measure_ids) if tier.bar_measure_ids else 'no adjudicated bar'})"
        for tier in order.tiers
    )
    unranked = (
        f" Bars on no ranked axis, which the order does not place: {_listed(order.unranked_bar_measure_ids)}."
        if order.unranked_bar_measure_ids
        else ""
    )
    return f"The campaign ranks its merit axes, strongest first, to break a tie no bar decides: {tiers}.{unranked}"


def _production_footing_sentence(bundle: AnalysisContextBundle, arm: Callable[[str], str]) -> str | None:
    """Say what the arms' runs set away from the subject's production configuration; None when nothing moved.

    The settings left unchecked are said once for the arms that share them (``every arm`` where all do), so a long
    list never prints once per arm; then each arm's own moves in one short clause each."""
    unchecked_by: dict[tuple[str, ...], list[str]] = {}
    own: list[str] = []
    arms_held = sorted(bundle.arm_production_footings.items())
    for variant_key, pooled in arms_held:
        footings = list(pooled.runs.values())
        # Each setting by the name every page calls its lever (`lever_names`), never by its id.
        settings = {
            (lever_reader_name(lever, bundle.lever_names), level)
            for f in footings
            if f is not None
            for lever, level in f.moved.items()
        }
        moved = sorted({f"{name} to {level}" for name, level in settings})
        unchecked = tuple(
            sorted(
                {lever_reader_name(name, bundle.lever_names) for f in footings if f is not None for name in f.unchecked}
            )
        )
        unread = sum(1 for f in footings if f is None)
        if unchecked:
            unchecked_by.setdefault(unchecked, []).append(arm(variant_key))
        said = [
            *([f"set {', '.join(moved)}"] if moved else []),
            *([f"{unread} run(s) nobody checked"] if unread else []),
        ]
        if said:
            own.append(f"arm {arm(variant_key)} {' and '.join(said)}")
    if not unchecked_by and not own:
        return None
    parts = [
        f"{'every arm' if len(names) == len(arms_held) and len(names) > 1 else _arm_words(names)} left unchecked "
        f"whether {', '.join(unchecked)} is production's"
        for unchecked, names in unchecked_by.items()
    ]
    return (
        "A per-reply cost is what production would spend only where its arm's runs moved nothing off the subject's "
        "production configuration: " + "; ".join([*parts, *own]) + "."
    )


def _arm_words(names: Sequence[str]) -> str:
    """``arm a`` or ``arms a and b``."""
    return f"{'arm' if len(names) == 1 else 'arms'} {_listed_words(names)}"


def _aliased_sentence(group: AliasedFactors, bundle: AnalysisContextBundle) -> str:
    """The one sentence naming a group of factors that moved in lockstep, each by the name every page calls it
    (:func:`~threetears.evals.analysis.style.lever_reader_name` over ``lever_names``), worded from the group's facts."""
    names = _listed_words([lever_reader_name(factor, bundle.lever_names) for factor in group.factors])
    across = (
        f"all {group.n_runs} runs"
        if group.n_runs == len(bundle.run_ids)
        else f"the {group.n_runs} runs that recorded them"
    )
    return (
        f"{names[:1].upper()}{names[1:]} move together across {across}, splitting them into the same {group.n_levels} "
        f"groups; no comparison separates them, so a difference across them belongs to all {len(group.factors)} at "
        "once."
    )


def _pair_scan_completeness(scan: FactorPairScan) -> str:
    """What the factor-pair scan examined and what it shows, worded from its counts."""
    factors = len(scan.factors)
    if factors < 2:
        return f"{factors} factor varied, so no pair of factors could co-vary."
    outside = scan.n_covarying - scan.n_covarying_in_groups
    shown = len(scan.pivots) + scan.pivots_omitted
    return (
        f"{scan.n_pairs_examined} factor pair(s) examined over the {factors} factors that varied; "
        f"{scan.n_covarying} co-vary"
        + (
            f", {scan.n_covarying_in_groups} of them inside a group that moves in lockstep"
            if scan.n_covarying_in_groups
            else ""
        )
        + (
            f"; the {outside} outside any group are shown in {shown} pivot(s), a group's members sharing one"
            if outside
            else ""
        )
        + (
            f", of which {scan.pivots_omitted} with the fewest unrun combinations are left out"
            if scan.pivots_omitted
            else ""
        )
        + "."
    )


def _listed_words(names: Sequence[str]) -> str:
    """Names as a sentence lists them: ``a``, ``a and b``, ``a, b and c``."""
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"


def _some_arms(
    bundle: AnalysisContextBundle, cells: Sequence[CellCoordinate], lead: str, arm: Callable[[str], str]
) -> str:
    """`` <lead>: <arms>.`` naming the arms a sentence holds for, or nothing when it holds for every arm."""
    keys = sorted({cell.variant_key for cell in cells})
    if set(keys) >= {board.variant_key for board in bundle.scoreboard}:
        return ""
    return f" {lead}: {'; '.join(arm(key) for key in keys)}."


__all__ = [
    "CHECK_PROOF_WORDS",
    "IN_THE_WRITTEN_REPORT",
    "NOT_SHOWN_TO_MEASURE",
    "NO_ANALYSIS",
]
