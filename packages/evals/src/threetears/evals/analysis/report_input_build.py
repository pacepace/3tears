"""The report input built from a campaign's own bundle, and the planner's budgeted view of it.

:func:`report_input` reads a :class:`~threetears.evals.analysis.campaign_bundle.CampaignBundle` and returns the
:class:`~threetears.evals.analysis.report_input.ReportInput` the report planner and the renderer read. It is the
package's port of the report kit's adapter (``docs/eval-report-kit/adapters/bundle_to_report.py``), reading the
bundle's typed objects where the kit read a writer's message: no figure is parsed out of prose.

**Statistics are copied through, never recomputed.** Every reading, interval, contrast, floor and bar status, judge
tier and per-arm figure is the pipeline's own. Two things here are orderings, not figures: which failed trials are
shown as examples, and in what order (where the candidates lose most first), and which strata come first (most cases
first).

**Every arm, reading, contrast and caveat the bundle holds reaches the input.** An arm is named by its campaign label
everywhere. A choice's pair readings sit on their aspect, a measurement's readings on its measures, and a contrast
the pipeline could not test is carried with its reason rather than dropped. On more than one aspect the by-arm maps
read the lead aspect (the first asked, else the first) and every other aspect's figures sit beside them.

**What the input does not carry.** A categorical or free-text measure is described, with no figure: it has none to
copy. Where one arm ran an aspect under more than one rig, its figures are its first rig's, and ``caveats.unreported``
says so.

:func:`planner_view` is what the planner reads instead of the whole input: the input as compact JSON, held to a
budget of :data:`~threetears.evals.analysis.campaign_bundle.CAMPAIGN_MESSAGE_CHARS_PER_ASPECT` per aspect. Over it,
code trims in a fixed order and says what it cut (``trimmed``); figure keys still resolve against the whole input.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from threetears.evals.analysis.arms import writer_arms
from threetears.evals.analysis.bundle.caps import capped_text
from threetears.evals.analysis.bundle.schema import AnalysisContextBundle, TrialRow
from threetears.evals.analysis.campaign_bundle import CAMPAIGN_MESSAGE_CHARS_PER_ASPECT, KIND_MEANS, CampaignBundle
from threetears.evals.analysis.campaign_readings import AspectReading, PairReading
from threetears.evals.analysis.depth import REPEATS_FLOOR
from threetears.evals.analysis.depth import Precision as BundlePrecision
from threetears.evals.analysis.depth_reading import aspect_depth
from threetears.evals.analysis.numbers import format_number
from threetears.evals.analysis.report_input import (
    ApparatusConfound,
    Arm,
    Aspect,
    BarRead,
    Caveats,
    Contrast,
    ContrastFamily,
    Criterion,
    Depth,
    Example,
    FloorRead,
    Judged,
    LateCriterion,
    Measure,
    Meta,
    PairRead,
    PassK,
    Precision,
    PriorInsight,
    Reading,
    ReportInput,
    ScoreboardRow,
    ShortRun,
    Stat,
    Stratum,
)
from threetears.evals.analysis.style import measure_name
from threetears.evals.kernel.campaign import VariantIndexEntry
from threetears.evals.kernel.errors import RefusedInput
from threetears.evals.kernel.host.sweepables import CANDIDATE_MODEL_LEVER
from threetears.evals.kernel.metrics import goal_check_of
from threetears.evals.kernel.surface import CellFacts

__all__ = ["planner_view", "report_input"]

#: How many failed trials the input shows as examples, at most.
EXAMPLES = 12

#: How much of a shown trial's excerpt the input keeps: the kit's cut.
EXAMPLE_EXCERPT_CHARS = 420

#: What a measure's definition is cut to, as the writer's view cuts it.
MEANING_CHARS = 200

#: The id of the stratum holding the cases that declare none.
UNSTRATIFIED = "unstratified"

#: What a planner's view cuts each shown trial's excerpt to, first of all its trims.
PLANNER_EXCERPT_CHARS = 200

#: The examples a planner's view keeps once it has to cut them.
PLANNER_EXAMPLES = 4

#: The strata a planner's view keeps once it has to cut them, the largest first.
PLANNER_STRATA = 12

#: The significant figures a figure in the planner's view keeps: the planner cites figures by key, never by value.
PLANNER_FIGURES = 4

#: A JSON escape left in a transcript's text, where a tool call's arguments were written as JSON: a surrogate pair, one
#: code point, or a one-character escape.
_ESCAPE = re.compile(r'\\u[dD][89abAB][0-9a-fA-F]{2}\\u[dD][c-fC-F][0-9a-fA-F]{2}|\\u[0-9a-fA-F]{4}|\\[nrt"\\/]')

#: The comparison verdicts that separate an arm from the control or show it alike: listed first.
_SEPARATED = frozenset({"improved", "regressed", "equivalent"})


# ---------------------------------------------------------------------- the report input


def report_input(bundle: CampaignBundle, *, provenance: str | None = None, examples: int = EXAMPLES) -> ReportInput:
    """The report input for one campaign, read off its bundle.

    :param bundle: The campaign's bundle.
    :ptype bundle: CampaignBundle
    :param provenance: Where the bundle came from, as the masthead's provenance line says it; None says nothing.
    :ptype provenance: str | None
    :param examples: The most failed trials to show.
    :ptype examples: int
    :return: The input.
    :rtype: ReportInput
    """
    return _Builder(bundle).build(provenance=provenance, examples=examples)


class _Builder:
    """Reads one campaign bundle into a report input, naming every arm by its campaign label."""

    def __init__(self, bundle: CampaignBundle) -> None:
        self.bundle = bundle
        keys = {entry.variant_key for aspect in bundle.aspects for entry in aspect.variant_index}
        keys |= {key for arm in bundle.arms for key in bundle.arm_keys(arm)}
        self.labels: dict[str, str] = {key: bundle.label_of(key) for key in sorted(keys)}
        self.trial_ids = {where: trial for trial, where in bundle.trial_ids().items()}
        asked = [aspect for aspect in bundle.aspects if aspect.question_declared]
        #: The aspect the by-arm maps read: the first the campaign asks about, else the first.
        self.lead: AnalysisContextBundle = asked[0] if asked else bundle.aspects[0]
        #: The aspects in the order the input reads them: the lead first, then the rest in the campaign's order.
        self.ordered = [self.lead, *(aspect for aspect in bundle.aspects if aspect is not self.lead)]
        self.unreported: list[str] = []

    def arm(self, variant_key: str) -> str:
        """The label of the arm ``variant_key`` names."""
        if variant_key not in self.labels:
            self.labels[variant_key] = self.bundle.label_of(variant_key)
        return self.labels[variant_key]

    def build(self, *, provenance: str | None, examples: int) -> ReportInput:
        """The whole input."""
        bundle = self.bundle
        judged = self.judged()
        primary = self.primary_judged(judged)
        strata = self.strata()
        return ReportInput.model_validate(
            {
                "schema": "report-input/1",
                "meta": self.meta(provenance),
                "arms": self.arms(),
                "aspects": [self.aspect(aspect) for aspect in bundle.aspects],
                "floors": self.floors(),
                "bars": self.bars(),
                "guardrails": self.guardrails(),
                "criteria": self.criteria(),
                "scoreboard": self.scoreboard(self.lead),
                "measures": self.measures(),
                "judged": judged,
                "primary_judged": primary,
                "contrasts": self.contrasts(),
                "contrast_families": self.families(),
                "strata": strata,
                "examples": self.examples(primary, strata, examples),
                "caveats": self.caveats(),
                "prior_insights": [
                    PriorInsight(statement=insight.statement, confidence=insight.confidence)
                    for insight in bundle.aspects[0].prior_insights
                ],
            }
        )

    # ------------------------------------------------------------------ what the campaign was

    def meta(self, provenance: str | None) -> Meta:
        """The masthead's facts: the campaign, its question and kind, its design and when it ran."""
        bundle = self.bundle
        created = [summary.created_at for aspect in bundle.aspects for summary in aspect.run_summaries]
        return Meta(
            campaign=bundle.name,
            question=bundle.declaration.question or None,
            behavior=bundle.behavior,
            kind=bundle.kind,
            kind_means=KIND_MEANS[bundle.kind],
            design_shape=self.lead.design.shape,
            run_date=max(created)[:10] if created else None,
            provenance=provenance,
            synthetic=False,
        )

    def arms(self) -> list[Arm]:
        """Every arm by its label: its role, served model, variant key, settings and reasoning share.

        A candidate's settings are its model and every lever it holds at another level than the control; the
        control's, and a choice's arms', are their model and the levers the arms differ on.
        """
        bundle = self.bundle
        entries = {entry.variant_key: entry for aspect in bundle.aspects for entry in aspect.variant_index}
        described = writer_arms(list(entries.values()))
        listed: list[dict[str, Any]] = described["arms"]  # type: ignore[assignment]  # writer_arms types its values as object
        differing = {arm["variant_key"]: dict(arm["levels"]) for arm in listed}
        served = self.served_models()
        ratios = self.reasoning_ratios()
        control = bundle.control()
        control_levers = self.levers(control.variant_key, entries) if control is not None else {}
        arms: list[Arm] = []
        seen: set[str] = set()
        roles = {key: arm.role for arm in bundle.arms for key in bundle.arm_keys(arm)}
        for key in [*(arm.variant_key for arm in bundle.arms), *entries]:
            label = self.arm(key)
            if label in seen:
                continue
            seen.add(label)
            levers = self.levers(key, entries)
            model = served.get(label) or levers.pop(CANDIDATE_MODEL_LEVER, (None, None))[1]
            levers.pop(CANDIDATE_MODEL_LEVER, None)
            if roles.get(key) != "control" and control is not None:
                settings = {
                    axis: display
                    for axis, (digest, display) in levers.items()
                    if control_levers.get(axis, (None, None))[0] != digest
                }
            else:
                settings = {axis: display for axis, display in differing.get(key, {}).items()}
                settings.pop(CANDIDATE_MODEL_LEVER, None)
            arms.append(
                Arm(
                    id=label,
                    name=label,
                    role="control" if roles.get(key) == "control" else "challenger",
                    served_model=model,
                    variant_key=key,
                    settings={"model": model, **settings},
                    reasoning_ratio=ratios.get(label),
                )
            )
        return arms

    def levers(self, variant_key: str, entries: Mapping[str, VariantIndexEntry]) -> dict[str, tuple[str, str]]:
        """Each lever an arm ran, applicable to its kind: ``axis -> (digest, display)``; from its campaign arm, else
        its variant's index entry."""
        arm = next((arm for arm in self.bundle.arms if arm.variant_key == variant_key), None)
        held = arm.levers if arm is not None and arm.levers is not None else None
        if held is None:
            entry = entries.get(variant_key)
            held = entry.named_levers if entry is not None and entry.levels_unavailable is None else {}
        return {axis: (level.content_hash, level.display) for axis, level in held.items() if level.not_of_kind is None}

    def served_models(self) -> dict[str, str]:
        """The model that served each arm, by label, from the first aspect that recorded one."""
        served: dict[str, str] = {}
        for aspect in self.ordered:
            for row in aspect.arm_served_models:
                if row.served_models:
                    served.setdefault(self.arm(row.variant_key), ", ".join(row.served_models))
        return served

    def reasoning_ratios(self) -> dict[str, float]:
        """Each arm's share of output spent reasoning, by label, from the first aspect that measured it."""
        ratios: dict[str, float] = {}
        for aspect in self.ordered:
            for reading in aspect.arm_mechanisms:
                if reading.covariate == "reasoning_ratio" and reading.mean is not None:
                    ratios.setdefault(self.arm(reading.variant_key), reading.mean)
        return ratios

    # ------------------------------------------------------------------ aspects and their readings

    def aspect(self, aspect: AnalysisContextBundle) -> Aspect:
        """One aspect: its question, margin, readings, pair readings and depth."""
        declared = aspect.declared
        label = aspect.aspect
        depth = aspect_depth(aspect, self.arm)
        pass_k = [
            PassK(
                where=subject.where,
                k=subject.k,
                common_depth=subject.common_depth,
                shares_no_case=subject.shares_no_case,
                unmeasured_for=subject.unmeasured_for,
                said=subject.lines(),
            )
            for subject in depth.subjects
            if subject.compared and subject.common_depth is not None
        ]
        return Aspect(
            id=label,
            name=aspect.short_name or label,
            margin=declared.margin if declared is not None else None,
            asked=aspect.question_declared,
            exploratory_measures=list(aspect.reading_scope.exploratory_measures),
            readings=[self.reading(row) for row in self.bundle.readings.on_aspect(label)],
            depth={
                arm: Depth(
                    cases=held.cases,
                    trials=held.trials,
                    repeats_per_case=[held.repeats_min, held.repeats_median, held.repeats_max],
                )
                for arm, held in depth.arms.items()
            }
            or None,
            pass_k=pass_k or None,
            pairs=[self.pair(pair) for pair in self.bundle.readings.pairs if pair.aspect == label],
            scoreboard=None if aspect is self.lead else self.scoreboard(aspect),
        )

    def reading(self, row: AspectReading) -> Reading:
        """One candidate's reading on its aspect, as code read it."""
        return Reading(
            arm=row.arm_label,
            reading=row.reading,
            change=row.change,
            within_interval=row.within_interval,
            gain_interval=row.gain_interval,
            gain_confidence=row.gain_confidence,
            margin=row.margin,
            control_passed=f"{row.control_passed} of {row.control_trials}",
            arm_passed=f"{row.arm_passed} of {row.arm_trials}",
            cases=[row.control_cases, row.arm_cases],
            confirmatory=row.confirmatory,
            why=row.reason or None,
            precision=self.precision(row.precision),
        )

    def pair(self, pair: PairReading) -> PairRead:
        """Two arms of a choice on one aspect, as code read them."""
        return PairRead(
            arms=(pair.first_label, pair.second_label),
            change=pair.change,
            interval=pair.interval,
            confidence=pair.confidence,
            p_holm=pair.p_holm,
            ahead=pair.ahead_label,
            confirmatory=pair.confirmatory,
            passed=(f"{pair.first_passed} of {pair.first_trials}", f"{pair.second_passed} of {pair.second_trials}"),
            cases=(pair.first_cases, pair.second_cases),
            why=pair.reason or None,
            precision=self.precision(pair.precision),
        )

    def precision(self, precision: BundlePrecision | None) -> Precision | None:
        """What limits a reading's interval, each arm by its label."""
        if precision is None:
            return None
        return Precision(
            half_width=precision.half_width,
            own_half_width={self.arm(key): width for key, width in precision.half_widths.items()},
            limited_by=self.arm(precision.limited_by) if precision.limited_by is not None else None,
            between_case_share=precision.between_share,
            means=REPEATS_FLOOR,
        )

    def floors(self) -> list[FloorRead]:
        """Every floor's status on every candidate, as code read it."""
        return [
            FloorRead(
                floor=status.floor,
                aspect=status.aspect,
                arm=status.arm_label,
                status=status.status,
                reading=status.reading,
                speaks_for=status.reading
                if status.reading is not None
                else ("the pass rate" if status.margin is not None else "the cost per reply"),
                interval=status.interval,
                worst_drop=status.worst_drop,
                margin=status.margin,
                short_by=status.short_by,
                reason=status.reason,
            )
            for status in self.bundle.readings.floors
        ]

    def bars(self) -> list[BarRead]:
        """Every arm against every absolute bar, as code read it."""
        return [
            BarRead(
                aspect=status.aspect,
                arm=status.arm_label,
                measure=status.measure,
                bar=status.bar,
                direction=status.direction,
                status=status.status,
                value=status.value,
                interval=status.interval,
                why=status.reason or None,
            )
            for status in self.bundle.readings.bars
        ]

    def guardrails(self) -> list[dict[str, object]]:
        """Every guardrail check on every aspect, as the pipeline read it, each arm by its label."""
        return [
            {
                "aspect": aspect.aspect,
                "arm": self.arm(check.contrast.variant_key),
                "control": self.arm(check.control.variant_key),
                "reading": check.reading,
                "name": check.name,
                "higher_is_better": check.higher_is_better,
                "delta": check.delta,
                "interval": list(check.interval) if check.interval is not None else None,
                "margin": check.margin,
                "margin_declared": check.margin_declared,
                "decision": check.decision,
            }
            for aspect in self.bundle.aspects
            for check in aspect.guardrails.checks
        ]

    def criteria(self) -> list[Criterion]:
        """Every absolute threshold the campaign declares: each aspect's bars. A floor is read against the control,
        not a threshold, so it is a floor, never a criterion."""
        return [
            Criterion(
                id=f"{aspect.aspect}/bar/{bar.measure_id}",
                kind="bar",
                measure=bar.measure_id,
                direction=bar.direction,
                threshold=bar.threshold,
            )
            for aspect in self.bundle.aspects
            if aspect.declared is not None
            for bar in aspect.declared.design.bars
        ]

    def scoreboard(self, aspect: AnalysisContextBundle) -> dict[str, ScoreboardRow]:
        """How each arm's trials came out on one aspect, by label."""
        return {
            self.arm(row.variant_key): ScoreboardRow(
                ran=row.ran,
                lost=row.lost,
                usable=row.usable,
                passed=row.passed,
                failed=row.failed,
                cases=row.cases,
                reply_cost_usd=row.reply_cost_usd,
                test_cost_usd=row.test_cost_usd,
            )
            for row in aspect.scoreboard
        }

    # ------------------------------------------------------------------ measures and judged dimensions

    def cells(self, aspect: AnalysisContextBundle) -> dict[str, CellFacts]:
        """Each arm's cell on one aspect, by label: its first rig's where it ran under more than one, which
        ``caveats.unreported`` then says."""
        cells: dict[str, CellFacts] = {}
        rigs: dict[str, int] = {}
        for cell in aspect.cell_measures:
            label = self.arm(cell.variant_key)
            cells.setdefault(label, cell)
            rigs[label] = rigs.get(label, 0) + 1
        for label, count in rigs.items():
            if count > 1:
                note = f"{label} ran {aspect.aspect} under {count} rigs; its figures here are its first rig's."
                if note not in self.unreported:
                    self.unreported.append(note)
        return cells

    def measures(self) -> dict[str, Measure]:
        """Every measure the cells carry: what it is, and each arm's figure on the lead aspect and on every other.

        A categorical or free-text measure is described where the catalog describes it, with no figure. A
        measurement's readings carry figures no cell does (the trials' own pass rate): each joins its measure.
        """
        facts: dict[str, dict[str, Any]] = {}
        figures: dict[str, dict[str, dict[str, Stat]]] = {}
        for aspect in self.ordered:
            for label, cell in self.cells(aspect).items():
                for summary in cell.measures.measures:
                    if summary.texts or summary.categories:
                        if summary.name in aspect.measure_catalog:
                            facts.setdefault(summary.name, self.measure_facts(aspect, summary.name))
                        continue
                    point = summary.rate if summary.rate is not None else summary.mean
                    tail = summary.bad_tail() if summary.higher_is_better is False else None
                    figures.setdefault(summary.name, {}).setdefault(aspect.aspect, {})[label] = Stat(
                        mean=point,
                        ci=(summary.ci_low, summary.ci_high)
                        if summary.ci_low is not None and summary.ci_high is not None
                        else None,
                        n=summary.n,
                        n_cases=summary.n_independent,
                        p95=tail,
                    )
                    facts.setdefault(summary.name, self.measure_facts(aspect, summary.name))
            for measured in self.bundle.readings.measures:
                if measured.aspect != aspect.aspect:
                    continue
                held = figures.setdefault(measured.measure, {}).setdefault(aspect.aspect, {})
                if measured.arm_label not in held:
                    held[measured.arm_label] = Stat(
                        mean=measured.value, ci=measured.interval, n=measured.trials, n_cases=measured.cases
                    )
                facts.setdefault(
                    measured.measure,
                    {"label": measured.name, "higher_is_better": measured.higher_is_better},
                )
        return {
            name: Measure(
                **held,
                per_arm=figures.get(name, {}).get(self.lead.aspect, {}),
                by_aspect={label: arms for label, arms in figures.get(name, {}).items() if label != self.lead.aspect},
            )
            for name, held in sorted(facts.items())
        }

    def measure_facts(self, aspect: AnalysisContextBundle, name: str) -> dict[str, Any]:
        """What one measure is, as the page names it: its name, unit, direction, merit axis and definition."""
        facts = aspect.measure_catalog.get(name)
        check = goal_check_of(name)
        label = (aspect.check_name(name) or check) if check is not None else None
        if facts is None:
            return {"label": label}
        return {
            "label": label if label is not None else measure_name(name, facts.reader_name),
            "unit": facts.unit,
            "higher_is_better": facts.higher_is_better,
            "merit_axis": facts.merit_axis,
            "means": capped_text(facts.reader_prose or facts.description or "", MEANING_CHARS) or None,
        }

    def judged(self) -> dict[str, Judged]:
        """Every judged dimension: its scale, tier, judge and calibration, and each arm's mean on the lead aspect and
        on every other."""
        tiers = {tier.rubric_dim: tier for aspect in reversed(self.ordered) for tier in aspect.judge_evidence_tiers}
        figures: dict[str, dict[str, dict[str, Stat]]] = {}
        evidence: dict[str, str] = {}
        for aspect in self.ordered:
            for label, cell in self.cells(aspect).items():
                for reading in cell.judged:
                    figures.setdefault(reading.dimension, {}).setdefault(aspect.aspect, {})[label] = Stat(
                        mean=reading.mean, sem=reading.sem, n=reading.n, n_cases=reading.n_independent
                    )
                    evidence.setdefault(reading.dimension, reading.evidence_tier)
        judged: dict[str, Judged] = {}
        for dimension, by_aspect in figures.items():
            tier = tiers.get(dimension)
            judged[dimension] = Judged(
                scale=tier.scale if tier is not None else None,
                tier=evidence[dimension],
                judge_model=tier.judge_model if tier is not None else None,
                calibration=tier.calibration.state if tier is not None else None,
                calibration_ratings=tier.calibration.n if tier is not None else None,
                agreement=tier.calibration.agreement if tier is not None else None,
                per_arm=by_aspect.get(self.lead.aspect, {}),
                by_aspect={label: arms for label, arms in by_aspect.items() if label != self.lead.aspect},
            )
        return judged

    def primary_judged(self, judged: Mapping[str, Judged]) -> str | None:
        """The judged dimension the examples are failures on: one a floor reads, else the first pass/fail one, else
        the first; a reserved axis (``__outcome__``) only when nothing else is judged."""
        named = [dimension for dimension in judged if not dimension.startswith("__")] or list(judged)
        floored = [status.reading for status in self.bundle.readings.floors if status.reading in named]
        passing = [dimension for dimension in named if judged[dimension].scale == "pass_fail"]
        chosen = floored or passing or named
        return chosen[0] if chosen else None

    # ------------------------------------------------------------------ contrasts

    def contrasts(self) -> list[Contrast]:
        """Every corrected contrast against the control, aspect by aspect, each aspect's separated ones first."""
        contrasts: list[Contrast] = []
        for aspect in self.bundle.aspects:
            rows: list[tuple[tuple[int, float], Contrast]] = []
            for family in aspect.multiple_comparisons.families:
                level = _level(family.interval_level)
                for comparison in family.comparisons:
                    rows.append(
                        (
                            (
                                0 if comparison.verdict in _SEPARATED else 1,
                                comparison.p_adjusted if comparison.p_adjusted is not None else 1.0,
                            ),
                            Contrast(
                                arm=self.arm(comparison.contrast.variant_key),
                                control=self.arm(comparison.control.variant_key),
                                kind=comparison.reading,
                                measure=comparison.name,
                                aspect=aspect.aspect,
                                verdict=comparison.verdict,
                                delta=comparison.delta,
                                interval=comparison.interval,
                                level=level,
                                p_adjusted=comparison.p_adjusted,
                                materiality=comparison.materiality,
                                confounds=[confound.dimension for confound in comparison.mechanism_confounds],
                                reason=comparison.untested_reason or comparison.not_separated_reason,
                            ),
                        )
                    )
            contrasts.extend(contrast for _, contrast in sorted(rows, key=lambda row: row[0]))
        return contrasts

    def families(self) -> list[ContrastFamily]:
        """Every family of contrasts corrected together, with the level each of its intervals is read at."""
        return [
            ContrastFamily(
                aspect=aspect.aspect,
                merit_axes=list(family.merit_axes),
                correction=family.correction,
                alpha=family.alpha,
                family_size=family.family_size,
                disclosure=family.disclosure,
                level=_level(family.interval_level),
            )
            for aspect in self.bundle.aspects
            for family in aspect.multiple_comparisons.families
        ]

    # ------------------------------------------------------------------ strata and examples

    def strata(self) -> list[Stratum]:
        """Every stratum of every aspect with each arm's judged means in it, the most cases first."""
        strata: list[Stratum] = []
        for aspect in self.ordered:
            rows: dict[str, dict[str, Any]] = {}
            for label, cell in self.cells(aspect).items():
                for held in cell.strata:
                    name = held.stratum if held.stratum is not None else UNSTRATIFIED
                    row = rows.setdefault(name, {"n_cases": held.n_cases, "per_arm": {}})
                    row["per_arm"][label] = {
                        reading.dimension: Stat(mean=reading.mean, sem=reading.sem, n=reading.n)
                        for reading in held.judged
                    }
            strata.extend(
                Stratum(id=name, aspect=aspect.aspect, n_cases=row["n_cases"], per_arm=row["per_arm"])
                for name, row in rows.items()
            )
        return sorted(strata, key=lambda stratum: -(stratum.n_cases or 0))

    def examples(self, primary: str | None, strata: Sequence[Stratum], limit: int) -> list[Example]:
        """Failed trials to show: on the primary judged dimension, one per arm and stratum, candidates first and the
        strata where the candidates lose most on it first; with no judged dimension, failed trials, one per arm,
        stratum and cause, candidates first."""
        control = self.bundle.control()
        control_label = control.label if control is not None else None
        loss = _loss(strata, primary, control_label) if primary is not None else {}
        trials = [(aspect, trial) for aspect in self.ordered for trial in aspect.trials]

        def failed(trial: TrialRow) -> bool:
            if primary is not None:
                return trial.judged.get(primary) == 0
            return trial.state == "failed"

        def order(item: tuple[AnalysisContextBundle, TrialRow]) -> tuple[bool, float, str]:
            aspect, trial = item
            return (
                self.arm(trial.arm) == control_label,
                -loss.get((aspect.aspect, trial.stratum or UNSTRATIFIED), 0.0),
                trial.stratum or "",
            )

        shown: list[Example] = []
        seen: set[tuple[str, str, str | None, str | None]] = set()
        for aspect, trial in sorted(trials, key=order):
            arm = self.arm(trial.arm)
            key = (aspect.aspect, arm, trial.stratum, None if primary is not None else trial.cause)
            if not failed(trial) or key in seen:
                continue
            seen.add(key)
            shown.append(
                Example(
                    trial=self.trial_ids.get((aspect.aspect, trial.trial_id), trial.trial_id),
                    case=trial.case_label,
                    stratum=trial.stratum,
                    aspect=aspect.aspect,
                    arm=arm,
                    repeat=trial.repeat,
                    reason=_reason(trial, primary),
                    excerpt=_unescaped(trial.excerpt)[:EXAMPLE_EXCERPT_CHARS] or None,
                    outcome=trial.cause,
                )
            )
            if len(shown) >= limit:
                break
        return shown

    # ------------------------------------------------------------------ caveats

    def caveats(self) -> Caveats:
        """Every disclosure the flag and the method section may need, each said once across the aspects."""
        late: dict[tuple[str, str | None, str], None] = {}
        tiers: dict[str, str] = {}
        confounds: dict[str, str] = {}
        apparatus: dict[str, ApparatusConfound] = {}
        latency: dict[str, None] = {}
        short: dict[str, ShortRun] = {}
        unreported: dict[str, None] = {}
        for aspect in self.ordered:
            for criterion in aspect.reading_scope.late_criteria:
                late.setdefault((criterion.criterion, criterion.name, criterion.page_date() or "not recorded"))
            for tier in aspect.judge_evidence_tiers:
                tiers.setdefault(tier.rubric_dim, tier.tier)
            for dimension in _confounds_named(aspect):
                if dimension in aspect.confound_catalog:
                    confounds.setdefault(dimension, aspect.confound_catalog[dimension])
            for confound in aspect.apparatus_confounds:
                apparatus.setdefault(
                    confound.dimension,
                    ApparatusConfound(
                        dimension=confound.dimension,
                        kind=confound.kind,
                        status=confound.status,
                        level_values=dict(confound.level_values),
                        threshold=confound.threshold,
                    ),
                )
            if aspect.latency_contended:
                latency.setdefault(aspect.latency_contended)
            runs = _run_arms(aspect)
            for run, note in aspect.short_runs.items():
                key = runs.get(run)
                short.setdefault(run, ShortRun(run=run, arm=self.arm(key) if key is not None else None, note=note))
            for observation in aspect.telemetry.measures.unreported_observations:
                unreported.setdefault(observation)
        for note in self.unreported:
            unreported.setdefault(note)
        return Caveats(
            late_criteria=[
                LateCriterion(criterion=criterion, name=name, set_at=set_at) for criterion, name, set_at in late
            ],
            judge_tiers=tiers,
            confounds=confounds,
            apparatus_confounds=list(apparatus.values()),
            latency=" ".join(latency) or None,
            short_runs=list(short.values()),
            unreported=list(unreported),
        )


def _level(level: float | None) -> str | None:
    """An interval level as the family's disclosure spells it, ``99.69%``; None where none was read."""
    return None if level is None else f"{format_number(100 * level)}%"


def _loss(strata: Sequence[Stratum], dimension: str, control: str | None) -> dict[tuple[str, str], float]:
    """Where the candidates lose most on ``dimension``: each stratum's cases times every candidate's drop below the
    control there, by ``(aspect, stratum)``. An ordering of the examples, never a figure the input reports."""
    loss: dict[tuple[str, str], float] = {}
    for stratum in strata:
        held = stratum.per_arm.get(control or "", {}).get(dimension)
        if held is None or held.mean is None:
            continue
        control_mean = held.mean
        means = [
            figures[dimension].mean
            for arm, figures in stratum.per_arm.items()
            if arm != control and dimension in figures
        ]
        drops = [max(0.0, control_mean - mean) for mean in means if mean is not None]
        loss[stratum.aspect or "", stratum.id] = (stratum.n_cases or 0) * sum(drops)
    return loss


def _unescaped(text: str) -> str:
    """``text`` with the JSON escapes a transcript carries read as the characters they stand for, so an example reads
    as the reply did: ``\\u2014`` as a dash, ``\\n`` as a new line. A lone surrogate is left as written."""

    def one(match: re.Match[str]) -> str:
        decoded: str = json.loads(f'"{match.group(0)}"')
        return match.group(0) if any(0xD800 <= ord(char) <= 0xDFFF for char in decoded) else decoded

    return _ESCAPE.sub(one, text)


def _reason(trial: TrialRow, primary: str | None) -> str | None:
    """Why a shown trial failed: the judge's reason on the primary dimension, else the checks it failed, else its
    error."""
    if primary is not None:
        return next((judged.reason.strip() or None for judged in trial.judge if judged.dim == primary), None)
    if trial.failed_checks:
        return "; ".join(check.name for check in trial.failed_checks)
    return trial.error_words


def _run_arms(aspect: AnalysisContextBundle) -> dict[str, str]:
    """Each run the aspect's design placed, and the variant key of the arm it ran."""
    design = aspect.design
    placed = [design.control_arm, *design.contrasts] if design.control_arm is not None else list(design.contrasts)
    return {run: arm.variant_key for arm in placed for run in arm.run_ids}


def _confounds_named(aspect: AnalysisContextBundle) -> Iterable[str]:
    """Every confound the aspect names on its readings: its apparatus confounds, its levers' and its design's, and
    its contrasts'. The catalog's other entries explain confounds nothing here reads."""
    yield from (confound.dimension for confound in aspect.apparatus_confounds)
    for lever in aspect.coverage:
        yield from (confound.dimension for confound in lever.confounded_by)
    design = aspect.design
    for arm in [design.control_arm, *design.contrasts]:
        if arm is not None:
            yield from (confound.dimension for confound in arm.mechanism_confounds)
    for family in aspect.multiple_comparisons.families:
        for comparison in family.comparisons:
            yield from (confound.dimension for confound in comparison.mechanism_confounds)


# ---------------------------------------------------------------------- the planner's view


def planner_view(report: ReportInput, *, budget: int | None = None) -> str:
    """The report input as the planner reads it: compact JSON within ``budget`` characters, saying what it cut.

    Each figure is kept to :data:`PLANNER_FIGURES` significant figures (the planner cites a figure by its key, which
    resolves against the whole input), and on one aspect no entry names it. Over budget, code trims in a fixed order, each step only as far as it must:
    every shown trial's excerpt to :data:`PLANNER_EXCERPT_CHARS` characters; the examples past the first
    :data:`PLANNER_EXAMPLES`; the strata past the :data:`PLANNER_STRATA` largest; the contrasts outside the asked
    aspects, least separated last aspect first; then, smallest stratum first, each kept stratum's figures on anything
    but the primary judged dimension, the one its examples fail on. Never trimmed: the arms, the readings and pair
    readings, the floors, the caveats, and every contrast on an asked aspect or on a judged dimension. What was cut is
    said in ``trimmed``, which counts toward the budget.

    :param report: The report input.
    :ptype report: ReportInput
    :param budget: The most characters the view may take; None for
        :data:`~threetears.evals.analysis.campaign_bundle.CAMPAIGN_MESSAGE_CHARS_PER_ASPECT` per aspect.
    :ptype budget: int | None
    :return: The view, as compact JSON text.
    :rtype: str
    :raises RefusedInput: The view is still over budget once every trim is made: what is left is never trimmed.
    """
    limit = budget if budget is not None else CAMPAIGN_MESSAGE_CHARS_PER_ASPECT * max(1, len(report.aspects))
    view: dict[str, Any] = _rounded(report.model_dump(mode="json", by_alias=True, exclude_none=True))
    if len(report.aspects) == 1:
        # One aspect: every contrast, family, stratum and example is on it, so naming it on each says nothing.
        for section in ("contrasts", "contrast_families", "strata", "examples"):
            for entry in view.get(section, []):
                entry.pop("aspect", None)
    if len(_compact(view)) <= limit:
        return _compact(view)
    cut = {"excerpts": 0, "examples": 0, "strata": 0, "contrasts": 0, "stratum figures": 0}
    view["trimmed"] = _trim_note(dict.fromkeys(cut, 10**6), limit, report.primary_judged)

    def over() -> bool:
        return len(_compact(view)) > limit

    examples: list[dict[str, Any]] = view.get("examples", [])
    shortened: list[int] = []
    for index, example in enumerate(examples):
        if not over():
            break
        excerpt = example.get("excerpt")
        if excerpt is not None and len(excerpt) > PLANNER_EXCERPT_CHARS:
            example["excerpt"] = capped_text(excerpt, PLANNER_EXCERPT_CHARS)
            shortened.append(index)
    while over() and len(examples) > PLANNER_EXAMPLES:
        examples.pop()
        cut["examples"] += 1
    # Counted over the examples kept, so the note never counts an excerpt whose example it then left out.
    cut["excerpts"] = sum(1 for index in shortened if index < len(examples))
    strata: list[dict[str, Any]] = view.get("strata", [])
    if over() and len(strata) > PLANNER_STRATA:
        strata.sort(key=lambda stratum: -int(stratum.get("n_cases", 0)))
        while over() and len(strata) > PLANNER_STRATA:
            strata.pop()
            cut["strata"] += 1
    asked = {aspect.id for aspect in report.aspects if aspect.asked} or {aspect.id for aspect in report.aspects[:1]}
    contrasts: list[dict[str, Any]] = view.get("contrasts", [])
    for index in reversed(range(len(contrasts))):
        if not over():
            break
        contrast = contrasts[index]
        if contrast.get("kind") == "judged" or "aspect" not in contrast or contrast["aspect"] in asked:
            continue
        del contrasts[index]
        cut["contrasts"] += 1
    primary = report.primary_judged
    for stratum in reversed(strata) if primary is not None else ():
        if not over():
            break
        held: dict[str, dict[str, Any]] = stratum.get("per_arm", {})
        if any(set(figures) - {primary} for figures in held.values()):
            stratum["per_arm"] = {
                arm: {name: figure for name, figure in figures.items() if name == primary}
                for arm, figures in held.items()
            }
            cut["stratum figures"] += 1
    if over():
        never = ("schema", "meta", "arms", "aspects", "floors", "caveats")
        protected: dict[str, Any] = {key: view[key] for key in never if key in view} | {"contrasts": contrasts}
        raise RefusedInput(
            f"the planner's view of campaign {report.meta.campaign!r} is {len(_compact(view))} characters after "
            f"every trim, over its budget of {limit}; what is never trimmed (arms, readings, floors, caveats, the "
            f"asked aspects' and judged dimensions' contrasts) takes {len(_compact(protected))} of them",
            remedy="raise the planner's budget for this campaign, or split it into campaigns of fewer arms or aspects",
        )
    view["trimmed"] = _trim_note(cut, limit, primary)
    return _compact(view)


def _trim_note(cut: Mapping[str, int], limit: int, primary: str | None) -> str:
    """What the planner's view says it cut to fit its budget."""
    said = {
        "excerpts": f"{cut['excerpts']} example excerpts cut to {PLANNER_EXCERPT_CHARS} characters",
        "examples": f"{cut['examples']} examples past the first {PLANNER_EXAMPLES}",
        "strata": f"{cut['strata']} strata past the {PLANNER_STRATA} largest",
        "contrasts": f"{cut['contrasts']} contrasts outside the asked aspects",
        "stratum figures": f"{cut['stratum figures']} strata's figures on anything but {primary}",
    }
    return (
        f"To fit the planner's budget of {limit} characters, this view leaves out: "
        + "; ".join(said[what] for what, count in cut.items() if count)
        + ". Every figure key still resolves against the whole report input."
    )


def _compact(node: Any) -> str:
    """The compact JSON a budget is counted in and the view is sent as."""
    return json.dumps(node, separators=(",", ":"), ensure_ascii=False)


def _rounded(node: Any) -> Any:
    """``node`` with every float kept to :data:`PLANNER_FIGURES` significant figures; integers as they are."""
    if isinstance(node, dict):
        return {key: _rounded(value) for key, value in node.items()}
    if isinstance(node, list):
        return [_rounded(value) for value in node]
    if isinstance(node, float) and math.isfinite(node) and node != 0:
        return round(node, PLANNER_FIGURES - 1 - math.floor(math.log10(abs(node))))
    return node
