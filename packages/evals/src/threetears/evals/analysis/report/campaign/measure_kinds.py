"""Which measures a campaign's scorecard reads as rates, which as money on which ruler, and which it shows at all.

What a measure is, not how its figures are spelled: the page's forms (:mod:`~threetears.evals.analysis.report.forms`)
read these to decide each measure's form, and the scorecard reads them to lay its columns out, so both classify a
measure alike.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from threetears.evals.analysis.bundle.surface import bundle_decision_surface
from threetears.evals.analysis.campaign_readings import CampaignReadings, MeasureReading
from threetears.evals.analysis.quantities import is_money
from threetears.evals.analysis.report.cost_unit import MoneyRuler, money_ruler
from threetears.evals.kernel.analysis_measures import MeasureSummary
from threetears.evals.kernel.metrics import (
    ACCURACY_MEASURE,
    MATCH_MEASURE,
    METRIC_DESCRIPTORS,
    MetricDescriptor,
    reads_as_rate,
)
from threetears.evals.kernel.surface import DecisionSurface

if TYPE_CHECKING:
    # The module that defines a page loads the parts that read these, so the type is named for the checker alone.
    from threetears.evals.analysis.report.model import Report

__all__ = ["measure_summary", "money_rulers", "rate_measures", "shown_measures"]


def _measure_described(reading: MeasureReading, reports: Mapping[str, Report]) -> MetricDescriptor | None:
    """A measure's description on its aspect: the bundle's catalog's, else the engine's own.

    :param reading: The arm's reading of the measure.
    :ptype reading: MeasureReading
    :param reports: Each aspect's own page, by its label.
    :ptype reports: Mapping[str, Report]
    :return: The description, or None.
    :rtype: MetricDescriptor | None
    """
    catalog = reports[reading.aspect].bundle.measure_catalog
    return catalog.get(reading.measure) or METRIC_DESCRIPTORS.get(reading.measure)


def measure_summary(
    reading: MeasureReading, reports: Mapping[str, Report], surfaces: dict[str, DecisionSurface]
) -> MeasureSummary | None:
    """A measure's summary on its arm's one cell on its aspect (a classifier's accuracy is its label match's); None
    where the arm ran under several setups, whose cells the fold holds apart, or the cell carries no such measure.

    :param reading: The arm's reading of the measure.
    :ptype reading: MeasureReading
    :param reports: Each aspect's own page, by its label.
    :ptype reports: Mapping[str, Report]
    :param surfaces: Each aspect's decision surface read so far, by its label, which this fills as it reads.
    :ptype surfaces: dict[str, DecisionSurface]
    :return: The summary, or None.
    :rtype: MeasureSummary | None
    """
    surface = surfaces.get(reading.aspect)
    if surface is None:
        surface = surfaces[reading.aspect] = bundle_decision_surface(reports[reading.aspect].bundle)
    cells = [cell for cell in surface.cells if cell.variant_key == reading.arm]
    if len(cells) != 1:
        return None
    held = {summary.name: summary for summary in cells[0].measures.measures}
    return held.get(reading.measure) or (held.get(MATCH_MEASURE) if reading.measure == ACCURACY_MEASURE else None)


def rate_measures(
    readings: CampaignReadings, reports: Mapping[str, Report], surfaces: dict[str, DecisionSurface] | None = None
) -> frozenset[str]:
    """The measures the scorecard reads as rates, by key, each stated in percent beside the pass rate: the pass rate
    itself, every measure that reads as a rate (:func:`~threetears.evals.kernel.metrics.reads_as_rate`: a check, a goal
    check's pass rate, a classifier's accuracy), and every one its cells summarise as a rate (a classifier's precision
    and recall), so a row never reads ``68.9`` beside ``0.9`` (the review's P18).

    :param readings: What code read off the campaign.
    :ptype readings: CampaignReadings
    :param reports: Each aspect's own page, by its label.
    :ptype reports: Mapping[str, Report]
    :param surfaces: Each aspect's decision surface read so far, which this fills as it reads; None for none.
    :ptype surfaces: dict[str, DecisionSurface] | None
    :return: The rates' keys.
    :rtype: frozenset[str]
    """
    held = {} if surfaces is None else surfaces
    return frozenset(
        reading.measure
        for reading in readings.measures
        if reading.passed is not None
        or reads_as_rate("measure", reading.measure, descriptor=_measure_described(reading, reports))
        or ((summary := measure_summary(reading, reports, held)) is not None and summary.rate is not None)
    )


#: How many replies a money measure's figures are drawn per, at the least.
_PER_THOUSAND_REPLIES = 1000


def money_rulers(readings: CampaignReadings, reports: Mapping[str, Report]) -> dict[str, MoneyRuler]:
    """Each money measure's one ruler, by its key, chosen from every figure the scorecard states of it, its intervals
    included (:func:`~threetears.evals.analysis.report.cost_unit.money_ruler`): money per reply drawn per 1,000 replies
    (per 1,000,000 where a figure is under a cent there), the count in the header, never a mark on the figure, as the
    measure's figures read on every row.

    :param readings: What code read off the campaign.
    :ptype readings: CampaignReadings
    :param reports: Each aspect's own page, by its label.
    :ptype reports: Mapping[str, Report]
    :return: The rulers, by measure key.
    :rtype: dict[str, MoneyRuler]
    """
    held: dict[str, list[float]] = {}
    for reading in readings.measures:
        described = _measure_described(reading, reports)
        if reading.passed is not None or described is None or not is_money(described.unit or ""):
            continue
        figures = held.setdefault(reading.measure, [])
        figures += [figure for figure in (reading.value, *(reading.interval or ())) if figure is not None]
    return {key: money_ruler(figures, per=1, start=_PER_THOUSAND_REPLIES) for key, figures in held.items()}


def shown_measures(readings: CampaignReadings) -> list[MeasureReading]:
    """The measures a measurement's scorecard shows: every one read, but a classifier's per-trial label match where its
    accuracy is read on the same aspect and arm, since the two are one figure under two names.

    :param readings: What code read off the campaign.
    :ptype readings: CampaignReadings
    :return: The readings shown, in their order.
    :rtype: list[MeasureReading]
    """
    accurate = {(reading.aspect, reading.arm) for reading in readings.measures if reading.measure == ACCURACY_MEASURE}
    return [
        reading
        for reading in readings.measures
        if not (reading.measure == MATCH_MEASURE and (reading.aspect, reading.arm) in accurate)
    ]
