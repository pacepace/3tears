"""The cell resolvers (the heatmap, the difference heatmap, the confusion matrix): none yet.

Until a kind is registered here, the dispatcher
(:func:`~threetears.evals.analysis.report.plan_page.resolve.resolve_chart`) refuses it, naming the kind and its
source: a chart is never drawn empty.
"""

from __future__ import annotations

from threetears.evals.analysis.report.plan_page.charting import KindResolver

__all__ = ["RESOLVERS"]

#: The kinds this module resolves: none yet.
RESOLVERS: dict[str, KindResolver] = {}
