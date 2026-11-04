"""The parts a page draws with: each chart's resolver, the arms' look, the words code writes, and the tables.

The report document's page (:mod:`~threetears.evals.analysis.report.doc_page`) and the analyst's blocks
(:mod:`~threetears.evals.analysis.analyst.blocks`) draw with them; the planned page they were written for is retired
with the report planner. The parts, each its own module:

- :mod:`~threetears.evals.analysis.report.plan_page.model`: the parts as documents (the masthead, the confidence, a
  decision slot, the options table, a chart, the arms key), model prose typed, code's lines plain text.
- :mod:`~threetears.evals.analysis.report.plan_page.look`: each arm's tag, colour and mark, and how a quantity is
  stated on a chart.
- :mod:`~threetears.evals.analysis.report.plan_page.words`: the words code writes on a page.
- :mod:`~threetears.evals.analysis.report.plan_page.resolve`: a chart's query to the kit's chart data, dispatched on
  the chart's kind through each ``resolve_*`` module's registry.
- :mod:`~threetears.evals.analysis.report.plan_page.tables`: the arms table and the arms key;
  :mod:`~threetears.evals.analysis.report.plan_page.full_results`: the Full results tables.

This package's own init imports nothing: the report's blocks read the arm look from
:mod:`~threetears.evals.analysis.report.plan_page.look`, which they sit below, so loading it must not load what reads
the blocks.
"""
