"""A campaign page built from its report plan: the planned page, in GENERATOR §2 order.

The plan says what the page says; code says how it looks and what each chart draws. The parts, each its own module:

- :mod:`~threetears.evals.analysis.report.plan_page.model`: :class:`PlannedPage` and its parts, planner text typed
  model prose, code's lines plain text.
- :mod:`~threetears.evals.analysis.report.plan_page.build`: :func:`plan_page`, the one builder, from a stored plan
  and the report input it was written against; no campaign and no bundle.
- :mod:`~threetears.evals.analysis.report.plan_page.look`: each arm's tag, colour and mark, and how a quantity is
  stated on a chart.
- :mod:`~threetears.evals.analysis.report.plan_page.words`: the words code writes on the page.
- :mod:`~threetears.evals.analysis.report.plan_page.resolve`: a chart's query to the kit's chart data, dispatched on
  the chart's kind through each ``resolve_*`` module's registry.
- :mod:`~threetears.evals.analysis.report.plan_page.tables`: the options table and the arms key.
- :mod:`~threetears.evals.analysis.report.plan_page.html` and
  :mod:`~threetears.evals.analysis.report.plan_page.markdown`: the page as HTML (default or static) and as Markdown.

This package's own init imports nothing: the report's blocks read the arm look from
:mod:`~threetears.evals.analysis.report.plan_page.look`, which they sit below, so loading it must not load the page
builder that reads the blocks.
"""
