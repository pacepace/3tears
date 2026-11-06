"""A campaign's page, one module per part of it, laid out by :mod:`.page`.

- :mod:`.page` — :func:`~.page.build_campaign_report`: the parts in reading order, one call each, then one fold per
  aspect.
- :mod:`.warnings` — what warns, under the answer: one line per arm for the bars it misses, the floors it breaks and
  the guardrails it is not shown safe on.
- :mod:`.provenance` — the question, each report-wide caveat once, and Design & provenance.
- :mod:`.scorecard` — code's readings by the campaign's kind: a change's table and what would settle it, a choice's
  pairs and frontier, a measurement's measures, every kind's bars, and what each aspect's fold says code read.
- :mod:`.depth_blocks` — how deep each arm ran, pass^k at the common depth, the opt-in model, and what limits each
  comparison's interval.
- :mod:`.aggregate` — figures across aspects; none yet.
- :mod:`.memo_blocks` — the writer's and the person's words: the summary, the recommendation, the findings with
  their trace, the next steps, each aspect's summary, and the call.
- :mod:`.names` — what the page calls an aspect.
- :mod:`.common` — what several parts share: the memo's words as printed, the arms' names, a floor by its key, a rate
  by its name, a change in points.

:mod:`threetears.evals.analysis.report.campaign_page` re-exports the page's public names.
"""
