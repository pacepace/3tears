"""The read lenses over runs that already exist, one module per lens, each over the score projection.

Every lens consumes :func:`~threetears.evals.analysis.reporting.project_score_records` (or the placement beneath it)
and is recomputed per query; nothing here is stored. A host imports them from :mod:`threetears.evals.analysis`,
never from here; :mod:`threetears.evals.analysis.reads` is where they meet storage.

- ``comparison_sets`` — which runs may honestly be compared with each other.
- ``pivot`` — any two factors as axes; ``frontier`` — the cheapest variant clearing the bar, per subject;
  ``history`` — per-measure longitudinal series with regression flags.
- ``program_budget`` — spend, never excluding what quality excludes; ``orphaned_runs`` — spend no campaign claims.
- ``export`` — the projection's rows as CSV or JSON; ``cost_estimate`` — a proposed sweep's predicted cost.
- ``aggregation`` and ``contestants`` — the roll-up vocabulary and the contestant key the lenses above share.
"""
