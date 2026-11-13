"""An aspect's page, one module per part, each reading only the parts before it.

:mod:`~threetears.evals.analysis.report.build` lays the page out from these parts and names what hosts import from
them; the parts, in the order each reads the ones before it:

- :mod:`~threetears.evals.analysis.report.aspect.common`: what every part reads (the control, the names, a cell, an
  interval's words).
- :mod:`~threetears.evals.analysis.report.aspect.gates`: the guardrails and the bars.
- :mod:`~threetears.evals.analysis.report.aspect.contrasts`: the contrasts against the control, and the aspect's
  verdict over them.
- :mod:`~threetears.evals.analysis.report.aspect.readings`: the readings by arm, stratum and label.
- :mod:`~threetears.evals.analysis.report.aspect.scoreboard`: the first screen's scoreboard.
- :mod:`~threetears.evals.analysis.report.aspect.head`: the first screen's lines above it.
- :mod:`~threetears.evals.analysis.report.aspect.method`: what was not tested, and the methods.
- :mod:`~threetears.evals.analysis.report.aspect.trials`: the cases and the trials.
- :mod:`~threetears.evals.analysis.report.aspect.question`: the floors, and the screen of what it means laid out.

This package's own init imports nothing, so loading one part loads only the parts it reads.
"""
