"""The figure-key grammar, one module per part, each reading only the parts before it.

:mod:`~threetears.evals.analysis.report.figure_keys` names what hosts import from them and teaches the grammar; the
parts, in the order each reads the ones before it:

- :mod:`~threetears.evals.analysis.report.keys.figure`: a key, the figure it resolves to, its scales and filters.
- :mod:`~threetears.evals.analysis.report.keys.vocabulary`: the slots a path is matched against, a family's shape,
  and the readers every family shares.
- :mod:`~threetears.evals.analysis.report.keys.names`: the one name a reader is shown for each id.
- :mod:`~threetears.evals.analysis.report.keys.reads`: what each family reads, and the values derived here.
- :mod:`~threetears.evals.analysis.report.keys.weakest`: the weakest margin.
- :mod:`~threetears.evals.analysis.report.keys.counts`: what a sentence counts or compares.
- :mod:`~threetears.evals.analysis.report.keys.prices`: prices, the price frontier and costs apart.
- :mod:`~threetears.evals.analysis.report.keys.separation`: spreads, differences and the cases that settle them.
- :mod:`~threetears.evals.analysis.report.keys.settle`: settling a reading against a margin, a floor or a pass rate.
- :mod:`~threetears.evals.analysis.report.keys.levers`: one setting's effect between two arms.
- :mod:`~threetears.evals.analysis.report.keys.families`: every family, in the one order the grammar reads them.
- :mod:`~threetears.evals.analysis.report.keys.resolve`: matching a path, each input's cache, and the public reads.
- :mod:`~threetears.evals.analysis.report.keys.grammar`: the grammar as the analyst's reference teaches it.

This package's own init imports nothing, so loading one part loads only the parts it reads.
"""
