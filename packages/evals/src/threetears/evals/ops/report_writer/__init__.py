"""The report writer: the sessions of tool turns that write a campaign report from a frozen snapshot (plan §4).

:mod:`~threetears.evals.ops.report_writer.session` is one model's conversation with one mounted tool, held to its
budgets, stored and resumed; :mod:`~threetears.evals.ops.report_writer.analyst` is the analyst's session over it (the
brief, the submission and its bounces); :mod:`~threetears.evals.ops.report_writer.tally` is what writing the report
cost, and the dollar budget that admits every turn. The loop never reads an environment variable: the model, its
client and every limit come from its caller.
"""
