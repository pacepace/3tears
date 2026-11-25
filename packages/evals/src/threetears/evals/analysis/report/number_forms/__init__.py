"""A page's number forms, one module per part, each reading only the parts before it.

:mod:`~threetears.evals.analysis.report.forms` holds :class:`~threetears.evals.analysis.report.forms.PageForms`,
which decides every figure through these parts, and names what hosts import from them; the parts, in the order each
reads the ones before it:

- :mod:`~threetears.evals.analysis.report.number_forms.kinds`: how a measure reads, its kind, a figure's role and
  which page a set of forms is for.
- :mod:`~threetears.evals.analysis.report.number_forms.policy`: the number policy's primitives.
- :mod:`~threetears.evals.analysis.report.number_forms.intervals`: an interval's level, an interval's words and a
  count, as every surface prints them.
- :mod:`~threetears.evals.analysis.report.number_forms.sets`: which figures share a measure's set, and the half-widths
  each page states.
- :mod:`~threetears.evals.analysis.report.number_forms.printed`: what a page's forms hand out, and one measure's form.
- :mod:`~threetears.evals.analysis.report.number_forms.written`: the written page's decisions.
- :mod:`~threetears.evals.analysis.report.number_forms.code_page`: the code page's decisions.

This package's own init imports nothing, so loading one part loads only the parts it reads.
"""
