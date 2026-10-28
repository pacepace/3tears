"""A campaign page built from its report document (``report-doc/1``): the analyst's claims, in its order, drawn.

The document says what the page says; code says how it looks, what each block draws and every part no analyst
writes. The parts, each its own module:

- :mod:`~threetears.evals.analysis.report.doc_page.model`: :class:`DocPage` and its parts, the analyst's text typed
  model prose, code's lines plain text.
- :mod:`~threetears.evals.analysis.report.doc_page.build`: :func:`build`, the one builder, from a document and the
  report input it was written from, strict (generation) or reading (a stored document, which degrades rather than
  refuses).
- :mod:`~threetears.evals.analysis.report.doc_page.words`: the words code writes on the page.
- :mod:`~threetears.evals.analysis.report.doc_page.html` and
  :mod:`~threetears.evals.analysis.report.doc_page.markdown`: the page as HTML (default or static) and as Markdown.
- :mod:`~threetears.evals.analysis.report.doc_page.text`: the page as a reader reads it, as plain text with every
  block's values table: the editor's input, and the text above the appendix that no internal id may appear in.

This package's own init imports nothing, as the planned page's does not.
"""
