"""The harness: the report writer evaluated as the thing under test, on the package's own store, from recordings.

The quick path (:mod:`threetears.evals.quick`) evaluates a callable; this package evaluates the report writer, the
analyst and its editor that write a campaign's page. It sits above ``quick`` and composes the engine's writer
machinery (``ops``, ``actions``) with the quick path's store:

- :class:`WriterEval` is the eval: arms of :class:`Writer` (a model, an :class:`AnalystSkill`, an effort) over a
  directory of portable writer cases, replayed from turn recordings, filed as one campaign and read for promotion;
  ``await writer_eval.run()`` returns a :class:`WriterResult`, and ``python -m threetears.evals FILE`` runs every
  writer eval a file defines.
- :class:`WriterCaseFile` is one portable case (a frozen report snapshot and what minting it names), and
  :func:`read_writer_cases` reads a directory of them.

**This module is the package's public root.** An eval file imports from here and from no module below it, and only
the names in ``__all__``.
"""

from __future__ import annotations

from threetears.evals.analysis.analyst.skill import AnalystSkill, SkillText
from threetears.evals.harness.writer_eval import Writer, WriterEval, WriterResult, WriterSide
from threetears.evals.quick.writer_cases import WriterCaseFile, read_writer_cases

__all__ = [
    "AnalystSkill",
    "SkillText",
    "Writer",
    "WriterCaseFile",
    "WriterEval",
    "WriterResult",
    "WriterSide",
    "read_writer_cases",
]
