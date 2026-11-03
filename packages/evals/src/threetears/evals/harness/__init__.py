"""The writer harness: the report writer measured as the thing under test, over the quick path's own store.

The quick path (:mod:`threetears.evals.quick`) evaluates a callable; this package evaluates the report writer, the
analyst and its editor that write a campaign's page. An eval file imports its names from here:

- :class:`WriterEval` is the eval: arms of :class:`Writer` (a model, an :class:`AnalystSkill`, an effort, its
  revisions) over a directory of portable writer cases, replayed from turn recordings, filed as one campaign and read
  for promotion. ``await writer_eval.run()`` returns a :class:`WriterResult`; ``python -m threetears.evals FILE``
  runs every writer eval a file defines. Its judge is any :class:`WriterJudge`, the shipped ``Model`` among them.
- :class:`WriterCaseFile` is one portable case (a frozen report snapshot and what minting it names), and
  :func:`read_writer_cases` reads a directory of them; minting one names a :class:`WriterSplit` and answers a
  :class:`MintedWriterCase`, on a :data:`WriterSide`.

Four modules build it, each for one part of a writer eval:

- :mod:`~threetears.evals.harness.writer_eval`: the eval, its arms and its result;
- :mod:`~threetears.evals.harness.writer_host`: a host that launches report writer arms over a store, beside the
  quick host's runs in it;
- :mod:`~threetears.evals.harness.writer_turns`: an arm's writer models, replayed from one recording per case, or
  recorded once;
- :mod:`~threetears.evals.harness.writer_cases`: a writer case frozen into a portable file, read back and minted
  into any scope.

This package sits above ``quick``: it composes the quick path with the surfaces above the engine (``ops``,
``actions``) and the replay format (``testing``), and nothing in the package imports it. It may import ``schema``,
``kernel``, ``run``, ``analysis``, ``storage``, ``testing``, ``quick``, ``ops``, ``actions`` and itself, and
``providers`` only as ``quick`` does, inside the function that builds a ``Model``, so importing it loads no client.

**This module is the package's public root.** An eval file imports from here and from no module below it, and only
the names in ``__all__``.
"""

from __future__ import annotations

from threetears.evals.analysis.analyst.skill import AnalystSkill, SkillText
from threetears.evals.harness.writer_cases import WriterCaseFile, WriterCaseFormat, read_writer_cases
from threetears.evals.harness.writer_eval import Writer, WriterEval, WriterResult
from threetears.evals.harness.writer_host import WriterJudge
from threetears.evals.ops.report_writer.writer_bank import MintedWriterCase, WriterSplit
from threetears.evals.ops.report_writer.writer_kind import WriterSide

__all__ = [
    "AnalystSkill",
    "MintedWriterCase",
    "SkillText",
    "Writer",
    "WriterCaseFile",
    "WriterCaseFormat",
    "WriterEval",
    "WriterJudge",
    "WriterResult",
    "WriterSide",
    "WriterSplit",
    "read_writer_cases",
]
