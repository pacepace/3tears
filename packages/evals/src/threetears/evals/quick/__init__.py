"""Batteries: run an eval in one call, and drive the engine from a command line.

The quickest way in. :func:`run_eval` takes a list of cases, an async candidate and scorer
functions, builds what a host would — a kind over the callable, a profile with one measure per
scorer, the in-memory reference store — launches through the engine's own launch path, and returns
an :class:`EvalSummary`. :func:`run_cli` is ``python -m threetears.evals``: ``run``, ``ls`` and
``report`` over a host named ``module:factory``, or mounted under a product's own CLI with its host
factory.

This package composes the others and is composed by nothing: it may import ``contracts``, ``run``,
``analysis`` and ``storage``, and no package of the engine imports it.

**This module is the package's public root.** A host imports from here and from no module below it,
and only the names in ``__all__``.
"""

from __future__ import annotations

from threetears.evals.quick.cli import DEFAULT_PROG, HostFactory, build_parser, run_cli
from threetears.evals.quick.one_call import CALLABLE_KIND, Candidate, Scorer, callable_host, run_eval
from threetears.evals.quick.summary import EvalSummary, MeasureSummary, summarize_run

__all__ = [
    "CALLABLE_KIND",
    "DEFAULT_PROG",
    "Candidate",
    "EvalSummary",
    "HostFactory",
    "MeasureSummary",
    "Scorer",
    "build_parser",
    "callable_host",
    "run_cli",
    "run_eval",
    "summarize_run",
]
