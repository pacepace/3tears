"""Batteries: run an eval in one call, and drive the engine from a command line.

The quickest way in. :func:`run_eval` takes a list of cases, an async candidate and scorer
functions, builds what a host would — a kind over the callable, a profile with one measure per
scorer, the in-memory reference store — launches through the engine's own launch path, and returns
an :class:`EvalSummary`. Handed each case's expected label (``expected=``), it grades the candidate as a
classifier, and the summary carries its confusion matrix (:class:`ConfusionCount`) and each label's
precision, recall and F1 (:class:`LabelStatistics`). Handed a :class:`Judge`, it also grades each answer
with a model against a rubric. Handed a :class:`World`, each case's starting state and goal-state checks, it
seeds every cell's world, hands the candidate :class:`WorldTools` that act on it, and grades the state it
leaves (:class:`GoalCheckSummary`). :func:`compare` runs two or more candidates over one case list the same way,
each as one arm, and returns a :class:`Comparison` whose campaign report tests every arm against the one
named the control. :func:`run_cli` is ``python -m threetears.evals``: ``run``, ``ls``, ``report``, ``bundle``
and ``spend`` over a host named ``module:factory``, or mounted under a product's own CLI with its host
factory and any subcommands of its own (:class:`HostCommand`).

This package composes the others and is composed by nothing: it may import ``contracts``, ``run``,
``analysis`` and ``storage``, and no package of the engine imports it.

**This module is the package's public root.** A host imports from here and from no module below it,
and only the names in ``__all__``.
"""

from __future__ import annotations

from threetears.evals.analysis.confusion import ConfusionCount, LabelStatistics
from threetears.evals.quick.cli import (
    DEFAULT_PROG,
    ENGINE_COMMANDS,
    EXIT_FAILED,
    EXIT_OK,
    EXIT_REFUSED,
    EXIT_RUN_DID_NOT_COMPLETE,
    HostCommand,
    HostFactory,
    build_parser,
    run_cli,
)
from threetears.evals.quick.compare import Comparison, compare
from threetears.evals.quick.one_call import (
    CALLABLE_KIND,
    CALLABLE_KIND_CONTRACT,
    CALLABLE_UNSEATED,
    JUDGED_CALLABLE_KIND,
    JUDGED_CALLABLE_KIND_CONTRACT,
    JUDGED_CALLABLE_UNSEATED,
    UNUSABLE_ANSWER,
    Candidate,
    ExpectedLabel,
    Scorer,
    callable_host,
    run_eval,
)
from threetears.evals.quick.judged import CaseMaterial, Judge
from threetears.evals.quick.world import CaseSeed, Dimension, ToolRefused, World, WorldCandidate, WorldTool, WorldTools
from threetears.evals.ops.summary import DimensionSummary, EvalSummary, GoalCheckSummary, MeasureSummary, summarize_run

__all__ = [
    "CALLABLE_KIND",
    "CALLABLE_KIND_CONTRACT",
    "CALLABLE_UNSEATED",
    "DEFAULT_PROG",
    "ENGINE_COMMANDS",
    "EXIT_FAILED",
    "EXIT_OK",
    "EXIT_REFUSED",
    "EXIT_RUN_DID_NOT_COMPLETE",
    "JUDGED_CALLABLE_KIND",
    "JUDGED_CALLABLE_KIND_CONTRACT",
    "JUDGED_CALLABLE_UNSEATED",
    "UNUSABLE_ANSWER",
    "Candidate",
    "CaseMaterial",
    "CaseSeed",
    "Comparison",
    "ConfusionCount",
    "Dimension",
    "DimensionSummary",
    "EvalSummary",
    "ExpectedLabel",
    "GoalCheckSummary",
    "HostCommand",
    "HostFactory",
    "Judge",
    "LabelStatistics",
    "MeasureSummary",
    "Scorer",
    "ToolRefused",
    "World",
    "WorldCandidate",
    "WorldTool",
    "WorldTools",
    "build_parser",
    "callable_host",
    "compare",
    "run_cli",
    "run_eval",
    "summarize_run",
]
