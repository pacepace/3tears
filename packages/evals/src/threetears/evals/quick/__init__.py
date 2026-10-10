"""Batteries: run an eval in one call, and drive the engine from a command line.

The quickest way in. :func:`run_eval` takes a list of cases, an async candidate and scorer
functions, builds what a host would — a kind over the callable, a profile with one measure per
scorer, the in-memory reference store — launches through the engine's own launch path, and returns
an :class:`EvalSummary`. Handed each case's expected label (``expected=``), it grades the candidate as a
classifier, and the summary carries its confusion matrix (:class:`ConfusionCount`) and each label's
precision, recall and F1 (:class:`LabelStatistics`). Handed a :class:`Judge`, it also grades each answer
with a model against a rubric. A candidate that calls a paid model returns an :class:`Answer` to report
what each answer spent, which the summary and the results' ``cost_usd`` carry. :func:`compare` runs two or
more candidates over one case list the same way, each as one arm, and returns a :class:`Comparison` whose
campaign report tests every arm against the one named the control; keyed by their level of each of several
factors (``factors=``), its arms are the cells of a factorial design, each factor a lever of its own. A scorer or
a judged dimension no arm may get worse on is declared a :class:`Guardrail` (``guardrails=``), decided for each
arm against the control apart from every contrast: held, breached or undecided. A candidate that calls tools declares them (``tools=``, each a
:data:`Tool`) and is handed them beside each case (:data:`ToolUsingCandidate`); a run with
``cassette_mode='capture'`` records what they answered, and ``'replay'`` serves that recording to every arm
in place of calling them. Handed a :class:`World`, each case's starting state and goal-state checks,
:func:`run_eval` seeds every cell's world, hands the candidate :class:`WorldTools` that act on it, and
grades the state it leaves (:class:`GoalCheckSummary`).
:func:`run_cli` is ``python -m threetears.evals``: ``run``, ``ls``, ``report``, ``bundle``, ``spend``, ``gate`` and ``frontier`` over
a host named ``module:factory``, or mounted under a product's own CLI with its host factory and any
subcommands of its own (:class:`HostCommand`).

**A host of your own** uses the same pieces without :func:`callable_host`: ``@measure(...)`` declares a measure on
the function that computes it (a :class:`Measure`, whose ``descriptor`` the host registers),
:func:`callable_kind` is the kind over a plain candidate and its scorers, declared under
:func:`callable_kind_contracts`, and a :class:`World`'s ``registry`` and ``bindings(state)`` are its world's
declaration and handles. Each is an ordinary contract object, mixed freely with ones the host writes by hand.

This package composes the others and is composed by nothing: it may import ``schema``, ``kernel``,
``run``, ``analysis`` and ``storage``, and no package of the engine imports it.

**This module is the package's public root.** A host imports from here and from no module below it,
and only the names in ``__all__``.
"""

from __future__ import annotations

from threetears.evals.analysis.confusion import ConfusionCount, LabelStatistics
from threetears.evals.quick.answer import Answer
from threetears.evals.quick.cli import (
    DEFAULT_PROG,
    ENGINE_COMMANDS,
    EXIT_FAILED,
    EXIT_GATE_FAILED,
    EXIT_OK,
    EXIT_REFUSED,
    EXIT_RUN_DID_NOT_COMPLETE,
    HostCommand,
    HostFactory,
    build_parser,
    run_cli,
)
from threetears.evals.quick.compare import ArmKey, Comparison, compare
from threetears.evals.quick.one_call import (
    ARM_LEVER,
    CALLABLE_KIND,
    CALLABLE_KIND_CONTRACT,
    CALLABLE_UNSEATED,
    DEFAULT_QUICK_SCOPE,
    JUDGED_CALLABLE_KIND,
    JUDGED_CALLABLE_KIND_CONTRACT,
    JUDGED_CALLABLE_UNSEATED,
    SHARED_ARM_MODEL,
    UNUSABLE_ANSWER,
    Candidate,
    ExpectedLabel,
    Scorer,
    callable_host,
    callable_kind,
    callable_kind_contracts,
    run_eval,
)
from threetears.evals.quick.guardrails import Guardrail, GuardrailDirection
from threetears.evals.quick.judged import CaseMaterial, Judge
from threetears.evals.quick.measures import Measure, measure
from threetears.evals.quick.tools import CandidateTools, Tool, ToolUsingCandidate
from threetears.evals.quick.world import CaseSeed, Dimension, ToolRefused, World, WorldCandidate, WorldTool, WorldTools
from threetears.evals.analysis.summary import (
    CaseOutcome,
    CaseResult,
    DimensionSummary,
    EvalSummary,
    GoalCheckSummary,
    JudgeGrade,
    RunMeasureSummary,
    summarize_run,
)

__all__ = [
    "ARM_LEVER",
    "CALLABLE_KIND",
    "CALLABLE_KIND_CONTRACT",
    "CALLABLE_UNSEATED",
    "DEFAULT_PROG",
    "DEFAULT_QUICK_SCOPE",
    "ENGINE_COMMANDS",
    "EXIT_FAILED",
    "EXIT_GATE_FAILED",
    "EXIT_OK",
    "EXIT_REFUSED",
    "EXIT_RUN_DID_NOT_COMPLETE",
    "JUDGED_CALLABLE_KIND",
    "JUDGED_CALLABLE_KIND_CONTRACT",
    "JUDGED_CALLABLE_UNSEATED",
    "SHARED_ARM_MODEL",
    "UNUSABLE_ANSWER",
    "Answer",
    "ArmKey",
    "Candidate",
    "CandidateTools",
    "CaseMaterial",
    "CaseOutcome",
    "CaseResult",
    "CaseSeed",
    "Comparison",
    "ConfusionCount",
    "Dimension",
    "DimensionSummary",
    "EvalSummary",
    "ExpectedLabel",
    "GoalCheckSummary",
    "Guardrail",
    "GuardrailDirection",
    "HostCommand",
    "HostFactory",
    "Judge",
    "JudgeGrade",
    "LabelStatistics",
    "Measure",
    "RunMeasureSummary",
    "Scorer",
    "Tool",
    "ToolUsingCandidate",
    "ToolRefused",
    "World",
    "WorldCandidate",
    "WorldTool",
    "WorldTools",
    "build_parser",
    "callable_host",
    "callable_kind",
    "callable_kind_contracts",
    "compare",
    "measure",
    "run_cli",
    "run_eval",
    "summarize_run",
]
