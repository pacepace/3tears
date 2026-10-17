"""The grader slot: what counts as right, in four forms, and :func:`grader` for what a reading declares beside it.

An :class:`~threetears.evals.quick.contract.Eval`'s ``grader`` is one grader or a list of them, each one of:

- **A function** ``(case, out) -> bool | float``. Its ``__name__`` names its measure and the first line of its
  docstring describes it; higher is better, and one annotated ``-> bool`` is a pass/fail on 0 to 1.
- **A field name** (``"category"``): the answer's ``category`` (``out.category`` on a pydantic model,
  ``out["category"]`` on a mapping, the answer itself when it is a plain value) against the case's ``category``. A
  field is graded as a label: the first field listed is the run's classifier, so the run carries its confusion
  matrix, each label's precision, recall and F1, and ``accuracy``
  (:attr:`~threetears.evals.analysis.EvalSummary.confusion`); every other field is a pass/fail of its own name, and
  :meth:`~threetears.evals.quick.result.Result.labels` counts its matrix the same way. A label is the value's
  text: a string as written, anything else as JSON (``true``, ``3``).
- **An expression** (``"state.light == 'on'"``, ``'calls("room.switch") <= 1'``): a goal-state check over the world
  the runner left and the calls it made, accepted only when the runner is a world's.
- **A** :class:`~threetears.evals.quick.judged.Judge`, scoring one dimension per rubric key.

Any function or field may be wrapped by :func:`grader` to declare what the reading needs beside its grade: a margin
for equivalence (``margin=``), a guardrail no arm may get worse on by more than its margin (``guard=``), or the range
a numeric grade lies in (``range=``). Every grade is higher-is-better, so a guardrail's direction is never stated.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, overload

from pydantic import BaseModel

from threetears.evals.kernel import ACCURACY_MEASURE, METRIC_DESCRIPTORS
from threetears.evals.quick.guardrails import Guardrail
from threetears.evals.quick.judged import Judge
from threetears.evals.quick.world import WorldOutcome

#: A function grader: takes the case and the runner's answer, returns a number (``True``/``False`` as 1 and 0).
GraderFunction = Callable[[Mapping[str, Any], Any], float | bool]

#: What an answer with no usable label in a field is counted under: no label, a blank one, or a value that is no
#: label (a list, a mapping). Its own cell in the confusion matrix, never a match.
UNUSABLE_ANSWER = "(unusable answer)"

#: What :func:`field_value` returns for an answer that has no such field.
_MISSING = object()


def _a_number(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int | float) and math.isfinite(value)


@dataclass(frozen=True)
class Grader:
    """A function or a field with what its reading declares: a margin, a guardrail, or a range.

    Built by :func:`grader`; listed in an eval's ``grader`` as the grader it wraps would be.

    :ivar target: The function, or the field name.
    :ivar margin: The most a difference in it may be and still be too small to act on, so a contrast on it can read
        ``equivalent``; ``None`` declares none.
    :ivar guard: Makes it a guardrail with this margin: decided for each arm against the control apart from every
        contrast, ``held``, ``breached`` or ``undecided``; ``None`` keeps it a quality reading.
    :ivar range: The lowest and highest grade a function returning a number can give; ``None`` for a pass/fail.
    """

    target: Callable[[Mapping[str, Any], Any], float | bool] | str
    margin: float | None = None
    guard: float | None = None
    range: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        """Refuse a declaration no reading could carry.

        :raises ValueError: A target that is neither a function nor a field name; a margin or guard that is not a
            positive number; both a margin and a guard; or a range on a field, or one that is not two numbers, low
            first.
        """
        if isinstance(self.target, Judge):
            raise ValueError("a Judge declares its guardrails itself (Judge(guard={'dimension': 0.02})); list it bare")
        if not (isinstance(self.target, str) and self.target.isidentifier()) and not callable(self.target):
            raise ValueError(f"grader() wraps a function or a field name, not {self.target!r}")
        for name in ("margin", "guard"):
            value = getattr(self, name)
            if value is not None and (not _a_number(value) or value <= 0):
                raise ValueError(f"grader({name}=...) is a positive number in the reading's own units, not {value!r}")
        if self.margin is not None and self.guard is not None:
            raise ValueError(
                "a guardrail is never contrasted, so a margin for equivalence has no use on it: pass margin= or guard="
            )
        if self.range is not None:
            if isinstance(self.target, str):
                raise ValueError(f"the field {self.target!r} is graded pass/fail, on 0 to 1 already; it takes no range")
            low_high = self.range
            if (
                not isinstance(low_high, tuple | list)
                or len(low_high) != 2
                or not all(_a_number(end) for end in low_high)
                or not low_high[0] < low_high[1]
            ):
                raise ValueError(
                    f"grader(range=...) is the lowest and highest grade it can return, low first: (1, 5), not "
                    f"{low_high!r}"
                )
            object.__setattr__(self, "range", (float(low_high[0]), float(low_high[1])))


@overload
def grader(
    target: None = None,
    /,
    *,
    margin: float | None = None,
    guard: float | None = None,
    range: tuple[float, float] | None = None,
) -> Callable[[Callable[[Mapping[str, Any], Any], float | bool]], Grader]: ...


@overload
def grader(
    target: Callable[[Mapping[str, Any], Any], float | bool] | str,
    /,
    *,
    margin: float | None = None,
    guard: float | None = None,
    range: tuple[float, float] | None = None,
) -> Grader: ...


def grader(
    target: Callable[[Mapping[str, Any], Any], float | bool] | str | None = None,
    /,
    *,
    margin: float | None = None,
    guard: float | None = None,
    range: tuple[float, float] | None = None,
) -> Grader | Callable[[Callable[[Mapping[str, Any], Any], float | bool]], Grader]:
    """Declare what a reading needs beside its grade: ``@grader(guard=0.02)``, ``grader("total", margin=0.05)``.

    :param target: The function or field name to wrap; ``None`` returns a decorator that wraps the function it
        decorates.
    :ptype target: Callable[[Mapping[str, Any], Any], float | bool] | str | None
    :param margin: The equivalence margin, in the reading's own units (0.05 is five points on a pass rate).
    :ptype margin: float | None
    :param guard: Makes the reading a guardrail with this margin.
    :ptype guard: float | None
    :param range: The lowest and highest grade a function returning a number gives.
    :ptype range: tuple[float, float] | None
    :return: The :class:`Grader`, or, with no ``target``, the decorator building one.
    :rtype: Grader | Callable[[Callable[[Mapping[str, Any], Any], float | bool]], Grader]
    :raises ValueError: As :class:`Grader` refuses.
    """
    if target is None:

        def wrap(function: GraderFunction) -> Grader:
            return Grader(function, margin=margin, guard=guard, range=range)

        return wrap
    return Grader(target, margin=margin, guard=guard, range=range)


def field_value(out: Any, name: str) -> Any:
    """The answer's value of one field: an attribute of a pydantic model, a key of a mapping, or the answer itself.

    :param out: The runner's answer; a world runner's is read through its ``answer``.
    :ptype out: Any
    :param name: The field.
    :ptype name: str
    :return: The value as JSON would carry it, or a sentinel the label reads as unusable when the answer has no such
        field.
    :rtype: Any
    """
    if isinstance(out, WorldOutcome):
        out = out.answer
    if isinstance(out, BaseModel):
        out = out.model_dump(mode="json")
    if isinstance(out, Mapping):
        return out.get(name, _MISSING)
    if out is None or isinstance(out, str | int | float | bool):
        return out
    return _MISSING


def label_text(value: Any) -> str | None:
    """A value as a label: a non-blank string as written, a number or a bool as JSON; ``None`` when it is no label."""
    if value is _MISSING or value is None:
        return None
    if isinstance(value, str):
        return value if value.strip() else None
    if isinstance(value, bool) or _a_number(value):
        return json.dumps(value)
    return None


def predicted_label(out: Any, name: str | None) -> str:
    """The label an answer gives in field ``name`` (or as itself, with no field): :data:`UNUSABLE_ANSWER` when none."""
    value = out if name is None else field_value(out, name)
    if name is None and not isinstance(value, str):
        return UNUSABLE_ANSWER
    return label_text(value) or UNUSABLE_ANSWER


def expected_label(case: Mapping[str, Any], name: str) -> str:
    """The label a case expects in field ``name``.

    :raises ValueError: The case has no such field, or its value is no label (blank, ``None``, a list, a mapping), or is
        :data:`UNUSABLE_ANSWER`.
    """
    if name not in case:
        raise ValueError(f"the field grader {name!r} reads the case's {name!r}, which it lacks")
    label = label_text(case[name])
    if label is None:
        raise ValueError(
            f"the case's {name!r} is {case[name]!r}; a field grader's label is a non-blank string, a number or a bool"
        )
    if label == UNUSABLE_ANSWER:
        raise ValueError(f"the case's {name!r} is {UNUSABLE_ANSWER!r}, the label an unusable answer is counted under")
    return label


def field_scorer(name: str) -> GraderFunction:
    """A field graded as a pass/fail of its own name: the answer's label in it is the case's.

    A case whose field is ``null`` expects the field absent: an answer with no value there (``None``, or no such
    field) passes, and any value fails, shown with the answer as every miss is. Only the classifier, which counts
    labels, refuses a ``null``.

    :param name: The field.
    :ptype name: str
    :return: The grading function, named ``name`` and annotated ``-> bool``.
    :rtype: GraderFunction
    """

    def matches(case: Mapping[str, Any], out: Any) -> bool:
        if name in case and case[name] is None:
            value = field_value(out, name)
            return value is None or value is _MISSING
        return predicted_label(out, name) == expected_label(case, name)

    matches.__name__ = name
    matches.__qualname__ = name
    matches.__doc__ = f"Whether the answer's {name} is the case's."
    matches.__annotations__ = {"case": "Mapping[str, Any]", "out": "Any", "return": "bool"}
    return matches


@dataclass
class GraderPlan:
    """An eval's graders, sorted by what reads them.

    :ivar scorers: Every function grader, and every field after the classifier, as the measures they report.
    :ivar classifier: The field the run classifies on (the first field listed), or ``None``.
    :ivar fields: Every field grader, in order, the classifier first.
    :ivar goal_checks: Every expression grader.
    :ivar judge: The judge, or ``None``.
    :ivar margins: Each scorer's equivalence margin, by name.
    :ivar ranges: Each numeric scorer's range, by name.
    :ivar guardrails: Each scorer declared a guardrail, by name.
    :ivar classifier_margin: The classifier's accuracy margin, declared on the runs; ``None`` for none.
    """

    scorers: list[GraderFunction] = field(default_factory=list)
    classifier: str | None = None
    fields: list[str] = field(default_factory=list)
    goal_checks: list[str] = field(default_factory=list)
    judge: Judge | None = None
    margins: dict[str, float] = field(default_factory=dict)
    ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    guardrails: dict[str, Guardrail] = field(default_factory=dict)
    classifier_margin: float | None = None

    @property
    def run_margins(self) -> dict[str, float] | None:
        """The margins declared on every run: the classifier's accuracy margin, when it has one."""
        return None if self.classifier_margin is None else {ACCURACY_MEASURE: self.classifier_margin}


def plan_graders(graders: Any) -> GraderPlan:
    """Sort an eval's ``grader`` slot into what reads each grader.

    :param graders: One grader, or a list of them, each a function, a field name, an expression, a
        :class:`~threetears.evals.quick.judged.Judge` or a :class:`Grader`.
    :ptype graders: Any
    :return: The plan.
    :rtype: GraderPlan
    :raises ValueError: No grader; two judges; a grader of no known form; a field named after an engine core measure; or
        a name graded twice.
    """
    listed: Sequence[Any] = graders if isinstance(graders, list | tuple) else [graders]
    if not listed:
        raise ValueError(
            "an eval needs at least one grader — a function, a field name, an expression or a Judge: a run nothing "
            "grades measures nothing"
        )
    plan = GraderPlan()
    seen: list[str] = []
    for given in listed:
        declared = given if isinstance(given, Grader) else None
        target = declared.target if declared is not None else given
        if isinstance(target, Judge):
            if plan.judge is not None:
                raise ValueError("an eval is graded by one judge; list one Judge, with every dimension in its rubric")
            plan.judge = target
            continue
        if isinstance(target, str):
            if not target.isidentifier():
                if declared is not None:
                    raise ValueError(
                        f"grader() declares a reading on a function or a field, not the expression {target!r}"
                    )
                plan.goal_checks.append(target)
                continue
            if target in METRIC_DESCRIPTORS:
                raise ValueError(
                    f"a field grader's reading is named after its field, and {target!r} is an engine core measure's "
                    "name; grade it with a function of another name"
                )
            seen.append(target)
            plan.fields.append(target)
            guard = declared.guard if declared is not None else None
            margin = declared.margin if declared is not None else None
            if plan.classifier is None and guard is None:
                plan.classifier = target
                plan.classifier_margin = margin
                continue
            target = field_scorer(target)
        elif not callable(target):
            raise ValueError(
                f"a grader is a function (case, out) -> bool | float, a field name, an expression over a world, a "
                f"Judge or a grader(...), not {target!r}"
            )
        else:
            name = getattr(target, "__name__", None)
            if not isinstance(name, str) or not name.isidentifier():
                raise ValueError(
                    f"a grader's __name__ names its measure, and {target!r} has none a measure can carry; write each "
                    "grader as a def"
                )
            seen.append(name)
        plan.scorers.append(target)
        name = target.__name__
        if declared is not None:
            if declared.margin is not None:
                plan.margins[name] = declared.margin
            if declared.range is not None:
                plan.ranges[name] = declared.range
            if declared.guard is not None:
                plan.guardrails[name] = Guardrail(margin=declared.guard, direction="higher_is_better")
    if repeated := sorted({name for name in seen if seen.count(name) > 1}):
        raise ValueError(f"{', '.join(repeated)} is graded more than once; each grader is one reading")
    return plan


__all__ = [
    "UNUSABLE_ANSWER",
    "Grader",
    "GraderFunction",
    "GraderPlan",
    "expected_label",
    "field_scorer",
    "field_value",
    "grader",
    "label_text",
    "plan_graders",
    "predicted_label",
]
