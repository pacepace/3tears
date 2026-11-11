"""``Judge``: a model grading an eval's answers against a rubric, through the engine's own judge.

A function grader is code, and some answers have no code that grades them: whether a reply is helpful, whether it
says only what its source supports. Listed in an eval's ``grader``, a :class:`Judge` asks a model instead, one rubric
dimension per call, and each dimension's scores are reported beside the other graders' measures. Nothing here
judges: it declares what the engine's judge reads and who calls the model.

- **The model** is an id, a :class:`~threetears.evals.providers.Model` (whose ``recording=`` covers the judge's
  calls), or any completion client that names its model; :func:`judge_client` reads it.
- **The rubric** is a mapping of dimension name to what the dimension measures, each scored pass/fail, or the
  engine's own :class:`~threetears.evals.schema.RubricDim` values for another scale. A bare name is namespaced under
  ``answer.`` (``helpful`` becomes ``answer.helpful``), because a judge config binds to a dimension by name across
  every template and a bare one would bind to every product's ``helpful``. The rubric is the template's, so it is
  part of what the case set's id is a digest of.
- **The judge is declared by the Judge itself.** Every dimension it builds, or is handed without a judge of its own,
  is declared ``"builtin"``: scored by the engine's own prompt on :attr:`Judge.client`. A
  :class:`~threetears.evals.schema.RubricDim` handed in declaring ``"configured"`` keeps it, and the run is held to
  it: refused unless the store it runs on holds an active judge config for that dimension.
- **The judge** is :class:`~threetears.evals.run.JudgeService`, built by
  :func:`~threetears.evals.run.build_judge_service` as for any judged run, over a client factory that hands it
  :attr:`Judge.client` for the judge role and nothing else. It scores the answer as a document: the rubric dimensions
  alone, each read against the case material, with no conversation axes.
- **The evidence** is the kind's to render (:func:`judge_evidence`): the answer as text, judged against
  :attr:`Judge.case_material`'s rendering of its case, or the case as JSON. It is stored on each cell's trace, so a
  re-judge sends exactly what the first judge read.
- **The spend** is the client's to report. Each call's tokens and dollars land on the result's ``judge`` usage row;
  a client that prices nothing leaves the judge's spend unknown, never zero.
- **A guarded dimension** (``guard=``) is a guardrail: scored on the boundary axis, kept out of every contrast, and
  decided for each arm against the control on the margin given.

**Examples check it in the run, and labels measure it.** A judge is checked before it is leaned on, two ways. The
first items to check it on are hard negatives: an answer that must fail a dimension, and one that looks wrong and
must pass it. Each is an :class:`Example`, and an eval judged by the Judge grades each one ``k`` times as part of its
run, through the call its answers are graded by, and reports how many it graded as they must be
(:class:`JudgeExamples`); it also stores them as authored judge cases
(:class:`~threetears.evals.kernel.AuthoredJudgeCase`), for a host's judge campaign to measure the judge on.
``labels=`` names a CSV of a person's verdicts on judged answers (the ``to_label.csv`` a written result leaves,
filled in), and the result reports how far the judge agrees with them.

**The caller owns the client.** The engine releases every client it is handed when a run ends; the judge's is lent
to it through :class:`BorrowedJudgeClient`, whose release does nothing, so one client serves every call that names it
and is closed by whoever opened it.
"""

from __future__ import annotations

import inspect
import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Self

from pydantic import ValidationError

from threetears.evals.analysis.report.forms import PageForms
from threetears.evals.analysis.summary import RoleSpend
from threetears.evals.schema import SCALES, CompletionClient, CompletionResult, JudgeEvidence, RubricDim
from threetears.evals.kernel.eval_file import beside_eval_file
from threetears.evals.kernel.host import CompletionClients, CompletionRole
from threetears.evals.kernel import AuthoredJudgeCase, AuthoredVerdict

#: Renders the material one case's answer is judged against: takes the case, returns non-blank text.
CaseMaterial = Callable[[Mapping[str, Any]], str]

#: What the judge reads in place of an answer that renders as blank text, which no judge could quote.
EMPTY_ANSWER = "(the candidate's answer was empty)"


#: Who an example's verdict is recorded as written by when it names no author: the label's rater.
EXAMPLE_AUTHOR = "an Example"

#: The namespace a rubric key is stored under: ``helpful`` is stored as ``answer.helpful``.
DIMENSION_CONTEXT = "answer"


@dataclass(frozen=True)
class Example:
    """An answer the judge must grade a known way: one that must fail a dimension, or one that looks wrong and must
    pass.

    :ivar dimension: The rubric dimension it is graded on, as :attr:`Judge.rubric` names it (bare, or namespaced).
    :ivar answer: The answer, rendered as the judge reads a runner's (a string as written, anything else as JSON).
    :ivar expected: The verdict it must get: ``"pass"`` or ``"fail"`` on a pass/fail dimension, a score from 1 to 5 on
        an ordinal one. One the dimension's scale cannot hold is refused when the :class:`Judge` is built.
    :ivar note: Why: the reason its verdict is what it is, in the author's words.
    :ivar case: The case it answers, rendered by :attr:`Judge.case_material` as a run's case is.
    :ivar author: Who wrote it, recorded as the verdict's rater.
    """

    dimension: str
    answer: Any
    expected: AuthoredVerdict
    note: str
    case: Mapping[str, Any] = field(default_factory=dict)
    author: str = EXAMPLE_AUTHOR


#: How long an example's answer is shown where a miss quotes it, an ellipsis marking the cut.
_QUOTED_ANSWER_CHARS = 120


@dataclass(frozen=True)
class ExampleGrade:
    """How the judge graded one :class:`Example` on each repeat of the run, beside the verdict it must get.

    :ivar example: The example, as written.
    :ivar dimension: Its dimension as the rubric stores it (``answer.grounded``).
    :ivar expected: The verdict it must get, as a verdict is written here: ``"pass"`` or ``"fail"``, or a score from
        ``"1"`` to ``"5"``.
    :ivar verdicts: The verdict the judge gave on each repeat that it answered, in order, written as ``expected`` is,
        or ``"can't tell"`` where it answered that it could not tell.
    :ivar errors: Why each repeat the judge gave no verdict on gave none (its call failed, or its reply would not
        parse), in order: the rig's or the provider's, so never read as the judge grading it wrong. With
        ``verdicts``, one per repeat asked: the eval's ``k``, or fewer where its cost cap stopped the grading.
    """

    example: Example
    dimension: str
    expected: str
    verdicts: tuple[str, ...]
    errors: tuple[str, ...] = ()

    @property
    def as_it_must(self) -> bool:
        """Whether every repeat asked gave ``expected``: a verdict on each, every one the one it must get.

        A repeat with any other verdict is a miss: a judge that grades a known answer wrong one time in ``k`` grades
        the run's answers wrong as often."""
        return bool(self.verdicts) and not self.errors and all(verdict == self.expected for verdict in self.verdicts)

    @property
    def missed(self) -> bool:
        """Whether the judge gave it a verdict it must not get on some repeat (``"can't tell"`` included)."""
        return any(verdict != self.expected for verdict in self.verdicts)

    def miss_text(self) -> str:
        """The miss in words: the answer quoted, what it was graded on each repeat, and what it must get and why.

        :return: ``"Done: your year is refunded in full." graded pass on grounded, must fail (no refund after 30
            days)``; with repeats that differ, each verdict with how many of them gave it.
        :rtype: str
        """
        answer = " ".join(_answer_text(self.example.answer).split())
        if len(answer) > _QUOTED_ANSWER_CHARS:
            answer = answer[: _QUOTED_ANSWER_CHARS - 1] + "…"
        counted = {verdict: self.verdicts.count(verdict) for verdict in self.verdicts}
        if len(counted) == 1:
            (got,) = counted
        else:
            got = " and ".join(f"{verdict} {count} of {len(self.verdicts)} times" for verdict, count in counted.items())
        return f'"{answer}" graded {got} on {self.example.dimension}, must {self.expected} ({self.example.note})'


@dataclass(frozen=True)
class JudgeExamples:
    """How the judge graded its examples in the run: each one on every repeat, and what that cost.

    Reported beside the judge's agreement with people's labels, never pooled into it: the examples are a few answers
    written to be hard, not a sample of the run's, and a handful reaches no tier's floor.

    :ivar grades: Each example's grades, in the order the judge lists them.
    :ivar repeats: How many times each was to be graded: the eval's ``k``.
    :ivar spend: What the judge's calls on them cost, under the ``judge`` role, apart from every arm's results: calls
        made outside any run, each written to the out-of-run ledger under purpose ``judge``.
    :ivar stopped: Why the grading stopped before every repeat of every example was asked, as a clause that follows
        "since" (the eval's cost cap, with what the eval landed against it; spend nobody priced; a grading that
        failed, with its cause), or ``None`` where nothing did.
    """

    grades: tuple[ExampleGrade, ...]
    repeats: int
    spend: RoleSpend
    stopped: str | None = None

    @property
    def as_they_must(self) -> int:
        """How many examples were graded as they must be on every repeat (:attr:`ExampleGrade.as_it_must`)."""
        return sum(self._in_full(grade) for grade in self.grades)

    def _in_full(self, grade: ExampleGrade) -> bool:
        """Whether ``grade`` was graded as it must be on every one of the eval's repeats."""
        return grade.as_it_must and len(grade.verdicts) == self.repeats

    @property
    def misses(self) -> tuple[ExampleGrade, ...]:
        """The examples the judge gave a verdict they must not get, in order."""
        return tuple(grade for grade in self.grades if grade.missed)

    @property
    def unread(self) -> tuple[ExampleGrade, ...]:
        """The examples neither graded as they must be nor missed: a repeat gave no verdict, or was never asked."""
        return tuple(grade for grade in self.grades if not self._in_full(grade) and not grade.missed)

    def _over(self) -> str:
        """What the count is over (every repeat, where there are several), and why any example went unread."""
        over = "" if self.repeats == 1 else f" on every one of {self.repeats} repeats"
        if unread := self.unread:
            errors = [error for grade in unread for error in grade.errors]
            why = self.stopped if self.stopped is not None else f"the judge's call failed: {errors[0]}"
            over += f"; {len(unread)} not graded in full, since {why}"
        return over

    @property
    def _none_graded(self) -> bool:
        """Whether the grading stopped before the judge gave any example a verdict or an error."""
        return self.stopped is not None and not any(grade.verdicts or grade.errors for grade in self.grades)

    def _cost(self) -> str:
        """What the calls cost, in parentheses after a leading space, or nothing where none was made."""
        return f" ({self.spend.money_text()})" if self.spend.calls else ""

    def lines(self) -> list[str]:
        """The quiet print's lines: how many were graded as they must be and what that cost, then each miss.

        :return: ``judge: 3 of 4 examples graded as they must be on every one of 2 repeats ($0.0010 over 8 call(s))``,
            then one indented ``missed:`` line per miss; ``judge: 4 examples not graded, since ...`` where the grading
            stopped before any was.
        :rtype: list[str]
        """
        if self._none_graded:
            return [f"judge: {len(self.grades)} examples not graded, since {self.stopped}{self._cost()}"]
        counted = f"{self.as_they_must} of {len(self.grades)} examples graded as they must be{self._over()}"
        return [f"judge: {counted}{self._cost()}", *(f"  missed: {grade.miss_text()}" for grade in self.misses)]

    def sentence(self) -> str:
        """What a page says: one sentence with what the calls cost, as the print says it, then its misses.

        :return: ``The judge graded 3 of 4 of its examples as they must be on every one of 2 repeats ($0.0010 over 8
            call(s)). Missed: ...``.
        :rtype: str
        """
        if self._none_graded:
            return f"The judge's {len(self.grades)} examples were not graded, since {self.stopped}{self._cost()}."
        said = (
            f"The judge graded {self.as_they_must} of {len(self.grades)} of its examples as they must "
            f"be{self._over()}{self._cost()}."
        )
        if self.misses:
            said += f" Missed: {'; '.join(grade.miss_text() for grade in self.misses)}."
        return said


class Judge:
    """A model that grades each answer on a rubric, one call per dimension: a grader in an eval's ``grader`` list.

    :ivar client: The completion client the judge calls. Lent to the engine for the run and never closed by it. The
        judge asks for a sampling temperature on every call (0, unless a judge config states another): when the
        client's ``generate`` takes a ``temperature`` keyword it is passed on every call, and the client reports
        what it actually sent on each completion's ``temperature``, which every score records as part of who
        judged.
    :ivar model: The model ``client`` calls, as the run records it: the run's judge pin, and the arm's judge in every
        comparison of two runs.
    :ivar rubric: The dimensions as given: each a name and what it measures (a pass/fail), or
        :class:`~threetears.evals.schema.RubricDim` values for a scoring guide or a 1-5 scale.
    :ivar examples: Answers the judge must grade a known way (:class:`Example`), graded in every run it judges.
    :ivar labels: The CSV of a person's labels the judge is calibrated against, or ``None``.
    :ivar case_material: Renders what an answer is judged against from its case, or ``None`` for the case as JSON.
    :ivar guard: Each dimension no arm may get worse on, by its rubric key, with the margin it may: scored on the
        boundary axis and decided against the control apart from every contrast.
    """

    def __init__(
        self,
        model: str | CompletionClient,
        rubric: Mapping[str, str] | Sequence[RubricDim],
        *,
        examples: Sequence[Example] = (),
        labels: str | os.PathLike[str] | None = None,
        case_material: Callable[[Mapping[str, Any]], str] | None = None,
        guard: Mapping[str, float] | None = None,
    ) -> None:
        """Declare the judge.

        :param model: What judges: a model id (``"deepseek/deepseek-v4-flash"``), a
            :class:`~threetears.evals.providers.Model` (its ``recording=`` with it), or any completion client
            that names its model (``model_name``).
        :ptype model: str | CompletionClient
        :param rubric: The dimensions, each a name and what it measures, scored pass/fail; or
            :class:`~threetears.evals.schema.RubricDim` values, each stating its own scale. A bare name is stored
            under ``answer.`` (``helpful`` is ``answer.helpful``), since a judge config binds to a dimension by
            name across every template. The rubric is the template's, so it is part of what the case set's id
            digests.
        :ptype rubric: Mapping[str, str] | Sequence[RubricDim]
        :param examples: Hard negatives, and answers that look wrong and must pass. An eval judged by this judge
            grades each one ``k`` times in its run, as it grades the run's answers, and its result says how many it
            graded as they must be (``Result.judge_examples``); it also stores them as authored judge cases, the
            items a host's judge campaign calibrates it on (:meth:`authored_cases`).
        :ptype examples: Sequence[Example]
        :param labels: A CSV of a person's verdicts on judged answers, in the columns
            :meth:`~threetears.evals.quick.contract.Result.write` writes ``to_label.csv`` in: the result reports
            how far the judge agrees with them (kappa, its lower bound and the evidence tier). A file that is not
            there yet reads as no labels, said by name. A relative path is read beside the eval file being
            loaded, else in the working directory.
        :ptype labels: str | os.PathLike[str] | None
        :param case_material: Renders what an answer is judged against from its case. ``None`` shows the judge the case
            as JSON, less every top-level field holding the classifier's expected label.
        :ptype case_material: Callable[[Mapping[str, Any]], str] | None
        :param guard: Each dimension no arm may get worse on than the control by more than its margin, by rubric key
            (``{"grounded": 0.02}``).
        :ptype guard: Mapping[str, float] | None
        :raises ValueError: A client naming no model; an empty rubric, a dimension the engine refuses, or two under one
            name; an example naming no dimension, giving a verdict its scale cannot hold, or whose case cannot be
            rendered; or a guard on no dimension, or one not narrower than the dimension's scale.
        """
        self.client, self.model = judge_client(model)
        self.rubric = rubric
        self.examples = tuple(examples)
        self.labels = None if labels is None else beside_eval_file(labels)
        self.case_material = case_material
        self.guard = dict(guard or {})
        dims = self._build_dims()
        if not dims:
            raise ValueError("a judge needs at least one rubric dimension; a rubric with none scores nothing")
        names = [dim.name for dim in dims]
        if repeated := sorted({name for name in names if names.count(name) > 1}):
            raise ValueError(f"rubric dimensions named {', '.join(repeated)} more than once; each name is one score")
        by_name = {dim.name: dim for dim in dims}
        for key, margin in self.guard.items():
            dim = by_name.get(self.dim_name(key))
            if dim is None:
                raise ValueError(
                    f"Judge(guard=...) names {key!r}, which is no dimension of the rubric "
                    f"({', '.join(sorted(by_name))})"
                )
            low, high = SCALES[dim.scale].value_range
            if isinstance(margin, bool) or not isinstance(margin, int | float) or not 0 < margin < high - low:
                forms = PageForms.off_page()
                ends = [forms.echoed(end, measure=dim.name, kind="judged").text for end in (low, high)]
                raise ValueError(
                    f"the guard on {key!r} is how much worse than the control an arm may be on it, a positive number "
                    f"narrower than its scale ({ends[0]} to {ends[1]}), not {margin!r}"
                )
        guarded = {self.dim_name(key) for key in self.guard}
        # A guarded dimension is a guardrail: scored on the boundary axis, which the judge stamps on every score.
        self._dims = tuple(dim.model_copy(update={"axis": "boundary"}) if dim.name in guarded else dim for dim in dims)
        # Built once here, against a stand-in scenario, so a bad example is refused where it is written.
        self.authored_cases(intent="the examples are checked when the judge is built")

    def __repr__(self) -> str:
        """The judge, by model and dimensions."""
        return f"Judge({self.model!r}, {[dim.name for dim in self._dims]!r})"

    @property
    def guard_margins(self) -> dict[str, float]:
        """Each guarded dimension's margin, by its stored name."""
        return {self.dim_name(key): float(margin) for key, margin in self.guard.items()}

    def _build_dims(self) -> list[RubricDim]:
        # The Judge is the declaration: the engine's built-in prompt, on this judge's client, for every dimension that
        # states no judge of its own — so the quick path asks nothing of a caller that a host's template must state.
        if isinstance(self.rubric, Mapping):
            return [
                RubricDim(name=self.dim_name(name), description=description, scale="pass_fail", judge="builtin")
                for name, description in self.rubric.items()
            ]
        return [dim if dim.judge is not None else dim.model_copy(update={"judge": "builtin"}) for dim in self.rubric]

    def dim_name(self, name: str) -> str:
        """``name`` as a stored dimension name: as given when it carries a context, under :attr:`context` when bare.

        :param name: A rubric key.
        :ptype name: str
        :return: The namespaced name: under ``answer.`` when bare.
        :rtype: str
        """
        return name if "." in name else f"{DIMENSION_CONTEXT}.{name}"

    def authored_cases(
        self, *, intent: str, evidence: Sequence[JudgeEvidence] | None = None
    ) -> list[AuthoredJudgeCase]:
        """The :attr:`examples` as authored judge cases: each answer as the judge reads it, its verdict its label.

        :param intent: The scenario the judge reads beside each one: the run's template intent, so an example is asked
            exactly as the run's answers are.
        :ptype intent: str
        :param evidence: What the judge reads of each example, in order, as a run's cell renders it (its case less the
            keys a run withholds, a classifier's label left out); ``None`` renders each example's case as written.
        :ptype evidence: Sequence[JudgeEvidence] | None
        :return: One case per example, in order.
        :rtype: list[AuthoredJudgeCase]
        :raises ValueError: An example names no dimension of the rubric, gives a verdict its dimension's scale cannot
            hold, or has a case :attr:`case_material` cannot render; or ``evidence`` is not one per example.
        """
        if evidence is not None and len(evidence) != len(self.examples):
            raise ValueError(f"{len(evidence)} evidence for {len(self.examples)} example(s); one per example")
        dims = {dim.name: dim for dim in self._dims}
        authored: list[AuthoredJudgeCase] = []
        for index, example in enumerate(self.examples):
            criterion = dims.get(self.dim_name(example.dimension))
            if criterion is None:
                raise ValueError(
                    f"judge example {index} is graded on {example.dimension!r}, which is no dimension of the rubric "
                    f"({', '.join(sorted(dims))})"
                )
            read = judge_evidence(self, example.case, example.answer) if evidence is None else evidence[index]
            try:
                authored.append(
                    AuthoredJudgeCase(
                        criterion=criterion,
                        output=read.artifact,
                        case_material=read.case_material,
                        intent=intent,
                        expected=example.expected,
                        author=example.author,
                        note=example.note,
                    )
                )
            except ValidationError as refused:
                reasons = "; ".join(str(error["msg"]).removeprefix("Value error, ") for error in refused.errors())
                raise ValueError(f"judge example {index}: {reasons}") from refused
        return authored

    @property
    def dims(self) -> tuple[RubricDim, ...]:
        """The rubric as the template stores it, in the order given, each dimension's judge declared."""
        return self._dims

    def clients(self) -> CompletionClients:
        """The host client factory a run judged by this judge is built with: this judge's client, lent.

        :return: A factory answering the judge role on :attr:`model` with :attr:`client`, and refusing every
            other role and model, since the client calls one model and no other role is in a
            judged eval.
        :rtype: CompletionClients
        """

        def lend(role: CompletionRole, model: str | None, *, temperature: float | None = None) -> BorrowedJudgeClient:
            if role != "judge":
                raise ValueError(f"a judged eval calls no model in the {role!r} role; its judge is its only model")
            if model is not None and model != self.model:
                raise ValueError(
                    f"a judge config asked for {model!r}, and this judge's client calls {self.model!r}; "
                    "a Judge scores every dimension on its one client"
                )
            return BorrowedJudgeClient(self.client, model_name=self.model, temperature=temperature)

        return lend


class BorrowedJudgeClient:
    """A caller's completion client, lent to one run: every call forwarded, the release left to its owner.

    Satisfies :class:`~threetears.evals.schema.BoundCompletionClient`. Its price ceiling is unknown —
    a run's judge is bounded by its spend as it arrives, never priced before a call is made. Bound to the
    temperature the engine asked for, as every client the host factory builds is, and passes it on to the
    owner's client when that client's ``generate`` takes a ``temperature`` keyword (:attr:`passes_temperature`);
    what was sent is what the owner's completion reports, which the judge records.
    """

    def __init__(self, client: CompletionClient, *, model_name: str, temperature: float | None = None) -> None:
        """Lend ``client`` under the model it calls, at the temperature the engine asked for.

        :param client: The owner's client.
        :ptype client: CompletionClient
        :param model_name: The model it calls.
        :ptype model_name: str
        :param temperature: The sampling temperature the engine asked this client for; ``None`` for the
            provider's default.
        :ptype temperature: float | None
        """
        self._client = client
        self.model_name = model_name
        self.temperature = temperature
        self.passes_temperature = _takes_temperature(client)

    async def generate(
        self, *, system: str, user: str, response_format: dict[str, Any] | None = None
    ) -> CompletionResult:
        """Forward one call to the owner's client.

        :param system: The judge's system prompt.
        :ptype system: str
        :param user: The judge's user prompt.
        :ptype user: str
        :param response_format: The provider directive, as the engine sends it.
        :ptype response_format: dict[str, Any] | None
        :return: The owner's client's completion.
        :rtype: CompletionResult
        """
        if self.passes_temperature:
            generate: Any = self._client.generate
            completion: CompletionResult = await generate(
                system=system, user=user, response_format=response_format, temperature=self.temperature
            )
            return completion
        return await self._client.generate(system=system, user=user, response_format=response_format)

    def price_ceiling(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> float | None:
        """Unknown: the owner's client states no rate.

        :param system: Unread.
        :ptype system: str
        :param user: Unread.
        :ptype user: str
        :param response_format: Unread.
        :ptype response_format: dict[str, Any] | None
        :return: ``None``.
        :rtype: float | None
        """
        return None

    async def aclose(self) -> None:
        """Nothing: the client is its owner's to close."""

    async def __aenter__(self) -> Self:
        """Enter a scope whose exit releases nothing."""
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        """Leave the client open for its owner."""


def judge_client(model: str | CompletionClient) -> tuple[CompletionClient, str]:
    """The client a judge calls and the model the run records it as, from what ``Judge(model=...)`` was handed.

    :param model: A :class:`~threetears.evals.providers.Model`; any completion client that names its model
        (``model_name``); or a model id, which builds a :class:`~threetears.evals.providers.Model` that calls it.
    :ptype model: str | CompletionClient
    :return: The client, and its model's name.
    :rtype: tuple[CompletionClient, str]
    :raises ValueError: A client that names no model, or a model id the completion client refuses (no key for its
        provider, naming the variable).
    """
    if isinstance(model, str):
        from threetears.evals.providers import Model  # the completion client, loaded only when a judge names an id

        built = Model(model)
        return built, built.model_name
    name = getattr(model, "model_name", None)
    if not isinstance(name, str) or not name.strip():
        raise ValueError(
            f"Judge(model=...) takes a model id, a Model, or a client that names its model (model_name); "
            f"{type(model).__name__} names none, and the run records which model judged"
        )
    return model, name


def _takes_temperature(client: CompletionClient) -> bool:
    """Whether ``client.generate`` accepts a ``temperature`` keyword, by name or through ``**kwargs``."""
    try:
        parameters = inspect.signature(client.generate).parameters.values()
    except TypeError, ValueError:
        return False
    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        or (parameter.name == "temperature" and parameter.kind is not inspect.Parameter.POSITIONAL_ONLY)
        for parameter in parameters
    )


def judge_evidence(judge: Judge, case: Mapping[str, Any], answer: Any, *, expected: str | None = None) -> JudgeEvidence:
    """What the judge reads of one answer: the answer as text, against its case's material.

    The evidence is stored on the cell's trace exactly as sent, so a re-judge sends the same material.

    :param judge: The judge, whose ``case_material`` renders the case.
    :ptype judge: Judge
    :param case: The case the answer was given for.
    :ptype case: Mapping[str, Any]
    :param answer: The candidate's answer: a string as written, anything else as JSON (its ``repr`` when
        JSON cannot hold it).
    :ptype answer: Any
    :param expected: The case's expected label on a classifier run, ``None`` otherwise. With no
        ``case_material``, every top-level field holding it is left out of the case shown (the answer key).
    :ptype expected: str | None
    :return: The evidence.
    :rtype: JudgeEvidence
    :raises ValueError: ``case_material`` raised, or rendered no text.
    """
    if judge.case_material is None:
        shown = {key: value for key, value in case.items() if expected is None or value != expected}
        material = json.dumps(shown, indent=2, sort_keys=True, default=repr)
    else:
        try:
            material = judge.case_material(case)
        # prawduct:ok-broad-except — case_material is the caller's code: what it raises is named, the case with it
        except Exception as raised:
            raise ValueError(f"the judge's case_material raised {type(raised).__name__}: {raised}") from raised
        if not isinstance(material, str) or not material.strip():
            raise ValueError(
                f"the judge's case_material gave {material!r}; the material an answer is judged against is text"
            )
    return JudgeEvidence(case_material=material, artifact=_answer_text(answer))


def _answer_text(answer: Any) -> str:
    """The answer as the judge reads it."""
    if isinstance(answer, str):
        text = answer
    else:
        try:
            text = json.dumps(answer, indent=2, sort_keys=True)
        except TypeError, ValueError:
            text = repr(answer)
    return text if text.strip() else EMPTY_ANSWER


__all__ = [
    "EMPTY_ANSWER",
    "DIMENSION_CONTEXT",
    "EXAMPLE_AUTHOR",
    "BorrowedJudgeClient",
    "CaseMaterial",
    "Judge",
    "Example",
    "ExampleGrade",
    "JudgeExamples",
    "judge_evidence",
]
