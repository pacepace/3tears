"""``Judge``: a model grading ``run_eval``'s answers against a rubric, through the engine's own judge.

A scorer is code, and some answers have no code that grades them: whether a reply is helpful,
whether it says only what its source supports. Handed a :class:`Judge`, :func:`~threetears.evals.quick.run_eval`
asks a model instead, one rubric dimension per call, and reports each dimension's scores beside the
scorers' measures. Nothing here judges: it declares what the engine's judge reads and who calls the model.

- **The rubric** is a mapping of dimension name to what the dimension measures, or the engine's own
  :class:`~threetears.evals.contracts.RubricDim` values. A bare name is namespaced under
  :attr:`Judge.context` (``helpful`` becomes ``answer.helpful``), because a judge config binds to a
  dimension by name across every template and a bare one would bind to every product's ``helpful``.
  The rubric is the template's, so it is part of what the case set's id is a digest of.
- **The judge** is :class:`~threetears.evals.run.JudgeService`, built by
  :func:`~threetears.evals.run.build_judge_service` as for any judged run, over a client factory that
  hands it :attr:`Judge.client` for the judge role and nothing else. It scores the answer as a document
  (:attr:`~threetears.evals.contracts.JudgedArtifact.DOCUMENT`): the rubric dimensions alone, each read
  against the case material, with no conversation axes.
- **The evidence** is the kind's to render (:func:`judge_evidence`): the answer as text, judged against
  :attr:`Judge.case_material`'s rendering of its case, or the case as JSON. It is stored on each cell's
  trace, so a re-judge sends exactly what the first judge read.
- **The spend** is the client's to report. Each call's tokens and dollars land on the result's ``judge``
  usage row, priced by the client's own ``cost_usd`` and qualified by its ``price_source``; a client that
  prices nothing leaves the judge's spend unknown, never zero.

**The caller owns the client.** The engine releases every client it is handed when a run ends; the
judge's is lent to it through :class:`BorrowedJudgeClient`, whose release does nothing, so one client
serves every call that names it and is closed by whoever opened it.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Self

from threetears.evals.contracts import (
    CompletionClient,
    CompletionResult,
    JudgeEvidence,
    RubricDim,
    RubricScale,
)
from threetears.evals.contracts.host import CompletionClients, CompletionRole

#: Renders the material one case's answer is judged against: takes the case, returns non-blank text.
CaseMaterial = Callable[[Mapping[str, Any]], str]

#: What the judge reads in place of an answer that renders as blank text, which no judge could quote.
EMPTY_ANSWER = "(the candidate's answer was empty)"


@dataclass(frozen=True)
class Judge:
    """A model that grades each answer on a rubric, one call per dimension.

    Attributes:
        client: The completion client the judge calls, already bound to ``model``. Lent to the engine
            for the run and never closed by it. The judge asks for a sampling temperature on every call
            (0, unless a judge config states another): when the client's ``generate`` takes a
            ``temperature`` keyword it is passed on every call, and the client reports what it actually sent
            on each completion's ``temperature`` (``None`` when its model refuses one), which every score
            records as part of who judged. A client whose ``generate`` takes none is called without it, and
            its scores record what its completions report: the model's default when they say none was sent.
        model: The model ``client`` calls, as the run records it: the run's judge pin, and the arm's
            judge in every comparison of two runs.
        rubric: The dimensions, each a name and what it measures, or :class:`RubricDim` values for a
            scoring guide or a pass/fail scale.
        scale: How a dimension given as a name and a description is answered: ``"ordinal"`` (1 to 5)
            or ``"pass_fail"``. A :class:`RubricDim` states its own.
        context: The namespace a bare dimension name is given: what the dimensions are scored against.
        case_material: Renders what an answer is judged against from its case — a question and the
            source it must be answered from, say. ``None`` shows the judge the case as JSON, less the answer
            key: on a classifier run (``run_eval(expected=...)``), every top-level field whose value is the
            case's expected label is left out, so the judge never grades against the label it is meant to
            judge without. To grade against a reference answer, render it here: what this returns is sent
            as it is, the expected label included if it includes it.
    """

    client: CompletionClient
    model: str
    rubric: Mapping[str, str] | Sequence[RubricDim]
    scale: RubricScale = "ordinal"
    context: str = "answer"
    case_material: CaseMaterial | None = None
    _dims: tuple[RubricDim, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Build the rubric's dimensions once, refusing a judge no run could record.

        Raises:
            ValueError: A blank model, an empty rubric, a dimension the engine refuses (a name with more
                than one ``.``, a blank description), or two dimensions under one name.
        """
        if not self.model.strip():
            raise ValueError("a judge names the model its client calls; the run records it as the judge")
        dims = tuple(self._build_dims())
        if not dims:
            raise ValueError("a judge needs at least one rubric dimension; a rubric with none scores nothing")
        names = [dim.name for dim in dims]
        if repeated := sorted({name for name in names if names.count(name) > 1}):
            raise ValueError(f"rubric dimensions named {', '.join(repeated)} more than once; each name is one score")
        object.__setattr__(self, "_dims", dims)

    def _build_dims(self) -> list[RubricDim]:
        if isinstance(self.rubric, Mapping):
            return [
                RubricDim(name=self.dim_name(name), description=description, scale=self.scale)
                for name, description in self.rubric.items()
            ]
        return list(self.rubric)

    def dim_name(self, name: str) -> str:
        """``name`` as a stored dimension name: as given when it carries a context, under :attr:`context` when bare.

        Args:
            name: A rubric key.

        Returns:
            The namespaced name.
        """
        return name if "." in name else f"{self.context}.{name}"

    @property
    def dims(self) -> tuple[RubricDim, ...]:
        """The rubric as the template stores it, in the order given."""
        return self._dims

    def clients(self) -> CompletionClients:
        """The host client factory a run judged by this judge is built with: this judge's client, lent.

        Returns:
            A factory answering the judge role on :attr:`model` with :attr:`client`, and refusing every
            other role and model, since the client calls one model and no other role is in a
            ``run_eval`` run.
        """

        def lend(role: CompletionRole, model: str | None, *, temperature: float | None = None) -> BorrowedJudgeClient:
            if role != "judge":
                raise ValueError(f"a run_eval run calls no model in the {role!r} role; its judge is its only model")
            if model is not None and model != self.model:
                raise ValueError(
                    f"a judge config asked for {model!r}, and this judge's client calls {self.model!r}; "
                    "a run_eval judge scores every dimension on its one client"
                )
            return BorrowedJudgeClient(self.client, model_name=self.model, temperature=temperature)

        return lend


class BorrowedJudgeClient:
    """A caller's completion client, lent to one run: every call forwarded, the release left to its owner.

    Satisfies :class:`~threetears.evals.contracts.BoundCompletionClient`. Its price ceiling is unknown —
    a run's judge is bounded by its spend as it arrives, never priced before a call is made. Bound to the
    temperature the engine asked for, as every client the host factory builds is, and passes it on to the
    owner's client when that client's ``generate`` takes a ``temperature`` keyword (:attr:`passes_temperature`);
    what was sent is what the owner's completion reports, which the judge records.
    """

    def __init__(self, client: CompletionClient, *, model_name: str, temperature: float | None = None) -> None:
        """Lend ``client`` under the model it calls, at the temperature the engine asked for.

        Args:
            client: The owner's client.
            model_name: The model it calls.
            temperature: The sampling temperature the engine asked this client for; ``None`` for the
                provider's default.
        """
        self._client = client
        self.model_name = model_name
        self.temperature = temperature
        self.passes_temperature = _takes_temperature(client)

    async def generate(
        self, *, system: str, user: str, response_format: dict[str, Any] | None = None
    ) -> CompletionResult:
        """Forward one call to the owner's client.

        Args:
            system: The judge's system prompt.
            user: The judge's user prompt.
            response_format: The provider directive, as the engine sends it.

        Returns:
            The owner's client's completion.
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

        Args:
            system: Unread.
            user: Unread.
            response_format: Unread.

        Returns:
            ``None``.
        """
        return None

    async def aclose(self) -> None:
        """Nothing: the client is its owner's to close."""

    async def __aenter__(self) -> Self:
        """Enter a scope whose exit releases nothing."""
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        """Leave the client open for its owner."""


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

    Args:
        judge: The judge, whose ``case_material`` renders the case.
        case: The case the answer was given for.
        answer: The candidate's answer: a string as written, anything else as JSON (its ``repr`` when
            JSON cannot hold it).
        expected: The case's expected label on a classifier run, ``None`` otherwise. With no
            ``case_material``, every top-level field holding it is left out of the case shown (the answer key).

    Returns:
        The evidence.

    Raises:
        ValueError: ``case_material`` raised, or rendered no text.
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


__all__ = ["EMPTY_ANSWER", "BorrowedJudgeClient", "CaseMaterial", "Judge", "judge_evidence"]
