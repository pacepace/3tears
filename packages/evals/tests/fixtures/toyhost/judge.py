"""The toy host's judged variant — a rubric, a scripted judge, and the engine's own judge service.

The standard toy template declares no rubric: its grader is a comparison rule, and the profile
says so by seating none of the core's model-judge apparatus on the extractor's kind contract. A second consumer grading with a
hand-written rubric today reaches for a model grader early, so this is the other shape: the same
extractor, the same invoices, and one rubric dimension scored by the engine's
:class:`~threetears.evals.run.judge_service.JudgeService` against the evidence the kind renders.

The judge is scripted, not a model, so the path runs with no network and a deterministic answer.
It reads the two sections the service renders for a document candidate — the case material and
the output under review — and scores how many of the invoice's fields the output reproduces. That
makes its score a function of the evidence it was actually handed, so a kind that stops rendering
evidence, or a service that stops forwarding it, changes the score rather than passing silently.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Self

from threetears.evals.schema import DEFAULT_JUDGE_TEMPERATURE, CompletionClient, EvalTemplate, RubricDim, StopReason
from threetears.evals.kernel import withhold_failure_detail
from threetears.evals.run import JudgeService
from packages.evals.tests.fixtures.toyhost.kind import INVOICE_FIELDS
from packages.evals.tests.fixtures.toyhost.run import toyhost_template

#: The one rubric dimension the judged variant scores.
FAITHFULNESS_DIM = "extraction.field_faithfulness"

#: The model id the scripted judge reports. Not a real model: the path proves the engine records
#: whichever id the host's client reports, not that any particular model was called.
TOY_JUDGE_MODEL = "toy/scripted-judge"

#: What the scripted judge charges per call, so a judged cell's spend is visibly non-zero.
TOY_JUDGE_COST_USD = 0.0004

#: Where that price comes from: the toy host's own script, reported the way a real client reports
#: its provider. The engine stores this and supplies none of its own.
TOY_JUDGE_PRICE_SOURCE = "toyhost_script"

#: The heading the judge service puts above the candidate's artifact in a document prompt, and the
#: one above the case material. The scripted judge splits on them to score what it was shown.
_OUTPUT_HEADING = "# Output under review\n"
_MATERIAL_HEADING = "# Case material (the evidence the output must be judged against)\n"


def toyhost_rubric() -> list[RubricDim]:
    """The judged variant's rubric: one ordinal dimension over the extraction's faithfulness.

    Returns:
        The rubric.
    """
    return [
        RubricDim(
            name=FAITHFULNESS_DIM,
            description="Every field the extraction reports matches what the invoice says, character for character.",
            scale="ordinal",
            scoring_guide={
                "1": "no field matches the invoice",
                "5": "every declared field matches the invoice exactly",
            },
        )
    ]


def toyhost_judged_template() -> EvalTemplate:
    """The standard toy template with a rubric, under its own id.

    Returns:
        The judged template.
    """
    template = toyhost_template()
    return template.model_copy(update={"id": f"{template.id}:judged", "rubric": toyhost_rubric()})


@dataclass(frozen=True)
class ToyJudgeCompletion:
    """One scripted judge reply — a frozen value satisfying :class:`~threetears.evals.schema.CompletionResult`.

    ``stop_reason`` is the engine's word, never a provider's: a real client maps its provider's
    finish reason (OpenAI's ``stop`` / ``length``) onto :data:`~threetears.evals.schema.StopReason`.
    """

    content: str
    #: ``None`` is a provider that reported no count — unknown, never zero.
    input_tokens: int | None
    output_tokens: int | None
    cost_usd: float | None
    model: str
    price_source: str | None
    reasoning_tokens: int | None = None
    stop_reason: StopReason = "end_turn"
    served_model: str | None = None
    #: What the request was sent at; ``None`` is a request sent with none, the model's own default applying.
    temperature: float | None = DEFAULT_JUDGE_TEMPERATURE


class ScriptedJudgeClient:
    """A judge client satisfying :class:`~threetears.evals.schema.CompletionClient`, scoring from its prompt."""

    def __init__(
        self, *, reports_token_counts: bool = True, temperature: float | None = DEFAULT_JUDGE_TEMPERATURE
    ) -> None:
        """Start with no calls recorded.

        Args:
            reports_token_counts: ``False`` is a provider that reports its price but omits the token
                counts, which the engine must carry as unknown rather than as zero.
            temperature: What every request is sent at, as each completion reports it; ``None`` is a model
                that refuses a temperature and is sent none.
        """
        #: Every ``(system, user)`` prompt pair the client was sent, in call order.
        self.calls: list[tuple[str, str]] = []
        self._reports_token_counts = reports_token_counts
        self._temperature = temperature

    async def generate(
        self, *, system: str, user: str, response_format: dict[str, str] | None = None
    ) -> ToyJudgeCompletion:
        """Score the output under review against the case material it was judged beside.

        Args:
            system: The judge's system prompt.
            user: The judge's user prompt, carrying the case material and the output.
            response_format: The requested wire format; the reply is always one JSON object.

        Returns:
            A reply scoring the rubric dimension 1–5 by the share of invoice fields reproduced, or
            answering ``cannot tell`` when the prompt carries no output to score.
        """
        self.calls.append((system, user))
        if _OUTPUT_HEADING not in user or _MATERIAL_HEADING not in user:
            reply = {
                "reasoning": "the prompt carries no document to score",
                "criteria_scores": {FAITHFULNESS_DIM: "cannot tell"},
            }
        else:
            material, output = user.split(_MATERIAL_HEADING, 1)[1].split(_OUTPUT_HEADING, 1)
            fields = json.loads(output)
            expected = dict(line.split(": ", 1) for line in material.strip().splitlines()[1:])
            matched = sum(1 for name in INVOICE_FIELDS if fields.get(name) == expected.get(name))
            reply = {
                "reasoning": f"{matched} of {len(INVOICE_FIELDS)} fields match the invoice",
                "criteria_scores": {FAITHFULNESS_DIM: 1 + round(4 * matched / len(INVOICE_FIELDS))},
            }
        return ToyJudgeCompletion(
            content=json.dumps(reply),
            input_tokens=len(user) // 4 if self._reports_token_counts else None,
            output_tokens=40 if self._reports_token_counts else None,
            cost_usd=TOY_JUDGE_COST_USD,
            price_source=TOY_JUDGE_PRICE_SOURCE,
            model=TOY_JUDGE_MODEL,
            temperature=self._temperature,
        )

    async def aclose(self) -> None:
        """Nothing to release: the scripted client holds no connection."""

    async def __aenter__(self) -> Self:
        """Enter a scope that releases the client on exit."""
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        """Release on the way out."""
        await self.aclose()


def toyhost_judge_service(client: ScriptedJudgeClient) -> JudgeService:
    """The engine's judge service, bound to the scripted client for every model it is asked for.

    Args:
        client: The scripted judge.

    Returns:
        The service.
    """

    def judge_client(_model: str | None, _temperature: float | None) -> CompletionClient:
        # Typed against the port, so a type checker proves the scripted client satisfies it.
        return client

    return JudgeService(client_factory=judge_client, failure_describer=withhold_failure_detail)


__all__ = [
    "FAITHFULNESS_DIM",
    "TOY_JUDGE_COST_USD",
    "TOY_JUDGE_MODEL",
    "TOY_JUDGE_PRICE_SOURCE",
    "ScriptedJudgeClient",
    "ToyJudgeCompletion",
    "toyhost_judge_service",
    "toyhost_judged_template",
    "toyhost_rubric",
]
