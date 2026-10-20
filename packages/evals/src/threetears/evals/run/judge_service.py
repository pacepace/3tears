"""Stateless single-dimension judge service.

One call → one dimension → one :class:`RubricScore`. The judge never holds
scoring state between calls; the only mutable state is a client cache (pure
construction-memoization). **Calls may run concurrently on one cached client** —
the runner gathers a result's ``2 + N`` calls under ``eval.judge_concurrency`` —
which is sound only because the port requires each ``generate()`` to build its
own request rather than share instance state
(:class:`~threetears.evals.schema.completion.CompletionClient`); the runner was once serial
for exactly the want of that guarantee.

Three scoring entry points:

- :meth:`JudgeService.score_transcript` — *decision quality*: were the
  candidate's reasoning + tool-param choices well-chosen **given the context it
  had**? Stable when externals drift; does **not** see the goal-state outcomes
  (axis separation — that's the outcome judge's job).
- :meth:`JudgeService.score_outcome` — *intent satisfaction*: did the final
  state / response satisfy what the user wanted? Sees the goal-state outcomes
  and is told to judge the result, not the path.
- :meth:`JudgeService.score_dimension` — one template :class:`RubricDim`
  (tone, groundedness, …), built from its description + scoring guide.

**What every call reads is the kind's, not the engine's.** The candidate's kind renders the
judge's evidence — who the candidate is, what its output is judged against, and the output itself
(:class:`~threetears.evals.schema.models.JudgeEvidence`) — and this service places those three
strings in the prompt without reading them. The engine renders no transcript and no subject of its
own: only the kind knows how its turns and tool calls read as text, and which facts in play a judge
should see that the candidate's interlocutors did not. The kind's declaration
(:class:`~threetears.evals.schema.models.JudgedArtifact`) picks the axes and the wording.

Each call resolves a versioned :class:`JudgeConfig` (pre-resolved per dim at run
start, keyed by ``rubric_dim_id``); when one exists its ``prompt_template`` is
used as the judging instructions and its ``model`` / ``temperature`` select the
client. With no config, a built-in default prompt + the default judge client
are used. **Which of the two a rubric dim gets is its author's declaration, not
this service's finding** (``RubricDim.judge``): the launch refuses a dim declared
``"configured"`` that resolves no config, or ``"builtin"`` beside one
(:func:`~threetears.evals.run.authoring.refuse_undeclared_judges`), so the configs
this service is built with are exactly the declared ones. The built-in prompt is sent
at :data:`~threetears.evals.schema.models.DEFAULT_JUDGE_TEMPERATURE` —
the same temperature a config defaults to, so configuring a dim's prompt never
changes how it is sampled (#633). Every score records the temperature its call
was actually sent at. **The built-in prompt has an identity, as a config does**: each
call records ``builtin:<sha12>`` (:func:`builtin_judge_config_ids`), a fingerprint of the
engine's wording, where a configured call records its config's id — so a change to the
built-in wording is a new judge to every reader that keys a judge by its config.

The fixed JSON-format instruction is appended to **both** so the response stays
parseable regardless of an operator's prompt wording. That suffix is therefore part
of a configured judge's prompt too, so a configured call records ``<config id>@<sha12>``
(:func:`judge_config_ids_for`): the config's id and a fingerprint of the suffix's wording.
A release that rewords the suffix makes every configured judge a new judge, as a new
config version would. Rules meant for the built-in judge still go in the built-in
instructions, never in the suffix, which would change every operator's judge with them.

The multi-dim composite prompt builder (``judge_prompts.build_rubric_prompt``)
is retired; the per-axis / per-dim builders here replace it. The underlying LLM
call + parse + cost-log still flow through
:func:`threetears.evals.run.judge.run_judge_llm` (public, for a host's own scorers).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from typing import Any, NamedTuple, Self

from threetears.evals.schema.models import (
    DEFAULT_JUDGE_TEMPERATURE,
    OUTCOME_DIM_ID,
    SCALE_LEVELS,
    TRANSCRIPT_DIM_ID,
    GoalStateOutcome,
    JudgeConfig,
    JudgeCutShort,
    JudgedArtifact,
    JudgeEvidence,
    LabelKey,
    RoleUsage,
    RubricDim,
    RubricAxis,
    RubricScale,
    RubricScore,
    KnownUsageRole,
    builtin_judge_config_id,
    configured_judge_config_id,
    label_key_of,
)
from threetears.evals.kernel.host.eval_host import CompletionClients
from threetears.evals.schema.completion import BoundCompletionClient, ProviderFailureDescriber
from threetears.evals.kernel.capped_completion import call_ceiling
from threetears.evals.kernel.usage_capture import CallUsage, RoleUsageLedger, blended_cost
from threetears.evals.run.judge import (
    CANNOT_TELL,
    JUDGE_CALL_ATTEMPTS,
    ORDINAL_ANSWERS,
    judge_reply_format,
    run_judge_llm,
)
from threetears.evals.schema.logs import contain, get_logger

log = get_logger(__name__)

#: Builds a judge LLM client for a ``(model, temperature)`` pair. ``model`` is
#: ``None`` when this dim states no model of its own — what the factory
#: substitutes is the caller's business, and the run-scoped one supplies the
#: run's judge pin. ``temperature`` is what the call is requested at: the dim's
#: config's, else :data:`~threetears.evals.schema.models.DEFAULT_JUDGE_TEMPERATURE`
#: — the service never asks for the provider default, and a client whose model
#: refuses a temperature sends none and reports that on its completion. The
#: service caches the result so identical configs reuse one client.
JudgeClientFactory = Callable[[str | None, float | None], Any]


def judge_clients_for_run(clients: CompletionClients, judge_model: str | None) -> JudgeClientFactory:
    """The judge client factory for a run pinned to ``judge_model``, over the host's client factory.

    The run-scoped half of the judge's model cascade: a dim whose configuration names a model is
    scored by that model, and every other dim by the run's pin. A ``None`` pin asks the host for the
    judge role's default.

    :param clients: The host's completion-client factory.
    :ptype clients: CompletionClients
    :param judge_model: The run's judge pin, resolved, or ``None``.
    :ptype judge_model: str | None
    :return: The factory a :class:`JudgeService` is built with.
    :rtype: JudgeClientFactory
    """

    def build(model: str | None, temperature: float | None) -> BoundCompletionClient:
        return clients("judge", model if model is not None else judge_model, temperature=temperature)

    return build


@dataclass(frozen=True)
class JudgeContext:
    """Everything a single-dim judge call needs about the result under review.

    One context is built per ``EvalResult`` and reused across the transcript / outcome / per-dim
    calls for that result. Every field is something the cell recorded — the case, its goal-state
    outcomes, the kind's declaration and the evidence the kind rendered — so a re-judge built
    from the stored trace asks exactly the question the first judge was asked.
    """

    case_id: str
    intent: str
    variation: dict[str, str]
    goal_outcomes: list[GoalStateOutcome]
    #: The kind's declaration: which axes are scored and how the evidence is worded. Never
    #: :attr:`~JudgedArtifact.UNJUDGED` — there is nothing to judge for such a kind.
    judged_artifact: JudgedArtifact
    #: What the judge reads, as the kind rendered it.
    judge_evidence: JudgeEvidence

    def __post_init__(self) -> None:
        """Refuse a context for a kind that declared nothing a judge reads.

        :raises ValueError: ``judged_artifact`` is :attr:`~JudgedArtifact.UNJUDGED`.
        """
        if self.judged_artifact is JudgedArtifact.UNJUDGED:
            raise ValueError(
                "an unjudged kind's cell has nothing for a judge to read; no judge context is built for it"
            )


@dataclass
class JudgeOutcome:
    """Result of one single-dim judge call.

    ``score`` is ``None`` exactly when ``error`` or ``cannot_tell`` is set: the call
    failed (LLM call raised, parse failed, an error envelope), or the judge answered
    that the evidence does not let it score the dim. ``config_id`` names what asked:
    the versioned :class:`JudgeConfig` used with the reply format it was sent
    (:func:`judge_config_ids_for`), or the built-in prompt's identity
    (:func:`builtin_judge_config_ids`) when no config supplied the instructions. It is
    ``None`` only for an outcome a caller made up for a call that never reached the
    service (one that raised, or a cost cap refused).
    """

    score: RubricScore | None
    config_id: str | None = None
    error: str | None = None
    #: Tokens + cost this call actually observed — the source for both the R3 ``judge``
    #: usage row and the run's judge spend. It replaced an earlier ``cost_usd: float``
    #: because that field was only populated on the scoring path, so a dim that burned
    #: tokens and then failed to parse reported ``0.0`` and its real spend vanished from
    #: the run total. ``None`` only when the client returned nothing at all.
    usage: CallUsage | None = None
    #: The judge's reason, when it answered it cannot tell: what the evidence lacked. Not a
    #: failure — the judge worked and said the transcript does not decide the dim — so it is
    #: never folded into the errors that exclude a result.
    cannot_tell: str | None = None
    #: The call was refused for the calling account (out of credit, the key refused), as the
    #: host's describer read what the client raised. Still an ``error`` — the dim is unscored and
    #: the result excluded — and also what stops the run, since every later call is behind the key.
    account_refused: bool = False
    #: The reply the provider cut short, as facts: its stop reason, what the call was sent with and its token
    #: counts. Set only beside ``error`` (the call's reply was cut off and not re-bought), and folded onto the
    #: result's ``judge_cut_short`` so the exclusion it causes is counted as ``truncated``, not as a fault.
    cut_short: JudgeCutShort | None = None


#: The one role a judge call spends under — what a fold of judge calls sums its cost over.
_JUDGE_ROLE: tuple[KnownUsageRole, ...] = ("judge",)


class JudgeFold(NamedTuple):
    """What a set of judge calls left beside their scores: configs, errors and spend."""

    #: dim -> the versioned judge config that scored it, for each call that named one.
    config_ids: dict[str, str]
    #: ``(dim, error)`` for each call that produced no score, in call order.
    errors: list[tuple[str, str]]
    #: The judge role's usage rows, one per model.
    usage: list[RoleUsage]
    #: Their total cost, or ``None`` when any call went unpriced — unknown, never zero.
    cost_usd: float | None
    #: dim -> the judge's reason, for each call answering it could not tell.
    cannot_tell: dict[str, str]
    #: dim -> the cut-off reply, for each call whose reply the provider cut short (each also in ``errors``).
    cut_short: dict[str, JudgeCutShort]


def fold_judge_outcomes(outcomes: list[tuple[str, JudgeOutcome]]) -> JudgeFold:
    """Fold judge calls' outcomes into their configs, errors and spend.

    Shared by the cell's judge phase and a later re-judge, so both read an outcome the same
    way. **A call that returned no score is an error even when it names none**: the judge
    service sets one on every failure, so the unnamed case is a contract breach, and letting
    it through silently would leave a dim unscored while nothing says it failed.

    Judge cost comes off the usage ledger rather than a per-call cost field, because a dim
    that burned tokens and then failed to parse is spend the run still paid. It is ``None``
    when any call went unpriced (:func:`~threetears.evals.kernel.usage_capture.blended_cost`).

    :param outcomes: ``(dim_id, outcome)`` for every call made, in call order.
    :ptype outcomes: list[tuple[str, JudgeOutcome]]
    :return: The fold.
    :rtype: JudgeFold
    """
    config_ids: dict[str, str] = {}
    errors: list[tuple[str, str]] = []
    cannot_tell: dict[str, str] = {}
    cut_short: dict[str, JudgeCutShort] = {}
    # Per-dim judge configs may each pin their own model, so the ledger splits rows by
    # model rather than blending the dims into one — see threetears.evals.kernel.usage_capture.
    ledger = RoleUsageLedger(role="judge")
    for dim_id, outcome in outcomes:
        if outcome.usage is not None:
            ledger.add_llm_result(outcome.usage)
        if outcome.config_id:
            config_ids[dim_id] = outcome.config_id
        if outcome.cannot_tell is not None:
            cannot_tell[dim_id] = outcome.cannot_tell
        elif outcome.score is None:
            errors.append((dim_id, outcome.error or "the judge returned neither a score nor an error"))
            if outcome.cut_short is not None:
                cut_short[dim_id] = outcome.cut_short
    rows = ledger.rows()
    return JudgeFold(
        config_ids=config_ids,
        errors=errors,
        usage=rows,
        cost_usd=blended_cost(rows, _JUDGE_ROLE),
        cannot_tell=cannot_tell,
        cut_short=cut_short,
    )


# ---------------------------------------------------------------------------
# Built-in default judging instructions (used when no JudgeConfig is active)
# ---------------------------------------------------------------------------
#
# **This wording is an instrument, and it has an identity.** Every built-in prompt is recorded under
# a ``builtin:<sha12>`` id where a config's id would stand (:func:`builtin_judge_config_ids`), a
# fingerprint of the fixed text below as the dimension's family renders it. Change a word here and
# every dim judged with it is judged by a new judge: stored judge profiles, agreement figures and
# drift readings measured on the old wording stop applying to it, as they would for a new config.

#: Where a judge's own recall stands against what the candidate looked up (#1895). Appended to
#: every built-in prompt, never to the shared JSON-format suffix: that suffix is sent after an operator's
#: configured prompt too, and a change there would change a configured judge while its config id stayed
#: the same.
_SOURCED_FACTS_RULE = (
    "Facts the evidence shows the candidate obtained from its tools or sources, and the case material, outrank "
    "your own recall. Never call a claim fabricated or unsupported only because you cannot verify it from memory: "
    "your knowledge stops at a cutoff, and a recent fact can be true and unknown to you. When the evidence neither "
    "shows where a claim came from nor lets you check it, say so rather than asserting fabrication, and give the "
    "cannot-tell answer described below if that leaves no honest score."
)

_TRANSCRIPT_INSTRUCTIONS = (
    "You are evaluating the DECISION QUALITY of a candidate's turn-by-turn "
    "behavior in a conversation. Score how well-chosen the candidate's reasoning, tool selection, and "
    "tool parameters were GIVEN THE INFORMATION AVAILABLE TO IT at each step.\n\n"
    "Judge the choices, not the outcomes: do NOT penalize the candidate for tool "
    "failures, empty search results, or world state outside its control — a sound "
    "decision with an unlucky result still scores well. You are NOT assessing whether "
    "the final result satisfied the user; a separate judge does that.\n\n" + _SOURCED_FACTS_RULE
)

_OUTCOME_INSTRUCTIONS = (
    "You are evaluating whether the candidate ACHIEVED THE USER'S INTENT. Score how "
    "well the final state and the candidate's responses satisfied what the user "
    "actually wanted.\n\n"
    "Judge the result, not the path: a clumsy route to a correct outcome still scores "
    "well; elegant reasoning that left the intent unmet scores poorly. The objective "
    "goal-state checks below are already computed — use them as evidence, do not "
    "re-derive them.\n\n" + _SOURCED_FACTS_RULE
)

#: How each scale is asked for: the scale sentence, and what ``criteria_scores`` maps the key to. An ordinal score is
#: asked for as a string, in the words of the strict reply format's enum (:data:`~threetears.evals.run.judge.ORDINAL_ANSWERS`),
#: so the prompt and the schema sent beside it ask for the same reply.
_SCALE_WORDING: dict[str, tuple[str, str]] = {
    "ordinal": (
        "Score on a scale of 1 (worst) to 5 (best).",
        "score, written as a string: "
        + ", ".join(f'"{answer}"' for answer in ORDINAL_ANSWERS[:-1])
        + f' or "{ORDINAL_ANSWERS[-1]}"',
    ),
    "pass_fail": ("Answer pass or fail.", 'answer, the string "pass" or the string "fail"'),
}

_JSON_FORMAT_TEMPLATE = (
    "\n\n{scale_sentence} Return a JSON object with "
    "exactly two keys:\n"
    '  - "reasoning": one short paragraph citing specific {evidence}.\n'
    '  - "criteria_scores": an object with the single key "{dim_id}" mapped to your '
    "{scale_answer}.\n"
    "If the {material} does not let you score this dimension at all, map the key to the string "
    '"{cannot_tell}" instead of a score, and say in "reasoning" what is missing. Use it only when '
    "no score would be honest, never to avoid a low one."
)


def _dimension_instructions(dim: RubricDim, *, document: bool = False) -> str:
    """Build the default judging instructions for a template rubric dimension.

    :param dim: The dimension being scored.
    :ptype dim: RubricDim
    :param document: Whether the candidate's output is a document judged against case material,
        rather than its side of a conversation.
    :ptype document: bool
    :return: The system instructions, before the JSON-format suffix.
    :rtype: str
    """
    return _dimension_wording(
        name=dim.name, description=dim.description, scale=dim.scale, scoring_guide=dim.scoring_guide, document=document
    )


def _dimension_wording(
    *, name: str, description: str, scale: RubricScale, scoring_guide: dict[str, str], document: bool
) -> str:
    """Lay out a built-in dimension prompt from the parts a rubric dimension fills.

    Taking the parts rather than a :class:`RubricDim` is what lets the built-in prompt's identity
    be rendered with a placeholder in each (:func:`_builtin_dimension_config_id`), so it fingerprints
    the engine's wording and never a rubric's.
    """
    subject = "a candidate's output" if document else "a candidate's side of a conversation"
    lines = [
        f"You are scoring {subject} on the subjective rubric dimension '{name}'.",
        description,
    ]
    if scoring_guide:
        lines.append("Scoring guide:")
        for k in SCALE_LEVELS[scale]:
            if k in scoring_guide:
                lines.append(f"  {k}: {scoring_guide[k]}")
    lines.append(
        "Be honest and specific — vague scoring is useless to operators reviewing the "
        "run. Do NOT re-judge objective goal-state outcomes; they are scored separately "
        "and shown only for context."
    )
    lines.append(_SOURCED_FACTS_RULE)
    return "\n".join(lines)


def _system_prompt(*, dim_id: str, instructions: str, scale: RubricScale, document: bool) -> str:
    """A call's system prompt: its instructions with the JSON-format contract appended.

    ``document`` says the candidate produced a document rather than a conversation: it changes
    what the JSON contract tells the judge to cite, never what is scored.
    """
    evidence = "passages of the output and of the case material" if document else "moments in the transcript"
    material = "output and case material" if document else "transcript"
    scale_sentence, scale_answer = _SCALE_WORDING[scale]
    return instructions + _JSON_FORMAT_TEMPLATE.format(
        dim_id=dim_id,
        evidence=evidence,
        material=material,
        cannot_tell=CANNOT_TELL,
        scale_sentence=scale_sentence,
        scale_answer=scale_answer,
    )


#: The built-in prompts of the two axes, as sent: an axis prompt has no part a rubric fills, so its
#: identity is a fingerprint of the whole system prompt.
_BUILTIN_TRANSCRIPT_CONFIG_ID = builtin_judge_config_id(
    _system_prompt(dim_id=TRANSCRIPT_DIM_ID, instructions=_TRANSCRIPT_INSTRUCTIONS, scale="ordinal", document=False)
)
_BUILTIN_OUTCOME_CONFIG_ID = builtin_judge_config_id(
    _system_prompt(dim_id=OUTCOME_DIM_ID, instructions=_OUTCOME_INSTRUCTIONS, scale="ordinal", document=False)
)

#: What stands in for each part of a built-in dimension prompt a rubric dimension fills, when its identity is rendered.
_PLACEHOLDER_DIM = "{dimension}"


@cache
def _builtin_dimension_config_id(scale: RubricScale, document: bool) -> str:
    """The identity of the built-in prompt a rubric dimension of ``scale`` is judged with, for a document or not.

    The prompt rendered with a placeholder for each part the dimension fills — its name, its description,
    every level of a scoring guide — so the fingerprint covers every word the engine contributes (a
    guide's heading included, which a dimension with no guide never shows) and none the rubric does: a
    dimension's own wording is its criterion, which a stored profile keys on separately, not its judge.
    """
    instructions = _dimension_wording(
        name=_PLACEHOLDER_DIM,
        description="{description}",
        scale=scale,
        scoring_guide={level: f"{{guide for {level}}}" for level in SCALE_LEVELS[scale]},
        document=document,
    )
    return builtin_judge_config_id(
        _system_prompt(dim_id=_PLACEHOLDER_DIM, instructions=instructions, scale=scale, document=document)
    )


@cache
def _reply_format_wording(scale: RubricScale, document: bool) -> str:
    """The reply-format contract appended to every prompt of ``scale``, for a document or not, with a placeholder key.

    What a configured judge reads beside its config's own instructions, and so the rest of its identity
    (:func:`~threetears.evals.schema.models.configured_judge_config_id`): the suffix with no instructions before it,
    its one per-dimension part (the reply's key) a placeholder.
    """
    return _system_prompt(dim_id=_PLACEHOLDER_DIM, instructions="", scale=scale, document=document)


def _configured_id(config: JudgeConfig, scale: RubricScale, document: bool) -> str:
    """A configured judge's recorded identity: its config's id and the fingerprint of the suffix it was sent."""
    return configured_judge_config_id(config.id, _reply_format_wording(scale, document))


def judge_config_ids_for(
    rubric: Sequence[RubricDim], configs: Mapping[str, str], *, judged_artifact: JudgedArtifact
) -> dict[str, str]:
    """What each dim a result is judged on would be asked by today, as a result records it.

    A dim with a config is asked by that config's prompt and the engine's reply-format suffix
    (:func:`~threetears.evals.schema.models.configured_judge_config_id`); every other dim by the built-in prompt
    (:func:`builtin_judge_config_ids`). What a caller re-asking a dim as the judge that first asked it compares the
    result's record with.

    :param rubric: The template's rubric.
    :ptype rubric: Sequence[RubricDim]
    :param configs: ``{dim_id: JudgeConfig id}`` for the dims a config asks, as a run records its set.
    :ptype configs: Mapping[str, str]
    :param judged_artifact: What the kind's judge reads, which picks the axes and the wording.
    :ptype judged_artifact: JudgedArtifact
    :return: ``{dim_id: recorded id}``, in judge-phase order.
    :rtype: dict[str, str]
    :raises ValueError: ``judged_artifact`` declares a kind no judge reads.
    """
    builtin = builtin_judge_config_ids(rubric, judged_artifact=judged_artifact)
    document = judged_artifact is JudgedArtifact.DOCUMENT
    scales: dict[str, RubricScale] = {dim.name: dim.scale for dim in rubric}
    return {
        dim: (
            configured_judge_config_id(configs[dim], _reply_format_wording(scales.get(dim, "ordinal"), document))
            if dim in configs
            else builtin_id
        )
        for dim, builtin_id in builtin.items()
    }


def builtin_judge_config_ids(rubric: Sequence[RubricDim], *, judged_artifact: JudgedArtifact) -> dict[str, str]:
    """The identity of the built-in prompt each dim a result is judged on would be asked with, today.

    What a result records for a dim no :class:`JudgeConfig` scored (``EvalResult.judge_config_ids``), and what a
    caller re-asking a dim as the judge that first asked it compares the recorded one with: two built-in wordings are
    two judges (:func:`~threetears.evals.schema.models.builtin_judge_config_id`).

    :param rubric: The template's rubric.
    :ptype rubric: Sequence[RubricDim]
    :param judged_artifact: What the kind's judge reads, which picks the axes and the wording.
    :ptype judged_artifact: JudgedArtifact
    :return: ``{dim_id: builtin id}`` for the two axes (a conversation's) and every rubric dim, in judge-phase order.
    :rtype: dict[str, str]
    :raises ValueError: ``judged_artifact`` declares a kind no judge reads.
    """
    if judged_artifact is JudgedArtifact.UNJUDGED:
        raise ValueError("an unjudged kind has no judge, so no built-in prompt asks it anything")
    document = judged_artifact is JudgedArtifact.DOCUMENT
    axes = (
        {}
        if document
        else {TRANSCRIPT_DIM_ID: _BUILTIN_TRANSCRIPT_CONFIG_ID, OUTCOME_DIM_ID: _BUILTIN_OUTCOME_CONFIG_ID}
    )
    return {**axes, **{dim.name: _builtin_dimension_config_id(dim.scale, document) for dim in rubric}}


class JudgeRequest(NamedTuple):
    """One judge call as it is sent: what :meth:`JudgeService._score` hands the client.

    Built by the service's ``*_request`` methods, which the scoring entry points send and a host's
    capture of judge prompts can read, so what is captured is what a judge was sent rather than a
    second rendering of it.
    """

    #: The dimension scored: a template rubric dim's name or a reserved axis id.
    dim_id: str
    #: The versioned config that supplied the instructions and the client, or ``None`` for the default.
    config: JudgeConfig | None
    #: The instructions with the fixed JSON-format contract appended.
    system_prompt: str
    #: The evidence: scenario, variation, the kind's subject, goal outcomes when included, case material, artifact.
    user_prompt: str
    #: How the dimension is answered.
    scale: RubricScale
    #: The cost-log label for the call.
    label: str
    #: Whether the evidence carries the goal-state outcomes (the transcript axis's does not).
    include_goal_outcomes: bool
    #: The directive the call sends: its reply as a strict schema (:func:`~threetears.evals.run.judge.judge_reply_format`),
    #: built from the same criterion and offer :func:`run_judge_llm` sends it from, so what is priced is what is sent.
    response_format: dict[str, Any]
    #: What asks: ``config``'s id with the reply format's fingerprint (``<config id>@<sha12>``), or the built-in
    #: prompt's identity (:func:`builtin_judge_config_ids`) when no config supplied the instructions — what the
    #: call's outcome records, so two wordings are never one judge.
    judge_config_id: str
    #: The dimension's rubric axis, stamped onto its score: a boundary dimension is a guardrail.
    axis: RubricAxis = "capability"
    #: What the call reads — the evidence's fingerprint and the criterion's — stamped onto its score, so a
    #: person's label of the same output on the same criterion pairs with it on any result (#628).
    label_key: LabelKey | None = None


def _criterion(dim_id: str, scale: RubricScale) -> dict[str, Any]:
    """The one criterion a single-dimension call scores, as :func:`run_judge_llm` reads it and its schema is built."""
    return {"name": dim_id, "weight": 1.0, "scale": scale}


def _request(
    *,
    dim_id: str,
    config: JudgeConfig | None,
    instructions: str,
    builtin_config_id: str,
    user_prompt: str,
    label: str,
    scale: RubricScale,
    include_goal_outcomes: bool,
    document: bool = False,
    axis: RubricAxis = "capability",
    label_key: LabelKey | None = None,
) -> JudgeRequest:
    """Compose one call's system prompt from its instructions and the JSON-format contract.

    ``scale`` is how the dimension is answered, and is required rather than defaulted: a
    pass/fail dimension scored as 1–5 would be refused on every answer the judge was told to give.

    ``builtin_config_id`` is the identity of the built-in prompt the dim is asked with when ``config`` is ``None``,
    and is what the call records as having asked it then.
    """
    return JudgeRequest(
        dim_id=dim_id,
        config=config,
        system_prompt=_system_prompt(dim_id=dim_id, instructions=instructions, scale=scale, document=document),
        user_prompt=user_prompt,
        scale=scale,
        label=label,
        include_goal_outcomes=include_goal_outcomes,
        response_format=judge_reply_format([_criterion(dim_id, scale)], cannot_tell_offered=True),
        judge_config_id=_configured_id(config, scale, document) if config is not None else builtin_config_id,
        axis=axis,
        label_key=label_key,
    )


# ---------------------------------------------------------------------------
# Judge service
# ---------------------------------------------------------------------------


class JudgeService:
    """Stateless single-dim judge. See module docstring for the contract."""

    def __init__(
        self,
        *,
        client_factory: JudgeClientFactory,
        configs: dict[str, JudgeConfig] | None = None,
        failure_describer: ProviderFailureDescriber,
    ) -> None:
        """Bind the service to a client factory and pre-resolved configs.

        :param client_factory: Builds a judge client for a ``(model, temperature)``
            pair. Called at most once per distinct pair (results cached).
        :ptype client_factory: JudgeClientFactory
        :param configs: Active :class:`JudgeConfig` per ``rubric_dim_id``, resolved
            once at run start. A dim absent from the map uses the built-in
            default prompt + default client. ``None`` ⇒ all defaults.
        :ptype configs: dict[str, JudgeConfig] | None
        :param failure_describer: The host's reading of what a judge client raises — the only thing
            that can say a call was refused for the calling account. Required: a reading that
            cannot tell would quietly turn an account refusal into one excluded dim after
            another, so a caller with no reading of its own names
            :func:`~threetears.evals.kernel.provider.withhold_failure_detail` explicitly.
        :ptype failure_describer: ProviderFailureDescriber
        """
        self._client_factory = client_factory
        self._failure_describer = failure_describer
        self._configs = dict(configs or {})
        self._client_cache: dict[tuple[str | None, float | None], Any] = {}

    async def aclose(self) -> None:
        """Release every client this service minted, and empty the cache.

        **This is the lifecycle the cached judge clients attach to.** Each one
        owns an httpx pool the collector does not close deterministically, and
        :meth:`_client_for` cannot release its own mint — the next dim to ask for
        the same ``(model, temperature)`` is about to reuse it. So the release is
        the service's, and it belongs to whoever owns the service: the eval run.

        The cache is emptied as it drains, which makes this idempotent and makes
        a second call a no-op rather than a double close. A client whose release
        raises does not strand the rest — the first failure is re-raised after
        every client has been offered a release, because a run teardown that
        gives up halfway leaks the remainder silently.
        """
        clients = list(self._client_cache.values())
        self._client_cache.clear()
        first_error: BaseException | None = None
        for client in clients:
            try:
                await client.aclose()
            except (
                Exception
            ) as exc:  # prawduct:allow prawduct/broad-except -- one client's teardown must not strand the others
                # NOSILENT: the first failure is raised once every client has been offered its release.
                contain(exc, log=log, what="releasing a judge client failed; its pool may leak")
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

    async def __aenter__(self) -> Self:
        """Enter a scope that releases the judge clients on exit."""
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        """Release the judge clients, whether or not the body raised.

        A teardown failure is logged, never raised: the run's own outcome is the
        more important error, and replacing it with a socket-teardown one would
        report the wrong failure for the run.
        """
        try:
            await self.aclose()
        except (
            Exception
        ) as unreleased:  # prawduct:allow prawduct/broad-except -- teardown must never displace the body's exception
            # NOSILENT: the service holds no run id, and a client pool that will not close is this process's leak,
            # not something the run measured — an error_details entry would count it as one of the run's faults.
            contain(unreleased, log=log, what="releasing the judge clients failed; a pool may leak", level="error")

    async def score_transcript(self, context: JudgeContext) -> JudgeOutcome:
        """Score the transcript (decision-quality) axis — excludes goal outcomes."""
        return await self._score(self.transcript_request(context), case_id=context.case_id)

    async def score_outcome(self, context: JudgeContext) -> JudgeOutcome:
        """Score the outcome (intent-satisfaction) axis — includes goal outcomes."""
        return await self._score(self.outcome_request(context), case_id=context.case_id)

    async def score_dimension(self, dim: RubricDim, context: JudgeContext) -> JudgeOutcome:
        """Score one template rubric dimension."""
        return await self._score(self.dimension_request(dim, context), case_id=context.case_id)

    async def score_request(self, request: JudgeRequest, *, case_id: str) -> JudgeOutcome:
        """Send one call already built by a ``*_request`` method — what a caller that priced it first sends.

        :param request: The call, as built.
        :ptype request: JudgeRequest
        :param case_id: The test case, for log context.
        :ptype case_id: str
        :return: The call's outcome.
        :rtype: JudgeOutcome
        """
        return await self._score(request, case_id=case_id)

    def client_for(self, request: JudgeRequest) -> Any:
        """The client ``request`` will be sent on — the one the service caches for its config's model and temperature.

        What a caller pricing a call before it is made asks: a call's price is its client's.

        :param request: The call.
        :ptype request: JudgeRequest
        :return: The client, built on first ask and cached like every other.
        :rtype: Any
        """
        return self._client_for(request.config)

    def price_ceiling(self, request: JudgeRequest) -> float | None:
        """The most sending ``request`` can cost, every attempt included, or ``None`` when its client cannot say.

        What a caller holding the call against a cost cap reserves before sending it (the cell's judge phase,
        :func:`~threetears.evals.run.runner.judge_dims`, and the judge kind): the client's ceiling for the pair
        as :func:`~threetears.evals.run.judge.run_judge_llm` sends it, times every attempt its parse retry can
        make (:data:`~threetears.evals.run.judge.JUDGE_CALL_ATTEMPTS`).

        :param request: The call, as built.
        :ptype request: JudgeRequest
        :return: The ceiling in dollars, or ``None`` when the client has no ``price_ceiling`` or answers ``None``.
        :rtype: float | None
        """
        return call_ceiling(
            self.client_for(request),
            system=request.system_prompt,
            user=request.user_prompt,
            response_format=request.response_format,
            attempts=JUDGE_CALL_ATTEMPTS,
        )

    # ------------------------------------------------------------------
    # Requests — what each call sends, built without calling anything
    # ------------------------------------------------------------------

    def transcript_request(self, context: JudgeContext) -> JudgeRequest:
        """The transcript axis's call as it is sent: its config, system prompt and user prompt."""
        return self._transcript_call(
            self._build_user_prompt(context, include_goal_outcomes=False),
            label_key_of(context.judge_evidence, TRANSCRIPT_DIM_ID),
        )

    def outcome_request(self, context: JudgeContext) -> JudgeRequest:
        """The outcome axis's call as it is sent: its config, system prompt and user prompt."""
        return self._outcome_call(
            self._build_user_prompt(context, include_goal_outcomes=True),
            label_key_of(context.judge_evidence, OUTCOME_DIM_ID),
        )

    def dimension_request(self, dim: RubricDim, context: JudgeContext) -> JudgeRequest:
        """One rubric dimension's call as it is sent: its config, system prompt and user prompt."""
        return self._dimension_call(
            dim,
            self._build_user_prompt(context, include_goal_outcomes=True),
            label_key_of(context.judge_evidence, dim),
            document=context.judged_artifact is JudgedArtifact.DOCUMENT,
        )

    def floor_requests(
        self, rubric: Sequence[RubricDim], *, intent: str, judged_artifact: JudgedArtifact
    ) -> list[JudgeRequest]:
        """Each call a result's judging consists of, at the smallest it could be sent — buildable before any run.

        What a launch prices a subject arm's judge by before anything is spent
        (:func:`~threetears.evals.run.price_subject_call_floor`): a result's judge calls carry the transcript, the
        kind's rendered evidence, the case's variation and the goal-state outcomes, none of which exist before
        the run, but every other part of each call is fixed by the template and its judge configs. Each request
        here is the call :func:`~threetears.evals.run.runner.judge_requests` builds for the dim, with every part a
        cell fills from its result left out — the variation, the goal outcomes, the subject, the case material
        and the artifact — and everything else sent as sent: the system prompt (the config's instructions or the
        default ones, with the reply contract), the template's intent, every heading, and the reply format.

        So each real call for a dim is its floor request with text inserted, never removed: a client whose
        ceiling does not fall as a prompt grows (one priced from a bound on the prompt's length, as
        :meth:`~threetears.evals.schema.completion.PricedCompletion.price_ceiling` is) prices every real call at
        least as high as its floor request.

        :param rubric: The template's rubric, whose dims are the per-dim calls.
        :ptype rubric: Sequence[RubricDim]
        :param intent: The template's intent, which every call states.
        :ptype intent: str
        :param judged_artifact: What the kind's judge reads, which picks the axes and the wording.
        :ptype judged_artifact: JudgedArtifact
        :return: One request per call, in :func:`~threetears.evals.run.runner.judge_requests`'s order, each carrying
            no label key: it reads no output.
        :rtype: list[JudgeRequest]
        :raises ValueError: ``judged_artifact`` declares a kind no judge reads.
        """
        if judged_artifact is JudgedArtifact.UNJUDGED:
            raise ValueError("an unjudged kind has no judge to price; its grade is the kind's own measures")
        document = judged_artifact is JudgedArtifact.DOCUMENT

        def floor(*, include_goal_outcomes: bool) -> str:
            # A document's goal block is shown only when its kind's checks are non-empty, so the floor leaves it
            # out; a conversation's is always shown, so its heading is kept and its body left empty.
            goals = "" if include_goal_outcomes and not document else None
            return _user_prompt(
                intent=intent, variation="", subject=None, goals=goals, case_material="", artifact="", document=document
            )

        axes = (
            []
            if document
            else [
                self._transcript_call(floor(include_goal_outcomes=False), None),
                self._outcome_call(floor(include_goal_outcomes=True), None),
            ]
        )
        return [
            *axes,
            *[self._dimension_call(dim, floor(include_goal_outcomes=True), None, document=document) for dim in rubric],
        ]

    def judge_config_ids(self, rubric: Sequence[RubricDim], *, judged_artifact: JudgedArtifact) -> dict[str, str]:
        """What asks each dim a result is judged on: its config's id and reply format, or the built-in prompt's
        identity.

        What a result records for each dim its judge phase asks (``EvalResult.judge_config_ids``), whether the dim's
        call returns, raises or is cut off with its cell — the same ids its calls' outcomes carry.

        :param rubric: The template's rubric.
        :ptype rubric: Sequence[RubricDim]
        :param judged_artifact: What the kind's judge reads, which picks the axes and the wording.
        :ptype judged_artifact: JudgedArtifact
        :return: ``{dim_id: config id or builtin id}``, in judge-phase order.
        :rtype: dict[str, str]
        :raises ValueError: ``judged_artifact`` declares a kind no judge reads.
        """
        return judge_config_ids_for(
            rubric, {dim: config.id for dim, config in self._configs.items()}, judged_artifact=judged_artifact
        )

    def _transcript_call(self, user_prompt: str, label_key: LabelKey | None) -> JudgeRequest:
        """The transcript axis's call over ``user_prompt``."""
        config = self._configs.get(TRANSCRIPT_DIM_ID)
        return _request(
            dim_id=TRANSCRIPT_DIM_ID,
            config=config,
            instructions=config.prompt_template if config else _TRANSCRIPT_INSTRUCTIONS,
            builtin_config_id=_BUILTIN_TRANSCRIPT_CONFIG_ID,
            user_prompt=user_prompt,
            label="EvalTranscript",
            scale="ordinal",
            include_goal_outcomes=False,
            label_key=label_key,
        )

    def _outcome_call(self, user_prompt: str, label_key: LabelKey | None) -> JudgeRequest:
        """The outcome axis's call over ``user_prompt``."""
        config = self._configs.get(OUTCOME_DIM_ID)
        return _request(
            dim_id=OUTCOME_DIM_ID,
            config=config,
            instructions=config.prompt_template if config else _OUTCOME_INSTRUCTIONS,
            builtin_config_id=_BUILTIN_OUTCOME_CONFIG_ID,
            user_prompt=user_prompt,
            label="EvalOutcome",
            scale="ordinal",
            include_goal_outcomes=True,
            label_key=label_key,
        )

    def _dimension_call(
        self, dim: RubricDim, user_prompt: str, label_key: LabelKey | None, *, document: bool
    ) -> JudgeRequest:
        """One rubric dimension's call over ``user_prompt``."""
        config = self._configs.get(dim.name)
        return _request(
            dim_id=dim.name,
            config=config,
            instructions=config.prompt_template if config else _dimension_instructions(dim, document=document),
            builtin_config_id=_builtin_dimension_config_id(dim.scale, document),
            user_prompt=user_prompt,
            label="EvalRubric",
            scale=dim.scale,
            include_goal_outcomes=True,
            document=document,
            axis=dim.axis,
            label_key=label_key,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _score(self, request: JudgeRequest, *, case_id: str) -> JudgeOutcome:
        """Send one single-dim judge call and parse it into a :class:`JudgeOutcome`."""
        dim_id = request.dim_id
        client = self._client_for(request.config)
        config_id = request.judge_config_id

        result = await run_judge_llm(
            client=client,
            system_prompt=request.system_prompt,
            user_prompt=request.user_prompt,
            criteria_dicts=[_criterion(dim_id, request.scale)],
            label=request.label,
            case_id=case_id,
            cannot_tell_offered=True,
            failure_describer=self._failure_describer,
        )
        # A judge call that failed to parse, omitted its score, or errored still burned
        # tokens — every return below carries the spend so an unscored dim can't hide the
        # dollars the program paid for it.
        usage = result.get("judge_usage") if isinstance(result, dict) else None

        if result is None or "error" in result:
            error_text = "no result" if result is None else (result.get("error") or "unknown judge error")
            log.warning("Judge dim=%s failed for case=%s: %s", dim_id, case_id, error_text)
            account_refused = result is not None and result.get("account_refused") is True
            return JudgeOutcome(
                score=None,
                config_id=config_id,
                error=error_text,
                usage=usage,
                account_refused=account_refused,
                cut_short=None if result is None else result.get("judge_cut_short"),
            )

        if dim_id in (result.get("criteria_cannot_tell") or []):
            reason = result["reasoning"] or "the judge gave no reason"
            log.info("Judge dim=%s could not tell for case=%s: %s", dim_id, case_id, reason)
            return JudgeOutcome(score=None, config_id=config_id, usage=usage, cannot_tell=reason)

        # The judge's answer as its stored score, which ``run_judge_llm`` has already refused unless
        # it is on the dimension's scale. Read as given: nothing here rounds or clamps it.
        score_int = (result.get("criteria_ordinal_scores") or {}).get(dim_id)
        if score_int is None:
            log.warning("Judge dim=%s returned no score for case=%s", dim_id, case_id)
            return JudgeOutcome(
                score=None, config_id=config_id, error=f"judge omitted score for '{dim_id}'", usage=usage
            )

        return JudgeOutcome(
            score=RubricScore(
                dim=dim_id,
                scale=request.scale,
                axis=request.axis,
                score=score_int,
                # A string by the time it gets here: ``run_judge_llm`` refuses a reply whose
                # reasoning is not one as a parse failure, so it cannot fail this construction
                # after the call's spend was paid.
                reasoning=result["reasoning"],
                # Who scored it, as the provider said — not the client's model, which is the
                # request and may be a floating alias.
                served_model=result.get("judge_served_model"),
                # What the call was sent at, as the client reported it — not what was asked for, since
                # a model that refuses a temperature is sent none.
                judge_temperature=result["judge_temperature"],
                # And the request settings it was sent with, as the client reported them; None = unverified.
                judge_sent_settings=result.get("judge_sent_settings"),
                # What was read on what criterion: the key a person's label of this output pairs by.
                output_fingerprint=None if request.label_key is None else request.label_key.output_fingerprint,
                criterion_fingerprint=None if request.label_key is None else request.label_key.criterion_fingerprint,
            ),
            config_id=config_id,
            usage=usage,
        )

    def _client_for(self, config: JudgeConfig | None) -> Any:
        """Return (and cache) the judge client for a config's model + temperature.

        The narrow end of the judge-model cascade (role default < run pin <
        per-dim ``JudgeConfig.model``): a config's model is passed through when
        set, and ``None`` otherwise — which is how "no opinion" reaches the
        factory that supplies the run pin. Blank and absent deliberately agree,
        so configuring a dim's *prompt* never changes which model scores it.

        **The clients minted here belong to the service, not to this function.**
        Each owns an httpx pool, and a ``finally`` at the mint would close a client
        the next dim is about to reuse — so the release is :meth:`aclose`, called
        by whoever owns the service (eval run teardown).

        The cache is still unbounded by construction: one entry per distinct
        ``(model, temperature)`` a run touches. That is bounded in practice by the
        number of judge configs a template declares, and every entry is released
        together at teardown, so it is a footprint rather than a leak.
        """
        if config is None:
            # The same temperature a config defaults to — never the provider's default, which made a dim's
            # sampling depend on whether anyone had written it a config (#633).
            key: tuple[str | None, float | None] = (None, DEFAULT_JUDGE_TEMPERATURE)
        else:
            key = (config.model or None, config.temperature)
        if key not in self._client_cache:
            self._client_cache[key] = self._client_factory(*key)
        return self._client_cache[key]

    def _build_user_prompt(self, context: JudgeContext, *, include_goal_outcomes: bool) -> str:
        """Place the evidence the judge scores against, as the kind rendered it.

        ``include_goal_outcomes`` is the axis-separation lever: the transcript
        judge never sees the objective final-state checks; the outcome judge
        and the per-dim judges do.

        Nothing here reads the evidence's three strings: they are placed under their headings
        whole, so what the judge reads is exactly what the kind rendered and what a re-judge
        later sends again.
        """
        evidence = context.judge_evidence
        document = context.judged_artifact is JudgedArtifact.DOCUMENT
        goals: str | None = None
        if document:
            # A document's mechanical facts are the kind's own checks, shown only when it has any:
            # a document kind that grades nothing mechanically declares no goal-state checks to say
            # are missing.
            if include_goal_outcomes and context.goal_outcomes:
                goals = _render_goal_outcomes(context.goal_outcomes)
        elif include_goal_outcomes:
            goals = _render_goal_outcomes(context.goal_outcomes)
        return _user_prompt(
            intent=context.intent,
            variation=_render_variation(context.variation),
            subject=evidence.subject,
            goals=goals,
            case_material=evidence.case_material,
            artifact=evidence.artifact,
            document=document,
        )


def _user_prompt(
    *,
    intent: str,
    variation: str,
    subject: str | None,
    goals: str | None,
    case_material: str,
    artifact: str,
    document: bool,
) -> str:
    """Lay out a judge call's user prompt from its parts, each already rendered.

    The one layout a real call (:meth:`JudgeService._build_user_prompt`) and its floor
    (:meth:`JudgeService.floor_requests`) share, so a floor is a real call with its result's parts left
    empty — what makes it a lower bound — and cannot drift from it.

    :param intent: The template's intent.
    :ptype intent: str
    :param variation: The case's variation, rendered.
    :ptype variation: str
    :param subject: The kind's rendering of the candidate, or ``None`` to leave its section out.
    :ptype subject: str | None
    :param goals: The goal-state outcomes, rendered, or ``None`` to leave their section out.
    :ptype goals: str | None
    :param case_material: What the output is judged against.
    :ptype case_material: str
    :param artifact: The output under review, or the transcript.
    :ptype artifact: str
    :param document: Whether the output is a document, which picks the headings.
    :ptype document: bool
    :return: The user prompt.
    :rtype: str
    """
    sections = ["# Scenario", f"**Intent:** {intent}", "", "# Variation params", variation, ""]
    if subject is not None:
        sections += ["# Candidate under test", subject, ""]
    if goals is not None:
        heading = (
            "# Objective checks (already scored — evidence only)"
            if document
            else "# Objective goal-state outcomes (already scored — evidence only)"
        )
        sections += [heading, goals, ""]
    sections += [
        "# Case material (the evidence the output must be judged against)",
        case_material,
        "",
        "# Output under review" if document else "# Transcript",
        artifact,
    ]
    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Evidence rendering — only the engine's own records; the kind renders the rest
# ---------------------------------------------------------------------------


def _render_variation(variation: dict[str, str]) -> str:
    """Render the test case's variation params."""
    if not variation:
        return "(none)"
    return "\n".join(f"- {k}: {v}" for k, v in variation.items())


def _render_goal_outcomes(outcomes: list[GoalStateOutcome]) -> str:
    """Render the objective goal-state checks for the outcome / per-dim judges."""
    if not outcomes:
        return "(no goal-state checks declared)"
    return "\n".join(f"{'✓' if o.passed else '✗'} {o.expression} → {o.detail}" for o in outcomes)


__all__ = [
    "JudgeClientFactory",
    "JudgeContext",
    "JudgeOutcome",
    "JudgeRequest",
    "JudgeService",
    "builtin_judge_config_ids",
    "judge_config_ids_for",
]
