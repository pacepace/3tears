"""Stateless single-dimension judge service.

One call → one dimension → one :class:`RubricScore`. The judge never holds
scoring state between calls; the only mutable state is a client cache (pure
construction-memoization). **Calls may run concurrently on one cached client** —
the runner gathers a result's ``2 + N`` calls under ``eval.judge_concurrency`` —
which is sound only because the port requires each ``generate()`` to build its
own request rather than share instance state
(:class:`~threetears.evals.contracts.provider.CompletionClient`); the runner was once serial
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
(:class:`~threetears.evals.contracts.models.JudgeEvidence`) — and this service places those three
strings in the prompt without reading them. The engine renders no transcript and no subject of its
own: only the kind knows how its turns and tool calls read as text, and which facts in play a judge
should see that the candidate's interlocutors did not. The kind's declaration
(:class:`~threetears.evals.contracts.models.JudgedArtifact`) picks the axes and the wording.

Each call resolves a versioned :class:`JudgeConfig` (pre-resolved per dim at run
start, keyed by ``rubric_dim_id``); when one exists its ``prompt_template`` is
used as the judging instructions and its ``model`` / ``temperature`` select the
client. With no config, a built-in default prompt + the default judge client
are used. The fixed JSON-format instruction is appended to **both** so the
response stays parseable regardless of an operator's prompt wording.

The multi-dim composite prompt builder (``judge_prompts.build_rubric_prompt``)
is retired; the per-axis / per-dim builders here replace it. The underlying LLM
call + parse + cost-log still flow through
:func:`threetears.evals.run.judge.run_judge_llm` (public, for a host's own scorers).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple, Self

from threetears.evals.contracts.models import (
    OUTCOME_DIM_ID,
    SCALE_LEVELS,
    TRANSCRIPT_DIM_ID,
    GoalStateOutcome,
    JudgeConfig,
    JudgedArtifact,
    JudgeEvidence,
    RoleUsage,
    RubricDim,
    RubricScale,
    RubricScore,
    UsageRole,
)
from threetears.evals.contracts.host.eval_host import CompletionClients
from threetears.evals.contracts.provider import BoundCompletionClient, ProviderFailureDescriber
from threetears.evals.contracts.usage_capture import CallUsage, RoleUsageLedger, blended_cost
from threetears.evals.run.judge import CANNOT_TELL, run_judge_llm
from threetears.observe import get_logger

log = get_logger(__name__)

#: Builds a judge LLM client for a ``(model, temperature)`` pair. ``model`` is
#: ``None`` when this dim states no model of its own — what the factory
#: substitutes is the caller's business, and the run-scoped one supplies the
#: run's judge pin. ``temperature`` is ``None`` for the provider default. The
#: service caches the result so identical configs reuse one client.
JudgeClientFactory = Callable[[str | None, float | None], Any]


def judge_clients_for_run(clients: CompletionClients, judge_model: str | None) -> JudgeClientFactory:
    """The judge client factory for a run pinned to ``judge_model``, over the host's client factory.

    The run-scoped half of the judge's model cascade: a dim whose configuration names a model is
    scored by that model, and every other dim by the run's pin. A ``None`` pin asks the host for the
    judge role's default.

    Args:
        clients: The host's completion-client factory.
        judge_model: The run's judge pin, resolved, or ``None``.

    Returns:
        The factory a :class:`JudgeService` is built with.
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

        Raises:
            ValueError: ``judged_artifact`` is :attr:`~JudgedArtifact.UNJUDGED`.
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
    that the evidence does not let it score the dim. ``config_id`` names the
    versioned :class:`JudgeConfig` used, or ``None`` when the built-in default
    prompt scored the dim.
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


#: The one role a judge call spends under — what a fold of judge calls sums its cost over.
_JUDGE_ROLE: tuple[UsageRole, ...] = ("judge",)


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


def fold_judge_outcomes(outcomes: list[tuple[str, JudgeOutcome]]) -> JudgeFold:
    """Fold judge calls' outcomes into their configs, errors and spend.

    Shared by the cell's judge phase and a later re-judge, so both read an outcome the same
    way. **A call that returned no score is an error even when it names none**: the judge
    service sets one on every failure, so the unnamed case is a contract breach, and letting
    it through silently would leave a dim unscored while nothing says it failed.

    Judge cost comes off the usage ledger rather than a per-call cost field, because a dim
    that burned tokens and then failed to parse is spend the run still paid. It is ``None``
    when any call went unpriced (:func:`~threetears.evals.contracts.usage_capture.blended_cost`).

    Args:
        outcomes: ``(dim_id, outcome)`` for every call made, in call order.

    Returns:
        The fold.
    """
    config_ids: dict[str, str] = {}
    errors: list[tuple[str, str]] = []
    cannot_tell: dict[str, str] = {}
    # Per-dim judge configs may each pin their own model, so the ledger splits rows by
    # model rather than blending the dims into one — see threetears.evals.contracts.usage_capture.
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
    rows = ledger.rows()
    return JudgeFold(
        config_ids=config_ids,
        errors=errors,
        usage=rows,
        cost_usd=blended_cost(rows, _JUDGE_ROLE),
        cannot_tell=cannot_tell,
    )


# ---------------------------------------------------------------------------
# Built-in default judging instructions (used when no JudgeConfig is active)
# ---------------------------------------------------------------------------

_TRANSCRIPT_INSTRUCTIONS = (
    "You are evaluating the DECISION QUALITY of a candidate's turn-by-turn "
    "behavior in a conversation. Score how well-chosen the candidate's reasoning, tool selection, and "
    "tool parameters were GIVEN THE INFORMATION AVAILABLE TO IT at each step.\n\n"
    "Judge the choices, not the outcomes: do NOT penalize the candidate for tool "
    "failures, empty search results, or world state outside its control — a sound "
    "decision with an unlucky result still scores well. You are NOT assessing whether "
    "the final result satisfied the user; a separate judge does that."
)

_OUTCOME_INSTRUCTIONS = (
    "You are evaluating whether the candidate ACHIEVED THE USER'S INTENT. Score how "
    "well the final state and the candidate's responses satisfied what the user "
    "actually wanted.\n\n"
    "Judge the result, not the path: a clumsy route to a correct outcome still scores "
    "well; elegant reasoning that left the intent unmet scores poorly. The objective "
    "goal-state checks below are already computed — use them as evidence, do not "
    "re-derive them."
)

#: How each scale is asked for: the scale sentence, and what ``criteria_scores`` maps the key to.
_SCALE_WORDING: dict[str, tuple[str, str]] = {
    "ordinal": ("Score on an integer scale of 1 (worst) to 5 (best).", "integer score (1-5)"),
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

    Args:
        dim: The dimension being scored.
        document: Whether the candidate's output is a document judged against case material,
            rather than its side of a conversation.

    Returns:
        The system instructions, before the JSON-format suffix.
    """
    subject = "a candidate's output" if document else "a candidate's side of a conversation"
    lines = [
        f"You are scoring {subject} on the subjective rubric dimension '{dim.name}'.",
        dim.description,
    ]
    if dim.scoring_guide:
        lines.append("Scoring guide:")
        for k in SCALE_LEVELS[dim.scale]:
            if k in dim.scoring_guide:
                lines.append(f"  {k}: {dim.scoring_guide[k]}")
    lines.append(
        "Be honest and specific — vague scoring is useless to operators reviewing the "
        "run. Do NOT re-judge objective goal-state outcomes; they are scored separately "
        "and shown only for context."
    )
    return "\n".join(lines)


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


def _request(
    *,
    dim_id: str,
    config: JudgeConfig | None,
    instructions: str,
    user_prompt: str,
    label: str,
    scale: RubricScale,
    include_goal_outcomes: bool,
    document: bool = False,
) -> JudgeRequest:
    """Compose one call's system prompt from its instructions and the JSON-format contract.

    ``scale`` is how the dimension is answered, and is required rather than defaulted: a
    pass/fail dimension scored as 1–5 would be refused on every answer the judge was told to give.

    ``document`` says the candidate produced a document rather than a conversation: it changes
    what the JSON contract tells the judge to cite, never what is scored.
    """
    evidence = "passages of the output and of the case material" if document else "moments in the transcript"
    material = "output and case material" if document else "transcript"
    scale_sentence, scale_answer = _SCALE_WORDING[scale]
    system_prompt = instructions + _JSON_FORMAT_TEMPLATE.format(
        dim_id=dim_id,
        evidence=evidence,
        material=material,
        cannot_tell=CANNOT_TELL,
        scale_sentence=scale_sentence,
        scale_answer=scale_answer,
    )
    return JudgeRequest(
        dim_id=dim_id,
        config=config,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        scale=scale,
        label=label,
        include_goal_outcomes=include_goal_outcomes,
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

        Args:
            client_factory: Builds a judge client for a ``(model, temperature)``
                pair. Called at most once per distinct pair (results cached).
            configs: Active :class:`JudgeConfig` per ``rubric_dim_id``, resolved
                once at run start. A dim absent from the map uses the built-in
                default prompt + default client. ``None`` ⇒ all defaults.
            failure_describer: The host's reading of what a judge client raises — the only thing
                that can say a call was refused for the calling account. Required: a reading that
                cannot tell would quietly turn an account refusal into one excluded dim after
                another, so a caller with no reading of its own names
                :func:`~threetears.evals.contracts.provider.withhold_failure_detail` explicitly.
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
                log.warning("releasing a judge client failed; its pool may leak", exc_info=True)
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
        except Exception:  # prawduct:allow prawduct/broad-except -- teardown must never displace the body's exception
            log.warning("releasing the judge clients failed; a pool may leak", exc_info=True)

    async def score_transcript(self, context: JudgeContext) -> JudgeOutcome:
        """Score the transcript (decision-quality) axis — excludes goal outcomes."""
        return await self._score(self.transcript_request(context), case_id=context.case_id)

    async def score_outcome(self, context: JudgeContext) -> JudgeOutcome:
        """Score the outcome (intent-satisfaction) axis — includes goal outcomes."""
        return await self._score(self.outcome_request(context), case_id=context.case_id)

    async def score_dimension(self, dim: RubricDim, context: JudgeContext) -> JudgeOutcome:
        """Score one template rubric dimension."""
        return await self._score(self.dimension_request(dim, context), case_id=context.case_id)

    # ------------------------------------------------------------------
    # Requests — what each call sends, built without calling anything
    # ------------------------------------------------------------------

    def transcript_request(self, context: JudgeContext) -> JudgeRequest:
        """The transcript axis's call as it is sent: its config, system prompt and user prompt."""
        config = self._configs.get(TRANSCRIPT_DIM_ID)
        return _request(
            dim_id=TRANSCRIPT_DIM_ID,
            config=config,
            instructions=config.prompt_template if config else _TRANSCRIPT_INSTRUCTIONS,
            user_prompt=self._build_user_prompt(context, include_goal_outcomes=False),
            label="EvalTranscript",
            scale="ordinal",
            include_goal_outcomes=False,
        )

    def outcome_request(self, context: JudgeContext) -> JudgeRequest:
        """The outcome axis's call as it is sent: its config, system prompt and user prompt."""
        config = self._configs.get(OUTCOME_DIM_ID)
        return _request(
            dim_id=OUTCOME_DIM_ID,
            config=config,
            instructions=config.prompt_template if config else _OUTCOME_INSTRUCTIONS,
            user_prompt=self._build_user_prompt(context, include_goal_outcomes=True),
            label="EvalOutcome",
            scale="ordinal",
            include_goal_outcomes=True,
        )

    def dimension_request(self, dim: RubricDim, context: JudgeContext) -> JudgeRequest:
        """One rubric dimension's call as it is sent: its config, system prompt and user prompt."""
        config = self._configs.get(dim.name)
        document = context.judged_artifact is JudgedArtifact.DOCUMENT
        return _request(
            dim_id=dim.name,
            config=config,
            instructions=config.prompt_template if config else _dimension_instructions(dim, document=document),
            user_prompt=self._build_user_prompt(context, include_goal_outcomes=True),
            label="EvalRubric",
            scale=dim.scale,
            include_goal_outcomes=True,
            document=document,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _score(self, request: JudgeRequest, *, case_id: str) -> JudgeOutcome:
        """Send one single-dim judge call and parse it into a :class:`JudgeOutcome`."""
        dim_id = request.dim_id
        client = self._client_for(request.config)
        config_id = request.config.id if request.config else None

        result = await run_judge_llm(
            client=client,
            system_prompt=request.system_prompt,
            user_prompt=request.user_prompt,
            criteria_dicts=[{"name": dim_id, "weight": 1.0, "scale": request.scale}],
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
                score=None, config_id=config_id, error=error_text, usage=usage, account_refused=account_refused
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
                score=score_int,
                # A string by the time it gets here: ``run_judge_llm`` refuses a reply whose
                # reasoning is not one as a parse failure, so it cannot fail this construction
                # after the call's spend was paid.
                reasoning=result["reasoning"],
                # Who scored it, as the provider said — not the client's model, which is the
                # request and may be a floating alias.
                served_model=result.get("judge_served_model"),
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
            key: tuple[str | None, float | None] = (None, None)
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
        sections = [
            "# Scenario",
            f"**Intent:** {context.intent}",
            "",
            "# Variation params",
            _render_variation(context.variation),
            "",
        ]
        if evidence.subject is not None:
            sections += ["# Candidate under test", evidence.subject, ""]
        if document:
            # A document's mechanical facts are the kind's own checks, shown only when it has any:
            # a document kind that grades nothing mechanically declares no goal-state checks to say
            # are missing.
            if include_goal_outcomes and context.goal_outcomes:
                sections += [
                    "# Objective checks (already scored — evidence only)",
                    _render_goal_outcomes(context.goal_outcomes),
                    "",
                ]
        elif include_goal_outcomes:
            sections += [
                "# Objective goal-state outcomes (already scored — evidence only)",
                _render_goal_outcomes(context.goal_outcomes),
                "",
            ]
        sections += [
            "# Case material (the evidence the output must be judged against)",
            evidence.case_material,
            "",
            "# Output under review" if document else "# Transcript",
            evidence.artifact,
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
]
