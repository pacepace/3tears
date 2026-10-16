"""What the eval engine needs from a model provider: the completion port and the shapes it hands back.

The port is :class:`CompletionClient` — ``async generate(*, system, user, response_format=None)``
plus a lifecycle — built by the host and handed to the engine; the engine never builds one and
never names a provider. Beside it sit what a completion hands back (:class:`CompletionResult`) and
why it stopped (:data:`StopReason`); the one-key wire directive for JSON-object output
(:data:`JSON_OBJECT_RESPONSE_FORMAT`); and the failure half of the port (:class:`ProviderFailure` /
:class:`ProviderFailureDescriber`): the client says what a call returns, and the describer says what
the engine may know when the call raised. The engine never reads an exception's own text, because
on an OpenAI-shaped provider ``str(exc)`` is the parsed response envelope, account fields included.

How the engine reads what a completion hands back — failure description and logging, the
cut-short stop reasons, the JSON parsers — is :mod:`threetears.evals.kernel.provider`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final, Protocol, Self

from threetears.evals.schema.models import ClientRequestSettings, StopReason

#: ``response_format`` directive asking for any JSON object, with no shape. No package call sends it: every call
#: whose reply is read into a typed shape (the judge, the analysis generator, the proposer on either axis, the
#: variation writer, the simulated user) sends that shape as a strict ``json_schema`` directive
#: (:func:`~threetears.evals.kernel.strict_response_format`). Kept for a host's own calls, and for its client's
#: translation of the strict directive to a provider that has only JSON mode.
JSON_OBJECT_RESPONSE_FORMAT: dict[str, Any] = {"type": "json_object"}


#: The longest one request capped at ``max_tokens`` output tokens can take on the host's client, in
#: seconds: every provider call the client makes for that one request and every wait between them
#: (re-sends of a severed body, SDK retries, their back-off sleeps). The host answers it, because only
#: the host knows its client: how many calls one request can become, how long it sleeps between them,
#: and how slowly a finishing call may write. Every wall-clock ceiling the engine derives over a
#: request (``Ceilings().generation_ceiling_s``, ``Ceilings().judge_phase_ceiling_s``
#: of :class:`~threetears.evals.run.Ceilings`, and :func:`~threetears.evals.analysis.reporter_cell_timeout_s`) is a count of requests times this, so
#: an answer below the client's real worst case cancels requests that were still going to finish,
#: after they were billed.
RequestCeiling = Callable[[int], float]


class CompletionResult(Protocol):
    """What eval reads off one completion, whatever produced it.

    A structural contract over the host's concrete result type. Every member is something an eval
    call site reads, and every member is read-only — so a frozen dataclass, a frozen Pydantic model
    or a plain object with these attributes all satisfy it. ``cost_usd``, ``input_tokens``,
    ``output_tokens`` and ``reasoning_tokens`` are ``float | None`` / ``int | None`` because a
    provider that reported nothing is a different fact from one that reported zero, and per-role
    usage rows are built on that distinction: a ``None`` count is carried as unknown into every
    usage row and rollup, never summed as zero.

    ``stop_reason`` is a NORMALIZED value (:data:`StopReason`), not the provider's own string. An
    implementation that passes a raw provider string through (OpenAI's ``length``, say) reads as
    finished here, and the truncation surfaces downstream as an unexplained parse failure — after a
    repair round-trip that buys a second full generation into the identical cap.

    **One optional member, not declared here: ``sent_settings``** (:data:`OPTIONAL_COMPLETION_ATTRIBUTES`). A
    :class:`~threetears.evals.schema.ClientRequestSettings` holding what the request that produced this
    completion ACTUALLY carried — its output cap, its reasoning effort word or budget, and whether it
    enforced every parameter (``strict_output``) — after every change the client made to it: a setting sent
    as the closest one the model accepts, one dropped because the model refused it, a strict call resent
    without enforcement (``strict_output=False``). ``None``, or no such attribute at all, is "the client does
    not say", which the engine records as unverified and never refuses. It is not a protocol property
    because the conformance check requires every property, and a host whose completion has no such member
    must keep conforming; the engine reads it through :func:`sent_settings_of` alone. One value per
    completion, never what was asked: a client that copies the settings it was built with here reports a
    mapping nobody checked, and makes a judge sent one thing compare equal to a judge sent another.
    """

    @property
    def content(self) -> str:
        """The completion's text."""
        ...

    @property
    def input_tokens(self) -> int | None:
        """Prompt tokens the provider counted, or ``None`` when it reported no count."""
        ...

    @property
    def output_tokens(self) -> int | None:
        """Completion tokens the provider counted, reasoning included, or ``None`` when it reported no count."""
        ...

    @property
    def cost_usd(self) -> float | None:
        """What the call cost, or ``None`` when nothing priced it."""
        ...

    @property
    def price_source(self) -> str | None:
        """Where ``cost_usd`` came from, as the host names it, or ``None`` when it can name none.

        The provider's own reported figure, a rate card the host applied, a scripted price. Eval
        stores it beside the dollars it qualifies (``RoleUsage.price_source``) so they stay
        re-derivable at current rates, and never supplies one of its own: only the client knows
        which provider priced the call.
        """
        ...

    @property
    def model(self) -> str:
        """The model the call was attributed to, for spend."""
        ...

    @property
    def served_model(self) -> str | None:
        """The model the provider's RESPONSE named as having answered, or ``None`` when it named none.

        Never the id that was requested. ``model`` may be filled from the request when the response
        is silent, which is harmless for attributing spend and wrong as evidence of who answered: a
        floating alias is resolved on the provider's side, so the request names a pointer and only
        the response names the model. The judge records this as the model that scored, and the usage
        ledger as the model that answered each row's calls (``RoleUsage.served_model``), which is how
        two candidate runs launched on one alias and served by different models are told apart. An
        implementation that copies the request here makes two different models compare equal.
        """
        ...

    @property
    def reasoning_tokens(self) -> int | None:
        """The reasoning share of ``output_tokens``, or ``None`` when the provider reported no split."""
        ...

    @property
    def temperature(self) -> float | None:
        """The sampling temperature the request was actually SENT with, or ``None`` when it was sent with none.

        ``None`` is the model's own default applying: a model that refuses a temperature (some reasoning
        models do) is sent none whatever the caller asked for, and only the client that built the request
        knows it did that. The judge records this on every score as part of the judge's identity, so an
        implementation that reports the requested value when it dropped it makes two different judges
        compare equal.
        """
        ...

    @property
    def stop_reason(self) -> StopReason:
        """Why the completion stopped, mapped onto the engine's vocabulary."""
        ...


#: Every member of :class:`CompletionResult`, read off the protocol itself so the list cannot fall
#: behind it. What :func:`~threetears.evals.testing.check_completion_conformance` holds a host's
#: completion type to.
COMPLETION_RESULT_ATTRIBUTES: tuple[str, ...] = tuple(
    name for name, member in vars(CompletionResult).items() if isinstance(member, property)
)


#: The members a completion MAY carry beyond :class:`CompletionResult`'s, each read by the engine through a reader of
#: its own that takes an absent attribute as "not reported": ``sent_settings``, read by :func:`sent_settings_of`.
#: Not protocol properties, so :data:`COMPLETION_RESULT_ATTRIBUTES` and the conformance check's required set never
#: hold them; :func:`~threetears.evals.testing.check_completion_conformance` checks one's type when it is present.
OPTIONAL_COMPLETION_ATTRIBUTES: Final[tuple[str, ...]] = ("sent_settings",)


class CompletionProtocolDefect(TypeError):
    """A host's completion does not meet the :class:`CompletionResult` protocol, found while the engine read it.

    The host's code, not a judging outcome: every call made through the same client returns the same type, so every
    cell of a run would meet it. It therefore fails the run it surfaces in — passed through every per-dimension and
    per-phase catch on the judge path, as
    :class:`~threetears.evals.kernel.candidate_kind.CandidateKindDefect` is — rather than becoming an error on each
    dimension of a run that then finishes. A ``TypeError``, so a caller already catching one keeps catching it. The
    message names :func:`~threetears.evals.testing.check_completion_conformance`, which a host runs on its own
    completion type to find the defect before a run does.
    """

    def __init__(self, completion: object, member: str, why: str, *, optional: bool = False) -> None:
        """Name the completion's type, the member it lacks or fills wrong, and why the engine reads that member.

        :param completion: The completion as the client returned it.
        :ptype completion: object
        :param member: The :class:`CompletionResult` member it does not carry, or the optional member
            (:data:`OPTIONAL_COMPLETION_ATTRIBUTES`) it holds a value of the wrong type in.
        :ptype member: str
        :param why: What the engine reads the member for, or what is wrong with its value, as a clause.
        :ptype why: str
        :param optional: ``member`` is an optional member the completion carries with a wrong value, not a protocol
            member it lacks.
        :ptype optional: bool
        """
        self.completion_type = type(completion).__name__
        self.member = member
        # What the calls already made, and billed, before the defect was found reported — set by the caller that
        # made them (the judge counts its attempts), so the spend goes up with the error and is counted where the
        # run's spend is summed rather than vanishing with the frame. Named as a completion names them
        # (USAGE_LEDGER_ATTRIBUTES), so the error reads as their usage. ``calls`` is 0 when no caller counted them;
        # every other figure is ``None`` when unreported: unknown, never zero.
        self.calls = 0
        self.model: str | None = None
        self.served_model: str | None = None
        self.input_tokens: int | None = None
        self.output_tokens: int | None = None
        self.reasoning_tokens: int | None = None
        self.cost_usd: float | None = None
        self.price_source: str | None = None
        problem = (
            f"{self.completion_type}'s optional `{member}` is not what the CompletionResult protocol documents: {why}"
            if optional
            else f"{self.completion_type} carries no `{member}`, a member of the CompletionResult protocol: {why}"
        )
        super().__init__(
            f"{problem} — threetears.evals.testing.check_completion_conformance holds a host's completion type to the "
            "protocol"
        )


def sent_settings_of(completion: object) -> ClientRequestSettings | None:
    """What the request that produced ``completion`` actually carried, as its client reports it, or ``None``.

    The one reader of the optional ``sent_settings`` member (:data:`OPTIONAL_COMPLETION_ATTRIBUTES`). ``None`` —
    an absent attribute or one holding ``None`` — is the client not saying: unverified, which the engine records and
    discloses and never refuses. A value of any other type is not an absence: the host's completion type declares
    the member and fills it wrong, and every call through its client would, so it is refused by name, as a missing
    protocol member is.

    :param completion: A completion, as a client returned it (or a usage record standing for one).
    :ptype completion: object
    :return: The settings the request was sent with, or ``None`` when the client does not report them.
    :rtype: ClientRequestSettings | None
    :raises CompletionProtocolDefect: ``sent_settings`` is present and is neither ``None`` nor a
        :class:`~threetears.evals.schema.ClientRequestSettings`.
    """
    sent = getattr(completion, "sent_settings", None)
    if sent is None or isinstance(sent, ClientRequestSettings):
        return sent
    raise CompletionProtocolDefect(
        completion,
        "sent_settings",
        f"it holds a {type(sent).__name__}, and the member is what the request carried as a ClientRequestSettings, "
        "or None when the client does not report it",
        optional=True,
    )


#: The :class:`CompletionResult` attributes the usage ledger reads off a completion
#: (:meth:`~threetears.evals.kernel.usage_capture.RoleUsageLedger.add_llm_result`), declared once.
#:
#: The ledger reads each through ``getattr`` with a default, which is right for test doubles that
#: supply part of the set and is also what makes a host's rename silent: every row from a
#: completion type that renamed one of these degrades to "unreported", with no error. So the ledger
#: and the conformance check both read this one tuple, and a host's suite catches a rename with
#: :func:`~threetears.evals.testing.check_completion_conformance`. ``calls`` is not here: it is an
#: eval-side count of retries folded into one record, declared on
#: :class:`~threetears.evals.kernel.usage_capture.CallUsage` alone.
USAGE_LEDGER_ATTRIBUTES: tuple[str, ...] = (
    "model",
    "served_model",
    "input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "cost_usd",
    "price_source",
)


class CompletionGenerator(Protocol):
    """The one call a consumer of a completion client makes, without the client's lifecycle.

    What a function that is handed a client and never owns it types against — so a wrapper that
    forwards calls to a client it does not own (and so must not close) satisfies it.
    """

    async def generate(
        self,
        *,
        system: str,
        user: str,
        response_format: dict[str, Any] | None = None,
    ) -> CompletionResult:
        """Send a prompt pair and return the completion."""
        ...


class CompletionClient(Protocol):
    """The completion port eval is constructed with.

    The host injects an implementation; eval never builds one and never names a
    provider. ``response_format`` is an optional provider directive for callers that parse structured
    output: every engine call that reads its reply into a typed shape sends a strict ``json_schema`` directive
    (``{"type": "json_schema", "json_schema": {"name", "strict": True, "schema"}}``), which a client passes to a
    provider that enforces it, or translates to the nearest mode its provider has (JSON mode, a tool's input
    schema); the engine validates the reply on return either way. ``None`` lets the provider default apply.

    **``generate`` must be safe to call concurrently on one instance.** The judge
    caches one client per ``(model, temperature)`` and scores a result's dimensions
    at once on it (``eval.judge_concurrency``), so an implementation that binds a
    call's request to instance state — and appends the reply there after the
    response — would let two in-flight calls corrupt each other's retries. Build
    each call's request locally.
    """

    async def generate(
        self,
        *,
        system: str,
        user: str,
        response_format: dict[str, Any] | None = None,
    ) -> CompletionResult:
        """Send a prompt pair and return the completion.

        Every engine call site passes the three by keyword, so an implementation may make them
        keyword-only. Concurrent calls on one instance must each send exactly their own prompt
        pair — see the class docstring.
        """
        ...

    async def aclose(self) -> None:
        """Release whatever transport this client owns.

        Declared here because eval BUILDS one of these per unit of work — a
        proposal, an analysis generation — and therefore has to be able to
        release one. An implementation that owns nothing satisfies this with a
        no-op; one that owns a connection pool must, or eval leaks it at the rate
        it works.

        Not optional, and not discoverable by duck-typing at the call site: an
        implementation conforming to a ``generate``-only contract would fail with
        ``AttributeError`` deep inside a run, which reads as an eval bug rather
        than a missing method on the injected client.
        """
        ...

    async def __aenter__(self) -> Self:
        """Enter a scope that releases on exit.

        Eval consumes clients with ``async with`` at most sites, so the contract
        declares that form rather than leaving each caller to discover whether
        the object it was handed supports it.
        """
        ...

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        """Release on the way out, whether or not the body raised."""
        ...


class BoundCompletionClient(CompletionClient, Protocol):
    """A completion client built for one model, which it names.

    What a host's client factory returns
    (:class:`~threetears.evals.kernel.host.eval_host.CompletionClients`). A caller that asked
    for a role's default model reads back which model that resolved to, so what it records — a
    generation's provenance, a pre-spend disclosure — names the model the client will actually
    call rather than a re-derivation of the host's cascade.
    """

    #: The model this client calls, as the host resolved it.
    model_name: str

    def price_ceiling(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> float | None:
        """The most one :meth:`generate` with this prompt pair can cost, in dollars, or ``None`` when it cannot say.

        See :meth:`PricedCompletion.price_ceiling`: a client the engine calls outside any run — a case
        generation's writer, a rubric proposer — is priced through this before the call is made.
        """
        ...


class PricedCompletion(Protocol):
    """A completion the engine can price before it makes it: the call, the model, and the call's ceiling.

    What every call the engine makes OUTSIDE a run goes through
    (:class:`~threetears.evals.kernel.out_of_run.OutOfRunBudget`): a case generation's ``llm``
    axis writer and the rubric proposer. A run's calls are bounded by its cost cap as their spend
    arrives; an out-of-run call has no run around it, so it is priced before it is made and refused
    when the price would pass the cap. The engine knows neither a model's rates nor the output cap
    the host built the client with, so the price is the client's answer — never the engine's guess.

    :class:`BoundCompletionClient` and :class:`VariationLLM` both satisfy it.
    """

    #: The model this client calls, as the host resolved it — what the call is priced and recorded as.
    model_name: str

    def price_ceiling(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> float | None:
        """The most one :meth:`generate` with this prompt pair can cost, in dollars, or ``None`` when it cannot say.

        A CEILING, not an estimate: the host's rate for :attr:`model_name` applied to a bound on the
        prompt's tokens and to the output cap it built this client with, reasoning included, times
        every attempt the client makes for one request (a re-sent severed body is billed twice). A
        figure below what the call can cost lets a call past the cap it was admitted under.

        ``None`` is "this client cannot bound the call" — no rate for its model — and an out-of-run
        call under an enforced cap is then refused rather than made, since unknown is not $0.

        :param system: The system prompt the call would send.
        :ptype system: str
        :param user: The user message the call would send.
        :ptype user: str
        :param response_format: The provider directive the call would send, or ``None``.
        :ptype response_format: dict[str, Any] | None
        :return: The ceiling in dollars, or ``None``.
        :rtype: float | None
        """
        ...

    async def generate(
        self, *, system: str, user: str, response_format: dict[str, Any] | None = None
    ) -> CompletionResult:
        """Send a prompt pair and return the completion.

        Read as a :class:`CompletionResult` — by that protocol's attribute names, ``served_model`` for the
        model the response named among them — for the call's ledger row. One protocol, whatever the client's
        shape: an attribute the result lacks reads as unreported, never zero, and one named otherwise (a
        simulator-shaped ``model`` standing in for ``served_model``) is not read.
        """
        ...  # pragma: no cover — protocol


class SimulatorLLM(Protocol):
    """The one-shot text-generation port the simulator role and the variation generator call.

    Narrower than :class:`CompletionClient`: one async ``generate(*, system, user)`` returning an
    object with a ``content`` attribute, and an optional ``response_format`` provider directive.
    Its callers on the simulator role are a simulated actor, which sends its reply schema there, and
    the ``llm_decided`` turn scheduler, which sends a schema whose one field is an ``enum`` of the
    answers legal at that pick; the variation generator's ``llm`` axis passes
    ``{"type": "json_object"}`` to force JSON mode. A host's full chat client is compatible -- its
    tool calls and conversation state are simply not needed for one utterance or one pick. Usage is
    read off the returned object by attribute, by :class:`CompletionResult`'s names (the simulator reads
    ``model``, ``input_tokens``, ``output_tokens``, ``reasoning_tokens``, ``cost_usd``, ``price_source``;
    a variation writer's out-of-run ledger reads those and ``served_model`` and ``stop_reason``,
    :class:`PricedCompletion`), and an attribute it lacks reads as unreported, never zero. ``model`` is the
    model the call was attributed to, which a client may fill from the request; ``served_model`` is the one
    the response named, and the two are not interchangeable.

    The variation generator types against :class:`VariationLLM`, this port with the model it calls.

    Declared in the schema rather than beside the simulator because two packages call it: the run
    package's simulator and the gen package's variation generator. The dependency matrix lets gen
    reach the schema and never run, so a port both sides type against has to live here.
    """

    async def generate(
        self, *, system: str, user: str, response_format: dict[str, Any] | None = None
    ) -> Any: ...  # pragma: no cover — protocol


class VariationLLM(SimulatorLLM, Protocol):
    """The client the variation generator writes an ``llm`` axis's values with, naming the model it calls.

    :class:`SimulatorLLM`'s one call, plus the model the client resolved to: the generated cases
    are the run's stimulus, so the run records which model wrote them
    (:attr:`~threetears.evals.schema.models.VariationCounts.variation_model`), read off the client
    that made the calls rather than restated by whoever built it. A host's
    :class:`BoundCompletionClient` for the ``variation`` role
    (:data:`~threetears.evals.kernel.host.eval_host.CompletionRole`) satisfies it.
    """

    #: The model this client calls, as the host resolved it.
    model_name: str

    def price_ceiling(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> float | None:
        """The most one :meth:`generate` with this prompt pair can cost — see :meth:`PricedCompletion.price_ceiling`.

        A generation's calls run outside every run, so each is priced before it is made.
        """
        ...


@dataclass(frozen=True, slots=True)
class ProviderFailure:
    """What eval is allowed to know about a completion call that raised.

    The failure half of the completion port. :class:`CompletionClient` says what
    eval asks a provider for and :class:`CompletionResult` what it hands back; this
    says what eval reads when the asking raised. Eval never touches the exception's
    own text, because on an OpenAI-shaped provider ``str(exc)`` for a status error
    IS the parsed response envelope, and on some routing providers that envelope carries the
    account's id.

    ``description`` is caller-facing text. It is logged, and it is persisted on the
    stored result an operator later reads in a report, so it has to carry the
    actionable half of the failure — the status and the provider's own message —
    while carrying nothing else.

    ``payload_withheld`` says whether ``description`` deliberately omits something
    the exception's own text carries — or the text of an exception it chains, which a
    traceback prints too. **It is not a label; it is the flag a caller
    must consult before logging a traceback**, whose last line is ``str(exc)`` — the
    text the description was written to replace. A describer that redacts and then
    reports ``False`` re-opens through the logs exactly what the stored detail is
    keeping out, and logs are inside the same containment boundary as storage. The
    bool exists because that decision cannot be made from ``description`` alone:
    redacted text and honest text look alike.

    ``account_refused`` says the provider refused the call for the calling ACCOUNT — out
    of credit, or the key refused — rather than for anything about the model or the
    request. Every model behind the same key gets the same answer, so a caller that
    runs a candidate's own call reads it as an apparatus fault and stops the run
    (``CandidateOutput.account_refused``) instead of charging the candidate. It is a
    field of the port rather than a type eval matches on because only the host knows
    its client's error types; a describer that cannot tell says ``False``, and the
    failure is then charged as any other.
    """

    description: str
    payload_withheld: bool
    account_refused: bool


class ProviderFailureDescriber(Protocol):
    """The host's mapping from its own exceptions onto :class:`ProviderFailure`.

    Injected the way the client is: eval is *constructed with* one and names no
    provider exception type of its own. A host maps its SDK's error hierarchy —
    the status-carrying class whose body must not be repeated, and everything
    else, whose text is its own and is worth keeping.

    **Total over ``BaseException``, not over the host's own error types.** Eval
    hands it whatever the injected client raised, which includes exceptions the
    provider never produced: a timeout from eval's own code, a bug in the callable
    the host supplied. A describer that assumes its SDK's shape and raises on
    anything else does not abort the work — a caller that has already paid for the
    call contains that and falls back to :func:`withhold_failure_detail` — but the
    diagnosis for that case is lost, and no fallback can recover what the describer
    was supposed to say.
    """

    def __call__(self, exc: BaseException) -> ProviderFailure:
        """Describe one raised exception without repeating a provider payload."""
        ...


__all__ = [
    "BoundCompletionClient",
    "COMPLETION_RESULT_ATTRIBUTES",
    "CompletionClient",
    "CompletionGenerator",
    "CompletionProtocolDefect",
    "CompletionResult",
    "JSON_OBJECT_RESPONSE_FORMAT",
    "OPTIONAL_COMPLETION_ATTRIBUTES",
    "PricedCompletion",
    "ProviderFailure",
    "ProviderFailureDescriber",
    "RequestCeiling",
    "SimulatorLLM",
    "StopReason",
    "USAGE_LEDGER_ATTRIBUTES",
    "VariationLLM",
    "sent_settings_of",
]
