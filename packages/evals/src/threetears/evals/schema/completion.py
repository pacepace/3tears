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
from typing import Any, Literal, Protocol, Self

#: ``response_format`` directive forcing JSON-object output, passed to
#: :meth:`CompletionClient.generate` by callers that parse a structured JSON
#: *object* (the judge, the analysis generator, the proposer/boundary-proposer).
#: Not for callers that expect a JSON *array* — ``{"type": "json_object"}``
#: forbids one, which is why ``variation_gen`` asks for an object wrapping its
#: array rather than passing this and hoping.
JSON_OBJECT_RESPONSE_FORMAT: dict[str, Any] = {"type": "json_object"}


#: The longest one request capped at ``max_tokens`` output tokens can take on the host's client, in
#: seconds: every provider call the client makes for that one request and every wait between them
#: (re-sends of a severed body, SDK retries, their back-off sleeps). The host answers it, because only
#: the host knows its client: how many calls one request can become, how long it sleeps between them,
#: and how slowly a finishing call may write. Every wall-clock ceiling the engine derives over a
#: request (:func:`~threetears.evals.analysis.generation_ceiling_s`,
#: :func:`~threetears.evals.analysis.judge_phase_ceiling_s`,
#: :func:`~threetears.evals.analysis.reporter_cell_timeout_s`) is a count of requests times this, so
#: an answer below the client's real worst case cancels requests that were still going to finish,
#: after they were billed.
RequestCeiling = Callable[[int], float]


#: Why a completion stopped, in the engine's words. A host maps its provider's own finish reason
#: onto one of these before eval sees it: ``"end_turn"`` is a completion that finished, and every
#: other member (:data:`INCOMPLETE_STOP_REASONS`) is one that was cut short — ``"max_tokens"`` at the
#: output cap (OpenAI's ``length``), ``"content_filter"`` by the provider's filter, ``"error"`` when
#: the provider returned no choices at all.
StopReason = Literal["end_turn", "max_tokens", "content_filter", "error"]


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
    provider. ``response_format`` is an optional provider directive (e.g.
    :data:`JSON_OBJECT_RESPONSE_FORMAT`) for callers that parse structured
    output; ``None`` lets the provider default apply.

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

        Args:
            system: The system prompt the call would send.
            user: The user message the call would send.
            response_format: The provider directive the call would send, or ``None``.

        Returns:
            The ceiling in dollars, or ``None``.
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
    "CompletionResult",
    "JSON_OBJECT_RESPONSE_FORMAT",
    "PricedCompletion",
    "ProviderFailure",
    "ProviderFailureDescriber",
    "RequestCeiling",
    "SimulatorLLM",
    "StopReason",
    "USAGE_LEDGER_ATTRIBUTES",
    "VariationLLM",
]
