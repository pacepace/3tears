"""What the eval engine needs from a model provider, and how it reads what one hands back.

The port is :class:`CompletionClient` — ``async generate(*, system, user, response_format=None)``
plus a lifecycle — built by the host and handed to the engine; the engine never builds one and
never names a provider. Around it sit the things every caller of the port shares:

* what a completion hands back (:class:`CompletionResult`) and why it stopped (:data:`StopReason`,
  with :data:`INCOMPLETE_STOP_REASONS` the members meaning it was cut short);
* the one-key wire directive for JSON-object output (:data:`JSON_OBJECT_RESPONSE_FORMAT`), and the
  JSON-object and JSON-array parsers;
* the failure half of the port (:class:`ProviderFailure` / :class:`ProviderFailureDescriber`): the
  client says what a call returns, and the describer says what the engine may know when the call
  raised. The engine never reads an exception's own text, because on an OpenAI-shaped provider
  ``str(exc)`` is the parsed response envelope, account fields included.

``sum_optional_tokens`` is four lines of tri-state arithmetic, and the tri-state is the part that
must not drift: ``None`` is "not measured", not zero, and per-role cost attribution depends on
being able to say so.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Literal, Protocol, Self

from threetears.observe import get_logger

log = get_logger(__name__)

#: ``response_format`` directive forcing JSON-object output, passed to
#: :meth:`CompletionClient.generate` by callers that parse a structured JSON
#: *object* (the judge, the analysis generator, the proposer/boundary-proposer).
#: Not for callers that expect a JSON *array* — ``{"type": "json_object"}``
#: forbids one, which is why ``variation_gen`` asks for an object wrapping its
#: array rather than passing this and hoping.
JSON_OBJECT_RESPONSE_FORMAT: dict[str, Any] = {"type": "json_object"}

#: Full provider calls one completed request can cost: the host's client re-sends a request whose
#: response body fails to parse, once, because a severed body is a completion already billed. A
#: caller bounding a request's wall clock multiplies by this. Duplicated from the host client's own
#: ``BODY_PARSE_ATTEMPTS`` for the same reason as the literal above; a test pins the two equal.
PROVIDER_REQUEST_ATTEMPTS = 2


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
    or a plain object with these attributes all satisfy it. ``cost_usd`` and ``reasoning_tokens``
    are ``float | None`` / ``int | None`` because a provider that reported nothing is a different
    fact from one that reported zero, and per-role usage rows are built on that distinction.

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
    def input_tokens(self) -> int:
        """Prompt tokens the provider counted."""
        ...

    @property
    def output_tokens(self) -> int:
        """Completion tokens the provider counted, reasoning included."""
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
        the response names the model. The judge records this as the model that scored, so an
        implementation that copies the request here makes two different scorers compare equal.
        """
        ...

    @property
    def reasoning_tokens(self) -> int | None:
        """The reasoning share of ``output_tokens``, or ``None`` when the provider reported no split."""
        ...

    @property
    def stop_reason(self) -> StopReason:
        """Why the completion stopped, mapped onto the engine's vocabulary."""
        ...


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
    (:class:`~threetears.evals.contracts.host.eval_host.CompletionClients`). A caller that asked
    for a role's default model reads back which model that resolved to, so what it records — a
    generation's provenance, a pre-spend disclosure — names the model the client will actually
    call rather than a re-derivation of the host's cascade.
    """

    #: The model this client calls, as the host resolved it.
    model_name: str


class SimulatorLLM(Protocol):
    """The one-shot text-generation port the simulated user and the variation generator call.

    Narrower than :class:`CompletionClient`: one async ``generate(*, system, user)`` returning an
    object with a ``content`` attribute, and an optional ``response_format`` provider directive.
    The simulated user sends its reply schema there; the variation generator's ``llm`` axis passes
    ``{"type": "json_object"}`` to force JSON mode. A host's full chat client is
    compatible -- its tool calls and conversation state are simply not needed for one utterance
    per simulated turn.

    Declared in contracts rather than beside the simulator because two packages call it: the run
    package's simulator and the gen package's variation generator. The dependency matrix lets gen
    reach contracts and never run, so a port both sides type against has to live here.
    """

    async def generate(
        self, *, system: str, user: str, response_format: dict[str, Any] | None = None
    ) -> Any: ...  # pragma: no cover — protocol


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


#: What a caller with no injected describer is told in place of the exception's own
#: text. Parenthesised at the call site below so it reads as the absence it is,
#: rather than as something a provider said.
_NO_DESCRIBER_NOTICE = (
    "detail withheld — no ProviderFailureDescriber was supplied, and an unrecognised exception's "
    "text may be the provider's own response body"
)


def withhold_failure_detail(exc: BaseException) -> ProviderFailure:
    """Describe a failure by its class and nothing else — the safe default.

    What a caller gets when the host injected no describer, and deliberately the
    *lossy* answer. Eval cannot tell a provider status error from a timeout without
    naming a provider SDK, and that import is the edge this port exists to cut; so
    with no host mapping in hand, the exception's class is the only thing it can
    say that is certainly not the envelope.

    **Lossy and loud, rather than plausible and silent.** The shape this replaced
    was an error path that degraded to ``str(exc)`` for any client the host had not
    anticipated: right-looking text with the account id in it, and nothing anywhere
    saying the mapping was missing. An operator reading "detail withheld — no
    ProviderFailureDescriber was supplied" knows what is unwired. An operator
    reading a plausible provider message does not, and neither does a test.

    Args:
        exc: The raised exception.

    Returns:
        The exception's class name and a statement of what was withheld, with
        ``payload_withheld`` set — so the caller drops the traceback too, whose
        last line is the text this declined to repeat — and ``account_refused``
        unset, since telling an account's refusal apart needs the host's types.
    """
    return ProviderFailure(
        f"{type(exc).__name__}: ({_NO_DESCRIBER_NOTICE})", payload_withheld=True, account_refused=False
    )


def describe_failure(
    describer: ProviderFailureDescriber, exc: BaseException, *, logger: logging.Logger, where: str
) -> ProviderFailure:
    """Describe a failed call through the host's describer, containing a describer that itself raises.

    What :class:`ProviderFailureDescriber` asks of a caller that has already paid for the call:
    this runs on the way out of a failure the caller must still record, and a second exception
    here would replace the first. A raising describer is logged by its class alone — neither its
    own text nor a traceback, since raising inside an ``except`` chains the failure being
    described, whose ``str`` may be the provider envelope — and the failure is then described by
    :func:`withhold_failure_detail`.

    Args:
        describer: The host's describer.
        exc: What the call raised.
        logger: The caller's logger, so the line lands with the caller's other records.
        where: What was being described, named in that line.

    Returns:
        The describer's answer, or the withheld description when the describer raised.
    """
    try:
        return describer(exc)
    # prawduct:ok-broad-except — a broken describer must not replace the failure being recorded
    except Exception as describer_exc:
        logger.error(
            "%s failure describer raised %s describing %s — keeping the class only",
            where,
            type(describer_exc).__name__,
            type(exc).__name__,
        )
        return withhold_failure_detail(exc)


def log_provider_failure(
    logger: logging.Logger,
    failure: ProviderFailure,
    exc: BaseException,
    message: str,
    *args: object,
    level: int = logging.ERROR,
) -> None:
    """Log a described failure, with its traceback only when the description withheld nothing.

    The step every caller of :func:`describe_failure` takes next, held in one place so no caller
    can take it differently: when ``failure.payload_withheld`` is set, a traceback's last line is
    ``str(exc)`` — the provider envelope, account id included, that the description was written to
    keep out — and logs sit inside the same containment boundary as storage. So that line carries
    the description and no traceback; an honest description keeps the traceback, which is then
    the cheapest diagnosis there is.

    Args:
        logger: The caller's logger, so the line lands with the caller's other records.
        failure: What :func:`describe_failure` returned for ``exc``.
        exc: The exception that was described.
        message: A %-format naming what failed; ``": <description>"`` is appended to it.
        *args: Arguments for ``message``.
        level: The level to log at.
    """
    logger.log(
        level, message + ": %s", *args, failure.description, exc_info=exc if traceback_is_safe(failure) else None
    )


def traceback_is_safe(failure: ProviderFailure) -> bool:
    """Whether the described exception's traceback may be printed — here, or by whatever it escapes to.

    The one reading of ``payload_withheld``: :func:`log_provider_failure` asks it before logging a
    traceback, and a caller deciding whether to let the exception propagate to a layer that logs
    tracebacks (a web server) asks the same question, since that layer prints ``str(exc)`` too.

    Args:
        failure: What :func:`describe_failure` returned for the exception.

    Returns:
        ``True`` when the description withheld nothing a traceback would print.
    """
    return not failure.payload_withheld


def describe_and_log_failure(
    describer: ProviderFailureDescriber,
    exc: BaseException,
    *,
    logger: logging.Logger,
    where: str,
    message: str,
    args: tuple[object, ...] = (),
    level: int = logging.ERROR,
) -> ProviderFailure:
    """Describe a failed call (:func:`describe_failure`) and log it (:func:`log_provider_failure`).

    For a caller whose log line does not depend on what the description says; one that chooses
    its message or level from the failure (an account's refusal read apart) calls the two halves.

    Args:
        describer: The host's describer.
        exc: What the call raised.
        logger: The caller's logger.
        where: What was being described, named if the describer itself raises.
        message: A %-format naming what failed; ``": <description>"`` is appended to it.
        args: Arguments for ``message``.
        level: The level to log at.

    Returns:
        The failure, as :func:`describe_failure` returned it.
    """
    failure = describe_failure(describer, exc, logger=logger, where=where)
    log_provider_failure(logger, failure, exc, message, *args, level=level)
    return failure


#: What each cut-short reason MEANS, and what to do about it — the vocabulary and its
#: wording as one object, so a member cannot exist without a sentence that describes it.
#: Keeping them apart is how a new reason acquires another member's diagnosis: the reader
#: is told "no choices at all" about a completion that had choices, which is the miscabled
#: cause :func:`describe_incomplete_completion` exists to prevent.
_CUT_SHORT_CAUSES: dict[StopReason, str] = {
    "max_tokens": (
        "the completion was TRUNCATED at the output cap (it reached the cap, or the "
        "provider returned finish_reason=length) — raise the cap for this call or choose a model whose "
        "output fits it; retrying the same request unchanged will truncate again"
    ),
    "content_filter": (
        "the completion was CUT SHORT by the provider's content filter (finish_reason="
        "content_filter) — the request needs changing, not retrying"
    ),
    "error": "the provider returned no choices at all, so nothing was generated to parse",
}

#: The :attr:`CompletionResult.stop_reason` values meaning the completion was CUT
#: SHORT rather than finished. Part of the port, not a host detail: a host maps its
#: provider's own finish reasons onto this vocabulary, and eval reads truncation off
#: nothing else. DERIVED from the causes above rather than listed beside them — a
#: member with no wording cannot exist, so the exhaustiveness this port needs holds by
#: construction instead of by a test remembering to check it.
INCOMPLETE_STOP_REASONS: frozenset[StopReason] = frozenset(_CUT_SHORT_CAUSES)


def describe_incomplete_completion(result: CompletionResult) -> str | None:
    """Say why a completion is unusable when the provider reported it was cut short.

    Returns ``None`` for a normally-finished call, so a caller can append this to its
    own parse/validation failure unconditionally and add a clause only when there is
    one.

    **Why callers need this.** A caller that reads only ``result.content`` sees an
    empty or half-written string and reports whatever its own parser said — "not
    valid JSON", true and causally useless. The truncation was already known one
    layer up and had to be recovered from the logs to diagnose a billed generation that
    failed. A parse failure downstream of a reported truncation IS the truncation.

    **Observed numbers, not an inferred mechanism.** An empty completion after a
    truncating finish is commonly reasoning tokens eating the whole budget, but this
    reports what the provider actually said — ``output_tokens`` and the
    ``reasoning_tokens`` subset — and says "unreported" where the provider reported
    no split, rather than asserting a cause it cannot see. ``reasoning_tokens`` is
    ``None`` for unknown and ``0`` for a reported zero; collapsing those would invent
    a fact.

    Args:
        result: The completed call to describe.

    Returns:
        A sentence naming the stop reason and the token counts behind it, or ``None``
        when the call finished normally.
    """
    if result.stop_reason not in INCOMPLETE_STOP_REASONS:
        return None

    if result.reasoning_tokens is None:
        split = f"{result.output_tokens} output token(s), reasoning split unreported"
    else:
        split = f"{result.output_tokens} output token(s), of which {result.reasoning_tokens} were reasoning"

    cause = _CUT_SHORT_CAUSES[result.stop_reason]

    empty = " The completion is empty, so nothing at all was returned to parse." if not result.content else ""
    return f"{cause} [{split}].{empty}"


def sum_optional_tokens(*values: int | None) -> int | None:
    """Total token counts that may be unreported, without inventing observations.

    Returns the sum of the values that were actually reported, or ``None`` when
    none of them were. A partially-reported total is the sum of what IS known: a
    turn with five reporting rounds and one silent round is better described by
    the five real measurements than by discarding them to signal incompleteness.
    This mirrors how cost is already accumulated (sum the observed, remember
    whether anything was).

    Args:
        *values: Token counts, each either an observation or ``None`` for
            "the provider reported nothing".

    Returns:
        The sum of the observed values, or ``None`` if none were observed.
    """
    observed = [v for v in values if v is not None]
    return sum(observed) if observed else None


_CODE_FENCE = re.compile(r"```(?:json)?\s*\n?(.*?)```", re.DOTALL)


def _strip_code_fence(text: str) -> str:
    """The body of the first fenced code block in ``text``, or ``text`` stripped when there is none.

    Shared by both parsers below, so a fence one of them reads the other reads too.
    """
    text = text.strip()
    block = _CODE_FENCE.search(text)
    return block.group(1).strip() if block else text


def extract_json(content: str) -> dict[str, Any]:
    """Extract a JSON object from provider output.

    Handles markdown code blocks and surrounding text via two strategies in
    order: direct parse, then ``json.JSONDecoder.raw_decode`` from the first
    ``{`` — which finds the first complete object, so leading and trailing prose
    are both tolerated.

    ``raw_decode`` subsumes an outermost-brace slice, which is why there is no third
    strategy: a JSON object is self-delimiting, so whenever
    ``text[first_brace:last_brace+1]`` parses it IS the complete object starting at
    ``first_brace``, and ``raw_decode`` at that offset returns the identical dict. It
    also handles what the slice cannot (an object followed by prose containing a
    brace, or by a second object). A tier nothing can reach would be an untraceable
    fallback.

    It raises rather than returning ``None`` because every caller would turn ``None``
    into an error on the next line.

    Top-level arrays / strings / numbers are rejected — callers expect an object.
    A bare object nested inside an array is still found, because ``raw_decode``
    starts at the first brace rather than at the start of the document.

    This is the one copy. It lives here, beside :func:`extract_json_array`, because
    eval's judge, proposer and analysis generator parse with it and the package must
    carry it when it leaves; the host's own LLM callers (its scoring stages and its
    other generators) import it from here, the allowed direction.

    Args:
        content: The raw model output to scrape an object out of.

    Returns:
        The parsed JSON object.

    Raises:
        ValueError: If no valid JSON object can be extracted.
    """
    text = _strip_code_fence(content)

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        # NOSILENT: one candidate parse of several; the function raises below when none succeeds
        pass

    try:
        decoder = json.JSONDecoder()
        idx = text.find("{")
        if idx != -1:
            data, _ = decoder.raw_decode(text, idx)
            if isinstance(data, dict):
                return data
    except json.JSONDecodeError, ValueError:
        # NOSILENT: one candidate parse of several; the function raises below when none succeeds
        pass

    # "LLM response", not "judge response": the analysis generator propagates this message
    # verbatim to its caller, so a judge-specific noun once reported an analysis-generation
    # failure as the judge's, a component that had not run. The noun has to fit every caller.
    raise ValueError(f"No valid JSON object found in LLM response ({len(content)} chars)")


def extract_json_array(text: str) -> list[dict[str, Any]]:
    """Extract a JSON array from provider output, handling markdown code blocks.

    Raises rather than returning ``[]`` when nothing parses: the one caller has
    already paid for the generation, and an unreadable batch reported as an empty
    one read as "the model had nothing to say" when the model had said something
    the parser could not read.

    A bare object is wrapped in a one-element list — a model asked for "an array
    of cases" that returns a single case returned one case.

    It reads what :func:`extract_json` reads, by the same two strategies: the fence
    is stripped by the same helper, and past a direct parse it calls ``raw_decode`` from
    the first ``[``, so an array followed by prose that itself holds a bracket (a
    trailing note citing "[1]") still parses rather than discarding a paid batch.

    It is the array half of the reply parsing every provider caller shares with
    :func:`extract_json`, and nothing in it knows what the array holds.

    Args:
        text: The raw model output to scrape an array out of.

    Returns:
        The parsed array.

    Raises:
        ValueError: No JSON array (or object) could be extracted from ``text``.
    """
    text = _strip_code_fence(text)

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            return [parsed]
        raise ValueError(f"the response is a JSON {type(parsed).__name__}, not an array of cases")
    except json.JSONDecodeError:
        # NOSILENT: one candidate parse of several; the function raises below when none succeeds
        pass

    start = text.find("[")
    if start != -1:
        try:
            parsed, _ = json.JSONDecoder().raw_decode(text, start)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            # NOSILENT: one candidate parse of several; the function raises below when none succeeds
            pass

    raise ValueError(f"no JSON array could be extracted from the response ({len(text)} chars): {text[:200]!r}")


__all__ = [
    "INCOMPLETE_STOP_REASONS",
    "JSON_OBJECT_RESPONSE_FORMAT",
    "BoundCompletionClient",
    "CompletionClient",
    "CompletionGenerator",
    "CompletionResult",
    "ProviderFailure",
    "ProviderFailureDescriber",
    "SimulatorLLM",
    "StopReason",
    "describe_failure",
    "describe_incomplete_completion",
    "extract_json",
    "extract_json_array",
    "sum_optional_tokens",
    "withhold_failure_detail",
]
