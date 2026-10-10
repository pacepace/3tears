"""How the eval engine reads what a model provider hands back through the completion port.

The port itself — :class:`~threetears.evals.contracts.completion.CompletionClient`, the
:class:`~threetears.evals.contracts.completion.CompletionResult` it returns and the failure
describer — is :mod:`threetears.evals.contracts.completion`. Here is what every caller of it shares:

* which stop reasons mean a completion was cut short (:data:`INCOMPLETE_STOP_REASONS`) and how that
  is said (:func:`describe_incomplete_completion`);
* the JSON-object and JSON-array parsers;
* describing and logging a raised call without repeating the provider's payload
  (:func:`describe_failure`, :func:`withhold_failure_detail`).

``sum_optional_tokens`` is four lines of tri-state arithmetic, and the tri-state is the part that
must not drift: ``None`` is "not measured", not zero, and per-role cost attribution depends on
being able to say so.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from threetears.evals.contracts.completion import (
    CompletionResult,
    ProviderFailure,
    ProviderFailureDescriber,
    StopReason,
)
from threetears.observe import get_logger

log = get_logger(__name__)


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
    no count or no split, rather than asserting a cause it cannot see. Both are
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

    output = (
        "output token count unreported" if result.output_tokens is None else f"{result.output_tokens} output token(s)"
    )
    if result.reasoning_tokens is None:
        split = f"{output}, reasoning split unreported"
    else:
        split = f"{output}, of which {result.reasoning_tokens} were reasoning"

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
    "describe_failure",
    "describe_incomplete_completion",
    "extract_json",
    "extract_json_array",
    "sum_optional_tokens",
    "withhold_failure_detail",
]
