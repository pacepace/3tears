"""Shared bounded-retry-on-structured-LLM-output helper.

The same retry/logging shape was independently duplicated across
``extraction.generate_candidates``/``generate_row_candidates``,
``eval_loop._judge_candidates``/``_judge_row_candidates``,
``enrichment.run_enrichment``, and
an application-side query-matching module's own disambiguation calls --
seven near-identical copies of: build a structured-output model call, retry
on exception with linear backoff, degrade to ``None`` (never raise) after
every attempt fails, log a WARNING per failed attempt and one ERROR on total
failure.

Two faces over that one loop. :func:`bounded_retry_structured_call` keeps the
degrade-to-``None`` shape for callers whose honest reading of "no answer" is
"nothing here" and that store nothing on the strength of it.
:func:`bounded_retry_structured_call_or_raise` raises
:class:`StructuredCallExhaustedError` carrying the last cause, for a caller that
PERSISTS the outcome: ``enrichment.enrich_extraction`` stored ``{}`` for a pass
whose every attempt failed, which a reader could not tell from "the model had
nothing to add", and ``None`` is exactly as unable to say it failed. The same
turned out to hold for candidate generation, the judges, schema discovery and
direct extraction, so every caller in this package uses the raising form now
except ``challenge.classify_failed_page``, whose verdict is advisory: without it
a failed extraction is recorded as ``"failed"``, which is what it is.

Lives inside this package rather than in a new neutral top-level utilities
package, even though one of the seven callers was an unrelated query-matching
module: dependency flows one way (a consumer may import
``threetears.scrape.*``; this package imports nothing back), so the sharing
costs nothing structurally. That the consumer ends up depending on a package
literally named "scrape" for a generic retry helper is a real, acknowledged
oddity -- accepted as lower-friction than inventing a second package for one
shared helper.

This module depends only on ``threetears.models``/``threetears.observe`` and
the stdlib: no consuming application's config or store is reachable from
here, which is what keeps it importable by anything.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel
from threetears.models import LlmPurpose, create_chat_model
from threetears.observe import get_logger

__all__ = ["StructuredCallExhaustedError", "bounded_retry_structured_call", "bounded_retry_structured_call_or_raise"]

log = get_logger(__name__)


class StructuredCallExhaustedError(RuntimeError):
    """Every attempt of a bounded structured-output call failed.

    Raised by :func:`bounded_retry_structured_call_or_raise` for a caller that has to
    RECORD a failure rather than read it as "nothing here". The last attempt's own
    exception is :attr:`last_error` and is also chained as ``__cause__``.

    :param log_label: the call site's label, as given to the call
    :ptype log_label: str
    :param attempts: how many attempts were made, every one of which failed
    :ptype attempts: int
    :param model_id: the model that was invoked
    :ptype model_id: str
    :param last_error: the exception the final attempt raised
    :ptype last_error: Exception
    """

    def __init__(self, log_label: str, *, attempts: int, model_id: str, last_error: Exception) -> None:
        """Record which call failed, how often, and why the last attempt did.

        :param log_label: the call site's label, as given to the call
        :ptype log_label: str
        :param attempts: how many attempts were made, every one of which failed
        :ptype attempts: int
        :param model_id: the model that was invoked
        :ptype model_id: str
        :param last_error: the exception the final attempt raised
        :ptype last_error: Exception
        :return: nothing
        :rtype: None
        """
        self.log_label = log_label
        self.attempts = attempts
        self.model_id = model_id
        self.last_error = last_error
        super().__init__(
            f"{log_label}: all {attempts} attempts failed; last: {type(last_error).__name__}: {last_error}"
        )


async def bounded_retry_structured_call_or_raise[T: BaseModel](
    prompt: str | list[Any],
    response_model: type[T],
    *,
    model_id: str,
    api_key: str,
    purpose: LlmPurpose,
    temperature: float,
    timeout: float,
    attempts: int,
    backoff_seconds: float,
    log_label: str,
    is_acceptable: Callable[[T], bool] | None = None,
    provider: str | None = None,
) -> T:
    """Invoke a structured-output LLM call, retried on transient failure; raise when every attempt fails.

    Requests *response_model* via ``with_structured_output(..., method="json_schema")``
    -- deliberately not LangChain's default ``"function_calling"``, which
    proved materially less reliable in practice across the providers this
    package calls -- retrying on any exception
    with linear backoff (``backoff_seconds * (attempt + 1)``). A WARNING is logged per
    failed attempt, and exhaustion is logged here, once, at ERROR with the last cause --
    where it happens -- so a caller that catches :class:`StructuredCallExhaustedError`
    records the failure without logging it a second time. ``asyncio.CancelledError`` is
    not an attempt and propagates untouched.

    :param prompt: the fully-built prompt text for this call, OR a pre-built list of
        LangChain messages (e.g. one ``HumanMessage`` with multimodal image+text
        content blocks, as the vision extraction path uses) -- passed straight
        through to ``ainvoke()``, which accepts either shape natively
    :ptype prompt: str | list[Any]
    :param response_model: pydantic model the structured output is forced into
    :ptype response_model: type[T]
    :param model_id: the model to invoke
    :ptype model_id: str
    :param api_key: OpenRouter API key
    :ptype api_key: str
    :param purpose: ``LlmPurpose`` routing tag for this call
    :ptype purpose: LlmPurpose
    :param temperature: sampling temperature
    :ptype temperature: float
    :param timeout: per-attempt call timeout in seconds
    :ptype timeout: float
    :param attempts: bounded retry count for transient failures; at least one
    :ptype attempts: int
    :param backoff_seconds: base backoff between retries (multiplied by attempt number)
    :ptype backoff_seconds: float
    :param log_label: prefix identifying this call site in WARNING log lines
        (e.g. ``"scrape judge"``, ``"query_agent match disambiguation"``)
    :ptype log_label: str
    :param is_acceptable: optional post-parse validity check; a successfully
        parsed result this rejects is treated as retry-worthy on every attempt
        except the last (the last attempt's result is returned even if
        rejected -- something is better than nothing once retries are
        exhausted). ``None`` accepts any successfully parsed result.
    :ptype is_acceptable: Callable[[T], bool] | None
    :param provider: optional explicit provider override forwarded to
        ``create_chat_model`` (e.g. ``"openrouter"`` to route a model id not
        pre-registered under its natural provider -- see ``defaults.py``'s own
        registry); ``None`` uses the registry's own resolution
    :ptype provider: str | None
    :return: the validated result
    :rtype: T
    :raises ValueError: if *attempts* is less than one
    :raises StructuredCallExhaustedError: if every attempt raised
    """
    if attempts < 1:
        raise ValueError(f"{log_label}: attempts must be at least 1, got {attempts}")
    last_exc: Exception | None = None
    result: T | None = None
    for attempt in range(attempts):
        try:
            model = create_chat_model(
                model_id,
                api_key=api_key,
                purpose=purpose,
                temperature=temperature,
                timeout=timeout,
                provider=provider,
            )
            structured_model = model.with_structured_output(response_model, method="json_schema")
            parsed = await structured_model.ainvoke(prompt)
            candidate = parsed if isinstance(parsed, response_model) else response_model.model_validate(parsed)
            if is_acceptable is not None and not is_acceptable(candidate) and attempt < attempts - 1:
                log.warning(
                    "%s attempt %d/%d returned an unusable result -- retrying",
                    log_label,
                    attempt + 1,
                    attempts,
                    extra={"extra_data": {"model_id": model_id}},
                )
                await asyncio.sleep(backoff_seconds * (attempt + 1))
                continue
            result = candidate
            break
        except Exception as exc:  # noqa: BLE001 -- prawduct:allow prawduct/broad-except -- a
            # retryable attempt failure; the last one is raised below as the typed exhaustion
            # error's cause, never swallowed. CancelledError is a BaseException and is not caught.
            last_exc = exc
            log.warning(
                "%s attempt %d/%d failed: %s",
                log_label,
                attempt + 1,
                attempts,
                exc,
                extra={"extra_data": {"model_id": model_id}},
            )
            if attempt < attempts - 1:
                await asyncio.sleep(backoff_seconds * (attempt + 1))
    if result is None:
        # Reaching here means the final attempt raised: a rejected-but-parsed final result is
        # returned above, so ``last_exc`` is always set. The fallback keeps the type honest.
        failure = last_exc if last_exc is not None else RuntimeError("no attempt produced a result")
        log.error(
            "%s failed after %d attempts: %s: %s",
            log_label,
            attempts,
            type(failure).__name__,
            failure,
            extra={"extra_data": {"model_id": model_id}},
        )
        raise StructuredCallExhaustedError(
            log_label, attempts=attempts, model_id=model_id, last_error=failure
        ) from failure
    return result


async def bounded_retry_structured_call[T: BaseModel](
    prompt: str | list[Any],
    response_model: type[T],
    *,
    model_id: str,
    api_key: str,
    purpose: LlmPurpose,
    temperature: float,
    timeout: float,
    attempts: int,
    backoff_seconds: float,
    log_label: str,
    degraded_to: str,
    is_acceptable: Callable[[T], bool] | None = None,
    provider: str | None = None,
) -> T | None:
    """Invoke a structured-output LLM call, retried on transient failure; ``None`` when every attempt fails.

    :func:`bounded_retry_structured_call_or_raise` with exhaustion answered as ``None``;
    the raising form has already logged the failure at ERROR, so this adds only an INFO
    line naming the degrade. For a caller whose honest reading of "no answer" is
    "nothing here" (e.g. no candidates / no winner / no match) and that stores nothing
    on the strength of it. A caller that PERSISTS the outcome must use the raising form
    instead: ``None`` cannot say that it failed, and a stored "nothing" that was really a
    failure is indistinguishable from a real one.

    :param prompt: the prompt text, or a pre-built list of LangChain messages
    :ptype prompt: str | list[Any]
    :param response_model: pydantic model the structured output is forced into
    :ptype response_model: type[T]
    :param model_id: the model to invoke
    :ptype model_id: str
    :param api_key: OpenRouter API key
    :ptype api_key: str
    :param purpose: ``LlmPurpose`` routing tag for this call
    :ptype purpose: LlmPurpose
    :param temperature: sampling temperature
    :ptype temperature: float
    :param timeout: per-attempt call timeout in seconds
    :ptype timeout: float
    :param attempts: bounded retry count for transient failures; at least one
    :ptype attempts: int
    :param backoff_seconds: base backoff between retries (multiplied by attempt number)
    :ptype backoff_seconds: float
    :param log_label: prefix identifying this call site in WARNING/ERROR log lines
    :ptype log_label: str
    :param degraded_to: noun phrase describing the honest-empty degrade, used
        only in the INFO line naming it (e.g. ``"no candidates"``, ``"no match"``)
    :ptype degraded_to: str
    :param is_acceptable: optional post-parse validity check, as for the raising form
    :ptype is_acceptable: Callable[[T], bool] | None
    :param provider: optional explicit provider override forwarded to ``create_chat_model``
    :ptype provider: str | None
    :return: the validated result, or ``None`` after every attempt failed
    :rtype: T | None
    :raises ValueError: if *attempts* is less than one
    """
    result: T | None = None
    try:
        result = await bounded_retry_structured_call_or_raise(
            prompt,
            response_model,
            model_id=model_id,
            api_key=api_key,
            purpose=purpose,
            temperature=temperature,
            timeout=timeout,
            attempts=attempts,
            backoff_seconds=backoff_seconds,
            log_label=log_label,
            is_acceptable=is_acceptable,
            provider=provider,
        )
    except StructuredCallExhaustedError:
        log.info(
            "%s: treating the failure as %s",
            log_label,
            degraded_to,
            extra={"extra_data": {"model_id": model_id}},
        )
    return result
