"""Shared bounded-retry-on-structured-LLM-output helper (backlog SCR-K7M3).

The same retry/logging shape was independently duplicated across
``extraction.generate_candidates``/``generate_row_candidates``,
``eval_loop._judge_candidates``/``_judge_row_candidates``,
``enrichment.run_enrichment``, and
``query_agent/matching.py``'s ``_invoke_match_disambiguation``/
``_invoke_multi_match_disambiguation`` -- seven near-identical copies of:
build a structured-output model call, retry on exception with linear
backoff, degrade to ``None`` (never raise) after every attempt fails, log a
WARNING per failed attempt and one ERROR on total failure.

Lives inside ``src/faidh/scrape/`` (not a new top-level module) so it stays
covered by ``tests/enforcement/test_scrape_no_faidh_imports.py``'s existing
zero-faidh-imports guard for free -- the only location in this repo a
faidh-side module (``query_agent/matching.py``) can import a shared helper
from without ``scrape/`` importing back into ``faidh.*`` (the enforcement
test only checks the ``scrape/`` -> ``faidh.*`` direction, matching
``WarnActPlugin``'s existing precedent of importing ``threetears.scrape.*``).
This creates a real, acknowledged cross-domain dependency (query-matching
code depending on a package literally named "scrape") -- accepted as the
lower-friction choice over inventing a second top-level neutral utilities
package for one shared helper (YAGNI).

Zero faidh imports (see ``scrape/__init__.py``).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel
from threetears.models import LlmPurpose, create_chat_model
from threetears.observe import get_logger

__all__ = ["bounded_retry_structured_call"]

log = get_logger(__name__)


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
    """Invoke a structured-output LLM call, retried on transient failure. Never raises.

    Requests *response_model* via ``with_structured_output(..., method="json_schema")``
    (this codebase's proven-reliable posture over the default
    ``"function_calling"`` -- see ``learnings.md``), retrying on any exception
    with linear backoff (``backoff_seconds * (attempt + 1)``). Degrades to
    ``None`` only after every attempt fails -- callers treat ``None`` as an
    honest "nothing here" result (e.g. no candidates / no winner / no match),
    never as a crash.

    :param prompt: the fully-built prompt text for this call, OR a pre-built list of
        LangChain messages (e.g. one ``HumanMessage`` with multimodal image+text
        content blocks, scrape-task-06's vision extraction path) -- passed straight
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
    :param attempts: bounded retry count for transient failures
    :ptype attempts: int
    :param backoff_seconds: base backoff between retries (multiplied by attempt number)
    :ptype backoff_seconds: float
    :param log_label: prefix identifying this call site in WARNING/ERROR log lines
        (e.g. ``"scrape judge"``, ``"query_agent match disambiguation"``)
    :ptype log_label: str
    :param degraded_to: noun phrase describing the honest-empty degrade, used
        only in the final ERROR log line (e.g. ``"no candidates"``, ``"no match"``)
    :ptype degraded_to: str
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
    :return: the validated result, or ``None`` after every attempt failed
    :rtype: T | None
    """
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
                result = None
                await asyncio.sleep(backoff_seconds * (attempt + 1))
                continue
            result = candidate
            break
        except Exception as exc:  # noqa: BLE001 -- prawduct:allow prawduct/broad-except -- honest-None
            # contract shared by every caller: a failed structured-output call must never
            # raise into the caller's pipeline, only degrade to "nothing here" (see this
            # module's own docstring).
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
    if result is None and last_exc is not None:
        log.error(
            "%s degraded after %d attempts -- treating as %s: %s",
            log_label,
            attempts,
            degraded_to,
            last_exc,
            extra={"extra_data": {"model_id": model_id}},
        )
    return result
