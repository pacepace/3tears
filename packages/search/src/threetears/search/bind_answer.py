"""Bind for answers -- a grounded answer into what the caller consumes.

The answer twin of :mod:`threetears.search.bind`, and built the same way: one
path renders both registers, the prose a model reads and the structured
projection a program reads under
:data:`~threetears.search.contracts.metadata.ANSWER_METADATA_KEY`, so the two
cannot disagree (check 14). This module is that key's one construction site
(AD8): :func:`project_answer_metadata` and
:func:`project_answer_failure_metadata` own its schema version, and
``tests/enforcement/test_one_search_result_shape.py`` holds every other module
to reading it.

**The first line says whose words follow** (AR-G2): ``Answer from brave, 3
sources:``. A model handed the provider's synthesis must not mistake it for
retrieved text, so the attribution leads rather than trailing as a footnote --
and an answer citing nothing says so in that same first line (AR-A3), because
an uncited paragraph that reads as complete is the defect P8 exists for. The
text follows unchanged, then the sources, numbered in citation order, each
with :func:`~threetears.search.bind.result_facts`' line, so a reader can go
and read the pages rather than trust the paragraph (AR-G1).

**Nothing raises past here** (D10): every typed failure becomes a failed
result carrying its spend under the same key.

This module MUST NOT import ``agent-tools``; the ``web_answer`` builtin
consumes it, not the reverse.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from threetears.observe import contain, get_logger
from threetears.observe.text_block import one_line, under
from threetears.search.answer import answer
from threetears.search.bind import result_facts
from threetears.search.contracts import (
    ANSWER_METADATA_KEY,
    ANSWER_SCHEMA_VERSION,
    EGRESS_DIRECT,
    AnswerMetadata,
    AnswerProvider,
    AnswerRequest,
    BudgetPort,
    FailureRecord,
    GroundedAnswer,
    RateLimiterPort,
    SearchFailure,
    Spend,
    TransportFailed,
)

__all__ = ["RenderedAnswer", "bind_answer"]

_logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class RenderedAnswer:
    """One answer, rendered for a caller, in both registers.

    The :class:`~threetears.search.bind.RenderedSearch` shape: a seam value
    with exactly the four fields a ``ToolResult`` needs, so the tool builds
    one without this package importing it.
    """

    #: whether the answer completed. False for every typed failure; True for
    #: an answer that cites nothing, which says so (AR-A3).
    success: bool
    #: the prose a model reads.
    content: str
    #: the failure message, when there was one; None on success.
    error: str | None = None
    #: the structured projection, keyed by :data:`ANSWER_METADATA_KEY`
    #: (AD8). Present on failures too, carrying spend (SR-E3, D10).
    metadata: dict[str, Any] = field(default_factory=dict)


def _first_line(result: GroundedAnswer, provider: str) -> str:
    """Say whose synthesis follows, and how many sources it cites (AR-G2, AR-A3).

    :param result: the answer
    :ptype result: GroundedAnswer
    :param provider: the product name that wrote it
    :ptype provider: str
    :return: the line
    :rtype: str
    """
    count = len(result.sources)
    if not count:
        return f"Answer from {provider}, citing no sources -- nothing here is checked against a page:"
    return f"Answer from {provider}, {count} source{'s' if count != 1 else ''}:"


def render_answer_prose(result: GroundedAnswer, *, provider: str, zone: str | None = None) -> str:
    """Render an answer as the prose a model reads.

    :param result: the answer
    :ptype result: GroundedAnswer
    :param provider: the product name that wrote it, named in the first line
    :ptype provider: str
    :param zone: the IANA zone a source's time of day is shown in
    :ptype zone: str | None
    :return: the first line, the answer text, the numbered sources, and a
        closing note naming any notice or unhonoured criterion
    :rtype: str
    """
    sections = [_first_line(result, provider), result.text.strip()]
    if result.sources:
        lines = ["Sources:"]
        for position, source in enumerate(result.sources, 1):
            lines.append(f"{position}. {source.identity}")
            lines.append(f"   {result_facts(source, zone)}")
            if source.snippet:
                lines.extend(under(source.snippet, "   "))
        sections.append("\n".join(lines))
    notes = _footnotes(result)
    if notes:
        sections.append(notes)
    return "\n\n".join(section for section in sections if section)


def _footnotes(result: GroundedAnswer) -> str:
    """State what the answer could not do, for a reader who only gets prose.

    :param result: the answer
    :ptype result: GroundedAnswer
    :return: the note, or the empty string when there is nothing to say
    :rtype: str
    """
    notes = [
        f"- the {disposition.criterion_key} constraint was not applied"
        + (f": {one_line(disposition.detail)}" if disposition.detail else "")
        for disposition in result.dispositions
        if disposition.disposition in {"unsatisfied", "ignored-unknown"}
    ]
    notes.extend(f"- {notice}" for notice in (one_line(str(n)) for n in result.notices) if notice)
    if not notes:
        return ""
    return "What this answer could not do:\n" + "\n".join(notes)


def project_answer_metadata(question: str, result: GroundedAnswer) -> dict[str, Any]:
    """Project an answer under its named key (AD8, D22).

    The one place the payload is built, stamped with the schema version it
    is written at, so the shape cannot drift per caller.

    :param question: the question the answer answers
    :ptype question: str
    :param result: the answer
    :ptype result: GroundedAnswer
    :return: a one-key mapping ready to merge into ``ToolResult.metadata``, at
        :data:`ANSWER_SCHEMA_VERSION`
    :rtype: dict[str, Any]
    """
    projection = AnswerMetadata(
        schema_version=ANSWER_SCHEMA_VERSION, question=question, answer=result, spend=result.spend
    )
    return {ANSWER_METADATA_KEY: projection.to_metadata()}


def project_answer_failure_metadata(question: str, record: FailureRecord) -> dict[str, Any]:
    """Project a typed failure under the same named key (AD8, D10).

    The failure twin of :func:`project_answer_metadata`, exported for the
    reason :func:`threetears.search.bind.project_failure_metadata` is: a
    refusal decided before any provider was called projects through here
    rather than assembling the key itself.

    :param question: the question the failed call was answering
    :ptype question: str
    :param record: the failure's wire record, spend included
    :ptype record: FailureRecord
    :return: a one-key mapping ready to merge into ``ToolResult.metadata``
    :rtype: dict[str, Any]
    """
    projection = AnswerMetadata(
        schema_version=ANSWER_SCHEMA_VERSION, question=question, spend=record.spend, failure=record
    )
    return {ANSWER_METADATA_KEY: projection.to_metadata()}


def bind_grounded_answer(
    question: str, result: GroundedAnswer, *, provider: str, zone: str | None = None
) -> RenderedAnswer:
    """Render one successful answer in both registers.

    :param question: the question the answer answers
    :ptype question: str
    :param result: the answer
    :ptype result: GroundedAnswer
    :param provider: the product name that wrote it
    :ptype provider: str
    :param zone: the IANA zone a source's time of day is shown in
    :ptype zone: str | None
    :return: the rendered answer; ``success`` is True even with no sources
    :rtype: RenderedAnswer
    """
    return RenderedAnswer(
        success=True,
        content=render_answer_prose(result, provider=provider, zone=zone),
        error=None,
        metadata=project_answer_metadata(question, result),
    )


def bind_answer_failure(question: str, failure: SearchFailure) -> RenderedAnswer:
    """Render a typed failure as a failed result that still accounts (D10).

    :param question: the question the failed call was answering
    :ptype question: str
    :param failure: the typed failure
    :ptype failure: SearchFailure
    :return: the rendered failure, carrying spend under the named key
    :rtype: RenderedAnswer
    """
    record = failure.to_record()
    message = f"answer failed ({record.failure_class}): {record.message}"
    if record.remediation:
        message = f"{message}\n{record.remediation}"
    return RenderedAnswer(
        success=False,
        content=message,
        error=message,
        metadata=project_answer_failure_metadata(question, record),
    )


async def bind_answer(
    request: AnswerRequest,
    *,
    provider: AnswerProvider,
    timeout_seconds: float | None = None,
    budget: BudgetPort | None = None,
    limiter: RateLimiterPort | None = None,
    egress: str = EGRESS_DIRECT,
    zone: str | None = None,
) -> RenderedAnswer:
    """Ask for an answer and render it, whatever happens.

    The one answer entry point with the D10 guarantee: no exception leaves it.
    The ports pass straight through to :func:`threetears.search.answer.answer`,
    which enforces them; this renders.

    :param request: what the caller asked
    :ptype request: AnswerRequest
    :param provider: the answers adapter to ask
    :ptype provider: AnswerProvider
    :param timeout_seconds: bound for this call (SR-G2); None applies the
        answer call layer's default (SR-G1)
    :ptype timeout_seconds: float | None
    :param budget: the local refusal authority (D4, D5); None consults none
    :ptype budget: BudgetPort | None
    :param limiter: the pacing seam for ``(provider instance, egress)``;
        None leaves the call unpaced
    :ptype limiter: RateLimiterPort | None
    :param egress: which exit the provider's transport leaves by (D20)
    :ptype egress: str
    :param zone: the IANA zone the prose shows a source's time in
    :ptype zone: str | None
    :return: the rendered answer -- successful, or a failure carrying its spend
    :rtype: RenderedAnswer
    """
    try:
        result = await answer(
            request,
            provider=provider,
            timeout_seconds=timeout_seconds,
            budget=budget,
            limiter=limiter,
            egress=egress,
        )
    except SearchFailure as failure:
        _logger.warning(
            "answer failed against %s: %s: %s", provider.provider_instance, failure.failure_class, failure.message
        )
        return bind_answer_failure(request.question, failure)
    except Exception as exc:
        # the call layer already maps unmapped adapter exceptions, so reaching
        # this is a defect in it or in a provider property -- logged as one,
        # and still rendered, because D10's guarantee is unconditional.
        contain(
            exc,
            log=_logger,
            what=f"answer raised past the typed taxonomy against {provider.provider_instance}",
            level="error",
        )
        unmapped = TransportFailed(
            f"answer failed with an unmapped {type(exc).__name__}: {exc}",
            spend=Spend(),
            provider_instance=provider.provider_instance,
        )
        return bind_answer_failure(request.question, unmapped)
    return bind_grounded_answer(request.question, result, provider=provider.provider, zone=zone)
