"""Bind -- candidates into what the caller actually consumes.

Two bindings ship, and one path produces both, because a face-specific
response shape is a regression by definition (check 14): the prose a model
reads, and the structured projection a program reads under
:data:`~threetears.search.contracts.metadata.SEARCH_RESULTS_METADATA_KEY`.
The same call renders both, so an agent tool, an MCP face and a direct
embedded caller are looking at one answer in two registers rather than two
answers that can disagree.

**The prose is a migration, not a rewrite.** Its shape is the one existing
callers already read -- a numbered list of ``title`` / ``URL:`` / snippet
blocks, ``No results found.`` when there are none -- so a model prompted
against the old tool sees no change. What changed is underneath: the lines
are rendered from typed candidates instead of from a provider's JSON, which
is what lets the structure ride alongside instead of being re-parsed out of
the text. Prose for a request with no criteria is byte-identical to the old
renderer's; the only addition is a closing note when a criterion could not
be honoured, and the old renderer had no criteria to fail to honour.

**Nothing raises past here** (D10). Every typed failure becomes a failed
result carrying its spend, because the far side of the tool envelope has no
way to receive an exception -- and because "search failed" with no
accounting is how a run overspends without anyone able to say where.

This module MUST NOT import ``agent-tools``: the ``TearsTool`` gutting
consumes these helpers, not the reverse. So the render returns a plain
frozen value with the four fields a ``ToolResult`` needs, and the tool
constructs its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final

from threetears.observe import get_logger
from threetears.search.call import search
from threetears.search.contracts import (
    SEARCH_RESULTS_METADATA_KEY,
    CandidateSet,
    SearchFailure,
    SearchProvider,
    SearchRequest,
    SearchResultsMetadata,
    Spend,
    TransportFailed,
)

__all__ = [
    "NO_RESULTS_PROSE",
    "PROSE_MAX_CANDIDATES",
    "RenderedSearch",
    "bind_candidate_set",
    "bind_failure",
    "bind_search",
    "project_metadata",
    "render_prose",
]

_logger = get_logger(__name__)

#: what the prose binding says when a search succeeded and found nothing.
#: The exact string the existing web-search tool returns, kept so a model
#: prompted against that tool reads the same words (SR-J2 -- and zero results
#: is a success, so this is the success prose).
NO_RESULTS_PROSE: Final[str] = "No results found."

#: candidates the prose binding renders. The structured projection carries
#: every candidate; prose is bounded because it is going into a context
#: window, and ten is the number the tool it replaces used.
PROSE_MAX_CANDIDATES: Final[int] = 10


@dataclass(frozen=True, slots=True)
class RenderedSearch:
    """One search, rendered for a caller, in both registers.

    A seam value rather than a contract type: it crosses from this package
    into whatever binds it -- an ``agent-tools`` ``ToolResult``, an MCP
    ``structuredContent`` pair, a direct embedded caller -- and never rides
    a payload itself. The four fields are exactly what a ``ToolResult``
    needs, so the tool constructs one without this package importing it.
    """

    #: whether the search completed. False for every typed failure; True for
    #: a search that ran and found nothing (SR-J2).
    success: bool
    #: the prose a model reads.
    content: str
    #: the failure message, when there was one; None on success.
    error: str | None = None
    #: the structured projection, keyed by
    #: :data:`SEARCH_RESULTS_METADATA_KEY` (D22). Present on failures too,
    #: carrying spend (SR-E3, D10).
    metadata: dict[str, Any] = field(default_factory=dict)


def render_prose(candidate_set: CandidateSet, *, max_candidates: int = PROSE_MAX_CANDIDATES) -> str:
    """Render candidates as the prose a model reads.

    :param candidate_set: what the search returned
    :ptype candidate_set: CandidateSet
    :param max_candidates: how many candidates to render
    :ptype max_candidates: int
    :return: the numbered list, or :data:`NO_RESULTS_PROSE` when the search
        found nothing. Unhonoured criteria and provider degradations are
        named in a closing note rather than left for the caller to notice
    :rtype: str
    """
    lines: list[str] = []
    for position, candidate in enumerate(candidate_set.candidates[:max_candidates], 1):
        lines.append(f"{position}. {candidate.title or 'Untitled'}")
        lines.append(f"   URL: {candidate.identity}")
        if candidate.snippet:
            lines.append(f"   {candidate.snippet}")
        lines.append("")
    body = "\n".join(lines).strip() if lines else NO_RESULTS_PROSE
    footnotes = _footnotes(candidate_set)
    return f"{body}\n\n{footnotes}" if footnotes else body


def _footnotes(candidate_set: CandidateSet) -> str:
    """State what the search could not do, for a reader who only gets prose.

    A model handed ten results has no way to know the domain filter was
    ignored or that two engines were down. Structure-reading consumers read
    the dispositions and notices; this is the same facts in the register the
    prose reader has (SR-B3, P8).

    :param candidate_set: what the search returned
    :ptype candidate_set: CandidateSet
    :return: the note, or the empty string when everything was honoured
    :rtype: str
    """
    unmet = [
        disposition
        for disposition in candidate_set.dispositions
        if disposition.disposition in {"unsatisfied", "ignored-unknown"}
    ]
    notes: list[str] = []
    for disposition in unmet:
        detail = f": {disposition.detail}" if disposition.detail else ""
        notes.append(f"- {disposition.criterion_key} was not applied ({disposition.disposition}){detail}")
    notes.extend(f"- {notice}" for notice in candidate_set.notices)
    if not notes:
        return ""
    return "Note on this result set:\n" + "\n".join(notes)


def project_metadata(query: str, candidate_set: CandidateSet) -> dict[str, Any]:
    """Project the structured result under its named key (D22).

    An explicit border projection, following ``ObjectHandle.to_metadata``:
    the payload is built by a named method that owns its schema version, not
    dumped at the call site where the shape would drift per caller.

    :param query: the query this result answers
    :ptype query: str
    :param candidate_set: what the search returned
    :ptype candidate_set: CandidateSet
    :return: a one-key mapping ready to merge into ``ToolResult.metadata``
    :rtype: dict[str, Any]
    """
    projection = SearchResultsMetadata.from_candidate_set(query=query, candidate_set=candidate_set)
    return {SEARCH_RESULTS_METADATA_KEY: projection.to_metadata()}


def bind_candidate_set(
    query: str, candidate_set: CandidateSet, *, max_candidates: int = PROSE_MAX_CANDIDATES
) -> RenderedSearch:
    """Render one successful search in both registers.

    :param query: the query this result answers
    :ptype query: str
    :param candidate_set: what the search returned
    :ptype candidate_set: CandidateSet
    :param max_candidates: how many candidates the prose renders
    :ptype max_candidates: int
    :return: the rendered result; ``success`` is True even for zero
        candidates (SR-J2)
    :rtype: RenderedSearch
    """
    return RenderedSearch(
        success=True,
        content=render_prose(candidate_set, max_candidates=max_candidates),
        error=None,
        metadata=project_metadata(query, candidate_set),
    )


def bind_failure(query: str, failure: SearchFailure) -> RenderedSearch:
    """Render a typed failure as a failed result that still accounts (D10).

    The prose names the failure class rather than prefixing a string an
    upstream layer would have to match on -- string-prefix error detection
    is one of the defects this package exists to retire. The remediation is
    included when the cause is known, because the reader of that sentence is
    usually the person who can fix it.

    :param query: the query the failed search was answering
    :ptype query: str
    :param failure: the typed failure
    :ptype failure: SearchFailure
    :return: the rendered failure, carrying spend under the named key
    :rtype: RenderedSearch
    """
    record = failure.to_record()
    message = f"search failed ({record.failure_class}): {record.message}"
    if record.remediation:
        message = f"{message}\n{record.remediation}"
    projection = SearchResultsMetadata.from_failure(query=query, failure=record)
    return RenderedSearch(
        success=False,
        content=message,
        error=message,
        metadata={SEARCH_RESULTS_METADATA_KEY: projection.to_metadata()},
    )


async def bind_search(
    request: SearchRequest,
    *,
    provider: SearchProvider,
    timeout_seconds: float | None = None,
    max_candidates: int = PROSE_MAX_CANDIDATES,
) -> RenderedSearch:
    """Run a search and render it, whatever happens.

    The one entry point with the D10 guarantee: no exception leaves it. A
    caller across the tool envelope cannot receive one, and a caller that
    can would still rather have the spend.

    :param request: what the caller asked for
    :ptype request: SearchRequest
    :param provider: the provider adapter to ask
    :ptype provider: SearchProvider
    :param timeout_seconds: bound for this call (SR-G2); None applies
        Call's default (SR-G1)
    :ptype timeout_seconds: float | None
    :param max_candidates: how many candidates the prose renders
    :ptype max_candidates: int
    :return: the rendered result -- successful, or a failure carrying its
        spend
    :rtype: RenderedSearch
    """
    try:
        result = await search(request, provider=provider, timeout_seconds=timeout_seconds)
    except SearchFailure as failure:
        _logger.warning(
            "search failed against %s: %s: %s", provider.provider_instance, failure.failure_class, failure.message
        )
        return bind_failure(request.query, failure)
    except Exception as exc:
        # Call already maps unmapped adapter exceptions, so reaching this is
        # a defect in Call or in a provider property. It is logged as one --
        # and still rendered, because D10's guarantee is unconditional: the
        # alternative is an exception crossing a wire that cannot carry it.
        _logger.exception("search raised past the typed taxonomy against %s", provider.provider_instance)
        unmapped = TransportFailed(
            f"search failed with an unmapped {type(exc).__name__}: {exc}",
            spend=Spend(),
            provider_instance=provider.provider_instance,
        )
        return bind_failure(request.query, unmapped)
    return bind_candidate_set(request.query, result, max_candidates=max_candidates)
