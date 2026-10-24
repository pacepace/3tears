"""Brave's Answers, through the injected streaming transport and nothing else.

One provider's API, mapped onto the answer contract: Brave's OpenAI-compatible
``/chat/completions`` endpoint in, a :class:`~threetears.search.contracts.answer.GroundedAnswer`
out -- the answer text with every tag removed, the cited pages as candidates,
the points where each is cited, and the spend the provider itself reported.
What this module deliberately does NOT do is open a client, read an
environment variable, or write any of the answer itself.

**It always streams** (``answers-requirements-design.md`` AD5). Brave returns
citations only on a streamed answer: the blocking response is bare text, which
is exactly the uncited paragraph AG2 refuses. So the adapter takes a
:class:`~threetears.search.contracts.transport.StreamingTransport` and nothing
else, and a host with no streaming transport cannot construct one.

**What arrives, as captured live** (design §4.2, "Verified live"). A stream of
SSE events, each ``data: {...}`` an OpenAI-style chunk whose
``choices[0].delta.content`` carries a piece of the answer, and a final
``data: [DONE]``. Inside the concatenated content, two kinds of tag:

- ``<citation>{json}</citation>`` -- ``start_index``, ``end_index``,
  ``number``, ``url``, ``favicon``, ``snippet``. The offsets are **code points
  into the text with every tag removed**, and Brave cites a zero-width point,
  ``start_index == end_index``, after the claim it supports;
- ``<usage>{json}</usage>`` -- the bill: ``X-Request-Queries``, the token
  counts, the per-dimension ``*-Cost`` keys and ``X-Request-Total-Cost``, the
  costs as JSON numbers.

The framing is parsed here, never in the transport (AR-F2): bytes become UTF-8
text through an incremental decoder, so a character split across chunks is
whole before it is read; lines become events; events' content runs through a
small tag state machine that buffers a tag until its closer arrives, wherever
the chunk and event boundaries fell. The live captures had every tag inside
one event; the provider promises nothing, so the parser assumes nothing.

**A tag is ``<name>`` followed by ``{``.** Answer prose can hold a ``<`` --
``x < y``, a Markdown ``<br>`` -- and treating every one as a tag would hold
the rest of the answer hostage to a closer that never comes. A tag name this
module does not read is removed from the text, kept verbatim on the answer's
provenance, and named in a notice (AR-A5): provider machinery is not answer
text, and dropping it silently would lose what the provider said (P2).

**Spend is the provider's report, never a computation** (AD10, AR-C1, AR-C2):
``money`` is ``X-Request-Total-Cost``, parsed as :class:`~decimal.Decimal` at
JSON-decode time so no float rounding touches it; ``provider_units`` is
``X-Request-Queries`` as ``brave:searches``; ``calls`` is 1. A stream that
ends without a ``<usage>`` tag records the call and no money, and says the
bill is unknown. A ``<usage>`` the adapter cannot read -- unparseable, or
without its cost keys -- is :class:`MalformedResponse`: a bill we cannot
state fails closed.

**Failures** (AD9, AR-D2, AR-E1, AR-E2) speak the search taxonomy and nothing
else. Brave answers every refusal in one envelope, ``{"type":
"ErrorResponse", "error": {"code", "detail", "status", "meta"}}``, and the
``code`` decides the class before the status does, because Brave's statuses
are coarse: an invalid key is a **422** ``SUBSCRIPTION_TOKEN_INVALID`` -- an
auth refusal, not a malformed request -- beside the 422 ``VALIDATION`` that
is one. A stream broken after its first byte is never retried (the
transport's rule) and carries ``calls=1``, because the provider had begun
answering and billed the search.
"""

from __future__ import annotations

import codecs
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Final

from pydantic import JsonValue

from threetears.observe import get_logger
from threetears.search.adapters._common import _as_float, _as_str, _DispositionPlan, attributed_failure, parsed_base_url
from threetears.search.contracts import (
    CRITERION_CARRIER,
    CRITERION_DOMAINS_EXCLUDE,
    CRITERION_DOMAINS_INCLUDE,
    CRITERION_LANGUAGE,
    CRITERION_MAX_RESULTS,
    CRITERION_MIN_RESOLUTION,
    CRITERION_RIGHTS_CLASS,
    CRITERION_TIME_RANGE,
    FIDELITY_SNIPPET,
    PRICING_PER_WEIGHTED_UNIT,
    PRODUCER_API_PROVIDER,
    Anchor,
    AnswerCapabilities,
    AnswerRequest,
    AuthFailed,
    Candidate,
    Criterion,
    GroundedAnswer,
    LocalCapExceeded,
    Locator,
    MalformedResponse,
    Provenance,
    QuotaExhausted,
    RateLimited,
    SearchFailure,
    Spend,
    StreamedResponse,
    StreamingTransport,
    TimedOut,
    TransportFailed,
    register_capabilities,
)

__all__ = ["BraveAnswersAdapter"]

_logger = get_logger(__name__)

#: product name; the key Brave's answer capabilities are registered under.
BRAVE_PROVIDER: Final[str] = "brave"

#: What Brave's answers meter in: the web searches an answer ran.
BRAVE_METERED_UNIT: Final[str] = "searches"

#: namespace for Brave-specific criteria.
BRAVE_NAMESPACE: Final[str] = "brave"

#: Brave's published API root. A compiled-in product constant, not an
#: environment default (D21, SR-K1, AR-B3).
BRAVE_API_BASE_URL: Final[str] = "https://api.search.brave.com/res/v1"

#: the model name a request sends. The response names the model that served
#: it (``brave-pro`` in the live capture), which provenance keeps.
BRAVE_ANSWERS_MODEL: Final[str] = "brave"

#: the country the provider's searches are run from; a two-letter code.
BRAVE_PARAM_COUNTRY: Final[str] = f"{BRAVE_NAMESPACE}:country"
#: the provider's safesearch level; one of :data:`BRAVE_SAFESEARCH_LEVELS`.
BRAVE_PARAM_SAFESEARCH: Final[str] = f"{BRAVE_NAMESPACE}:safesearch"
#: how much retrieved context the answer is written from; one of
#: :data:`BRAVE_SEARCH_CONTEXT_SIZES`.
BRAVE_PARAM_SEARCH_CONTEXT_SIZE: Final[str] = f"{BRAVE_NAMESPACE}:search-context-size"
#: the most tokens the answer may run to; a positive integer.
BRAVE_PARAM_MAX_COMPLETION_TOKENS: Final[str] = f"{BRAVE_NAMESPACE}:max-completion-tokens"

#: the safesearch levels Brave accepts.
BRAVE_SAFESEARCH_LEVELS: Final[tuple[str, ...]] = ("off", "moderate", "strict")

#: the search context sizes Brave accepts.
BRAVE_SEARCH_CONTEXT_SIZES: Final[tuple[str, ...]] = ("low", "medium", "high")

#: the shortest bound Brave says a single-mode answer needs, in seconds.
BRAVE_MINIMUM_TIMEOUT_SECONDS: Final[float] = 30.0

#: the bound an adapter applies when neither its host nor the call names one:
#: Brave's floor with room for a slow answer.
BRAVE_DEFAULT_TIMEOUT_SECONDS: Final[float] = 45.0

#: the cap on one answer's whole stream, when the host names none. A captured
#: twelve-citation answer is about 20 KB; this is a hundred times that.
BRAVE_DEFAULT_MAX_STREAM_BYTES: Final[int] = 2_000_000

#: the usage report's keys this adapter reads. Total cost and queries are the
#: bill (AD10), so a report missing either is unreadable (AR-E2).
_USAGE_TOTAL_COST: Final[str] = "X-Request-Total-Cost"
_USAGE_QUERIES: Final[str] = "X-Request-Queries"
_USAGE_TOKENS_IN: Final[str] = "X-Request-Tokens-In"
_USAGE_TOKENS_OUT: Final[str] = "X-Request-Tokens-Out"

#: the tag names this adapter reads; any other is an unknown tag (AR-A5).
_CITATION_TAG: Final[str] = "citation"
_USAGE_TAG: Final[str] = "usage"

#: the SSE data line that ends the stream.
_DONE: Final[str] = "[DONE]"

#: Brave's ``error.code`` for a key it does not recognise. Captured live: a 422.
_CODE_TOKEN_INVALID: Final[str] = "SUBSCRIPTION_TOKEN_INVALID"
#: Brave's ``error.code`` for a request body it could not validate. Captured live: a 422.
_CODE_VALIDATION: Final[str] = "VALIDATION"
#: Brave's ``error.code`` for the per-second pace. From Brave's published error
#: codes, not a capture: it could not be provoked without a burst on a paid key.
_CODE_RATE_LIMITED: Final[str] = "RATE_LIMITED"
#: Brave's ``error.code`` for the account's monthly credit limit. From Brave's
#: published error codes, not a capture: the limit could not be provoked, so
#: this mapping stays unverified until a live refusal is recorded.
_CODE_QUOTA_LIMITED: Final[str] = "QUOTA_LIMITED"

#: the most of an error body read for its envelope; an envelope is a few hundred bytes.
_ERROR_BODY_BYTES: Final[int] = 64_000

#: an opening tag's name; a tag is this followed by ``{``.
_TAG_OPENER: Final[re.Pattern[str]] = re.compile(r"<([a-z][a-z0-9_-]*)>")
#: what an opening tag looks like before its ``>`` has arrived.
_TAG_OPENING: Final[re.Pattern[str]] = re.compile(r"<[a-z][a-z0-9_-]*")

#: Quota exhaustion is account-level and unfixable by rephrasing (SR-D3).
BRAVE_QUOTA_REMEDIATION: Final[str] = (
    "Brave's account-level credit limit is spent. No rephrasing or retry clears it -- raise the "
    "limit or wait for the month to roll over. This is the provider's refusal, not a locally-configured "
    "cap (SR-D3): stop answering on this key rather than pacing against it."
)

#: The remediation for a body Brave could not validate. The adapter builds the
#: body itself, so the usual cause is a namespaced value Brave refused.
BRAVE_VALIDATION_REMEDIATION: Final[str] = (
    f"Brave refused the request body. This adapter builds it from the question and the request's "
    f"criteria, so check the '{BRAVE_PARAM_COUNTRY}', '{BRAVE_PARAM_SAFESEARCH}', "
    f"'{BRAVE_PARAM_SEARCH_CONTEXT_SIZE}' and '{BRAVE_PARAM_MAX_COMPLETION_TOKENS}' values this deployment sent."
)

#: Brave's answer capability declaration (AR-B2, SR-B4). Registered at import
#: under the ``answer`` kind, beside whatever search declaration Brave has.
BRAVE_ANSWERS_CAPABILITIES: Final[AnswerCapabilities] = AnswerCapabilities(
    provider=BRAVE_PROVIDER,
    modes=("single",),
    citations_in_modes=("single",),
    pushdown_criteria=(CRITERION_LANGUAGE,),
    unsatisfiable_criteria=(
        CRITERION_TIME_RANGE,
        CRITERION_DOMAINS_INCLUDE,
        CRITERION_DOMAINS_EXCLUDE,
        CRITERION_CARRIER,
        CRITERION_MIN_RESOLUTION,
        CRITERION_RIGHTS_CLASS,
        CRITERION_MAX_RESULTS,
    ),
    namespaced_parameters=(
        BRAVE_PARAM_COUNTRY,
        BRAVE_PARAM_MAX_COMPLETION_TOKENS,
        BRAVE_PARAM_SAFESEARCH,
        BRAVE_PARAM_SEARCH_CONTEXT_SIZE,
    ),
    pricing_model=PRICING_PER_WEIGHTED_UNIT,
    metered_unit=BRAVE_METERED_UNIT,
    minimum_timeout_seconds=BRAVE_MINIMUM_TIMEOUT_SECONDS,
)

register_capabilities(BRAVE_ANSWERS_CAPABILITIES)


def _as_count(value: object) -> int | None:
    """Read a provider count as a non-negative int, or ``None``.

    :param value: a usage or citation value
    :ptype value: object
    :return: the count, or None when it is not a non-negative integer
    :rtype: int | None
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _as_cost(value: object) -> Decimal | None:
    """Read a usage cost as a non-negative Decimal, or ``None``.

    The tag's JSON is decoded with ``parse_float=Decimal``, so a cost arrives
    here as the provider wrote it, never through a float (AR-C2).

    :param value: a usage value
    :ptype value: object
    :return: the amount, or None when it is not a non-negative number
    :rtype: Decimal | None
    """
    if isinstance(value, bool) or not isinstance(value, int | Decimal):
        return None
    amount = Decimal(value)
    return amount if amount >= 0 else None


def _provider_fact(value: object) -> str:
    """Render one provider value for :attr:`Provenance.provider_ids`, which holds strings.

    :param value: a JSON value from a citation
    :ptype value: object
    :return: the string itself, or the value as JSON
    :rtype: str
    """
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str)


class _EventReader:
    """Bytes in, SSE ``data`` payloads out, whatever the chunk boundaries.

    Decodes UTF-8 incrementally -- a code point split across two chunks is
    held until it is whole -- and frames lines into events by the SSE rules:
    ``data:`` lines join with a newline, a blank line ends the event, a line
    starting ``:`` is a comment, and any other field is not this adapter's.
    """

    def __init__(self) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")()
        self._partial = ""
        self._data: list[str] = []

    def feed(self, chunk: bytes, *, final: bool = False) -> list[str]:
        """Read one chunk, returning each event it completed.

        :param chunk: the bytes the socket delivered
        :ptype chunk: bytes
        :param final: whether this is the end of the stream, which completes
            a last event that no blank line closed
        :ptype final: bool
        :return: the completed events' data, in order
        :rtype: list[str]
        :raises UnicodeDecodeError: when the bytes are not UTF-8
        """
        lines = (self._partial + self._decoder.decode(chunk, final=final)).split("\n")
        self._partial = "" if final else lines.pop()
        events: list[str] = []
        for line in lines:
            self._line(line.removesuffix("\r"), events)
        if final and self._data:
            events.append("\n".join(self._data))
            self._data = []
        return events

    def _line(self, line: str, events: list[str]) -> None:
        """Fold one line into the event under construction.

        :param line: one line, its terminator removed
        :ptype line: str
        :param events: where a completed event's data is appended
        :ptype events: list[str]
        """
        if not line:
            if self._data:
                events.append("\n".join(self._data))
                self._data = []
            return
        if line.startswith(":"):
            return
        name, _, value = line.partition(":")
        if name == "data":
            self._data.append(value.removeprefix(" "))


@dataclass(frozen=True, slots=True)
class _Tag:
    """One closed ``<name>{json}</name>`` tag, as it arrived."""

    #: the tag's name.
    name: str
    #: what lay between the opener and the closer.
    body: str

    @property
    def raw(self) -> str:
        """The tag verbatim, for provenance and a failure's detail.

        :return: ``<name>body</name>``
        :rtype: str
        """
        return f"<{self.name}>{self.body}</{self.name}>"


class _TagReader:
    """The tag state machine: answer content in, text and closed tags out.

    Outside a tag, text passes through until a ``<``; there it waits until it
    can tell whether ``<name>{`` follows. Inside a tag it buffers until the
    matching closer. Nothing is decided on a chunk or event boundary, because
    neither means anything to the content.
    """

    def __init__(self) -> None:
        self._text: list[str] = []
        self._pending = ""
        self._open: str | None = None

    @property
    def text(self) -> str:
        """The answer text read so far, with every tag removed.

        :return: the text
        :rtype: str
        """
        return "".join(self._text)

    def feed(self, content: str) -> list[_Tag]:
        """Read a piece of the content, returning each tag it closed.

        :param content: one event's ``delta.content``
        :ptype content: str
        :return: the tags closed by this piece, in order
        :rtype: list[_Tag]
        """
        self._pending += content
        closed: list[_Tag] = []
        while self._pending:
            if self._open is not None:
                if not self._read_inside(self._open, closed):
                    break
            elif not self._read_outside():
                break
        return closed

    def finish(self) -> str | None:
        """End the content: anything held outside a tag is text.

        :return: the unterminated tag's raw content when the content ended
            inside one; None when it ended cleanly
        :rtype: str | None
        """
        if self._open is not None:
            return f"<{self._open}>{self._pending}"
        self._text.append(self._pending)
        self._pending = ""
        return None

    def _read_outside(self) -> bool:
        """Advance outside a tag.

        :return: whether progress was made; False when more content is needed
        :rtype: bool
        """
        at = self._pending.find("<")
        if at < 0:
            self._text.append(self._pending)
            self._pending = ""
            return True
        if at:
            self._text.append(self._pending[:at])
            self._pending = self._pending[at:]
        opener = _TAG_OPENER.match(self._pending)
        if opener is None:
            if self._pending == "<" or _TAG_OPENING.fullmatch(self._pending):
                return False
            # a '<' that cannot begin a tag is the answer's own text
            self._text.append("<")
            self._pending = self._pending[1:]
            return True
        after = opener.end()
        if after == len(self._pending):
            return False
        if self._pending[after] != "{":
            self._text.append("<")
            self._pending = self._pending[1:]
            return True
        self._open = opener.group(1)
        self._pending = self._pending[after:]
        return True

    def _read_inside(self, name: str, closed: list[_Tag]) -> bool:
        """Advance inside a tag, closing it when its closer has arrived.

        :param name: the open tag's name
        :ptype name: str
        :param closed: where a closed tag is appended
        :ptype closed: list[_Tag]
        :return: whether the tag closed; False when more content is needed
        :rtype: bool
        """
        closer = f"</{name}>"
        at = self._pending.find(closer)
        if at < 0:
            return False
        closed.append(_Tag(name=name, body=self._pending[:at]))
        self._pending = self._pending[at + len(closer) :]
        self._open = None
        return True


@dataclass(slots=True)
class _Usage:
    """The provider's own bill, read from its ``<usage>`` tag (AD10)."""

    #: ``X-Request-Total-Cost``, in USD.
    money: Decimal
    #: ``X-Request-Queries``: the searches the answer ran.
    searches: Decimal
    #: ``X-Request-Tokens-In``, when reported.
    tokens_in: int | None
    #: ``X-Request-Tokens-Out``, when reported.
    tokens_out: int | None


@dataclass(slots=True)
class _Reading:
    """Everything one stream has said so far. Never leaves this module."""

    question: str
    retrieved_at: datetime
    egress: str
    events: _EventReader = field(default_factory=_EventReader)
    tags: _TagReader = field(default_factory=_TagReader)
    sources: list[Candidate] = field(default_factory=list)
    #: source index by citation ``number``, so a source cited twice is one source.
    by_number: dict[str, int] = field(default_factory=dict)
    #: (start, end, source index) as the provider stated each citation point.
    points: list[tuple[int, int, int]] = field(default_factory=list)
    #: citations whose offsets were absent or not integers.
    unanchored: int = 0
    unknown: list[_Tag] = field(default_factory=list)
    usage: _Usage | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    provider_ids: dict[str, str] = field(default_factory=dict)
    received: int = 0
    done: bool = False


class _Plan(_DispositionPlan):
    """The request as Brave will receive it, with its honest dispositions.

    Not a contract type -- it never leaves this module. The dispositions list
    and :meth:`answer` come from :class:`_DispositionPlan`, shared with every
    other adapter's ``_Plan`` (SR-B2/SR-B3).
    """

    def __init__(self, body: dict[str, JsonValue]) -> None:
        """Start a plan from the body every question carries.

        :param body: the base wire body
        :ptype body: dict[str, JsonValue]
        """
        super().__init__()
        self.body = body


class BraveAnswersAdapter:
    """Brave's Answers API, behind the answer provider seam.

    Satisfies :class:`~threetears.search.contracts.answer.AnswerProvider`
    structurally. Construct one per configured *key*: two keys are two
    instances, paced and exhausted separately (D8, SR-N4).
    """

    def __init__(
        self,
        *,
        api_key: str,
        transport: StreamingTransport,
        provider_instance: str | None = None,
        base_url: str = BRAVE_API_BASE_URL,
        usd_per_search: Decimal | None = None,
        usd_per_million_tokens: Decimal | None = None,
        timeout_seconds: float = BRAVE_DEFAULT_TIMEOUT_SECONDS,
        max_stream_bytes: int = BRAVE_DEFAULT_MAX_STREAM_BYTES,
    ) -> None:
        """Bind to one configured Brave key.

        :param api_key: the Brave API key, resolved by the host. Required and
            never defaulted: this package reads no environment variable and no
            secret store (D21, SR-K1, AR-B3)
        :ptype api_key: str
        :param transport: the injected streaming transport every answer is read
            through (AD5). There is no blocking path: citations exist only on a
            stream
        :ptype transport: StreamingTransport
        :param provider_instance: name for this configured key, used in
            provenance and as the pacing key. Defaults to the base URL's host;
            a deployment running more than one key MUST name them
        :ptype provider_instance: str | None
        :param base_url: Brave's API root, or the gateway a deployment fronts
            it with. A deployment fact with a default, never a caller's (D21)
        :ptype base_url: str
        :param usd_per_search: what this account pays per search, if the host
            knows. With ``usd_per_million_tokens``, it prices the estimate a
            budget checks before the call (AR-C3); the bill itself is always
            the provider's report
        :ptype usd_per_search: Decimal | None
        :param usd_per_million_tokens: what this account pays per million
            tokens, in or out. Given with ``usd_per_search`` or not at all
        :ptype usd_per_million_tokens: Decimal | None
        :param timeout_seconds: the bound on a whole answer when the call names
            none. Brave's own floor is :data:`BRAVE_MINIMUM_TIMEOUT_SECONDS`
        :ptype timeout_seconds: float
        :param max_stream_bytes: the cap on one answer's whole stream (SR-G5)
        :ptype max_stream_bytes: int
        :raises ValueError: when ``api_key`` is empty, ``base_url`` is not an
            absolute http(s) URL, one rate is given without the other, a rate
            is negative, or a bound is not positive
        """
        if not api_key:
            raise ValueError(
                "BraveAnswersAdapter requires a host-supplied api_key: this package reads no environment "
                "variable and no secret store (D21, SR-K1)"
            )
        parsed = parsed_base_url(base_url)
        if (usd_per_search is None) != (usd_per_million_tokens is None):
            raise ValueError(
                "usd_per_search and usd_per_million_tokens are given together or not at all: half a rate card "
                "prices an estimate nobody stated"
            )
        for name, rate in (("usd_per_search", usd_per_search), ("usd_per_million_tokens", usd_per_million_tokens)):
            if rate is not None and rate < 0:
                raise ValueError(f"{name} cannot be negative, got {rate!r}")
        if timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be positive, got {timeout_seconds!r}")
        if max_stream_bytes < 1:
            raise ValueError(f"max_stream_bytes must be positive, got {max_stream_bytes!r}")
        self._api_key = api_key
        self._transport = transport
        self._base_url = base_url.rstrip("/")
        self._provider_instance = provider_instance or (parsed.hostname or self._base_url)
        self._usd_per_search = usd_per_search
        self._usd_per_million_tokens = usd_per_million_tokens
        self._timeout_seconds = timeout_seconds
        self._max_stream_bytes = max_stream_bytes

    @property
    def provider(self) -> str:
        """Product name.

        :return: :data:`BRAVE_PROVIDER`
        :rtype: str
        """
        return BRAVE_PROVIDER

    @property
    def provider_instance(self) -> str:
        """Name of the configured key this adapter answers with.

        :return: the instance name
        :rtype: str
        """
        return self._provider_instance

    @property
    def capabilities(self) -> AnswerCapabilities:
        """What Brave's answers can express (AR-B2).

        :return: :data:`BRAVE_ANSWERS_CAPABILITIES`
        :rtype: AnswerCapabilities
        """
        return BRAVE_ANSWERS_CAPABILITIES

    def estimate(self, request: AnswerRequest) -> Spend:
        """State what one answer is expected to cost, before it is sent (AR-C3).

        One call and one search, always. Money only when the host stated its
        rates: the search at its price, plus the completion at the token rate
        when the request caps it, since the cap is the most the reply can
        bill. The prompt's tokens -- most of a live answer's cost -- cannot be
        known before the provider has searched, so this is a floor.

        :param request: what the caller is about to ask
        :ptype request: AnswerRequest
        :return: the estimate to hand ``BudgetPort.check``
        :rtype: Spend
        """
        money = Decimal("0")
        if self._usd_per_search is not None and self._usd_per_million_tokens is not None:
            money = self._usd_per_search
            cap = next(
                (
                    _as_count(criterion.value)
                    for criterion in request.criteria
                    if criterion.key == BRAVE_PARAM_MAX_COMPLETION_TOKENS
                ),
                None,
            )
            if cap:
                money += self._usd_per_million_tokens * Decimal(cap) / Decimal(1_000_000)
        return Spend(
            money=money,
            calls=1,
            provider_units=Decimal(1),
            provider_unit=BRAVE_ANSWERS_CAPABILITIES.qualified_unit,
        )

    async def answer(self, request: AnswerRequest, *, timeout_seconds: float | None = None) -> GroundedAnswer:
        """Answer one question, with its sources, from one streamed reply.

        :param request: what the caller asked
        :ptype request: AnswerRequest
        :param timeout_seconds: bound on the whole answer (SR-G2); None
            applies this adapter's configured bound (SR-G1)
        :ptype timeout_seconds: float | None
        :return: the answer -- text, sources, anchors, spend, notices. Zero
            sources is a success that says so (AR-A3)
        :rtype: GroundedAnswer
        :raises SearchFailure: one of the typed classes, carrying spend
        """
        try:
            plan = self._plan(request)
            response = await self._open(plan, timeout_seconds=timeout_seconds)
            async with response:
                await self._raise_for_status(response)
                reading = await self._read(request, response)
            answer = self._answer(reading, plan, response)
        except SearchFailure as failure:
            attributed = self._attributed(failure)
            if attributed is failure:
                raise
            raise attributed from failure
        return answer

    # -- request planning ---------------------------------------------------

    def _plan(self, request: AnswerRequest) -> _Plan:
        """Map the question and criteria onto the wire body and dispositions.

        The question is sent as the one user message and wrapped in nothing
        (AR-K1): it is untrusted user content, and a template around it would
        be text this package wrote into the provider's prompt.

        :param request: the caller's request
        :ptype request: AnswerRequest
        :return: the plan the exchange reads
        :rtype: _Plan
        :raises LocalCapExceeded: when the question is empty -- refused here,
            before a billed answer is spent on nothing
        """
        question = request.question.strip()
        if not question:
            raise LocalCapExceeded(
                "brave requires a non-empty question",
                spend=Spend(),
                provider_instance=self._provider_instance,
                remediation="send a question with at least one non-whitespace character",
                scope="question-empty",
            )
        plan = _Plan(
            {
                "model": BRAVE_ANSWERS_MODEL,
                "messages": [{"role": "user", "content": request.question}],
                "stream": True,
                "enable_citations": True,
            }
        )
        for criterion in request.criteria:
            self._plan_criterion(plan, criterion)
        return plan

    def _plan_criterion(self, plan: _Plan, criterion: Criterion) -> None:
        """Fold one criterion into ``plan``.

        :param plan: the plan under construction
        :ptype plan: _Plan
        :param criterion: the criterion to handle
        :ptype criterion: Criterion
        """
        key = criterion.key
        if key == CRITERION_LANGUAGE:
            language = _as_str(criterion.value)
            if language is None:
                plan.answer(key, "unsatisfied", f"'language' takes a language tag; {criterion.value!r} is not one")
                return
            plan.body["language"] = language
            plan.answer(key, "pushdown", "sent as Brave's 'language'")
        elif key in BRAVE_ANSWERS_CAPABILITIES.unsatisfiable_criteria:
            plan.answer(
                key,
                "unsatisfied",
                f"Brave's answers take no '{key}' parameter, and a synthesised answer cannot be filtered after it "
                f"arrives; scope it in the question, or search instead",
            )
        elif key in BRAVE_ANSWERS_CAPABILITIES.namespaced_parameters:
            self._plan_namespaced(plan, criterion)
        else:
            plan.answer(key, "ignored-unknown", f"'{key}' is not a criterion this adapter recognises")

    def _plan_namespaced(self, plan: _Plan, criterion: Criterion) -> None:
        """Push a ``brave:``-namespaced parameter down to the provider.

        :param plan: the plan under construction
        :ptype plan: _Plan
        :param criterion: the namespaced criterion
        :ptype criterion: Criterion
        """
        key = criterion.key
        if key == BRAVE_PARAM_MAX_COMPLETION_TOKENS:
            tokens = _as_count(criterion.value)
            if not tokens:
                plan.answer(key, "unsatisfied", f"'{key}' takes a positive integer; {criterion.value!r} is not one")
                return
            plan.body["max_completion_tokens"] = tokens
            plan.answer(key, "pushdown", "sent as Brave's 'max_completion_tokens'")
            return
        value = str(criterion.value)
        if key == BRAVE_PARAM_COUNTRY:
            plan.body["country"] = value
            plan.answer(key, "pushdown", "sent as Brave's 'country'")
            return
        if key == BRAVE_PARAM_SAFESEARCH:
            if not self._accepts(plan, key, value, BRAVE_SAFESEARCH_LEVELS, "safesearch level"):
                return
            plan.body["safesearch"] = value
            plan.answer(key, "pushdown", "sent as Brave's 'safesearch'")
            return
        if not self._accepts(plan, key, value, BRAVE_SEARCH_CONTEXT_SIZES, "search context size"):
            return
        plan.body["web_search_options"] = {"search_context_size": value}
        plan.answer(key, "pushdown", "sent as Brave's 'web_search_options.search_context_size'")

    def _accepts(self, plan: _Plan, key: str, value: str, accepted: tuple[str, ...], what: str) -> bool:
        """Answer whether ``value`` is in ``accepted``, reporting it if not.

        :param plan: the plan under construction
        :ptype plan: _Plan
        :param key: the criterion key being answered for
        :ptype key: str
        :param value: the requested value
        :ptype value: str
        :param accepted: the values Brave accepts
        :ptype accepted: tuple[str, ...]
        :param what: what the value names, for the message
        :ptype what: str
        :return: whether the value is acceptable
        :rtype: bool
        """
        if value in accepted:
            return True
        plan.answer(key, "unsatisfied", f"'{value}' is not a Brave {what}; accepted: {', '.join(accepted)}")
        return False

    # -- the exchange -------------------------------------------------------

    async def _open(self, plan: _Plan, *, timeout_seconds: float | None) -> StreamedResponse:
        """Open the stream, mapping every failure before the first byte onto the taxonomy.

        :param plan: the planned request
        :ptype plan: _Plan
        :param timeout_seconds: per-call bound, or None for this adapter's own
        :ptype timeout_seconds: float | None
        :return: the open stream, head read
        :rtype: StreamedResponse
        :raises SearchFailure: whatever the transport raised, typed
        """
        headers = {"Accept": "text/event-stream", "X-Subscription-Token": self._api_key}
        try:
            return await self._transport.stream(
                "POST",
                f"{self._base_url}/chat/completions",
                headers=headers,
                json_body=plan.body,
                timeout_seconds=self._timeout_seconds if timeout_seconds is None else timeout_seconds,
                max_bytes=self._max_stream_bytes,
            )
        except SearchFailure as failure:
            # a transport that speaks the taxonomy knows the attempts, time and
            # bytes; keep its record and add which key.
            raise self._attributed(failure) from failure
        except TimeoutError as exc:
            raise TimedOut(
                f"brave instance {self._provider_instance} did not answer within the call's bound",
                spend=Spend(calls=0),
                provider_instance=self._provider_instance,
            ) from exc
        except Exception as exc:
            # the protocol lets an implementation raise whatever it likes, and an
            # adapter must not import an HTTP library to name those exceptions.
            # Unrecognised means "gave up at the transport" (SR-J1).
            raise TransportFailed(
                f"transport failed reaching brave instance {self._provider_instance}: {type(exc).__name__}: {exc}",
                spend=Spend(calls=0),
                provider_instance=self._provider_instance,
            ) from exc

    def _attributed(self, failure: SearchFailure) -> SearchFailure:
        """Re-stamp a failure with what only this adapter can attribute (D8/D20, SR-A3).

        :param failure: the failure about to leave this adapter
        :ptype failure: SearchFailure
        :return: the same failure class, fully attributed
        :rtype: SearchFailure
        """
        return attributed_failure(
            failure, provider_instance=self._provider_instance, egress_name=self._transport.egress_name
        )

    async def _raise_for_status(self, response: StreamedResponse) -> None:
        """Map a non-2xx answer onto the taxonomy, by Brave's ``error.code`` first.

        :param response: the open stream, head read
        :ptype response: StreamedResponse
        :raises SearchFailure: for every status outside 2xx
        """
        status = response.status_code
        if 200 <= status < 300:
            return
        body, unread = await self._error_body(response)
        code, detail = self._envelope(body)
        spend = Spend(calls=1, wall_clock_seconds=response.elapsed_seconds, bytes_transferred=len(body))
        message = f"brave instance {self._provider_instance} answered HTTP {status}"
        if code is not None:
            message += f" {code}"
        if detail:
            message += f": {detail}"
        if unread is not None:
            message += f" (its error body could not be read: {unread})"
        if code == _CODE_TOKEN_INVALID or (code is None and status in {401, 403, 407}):
            raise AuthFailed(
                message,
                spend=spend,
                provider_instance=self._provider_instance,
                remediation="Brave rejected the API key this deployment supplied",
            )
        if code == _CODE_QUOTA_LIMITED or (code is None and status == 402):
            # an account out of credit is an outage for every answer on the
            # key, not a per-question warning, so it is alertable.
            _logger.error(
                "brave instance %s reports its credit limit spent (HTTP %d); answers on this key will keep "
                "being refused until the limit is raised",
                self._provider_instance,
                status,
            )
            raise QuotaExhausted(
                message, spend=spend, provider_instance=self._provider_instance, remediation=BRAVE_QUOTA_REMEDIATION
            )
        if code == _CODE_RATE_LIMITED or status == 429:
            raise RateLimited(
                message,
                spend=spend,
                provider_instance=self._provider_instance,
                remediation="Brave is pacing this key; lower the configured rate for this instance",
                retry_after_seconds=_as_float(response.headers.get("retry-after")),
            )
        if code == _CODE_VALIDATION:
            raise TransportFailed(
                message,
                spend=spend,
                provider_instance=self._provider_instance,
                remediation=BRAVE_VALIDATION_REMEDIATION,
                status=status,
            )
        if status in {408, 504}:
            raise TimedOut(message, spend=spend, provider_instance=self._provider_instance)
        raise TransportFailed(message, spend=spend, provider_instance=self._provider_instance, status=status)

    async def _error_body(self, response: StreamedResponse) -> tuple[bytes, str | None]:
        """Read an error answer's body, which carries Brave's envelope.

        :param response: the open stream of an error answer
        :ptype response: StreamedResponse
        :return: the body read, and why the rest could not be when it could not
        :rtype: tuple[bytes, str | None]
        """
        body = bytearray()
        try:
            async for chunk in response:
                body += chunk
                if len(body) >= _ERROR_BODY_BYTES:
                    break
        except SearchFailure as failure:
            # the status already decides the failure; the body only adds its
            # detail, so a body that broke is named in the message, not raised.
            return bytes(body), f"{failure.failure_class}: {failure.message}"
        return bytes(body), None

    def _envelope(self, body: bytes) -> tuple[str | None, str | None]:
        """Read Brave's ``ErrorResponse`` envelope, when the body is one.

        The validation failure's ``meta.errors`` echo the request's input,
        which carries the question -- user content (D11) -- so only each
        error's location and message are kept.

        :param body: an error answer's body
        :ptype body: bytes
        :return: ``error.code`` and a detail line; None for what is absent
        :rtype: tuple[str | None, str | None]
        """
        try:
            payload = json.loads(body)
        # NOSILENT: a non-JSON error body has no envelope; the status alone decides
        except ValueError:
            return None, None
        error = payload.get("error") if isinstance(payload, dict) else None
        if not isinstance(error, dict):
            return None, None
        detail = _as_str(error.get("detail"))
        meta = error.get("meta")
        problems = meta.get("errors") if isinstance(meta, dict) else None
        if isinstance(problems, list):
            located = [
                f"{'.'.join(str(part) for part in problem.get('loc', []))}: {problem.get('msg')}"
                for problem in problems
                if isinstance(problem, dict)
            ]
            if located:
                detail = f"{detail or 'invalid request'} ({'; '.join(located)})"
        return _as_str(error.get("code")), detail

    # -- the stream -----------------------------------------------------------

    async def _read(self, request: AnswerRequest, response: StreamedResponse) -> _Reading:
        """Read the stream to ``[DONE]``, folding every event into a reading.

        A failure of the stream itself -- the cap, the deadline, a broken
        connection -- leaves as the transport's class with the bill this
        adapter can state: the call, and the provider's report if it had
        already arrived (AR-E1).

        :param request: the request being answered
        :ptype request: AnswerRequest
        :param response: the open, successful stream
        :ptype response: StreamedResponse
        :return: everything the stream said
        :rtype: _Reading
        :raises SearchFailure: a broken stream, or a malformed one
        """
        reading = _Reading(question=request.question, retrieved_at=datetime.now(UTC), egress=response.egress)
        chunks = aiter(response)
        while True:
            try:
                chunk = await anext(chunks)
            # NOSILENT: the body's end is the stream's normal end; whether it was whole is [DONE]'s to say, below
            except StopAsyncIteration:
                break
            except SearchFailure as failure:
                raise self._broken(failure, reading, response) from failure
            except Exception as exc:
                # a non-taxonomy transport's mid-body failure: the stream broke
                # after it began, which is what TransportFailed says (SR-J1).
                raise TransportFailed(
                    f"stream from brave instance {self._provider_instance} broke after "
                    f"{reading.received} byte(s): {type(exc).__name__}: {exc}{self._unbilled(reading)}",
                    spend=self._spend(reading, response),
                    provider_instance=self._provider_instance,
                ) from exc
            reading.received += len(chunk)
            self._events(reading, response, chunk, final=False)
        self._events(reading, response, b"", final=True)
        if not reading.done:
            raise MalformedResponse(
                f"brave instance {self._provider_instance} ended its stream without [DONE]; what arrived is not "
                f"the whole answer{self._unbilled(reading)}",
                spend=self._spend(reading, response),
                provider_instance=self._provider_instance,
            )
        unterminated = reading.tags.finish()
        if unterminated is not None:
            raise MalformedResponse(
                f"brave instance {self._provider_instance} ended its answer inside a tag: {unterminated!r}",
                spend=self._spend(reading, response),
                provider_instance=self._provider_instance,
            )
        return reading

    def _events(self, reading: _Reading, response: StreamedResponse, chunk: bytes, *, final: bool) -> None:
        """Feed one chunk through the event framing, and each event through the tag reader.

        :param reading: the reading so far
        :ptype reading: _Reading
        :param response: the stream, for the spend a failure carries
        :ptype response: StreamedResponse
        :param chunk: the bytes the socket delivered
        :ptype chunk: bytes
        :param final: whether the stream has ended
        :ptype final: bool
        :raises MalformedResponse: when the bytes are not UTF-8, an event is
            not a JSON object, or content follows ``[DONE]``
        """
        try:
            events = reading.events.feed(chunk, final=final)
        except UnicodeDecodeError as exc:
            raise MalformedResponse(
                f"brave instance {self._provider_instance} streamed bytes that are not UTF-8: {exc}",
                spend=self._spend(reading, response),
                provider_instance=self._provider_instance,
            ) from exc
        for data in events:
            if data == _DONE:
                reading.done = True
                continue
            if reading.done:
                raise MalformedResponse(
                    f"brave instance {self._provider_instance} sent an event after [DONE]: {data[:200]!r}",
                    spend=self._spend(reading, response),
                    provider_instance=self._provider_instance,
                )
            self._event(reading, response, data)

    def _event(self, reading: _Reading, response: StreamedResponse, data: str) -> None:
        """Fold one event's chunk into the reading.

        :param reading: the reading so far
        :ptype reading: _Reading
        :param response: the stream, for the spend a failure carries
        :ptype response: StreamedResponse
        :param data: one event's data
        :ptype data: str
        :raises MalformedResponse: when the event is not a JSON object, or a
            tag it closed cannot be read
        """
        try:
            event = json.loads(data)
        except ValueError as exc:
            raise MalformedResponse(
                f"brave instance {self._provider_instance} sent an event that is not JSON: {data[:200]!r}",
                spend=self._spend(reading, response),
                provider_instance=self._provider_instance,
            ) from exc
        if not isinstance(event, dict):
            raise MalformedResponse(
                f"brave instance {self._provider_instance} sent an event that is not a JSON object: {data[:200]!r}",
                spend=self._spend(reading, response),
                provider_instance=self._provider_instance,
            )
        for name in ("model", "id"):
            value = _as_str(event.get(name))
            if value is not None:
                reading.provider_ids.setdefault(name, value)
        usage = event.get("usage")
        if isinstance(usage, dict):
            reading.prompt_tokens = _as_count(usage.get("prompt_tokens"))
            reading.completion_tokens = _as_count(usage.get("completion_tokens"))
        choices = event.get("choices")
        for choice in choices if isinstance(choices, list) else []:
            delta = choice.get("delta") if isinstance(choice, dict) else None
            content = delta.get("content") if isinstance(delta, dict) else None
            if isinstance(content, str) and content:
                for tag in reading.tags.feed(content):
                    self._tag(reading, response, tag)

    def _tag(self, reading: _Reading, response: StreamedResponse, tag: _Tag) -> None:
        """Read one closed tag: a citation, the usage report, or an unknown tag.

        :param reading: the reading so far
        :ptype reading: _Reading
        :param response: the stream, for the spend a failure carries
        :ptype response: StreamedResponse
        :param tag: the tag, as it arrived
        :ptype tag: _Tag
        :raises MalformedResponse: when the tag's JSON does not parse as an
            object, a citation has no URL, or the usage report has no bill
            (AR-E2), carrying the raw tag
        """
        try:
            payload = json.loads(tag.body, parse_float=Decimal)
        except ValueError as exc:
            raise self._malformed_tag(reading, response, tag, "its JSON does not parse") from exc
        if not isinstance(payload, dict):
            raise self._malformed_tag(reading, response, tag, "its JSON is not an object")
        if tag.name == _CITATION_TAG:
            self._citation(reading, response, tag, payload)
        elif tag.name == _USAGE_TAG:
            reading.usage = self._usage(reading, response, tag, payload)
        else:
            reading.unknown.append(tag)

    def _malformed_tag(self, reading: _Reading, response: StreamedResponse, tag: _Tag, why: str) -> MalformedResponse:
        """The failure for a tag that cannot be read, carrying it raw (AR-E2).

        :param reading: the reading so far
        :ptype reading: _Reading
        :param response: the stream, for the spend
        :ptype response: StreamedResponse
        :param tag: the unreadable tag
        :ptype tag: _Tag
        :param why: what is wrong with it
        :ptype why: str
        :return: the failure to raise
        :rtype: MalformedResponse
        """
        return MalformedResponse(
            f"brave instance {self._provider_instance} sent a <{tag.name}> tag that cannot be read: {why}: "
            f"{tag.raw}{self._unbilled(reading)}",
            spend=self._spend(reading, response),
            provider_instance=self._provider_instance,
        )

    def _citation(
        self, reading: _Reading, response: StreamedResponse, tag: _Tag, payload: Mapping[str, object]
    ) -> None:
        """Turn one citation into a source and the point it is cited at.

        A source cited twice -- the same ``number`` -- is one source with two
        points. Every fact the citation carried beyond its URL and snippet
        stays on the source's provenance (AR-A2, P2).

        :param reading: the reading so far
        :ptype reading: _Reading
        :param response: the stream, for the spend a failure carries
        :ptype response: StreamedResponse
        :param tag: the citation tag, for a failure's detail
        :ptype tag: _Tag
        :param payload: the citation's decoded JSON
        :ptype payload: Mapping[str, object]
        :raises MalformedResponse: when the citation names no URL
        """
        url = _as_str(payload.get("url"))
        if url is None:
            raise self._malformed_tag(reading, response, tag, "it names no url")
        number = payload.get("number")
        key = None if number is None else _provider_fact(number)
        index = reading.by_number.get(key) if key is not None else None
        if index is None:
            index = len(reading.sources)
            reading.sources.append(self._source(reading, url, payload))
            if key is not None:
                reading.by_number[key] = index
        start, end = _as_count(payload.get("start_index")), _as_count(payload.get("end_index"))
        if start is None or end is None:
            reading.unanchored += 1
            return
        reading.points.append((start, end, index))

    def _source(self, reading: _Reading, url: str, payload: Mapping[str, object]) -> Candidate:
        """Build one cited source as a candidate whose page is unread (AD7, AR-A2).

        :param reading: the reading so far
        :ptype reading: _Reading
        :param url: the cited URL
        :ptype url: str
        :param payload: the citation's decoded JSON
        :ptype payload: Mapping[str, object]
        :return: the candidate
        :rtype: Candidate
        """
        facts = {
            name: _provider_fact(value)
            for name, value in payload.items()
            if name not in {"url", "snippet"} and value is not None
        }
        return Candidate(
            identity=url,
            locators=(Locator(url=url, rel="canonical"),),
            provenance=Provenance(
                query=reading.question,
                provider_instance=self._provider_instance,
                provider_ids=facts,
                retrieved_at=reading.retrieved_at,
                egress=reading.egress,
                producer=PRODUCER_API_PROVIDER,
            ),
            # the cited page's own text, extracted -- not model-written (design
            # §4.2), so it stands as a SERP snippet does (AR-G1).
            snippet=_as_str(payload.get("snippet")),
            fidelity_available=FIDELITY_SNIPPET,
            fidelity_achieved=FIDELITY_SNIPPET,
            content=None,
        )

    def _usage(self, reading: _Reading, response: StreamedResponse, tag: _Tag, payload: Mapping[str, object]) -> _Usage:
        """Read the provider's bill from its ``<usage>`` tag (AD10, AR-C2, AR-E2).

        :param reading: the reading so far
        :ptype reading: _Reading
        :param response: the stream, for the spend a failure carries
        :ptype response: StreamedResponse
        :param tag: the usage tag, for a failure's detail
        :ptype tag: _Tag
        :param payload: the tag's JSON, decoded with Decimal floats
        :ptype payload: Mapping[str, object]
        :return: the bill
        :rtype: _Usage
        :raises MalformedResponse: when the total cost or the query count is
            absent or not a non-negative number -- a bill we cannot state
        """
        money = _as_cost(payload.get(_USAGE_TOTAL_COST))
        searches = _as_cost(payload.get(_USAGE_QUERIES))
        if money is None or searches is None:
            missing = [
                name for name, value in ((_USAGE_TOTAL_COST, money), (_USAGE_QUERIES, searches)) if value is None
            ]
            raise self._malformed_tag(reading, response, tag, f"it states no readable {' or '.join(missing)}")
        return _Usage(
            money=money,
            searches=searches,
            tokens_in=_as_count(payload.get(_USAGE_TOKENS_IN)),
            tokens_out=_as_count(payload.get(_USAGE_TOKENS_OUT)),
        )

    # -- spend and the answer --------------------------------------------------

    def _spend(self, reading: _Reading, response: StreamedResponse) -> Spend:
        """What the call has consumed, by the provider's report where it has made one (AD10).

        ``calls`` is 1 once the stream is open: the provider has begun
        answering and billed the search (AR-E1). Money and searches are the
        ``<usage>`` report's, and zero until it arrives -- never estimated.

        :param reading: the reading so far
        :ptype reading: _Reading
        :param response: the stream, for its elapsed time
        :ptype response: StreamedResponse
        :return: the spend
        :rtype: Spend
        """
        usage = reading.usage
        if usage is None:
            return Spend(calls=1, wall_clock_seconds=response.elapsed_seconds, bytes_transferred=reading.received)
        return Spend(
            money=usage.money,
            currency="USD",
            wall_clock_seconds=response.elapsed_seconds,
            calls=1,
            provider_units=usage.searches,
            provider_unit=BRAVE_ANSWERS_CAPABILITIES.qualified_unit if usage.searches else None,
            bytes_transferred=reading.received,
        )

    def _unbilled(self, reading: _Reading) -> str:
        """The clause a failure's message adds when the provider's bill never arrived (AR-C1).

        :param reading: the reading so far
        :ptype reading: _Reading
        :return: the clause, or the empty string when the bill was read
        :rtype: str
        """
        if reading.usage is not None:
            return ""
        return "; brave had begun answering, so the call is billed, and its bill is unknown"

    def _broken(self, failure: SearchFailure, reading: _Reading, response: StreamedResponse) -> SearchFailure:
        """Re-state a failure that broke the stream mid-body with this call's bill (AR-E1).

        The class and the transport's own facts stay; the spend becomes the
        call's -- one billed call, and the provider's report if it arrived --
        with the transport's wall-clock and bytes.

        :param failure: what the transport raised while the body was read
        :ptype failure: SearchFailure
        :param reading: the reading so far
        :ptype reading: _Reading
        :param response: the stream
        :ptype response: StreamedResponse
        :return: the same failure class, carrying the call's spend
        :rtype: SearchFailure
        """
        spend = self._spend(reading, response).model_copy(
            update={
                "wall_clock_seconds": failure.spend.wall_clock_seconds or response.elapsed_seconds,
                "bytes_transferred": max(failure.spend.bytes_transferred, reading.received),
            }
        )
        record = failure.to_record()
        return record.model_copy(
            update={"spend": spend, "message": f"{record.message}{self._unbilled(reading)}"}
        ).to_failure()

    def _answer(self, reading: _Reading, plan: _Plan, response: StreamedResponse) -> GroundedAnswer:
        """Assemble the answer from a complete reading.

        :param reading: everything the stream said, ended by ``[DONE]``
        :ptype reading: _Reading
        :param plan: the plan, for its dispositions
        :ptype plan: _Plan
        :param response: the closed stream, for its final elapsed time
        :ptype response: StreamedResponse
        :return: the answer
        :rtype: GroundedAnswer
        """
        text = reading.tags.text
        anchors = [
            Anchor(start=start, end=end, source_index=index)
            for start, end, index in reading.points
            if start <= end <= len(text)
        ]
        unanchored = reading.unanchored + len(reading.points) - len(anchors)
        notices: list[str] = []
        if not reading.sources:
            notices.append(
                "brave cited no sources for this answer: its text is the provider's synthesis with nothing to "
                "check it against"
            )
        elif unanchored:
            notices.append(
                f"{unanchored} of brave's citations gave no usable offset into the answer text, so they anchor "
                f"nowhere; nothing was inferred from the wording"
            )
        provider_ids = dict(reading.provider_ids)
        for position, tag in enumerate(reading.unknown):
            provider_ids[f"unknown-tag-{position}"] = tag.raw
            notices.append(
                f"brave sent a <{tag.name}> tag this adapter does not read; it was removed from the text and kept "
                f"verbatim on the answer's provenance"
            )
        if reading.usage is None:
            notices.append(
                "brave reported no usage for this answer, so its bill is unknown: the spend records the call and "
                "no money"
            )
        prompt_tokens, completion_tokens = reading.prompt_tokens, reading.completion_tokens
        if reading.usage is not None:
            # the final event's usage object first; the bill's own counts when it sent none
            prompt_tokens = reading.usage.tokens_in if prompt_tokens is None else prompt_tokens
            completion_tokens = reading.usage.tokens_out if completion_tokens is None else completion_tokens
        return GroundedAnswer(
            text=text,
            sources=tuple(reading.sources),
            anchors=tuple(anchors),
            provenance=Provenance(
                query=reading.question,
                provider_instance=self._provider_instance,
                provider_ids=provider_ids,
                retrieved_at=reading.retrieved_at,
                egress=reading.egress,
                producer=PRODUCER_API_PROVIDER,
            ),
            spend=self._spend(reading, response),
            notices=tuple(notices),
            dispositions=tuple(plan.dispositions),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
