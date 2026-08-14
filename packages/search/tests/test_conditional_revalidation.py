"""D30 / SR-M4: spend a validator, and be told nothing changed.

The consumer holds the bytes (D7) and owns retention (D12), so it holds -- or
could hold -- the ``ETag`` / ``Last-Modified`` those bytes arrived with. Nothing
in the stack let it spend them, so every re-read of an unchanged page paid full
freight: a fetch, and on the scrape path a render *and* an LLM extraction.

**This is not caching, and the distinction is load-bearing.** Nothing is stored
anywhere in the capability, not even for the duration of a call. D14 is
untouched. The validators live where the bytes live, which is with the caller.

The tests here follow §5 of ``docs/search-task-01-conditional-revalidation.md``,
including the one it singles out: **no validator supplied must mean no
conditional headers on the wire.** An unconditional fetch that silently became
conditional would return ``304`` to a caller holding no copy -- that is, would
return nothing at all, successfully. That test reads the bytes the server
actually received rather than asserting a parameter was passed, because the
whole point is what crossed the socket.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from threetears.media.contracts import (
    EXTRACTION_STATUS_COMPLETE,
    EXTRACTION_STATUS_UNCHANGED,
)
from threetears.search.contracts import (
    EGRESS_DIRECT,
    Candidate,
    ContentSlot,
    Locator,
    Provenance,
    TransportResponse,
)
from threetears.search.extract import EXTRACTION_STATUS_FACET, extract
from threetears.search.standalone import StandaloneTransport
from threetears.search.testing.http_server import LocalHttpServer, Reply

pytest.importorskip("trafilatura", reason="conditional revalidation drives the real extractor")

_ETAG = '"v1-abcdef"'
_LAST_MODIFIED = "Wed, 13 Aug 2026 10:00:00 GMT"

_PAGE = (
    b"<html><head><title>Capybara</title></head><body>"
    b"<article><p>" + b"The capybara is the largest living rodent. " * 12 + b"</p></article>"
    b"</body></html>"
)
_PAGE_V2 = _PAGE.replace(b"largest living rodent", b"largest living rodent, still")


def _candidate(url: str, *, content: ContentSlot | None = None) -> Candidate:
    """A candidate for ``url``, optionally already carrying a copy."""
    return Candidate(
        identity=url,
        locators=(Locator(url=url),),
        provenance=Provenance(
            query=url,
            provider_instance="test",
            retrieved_at=datetime(2026, 8, 14, tzinfo=UTC),
        ),
        content=content,
    )


def _held_copy(*, etag: str | None = _ETAG, last_modified: str | None = None) -> ContentSlot:
    """The caller's existing copy, as it would have been stored."""
    return ContentSlot(
        text="The capybara is the largest living rodent.",
        origin="later-fetch",
        etag=etag,
        last_modified=last_modified,
    )


def _transport() -> StandaloneTransport:
    return StandaloneTransport(allowed_hosts=("127.0.0.1",), allow_private_addresses=True)


def _status(candidate: Candidate) -> object:
    return candidate.facets.get(EXTRACTION_STATUS_FACET)


class TestA304OverARealSocket:
    """The payoff path, end to end, through the real transport."""

    async def test_a_304_reports_unchanged_and_returns_the_copy_untouched(self) -> None:
        held = _held_copy()
        async with LocalHttpServer((Reply(status=304, body=b"", headers={"ETag": _ETAG}),)) as server:
            result = await extract(
                _candidate(f"{server.base_url}/page", content=held),
                transport=_transport(),
                respect_robots=False,
                revalidate=True,
            )

        assert _status(result) == EXTRACTION_STATUS_UNCHANGED
        assert result.content is not None
        assert result.content.text == held.text, "a 304 rebuilt the content slot from a body it never received"
        assert result.content.etag == held.etag

    async def test_the_conditional_header_actually_crossed_the_socket(self) -> None:
        """Read the request bytes, not the parameters."""
        async with LocalHttpServer((Reply(status=304, body=b""),)) as server:
            await extract(
                _candidate(f"{server.base_url}/page", content=_held_copy()),
                transport=_transport(),
                respect_robots=False,
                revalidate=True,
            )
            requests = list(server.requests)

        assert any("if-none-match" in request.lower() for request in requests), requests

    async def test_last_modified_is_echoed_back_verbatim(self) -> None:
        """The header's own string, unparsed -- re-rendering it risks a miss."""
        async with LocalHttpServer((Reply(status=304, body=b""),)) as server:
            await extract(
                _candidate(
                    f"{server.base_url}/page",
                    content=_held_copy(etag=None, last_modified=_LAST_MODIFIED),
                ),
                transport=_transport(),
                respect_robots=False,
                revalidate=True,
            )
            requests = list(server.requests)

        assert any(_LAST_MODIFIED in request for request in requests), requests

    async def test_origin_records_that_the_copy_was_revalidated(self) -> None:
        """SR-A2's 'where did this content come from' stays answerable."""
        async with LocalHttpServer((Reply(status=304, body=b""),)) as server:
            result = await extract(
                _candidate(f"{server.base_url}/page", content=_held_copy()),
                transport=_transport(),
                respect_robots=False,
                revalidate=True,
            )

        assert result.content is not None
        assert result.content.origin == "revalidated"


class TestA200AfterA304:
    """The page moved on: content replaced, new validators stored."""

    async def test_a_200_replaces_the_content_and_stores_the_new_validators(self) -> None:
        new_etag = '"v2-123456"'
        reply = Reply(
            status=200,
            body=_PAGE_V2,
            headers={"Content-Type": "text/html", "ETag": new_etag, "Last-Modified": _LAST_MODIFIED},
        )
        async with LocalHttpServer((reply,)) as server:
            result = await extract(
                _candidate(f"{server.base_url}/page", content=_held_copy()),
                transport=_transport(),
                respect_robots=False,
                revalidate=True,
            )

        assert _status(result) == EXTRACTION_STATUS_COMPLETE
        assert result.content is not None
        assert "still" in result.content.text
        assert result.content.etag == new_etag
        assert result.content.last_modified == _LAST_MODIFIED
        assert result.content.origin == "later-fetch"


class TestNoValidatorMeansNoConditionalRequest:
    """The regression §5 singles out, and the reason it matters.

    A fetch that silently became conditional would answer ``304`` to a caller
    holding nothing -- success, with no content, forever.
    """

    async def test_a_plain_fetch_sends_no_conditional_headers(self) -> None:
        reply = Reply(status=200, body=_PAGE, headers={"Content-Type": "text/html", "ETag": _ETAG})
        async with LocalHttpServer((reply,)) as server:
            await extract(
                _candidate(f"{server.base_url}/page"),
                transport=_transport(),
                respect_robots=False,
            )
            requests = list(server.requests)

        joined = " ".join(requests).lower()
        assert "if-none-match" not in joined, requests
        assert "if-modified-since" not in joined, requests

    async def test_a_plain_fetch_still_stores_the_validators_it_was_given(self) -> None:
        """Unconditional now, revalidatable later -- the point of storing them."""
        reply = Reply(status=200, body=_PAGE, headers={"Content-Type": "text/html", "ETag": _ETAG})
        async with LocalHttpServer((reply,)) as server:
            result = await extract(
                _candidate(f"{server.base_url}/page"),
                transport=_transport(),
                respect_robots=False,
            )

        assert result.content is not None
        assert result.content.etag == _ETAG

    async def test_revalidate_without_a_validator_makes_no_call_at_all(self) -> None:
        """SR-A2 keeps its early return: asked to CHECK, with nothing to check.

        Deliberately not an unconditional re-fetch -- that would spend exactly
        the bytes the request exists to avoid.
        """
        async with LocalHttpServer((Reply(status=200, body=_PAGE),)) as server:
            held = ContentSlot(text="already have this", origin="provider-response")
            result = await extract(
                _candidate(f"{server.base_url}/page", content=held),
                transport=_transport(),
                respect_robots=False,
                revalidate=True,
            )
            requests = list(server.requests)

        assert requests == [], "a candidate with no validator was fetched anyway"
        assert result.content is not None
        assert result.content.text == "already have this"


class TestOptInIsNotInferred:
    async def test_content_plus_validator_without_opt_in_makes_no_call(self) -> None:
        """The default is byte-for-byte today's behaviour (SR-A2).

        Inferring revalidation from a stored ``etag`` would turn every
        content-carrying candidate into a network call -- including Tavily's,
        whose content came with the search response and has nothing upstream to
        revalidate against.
        """
        async with LocalHttpServer((Reply(status=200, body=_PAGE),)) as server:
            result = await extract(
                _candidate(f"{server.base_url}/page", content=_held_copy()),
                transport=_transport(),
                respect_robots=False,
            )
            requests = list(server.requests)

        assert requests == []
        assert _status(result) is None


class TestAnUnconditional304IsNotSuccess:
    """A 304 nobody asked for is a server bug, not an answer.

    Treating it as ``unchanged`` would hand the caller a candidate with no
    content and no failure -- the worst of both readings.
    """

    async def test_an_unrequested_304_is_a_failure_not_unchanged(self) -> None:
        async with LocalHttpServer((Reply(status=304, body=b""),)) as server:
            result = await extract(
                _candidate(f"{server.base_url}/page"),
                transport=_transport(),
                respect_robots=False,
            )

        assert _status(result) != EXTRACTION_STATUS_UNCHANGED


class TestValidatorHygiene:
    """A blank validator is worse than none: echoed back it matches nothing."""

    async def test_a_blank_etag_is_stored_as_none(self) -> None:
        reply = Reply(status=200, body=_PAGE, headers={"Content-Type": "text/html", "ETag": "   "})
        async with LocalHttpServer((reply,)) as server:
            result = await extract(
                _candidate(f"{server.base_url}/page"),
                transport=_transport(),
                respect_robots=False,
            )

        assert result.content is not None
        assert result.content.etag is None

    async def test_an_absent_etag_is_stored_as_none(self) -> None:
        reply = Reply(status=200, body=_PAGE, headers={"Content-Type": "text/html"})
        async with LocalHttpServer((reply,)) as server:
            result = await extract(
                _candidate(f"{server.base_url}/page"),
                transport=_transport(),
                respect_robots=False,
            )

        assert result.content is not None
        assert result.content.etag is None
        assert result.content.last_modified is None


class _IgnoringHeavyFetcher:
    """A heavy fetcher that drops ``headers``, as an old implementer would.

    # parity-with: threetears.search.contracts.transport.HeavyFetcher
    """

    def __init__(self) -> None:
        self.seen_headers: object = "not-called"

    async def fetch_rendered(
        self,
        url: str,
        *,
        max_bytes: int,
        timeout_seconds: float | None = None,
        headers: object = None,
    ) -> TransportResponse:
        """Answer 200 with a body, whatever was asked -- the conformant default."""
        self.seen_headers = headers
        return TransportResponse(
            status_code=200,
            body=_PAGE,
            final_url=url,
            egress=EGRESS_DIRECT,
            elapsed_seconds=0.01,
            headers={"content-type": "text/html"},
        )


class TestAHeavyFetcherThatIgnoresHeaders:
    """§3.1's rule: ignoring ``headers`` is fine; reporting 304 anyway is not."""

    async def test_it_never_reports_unchanged(self) -> None:
        fetcher = _IgnoringHeavyFetcher()

        result = await extract(
            _candidate("https://example.test/page", content=_held_copy()),
            transport=_transport(),
            heavy_fetcher=fetcher,
            respect_robots=False,
            revalidate=True,
        )

        assert _status(result) == EXTRACTION_STATUS_COMPLETE
        assert _status(result) != EXTRACTION_STATUS_UNCHANGED

    async def test_the_validators_still_reached_it(self) -> None:
        """Additive with a default: the parameter arrives, honouring is optional."""
        fetcher = _IgnoringHeavyFetcher()

        await extract(
            _candidate("https://example.test/page", content=_held_copy()),
            transport=_transport(),
            heavy_fetcher=fetcher,
            respect_robots=False,
            revalidate=True,
        )

        assert fetcher.seen_headers == {"If-None-Match": _ETAG}
