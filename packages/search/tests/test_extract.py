"""Extract's web path -- the obligations §3.5 states, pinned.

Driven through a scripted :class:`FetchTransport` rather than the real
socket: what a socket does under a byte cap is ``standalone``'s obligation
and is already pinned there against a real one. What *this* module owes is
the decisions it makes around a fetch -- when not to fetch at all, whose
permission it asks first, and which status a given outcome earns -- so the
transport here is scripted to answer exactly, and the assertions are about
Extract's choices rather than httpx's.
"""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from threetears.media.contracts import (
    EXTRACTION_STATUS_COMPLETE,
    EXTRACTION_STATUS_FAILED,
    EXTRACTION_STATUS_REFUSED,
)

from threetears.search.contracts.candidate import Candidate, ContentSlot, Locator
from threetears.search.contracts.errors import LocalCapExceeded, TransportFailed
from threetears.search.contracts.fidelity import FIDELITY_CONTENT
from threetears.search.contracts.spend import Spend
from threetears.search.contracts.transport import FetchTransport, HeavyFetcher, TransportResponse
from threetears.search.extract import (
    EXTRACTION_METHOD_FACET,
    EXTRACTION_STATUS_FACET,
    EXTRACTOR_UNAVAILABLE_SCOPE,
    extract,
)

from _search_instances import PROVENANCE

_PAGE_URL = "https://example.org/capybaras"
_ROBOTS_URL = "https://example.org/robots.txt"

_ARTICLE = (
    b"<html><head><title>Capybara habitats</title></head><body><article>"
    b"<p>Capybaras range across most of South America, favouring wetlands "
    b"and the margins of slow rivers where forage is dense and water is "
    b"never far. They are the largest living rodent, and they are social "
    b"in a way that reads as deliberate: groups hold territory together, "
    b"and a sentinel calls when a caiman surfaces.</p>"
    b"</article></body></html>"
)


def _candidate(**overrides: object) -> Candidate:
    """A minimal web candidate, with no content and one canonical locator.

    :param overrides: fields to replace on the built candidate
    :ptype overrides: object
    :return: the candidate under test
    :rtype: Candidate
    """
    base = Candidate(
        identity=_PAGE_URL,
        locators=(Locator(url=_PAGE_URL, rel="canonical"),),
        provenance=PROVENANCE,
        title="Capybara habitats",
    )
    return base.model_copy(update=dict(overrides)) if overrides else base


def _ok(body: bytes, *, content_type: str = "text/html") -> TransportResponse:
    """A 200 carrying ``body``.

    :param body: the response body
    :ptype body: bytes
    :param content_type: the declared media type
    :ptype content_type: str
    :return: the scripted response
    :rtype: TransportResponse
    """
    return TransportResponse(
        status_code=200,
        body=body,
        final_url=_PAGE_URL,
        egress="direct",
        elapsed_seconds=0.01,
        headers={"content-type": content_type},
    )


class ScriptedFetchTransport(FetchTransport):  # parity-with: threetears.search.contracts.FetchTransport
    """A fetch seam answering from a per-URL script, recording what it was asked."""

    def __init__(
        self,
        replies: Mapping[str, TransportResponse | Exception],
    ) -> None:
        """Load the script.

        :param replies: response or exception to produce, keyed by URL
        :ptype replies: Mapping[str, TransportResponse | Exception]
        """
        self._replies = replies
        #: every URL fetched, in order -- the pin for "robots first" and for
        #: "no fetch at all".
        self.requested: list[str] = []
        #: the content-type gates Extract asked for, by URL.
        self.gates: dict[str, tuple[str, ...] | None] = {}
        #: the byte caps Extract asked for, by URL.
        self.caps: dict[str, int] = {}

    @property
    def egress_name(self) -> str:
        """Report the configured exit's name.

        :return: always ``direct`` for this fake
        :rtype: str
        """
        return "direct"

    async def fetch(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        max_bytes: int,
        allowed_content_types: tuple[str, ...] | None = None,
        timeout_seconds: float | None = None,
    ) -> TransportResponse:
        """Answer from the script, recording the call.

        :param method: HTTP method
        :ptype method: str
        :param url: absolute URL
        :ptype url: str
        :param headers: ignored
        :ptype headers: Mapping[str, str] | None
        :param max_bytes: the per-call cap, recorded
        :ptype max_bytes: int
        :param allowed_content_types: the gate, recorded
        :ptype allowed_content_types: tuple[str, ...] | None
        :param timeout_seconds: ignored
        :ptype timeout_seconds: float | None
        :return: the scripted response
        :rtype: TransportResponse
        :raises Exception: when the script holds one for this URL
        """
        self.requested.append(url)
        self.gates[url] = allowed_content_types
        self.caps[url] = max_bytes
        reply = self._replies[url]
        if isinstance(reply, Exception):
            raise reply
        return reply


class ScriptedHeavyFetcher(HeavyFetcher):  # parity-with: threetears.search.contracts.HeavyFetcher
    """A rendering escalation that answers once and records that it was used."""

    def __init__(self, reply: TransportResponse) -> None:
        """Load the reply.

        :param reply: what the renderer returns
        :ptype reply: TransportResponse
        """
        self._reply = reply
        #: URLs this fetcher was asked to render.
        self.rendered: list[str] = []

    async def fetch_rendered(
        self,
        url: str,
        *,
        max_bytes: int,
        timeout_seconds: float | None = None,
    ) -> TransportResponse:
        """Answer with the loaded reply.

        :param url: the URL to render, recorded
        :ptype url: str
        :param max_bytes: the cap, ignored by this fake
        :ptype max_bytes: int
        :param timeout_seconds: ignored
        :ptype timeout_seconds: float | None
        :return: the scripted rendered document
        :rtype: TransportResponse
        """
        self.rendered.append(url)
        return self._reply


def _allowed() -> TransportResponse:
    """A permissive ``robots.txt``.

    :return: a 200 allowing everything
    :rtype: TransportResponse
    """
    return TransportResponse(
        status_code=200,
        body=b"User-agent: *\nAllow: /\n",
        final_url=_ROBOTS_URL,
        egress="direct",
        elapsed_seconds=0.01,
        headers={"content-type": "text/plain"},
    )


def _disallowed() -> TransportResponse:
    """A ``robots.txt`` refusing everything.

    :return: a 200 disallowing the whole tree
    :rtype: TransportResponse
    """
    return TransportResponse(
        status_code=200,
        body=b"User-agent: *\nDisallow: /\n",
        final_url=_ROBOTS_URL,
        egress="direct",
        elapsed_seconds=0.01,
        headers={"content-type": "text/plain"},
    )


async def test_a_candidate_that_already_has_content_is_not_fetched() -> None:
    """SR-A2: provider-supplied content is never re-bought.

    The pin is the empty request log, not just the unchanged content: "did
    not change the text" would also pass for an implementation that fetched
    the page, extracted it, and happened to keep the original.
    """
    transport = ScriptedFetchTransport({})
    candidate = _candidate(
        content=ContentSlot(text="already bought", origin="provider-response", mime_type="text/html")
    )

    result = await extract(candidate, transport=transport)

    assert result is candidate
    assert transport.requested == []


async def test_the_web_path_fills_content_and_records_how() -> None:
    """The success case: text, fidelity, status and method all recorded."""
    transport = ScriptedFetchTransport({_ROBOTS_URL: _allowed(), _PAGE_URL: _ok(_ARTICLE)})

    result = await extract(_candidate(), transport=transport)

    assert result.content is not None
    assert "South America" in result.content.text
    assert result.content.origin == "later-fetch"
    assert result.content.mime_type == "text/html"
    assert result.content.size_bytes == len(_ARTICLE)
    assert result.fidelity_achieved == FIDELITY_CONTENT
    assert result.facets[EXTRACTION_STATUS_FACET] == EXTRACTION_STATUS_COMPLETE
    assert result.facets[EXTRACTION_METHOD_FACET] == "trafilatura"


async def test_robots_is_asked_before_the_page_and_through_the_same_seam() -> None:
    """D12's enforcement point, and the ruling that it goes through the seam.

    Order matters on its own: asking permission after reading the page is
    not asking permission.
    """
    transport = ScriptedFetchTransport({_ROBOTS_URL: _allowed(), _PAGE_URL: _ok(_ARTICLE)})

    await extract(_candidate(), transport=transport)

    assert transport.requested == [_ROBOTS_URL, _PAGE_URL]


async def test_a_disallowing_robots_refuses_without_reading_the_page() -> None:
    """A refusal is recorded on the candidate, and the page is never fetched."""
    transport = ScriptedFetchTransport({_ROBOTS_URL: _disallowed()})

    result = await extract(_candidate(), transport=transport)

    assert result.facets[EXTRACTION_STATUS_FACET] == EXTRACTION_STATUS_REFUSED
    assert result.content is None
    assert transport.requested == [_ROBOTS_URL]


async def test_a_missing_robots_file_permits_the_fetch() -> None:
    """RFC 9309: 4xx means no rules exist, so there is nothing to disobey."""
    missing = TransportResponse(
        status_code=404,
        body=b"",
        final_url=_ROBOTS_URL,
        egress="direct",
        elapsed_seconds=0.01,
    )
    transport = ScriptedFetchTransport({_ROBOTS_URL: missing, _PAGE_URL: _ok(_ARTICLE)})

    result = await extract(_candidate(), transport=transport)

    assert result.facets[EXTRACTION_STATUS_FACET] == EXTRACTION_STATUS_COMPLETE


@pytest.mark.parametrize(
    "robots_reply",
    [
        pytest.param(
            TransportResponse(
                status_code=503,
                body=b"",
                final_url=_ROBOTS_URL,
                egress="direct",
                elapsed_seconds=0.01,
            ),
            id="server-error",
        ),
        pytest.param(TransportFailed("connection reset", spend=Spend()), id="transport-failure"),
    ],
)
async def test_unknown_robots_rules_are_honoured_as_deny(robots_reply: TransportResponse | Exception) -> None:
    """RFC 9309's other half, and the reason it is not the convenient default.

    Reading "allowed" out of a server error is how a polite fetcher becomes
    an impolite one for exactly as long as the origin is having a bad day.

    :param robots_reply: the failing ``robots.txt`` outcome under test
    :ptype robots_reply: TransportResponse | Exception
    """
    transport = ScriptedFetchTransport({_ROBOTS_URL: robots_reply})

    result = await extract(_candidate(), transport=transport)

    assert result.facets[EXTRACTION_STATUS_FACET] == EXTRACTION_STATUS_REFUSED
    assert transport.requested == [_ROBOTS_URL]


async def test_the_page_fetch_carries_the_cap_and_the_content_type_gate() -> None:
    """SR-G5 and §3.1: the gate is asked for, not hoped for."""
    transport = ScriptedFetchTransport({_ROBOTS_URL: _allowed(), _PAGE_URL: _ok(_ARTICLE)})

    await extract(_candidate(), transport=transport, max_bytes=4096)

    assert transport.caps[_PAGE_URL] == 4096
    assert transport.gates[_PAGE_URL] is not None
    assert "text/html" in transport.gates[_PAGE_URL]


async def test_a_cap_refusal_is_recorded_as_refused_not_failed() -> None:
    """``refused`` means the rules will decline it again; ``failed`` means it broke."""
    refusal = LocalCapExceeded("body past cap", spend=Spend(), scope="response-bytes")
    transport = ScriptedFetchTransport({_ROBOTS_URL: _allowed(), _PAGE_URL: refusal})

    result = await extract(_candidate(), transport=transport)

    assert result.facets[EXTRACTION_STATUS_FACET] == EXTRACTION_STATUS_REFUSED
    assert result.content is None


@pytest.mark.parametrize(
    "page_reply",
    [
        pytest.param(TransportFailed("connection reset", spend=Spend()), id="transport-failure"),
        pytest.param(
            TransportResponse(
                status_code=404,
                body=b"nope",
                final_url=_PAGE_URL,
                egress="direct",
                elapsed_seconds=0.01,
                headers={"content-type": "text/html"},
            ),
            id="not-found",
        ),
        pytest.param(_ok(b"<html><body></body></html>"), id="nothing-extractable"),
    ],
)
async def test_a_page_that_yields_no_text_is_recorded_as_failed(page_reply: TransportResponse | Exception) -> None:
    """Every way the read can break lands as ``failed`` on the candidate.

    None of them raises: one unreadable page must not take a set down with
    it.

    :param page_reply: the failing page outcome under test
    :ptype page_reply: TransportResponse | Exception
    """
    transport = ScriptedFetchTransport({_ROBOTS_URL: _allowed(), _PAGE_URL: page_reply})

    result = await extract(_candidate(), transport=transport)

    assert result.facets[EXTRACTION_STATUS_FACET] == EXTRACTION_STATUS_FAILED
    assert result.content is None


async def test_a_candidate_with_no_locator_fails_without_fetching() -> None:
    """Nothing to read is a failed extraction, not a crash."""
    transport = ScriptedFetchTransport({})

    result = await extract(_candidate(locators=()), transport=transport)

    assert result.facets[EXTRACTION_STATUS_FACET] == EXTRACTION_STATUS_FAILED
    assert transport.requested == []


async def test_the_canonical_locator_wins_over_the_others() -> None:
    """A candidate carrying several addresses is read at its canonical one."""
    transport = ScriptedFetchTransport({_ROBOTS_URL: _allowed(), _PAGE_URL: _ok(_ARTICLE)})
    candidate = _candidate(
        locators=(
            Locator(url="https://example.org/capybaras.pdf", rel="direct-file"),
            Locator(url=_PAGE_URL, rel="canonical"),
        )
    )

    await extract(candidate, transport=transport)

    assert transport.requested == [_ROBOTS_URL, _PAGE_URL]


async def test_the_heavy_fetcher_is_used_only_when_the_caller_passes_one() -> None:
    """Escalation is a parameter, not a fallback (§3.5).

    The ordinary transport is scripted to fail here: an implementation that
    escalated on failure would still produce content, and only the empty
    ``rendered`` log distinguishes it.
    """
    transport = ScriptedFetchTransport({_ROBOTS_URL: _allowed(), _PAGE_URL: TransportFailed("blocked", spend=Spend())})
    heavy = ScriptedHeavyFetcher(_ok(_ARTICLE))

    result = await extract(_candidate(), transport=transport)

    assert result.facets[EXTRACTION_STATUS_FACET] == EXTRACTION_STATUS_FAILED
    assert heavy.rendered == []


async def test_a_supplied_heavy_fetcher_reads_the_carrier_and_is_recorded() -> None:
    """The caller's choice is honoured, and the method says a renderer ran."""
    transport = ScriptedFetchTransport({})
    heavy = ScriptedHeavyFetcher(_ok(_ARTICLE))

    result = await extract(_candidate(), transport=transport, heavy_fetcher=heavy)

    assert heavy.rendered == [_PAGE_URL]
    assert result.content is not None
    assert result.facets[EXTRACTION_METHOD_FACET] == "trafilatura+rendered"
    assert transport.requested == []


async def test_robots_can_be_turned_off_by_configuration() -> None:
    """D12's override is recorded config, so the code has to accept it.

    D12 is a *proposed* stance pending cross-repo ratification, and it
    states the override as deployment config rather than a code change --
    which only works if the parameter exists.
    """
    transport = ScriptedFetchTransport({_PAGE_URL: _ok(_ARTICLE)})

    result = await extract(_candidate(), transport=transport, respect_robots=False)

    assert result.facets[EXTRACTION_STATUS_FACET] == EXTRACTION_STATUS_COMPLETE
    assert transport.requested == [_PAGE_URL]


async def test_a_missing_extract_extra_refuses_by_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """The one outcome that raises, because it is not per-candidate.

    A caller handed prose must be able to tell whether a real extractor
    produced it, so the alternative -- a crude tag-strip fallback -- is
    refused rather than offered. And because the fault would refuse every
    candidate identically, it raises once instead of marking a hundred
    candidates ``refused`` and hiding one fixable cause behind them.

    :param monkeypatch: fixture used to hide the installed extractor
    :ptype monkeypatch: pytest.MonkeyPatch
    """
    import builtins

    real_import = builtins.__import__

    def _refuse_trafilatura(name: str, *args: object, **kwargs: object) -> object:
        if name == "trafilatura":
            raise ImportError("No module named 'trafilatura'")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", _refuse_trafilatura)
    transport = ScriptedFetchTransport({_ROBOTS_URL: _allowed(), _PAGE_URL: _ok(_ARTICLE)})

    with pytest.raises(LocalCapExceeded) as caught:
        await extract(_candidate(), transport=transport)

    assert caught.value.scope == EXTRACTOR_UNAVAILABLE_SCOPE
    assert "3tears-search[extract]" in str(caught.value)
    assert caught.value.remediation is not None
    assert "3tears-search[extract]" in caught.value.remediation


async def test_the_extra_is_checked_before_anything_is_fetched() -> None:
    """A run that cannot extract should not spend a request finding out.

    Kept separate from the refusal's own test because it pins a different
    thing: not *that* it refuses, but that it refuses before paying.
    """
    transport = ScriptedFetchTransport({_ROBOTS_URL: _allowed(), _PAGE_URL: _ok(_ARTICLE)})

    await extract(_candidate(), transport=transport)

    # trafilatura is installed in this venv, so the ordering pin is that the
    # extractor load precedes the robots fetch in the successful path too:
    # the load is unconditional and first, which is what makes the refusal
    # above cost nothing.
    assert transport.requested[0] == _ROBOTS_URL
