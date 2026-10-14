"""Recorded pages for the fetch path: a :class:`FetchTransport` that answers from fixtures.

A consumer testing what its fetch tool says about a page -- a 403, a robots
file served as HTML, a PDF past its cap, a thin page's frame -- needs the
web's answers without the web. :class:`RecordedFetchTransport` answers each
URL from a :class:`RecordedPage` and keeps the two rules the real transport
keeps before a body is read, so a test cannot pass on a fixture the real
transport would have refused:

- the byte cap: a body past ``max_bytes`` is refused as
  :class:`~threetears.search.contracts.errors.LocalCapExceeded` with scope
  ``response-bytes``, never cut;
- the content-type gate: a successful answer whose declared type matches no
  allowed prefix is refused with scope ``content-type``.

A URL with no recording is answered 404 and listed on
:attr:`RecordedFetchTransport.unrecorded`, so a test that did not expect the
fetch fails on its own assertion about what was fetched, rather than on an
error from the fake.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from threetears.search.contracts import EGRESS_DIRECT, FetchTransport, LocalCapExceeded, Spend, TransportResponse

__all__ = ["RecordedFetchTransport", "RecordedPage"]

#: The scopes :class:`~threetears.search.standalone.StandaloneTransport` refuses under.
_RESPONSE_BYTES_SCOPE = "response-bytes"
_CONTENT_TYPE_SCOPE = "content-type"


@dataclass(frozen=True, slots=True)
class RecordedPage:
    """One recorded answer: a status, headers, a body -- or the failure the fetch raised."""

    #: HTTP status.
    status_code: int = 200
    #: the body as it came.
    body: bytes = b""
    #: the declared media type, with any parameters (``text/html; charset=utf-8``).
    content_type: str | None = "text/html"
    #: other response headers, lower-cased keys.
    headers: Mapping[str, str] = field(default_factory=dict)
    #: the URL that answered, when it is not the one asked for (a redirect).
    final_url: str | None = None
    #: raised instead of answering -- ``TimedOut``, ``TransportFailed``.
    failure: BaseException | None = None


class RecordedFetchTransport(FetchTransport):  # parity-with: threetears.search.contracts.FetchTransport
    """Answers each fetch from its recording, under the real transport's cap and gate."""

    def __init__(self, pages: Mapping[str, RecordedPage], *, egress_name: str = EGRESS_DIRECT) -> None:
        """Load the recordings.

        :param pages: the recorded answer for each URL
        :ptype pages: Mapping[str, RecordedPage]
        :param egress_name: the exit name every answer reports
        :ptype egress_name: str
        """
        self._pages = dict(pages)
        self._egress_name = egress_name
        #: every fetch, in order: ``{"url", "max_bytes", "allowed_content_types", "timeout_seconds",
        #: "headers"}``.
        self.calls: list[dict[str, object]] = []
        #: the URLs fetched that had no recording, each answered 404.
        self.unrecorded: list[str] = []

    @property
    def egress_name(self) -> str:
        """The exit name every answer reports.

        :return: the configured name
        :rtype: str
        """
        return self._egress_name

    @property
    def fetched(self) -> list[str]:
        """The URLs fetched, in order.

        :return: the URLs
        :rtype: list[str]
        """
        return [str(call["url"]) for call in self.calls]

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
        """Answer from the recording, refusing what the real transport refuses.

        :param method: HTTP method, unused
        :ptype method: str
        :param url: the URL asked for
        :ptype url: str
        :param headers: request headers, recorded
        :ptype headers: Mapping[str, str] | None
        :param max_bytes: this call's cap
        :ptype max_bytes: int
        :param allowed_content_types: the gate
        :ptype allowed_content_types: tuple[str, ...] | None
        :param timeout_seconds: this call's bound, recorded
        :ptype timeout_seconds: float | None
        :return: the recorded answer; a 404 for a URL with no recording
        :rtype: TransportResponse
        :raises LocalCapExceeded: past the cap, or outside the gate
        """
        del method
        self.calls.append(
            {
                "url": url,
                "max_bytes": max_bytes,
                "allowed_content_types": allowed_content_types,
                "timeout_seconds": timeout_seconds,
                "headers": dict(headers or {}),
            }
        )
        page = self._pages.get(url)
        if page is None:
            self.unrecorded.append(url)
            page = RecordedPage(status_code=404, body=b"", content_type="text/plain")
        if page.failure is not None:
            raise page.failure
        media_type = (page.content_type or "").split(";", 1)[0].strip().lower()
        if allowed_content_types and 200 <= page.status_code < 300:
            if not media_type or not any(media_type.startswith(p.lower()) for p in allowed_content_types):
                raise LocalCapExceeded(
                    f"response from {url} declared content type {media_type!r}, which this read does not accept",
                    spend=Spend(),
                    scope=_CONTENT_TYPE_SCOPE,
                )
        if len(page.body) > max_bytes:
            raise LocalCapExceeded(
                f"response from {url} is {len(page.body)} bytes, past this read's {max_bytes}-byte cap",
                spend=Spend(),
                scope=_RESPONSE_BYTES_SCOPE,
            )
        response_headers = {key.lower(): value for key, value in page.headers.items()}
        if page.content_type is not None:
            response_headers["content-type"] = page.content_type
        return TransportResponse(
            status_code=page.status_code,
            body=page.body,
            final_url=page.final_url or url,
            egress=self._egress_name,
            elapsed_seconds=0.01,
            headers=response_headers,
        )
