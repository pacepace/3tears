"""Tavily JSON payloads, in the shape the search endpoint answers with.

Kept in one module so the adapter suite and any later conformance case drive
the same fixtures: a payload that drifts per test file is a payload that stops
describing the provider.

Field names and shapes follow Tavily's ``POST /search`` response: the envelope
carries ``query`` / ``answer`` / ``images`` / ``results`` / ``response_time`` /
``request_id``, and each result carries ``title`` / ``url`` / ``content`` /
``score`` / ``raw_content``, plus ``published_date`` for ``news`` results --
which Tavily reports in RFC 2822 rather than the ISO 8601 its general results
use.
"""

from __future__ import annotations

import json
from typing import Any

__all__ = [
    "CONTENT_RESULT",
    "MALFORMED_BODY",
    "NEWS_RESULT",
    "REQUEST_ID",
    "TWO_RESULTS",
    "TWO_RESULTS_BODY",
    "WEB_RESULT",
    "ZERO_RESULTS_BODY",
    "body",
]

#: Tavily's own id for one exchange, echoed on every candidate's provenance.
REQUEST_ID: str = "a1b2c3d4-0000-4000-8000-000000000001"

#: an ordinary web result: snippet only, no page text, no published date.
WEB_RESULT: dict[str, Any] = {
    "title": "Capybara",
    "url": "https://example.org/capybara",
    "content": "The capybara is the largest living rodent.",
    "score": 0.94,
    "raw_content": None,
}

#: a news result, with the RFC 2822 publication date Tavily's news topic
#: reports (its general results use ISO 8601 instead).
NEWS_RESULT: dict[str, Any] = {
    "title": "Capybara boom in the wetlands",
    "url": "https://news.example.net/capybara-boom",
    "content": "Wetland surveys counted a record number of capybaras.",
    "score": "0.5",
    "raw_content": None,
    "published_date": "Wed, 21 Aug 2024 07:00:00 GMT",
}

#: a result carrying the page text Tavily was asked to include -- the SR-A2
#: case, where the information arrived with the search and re-fetching it
#: would be paying twice.
CONTENT_RESULT: dict[str, Any] = {
    "title": "Capybara husbandry",
    "url": "https://example.org/husbandry",
    "content": "Housing, diet and water requirements.",
    "score": 0.81,
    "raw_content": "Capybaras need open water deep enough to submerge in.",
    "published_date": "2026-02-01T00:00:00",
}

#: the two ordinary results together, in provider order.
TWO_RESULTS: tuple[dict[str, Any], ...] = (WEB_RESULT, NEWS_RESULT)


def body(results: tuple[dict[str, Any], ...] = TWO_RESULTS, *, request_id: str | None = REQUEST_ID) -> bytes:
    """Render a Tavily response envelope around ``results``.

    :param results: the result objects to carry
    :ptype results: tuple[dict[str, Any], ...]
    :param request_id: Tavily's exchange id, or None to omit it
    :ptype request_id: str | None
    :return: the JSON body bytes
    :rtype: bytes
    """
    payload: dict[str, Any] = {
        "query": "capybara",
        "follow_up_questions": None,
        "answer": None,
        "images": [],
        "results": list(results),
        "response_time": 1.23,
    }
    if request_id is not None:
        payload["request_id"] = request_id
    return json.dumps(payload).encode("utf-8")


#: a successful two-result response.
TWO_RESULTS_BODY: bytes = body()

#: a successful response that found nothing (SR-J2's fixture).
ZERO_RESULTS_BODY: bytes = body(())

#: well-formed JSON that is not the shape the API promises -- no ``results``
#: list at all, which is what an error envelope or a gateway page produces.
MALFORMED_BODY: bytes = json.dumps({"query": "capybara", "detail": "unauthorized"}).encode("utf-8")
