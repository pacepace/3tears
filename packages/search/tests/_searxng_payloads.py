"""SearXNG JSON payloads, in the shape a real instance answers with.

Kept in one module so the adapter suite, the conformance case and the Call /
Bind suites all drive the same fixtures: a payload that drifts per test file
is a payload that stops describing the provider.

Field names and shapes follow SearXNG's ``format=json`` response: results
carry ``url`` / ``title`` / ``content`` / ``engine`` / ``engines`` /
``positions`` / ``score`` / ``category`` / ``template``, image results add
``img_src`` / ``thumbnail_src`` / ``resolution`` / ``img_format``, and the
envelope carries ``unresponsive_engines`` for engines that did not answer.
"""

from __future__ import annotations

import json
from typing import Any

__all__ = [
    "IMAGE_RESULT",
    "MALFORMED_BODY",
    "TWO_RESULTS",
    "TWO_RESULTS_BODY",
    "WEB_RESULT",
    "ZERO_RESULTS_BODY",
    "body",
]

#: an ordinary web result, with a naive published date (which is what
#: SearXNG's engines commonly report).
WEB_RESULT: dict[str, Any] = {
    "url": "https://example.org/capybara",
    "title": "Capybara",
    "content": "The capybara is the largest living rodent.",
    "engine": "duckduckgo",
    "engines": ["duckduckgo", "brave"],
    "positions": [1, 3],
    "score": 2.5,
    "category": "general",
    "template": "default.html",
    "publishedDate": "2026-02-01T00:00:00",
}

#: an image result: the page is the canonical locator, the file is a
#: direct-file locator, and the provider reports a resolution and a licence.
IMAGE_RESULT: dict[str, Any] = {
    "url": "https://images.example.net/page/capy",
    "title": "Capybara at dusk",
    "content": "",
    "engine": "bing images",
    "engines": ["bing images"],
    "positions": [2],
    "score": 1.0,
    "category": "images",
    "template": "images.html",
    "img_src": "https://cdn.example.net/capy.jpg",
    "thumbnail_src": "https://cdn.example.net/capy_thumb.jpg",
    "resolution": "1920x1080",
    "img_format": "jpeg",
    "license": "CC BY 2.0",
}

#: the two results together, in provider order.
TWO_RESULTS: tuple[dict[str, Any], ...] = (WEB_RESULT, IMAGE_RESULT)


def body(results: tuple[dict[str, Any], ...] = TWO_RESULTS, *, unresponsive: tuple[list[str], ...] = ()) -> bytes:
    """Render a SearXNG response envelope around ``results``.

    :param results: the result objects to carry
    :ptype results: tuple[dict[str, Any], ...]
    :param unresponsive: ``[engine, reason]`` pairs for engines that did not
        answer
    :ptype unresponsive: tuple[list[str], ...]
    :return: the JSON body bytes
    :rtype: bytes
    """
    payload: dict[str, Any] = {
        "query": "capybara",
        "number_of_results": len(results),
        "results": list(results),
        "answers": [],
        "corrections": [],
        "infoboxes": [],
        "suggestions": ["capybara habitat"],
        "unresponsive_engines": [list(entry) for entry in unresponsive],
    }
    return json.dumps(payload).encode("utf-8")


#: a successful two-result response.
TWO_RESULTS_BODY: bytes = body()

#: a successful response that found nothing (SR-J2's fixture).
ZERO_RESULTS_BODY: bytes = body(())

#: well-formed JSON that is not the shape the API promises -- no ``results``
#: list at all, which is what a misconfigured instance or a proxy error page
#: produces.
MALFORMED_BODY: bytes = json.dumps({"query": "capybara", "error": "formats"}).encode("utf-8")
