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
import math
from collections.abc import Sequence
from typing import Any

__all__ = [
    "IMAGE_RESULT",
    "MALFORMED_BODY",
    "TWO_RESULTS",
    "TWO_RESULTS_BODY",
    "WEB_RESULT",
    "ZERO_RESULTS_BODY",
    "body",
    "searx_score",
]


def searx_score(positions: Sequence[int], engine_weights: Sequence[float] | None = None) -> float:
    """SearXNG's own ``searx/results.py:calculate_score``, for the default case.

    Every fixture score below is computed by this rather than written as a
    literal, because a literal is a second place for the value to drift from
    the formula it claims to follow -- and it did: ``WEB_RESULT`` read 2.5
    against positions ``[1, 3]`` (the formula gives 2.667) and
    ``IMAGE_RESULT`` read 1.0 against ``[2]`` (gives 0.5) until SR-A4 was
    settled against a live instance on 2026-08-12. The adapter only passes
    the field through, so nothing failed -- but a fixture whose docstring
    says it mirrors what SearXNG reports is where the next person goes to
    learn the formula, and that one taught a wrong one.

    Models default-priority engines only. A ``priority: high`` engine
    contributes ``weight`` per position instead of ``weight / position``, and
    ``priority: low`` contributes nothing at all; no fixture needs either
    yet, and adding one means extending this rather than hand-computing it.

    :param positions: the rank each contributing engine gave the result, 1-based
    :ptype positions: Sequence[int]
    :param engine_weights: each contributing engine's configured weight;
        defaults to 1.0 per position, which is SearXNG's own default
    :ptype engine_weights: Sequence[float] | None
    :return: the fused score, unbounded above (SR-A4)
    :rtype: float
    """
    weights = tuple(engine_weights) if engine_weights is not None else (1.0,) * len(positions)
    weight = math.prod(weights) * len(positions)
    return sum(weight / position for position in positions)


#: the ranks two default-weight engines gave the web result.
_WEB_POSITIONS = [1, 3]

#: an ordinary web result, with a naive published date (which is what
#: SearXNG's engines commonly report).
WEB_RESULT: dict[str, Any] = {
    "url": "https://example.org/capybara",
    "title": "Capybara",
    "content": "The capybara is the largest living rodent.",
    "engine": "duckduckgo",
    "engines": ["duckduckgo", "brave"],
    "positions": _WEB_POSITIONS,
    "score": searx_score(_WEB_POSITIONS),
    "category": "general",
    "template": "default.html",
    "publishedDate": "2026-02-01T00:00:00",
}

#: the rank the single image engine gave the image result.
_IMAGE_POSITIONS = [2]

#: an image result: the page is the canonical locator, the file is a
#: direct-file locator, and the provider reports a resolution and a licence.
IMAGE_RESULT: dict[str, Any] = {
    "url": "https://images.example.net/page/capy",
    "title": "Capybara at dusk",
    "content": "",
    "engine": "bing images",
    "engines": ["bing images"],
    "positions": _IMAGE_POSITIONS,
    "score": searx_score(_IMAGE_POSITIONS),
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
