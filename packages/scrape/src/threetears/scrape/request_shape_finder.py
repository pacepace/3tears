"""Verified-request-shape discovery: the "how" sibling to ``page_finder.py``'s
"what."

``find_target_page`` resolves which page/URL answers a query, and verifies
its structure deterministically -- but explicitly, by its own docstring,
"never verifies to camoufox/network_capture" (a stateless HTTP fetch can't
observe an authenticated in-session XHR/fetch call at all). For any target
whose real data arrives that way, the ACTUAL request shape -- method, URL,
body/query-param structure -- has been a model-inference gap: whoever builds
the plugin either already knows the target's real API (rare), or hand-guesses
a shape from a library's remembered conventions and iterates against the live
target until something returns 200. That second path is exactly the case
FIX_PROTOCOL exists to catch: the binding (the real request shape) was never
captured as a signal, so the model had to infer it -- and a wrong guess that
accidentally returns 200 with subtly incorrect data is worse than a loud
failure, because nothing flags it as wrong.

``capture_request_shape`` closes that gap the same way ``find_target_page``
closes the URL-discovery one: drive a REAL browser session to the target and
CAPTURE what it actually does, rather than modeling what it probably does.
Reuses ``ScrapeDriver.render(capture_network=True)`` (already used live by
``NetworkCaptureDriver`` for extraction) for discovery instead -- same
mechanism, different purpose: report the real captured calls back to the
caller as plain data, not synthesize HTML from them.

Deliberately does not pick "the one real call" out of the results -- a
caller's own real data call is not reliably the largest, the first, or
matched by any single generic heuristic across different targets (verified
against this module's own first real case: Google Trends' actual data call
returned a small widget-config body, while incidental picker/taxonomy
prefetch calls captured alongside it returned much larger lists -- a
"largest list wins" heuristic, ``NetworkCaptureDriver``'s own, would have
picked the WRONG call here). Returns everything captured; the caller
(a human, or a subsequent LLM coercion step mirroring ``page_finder.py``'s
own) decides which call is the real one -- "plain data, caller decides,"
the same discipline ``PageFinderResult`` already established.

Zero target-specific logic anywhere in this module (no XSSI-prefix handling,
no vendor names) -- that's the general test for whether this is real platform
capability or one target's fix wearing a general name: would THIS module help
a different, unrelated target of the same class (an authenticated in-session
XHR/fetch API)? If any line here only makes sense for one real target, it
does not belong in this file.

Zero faidh imports (matches ``page_finder.py``'s own discipline).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from threetears.observe import get_logger

from .driver import NavStep, RenderedPage, ScrapeDriver

__all__ = ["CapturedRequestShape", "RequestShapeResult", "capture_request_shape"]

log = get_logger(__name__)

_DEFAULT_SETTLE_WAIT_MS = 10_000


@dataclass(frozen=True)
class CapturedRequestShape:
    """One real captured XHR/fetch call, with its body parsed where possible.

    Plain data -- no interpretation of which call (if any) is "the real
    data call" happens here or anywhere else in this module.
    """

    url: str
    method: str
    status: int
    content_type: str
    #: The real, unparsed response body exactly as captured -- always
    #: present, even when ``body_shape`` is ``None`` (unparseable).
    body: str
    #: ``json.loads(body)`` when it parses, honoring a known JSON-hijacking
    #: prefix if the driver-level capture already stripped/preserved one
    #: (see ``NetworkCaptureDriver``'s sibling sidecar fix) -- ``None`` when
    #: the body isn't JSON-shaped at all, not fabricated as ``{}``.
    body_shape: Any = None


@dataclass(frozen=True)
class RequestShapeResult:
    """Plain-data result of one real browser render's captured network calls.

    Never persisted, never forces a conclusion about which call matters --
    mirrors ``PageFinderResult``'s own "caller decides" contract.
    """

    page_status: int
    final_url: str
    calls: tuple[CapturedRequestShape, ...] = field(default_factory=tuple)


def _parse_body_shape(body: str) -> Any:
    """Return ``json.loads(body)`` if it parses, honoring a leading
    anti-JSON-hijacking prefix already visible in the raw text (e.g. ``)]}'``)
    by skipping to the first ``{``/``[`` before attempting to parse -- generic
    handling of a documented, widely-used convention (not any one vendor's),
    matching this module's own no-target-specific-logic discipline.

    :param body: the real, unparsed captured response body.
    :ptype body: str
    :return: the parsed value, or ``None`` if nothing JSON-shaped is found.
    :rtype: Any
    """
    brace_index = min((i for i in (body.find("{"), body.find("[")) if i != -1), default=-1)
    if brace_index == -1:
        result = None
    else:
        try:
            result = json.loads(body[brace_index:])
        except json.JSONDecodeError:
            result = None
    return result


async def capture_request_shape(
    url: str,
    *,
    driver: ScrapeDriver,
    settle_wait_ms: int = _DEFAULT_SETTLE_WAIT_MS,
    nav_steps: list[NavStep] | None = None,
) -> RequestShapeResult:
    """Render *url* through a real browser-capable *driver* and report every
    real XHR/fetch call it made, verified -- not guessed.

    :param url: the target page to render.
    :ptype url: str
    :param driver: a real browser-capable ``ScrapeDriver`` (e.g.
        ``NodriverSidecarDriver``) -- matches ``NetworkCaptureDriver``'s own
        injected-``inner``-driver shape; there is no sensible zero-config
        default, the whole point is a real session.
    :ptype driver: ScrapeDriver
    :param settle_wait_ms: how long to let the page's own client-side JS run
        before inspecting captured calls -- a single-page-app's data calls
        often fire well after the initial document load event fires (a real
        page's own load event is not the same as "the app has fetched its
        real data yet"). Appended as a trailing ``wait_ms`` nav step.
    :ptype settle_wait_ms: int
    :param nav_steps: additional real browser actions to perform before the
        settle wait (e.g. a click/fill sequence to reach the data view) --
        run in order, ahead of the settle wait, matching
        ``ScrapeDriver.render``'s own ``nav_steps`` ordering contract.
    :ptype nav_steps: list[NavStep] | None
    :return: every real captured call from this one render, each with its
        body parsed where possible.
    :rtype: RequestShapeResult
    """
    page: RenderedPage = await driver.render(
        url,
        capture_network=True,
        nav_steps=[*(nav_steps or []), NavStep(action="wait_ms", ms=settle_wait_ms)],
    )
    calls = tuple(
        CapturedRequestShape(
            url=call.url,
            method=call.method,
            status=call.status,
            content_type=call.content_type,
            body=call.body,
            body_shape=_parse_body_shape(call.body),
        )
        for call in page.network_calls
    )
    log.info(
        "capture_request_shape: url=%s status=%s calls_captured=%d",
        url,
        page.status,
        len(calls),
    )
    return RequestShapeResult(page_status=page.status, final_url=page.final_url, calls=calls)
