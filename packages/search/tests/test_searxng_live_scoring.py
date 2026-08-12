"""SR-A4 against the real scorer: SearXNG's ``score`` is the formula we claim.

The offline suite drives the adapter over recorded payloads. That proves the
adapter reads ``score`` and passes it through; it cannot prove the number
*means* what :meth:`SearxngAdapter._scores`' docstring says it means, because
the fixture asserting the meaning is one we wrote. Only SearXNG computing it
can settle that, which is what this module is for.

**What is asserted, and why it is not "fusion happens".** SearXNG's
``calculate_score`` is deterministic. The engines it federates are not: they
rate-limit, they drop out, and a run that saw two engines agree will not
reliably see it again. So the pin is the *invariant* --

    for every result carrying ``positions``, ``score`` equals
    ``searx_score(positions)``

-- which holds over whatever came back, one engine or five, and fails the
moment the formula the family encodes stops matching the one SearXNG runs.
Asserting that a fused score appeared would be asserting the weather.

**A fused capture, recorded because it cannot be demanded.** On 2026-08-12
this instance returned, before its engines were rate-limited::

    score     = 4.642857142857142
    engines   = ["duckduckgo", "brave"]
    positions = [1, 21, 2]

``searx_score([1, 21, 2])`` is ``3/1 + 3/21 + 3/2`` = 4.642857142857143, which
matches to floating point. That is the observation SR-A4's residue was waiting
on: a score above 1.0, produced by real engine fusion, agreeing with the
formula. It also settled something the formula's prose left ambiguous --
**two engines produced three positions**, so ``len(positions)`` drives the
weight and ``len(engines)`` does not. A ``searx_score(engines)`` would have
been wrong and would have looked right on every single-engine result.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Any

import pytest

from ._searxng_payloads import searx_score

pytestmark = pytest.mark.integration


def _search(base_url: str, query: str) -> dict[str, Any]:
    """Ask the instance for one query in JSON.

    Uses ``urllib`` rather than the package's own transport on purpose: this
    module is checking SearXNG, and routing through the thing under test
    elsewhere would let an adapter bug present as a scorer disagreement.

    :param base_url: the container's base URL
    :ptype base_url: str
    :param query: the search text
    :ptype query: str
    :return: the decoded JSON body
    :rtype: dict[str, Any]
    """
    url = f"{base_url}/search?{urllib.parse.urlencode({'q': query, 'format': 'json'})}"
    with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310
        payload: dict[str, Any] = json.loads(response.read())
    return payload


def test_the_live_score_is_the_formula_the_family_encodes(searxng_container: str) -> None:
    """Every scored result must equal ``searx_score`` of its own positions.

    **What a pass here does and does not prove.** On single-position results
    -- which is all a throttled instance returns -- the ``len(positions)``
    factor is invisible: ``len(p) * (1/p)`` and ``1/p`` are the same number
    when ``p`` has one entry. Checked by mutation against 18 real captured
    results: dropping that factor produced **zero** mismatches, while a
    0-based reading produced 18 and a sum-of-weights reading produced 16. So
    this pin catches a formula that is wrong in shape and cannot catch one
    that is wrong only in the multiplier.

    The sibling below is what pins the multiplier, and it skips rather than
    passes when the data cannot settle it. Neither is written to look
    stronger than the results it was handed.

    :param searxng_container: base URL of the session's SearXNG
    :ptype searxng_container: str
    :return: nothing
    :rtype: None
    """
    scored: list[dict[str, Any]] = []
    # Several queries because any one of them may come back empty when the
    # upstream engines are throttling; the assertion below is unaffected by
    # which of them answered.
    for query in ("capybara habitat", "postgres partial index", "rust borrow checker"):
        scored.extend(
            result
            for result in _search(searxng_container, query).get("results", [])
            if "positions" in result and "score" in result
        )

    if not scored:
        pytest.skip("the instance returned no scored results; its engines are throttled")

    mismatched = [
        (result["positions"], result["score"], searx_score(result["positions"]))
        for result in scored
        if abs(searx_score(result["positions"]) - float(result["score"])) > 1e-9
    ]
    assert not mismatched, (
        f"SearXNG's score disagrees with searx_score on {len(mismatched)} of {len(scored)} results: "
        f"{mismatched[:3]}. The family's copy of calculate_score has drifted from the one SearXNG runs."
    )


def test_the_weight_follows_positions_and_not_engine_count(searxng_container: str) -> None:
    """Where a result carries both, ``positions`` is what the score is built from.

    Skips rather than fails when no such result appears: one engine returning
    a page once is the common case, and a throttled instance returns only
    those. What must never happen is a result whose score matches an
    engine-count reading of the formula and not a position-count one.

    :param searxng_container: base URL of the session's SearXNG
    :ptype searxng_container: str
    :return: nothing
    :rtype: None
    """
    for query in ("capybara habitat", "python asyncio timeout", "sqlite wal mode"):
        for result in _search(searxng_container, query).get("results", []):
            positions = result.get("positions")
            engines = result.get("engines")
            if not positions or not engines or len(positions) == len(engines):
                continue
            assert float(result["score"]) == pytest.approx(searx_score(positions)), (
                f"score {result['score']} matches neither reading for positions={positions} engines={engines}"
            )
            return
    pytest.skip("no result arrived with a position count differing from its engine count")
