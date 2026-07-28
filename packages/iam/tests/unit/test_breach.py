"""k-anonymized breach-password screening."""

from __future__ import annotations

import hashlib

import httpx
import pytest

from threetears.iam.breach import BreachCorpus, RangeApiBreachCorpus, sha1_prefix_suffix


def test_prefix_suffix_splits_an_uppercase_sha1() -> None:
    prefix, suffix = sha1_prefix_suffix("password")
    expected = hashlib.sha1(b"password", usedforsecurity=False).hexdigest().upper()
    assert prefix == expected[:5]
    assert suffix == expected[5:]
    assert len(prefix) == 5
    assert len(suffix) == 35


def test_seeded_corpus_matches_its_own_entries() -> None:
    corpus = BreachCorpus(seed_passwords=["hunter2", "letmein"])
    assert corpus.is_breached("hunter2")
    assert corpus.is_breached("letmein")
    assert not corpus.is_breached("something-nobody-has-used")


def test_empty_corpus_reports_nothing_breached() -> None:
    # The default is an empty index rather than a token handful of hardcoded passwords:
    # a partial corpus reads as coverage while providing almost none.
    assert not BreachCorpus().is_breached("password")


def test_only_the_prefix_crosses_the_lookup_boundary() -> None:
    seen: list[str] = []

    def lookup(prefix: str) -> frozenset[str]:
        seen.append(prefix)
        return frozenset()

    BreachCorpus(lookup_by_prefix=lookup).is_breached("a-secret-password")

    full_digest = hashlib.sha1(b"a-secret-password", usedforsecurity=False).hexdigest().upper()
    assert seen == [full_digest[:5]]
    # The whole point of k-anonymity: the boundary never sees enough to identify the candidate.
    assert full_digest not in seen
    assert full_digest[5:] not in seen


def test_supplied_lookup_overrides_the_seed_index() -> None:
    corpus = BreachCorpus(seed_passwords=["hunter2"], lookup_by_prefix=lambda _: frozenset())
    assert not corpus.is_breached("hunter2")


async def test_range_api_matches_a_listed_suffix() -> None:
    prefix, suffix = sha1_prefix_suffix("hunter2")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith(f"/{prefix}")
        return httpx.Response(200, text=f"0000000000000000000000000000000000A:3\r\n{suffix}:42\r\n")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        assert await RangeApiBreachCorpus(client).is_breached("hunter2")


async def test_range_api_reports_an_unlisted_suffix_as_clean() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="0000000000000000000000000000000000A:3\r\n")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        assert not await RangeApiBreachCorpus(client).is_breached("hunter2")


@pytest.mark.parametrize("status", [429, 500, 503])
async def test_range_api_fails_open_on_an_error_response(status: int) -> None:
    # Documented and deliberate for this control only: an outage that blocked every
    # password change would do more damage than the unscreened window it prevents.
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(status)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        assert not await RangeApiBreachCorpus(client).is_breached("hunter2")


async def test_range_api_fails_open_on_a_transport_error() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        assert not await RangeApiBreachCorpus(client).is_breached("hunter2")


async def test_range_api_skips_malformed_lines_without_losing_the_match() -> None:
    prefix, suffix = sha1_prefix_suffix("hunter2")

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=f"\r\n   \r\nnot-a-real-line\r\n{suffix}:9\r\n")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        assert await RangeApiBreachCorpus(client).is_breached("hunter2")
    assert len(prefix) == 5
