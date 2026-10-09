"""answers computed once per version, read by every replica, and retired when the version moves.

Two replicas are two collections on two registries sharing one L2 bucket under one key scope, as the
replicas of one tool pod do.
"""

from __future__ import annotations

import asyncio
import gzip

import pytest

from threetears.core.collections.registry import CollectionRegistry
from threetears.core.collections.versioned_answers import VersionedAnswers
from threetears.core.config import DefaultCoreConfig
from threetears.core.testing.kv import FakeNatsClient

_SCOPE = "tool_pod-answersunit"
_TABLE = "report_answers"


def _replica(nats: FakeNatsClient) -> VersionedAnswers:
    registry = CollectionRegistry()
    registry.configure(l1_backend=None, l2_client=nats, l3_pool=None, kv_key_scope=_SCOPE)  # type: ignore[arg-type]
    return VersionedAnswers(registry, DefaultCoreConfig(), nats, table_name=_TABLE)


class _Computer:
    """counts its computations and answers ``text``."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    async def __call__(self) -> str:
        self.calls += 1
        return self.text


def _text(compressed: bytes) -> str:
    return gzip.decompress(compressed).decode("utf-8")


@pytest.mark.asyncio
async def test_an_answer_is_computed_once_and_served_again_compressed() -> None:
    answers = _replica(FakeNatsClient())
    compute = _Computer('{"rows": [1, 2, 3]}')
    first = await answers.answer("v1", "rows|contest|state=VA", compute)
    second = await answers.answer("v1", "rows|contest|state=VA", compute)
    assert _text(first) == _text(second) == '{"rows": [1, 2, 3]}'
    assert compute.calls == 1


@pytest.mark.asyncio
async def test_a_second_replica_serves_an_answer_it_did_not_compute() -> None:
    nats = FakeNatsClient()
    computed_here = _Computer("the answer")
    await _replica(nats).answer("v1", "rows|contest|", computed_here)
    never = _Computer("must not run")
    served = await _replica(nats).answer("v1", "rows|contest|", never)
    assert _text(served) == "the answer"
    assert (computed_here.calls, never.calls) == (1, 0)


@pytest.mark.asyncio
async def test_another_version_or_request_is_another_answer() -> None:
    answers = _replica(FakeNatsClient())
    compute = _Computer("x")
    await answers.answer("v1", "a", compute)
    await answers.answer("v2", "a", compute)
    await answers.answer("v1", "b", compute)
    assert compute.calls == 3


@pytest.mark.asyncio
async def test_a_failure_reaches_the_caller_and_is_not_cached() -> None:
    answers = _replica(FakeNatsClient())

    async def refuses() -> str:
        raise LookupError("v1 is not the current version")

    with pytest.raises(LookupError, match="not the current version"):
        await answers.answer("v1", "a", refuses)
    compute = _Computer("now it answers")
    assert _text(await answers.answer("v1", "a", compute)) == "now it answers"
    assert compute.calls == 1


@pytest.mark.asyncio
async def test_concurrent_callers_on_one_replica_compute_once() -> None:
    answers = _replica(FakeNatsClient())
    calls = 0

    async def slow() -> str:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.02)
        return "slow"

    got = await asyncio.gather(*(answers.answer("v1", "a", slow) for _ in range(5)))
    assert {_text(g) for g in got} == {"slow"}
    assert calls == 1


@pytest.mark.asyncio
async def test_retiring_deletes_every_other_versions_answers_and_keeps_the_current() -> None:
    nats = FakeNatsClient()
    answers = _replica(nats)
    for version in ("v1", "v2", "v3"):
        for request in ("a", "b"):
            await answers.answer(version, request, _Computer(f"{version}{request}"))
    assert await answers.retire_all_but("v3") == 4
    compute = _Computer("again")
    await answers.answer("v3", "a", compute)
    await answers.answer("v1", "a", compute)
    # v3's answer was kept; v1's was retired and computed again
    assert compute.calls == 1


@pytest.mark.asyncio
async def test_a_new_current_version_retires_the_old_once() -> None:
    answers = _replica(FakeNatsClient())
    await answers.answer("v1", "a", _Computer("old"))
    answers.current_version("v2")
    answers.current_version("v2")
    await asyncio.sleep(0.01)
    compute = _Computer("recomputed")
    await answers.answer("v1", "a", compute)
    assert compute.calls == 1


def test_a_version_that_cannot_be_a_key_segment_is_refused() -> None:
    answers = _replica(FakeNatsClient())
    with pytest.raises(ValueError, match="version key segment"):
        answers.key_of("v1_2", "a")
    with pytest.raises(ValueError, match="version key segment"):
        answers.key_of("v 1", "a")


def test_the_request_is_digested_into_the_key() -> None:
    answers = _replica(FakeNatsClient())
    version, digest = answers.key_of("v1", "rows|contest|state=VA & county=Loudoun")
    assert version == "v1"
    assert len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)


@pytest.mark.asyncio
async def test_retiring_never_lists_keys_a_pods_grant_cannot_list() -> None:
    """a pod's grant on the shared bucket is key-addressed: no consumer, so no listing."""
    nats = FakeNatsClient()
    bucket = await nats.kv_bucket(name="collections")

    async def refused(*, prefix: str = "") -> list[str]:
        raise AssertionError(f"listed keys under {prefix!r}; a pod's grant refuses the consumer")

    bucket.list_keys = refused  # type: ignore[method-assign]
    answers = _replica(nats)
    await answers.answer("v1", "a", _Computer("old"))
    await answers.answer("v2", "a", _Computer("new"))
    assert await answers.retire_all_but("v2") == 1
    # the index forgets the retired version and keeps the current one
    assert await answers.retire_all_but("v2") == 0
    assert [key for key in bucket.keys() if key.endswith(".v1_" + answers.key_of("v1", "a")[1])] == []


def test_a_version_with_a_dot_is_refused() -> None:
    with pytest.raises(ValueError, match="version key segment"):
        _replica(FakeNatsClient()).key_of("v1.2", "a")
