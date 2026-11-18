"""the L2 size gauges: each bucket's kind, its bytes and keys by declared policy, its budget, and a sizing failure.

The streams below are shaped as cobalt dev's ``STREAM.LIST`` answered them on 2026-11-15 (N1's inventory).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from datetime import timedelta
from typing import Any

import pytest
from prometheus_client import REGISTRY

from threetears.core.eviction import Policy, Tier, policies_declared
from threetears.core.l2_sizes import L2Sizes, kind_of
from threetears.nats.errors import RequestTimeoutError
from threetears.nats.subjects import Subject

_NS = "aibots"
_AGENT = "agent_pod-06a8e0804ffd7b51800083b79e123597"
_TOOL = "tool_pod-06ad00763a0a7ea5800050c45f5dd589"
_GiB = 1024**3


def _stream(
    name: str, *, held: int, max_bytes: int = -1, storage: str = "memory", keys: int = 1, messages: int = 1
) -> dict[str, Any]:
    return {
        "config": {"name": name, "storage": storage, "max_bytes": max_bytes, "num_replicas": 3},
        "state": {"bytes": held, "num_subjects": keys, "messages": messages},
    }


class _Lister:
    """answers ``STREAM.LIST`` from a fixed set of streams, ``limit`` per page, as the server pages it."""

    def __init__(self, streams: Sequence[Mapping[str, Any]], *, limit: int = 256) -> None:
        self.streams: Sequence[Mapping[str, Any]] = list(streams)
        self.limit = limit
        self.asked: list[int] = []
        self.fail = False

    async def request_raw(
        self, *, subject: Subject, payload: bytes, timeout: timedelta = timedelta(seconds=1)
    ) -> bytes:
        assert subject.path == "$JS.API.STREAM.LIST"
        if self.fail:
            raise RequestTimeoutError("no answer")
        offset = int(json.loads(payload)["offset"])
        self.asked.append(offset)
        page = self.streams[offset : offset + self.limit]
        return json.dumps({"total": len(self.streams), "offset": offset, "limit": self.limit, "streams": page}).encode()


def _held(name: str, policy: Policy) -> float | None:
    return REGISTRY.get_sample_value("threetears_cache_held_bytes", {"tier": "l2", "name": name, "policy": policy})


def _entries(name: str, policy: Policy) -> float | None:
    return REGISTRY.get_sample_value("threetears_cache_held_entries", {"tier": "l2", "name": name, "policy": policy})


def _budget(name: str) -> float | None:
    return REGISTRY.get_sample_value("threetears_cache_budget_bytes", {"tier": "l2", "name": name})


def _sizing_failures(name: str) -> float:
    return REGISTRY.get_sample_value("threetears_cache_sizing_failures_total", {"tier": "l2", "name": name}) or 0.0


@pytest.mark.parametrize(
    ("stream", "kind"),
    [
        ("KV_aibots-leases", "leases"),
        ("KV_aibots-collections", "collections"),
        ("aibots-audit", "audit"),
        ("aibots-tools-results", "tools-results"),
        ("KV_tool_catalog", "tool_catalog"),
        ("KV_aibots_agent_config", "agent_config"),
        (f"KV_aibots-{_AGENT}-workspace-locks", "workspace-locks"),
        (f"KV_aibots-{_AGENT}-run_claims", "agent-coordination"),
        (f"KV_aibots-{_AGENT}-survey-quota-cells", "agent-coordination"),
        (f"OBJ_aibots-{_TOOL}-objects", "objects"),
        (f"KV_aibots-{_TOOL}-pointers", "pointers"),
        ("KV_aibots-identity-dpop-nonces", "identity-dpop-nonces"),
    ],
)
def test_a_streams_kind_is_its_name_without_prefix_namespace_or_pod_scope(stream: str, kind: str) -> None:
    assert kind_of(stream, namespace=_NS) == kind


#: the buckets a 3tears package names, as their declarers pass them
_3TEARS_KINDS = (
    "collections",
    "leases",
    "epochs",
    "epoch-changes",
    "data-versions",
    "pop_nonces",
    "proxy_assertion_nonces",
    "gateway-door-proof-nonces",
    "ratelimits",
    "scheduler-locks",
    "derived-build-locks",
    "geo-tile-build-locks",
    "workspace-locks",
    "objects",
    "pointers",
    "tools-results",
    "tool_catalog",
    "agent_router_catalog",
    "agent_config",
    "agents-turn",
    "channels-deliver",
    "audit",
    "checkpoints",
    "agent-coordination",
)


def test_every_3tears_kind_declares_a_policy_with_its_reason() -> None:
    declared = policies_declared(Tier.L2)
    for kind in _3TEARS_KINDS:
        assert declared[kind].why.strip(), kind
    # the one cache among them: everything else is state that must stay whole
    evictable = {kind for kind in _3TEARS_KINDS if declared[kind].policy is Policy.EVICTABLE}
    assert evictable == {"collections"}


def test_a_sample_sets_each_kinds_bytes_keys_and_budget_and_the_tiers() -> None:
    lister = _Lister(
        [
            _stream("KV_aibots-collections", held=55_397_810, max_bytes=512 * 1024**2, keys=118_284, messages=118_284),
            _stream(f"OBJ_aibots-{_TOOL}-objects", held=356_782_105, max_bytes=3 * _GiB, keys=40, messages=5960),
            _stream(f"KV_aibots-{_AGENT}-workspace-locks", held=100, max_bytes=1024**2, keys=2),
            _stream("KV_aibots-agent_pod-06a639efd6b37bc68000da7c82fddea2-workspace-locks", held=50, max_bytes=1024**2),
            _stream("KV_aibots-pop_nonces", held=7, keys=1),
            _stream("aibots-audit", held=267_645_533, storage="file"),
            _stream("KV_aibots-not-a-declared-kind", held=999),
        ],
        limit=3,
    )
    sizes = asyncio.run(L2Sizes(lister, namespace=_NS, budget_bytes=6 * _GiB).sample_once())

    assert lister.asked == [0, 3, 6]
    assert len(sizes) == 7
    assert _held("collections", Policy.EVICTABLE) == 55_397_810
    assert _entries("collections", Policy.EVICTABLE) == 118_284
    assert _budget("collections") == 512 * 1024**2
    # an Object Store counts its messages (chunks), a KV bucket its keys
    assert _entries("objects", Policy.KEEP) == 5960
    # two agents' lock buckets are one kind: summed, bytes, keys and bounds alike
    assert _held("workspace-locks", Policy.KEEP) == 150
    assert _entries("workspace-locks", Policy.KEEP) == 3
    assert _budget("workspace-locks") == 2 * 1024**2
    # unbounded: held, but no bound to add to the budget
    assert _held("pop_nonces", Policy.KEEP) == 7
    assert _budget("pop_nonces") == 0
    # a file stream is outside the memory budget, and an undeclared kind has no policy label: neither is exported
    assert _held("audit", Policy.KEEP) is None
    assert _held("not-a-declared-kind", Policy.KEEP) is None
    assert _budget("") == 6 * _GiB
    assert _held("", Policy.EVICTABLE) == 55_397_810
    assert _held("", Policy.KEEP) == 356_782_105 + 150 + 7


def test_a_kind_gone_since_the_last_sample_reads_zero() -> None:
    lister = _Lister([_stream("KV_aibots-derived-build-locks", held=300, max_bytes=1024**2)])
    sizes = L2Sizes(lister, namespace=_NS, budget_bytes=6 * _GiB)
    asyncio.run(sizes.sample_once())
    assert _held("derived-build-locks", Policy.KEEP) == 300
    lister.streams = []
    asyncio.run(sizes.sample_once())
    assert _held("derived-build-locks", Policy.KEEP) == 0


def test_keep_reaching_a_budget_is_one_sizing_failure_until_it_falls_back() -> None:
    full = [_stream("KV_aibots-scheduler-locks", held=1024**2, max_bytes=1024**2)]
    lister = _Lister(full)
    sizes = L2Sizes(lister, namespace=_NS, budget_bytes=6 * _GiB)
    before = _sizing_failures("scheduler-locks")
    asyncio.run(sizes.sample_once())
    asyncio.run(sizes.sample_once())
    assert _sizing_failures("scheduler-locks") == before + 1
    lister.streams = [_stream("KV_aibots-scheduler-locks", held=10, max_bytes=1024**2)]
    asyncio.run(sizes.sample_once())
    lister.streams = full
    asyncio.run(sizes.sample_once())
    assert _sizing_failures("scheduler-locks") == before + 2


def test_a_full_cache_is_not_a_sizing_failure() -> None:
    lister = _Lister([_stream("KV_aibots-collections", held=512 * 1024**2, max_bytes=512 * 1024**2)])
    before = _sizing_failures("collections")
    asyncio.run(L2Sizes(lister, namespace=_NS, budget_bytes=6 * _GiB).sample_once())
    assert _sizing_failures("collections") == before


def test_the_tiers_keep_bytes_at_its_budget_are_a_sizing_failure() -> None:
    lister = _Lister([_stream(f"OBJ_aibots-{_TOOL}-objects", held=2048, max_bytes=3 * _GiB, messages=2)])
    before = _sizing_failures("")
    asyncio.run(L2Sizes(lister, namespace=_NS, budget_bytes=1024).sample_once())
    assert _sizing_failures("") == before + 1


def test_a_sample_nats_does_not_answer_is_logged_and_the_next_one_runs() -> None:
    lister = _Lister([_stream("KV_aibots-geo-tile-build-locks", held=42, max_bytes=1024**2)])
    lister.fail = True

    async def run() -> None:
        sizes = L2Sizes(lister, namespace=_NS, budget_bytes=6 * _GiB, interval=0.01)
        sizes.start()
        await asyncio.sleep(0.03)
        lister.fail = False
        for _ in range(100):
            if lister.asked:
                break
            await asyncio.sleep(0.01)
        await sizes.stop()

    asyncio.run(run())
    assert lister.asked
    assert _held("geo-tile-build-locks", Policy.KEEP) == 42
