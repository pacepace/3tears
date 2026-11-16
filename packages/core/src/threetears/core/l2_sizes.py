"""How big each L2 (NATS) bucket is, against the cluster's memory budget, as the one cache family's gauges.

**Why.** Every L2 bucket and stream lives in ONE shared NATS cluster, on memory storage, three replicas
(``nats-memory-only``, ``nats-buckets-replicated``). So a byte held costs a byte on each of the three servers, and
every product's buckets draw on each server's one JetStream ``max_memory_store``. The L2's budget is that store, not
any pod's memory: ``max_memory_store x servers / replicas`` (6 GiB x 3 / 3 = 6 GiB on cobalt dev and prod,
2026-11-15). A bounded memory stream reserves its whole ``max_bytes`` on every server holding it; an unbounded one
reserves nothing and grows until the server's store is full, and then every memory stream on it refuses writes.

**What it exports.** :mod:`threetears.core.eviction`'s family, ``tier="l2"``: per declared bucket kind (``name``)
``threetears_cache_held_bytes`` and ``threetears_cache_held_entries`` (keys for a KV bucket, messages otherwise) by
``policy``, and ``threetears_cache_budget_bytes`` (the sum of its streams' ``max_bytes``); tier-wide (``name=""``)
the budget the process was given and the held bytes by policy; ``threetears_cache_sizing_failures_total`` when what
must be KEPT alone reaches a budget (a bucket's or the tier's). The servers' own store use against their maximum is
the NATS exporter's (``nats_varz_jetstream_stats_memory``, ``..._reserved_memory``, ``nats_server_max_memory``),
already scraped beside these; it is not measured again here. Metrics only: alerting is configured in Grafana.

**The kinds.** A bucket's kind is its name without the ``KV_``/``OBJ_`` stream prefix, the namespace and a pod's
scope (:func:`kind_of`): ``KV_aibots-agent_pod-<hex>-workspace-locks`` is ``workspace-locks``. Each kind declares
its eviction policy with its reason (:func:`threetears.core.eviction.policy_declare`: this module for the buckets a
3tears package names; a product declares its own the same way). A live stream whose kind is undeclared is not
exported, since the family has no "no policy" label, and is logged once; a product's test refuses one (the hub's
``test_nats_buckets_declare_their_eviction_policy``).

**Only memory streams count.** A FILE stream is outside the memory budget (and against the memory-only rule); it is
logged once and not exported, so the tier's held bytes are what the budget is spent on.

**Cost.** One ``STREAM.LIST`` page per 256 streams a minute, IO on the loop; decoding the answer is the only CPU:
71 streams on cobalt dev answer about 140 KB, which ``json.loads`` decodes in about 1 ms (N1, 2026-11-15) -- once a
minute, so it stays in the serving process.
"""

from __future__ import annotations

import asyncio
import enum
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Final, Protocol

from threetears.core.eviction import (
    BUDGET_BYTES,
    HELD_BYTES,
    HELD_ENTRIES,
    SIZING_FAILURES,
    Declared,
    Policy,
    Tier,
    policies_declared,
    policy_declare,
)
from threetears.nats.errors import NatsClientError
from threetears.nats.subjects import Subject
from threetears.observe import contain, get_logger
from threetears.observe.background import cancel_and_wait, spawn_background

__all__ = ["L2Sizes", "kind_of"]

log = get_logger(__name__)

#: how often the sizes are read: a bucket's size moves with loads and pods, minutes apart
_INTERVAL_SECONDS: Final = 60.0

#: the ceiling on one ``STREAM.LIST`` page's request
_REQUEST_TIMEOUT: Final = timedelta(seconds=10)

#: a pod's own scope inside a bucket name, ``agent_pod-<32 hex>-`` or ``tool_pod-<32 hex>-``
#: (:mod:`threetears.nats.subject_permissions` composes them so)
_POD_SCOPE: Final = re.compile(r"^(agent_pod|tool_pod)-[0-9a-f]{32}-")

#: the stream-name prefixes JetStream gives a KV bucket and an Object Store
_STREAM_PREFIXES: Final = ("KV_", "OBJ_")

#: the kind of an agent's own coordination bucket whose suffix the agent chose (a run claim, a quota cell): the hub
#: declares each from the agent's grant (``AgentBucketGrant``), so its suffix is the agent's, not the platform's
_AGENT_COORDINATION: Final = "agent-coordination"


class _L2Kind(enum.StrEnum):
    """the L2 buckets and streams a 3tears package names, by kind (the name a declarer passes, unprefixed)."""

    COLLECTIONS = "collections"
    LEASES = "leases"
    EPOCHS = "epochs"
    EPOCH_CHANGES = "epoch-changes"
    DATA_VERSIONS = "data-versions"
    POP_NONCES = "pop_nonces"
    PROXY_ASSERTION_NONCES = "proxy_assertion_nonces"
    GATEWAY_DOOR_PROOF_NONCES = "gateway-door-proof-nonces"
    RATELIMITS = "ratelimits"
    SCHEDULER_LOCKS = "scheduler-locks"
    DERIVED_BUILD_LOCKS = "derived-build-locks"
    GEO_TILE_BUILD_LOCKS = "geo-tile-build-locks"
    WORKSPACE_LOCKS = "workspace-locks"
    POD_OBJECTS = "objects"
    POD_POINTERS = "pointers"
    TOOLS_RESULTS = "tools-results"
    TOOL_CATALOG = "tool_catalog"
    AGENT_ROUTER_CATALOG = "agent_router_catalog"
    AGENT_CONFIG = "agent_config"
    AGENTS_TURN = "agents-turn"
    CHANNELS_DELIVER = "channels-deliver"
    AUDIT = "audit"
    CHECKPOINTS = "checkpoints"
    AGENT_COORDINATION = _AGENT_COORDINATION


#: why each must be kept whole, said once for the shapes several kinds share
_REPLAY: Final = "a replay guard: a dropped nonce admits the replay it guards against"
_LOCK: Final = "a lock or claim: dropping one lets two holders run the same work"
_REGISTRY: Final = "a registry: a dropped entry hides what registered until it registers again"
_IN_FLIGHT: Final = "a message in flight: dropping it loses an answer or a delivery"

policy_declare(
    Tier.L2,
    _L2Kind,
    {
        _L2Kind.COLLECTIONS: Declared(
            Policy.EVICTABLE, "the platform's entity cache: a dropped entry is a miss read from L3 and written back"
        ),
        _L2Kind.LEASES: Declared(Policy.KEEP, "holds and leases: a dropped one releases a hold early (2026-11-02)"),
        _L2Kind.EPOCHS: Declared(Policy.KEEP, "each table's epoch: readers trust it to say what changed"),
        _L2Kind.EPOCH_CHANGES: Declared(
            Policy.KEEP, "the epoch change log: a follower that missed a broadcast reads it"
        ),
        _L2Kind.DATA_VERSIONS: Declared(Policy.KEEP, "each dataset's published version: readers flip on it"),
        _L2Kind.POP_NONCES: Declared(Policy.KEEP, _REPLAY),
        _L2Kind.PROXY_ASSERTION_NONCES: Declared(Policy.KEEP, _REPLAY),
        _L2Kind.GATEWAY_DOOR_PROOF_NONCES: Declared(Policy.KEEP, _REPLAY),
        _L2Kind.RATELIMITS: Declared(Policy.KEEP, "rate counts: a dropped count lets a caller past its limit"),
        _L2Kind.SCHEDULER_LOCKS: Declared(Policy.KEEP, _LOCK),
        _L2Kind.DERIVED_BUILD_LOCKS: Declared(Policy.KEEP, _LOCK),
        _L2Kind.GEO_TILE_BUILD_LOCKS: Declared(Policy.KEEP, _LOCK),
        _L2Kind.WORKSPACE_LOCKS: Declared(Policy.KEEP, _LOCK),
        _L2Kind.POD_OBJECTS: Declared(
            Policy.KEEP,
            "a tool pod's own Object Store, what it serves: the pod retires whole objects itself; a dropped chunk "
            "breaks an object",
        ),
        _L2Kind.POD_POINTERS: Declared(Policy.KEEP, "a tool pod's pointers to the objects it serves"),
        _L2Kind.TOOLS_RESULTS: Declared(Policy.KEEP, _IN_FLIGHT),
        _L2Kind.TOOL_CATALOG: Declared(Policy.KEEP, _REGISTRY),
        _L2Kind.AGENT_ROUTER_CATALOG: Declared(Policy.KEEP, _REGISTRY),
        _L2Kind.AGENT_CONFIG: Declared(Policy.KEEP, "each agent's turn settings: the router reads them on every turn"),
        _L2Kind.AGENTS_TURN: Declared(Policy.KEEP, _IN_FLIGHT),
        _L2Kind.CHANNELS_DELIVER: Declared(Policy.KEEP, _IN_FLIGHT),
        _L2Kind.AUDIT: Declared(Policy.KEEP, "the audit trail: a dropped record is a gap in it"),
        _L2Kind.CHECKPOINTS: Declared(
            Policy.KEEP, "a conversation's graph state: a dropped one loses the conversation"
        ),
        _L2Kind.AGENT_COORDINATION: Declared(
            Policy.KEEP, "an agent's own coordination state (claims, tickets, quota cells, nonces), declared by the hub"
        ),
    },
)


def kind_of(stream: str, *, namespace: str) -> str:
    """the kind of the bucket or stream named ``stream``: its name without the KV/Object Store prefix, the
    ``namespace`` and a pod's scope. An agent pod's bucket whose suffix is no declared kind is the agent's own
    coordination bucket (``agent-coordination``).

    :param stream: the JetStream stream's name (``KV_aibots-leases``, ``aibots-audit``, ``KV_tool_catalog``)
    :ptype stream: str
    :param namespace: the deployment's namespace (``aibots``)
    :ptype namespace: str
    :return: the kind (``leases``, ``audit``, ``tool_catalog``)
    :rtype: str
    """
    name = stream
    for prefix in _STREAM_PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    for joiner in ("-", "_"):
        if name.startswith(namespace + joiner):
            name = name[len(namespace) + 1 :]
            break
    scope = _POD_SCOPE.match(name)
    if scope is None:
        return name
    suffix = name[scope.end() :]
    if scope.group(1) == "agent_pod" and suffix not in policies_declared(Tier.L2):
        return _AGENT_COORDINATION
    return suffix


@dataclass(frozen=True)
class _StreamSize:
    """one live stream as the sampler read it.

    :ivar stream: its name
    :ivar kind: its kind (:func:`kind_of`)
    :ivar memory: whether it is on memory storage
    :ivar held_bytes: the bytes it holds (one replica's)
    :ivar entries: its keys (a KV bucket) or messages
    :ivar max_bytes: its bound; ``None`` when unbounded
    """

    stream: str
    kind: str
    memory: bool
    held_bytes: int
    entries: int
    max_bytes: int | None


class _Requester(Protocol):
    """the slice of :class:`~threetears.nats.NatsClient` the sampler asks through."""

    async def request_raw(self, *, subject: Subject, payload: bytes, timeout: timedelta = ...) -> bytes:
        """ask ``subject`` with ``payload`` and answer the reply's bytes.

        :param subject: the subject asked (``$JS.API.STREAM.LIST``)
        :ptype subject: Subject
        :param payload: the request's body
        :ptype payload: bytes
        :param timeout: how long to wait for the reply
        :ptype timeout: timedelta
        :return: the reply's body
        :rtype: bytes
        """
        ...


def _stream_size(info: Mapping[str, Any], *, namespace: str) -> _StreamSize:
    """one ``STREAM.LIST`` entry as a :class:`_StreamSize`.

    :param info: the entry (``config`` and ``state``)
    :ptype info: Mapping[str, Any]
    :param namespace: the deployment's namespace
    :ptype namespace: str
    :return: its size
    :rtype: _StreamSize
    """
    config, state = info["config"], info["state"]
    name = str(config["name"])
    max_bytes = int(config.get("max_bytes", -1))
    # a KV bucket keeps one revision per key, so its subjects are its keys; any other stream counts messages
    entries = int(state.get("num_subjects", 0)) if name.startswith("KV_") else int(state.get("messages", 0))
    return _StreamSize(
        stream=name,
        kind=kind_of(name, namespace=namespace),
        memory=config.get("storage") == "memory",
        held_bytes=int(state.get("bytes", 0)),
        entries=entries,
        max_bytes=max_bytes if max_bytes > 0 else None,
    )


class L2Sizes:
    """the L2's size gauges for one deployment: every stream its account holds, read every ``interval``.

    Run by ONE process per deployment that may list every stream (the hub), so a gauge is not summed over replicas
    that all read the same streams: a sum over its ``name`` is the deployment's.

    :param client: the connected client to ask through
    :ptype client: _Requester
    :param namespace: the deployment's namespace
    :ptype namespace: str
    :param budget_bytes: the tier's budget, ``max_memory_store x servers / replicas``, from the deployment's config
    :ptype budget_bytes: int
    :param interval: seconds between samples
    :ptype interval: float
    """

    def __init__(
        self, client: _Requester, *, namespace: str, budget_bytes: int, interval: float = _INTERVAL_SECONDS
    ) -> None:
        self._client = client
        self._namespace = namespace
        self._budget = budget_bytes
        self._interval = interval
        self._task: asyncio.Task[None] | None = None
        #: the (kind, policy) label sets the last sample set, so a kind gone since reads 0 rather than its last size
        self._exported: set[tuple[str, str]] = set()
        #: what has been logged once already: undeclared kinds, file streams, unbounded memory streams
        self._logged: set[tuple[str, str]] = set()
        #: the budgets KEEP alone has reached, counted once on reaching rather than every sample after
        self._failing: set[str] = set()

    async def read(self) -> list[_StreamSize]:
        """every stream of the account, one ``STREAM.LIST`` page at a time.

        :return: each stream's size
        :rtype: list[_StreamSize]
        :raises NatsClientError: a page was not answered
        :raises ValueError: a page did not decode, or answered an error
        """
        found: list[_StreamSize] = []
        while True:
            raw = await self._client.request_raw(
                subject=Subject.raw("$JS.API.STREAM.LIST"),
                payload=json.dumps({"offset": len(found)}).encode("utf-8"),
                timeout=_REQUEST_TIMEOUT,
            )
            page = json.loads(raw)
            if "error" in page:
                raise ValueError(f"STREAM.LIST answered an error: {page['error']}")
            streams = page.get("streams") or []
            found.extend(_stream_size(info, namespace=self._namespace) for info in streams)
            if not streams or len(found) >= int(page.get("total", 0)):
                return found

    async def sample_once(self) -> list[_StreamSize]:
        """read every stream now and set the gauges.

        :return: each stream's size, declared or not
        :rtype: list[_StreamSize]
        :raises NatsClientError: a page was not answered
        :raises ValueError: a page did not decode, or answered an error
        """
        sizes = await self.read()
        self._export(sizes)
        return sizes

    def _export(self, sizes: list[_StreamSize]) -> None:
        """set each declared kind's gauges and the tier's, zero kinds gone since the last sample, log what is not
        exported once, and count a budget KEEP alone reached.

        :param sizes: each stream's size
        :ptype sizes: list[_StreamSize]
        :return: nothing
        :rtype: None
        """
        policies = policies_declared(Tier.L2)
        held: dict[tuple[str, str], list[int]] = {}
        budgets: dict[str, int] = {}
        for size in sizes:
            declared = policies.get(size.kind)
            if declared is None:
                self._log_once("undeclared", size.stream, "l2 sizes: a stream whose kind declares no eviction policy")
                continue
            if not size.memory:
                self._log_once("file", size.stream, "l2 sizes: a FILE stream, outside the L2 memory budget")
                continue
            if size.max_bytes is None:
                self._log_once("unbounded", size.stream, "l2 sizes: an unbounded memory stream reserves no budget")
            key = (size.kind, declared.policy.value)
            bytes_entries = held.setdefault(key, [0, 0])
            bytes_entries[0] += size.held_bytes
            bytes_entries[1] += size.entries
            budgets[size.kind] = budgets.get(size.kind, 0) + (size.max_bytes or 0)
        tier = Tier.L2.value
        by_policy = dict.fromkeys((policy.value for policy in Policy), 0)
        for (kind, policy), (held_bytes, entries) in held.items():
            HELD_BYTES.labels(tier=tier, name=kind, policy=policy).set(held_bytes)
            HELD_ENTRIES.labels(tier=tier, name=kind, policy=policy).set(entries)
            by_policy[policy] += held_bytes
            self._sizing(kind, kept=held_bytes if policy == Policy.KEEP.value else 0, budget=budgets[kind])
        for kind, policy in self._exported - set(held):
            HELD_BYTES.labels(tier=tier, name=kind, policy=policy).set(0)
            HELD_ENTRIES.labels(tier=tier, name=kind, policy=policy).set(0)
        self._exported = set(held)
        for kind, budget in budgets.items():
            BUDGET_BYTES.labels(tier=tier, name=kind).set(budget)
        BUDGET_BYTES.labels(tier=tier, name="").set(self._budget)
        for policy, held_bytes in by_policy.items():
            HELD_BYTES.labels(tier=tier, name="", policy=policy).set(held_bytes)
        self._sizing("", kept=by_policy[Policy.KEEP.value], budget=self._budget)

    def _sizing(self, name: str, *, kept: int, budget: int) -> None:
        """count and log a sizing failure when what must be KEPT reaches ``budget`` (once on reaching it).

        A bounded KEEP bucket at its bound refuses every new key; the tier's KEEP bytes at the tier's budget leave
        nothing for any cache. Neither is ever an eviction.

        :param name: the kind, ``""`` for the tier
        :ptype name: str
        :param kept: the KEEP bytes held
        :ptype kept: int
        :param budget: the budget; 0 for an unbounded kind, which has none to reach
        :ptype budget: int
        :return: nothing
        :rtype: None
        """
        if budget <= 0 or kept < budget:
            self._failing.discard(name)
            return
        if name in self._failing:
            return
        self._failing.add(name)
        SIZING_FAILURES.labels(tier=Tier.L2.value, name=name).inc()
        log.error(
            "l2 sizes: what must be kept has reached its budget (a sizing failure, never an eviction)",
            extra={"extra_data": {"name": name or "(tier)", "kept_bytes": kept, "budget_bytes": budget}},
        )

    def _log_once(self, why: str, stream: str, what: str) -> None:
        if (why, stream) not in self._logged:
            self._logged.add((why, stream))
            log.warning(what, extra={"extra_data": {"stream": stream}})

    def start(self) -> None:
        """sample every ``interval`` from now on, the first at once, until :meth:`stop`.

        :return: nothing
        :rtype: None
        """
        if self._task is None:
            self._task = spawn_background(self._run(), name="l2-sizes", logger=log)

    async def stop(self) -> None:
        """stop sampling.

        :return: nothing
        :rtype: None
        """
        task, self._task = self._task, None
        if task is not None:
            await cancel_and_wait(task)

    async def _run(self) -> None:
        """sample, then wait ``interval``, until cancelled; a failed sample is logged and the next one tries again.

        :return: nothing
        :rtype: None
        """
        while True:
            try:
                await self.sample_once()
            except (NatsClientError, ValueError, KeyError) as exc:
                # NOSILENT: NATS did not answer, or answered what does not decode: this minute has no sizes; the
                # gauges keep their last values and the next sample tries again
                contain(exc, log=log, what="l2 sizes: the streams could not be read this sample")
            await asyncio.sleep(self._interval)
