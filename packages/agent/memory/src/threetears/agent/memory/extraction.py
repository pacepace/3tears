"""Memory extraction -- distills conversation turns into persistent memories.

Gating (heuristic, a read of the conversation's cooldown, LLM worthiness,
then an atomic claim of the cooldown), followed by LLM-driven extraction,
embedding, similar-memory lookup, and LLM resolution (ADD/UPDATE/DELETE/NOOP).
extract() answers an :class:`ExtractionResult` saying what happened; it
raises only when its task is cancelled.

All memory-table writes go through :class:`MemoriesCollection` (save
new via the entity lifecycle; updates through
:meth:`MemoriesCollection.save_entity`; deletes through
:meth:`MemoriesCollection.delete` — hard-delete only under the
unified model, CASCADE FKs propagate to chunks + media); similar-
memory lookups use :meth:`MemoriesCollection.find_similar_for_dedup`.
The extractor holds no pool reference — Collections carry their pool
internally.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable
from uuid import UUID

from langchain_core.messages import HumanMessage, SystemMessage
from threetears.langgraph.fence import mint_nonce, untrusted_fence, with_fence_rules
from uuid_utils import uuid7

from threetears.agent.memory.authorize import (
    ACTION_MEMORY_EXTRACT,
    MemoryAuthorizerDependencies,
    authorize_memory_access,
)
from langchain_core.embeddings import Embeddings

from threetears.agent.memory.collections import MemoriesCollection
from threetears.agent.memory.embedding_utils import _safe_aembed_query
from threetears.agent.memory.entities import MemoryEntity
from threetears.agent.memory.prompts import ExtractionPrompts
from threetears.agent.memory.types import MemoryConfig, MemoryType
from threetears.nats.errors import KvError
from threetears.observe import get_logger, traced

__all__ = [
    "ChatModelFactory",
    "ExtractionGate",
    "ExtractionOutcome",
    "ExtractionResult",
    "MemoryExtractor",
]

log = get_logger(__name__)

_VALID_MEMORY_TYPES = {t.value for t in MemoryType}

#: JetStream's answer to a per-message TTL on a stream without ``allow_msg_ttl``
#: (``JSMessageTTLDisabledErr``, "per-message TTL is disabled"). observed on the
#: wire against nats-server; nats-py raises it as ``BadRequestError``, which
#: :class:`~threetears.nats.NatsKvBucket` wraps in ``KvError``.
_MSG_TTL_DISABLED_ERR_CODE = 10166


class ExtractionOutcome(StrEnum):
    """what one :meth:`MemoryExtractor.extract` call came to.

    :cvar STORED: the pipeline reached the write stage and at least one write
        landed, or resolution decided every candidate needed no write
    :cvar SKIPPED: a gate stopped the turn before anything was written; the
        result's ``gate`` names which one
    :cvar FAILED: something went wrong; the result's ``reason`` says what
    """

    STORED = "stored"
    SKIPPED = "skipped"
    FAILED = "failed"


class ExtractionGate(StrEnum):
    """the gate that stopped a skipped turn.

    :cvar HEURISTIC: message length or turn-count thresholds
    :cvar RATE_LIMIT: the conversation's extraction cooldown is running, or a
        concurrent turn claimed it first
    :cvar WORTHINESS: the worthiness model judged the turn not worth remembering
    :cvar NOTHING_FOUND: the extraction model found nothing to remember
    """

    HEURISTIC = "heuristic"
    RATE_LIMIT = "rate_limit"
    WORTHINESS = "worthiness"
    NOTHING_FOUND = "nothing_found"


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    """what :meth:`MemoryExtractor.extract` did with one conversation turn.

    :ivar outcome: stored, skipped or failed
    :ivar stored: memories written (added or updated) by this turn
    :ivar gate: the gate that stopped a skipped turn; ``None`` otherwise
    :ivar reason: human-readable detail -- the gate's own reason, the failure,
        or the per-action tally of a stored turn. for a person, never parsed
    """

    outcome: ExtractionOutcome
    stored: int = 0
    gate: ExtractionGate | None = None
    reason: str = ""


def _invoke_identity_kwargs(
    user_id: UUID | None,
    conversation_id: UUID | None,
) -> dict[str, Any]:
    """build the identity kwargs a gateway-routed chat model requires on invoke.

    the ``GatewayChatModel`` resolves the invoking ``user_id`` (required
    for ``model.invoke`` authorization + usage attribution) and the
    optional ``conversation_id`` from the ``ainvoke`` kwargs; without a
    ``user_id`` it raises ``user_id required``. the agent id is baked onto
    the model instance, so only these per-call dimensions are threaded here.
    omit ``None`` values so a stubbed/non-gateway model is unaffected.

    :param user_id: invoking user identifier
    :ptype user_id: UUID | None
    :param conversation_id: conversation identifier for usage attribution
    :ptype conversation_id: UUID | None
    :return: kwargs to splat into ``model.ainvoke``
    :rtype: dict[str, Any]
    """
    kwargs: dict[str, Any] = {}
    if user_id is not None:
        kwargs["user_id"] = user_id
    if conversation_id is not None:
        kwargs["conversation_id"] = conversation_id
    return kwargs


def _refuses_per_message_ttl(exc: KvError) -> bool:
    """whether a KV failure is the server refusing a per-message TTL.

    walks the ``__cause__`` chain for JetStream's ``per-message TTL is disabled``
    error code; the wrapper's ``KvError`` carries nats-py's ``BadRequestError``
    as its cause.

    :param exc: the KV failure
    :ptype exc: KvError
    :return: ``True`` when the stream refused the TTL for want of ``allow_msg_ttl``
    :rtype: bool
    """
    refused = False
    cause: BaseException | None = exc
    while cause is not None and not refused:
        refused = getattr(cause, "err_code", None) == _MSG_TTL_DISABLED_ERR_CODE
        cause = cause.__cause__
    return refused


@dataclass(slots=True)
class _ActionTally:
    """counts of what :meth:`MemoryExtractor._execute_actions` did with the resolved actions.

    :ivar added: new memories written
    :ivar updated: existing memories rewritten
    :ivar deleted: existing memories removed
    :ivar skipped: NOOPs, and UPDATE / DELETE targets that were absent or not the user's
    :ivar failed: actions that raised
    """

    added: int = 0
    updated: int = 0
    deleted: int = 0
    skipped: int = 0
    failed: int = 0

    def as_result(self) -> ExtractionResult:
        """the turn's result: failed when every attempted action raised, stored otherwise.

        :return: the extraction result this tally amounts to
        :rtype: ExtractionResult
        """
        applied = self.added + self.updated + self.deleted
        detail = (
            f"added={self.added} updated={self.updated} deleted={self.deleted} "
            f"unchanged={self.skipped} failed={self.failed}"
        )
        outcome = ExtractionOutcome.FAILED if self.failed and not applied else ExtractionOutcome.STORED
        return ExtractionResult(outcome=outcome, stored=self.added + self.updated, reason=detail)


@dataclass(slots=True)
class _Claim:
    """the cooldown key one turn claimed, so the turn can give it back when it comes to nothing.

    :ivar revision: the KV revision the claim created; ``None`` until a claim lands, and when the
        claim went through a replaced :meth:`MemoryExtractor.claim_rate_limit` that cannot say
    """

    revision: int | None = None


def _is_own(bound: Any, own: Any) -> bool:
    """whether a stage hook is still :class:`MemoryExtractor`'s own, not a replacement.

    the pipeline runs a stage's stricter private form only when the public hook was not
    overridden or replaced: a caller's own hook keeps its own contract.

    :param bound: the hook as the instance resolves it
    :ptype bound: Any
    :param own: :class:`MemoryExtractor`'s function for that hook
    :ptype own: Any
    :return: ``True`` when the hook is the class's own
    :rtype: bool
    """
    return getattr(bound, "__func__", None) is own


class _ExtractionModelFailed(RuntimeError):
    """the extraction model's call failed, or its reply was not the JSON list it was asked for.

    raised by the pipeline so the turn answers :attr:`ExtractionOutcome.FAILED` with the reason,
    where the public :meth:`MemoryExtractor.extract_candidates` hook keeps answering ``[]``.
    """


@runtime_checkable
class ChatModelFactory(Protocol):
    """Protocol for creating chat models for extraction purposes."""

    async def create_chat_model(self, purpose: str = "extraction") -> Any:
        """Create a chat model (returns LangChain BaseChatModel or compatible).

        purpose: "extraction", "worthiness", "resolution"
        """
        ...


class MemoryExtractor:
    """Extracts memorable facts from conversation turns and persists them."""

    def __init__(
        self,
        config: MemoryConfig,
        embedding_provider: Embeddings,
        chat_model_factory: ChatModelFactory,
        authorizer: MemoryAuthorizerDependencies,
        memories_collection: MemoriesCollection,
        nats_client: Any = None,
        prompts: ExtractionPrompts | None = None,
        rate_limit_bucket: str = "ratelimits",
        summary_callback: Callable[[str, str], Awaitable[None]] | None = None,
        on_memory_created: Callable[["MemoryEntity"], Awaitable[None]] | None = None,
        rate_limit_bucket_create_if_missing: bool = True,
    ) -> None:
        """initialize the extractor with the memories Collection + rbac authorizer.

        :param config: memory extraction configuration
        :ptype config: MemoryConfig
        :param embedding_provider: embedding provider for candidate vectors
        :ptype embedding_provider: Embeddings
        :param chat_model_factory: factory producing chat models for
            worthiness / extraction / resolution stages
        :ptype chat_model_factory: ChatModelFactory
        :param authorizer: rbac authorizer dependency bundle; required
        :ptype authorizer: MemoryAuthorizerDependencies
        :param memories_collection: three-tier memories collection;
            required. all memory writes / similar-memory lookups go
            through this collection
        :ptype memories_collection: MemoriesCollection
        :param nats_client: NATS KV client for per-conversation rate limits
        :ptype nats_client: Any
        :param prompts: extraction prompt bundle (defaults to
            :class:`ExtractionPrompts`)
        :ptype prompts: ExtractionPrompts | None
        :param rate_limit_bucket: KV bucket hosting the per-conversation
            rate-limit keys
        :ptype rate_limit_bucket: str
        :param summary_callback: optional coroutine invoked with
            ``(memory_id, content)`` after each ADD / UPDATE
        :ptype summary_callback: Callable[[str, str], Awaitable[None]] | None
        :param on_memory_created: optional coroutine invoked once per
            newly-committed memory, after ``save_entity`` returns. fires
            ONLY for the ADD action -- not UPDATE / DELETE -- so a
            consumer subscribed to "new memory created" sees one
            invocation per row that lands in storage. the consumer
            receives the full :class:`MemoryEntity` so it can build a
            push notification (server-sent event, websocket frame,
            slack message, etc.) with no follow-up database read. the
            entity is a detached snapshot of the row as written: its
            fields do not depend on the cache still holding the row,
            and setting one does not write it back. an
            exception inside the callback is logged at WARNING and
            swallowed so a flaky downstream push doesn't break the
            extraction pipeline (the row is already committed; the
            push is best-effort)
        :ptype on_memory_created: Callable[[MemoryEntity], Awaitable[None]] | None
        :param rate_limit_bucket_create_if_missing: ``True`` (the default) creates the rate-limit
            bucket when absent; ``False`` only BINDS one another identity declared and never
            issues STREAM.CREATE -- what an agent pod passes, since a pod holds no
            stream-management verb and the hub declares the bucket
        :ptype rate_limit_bucket_create_if_missing: bool
        """
        self._config = config
        self._embedding_provider = embedding_provider
        self._chat_model_factory = chat_model_factory
        self._nats_client = nats_client
        self._prompts = prompts or ExtractionPrompts()
        self._rate_limit_bucket = rate_limit_bucket
        self._rate_limit_bucket_create_if_missing = rate_limit_bucket_create_if_missing
        self._summary_callback = summary_callback
        self._on_memory_created = on_memory_created
        self._authorizer = authorizer
        self._memories = memories_collection

    @traced(record_args=False)
    async def extract(
        self,
        user_id: UUID,
        conversation_id: UUID,
        message_id_source: UUID,
        user_message: str,
        assistant_response: str,
        turn_count: int,
        *,
        agent_id: UUID,
        customer_id: UUID,
    ) -> ExtractionResult:
        """extract memories from one conversation turn and say what happened.

        gates run cheapest first: heuristics, a READ of the conversation's
        cooldown (so a turn inside it costs no worthiness call), the worthiness
        model, then an atomic CLAIM of the cooldown. the claim comes after
        worthiness on purpose: taking it earlier let an unworthy turn -- or one
        cancelled by the next message -- block the whole cooldown window, and
        creating it atomically means two turns racing past the read cannot both
        extract.

        safe to run as a background task: every failure is caught, logged and
        answered as :attr:`ExtractionOutcome.FAILED`. cancellation is the one
        exception -- it is logged once at WARNING and re-raised, because a
        cancelled task must stay cancelled.

        a turn that claimed the cooldown and then FAILED or was cancelled gives
        the key back (:meth:`_release_claim`), so a turn that stored nothing --
        one cancelled by the next message included -- does not block the window.
        a STORED or NOTHING_FOUND turn keeps it.

        :param user_id: user who sent message
        :ptype user_id: UUID
        :param conversation_id: conversation this turn belongs to
        :ptype conversation_id: UUID
        :param message_id_source: source message ID for extracted memories
        :ptype message_id_source: UUID
        :param user_message: raw user message text
        :ptype user_message: str
        :param assistant_response: raw assistant response text
        :ptype assistant_response: str
        :param turn_count: number of turns in conversation so far
        :ptype turn_count: int
        :param agent_id: agent UUID owning memory namespace (required)
        :ptype agent_id: UUID
        :param customer_id: customer UUID owning memory namespace (required)
        :ptype customer_id: UUID
        :return: stored count, the gate that stopped the turn, or the failure
        :rtype: ExtractionResult
        :raises asyncio.CancelledError: when the task running it is cancelled
        """
        claim = _Claim()
        try:
            result = await self._run_extraction(
                user_id=user_id,
                conversation_id=conversation_id,
                message_id_source=message_id_source,
                user_message=user_message,
                assistant_response=assistant_response,
                turn_count=turn_count,
                agent_id=agent_id,
                customer_id=customer_id,
                claim=claim,
            )
        except asyncio.CancelledError:
            # convert at border: log extra_data fields
            log.warning(
                "memory extraction cancelled",
                extra={"extra_data": {"conversation_id": str(conversation_id), "agent_id": str(agent_id)}},
            )
            # shielded: the release must land even though this task is being cancelled
            await asyncio.shield(self._release_claim(conversation_id, claim))
            raise
        except Exception as exc:
            log.error(
                "memory extraction failed: %s",
                exc,
                exc_info=True,
                extra={"extra_data": {"conversation_id": str(conversation_id), "agent_id": str(agent_id)}},
            )
            result = ExtractionResult(outcome=ExtractionOutcome.FAILED, reason=f"{type(exc).__name__}: {exc}")
        if result.outcome is ExtractionOutcome.FAILED:
            await self._release_claim(conversation_id, claim)
        return result

    async def _release_claim(self, conversation_id: UUID, claim: _Claim) -> None:
        """give back the cooldown key this turn claimed, and only that one.

        the delete is guarded by the revision the claim created, so a key another turn claimed
        after this one's expired is never removed. a release that cannot reach NATS is logged:
        the key then simply runs its course.

        :param conversation_id: conversation UUID the key belongs to
        :ptype conversation_id: UUID
        :param claim: what this turn claimed
        :ptype claim: _Claim
        """
        if claim.revision is None or self._nats_client is None:
            return
        try:
            bucket = await self._nats_client.kv_bucket(
                name=self._rate_limit_bucket, create_if_missing=self._rate_limit_bucket_create_if_missing
            )
            released = await bucket.delete(key=self._rate_limit_key(conversation_id), revision=claim.revision)
        except Exception as exc:
            log.warning("could not release the extraction cooldown of a turn that stored nothing: %s", exc)
            return
        claim.revision = None
        # convert at border: log extra_data fields
        log.debug(
            "released the extraction cooldown of a turn that stored nothing",
            extra={"extra_data": {"conversation_id": str(conversation_id), "released": released}},
        )

    async def _run_extraction(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        message_id_source: UUID,
        user_message: str,
        assistant_response: str,
        turn_count: int,
        agent_id: UUID,
        customer_id: UUID,
        claim: _Claim,
    ) -> ExtractionResult:
        """run the gated pipeline for one turn; failures propagate to :meth:`extract`.

        each gate that stops the turn is a guard clause answering its own
        :class:`ExtractionResult`.

        :param user_id: user who sent message
        :ptype user_id: UUID
        :param conversation_id: conversation this turn belongs to
        :ptype conversation_id: UUID
        :param message_id_source: source message ID for extracted memories
        :ptype message_id_source: UUID
        :param user_message: raw user message text
        :ptype user_message: str
        :param assistant_response: raw assistant response text
        :ptype assistant_response: str
        :param turn_count: number of turns in conversation so far
        :ptype turn_count: int
        :param agent_id: agent UUID owning memory namespace
        :ptype agent_id: UUID
        :param customer_id: customer UUID owning memory namespace
        :ptype customer_id: UUID
        :param claim: filled with the cooldown key's revision once this turn claims it
        :ptype claim: _Claim
        :return: what the turn came to
        :rtype: ExtractionResult
        :raises _ExtractionModelFailed: when the extraction model's call or reply failed
        :raises MemoryAccessDenied: when the agent may not extract into its namespace
        :raises KvError: when the rate-limit bucket refuses a per-key lifetime
        """
        # agent-internal turn extraction: authorize AGENT-ONLY so the
        # owner short-circuit gates it (the agent owns its memory
        # namespace by construction). ``caller_user_id`` is left None on
        # purpose -- passing the user would force user ∩ agent
        # intersection, which denies until the user holds a memory grant
        # that only a successful write creates (a bootstrap deadlock).
        # ``user_id`` still scopes the stored memory rows below.
        await authorize_memory_access(
            action=ACTION_MEMORY_EXTRACT,
            agent_id=agent_id,
            customer_id=customer_id,
            caller_user_id=None,
            caller_agent_id=agent_id,
            deps=self._authorizer,
        )

        passed, reason = self.check_heuristic_gates(user_message, assistant_response, turn_count)
        if not passed:
            log.debug("memory extraction skipped by heuristic gate: %s", reason)
            return ExtractionResult(outcome=ExtractionOutcome.SKIPPED, gate=ExtractionGate.HEURISTIC, reason=reason)

        open_window, cooldown = await self.check_rate_limit(conversation_id)
        if not open_window:
            log.debug("memory extraction skipped by rate limit, cooldown=%d", cooldown)
            return ExtractionResult(
                outcome=ExtractionOutcome.SKIPPED,
                gate=ExtractionGate.RATE_LIMIT,
                reason=f"cooldown running ({cooldown}s)",
            )

        worthy, worthiness_reason = await self.check_worthiness(
            user_message,
            assistant_response,
            user_id=user_id,
            conversation_id=conversation_id,
        )
        if not worthy:
            log.debug("memory extraction skipped by worthiness gate: %s", worthiness_reason)
            return ExtractionResult(
                outcome=ExtractionOutcome.SKIPPED,
                gate=ExtractionGate.WORTHINESS,
                reason=worthiness_reason,
            )

        if _is_own(self.claim_rate_limit, MemoryExtractor.claim_rate_limit):
            claimed, cooldown, claim.revision = await self._claim(conversation_id)
        else:
            claimed, cooldown = await self.claim_rate_limit(conversation_id)
        if not claimed:
            log.debug("memory extraction skipped: a concurrent turn claimed the cooldown, cooldown=%d", cooldown)
            return ExtractionResult(
                outcome=ExtractionOutcome.SKIPPED,
                gate=ExtractionGate.RATE_LIMIT,
                reason=f"cooldown claimed by a concurrent turn ({cooldown}s)",
            )

        if _is_own(self.extract_candidates, MemoryExtractor.extract_candidates):
            # the pipeline's own stage raises on a failed call, so an outage answers FAILED
            candidates_raw: list[dict[str, str]] = await self._extract_candidates(
                user_message, assistant_response, user_id=user_id, conversation_id=conversation_id
            )
        else:
            candidates_raw = await self.extract_candidates(
                user_message,
                assistant_response,
                user_id=user_id,
                conversation_id=conversation_id,
            )
        if not candidates_raw:
            log.debug("no memories extracted from conversation turn")
            return ExtractionResult(
                outcome=ExtractionOutcome.SKIPPED,
                gate=ExtractionGate.NOTHING_FOUND,
                reason="the extraction model found nothing to remember",
            )

        # each candidate's embedding and dedup search depend on nothing but
        # the candidate, so they run at once; the order of candidates is kept.
        prepared = await asyncio.gather(
            *(self._prepare_candidate(mem, user_id=user_id, agent_id=agent_id) for mem in candidates_raw)
        )
        candidates = [candidate for candidate in prepared if candidate is not None]
        if not candidates:
            return ExtractionResult(
                outcome=ExtractionOutcome.FAILED,
                reason=f"could not embed any of {len(candidates_raw)} candidate(s)",
            )

        actions = await self.resolve_actions(candidates, user_id=user_id, conversation_id=conversation_id)
        tally = await self._execute_actions(
            actions,
            candidates,
            user_id,
            conversation_id,
            message_id_source,
            agent_id=agent_id,
            customer_id=customer_id,
        )
        return tally.as_result()

    async def _prepare_candidate(
        self,
        mem: dict[str, str],
        *,
        user_id: UUID,
        agent_id: UUID,
    ) -> dict[str, Any] | None:
        """embed one extracted candidate and find the memories it may duplicate.

        :param mem: the candidate's ``type`` and ``content``
        :ptype mem: dict[str, str]
        :param user_id: owning user UUID (row filter for the dedup search)
        :ptype user_id: UUID
        :param agent_id: partition column on memories
        :ptype agent_id: UUID
        :return: the candidate with its embedding and similar memories, or ``None`` when it
            could not be embedded (the embedding helper logs why)
        :rtype: dict[str, Any] | None
        """
        embedding = await _safe_aembed_query(self._embedding_provider, mem["content"])
        if embedding is None:
            return None
        similar = await self._get_similar_memories(embedding, user_id, agent_id)
        return {
            "type": mem["type"],
            "content": mem["content"],
            "embedding": embedding,
            "similar_memories": similar,
        }

    def check_heuristic_gates(
        self,
        user_message: str,
        assistant_response: str,
        turn_count: int,
    ) -> tuple[bool, str]:
        """layer 1 extension point: free, instant heuristic pre-filters.

        public stage hook on :class:`MemoryExtractor`. override in
        subclasses or replace via duck typing to customize the
        heuristic gate; tests stub this to bypass length / turn
        thresholds. stability contract: signature and return shape
        are part of public api.

        :param user_message: raw user message
        :ptype user_message: str
        :param assistant_response: raw assistant response
        :ptype assistant_response: str
        :param turn_count: conversation turn count
        :ptype turn_count: int
        :return: (passed, reason) pair
        :rtype: tuple[bool, str]
        """
        if len(user_message.strip()) < self._config.extraction_min_user_message_length:
            return False, "user_message_too_short"
        if len(assistant_response.strip()) < self._config.extraction_min_assistant_response_length:
            return False, "assistant_response_too_short"
        if turn_count < self._config.extraction_min_conversation_turns:
            return False, f"too_few_turns ({turn_count})"
        return True, "passed"

    def _rate_limit_key(self, conversation_id: UUID) -> str:
        """the KV key holding one conversation's extraction cooldown.

        :param conversation_id: conversation UUID
        :ptype conversation_id: UUID
        :return: KV key
        :rtype: str
        """
        # convert at border: KV key
        return f"memory.last_extract.{conversation_id}"

    async def check_rate_limit(
        self,
        conversation_id: UUID,
    ) -> tuple[bool, int]:
        """layer 2 extension point: READ the conversation's extraction cooldown.

        public stage hook on :class:`MemoryExtractor`, run BEFORE worthiness so a
        turn inside the cooldown costs no worthiness call. it only reads: the key
        is taken by :meth:`claim_rate_limit`, after worthiness says yes, so a turn
        that stops here or at worthiness never blocks the next one. fail-open on
        NATS errors is part of the contract. stability contract: signature and
        return shape are part of public api.

        :param conversation_id: conversation UUID to rate-limit
        :ptype conversation_id: UUID
        :return: ``(True, 0)`` when no cooldown is running, ``(False, cooldown)``
            when one is
        :rtype: tuple[bool, int]
        """
        cooldown = self._config.extraction_rate_limit_cooldown_seconds
        if self._nats_client is None or cooldown <= 0:
            return True, 0
        try:
            bucket = await self._nats_client.kv_bucket(
                name=self._rate_limit_bucket, create_if_missing=self._rate_limit_bucket_create_if_missing
            )
            running = await bucket.get(key=self._rate_limit_key(conversation_id)) is not None
        except Exception as exc:
            log.warning("rate limit read failed, allowing extraction: %s", exc)
            running = False
        return (False, cooldown) if running else (True, 0)

    async def claim_rate_limit(
        self,
        conversation_id: UUID,
    ) -> tuple[bool, int]:
        """layer 4 extension point: atomically START the conversation's cooldown.

        public stage hook on :class:`MemoryExtractor`, run AFTER worthiness.
        a KV ``create`` lands only when no live key exists, so of two turns that
        both passed :meth:`check_rate_limit` exactly one claims the window.

        the lifetime rides the KEY (a per-message TTL), not the bucket: the
        bucket is opened by name alone, because a bucket-level ``ttl`` is the
        stream's ``max_age`` -- set once by whoever created the bucket, and on a
        shared bucket nothing to do with this cooldown. a bucket whose stream
        cannot carry a per-message TTL (``allow_msg_ttl`` off, on a handle that
        could not reconcile it) refuses the write; that is a deployment defect,
        logged at ERROR naming the bucket and raised -- never answered by a key
        that would outlive its cooldown. any other NATS failure fails open.
        stability contract: signature and return shape are part of public api.

        :param conversation_id: conversation UUID to rate-limit
        :ptype conversation_id: UUID
        :return: ``(True, 0)`` when this turn claimed the cooldown,
            ``(False, cooldown)`` when a concurrent turn already had
        :rtype: tuple[bool, int]
        :raises KvError: when the bucket refuses a per-key TTL
        """
        claimed, cooldown, _revision = await self._claim(conversation_id)
        return claimed, cooldown

    async def _claim(self, conversation_id: UUID) -> tuple[bool, int, int | None]:
        """:meth:`claim_rate_limit`, with the revision the claim created.

        :param conversation_id: conversation UUID to rate-limit
        :ptype conversation_id: UUID
        :return: ``(claimed, cooldown, revision)``; the revision is ``None`` when no key was
            written -- the limit is off, or NATS failed and the claim failed open
        :rtype: tuple[bool, int, int | None]
        :raises KvError: when the bucket refuses a per-key TTL
        """
        cooldown = self._config.extraction_rate_limit_cooldown_seconds
        if self._nats_client is None or cooldown <= 0:
            return True, 0, None
        revision: int | None = None
        bucket_name = self._rate_limit_bucket
        try:
            bucket = await self._nats_client.kv_bucket(
                name=bucket_name, create_if_missing=self._rate_limit_bucket_create_if_missing
            )
            bucket_name = bucket.name
            revision = await bucket.create(
                key=self._rate_limit_key(conversation_id),
                value=b"1",
                ttl=timedelta(seconds=cooldown),
            )
            claimed = revision is not None
        except KvError as exc:
            if not _refuses_per_message_ttl(exc):
                log.warning("rate limit claim failed, allowing extraction: %s", exc)
                claimed = True
            else:
                log.error(
                    "memory extraction rate-limit bucket %s refuses per-key TTLs (allow_msg_ttl is off), so no "
                    "cooldown can be set. open it with create_if_missing=True from an identity allowed to "
                    "update the stream -- that reconciles allow_msg_ttl in place -- or recreate it: %s",
                    bucket_name,
                    exc,
                    extra={"extra_data": {"bucket": bucket_name}},
                )
                raise
        except Exception as exc:
            log.warning("rate limit claim failed, allowing extraction: %s", exc)
            claimed = True
        return (True, 0, revision) if claimed else (False, cooldown, None)

    async def check_worthiness(
        self,
        user_message: str,
        assistant_response: str,
        *,
        user_id: UUID | None = None,
        conversation_id: UUID | None = None,
    ) -> tuple[bool, str]:
        """layer 3 extension point: cheap LLM call gating extraction.

        public stage hook on :class:`MemoryExtractor`. override in
        subclasses or replace via duck typing to customize the
        worthiness gate; tests stub this with a canned LLM response.
        fail-open on parse or LLM errors is part of the contract.
        stability contract: signature and return shape are part of
        public api.

        :param user_message: raw user message
        :ptype user_message: str
        :param assistant_response: raw assistant response
        :ptype assistant_response: str
        :return: (worthy, reason) pair
        :rtype: tuple[bool, str]
        """
        try:
            model = await self._chat_model_factory.create_chat_model(
                purpose="worthiness",
            )
            prompt = self._prompts.worthiness.format(
                user_message=user_message[:500],
                assistant_response_preview=assistant_response[:500],
            )
            response = await model.ainvoke(
                [
                    SystemMessage(
                        content="You decide whether a conversation turn holds something worth "
                        "remembering. Return only valid JSON.",
                    ),
                    HumanMessage(content=prompt),
                ],
                **_invoke_identity_kwargs(user_id, conversation_id),
            )
            content = self._get_response_content(response)
            result = json.loads(self._strip_code_block(content))
            worthy = result.get("worthy", False)
            reason = result.get("reason", "no_reason")
            return bool(worthy), str(reason)
        except json.JSONDecodeError, KeyError:
            log.warning("Failed to parse worthiness gate response, allowing extraction")
            return True, "parse_error"
        except Exception as exc:
            log.warning("Worthiness gate LLM call failed, allowing extraction: %s", exc)
            return True, "llm_error"

    @traced()
    async def extract_candidates(
        self,
        user_message: str,
        assistant_response: str,
        *,
        user_id: UUID | None = None,
        conversation_id: UUID | None = None,
    ) -> list[dict[str, str]]:
        """extraction stage extension point: LLM-driven candidate extraction.

        public stage hook on :class:`MemoryExtractor`. override in
        subclasses or replace via duck typing to customize candidate
        extraction; tests stub this with canned candidate lists.
        returns ``[]`` on parse or LLM errors (fail-closed on this
        stage is part of the contract). stability contract:
        signature and return shape are part of public api.

        :param user_message: raw user message
        :ptype user_message: str
        :param assistant_response: raw assistant response
        :ptype assistant_response: str
        :return: list of candidate memory dicts
        :rtype: list[dict[str, str]]
        """
        try:
            candidates = await self._extract_candidates(
                user_message, assistant_response, user_id=user_id, conversation_id=conversation_id
            )
        except _ExtractionModelFailed as exc:
            log.warning("%s", exc)
            candidates = []
        return candidates

    async def _extract_candidates(
        self,
        user_message: str,
        assistant_response: str,
        *,
        user_id: UUID | None,
        conversation_id: UUID | None,
    ) -> list[dict[str, str]]:
        """:meth:`extract_candidates`, raising when the model's call or reply fails.

        ``[]`` means the model answered and found nothing; a failed call, a reply that is not
        JSON, or JSON that is not a list raises, so the pipeline can tell an outage from a quiet
        turn.

        :param user_message: raw user message
        :ptype user_message: str
        :param assistant_response: raw assistant response
        :ptype assistant_response: str
        :param user_id: the user, for the model call's identity
        :ptype user_id: UUID | None
        :param conversation_id: the conversation, for the model call's identity
        :ptype conversation_id: UUID | None
        :return: list of candidate memory dicts
        :rtype: list[dict[str, str]]
        :raises _ExtractionModelFailed: when the call or its reply failed
        """
        try:
            model = await self._chat_model_factory.create_chat_model(purpose="extraction")
            prompt = self._prompts.extraction.format(
                user_message=user_message[:2000],
                assistant_response=assistant_response[:2000],
            )
            response = await model.ainvoke(
                [
                    SystemMessage(
                        content="You write down what is worth remembering from a conversation. Return only valid JSON.",
                    ),
                    HumanMessage(content=prompt),
                ],
                **_invoke_identity_kwargs(user_id, conversation_id),
            )
            content = self._get_response_content(response)
            memories = json.loads(self._strip_code_block(content))
        except json.JSONDecodeError as exc:
            raise _ExtractionModelFailed(f"the extraction model's reply was not valid JSON: {exc}") from exc
        except Exception as exc:
            raise _ExtractionModelFailed(f"the extraction model call failed: {type(exc).__name__}: {exc}") from exc
        if not isinstance(memories, list):
            raise _ExtractionModelFailed(f"the extraction model's reply was {type(memories).__name__}, not a list")
        valid: list[dict[str, str]] = []
        for mem in memories:
            if (
                isinstance(mem, dict)
                and isinstance(mem.get("type"), str)
                and isinstance(mem.get("content"), str)
                and mem["type"] in _VALID_MEMORY_TYPES
                and mem["content"].strip()
            ):
                valid.append({"type": mem["type"], "content": mem["content"].strip()})
        return valid

    async def _get_similar_memories(
        self,
        embedding: list[float],
        user_id: UUID,
        agent_id: UUID,
    ) -> list[dict[str, Any]]:
        """Query existing memories similar to a candidate via the Collection.

        :param embedding: candidate embedding vector
        :ptype embedding: list[float]
        :param user_id: owning user UUID (row filter)
        :ptype user_id: UUID
        :param agent_id: partition column on memories; required
        :ptype agent_id: UUID
        :return: list of similar memory dicts
        :rtype: list[dict[str, Any]]
        """
        rows = await self._memories.find_similar_for_dedup(
            user_id=user_id,
            agent_id=agent_id,
            embedding=embedding,
            top_k=self._config.similar_memory_top_k,
            threshold=self._config.similar_memory_threshold,
        )
        return [
            {
                "memory_id": str(row["memory_id"]),
                "content": row["content"],
                "type_memory": row["type_memory"],
                "similarity": float(row["similarity"]),
            }
            for row in rows
        ]

    @traced()
    async def resolve_actions(
        self,
        candidates: list[dict[str, Any]],
        *,
        user_id: UUID | None = None,
        conversation_id: UUID | None = None,
    ) -> list[dict[str, Any]]:
        """resolution stage extension point: decide ADD/UPDATE/DELETE/NOOP.

        public stage hook on :class:`MemoryExtractor`. override in
        subclasses or replace via duck typing to customize action
        resolution; tests stub this with canned action lists. fast
        path when no candidate has similar memories returns all ADD
        without an LLM call. fail-closed fallback to all-ADD on parse
        or LLM errors is part of the contract. stability contract:
        signature and return shape are part of public api.

        :param candidates: candidate memories with similar-memory context
        :ptype candidates: list[dict[str, Any]]
        :return: list of validated action dicts
        :rtype: list[dict[str, Any]]
        """
        has_any_similar = any(c["similar_memories"] for c in candidates)
        if not has_any_similar:
            return [{"index": i, "action": "ADD"} for i in range(len(candidates))]

        try:
            model = await self._chat_model_factory.create_chat_model(
                purpose="resolution",
            )
            prompt = self._build_resolution_prompt(candidates)
            response = await model.ainvoke(
                with_fence_rules(
                    [
                        SystemMessage(
                            content="You decide what to do with new memories. Return only valid JSON.",
                        ),
                        HumanMessage(content=prompt),
                    ]
                ),
                **_invoke_identity_kwargs(user_id, conversation_id),
            )
            content = self._get_response_content(response)
            actions = json.loads(self._strip_code_block(content))
            if not isinstance(actions, list):
                return [{"index": i, "action": "ADD"} for i in range(len(candidates))]

            valid_actions: list[dict[str, Any]] = []
            seen_indices: set[int] = set()
            for act in actions:
                if not isinstance(act, dict):
                    continue
                idx = act.get("index")
                action = act.get("action", "").upper()
                if not isinstance(idx, int) or idx < 0 or idx >= len(candidates):
                    continue
                if action not in ("ADD", "UPDATE", "DELETE", "NOOP"):
                    continue
                if idx in seen_indices:
                    continue
                seen_indices.add(idx)

                validated: dict[str, Any] = {"index": idx, "action": action}
                if action == "UPDATE":
                    mid = act.get("memory_id")
                    new_content = act.get("content")
                    new_type = act.get("type")
                    if mid and new_content:
                        validated["memory_id"] = str(mid)
                        validated["content"] = str(new_content).strip()
                        validated["type"] = (
                            str(new_type)
                            if new_type and str(new_type) in _VALID_MEMORY_TYPES
                            else candidates[idx]["type"]
                        )
                    else:
                        validated["action"] = "NOOP"
                elif action == "DELETE":
                    mid = act.get("memory_id")
                    if mid:
                        validated["memory_id"] = str(mid)
                    else:
                        validated["action"] = "NOOP"
                valid_actions.append(validated)

            for i in range(len(candidates)):
                if i not in seen_indices:
                    valid_actions.append({"index": i, "action": "ADD"})

            return valid_actions

        except json.JSONDecodeError, KeyError:
            log.warning("Failed to parse memory resolution response, falling back to ADD all")
            return [{"index": i, "action": "ADD"} for i in range(len(candidates))]
        except Exception as exc:
            log.warning("Memory resolution LLM call failed: %s, falling back to ADD all", exc)
            return [{"index": i, "action": "ADD"} for i in range(len(candidates))]

    def _build_resolution_prompt(
        self,
        candidates: list[dict[str, Any]],
    ) -> str:
        """Build the candidates section for the resolution prompt.

        :param candidates: candidates with similar-memory context
        :ptype candidates: list[dict[str, Any]]
        :return: prompt string
        :rtype: str
        """
        # An existing memory is read back from storage, and what was stored came
        # from conversations and tools: it is fenced as material.
        nonce = mint_nonce()
        sections = []
        for i, c in enumerate(candidates):
            section = f"Candidate {i}:\n  Type: {c['type']}\n  Content: {c['content']}"
            if c["similar_memories"]:
                existing = "\n".join(
                    f"    - ID: {m['memory_id']} | [{m['type_memory']}] "
                    f"{m['content']} (similarity: {m['similarity']:.0%})"
                    for m in c["similar_memories"]
                )
                section += f"\n  Existing similar memories:\n{untrusted_fence(nonce, existing)}"
            else:
                section += "\n  No similar existing memories found."
            sections.append(section)

        return self._prompts.resolution.format(
            candidates_section="\n\n".join(sections),
        )

    @traced()
    async def _execute_actions(
        self,
        actions: list[dict[str, Any]],
        candidates: list[dict[str, Any]],
        user_id: UUID,
        conversation_id: UUID,
        message_id_source: UUID,
        *,
        agent_id: UUID,
        customer_id: UUID,
    ) -> _ActionTally:
        """execute ADD/UPDATE/DELETE/NOOP actions via the Collection, counting what landed.

        a failing action is logged and counted, and the rest still run.

        :param actions: resolved action list from resolve_actions
        :ptype actions: list[dict[str, Any]]
        :param candidates: candidate memories with embeddings
        :ptype candidates: list[dict[str, Any]]
        :param user_id: user who owns these memories
        :ptype user_id: UUID
        :param conversation_id: conversation this extraction belongs to
        :ptype conversation_id: UUID
        :param message_id_source: source message ID
        :ptype message_id_source: UUID
        :param agent_id: agent UUID to tag new memories with
        :ptype agent_id: UUID
        :param customer_id: customer UUID to tag new memories with
        :ptype customer_id: UUID
        :return: counts of added, updated, deleted, unchanged and failed actions
        :rtype: _ActionTally
        """
        now = datetime.now(UTC)
        tally = _ActionTally()
        #: (memory id, content) for every row written, summarised after the loop.
        summaries: list[tuple[str, str]] = []

        for act in actions:
            idx = act["index"]
            action = act["action"]
            candidate = candidates[idx]

            try:
                if action == "ADD":
                    memory_id = uuid7()
                    new_data: dict[str, Any] = {
                        "memory_id": memory_id,
                        "agent_id": agent_id,
                        "customer_id": customer_id,
                        "user_id": user_id,
                        "conversation_id": conversation_id,
                        "message_id_source": message_id_source,
                        "type_memory": candidate["type"],
                        "content": candidate["content"],
                        "embedding": candidate["embedding"],
                        # honor the per-user salience_seed knob (design §3);
                        # without this the DB server default (0.5) applies and
                        # a tuned seed is a silent no-op. Mirrors how dream.py
                        # threads consolidation_gist_salience_seed + how
                        # agent/intention threads its salience_seed.
                        "salience": self._config.salience_seed,
                        "date_created": now,
                        "date_updated": now,
                    }
                    new_entity: MemoryEntity = self._memories.create(new_data)
                    await self._memories.save_entity(new_entity)
                    tally.added += 1
                    summaries.append(
                        (
                            str(memory_id),  # convert at border: summary_callback Callable[[str, str], ...] contract
                            candidate["content"],
                        )
                    )
                    if self._on_memory_created:
                        # Best-effort push: the row is already committed,
                        # so a failing callback (downstream WS down,
                        # transport flake, consumer bug) must not break
                        # the extraction pipeline. Log + swallow.
                        #
                        # The log shape (event key + structured fields)
                        # is the metric surface for ops: Loki LogQL like
                        # ``rate(count_over_time({container="..."}
                        # |= "on_memory_created callback failed"
                        # [5m]))`` aggregates failures into a per-minute
                        # rate, sub-categorizable by ``error_type``.
                        # Keeps the framework dep-free of Prometheus
                        # while giving metric-grade observability --
                        # downstream products that prefer
                        # prometheus-client wrap the callback and
                        # increment their own Counter on the exception.
                        try:
                            # A snapshot of the committed row, not
                            # ``new_entity``. The live handle reads its
                            # fields from L1 only: a turn that surfaces
                            # this memory bumps its salience, which evicts
                            # the row, and the handle then reads every
                            # field as None. That happened while the
                            # summary callback, which ran here, waited
                            # minutes on a model.
                            await self._on_memory_created(
                                MemoryEntity(dict(new_data), is_new=False),
                            )
                        except Exception as cb_exc:
                            # convert at border: callback-failed log extra_data fields
                            log_user_id = str(user_id)
                            log_conversation_id = str(conversation_id)
                            log.warning(
                                "on_memory_created callback failed",
                                extra={
                                    "extra_data": {
                                        "memory_id": str(memory_id),
                                        "user_id": log_user_id,
                                        "conversation_id": log_conversation_id,
                                        "error_type": type(cb_exc).__name__,
                                        "error": str(cb_exc),
                                    },
                                },
                            )

                elif action == "UPDATE":
                    updated_content = act["content"]
                    updated_type = act.get("type", candidate["type"])
                    new_embedding = await _safe_aembed_query(
                        self._embedding_provider,
                        updated_content,
                    )
                    if new_embedding is None:
                        # the embedding helper already logged why; the rewrite cannot land without one.
                        tally.failed += 1
                        continue
                    memory_uuid = UUID(act["memory_id"])
                    update_entity: MemoryEntity | None = await self._memories.get(
                        (agent_id, memory_uuid),
                    )
                    if update_entity is None or update_entity.user_id != user_id:
                        tally.skipped += 1
                        continue
                    update_entity.content = updated_content
                    update_entity.type_memory = updated_type
                    update_entity.embedding = new_embedding
                    await self._memories.save_entity(update_entity)
                    tally.updated += 1
                    summaries.append((act["memory_id"], updated_content))

                elif action == "DELETE":
                    memory_uuid = UUID(act["memory_id"])
                    delete_entity: MemoryEntity | None = await self._memories.get(
                        (agent_id, memory_uuid),
                    )
                    if delete_entity is None or delete_entity.user_id != user_id:
                        tally.skipped += 1
                        continue
                    # Hard-delete under the unified model; v017's CASCADE
                    # FKs propagate to any chunks + media attached.
                    await self._memories.delete((agent_id, memory_uuid))
                    tally.deleted += 1

                else:
                    tally.skipped += 1

            except Exception as exc:
                log.warning(
                    "Failed to execute memory action %s: %s",
                    action,
                    exc,
                )
                tally.failed += 1
                continue

        await self._summarise(summaries)
        return tally

    async def _summarise(self, written: list[tuple[str, str]]) -> None:
        """run the summary callback for every row written, all at once.

        each call is a model call in a consumer that summarises (metallm's
        waited out a 120 s deadline twice in one run), so one after another
        they cost the sum and at once the longest. they run after every
        action and push, so a slow summary holds neither. a failing one is
        logged; its row is already written.

        :param written: ``(memory_id, content)`` per row added or updated
        :ptype written: list[tuple[str, str]]
        :return: nothing
        :rtype: None
        """
        callback = self._summary_callback
        if callback is None or not written:
            return None
        outcomes = await asyncio.gather(
            *(callback(memory_id, content) for memory_id, content in written),
            return_exceptions=True,
        )
        for (memory_id, _content), outcome in zip(written, outcomes, strict=True):
            if isinstance(outcome, BaseException):
                log.warning(
                    "summary callback failed",
                    extra={
                        "extra_data": {
                            "memory_id": memory_id,
                            "error_type": type(outcome).__name__,
                            "error": str(outcome),
                        },
                    },
                )
        return None

    @staticmethod
    def _get_response_content(response: Any) -> str:
        """Extract string content from an LLM response.

        :param response: raw LLM response object
        :ptype response: Any
        :return: stripped content string
        :rtype: str
        """
        raw = response.content if hasattr(response, "content") else str(response)
        return (raw if isinstance(raw, str) else str(raw)).strip()

    @staticmethod
    def _strip_code_block(content: str) -> str:
        """Strip markdown code block wrappers if present.

        :param content: raw content possibly wrapped in markdown fence
        :ptype content: str
        :return: inner content with fence stripped
        :rtype: str
        """
        content = content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            if len(lines) > 2:
                content = "\n".join(lines[1:-1]).strip()
        # the model sometimes prepends reasoning prose to the JSON (seen on the
        # worthiness gate); extract the outermost object/array so json.loads
        # sees only the payload. picks whichever bracket type opens first.
        obj_start = content.find("{")
        arr_start = content.find("[")
        starts = [i for i in (obj_start, arr_start) if i != -1]
        if starts:
            start = min(starts)
            closer = "}" if content[start] == "{" else "]"
            end = content.rfind(closer)
            if end > start:
                content = content[start : end + 1]
        return content
