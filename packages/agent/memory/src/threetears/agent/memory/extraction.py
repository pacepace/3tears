"""Memory extraction -- distills conversation turns into persistent memories.

Three-layer gating (heuristic, rate limit, LLM worthiness), followed by
LLM-driven extraction, embedding, similar-memory lookup, and LLM resolution
(ADD/UPDATE/DELETE/NOOP). Fire-and-forget safe: extract() never raises.

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

import json
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable
from uuid import UUID

from langchain_core.messages import HumanMessage, SystemMessage
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
from threetears.observe import get_logger, traced

__all__ = [
    "ChatModelFactory",
    "MemoryExtractor",
]

log = get_logger(__name__)

_VALID_MEMORY_TYPES = {t.value for t in MemoryType}


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
            slack message, etc.) with no follow-up database read. an
            exception inside the callback is logged at WARNING and
            swallowed so a flaky downstream push doesn't break the
            extraction pipeline (the row is already committed; the
            push is best-effort)
        :ptype on_memory_created: Callable[[MemoryEntity], Awaitable[None]] | None
        """
        self._config = config
        self._embedding_provider = embedding_provider
        self._chat_model_factory = chat_model_factory
        self._nats_client = nats_client
        self._prompts = prompts or ExtractionPrompts()
        self._rate_limit_bucket = rate_limit_bucket
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
    ) -> None:
        """Extract memories from conversation turn. Fire-and-forget safe.

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
        :return: nothing
        :rtype: None
        :raises: never (fire-and-forget safe, logs errors internally)
        """
        try:
            await authorize_memory_access(
                action=ACTION_MEMORY_EXTRACT,
                agent_id=agent_id,
                customer_id=customer_id,
                caller_user_id=user_id,
                caller_agent_id=agent_id,
                deps=self._authorizer,
            )

            passed, reason = self.check_heuristic_gates(
                user_message,
                assistant_response,
                turn_count,
            )
            if not passed:
                log.debug(
                    "Memory extraction skipped by heuristic gate: %s",
                    reason,
                )
                return

            rate_passed, cooldown = await self.check_rate_limit(conversation_id)
            if not rate_passed:
                log.debug(
                    "Memory extraction skipped by rate limit, cooldown=%d",
                    cooldown,
                )
                return

            worthy, worthiness_reason = await self.check_worthiness(
                user_message,
                assistant_response,
            )
            if not worthy:
                log.debug(
                    "Memory extraction skipped by worthiness gate: %s",
                    worthiness_reason,
                )
                return

            candidates_raw = await self.extract_candidates(
                user_message,
                assistant_response,
            )
            if not candidates_raw:
                log.debug("No memories extracted from conversation turn")
                return

            candidates: list[dict[str, Any]] = []
            for mem in candidates_raw:
                embedding = await _safe_aembed_query(
                    self._embedding_provider,
                    mem["content"],
                )
                if embedding is None:
                    continue
                similar = await self._get_similar_memories(
                    embedding,
                    user_id,
                    agent_id,
                )
                candidates.append(
                    {
                        "type": mem["type"],
                        "content": mem["content"],
                        "embedding": embedding,
                        "similar_memories": similar,
                    }
                )

            if not candidates:
                return

            actions = await self.resolve_actions(candidates)

            await self._execute_actions(
                actions,
                candidates,
                user_id,
                conversation_id,
                message_id_source,
                agent_id=agent_id,
                customer_id=customer_id,
            )

        except Exception as exc:
            log.error(
                "Memory extraction failed: %s",
                exc,
                exc_info=True,
            )

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

    async def check_rate_limit(
        self,
        conversation_id: UUID,
    ) -> tuple[bool, int]:
        """layer 2 extension point: rate limiter via NATS KV create().

        public stage hook on :class:`MemoryExtractor`. override in
        subclasses or replace via duck typing to customize rate
        limiting; tests stub this to bypass the NATS bucket. fail-open
        on NATS errors is part of the contract. stability contract:
        signature and return shape are part of public api.

        :param conversation_id: conversation UUID to rate-limit
        :ptype conversation_id: UUID
        :return: (passed, cooldown) pair
        :rtype: tuple[bool, int]
        """
        if self._nats_client is None:
            return True, 0
        try:
            key = f"memory.last_extract.{conversation_id}"
            bucket = self._nats_client.bucket_name(self._rate_limit_bucket)
            was_created = await self._nats_client.create(bucket, key, b"1")
            if not was_created:
                return False, self._config.extraction_rate_limit_cooldown_seconds
            return True, 0
        except Exception as exc:
            log.warning("Rate limit check failed, allowing extraction: %s", exc)
            return True, 0

    async def check_worthiness(
        self,
        user_message: str,
        assistant_response: str,
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
                        content="You evaluate whether conversation turns contain "
                        "memorable user information. Return only valid JSON.",
                    ),
                    HumanMessage(content=prompt),
                ]
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
            model = await self._chat_model_factory.create_chat_model(
                purpose="extraction",
            )
            prompt = self._prompts.extraction.format(
                user_message=user_message[:2000],
                assistant_response=assistant_response[:2000],
            )
            response = await model.ainvoke(
                [
                    SystemMessage(
                        content="You extract structured memories from conversations. Return only valid JSON.",
                    ),
                    HumanMessage(content=prompt),
                ]
            )
            content = self._get_response_content(response)
            memories = json.loads(self._strip_code_block(content))
            if not isinstance(memories, list):
                return []

            valid: list[dict[str, str]] = []
            for mem in memories:
                if (
                    isinstance(mem, dict)
                    and isinstance(mem.get("type"), str)
                    and isinstance(mem.get("content"), str)
                    and mem["type"] in _VALID_MEMORY_TYPES
                    and mem["content"].strip()
                ):
                    valid.append(
                        {
                            "type": mem["type"],
                            "content": mem["content"].strip(),
                        }
                    )
            return valid
        except json.JSONDecodeError, KeyError:
            log.warning("Failed to parse memory extraction LLM response")
            return []
        except Exception as exc:
            log.warning("Memory extraction LLM call failed: %s", exc)
            return []

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
                [
                    SystemMessage(
                        content="You are a memory manager that decides how to handle "
                        "new memories. Return only valid JSON.",
                    ),
                    HumanMessage(content=prompt),
                ]
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
        sections = []
        for i, c in enumerate(candidates):
            section = f"Candidate {i}:\n  Type: {c['type']}\n  Content: {c['content']}"
            if c["similar_memories"]:
                section += "\n  Existing similar memories:"
                for m in c["similar_memories"]:
                    section += (
                        f"\n    - ID: {m['memory_id']} | [{m['type_memory']}] "
                        f"{m['content']} (similarity: {m['similarity']:.0%})"
                    )
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
    ) -> None:
        """Execute ADD/UPDATE/DELETE/NOOP actions via the Collection.

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
        :return: nothing
        :rtype: None
        """
        now = datetime.now(UTC)

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
                        "date_created": now,
                        "date_updated": now,
                    }
                    new_entity: MemoryEntity = self._memories.create(new_data)
                    await self._memories.save_entity(new_entity)
                    if self._summary_callback:
                        await self._summary_callback(
                            str(memory_id),  # convert at border: summary_callback Callable[[str, str], ...] contract
                            candidate["content"],
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
                            await self._on_memory_created(new_entity)
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
                    if new_embedding is not None:
                        memory_uuid = UUID(act["memory_id"])
                        update_entity: MemoryEntity | None = await self._memories.get(
                            (agent_id, memory_uuid),
                        )
                        if update_entity is None or update_entity.user_id != user_id:
                            continue
                        update_entity.content = updated_content
                        update_entity.type_memory = updated_type
                        update_entity.embedding = new_embedding
                        await self._memories.save_entity(update_entity)
                        if self._summary_callback:
                            await self._summary_callback(
                                act["memory_id"],
                                updated_content,
                            )

                elif action == "DELETE":
                    memory_uuid = UUID(act["memory_id"])
                    delete_entity: MemoryEntity | None = await self._memories.get(
                        (agent_id, memory_uuid),
                    )
                    if delete_entity is None or delete_entity.user_id != user_id:
                        continue
                    # Hard-delete under the unified model; v017's CASCADE
                    # FKs propagate to any chunks + media attached.
                    await self._memories.delete((agent_id, memory_uuid))

            except Exception as exc:
                log.warning(
                    "Failed to execute memory action %s: %s",
                    action,
                    exc,
                )
                continue

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
        if content.startswith("```"):
            lines = content.split("\n")
            if len(lines) > 2:
                return "\n".join(lines[1:-1])
        return content
