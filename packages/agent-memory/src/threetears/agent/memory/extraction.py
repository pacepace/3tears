"""Memory extraction -- distills conversation turns into persistent memories.

Three-layer gating (heuristic, rate limit, LLM worthiness), followed by
LLM-driven extraction, embedding, similar-memory lookup, and LLM resolution
(ADD/UPDATE/DELETE/NOOP). Fire-and-forget safe: extract() never raises.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable
from uuid import UUID

from langchain_core.messages import HumanMessage, SystemMessage
from uuid_utils import uuid7

from threetears.agent.memory.embedding import EmbeddingProvider
from threetears.agent.memory.prompts import ExtractionPrompts
from threetears.agent.memory.types import MemoryConfig, MemoryType
from threetears.core.logging import get_logger

_logger = get_logger(__name__)

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
        embedding_provider: EmbeddingProvider,
        chat_model_factory: ChatModelFactory,
        nats_client: Any = None,
        prompts: ExtractionPrompts | None = None,
        rate_limit_bucket: str = "ratelimits",
        summary_callback: Callable[[str, str], Awaitable[None]] | None = None,
    ) -> None:
        self._config = config
        self._embedding_provider = embedding_provider
        self._chat_model_factory = chat_model_factory
        self._nats_client = nats_client
        self._prompts = prompts or ExtractionPrompts()
        self._rate_limit_bucket = rate_limit_bucket
        self._summary_callback = summary_callback

    async def extract(
        self,
        pool: Any,
        user_id: UUID,
        conversation_id: UUID,
        message_id_source: UUID,
        user_message: str,
        assistant_response: str,
        turn_count: int,
    ) -> None:
        """Extract memories from a conversation turn. Fire-and-forget safe."""
        try:
            # Layer 1: Heuristic gates
            passed, reason = self._check_heuristic_gates(
                user_message,
                assistant_response,
                turn_count,
            )
            if not passed:
                _logger.debug(
                    "Memory extraction skipped by heuristic gate: %s",
                    reason,
                )
                return

            # Layer 2: Rate limit
            rate_passed, cooldown = await self._check_rate_limit(conversation_id)
            if not rate_passed:
                _logger.debug(
                    "Memory extraction skipped by rate limit, cooldown=%d",
                    cooldown,
                )
                return

            # Layer 3: Worthiness gate
            worthy, worthiness_reason = await self._check_worthiness(
                user_message,
                assistant_response,
            )
            if not worthy:
                _logger.debug(
                    "Memory extraction skipped by worthiness gate: %s",
                    worthiness_reason,
                )
                return

            # Extract candidates
            candidates_raw = await self._extract_candidates(
                user_message,
                assistant_response,
            )
            if not candidates_raw:
                _logger.debug("No memories extracted from conversation turn")
                return

            # Embed and find similar memories
            candidates: list[dict[str, Any]] = []
            for mem in candidates_raw:
                embedding, _tokens = await self._embedding_provider.embed_text(
                    mem["content"],
                )
                if embedding is None:
                    continue
                similar = await self._get_similar_memories(pool, embedding, user_id)
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

            # Resolve actions
            actions = await self._resolve_actions(candidates)

            # Execute
            await self._execute_actions(
                actions,
                candidates,
                pool,
                user_id,
                conversation_id,
                message_id_source,
            )

        except Exception as exc:
            _logger.error(
                "Memory extraction failed: %s",
                exc,
                exc_info=True,
            )

    def _check_heuristic_gates(
        self,
        user_message: str,
        assistant_response: str,
        turn_count: int,
    ) -> tuple[bool, str]:
        """Layer 1: Free, instant heuristic pre-filters."""
        if len(user_message.strip()) < self._config.extraction_min_user_message_length:
            return False, "user_message_too_short"
        if len(assistant_response.strip()) < self._config.extraction_min_assistant_response_length:
            return False, "assistant_response_too_short"
        if turn_count < self._config.extraction_min_conversation_turns:
            return False, f"too_few_turns ({turn_count})"
        return True, "passed"

    async def _check_rate_limit(
        self,
        conversation_id: UUID,
    ) -> tuple[bool, int]:
        """Layer 2: Rate limiter via NATS KV create()."""
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
            _logger.warning("Rate limit check failed, allowing extraction: %s", exc)
            return True, 0

    async def _check_worthiness(
        self,
        user_message: str,
        assistant_response: str,
    ) -> tuple[bool, str]:
        """Layer 3: Cheap LLM call to decide if extraction is worth running."""
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
        except (json.JSONDecodeError, KeyError):
            _logger.warning("Failed to parse worthiness gate response, allowing extraction")
            return True, "parse_error"
        except Exception as exc:
            _logger.warning("Worthiness gate LLM call failed, allowing extraction: %s", exc)
            return True, "llm_error"

    async def _extract_candidates(
        self,
        user_message: str,
        assistant_response: str,
    ) -> list[dict[str, str]]:
        """Use LLM to extract candidate memories from a conversation turn."""
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
        except (json.JSONDecodeError, KeyError):
            _logger.warning("Failed to parse memory extraction LLM response")
            return []
        except Exception as exc:
            _logger.warning("Memory extraction LLM call failed: %s", exc)
            return []

    async def _get_similar_memories(
        self,
        pool: Any,
        embedding: list[float],
        user_id: UUID,
    ) -> list[dict[str, Any]]:
        """Query pgvector for existing memories similar to a candidate."""
        embedding_str = json.dumps(embedding)
        rows = await pool.fetch(
            """
            SELECT memory_id, content, type_memory,
                   1 - (embedding <=> $1::vector) AS similarity
            FROM memories
            WHERE user_id = $2 AND is_deleted = false
            ORDER BY embedding <=> $1::vector
            LIMIT $3
            """,
            embedding_str,
            user_id,
            self._config.similar_memory_top_k,
        )
        return [
            {
                "memory_id": str(row["memory_id"]),
                "content": row["content"],
                "type_memory": row["type_memory"],
                "similarity": float(row["similarity"]),
            }
            for row in rows
            if float(row["similarity"]) >= self._config.similar_memory_threshold
        ]

    async def _resolve_actions(
        self,
        candidates: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Decide ADD/UPDATE/DELETE/NOOP for each candidate.

        Fast path: if no candidates have similar memories, return all ADD
        without an LLM call.
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

            # Any candidates not covered default to ADD
            for i in range(len(candidates)):
                if i not in seen_indices:
                    valid_actions.append({"index": i, "action": "ADD"})

            return valid_actions

        except (json.JSONDecodeError, KeyError):
            _logger.warning("Failed to parse memory resolution response, falling back to ADD all")
            return [{"index": i, "action": "ADD"} for i in range(len(candidates))]
        except Exception as exc:
            _logger.warning("Memory resolution LLM call failed: %s, falling back to ADD all", exc)
            return [{"index": i, "action": "ADD"} for i in range(len(candidates))]

    def _build_resolution_prompt(
        self,
        candidates: list[dict[str, Any]],
    ) -> str:
        """Build the candidates section for the resolution prompt."""
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

    async def _execute_actions(
        self,
        actions: list[dict[str, Any]],
        candidates: list[dict[str, Any]],
        pool: Any,
        user_id: UUID,
        conversation_id: UUID,
        message_id_source: UUID,
    ) -> None:
        """Execute ADD/UPDATE/DELETE/NOOP actions against the database."""
        now = datetime.now(UTC)

        for act in actions:
            idx = act["index"]
            action = act["action"]
            candidate = candidates[idx]

            try:
                if action == "ADD":
                    memory_id = uuid7()
                    embedding_str = json.dumps(candidate["embedding"])
                    await pool.execute(
                        """
                        INSERT INTO memories (
                            memory_id, user_id, conversation_id, message_id_source,
                            type_memory, content, embedding, is_deleted,
                            date_created, date_updated
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7::vector, false, $8, $8)
                        """,
                        memory_id,
                        user_id,
                        conversation_id,
                        message_id_source,
                        candidate["type"],
                        candidate["content"],
                        embedding_str,
                        now,
                    )
                    if self._summary_callback:
                        await self._summary_callback(
                            str(memory_id),
                            candidate["content"],
                        )

                elif action == "UPDATE":
                    updated_content = act["content"]
                    updated_type = act.get("type", candidate["type"])
                    new_embedding, _tokens = await self._embedding_provider.embed_text(
                        updated_content,
                    )
                    if new_embedding is not None:
                        embedding_str = json.dumps(new_embedding)
                        await pool.execute(
                            """
                            UPDATE memories
                            SET content = $1, type_memory = $2,
                                embedding = $3::vector, date_updated = $4
                            WHERE memory_id = $5::uuid AND user_id = $6
                              AND is_deleted = false
                            """,
                            updated_content,
                            updated_type,
                            embedding_str,
                            now,
                            act["memory_id"],
                            user_id,
                        )
                        if self._summary_callback:
                            await self._summary_callback(
                                act["memory_id"],
                                updated_content,
                            )

                elif action == "DELETE":
                    await pool.execute(
                        """
                        UPDATE memories
                        SET is_deleted = true, date_deleted = $1
                        WHERE memory_id = $2::uuid AND user_id = $3
                          AND is_deleted = false
                        """,
                        now,
                        act["memory_id"],
                        user_id,
                    )

                # NOOP: nothing to do

            except Exception as exc:
                _logger.warning(
                    "Failed to execute memory action %s: %s",
                    action,
                    exc,
                )
                continue

    @staticmethod
    def _get_response_content(response: Any) -> str:
        """Extract string content from an LLM response."""
        raw = response.content if hasattr(response, "content") else str(response)
        return (raw if isinstance(raw, str) else str(raw)).strip()

    @staticmethod
    def _strip_code_block(content: str) -> str:
        """Strip markdown code block wrappers if present."""
        if content.startswith("```"):
            lines = content.split("\n")
            if len(lines) > 2:
                return "\n".join(lines[1:-1])
        return content
