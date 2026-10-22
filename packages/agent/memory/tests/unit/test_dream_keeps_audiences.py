"""a dream never merges memories that carry different host labels.

the labels in ``tags`` decide which audience may read a memory (a host's read filter), and the
package does not know what they mean, so a gist of memories from two audiences has no audience
it could be given: the cluster is skipped before the reflector is asked. driven through
``run_consolidation``, as the members-section test is.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from threetears.agent.memory.dream import DreamService
from threetears.agent.memory.types import MemoryConfig

_AID = uuid.UUID("11111111-1111-1111-1111-111111111111")


class _CountingReflectorFactory:
    """a reflector whose model counts the clusters it is asked to merge, and merges none."""

    def __init__(self) -> None:
        self.asked = 0

    async def create_chat_model(self, purpose: str = "consolidation") -> Any:
        _ = purpose
        model = MagicMock()

        async def _ainvoke(messages: list[Any], **kwargs: Any) -> Any:
            _ = messages, kwargs
            self.asked += 1
            response = MagicMock()
            response.content = "{}"
            return response

        model.ainvoke = _ainvoke
        return model


def _member(tags: list[str]) -> dict[str, Any]:
    return {
        "memory_id": uuid.uuid4(),
        "customer_id": None,
        "user_id": None,
        "conversation_id": uuid.uuid4(),
        "type_memory": "fact",
        "content": "the lighthouse keeper logs the tide",
        "embedding": [1.0, 0.0],
        "date_created": datetime.now(UTC),
        "tags": tags,
    }


async def _asked(tag_sets: list[list[str]]) -> int:
    reflector = _CountingReflectorFactory()
    memories = MagicMock()
    memories.find_active_for_consolidation = AsyncMock(return_value=[_member(tags) for tags in tag_sets])
    service = DreamService(
        config=MemoryConfig(),
        embedding_provider=MagicMock(),
        chat_model_factory=reflector,
        memories_collection=memories,
        consolidations_collection=MagicMock(),
    )
    await service.run_consolidation(_AID)
    return reflector.asked


async def test_memories_in_different_audiences_are_never_put_to_the_reflector() -> None:
    assert await _asked([["audience:staff"], ["audience:public"]]) == 0


async def test_memories_in_one_audience_are_still_merged() -> None:
    assert await _asked([["audience:staff"], ["audience:staff"]]) == 1


async def test_the_packages_own_bookkeeping_tags_are_not_an_audience() -> None:
    """``unattended`` and ``made_by:`` are this package's marks, not host labels: they never split a cluster."""
    assert await _asked([["audience:staff", "unattended", "made_by:agent"], ["audience:staff", "made_by:person"]]) == 1
