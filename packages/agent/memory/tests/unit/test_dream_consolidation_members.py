"""exact-output test for the members section of the consolidation prompt :class:`DreamService` sends.

each cluster member is its ``- [type]`` heading with every line of its content indented under
it and no blank line inside, fenced as material. driven through ``run_consolidation`` with the
reflector's request recorded.
"""

from __future__ import annotations

import re
import uuid
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from threetears.agent.memory.dream import DreamService
from threetears.agent.memory.types import MemoryConfig

_FENCE = re.compile(r"<untrusted nonce=(\w+)>\n(.*?)\n</untrusted nonce=\1>", re.DOTALL)
_AID = uuid.UUID("11111111-1111-1111-1111-111111111111")


class _RecordingReflectorFactory:
    """a reflector factory whose model records its messages and answers with nothing usable."""

    def __init__(self) -> None:
        self.asked: list[list[Any]] = []

    async def create_chat_model(self, purpose: str = "consolidation") -> Any:
        _ = purpose
        model = MagicMock()

        async def _ainvoke(messages: list[Any], **kwargs: Any) -> Any:
            _ = kwargs
            self.asked.append(list(messages))
            response = MagicMock()
            response.content = "{}"
            return response

        model.ainvoke = _ainvoke
        return model


def _member(content: str, type_memory: str) -> dict[str, Any]:
    return {
        "memory_id": uuid.uuid4(),
        "customer_id": None,
        "user_id": None,
        "conversation_id": uuid.uuid4(),
        "type_memory": type_memory,
        "content": content,
        "embedding": [1.0, 0.0],
        "date_created": datetime.now(UTC),
    }


async def _members_section(contents: list[tuple[str, str]]) -> list[str]:
    reflector = _RecordingReflectorFactory()
    memories = MagicMock()
    memories.find_active_for_consolidation = AsyncMock(return_value=[_member(c, t) for c, t in contents])
    service = DreamService(
        config=MemoryConfig(),
        embedding_provider=MagicMock(),
        chat_model_factory=reflector,
        memories_collection=memories,
        consolidations_collection=MagicMock(),
    )
    await service.run_consolidation(_AID)
    [[_system, human]] = reflector.asked
    [(_nonce, body)] = _FENCE.findall(str(human.content))
    return body.split("\n")


class TestConsolidationMembers:
    async def test_every_line_of_a_member_sits_indented_under_its_type_heading(self) -> None:
        section = await _members_section(
            [("first line\n\n  indented line\nlast line", "fact"), ("other\n\nthing", "preference")]
        )
        assert section == [
            "- [fact] first line",
            "    indented line",
            "  last line",
            "- [preference] other",
            "  thing",
        ]

    async def test_a_member_with_blank_content_still_shows_its_type_heading(self) -> None:
        section = await _members_section([("  \n\n", "fact"), ("real\ntext", "fact")])
        assert section == ["- [fact]", "- [fact] real", "  text"]
