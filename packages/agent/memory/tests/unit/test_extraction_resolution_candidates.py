"""exact-output tests for the candidates section of the resolution prompt :class:`MemoryExtractor` sends.

each candidate is ``Candidate i:``, ``  Type:`` and a ``  Content:`` line with every further
line of its content indented under it, no blank line inside the candidate; an existing similar
memory is joined to one line. driven through ``resolve_actions`` with the model's request recorded.
"""

from __future__ import annotations

import re
from typing import Any
from unittest.mock import MagicMock

from threetears.agent.memory.extraction import MemoryExtractor
from threetears.agent.memory.types import MemoryConfig

_FENCE = re.compile(r"<untrusted nonce=(\w+)>\n(.*?)\n</untrusted nonce=\1>", re.DOTALL)


class _RecordingFactory:
    """a chat-model factory whose model records its messages and answers with an empty list."""

    def __init__(self) -> None:
        self.asked: list[list[Any]] = []

    async def create_chat_model(self, purpose: str = "resolution") -> Any:
        _ = purpose
        model = MagicMock()

        async def _ainvoke(messages: list[Any], **kwargs: Any) -> Any:
            _ = kwargs
            self.asked.append(list(messages))
            response = MagicMock()
            response.content = "[]"
            return response

        model.ainvoke = _ainvoke
        return model


def _candidate(content: str, similar: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {"type": "fact", "content": content, "embedding": [1.0], "similar_memories": similar or []}


def _similar(content: str) -> dict[str, Any]:
    return {"memory_id": "m1", "content": content, "type_memory": "fact", "similarity": 0.9}


async def _prompt(candidates: list[dict[str, Any]]) -> str:
    factory = _RecordingFactory()
    extractor = MemoryExtractor(
        config=MemoryConfig(),
        embedding_provider=MagicMock(),
        chat_model_factory=factory,  # type: ignore[arg-type]
        authorizer=MagicMock(),
        memories_collection=MagicMock(),
    )
    await extractor.resolve_actions(candidates)
    [[_system, human]] = factory.asked
    return str(human.content)


class TestResolutionCandidates:
    async def test_every_line_of_candidate_content_sits_indented_under_its_content_line(self) -> None:
        text = await _prompt([_candidate("first line\n\n  indented line\nlast line", [_similar("x")])])
        assert "Candidate 0:\n  Type: fact\n  Content: first line\n      indented line\n    last line\n" in text

    async def test_no_blank_line_falls_inside_a_candidate(self) -> None:
        text = await _prompt([_candidate("a\n\nb", [_similar("x")]), _candidate("c\n\nd", [_similar("y")])])
        [first, second] = text.split("Candidate 0:\n")[1].split("\n\nCandidate 1:\n")
        assert "\n\n" not in first.split("  Existing similar memories:")[0]
        assert second.startswith("  Type: fact\n  Content: c\n    d\n  Existing similar memories:")

    async def test_a_candidate_with_blank_content_still_shows_its_content_line(self) -> None:
        text = await _prompt([_candidate(" \n\n", [_similar("x")])])
        assert "Candidate 0:\n  Type: fact\n  Content:\n  Existing similar memories:" in text

    async def test_a_similar_memory_is_joined_to_one_line(self) -> None:
        text = await _prompt([_candidate("c", [_similar("one\n\n  two\nthree")])])
        [(_nonce, body)] = _FENCE.findall(text)
        assert body == "    - ID: m1 | [fact] one two three (similarity: 90%)"
