"""Web answer: a provider's cited answer, on the search leaf (answers-requirements-design.md §10.5).

``threetears.web_answer`` takes one question and returns Brave's cited answer:
the provider's synthesis, named as the provider's in its first line, then the
pages it cites. It is :func:`threetears.search.bind_answer.bind_answer` over a
:class:`~threetears.search.adapters.brave_answers.BraveAnswersAdapter` and an
injected streaming transport, so bounded retry before the first byte, the
whole-stream bound and byte cap, the SSRF guards and the typed failures all
come from the leaf; nothing here opens a client or reads the environment.

**Prose and structure, as web_search has them.** ``content`` is the leaf's
rendering. :attr:`ToolResult.metadata` carries the typed answer -- text,
sources as candidates, anchors, spend, notices, dispositions -- or, on a
failure, the failure record, under
:data:`~threetears.search.contracts.ANSWER_METADATA_KEY` (AD8), which survives
the NATS hop because ``CallResponse.metadata`` carries it.

**Opt-in by its own alias, never inside ``web``** (AD13). Every call costs
money, so a persona granted ``web`` for a free SearXNG search does not
acquire this; a grant of ``answers`` does.

**The answer is not grounding text** (AR-G1). The sources are what a caller
reads before trusting a claim; the prose says so in its first line (AR-G2).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.tools import StructuredTool

from threetears.agent.tools.base_tool import MCPToolDefinition, TearsTool, ToolResult
from threetears.search.adapters.brave_answers import BraveAnswersAdapter
from threetears.search.bind_answer import RenderedAnswer, bind_answer
from threetears.search.contracts import AnswerRequest

if TYPE_CHECKING:
    from threetears.search.contracts import BudgetPort, RateLimiterPort, StreamingTransport

# ``create_web_answer_tool`` is reached by name, through ``register_builtins``'s
# lazy import, which keeps this module unimported until a registry asks for it.
__all__ = ["WebAnswerTool"]


def create_web_answer_tool(config: dict[str, Any], description: str) -> StructuredTool:
    """Factory: create a web answer tool.

    delegates to :func:`threetears.agent.tools.langchain_adapter.to_langchain_tool`
    so the StructuredTool path and the NATS-dispatched ToolServer path share
    :meth:`WebAnswerTool.execute` as their single execution body. Config must
    include ``api_key``, the host-resolved Brave key.

    Optional config keys, all deployment facts rather than per-call choices:
    ``transport`` (a prebuilt
    :class:`~threetears.search.contracts.transport.StreamingTransport`),
    ``budget``, ``limiter`` and ``timeout_seconds``; and ``_user_timezone``,
    the zone of the person the turn is with, which a host injects.

    :param config: tool configuration; ``api_key`` is required
    :ptype config: dict[str, Any]
    :param description: the description the LLM sees
    :ptype description: str
    :return: the wrapped tool
    :rtype: StructuredTool
    :raises ValueError: when ``api_key`` is absent
    """
    from threetears.agent.tools.langchain_adapter import to_langchain_tool

    api_key = config.get("api_key")
    if not api_key:
        raise ValueError("web_answer requires 'api_key' in config")
    return to_langchain_tool(
        WebAnswerTool(
            api_key=api_key,
            transport=config.get("transport"),
            timeout_seconds=config.get("timeout_seconds"),
            budget=config.get("budget"),
            limiter=config.get("limiter"),
            zone=config.get("_user_timezone") or None,
        ),
        description=description,
    )


class WebAnswerTool(TearsTool):
    """TearsTool wrapper for Brave's cited answers, over the search leaf.

    answers one question through a configured Brave key and returns rendered
    prose plus the typed answer under
    :data:`~threetears.search.contracts.ANSWER_METADATA_KEY`.
    """

    _INPUT_SCHEMA: dict[str, Any] = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": (
                    "The question to answer, in full. The answer comes back with the pages it cites; "
                    "read a page before relying on a claim."
                ),
            },
        },
        "required": ["question"],
    }

    def __init__(
        self,
        api_key: str,
        *,
        transport: StreamingTransport | None = None,
        timeout_seconds: float | None = None,
        budget: BudgetPort | None = None,
        limiter: RateLimiterPort | None = None,
        zone: str | None = None,
    ) -> None:
        """initialize web answer tool against one configured Brave key.

        The transport is injectable and defaults to the leaf's standalone
        transport: an answer arrives as a stream, and the host-side traced
        client buffers, so it cannot carry one (AD11). Nothing here reads the
        environment: the key is deployment config arriving through the caller
        (D21, SR-K1).

        :param api_key: the Brave API key, resolved by the host
        :ptype api_key: str
        :param transport: the streaming transport every answer is read
            through; None builds a
            :class:`~threetears.search.standalone.StandaloneTransport`
        :ptype transport: StreamingTransport | None
        :param timeout_seconds: bound for one whole answer (SR-G2); None
            applies the leaf's answer default
        :ptype timeout_seconds: float | None
        :param budget: the refusal authority consulted before the call and
            debited after it (D4, D5); None makes no consultation at all
        :ptype budget: BudgetPort | None
        :param limiter: the pacing seam for this key's egress (D8); None
            leaves the call unpaced. Brave's two-per-second default is this
            limiter's configuration
        :ptype limiter: RateLimiterPort | None
        :param zone: the IANA zone of the person the agent works for; a
            source's time of day is shown in it
        :ptype zone: str | None
        :return: nothing
        :rtype: None
        :raises ValueError: when ``api_key`` is empty
        """
        if transport is None:
            from threetears.search.standalone import StandaloneTransport

            transport = StandaloneTransport()
        self._transport = transport
        self._provider = BraveAnswersAdapter(api_key=api_key, transport=transport)
        self._timeout_seconds = timeout_seconds
        self._budget = budget
        self._limiter = limiter
        self._zone = zone

    async def execute(self, **kwargs: Any) -> ToolResult:
        """answer one question.

        Never raises: :func:`~threetears.search.bind_answer.bind_answer`
        carries D10's unconditional guarantee, so a refusal, a timeout or a
        broken stream all arrive here as a rendered failure that still
        accounts for what it spent. An empty question is answered before
        anything is sent, and costs nothing.

        :param kwargs: ``question``
        :ptype kwargs: Any
        :return: result carrying rendered prose, and the typed answer under
            the answer metadata key
        :rtype: ToolResult
        """
        question = str(kwargs.get("question") or "").strip()
        if not question:
            return _refused("web_answer needs a question: say what to answer.")
        rendered: RenderedAnswer = await bind_answer(
            AnswerRequest(question=question),
            provider=self._provider,
            timeout_seconds=self._timeout_seconds,
            budget=self._budget,
            limiter=self._limiter,
            egress=self._transport.egress_name,
            zone=self._zone,
        )
        return ToolResult(
            success=rendered.success,
            content=rendered.content,
            metadata=rendered.metadata,
            error=rendered.error,
        )

    def mcp_schema(self) -> MCPToolDefinition:
        """return MCP-compatible tool definition for web answer.

        :return: tool definition with name, version, description, input schema
        :rtype: MCPToolDefinition
        """
        result = MCPToolDefinition(
            name=self.mcp_name(),
            version=self.mcp_version(),
            description="answer a question from the web with Brave, citing the pages the answer rests on",
            input_schema=self._INPUT_SCHEMA,
        )
        return result

    def mcp_name(self) -> str:
        """return namespaced tool name.

        :return: namespaced tool name
        :rtype: str
        """
        return "threetears.web_answer"

    def mcp_version(self) -> str:
        """return tool version.

        :return: version string
        :rtype: str
        """
        return "1.0"


def _refused(message: str) -> ToolResult:
    """An answer that was not asked for, because the question is unusable; the message says what to send.

    :param message: what was wrong
    :ptype message: str
    :return: the failed result
    :rtype: ToolResult
    """
    return ToolResult(success=False, content=message, metadata={}, error=message)
