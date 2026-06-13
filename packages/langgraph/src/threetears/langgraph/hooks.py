"""hook protocols for agent_node and tool_node customization.

the canonical :func:`threetears.langgraph.nodes.agent_node` and
:func:`threetears.langgraph.nodes.tool_node` are deliberately small.
downstream callers (the aibots-agents SDK, audience_builder, any
future agent runtime) extend them via hook objects rather than by
forking the node body. a hook is a narrow protocol -- pure
before/after transforms for agent_node, and pure start/end/heartbeat
emitters for tool_node. hooks do NOT own graph state; they reach
into ``configurable`` or receive explicit state objects.

::

    class MyHook:
        async def before_invoke(self, messages, config, state):
            return rewritten_messages, config

        async def after_invoke(self, response, config, state):
            return response

    agent_node(state, config, hooks=[MyHook()])

composition: :func:`compose_agent_node_hooks` and
:func:`compose_tool_node_hooks` return adapters that fan a node's
single hook-call site across a sequence of hooks in order. the
primitives accept a ``Sequence`` of hooks directly; the compose
helpers exist for callers that want to assemble a pipeline once and
pass it through multiple graph builds.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable
from weakref import WeakKeyDictionary

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from threetears.langgraph.caching import (
    ChatModelCapabilities,
    annotate_system_prompt,
    compute_tool_key,
    detect_capabilities,
    extract_cache_usage,
    should_bind_tools_fresh,
)
from threetears.observe import get_logger

__all__ = [
    "AgentNodeHook",
    "PromptCachingHook",
    "ToolNodeHook",
    "compose_agent_node_hooks",
    "compose_tool_node_hooks",
    "summarize_args",
]

log = get_logger(__name__)


@runtime_checkable
class AgentNodeHook(Protocol):
    """protocol for before/after hooks on :func:`agent_node`.

    hooks are pure transformations: :meth:`before_invoke` rewrites the
    message list and/or the config just before the model is invoked;
    :meth:`after_invoke` inspects (and optionally rewrites) the raw
    model response before it is wrapped into a state-update dict.
    neither method is required -- default implementations return their
    inputs unchanged, so concrete hooks override only what they need.

    contract:

    - :meth:`before_invoke` MUST NOT mutate ``messages[0]`` when it is a
      :class:`langchain_core.messages.SystemMessage`. the system
      prefix is the stable cache-key head and downstream
      prompt-caching integrations (langgraph-task-01) rely on it.
    - hooks MUST NOT hold graph state across invocations in closure
      variables. persistent state belongs on ``configurable`` or an
      explicit state object passed through the graph.
    """

    async def before_invoke(
        self,
        messages: list[BaseMessage],
        config: RunnableConfig,
        state: dict[str, Any],
    ) -> tuple[list[BaseMessage], RunnableConfig]:
        """rewrite messages and/or config before :meth:`chat_model.ainvoke`.

        default implementation returns inputs unchanged.

        :param messages: full message list assembled by agent_node,
            including system prompt at index 0 when present
        :ptype messages: list[BaseMessage]
        :param config: LangGraph runtime config dict
        :ptype config: RunnableConfig
        :param state: current agent state (read-only view for hooks)
        :ptype state: dict[str, Any]
        :return: ``(messages, config)`` tuple for the next hook or the
            model invocation
        :rtype: tuple[list[BaseMessage], RunnableConfig]
        """
        ...

    async def after_invoke(
        self,
        response: Any,
        config: RunnableConfig,
        state: dict[str, Any],
    ) -> Any:
        """inspect or rewrite the raw model response after ``ainvoke``.

        default implementation returns the response unchanged.

        :param response: raw return value from ``chat_model.ainvoke``
            (typically an :class:`AIMessage` or compatible chunk)
        :ptype response: Any
        :param config: LangGraph runtime config dict
        :ptype config: RunnableConfig
        :param state: current agent state (read-only view for hooks)
        :ptype state: dict[str, Any]
        :return: response object (possibly rewritten) for the next hook
            or for wrapping into the state-update dict
        :rtype: Any
        """
        ...


@runtime_checkable
class ToolNodeHook(Protocol):
    """protocol for dispatch-level hooks on :func:`tool_node`.

    :meth:`before_dispatch` rewrites or filters the tool_call list
    immediately before dispatch. :meth:`on_tool_start` and
    :meth:`on_tool_end` are pure emitters -- they observe the call
    without returning a value. :meth:`on_heartbeat` observes periodic
    liveness ticks during long tool calls; implementations that do
    not care about heartbeats leave the method as a no-op. as with
    :class:`AgentNodeHook`, hooks MUST NOT hold graph state across
    invocations in closure variables.
    """

    async def before_dispatch(
        self,
        tool_calls: list[dict[str, Any]],
        config: RunnableConfig,
        state: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], RunnableConfig]:
        """rewrite tool_calls and/or config before dispatch.

        default implementation returns inputs unchanged. rewriting
        lets hooks add synthesized tool calls, filter disallowed
        ones, or amend arguments.

        :param tool_calls: list of tool_call dicts from the last
            :class:`AIMessage`; each entry has ``id``, ``name``,
            ``args`` keys
        :ptype tool_calls: list[dict[str, Any]]
        :param config: LangGraph runtime config dict
        :ptype config: RunnableConfig
        :param state: current agent state (read-only view for hooks)
        :ptype state: dict[str, Any]
        :return: ``(tool_calls, config)`` tuple for the next hook or
            dispatch
        :rtype: tuple[list[dict[str, Any]], RunnableConfig]
        """
        ...

    async def on_tool_start(
        self,
        tool_call: dict[str, Any],
        config: RunnableConfig,
        state: dict[str, Any],
    ) -> None:
        """fired immediately before a single tool is dispatched.

        default implementation is a no-op. emitters publish
        ``tool_call_start`` envelopes, open timing spans, etc.

        :param tool_call: single tool_call dict with ``id``, ``name``,
            ``args`` keys
        :ptype tool_call: dict[str, Any]
        :param config: LangGraph runtime config dict
        :ptype config: RunnableConfig
        :param state: current agent state (read-only view for hooks)
        :ptype state: dict[str, Any]
        :return: nothing
        :rtype: None
        """
        ...

    async def on_tool_end(
        self,
        tool_call: dict[str, Any],
        result: Any,
        success: bool,
        elapsed_ms: int,
        config: RunnableConfig,
        state: dict[str, Any],
    ) -> None:
        """fired immediately after a single tool call completes.

        default implementation is a no-op. ``success=False`` fires when
        the tool raised or when the tool name did not resolve.

        :param tool_call: single tool_call dict with ``id``, ``name``,
            ``args`` keys
        :ptype tool_call: dict[str, Any]
        :param result: the value returned by the tool (or an error
            string when ``success=False``)
        :ptype result: Any
        :param success: whether tool dispatch succeeded
        :ptype success: bool
        :param elapsed_ms: wall-clock milliseconds from dispatch to
            completion
        :ptype elapsed_ms: int
        :param config: LangGraph runtime config dict
        :ptype config: RunnableConfig
        :param state: current agent state (read-only view for hooks)
        :ptype state: dict[str, Any]
        :return: nothing
        :rtype: None
        """
        ...

    async def on_heartbeat(
        self,
        tool_call: dict[str, Any],
        elapsed_seconds: float,
        config: RunnableConfig,
        state: dict[str, Any],
    ) -> None:
        """fired periodically while a long tool call is running.

        default implementation is a no-op. emitters publish
        ``tool_call_progress`` envelopes so that upstream stream
        consumers see a liveness signal and do not trip on gap
        timeouts. tool_node installs a heartbeat loop only when at
        least one hook implements a non-default ``on_heartbeat``;
        implementations that do not care pay no cost.

        :param tool_call: single tool_call dict with ``id``, ``name``,
            ``args`` keys
        :ptype tool_call: dict[str, Any]
        :param elapsed_seconds: wall-clock seconds since tool dispatch
            started; monotonic
        :ptype elapsed_seconds: float
        :param config: LangGraph runtime config dict
        :ptype config: RunnableConfig
        :param state: current agent state (read-only view for hooks)
        :ptype state: dict[str, Any]
        :return: nothing
        :rtype: None
        """
        ...


class _ComposedAgentNodeHook:
    """adapter that fans agent_node hook calls across a sequence.

    hidden implementation behind :func:`compose_agent_node_hooks`.
    calls :meth:`AgentNodeHook.before_invoke` on each wrapped hook in
    order, threading the ``(messages, config)`` tuple through. fires
    :meth:`AgentNodeHook.after_invoke` in the same order (NOT
    reversed -- hooks are peers, not wrapping decorators; the sequence
    order is the one contract).
    """

    def __init__(self, hooks: Sequence[AgentNodeHook]) -> None:
        """capture the hook sequence for later dispatch.

        :param hooks: sequence of :class:`AgentNodeHook` instances
        :ptype hooks: Sequence[AgentNodeHook]
        """
        self._hooks: tuple[AgentNodeHook, ...] = tuple(hooks)

    async def before_invoke(
        self,
        messages: list[BaseMessage],
        config: RunnableConfig,
        state: dict[str, Any],
    ) -> tuple[list[BaseMessage], RunnableConfig]:
        """sequentially invoke :meth:`before_invoke` on each hook.

        :param messages: message list to thread through hooks
        :ptype messages: list[BaseMessage]
        :param config: config dict to thread through hooks
        :ptype config: RunnableConfig
        :param state: read-only state view passed to each hook
        :ptype state: dict[str, Any]
        :return: final ``(messages, config)`` after all hooks applied
        :rtype: tuple[list[BaseMessage], RunnableConfig]
        """
        current_messages = messages
        current_config = config
        for hook in self._hooks:
            current_messages, current_config = await hook.before_invoke(
                current_messages,
                current_config,
                state,
            )
        return current_messages, current_config

    async def after_invoke(
        self,
        response: Any,
        config: RunnableConfig,
        state: dict[str, Any],
    ) -> Any:
        """sequentially invoke :meth:`after_invoke` on each hook.

        :param response: response object to thread through hooks
        :ptype response: Any
        :param config: config dict passed to each hook
        :ptype config: RunnableConfig
        :param state: read-only state view passed to each hook
        :ptype state: dict[str, Any]
        :return: final response after all hooks applied
        :rtype: Any
        """
        current_response = response
        for hook in self._hooks:
            current_response = await hook.after_invoke(
                current_response,
                config,
                state,
            )
        return current_response


class _ComposedToolNodeHook:
    """adapter that fans tool_node hook calls across a sequence.

    hidden implementation behind :func:`compose_tool_node_hooks`.
    sequences ``before_dispatch`` threads; ``on_tool_start``,
    ``on_tool_end``, ``on_heartbeat`` fan out in order and are
    independent (failure in one emitter is swallowed at the node
    level -- emitters should not crash the graph).
    """

    def __init__(self, hooks: Sequence[ToolNodeHook]) -> None:
        """capture the hook sequence for later dispatch.

        :param hooks: sequence of :class:`ToolNodeHook` instances
        :ptype hooks: Sequence[ToolNodeHook]
        """
        self._hooks: tuple[ToolNodeHook, ...] = tuple(hooks)

    async def before_dispatch(
        self,
        tool_calls: list[dict[str, Any]],
        config: RunnableConfig,
        state: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], RunnableConfig]:
        """sequentially invoke :meth:`before_dispatch` on each hook.

        :param tool_calls: tool_call list to thread through hooks
        :ptype tool_calls: list[dict[str, Any]]
        :param config: config dict to thread through hooks
        :ptype config: RunnableConfig
        :param state: read-only state view passed to each hook
        :ptype state: dict[str, Any]
        :return: final ``(tool_calls, config)`` after all hooks
        :rtype: tuple[list[dict[str, Any]], RunnableConfig]
        """
        current_tool_calls = tool_calls
        current_config = config
        for hook in self._hooks:
            current_tool_calls, current_config = await hook.before_dispatch(
                current_tool_calls,
                current_config,
                state,
            )
        return current_tool_calls, current_config

    async def on_tool_start(
        self,
        tool_call: dict[str, Any],
        config: RunnableConfig,
        state: dict[str, Any],
    ) -> None:
        """fan :meth:`on_tool_start` across every hook in order.

        :param tool_call: single tool_call dict
        :ptype tool_call: dict[str, Any]
        :param config: config dict
        :ptype config: RunnableConfig
        :param state: read-only state view
        :ptype state: dict[str, Any]
        :return: nothing
        :rtype: None
        """
        for hook in self._hooks:
            await hook.on_tool_start(tool_call, config, state)

    async def on_tool_end(
        self,
        tool_call: dict[str, Any],
        result: Any,
        success: bool,
        elapsed_ms: int,
        config: RunnableConfig,
        state: dict[str, Any],
    ) -> None:
        """fan :meth:`on_tool_end` across every hook in order.

        :param tool_call: single tool_call dict
        :ptype tool_call: dict[str, Any]
        :param result: tool result value
        :ptype result: Any
        :param success: whether tool dispatch succeeded
        :ptype success: bool
        :param elapsed_ms: wall-clock milliseconds for the call
        :ptype elapsed_ms: int
        :param config: config dict
        :ptype config: RunnableConfig
        :param state: read-only state view
        :ptype state: dict[str, Any]
        :return: nothing
        :rtype: None
        """
        for hook in self._hooks:
            await hook.on_tool_end(
                tool_call,
                result,
                success,
                elapsed_ms,
                config,
                state,
            )

    async def on_heartbeat(
        self,
        tool_call: dict[str, Any],
        elapsed_seconds: float,
        config: RunnableConfig,
        state: dict[str, Any],
    ) -> None:
        """fan :meth:`on_heartbeat` across every hook in order.

        :param tool_call: single tool_call dict
        :ptype tool_call: dict[str, Any]
        :param elapsed_seconds: seconds since tool dispatch started
        :ptype elapsed_seconds: float
        :param config: config dict
        :ptype config: RunnableConfig
        :param state: read-only state view
        :ptype state: dict[str, Any]
        :return: nothing
        :rtype: None
        """
        for hook in self._hooks:
            await hook.on_heartbeat(tool_call, elapsed_seconds, config, state)


def compose_agent_node_hooks(
    hooks: Sequence[AgentNodeHook],
) -> AgentNodeHook:
    """compose a sequence of :class:`AgentNodeHook` into a single hook.

    returns an adapter that calls :meth:`before_invoke` and
    :meth:`after_invoke` on each wrapped hook in sequence order.
    passing an empty sequence returns an adapter that is a pure
    no-op pass-through.

    :param hooks: sequence of :class:`AgentNodeHook` instances to
        fan across
    :ptype hooks: Sequence[AgentNodeHook]
    :return: single :class:`AgentNodeHook` that dispatches to all
        wrapped hooks in order
    :rtype: AgentNodeHook
    """
    return _ComposedAgentNodeHook(hooks)


def compose_tool_node_hooks(
    hooks: Sequence[ToolNodeHook],
) -> ToolNodeHook:
    """compose a sequence of :class:`ToolNodeHook` into a single hook.

    returns an adapter that calls :meth:`before_dispatch`,
    :meth:`on_tool_start`, :meth:`on_tool_end`, and
    :meth:`on_heartbeat` on each wrapped hook in sequence order.
    passing an empty sequence returns an adapter that is a pure
    no-op pass-through.

    :param hooks: sequence of :class:`ToolNodeHook` instances to
        fan across
    :ptype hooks: Sequence[ToolNodeHook]
    :return: single :class:`ToolNodeHook` that dispatches to all
        wrapped hooks in order
    :rtype: ToolNodeHook
    """
    return _ComposedToolNodeHook(hooks)


@dataclass
class _BoundModelCache:
    """cached tuple of ``(tool_key, bound_model)`` per chat-model instance.

    internal representation behind the module-level
    :data:`_BOUND_MODEL_CACHE`. a stable :class:`dataclass` rather
    than a bare tuple keeps field access readable at the call site
    and makes adding future fields (e.g. ``last_used_at`` for
    eviction) non-breaking.

    :param tool_key: 16-hex-char digest matching the bound tools at
        bind time
    :ptype tool_key: str
    :param bound_model: the object returned by
        ``chat_model.bind_tools(sorted_tools)``; may be the original
        model when no tools are bound
    :ptype bound_model: Any
    """

    tool_key: str
    bound_model: Any


_BOUND_MODEL_CACHE: WeakKeyDictionary[Any, _BoundModelCache] = WeakKeyDictionary()
"""module-level cache keyed on the original chat-model instance.

using a :class:`WeakKeyDictionary` means the cache entry is
evicted automatically when the chat_model is garbage-collected;
this is the right semantic for long-lived agent pods that
occasionally rebuild their chat_model (config reload, rotation)
and should not leak stale bound-model references.

the dict is intentionally module-level rather than per-hook so
that multiple :class:`PromptCachingHook` instances wired into
the same process share the memoization -- the node creates a
fresh hook per graph build in some code paths, and a per-hook
cache would rebind tools on every build.
"""


class PromptCachingHook:
    """annotate the system prompt and memoize tool binding.

    :class:`AgentNodeHook` implementation. before each
    ``ainvoke``:

    1. detects the chat model's caching capabilities via
       :func:`threetears.langgraph.caching.detect_capabilities`.
    2. replaces the bare-string :class:`SystemMessage` at index 0
       (inserted by :func:`threetears.langgraph.nodes.agent_node`)
       with the structured-content form carrying
       ``cache_control={"type": "ephemeral"}`` when the model
       supports anthropic prompt caching; leaves it as-is
       otherwise.
    3. computes a stable tool-key digest
       (:func:`threetears.langgraph.caching.compute_tool_key`),
       consults the module-level :data:`_BOUND_MODEL_CACHE`, and
       either reuses the cached bound-model reference or calls
       ``chat_model.bind_tools(sorted_tools)`` fresh. when the
       cache hits, it rewrites ``configurable["chat_model"]`` to
       the pre-bound model and empties ``configurable["tools"]``
       so the node's own binding branch becomes a no-op.

    after each ``ainvoke``:

    4. runs :func:`threetears.langgraph.caching.extract_cache_usage`
       on the response and stashes the normalized dict on
       ``response.usage_metadata["cache_usage"]`` so downstream
       callers (tests, gateway telemetry) read the cache-hit
       counts without shape-juggling.

    for non-caching chat models the hook still performs the
    tool-binding memoization path -- that's a plain latency win
    with no provider-side caching interaction.
    """

    def __init__(self) -> None:
        """initialize the hook with no captured state.

        the hook is deliberately stateless -- all memoization lives
        on the module-level :data:`_BOUND_MODEL_CACHE`. instances
        are cheap to create and safe to drop at any time.

        :return: nothing
        :rtype: None
        """

    async def before_invoke(
        self,
        messages: list[BaseMessage],
        config: RunnableConfig,
        state: dict[str, Any],
    ) -> tuple[list[BaseMessage], RunnableConfig]:
        """annotate the system prompt and swap in a memoized bound model.

        the hook reads the original ``chat_model`` and ``tools``
        from ``configurable`` (the node has not yet captured them
        because the node re-reads them after the hook chain, see
        :func:`threetears.langgraph.nodes.agent_node`) and, when
        the capability detection flags caching support, rewrites
        ``messages[0]`` to the structured-content form. the tool
        memoization path runs regardless of caching support.

        :param messages: full message list including system prefix
        :ptype messages: list[BaseMessage]
        :param config: runtime config dict
        :ptype config: RunnableConfig
        :param state: agent state dict (read-only view)
        :ptype state: dict[str, Any]
        :return: ``(messages, config)`` with the possibly-annotated
            system prompt and possibly-swapped chat_model
        :rtype: tuple[list[BaseMessage], RunnableConfig]
        """
        configurable_raw = config.get("configurable", {})
        configurable: dict[str, Any] = dict(configurable_raw)
        chat_model = configurable.get("chat_model")
        tools = list(configurable.get("tools", []) or [])

        result_messages = list(messages)
        result_config: RunnableConfig = config

        if chat_model is None:
            # no model to annotate against; pass the inputs through
            # rather than raise. the node itself raises on missing
            # chat_model a few lines later.
            return result_messages, result_config

        caps = detect_capabilities(chat_model)
        result_messages = _rewrite_system_prompt_for_cache(result_messages, caps)

        bound_model, rewrite_tools = _memoize_bound_model(chat_model, tools)

        configurable["chat_model"] = bound_model
        configurable["tools"] = rewrite_tools
        # preserve any other hook wiring (_hooks, call_context, nc,
        # heartbeat) under the spread.
        result_config = {**config, "configurable": configurable}
        return result_messages, result_config

    async def after_invoke(
        self,
        response: Any,
        config: RunnableConfig,
        state: dict[str, Any],
    ) -> Any:
        """normalize cache-usage counters onto the response.

        sets ``response.usage_metadata["cache_usage"]`` to the
        dict produced by
        :func:`threetears.langgraph.caching.extract_cache_usage`.
        when the response has no ``usage_metadata`` the hook
        attaches a fresh one carrying only the ``cache_usage``
        block; the helper already handles the all-zero case so
        the presence of the key is uniform for downstream readers.

        :param response: model response
        :ptype response: Any
        :param config: runtime config dict
        :ptype config: RunnableConfig
        :param state: agent state dict (read-only view)
        :ptype state: dict[str, Any]
        :return: response with ``usage_metadata["cache_usage"]``
            populated
        :rtype: Any
        """
        usage = extract_cache_usage(response)
        existing = getattr(response, "usage_metadata", None)
        if isinstance(existing, dict):
            existing["cache_usage"] = usage
        else:
            try:
                response.usage_metadata = {"cache_usage": usage}
            except AttributeError:
                # some response shapes (e.g. plain mocks) refuse
                # attribute assignment; logging once keeps the hook
                # non-fatal and keeps loki readable.
                log.debug(
                    "response %s does not accept usage_metadata assignment",
                    type(response).__name__,
                )
        return response


def _rewrite_system_prompt_for_cache(
    messages: list[BaseMessage],
    caps: ChatModelCapabilities,
) -> list[BaseMessage]:
    """replace the system message at index 0 with an annotated copy.

    when ``messages[0]`` is a :class:`SystemMessage` and the
    capability record flags anthropic cache_control support, the
    function extracts the string content, pipes it through
    :func:`annotate_system_prompt`, and returns a new list with
    the annotated message at index 0. messages already carrying
    structured content are left alone (idempotency when the hook
    runs twice). non-caching caps return the list unchanged.

    :param messages: full message list; index 0 may or may not be
        a :class:`SystemMessage`
    :ptype messages: list[BaseMessage]
    :param caps: capability record from :func:`detect_capabilities`
    :ptype caps: ChatModelCapabilities
    :return: message list with index 0 optionally rewritten
    :rtype: list[BaseMessage]
    """
    result = list(messages)
    if not caps.supports_anthropic_cache_control:
        return result
    if not result or not isinstance(result[0], SystemMessage):
        return result
    existing = result[0]
    if isinstance(existing.content, list):
        # already structured; assume caller (or a prior run of
        # this hook) placed cache_control on it.
        return result
    prompt_text = existing.content if isinstance(existing.content, str) else str(existing.content)
    result[0] = annotate_system_prompt(prompt_text, caps)
    return result


def _memoize_bound_model(
    chat_model: Any,
    tools: list[Any],
) -> tuple[Any, list[Any]]:
    """return the bound model and the tool list to pass downstream.

    when ``tools`` is non-empty, calls :func:`compute_tool_key`
    and either reuses the cached bound model (when
    :func:`should_bind_tools_fresh` returns False) or calls
    ``chat_model.bind_tools(sorted_tools)`` and stores the result
    on :data:`_BOUND_MODEL_CACHE`. on the cache-hit path the
    returned tool list is empty so the node's own binding branch
    becomes a no-op and the cached binding is the one used.

    when ``tools`` is empty, the function returns
    ``(chat_model, [])`` unchanged -- there is nothing to bind.

    :param chat_model: original chat-model instance (the dict key
        for :data:`_BOUND_MODEL_CACHE`)
    :ptype chat_model: Any
    :param tools: list of tool instances to bind
    :ptype tools: list[Any]
    :return: ``(model_for_node, tools_for_node)`` tuple
    :rtype: tuple[Any, list[Any]]
    """
    if not tools:
        return chat_model, []
    sorted_tools = sorted(tools, key=lambda t: getattr(t, "name", ""))
    current_key = compute_tool_key(sorted_tools)
    result: tuple[Any, list[Any]]
    try:
        cached = _BOUND_MODEL_CACHE.get(chat_model)
    except TypeError:
        # the cache is identity-keyed via a WeakKeyDictionary, which
        # requires a hashable model. a non-conformant (unhashable) chat
        # model must NOT crash the caller's turn over a missed optimization
        # — degrade to bind-fresh + warn so the turn still completes. the
        # right fix is to make the model identity-hashable (the langchain
        # contract); this guard keeps a shared-library cache from ever
        # turning a perf optimization into a turn-killing exception.
        log.warning(
            "chat model %s is unhashable; skipping bound-model cache "
            "(bind-fresh). make the model identity-hashable to restore "
            "the memoization.",
            type(chat_model).__name__,
        )
        result = (chat_model.bind_tools(sorted_tools), [])
    else:
        prev_key = cached.tool_key if cached is not None else None
        bound_model: Any
        if should_bind_tools_fresh(prev_key, current_key):
            bound_model = chat_model.bind_tools(sorted_tools)
            _BOUND_MODEL_CACHE[chat_model] = _BoundModelCache(
                tool_key=current_key,
                bound_model=bound_model,
            )
        else:
            assert cached is not None  # noqa: S101 - guarded by should_bind_tools_fresh
            bound_model = cached.bound_model
        result = (bound_model, [])
    return result


def summarize_args(args: dict[str, Any], max_length: int = 100) -> str:
    """build a truncated summary of tool-call arguments for observation.

    publishes only the argument *keys* with elided values so a
    downstream observer (tool-call-start envelope, audit event, log
    line) sees the shape of the call without leaking sensitive
    contents (SQL fragments, passwords, PII). caller-tunable
    ``max_length`` clamps the summary length so a tool with many keys
    cannot blow up the wire envelope.

    :param args: tool-call arguments dict (typically the ``args`` key
        on a LangChain tool_call dict)
    :ptype args: dict[str, Any]
    :param max_length: maximum returned summary length in characters
    :ptype max_length: int
    :return: truncated string representation
    :rtype: str
    """
    keys = list(args.keys())
    if not keys:
        return "(no arguments)"
    summary = ", ".join(f"{k}=..." for k in keys[:3])
    if len(keys) > 3:
        summary += f" (+{len(keys) - 3} more)"
    return summary[:max_length]
