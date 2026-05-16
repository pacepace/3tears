"""typed agent-runtime events dispatched through langchain custom events.

before this module landed, every product that wanted server-push side-
channel events from inside a langgraph node had to ``adispatch_custom_event``
by magic string and re-implement the consumer-side dispatch table. payload
shapes drifted between producer and consumer; new fields landed in the
producer site but the consumer-side ``.get(...)`` quietly defaulted. this
module is the single source of truth for agent-runtime event envelopes
any 3tears app emits or observes from inside a langgraph run.

the wire vocabulary is in-process: events live between ``adispatch_custom_event``
in a graph node and the ``astream_events('v2')`` consumer reading
``on_custom_event`` ticks. each envelope is a pydantic ``BaseModel`` with a
``Literal`` ``type`` discriminator so a :class:`FrameworkEventRegistry`
parse-by-name produces the concrete subclass.

products that build their own outgoing-wire serialization (websocket
messages, sse frames, slack blocks) consume the typed event and map to
their own wire format. the framework owns the in-process schema; the
product owns the wire shape. for the existing token-streaming wire path
see :mod:`threetears.langgraph.streaming`.

the registry pattern mirrors :data:`StreamEvent` in ``streaming.py``: a
discriminated union over every framework event so an unknown ``type`` from
an outdated producer surfaces as a :class:`pydantic.ValidationError`
instead of silently misparsing.

naming convention: every event uses ``noun_verb`` form with the verb in
PAST TENSE (the event represents a completed action). this distinguishes
events from tools, which use ``noun_verb`` with imperative verbs (e.g.
``memory_add``, ``conversation_summarize``).
"""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.callbacks import adispatch_custom_event
from pydantic import BaseModel, Field

__all__ = [
    "FrameworkEvent",
    "FrameworkEventRegistry",
    "ImageGeneratedEvent",
    "PromptBuiltEvent",
    "ReasoningStreamedEvent",
    "ResponseCompletedEvent",
    "ResponseFailedEvent",
    "ToolCompletedEvent",
    "ToolDispatchedEvent",
    "ToolStartedEvent",
    "WorkflowCompletedEvent",
    "WorkflowStartedEvent",
    "WorkflowStepCompletedEvent",
    "default_registry",
    "dispatch_event",
]


class FrameworkEvent(BaseModel):
    """base class for every typed framework event.

    every subclass declares a ``type: Literal["..."]`` discriminator that
    doubles as the langchain custom-event name. concrete subclasses MUST
    NOT override ``type`` with a non-literal default; the parser keys on
    the literal value and a non-literal default breaks registration.

    payload fields use plain attribute access -- the in-process consumer
    receives the typed object, not a dict -- so adding a new field to a
    framework event is a non-breaking change for older consumers that
    ignore the new attribute.
    """


# ---------------------------------------------------------------------------
# Agent-runtime events (langgraph layer)
# ---------------------------------------------------------------------------


class PromptBuiltEvent(FrameworkEvent):
    """fired once per agent turn just before the model dispatches.

    carries the assembled system prompt + post-history jailbreak so a
    prompt-inspector ui can render the exact text the model is about to
    see. emitting this event before the first model token means the
    inspector affordance appears at the start of streaming, not after.

    :ivar system_prompt: full assembled system prompt sent to the model
    :ivar jailbreak_text: post-history jailbreak text, or ``None`` when no
        jailbreak is injected on this turn
    :ivar jailbreak_role: role under which the jailbreak is injected
        (``'system'`` or ``'user'``), or ``None``
    """

    type: Literal["prompt_built"] = "prompt_built"
    system_prompt: str
    jailbreak_text: str | None = None
    jailbreak_role: str | None = None


class ToolDispatchedEvent(FrameworkEvent):
    """fired when the model has emitted a tool call and dispatch is starting.

    consumers typically use this to reset per-round streamed-content
    tracking, since a new round begins after this tool call resolves.

    :ivar tool_name: name of the tool the model called
    """

    type: Literal["tool_dispatched"] = "tool_dispatched"
    tool_name: str


class ToolStartedEvent(FrameworkEvent):
    """fired as a tool function begins executing.

    paired with exactly one :class:`ToolCompletedEvent` per invocation.
    the ``tool_args`` payload is the parsed dict the model emitted; the
    consumer typically renders only the tool name in a status indicator
    but the args are available for debug / audit surfaces.

    :ivar tool_name: name of the tool being executed
    :ivar tool_args: parsed argument dict from the model's tool call
    """

    type: Literal["tool_started"] = "tool_started"
    tool_name: str
    tool_args: dict[str, Any] = Field(default_factory=dict)


class ToolCompletedEvent(FrameworkEvent):
    """fired when a tool function has finished executing.

    fires regardless of success or failure -- ``tool_status`` carries
    the outcome discriminator. ui consumers typically use this to
    render the tool's final-state indicator (a checkmark or an x).

    :ivar tool_name: name of the tool that ran
    :ivar tool_status: ``'completed'`` or ``'failed'``
    :ivar tool_duration_ms: wall-clock execution time in milliseconds
    """

    type: Literal["tool_completed"] = "tool_completed"
    tool_name: str
    tool_status: str = "completed"
    tool_duration_ms: int = 0


class ReasoningStreamedEvent(FrameworkEvent):
    """fired per-token for reasoning-class models (extended thinking traces).

    a reasoning-class model emits its chain-of-thought through this event
    stream (one event per token / chunk) so the consumer can render a
    dedicated "show reasoning" section rather than mixing it into the
    main response stream. the full trace is also persisted to the
    assistant message's reasoning content column for page-reload fidelity;
    this live channel is the per-token live render.

    :ivar token: token text fragment from the reasoning trace
    """

    type: Literal["reasoning_streamed"] = "reasoning_streamed"
    token: str


class ResponseCompletedEvent(FrameworkEvent):
    """fired once per agent turn when the model has produced its final answer.

    this is the cross-check envelope: the consumer compares the node's
    view of the response (``response_content``) with its own view of
    what it streamed to the client so a divergence (blank live ui +
    saved-content path) surfaces as observability rather than silent
    drift.

    the event is graph-node-agnostic -- emitted by whichever node assembles
    the final assistant message (in metallm this is the personality node;
    other 3tears products may emit from a different node).

    :ivar response_content: final accumulated assistant message text
    :ivar reasoning_content: reasoning-class trace text, or ``None`` for
        non-reasoning models / fallback paths where the reasoning IS
        the response content
    :ivar token_count_prompt: total prompt tokens consumed across rounds
    :ivar token_count_completion: total completion tokens emitted across rounds
    :ivar cache_read_input_tokens: cached-prompt tokens reused (provider-
        supplied; 0 when the provider does not surface cache metadata)
    :ivar cache_creation_input_tokens: cached-prompt tokens written this turn
    :ivar system_prompt: system prompt sent to the model, for inspector ui
    :ivar jailbreak_text: post-history jailbreak text, or ``None``
    :ivar jailbreak_role: jailbreak role discriminator, or ``None``
    :ivar tools_used: mapping of tool name to invocation count this turn
    :ivar tool_llm_id: provider id of the model that handled tool routing,
        or ``None`` when not separately tracked
    :ivar tool_invocation_id: identifier of the tool-routing invocation row,
        or ``None``
    """

    type: Literal["response_completed"] = "response_completed"
    response_content: str
    reasoning_content: str | None = None
    token_count_prompt: int = 0
    token_count_completion: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    system_prompt: str | None = None
    jailbreak_text: str | None = None
    jailbreak_role: str | None = None
    tools_used: dict[str, int] = Field(default_factory=dict)
    tool_llm_id: str | None = None
    tool_invocation_id: str | None = None


class ResponseFailedEvent(FrameworkEvent):
    """fired when the agent node generated a friendly fallback message
    because the provider failed mid-stream.

    consumers stream this content to the live ui as the assistant's
    response so the user does not see an empty bubble while the saved
    fallback waits behind a page reload.

    :ivar content: human-readable fallback message text
    """

    type: Literal["response_failed"] = "response_failed"
    content: str


class WorkflowStartedEvent(FrameworkEvent):
    """fired when an agent declares a multi-step workflow plan.

    :ivar plan: free-text overview of the plan
    :ivar steps: ordered list of step descriptions
    :ivar total_steps: convenience count, equal to ``len(steps)``
    """

    type: Literal["workflow_started"] = "workflow_started"
    plan: str = ""
    steps: list[str] = Field(default_factory=list)
    total_steps: int = 0


class WorkflowStepCompletedEvent(FrameworkEvent):
    """fired when an individual workflow step finishes.

    :ivar step_index: 0-indexed position of the completed step in the plan
    :ivar step_description: text of the step that completed
    :ivar step_status: completion status, typically ``'completed'`` or
        ``'failed'``
    :ivar tool_name: tool that drove the step, or ``None`` for pure-llm
        steps
    """

    type: Literal["workflow_step_completed"] = "workflow_step_completed"
    step_index: int = 0
    step_description: str = ""
    step_status: str = "completed"
    tool_name: str | None = None


class WorkflowCompletedEvent(FrameworkEvent):
    """fired when the entire workflow plan has finished.

    :ivar status: aggregate status, typically ``'completed'``,
        ``'partial'``, or ``'failed'``
    :ivar completed_steps: count of steps that finished successfully
    :ivar total_steps: total steps declared in the plan
    """

    type: Literal["workflow_completed"] = "workflow_completed"
    status: str = "completed"
    completed_steps: int = 0
    total_steps: int = 0


class ImageGeneratedEvent(FrameworkEvent):
    """fired when an image-generation tool has produced renderable markdown.

    this is the streaming path for image output -- the markdown is injected
    into the live response stream so the client renders the image before
    the model continues its narration around it. for v0.7.0 this event
    lives in the langgraph package alongside other agent-runtime events;
    a future agent/media package may re-home it without breaking
    consumers (re-exported via the registry).

    :ivar markdown: image markdown ready to inject into the response stream
    """

    type: Literal["image_generated"] = "image_generated"
    markdown: str


# ---------------------------------------------------------------------------
# Dispatch + registry
# ---------------------------------------------------------------------------


async def dispatch_event(
    event: FrameworkEvent,
    *,
    config: dict[str, Any] | None = None,
) -> None:
    """publish a typed framework event via langchain's custom-event channel.

    the typed event is serialized to a dict with ``model_dump(mode='json')``;
    the ``type`` field becomes the langchain event name and the remaining
    fields are the payload. consumers reading ``on_custom_event`` see the
    same shape they would from a hand-rolled ``adispatch_custom_event``,
    so the consumer side can adopt typed parsing incrementally without
    forcing every producer to migrate first.

    :param event: typed framework event to dispatch
    :ptype event: FrameworkEvent
    :param config: langchain runtime config (passed through verbatim)
    :ptype config: dict[str, Any] | None
    :return: nothing
    :rtype: None
    """
    payload = event.model_dump(mode="json")
    name = payload.pop("type")
    await adispatch_custom_event(name, payload, config=config)


class FrameworkEventRegistry:
    """maps custom-event names to their typed :class:`FrameworkEvent` class.

    consumers reading ``on_custom_event`` from ``astream_events('v2')``
    pass the event name + data into :meth:`parse` and receive either a
    concrete typed instance or ``None`` for an unknown name. ``None`` is
    the signal that the event is product-bespoke (or comes from a producer
    newer than the consumer); the consumer can fall through to legacy
    handling without losing the message.

    a default registry pre-populated with every framework event is exposed
    as :data:`default_registry`. products that want to register their own
    typed events can either instantiate a fresh registry or call
    :meth:`register` on the default; framework events do not collide with
    product events as long as the ``type`` discriminators are distinct.
    """

    def __init__(self) -> None:
        """initialize an empty registry."""
        self._by_name: dict[str, type[FrameworkEvent]] = {}

    def register(self, event_cls: type[FrameworkEvent]) -> None:
        """register one typed event class by its ``type`` discriminator.

        the class must declare ``type`` as a ``Literal[...]`` field with
        a string default (the wire name). raises :class:`ValueError` if
        the discriminator cannot be read or if a class is already
        registered under the same name (silent overwrites are a debugging
        nightmare across packages).

        :param event_cls: concrete :class:`FrameworkEvent` subclass
        :ptype event_cls: type[FrameworkEvent]
        :return: nothing
        :rtype: None
        :raises ValueError: when the class lacks a literal ``type`` default
            or collides with an already-registered name
        """
        field = event_cls.model_fields.get("type")
        if field is None:
            raise ValueError(
                f"{event_cls.__name__} has no 'type' field; cannot register",
            )
        name = field.default
        if not isinstance(name, str):
            raise ValueError(
                f"{event_cls.__name__}.type must be a Literal[str] with a string default; got {name!r}",
            )
        existing = self._by_name.get(name)
        if existing is not None and existing is not event_cls:
            raise ValueError(
                f"event name '{name}' already registered to {existing.__name__}; cannot rebind to {event_cls.__name__}",
            )
        self._by_name[name] = event_cls

    def parse(
        self,
        name: str,
        data: dict[str, Any],
    ) -> FrameworkEvent | None:
        """parse a raw custom-event payload into its typed envelope.

        the langchain custom-event shape is ``{name: str, data: dict}``.
        this method takes both and produces a concrete typed instance by
        merging ``{"type": name}`` back into ``data`` before pydantic
        validation, so the round-trip with :func:`dispatch_event` is
        symmetric.

        :param name: langchain custom-event name (the ``type`` discriminator)
        :ptype name: str
        :param data: payload dict (sans the ``type`` field)
        :ptype data: dict[str, Any]
        :return: typed event instance or ``None`` when name is unknown
        :rtype: FrameworkEvent | None
        :raises pydantic.ValidationError: when the payload does not match
            the registered class's schema
        """
        cls = self._by_name.get(name)
        if cls is None:
            return None
        return cls.model_validate({"type": name, **data})

    def names(self) -> list[str]:
        """return the sorted list of currently-registered event names.

        primarily an enforcement-test surface so a static checker can
        confirm "every framework event has a registry entry" without
        importing every event class individually.

        :return: sorted list of registered event names
        :rtype: list[str]
        """
        return sorted(self._by_name.keys())


default_registry = FrameworkEventRegistry()
"""shared registry pre-populated with every framework event class.

products typically use :data:`default_registry` directly. the registry is
mutable; downstream packages (``threetears.agent.memory.events``, etc.)
register their own event classes into this same instance on import so a
consumer parsing an ``on_custom_event`` tick sees every framework event
regardless of which package owns it.
"""


def _register_langgraph_events() -> None:
    """internal: register every event class defined in this module.

    called at import time so :data:`default_registry` is populated before
    any consumer imports it.

    :return: nothing
    :rtype: None
    """
    for cls in (
        PromptBuiltEvent,
        ToolDispatchedEvent,
        ToolStartedEvent,
        ToolCompletedEvent,
        ReasoningStreamedEvent,
        ResponseCompletedEvent,
        ResponseFailedEvent,
        WorkflowStartedEvent,
        WorkflowStepCompletedEvent,
        WorkflowCompletedEvent,
        ImageGeneratedEvent,
    ):
        default_registry.register(cls)


_register_langgraph_events()
