"""OpenRouter chat factory backed by ``langchain_openrouter``.

LangChain-native shape (3tears v0.6.0+): :func:`create_openrouter_chat`
returns a fully-configured ``ChatOpenRouter`` instance. ``ChatOpenRouter``
expects the request timeout in milliseconds — the factory accepts seconds
to match the rest of the API surface and converts internally.

Tool-name translation: OpenRouter routes some upstream models through
backends with strict tool-name validators (Bedrock requires
``^[a-zA-Z0-9_-]{1,128}$`` -- no dots), but the canonical 3tears tool
name is the dotted ``threetears.X`` form. The factory returns a
:class:`_NameTranslatingChatOpenRouter` subclass -- a thin binding of the
shared
:class:`~threetears.models.providers._name_translation_mixin.NameTranslatingChatMixin`
(identical hooks across the OpenAI / OpenRouter / Anthropic wrappers) --
that translates dot-to-underscore on outgoing tool specs / history and
underscore-to-dot on incoming ``tool_calls``, so application code never
sees the wire form. The translation is keyed by the dotted -> underscored
mapping built at ``bind_tools`` time, so it round-trips losslessly even
for tools whose names contain underscores already (e.g.
``threetears.web_search`` -> ``threetears_web_search`` and back).

Structured output is the same class of wire quirk:
:func:`openrouter_structured_output_kwargs` builds the OpenRouter-native
directive so no consumer has to know the provider's spelling of "return
json matching this schema". Dispatch across providers lives in
:mod:`threetears.models.providers.structured_output`.

What an answer's ``response_metadata`` carries from OpenRouter, beside LangChain's own keys
(``provider``, ``cost``, ``cost_details`` among them):

- ``openrouter_usage``: the reply's ``usage`` block, OpenRouter's wire shape as sent -- every
  field OpenRouter sent and the SDK parses (``is_byok``, ``server_tool_use_details``, every
  ``prompt_tokens_details`` / ``completion_tokens_details`` count), and none it did not send: a
  count missing from the dict was not reported, which is not the same as ``0``. Absent when the
  reply carried no usage.
- ``openrouter_response``: the reply's response-level fields LangChain drops --
  ``openrouter_metadata``, ``service_tier``, ``system_fingerprint`` -- each only when OpenRouter
  sent it with a value; absent when it sent none of them. A ``null`` counts as not sent: the SDK
  parses a stream chunk's ``null`` ``system_fingerprint`` as unsent, so the two cannot be told
  apart on every path, and a key here always holds a value.

On a non-streamed answer both are on the message; on a streamed one they ride the usage chunk
(so the aggregated message carries them too), or a trailing empty chunk when the stream had no
usage chunk.

**BYOK cost.** ``openrouter_usage["is_byok"]`` is ``True`` when the call ran on the caller's own
provider key. ``cost`` is then only OpenRouter's fee: the upstream provider bills its share to
the caller's own account, reported as ``cost_details["upstream_inference_cost"]``, so the call's
full cost is ``cost + cost_details["upstream_inference_cost"]``. When ``is_byok`` is ``False``,
``cost`` is the whole charge. When ``is_byok`` is absent OpenRouter did not say.

Why models carries these: ``langchain-openrouter`` (0.2.9) builds ``response_metadata`` from the
SDK's reply and keeps only part of it (``cost``, ``cost_details``, and token counts flattened into
``usage_metadata`` with ``or 0``, so a missing count reads as zero). See
:class:`_ReplyObservingClient`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, Self

from pydantic import PrivateAttr, model_validator

from threetears.models.capabilities import ModelCapabilities, register_capabilities
from threetears.models.enums import ModelStatus, ModelTier, ModelType
from threetears.models.providers._name_translation_mixin import NameTranslatingChatMixin
from threetears.models.providers.structured_output import ensure_valid_json_schema

if TYPE_CHECKING:
    from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
    from langchain_core.messages import BaseMessage
    from langchain_core.outputs import ChatGenerationChunk
    from langchain_core.outputs import ChatResult as LcChatResult
    from langchain_openrouter import ChatOpenRouter

__all__ = [
    "OPENROUTER_PROVIDER_NAME",
    "create_openrouter_chat",
    "declare_served_provider",
    "openrouter_structured_output_kwargs",
]


OPENROUTER_PROVIDER_NAME = "openrouter"


def openrouter_structured_output_kwargs(
    json_schema: dict[str, Any],
    *,
    name: str = "response",
    strict: bool = True,
) -> dict[str, Any]:
    """builds the OpenRouter-native structured-output bind kwargs.

    ``ChatOpenRouter`` merges bound kwargs straight into the OpenRouter
    SDK's ``chat.send(...)`` call, which accepts ``response_format`` and
    ``provider`` as top-level request params -- so the OpenAI-shaped
    ``response_format`` block goes on directly, no ``extra_body`` nesting.

    ``provider={"require_parameters": True}`` is not optional: it is what
    makes OpenRouter REJECT the request when the routed upstream model
    cannot honour ``response_format``. Without it OpenRouter silently
    drops the unsupported param and the model returns prose, which is the
    exact failure structured output exists to prevent. The factory's model
    merges this block INTO its own routing (``openrouter_provider``: an
    ``only`` / ``ignore`` policy) rather than letting it replace it, so a
    structured ask is routed exactly as a plain one.

    ``require_parameters`` guards only whether the routed model CAN honour
    ``response_format``; it says nothing about the schema being coherent.
    OpenRouter accepts a malformed schema and answers successfully (a
    live ``google/gemini-2.5-flash`` call with ``{"type": "object",
    "properties": "oops"}`` returned ``{}``), so the schema is checked
    locally by :func:`~threetears.models.providers.structured_output.ensure_valid_json_schema`
    before it is embedded -- otherwise a caller typo is indistinguishable
    from a real answer.

    Returns bind KWARGS rather than a bound model on purpose: the caller
    may be applying this to a model that is already a ``bind_tools``
    ``RunnableBinding``, which exposes ``.bind(**kwargs)`` but none of the
    wrapper class's own methods.

    :param json_schema: json-schema the response must satisfy
    :ptype json_schema: dict[str, Any]
    :param name: schema name reported to the provider
    :ptype name: str
    :param strict: whether the provider must enforce the schema exactly
    :ptype strict: bool
    :return: kwargs to pass to ``model.bind(**kwargs)``
    :rtype: dict[str, Any]
    :raises StructuredOutputSchemaError: when json_schema is not a valid
        json-schema
    """
    ensure_valid_json_schema(json_schema, provider_type=OPENROUTER_PROVIDER_NAME)

    return {
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": name,
                "schema": json_schema,
                "strict": strict,
            },
        },
        "provider": {"require_parameters": True},
    }


def declare_served_provider() -> None:
    """Teach the OpenRouter SDK's response models the ``provider`` field OpenRouter sends.

    OpenRouter names the upstream that actually served the call (``"provider": "Anthropic"``)
    on every response and every stream chunk. The ``openrouter`` SDK's ``ChatResult`` and
    ``ChatStreamChunk`` do not declare it, so parsing drops it, and ``langchain-openrouter``
    (0.2.9+, which copies it into ``response_metadata`` / the final chunk's
    ``generation_info``) finds nothing. Without it nobody can tell which upstream answered, which
    is the question an ``only`` / ``ignore`` routing policy exists to settle.

    Idempotent. ``ChatStreamingResponse`` wraps each chunk and holds a copy of its schema taken
    at import, so it is rebuilt after the chunk is. A class that moved or stopped being a
    pydantic model raises -- silently skipping it would drop the field again, invisibly.

    :raises ImportError: when the ``openrouter`` SDK no longer has these modules or classes
    :raises TypeError: when one of them is no longer a pydantic model
    """
    from openrouter.components.chatresult import ChatResult as SdkChatResult
    from openrouter.components.chatstreamchunk import ChatStreamChunk as SdkChatStreamChunk
    from openrouter.components.chatstreamingresponse import ChatStreamingResponse as SdkChatStreamingResponse
    from pydantic import BaseModel
    from pydantic.fields import FieldInfo

    for model in (SdkChatResult, SdkChatStreamChunk, SdkChatStreamingResponse):
        if not (isinstance(model, type) and issubclass(model, BaseModel)):
            raise TypeError(f"openrouter's {model!r} is no longer a pydantic model; the served provider cannot be read")
    for model in (SdkChatResult, SdkChatStreamChunk):
        if "provider" in model.model_fields:
            continue
        served: Any = str | None
        model.model_fields["provider"] = FieldInfo(annotation=served, default=None)
        model.__annotations__["provider"] = served
        model.model_rebuild(force=True)
        SdkChatStreamingResponse.model_rebuild(force=True)


#: the response-level fields of an OpenRouter reply that LangChain drops and the SDK parses.
_RESPONSE_FIELDS = ("openrouter_metadata", "service_tier", "system_fingerprint")


class _ReplySlot:
    """One OpenRouter call's ``usage`` and response-level fields, as that call's own replies sent them.

    Made by the call itself (``_RoutingChatOpenRouter``'s ``_generate`` / ``_agenerate`` /
    ``_stream`` / ``_astream``) and handed to the SDK wrapper through :data:`_CALL_REPLY` for
    exactly as long as it takes the call to reach the SDK: the wrapper binds the slot to the
    reply (or stream) it returns, so every later read is of that call's own reply.

    Sent versus absent: each reply is read through the SDK's own serializer with
    ``exclude_unset=True``. The SDK marks a field OpenRouter did not send as unset, and its
    serializer, run without ``exclude_unset``, writes such a nested optional field back as
    ``null``; with it, only what was sent comes out, a sent ``null`` included.

    :ivar usage: the ``usage`` seen and not yet put on a message, or ``None``
    :ptype usage: dict[str, Any] | None
    :ivar response: the response-level fields seen and not yet put on a message
    :ptype response: dict[str, Any]
    """

    __slots__ = ("response", "usage")

    def __init__(self) -> None:
        self.usage: dict[str, Any] | None = None
        self.response: dict[str, Any] = {}

    def observe(self, reply: Any) -> None:  # noqa: ANN401 -- an SDK ChatResult or ChatStreamChunk
        """keep what ``reply`` sent: its ``usage``, and its response-level fields that carry a value.

        A reply with no ``usage`` (every stream chunk but the last) leaves the usage seen so far.
        A reply that is not a pydantic model, or whose usage moved, is read as having sent none:
        the fields are information, and their absence breaks nothing.

        :param reply: one parsed reply from the SDK
        :ptype reply: Any
        """
        dump = getattr(reply, "model_dump", None)
        if not callable(dump):
            return
        sent = dump(mode="json", by_alias=True, exclude_unset=True)
        if not isinstance(sent, dict):
            return
        usage = sent.get("usage")
        if isinstance(usage, dict):
            self.usage = usage
        for name in _RESPONSE_FIELDS:
            if sent.get(name) is not None:
                self.response[name] = sent[name]

    def pending(self) -> bool:
        """whether something seen has not yet been put on a message.

        :return: ``True`` while a usage or a response-level field waits
        :rtype: bool
        """
        return self.usage is not None or bool(self.response)

    def take(self) -> dict[str, Any]:
        """the ``response_metadata`` keys for what was seen, emptying the slot.

        :return: ``openrouter_usage`` and ``openrouter_response``, each only when seen
        :rtype: dict[str, Any]
        """
        keys: dict[str, Any] = {}
        if self.usage is not None:
            keys["openrouter_usage"] = self.usage
        if self.response:
            keys["openrouter_response"] = self.response
        self.usage = None
        self.response = {}
        return keys

    def mark_chunk(self, chunk: ChatGenerationChunk) -> ChatGenerationChunk:
        """put what was seen on the stream's usage chunk, ``chunk``, if this is it.

        ``langchain-openrouter`` emits the SDK's usage-only chunk as a chunk with
        ``usage_metadata`` and ``cost`` / ``cost_details`` in its ``response_metadata``; the two
        keys join them there.

        :param chunk: the next chunk LangChain yielded for this call
        :ptype chunk: ChatGenerationChunk
        :return: ``chunk``, its metadata carrying the keys when it is the usage chunk
        :rtype: ChatGenerationChunk
        """
        if not self.pending() or getattr(chunk.message, "usage_metadata", None) is None:
            return chunk
        chunk.message.response_metadata.update(self.take())
        return chunk

    def trailing_chunk(self) -> ChatGenerationChunk | None:
        """an empty chunk carrying what no usage chunk took, at the end of a stream.

        A stream that ended without a usage chunk still sent its response-level fields (and
        possibly a usage LangChain did not emit); they arrive on this chunk rather than nowhere.

        :return: the chunk, or ``None`` when nothing waits
        :rtype: ChatGenerationChunk | None
        """
        from langchain_core.messages import AIMessageChunk
        from langchain_core.outputs import ChatGenerationChunk

        if not self.pending():
            return None
        return ChatGenerationChunk(message=AIMessageChunk(content="", response_metadata=self.take()))

    def mark_result(self, result: LcChatResult) -> LcChatResult:
        """put what was seen on every AI message of a non-streamed ``result``.

        Each message gets its own copy: one usage block covers every choice, as LangChain's
        ``cost`` / ``cost_details`` do.

        :param result: the call's result, as LangChain built it
        :ptype result: LcChatResult
        :return: ``result``, its AI messages carrying the keys when something was seen
        :rtype: LcChatResult
        """
        import copy

        from langchain_core.messages import AIMessage

        keys = self.take()
        if not keys:
            return result
        for generation in result.generations:
            if isinstance(generation.message, AIMessage):
                generation.message.response_metadata.update(copy.deepcopy(keys))
        return result


#: the slot of the OpenRouter call in this context that has not yet reached the SDK. Set by the
#: call just before it enters LangChain, read by :class:`_ReplyObservingChat` when LangChain sends,
#: and reset by the call in the same step, so it is never set across a yield or an ``await`` on
#: anything but the call's own send. Each asyncio task and each thread has its own value.
_CALL_REPLY: ContextVar[_ReplySlot | None] = ContextVar("threetears_openrouter_call_reply", default=None)


class _ObservedStream:
    """the SDK's sync event stream, each chunk shown to one call's slot as it passes.

    :ivar _inner: the SDK's stream
    :ivar _slot: the call's slot
    """

    def __init__(self, inner: Any, slot: _ReplySlot) -> None:  # noqa: ANN401 -- the SDK's EventStream
        self._inner = inner
        self._slot = slot

    def __iter__(self) -> _ObservedStream:
        return self

    def __next__(self) -> Any:  # noqa: ANN401 -- an SDK ChatStreamChunk
        chunk = next(self._inner)
        self._slot.observe(chunk)
        return chunk

    def __enter__(self) -> _ObservedStream:
        self._inner.__enter__()
        return self

    def __exit__(self, *exc: object) -> Any:  # noqa: ANN401 -- the SDK stream's own answer
        return self._inner.__exit__(*exc)

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401 -- the SDK stream's attributes
        if name == "_inner":
            raise AttributeError(name)
        return getattr(self._inner, name)


class _ObservedAsyncStream:
    """the SDK's async event stream, each chunk shown to one call's slot as it passes.

    :ivar _inner: the SDK's stream
    :ivar _slot: the call's slot
    """

    def __init__(self, inner: Any, slot: _ReplySlot) -> None:  # noqa: ANN401 -- the SDK's EventStreamAsync
        self._inner = inner
        self._slot = slot

    def __aiter__(self) -> _ObservedAsyncStream:
        return self

    async def __anext__(self) -> Any:  # noqa: ANN401 -- an SDK ChatStreamChunk
        chunk = await anext(self._inner)
        self._slot.observe(chunk)
        return chunk

    async def __aenter__(self) -> _ObservedAsyncStream:
        await self._inner.__aenter__()
        return self

    async def __aexit__(self, *exc: object) -> Any:  # noqa: ANN401 -- the SDK stream's own answer
        return await self._inner.__aexit__(*exc)

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401 -- the SDK stream's attributes
        if name == "_inner":
            raise AttributeError(name)
        return getattr(self._inner, name)


class _ReplyObservingChat:
    """the SDK's ``chat`` sub-client, its replies shown to the slot of the call that sent them.

    The slot is read once, when LangChain calls ``send`` / ``send_async``, and bound to what that
    send returns; a send with no slot (a call path that does not go through the model's own
    generate or stream) passes through untouched.

    :ivar _inner: the SDK's ``chat``
    """

    def __init__(self, inner: Any) -> None:  # noqa: ANN401 -- the SDK's Chat
        self._inner = inner

    def send(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401 -- the SDK's own signature
        """the SDK's ``chat.send``, its reply shown to the calling slot.

        :param args: passthrough
        :ptype args: Any
        :param kwargs: passthrough
        :ptype kwargs: Any
        :return: the SDK's reply; a stream comes back wrapped so each chunk is seen
        :rtype: Any
        """
        slot = _CALL_REPLY.get()
        reply = self._inner.send(*args, **kwargs)
        if slot is None:
            return reply
        if hasattr(reply, "__next__"):
            return _ObservedStream(reply, slot)
        slot.observe(reply)
        return reply

    async def send_async(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401 -- the SDK's own signature
        """the SDK's ``chat.send_async``, its reply shown to the calling slot.

        :param args: passthrough
        :ptype args: Any
        :param kwargs: passthrough
        :ptype kwargs: Any
        :return: the SDK's reply; a stream comes back wrapped so each chunk is seen
        :rtype: Any
        """
        slot = _CALL_REPLY.get()
        reply = await self._inner.send_async(*args, **kwargs)
        if slot is None:
            return reply
        if hasattr(reply, "__anext__"):
            return _ObservedAsyncStream(reply, slot)
        slot.observe(reply)
        return reply

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401 -- the SDK chat's other attributes
        if name == "_inner":
            raise AttributeError(name)
        return getattr(self._inner, name)


class _ReplyObservingClient:
    """the ``openrouter`` SDK client, with its ``chat`` replies observed for what LangChain drops.

    Why this lives in models: OpenRouter's reply carries ``usage`` fields (``is_byok``,
    ``server_tool_use_details``, the token-detail counts, sent or not) and response-level fields
    (``openrouter_metadata``, ``service_tier``, ``system_fingerprint``) that the SDK parses and
    ``langchain-openrouter`` (0.2.9) drops or flattens while building ``response_metadata``.
    Without ``is_byok`` a BYOK caller's ``cost`` (OpenRouter's fee alone) reads as the whole
    charge. LangChain's package is not patched: the model's public ``client`` is this wrapper,
    every reply passing through it is shown to the slot of the call that sent it, and the call
    puts what was sent on its own message, under keys of its own (``openrouter_usage``,
    ``openrouter_response``) that LangChain does not write.

    Built once per model. Everything but ``chat`` is the SDK client's own.

    :ivar chat: the observed ``chat`` sub-client
    :ivar _inner: the SDK client
    """

    def __init__(self, inner: Any) -> None:  # noqa: ANN401 -- the openrouter.OpenRouter client
        """wrap ``inner``, refusing a client whose ``chat`` surface moved.

        :param inner: the SDK client LangChain built or the caller passed
        :ptype inner: Any
        :raises TypeError: when the client has no ``chat``, or ``chat`` has no callable ``send`` or
            ``send_async`` -- LangChain calls both, and wrapping something else would drop the
            flag invisibly
        """
        chat = getattr(inner, "chat", None)
        if chat is None:
            raise TypeError(
                f"the openrouter client {type(inner).__name__} has no `chat`; its replies cannot be observed"
            )
        for name in ("send", "send_async"):
            if not callable(getattr(chat, name, None)):
                raise TypeError(
                    f"the openrouter client's chat ({type(chat).__name__}) has no callable `{name}`; "
                    "its replies cannot be observed"
                )
        self._inner = inner
        self.chat = _ReplyObservingChat(chat)

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401 -- the SDK client's other attributes
        if name == "_inner":
            raise AttributeError(name)
        return getattr(self._inner, name)


#: The ``reasoning`` keys OpenRouter takes that the ``openrouter`` SDK's ``ChatRequestReasoning`` (0.10.8)
#: does not declare, and the type of each: a token budget, an on/off switch, and whether the reasoning
#: text is left out of the reply.
_UNDECLARED_REASONING_KEYS: dict[str, type] = {"max_tokens": int, "enabled": bool, "exclude": bool}


def _declare_reasoning_budget() -> None:
    """Teach the OpenRouter SDK's typed request the ``reasoning`` keys it drops.

    OpenRouter's ``reasoning`` object takes ``max_tokens``, ``enabled`` and ``exclude`` beside
    ``effort`` and ``summary``. The ``openrouter`` SDK's ``ChatRequestReasoning`` declares only the
    last two, and validating a request through it drops the rest without a word: a judge's token
    budget went out as ``"reasoning": {}``. Each missing key is declared the way the SDK declares its
    own optional ones -- unset by default, and an unset key is not sent -- so a request carries
    exactly the keys its caller set.

    The same pattern as :func:`declare_served_provider`, on the request side. Idempotent: a key the
    SDK already declares is left alone. ``ChatRequest`` and the operation request that wraps it hold
    a copy of the reasoning schema taken at import, so both are rebuilt after it. A class that moved
    or stopped being a pydantic model raises -- silently skipping it would drop the budget again,
    invisibly.

    :raises ImportError: when the ``openrouter`` SDK no longer has these modules or classes
    :raises TypeError: when one of them is no longer a pydantic model
    """
    from openrouter.components.chatrequest import ChatRequest as SdkChatRequest
    from openrouter.components.chatrequest import ChatRequestReasoning as SdkChatRequestReasoning
    from openrouter.operations.sendchatcompletionrequest import (
        SendChatCompletionRequestRequest as SdkSendChatCompletionRequest,
    )
    from openrouter.types import UNSET
    from pydantic import BaseModel
    from pydantic.fields import FieldInfo

    holders = (SdkChatRequest, SdkSendChatCompletionRequest)
    for model in (SdkChatRequestReasoning, *holders):
        if not (isinstance(model, type) and issubclass(model, BaseModel)):
            raise TypeError(f"openrouter's {model!r} is no longer a pydantic model; a reasoning budget cannot be sent")
    missing = {
        key: kind for key, kind in _UNDECLARED_REASONING_KEYS.items() if key not in SdkChatRequestReasoning.model_fields
    }
    if not missing:
        return
    for key, kind in missing.items():
        # The SDK's ``OptionalNullable[...]``, spelled out: the value, null, or unset.
        annotation: Any = kind | None | type(UNSET)
        SdkChatRequestReasoning.model_fields[key] = FieldInfo(annotation=annotation, default=UNSET)
        SdkChatRequestReasoning.__annotations__[key] = annotation
    SdkChatRequestReasoning.model_rebuild(force=True)
    for holder in holders:
        holder.model_rebuild(force=True)


def _build_translating_chat_class() -> type[ChatOpenRouter]:
    """build the :class:`ChatOpenRouter` subclass with name-translation hooks.

    Defined inside a function so the langchain-openrouter import is
    lazy -- :mod:`threetears.models.providers.openrouter` is imported
    eagerly at package load to populate the capability registry, and we
    must not require ``langchain-openrouter`` to be installed for that
    side-effect import to succeed (it is an optional dependency,
    selected via ``3tears-models[openrouter]``).

    :return: name-translating ChatOpenRouter subclass
    :rtype: type[ChatOpenRouter]
    """
    from langchain_openrouter import ChatOpenRouter

    declare_served_provider()
    _declare_reasoning_budget()

    class _RoutingChatOpenRouter(ChatOpenRouter):
        """``ChatOpenRouter`` that keeps the model's provider routing under a bound ``provider``,
        and puts what OpenRouter sent and LangChain drops on each answer.

        Sits below :class:`NameTranslatingChatMixin` in the MRO, so the mixin's call deadline and
        name translation wrap it and it wraps the SDK call itself.

        ``openrouter_usage`` / ``openrouter_response`` (see the module docstring): this model's
        ``client`` is a :class:`_ReplyObservingClient`. Each call makes its own
        :class:`_ReplySlot`, sets it in :data:`_CALL_REPLY` only until LangChain has sent the
        request (the first chunk, or the result), and puts what the slot saw on its own message.
        Two calls on one model never share a slot: the client wrapper binds the slot it finds at
        send time to that send's own reply, and each call's slot is set only in its own context
        (task or thread), and reset before the call yields anything.
        """

        @model_validator(mode="after")
        def _observe_replies(self) -> Self:
            """make the model's SDK client the reply-observing wrapper, once.

            :return: the model
            :rtype: Self
            :raises TypeError: when the SDK client's ``chat.send`` / ``chat.send_async`` moved
            """
            if not isinstance(self.client, _ReplyObservingClient):
                self.client = _ReplyObservingClient(self.client)
            return self

        def _with_model_routing(self, kwargs: dict[str, Any]) -> dict[str, Any]:
            """``kwargs`` with a bound ``provider`` block merged INTO the model's own routing.

            ``ChatOpenRouter`` builds each request as ``{**params, **kwargs}``, so a bound
            ``provider`` (the structured-output directive binds
            ``{"require_parameters": True}``) would REPLACE the model's whole routing block --
            its ``only`` and ``ignore`` lists -- and the request would go out with no provider
            restrictions at all. Merged instead: the model's routing, with the bound keys on top.

            :param kwargs: the call's keyword arguments, bound ones included
            :ptype kwargs: dict[str, Any]
            :return: the kwargs to send, with the merged ``provider`` block
            :rtype: dict[str, Any]
            """
            bound = kwargs.get("provider")
            routing = self._default_params.get("provider")
            if not isinstance(bound, dict) or not isinstance(routing, dict):
                return kwargs
            return {**kwargs, "provider": {**routing, **bound}}

        def _generate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: CallbackManagerForLLMRun | None = None,
            **kwargs: Any,
        ) -> LcChatResult:
            """The parent's generate, with the model's routing kept under a bound ``provider``,
            and what OpenRouter sent and LangChain dropped on each AI message of the result.

            :param messages: chat messages
            :ptype messages: list[BaseMessage]
            :param stop: optional stop sequences
            :ptype stop: list[str] | None
            :param run_manager: LangChain run manager
            :ptype run_manager: CallbackManagerForLLMRun | None
            :param kwargs: passthrough
            :ptype kwargs: Any
            :return: the parent's chat result, its AI messages carrying ``openrouter_usage`` / ``openrouter_response``
            :rtype: LcChatResult
            """
            slot = _ReplySlot()
            token = _CALL_REPLY.set(slot)
            try:
                result: LcChatResult = super()._generate(
                    messages, stop=stop, run_manager=run_manager, **self._with_model_routing(kwargs)
                )
            finally:
                _CALL_REPLY.reset(token)
            return slot.mark_result(result)

        async def _agenerate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: AsyncCallbackManagerForLLMRun | None = None,
            **kwargs: Any,
        ) -> LcChatResult:
            """The parent's async generate, with the model's routing kept under a bound ``provider``,
            and what OpenRouter sent and LangChain dropped on each AI message of the result.

            :param messages: chat messages
            :ptype messages: list[BaseMessage]
            :param stop: optional stop sequences
            :ptype stop: list[str] | None
            :param run_manager: LangChain run manager
            :ptype run_manager: AsyncCallbackManagerForLLMRun | None
            :param kwargs: passthrough
            :ptype kwargs: Any
            :return: the parent's chat result, its AI messages carrying ``openrouter_usage`` / ``openrouter_response``
            :rtype: LcChatResult
            """
            slot = _ReplySlot()
            token = _CALL_REPLY.set(slot)
            try:
                result: LcChatResult = await super()._agenerate(
                    messages, stop=stop, run_manager=run_manager, **self._with_model_routing(kwargs)
                )
            finally:
                _CALL_REPLY.reset(token)
            return slot.mark_result(result)

        def _stream(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: CallbackManagerForLLMRun | None = None,
            **kwargs: Any,
        ) -> Iterator[ChatGenerationChunk]:
            """The parent's stream, with the model's routing kept under a bound ``provider``,
            and what OpenRouter sent and LangChain dropped on the usage chunk.

            :param messages: chat messages
            :ptype messages: list[BaseMessage]
            :param stop: optional stop sequences
            :ptype stop: list[str] | None
            :param run_manager: LangChain run manager
            :ptype run_manager: CallbackManagerForLLMRun | None
            :param kwargs: passthrough
            :ptype kwargs: Any
            :return: the parent's chunks, the usage chunk carrying ``openrouter_usage`` / ``openrouter_response``
            :rtype: Iterator[ChatGenerationChunk]
            """
            slot = _ReplySlot()
            stream = super()._stream(messages, stop=stop, run_manager=run_manager, **self._with_model_routing(kwargs))
            token = _CALL_REPLY.set(slot)
            try:
                first = next(stream, None)
            finally:
                _CALL_REPLY.reset(token)
            if first is None:
                return
            yield slot.mark_chunk(first)
            for chunk in stream:
                yield slot.mark_chunk(chunk)
            tail = slot.trailing_chunk()
            if tail is not None:
                if run_manager:
                    run_manager.on_llm_new_token(tail.text, chunk=tail)
                yield tail

        async def _astream(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: AsyncCallbackManagerForLLMRun | None = None,
            **kwargs: Any,
        ) -> AsyncIterator[ChatGenerationChunk]:
            """The parent's async stream, with the model's routing kept under a bound ``provider``,
            and what OpenRouter sent and LangChain dropped on the usage chunk.

            :param messages: chat messages
            :ptype messages: list[BaseMessage]
            :param stop: optional stop sequences
            :ptype stop: list[str] | None
            :param run_manager: LangChain run manager
            :ptype run_manager: AsyncCallbackManagerForLLMRun | None
            :param kwargs: passthrough
            :ptype kwargs: Any
            :return: the parent's chunks, the usage chunk carrying ``openrouter_usage`` / ``openrouter_response``
            :rtype: AsyncIterator[ChatGenerationChunk]
            """
            slot = _ReplySlot()
            stream = super()._astream(messages, stop=stop, run_manager=run_manager, **self._with_model_routing(kwargs))
            token = _CALL_REPLY.set(slot)
            try:
                first = await anext(stream, None)
            finally:
                _CALL_REPLY.reset(token)
            if first is None:
                return
            yield slot.mark_chunk(first)
            async for chunk in stream:
                yield slot.mark_chunk(chunk)
            tail = slot.trailing_chunk()
            if tail is not None:
                if run_manager:
                    await run_manager.on_llm_new_token(tail.text, chunk=tail)
                yield tail

    class _NameTranslatingChatOpenRouter(NameTranslatingChatMixin, _RoutingChatOpenRouter):
        """``ChatOpenRouter`` with dot<->underscore tool-name translation at
        the wire boundary, hiding Bedrock-style provider quirks from the rest
        of the codebase.

        All translation hooks live in :class:`NameTranslatingChatMixin`
        (mixed in ahead of ``ChatOpenRouter`` so ``super()`` resolves to it);
        this subclass only supplies the per-instance reverse-map slot.

        :ivar _name_reverse_map: underscored-wire -> canonical-dotted map,
            populated at ``bind_tools`` time.
        :ptype _name_reverse_map: dict[str, str]
        """

        _name_reverse_map: dict[str, str] = PrivateAttr(default_factory=dict)

        def call_deadline_s(self) -> float | None:
            """The configured timeout, as the whole call's deadline.

            The OpenRouter SDK hands the timeout to httpx, which applies it to
            each read, and OpenRouter answers 200 at once and keeps the call
            open with keep-alive comments while the upstream works. So no read
            ever waited 120 s and the call ran as long as the upstream did: one
            summary call held a turn for 218 s. The mixin holds each call to this.
            """
            timeout_ms = getattr(self, "request_timeout", None)
            return timeout_ms / 1000 if timeout_ms else None

    return _NameTranslatingChatOpenRouter


def create_openrouter_chat(
    model_name: str,
    api_key: str,
    *,
    timeout: int = 120,
    max_retries: int = 2,
    **extra_kwargs: Any,
) -> ChatOpenRouter:
    """creates a configured ``ChatOpenRouter`` for OpenRouter-routed models.

    Returns the :class:`_NameTranslatingChatOpenRouter` subclass, not
    a vanilla ``ChatOpenRouter``. Application code interacts with it
    exactly the same way (it IS a ``ChatOpenRouter``); the only
    difference is the wire-side tool-name translation that hides
    Bedrock-style provider quirks from the rest of the codebase.

    :param model_name: OpenRouter model identifier (e.g. ``deepseek/deepseek-chat-v3-0324``)
    :ptype model_name: str
    :param api_key: OpenRouter API key
    :ptype api_key: str
    :param timeout: request timeout in seconds (converted to ms internally)
    :ptype timeout: int
    :param max_retries: maximum retry attempts for failed requests
    :ptype max_retries: int
    :param extra_kwargs: additional keyword arguments forwarded to ``ChatOpenRouter``
    :ptype extra_kwargs: Any
    :return: configured ``ChatOpenRouter`` (the name-translating subclass)
    :rtype: ChatOpenRouter
    """
    chat_cls = _build_translating_chat_class()

    # langchain-openrouter 0.1.0 defaults app_title="langchain" and forwards
    # it as `x_title` to the underlying openrouter SDK. openrouter 0.8+
    # renamed that kwarg to `x_open_router_title`, so the old name now
    # raises TypeError. setting both to None restores compatibility until
    # langchain-openrouter ships a fix; callers that need attribution can
    # pass app_title/app_url via extra_kwargs.
    kwargs: dict[str, Any] = {
        "model": model_name,
        "api_key": api_key,
        "timeout": timeout * 1000,
        "max_retries": max_retries,
        "app_title": None,
        "app_url": None,
    }
    kwargs.update(extra_kwargs)

    model: ChatOpenRouter = chat_cls(**kwargs)
    return model


# -- capability registration -------------------------------------------------

# representative OpenRouter ids. additional ids can be registered by host
# apps at boot via register_capabilities().
_OPENROUTER_CAPABILITIES: dict[str, ModelCapabilities] = {
    "deepseek/deepseek-chat-v3-0324": ModelCapabilities(
        model_name="deepseek/deepseek-chat-v3-0324",
        provider_name=OPENROUTER_PROVIDER_NAME,
        model_type=ModelType.CHAT,
        model_tier=ModelTier.LARGE,
        model_status=ModelStatus.ACTIVE,
        context_window=64_000,
        max_output_tokens=8_192,
        supports_streaming=True,
        supports_tools=True,
        supports_vision=False,
        requires_alternating_roles=True,
        # DeepSeek's direct API runs automatic context caching and surfaces
        # ``cached_tokens`` on the response without an opt-in marker; the
        # ``deepseek/`` slug routed through OpenRouter inherits the same
        # behavior. Same request shape as OpenAI auto-cache.
        supports_anthropic_cache_control=False,
        supports_openai_auto_cache=True,
        min_cacheable_tokens=0,
        cache_ttl_seconds=0,
    ),
    "deepseek/deepseek-r1": ModelCapabilities(
        model_name="deepseek/deepseek-r1",
        provider_name=OPENROUTER_PROVIDER_NAME,
        model_type=ModelType.CHAT,
        model_tier=ModelTier.LARGE,
        model_status=ModelStatus.ACTIVE,
        context_window=64_000,
        max_output_tokens=8_192,
        supports_streaming=True,
        supports_tools=False,
        supports_vision=False,
        requires_alternating_roles=True,
        supports_anthropic_cache_control=False,
        supports_openai_auto_cache=True,
        min_cacheable_tokens=0,
        cache_ttl_seconds=0,
    ),
    "deepseek/deepseek-v4-flash": ModelCapabilities(
        model_name="deepseek/deepseek-v4-flash",
        provider_name=OPENROUTER_PROVIDER_NAME,
        model_type=ModelType.CHAT,
        # "Flash" branding + real OpenRouter pricing confirms this is the cheap/fast
        # sibling of the deepseek-chat-v3-0324 entry above (~3-6x cheaper per token
        # despite a much larger context window) -- MEDIUM, not LARGE (verified
        # 2026-07-23 via GET https://openrouter.ai/api/v1/models).
        model_tier=ModelTier.MEDIUM,
        model_status=ModelStatus.ACTIVE,
        context_window=1_048_576,
        # OpenRouter's models endpoint reports top_provider.max_completion_tokens
        # as null for this id (no explicit cap published) -- 8_192 mirrors the
        # conservative default used for the other DeepSeek entries above rather
        # than an inferred true ceiling.
        max_output_tokens=8_192,
        supports_streaming=True,
        # Confirmed via the live models endpoint: "tools", "tool_choice", and
        # "structured_outputs" are all in this id's supported_parameters.
        supports_tools=True,
        # Confirmed: architecture.input_modalities is text-only for this id.
        supports_vision=False,
        # Not independently verified for this id -- inherited from the same
        # vendor-family requirement on the deepseek-chat-v3-0324/deepseek-r1
        # entries above.
        requires_alternating_roles=True,
        supports_anthropic_cache_control=False,
        # Confirmed: the live pricing block includes a real "input_cache_read"
        # rate for this id, same auto-cache signal as deepseek-chat-v3-0324.
        supports_openai_auto_cache=True,
        min_cacheable_tokens=0,
        cache_ttl_seconds=0,
        # "structured_outputs" is in this id's supported_parameters (the same
        # 2026-07-23 catalogue read as supports_tools above). the other
        # request-settings facts were not checked against the catalogue, so
        # they stay None.
        structured_output=True,
    ),
}


for _model_id, _caps in _OPENROUTER_CAPABILITIES.items():
    register_capabilities(_model_id, _caps)
