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
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import PrivateAttr

from threetears.models.capabilities import ModelCapabilities, register_capabilities
from threetears.models.enums import ModelStatus, ModelTier, ModelType
from threetears.models.providers._name_translation_mixin import NameTranslatingChatMixin
from threetears.models.providers.structured_output import ensure_valid_json_schema

if TYPE_CHECKING:
    from langchain_openrouter import ChatOpenRouter

__all__ = [
    "OPENROUTER_PROVIDER_NAME",
    "create_openrouter_chat",
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
    exact failure structured output exists to prevent.

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

    class _NameTranslatingChatOpenRouter(NameTranslatingChatMixin, ChatOpenRouter):
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

    return _NameTranslatingChatOpenRouter


def create_openrouter_chat(
    model_name: str,
    api_key: str,
    *,
    timeout: int = 120,
    max_retries: int = 2,
    **extra_kwargs: object,
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
    :ptype extra_kwargs: object
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
    kwargs: dict[str, object] = {
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
    ),
}


for _model_id, _caps in _OPENROUTER_CAPABILITIES.items():
    register_capabilities(_model_id, _caps)
