"""provider-agnostic tool-name translation for chat-model wire boundaries.

The canonical 3tears tool name is a dotted form (``threetears.calculator``,
``aibots.admin.agent_management``). Several upstream model APIs validate
tool names against ``^[a-zA-Z0-9_-]{1,128}$`` and reject the dot:

- Anthropic Messages API (``claude-*-direct``): ``invalid_request_error``
  with the regex above explicitly in the message.
- AWS Bedrock-routed Anthropic + most OpenAI-compat backends accessed via
  OpenRouter: same regex, surfaced as a 4xx from the upstream provider.
- OpenAI Responses / chat-completions tools surface: same regex, sometimes
  different text but the constraint is identical.

This module provides the wire-side translation primitives so EVERY chat
provider can wear the same translation layer without duplicating it. The
two pieces are:

- :class:`NameMangledToolProxy` -- a LangChain :class:`BaseTool` whose
  ``.name`` is the dot-replaced wire form, delegating execution to the
  original tool. The proxy preserves description and ``args_schema``
  verbatim, so the LLM still sees the unchanged tool spec aside from
  the name.
- :func:`build_name_translation` -- builds a ``(wire_tools, reverse_map)``
  tuple from an application-side tool list. ``wire_tools`` is the list
  the chat model binds to; ``reverse_map`` maps each underscored wire
  name back to its canonical dotted form so streaming / non-streaming
  responses can be un-translated before reaching application code.
- :func:`reverse_translate_message` -- mutates an
  ``AIMessage`` / ``AIMessageChunk`` in place, rewriting every
  ``tool_calls`` / ``tool_call_chunks`` / ``invalid_tool_calls`` name
  field through ``reverse_map``. Provider-specific
  ``_astream`` / ``_agenerate`` wrappers call this so consumers see
  canonical dotted tool names regardless of which provider's
  validator forced the wire-side rename.

Per-provider integration is a thin subclass that:

1. Overrides ``bind_tools`` to call :func:`build_name_translation` and
   store the resulting reverse map on ``_name_reverse_map`` (a
   :class:`PrivateAttr` on the subclass).
2. Overrides the response surface to call
   :func:`reverse_translate_message` on every message it returns. This
   must cover BOTH the streaming and non-streaming public entry points,
   because LangChain's ``ainvoke`` / ``invoke`` route internally through
   the PROTECTED ``_astream`` / ``_stream`` (not ``_agenerate`` /
   ``_generate``) whenever ``_should_stream()`` is true — e.g. when a
   streaming callback is attached (LangGraph's ``astream_events`` tap).
   The concrete subclasses therefore override the public ``astream`` +
   ``ainvoke`` + ``invoke`` (post-processing the result) AND
   ``_agenerate`` (for the pure non-streaming path). Overriding the
   protected ``_astream`` directly is deliberately avoided — wrapping it
   in another async generator drops ``on_chat_model_stream`` callbacks
   (see the ``astream`` docstrings in the provider modules).

See :mod:`threetears.models.providers.openrouter` and
:mod:`threetears.models.providers.anthropic` for the two concrete
integrations.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool
from pydantic import PrivateAttr

__all__ = [
    "NameMangledToolProxy",
    "build_name_translation",
    "mangle_tool_name",
    "reverse_translate_message",
]


def mangle_tool_name(name: str) -> str:
    """translate a dotted canonical tool name to its underscored wire form.

    Applied before sending tool specs to providers whose validators
    reject dots. The transform is the simplest one that satisfies
    every observed validator (``^[a-zA-Z0-9_-]{1,128}$``); existing
    underscores in the name are preserved so the round-trip is
    lossless when paired with the reverse map built by
    :func:`build_name_translation`.

    :param name: canonical tool name (may contain dots)
    :ptype name: str
    :return: dot-replaced form suitable for strict provider regex
        validators (Bedrock, Anthropic-direct, OpenAI Responses,
        OpenRouter)
    :rtype: str
    """
    return name.replace(".", "_")


class NameMangledToolProxy(BaseTool):
    """LangChain :class:`BaseTool` whose ``.name`` is the wire form,
    delegating execution to a dotted-named original tool.

    Used by every provider's name-translating chat subclass to swap
    each application-side tool for a wire-side proxy at
    ``bind_tools`` time. The proxy keeps the ``description`` and
    ``args_schema`` of the original (those flow to the LLM
    unchanged) but exposes the dot-replaced name so the bound tool
    spec passes strict provider validators.

    Application code never receives this proxy directly -- it lives
    only inside the bound runnable's tool list, and the response
    un-translation layer (:func:`reverse_translate_message`) rewrites
    tool-call names back to the dotted form before any consumer code
    sees them.
    """

    name: str
    description: str
    args_schema: type[Any] | None = None
    _delegate: BaseTool = PrivateAttr()

    def __init__(self, *, delegate: BaseTool, mangled_name: str) -> None:
        """proxy initializer that copies description/args from the delegate.

        :param delegate: original tool whose execution to forward to
        :ptype delegate: BaseTool
        :param mangled_name: dot-replaced wire name to expose
        :ptype mangled_name: str
        """
        super().__init__(
            name=mangled_name,
            description=delegate.description,
            args_schema=delegate.args_schema,
        )
        self._delegate = delegate

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        """forward async execution to the delegate's ``_arun``.

        Calling ``delegate._arun`` rather than ``delegate.arun`` is
        deliberate: ``arun`` is the LangChain wrapper that runs
        callbacks/validation around the subclass's ``_arun`` body.
        Our proxy already wears that wrapper at the proxy level
        (LangChain calls ``proxy.arun`` -> ``proxy._arun``); calling
        ``delegate.arun`` here would double-fire callbacks and
        re-process the input.

        :param args: positional forwarded to the delegate
        :ptype args: Any
        :param kwargs: keyword forwarded to the delegate
        :ptype kwargs: Any
        :return: delegate result
        :rtype: Any
        """
        return await self._delegate._arun(*args, **kwargs)

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        """forward sync execution to the delegate's ``_run``.

        Same double-wrapper rationale as :meth:`_arun`.

        :param args: positional forwarded to the delegate
        :ptype args: Any
        :param kwargs: keyword forwarded to the delegate
        :ptype kwargs: Any
        :return: delegate result
        :rtype: Any
        """
        return self._delegate._run(*args, **kwargs)


def build_name_translation(
    tools: list[Any],
) -> tuple[list[Any], dict[str, str]]:
    """build the wire-side translated tool list + reverse map for un-translation.

    Accepts two input shapes that both occur in production:

    - **BaseTool list** (3tears agent runtime path). Each
      :class:`BaseTool` whose ``.name`` contains a dot is wrapped
      in a :class:`NameMangledToolProxy` carrying the underscored
      wire name; the original tool is kept as the proxy's
      delegate so execution still hits the unchanged
      ``_run`` / ``_arun``.
    - **Tool-spec dict list** (gateway path). Each dict is in the
      provider-native shape that crosses the NATS gateway boundary
      (already serialized for the upstream API). The dict's
      ``name`` field is the mangle target. Two layouts are
      handled: the OpenAI ``{"type":"function","function":{"name":...}}``
      shape and the Anthropic / 3tears-canonical ``{"name":...}``
      flat shape. A shallow-copied dict is returned so the caller's
      original list is not mutated.

    Tools whose canonical name has no dot pass through unchanged
    (no translation needed). The reverse map keys on the
    underscored form so per-provider chat subclasses can rewrite
    ``tool_call`` names in streaming and non-streaming responses.

    :param tools: application-side bind_tools input
        (BaseTool objects OR provider-native tool spec dicts)
    :ptype tools: list[Any]
    :return: tuple of (wire-side tool list, underscored -> dotted
        reverse map)
    :rtype: tuple[list[Any], dict[str, str]]
    """
    wire_tools: list[Any] = []
    reverse_map: dict[str, str] = {}
    for tool in tools:
        if isinstance(tool, dict):
            wire_tool, mapping = _translate_dict_tool(tool)
            wire_tools.append(wire_tool)
            reverse_map.update(mapping)
            continue
        # BaseTool path
        canonical = tool.name
        if "." not in canonical:
            wire_tools.append(tool)
            continue
        mangled = mangle_tool_name(canonical)
        wire_tools.append(
            NameMangledToolProxy(
                delegate=tool,
                mangled_name=mangled,
            )
        )
        reverse_map[mangled] = canonical
    return wire_tools, reverse_map


def _translate_dict_tool(
    tool: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    """translate a single provider-native tool-spec dict.

    Two dict shapes appear in practice:

    - OpenAI / OpenAI-compat chat-completions ``tools`` entry:
      ``{"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}``
      -- the name lives at ``["function"]["name"]``.
    - Anthropic Messages and 3tears-canonical:
      ``{"name": "...", "description": "...", "input_schema": {...}}``
      -- the name lives at the top level.

    Other layouts (no recognised name field) pass through unchanged
    so a future provider extension does not silently drop tools.

    :param tool: provider-native tool spec
    :ptype tool: dict[str, Any]
    :return: tuple of (translated tool spec, single-entry reverse
        map for this tool; empty when no translation was needed)
    :rtype: tuple[dict[str, Any], dict[str, str]]
    """
    # detect the OpenAI-shape nesting first; otherwise fall back to
    # flat ``name`` (Anthropic / canonical).
    fn_block = tool.get("function") if isinstance(tool.get("function"), dict) else None
    if fn_block is not None:
        canonical = fn_block.get("name")
        if not isinstance(canonical, str) or "." not in canonical:
            return tool, {}
        mangled = mangle_tool_name(canonical)
        new_fn = {**fn_block, "name": mangled}
        new_tool = {**tool, "function": new_fn}
        return new_tool, {mangled: canonical}
    canonical = tool.get("name")
    if not isinstance(canonical, str) or "." not in canonical:
        return tool, {}
    mangled = mangle_tool_name(canonical)
    new_tool = {**tool, "name": mangled}
    return new_tool, {mangled: canonical}


def reverse_translate_message(
    message: Any,
    reverse_map: dict[str, str],
) -> None:
    """rewrite tool-call names on ``message`` from wire to canonical form.

    Mutates in place. Called for every chunk yielded from a chat
    model's ``_astream`` and every message in ``_agenerate``'s
    result. Touches three name-bearing fields:

    - ``tool_call_chunks`` -- partial streamed tool calls; the
      ``name`` field arrives once at the start of each call,
      subsequent chunks carry only ``args``.
    - ``tool_calls`` -- the merged, well-formed tool calls
      consumers iterate.
    - ``invalid_tool_calls`` -- the recovery target for malformed
      streaming; consumers (metallm, aibots-agents) attempt to
      re-parse them.

    No-op when the message has no tool-call fields or when the
    reverse map is empty (no tools were bound, so nothing to
    translate).

    :param message: target chat message (``AIMessage`` or
        ``AIMessageChunk``); duck-typed via attribute access
    :ptype message: Any
    :param reverse_map: underscored -> dotted reverse map produced
        by :func:`build_name_translation`
    :ptype reverse_map: dict[str, str]
    """
    if not reverse_map:
        return
    for tc_chunk in getattr(message, "tool_call_chunks", None) or []:
        name = tc_chunk.get("name")
        if name and name in reverse_map:
            tc_chunk["name"] = reverse_map[name]
    for tc in getattr(message, "tool_calls", None) or []:
        name = tc.get("name")
        if name and name in reverse_map:
            tc["name"] = reverse_map[name]
    for tc in getattr(message, "invalid_tool_calls", None) or []:
        name = tc.get("name")
        if name and name in reverse_map:
            tc["name"] = reverse_map[name]
