"""provider dispatch for structured-output bind kwargs.

Every provider spells "return json matching this schema" differently:
Anthropic uses ``output_config``, OpenRouter takes a top-level
``response_format`` plus a ``provider`` routing block, OpenAI needs the
same ``response_format`` nested under ``extra_body`` to stay on the plain
completions path. That is the same class of wire quirk as the tool-name
dot restriction handled by
:mod:`threetears.models.providers._name_translation_mixin`, and it is
owned here for the same reason: application code above the provider layer
should never see the wire form.

:func:`structured_output_kwargs` maps a provider type to the correct
per-provider builder. The builders return bind KWARGS rather than a bound
model because callers frequently apply structured output to a model that
is already a ``bind_tools`` ``RunnableBinding`` -- that object exposes
``.bind(**kwargs)`` but none of the provider wrapper's own methods.

Every builder rejects a malformed schema locally, before any provider is
contacted, so the caller learns the schema is wrong instead of receiving
a plausible-looking answer built from it. Anthropic gets that for free
from ``transform_schema``; the builders that embed the schema verbatim
call :func:`ensure_valid_json_schema` to reach the same guarantee.
"""

from __future__ import annotations

from typing import Any

from threetears.models.providers._structured_output_schema import (
    StructuredOutputError,
    StructuredOutputSchemaError,
    StructuredOutputUnsupportedError,
    ensure_valid_json_schema,
)
from threetears.models.providers.anthropic import anthropic_structured_output_kwargs
from threetears.models.providers.openai import openai_structured_output_kwargs
from threetears.models.providers.openrouter import openrouter_structured_output_kwargs

__all__ = [
    "StructuredOutputError",
    "StructuredOutputSchemaError",
    "StructuredOutputUnsupportedError",
    "ensure_valid_json_schema",
    "structured_output_kwargs",
]


def structured_output_kwargs(
    provider_type: str,
    json_schema: dict[str, Any],
    *,
    name: str = "response",
    strict: bool = True,
) -> dict[str, Any]:
    """builds the provider-native structured-output bind kwargs.

    Dispatches to the per-provider builder and returns kwargs suitable for
    ``model.bind(**kwargs)``. Never returns an empty dict: an empty result
    would drop the directive silently and let the model answer in prose,
    which is precisely the failure structured output exists to prevent, so
    an untranslatable provider raises instead.

    :param provider_type: resolved provider type (``anthropic``,
        ``openrouter``, ``openai``)
    :ptype provider_type: str
    :param json_schema: json-schema the response must satisfy
    :ptype json_schema: dict[str, Any]
    :param name: schema name reported to the provider
    :ptype name: str
    :param strict: whether the provider must enforce the schema exactly
    :ptype strict: bool
    :return: kwargs to pass to ``model.bind(**kwargs)``
    :rtype: dict[str, Any]
    :raises StructuredOutputUnsupportedError: when provider_type has no
        structured-output translation
    :raises StructuredOutputSchemaError: when json_schema cannot be
        translated for provider_type (raised by the per-provider builder)
    """
    if provider_type == "anthropic":
        result = anthropic_structured_output_kwargs(json_schema, name=name, strict=strict)
    elif provider_type == "openrouter":
        result = openrouter_structured_output_kwargs(json_schema, name=name, strict=strict)
    elif provider_type == "openai":
        result = openai_structured_output_kwargs(json_schema, name=name, strict=strict)
    else:
        raise StructuredOutputUnsupportedError(
            f"provider type '{provider_type}' has no structured output translation; "
            "supported provider types are: anthropic, openrouter, openai",
        )
    return result
