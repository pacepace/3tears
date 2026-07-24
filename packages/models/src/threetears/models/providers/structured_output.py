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

__all__ = [
    "StructuredOutputError",
    "StructuredOutputSchemaError",
    "StructuredOutputUnsupportedError",
    "ensure_valid_json_schema",
    "structured_output_kwargs",
]


class StructuredOutputError(ValueError):
    """base for structured-output failures caused by the CALLER's request.

    Exists so a consumer can catch every caller-error kind in one clause.
    The distinction that matters to a consumer is not WHICH way the
    request was wrong but that the request -- not the provider -- was at
    fault: these failures all happen LOCALLY, before any provider API
    call, so they must never be classified as a provider outage. A
    consumer that lets one kind fall through to a generic handler will
    read a local rejection as an unreachable provider, and any
    circuit-breaker keyed on that verdict will take the provider out of
    service for every caller over one caller's malformed request.

    Subclasses :class:`ValueError` to match the package's existing
    unresolvable-provider idiom in :mod:`threetears.models.factory`.
    """


class StructuredOutputUnsupportedError(StructuredOutputError):
    """raised when a provider type has no structured-output translation."""


class StructuredOutputSchemaError(StructuredOutputError):
    """raised when the caller's json schema cannot be translated for a provider.

    Sibling of :class:`StructuredOutputUnsupportedError`, deliberately not
    a subclass: the provider is perfectly capable of structured output,
    the supplied schema is what was rejected. Conflating the two makes the
    operator-facing message misreport the cause and sends whoever reads it
    looking at provider configuration instead of at the schema.
    """


def ensure_valid_json_schema(json_schema: dict[str, Any], *, provider_type: str) -> None:
    """validates a caller schema locally, raising a typed caller-error.

    Used by the provider builders that embed the caller's schema
    VERBATIM (openrouter, openai). Those wire shapes have no traversal
    step of their own, so without this check a malformed schema is
    forwarded intact -- and the providers do not reject it. Verified
    live against openrouter -> ``google/gemini-2.5-flash``:
    ``{"type": "not_a_real_json_type"}`` returned the bare string
    ``"Dublin"`` and ``{"type": "object", "properties": "oops"}``
    returned ``{}``, both as SUCCESSFUL completions. A caller typo
    therefore yields a silently degenerate shape, which is the exact
    failure structured output exists to prevent, so the schema is
    checked here instead of trusted to the provider.

    Deliberately validates that the schema IS a json-schema, not that it
    satisfies any provider's additional subset rules (e.g. OpenAI strict
    mode's ``additionalProperties: false``). Enforcing a provider's
    policy here would reject schemas that provider accepts today and
    would rot the moment the policy moves; rejecting non-schemas is a
    stable, provider-independent property.

    Anthropic does NOT route through this: its builder already validates
    while transforming, and layering a second validator ahead of it would
    reject schemas ``transform_schema`` accepts (a schema with no ``type``
    is valid json-schema but is rejected by Anthropic, and the reverse
    ordering would misreport which layer objected).

    :param json_schema: json-schema supplied by the caller
    :ptype json_schema: dict[str, Any]
    :param provider_type: provider type named in the failure message
    :ptype provider_type: str
    :return: nothing; raises when the schema is not a valid json-schema
    :rtype: None
    :raises StructuredOutputSchemaError: when json_schema is not a valid
        json-schema
    """
    # imported inside the function to match this module's per-branch
    # import discipline: the capability-registry side-effect import of
    # the provider modules must not pull validation machinery in.
    from jsonschema import Draft202012Validator
    from jsonschema.exceptions import SchemaError

    try:
        Draft202012Validator.check_schema(json_schema)
    except SchemaError as exc:
        # every structural rejection arrives as SchemaError: an
        # unrecognized "type", a non-mapping "properties", a non-mapping
        # schema, a bad nested subschema. carries ``message`` + the path
        # to the offending keyword rather than ``str(exc)``, which
        # re-renders the entire schema (1.8 kB for a two-key example)
        # and would drown the actionable line in the error response and
        # every log record carrying it.
        location = "/".join(str(part) for part in exc.absolute_path) or "<root>"
        raise StructuredOutputSchemaError(
            f"supplied json schema is not a valid json-schema and cannot be sent to {provider_type}: "
            f"at {location}: {exc.message}",
        ) from exc
    except RecursionError as exc:
        # validation descends once per nesting level with no depth guard,
        # so caller json nested past the interpreter frame limit exhausts
        # the stack (reachable well inside what json.loads accepts).
        # RecursionError is a RuntimeError, NOT a SchemaError, so it needs
        # naming explicitly or it escapes into a consumer's generic
        # handler and trips the circuit breaker over one caller's bad
        # schema. Deliberately explicit rather than a broad catch so a
        # genuine defect in this module still surfaces as itself.
        raise StructuredOutputSchemaError(
            f"supplied json schema is nested too deeply to validate for {provider_type}: RecursionError: {exc}",
        ) from exc


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
    # imported per branch (mirroring threetears.models.factory) so a
    # dispatch for one provider never drags another provider's module in.
    if provider_type == "anthropic":
        from threetears.models.providers.anthropic import anthropic_structured_output_kwargs

        result = anthropic_structured_output_kwargs(json_schema, name=name, strict=strict)
    elif provider_type == "openrouter":
        from threetears.models.providers.openrouter import openrouter_structured_output_kwargs

        result = openrouter_structured_output_kwargs(json_schema, name=name, strict=strict)
    elif provider_type == "openai":
        from threetears.models.providers.openai import openai_structured_output_kwargs

        result = openai_structured_output_kwargs(json_schema, name=name, strict=strict)
    else:
        raise StructuredOutputUnsupportedError(
            f"provider type '{provider_type}' has no structured output translation; "
            "supported provider types are: anthropic, openrouter, openai",
        )
    return result
