"""the structured-output caller errors, and the local json-schema check that raises one.

Kept apart from :mod:`threetears.models.providers.structured_output` because the provider modules
raise these and the dispatcher there imports the provider modules: defined here, below both, each
imports it without a cycle. :mod:`~threetears.models.providers.structured_output` re-exports every
name, and that is where callers import them from.
"""

from __future__ import annotations

from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

__all__ = [
    "StructuredOutputError",
    "StructuredOutputSchemaError",
    "StructuredOutputUnsupportedError",
    "ensure_valid_json_schema",
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
