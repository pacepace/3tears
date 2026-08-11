"""Shared plumbing for provider adapters -- the shape every adapter repeats.

Private to the adapters subpackage: nothing here is re-exported, and nothing
here encodes one provider's semantics (that stays in ``searxng.py`` and
``tavily.py``, each mapped onto the contract in its own module docstring).
What lives here is identical across every current adapter for a structural
reason, not a coincidental one:

- :func:`parsed_base_url` is D21/SR-K1 stated once -- a deployment's base
  URL is compiled-in configuration, never a caller or environment value,
  and a non-HTTP(S) scheme is refused here rather than at the socket.
- :class:`_DispositionPlan` is SR-B2/SR-B3 stated once: every criterion a
  request carries gets exactly one recorded answer, and the list an
  adapter's own ``_Plan`` appends to -- and the single method that appends
  to it -- is the same shape regardless of what that provider can express.
- :func:`attributed_failure` is D8/D20/SR-A3 stated once: what only the
  adapter (never the transport) can attribute to a failure -- which
  configured instance, which egress, when -- is the same three facts for
  every provider.
- :func:`decode_results_payload` is the shared shape behind every adapter's
  own ``_decode``: parse the body, require a JSON object carrying a
  ``results`` list, map anything else onto :class:`MalformedResponse`.
  Parameterised by the provider's own name and an optional remediation for
  a non-JSON body, which are the only things that vary between them today.
- ``_as_float``, ``_as_str`` and ``_string_list`` are payload readers with
  no provider semantics baked in: probing an untyped JSON value for a
  shape, and answering ``None``/``[]`` rather than raising or inventing one,
  is not a per-provider decision.

A helper that differs even slightly between the two adapters -- SearXNG's
``_Refused``-based criterion validation (SR-B3's teaching-error shape,
which Tavily does not share), Tavily's ISO/RFC-2822 published-date fallback,
either provider's own ``max-results`` handling -- stays local to that module
with a comment saying why, rather than being generalised into a parameter
neither adapter needs today.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from urllib.parse import ParseResult, urlparse

from threetears.search.contracts import (
    CriterionDisposition,
    Disposition,
    MalformedResponse,
    SearchFailure,
    Spend,
    TransportResponse,
)

__all__: list[str] = []


def _as_float(value: object) -> float | None:
    """Read ``value`` as a float, or ``None`` when it is not numeric.

    A bool is not a number here: ``True`` is a provider saying something
    other than "one", and reading it as 1.0 would invent a measurement.

    :param value: a header or payload value
    :ptype value: object
    :return: the float, or None when the value cannot be one
    :rtype: float | None
    """
    if isinstance(value, bool) or not isinstance(value, int | float | str):
        return None
    # Probing a provider value for a number: a non-numeric value is the provider
    # not reporting one, which the caller reads as an absence rather than as a
    # zero. Absence is the honest answer and there is nothing to log per result.
    try:
        return float(value)
    # NOSILENT: a non-numeric provider value means nothing was reported
    except ValueError:
        return None


def _as_str(value: object) -> str | None:
    """Read ``value`` as a non-empty string, or ``None``.

    :param value: a JSON value from a provider payload
    :ptype value: object
    :return: the string, or None when it is absent or not a string
    :rtype: str | None
    """
    return value if isinstance(value, str) and value else None


def _string_list(value: object) -> list[str]:
    """Read a provider or criterion value as a list of strings.

    :param value: a string, or a sequence of values to stringify
    :ptype value: object
    :return: the values as strings; empty when there are none
    :rtype: list[str]
    """
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return [str(item) for item in value]
    return []


def parsed_base_url(base_url: str) -> ParseResult:
    """Parse and validate a deployment's configured base URL (D21, SR-K1).

    ``base_url`` MUST come from deployment config, never from a caller or
    the environment -- this only checks its shape, not its provenance. A
    non-HTTP(S) scheme is refused here rather than at the socket, so a
    misconfigured deployment fails at construction with a message naming
    the value, not on the first search with a stack trace from the
    transport.

    :param base_url: the instance's or the key's base URL
    :ptype base_url: str
    :return: the parsed URL
    :rtype: ParseResult
    :raises ValueError: when ``base_url`` is not an absolute http(s) URL
    """
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"base_url must be an absolute http(s) URL from deployment config, got {base_url!r}")
    return parsed


class _DispositionPlan:
    """Base for a provider's own ``_Plan``: the dispositions list, shared.

    Not a contract type -- like every provider's ``_Plan`` it never leaves
    its module. SR-B2/SR-B3 -- every criterion the request carried gets
    exactly one recorded answer, and an answer is never "nothing" -- does
    not vary by provider even though what a provider can express does, so
    the list and the single place that appends to it live here once rather
    than once per adapter.
    """

    def __init__(self) -> None:
        self.dispositions: list[CriterionDisposition] = []

    def answer(self, key: str, disposition: Disposition, detail: str | None = None) -> None:
        """Record how one criterion was handled.

        :param key: the criterion key being answered for
        :ptype key: str
        :param disposition: how it was handled
        :ptype disposition: Disposition
        :param detail: specifics -- why unsatisfiable, which rule applied
        :ptype detail: str | None
        """
        self.dispositions.append(CriterionDisposition(criterion_key=key, disposition=disposition, detail=detail))


def attributed_failure(failure: SearchFailure, *, provider_instance: str, egress_name: str) -> SearchFailure:
    """Re-stamp a failure with what only the adapter can attribute (D8/D20, SR-A3).

    The transport knows attempts, elapsed and bytes; only the adapter knows
    which configured instance or key the call was for, which egress its
    transport's requests leave by, and -- when the transport did not
    already say -- when the failure happened. Whatever the failure already
    carries is kept: a transport that stamped its own egress said something
    truer than the adapter's view of it.

    :param failure: the failure about to leave the adapter
    :ptype failure: SearchFailure
    :param provider_instance: the configured instance or key this adapter reaches
    :ptype provider_instance: str
    :param egress_name: the transport's own egress name
    :ptype egress_name: str
    :return: the same failure class, fully attributed
    :rtype: SearchFailure
    """
    updates: dict[str, object] = {}
    if failure.provider_instance != provider_instance:
        updates["provider_instance"] = provider_instance
    if failure.egress is None:
        updates["egress"] = egress_name
    if failure.occurred_at is None:
        updates["occurred_at"] = datetime.now(UTC)
    if not updates:
        return failure
    return failure.to_record().model_copy(update=updates).to_failure()


def decode_results_payload(
    response: TransportResponse,
    spend: Spend,
    *,
    provider_name: str,
    provider_instance: str,
    not_json_remediation: str | None = None,
) -> Mapping[str, object]:
    """Parse a provider's JSON body, requiring a ``results`` list.

    The shared shape behind every adapter's own ``_decode``: parse, and map
    anything that is not a JSON object carrying a ``results`` list onto
    :class:`MalformedResponse`, naming the provider and instance that
    answered. ``not_json_remediation`` is the one thing that varies between
    today's adapters -- SearXNG's 403-vs-non-JSON confusion shares a root
    cause worth naming in the error; Tavily's non-JSON body has no such
    common cause to point at.

    :param response: the successful exchange
    :ptype response: TransportResponse
    :param spend: what the call consumed, carried onto any failure
    :ptype spend: Spend
    :param provider_name: the provider's name, for the message
    :ptype provider_name: str
    :param provider_instance: the configured instance or key, for the message
    :ptype provider_instance: str
    :param not_json_remediation: remediation for a non-JSON body, when the
        provider has a known common cause; None when it has none
    :ptype not_json_remediation: str | None
    :return: the decoded payload
    :rtype: Mapping[str, object]
    :raises MalformedResponse: when the body is not a JSON object, or
        carries no ``results`` list
    """
    try:
        payload = json.loads(response.body)
    except ValueError as exc:
        raise MalformedResponse(
            f"{provider_name} instance {provider_instance} answered {response.status_code} with a body "
            f"that is not JSON",
            spend=spend,
            provider_instance=provider_instance,
            remediation=not_json_remediation,
        ) from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise MalformedResponse(
            f"{provider_name} instance {provider_instance} answered JSON without a 'results' list",
            spend=spend,
            provider_instance=provider_instance,
        )
    return payload
