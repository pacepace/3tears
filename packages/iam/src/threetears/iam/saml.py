"""SAML 2.0 service-provider helpers: metadata, identity extraction, relay state.

The SAML equivalent of :mod:`threetears.iam.oidc`. Assertion verification
itself belongs to ``pysaml2`` -- signature validation over XML is not something
to reimplement, and reimplementing it is how XML signature-wrapping bugs get
born. What is here is everything around it.

**Requires the ``saml`` extra**, which pulls ``defusedxml`` and ``pysaml2``.
``pysaml2`` in turn needs the ``xmlsec1`` SYSTEM binary. A consumer doing only
password and OIDC should not inherit an apt-get line, so importing this module
without the extra raises a pointed :class:`ImportError` rather than a confusing
one three frames down.

**XML parsing here is defused, and nothing parsed here is trusted.** The two
peek functions read one attribute off untrusted input using ``defusedxml``,
which is what stops the billion-laughs and external-entity attacks that plain
``ElementTree`` is vulnerable to. Neither peeked value is ever treated as
proof: ``InResponseTo`` is only a lookup key, and the real check is ``pysaml2``
cross-verifying the signed assertion against the single outstanding request
that lookup found. A peeked value that does not correspond to a real, signed,
matching assertion fails that later check -- it does not bypass it.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import httpx

from threetears.iam.oidc import coerce_email_verified
from threetears.observe import get_logger

try:
    from defusedxml import ElementTree as DefusedElementTree
except ImportError as _exc:  # pragma: no cover - exercised by installing without the extra
    raise ImportError(
        "threetears.iam.saml requires the 'saml' extra: pip install '3tears-iam[saml]'. "
        "It also needs the xmlsec1 system binary on PATH."
    ) from _exc

__all__ = [
    "SamlError",
    "SamlIdentity",
    "SamlIdpMetadata",
    "SamlMetadataResolver",
    "extract_identity",
    "peek_entity_id",
    "peek_in_response_to",
    "relay_state_allowed",
]

log = get_logger(__name__)


class SamlError(Exception):
    """A SAML step failed and the login must be denied."""


@dataclass(frozen=True, slots=True)
class SamlIdpMetadata:
    """A fetched identity-provider metadata document.

    :ivar entity_id: the provider's ``entityID``, read off the document itself. Unsigned --
        the trust here is in the HTTPS transport the document arrived over, exactly as with
        an OIDC discovery document.
    :ivar metadata_xml: the raw document, to hand to ``pysaml2``'s inline metadata loader.
    """

    entity_id: str
    metadata_xml: str


@dataclass(frozen=True, slots=True)
class SamlIdentity:
    """The identity an assertion asserts, after verification.

    :ivar subject: the ``NameID``. SAML's analogue of OIDC's ``sub``.
    :ivar email: the asserted address, if any.
    :ivar email_verified: whether the provider asserted the address is verified.
    :ivar display_name: the asserted display name, if any.
    :ivar claims: the full attribute set. SAML attributes are inherently multi-valued, so
        these are lists -- which :func:`~threetears.iam.claim_mapping.resolve_claim_grants`
        already handles without adaptation.
    """

    subject: str
    email: str | None = None
    email_verified: bool = False
    display_name: str | None = None
    claims: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class AuthnResponseLike(Protocol):
    """The part of a verified ``pysaml2`` ``AuthnResponse`` this module reads.

    Structural rather than the concrete class, so :func:`extract_identity` can be tested
    without constructing a full signed SAML response, and so this module does not import
    ``pysaml2`` at all.
    """

    @property
    def ava(self) -> Mapping[str, Any] | None:
        """The attribute-value assertions."""
        ...

    @property
    def name_id(self) -> Any:
        """The subject ``NameID``, carrying a ``.text``."""
        ...


class SamlMetadataResolver:
    """Fetches and caches identity-provider metadata.

    Same caching behaviour, and same stated limitation, as
    :class:`~threetears.iam.oidc.OidcDiscoveryClient`: unbounded in time, so a provider's
    key rotation is picked up on the next restart unless :meth:`forget` is called.
    """

    def __init__(self, client: httpx.AsyncClient) -> None:
        """
        :param client: the HTTP client to fetch through, supplied by the caller so timeouts
            and lifecycle stay under its control.
        :ptype client: httpx.AsyncClient
        """
        self._client = client
        self._cache: dict[str, SamlIdpMetadata] = {}

    async def discover(self, idp_metadata_url: str) -> SamlIdpMetadata:
        """Fetch and cache the metadata at ``idp_metadata_url``.

        :raises SamlError: the fetch failed, or the document carries no ``entityID``.
        """
        cached = self._cache.get(idp_metadata_url)
        if cached is not None:
            return cached
        try:
            response = await self._client.get(idp_metadata_url)
            response.raise_for_status()
            metadata_xml = response.text
        except httpx.HTTPError as exc:
            raise SamlError(f"saml metadata fetch failed ({type(exc).__name__}).") from exc
        entity_id = peek_entity_id(metadata_xml)
        if entity_id is None:
            raise SamlError("saml metadata document has no entityID.")
        metadata = SamlIdpMetadata(entity_id=entity_id, metadata_xml=metadata_xml)
        self._cache[idp_metadata_url] = metadata
        return metadata

    def forget(self, idp_metadata_url: str) -> None:
        """Drop a cached document so the next :meth:`discover` refetches it."""
        self._cache.pop(idp_metadata_url, None)


def peek_entity_id(metadata_xml: str) -> str | None:
    """Read the top-level ``entityID`` off a metadata document, or ``None``.

    An untrusted, defused XML read. Malformed input returns ``None`` rather than raising --
    a misconfigured metadata URL must fail closed as a denial, not crash the caller -- but it
    is logged, because a metadata document that will not parse is a real operational signal.
    """
    root = _parse_xml(metadata_xml.encode("utf-8"), context="IdP metadata")
    if root is None:
        return None
    entity_id = root.attrib.get("entityID")
    # An entityID is a URI parsed off untrusted XML -- the attribute accessor is typed loosely,
    # so this is the parse boundary that pins its type.
    return str(entity_id) if entity_id is not None else None  # convert at border: a URI, not a UUID


def peek_in_response_to(saml_response_b64: str) -> str | None:
    """Read ``InResponseTo`` off a base64 SAML Response, verifying NOTHING.

    Used only as a lookup key for the outstanding-request store. That is what makes reading
    an unverified attribute safe here: the real check is ``pysaml2`` cross-verifying the
    signed assertion against the single outstanding entry this lookup found (module
    docstring). Malformed or non-base64 input returns ``None`` and is logged -- garbage on an
    externally-reachable field is exactly the probing a federation broker should see.
    """
    try:
        xml_bytes = base64.b64decode(saml_response_b64, validate=True)
    except ValueError, TypeError:
        log.warning("saml: SAMLResponse is not valid base64")
        return None
    root = _parse_xml(xml_bytes, context="SAMLResponse")
    if root is None:
        return None
    in_response_to = root.attrib.get("InResponseTo")
    return str(in_response_to) if in_response_to is not None else None


def relay_state_allowed(*, issued: str, presented: str) -> bool:
    """Whether a Response's ``RelayState`` is the one this connection issued.

    The allow-list is exactly one value: each authentication request gets its own
    self-issued relay state, never a caller-supplied one. This is the open-redirect defence
    -- a Response carrying a relay state the service did not issue must be rejected, never
    followed. Compared exactly, and an empty issued value matches nothing.
    """
    return bool(issued) and presented == issued


def extract_identity(response: AuthnResponseLike) -> SamlIdentity:
    """Pull the identity out of an ALREADY-VERIFIED assertion.

    Does no verification of its own -- the caller must have run
    ``Saml2Client.parse_authn_request_response`` first.

    :raises SamlError: the assertion carries no ``NameID``. SAML's Subject is the analogue of
        OIDC's ``sub``; without one there is no identity to resolve or provision.
    """
    name_id = response.name_id
    text = getattr(name_id, "text", None) if name_id is not None else None
    if not text:
        raise SamlError("saml assertion is missing a NameID.")
    attributes: Mapping[str, Any] = dict(response.ava or {})
    display = _first(attributes, "displayName") or _first(attributes, "name")
    email = _first(attributes, "email")
    return SamlIdentity(
        subject=str(text),
        email=str(email) if email else None,
        email_verified=coerce_email_verified(_first(attributes, "email_verified")),
        display_name=str(display) if display else None,
        claims=attributes,
    )


def _first(attributes: Mapping[str, Any], key: str) -> Any:
    """The first value of a multi-valued SAML attribute, or ``None``."""
    values = attributes.get(key)
    if not values:
        return None
    if isinstance(values, str):
        return values
    if isinstance(values, Sequence):
        return values[0] if values else None
    return values


def _parse_xml(payload: bytes, *, context: str) -> Any:
    """Defused-parse ``payload``, returning ``None`` on anything malformed.

    Broad by intent: this runs on externally-reachable input, and the correct answer to
    every parse failure is the same generic denial. Narrowing it would only mean an
    unanticipated parser error escapes as a 500 instead.
    """
    try:
        return DefusedElementTree.fromstring(payload)
    except Exception:
        log.warning("saml: failed to parse %s", context)
        return None
