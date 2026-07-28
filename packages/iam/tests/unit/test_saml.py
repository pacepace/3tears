"""SAML service-provider helpers."""

from __future__ import annotations

import base64
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx
import pytest

from threetears.iam.saml import (
    AuthnResponseLike,
    SamlError,
    SamlMetadataResolver,
    extract_identity,
    peek_entity_id,
    peek_in_response_to,
    relay_state_allowed,
)

_METADATA = (
    '<?xml version="1.0"?>'
    '<EntityDescriptor xmlns="urn:oasis:names:tc:SAML:2.0:metadata" entityID="https://idp.example/saml">'
    "</EntityDescriptor>"
)

#: The classic XML external-entity attack. defusedxml is what makes reading an attribute off
#: attacker-supplied XML safe; plain ElementTree would attempt the file read.
_XXE = (
    '<?xml version="1.0"?>'
    '<!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]>'
    '<EntityDescriptor entityID="&xxe;"></EntityDescriptor>'
)

#: Billion laughs -- exponential entity expansion, a denial of service against the parser.
_BILLION_LAUGHS = (
    '<?xml version="1.0"?>'
    "<!DOCTYPE lolz ["
    '<!ENTITY lol "lol">'
    '<!ENTITY lol2 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">'
    '<!ENTITY lol3 "&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;">'
    "]>"
    '<EntityDescriptor entityID="&lol3;"></EntityDescriptor>'
)


@dataclass(frozen=True)
class _NameId:
    """Stand-in for a pysaml2 NameID.

    # parity-with: threetears.iam.saml.AuthnResponseLike
    """

    text: str | None


@dataclass(frozen=True)
class _Response:
    """Stand-in for a verified pysaml2 AuthnResponse.

    # parity-with: threetears.iam.saml.AuthnResponseLike
    """

    name_id: _NameId | None
    ava: dict[str, Any] | None


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> SamlMetadataResolver:
    return SamlMetadataResolver(httpx.AsyncClient(transport=httpx.MockTransport(handler)))


def _response_b64(in_response_to: str | None = "req-1") -> str:
    attribute = f' InResponseTo="{in_response_to}"' if in_response_to is not None else ""
    xml = f'<?xml version="1.0"?><Response{attribute}></Response>'
    return base64.b64encode(xml.encode("utf-8")).decode("ascii")


def test_the_protocol_stand_in_matches() -> None:
    assert isinstance(_Response(name_id=_NameId("s"), ava={}), AuthnResponseLike)


def test_peek_entity_id_reads_the_attribute() -> None:
    assert peek_entity_id(_METADATA) == "https://idp.example/saml"


@pytest.mark.parametrize("payload", ["", "not xml", "<unclosed>", _XXE, _BILLION_LAUGHS])
def test_peek_entity_id_fails_closed_on_hostile_or_malformed_xml(payload: str) -> None:
    # Never raises, never expands an entity, never reads a file: a misconfigured or hostile
    # metadata URL is a denial, not a crash and not a file disclosure.
    assert peek_entity_id(payload) in (None, "")


def test_peek_in_response_to_reads_the_attribute() -> None:
    assert peek_in_response_to(_response_b64("req-42")) == "req-42"


def test_peek_in_response_to_returns_none_when_absent() -> None:
    assert peek_in_response_to(_response_b64(None)) is None


@pytest.mark.parametrize("payload", ["", "not-base64!!", "YQ==", base64.b64encode(_XXE.encode()).decode()])
def test_peek_in_response_to_fails_closed(payload: str) -> None:
    assert peek_in_response_to(payload) in (None, "")


def test_relay_state_matches_only_the_issued_value() -> None:
    # The open-redirect defence: a Response carrying a relay state this service did not issue
    # must be rejected, never followed.
    assert relay_state_allowed(issued="abc", presented="abc")
    assert not relay_state_allowed(issued="abc", presented="abd")
    assert not relay_state_allowed(issued="abc", presented="")


def test_an_empty_issued_relay_state_matches_nothing() -> None:
    # Otherwise a connection that never issued one would accept an empty presented value.
    assert not relay_state_allowed(issued="", presented="")


async def test_metadata_resolver_fetches_and_caches() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, text=_METADATA)

    resolver = _client(handler)
    metadata = await resolver.discover("https://idp.example/metadata")
    assert metadata.entity_id == "https://idp.example/saml"
    assert metadata.metadata_xml == _METADATA
    await resolver.discover("https://idp.example/metadata")
    assert len(calls) == 1

    resolver.forget("https://idp.example/metadata")
    await resolver.discover("https://idp.example/metadata")
    assert len(calls) == 2


async def test_metadata_resolver_rejects_a_document_without_an_entity_id() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text='<?xml version="1.0"?><EntityDescriptor></EntityDescriptor>')

    with pytest.raises(SamlError, match="entityID"):
        await _client(handler).discover("https://idp.example/metadata")


async def test_metadata_resolver_surfaces_a_transport_failure() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("unreachable")

    with pytest.raises(SamlError, match="fetch failed"):
        await _client(handler).discover("https://idp.example/metadata")


def test_extract_identity_reads_the_name_id_and_attributes() -> None:
    identity = extract_identity(
        _Response(
            name_id=_NameId("user@acme.com"),
            ava={"email": ["user@acme.com"], "displayName": ["Ada"], "email_verified": ["true"]},
        )
    )
    assert identity.subject == "user@acme.com"
    assert identity.email == "user@acme.com"
    assert identity.display_name == "Ada"
    assert identity.email_verified


def test_extract_identity_falls_back_to_the_name_attribute() -> None:
    identity = extract_identity(_Response(name_id=_NameId("u"), ava={"name": ["Grace"]}))
    assert identity.display_name == "Grace"


def test_extract_identity_defaults_email_verified_to_false() -> None:
    # SAML providers rarely assert it; assuming verified would let an attacker claim a local
    # account by asserting someone else's address.
    identity = extract_identity(_Response(name_id=_NameId("u"), ava={"email": ["a@b.c"]}))
    assert not identity.email_verified


@pytest.mark.parametrize("response", [_Response(name_id=None, ava={}), _Response(name_id=_NameId(None), ava={})])
def test_extract_identity_requires_a_name_id(response: _Response) -> None:
    # SAML's Subject is the analogue of OIDC's sub; without one there is nothing to resolve.
    with pytest.raises(SamlError, match="NameID"):
        extract_identity(response)


def test_extract_identity_tolerates_absent_attributes() -> None:
    identity = extract_identity(_Response(name_id=_NameId("u"), ava=None))
    assert identity.subject == "u"
    assert identity.email is None
    assert identity.display_name is None
    assert identity.claims == {}


def test_multi_valued_attributes_take_the_first_value() -> None:
    identity = extract_identity(_Response(name_id=_NameId("u"), ava={"email": ["first@b.c", "second@b.c"]}))
    assert identity.email == "first@b.c"
    # The full multi-valued set stays available for claim mapping.
    assert identity.claims["email"] == ["first@b.c", "second@b.c"]
