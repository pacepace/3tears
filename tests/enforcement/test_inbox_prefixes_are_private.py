"""Enforcement: no principal may publish into, or listen on, another principal's request inbox.

The ``request-reply`` sender posture (``threetears.enforcement.sender_posture``) trusts the replies on a
fresh inbox exactly as ``nc.request`` trusts its reply. That holds only while each connection's inbox prefix
(``_INBOX_{principal}_{id}``) is its own: a principal that could publish into another's inbox could forge
the replies to its requests, and one that could subscribe to it could read them. A responder reaches a
requester's inbox only through NATS ``allow_responses`` (the reply subject of a message it received), never
through a standing grant. This checks every principal's grants, as resolved and as minted into its user JWT,
against every other principal's inbox.
"""

from __future__ import annotations

import base64
import json

import pytest

from threetears.nats.subject_permissions import Principal, PrincipalPermissions, build_permissions
from threetears.nats.subjects import set_default_namespace
from threetears.nats.user_jwt import generate_account_seed, mint_user_jwt

pytestmark = pytest.mark.enforcement

_NS = "3tears"

#: two connections of each principal, so a principal's grants are checked against another of its own kind too
_IDS: dict[Principal, tuple[dict[str, str], dict[str, str]]] = {
    Principal.AGENT_POD: (
        {"agent_id": "019470a8-b5c3-7def-8123-0000000000aa", "pod_id": "01947100-0000-7000-8000-0000000000aa"},
        {"agent_id": "019470a8-b5c3-7def-8123-0000000000bb", "pod_id": "01947100-0000-7000-8000-0000000000bb"},
    ),
    Principal.TOOL_POD: (
        {"pod_id": "01947100-0000-7000-8000-0000000000cc"},
        {"pod_id": "01947100-0000-7000-8000-0000000000dd"},
    ),
    Principal.REGISTRY: ({"conn_id": "reg-1"}, {"conn_id": "reg-2"}),
    Principal.HUB: ({"conn_id": "hub-1"}, {"conn_id": "hub-2"}),
    Principal.GATEWAY: ({"conn_id": "gw-1"}, {"conn_id": "gw-2"}),
    Principal.CHANNEL_ADAPTER: ({"conn_id": "chan-1"}, {"conn_id": "chan-2"}),
    Principal.AGENT_ROUTER: ({"conn_id": "router-1"}, {"conn_id": "router-2"}),
    Principal.DATASET_EXECUTOR: ({"conn_id": "dsx-1"}, {"conn_id": "dsx-2"}),
}


@pytest.fixture(autouse=True)
def _bind_namespace() -> None:
    set_default_namespace(_NS)


def _matches(pattern: str, subject: str) -> bool:
    """whether the NATS subject pattern (``*`` one token, ``>`` the rest) covers ``subject``."""
    wanted, tokens = pattern.split("."), subject.split(".")
    for index, token in enumerate(wanted):
        if token == ">":
            return len(tokens) > index
        if index >= len(tokens) or (token != "*" and token != tokens[index]):
            return False
    return len(wanted) == len(tokens)


def _minted_publish(permissions: PrincipalPermissions) -> list[str]:
    token = mint_user_jwt(
        account_seed=generate_account_seed(),
        user_public_key="UTESTUSERPUBLICKEY",
        permissions=permissions,
        name="inbox-privacy",
        expires_in_seconds=300,
    )
    payload = token.split(".")[1]
    claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    allow: list[str] = claims["nats"]["pub"]["allow"]
    return allow


def _connections() -> list[tuple[str, PrincipalPermissions]]:
    return [
        (f"{principal.value}#{number}", build_permissions(principal, **ids))
        for principal, pair in _IDS.items()
        for number, ids in enumerate(pair)
    ]


def test_every_connection_has_an_inbox_of_its_own() -> None:
    """Flips if two connections share an inbox prefix, or a prefix is the global ``_INBOX``."""
    prefixes = [permissions.inbox_prefix for _, permissions in _connections()]
    assert len(set(prefixes)) == len(prefixes)
    assert all(prefix.startswith("_INBOX_") for prefix in prefixes)


@pytest.mark.parametrize("principal", list(Principal))
def test_no_principal_can_publish_into_or_listen_on_another_connections_inbox(principal: Principal) -> None:
    """Flips if a grant lets one connection forge or read the replies to another's requests."""
    connections = _connections()
    problems: list[str] = []
    for name, permissions in connections:
        if not name.startswith(f"{principal.value}#"):
            continue
        publish = {*permissions.publish, *_minted_publish(permissions)}
        for other, theirs in connections:
            if other == name:
                continue
            reply = f"{theirs.inbox_prefix}.AbCdEf123.1"
            problems += [
                f"{name} may publish {pattern!r} into {other}'s inbox"
                for pattern in publish
                if _matches(pattern, reply)
            ]
            problems += [
                f"{name} may subscribe {pattern!r} on {other}'s inbox"
                for pattern in permissions.subscribe
                if _matches(pattern, reply)
            ]
    assert problems == []
