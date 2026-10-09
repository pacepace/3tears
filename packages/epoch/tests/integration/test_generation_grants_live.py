"""Integration test: a pod's minted grant follows a table's write generation and cannot advance it, on a live broker.

``test_generation_grants`` checks the grant strings against the keys ``EpochGenerationSource`` and
``EpochGenerationReader`` open. A JetStream request a grant does not cover is never answered -- it
blocks to its deadline and reads as an unreachable broker -- so this runs the reader and the source
themselves, through ``NatsClient``, as an agent pod and as a tool pod minted by ``build_permissions``:

- the pod binds the epoch bucket the hub declared, reads a table's generation, and is pushed its
  latest value through ``watch_key``, which is how a pod follows the access tables;
- the pod's advance is refused and surfaces as ``GenerationUnavailableError``, and the bucket still
  holds the generation the hub wrote.

As in ``threetears.nats``'s live grant tests, the minted allow-lists are applied as config-mode
``authorization`` permissions: the grant strings are what is under test.

Gated on docker: a checkout without docker skips cleanly.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
from collections.abc import Iterator
from contextlib import aclosing
from datetime import timedelta
from pathlib import Path

import pytest

from threetears.core.exceptions import GenerationUnavailableError
from threetears.core.testing.containers import check_docker_available
from threetears.epoch import EpochGenerationReader, EpochGenerationSource
from threetears.nats import NatsClient
from threetears.nats.subject_permissions import Principal, PrincipalPermissions, build_permissions
from threetears.nats.subjects import get_default_namespace, set_default_namespace
from threetears.nats.user_jwt import generate_account_seed, mint_user_jwt

pytestmark = pytest.mark.integration

_NS = "gengrantlive"
_AGENT = "019470a8-b5c3-7def-8123-0000000000aa"
_POD = "01947100-0000-7000-8000-0000000000aa"
_TABLE = "group_members"
_ADMIN_PW = "admin-pw"  # noqa: S105 - ephemeral testcontainer credential
_POD_PW = "pod-pw"  # noqa: S105 - ephemeral testcontainer credential
_WAIT = 10.0


def _minted_allow_lists(permissions: PrincipalPermissions) -> tuple[list[str], list[str]]:
    """mint a real user JWT for ``permissions`` and return its (pub allow, sub allow) lists.

    :param permissions: the resolved allow-list to mint
    :ptype permissions: PrincipalPermissions
    :return: the JWT's publish allow-list and subscribe allow-list
    :rtype: tuple[list[str], list[str]]
    """
    token = mint_user_jwt(
        account_seed=generate_account_seed(),
        user_public_key="UTEST",
        permissions=permissions,
        name="generation-grant-live",
        expires_in_seconds=600,
    )
    payload_seg = token.split(".")[1]
    payload = json.loads(base64.urlsafe_b64decode(payload_seg + "=" * (-len(payload_seg) % 4)))
    return payload["nats"]["pub"]["allow"], payload["nats"]["sub"]["allow"]


def _server_config(pub_allow: list[str], sub_allow: list[str]) -> str:
    """a JetStream nats-server config with a full admin and the pod user under its minted allow-lists.

    :param pub_allow: the minted publish allow-list
    :ptype pub_allow: list[str]
    :param sub_allow: the minted subscribe allow-list
    :ptype sub_allow: list[str]
    :return: the nats-server configuration text
    :rtype: str
    """
    authorization = {
        "users": [
            {
                "user": "admin",
                "password": _ADMIN_PW,
                "permissions": {"publish": ">", "subscribe": ">", "allow_responses": True},
            },
            {
                "user": "pod",
                "password": _POD_PW,
                "permissions": {
                    "publish": {"allow": pub_allow},
                    "subscribe": {"allow": sub_allow},
                    "allow_responses": True,
                },
            },
        ]
    }
    return f"jetstream {{ store_dir: /tmp/js-store }}\nport: 4222\nauthorization {json.dumps(authorization)}\n"


@contextlib.contextmanager
def _nats_with_auth(config_text: str, conf_dir: Path) -> Iterator[str]:
    """start a JetStream nats-server with a custom ``authorization`` config; yield its URI.

    :param config_text: the server configuration
    :ptype config_text: str
    :param conf_dir: a directory to mount the configuration from
    :ptype conf_dir: Path
    :return: the server's URI, yielded
    :rtype: Iterator[str]
    """
    from testcontainers.nats import NatsContainer  # noqa: PLC0415

    (conf_dir / "nats.conf").write_text(config_text)
    container = (
        NatsContainer(jetstream=False)
        .with_volume_mapping(str(conf_dir), "/etc/nats", "ro")
        .with_command(["-c", "/etc/nats/nats.conf"])
    )
    container.start()
    try:
        yield container.nats_uri()
    finally:
        container.stop()


def _permissions(principal: Principal) -> PrincipalPermissions:
    """the minted permissions of one pod principal, in this test's namespace.

    :param principal: the agent pod or the tool pod
    :ptype principal: Principal
    :return: its permissions
    :rtype: PrincipalPermissions
    """
    previous = get_default_namespace()
    set_default_namespace(_NS)
    try:
        if principal is Principal.AGENT_POD:
            return build_permissions(principal, agent_id=_AGENT, pod_id=_POD, conn_id=_POD)
        if principal is Principal.REGISTRY:
            return build_permissions(principal, conn_id=_POD)
        return build_permissions(principal, pod_id=_POD, conn_id=_POD)
    finally:
        set_default_namespace(previous)


async def _connect(uri: str, *, user: str, password: str, inbox_prefix: str | None) -> NatsClient:
    """connect a 3tears client as one of the server's users.

    :param uri: the server URI
    :ptype uri: str
    :param user: the user
    :ptype user: str
    :param password: its password
    :ptype password: str
    :param inbox_prefix: the inbox prefix the user's subscribe grant covers, ``None`` for the default
    :ptype inbox_prefix: str | None
    :return: the connected client
    :rtype: NatsClient
    """
    extra = {} if inbox_prefix is None else {"inbox_prefix": inbox_prefix}
    return await NatsClient.connect(
        nats_url=uri,
        nats_subject_namespace=_NS,
        client_name=f"generation-grant-live-{user}",
        user=user,
        password=password,
        startup_timeout=timedelta(seconds=10),
        **extra,
    )


@pytest.mark.parametrize("principal", [Principal.AGENT_POD, Principal.TOOL_POD, Principal.REGISTRY])
async def test_a_pod_follows_a_generation_and_cannot_advance_it(principal: Principal, tmp_path: Path) -> None:
    """the reader binds, reads and watches under the pod's grant; the pod's own advance is refused."""
    if not check_docker_available():
        pytest.skip("Docker not available")

    permissions = _permissions(principal)
    pub_allow, sub_allow = _minted_allow_lists(permissions)

    with _nats_with_auth(_server_config(pub_allow, sub_allow), tmp_path) as uri:
        hub = await _connect(uri, user="admin", password=_ADMIN_PW, inbox_prefix=None)
        pod: NatsClient | None = None
        try:
            # the hub declares the bucket and writes the table's generation, as its broker will
            hub_source = EpochGenerationSource(hub)
            written = await hub_source.advance(_TABLE)

            pod = await _connect(uri, user="pod", password=_POD_PW, inbox_prefix=permissions.inbox_prefix)
            reader = EpochGenerationReader(pod)

            # === SUCCEEDS: bind, read, and the watch a pod follows the access tables by ==========
            assert await reader.read(_TABLE) == written
            async with aclosing(reader.watch(_TABLE)) as pushed:
                assert await asyncio.wait_for(anext(pushed), timeout=_WAIT) == written
                again = await hub_source.advance(_TABLE)
                assert await asyncio.wait_for(anext(pushed), timeout=_WAIT) == again

            # === REFUSED: the pod advancing the generation itself ================================
            with pytest.raises(GenerationUnavailableError):
                await EpochGenerationSource(pod).advance(_TABLE)
            assert await hub_source.current(_TABLE) == again
        finally:
            if pod is not None:
                await pod.shutdown(drain_timeout=timedelta(seconds=2))
            await hub.shutdown(drain_timeout=timedelta(seconds=2))
