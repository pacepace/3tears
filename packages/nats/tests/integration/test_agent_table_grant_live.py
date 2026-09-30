"""Integration test: a tool pod granted an agent's tables reaches exactly those keys, on a live broker.

The grant under test is :attr:`JsCapability.KV_TABLE_SCOPED` on the shared ``{ns}-collections``
bucket, as the tool-pod resolver declares it from :class:`AgentTableGrant` and :func:`mint_user_jwt`
renders it. A unit test can only say the strings look right; a JetStream request the grant does not
cover is never answered -- it blocks to its deadline and reads as an unreachable broker -- so each
claim is run against a real nats-server, through the same nats-py KV calls the collection layer's L2
path makes (bind, get, put, compare-and-set update, create, delete):

- a WRITE grant: the pod binds the bucket and reads, puts, updates at a revision, creates and deletes
  keys of the granted table under the OWNER's scope;
- a READ grant: the pod reads that table's keys and every write to them is refused;
- another table of the same owner, and the same table of another owner, are refused on read and on
  write;
- the pod's own scope still works beside the grants.

As in ``test_data_version_key_grant_live``, the minted allow-lists are applied as config-mode
``authorization`` permissions rather than through a live auth-callout responder: the grant strings are
what is under test, and this is the credential the server would enforce.

Gated on docker: a checkout without docker skips cleanly.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any
from uuid import UUID

import nats
import nats.errors
import nats.js.errors
import pytest

from threetears.core.testing.containers import check_docker_available
from threetears.nats.subject_permissions import (
    AgentTableGrant,
    Principal,
    PrincipalPermissions,
    build_permissions,
    kv_key_scope_for,
)
from threetears.nats.subjects import get_default_namespace, set_default_namespace
from threetears.nats.user_jwt import generate_account_seed, mint_user_jwt

pytestmark = pytest.mark.integration

_NS = "atlive"
_OWNER = UUID("019470a8-b5c3-7def-8123-0000000000aa")
_OTHER_OWNER = UUID("019470a8-b5c3-7def-8123-0000000000bb")
_POD = "01947100-0000-7000-8000-0000000000aa"
_WRITE_TABLE = "responses"
_READ_TABLE = "sessions"
_ADMIN_PW = "admin-pw"  # noqa: S105 - ephemeral testcontainer credential
_POD_PW = "pod-pw"  # noqa: S105 - ephemeral testcontainer credential

#: what a refused JetStream call surfaces as in nats-py: an unanswered request.
_REFUSED = (nats.errors.TimeoutError, nats.errors.NoRespondersError)


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
        name="agent-table-live",
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


@contextlib.asynccontextmanager
async def _connect(uri: str, *, user: str, password: str, inbox_prefix: str, errors: list[str]) -> AsyncIterator[Any]:
    """connect a raw nats client, routing async permission-violation errors into ``errors``.

    :param uri: the server URI
    :ptype uri: str
    :param user: the user to connect as
    :ptype user: str
    :param password: that user's password
    :ptype password: str
    :param inbox_prefix: the inbox prefix the user's subscribe grant covers
    :ptype inbox_prefix: str
    :param errors: collects every asynchronous error the client reports
    :ptype errors: list[str]
    :return: the connected client, yielded
    :rtype: AsyncIterator[Any]
    """

    async def _err_cb(exc: Exception) -> None:
        errors.append(str(exc))

    nc = await nats.connect(
        uri,
        user=user,
        password=password,
        inbox_prefix=inbox_prefix.encode(),
        error_cb=_err_cb,
        max_reconnect_attempts=0,
    )
    try:
        yield nc
    finally:
        await nc.close()


async def test_a_tool_pod_reaches_exactly_the_agent_tables_it_was_granted(tmp_path: Path) -> None:
    """both halves against a real broker: the granted table works, every other route is refused."""
    if not check_docker_available():
        pytest.skip("Docker not available")

    previous_ns = get_default_namespace()
    set_default_namespace(_NS)
    try:
        permissions = build_permissions(
            Principal.TOOL_POD,
            pod_id=_POD,
            conn_id=_POD,
            agent_table_grants=(
                AgentTableGrant(owner_agent_id=_OWNER, table=_WRITE_TABLE, writable=True),
                AgentTableGrant(owner_agent_id=_OWNER, table=_READ_TABLE, writable=False),
            ),
        )
        pub_allow, sub_allow = _minted_allow_lists(permissions)
    finally:
        set_default_namespace(previous_ns)

    bucket = f"{_NS}-collections"
    owner = kv_key_scope_for(Principal.AGENT_POD, agent_id=_OWNER)
    other_owner = kv_key_scope_for(Principal.AGENT_POD, agent_id=_OTHER_OWNER)
    own = kv_key_scope_for(Principal.TOOL_POD, pod_id=_POD)
    written = f"{owner}.{_WRITE_TABLE}.row-1"
    read_only = f"{owner}.{_READ_TABLE}.row-1"
    other_table = f"{owner}.conversations.row-1"
    other_owners = f"{other_owner}.{_WRITE_TABLE}.row-1"

    with _nats_with_auth(_server_config(pub_allow, sub_allow), tmp_path) as uri:
        admin_errors: list[str] = []
        async with _connect(
            uri, user="admin", password=_ADMIN_PW, inbox_prefix="_INBOX", errors=admin_errors
        ) as admin_nc:
            admin_js = admin_nc.jetstream()
            # the hub declares the shared bucket with allow_direct; a pod binds it and never creates
            # it, and without allow_direct a read is the body-carried STREAM.MSG.GET no grant narrows.
            admin_kv = await admin_js.create_key_value(bucket=bucket, direct=True)
            await admin_kv.put(written, b"w0")
            await admin_kv.put(read_only, b"r0")
            await admin_kv.put(other_table, b"c0")
            await admin_kv.put(other_owners, b"o0")

            errors: list[str] = []
            async with _connect(
                uri, user="pod", password=_POD_PW, inbox_prefix=permissions.inbox_prefix, errors=errors
            ) as nc:
                js = nc.jetstream(timeout=2)
                await js.account_info()
                kv = await js.key_value(bucket)

                # === SUCCEEDS: every L2 operation on the WRITE-granted table ==================
                entry = await kv.get(written)
                assert entry.value == b"w0"
                revision = await kv.put(written, b"w1")
                revision = await kv.update(written, b"w2", last=revision)
                with pytest.raises(nats.js.errors.KeyWrongLastSequenceError):
                    await kv.update(written, b"stale", last=revision - 1)
                created = f"{owner}.{_WRITE_TABLE}.row-2"
                await kv.create(created, b"n1")
                await kv.delete(created)
                with pytest.raises(nats.js.errors.KeyNotFoundError):
                    await kv.get(created)

                # === SUCCEEDS: a READ of the read-granted table ================================
                assert (await kv.get(read_only)).value == b"r0"

                # === SUCCEEDS: the pod's own scope, beside the grants ==========================
                await kv.put(f"{own}.widgets.w1", b"mine")
                assert (await kv.get(f"{own}.widgets.w1")).value == b"mine"
                assert not [e for e in errors if "permissions violation" in e.lower()], errors

                # === REFUSED: every write to the read-granted table ============================
                with pytest.raises(_REFUSED):
                    await kv.put(read_only, b"r1")
                with pytest.raises(_REFUSED):
                    await kv.delete(read_only)

                # === REFUSED: another table of the owner, and the owner's table elsewhere ======
                for key in (other_table, other_owners):
                    with pytest.raises(_REFUSED):
                        await kv.get(key)
                    with pytest.raises(_REFUSED):
                        await kv.put(key, b"x")

                # === REFUSED: a whole-bucket consumer, which could read every key ==============
                with pytest.raises(_REFUSED):
                    await nc.request(
                        f"$JS.API.CONSUMER.CREATE.KV_{bucket}",
                        json.dumps(
                            {
                                "stream_name": f"KV_{bucket}",
                                "config": {
                                    "filter_subject": f"$KV.{bucket}.{owner}.>",
                                    "deliver_subject": f"{permissions.inbox_prefix}.spy",
                                },
                            }
                        ).encode(),
                        timeout=2,
                    )

                await asyncio.sleep(0.4)  # let the async -ERR frames land in the error callback
                violations = [e for e in errors if "permissions violation" in e.lower()]
                for needle in (
                    f"$kv.{bucket}.{read_only}".lower(),
                    f"$kv.{bucket}.{other_table}".lower(),
                    f"$kv.{bucket}.{other_owners}".lower(),
                    f"$js.api.direct.get.kv_{bucket}.$kv.{bucket}.{other_table}".lower(),
                    f"$js.api.direct.get.kv_{bucket}.$kv.{bucket}.{other_owners}".lower(),
                ):
                    assert any(needle in e.lower() for e in violations), (needle, violations)

                # the granted table still works after the refusals
                assert (await kv.get(written)).value == b"w2"

            # nothing the pod was refused changed a value; what it was granted landed
            assert (await admin_kv.get(written)).value == b"w2"
            assert (await admin_kv.get(read_only)).value == b"r0"
            assert (await admin_kv.get(other_table)).value == b"c0"
            assert (await admin_kv.get(other_owners)).value == b"o0"
