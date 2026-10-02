"""Integration test: a tool pod granted an agent's coordination buckets reaches exactly those, on a live broker.

The grant under test is :attr:`JsCapability.KV_BUCKET_KEYS`, as the tool-pod resolver declares it from
:class:`AgentBucketGrant` and :func:`mint_user_jwt` renders it, plus the pod's own audit subtree
(``{ns}.audit.tool_pod.{pod}.>``). A unit test can only say the strings look right; a JetStream request
the grant does not cover is never answered -- it blocks to its deadline and reads as an unreachable
broker -- so each claim runs against a real nats-server, through the nats-py calls the 3tears KV client
(:class:`threetears.nats.kv.NatsKvBucket`) makes: bind, get, put, compare-and-set update, create, delete.

- a WRITE grant on a bucket created WITHOUT ``allow_direct`` (how every agent opens its coordination
  buckets): bind, the body-carried read, and every write;
- a READ grant on a bucket created WITH ``allow_direct``: bind and the subject-carried read, and every
  write refused;
- another owner's bucket of the same suffix, and an undeclared suffix of the same owner, refused at
  the bind;
- a key listing through a NAMED consumer filtered inside the granted bucket, which the survey's
  erasure sweep needs; no unnamed consumer, no purge, no stream create or update (``sources`` would
  copy any stream into one the pod reads), even on a granted bucket;
- the pod's own audit subject accepted by the audit stream; another pod's, and a platform audit
  subject, refused.

As in ``test_agent_table_grant_live``, the minted allow-lists are applied as config-mode
``authorization`` permissions: the grant strings are what is under test, and this is the credential the
server would enforce.

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
    AgentBucketGrant,
    Principal,
    PrincipalPermissions,
    build_permissions,
    coordination_bucket_name,
    kv_key_scope_for,
)
from threetears.nats.subjects import Subjects, get_default_namespace, set_default_namespace
from threetears.nats.user_jwt import generate_account_seed, mint_user_jwt

pytestmark = pytest.mark.integration

_NS = "abglive"
_OWNER = UUID("019470a8-b5c3-7def-8123-0000000000aa")
_OTHER_OWNER = UUID("019470a8-b5c3-7def-8123-0000000000bb")
_POD = "01947100-0000-7000-8000-0000000000aa"
_OTHER_POD = "01947100-0000-7000-8000-0000000000bb"
_WRITE_SUFFIX = "survey-quota-cells"
_READ_SUFFIX = "respondent_resume_handles"
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
        name="agent-bucket-live",
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


async def test_a_tool_pod_reaches_exactly_the_agent_buckets_it_was_granted(tmp_path: Path) -> None:
    """both halves against a real broker: the granted buckets work, every other route is refused."""
    if not check_docker_available():
        pytest.skip("Docker not available")

    previous_ns = get_default_namespace()
    set_default_namespace(_NS)
    try:
        permissions = build_permissions(
            Principal.TOOL_POD,
            pod_id=_POD,
            conn_id=_POD,
            agent_bucket_grants=(
                AgentBucketGrant(owner_agent_id=_OWNER, suffix=_WRITE_SUFFIX, writable=True),
                AgentBucketGrant(owner_agent_id=_OWNER, suffix=_READ_SUFFIX, writable=False),
            ),
        )
        pub_allow, sub_allow = _minted_allow_lists(permissions)
        own_audit = str(Subjects.tool_pod_audit_event(_POD, "collector.promoted"))
        other_pod_audit = str(Subjects.tool_pod_audit_event(_OTHER_POD, "collector.promoted"))
        platform_audit = str(Subjects.audit_event("identity.principal.merge"))
        audit_wildcard = str(Subjects.audit_wildcard())
    finally:
        set_default_namespace(previous_ns)

    owner_scope = kv_key_scope_for(Principal.AGENT_POD, agent_id=_OWNER)
    written = coordination_bucket_name(owner_scope, _WRITE_SUFFIX, ns=_NS)
    read_only = coordination_bucket_name(owner_scope, _READ_SUFFIX, ns=_NS)
    undeclared = coordination_bucket_name(owner_scope, "panel_reset_tickets", ns=_NS)
    other_owners = coordination_bucket_name(
        kv_key_scope_for(Principal.AGENT_POD, agent_id=_OTHER_OWNER), _WRITE_SUFFIX, ns=_NS
    )

    with _nats_with_auth(_server_config(pub_allow, sub_allow), tmp_path) as uri:
        admin_errors: list[str] = []
        async with _connect(
            uri, user="admin", password=_ADMIN_PW, inbox_prefix="_INBOX", errors=admin_errors
        ) as admin_nc:
            admin_js = admin_nc.jetstream()
            # the owning agent opens its coordination buckets through ``kv_bucket`` with no direct
            # flag, so they run WITHOUT allow_direct; the read-granted one is created WITH it so the
            # other read form is exercised too.
            admin_written = await admin_js.create_key_value(bucket=written)
            admin_read = await admin_js.create_key_value(bucket=read_only, direct=True)
            admin_undeclared = await admin_js.create_key_value(bucket=undeclared)
            admin_other = await admin_js.create_key_value(bucket=other_owners)
            await admin_written.put("cell-1", b"1")
            await admin_read.put("handle-1", b"h0")
            await admin_undeclared.put("ticket-1", b"t0")
            await admin_other.put("cell-1", b"9")
            await admin_js.add_stream(name=f"{_NS}-audit", subjects=[audit_wildcard])

            errors: list[str] = []
            async with _connect(
                uri, user="pod", password=_POD_PW, inbox_prefix=permissions.inbox_prefix, errors=errors
            ) as nc:
                js = nc.jetstream(timeout=2)
                await js.account_info()

                # === SUCCEEDS: every KV operation on the WRITE-granted bucket (no allow_direct) ====
                kv = await js.key_value(written)
                assert (await kv.get("cell-1")).value == b"1"
                revision = await kv.put("cell-1", b"2")
                revision = await kv.update("cell-1", b"3", last=revision)
                with pytest.raises(nats.js.errors.KeyWrongLastSequenceError):
                    await kv.update("cell-1", b"stale", last=revision - 1)
                await kv.create("cell-2", b"1")
                await kv.delete("cell-2")
                with pytest.raises(nats.js.errors.KeyNotFoundError):
                    await kv.get("cell-2")

                # === SUCCEEDS: a READ of the read-granted bucket (allow_direct) ====================
                read_kv = await js.key_value(read_only)
                assert (await read_kv.get("handle-1")).value == b"h0"

                # === SUCCEEDS: the pod's own audit subject, persisted by the audit stream =========
                ack = await js.publish(own_audit, b"{}")
                assert ack.stream == f"{_NS}-audit"
                assert not [e for e in errors if "permissions violation" in e.lower()], errors

                # === REFUSED: every write to the read-granted bucket =============================
                with pytest.raises(_REFUSED):
                    await read_kv.put("handle-1", b"h1")
                with pytest.raises(_REFUSED):
                    await read_kv.delete("handle-1")

                # === REFUSED: another owner's bucket, and an undeclared suffix of this owner ======
                for bucket in (other_owners, undeclared):
                    with pytest.raises(_REFUSED):
                        await js.key_value(bucket)
                    with pytest.raises(_REFUSED):
                        await js.get_msg(f"KV_{bucket}", subject=f"$KV.{bucket}.cell-1")

                # === SUCCEEDS: a named consumer filtered inside the granted bucket ===============
                created = await nc.request(
                    f"$JS.API.CONSUMER.CREATE.KV_{written}.w1.$KV.{written}.cell-1",
                    json.dumps(
                        {
                            "stream_name": f"KV_{written}",
                            "config": {"name": "w1", "filter_subject": f"$KV.{written}.cell-1", "ack_policy": "none"},
                        }
                    ).encode(),
                    timeout=2,
                )
                assert "error" not in json.loads(created.data), created.data

                # === REFUSED: an unnamed consumer, a purge and a stream create or update ==========
                for request_subject, body in (
                    (f"$JS.API.CONSUMER.CREATE.KV_{written}", {"stream_name": f"KV_{written}", "config": {}}),
                    (
                        f"$JS.API.CONSUMER.CREATE.KV_{written}.w2.$KV.{other_owners}.cell-1",
                        {
                            "stream_name": f"KV_{written}",
                            "config": {"name": "w2", "filter_subject": f"$KV.{other_owners}.cell-1"},
                        },
                    ),
                    (f"$JS.API.STREAM.PURGE.KV_{written}", {}),
                    (f"$JS.API.STREAM.UPDATE.KV_{written}", {"name": f"KV_{written}"}),
                    (
                        f"$JS.API.STREAM.CREATE.KV_{written}",
                        {"name": f"KV_{written}", "sources": [{"name": f"KV_{other_owners}"}]},
                    ),
                ):
                    with pytest.raises(_REFUSED):
                        await nc.request(request_subject, json.dumps(body).encode(), timeout=2)

                # === REFUSED: another pod's audit subject, and a platform audit subject ===========
                for subject in (other_pod_audit, platform_audit):
                    with pytest.raises(_REFUSED):
                        await js.publish(subject, b"{}", timeout=2)

                await asyncio.sleep(0.4)  # let the async -ERR frames land in the error callback
                violations = [e for e in errors if "permissions violation" in e.lower()]
                for needle in (
                    f"$kv.{read_only}.handle-1".lower(),
                    f"$js.api.stream.info.kv_{other_owners}".lower(),
                    f"$js.api.stream.info.kv_{undeclared}".lower(),
                    f"$js.api.stream.purge.kv_{written}".lower(),
                    other_pod_audit.lower(),
                    platform_audit.lower(),
                ):
                    assert any(needle in e.lower() for e in violations), (needle, violations)

                # the granted bucket still works after the refusals
                assert (await kv.get("cell-1")).value == b"3"

            # nothing the pod was refused changed a value; what it was granted landed
            assert (await admin_written.get("cell-1")).value == b"3"
            assert (await admin_read.get("handle-1")).value == b"h0"
            assert (await admin_undeclared.get("ticket-1")).value == b"t0"
            assert (await admin_other.get("cell-1")).value == b"9"
            info = await admin_js.stream_info(f"{_NS}-audit")
            assert info.state.messages == 1
