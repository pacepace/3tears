"""Integration test: a pod reaches only its OWN keys in the platform's shared pod buckets, on a live broker.

``{ns}-ratelimits`` and ``{ns}-proxy_assertion_nonces`` are one bucket every agent pod binds. The grant
under test is :attr:`JsCapability.KV_OWNER_KEYS`, as the agent-pod resolver declares it and
:func:`mint_user_jwt` renders it: every route narrowed to the pod's own ``{scope}.>``. A unit test can
only say the strings look right; a JetStream request the grant does not cover is never answered -- it
blocks to its deadline and reads as an unreachable broker -- so each claim runs against a real
nats-server, through the nats-py calls the 3tears KV client makes.

- the pod's own keys: bind, direct read, put, create, compare-and-set update, delete, and a key listing
  through a NAMED consumer filtered inside its own prefix;
- another agent's keys, and an unscoped key: every read, write and listing refused;
- the body-carried ``STREAM.MSG.GET`` (which could name any key) refused;
- the retired shared checkpoint bucket: not even bindable;
- the epoch bucket: readable, never writable.

As in ``test_agent_bucket_grant_live``, the minted allow-lists are applied as config-mode
``authorization`` permissions: the grant strings are what is under test.

Gated on docker: a checkout without docker skips cleanly.
"""

from __future__ import annotations

import base64
import contextlib
import json
from collections.abc import Iterator
from pathlib import Path
from uuid import UUID

import pytest

from threetears.core.testing.containers import check_docker_available
from threetears.nats.subject_permissions import (
    Principal,
    PrincipalPermissions,
    build_permissions,
    kv_key_scope_for,
)
from threetears.nats.subjects import get_default_namespace, set_default_namespace
from threetears.nats.user_jwt import generate_account_seed, mint_user_jwt

from .refusals import Violations, connect, refused, refused_request

pytestmark = pytest.mark.integration

_NS = "sharedlive"
_AGENT = UUID("019470a8-b5c3-7def-8123-0000000000aa")
_OTHER_AGENT = UUID("019470a8-b5c3-7def-8123-0000000000bb")
_POD = "01947100-0000-7000-8000-0000000000aa"
_ADMIN_PW = "admin-pw"  # noqa: S105 - ephemeral testcontainer credential
_POD_PW = "pod-pw"  # noqa: S105 - ephemeral testcontainer credential


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
        name="shared-pod-bucket-live",
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
    from threetears.core.testing.fixtures import nats_server, nats_server_uri  # noqa: PLC0415

    (conf_dir / "nats.conf").write_text(config_text)
    container = (
        nats_server().with_volume_mapping(str(conf_dir), "/etc/nats", "ro").with_command(["-c", "/etc/nats/nats.conf"])
    )
    container.start()
    try:
        yield nats_server_uri(container)
    finally:
        container.stop()


async def test_an_agent_pod_reaches_only_its_own_keys_in_the_shared_pod_buckets(tmp_path: Path) -> None:
    """both halves against a real broker: its own keys work, every other owner's are refused."""
    if not check_docker_available():
        pytest.skip("Docker not available")

    previous_ns = get_default_namespace()
    set_default_namespace(_NS)
    try:
        permissions = build_permissions(Principal.AGENT_POD, agent_id=str(_AGENT), pod_id=_POD, conn_id=_POD)
        pub_allow, sub_allow = _minted_allow_lists(permissions)
    finally:
        set_default_namespace(previous_ns)

    mine = kv_key_scope_for(Principal.AGENT_POD, agent_id=_AGENT)
    theirs = kv_key_scope_for(Principal.AGENT_POD, agent_id=_OTHER_AGENT)
    nonces = f"{_NS}-proxy_assertion_nonces"
    ratelimits = f"{_NS}-ratelimits"
    checkpoints = f"{_NS}-checkpoints"
    epochs = f"{_NS}-epochs"

    with _nats_with_auth(_server_config(pub_allow, sub_allow), tmp_path) as uri:
        async with connect(uri, user="admin", password=_ADMIN_PW, violations=Violations()) as admin_nc:
            admin_js = admin_nc.jetstream()
            # the hub declares every shared pod bucket WITH allow_direct: the owner-keyed grant
            # carries only the subject-carried read.
            admin_nonces = await admin_js.create_key_value(bucket=nonces, direct=True)
            admin_ratelimits = await admin_js.create_key_value(bucket=ratelimits, direct=True)
            admin_checkpoints = await admin_js.create_key_value(bucket=checkpoints, direct=True)
            admin_epochs = await admin_js.create_key_value(bucket=epochs, direct=True)
            await admin_nonces.put(f"{theirs}.digest-1", b"theirs")
            await admin_nonces.put("digest-1", b"unscoped")
            await admin_ratelimits.put(f"{theirs}.memory.last_extract.c1", b"1")
            await admin_checkpoints.put("customer/thread-1", b"another agent's conversation")
            await admin_epochs.put("sharedlive.mcp.rbac.epoch", b"7")

            violations = Violations()
            async with connect(
                uri, user="pod", password=_POD_PW, inbox_prefix=permissions.inbox_prefix, violations=violations
            ) as nc:
                js = nc.jetstream(timeout=2)
                await js.account_info()

                # === SUCCEEDS: every KV operation on the pod's OWN keys ===========================
                kv = await js.key_value(nonces)
                own = f"{mine}.digest-1"
                assert await kv.create(own, b"1") == 3
                assert (await kv.get(own)).value == b"1"
                revision = await kv.put(own, b"2")
                await kv.update(own, b"3", last=revision)
                limits = await js.key_value(ratelimits)
                await limits.create(f"{mine}.memory.last_extract.c1", b"1")
                assert (await limits.get(f"{mine}.memory.last_extract.c1")).value == b"1"
                await limits.delete(f"{mine}.memory.last_extract.c1")

                # === SUCCEEDS: a key listing through a NAMED consumer inside its own prefix =======
                created = await nc.request(
                    f"$JS.API.CONSUMER.CREATE.KV_{nonces}.kl1.$KV.{nonces}.{mine}.>",
                    json.dumps(
                        {
                            "stream_name": f"KV_{nonces}",
                            "config": {
                                "name": "kl1",
                                "filter_subject": f"$KV.{nonces}.{mine}.>",
                                "ack_policy": "none",
                            },
                        }
                    ).encode(),
                    timeout=2,
                )
                assert "error" not in json.loads(created.data), created.data

                # === SUCCEEDS: a read of the epoch bucket ========================================
                epoch_kv = await js.key_value(epochs)
                assert (await epoch_kv.get("sharedlive.mcp.rbac.epoch")).value == b"7"
                assert not violations.errors, violations.errors

                # === REFUSED: another agent's key, and an unscoped key, on every route ===========
                for key in (f"{theirs}.digest-1", "digest-1"):
                    named = f"$kv.{nonces}.{key}"
                    await refused(violations, kv.get(key), naming=named)
                    await refused(violations, kv.put(key, b"overwritten"), naming=named)
                    await refused(violations, kv.delete(key), naming=named)
                await refused(
                    violations,
                    limits.delete(f"{theirs}.memory.last_extract.c1"),
                    naming=f"$kv.{ratelimits}.{theirs}.memory.last_extract.c1",
                )

                # === REFUSED: the body-carried read, and a listing of the whole bucket ===========
                await refused(
                    violations,
                    js.get_msg(f"KV_{nonces}", subject=f"$KV.{nonces}.{theirs}.digest-1"),
                    naming=f"$js.api.stream.msg.get.kv_{nonces}",
                )
                for request_subject, body in (
                    (
                        f"$JS.API.CONSUMER.CREATE.KV_{nonces}.kl2.$KV.{nonces}.>",
                        {"stream_name": f"KV_{nonces}", "config": {"name": "kl2", "filter_subject": f"$KV.{nonces}.>"}},
                    ),
                    (
                        f"$JS.API.CONSUMER.CREATE.KV_{nonces}.kl3.$KV.{nonces}.{theirs}.>",
                        {
                            "stream_name": f"KV_{nonces}",
                            "config": {"name": "kl3", "filter_subject": f"$KV.{nonces}.{theirs}.>"},
                        },
                    ),
                    (f"$JS.API.CONSUMER.CREATE.KV_{nonces}", {"stream_name": f"KV_{nonces}", "config": {}}),
                    (f"$JS.API.STREAM.PURGE.KV_{nonces}", {}),
                ):
                    await refused_request(violations, nc, request_subject, json.dumps(body).encode())

                # === REFUSED: the retired shared checkpoint bucket, at the bind ===================
                await refused(violations, js.key_value(checkpoints), naming=f"$js.api.stream.info.kv_{checkpoints}")
                await refused(
                    violations,
                    js.get_msg(f"KV_{checkpoints}", subject=f"$KV.{checkpoints}.customer/thread-1"),
                    naming=f"$js.api.stream.msg.get.kv_{checkpoints}",
                )

                # === REFUSED: any write to the epoch bucket ======================================
                await refused(
                    violations,
                    epoch_kv.put("sharedlive.mcp.rbac.epoch", b"8"),
                    naming=f"$kv.{epochs}.sharedlive.mcp.rbac.epoch",
                )

                # the pod's own keys still work after the refusals
                assert (await kv.get(own)).value == b"3"

            # nothing the pod was refused changed a value
            assert (await admin_nonces.get(f"{theirs}.digest-1")).value == b"theirs"
            assert (await admin_nonces.get("digest-1")).value == b"unscoped"
            assert (await admin_ratelimits.get(f"{theirs}.memory.last_extract.c1")).value == b"1"
            assert (await admin_epochs.get("sharedlive.mcp.rbac.epoch")).value == b"7"
            assert (await admin_nonces.get(f"{mine}.digest-1")).value == b"3"
