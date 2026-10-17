"""Integration test: a pod reads and watches its OWN data-version key, and nothing else, on a live broker.

The grant under test is :attr:`JsCapability.KV_KEY_READ` on ``{ns}-data-versions``, as the agent-pod
resolver declares it and :func:`mint_user_jwt` renders it. A unit test can only say the strings look
right; a JetStream request the grant does not cover is never answered -- it blocks to its deadline and
reads as an unreachable broker -- so each claim is run against a real nats-server:

- the pod binds the bucket, direct-reads its own key, and WATCHES it through a named consumer whose
  filter rides in the create subject, receiving both the current value and a later write;
- nats-py's stock ``KeyValue.watch`` (an unnamed consumer, filter in the body only) is refused, which
  is the constraint a watcher built on this grant has to honour;
- another principal's key is refused on every route: the direct read, a named watch, an unnamed
  whole-bucket consumer, the body-carried ``STREAM.MSG.GET``;
- a named consumer whose SUBJECT names the pod's own key but whose BODY filters the whole bucket is
  refused by the server itself;
- the pod cannot write, even its own key.

As in ``test_user_jwt_scoped_grant_live``, the minted allow-lists are applied as config-mode
``authorization`` permissions rather than through a live auth-callout responder: the grant strings are
what is under test, and this is the credential the server would enforce.

Gated on docker: a checkout without docker skips cleanly.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
from collections.abc import Iterator
from pathlib import Path
from uuid import UUID

import nats.aio.msg
import nats.js.api
import nats.js.errors
import pytest

from threetears.core.testing.containers import check_docker_available
from threetears.nats.subject_permissions import (
    Principal,
    PrincipalPermissions,
    build_permissions,
    data_version_kv_key,
    data_versions_bucket_name,
)
from threetears.nats.subjects import get_default_namespace, set_default_namespace
from threetears.nats.user_jwt import generate_account_seed, mint_user_jwt

from .refusals import Violations, connect, refused, refused_request

pytestmark = pytest.mark.integration

_NS = "dvlive"
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
        name="data-version-live",
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


async def test_a_pod_reads_and_watches_only_its_own_data_version_key(tmp_path: Path) -> None:
    """both halves against a real broker: the own key works, every other route is refused."""
    if not check_docker_available():
        pytest.skip("Docker not available")

    previous_ns = get_default_namespace()
    set_default_namespace(_NS)
    try:
        permissions = build_permissions(Principal.AGENT_POD, agent_id=str(_AGENT), pod_id=_POD, conn_id=_POD)
        pub_allow, sub_allow = _minted_allow_lists(permissions)
    finally:
        set_default_namespace(previous_ns)

    bucket = data_versions_bucket_name(_NS)
    stream = f"KV_{bucket}"
    own = data_version_kv_key(_AGENT)
    other = data_version_kv_key(_OTHER_AGENT)

    with _nats_with_auth(_server_config(pub_allow, sub_allow), tmp_path) as uri:
        async with connect(uri, user="admin", password=_ADMIN_PW, violations=Violations()) as admin_nc:
            admin_js = admin_nc.jetstream()
            # the hub creates the bucket with allow_direct; without it a read is the body-carried
            # STREAM.MSG.GET, which no subject grant can narrow to one key.
            admin_kv = await admin_js.create_key_value(bucket=bucket, direct=True)
            await admin_kv.put(own, b"3")
            await admin_kv.put(other, b"9")

            violations = Violations()
            async with connect(
                uri, user="pod", password=_POD_PW, inbox_prefix=permissions.inbox_prefix, violations=violations
            ) as nc:
                js = nc.jetstream(timeout=3)
                await js.account_info()

                # === SUCCEEDS: bind, read, and a NAMED watch of the pod's own key ===========
                kv = await js.key_value(bucket)
                assert (await kv.get(own)).value == b"3"

                received: asyncio.Queue[bytes] = asyncio.Queue()

                async def on_message(msg: nats.aio.msg.Msg) -> None:
                    await received.put(msg.data)

                watch = await js.subscribe(
                    f"$KV.{bucket}.{own}",
                    stream=stream,
                    cb=on_message,
                    ordered_consumer=True,
                    deliver_policy=nats.js.api.DeliverPolicy.LAST_PER_SUBJECT,
                    config=nats.js.api.ConsumerConfig(name="dvwatch1"),
                )
                assert await asyncio.wait_for(received.get(), timeout=5) == b"3"
                await admin_kv.put(own, b"4")
                assert await asyncio.wait_for(received.get(), timeout=5) == b"4"
                await watch.unsubscribe()
                assert not violations.errors, violations.errors

                # === REFUSED: the stock watch, whose consumer is unnamed ====================
                await refused(violations, kv.watch(own), naming=f"$js.api.consumer.create.{stream}")

                # === REFUSED: every route to another principal's key, and every write ======
                await refused_request(violations, nc, f"$JS.API.DIRECT.GET.{stream}.$KV.{bucket}.{other}")
                await refused_request(
                    violations,
                    nc,
                    f"$JS.API.CONSUMER.CREATE.{stream}.spy.$KV.{bucket}.{other}",
                    body=json.dumps(
                        {
                            "stream_name": stream,
                            "config": {
                                "name": "spy",
                                "filter_subject": f"$KV.{bucket}.{other}",
                                "deliver_subject": f"{permissions.inbox_prefix}.spy",
                            },
                        }
                    ).encode(),
                )
                await refused_request(
                    violations,
                    nc,
                    f"$JS.API.CONSUMER.CREATE.{stream}",
                    body=json.dumps(
                        {
                            "stream_name": stream,
                            "config": {
                                "filter_subject": f"$KV.{bucket}.>",
                                "deliver_subject": f"{permissions.inbox_prefix}.spy",
                            },
                        }
                    ).encode(),
                )
                await refused_request(
                    violations,
                    nc,
                    f"$JS.API.STREAM.MSG.GET.{stream}",
                    body=json.dumps({"last_by_subj": f"$KV.{bucket}.{other}"}).encode(),
                )
                await refused_request(violations, nc, f"$KV.{bucket}.{own}", body=b"99")

                # === REFUSED BY THE SERVER: own key in the subject, whole bucket in the body ==
                reply = await nc.request(
                    f"$JS.API.CONSUMER.CREATE.{stream}.spy2.$KV.{bucket}.{own}",
                    json.dumps(
                        {
                            "stream_name": stream,
                            "config": {
                                "name": "spy2",
                                "filter_subject": f"$KV.{bucket}.>",
                                "deliver_subject": f"{permissions.inbox_prefix}.spy2",
                            },
                        }
                    ).encode(),
                    timeout=3,
                )
                assert "error" in json.loads(reply.data), reply.data

                # the pod's own read still works after being refused
                assert (await kv.get(own)).value == b"4"

            # nothing the pod attempted changed either value
            assert (await admin_kv.get(own)).value == b"4"
            assert (await admin_kv.get(other)).value == b"9"
