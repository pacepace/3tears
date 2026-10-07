"""PINNED RESIDUAL: a pod can put bytes of its choosing on ANY subject in its account. A known platform hole.

This test asserts the hole is THERE. It is not a contract the platform wants; it pins a property of
NATS that no grant can close, so the day it stops holding (a nats-server that checks a consumer's
deliver subject or a request's reply subject against the requester's permissions, or the platform
moving principals into separate accounts) this test fails and says to remove it and the residual's
note in ``docs/design-durable-coordination.md``.

**What holds today** (nats-server 2.12.6, the local stack's version; a real minted tool-pod JWT
applied as config permissions, as the sibling grant tests do):

- a NAMED push consumer the grant admits on the pod's OWN bucket (the filter rides in the create
  subject, where the server checks it) carries ``deliver_subject`` in the request BODY, which no
  subject permission sees. The server delivers the pod's own stored bytes to whatever subject the
  body names -- another pod's ``tools.internal.<id>``, another principal's inbox -- although the
  pod may not publish there;
- every request the grant admits (a direct get, a pull ``MSG.NEXT``, any JetStream API call) is
  answered on the request's REPLY subject, which NATS never checks against the requester's
  permissions; a direct get of a value the pod wrote returns that value verbatim.

So reading through pull consumers or direct gets closes nothing, and the read shape is not the
fix. Any pod that can write a value it can read back has this, through every L2 grant: its own KV
and Object Store buckets, its owner keys in the shared buckets, its scope of the collections bucket.
The defence is the RECEIVER's: a subject whose receiver trusts the subject alone (no signed token,
no correlation it minted) can be fed a pod's bytes. Pace, 2026-10-07: receivers are being audited;
their fixes are their own piece of work.

Gated on docker: a checkout without docker skips cleanly.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from threetears.core.testing.containers import check_docker_available
from threetears.nats import NatsClient
from threetears.nats.subject_permissions import (
    Principal,
    PrincipalPermissions,
    build_permissions,
    tool_pod_object_store_name,
)
from threetears.nats.subjects import get_default_namespace, set_default_namespace
from threetears.nats.user_jwt import generate_account_seed, mint_user_jwt

pytestmark = pytest.mark.integration

_NS = "bounce"
_POD = "01947100-0000-7000-8000-0000000000e7"
_OTHER_POD = "01947100-0000-7000-8000-0000000000e8"
_ADMIN_PW = "admin-pw"  # noqa: S105 - ephemeral testcontainer credential
_POD_PW = "pod-pw"  # noqa: S105 - ephemeral testcontainer credential
#: the local stack's nats-server, so the residual is pinned on the version that runs
_NATS_IMAGE = "nats:2.12.6"
_PAYLOAD = b'{"forged": "a pod chose these bytes"}'


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
        name="no-stream-management-live",
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
        NatsContainer(image=_NATS_IMAGE, jetstream=False)
        .with_volume_mapping(str(conf_dir), "/etc/nats", "ro")
        .with_command(["-c", "/etc/nats/nats.conf"])
    )
    container.start()
    try:
        yield container.nats_uri()
    finally:
        container.stop()


async def test_a_pod_can_bounce_its_own_bytes_to_a_subject_it_may_not_publish(tmp_path: Path) -> None:
    if not check_docker_available():
        pytest.skip("Docker not available")
    previous_ns = get_default_namespace()
    set_default_namespace(_NS)
    try:
        permissions = build_permissions(Principal.TOOL_POD, pod_id=_POD, conn_id=_POD, object_store=True)
        pub_allow, sub_allow = _minted_allow_lists(permissions)
        own = tool_pod_object_store_name(_POD, ns=_NS)
    finally:
        set_default_namespace(previous_ns)
    other_pods_calls = f"{_NS}.tools.internal.{_OTHER_POD}"
    other_inbox = "_INBOX_tool_pod_victim.x"
    reply_target = f"{_NS}.tools.internal.{_OTHER_POD}.reply"
    for target in (other_pods_calls, other_inbox, reply_target):
        assert target not in pub_allow, f"the pod may publish {target}; the probe would prove nothing"

    with _nats_with_auth(_server_config(pub_allow, sub_allow), tmp_path) as uri:
        hub = await NatsClient.connect(
            nats_url=uri, nats_subject_namespace=_NS, client_name="hub", user="admin", password=_ADMIN_PW
        )
        pod: NatsClient | None = None
        try:
            await hub.ensure_object_store(name=own, max_bytes=1024 * 1024, prefix_namespace=False)
            received: dict[str, list[bytes]] = {other_pods_calls: [], other_inbox: [], reply_target: []}
            for subject in received:

                async def note(msg: Any, subject: str = subject) -> None:
                    received[subject].append(bytes(msg.data))

                await hub.raw.subscribe(subject, cb=note)
            pod = await NatsClient.connect(
                nats_url=uri,
                nats_subject_namespace=_NS,
                client_name="pod",
                user="pod",
                password=_POD_PW,
                inbox_prefix=permissions.inbox_prefix,
            )
            store = await pod.object_store(name=own, prefix_namespace=False)
            await store.put("payload", _PAYLOAD)
            for name, deliver in (("c1", other_pods_calls), ("c2", other_inbox)):
                body = {
                    "stream_name": f"OBJ_{own}",
                    "config": {
                        "name": name,
                        "deliver_subject": deliver,
                        "filter_subject": f"$O.{own}.C.>",
                        "ack_policy": "none",
                        "deliver_policy": "all",
                    },
                }
                created = await pod.raw.request(
                    f"$JS.API.CONSUMER.CREATE.OBJ_{own}.{name}.$O.{own}.C.>", json.dumps(body).encode(), timeout=5
                )
                assert b"consumer_create_response" in created.data and b'"error"' not in created.data
            meta = base64.urlsafe_b64encode(b"payload").decode()
            await pod.raw.publish(f"$JS.API.DIRECT.GET.OBJ_{own}.$O.{own}.M.{meta}", b"", reply=reply_target)
            deadline = asyncio.get_running_loop().time() + 10
            while not all(received.values()) and asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(0.1)
            # THE RESIDUAL. If any of these fails, NATS or the platform closed the hole: delete this
            # test and the residual's note in docs/design-durable-coordination.md.
            assert received[other_pods_calls] == [_PAYLOAD], "a push consumer no longer delivers off-grant"
            assert received[other_inbox] == [_PAYLOAD], "a push consumer no longer delivers to another inbox"
            assert received[reply_target], "a reply no longer reaches a subject the requester may not publish"
        finally:
            if pod is not None:
                await pod.shutdown(drain_timeout=timedelta(seconds=1))
            await hub.shutdown(drain_timeout=timedelta(seconds=1))
