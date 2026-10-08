"""Integration test: a tool pod works inside the Object Store the hub declared for it, and manages nothing.

The pod's grant is minted by :func:`mint_user_jwt` from :func:`build_permissions` exactly as the auth
callout mints it, and applied as config-mode ``authorization`` permissions. A JetStream call the grant
does not cover is never answered, so each claim runs against a real nats-server:

- through the wrapper, the pod binds its own Object Store and pointer bucket, puts an object larger
  than a chunk, reads it back, describes and lists it, and watches its pointers by prefix;
- nats-py's own ``ObjectStore.get`` -- an unnamed ordered consumer -- is refused;
- deleting an object, purging, deleting or reshaping the bucket, and every route into another pod's
  bucket are refused, and the bucket is unchanged.

Gated on docker: a checkout without docker skips cleanly.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
from collections.abc import Iterator
from contextlib import aclosing
from datetime import timedelta
from pathlib import Path

import nats.errors
import pytest

from threetears.core.testing.containers import check_docker_available
from threetears.nats import NatsClient, ObjectStoreError
from threetears.nats.subject_permissions import (
    Principal,
    PrincipalPermissions,
    build_permissions,
    tool_pod_object_store_name,
    tool_pod_pointers_bucket_name,
)
from threetears.nats.subjects import get_default_namespace, set_default_namespace
from threetears.nats.user_jwt import generate_account_seed, mint_user_jwt

pytestmark = pytest.mark.integration

_NS = "objgrant"
_POD = "01947100-0000-7000-8000-0000000000f1"
_VICTIM = "01947100-0000-7000-8000-0000000000f2"
_ADMIN_PW = "admin-pw"  # noqa: S105 - ephemeral testcontainer credential
_POD_PW = "pod-pw"  # noqa: S105 - ephemeral testcontainer credential
_WAIT = 10.0

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
        NatsContainer(jetstream=False)
        .with_volume_mapping(str(conf_dir), "/etc/nats", "ro")
        .with_command(["-c", "/etc/nats/nats.conf"])
    )
    container.start()
    try:
        yield container.nats_uri()
    finally:
        container.stop()


async def test_a_tool_pod_works_inside_its_object_store_and_manages_nothing(tmp_path: Path) -> None:
    if not check_docker_available():
        pytest.skip("Docker not available")

    previous_ns = get_default_namespace()
    set_default_namespace(_NS)
    try:
        permissions = build_permissions(Principal.TOOL_POD, pod_id=_POD, conn_id=_POD, object_store=True)
        pub_allow, sub_allow = _minted_allow_lists(permissions)
        own = tool_pod_object_store_name(_POD, ns=_NS)
        pointers = tool_pod_pointers_bucket_name(_POD, ns=_NS)
        victim = tool_pod_object_store_name(_VICTIM, ns=_NS)
    finally:
        set_default_namespace(previous_ns)

    with _nats_with_auth(_server_config(pub_allow, sub_allow), tmp_path) as uri:
        hub = await NatsClient.connect(
            nats_url=uri,
            nats_subject_namespace=_NS,
            client_name="hub-declares",
            user="admin",
            password=_ADMIN_PW,
            startup_timeout=timedelta(seconds=10),
        )
        pod: NatsClient | None = None
        try:
            # === the HUB declares the pod's buckets, and another pod's ==============================
            await hub.ensure_object_store(name=own, max_bytes=16 * 1024 * 1024, prefix_namespace=False)
            await hub.ensure_kv_bucket(name=pointers, direct=True, owns_bucket=True, prefix_namespace=False)
            victim_store = await hub.ensure_object_store(name=victim, max_bytes=1024 * 1024, prefix_namespace=False)
            await victim_store.put("secret", b"another pod's rows")

            pod = await NatsClient.connect(
                nats_url=uri,
                nats_subject_namespace=_NS,
                client_name="pod-binds",
                user="pod",
                password=_POD_PW,
                inbox_prefix=permissions.inbox_prefix,
                startup_timeout=timedelta(seconds=10),
            )

            # === SUCCEEDS: every Object Store operation a pod has, through the wrapper ==============
            store = await pod.object_store(name=own, prefix_namespace=False)
            data = os.urandom(1_500_000)
            written = await store.put("candidates/TX/7", data)
            assert written.chunks > 1
            assert await store.get("candidates/TX/7") == data
            described = await store.info("candidates/TX/7")
            assert described is not None and described.size == len(data)
            assert [i.name for i in await store.list_objects(prefix="candidates/")] == ["candidates/TX/7"]
            assert await store.bytes_held() >= len(data)

            # === SUCCEEDS: the pointer bucket, watched by prefix ====================================
            bucket = await pod.kv_bucket(name=pointers.removeprefix(f"{_NS}-"), create_if_missing=False)
            assert bucket.name == pointers
            await bucket.put(key="snap.TX", value=b"7")
            async with aclosing(bucket.watch_prefix(prefix="snap.")) as watch:
                first = await asyncio.wait_for(anext(watch), timeout=_WAIT)
                assert first is not None and (first.key, first.value) == ("snap.TX", b"7")
                assert await asyncio.wait_for(anext(watch), timeout=_WAIT) is None

            # === REFUSED: nats-py's own read, an unnamed ordered consumer =========================
            stock = await pod.jetstream_context().object_store(own)
            # an unanswered create never raises on its own; the read simply never arrives
            with pytest.raises((*_REFUSED, TimeoutError)):
                await asyncio.wait_for(stock.get("candidates/TX/7"), timeout=5)

            # === REFUSED: deleting an object is a purge, the hub's ===================================
            with pytest.raises(ObjectStoreError):
                await store.delete("candidates/TX/7")

            # === REFUSED: every management verb, and every route into another pod's bucket ==========
            raw = pod.raw
            meta = base64.urlsafe_b64encode(b"secret").decode()
            for subject, body in (
                (f"$JS.API.STREAM.PURGE.OBJ_{own}", {}),
                (f"$JS.API.STREAM.DELETE.OBJ_{own}", {}),
                (f"$JS.API.STREAM.UPDATE.OBJ_{own}", {"name": f"OBJ_{own}", "sources": [{"name": f"OBJ_{victim}"}]}),
                (f"$JS.API.STREAM.CREATE.OBJ_{_NS}-exfil", {"name": f"OBJ_{_NS}-exfil", "subjects": []}),
                (f"$JS.API.STREAM.MSG.GET.OBJ_{own}", {"last_by_subj": f"$O.{own}.M.x"}),
                (f"$JS.API.STREAM.MSG.DELETE.OBJ_{own}", {"seq": 1}),
                (f"$JS.API.STREAM.INFO.OBJ_{victim}", {}),
                (f"$JS.API.DIRECT.GET.OBJ_{victim}.$O.{victim}.M.{meta}", None),
                (f"$JS.API.CONSUMER.CREATE.OBJ_{victim}.c1.$O.{victim}.>", {"stream_name": f"OBJ_{victim}"}),
            ):
                payload = b"" if body is None else json.dumps(body).encode()
                with pytest.raises(_REFUSED):
                    await raw.request(subject, payload, timeout=2)
            with pytest.raises(ObjectStoreError):
                await pod.object_store(name=victim, prefix_namespace=False)

            # === nothing changed ======================================================================
            admin_js = hub.raw.jetstream()
            assert not (await admin_js.stream_info(f"OBJ_{own}")).config.sources
            assert (await hub.object_store(name=own, prefix_namespace=False)) is not None
            assert await (await hub.object_store(name=own, prefix_namespace=False)).get("candidates/TX/7") == data
        finally:
            if pod is not None:
                await pod.shutdown(drain_timeout=timedelta(seconds=2))
            await hub.shutdown(drain_timeout=timedelta(seconds=2))
