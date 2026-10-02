"""Integration test: ``NatsKvBucket.watch_key`` works under a one-key grant, where the stock watch cannot.

The principal is an agent pod holding :attr:`JsCapability.KV_KEY_READ` on ``{ns}-data-versions``,
minted by :func:`mint_user_jwt` exactly as ``test_data_version_key_grant_live`` mints it: bind, a
DIRECT.GET of its own key, and ``CONSUMER.CREATE.{stream}.*.$KV.{bucket}.{key}``. A JetStream call
that grant does not cover is never answered -- it blocks to its deadline -- so each claim is proven
against a real nats-server:

- the pod, through the wrapper client, binds the bucket and watches its own key: it receives the
  current value, then each later write and the delete marker, and never another key's write;
- nats-py's stock ``KeyValue.watch`` on the same connection is refused. That refusal is why
  ``watch_key`` exists;
- when the server loses the watch's consumer, the watch notices the missing heartbeats, creates a
  replacement under a fresh name, and delivers the next write -- with no permissions violation;
- closing the watch drops the consumer's subscription, so nothing is left bound to it.

As in ``test_user_jwt_scoped_grant_live``, the minted allow-lists are applied as config-mode
``authorization`` permissions rather than through a live auth-callout responder: the grant strings
are what is under test, and this is the credential the server would enforce.

Gated on docker: a checkout without docker skips cleanly.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
from collections.abc import Iterator
from contextlib import aclosing
from datetime import timedelta
from pathlib import Path
from uuid import UUID

import nats
import nats.errors
import pytest

from threetears.core.testing.containers import check_docker_available
from threetears.nats import NatsClient
from threetears.nats.kv_watch import KvKeyUpdate
from threetears.nats.subject_permissions import (
    DATA_VERSIONS_BUCKET_SUFFIX,
    Principal,
    PrincipalPermissions,
    build_permissions,
    data_version_kv_key,
    data_versions_bucket_name,
)
from threetears.nats.subjects import get_default_namespace, set_default_namespace
from threetears.nats.user_jwt import generate_account_seed, mint_user_jwt

pytestmark = pytest.mark.integration

_NS = "kwlive"
_AGENT = UUID("019470a8-b5c3-7def-8123-0000000000cc")
_OTHER_AGENT = UUID("019470a8-b5c3-7def-8123-0000000000dd")
_POD = "01947100-0000-7000-8000-0000000000cc"
_ADMIN_PW = "admin-pw"  # noqa: S105 - ephemeral testcontainer credential
_POD_PW = "pod-pw"  # noqa: S105 - ephemeral testcontainer credential
_HEARTBEAT = timedelta(seconds=1)
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
        name="kv-watch-key-live",
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


async def _watch_consumers(admin_js: object, stream: str) -> list[str]:
    """the names of the key-watch consumers live on ``stream``, as the admin sees them.

    :param admin_js: the admin's JetStream context
    :ptype admin_js: object
    :param stream: the stream
    :ptype stream: str
    :return: consumer names
    :rtype: list[str]
    """
    infos = await admin_js.consumers_info(stream)  # type: ignore[attr-defined]
    return [info.name for info in infos if info.name.startswith("kw_")]


async def test_watch_key_works_under_the_one_key_grant_where_the_stock_watch_is_refused(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """current value, updates, delete marker, consumer replacement and close, under the real grant."""
    if not check_docker_available():
        pytest.skip("Docker not available")

    previous_ns = get_default_namespace()
    set_default_namespace(_NS)
    try:
        permissions = build_permissions(Principal.AGENT_POD, agent_id=str(_AGENT), pod_id=_POD, conn_id=_POD)
        pub_allow, sub_allow = _minted_allow_lists(permissions)
    finally:
        set_default_namespace(previous_ns)

    bucket_name = data_versions_bucket_name(_NS)
    stream = f"KV_{bucket_name}"
    own = data_version_kv_key(_AGENT)
    other = data_version_kv_key(_OTHER_AGENT)

    with _nats_with_auth(_server_config(pub_allow, sub_allow), tmp_path) as uri:
        admin_nc = await nats.connect(uri, user="admin", password=_ADMIN_PW, max_reconnect_attempts=0)
        pod: NatsClient | None = None
        try:
            admin_js = admin_nc.jetstream()
            admin_kv = await admin_js.create_key_value(bucket=bucket_name, direct=True)
            await admin_kv.put(own, b"3")
            await admin_kv.put(other, b"9")

            pod = await NatsClient.connect(
                nats_url=uri,
                nats_subject_namespace=_NS,
                client_name="kv-watch-key-live",
                user="pod",
                password=_POD_PW,
                inbox_prefix=permissions.inbox_prefix,
                startup_timeout=timedelta(seconds=10),
            )
            bucket = await pod.kv_bucket(name=DATA_VERSIONS_BUCKET_SUFFIX, create_if_missing=False, direct=True)

            with caplog.at_level(logging.INFO, logger="threetears.nats"):
                async with aclosing(bucket.watch_key(key=own, heartbeat=_HEARTBEAT, retry=_HEARTBEAT)) as watch:
                    # === the current value, then each write of the pod's own key =================
                    current = await asyncio.wait_for(anext(watch), timeout=_WAIT)
                    assert current.value == b"3"
                    assert current.revision == 1

                    await admin_kv.put(other, b"10")  # another principal's key: never delivered
                    revision = await admin_kv.put(own, b"4")
                    assert await asyncio.wait_for(anext(watch), timeout=_WAIT) == KvKeyUpdate(
                        key=own, value=b"4", revision=revision
                    )

                    await admin_kv.delete(own)
                    removal = await asyncio.wait_for(anext(watch), timeout=_WAIT)
                    assert removal.deleted

                    # === a lost consumer is replaced under a fresh name ============================
                    first = await _watch_consumers(admin_js, stream)
                    assert len(first) == 1, first
                    await admin_js.delete_consumer(stream, first[0])
                    await admin_kv.put(own, b"5")
                    # the replacement redelivers the latest message: b"5", written after the loss
                    replaced = await asyncio.wait_for(anext(watch), timeout=_WAIT)
                    assert replaced.value == b"5"
                    second = await _watch_consumers(admin_js, stream)
                    assert len(second) == 1, second
                    assert second[0] != first[0]

                # === closing the watch leaves nothing subscribed to its consumer =================
                for _ in range(50):
                    info = await admin_js.consumer_info(stream, second[0])
                    if not info.push_bound:
                        break
                    await asyncio.sleep(0.1)
                assert not info.push_bound

            violations = [r.getMessage() for r in caplog.records if "permissions violation" in r.getMessage().lower()]
            assert not violations, violations
            assert "could not be created" not in caplog.text
            # positive control: the capture sees this package's log lines, so the two absences above
            # are evidence rather than a logger nobody was listening to
            assert "replacing it" in caplog.text

            # === the stock watch, whose consumer is unnamed, is refused on the same connection ======
            stock_kv = await pod.raw.jetstream(timeout=3).key_value(bucket_name)
            with pytest.raises((nats.errors.TimeoutError, nats.errors.NoRespondersError)):
                await stock_kv.watch(own)
        finally:
            if pod is not None:
                await pod.shutdown(drain_timeout=timedelta(seconds=2))
            await admin_nc.close()
            set_default_namespace(previous_ns)
