"""Integration test: an agent pod binds and works inside the buckets the hub declared, and manages no stream.

``STREAM.CREATE`` and ``STREAM.UPDATE`` carry ``sources`` in the request BODY, where no subject
permission can see them, so a pod holding either against a stream of its own could copy ANY stream on
the bus -- another agent's coordination bucket here -- into one it reads. The pod's grant is minted by
:func:`mint_user_jwt` from :func:`build_permissions` exactly as the auth callout mints it, and applied as
config-mode ``authorization`` permissions, as the sibling live tests do. A JetStream call the grant
does not cover is never answered, so each claim runs against a real nats-server:

- the pod, through the wrapper client, binds its own coordination bucket the hub declared and gets,
  puts, compare-and-sets, deletes, watches a key and lists keys by prefix;
- a bind-only open asking for an entry lifetime, on a bucket the hub declared with none, writes every
  entry with that lifetime, and the entry expires;
- a bind-only open of a bucket the hub never declared fails fast, naming the bucket;
- every stream-management verb -- a create with ``sources`` pointing at another agent's bucket, an
  update adding ``sources`` to its own bucket, delete, purge, snapshot -- is refused, on its own
  buckets, on the streams it consumes and on a stream nobody declared, and nothing is copied.

Gated on docker: a checkout without docker skips cleanly.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import time
from collections.abc import Iterator
from contextlib import aclosing
from datetime import timedelta
from pathlib import Path
from uuid import UUID

import nats
import nats.errors
import pytest

from threetears.core.testing.containers import check_docker_available
from threetears.nats import KvError, NatsClient
from threetears.nats.kv import KvTimings
from threetears.nats.result_delivery import result_stream_name
from threetears.nats.subject_permissions import (
    Principal,
    PrincipalPermissions,
    build_permissions,
    coordination_bucket_name,
    kv_key_scope_for,
)
from threetears.nats.subjects import get_default_namespace, set_default_namespace
from threetears.nats.user_jwt import generate_account_seed, mint_user_jwt

pytestmark = pytest.mark.integration

_NS = "nomgmt"
_AGENT = UUID("019470a8-b5c3-7def-8123-0000000000e1")
_VICTIM = UUID("019470a8-b5c3-7def-8123-0000000000e2")
_POD = "01947100-0000-7000-8000-0000000000e1"
_ADMIN_PW = "admin-pw"  # noqa: S105 - ephemeral testcontainer credential
_POD_PW = "pod-pw"  # noqa: S105 - ephemeral testcontainer credential
_WAIT = 10.0

#: the suffixes the agent declares: one the hub creates, one with an entry lifetime, one it never creates.
_CHECKPOINTS = "checkpoints"
_NONCES = "entry_challenge_nonces"
_NEVER_DECLARED = "survey-quota-cells"

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


async def test_a_pod_works_inside_the_hubs_buckets_and_manages_no_stream(tmp_path: Path) -> None:
    """bind, read, write, watch and list inside; every stream-management verb refused, nothing copied."""
    if not check_docker_available():
        pytest.skip("Docker not available")

    previous_ns = get_default_namespace()
    set_default_namespace(_NS)
    try:
        permissions = build_permissions(
            Principal.AGENT_POD,
            agent_id=str(_AGENT),
            pod_id=_POD,
            conn_id=_POD,
            coordination_buckets=(_CHECKPOINTS, _NONCES, _NEVER_DECLARED),
        )
        pub_allow, sub_allow = _minted_allow_lists(permissions)
        results = result_stream_name()
    finally:
        set_default_namespace(previous_ns)

    scope = kv_key_scope_for(Principal.AGENT_POD, agent_id=_AGENT)
    victim_scope = kv_key_scope_for(Principal.AGENT_POD, agent_id=_VICTIM)
    own = coordination_bucket_name(scope, _CHECKPOINTS, ns=_NS)
    never_declared = coordination_bucket_name(scope, _NEVER_DECLARED, ns=_NS)
    victim = coordination_bucket_name(victim_scope, _CHECKPOINTS, ns=_NS)

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
            # === the HUB declares: uniform memory buckets, no bucket-wide expiry =====================
            for suffix in (_CHECKPOINTS, _NONCES):
                await hub.ensure_kv_bucket(name=f"{scope}-{suffix}", direct=True)
            victim_bucket = await hub.ensure_kv_bucket(name=f"{victim_scope}-{_CHECKPOINTS}", direct=True)
            await victim_bucket.put(key="thread-v.state", value=b"respondent text")
            await hub.ensure_jetstream_stream(name="audit", subjects=[f"{_NS}.audit.>"])
            admin_js = hub.raw.jetstream()
            await admin_js.add_stream(name=results, subjects=[f"{_NS}.results-probe.>"])

            pod = await NatsClient.connect(
                nats_url=uri,
                nats_subject_namespace=_NS,
                client_name="pod-binds",
                user="pod",
                password=_POD_PW,
                inbox_prefix=permissions.inbox_prefix,
                startup_timeout=timedelta(seconds=10),
                # a bind-only open waits for an absent bucket's declarer; shortened so the
                # never-declared case below fails in a second rather than the production 30s.
                kv_timings=KvTimings(bind_wait_for_declarer_seconds=1.0),
            )

            # === SUCCEEDS: every KV operation inside its own bucket, bind-only ======================
            bucket = await pod.kv_bucket(name=f"{scope}-{_CHECKPOINTS}", create_if_missing=False)
            revision = await bucket.put(key="thread-1.a", value=b"1")
            assert await bucket.get(key="thread-1.a") == b"1"
            assert await bucket.update(key="thread-1.a", value=b"2", revision=revision) is not None
            assert await bucket.create(key="thread-1.b", value=b"1") is not None
            await bucket.put(key="thread-2.a", value=b"1")
            assert await bucket.delete(key="thread-1.b")
            assert sorted(await bucket.list_keys(prefix="thread-1.")) == ["thread-1.a"]
            assert sorted(await bucket.list_keys()) == ["thread-1.a", "thread-2.a"]
            async with aclosing(bucket.watch_key(key="thread-2.a")) as watch:
                first = await asyncio.wait_for(anext(watch), timeout=_WAIT)
                assert first.value == b"1"

            # === SUCCEEDS: an entry lifetime rides each write when the hub set none =================
            nonces = await pod.kv_bucket(name=f"{scope}-{_NONCES}", ttl=timedelta(seconds=1), create_if_missing=False)
            await nonces.create(key="nonce-1", value=b"1")
            assert await nonces.get(key="nonce-1") == b"1"
            deadline = time.monotonic() + _WAIT
            while await nonces.get(key="nonce-1") is not None and time.monotonic() < deadline:
                await asyncio.sleep(0.25)
            assert await nonces.get(key="nonce-1") is None, "the entry lifetime was not applied"

            # === FAILS LOUDLY and BOUNDED, naming the bucket: a bucket the hub never declared ========
            # a bind-only open waits for the declarer (a NATS restart leaves every pod bucket absent
            # until the hub re-declares), so the bound here is that wait, shortened for the test --
            # and the failure at its end names the bucket rather than arriving as a deadline.
            started = time.monotonic()
            with pytest.raises(KvError, match=never_declared):
                await pod.kv_bucket(name=f"{scope}-{_NEVER_DECLARED}", create_if_missing=False)
            assert time.monotonic() - started < 5.0, "a missing bucket must fail loudly, not at a deadline"

            # === REFUSED: every stream-management verb, on every stream =============================
            raw = pod.raw
            copy_victim = {"sources": [{"name": f"KV_{victim}"}], "subjects": [], "storage": "memory"}
            for subject, body in (
                (f"$JS.API.STREAM.CREATE.KV_{never_declared}", {"name": f"KV_{never_declared}", **copy_victim}),
                (f"$JS.API.STREAM.UPDATE.KV_{own}", {"name": f"KV_{own}", **copy_victim}),
                (f"$JS.API.STREAM.CREATE.{_NS}-exfil", {"name": f"{_NS}-exfil", **copy_victim}),
                (f"$JS.API.STREAM.UPDATE.{results}", {"name": results, **copy_victim}),
                (f"$JS.API.STREAM.UPDATE.{_NS}-audit", {"name": f"{_NS}-audit", **copy_victim}),
                (f"$JS.API.STREAM.DELETE.KV_{own}", {}),
                (f"$JS.API.STREAM.PURGE.KV_{own}", {}),
                (f"$JS.API.STREAM.SNAPSHOT.KV_{own}", {"deliver_subject": f"{permissions.inbox_prefix}.snap"}),
                (f"$JS.API.CONSUMER.CREATE.KV_{own}", {"stream_name": f"KV_{own}", "config": {}}),
            ):
                with pytest.raises(_REFUSED):
                    await raw.request(subject, json.dumps(body).encode(), timeout=2)

            # === nothing was copied, and the pod's own bucket was not reshaped =====================
            own_info = await admin_js.stream_info(f"KV_{own}")
            assert not own_info.config.sources
            names = await admin_js.streams_info()
            assert f"KV_{never_declared}" not in {info.config.name for info in names}
            assert f"{_NS}-exfil" not in {info.config.name for info in names}
            assert await bucket.get(key="thread-1.a") == b"2"
        finally:
            if pod is not None:
                await pod.shutdown(drain_timeout=timedelta(seconds=2))
            await hub.shutdown(drain_timeout=timedelta(seconds=2))
