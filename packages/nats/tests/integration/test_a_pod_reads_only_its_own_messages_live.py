"""Integration test: a pod collects its own deliveries and reads nothing that belongs to another agent.

Two surfaces, each against a real nats-server running the allow-lists :func:`mint_user_jwt` mints
from :func:`build_permissions` exactly as the auth callout mints them, applied as config-mode
``authorization`` permissions as the sibling live tests do. A JetStream call the grant does not
cover is never answered, so every refusal below is a real broker's refusal:

- **the durable answer stream.** An agent pod collects the reply the registry delivers to IT,
  through the one consumer shape its grant admits -- named, filtered in the create subject, pushed
  to its own inbox. It cannot create a consumer over another agent's replies, a tool pod's results,
  or the whole stream, in the named, unnamed or durable form; it cannot pull from, inspect or delete
  the registry's collector; it holds no consumer grant on the channel-delivery or audit streams it
  publishes to; and a body whose filter disagrees with its subject is refused by the server. The
  registry's collector afterwards still holds every message, so nothing was consumed out from under
  it. A tool pod's result publish is still acknowledged, and it creates no consumer at all.
- **the agent-config hot cache.** A bucket created the legacy way (no direct reads) is reconciled
  in place the way the hub declares it; the agent pod then reads and watches its OWN key, and is
  refused another agent's key and any write, its own key included.

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
from uuid import UUID

import nats
import nats.errors
import pytest
from nats.js.api import AckPolicy, StorageType

from threetears.core.testing.containers import check_docker_available
from threetears.nats import NatsClient
from threetears.nats.kv import NatsKvBucket
from threetears.nats.result_delivery import result_stream_name
from threetears.nats.subject_permissions import (
    Principal,
    PrincipalPermissions,
    agent_config_bucket_name,
    build_permissions,
)
from threetears.nats.subjects import Subjects, get_default_namespace, set_default_namespace
from threetears.nats.user_jwt import generate_account_seed, mint_user_jwt

pytestmark = pytest.mark.integration

_NS = "ownread"
_AGENT = UUID("019470a8-b5c3-7def-8123-0000000000f1")
_OTHER_AGENT = UUID("019470a8-b5c3-7def-8123-0000000000f2")
_AGENT_POD = "01947100-0000-7000-8000-0000000000f1"
_TOOL_POD = "01947100-0000-7000-8000-0000000000f3"
_ADMIN_PW = "admin-pw"  # noqa: S105 - ephemeral testcontainer credential
_AGENT_PW = "agent-pw"  # noqa: S105 - ephemeral testcontainer credential
_TOOL_PW = "tool-pw"  # noqa: S105 - ephemeral testcontainer credential
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
        name="own-messages-live",
        expires_in_seconds=600,
    )
    payload_seg = token.split(".")[1]
    payload = json.loads(base64.urlsafe_b64decode(payload_seg + "=" * (-len(payload_seg) % 4)))
    return payload["nats"]["pub"]["allow"], payload["nats"]["sub"]["allow"]


def _user(name: str, password: str, permissions: PrincipalPermissions) -> dict[str, object]:
    """one static-authorization user carrying a minted allow-list.

    :param name: the user name
    :ptype name: str
    :param password: the user password
    :ptype password: str
    :param permissions: the resolved allow-list to mint and apply
    :ptype permissions: PrincipalPermissions
    :return: the user entry
    :rtype: dict[str, object]
    """
    pub_allow, sub_allow = _minted_allow_lists(permissions)
    return {
        "user": name,
        "password": password,
        "permissions": {
            "publish": {"allow": pub_allow},
            "subscribe": {"allow": sub_allow},
            "allow_responses": True,
        },
    }


def _server_config(users: list[dict[str, object]]) -> str:
    """a JetStream nats-server config with a full admin and each supplied scoped user.

    :param users: the scoped users
    :ptype users: list[dict[str, object]]
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
            *users,
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


async def _connect(uri: str, *, user: str, password: str, inbox_prefix: str | None = None) -> NatsClient:
    """connect the wrapper client as one configured user.

    :param uri: the server URI
    :ptype uri: str
    :param user: the user name
    :ptype user: str
    :param password: the user password
    :ptype password: str
    :param inbox_prefix: the scoped inbox the user's grant covers, or ``None`` for the default
    :ptype inbox_prefix: str | None
    :return: the connected client
    :rtype: NatsClient
    """
    extra = {"inbox_prefix": inbox_prefix} if inbox_prefix is not None else {}
    return await NatsClient.connect(
        nats_url=uri,
        nats_subject_namespace=_NS,
        client_name=f"{user}-own-messages",
        user=user,
        password=password,
        startup_timeout=timedelta(seconds=10),
        **extra,
    )


def _create_body(stream: str, name: str | None, filter_subject: str, **extra: object) -> bytes:
    """a consumer-create request body, as nats-py would send it.

    :param stream: the stream the consumer is on
    :ptype stream: str
    :param name: the consumer name, or ``None`` for an unnamed create
    :ptype name: str | None
    :param filter_subject: the body's filter
    :ptype filter_subject: str
    :param extra: further consumer config fields
    :ptype extra: object
    :return: the JSON body
    :rtype: bytes
    """
    config: dict[str, object] = {"filter_subject": filter_subject, "ack_policy": "none", **extra}
    if name is not None:
        config["name"] = name
    return json.dumps({"stream_name": stream, "config": config}).encode()


async def test_a_pod_collects_its_own_deliveries_and_no_other_agents(tmp_path: Path) -> None:
    """the agent pod collects its reply; every route to another principal's messages is refused."""
    if not check_docker_available():
        pytest.skip("Docker not available")

    previous_ns = get_default_namespace()
    set_default_namespace(_NS)
    try:
        agent_permissions = build_permissions(
            Principal.AGENT_POD, agent_id=str(_AGENT), pod_id=_AGENT_POD, conn_id=_AGENT_POD
        )
        tool_permissions = build_permissions(Principal.TOOL_POD, pod_id=_TOOL_POD, conn_id=_TOOL_POD)
        config = _server_config(
            [
                _user("agent", _AGENT_PW, agent_permissions),
                _user("tool", _TOOL_PW, tool_permissions),
            ]
        )
        results = result_stream_name()
        own_reply = Subjects.tools_reply(_AGENT, "call-1")
        other_reply = Subjects.tools_reply(_OTHER_AGENT, "call-1")
        tool_result = Subjects.tools_result(_TOOL_POD, "call-9")
        result_family = str(Subjects.tools_result_wildcard())
        reply_family = str(Subjects.tools_reply_wildcard())
    finally:
        set_default_namespace(previous_ns)
    channels = f"{_NS}-channels-deliver"
    audit = f"{_NS}-audit"

    with _nats_with_auth(config, tmp_path) as uri:
        hub = await _connect(uri, user="admin", password=_ADMIN_PW)
        agent: NatsClient | None = None
        tool: NatsClient | None = None
        try:
            # === the declarers: the registry's result stream and its collector, the hub's streams ===
            admin_js = hub.raw.jetstream()
            await admin_js.add_stream(name=results, subjects=[result_family, reply_family], storage=StorageType.MEMORY)
            await admin_js.add_stream(name=channels, subjects=[f"{_NS}.channels.deliver.*"], storage=StorageType.MEMORY)
            await admin_js.add_stream(name=audit, subjects=[f"{_NS}.audit.>"], storage=StorageType.MEMORY)
            await admin_js.publish(own_reply.path, b"the reply for this agent")
            await admin_js.publish(other_reply.path, b"another agent's reply")
            await admin_js.publish(f"{_NS}.channels.deliver.slack", b"another agent's answer")
            await admin_js.publish(f"{_NS}.audit.tool.call", b"another agent's audit event")
            await admin_js.add_consumer(results, durable_name="registry-collector", ack_policy=AckPolicy.EXPLICIT)

            agent = await _connect(uri, user="agent", password=_AGENT_PW, inbox_prefix=agent_permissions.inbox_prefix)
            tool = await _connect(uri, user="tool", password=_TOOL_PW, inbox_prefix=tool_permissions.inbox_prefix)

            # === SUCCEEDS: the agent collects the reply delivered to IT =============================
            waiter = await agent.jetstream_result_waiter(
                subject=own_reply, stream=results, wait_budget=timedelta(seconds=_WAIT)
            )
            try:
                assert await waiter.wait(timeout=timedelta(seconds=_WAIT)) == b"the reply for this agent"
            finally:
                await waiter.close()

            # === SUCCEEDS: a tool pod's result publish is acknowledged ==============================
            ack = await tool.raw.jetstream(timeout=_WAIT).publish(tool_result.path, b"68KB of results")
            assert ack.stream == results

            # === REFUSED by the grant: every route to a message that is not the pod's own ==========
            probes: list[tuple[NatsClient, str, bytes]] = [
                # a named create over another agent's reply, every reply, a tool pod's result
                (
                    agent,
                    f"$JS.API.CONSUMER.CREATE.{results}.c1.{other_reply.path}",
                    _create_body(results, "c1", other_reply.path),
                ),
                (
                    agent,
                    f"$JS.API.CONSUMER.CREATE.{results}.c2.{_NS}.tools.reply.*.call-1",
                    _create_body(results, "c2", f"{_NS}.tools.reply.*.call-1"),
                ),
                (
                    agent,
                    f"$JS.API.CONSUMER.CREATE.{results}.c3.{tool_result.path}",
                    _create_body(results, "c3", tool_result.path),
                ),
                # the unnamed and durable creates, whose filter rides only in the body
                (agent, f"$JS.API.CONSUMER.CREATE.{results}", _create_body(results, None, ">")),
                (
                    agent,
                    f"$JS.API.CONSUMER.DURABLE.CREATE.{results}.c4",
                    _create_body(results, None, ">", durable_name="c4"),
                ),
                # every verb that reaches the registry's collector by name, and the listings
                (
                    agent,
                    f"$JS.API.CONSUMER.MSG.NEXT.{results}.registry-collector",
                    b'{"batch": 10, "expires": 1000000000}',
                ),
                (agent, f"$JS.API.CONSUMER.INFO.{results}.registry-collector", b""),
                (agent, f"$JS.API.CONSUMER.DELETE.{results}.registry-collector", b""),
                (agent, f"$JS.API.CONSUMER.LIST.{results}", b"{}"),
                (agent, f"$JS.API.STREAM.INFO.{results}", b""),
                # the streams it only publishes to
                (
                    agent,
                    f"$JS.API.CONSUMER.CREATE.{channels}.c5.{_NS}.channels.deliver.slack",
                    _create_body(channels, "c5", f"{_NS}.channels.deliver.slack"),
                ),
                (
                    agent,
                    f"$JS.API.CONSUMER.CREATE.{audit}.c6.{_NS}.audit.tool.call",
                    _create_body(audit, "c6", f"{_NS}.audit.tool.call"),
                ),
                # a tool pod collects nothing at all
                (
                    tool,
                    f"$JS.API.CONSUMER.CREATE.{results}.c7.{tool_result.path}",
                    _create_body(results, "c7", tool_result.path),
                ),
                (
                    tool,
                    f"$JS.API.CONSUMER.MSG.NEXT.{results}.registry-collector",
                    b'{"batch": 10, "expires": 1000000000}',
                ),
            ]
            for client, subject, body in probes:
                with pytest.raises(_REFUSED):
                    await client.raw.request(subject, body, timeout=2)

            # === REFUSED by the server: a body filter that disagrees with the granted subject =======
            # the grant admits the subject; the server then holds the body to it. these are the
            # only routes past a subject grant, so each is proven refused rather than assumed.
            own_filter_subject = f"$JS.API.CONSUMER.CREATE.{results}.c8.{own_reply.path}"
            for body in (
                _create_body(results, "c8", other_reply.path),
                _create_body(results, "c8", f"{_NS}.tools.>"),
                _create_body(results, "c8", own_reply.path, filter_subjects=[other_reply.path]),
                json.dumps(
                    {"stream_name": results, "config": {"name": "c8", "filter_subjects": [other_reply.path]}}
                ).encode(),
            ):
                response = json.loads((await agent.raw.request(own_filter_subject, body, timeout=_WAIT)).data)
                assert "error" in response, response

            # === nothing was consumed out from under the collector, and no probe consumer exists ====
            collector = await admin_js.consumer_info(results, "registry-collector")
            assert collector.num_pending == 3, collector
            names = {info.name for info in await admin_js.consumers_info(results)}
            assert not names & {"c1", "c2", "c3", "c4", "c7", "c8"}, names
            for stream in (channels, audit):
                assert not await admin_js.consumers_info(stream), stream
        finally:
            for client in (agent, tool, hub):
                if client is not None:
                    await client.shutdown(drain_timeout=timedelta(seconds=2))


async def test_an_agent_pod_reads_and_watches_its_own_config_and_writes_none(tmp_path: Path) -> None:
    """the hub reconciles the cache to direct reads; the pod reads its own key and nothing more."""
    if not check_docker_available():
        pytest.skip("Docker not available")

    previous_ns = get_default_namespace()
    set_default_namespace(_NS)
    try:
        permissions = build_permissions(
            Principal.AGENT_POD, agent_id=str(_AGENT), pod_id=_AGENT_POD, conn_id=_AGENT_POD
        )
        config = _server_config([_user("agent", _AGENT_PW, permissions)])
        bucket_name = agent_config_bucket_name()
    finally:
        set_default_namespace(previous_ns)
    own_key = str(_AGENT)
    other_key = str(_OTHER_AGENT)

    with _nats_with_auth(config, tmp_path) as uri:
        hub = await _connect(uri, user="admin", password=_ADMIN_PW)
        agent: NatsClient | None = None
        try:
            # === the LEGACY bucket: created with nats-py's defaults, so no direct reads ============
            legacy = await hub.raw.jetstream().create_key_value(bucket=bucket_name)
            await legacy.put(own_key, b'{"system_prompt": "own"}')
            await legacy.put(other_key, b'{"system_prompt": "another agent"}')
            assert not (await hub.raw.jetstream().stream_info(f"KV_{bucket_name}")).config.allow_direct

            # === the hub declares it: reconciled in place, entries kept ============================
            await NatsKvBucket.open(
                client=hub,
                full_name=bucket_name,
                ttl=None,
                storage="file",
                create_if_missing=True,
                history=1,
                direct=True,
            )
            assert (await hub.raw.jetstream().stream_info(f"KV_{bucket_name}")).config.allow_direct

            agent = await _connect(uri, user="agent", password=_AGENT_PW, inbox_prefix=permissions.inbox_prefix)

            # === SUCCEEDS: bind, read and watch its OWN key ========================================
            kv = await agent.raw.jetstream(timeout=_WAIT).key_value(bucket_name)
            entry = await kv.get(own_key)
            assert entry.value == b'{"system_prompt": "own"}'
            bound = await NatsKvBucket.open(
                client=agent,
                full_name=bucket_name,
                ttl=None,
                storage="file",
                create_if_missing=False,
                history=1,
            )
            async with aclosing(bound.watch_key(key=own_key)) as updates:
                first = await asyncio.wait_for(anext(updates), timeout=_WAIT)
                assert first.value == b'{"system_prompt": "own"}'

            # === REFUSED: another agent's key, and every write -- its own key included =============
            short = await agent.raw.jetstream(timeout=2).key_value(bucket_name)
            with pytest.raises(_REFUSED):
                await short.get(other_key)
            for key in (own_key, other_key):
                with pytest.raises(_REFUSED):
                    await short.put(key, b'{"system_prompt": "rewritten"}')
            stream = f"KV_{bucket_name}"
            for subject, body in (
                (
                    f"$JS.API.CONSUMER.CREATE.{stream}.w1.$KV.{bucket_name}.{other_key}",
                    _create_body(stream, "w1", f"$KV.{bucket_name}.{other_key}"),
                ),
                (
                    f"$JS.API.CONSUMER.CREATE.{stream}.w2.$KV.{bucket_name}.>",
                    _create_body(stream, "w2", f"$KV.{bucket_name}.>"),
                ),
                (f"$JS.API.CONSUMER.CREATE.{stream}", _create_body(stream, None, f"$KV.{bucket_name}.>")),
                (
                    f"$JS.API.STREAM.MSG.GET.{stream}",
                    json.dumps({"last_by_subj": f"$KV.{bucket_name}.{other_key}"}).encode(),
                ),
            ):
                with pytest.raises(_REFUSED):
                    await agent.raw.request(subject, body, timeout=2)

            # === nothing was rewritten ===============================================================
            assert (await legacy.get(own_key)).value == b'{"system_prompt": "own"}'
            assert (await legacy.get(other_key)).value == b'{"system_prompt": "another agent"}'
        finally:
            for client in (agent, hub):
                if client is not None:
                    await client.shutdown(drain_timeout=timedelta(seconds=2))
