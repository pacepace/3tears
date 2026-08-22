"""Integration test: the durable-answer grants work, and fail closed, against a live broker.

The whole design rests on grant STRINGS. A pod that cannot publish its own result, or a registry that
cannot bind a consumer to collect it, does not fail with a denial that says so -- the JetStream
operation publishes to a ``$JS.API`` subject the connection lacks, the server drops it, and the op
hangs to its deadline. That reads as an unreachable broker, which is exactly how the original
incident read, so proving the grants on a real nats-server's subject matcher is the point.

Three things are proven end to end, with the SAME allow-lists
:func:`threetears.nats.subject_permissions.build_permissions` mints in production:

- a tool pod can publish a result under its OWN pod id and get a PubAck -- no silent timeout;
- the same connection is DENIED publishing under a PEER pod's id, which is what makes a standing
  grant safe where a standing grant on the requester's inbox tree was not;
- a registry connection can open the exact pull consumer :class:`JetStreamResultWaiter` opens and
  collect that result, which is the half a unit test cannot cover (the JS-API consumer grants).

The auth-callout responder is not stood up: the grant strings are what matters, so they are applied
directly as config-mode ``authorization`` permissions -- the same allow-list the responder mints.

Gated on docker: a checkout without docker skips cleanly.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
from collections.abc import AsyncIterator, Iterator
from datetime import timedelta
from pathlib import Path

import nats
import nats.errors
import nats.js.errors
import pytest

from threetears.core.testing.containers import check_docker_available
from threetears.nats import (
    NatsClient,
    Principal,
    Subjects,
    build_permissions,
    inbox_prefix_for,
    result_stream_name,
    set_default_namespace,
)
from threetears.nats.user_jwt import generate_account_seed, mint_user_jwt

pytestmark = pytest.mark.integration

_NS = "livegrant"
# uuids, and that is a CONTRACT rather than a test convenience: ``coll-task-07c`` gives a tool pod
# the scoped collections bucket, and the scope is derived from ``tool_pods.id`` by
# ``kv_key_scope_for``, which refuses anything non-uuid because a boundary derived from a display
# name is not provably collision-free. A slug pod id can no longer be granted at all.
_POD = "01947100-0000-7000-8000-0000000000aa"
_PEER_POD = "01947100-0000-7000-8000-0000000000bb"
_REGISTRY_CONN = "reg-1"
_ADMIN_PW = "admin-pw"  # noqa: S105 - ephemeral testcontainer credential
_POD_PW = "pod-pw"  # noqa: S105 - ephemeral testcontainer credential
_REGISTRY_PW = "registry-pw"  # noqa: S105 - ephemeral testcontainer credential


def _minted_permissions(principal: Principal, **ids: str) -> tuple[list[str], list[str]]:
    """mint a real user JWT for one principal and return its (pub allow, sub allow) lists.

    Feeding the MINTED strings into the server's static ``authorization`` proves the credential the
    responder would actually issue, rather than a hand-typed copy of it that could quietly diverge.

    :param principal: the connection identity class to mint for
    :ptype principal: Principal
    :param ids: the scoping ids that principal requires (pod_id / conn_id / agent_id)
    :ptype ids: str
    :return: the publish and subscribe allow-lists
    :rtype: tuple[list[str], list[str]]
    """
    token = mint_user_jwt(
        account_seed=generate_account_seed(),
        user_public_key="UTEST",  # sub is irrelevant for the static-permissions projection
        permissions=build_permissions(principal, **ids),
        name=f"{principal.value}-grant-test",
        expires_in_seconds=600,
    )
    payload_seg = token.split(".")[1]
    payload = json.loads(base64.urlsafe_b64decode(payload_seg + "=" * (-len(payload_seg) % 4)))
    nats_claim = payload["nats"]
    return nats_claim["pub"]["allow"], nats_claim["sub"]["allow"]


def _server_config(users: list[dict[str, object]]) -> str:
    """a nats-server config: JetStream on, an admin (full) plus each supplied scoped user."""
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


def _scoped_user(name: str, password: str, allow: tuple[list[str], list[str]]) -> dict[str, object]:
    """one static-authorization user carrying a minted allow-list."""
    pub_allow, sub_allow = allow
    return {
        "user": name,
        "password": password,
        "permissions": {
            "publish": {"allow": pub_allow},
            "subscribe": {"allow": sub_allow},
            "allow_responses": True,
        },
    }


@contextlib.contextmanager
def _nats_with_auth(config_text: str, conf_dir: Path) -> Iterator[str]:
    """start a JetStream nats-server with a custom ``authorization`` config; yield its URI."""
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
async def _raw(uri: str, *, user: str, password: str, inbox: str, errors: list[str]) -> AsyncIterator:
    """connect a raw nats client; route async permission-violation frames into ``errors``."""

    async def _err_cb(exc: Exception) -> None:
        errors.append(str(exc))

    nc = await nats.connect(
        uri,
        user=user,
        password=password,
        inbox_prefix=inbox.encode(),
        error_cb=_err_cb,
        max_reconnect_attempts=0,
    )
    try:
        yield nc
    finally:
        await nc.close()


async def test_a_pod_delivers_its_own_result_and_a_registry_collects_it(tmp_path: Path) -> None:
    if not check_docker_available():
        pytest.skip("Docker not available")

    set_default_namespace(_NS)
    pod_allow = _minted_permissions(Principal.TOOL_POD, pod_id=_POD)
    registry_allow = _minted_permissions(Principal.REGISTRY, conn_id=_REGISTRY_CONN)

    # sanity on the credential about to be enforced: the pod's standing grant is present and is
    # scoped to its OWN id, and no principal carries the broad control-plane hole.
    assert str(Subjects.tools_result_pod_wildcard(_POD)) in pod_allow[0]
    assert str(Subjects.tools_result_pod_wildcard(_PEER_POD)) not in pod_allow[0]
    assert "$JS.API.>" not in pod_allow[0]
    assert "$JS.API.>" not in registry_allow[0]

    stream = result_stream_name()
    result_subject = Subjects.tools_result(_POD, "call-1")
    peer_subject = Subjects.tools_result(_PEER_POD, "call-1")
    config = _server_config(
        [
            _scoped_user("pod", _POD_PW, pod_allow),
            _scoped_user("registry", _REGISTRY_PW, registry_allow),
        ]
    )

    with _nats_with_auth(config, tmp_path) as uri:
        # --- admin declares the stream the registry declares in production ---
        admin_errors: list[str] = []
        async with _raw(uri, user="admin", password=_ADMIN_PW, inbox="_INBOX", errors=admin_errors) as admin_nc:
            from nats.js.api import StorageType, StreamConfig  # noqa: PLC0415

            await admin_nc.jetstream().add_stream(
                StreamConfig(
                    name=stream,
                    subjects=[str(Subjects.tools_result_wildcard()), str(Subjects.tools_reply_wildcard())],
                    storage=StorageType.MEMORY,
                    max_msgs_per_subject=1,
                )
            )

        # --- the pod delivers its answer under its OWN id: this must genuinely ack ---
        pod_errors: list[str] = []
        pod_inbox = inbox_prefix_for(Principal.TOOL_POD, conn_id=_POD)
        async with _raw(uri, user="pod", password=_POD_PW, inbox=pod_inbox, errors=pod_errors) as pod_nc:
            pod_js = pod_nc.jetstream(timeout=8)
            ack = await pod_js.publish(result_subject.path, b"68KB of results")
            assert ack.stream == stream
            assert not [e for e in pod_errors if "permissions violation" in e.lower()], pod_errors

            # --- and is DENIED delivering under a PEER pod's id ---
            before = len(pod_errors)
            with pytest.raises((nats.errors.TimeoutError, nats.errors.NoRespondersError)):
                await pod_js.publish(peer_subject.path, b"forged")
            await asyncio.sleep(0.3)  # let the async -ERR frame land in the error callback
            violations = [e for e in pod_errors[before:] if "permissions violation" in e.lower()]
            assert violations, f"a peer's result subject was not refused: {pod_errors[before:]}"
            # the violation must name the FOREIGN pod's subject -- proof it was the grant, not some
            # unrelated failure, that refused the publish (nats-py lowercases the server's frame).
            assert any(_PEER_POD.lower() in e.lower() for e in violations), violations

        # --- the registry collects it through the SAME consumer the waiter opens ---
        client = await NatsClient.connect(
            nats_url=uri,
            nats_subject_namespace=_NS,
            client_name="registry-grant-test",
            user="registry",
            password=_REGISTRY_PW,
            inbox_prefix=inbox_prefix_for(Principal.REGISTRY, conn_id=_REGISTRY_CONN),
        )
        try:
            waiter = await client.jetstream_result_waiter(
                subject=result_subject,
                stream=stream,
                wait_budget=timedelta(seconds=10),
            )
            try:
                delivered = await waiter.wait(timeout=timedelta(seconds=10))
            finally:
                await waiter.close()
        finally:
            await client.shutdown()

        assert delivered == b"68KB of results"
