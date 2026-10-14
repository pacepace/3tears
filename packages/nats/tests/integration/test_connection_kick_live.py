"""Integration test: a system-account user closes one connection on demand, beside the auth-callout.

Owner ruling Q17: a pod keeps a long credential, and when it must lose access its connection is
closed at once -- ``$SYS.REQ.SERVER.<server_id>.KICK`` -- and its reconnect is refused by the
callout. Proven here against real nats-servers, both the version the local stack runs (2.12.6) and
the one the clusters run (2.14.2):

- a SYSTEM-account user loads beside config-mode ``auth_callout`` and reaches the system subjects,
  once it is listed in ``auth_users``; an ordinary account user cannot reach them at all;
- every admission names its connection (server id + client id), which an
  :class:`~threetears.nats.AdmissionRecorder` keeps;
- a kick of that connection closes it within milliseconds, and a callout that refuses the
  principal keeps it from coming back;
- the three answers a kick can get -- kicked, already gone, server gone -- are the ones
  :func:`~threetears.nats.kick_connection` reads;
- a CONNZ probe closes nothing, and its three answers -- held, not held, server gone -- are the ones
  :func:`~threetears.nats.probe_connection` reads.

Gated on docker: a checkout without docker skips cleanly.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import pytest

from threetears.core.testing.containers import check_docker_available
from threetears.nats import (
    AuthCalloutRequest,
    AuthCalloutResponder,
    GrantPolicy,
    KickOutcome,
    NatsClient,
    NatsConnectionRef,
    PrincipalPermissions,
    ProbeOutcome,
    PrincipalResolver,
    RefusedPrincipal,
    ResolvedPrincipal,
    Subject,
    SystemAccountUnavailableError,
    account_public_key,
    generate_account_seed,
    kick_connection,
    probe_connection,
    require_system_account,
)
from threetears.nats.subjects import get_default_namespace, set_default_namespace

pytestmark = pytest.mark.integration

#: the server the local compose stack runs, and the one cobalt-dev and cobalt-prod run.
_IMAGES = ("nats:2.12.6-alpine", "nats:2.14.2-alpine")

_NS = "kicklive"
_TOKEN = "kick-live-identity"  # noqa: S105 - the test resolver's only admissible credential
_ADMIN_PW = "admin-pw"  # noqa: S105 - ephemeral testcontainer credential
_SYSTEM_PW = "system-pw"  # noqa: S105 - ephemeral testcontainer credential
_INBOX = "_INBOX_kick_live"
_QUERY_SUBJECT = "svc.query"
#: the server's wait for a callout answer: short, so a refused reconnect is refused promptly.
_CALLOUT_TIMEOUT_SECONDS = 2
#: a server id no server holds (a well-formed server nkey).
_ABSENT_SERVER_ID = "NBOGUSBOGUSBOGUSBOGUSBOGUSBOGUSBOGUSBOGUSBOGUSBOGUSBOGUS"


class _FencedResolver(PrincipalResolver):
    """admits the test credential until the test fences it; then refuses it."""

    def __init__(self) -> None:
        """start admitting.

        :return: nothing
        :rtype: None
        """
        self.fenced = False
        self.refusals = 0

    async def resolve(self, request: AuthCalloutRequest) -> ResolvedPrincipal | RefusedPrincipal | None:
        """admit the test credential while it is not fenced.

        :param request: the decoded authorization request
        :ptype request: AuthCalloutRequest
        :return: the principal, or ``None`` once fenced or for a foreign credential
        :rtype: ResolvedPrincipal | RefusedPrincipal | None
        """
        result: ResolvedPrincipal | None = None
        if request.bootstrap_token == _TOKEN and not self.fenced:
            result = ResolvedPrincipal(conn_id="kick-live", name="kick-live-pod")
        elif request.bootstrap_token == _TOKEN:
            self.refusals += 1
        return result


class _PodGrant(GrantPolicy):
    """the pod's grant: its inbox and one service subject."""

    def permissions(self, principal: ResolvedPrincipal) -> PrincipalPermissions:
        """the pod's allow-list.

        :param principal: the admitted principal
        :ptype principal: ResolvedPrincipal
        :return: the allow-list minted into the pod's user JWT
        :rtype: PrincipalPermissions
        """
        return PrincipalPermissions(
            publish=(_QUERY_SUBJECT,),
            subscribe=(f"{_INBOX}.>",),
            allow_responses=True,
            inbox_prefix=_INBOX,
        )


class _Recorder:
    """keeps every admitted connection, as a consumer's recorder would."""

    def __init__(self) -> None:
        """record nothing yet.

        :return: nothing
        :rtype: None
        """
        self.admitted: list[NatsConnectionRef] = []

    async def record_admission(self, request: AuthCalloutRequest, principal: ResolvedPrincipal) -> None:
        """keep the connection the request names.

        :param request: the admitted request
        :ptype request: AuthCalloutRequest
        :param principal: the principal it was admitted as
        :ptype principal: ResolvedPrincipal
        :return: nothing
        :rtype: None
        :raises ValueError: when the request names no connection, which denies it
        """
        connection = request.connection
        if connection is None:
            raise ValueError("the authorization request names no connection")
        self.admitted.append(connection)


@dataclass
class _Stack:
    """a running server, its callout, a system-account client and a connected pod."""

    admin: NatsClient
    system: NatsClient
    resolver: _FencedResolver
    recorder: _Recorder
    pod: NatsClient


@contextlib.contextmanager
def _nats_with_config(image: str, config_text: str, conf_dir: Path) -> Iterator[str]:
    """start a nats-server of ``image`` with ``config_text``; yield its URI.

    :param image: the nats-server image
    :ptype image: str
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
        nats_server(image)
        .with_volume_mapping(str(conf_dir), "/etc/nats", "ro")
        .with_command(["-c", "/etc/nats/nats.conf"])
    )
    container.start()
    try:
        yield nats_server_uri(container)
    finally:
        container.stop()


def _config(account_seed: bytes) -> str:
    """config-mode auth_callout beside a SYSTEM account, as the platform renders it.

    :param account_seed: the callout account's signing seed
    :ptype account_seed: bytes
    :return: the server configuration
    :rtype: str
    """
    return (
        "port: 4222\n"
        "authorization {\n"
        f"  timeout: {_CALLOUT_TIMEOUT_SECONDS}\n"
        "  auth_callout {\n"
        f'    issuer: "{account_public_key(account_seed)}"\n'
        # the system user is listed here too: a user the callout does not bypass goes through it,
        # whatever account it belongs to, and the callout knows nothing of it
        "    auth_users: [ admin, hub_system ]\n"
        "  }\n"
        f'  users: [ {{ user: admin, password: "{_ADMIN_PW}" }} ]\n'
        "}\n"
        "system_account: SYS\n"
        "accounts {\n"
        f'  SYS {{ users: [ {{ user: hub_system, password: "{_SYSTEM_PW}" }} ] }}\n'
        "}\n"
    )


@contextlib.asynccontextmanager
async def _stack(image: str, tmp_path: Path) -> AsyncIterator[_Stack]:
    """the server, the callout serving it, a system-account client, and a pod the callout admitted.

    :param image: the nats-server image
    :ptype image: str
    :param tmp_path: a directory for the server configuration
    :ptype tmp_path: Path
    :return: the running stack, yielded
    :rtype: AsyncIterator[_Stack]
    """
    account_seed = generate_account_seed()
    previous_ns = get_default_namespace()
    with _nats_with_config(image, _config(account_seed), tmp_path) as uri:
        admin = await NatsClient.connect(
            nats_url=uri,
            nats_subject_namespace=_NS,
            client_name="kick-admin",
            user="admin",
            password=_ADMIN_PW,
            verify_jetstream=False,
        )
        resolver = _FencedResolver()
        recorder = _Recorder()
        responder = AuthCalloutResponder(
            nc=admin,
            account_seed=account_seed,
            resolver=resolver,
            policy=_PodGrant(),
            account_name="$G",
            admission_recorder=recorder,
        )
        await responder.start()

        async def _answer(msg: object) -> None:
            reply = getattr(msg, "reply_subject", None)
            assert reply is not None
            await admin.publish_raw_reply(reply_subject=reply, payload=b"answer")

        await admin.subscribe(Subject.raw(_QUERY_SUBJECT), cb=_answer)
        await admin.flush()
        system = await NatsClient.connect(
            nats_url=uri,
            nats_subject_namespace=_NS,
            client_name="kick-system",
            user="hub_system",
            password=_SYSTEM_PW,
            verify_jetstream=False,
        )
        pod = await NatsClient.connect(
            nats_url=uri,
            nats_subject_namespace=_NS,
            client_name="kick-pod",
            auth_token=lambda: _TOKEN,
            inbox_prefix=_INBOX,
            verify_jetstream=False,
            startup_timeout=timedelta(seconds=15),
        )
        try:
            yield _Stack(admin=admin, system=system, resolver=resolver, recorder=recorder, pod=pod)
        finally:
            await pod.shutdown(drain_timeout=timedelta(seconds=2))
            await system.shutdown(drain_timeout=timedelta(seconds=2))
            await admin.shutdown(drain_timeout=timedelta(seconds=2))
            set_default_namespace(previous_ns)


@pytest.mark.parametrize("image", _IMAGES)
async def test_a_system_user_loads_beside_the_callout_and_only_it_reaches_the_system(
    image: str, tmp_path: Path
) -> None:
    """the SYS login is served; the ordinary account's is not, which is what a kick must not use."""
    if not check_docker_available():
        pytest.skip("Docker not available")

    async with _stack(image, tmp_path) as stack:
        server = await require_system_account(stack.system)
        assert server.id.startswith("N")
        # the one admission the callout made was the pod's: the system user bypassed it
        assert len(stack.recorder.admitted) == 1
        assert stack.recorder.admitted[0].server_id == server.id
        with pytest.raises(SystemAccountUnavailableError):
            await require_system_account(stack.admin)


@pytest.mark.parametrize("image", _IMAGES)
async def test_a_kick_closes_the_connection_at_once_and_the_fence_keeps_it_out(image: str, tmp_path: Path) -> None:
    """milliseconds, not the credential's day; and a refused reconnect means it stays out."""
    if not check_docker_available():
        pytest.skip("Docker not available")

    async with _stack(image, tmp_path) as stack:
        pod = stack.pod
        assert await pod.request_raw(subject=Subject.raw(_QUERY_SUBJECT), payload=b"q") == b"answer"
        [connection] = stack.recorder.admitted

        stack.resolver.fenced = True
        kicked_at = time.monotonic()
        outcome = await kick_connection(stack.system, connection)
        while pod.is_connected and time.monotonic() - kicked_at < 1.0:
            await asyncio.sleep(0.001)
        closed_after_ms = (time.monotonic() - kicked_at) * 1000

        assert outcome is KickOutcome.KICKED
        assert not pod.is_connected, "the kicked connection is still up"
        assert closed_after_ms < 250, f"the connection closed {closed_after_ms:.1f}ms after the kick"

        # the pod's own reconnect goes back through the callout, which now refuses it
        for _ in range(100):
            if stack.resolver.refusals >= 2:
                break
            await asyncio.sleep(0.1)
        assert stack.resolver.refusals >= 2, "the kicked pod never tried to come back"
        assert not pod.is_connected, "a refused principal came back"
        assert len(stack.recorder.admitted) == 1, "the fence admitted the kicked pod again"
        print(  # noqa: T201 - the measured latency is the evidence this test exists for
            f"\n[{image}] kick answered {outcome.value}; connection closed {closed_after_ms:.1f}ms after the "
            f"kick was sent; reconnect refused {stack.resolver.refusals} time(s)"
        )


@pytest.mark.parametrize("image", _IMAGES)
async def test_a_kick_of_a_connection_that_is_gone_says_so(image: str, tmp_path: Path) -> None:
    """a closed connection, and a server that no longer exists, are both answers -- not errors."""
    if not check_docker_available():
        pytest.skip("Docker not available")

    async with _stack(image, tmp_path) as stack:
        [connection] = stack.recorder.admitted
        stack.resolver.fenced = True
        assert await kick_connection(stack.system, connection) is KickOutcome.KICKED

        again = await kick_connection(stack.system, connection)
        elsewhere = await kick_connection(
            stack.system, NatsConnectionRef(server_id=_ABSENT_SERVER_ID, client_id=connection.client_id)
        )

        assert again is KickOutcome.NOT_CONNECTED
        assert elsewhere is KickOutcome.SERVER_GONE


@pytest.mark.parametrize("image", _IMAGES)
async def test_the_kick_reply_shapes_are_the_ones_the_helper_reads(image: str, tmp_path: Path) -> None:
    """pins the raw replies, so a server release that changes them fails here, naming the change."""
    if not check_docker_available():
        pytest.skip("Docker not available")

    async with _stack(image, tmp_path) as stack:
        [connection] = stack.recorder.admitted
        subject = Subject.raw(f"$SYS.REQ.SERVER.{connection.server_id}.KICK")

        gone = json.loads(
            await stack.system.request_raw(subject=subject, payload=json.dumps({"cid": 999_999}).encode())
        )

        assert gone["error"] == {"code": 500, "description": "no such client or leafnode id"}
        assert gone["server"]["id"] == connection.server_id


@pytest.mark.parametrize("image", _IMAGES)
async def test_a_probe_closes_nothing_and_says_whether_the_connection_is_held(image: str, tmp_path: Path) -> None:
    """held while open, not held once closed, server gone for an id no server holds."""
    if not check_docker_available():
        pytest.skip("Docker not available")

    async with _stack(image, tmp_path) as stack:
        [connection] = stack.recorder.admitted

        held = await probe_connection(stack.system, connection)
        assert held is ProbeOutcome.HELD
        assert stack.pod.is_connected, "a probe closed the connection it asked about"
        assert await stack.pod.request_raw(subject=Subject.raw(_QUERY_SUBJECT), payload=b"q") == b"answer"

        stack.resolver.fenced = True
        assert await kick_connection(stack.system, connection) is KickOutcome.KICKED
        outcome = await probe_connection(stack.system, connection)
        for _ in range(200):
            if outcome is ProbeOutcome.NOT_HELD:
                break
            await asyncio.sleep(0.01)
            outcome = await probe_connection(stack.system, connection)
        elsewhere = await probe_connection(
            stack.system, NatsConnectionRef(server_id=_ABSENT_SERVER_ID, client_id=connection.client_id)
        )

        assert outcome is ProbeOutcome.NOT_HELD
        assert elsewhere is ProbeOutcome.SERVER_GONE


@pytest.mark.parametrize("image", _IMAGES)
async def test_the_connz_reply_shape_is_the_one_the_probe_reads(image: str, tmp_path: Path) -> None:
    """pins the raw CONNZ reply filtered by client id, so a server release that changes it fails here."""
    if not check_docker_available():
        pytest.skip("Docker not available")

    async with _stack(image, tmp_path) as stack:
        [connection] = stack.recorder.admitted
        subject = Subject.raw(f"$SYS.REQ.SERVER.{connection.server_id}.CONNZ")

        held = json.loads(
            await stack.system.request_raw(subject=subject, payload=json.dumps({"cid": connection.client_id}).encode())
        )
        absent = json.loads(
            await stack.system.request_raw(subject=subject, payload=json.dumps({"cid": 999_999}).encode())
        )

        assert held["server"]["id"] == connection.server_id
        assert [entry["cid"] for entry in held["data"]["connections"]] == [connection.client_id]
        assert "error" not in held
        assert absent["data"]["connections"] == []
        assert "error" not in absent
