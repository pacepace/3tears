"""Integration test: a renewal refused ON PURPOSE stops the pod at once; any other refusal does not.

nats-server tells a refused connection only ``-ERR 'Authorization Violation'`` -- the auth-callout's
reason reaches the server log and nowhere else -- so a pod cannot tell "your identity was superseded,
stop serving" from "the callout is down, try again". The ruling (Q16, 2026-09-30) needs both:

- the callout refuses a renewal because the pod is SUPERSEDED: the responder publishes a typed
  :class:`~threetears.nats.CredentialRefusal` to the pod's inbox over the connection the pod still
  holds, and the pod closes every connection at once (:meth:`NatsClient.abandon_on_refusal`);
- the callout is unreachable (it answers nothing, and the server times the callout out): the pod
  keeps its still-valid connection, requests on it still complete, it is not counted unhealthy, and
  the renewal is retried until the connection's own credential expires.

Both run against a real nats-server with config-mode ``auth_callout`` and the real
:class:`~threetears.nats.AuthCalloutResponder` serving the callout through ``handle_request``.

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
    CredentialRefusal,
    CredentialRefusalReason,
    GrantPolicy,
    IncomingMessage,
    NatsClient,
    NatsClientError,
    PrincipalPermissions,
    PrincipalResolver,
    RefusedPrincipal,
    ResolvedPrincipal,
    Subject,
    account_public_key,
    generate_account_seed,
)
from threetears.nats.subjects import get_default_namespace, set_default_namespace

pytestmark = pytest.mark.integration

_NS = "refusallive"
_TOKEN = "refusal-live-identity"  # noqa: S105 - the test resolver's only admissible credential
_ADMIN_PW = "admin-pw"  # noqa: S105 - ephemeral testcontainer credential
_INBOX = "_INBOX_refusal_live"
_POD = "pod-a"
_GENERATION = "g-7"
_QUERY_SUBJECT = "svc.query"
_TTL_SECONDS = 100
#: the server's wait for a callout answer: short, so an unreachable callout refuses promptly.
_CALLOUT_TIMEOUT_SECONDS = 2


class _ScriptedResolver(PrincipalResolver):
    """admits the first connection; then refuses renewals the way the test says."""

    def __init__(self) -> None:
        """start admitting.

        :return: nothing
        :rtype: None
        """
        self.calls = 0
        #: the refusal to return for every renewal, or ``None`` to admit it
        self.refuse_with: CredentialRefusal | None = None

    async def resolve(self, request: AuthCalloutRequest) -> ResolvedPrincipal | RefusedPrincipal | None:
        """admit the test credential, or refuse it on purpose once told to.

        :param request: the decoded authorization request
        :ptype request: AuthCalloutRequest
        :return: the principal, the scripted refusal, or ``None`` for a foreign credential
        :rtype: ResolvedPrincipal | RefusedPrincipal | None
        """
        self.calls += 1
        if request.bootstrap_token != _TOKEN:
            return None
        if self.calls > 1 and self.refuse_with is not None:
            return RefusedPrincipal(inbox_prefix=_INBOX, refusal=self.refuse_with)
        return ResolvedPrincipal(conn_id="refusal-live", name="refusal-live")


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


@dataclass
class _Stack:
    """a running server, the responder serving its callout, and a connected pod."""

    admin: NatsClient
    responder: AuthCalloutResponder
    resolver: _ScriptedResolver
    pod: NatsClient


@contextlib.contextmanager
def _nats_with_config(config_text: str, conf_dir: Path) -> Iterator[str]:
    """start a nats-server with ``config_text``; yield its URI.

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
async def _stack(tmp_path: Path) -> AsyncIterator[_Stack]:
    """a server with config-mode auth_callout, the real responder on it, and a pod admitted by it.

    :param tmp_path: a directory for the server configuration
    :ptype tmp_path: Path
    :return: the running stack, yielded
    :rtype: AsyncIterator[_Stack]
    """
    account_seed = generate_account_seed()
    admin_user = {"user": "admin", "password": _ADMIN_PW}
    config = (
        "port: 4222\n"
        "authorization {\n"
        f"  timeout: {_CALLOUT_TIMEOUT_SECONDS}\n"
        "  auth_callout {\n"
        f'    issuer: "{account_public_key(account_seed)}"\n'
        "    auth_users: [ admin ]\n"
        "  }\n"
        f"  users: [ {json.dumps(admin_user)} ]\n"
        "}\n"
    )
    previous_ns = get_default_namespace()
    with _nats_with_config(config, tmp_path) as uri:
        admin = await NatsClient.connect(
            nats_url=uri,
            nats_subject_namespace=_NS,
            client_name="refusal-admin",
            user="admin",
            password=_ADMIN_PW,
            verify_jetstream=False,
        )
        resolver = _ScriptedResolver()
        responder = AuthCalloutResponder(
            nc=admin,
            account_seed=account_seed,
            resolver=resolver,
            policy=_PodGrant(),
            account_name="$G",
            user_jwt_ttl_seconds=_TTL_SECONDS,
        )
        await responder.start()

        async def _answer(msg: IncomingMessage) -> None:
            assert msg.reply_subject is not None
            await admin.publish_raw_reply(reply_subject=msg.reply_subject, payload=b"answer")

        await admin.subscribe(Subject.raw(_QUERY_SUBJECT), cb=_answer)
        await admin.flush()
        pod = await NatsClient.connect(
            nats_url=uri,
            nats_subject_namespace=_NS,
            client_name="refusal-pod",
            auth_token=lambda: _TOKEN,
            inbox_prefix=_INBOX,
            verify_jetstream=False,
            startup_timeout=timedelta(seconds=15),
        )
        try:
            yield _Stack(admin=admin, responder=responder, resolver=resolver, pod=pod)
        finally:
            await pod.shutdown(drain_timeout=timedelta(seconds=2))
            await admin.shutdown(drain_timeout=timedelta(seconds=2))
            set_default_namespace(previous_ns)


def _is_this_runner(refusal: CredentialRefusal) -> bool:
    """what a pod checks: its own pod-session, and the generation it presents.

    :param refusal: the refusal received
    :ptype refusal: CredentialRefusal
    :return: whether it names this runner
    :rtype: bool
    """
    return refusal.pod_id == _POD and refusal.identity_generation == _GENERATION


async def _query(pod: NatsClient) -> bytes:
    """one request on the pod's current connection.

    :param pod: the pod
    :ptype pod: NatsClient
    :return: the reply
    :rtype: bytes
    """
    return await pod.request_raw(subject=Subject.raw(_QUERY_SUBJECT), payload=b"q", timeout=timedelta(seconds=5))


async def test_a_superseded_pod_stops_serving_at_once(tmp_path: Path) -> None:
    """the refusal names this runner: every connection is closed within moments, not at expiry."""
    if not check_docker_available():
        pytest.skip("Docker not available")

    async with _stack(tmp_path) as stack:
        pod = stack.pod
        await pod.abandon_on_refusal(inbox_prefix=_INBOX, is_mine=_is_this_runner)
        assert await _query(pod) == b"answer"

        # a refusal naming ANOTHER runner of this principal (an older generation) is not this one's
        stack.resolver.refuse_with = CredentialRefusal(
            reason=CredentialRefusalReason.SUPERSEDED, pod_id=_POD, identity_generation="g-6"
        )
        with pytest.raises(Exception):  # noqa: B017,PT011 -- nats-py's refusal type is not ours to pin
            await pod.renew_connection(retire_after=timedelta(seconds=30))
        await asyncio.sleep(1.0)
        assert not pod.is_closed, "a refusal for another runner closed this one"
        assert await _query(pod) == b"answer"

        # the refusal that names this runner
        stack.resolver.refuse_with = CredentialRefusal(
            reason=CredentialRefusalReason.SUPERSEDED, pod_id=_POD, identity_generation=_GENERATION
        )
        refused_at = time.monotonic()
        renewal = asyncio.create_task(pod.renew_connection(retire_after=timedelta(seconds=30)))
        for _ in range(100):
            if pod.is_closed:
                break
            await asyncio.sleep(0.05)
        closed_after = time.monotonic() - refused_at
        assert pod.is_closed, "a superseded pod kept serving"
        # the first refusal is answered in milliseconds; the renewal's own bound is 10s, the
        # credential's expiry 100s. closing well inside either is closing at the refusal.
        assert closed_after < 3.0, f"closed {closed_after:.1f}s after the refusal"
        # and the renewal attempt is stopped with it, rather than asking the callout until its bound
        with pytest.raises(NatsClientError, match="abandoned"):
            await asyncio.wait_for(renewal, timeout=2.0)


async def test_an_unreachable_callout_keeps_the_pod_serving(tmp_path: Path) -> None:
    """no refusal arrives: the pod keeps its valid connection, healthy, and keeps retrying."""
    if not check_docker_available():
        pytest.skip("Docker not available")

    async with _stack(tmp_path) as stack:
        pod = stack.pod
        await pod.abandon_on_refusal(inbox_prefix=_INBOX, is_mine=_is_this_runner)
        await stack.responder.stop()  # the callout is now unreachable: the server times it out
        await stack.admin.flush()

        for _ in range(3):
            with pytest.raises(Exception):  # noqa: B017,PT011 -- nats-py's refusal type is not ours to pin
                await pod.renew_connection(retire_after=timedelta(seconds=30))

        assert not pod.is_closed
        assert pod.is_connected
        assert pod.is_healthy, "a refused renewal counted against the connection still in use"
        assert await _query(pod) == b"answer"
