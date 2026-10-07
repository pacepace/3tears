"""Integration test: a changed grant reaches a live connection when the minter asks it to renew.

A connection's permissions are fixed at admission, and with a day-long credential the connection
that is open would keep an old grant for up to a day. The minter asks instead
(:class:`~threetears.nats.CredentialRenewalRequest` on the principal's inbox), and a client armed
with :meth:`~threetears.nats.NatsClient.renew_on_request` moves to a successor admitted with the
grant as it stands now -- make-before-break, so a request in flight across the move still completes.

Against a real nats-server (2.14.2) with config-mode ``auth_callout`` and the real
:class:`~threetears.nats.AuthCalloutResponder`.

Gated on docker: a checkout without docker skips cleanly.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path

import pytest

from threetears.core.testing.containers import check_docker_available
from threetears.nats import (
    AuthCalloutRequest,
    AuthCalloutResponder,
    CredentialRenewalReason,
    CredentialRenewalRequest,
    GrantPolicy,
    IncomingMessage,
    NatsClient,
    PrincipalPermissions,
    PrincipalResolver,
    RefusedPrincipal,
    ResolvedPrincipal,
    Subject,
    Subjects,
    account_public_key,
    generate_account_seed,
)
from threetears.nats.subjects import get_default_namespace, set_default_namespace

pytestmark = pytest.mark.integration

_IMAGE = "nats:2.14.2-alpine"
_NS = "renewlive"
_TOKEN = "renew-live-identity"  # noqa: S105 - the test resolver's only admissible credential
_ADMIN_PW = "admin-pw"  # noqa: S105 - ephemeral testcontainer credential
_INBOX = "_INBOX_renew_live"
_POD = "pod-a"


class _Resolver(PrincipalResolver):
    """admits the test credential."""

    async def resolve(self, request: AuthCalloutRequest) -> ResolvedPrincipal | RefusedPrincipal | None:
        """admit the test credential.

        :param request: the decoded authorization request
        :ptype request: AuthCalloutRequest
        :return: the principal, or ``None`` for a foreign credential
        :rtype: ResolvedPrincipal | RefusedPrincipal | None
        """
        return ResolvedPrincipal(conn_id="renew-live", name="renew-live") if request.bootstrap_token == _TOKEN else None


class _ChangingGrant(GrantPolicy):
    """the pod's grant: its inbox, one service subject, and a feed only once the test grants it."""

    def __init__(self) -> None:
        """start without the feed.

        :return: nothing
        :rtype: None
        """
        self.feed_granted = False

    def permissions(self, principal: ResolvedPrincipal) -> PrincipalPermissions:
        """the pod's allow-list as it stands now.

        :param principal: the admitted principal
        :ptype principal: ResolvedPrincipal
        :return: the allow-list minted into the pod's user JWT
        :rtype: PrincipalPermissions
        """
        subscribe = (f"{_INBOX}.>", "feed") if self.feed_granted else (f"{_INBOX}.>",)
        return PrincipalPermissions(
            publish=("svc.slow",), subscribe=subscribe, allow_responses=True, inbox_prefix=_INBOX
        )


@contextlib.contextmanager
def _nats(config_text: str, conf_dir: Path) -> Iterator[str]:
    """start a nats-server with ``config_text``; yield its URI.

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
        nats_server(_IMAGE)
        .with_volume_mapping(str(conf_dir), "/etc/nats", "ro")
        .with_command(["-c", "/etc/nats/nats.conf"])
    )
    container.start()
    try:
        yield nats_server_uri(container)
    finally:
        container.stop()


async def test_a_changed_grant_reaches_the_live_connection_without_losing_a_request(tmp_path: Path) -> None:
    if not check_docker_available():
        pytest.skip("Docker not available")

    account_seed = generate_account_seed()
    config = (
        "port: 4222\n"
        "authorization {\n"
        "  timeout: 2\n"
        f'  auth_callout {{ issuer: "{account_public_key(account_seed)}", auth_users: [ admin ] }}\n'
        f'  users: [ {{ user: admin, password: "{_ADMIN_PW}" }} ]\n'
        "}\n"
    )
    previous_ns = get_default_namespace()
    with _nats(config, tmp_path) as uri:
        admin = await NatsClient.connect(
            nats_url=uri,
            nats_subject_namespace=_NS,
            client_name="renew-admin",
            user="admin",
            password=_ADMIN_PW,
            verify_jetstream=False,
        )
        policy = _ChangingGrant()
        responder = AuthCalloutResponder(
            nc=admin, account_seed=account_seed, resolver=_Resolver(), policy=policy, account_name="$G"
        )
        await responder.start()
        release = asyncio.Event()

        async def _slow(msg: IncomingMessage) -> None:
            await release.wait()
            assert msg.reply_subject is not None
            await admin.publish_raw_reply(reply_subject=msg.reply_subject, payload=b"slow answer")

        await admin.subscribe(Subject.raw("svc.slow"), cb=_slow)
        await admin.flush()
        pod = await NatsClient.connect(
            nats_url=uri,
            nats_subject_namespace=_NS,
            client_name="renew-pod",
            auth_token=lambda: _TOKEN,
            inbox_prefix=_INBOX,
            verify_jetstream=False,
            startup_timeout=timedelta(seconds=15),
        )
        try:
            await pod.renew_on_request(inbox_prefix=_INBOX, is_mine=lambda request: request.pod_id in (None, _POD))
            before = pod.raw
            # a request in flight on the connection about to be replaced
            in_flight = asyncio.create_task(
                pod.request_raw(subject=Subject.raw("svc.slow"), payload=b"q", timeout=timedelta(seconds=20))
            )
            await asyncio.sleep(0.2)

            policy.feed_granted = True
            await admin.publish(
                subject=Subjects.credential_renewal_request(_INBOX),
                message=CredentialRenewalRequest(reason=CredentialRenewalReason.GRANTS_CHANGED, pod_id=_POD),
            )
            for _ in range(100):
                if pod.raw is not before:
                    break
                await asyncio.sleep(0.05)
            assert pod.raw is not before, "the pod did not renew on request"

            received: list[bytes] = []

            async def _on_feed(msg: IncomingMessage) -> None:
                received.append(bytes(msg.data))

            await pod.subscribe(Subject.raw("feed"), cb=_on_feed)
            await pod.flush()
            await admin.publish_raw(subject=Subject.raw("feed"), payload=b"granted now")
            release.set()
            answer = await in_flight
            for _ in range(50):
                if received:
                    break
                await asyncio.sleep(0.05)

            assert answer == b"slow answer", "the request in flight across the renewal was lost"
            assert received == [b"granted now"], "the new grant did not reach the live pod"
        finally:
            await pod.shutdown(drain_timeout=timedelta(seconds=2))
            await admin.shutdown(drain_timeout=timedelta(seconds=2))
            set_default_namespace(previous_ns)
