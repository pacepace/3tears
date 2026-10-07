"""Integration test: a connection learns its credential's lifetime from the server, and renews on it.

A tool pod gets no hub handshake, so it used to take its credential's lifetime from its own
environment (:func:`threetears.nats.credential_renewal.nats_user_jwt_ttl_seconds`). When that guess
is LONGER than what the auth-callout actually minted, the renewal is scheduled after the expiry: the
server ends the connection with an authorization error, nats-py treats that as fatal rather than
reconnecting, and the pod crash-loops. That happened on cobalt-dev when a hub minting 300s served a
pod whose default said a day.

The server that ends the connection is the one authority on when it will: it answers a
connection's ``$SYS.REQ.USER.INFO`` with that connection's own remaining lifetime. This proves,
against a real nats-server running config-mode ``auth_callout`` and the real
:class:`~threetears.nats.AuthCalloutResponder`, that:

- :meth:`~threetears.nats.NatsClient.credential_ttl_from_server` reports the lifetime the responder
  minted, not the one the pod was configured with;
- a renewal loop asked to consult the server renews before the MINTED expiry even though its
  configured lifetime is a day, so the connection is still serving after the minted lifetime has
  passed.

Gated on docker: a checkout without docker skips cleanly.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path
from typing import Any

import nats
import pytest

from threetears.core.testing.containers import check_docker_available
from threetears.nats import (
    AuthCalloutRequest,
    AuthCalloutResponder,
    GrantPolicy,
    JsResource,
    NatsClient,
    PrincipalPermissions,
    PrincipalResolver,
    ResolvedPrincipal,
    Subject,
    account_public_key,
    decode_auth_request,
    generate_account_seed,
)
from threetears.nats.subject_permissions import SERVER_USER_INFO_SUBJECT

pytestmark = pytest.mark.integration

_TOKEN = "ttl-live-identity"  # noqa: S105 - the test resolver's only admissible credential
_ADMIN_PW = "admin-pw"  # noqa: S105 - ephemeral testcontainer credential
_INBOX = "_INBOX_ttl_live"
_ECHO = "ttl.echo"
_NS = "ttllive"
_BUCKET = f"{_NS}-cfg"
_KEY = "k"
#: what the responder mints. Short, so the test outlives it.
_MINTED_TTL_SECONDS = 40
#: what the pod is configured to believe: the platform default that bit cobalt-dev.
_CONFIGURED_TTL_SECONDS = 86_400
#: the longest request the pod declares. Small, so the renewal fires well inside the minted TTL.
_LONGEST_REQUEST_SECONDS = 2.0


class _Resolver(PrincipalResolver):
    """admits the one test credential, counting authorizations."""

    def __init__(self) -> None:
        """start the count.

        :return: nothing
        :rtype: None
        """
        self.calls = 0

    async def resolve(self, request: AuthCalloutRequest) -> ResolvedPrincipal | None:
        """admit the test credential.

        :param request: the decoded authorization request
        :ptype request: AuthCalloutRequest
        :return: the pod principal, or ``None`` for any other credential
        :rtype: ResolvedPrincipal | None
        """
        if request.bootstrap_token != _TOKEN:
            return None
        self.calls += 1
        return ResolvedPrincipal(conn_id="ttl-live", name="ttl-live")


class _PodGrant(GrantPolicy):
    """the pod's grant: its inbox, one echo subject, and the server's user-info request."""

    def permissions(self, principal: ResolvedPrincipal) -> PrincipalPermissions:
        """the pod's allow-list.

        :param principal: the admitted principal
        :ptype principal: ResolvedPrincipal
        :return: the allow-list minted into the pod's user JWT
        :rtype: PrincipalPermissions
        """
        return PrincipalPermissions(
            publish=(_ECHO, SERVER_USER_INFO_SUBJECT),
            subscribe=(f"{_INBOX}.>",),
            allow_responses=False,
            inbox_prefix=_INBOX,
            # a JetStream resource, because the client confirms JetStream at connect.
            js_resources=(JsResource.kv_key_read(_BUCKET, key=_KEY),),
        )


def _server_config(account_pub: str) -> str:
    """a nats-server that delegates every connection but the admin's to the callout.

    :param account_pub: the account public key that signs callout responses
    :ptype account_pub: str
    :return: the nats-server configuration text
    :rtype: str
    """
    admin = {"user": "admin", "password": _ADMIN_PW}
    return (
        "port: 4222\n"
        "jetstream { store_dir: /tmp/js-store }\n"
        "authorization {\n"
        "  timeout: 10\n"
        "  auth_callout {\n"
        f'    issuer: "{account_pub}"\n'
        "    auth_users: [ admin ]\n"
        "  }\n"
        f"  users: [ {json.dumps(admin)} ]\n"
        "}\n"
    )


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


async def test_the_server_reports_the_minted_lifetime_and_the_renewal_runs_on_it(tmp_path: Path) -> None:
    """the pod believes a day, the hub mints 40s; the pod learns 40s and is still serving after it."""
    if not check_docker_available():
        pytest.skip("Docker not available")

    account_seed = generate_account_seed()
    resolver = _Resolver()
    responder = AuthCalloutResponder(
        nc=None,  # driven through build_response from the admin connection below
        account_seed=account_seed,
        resolver=resolver,
        policy=_PodGrant(),
        account_name="$G",
        user_jwt_ttl_seconds=_MINTED_TTL_SECONDS,
    )

    with _nats_with_config(_server_config(account_public_key(account_seed)), tmp_path) as uri:
        admin = await nats.connect(uri, user="admin", password=_ADMIN_PW, max_reconnect_attempts=0)
        pod: NatsClient | None = None
        try:

            async def _on_auth(msg: Any) -> None:
                request = decode_auth_request(bytes(msg.data).decode())
                await msg.respond((await responder.build_response(request)).encode())

            await admin.subscribe("$SYS.REQ.USER.AUTH", cb=_on_auth)

            async def _on_echo(msg: Any) -> None:
                await msg.respond(bytes(msg.data))

            await admin.subscribe(_ECHO, cb=_on_echo)
            await admin.flush()
            await admin.jetstream().create_key_value(bucket=_BUCKET, direct=True)

            pod = await NatsClient.connect(
                nats_url=uri,
                nats_subject_namespace=_NS,
                client_name="ttl-live",
                auth_token=lambda: _TOKEN,
                inbox_prefix=_INBOX,
                startup_timeout=timedelta(seconds=15),
            )

            learned = await pod.credential_ttl_from_server()
            assert learned is not None, "the server reported no lifetime for an expiring credential"
            # the minted lifetime, never the configured one; a second's slack for the round trip.
            assert _MINTED_TTL_SECONDS - 2 <= learned <= _MINTED_TTL_SECONDS, learned

            pod.renew_credential(
                ttl_seconds=lambda: _CONFIGURED_TTL_SECONDS,
                longest_request_seconds=_LONGEST_REQUEST_SECONDS,
                ask_server=True,
            )

            # past the minted expiry by a clear margin: without a renewal the server has ended the
            # connection by now, and nats-py does not come back from an expired authorization.
            await asyncio.sleep(_MINTED_TTL_SECONDS + 10)

            reply = await pod.request_raw(subject=Subject.raw(_ECHO), payload=b"still-here")
            assert reply == b"still-here"
            assert resolver.calls >= 2, "the connection outlived its minted lifetime without renewing"
        finally:
            if pod is not None:
                await pod.shutdown()
            await admin.close()
