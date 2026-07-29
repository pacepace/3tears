"""Integration test: the owner-routed forward grants admit and refuse against a live broker.

``Subjects.forward`` shipped with no principal granting it, so every message
``threetears.scrape.operator_control`` carries is refused by auth-callout on any deployment
that enforces these grants -- and the refusal presents as a timeout, which is why it went
unnoticed. The repair is a family-scoped subject plus a per-tool grant, and a grant is only
worth what a real nats-server's subject matcher makes of it. So this proves all three
directions against a live broker, with the MINTED credential applied as config-mode
``authorization``:

- the granted family genuinely round-trips: a ``serve_owner`` on the tool pod's credential
  and a ``forward`` on the hub's meet, the owner's handler runs, and its bytes come back;
- a family the pod was NOT authorized for is refused, and the assertion is on the server's
  Permissions Violation naming that exact subject. A denied subscribe and an unreachable
  broker both present as a silent timeout, so an assertion on the absence of a reply would
  pass against a dead test harness;
- the pod can open the KV bucket its display claim actually materialises. Without it
  ``claim_session`` is handed ``lease=None``, logs, and serves the display UNCLAIMED.

The permission set is applied directly as static ``authorization`` rather than by standing up
the auth-callout responder: the grant STRINGS are what this scope changes, and this is the
credential the server enforces either way. Mirrors
``test_user_jwt_scoped_grant_live.py``, which also owns the reason this file does not use the
session-scoped ``nats_container`` fixture -- a shared container cannot carry a per-test
``authorization`` block.

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
import pytest

from threetears.core.coordination import KVLease
from threetears.core.testing.containers import check_docker_available
from threetears.nats import NatsClient, Subjects, forward, serve_owner, set_default_namespace
from threetears.nats.subject_permissions import Principal, PrincipalPermissions, build_permissions
from threetears.nats.user_jwt import generate_account_seed, mint_user_jwt

pytestmark = pytest.mark.integration

_NS = "fwdgrant"
_POD_ID = "pod-alpha"
_HUB_CONN = "hub-1"

#: the tool this pod's ``allowed_namespaces`` row authorizes it to serve.
_GRANTED_TOOL = "tools.scrape-zone_alpha.1-0-0"
#: a peer network zone's tool, registered by a DIFFERENT pod set. serving it would be a reach
#: into a firewalled zone this pod was never placed in.
_FOREIGN_TOOL = "tools.scrape-zone_beta.1-0-0"

_ADMIN_PW = "admin-pw"  # noqa: S105 - ephemeral testcontainer credential
_POD_PW = "pod-pw"  # noqa: S105 - ephemeral testcontainer credential
_HUB_PW = "hub-pw"  # noqa: S105 - ephemeral testcontainer credential


def _pod_permissions() -> PrincipalPermissions:
    """the tool pod's allow-list, scoped to the one tool it is authorized to serve."""
    return build_permissions(Principal.TOOL_POD, pod_id=_POD_ID, tool_namespaces=(_GRANTED_TOOL,))


def _hub_permissions() -> PrincipalPermissions:
    """the hub's allow-list; it originates the forward for every tool it fronts."""
    return build_permissions(Principal.HUB, conn_id=_HUB_CONN)


def _minted_allow_lists(permissions: PrincipalPermissions, name: str) -> tuple[list[str], list[str]]:
    """mint a real user JWT and return its ``(pub allow, sub allow)`` lists.

    the grant strings the auth-callout responder would mint, read back out of the signed
    claim rather than hand-copied, so the server enforces the MINTED credential.
    """
    token = mint_user_jwt(
        account_seed=generate_account_seed(),
        user_public_key="UTEST",  # sub is irrelevant for the static-permissions projection
        permissions=permissions,
        name=name,
        expires_in_seconds=600,
    )
    payload_seg = token.split(".")[1]
    payload = json.loads(base64.urlsafe_b64decode(payload_seg + "=" * (-len(payload_seg) % 4)))
    nats_claim = payload["nats"]
    return nats_claim["pub"]["allow"], nats_claim["sub"]["allow"]


def _server_config() -> str:
    """a nats-server config: JetStream on, an admin plus the two minted principals."""
    pod_pub, pod_sub = _minted_allow_lists(_pod_permissions(), "tool-pod-grant-test")
    hub_pub, hub_sub = _minted_allow_lists(_hub_permissions(), "hub-grant-test")
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
                    "publish": {"allow": pod_pub},
                    "subscribe": {"allow": pod_sub},
                    "allow_responses": True,
                },
            },
            {
                "user": "hub",
                "password": _HUB_PW,
                "permissions": {
                    "publish": {"allow": hub_pub},
                    "subscribe": {"allow": hub_sub},
                    "allow_responses": True,
                },
            },
        ]
    }
    return f"jetstream {{ store_dir: /tmp/js-store }}\nport: 4222\nauthorization {json.dumps(authorization)}\n"


@contextlib.contextmanager
def _nats_with_auth(conf_dir: Path) -> Iterator[str]:
    """start a JetStream nats-server carrying the minted ``authorization``; yield its URI."""
    from testcontainers.nats import NatsContainer  # noqa: PLC0415

    (conf_dir / "nats.conf").write_text(_server_config())
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


async def _connect_wrapped(uri: str, *, user: str, password: str, permissions: PrincipalPermissions) -> NatsClient:
    """connect the canonical wrapper on one principal's credential + its scoped inbox."""
    set_default_namespace(_NS)
    return await NatsClient.connect(
        nats_url=uri,
        nats_subject_namespace=_NS,
        client_name=f"{user}-grant-test",
        user=user,
        password=password,
        inbox_prefix=permissions.inbox_prefix,
    )


@contextlib.asynccontextmanager
async def _connect_raw(uri: str, *, user: str, password: str, errors: list[str]) -> AsyncIterator[nats.NATS]:
    """connect a raw nats-py client, routing async permission violations into ``errors``.

    raw rather than wrapped because the refusal this test turns on is an asynchronous
    ``-ERR`` frame, and an error callback is the only place the server's Permissions
    Violation surfaces at all -- a denied subscribe raises nothing locally.
    """

    async def _err_cb(exc: Exception) -> None:
        errors.append(str(exc))

    nc = await nats.connect(
        uri,
        user=user,
        password=password,
        inbox_prefix=_pod_permissions().inbox_prefix.encode(),
        error_cb=_err_cb,
        max_reconnect_attempts=0,
    )
    try:
        yield nc
    finally:
        await nc.close()


async def test_granted_family_round_trips_and_foreign_family_is_refused(tmp_path: Path) -> None:
    if not check_docker_available():
        pytest.skip("Docker not available")

    set_default_namespace(_NS)
    granted_family = Subjects.hitl_forward_family(_GRANTED_TOOL)
    foreign_family = Subjects.hitl_forward_family(_FOREIGN_TOOL)
    session_key = "session-42"

    with _nats_with_auth(tmp_path) as uri:
        # --- the granted family: the pod really serves it and the hub really reaches it ---
        pod = await _connect_wrapped(uri, user="pod", password=_POD_PW, permissions=_pod_permissions())
        hub = await _connect_wrapped(uri, user="hub", password=_HUB_PW, permissions=_hub_permissions())
        async with pod, hub:
            served: list[bytes] = []

            async def handler(payload: bytes) -> bytes:
                served.append(payload)
                return b"state:" + payload

            async with serve_owner(pod, session_key, handler, family=granted_family):
                reply = await forward(
                    hub,
                    session_key,
                    b"read_state",
                    timeout=timedelta(seconds=5),
                    family=granted_family,
                )

            # the OWNER's handler ran (the message crossed the broker under the pod's own
            # credential) and its bytes came back to the hub through the reply inbox.
            assert served == [b"read_state"]
            assert reply == b"state:read_state"

        # --- a family this pod was not authorized for: refused, and provably so ---
        violations: list[str] = []
        async with _connect_raw(uri, user="pod", password=_POD_PW, errors=violations) as raw:
            foreign_subject = Subjects.forward_scoped(foreign_family, session_key)
            granted_subject = Subjects.forward_scoped(granted_family, session_key)

            async def _noop(_msg: object) -> None:
                return None

            await raw.subscribe(foreign_subject.path, cb=_noop)
            await raw.flush()
            await asyncio.sleep(0.3)  # let the async -ERR frame land in the error callback

            # the server REFUSED it, and named the foreign subject while doing so. asserting
            # that no message arrived would pass equally against a broker that was never
            # reachable; this cannot.
            refusals = [e for e in violations if "permissions violation" in e.lower()]
            assert refusals, f"expected a permissions violation, got: {violations}"
            foreign_token = foreign_subject.path.split(".")[2]
            assert any(foreign_token in e for e in refusals), refusals

            # ...and the refusal is the FAMILY doing its job, not a broken credential: the
            # same connection subscribes its own family with nothing new refused.
            before = len(violations)
            await raw.subscribe(granted_subject.path, cb=_noop)
            await raw.flush()
            await asyncio.sleep(0.3)
            new = [e for e in violations[before:] if "permissions violation" in e.lower()]
            assert not new, new


async def test_tool_pod_can_open_the_bucket_its_display_claim_uses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not check_docker_available():
        pytest.skip("Docker not available")

    # The namespace the grants render under. KVLease no longer reads it -- its default is a bare
    # suffix and the transport applies the prefix -- so this is set for the SUBJECT builders, exactly as a
    # deployed pod does; the wrapper then layers its own namespace prefix over it.
    monkeypatch.setenv("THREETEARS_NATS_SUBJECT_NAMESPACE", _NS)

    with _nats_with_auth(tmp_path) as uri:
        pod = await _connect_wrapped(uri, user="pod", password=_POD_PW, permissions=_pod_permissions())
        async with pod:
            lease = KVLease(pod, pod_id=_POD_ID)
            # the real acquire path: create the bucket, write the claim, read it back. a grant
            # naming a bucket nothing opens would surface here as a JS timeout, which is the
            # shape the missing grant took in production.
            handle = await lease.acquire("display/session-42", ttl_seconds=30, max_wait_seconds=5)
            assert handle.holder == _POD_ID
            await handle.refresh()
            await handle.release()
