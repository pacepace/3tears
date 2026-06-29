"""Integration test: the scoped JetStream grant works AND fails closed against a live broker.

The fail-closed-isolation fix (replacing the bare ``$JS.API.>`` control-plane grant with a per-stream
allow-list, :func:`threetears.nats.user_jwt._js_api_grants_for_stream`) is only trustworthy if the
grant strings behave on a REAL nats-server's subject matcher the way the unit tests assert they do.
The classic failure of an over-tight JS-API allow-list is a SILENT timeout: the KV op publishes a
request to a ``$JS.API`` subject the connection lacks, the server drops it, and the op hangs to its
deadline. So this proves BOTH directions against a live JetStream broker:

- the EXACT pub/sub allow-list :func:`mint_user_jwt` produces for a principal that declares one KV
  bucket is applied as that principal's static ``authorization`` permissions, and under it a full KV
  round-trip on the GRANTED bucket genuinely succeeds (bind, put, get, create, delete, status,
  account_info) -- no silent timeout;
- the same connection is DENIED the cross-tenant control subjects a bare ``$JS.API.>`` once exposed:
  ``$JS.API.STREAM.INFO.KV_<other>`` and ``$JS.API.STREAM.MSG.GET.KV_<other>`` (direct-read of another
  principal's backing stream) raise instead of returning data, and the server emits a Permissions
  Violation naming the foreign subject.

We do not stand up the auth-callout responder here: the grant STRINGS are what the fix changes, so we
apply them directly as config-mode ``authorization`` permissions (the same allow-list the responder
would mint) and connect with them -- a faithful, hermetic proof of the credential the server enforces.

Gated on docker: a checkout without docker skips cleanly.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import nats
import nats.errors
import nats.js.errors
import pytest

from threetears.core.testing.containers import check_docker_available
from threetears.nats.subject_permissions import PrincipalPermissions
from threetears.nats.user_jwt import generate_account_seed, mint_user_jwt

pytestmark = pytest.mark.integration

_GRANTED = "granted"  # the bucket the scoped principal declares -> stream KV_granted
_OTHER = "other"  # a peer bucket the scoped principal must NOT be able to touch -> KV_other
_INBOX = "_INBOX_scoped_jwt_test"
_ADMIN_PW = "admin-pw"  # noqa: S105 - ephemeral testcontainer credential
_SCOPED_PW = "scoped-pw"  # noqa: S105 - ephemeral testcontainer credential


def _scoped_perms() -> PrincipalPermissions:
    """a principal declaring exactly one KV bucket (``granted``) and no extra app subjects."""
    return PrincipalPermissions(
        publish=(),
        subscribe=(f"{_INBOX}.>",),
        allow_responses=True,
        inbox_prefix=_INBOX,
        kv_buckets=(_GRANTED,),
    )


def _minted_permissions() -> tuple[list[str], list[str]]:
    """mint a real user JWT for :func:`_scoped_perms` and return its (pub allow, sub allow) lists.

    these are the literal grant strings the auth-callout responder would mint; feeding them straight
    into the server's static ``authorization`` proves the MINTED credential -- not a hand-typed copy.
    """
    token = mint_user_jwt(
        account_seed=generate_account_seed(),
        user_public_key="UTEST",  # sub is irrelevant for the static-permissions projection
        permissions=_scoped_perms(),
        name="scoped-jwt-test",
        expires_in_seconds=600,
    )
    payload_seg = token.split(".")[1]
    payload = json.loads(base64.urlsafe_b64decode(payload_seg + "=" * (-len(payload_seg) % 4)))
    nats_claim = payload["nats"]
    return nats_claim["pub"]["allow"], nats_claim["sub"]["allow"]


def _server_config(pub_allow: list[str], sub_allow: list[str]) -> str:
    """a nats-server config: JetStream on, an admin (full) + the scoped user (minted allow-list)."""
    authorization = {
        "users": [
            {
                "user": "admin",
                "password": _ADMIN_PW,
                "permissions": {"publish": ">", "subscribe": ">", "allow_responses": True},
            },
            {
                "user": "scoped",
                "password": _SCOPED_PW,
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
async def _connect(uri: str, *, user: str, password: str, errors: list[str] | None = None) -> AsyncIterator:
    """connect a raw nats client; route async permission-violation errors into ``errors``."""

    async def _err_cb(exc: Exception) -> None:
        if errors is not None:
            errors.append(str(exc))

    nc = await nats.connect(
        uri,
        user=user,
        password=password,
        inbox_prefix=_INBOX.encode() if user == "scoped" else b"_INBOX",
        error_cb=_err_cb,
        max_reconnect_attempts=0,
    )
    try:
        yield nc
    finally:
        await nc.close()


async def test_scoped_grant_allows_own_bucket_and_denies_cross_bucket(tmp_path: Path) -> None:
    if not check_docker_available():
        pytest.skip("Docker not available")

    pub_allow, sub_allow = _minted_permissions()
    # sanity: the credential we are about to enforce is the scoped one, not the old broad hole.
    assert "$JS.API.>" not in pub_allow
    assert f"$KV.{_GRANTED}.>" in pub_allow

    with _nats_with_auth(_server_config(pub_allow, sub_allow), tmp_path) as uri:
        # --- admin pre-creates BOTH buckets and seeds the peer bucket with a secret value ---
        async with _connect(uri, user="admin", password=_ADMIN_PW) as admin_nc:
            admin_js = admin_nc.jetstream()
            await admin_js.create_key_value(bucket=_GRANTED)
            other_kv = await admin_js.create_key_value(bucket=_OTHER)
            await other_kv.put("peer_secret", b"do-not-leak")

        # --- scoped principal: full KV round-trip on the GRANTED bucket must genuinely work ---
        scoped_errors: list[str] = []
        async with _connect(uri, user="scoped", password=_SCOPED_PW, errors=scoped_errors) as nc:
            js = nc.jetstream(timeout=8)

            # account-level reachability probe (NatsClient.connect / KV-cache ping use this)
            await js.account_info()

            kv = await js.key_value(_GRANTED)  # bind: $JS.API.STREAM.INFO.KV_granted
            rev = await kv.put("k1", b"v1")  # publish $KV.granted.k1 + PubAck
            assert rev > 0
            entry = await kv.get("k1")  # DIRECT.GET / STREAM.MSG.GET on KV_granted
            assert entry.value == b"v1"
            rev2 = await kv.create("k2", b"v2")  # create == update w/ expected-seq publish
            assert rev2 > 0
            assert (await kv.get("k2")).value == b"v2"
            assert await kv.delete("k1") is True
            with pytest.raises(nats.js.errors.KeyNotFoundError):
                await kv.get("k1")
            status = await kv.status()  # status -> stream_info(KV_granted)
            assert status.bucket == _GRANTED

            # the granted ops must NOT have produced any permission violation (nats-py lowercases
            # the server's "-ERR Permissions Violation" frame, so compare case-insensitively)
            assert not any("permissions violation" in e.lower() for e in scoped_errors), scoped_errors

            # --- cross-bucket: reading the PEER bucket's backing stream must be DENIED ---
            before = len(scoped_errors)
            with pytest.raises((nats.errors.TimeoutError, nats.errors.NoRespondersError)):
                await js.stream_info(f"KV_{_OTHER}")  # $JS.API.STREAM.INFO.KV_other
            with pytest.raises((nats.errors.TimeoutError, nats.errors.NoRespondersError)):
                # direct-read the peer's secret value: $JS.API.STREAM.MSG.GET.KV_other
                # (get_msg is inherited from JetStreamManager onto the public JetStreamContext)
                await js.get_msg(f"KV_{_OTHER}", subject=f"$KV.{_OTHER}.peer_secret", direct=False)

            await asyncio.sleep(0.3)  # let the async -ERR frames land in the error callback
            new_errors = scoped_errors[before:]
            violations = [e for e in new_errors if "permissions violation" in e.lower()]
            assert violations, f"expected a permissions violation, got: {new_errors}"
            # the violation must name the FOREIGN stream's control subject -- proof it was the grant,
            # not some unrelated failure, that blocked the cross-bucket read (nats-py lowercases it).
            assert any(f"kv_{_OTHER}".lower() in e.lower() for e in violations), violations

            # the connection is still usable for its OWN bucket after the denied cross-bucket op
            assert (await kv.get("k2")).value == b"v2"
