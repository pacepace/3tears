"""Integration test: the scoped JetStream grant works AND fails closed against a live broker.

The fail-closed-isolation fix (replacing the bare ``$JS.API.>`` control-plane grant with a per-stream
allow-list, :func:`threetears.nats.user_jwt.js_api_grants_for_stream`) is only trustworthy if the
grant strings behave on a REAL nats-server's subject matcher the way the unit tests assert they do.
The classic failure of an over-tight JS-API allow-list is a SILENT timeout: the KV op publishes a
request to a ``$JS.API`` subject the connection lacks, the server drops it, and the op hangs to its
deadline. So this proves BOTH directions against a live JetStream broker.

**Two isolation boundaries, two tests, and the second is the one that matters.**

*Between buckets.* The EXACT pub/sub allow-list :func:`mint_user_jwt` produces for a principal that
declares one KV bucket is applied as that principal's static ``authorization`` permissions; under it
a full KV round-trip on the GRANTED bucket genuinely succeeds (bind, put, get, create, delete,
status, account_info) with no silent timeout, while the control subjects of a PEER bucket
(``$JS.API.STREAM.INFO.KV_<other>``, ``$JS.API.STREAM.MSG.GET.KV_<other>``) raise instead of
returning data and the server emits a Permissions Violation naming the foreign subject.

*Between principals INSIDE one bucket.* Pinning the stream name does nothing here, because
``{ns}-collections`` is one stream held by many principals. The second test puts two key scopes in
one shared bucket and drives every route to the other principal's key from a live connection: the
direct read, the body-carried ``STREAM.MSG.GET``, a consumer filtered on the whole bucket and
delivered to the caller's own inbox, ``PURGE`` / ``SNAPSHOT`` / ``UPDATE``, and a plain ``$KV.``
overwrite. All seven must be refused, and -- the half that is easy to forget -- the principal's OWN
scope must still read and write, because a grant that refused everything would pass the refusal half
and brick the fleet.

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
from threetears.nats.subject_permissions import JsResource, PrincipalPermissions
from threetears.nats.user_jwt import generate_account_seed, mint_user_jwt

pytestmark = pytest.mark.integration

_GRANTED = "granted"  # the bucket the scoped principal declares -> stream KV_granted
_OTHER = "other"  # a peer bucket the scoped principal must NOT be able to touch -> KV_other
_INBOX = "_INBOX_scoped_jwt_test"
_ADMIN_PW = "admin-pw"  # noqa: S105 - ephemeral testcontainer credential
_SCOPED_PW = "scoped-pw"  # noqa: S105 - ephemeral testcontainer credential

#: The SHARED bucket -- one bucket, many principals, which is the case pinning the stream name
#: does nothing for. Named like the real one so the subject shapes below read as they do live.
_SHARED = "collections"
_SHARED_STREAM = f"KV_{_SHARED}"
_A_SCOPE = "agent_pod-019470a8b5c37def8123456789abcdef"
_B_SCOPE = "agent_pod-ffffffffffffffffffffffffffffffff"
_SHARED_PW = "shared-pw"  # noqa: S105 - ephemeral testcontainer credential


def _scoped_perms() -> PrincipalPermissions:
    """a principal declaring exactly one KV bucket (``granted``) and no extra app subjects."""
    return PrincipalPermissions(
        publish=(),
        subscribe=(f"{_INBOX}.>",),
        allow_responses=True,
        inbox_prefix=_INBOX,
        js_resources=(JsResource.kv(_GRANTED, scope=None, writable=True),),
    )


def _key_scoped_perms() -> PrincipalPermissions:
    """principal A: one SHARED bucket, narrowed to A's own key scope."""
    return PrincipalPermissions(
        publish=(),
        subscribe=(f"{_INBOX}.>",),
        allow_responses=True,
        inbox_prefix=_INBOX,
        js_resources=(JsResource.kv(_SHARED, scope=_A_SCOPE, writable=True),),
    )


def _minted_allow_lists(permissions: PrincipalPermissions) -> tuple[list[str], list[str]]:
    """mint a real user JWT for ``permissions`` and return its (pub allow, sub allow) lists.

    these are the literal grant strings the auth-callout responder would mint; feeding them straight
    into the server's static ``authorization`` proves the MINTED credential -- not a hand-typed copy.

    :param permissions: the resolved allow-list to mint
    :ptype permissions: PrincipalPermissions
    :return: the JWT's publish allow-list and subscribe allow-list
    :rtype: tuple[list[str], list[str]]
    """
    token = mint_user_jwt(
        account_seed=generate_account_seed(),
        user_public_key="UTEST",  # sub is irrelevant for the static-permissions projection
        permissions=permissions,
        name="scoped-jwt-test",
        expires_in_seconds=600,
    )
    payload_seg = token.split(".")[1]
    payload = json.loads(base64.urlsafe_b64decode(payload_seg + "=" * (-len(payload_seg) % 4)))
    nats_claim = payload["nats"]
    return nats_claim["pub"]["allow"], nats_claim["sub"]["allow"]


def _minted_permissions() -> tuple[list[str], list[str]]:
    """the (pub, sub) allow-lists for the bucket-scoped principal."""
    return _minted_allow_lists(_scoped_perms())


def _server_config(
    pub_allow: list[str],
    sub_allow: list[str],
    *,
    user: str = "scoped",
    password: str = _SCOPED_PW,
) -> str:
    """a nats-server config: JetStream on, an admin (full) + one scoped user (minted allow-list).

    :param pub_allow: the minted publish allow-list to enforce
    :ptype pub_allow: list[str]
    :param sub_allow: the minted subscribe allow-list to enforce
    :ptype sub_allow: list[str]
    :param user: the scoped user's name
    :ptype user: str
    :param password: the scoped user's password
    :ptype password: str
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
                "user": user,
                "password": password,
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
        inbox_prefix=b"_INBOX" if user == "admin" else _INBOX.encode(),
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


async def _refused(nc, subject: str, *, body: bytes = b"") -> None:
    """assert the server REFUSES a raw request on ``subject``.

    A refused JetStream request is not answered at all -- the server drops the publish and emits an
    asynchronous ``-ERR`` on a channel the caller is not reading -- so the only synchronous evidence
    is the caller's own deadline. That is exactly why these probes have to be RUN rather than
    reasoned about: a grant that matches nothing and a grant that is absent produce the identical
    timeout, and so does a bucket nobody created.

    :param nc: the connected nats-py client to probe from
    :ptype nc: Any
    :param subject: the subject to request on
    :ptype subject: str
    :param body: the request payload (the bypasses carry their target in the BODY)
    :ptype body: bytes
    :return: nothing
    :rtype: None
    :raises AssertionError: if the request is answered rather than refused
    """
    with pytest.raises((nats.errors.TimeoutError, nats.errors.NoRespondersError)):
        await nc.request(subject, body, timeout=2)


async def test_key_scoped_grant_refuses_every_bypass_and_admits_its_own_scope(tmp_path: Path) -> None:
    """The shard's whole claim, run against a real broker rather than read off the grant strings.

    One SHARED bucket, two key scopes. Principal A holds the minted key-scoped grant. Each of the
    seven routes to principal B's key must be REFUSED, and A's own scope must work for both read
    and write -- both halves, because a grant that refused everything would pass the refusal half
    on its own and brick the fleet.
    """
    if not check_docker_available():
        pytest.skip("Docker not available")

    pub_allow, sub_allow = _minted_allow_lists(_key_scoped_perms())
    # sanity on the credential itself before the broker ever sees it
    assert f"$KV.{_SHARED}.{_A_SCOPE}.>" in pub_allow
    assert f"$KV.{_SHARED}.>" not in pub_allow
    assert not [s for s in sub_allow if s.startswith("$KV")], sub_allow

    config = _server_config(pub_allow, sub_allow, user="scopedkey", password=_SHARED_PW)
    a_key = f"{_A_SCOPE}.widgets.e1"
    b_key = f"{_B_SCOPE}.widgets.e1"

    with _nats_with_auth(config, tmp_path) as uri:
        # --- admin creates the shared bucket with allow_direct (coll-task-04a) and seeds BOTH ---
        async with _connect(uri, user="admin", password=_ADMIN_PW) as admin_nc:
            admin_js = admin_nc.jetstream()
            shared = await admin_js.create_key_value(bucket=_SHARED, direct=True)
            await shared.put(a_key, b"mine")
            await shared.put(b_key, b"do-not-leak")
            info = await admin_js.stream_info(_SHARED_STREAM)
            # without this the read is $JS.API.STREAM.MSG.GET with the key in the BODY, where no
            # subject permission can constrain it, and the whole narrowing is inert.
            assert info.config.allow_direct is True

        errors: list[str] = []
        async with _connect(uri, user="scopedkey", password=_SHARED_PW, errors=errors) as nc:
            js = nc.jetstream(timeout=4)
            await js.account_info()
            kv = await js.key_value(_SHARED)  # bind: $JS.API.STREAM.INFO.KV_collections

            # === SUCCEEDS: A's own scope, read and write =============================
            assert (await kv.get(a_key)).value == b"mine"
            assert await kv.put(f"{_A_SCOPE}.widgets.e2", b"written") > 0
            assert (await kv.get(f"{_A_SCOPE}.widgets.e2")).value == b"written"
            assert not [e for e in errors if "permissions violation" in e.lower()], errors

            # === REFUSED: every route to B's key =====================================
            before = len(errors)

            # 1. the direct read of a peer's key, on the literal subject nats-py builds
            await _refused(nc, f"$JS.API.DIRECT.GET.{_SHARED_STREAM}.$KV.{_SHARED}.{b_key}")
            # 2. BYPASS 1 -- the body-carried read; the key never appears in the subject
            await _refused(
                nc,
                f"$JS.API.STREAM.MSG.GET.{_SHARED_STREAM}",
                body=json.dumps({"last_by_subj": f"$KV.{_SHARED}.{b_key}"}).encode(),
            )
            # 3. BYPASS 3 -- a consumer filtered on the WHOLE bucket, delivered to A's own inbox
            await _refused(
                nc,
                f"$JS.API.CONSUMER.CREATE.{_SHARED_STREAM}",
                body=json.dumps(
                    {
                        "stream_name": _SHARED_STREAM,
                        "config": {
                            "filter_subject": f"$KV.{_SHARED}.>",
                            "deliver_subject": f"{_INBOX}.spy",
                        },
                    }
                ).encode(),
            )
            # 4-6. BYPASS 4 -- the verb sits at token 4, so ``STREAM.*`` covered all of these
            await _refused(nc, f"$JS.API.STREAM.PURGE.{_SHARED_STREAM}")
            await _refused(nc, f"$JS.API.STREAM.SNAPSHOT.{_SHARED_STREAM}")
            await _refused(nc, f"$JS.API.STREAM.UPDATE.{_SHARED_STREAM}")
            # 7. the data plane: writing over a peer's key. ``kv.put`` is ``js.publish``, a request
            #    for the PubAck, so a refused publish surfaces as the same unanswered deadline.
            await _refused(nc, f"$KV.{_SHARED}.{b_key}", body=b"overwritten")

            await asyncio.sleep(0.4)  # let the async -ERR frames land in the error callback
            violations = [e for e in errors[before:] if "permissions violation" in e.lower()]
            assert len(violations) >= 7, violations
            # the refusals must name the SHARED stream / bucket -- proof it was the grant that
            # blocked them and not some unrelated failure (nats-py lowercases the frame).
            for needle in (
                f"$js.api.direct.get.{_SHARED_STREAM}".lower(),
                f"$js.api.stream.msg.get.{_SHARED_STREAM}".lower(),
                f"$js.api.consumer.create.{_SHARED_STREAM}".lower(),
                f"$js.api.stream.purge.{_SHARED_STREAM}".lower(),
                f"$js.api.stream.snapshot.{_SHARED_STREAM}".lower(),
                f"$js.api.stream.update.{_SHARED_STREAM}".lower(),
                f"$kv.{_SHARED}.{_B_SCOPE}".lower(),
            ):
                assert any(needle in e.lower() for e in violations), (needle, violations)

            # === STILL SUCCEEDS after the refusals ====================================
            # the connection stays up through a permissions violation, which is the whole reason
            # the failure is so quiet; A must be unharmed by having been refused seven times.
            assert (await kv.get(a_key)).value == b"mine"
            # ...and B's value is still there, unread and unmodified.

        async with _connect(uri, user="admin", password=_ADMIN_PW) as admin_nc:
            admin_kv = await admin_nc.jetstream().key_value(_SHARED)
            assert (await admin_kv.get(b_key)).value == b"do-not-leak"
