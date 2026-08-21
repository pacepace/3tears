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
from threetears.nats.subject_permissions import (
    CROSS_PLATFORM_CACHE_INVALIDATE,
    JsResource,
    Principal,
    PrincipalPermissions,
    build_permissions,
    inbox_prefix_for,
    kv_key_scope_for,
)
from threetears.nats.subjects import get_default_namespace, set_default_namespace
from threetears.nats.user_jwt import generate_account_seed, js_api_grants_for_stream, mint_user_jwt

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
async def _connect(
    uri: str,
    *,
    user: str,
    password: str,
    errors: list[str] | None = None,
    inbox_prefix: str | None = None,
) -> AsyncIterator:
    """connect a raw nats client; route async permission-violation errors into ``errors``.

    ``inbox_prefix`` must be the prefix the principal's own subscribe allow-list covers. A minted
    pod credential carries only ``_INBOX_{principal}_{conn}.>``, so a client left on the global
    ``_INBOX`` cannot subscribe its own reply inbox and every request fails for a reason that has
    nothing to do with the grant under test.
    """

    async def _err_cb(exc: Exception) -> None:
        if errors is not None:
            errors.append(str(exc))

    if inbox_prefix is not None:
        prefix = inbox_prefix.encode()
    else:
        prefix = b"_INBOX" if user == "admin" else _INBOX.encode()
    nc = await nats.connect(
        uri,
        user=user,
        password=password,
        inbox_prefix=prefix,
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


# ---------------------------------------------------------------------------
# coll-task-07c TP-07: a tool pod's keys are refused to every OTHER principal
# ---------------------------------------------------------------------------
#
# The shard's whole security claim, run rather than read off the grant strings. Three credentials
# against ONE shared bucket:
#
#   toolpod A -- the credential the auth callout MINTS for ``build_permissions(TOOL_POD, ...)``,
#                which is what this shard changed;
#   toolpod B -- a SECOND minted tool pod, a different ``tool_pods.id``, therefore a different key
#                scope. this is the peer the shard exists to keep out;
#   tool_server -- the STATIC NATS user, whose shape ``coll-task-05b`` fixed in the hub's conf:
#                a coarse ``$KV.>`` / ``$JS.>`` publish allow plus a GENERATED deny that closes the
#                collections bucket and its backing stream. it holds no keys of its own there, so
#                the deny cannot strand its own data -- and it is the credential every tool pod's
#                own compose/k8s manifest still provisions.
#
# Both the minted peer AND the static user must be refused, because they fail differently: a minted
# peer is stopped by an ALLOW-list that does not name A's scope, and the static user is stopped by a
# DENY that NATS evaluates after a ``$KV.>`` allow which would otherwise cover everything.

_TP_NS = "tpprobe"
_TP_BUCKET = f"{_TP_NS}-collections"
_TP_STREAM = f"KV_{_TP_BUCKET}"
_TP_POD_A = "01947100-0000-7000-8000-0000000000aa"
_TP_POD_B = "01947100-0000-7000-8000-0000000000bb"
_TP_A_PW = "toolpod-a-pw"  # noqa: S105 - ephemeral testcontainer credential
_TP_B_PW = "toolpod-b-pw"  # noqa: S105 - ephemeral testcontainer credential
_TP_STATIC_PW = "tool-server-pw"  # noqa: S105 - ephemeral testcontainer credential
#: a SECOND bucket the static user is not denied, so "everything was refused" cannot be mistaken for
#: a broken credential. Named like the bucket a tool pod's ``KVLease`` display claim really uses.
_TP_OTHER_BUCKET = f"{_TP_NS}-leases"


def _tool_pod_allow_lists(pod_id: str) -> tuple[list[str], list[str]]:
    """the minted (pub, sub) allow-lists for one tool pod, straight from the resolver.

    :param pod_id: the pod's ``tool_pods.id``
    :ptype pod_id: str
    :return: the JWT's publish allow-list and subscribe allow-list
    :rtype: tuple[list[str], list[str]]
    """
    return _minted_allow_lists(build_permissions(Principal.TOOL_POD, pod_id=pod_id))


def _static_tool_server_grants() -> tuple[list[str], list[str], list[str]]:
    """the ``tool_server`` static user's shape: coarse allow, generated collections deny.

    Reproduces what ``aibots.hub.security.static_nats_grants`` renders into ``nats.conf`` for a user
    that holds NO keys in the shared bucket. The deny set is GENERATED from
    :func:`threetears.nats.user_jwt.js_api_grants_for_stream` rather than hand-typed, for the reason
    that function's own docstring gives: hand-deriving these shapes has failed twice, once on
    whole-token wildcards and once by omitting ``$JS.API.STREAM.MSG.*.{stream}``, the six-token
    terminal form that IS the body-carried read.

    :return: the publish allow, the subscribe allow, and the publish deny
    :rtype: tuple[list[str], list[str], list[str]]
    """
    app = [f"{_TP_NS}.>", "_INBOX.>", CROSS_PLATFORM_CACHE_INVALIDATE]
    publish = [f"{_TP_NS}.>", "$JS.>", "$KV.>", "_INBOX.>", CROSS_PLATFORM_CACHE_INVALIDATE]
    deny = [f"$KV.{_TP_BUCKET}.>", *js_api_grants_for_stream(_TP_STREAM)]
    return publish, app, deny


def _tool_pod_probe_config() -> str:
    """a nats-server config carrying admin + the two minted pods + the static user.

    :return: the nats-server configuration text
    :rtype: str
    """
    a_pub, a_sub = _tool_pod_allow_lists(_TP_POD_A)
    b_pub, b_sub = _tool_pod_allow_lists(_TP_POD_B)
    s_pub, s_sub, s_deny = _static_tool_server_grants()
    authorization = {
        "users": [
            {
                "user": "admin",
                "password": _ADMIN_PW,
                "permissions": {"publish": ">", "subscribe": ">", "allow_responses": True},
            },
            {
                "user": "toolpod_a",
                "password": _TP_A_PW,
                "permissions": {
                    "publish": {"allow": a_pub},
                    "subscribe": {"allow": a_sub},
                    "allow_responses": True,
                },
            },
            {
                "user": "toolpod_b",
                "password": _TP_B_PW,
                "permissions": {
                    "publish": {"allow": b_pub},
                    "subscribe": {"allow": b_sub},
                    "allow_responses": True,
                },
            },
            {
                "user": "tool_server",
                "password": _TP_STATIC_PW,
                "permissions": {
                    "publish": {"allow": s_pub, "deny": s_deny},
                    "subscribe": {"allow": s_sub},
                    "allow_responses": True,
                },
            },
        ]
    }
    return f"jetstream {{ store_dir: /tmp/js-store }}\nport: 4222\nauthorization {json.dumps(authorization)}\n"


async def _violations_naming(nc_errors: list[str], since: int, needles: tuple[str, ...]) -> list[str]:
    """let the async ``-ERR`` frames land, then return the permission violations they carried.

    A refused JetStream request is never answered: the server drops the publish and emits an
    asynchronous ``-ERR`` the caller is not reading, so the only synchronous evidence is the
    caller's own deadline -- which is indistinguishable from an unreachable broker or a bucket
    nobody created. Asserting the violation NAMES the subject is what tells those apart.

    :param nc_errors: the connection's collected async error strings
    :ptype nc_errors: list[str]
    :param since: index into ``nc_errors`` marking the start of this probe run
    :ptype since: int
    :param needles: lowercase subject fragments each of which must appear in some violation
    :ptype needles: tuple[str, ...]
    :return: the permission violations recorded since ``since``
    :rtype: list[str]
    :raises AssertionError: if any needle names no violation
    """
    await asyncio.sleep(0.4)
    violations = [e for e in nc_errors[since:] if "permissions violation" in e.lower()]
    for needle in needles:
        assert any(needle in e.lower() for e in violations), (needle, violations)
    return violations


async def test_a_tool_pods_keys_are_refused_to_every_other_principal(tmp_path: Path) -> None:
    """TP-07, live: the tool pod caches, and it caches SAFELY.

    Both halves, and the second is not optional: a grant that refused everything would pass the
    refusal half on its own and brick every tool pod on the platform. So principal A's own scope
    must genuinely read and write, and the static user must genuinely reach a bucket it is not
    denied, before either refusal means anything.
    """
    if not check_docker_available():
        pytest.skip("Docker not available")

    previous_namespace = get_default_namespace()
    set_default_namespace(_TP_NS)
    try:
        a_scope = kv_key_scope_for(Principal.TOOL_POD, pod_id=_TP_POD_A)
        b_scope = kv_key_scope_for(Principal.TOOL_POD, pod_id=_TP_POD_B)
        a_key = f"{a_scope}.widgets.e1"
        b_key = f"{b_scope}.widgets.e1"
        a_pub, _ = _tool_pod_allow_lists(_TP_POD_A)
        config = _tool_pod_probe_config()
    finally:
        set_default_namespace(previous_namespace)

    # sanity on the credential itself, before the broker ever sees it
    assert a_scope != b_scope
    assert f"$KV.{_TP_BUCKET}.{a_scope}.>" in a_pub
    assert f"$KV.{_TP_BUCKET}.>" not in a_pub
    assert f"$JS.API.DIRECT.GET.{_TP_STREAM}.$KV.{_TP_BUCKET}.{a_scope}.>" in a_pub
    assert f"$JS.API.STREAM.CREATE.{_TP_STREAM}" not in a_pub, "a pod must never hold the declare verbs"
    assert f"$JS.API.STREAM.UPDATE.{_TP_STREAM}" not in a_pub, "UPDATE is a read-all primitive here"

    a_inbox = inbox_prefix_for(Principal.TOOL_POD, conn_id=_TP_POD_A)
    b_inbox = inbox_prefix_for(Principal.TOOL_POD, conn_id=_TP_POD_B)

    with _nats_with_auth(config, tmp_path) as uri:
        # --- admin declares the shared bucket with allow_direct and seeds BOTH pods' keys --------
        async with _connect(uri, user="admin", password=_ADMIN_PW) as admin_nc:
            admin_js = admin_nc.jetstream()
            shared = await admin_js.create_key_value(bucket=_TP_BUCKET, direct=True)
            await shared.put(a_key, b"pod-a-owns-this")
            await shared.put(b_key, b"pod-b-owns-this")
            await admin_js.create_key_value(bucket=_TP_OTHER_BUCKET)
            info = await admin_js.stream_info(_TP_STREAM)
            # THE PREMISE, asserted first. With allow_direct false, nats-py reads a key by putting
            # it in the BODY of $JS.API.STREAM.MSG.GET, where no subject permission can see it, and
            # every refusal below would be vacuous.
            assert info.config.allow_direct is True

        # === TOOL POD A: its own scope genuinely works ==========================================
        a_errors: list[str] = []
        async with _connect(uri, user="toolpod_a", password=_TP_A_PW, errors=a_errors, inbox_prefix=a_inbox) as a_nc:
            a_js = a_nc.jetstream(timeout=4)
            await a_js.account_info()
            a_kv = await a_js.key_value(_TP_BUCKET)  # bind: $JS.API.STREAM.INFO
            assert (await a_kv.get(a_key)).value == b"pod-a-owns-this"
            assert await a_kv.put(f"{a_scope}.widgets.e2", b"written-by-a") > 0
            assert (await a_kv.get(f"{a_scope}.widgets.e2")).value == b"written-by-a"
            assert not [e for e in a_errors if "permissions violation" in e.lower()], a_errors

        # === TOOL POD B: a MINTED peer, refused on every route to A's key =======================
        b_errors: list[str] = []
        async with _connect(uri, user="toolpod_b", password=_TP_B_PW, errors=b_errors, inbox_prefix=b_inbox) as b_nc:
            b_js = b_nc.jetstream(timeout=4)
            await b_js.account_info()
            b_kv = await b_js.key_value(_TP_BUCKET)
            # its OWN key works -- so a refusal below is the SCOPE, not a broken credential
            assert (await b_kv.get(b_key)).value == b"pod-b-owns-this"

            before = len(b_errors)
            # 1. the direct read, on the literal subject nats-py builds for A's key
            await _refused(b_nc, f"$JS.API.DIRECT.GET.{_TP_STREAM}.$KV.{_TP_BUCKET}.{a_key}")
            # 2. the body-carried read: the key never appears in the subject at all
            await _refused(
                b_nc,
                f"$JS.API.STREAM.MSG.GET.{_TP_STREAM}",
                body=json.dumps({"last_by_subj": f"$KV.{_TP_BUCKET}.{a_key}"}).encode(),
            )
            # 3. a consumer filtered on the WHOLE bucket, delivered to B's own inbox
            await _refused(
                b_nc,
                f"$JS.API.CONSUMER.CREATE.{_TP_STREAM}",
                body=json.dumps(
                    {
                        "stream_name": _TP_STREAM,
                        "config": {
                            "filter_subject": f"$KV.{_TP_BUCKET}.>",
                            "deliver_subject": f"{b_inbox}.spy",
                        },
                    }
                ).encode(),
            )
            # 4-6. the verb sits at token 4, so a ``STREAM.*`` wildcard would have covered all three
            await _refused(b_nc, f"$JS.API.STREAM.PURGE.{_TP_STREAM}")
            await _refused(b_nc, f"$JS.API.STREAM.SNAPSHOT.{_TP_STREAM}")
            await _refused(b_nc, f"$JS.API.STREAM.UPDATE.{_TP_STREAM}")
            # 7. the data plane: overwriting A's key
            await _refused(b_nc, f"$KV.{_TP_BUCKET}.{a_key}", body=b"overwritten-by-b")

            peer_violations = await _violations_naming(
                b_errors,
                before,
                (
                    f"$js.api.direct.get.{_TP_STREAM}".lower(),
                    f"$js.api.stream.msg.get.{_TP_STREAM}".lower(),
                    f"$js.api.consumer.create.{_TP_STREAM}".lower(),
                    f"$js.api.stream.purge.{_TP_STREAM}".lower(),
                    f"$js.api.stream.snapshot.{_TP_STREAM}".lower(),
                    f"$js.api.stream.update.{_TP_STREAM}".lower(),
                    f"$kv.{_TP_BUCKET}.{a_scope}".lower(),
                ),
            )
            assert len(peer_violations) >= 7, peer_violations
            # a permissions violation does not close the connection -- which is exactly why the
            # failure is so quiet -- so B must be unharmed by having been refused seven times.
            assert (await b_kv.get(b_key)).value == b"pod-b-owns-this"

        # === THE STATIC USER: refused by a DENY layered under a coarse $KV.> allow ===============
        s_errors: list[str] = []
        async with _connect(
            uri, user="tool_server", password=_TP_STATIC_PW, errors=s_errors, inbox_prefix="_INBOX"
        ) as s_nc:
            s_js = s_nc.jetstream(timeout=4)
            await s_js.account_info()
            # NOT VACUOUS: the credential works on a bucket it is not denied. Without this, every
            # refusal below is equally explained by a connection that can do nothing at all.
            other_kv = await s_js.key_value(_TP_OTHER_BUCKET)
            assert await other_kv.put("claim", b"held") > 0
            assert not [e for e in s_errors if "permissions violation" in e.lower()], s_errors

            before = len(s_errors)
            # the deny closes the control plane, so it cannot even BIND the collections bucket
            await _refused(s_nc, f"$JS.API.STREAM.INFO.{_TP_STREAM}")
            await _refused(s_nc, f"$JS.API.DIRECT.GET.{_TP_STREAM}.$KV.{_TP_BUCKET}.{a_key}")
            await _refused(
                s_nc,
                f"$JS.API.STREAM.MSG.GET.{_TP_STREAM}",
                body=json.dumps({"last_by_subj": f"$KV.{_TP_BUCKET}.{a_key}"}).encode(),
            )
            await _refused(
                s_nc,
                f"$JS.API.CONSUMER.CREATE.{_TP_STREAM}",
                body=json.dumps(
                    {
                        "stream_name": _TP_STREAM,
                        "config": {
                            "filter_subject": f"$KV.{_TP_BUCKET}.>",
                            "deliver_subject": "_INBOX.spy",
                        },
                    }
                ).encode(),
            )
            # and the data plane, which the coarse ``$KV.>`` allow would otherwise have covered
            await _refused(s_nc, f"$KV.{_TP_BUCKET}.{a_key}", body=b"overwritten-by-static")

            static_violations = await _violations_naming(
                s_errors,
                before,
                (
                    f"$js.api.stream.info.{_TP_STREAM}".lower(),
                    f"$js.api.direct.get.{_TP_STREAM}".lower(),
                    f"$js.api.stream.msg.get.{_TP_STREAM}".lower(),
                    f"$js.api.consumer.create.{_TP_STREAM}".lower(),
                    f"$kv.{_TP_BUCKET}.{a_scope}".lower(),
                ),
            )
            assert len(static_violations) >= 5, static_violations

        # === A's data is exactly as A left it ===================================================
        async with _connect(uri, user="admin", password=_ADMIN_PW) as admin_nc:
            admin_kv = await admin_nc.jetstream().key_value(_TP_BUCKET)
            assert (await admin_kv.get(a_key)).value == b"pod-a-owns-this"
            assert (await admin_kv.get(f"{a_scope}.widgets.e2")).value == b"written-by-a"
