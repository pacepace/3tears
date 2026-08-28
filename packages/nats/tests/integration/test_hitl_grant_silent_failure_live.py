"""Integration test: the tool-namespace HITL grant delivers, and its absence fails SILENTLY.

The tool-pod HITL grants are minted from a pod's ``allowed_namespaces`` row
(:func:`threetears.nats.subject_permissions.build_permissions` with ``tool_namespaces=``), and every
test of them so far asserts on the JWT's grant STRINGS or on a foreign family being refused. Neither
proves the property the grant exists for: that the presence or absence of ``tool_namespaces`` is the
difference between a pod's owner-routed HITL subscription receiving messages and receiving nothing
at all, forever, with no exception anywhere.

Both halves run the SAME actions against the SAME broker on the SAME subject. The only variable is
the credential:

- WITH ``tool_namespaces``: the pod subscribes its owner-routed HITL subject and a message the hub
  publishes genuinely arrives;
- WITHOUT them: :meth:`NatsClient.subscribe` returns a live :class:`Subscription` and RAISES NOTHING,
  the connection stays up and healthy, and the message never arrives. The refusal exists only as an
  asynchronous ``-ERR`` frame, which is precisely the invisible failure
  :func:`threetears.nats.client._on_error`'s permissions-violation line was added to report -- so the
  same test asserts that line fired, naming that exact subject in its structured fields. Without that
  assertion the negative half would pass equally against a broker that was never reachable.

The minted permissions are applied as static config-mode ``authorization`` rather than by standing up
the auth-callout responder: the grant STRINGS are what the scope changes and this is the credential
the server enforces either way. The harness (docker gate, per-test ``nats.conf`` volume, minted
allow-lists read back out of the signed claim) is the one
``test_forward_grants_live.py`` / ``test_user_jwt_scoped_grant_live.py`` already use, including their
reason for not using the session-scoped ``nats_container`` fixture: a shared container cannot carry a
per-test ``authorization`` block.

Gated on docker: a checkout without docker skips cleanly.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

from threetears.core.testing.containers import check_docker_available
from threetears.nats import IncomingMessage, NatsClient, Subjects, set_default_namespace
from threetears.nats.client import _SUBJECT_CASE_LOWERCASED, _last_error_log
from threetears.nats.subject_permissions import Principal, PrincipalPermissions, build_permissions
from threetears.nats.user_jwt import generate_account_seed, mint_user_jwt

pytestmark = pytest.mark.integration

_NS = "hitlgrant"
_CLIENT_LOGGER = "threetears.nats.client"

#: the tool-name NODE the SERVING pod's ``allowed_namespaces`` row authorizes it for. a
#: NODE, never a registered tool namespace name: these grants are minted at CONNECT and a
#: tool leaf does not exist until REGISTRATION.
_OWNED_NODE = "scrape-zone_alpha"
#: the human-in-the-loop session both pods address. owner-routed on this key.
_SESSION_KEY = "session-42"

#: pod identities are UUIDs, and that is a CONTRACT rather than a convenience:
#: ``kv_key_scope_for`` derives a tool pod's L2 key scope from its id and REFUSES
#: anything that is not a uuid, because a scope built from an arbitrary display
#: name is not provably collision-free. these were slugs, so this whole module
#: raised at permission-build time and never reached the broker it exists to test.
_POD_SERVING = "01947100-0000-7000-8000-00000000fe01"
_POD_UNGRANTED = "01947100-0000-7000-8000-00000000fe02"
_HUB_CONN = "hub-1"

_ADMIN_PW = "admin-pw"  # noqa: S105 - ephemeral testcontainer credential
_SERVING_PW = "serving-pw"  # noqa: S105 - ephemeral testcontainer credential
_UNGRANTED_PW = "ungranted-pw"  # noqa: S105 - ephemeral testcontainer credential
_HUB_PW = "hub-pw"  # noqa: S105 - ephemeral testcontainer credential

#: how long the negative half waits for a message that must never come. long enough that a delivery
#: which merely raced the assertion would land inside it.
_SILENCE_WINDOW_SECONDS = 1.5


def _serving_permissions() -> PrincipalPermissions:
    """a tool pod whose row authorizes :data:`_OWNED_NODE` -- the HITL family is granted."""
    return build_permissions(Principal.TOOL_POD, pod_id=_POD_SERVING, tool_namespaces=(_OWNED_NODE,))


def _ungranted_permissions() -> PrincipalPermissions:
    """the SAME principal with no ``tool_namespaces`` at all -- no HITL family is granted.

    not a pod holding the wrong tool: a pod whose permissions were built without the argument, which
    is what a caller that forgets to pass ``allowed_namespaces`` through actually produces.
    """
    return build_permissions(Principal.TOOL_POD, pod_id=_POD_UNGRANTED)


def _hub_permissions() -> PrincipalPermissions:
    """the hub's allow-list; it originates the owner-routed message for every tool it fronts."""
    return build_permissions(Principal.HUB, conn_id=_HUB_CONN)


def _minted_allow_lists(permissions: PrincipalPermissions, name: str) -> tuple[list[str], list[str]]:
    """mint a real user JWT and return its ``(pub allow, sub allow)`` lists.

    read back out of the signed claim rather than hand-copied, so the server enforces the MINTED
    credential and a drift in the minter shows up here rather than being papered over by a literal.

    :param permissions: the principal's resolved allow-list
    :ptype permissions: PrincipalPermissions
    :param name: the JWT name, for readability in a failure
    :ptype name: str
    :return: the claim's publish-allow and subscribe-allow lists
    :rtype: tuple[list[str], list[str]]
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
    """a nats-server config: JetStream on, an admin plus the three minted principals.

    :return: the ``nats.conf`` text
    :rtype: str
    """
    serving_pub, serving_sub = _minted_allow_lists(_serving_permissions(), "hitl-serving-pod")
    ungranted_pub, ungranted_sub = _minted_allow_lists(_ungranted_permissions(), "hitl-ungranted-pod")
    hub_pub, hub_sub = _minted_allow_lists(_hub_permissions(), "hitl-hub")
    authorization = {
        "users": [
            {
                "user": "admin",
                "password": _ADMIN_PW,
                "permissions": {"publish": ">", "subscribe": ">", "allow_responses": True},
            },
            {
                "user": "serving",
                "password": _SERVING_PW,
                "permissions": {
                    "publish": {"allow": serving_pub},
                    "subscribe": {"allow": serving_sub},
                    "allow_responses": True,
                },
            },
            {
                "user": "ungranted",
                "password": _UNGRANTED_PW,
                "permissions": {
                    "publish": {"allow": ungranted_pub},
                    "subscribe": {"allow": ungranted_sub},
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
    """start a JetStream nats-server carrying the minted ``authorization``; yield its URI.

    :param conf_dir: a writable directory the container mounts read-only at ``/etc/nats``
    :ptype conf_dir: Path
    :return: an iterator yielding the broker URI
    :rtype: Iterator[str]
    """
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


async def _connect(uri: str, *, user: str, password: str, permissions: PrincipalPermissions) -> NatsClient:
    """connect the canonical wrapper on one principal's credential + its scoped inbox.

    the wrapper rather than a raw nats-py client on purpose: it is the wrapper that installs
    :func:`threetears.nats.client._on_error` as the error callback, and the negative half of this
    test turns on that callback's output.

    :param uri: the broker URI
    :ptype uri: str
    :param user: the config-mode user name
    :ptype user: str
    :param password: that user's password
    :ptype password: str
    :param permissions: the principal's allow-list, for its scoped inbox prefix
    :ptype permissions: PrincipalPermissions
    :return: a connected client
    :rtype: NatsClient
    """
    set_default_namespace(_NS)
    return await NatsClient.connect(
        nats_url=uri,
        nats_subject_namespace=_NS,
        client_name=f"{user}-hitl-grant-test",
        user=user,
        password=password,
        inbox_prefix=permissions.inbox_prefix,
    )


async def test_hitl_grant_delivers_and_its_absence_is_a_silent_dead_subscription(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """with tool namespaces the owner-routed message arrives; without them nothing does, silently."""
    if not check_docker_available():
        pytest.skip("Docker not available")

    set_default_namespace(_NS)
    family = Subjects.hitl_forward_family(_OWNED_NODE)
    subject = Subjects.forward_scoped(family, _SESSION_KEY)
    payload = b"open_tab"

    with _nats_with_auth(tmp_path) as uri:
        hub = await _connect(uri, user="hub", password=_HUB_PW, permissions=_hub_permissions())

        # --- WITH tool namespaces: the subscription is real and the message lands ---
        serving = await _connect(uri, user="serving", password=_SERVING_PW, permissions=_serving_permissions())
        async with hub, serving:
            received: list[bytes] = []

            async def _collect(msg: IncomingMessage) -> None:
                received.append(msg.data)

            await serving.subscribe(subject, cb=_collect)
            await asyncio.sleep(0.3)  # let the SUB register before the publish crosses
            await hub.publish_raw(subject=subject, payload=payload)
            await asyncio.sleep(_SILENCE_WINDOW_SECONDS)

            assert received == [payload], "the granted pod's owner-routed subscription received nothing"
            assert serving.is_healthy, "a working grant must leave the connection healthy"

            # --- WITHOUT them: same subject, same publish, and the capability is simply dead ---
            _last_error_log.clear()  # the rate limiter is module-global; start this half clean
            ungranted = await _connect(
                uri,
                user="ungranted",
                password=_UNGRANTED_PW,
                permissions=_ungranted_permissions(),
            )
            async with ungranted:
                stranded: list[bytes] = []

                async def _never(msg: IncomingMessage) -> None:  # pragma: no cover - must never run
                    stranded.append(msg.data)

                with caplog.at_level(logging.ERROR, logger=_CLIENT_LOGGER):
                    # subscribe SUCCEEDS -- this is the whole point. the server refuses it
                    # asynchronously and nothing is raised here, now or ever.
                    subscription = await ungranted.subscribe(subject, cb=_never)
                    assert subscription is not None

                    await asyncio.sleep(0.3)  # let the async -ERR frame land in the error callback
                    await hub.publish_raw(subject=subject, payload=payload)
                    await asyncio.sleep(_SILENCE_WINDOW_SECONDS)

                assert stranded == [], "a refused subscription must receive nothing"
                # ...and the connection is UP and reports itself healthy while serving nothing: a
                # permissions violation is not an authorization violation, so nothing restarts and
                # nothing degrades. the log line below is the only evidence that exists.
                assert ungranted.is_healthy

                violations = [
                    rec
                    for rec in caplog.records
                    if rec.name == _CLIENT_LOGGER and "PERMISSIONS VIOLATION" in rec.getMessage()
                ]
                assert violations, "the silent failure produced no report at all"

                # the report must be MACHINE-READABLE, and must name THIS subject -- asserting only
                # that no message arrived would pass against a broker that was never reachable.
                data = getattr(violations[0], "extra_data", None)
                assert data is not None, "the violation must carry structured fields"
                assert data["operation"] == "subscribe"
                assert data["subject"] == subject.path
                # the HITL subject is a namespace literal plus two sha256 digests, so nats-py's
                # lowercasing cannot mangle it -- the equality above holds even though the case
                # flag correctly refuses to promise that in general.
                assert data["subject_case"] == _SUBJECT_CASE_LOWERCASED

                # the sibling family is refused too: a pod without tool namespaces holds NEITHER of
                # the two families one session derives, so its display stream is dead as well.
                pipe_subject = Subjects.forward_scoped(Subjects.hitl_pipe_family(_OWNED_NODE), _SESSION_KEY)
                assert pipe_subject.path != subject.path
                _last_error_log.clear()
                with caplog.at_level(logging.ERROR, logger=_CLIENT_LOGGER):
                    mark = len(caplog.records)
                    await ungranted.subscribe(pipe_subject, cb=_never)
                    await asyncio.sleep(0.5)
                refused = [
                    getattr(rec, "extra_data", None)
                    for rec in caplog.records[mark:]
                    if rec.name == _CLIENT_LOGGER and "PERMISSIONS VIOLATION" in rec.getMessage()
                ]
                assert any(entry is not None and entry["subject"] == pipe_subject.path for entry in refused), refused
