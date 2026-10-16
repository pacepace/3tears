"""Live door-refusal regression: a door-request a pod has no valid proof for, delivered over a real
nats-server by each trick a grant cannot stop, is refused by the receiver's ``verify_door_request``
with no side effect, while the genuine request from the real sender still verifies.

This joins the two merged patterns (see ``forger_harness``): the delivery trick from
``test_a_pod_can_bounce_its_bytes_to_any_subject_residual_live`` and the receiver refusal from
``packages/core/tests/unit/security/test_nats_proxy_door_proof``. The receiver here is
``verify_door_request`` itself, the one call every door runs before it acts: a body it raises on is a
body the door would not act on, so a refusal is exactly "no side effect".

One door is covered end to end, ``l3``. Its genuine and forged bodies are built from the same
``NatsProxyL3Backend`` the unit tests use, so the wire shape is the real one. Further doors
(``hub.usage.track``, ``hub.claim_grant``, ``knowledge.draft``, the service-envelope doors) are the
same shape with their own genuine/forged bodies and are added here as each is grounded.

Gated on docker: a checkout without docker skips cleanly.
"""

from __future__ import annotations

import json
import time
from datetime import timedelta
from pathlib import Path

import pytest
from threetears.core.backends.nats_proxy import NatsProxyL3Backend
from threetears.core.security import (
    DOOR_PROOF_FIELD,
    PopKey,
    DelegatedTokenError,
    build_jwks,
    generate_signing_keypair,
    jwk_thumbprint,
    sign_identity_token,
    verify_door_request,
)
from threetears.core.security.identity_token import IdentityClaims
from threetears.core.testing.containers import check_docker_available
from threetears.nats import NatsClient
from threetears.nats.subject_permissions import Principal, build_permissions
from threetears.nats.subjects import get_default_namespace, set_default_namespace

from .forger_harness import (
    admin_client,
    minted_allow_lists,
    nats_with_auth,
    pod_client,
    server_config,
    settle,
    watch,
)

pytestmark = pytest.mark.integration

_NS = "forge"
_POD = "01947100-0000-7000-8000-0000000000f0"
_HUB, _HUB_PUB = generate_signing_keypair()
_JWKS = build_jwks({"hub-1": _HUB_PUB})
_AGENT, _AGENT_PUB = generate_signing_keypair()


def _token() -> str:
    """a hub-signed identity token bound to the agent key, as a real pod carries.

    :return: the signed token
    :rtype: str
    """
    now = int(time.time())
    return sign_identity_token(
        IdentityClaims(
            sub="agent-1",
            customer_id="cust-1",
            sid="sid-1",
            pod_id=_POD,
            iss="hub",
            iat=now,
            exp=now + 600,
            cnf=jwk_thumbprint(_AGENT_PUB),
        ),
        signing_key=_HUB,
        kid="hub-1",
    )


async def _door_bodies() -> tuple[bytes, bytes]:
    """a genuine l3 query body (proof by the bound key) and a forged one (proof by a key the token
    does not bind), both the real wire shape the backend sends.

    :return: ``(genuine, forged)`` request bytes
    :rtype: tuple[bytes, bytes]
    """
    from unittest.mock import AsyncMock, MagicMock  # noqa: PLC0415

    def _capture(pop: object) -> tuple[NatsProxyL3Backend, AsyncMock]:
        nc = MagicMock()
        nc.request_raw = AsyncMock(return_value=json.dumps({"success": True, "rows": []}).encode())
        backend = NatsProxyL3Backend(
            nats_client=nc,
            namespace_prefix=_NS,
            agent_id="agent-1",
            identity_token=_token,
            pop_key=pop,  # type: ignore[arg-type]
        )
        return backend, nc.request_raw

    genuine_backend, genuine_sent = _capture(lambda: PopKey(_AGENT))
    await genuine_backend.fetch("SELECT 1")
    genuine = bytes(genuine_sent.await_args_list[-1].kwargs["payload"])

    # the forger holds the pod's token but not the key it binds. A backend given a key the token does
    # not bind sends the token alone, with no proof (test_nats_proxy_door_proof pins this), so the
    # forgery carries the token over a body nobody proved: the door refuses it.
    thief, _ = generate_signing_keypair()
    forged_backend, forged_sent = _capture(lambda: PopKey(thief))
    await forged_backend.fetch("SELECT 1 WHERE forged")
    forged = bytes(forged_sent.await_args_list[-1].kwargs["payload"])
    assert DOOR_PROOF_FIELD not in json.loads(forged), "the forgery must not carry a valid proof"
    return genuine, forged


def _receiver_accepts(body_bytes: bytes) -> bool:
    """what the l3 door does with a delivered body: verify, and act only if it passes.

    :param body_bytes: the bytes the receiver got
    :ptype body_bytes: bytes
    :return: True if the door would act (the request verified), False if it refused
    :rtype: bool
    """
    try:
        verify_door_request(json.loads(body_bytes), jwks=_JWKS, hub_issuer="hub", door="l3", op="query")
    except (DelegatedTokenError, ValueError):
        # a missing or unbound proof (ProofAbsentError subclasses DelegatedTokenError), or a
        # malformed body: every one is the door refusing to act.
        return False
    return True


async def test_a_forged_l3_request_put_on_the_door_by_a_real_pod_is_refused_the_genuine_one_verifies(
    tmp_path: Path,
) -> None:
    if not check_docker_available():
        pytest.skip("Docker not available")
    genuine, forged = await _door_bodies()
    # sanity at the receiver before any bus: the door accepts the genuine and refuses the forged.
    assert _receiver_accepts(genuine) is True, "the genuine request must verify"
    assert _receiver_accepts(forged) is False, "the forged request must be refused"

    previous = get_default_namespace()
    set_default_namespace(_NS)
    try:
        pub_allow, sub_allow = minted_allow_lists(
            build_permissions(Principal.TOOL_POD, pod_id=_POD, conn_id=_POD, object_store=True)
        )
    finally:
        set_default_namespace(previous)

    # the pod's grant DOES admit the l3 door subject -- a pod reaches L3 through it legitimately. So
    # the subject grant is no defence here; the door's proof check is the whole defence. The forgery
    # is a plain publish of an unproven request to the door, "a direct publish where the grant allows
    # one" (the plan). A door whose subject the pod may NOT publish is reached by the bounce trick
    # instead (forger_harness, DoorCase.watch_subjects); those cases are added as each is grounded.
    door_subject = f"{_NS}.l3.query"
    assert door_subject in pub_allow, "the l3 door is published by the pod; this case is the direct-publish forgery"

    with nats_with_auth(server_config(pub_allow, sub_allow), tmp_path) as uri:
        async with admin_client(uri, _NS) as receiver:
            received = await watch(receiver, [door_subject])  # watch BEFORE publishing
            pod: NatsClient | None = None
            try:
                set_default_namespace(_NS)
                pod, _ = await pod_client(uri, _NS, _POD)
                await pod.raw.publish(door_subject, forged)
                await pod.raw.publish(door_subject, genuine)
                got = await settle(received, want=2, seconds=5)
            finally:
                set_default_namespace(previous)
                if pod is not None:
                    await pod.shutdown(drain_timeout=timedelta(seconds=1))
        bodies = got.get(door_subject, [])
        # both reached the door (the bus let them through); the receiver refuses the forged and
        # accepts the genuine. "Refused" is "would not act": a verify that raises is no side effect.
        assert any(_receiver_accepts(b) for b in bodies), "the genuine request did not reach the door and verify"
        forged_at_door = [b for b in bodies if not _receiver_accepts(b)]
        assert forged_at_door, "the forged request did not reach the door; the delivery proved nothing"
        for body in forged_at_door:
            assert _receiver_accepts(body) is False, "a forged l3 request verified at the door"
