"""the legitimate JWKS path, end to end on a real nats-server: a signed reply is taken, a token verifies.

A publisher serves its key set on the hub JWKS subject with a statement signed under its root key;
a provider pinned to that root, with the hub JWKS door REQUIRED, fetches it over request/reply,
adopts the keys, and a token signed under them verifies. The forged-delivery side of this door is
the live test owned by Pace; this one proves the real publisher still gets through.

Uses the session-scoped ``nats_container`` fixture; a checkout without docker skips cleanly.
"""

from __future__ import annotations

import json
import time
from datetime import timedelta
from typing import Any

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from threetears.nats import NatsClient, Subjects

from threetears.core.security.door_proof import DoorProofSwitch
from threetears.core.security.identity_token import (
    IdentityClaims,
    build_jwks,
    generate_signing_keypair,
    jwk_thumbprint,
    sign_identity_token,
    verify_identity_token,
)
from threetears.core.security.jwks_provider import HUB_JWKS_ISSUER, CachedHubJwksProvider
from threetears.core.security.jwks_trust import (
    HUB_JWKS_STATEMENT_WINDOW,
    JWKS_STATEMENT_FIELD,
    sign_jwks_statement,
)

pytestmark = pytest.mark.integration


async def test_a_signed_reply_over_real_nats_is_taken_with_the_door_required(nats_container: str) -> None:
    namespace = "jwkslive"
    root = Ed25519PrivateKey.generate()
    signing, public = generate_signing_keypair()
    kid = jwk_thumbprint(public)

    async with (
        await NatsClient.connect(nats_url=nats_container, nats_subject_namespace=namespace, client_name="hub") as hub,
        await NatsClient.connect(nats_url=nats_container, nats_subject_namespace=namespace, client_name="pod") as pod,
    ):

        async def serve(msg: Any) -> None:
            keys = build_jwks({kid: public})["keys"]
            statement = sign_jwks_statement(
                keys, signing_key=root, issuer=HUB_JWKS_ISSUER, now=int(time.time()), ttl=timedelta(minutes=5)
            )
            body = json.dumps({"keys": keys, JWKS_STATEMENT_FIELD: statement}).encode("utf-8")
            await hub.publish_raw_reply(reply_subject=msg.reply_subject, payload=body)

        # sender-posture: read-only -- the test's stand-in for the hub's public JWKS responder
        subscription = await hub.subscribe(subject=Subjects.hub_jwks(), cb=serve)
        try:
            window = HUB_JWKS_STATEMENT_WINDOW
            provider = CachedHubJwksProvider(
                pod,
                request_timeout_seconds=5.0,
                root_public_key=root.public_key(),
                statement_switch=DoorProofSwitch(window, today=lambda: window.required_from),
            )
            await provider.refresh()

            assert provider.is_warmed
            now = int(time.time())
            token = sign_identity_token(
                IdentityClaims(
                    sub="019470a8-b5c3-7def-8123-0000000000aa",
                    customer_id="019470a8-b5c3-7def-8123-0000000000dd",
                    sid="019470a8-b5c3-7def-8123-0000000000cc",
                    pod_id="019470a8-b5c3-7def-8123-0000000000bb",
                    iss="hub",
                    iat=now,
                    exp=now + 300,
                ),
                signing_key=signing,
                kid=kid,
            )
            assert verify_identity_token(token, jwks=provider(), issuer="hub").sub.endswith("aa")
        finally:
            await hub.unsubscribe(subscription)
