# 3tears-iam

`threetears.iam` -- the identity and access primitives every authenticating
service in the platform needs: password handling, OAuth2/OIDC, SAML, GitHub
sign-in, session tokens, DPoP, TOTP, WebAuthn, and the anti-automation
controls that keep all of it from being brute-forced.

## Why this exists

Two services in this ecosystem grew their own identity layers independently.
Both wrote argon2id password hashing with anti-enumeration timing. Both wrote
a GitHub OAuth2 authorization-code flow. Both wrote a NATS-KV login throttle,
a single-use SHA-256 ticket store, and a JWT mint/verify pair that pins its
claim set. Neither could use the other's, because each was welded to its own
database schema, its own transport, and its own config prefix.

That is the failure this package exists to stop. The protocol work -- RFC 7636
PKCE, RFC 9449 DPoP, RFC 6238 TOTP, OIDC discovery and `id_token` verification,
SAML assertion handling, the OAuth2 code exchange -- is the same everywhere.
Getting it subtly wrong is a security bug, and getting it subtly wrong twice
means fixing it twice, in two repos, on two schedules, and finding out the
second one was missed during an incident.

## Model

The package owns **protocol, crypto, and policy**. It owns nobody's database
schema and nobody's wire DTOs.

That line is deliberate. The two services that seeded this package disagree on
almost everything below the protocol layer -- one is NATS-RPC-native with a
multi-tenant Postgres `identity` schema, the other is a FastAPI app with its own
control plane -- and any attempt to unify their persistence would have produced
an abstraction neither could use. So state lives behind narrow Protocols
(`SingleUseTicketStore`, `AttemptLimiter`, `StateStore`), with a NATS-KV
implementation shipped for the common case and nothing stopping a caller from
supplying its own.

Everything else follows from that:

- **Pure functions where the protocol allows it.** PKCE verification, password
  policy, step-up freshness, claim mapping, and API-key hashing take arguments
  and return answers. No I/O, no clock you cannot inject, no global state.
- **Algorithms are pinned from literals, never read from the input.** A DPoP
  proof does not get to say which algorithm verifies it. An `id_token` does not
  get to select `none`. This mirrors `threetears.core.security.identity_token`'s
  discipline, and the pins are written so a static reader can audit them.
- **Fail closed by default, and without a side channel.** A malformed stored hash is an
  authentication failure, not a 500. The one place a caller may choose otherwise is
  `NatsKvAttemptLimiter`'s `fail_open`, which exists for a cheap throttle sitting in front
  of an authoritative check -- it defaults to closed, and a counter with nothing behind it
  must leave it that way. A rejected password never says *which*
  rule it broke when saying so would build an oracle. Errors carry structural
  reasons only -- never token strings, key material, or credentials -- so they
  are safe to log at a verification boundary.
- **Builds on core, does not fork it.** `jwk_thumbprint`, `build_jwks`,
  `generate_signing_keypair`, `ReplayGuard`, `RevocationGuard`, `WindowedCounter`
  and `seal`/`open_secret` already exist in `threetears.core`. This package
  imports them.

## Public surface

Imported per module -- `threetears.iam` itself exports only `__version__`, so reach for the
submodule that owns the thing:

```python
from threetears.iam.passwords import hash_password
from threetears.iam.tokens import SessionClaims, mint_session_token
from threetears.iam.stores.nats_kv import state_store, ticket_store
```

- **Passwords** (`.passwords`, `.breach`) -- `hash_password`, `verify_password`, `validate_new_password`,
  `normalize_password`, `PasswordVerifyResult`, `PasswordPolicyError`, plus
  `BreachCorpus` for k-anonymity breach screening. argon2id for new hashes,
  bcrypt verify-then-upgrade for migrated ones, NFKC normalization always.
- **OAuth2 / OIDC** (`.pkce`, `.oidc`, `.github`) -- `PkceChallenge` and the RFC 7636 verifier, `OidcDiscoveryClient`,
  `verify_id_token`, `OidcIdentity`, `GithubOAuth2Client`, `GithubProfile`.
- **SAML** (`.saml`, extra: `saml`) -- `SamlMetadataResolver`, assertion identity
  extraction, relay-state validation.
- **Sessions** (`.tokens`, `.rotation`) -- `SessionClaims`, `mint_session_token`,
  `verify_session_token` over EdDSA or HS256, `mint_token_pair`, `TokenPair`,
  `sole_audience`, and `rotate_refresh_token` with reuse detection.
- **Proof of possession** (`.dpop`) -- `validate_dpop_proof` (RFC 9449, ES256/P-256).
- **Second factors** (`.totp`, `.webauthn`) -- TOTP enrolment and verification, backup codes, and
  (extra: `webauthn`) passkey registration/assertion helpers.
- **Anti-automation** (`.stores`, `.clientip`) -- the `AttemptLimiter` Protocol and its
  `NatsKvAttemptLimiter` implementation over `threetears.core.coordination.WindowedCounter`,
  plus `resolve_client_ip` for trusted-proxy-aware rate-limit keying.
- **Storage seams** (`.stores`) -- `SingleUseTicketStore` and `StateStore` Protocols,
  `hash_ticket`/`new_ticket_secret`, the `threetears.iam.stores.nats_kv` implementations with
  their `state_store`/`ticket_store` factories, and in-memory doubles in
  `threetears.iam.stores.memory` for consumer tests.

## Install

```bash
pip install 3tears-iam
pip install '3tears-iam[saml]'      # adds pysaml2; needs the xmlsec1 system binary
pip install '3tears-iam[webauthn]'  # adds passkey support
```

## Versioning policy

`3tears-iam` versions in lockstep with the rest of the 3tears monorepo: every
package shares one version, tracking the framework git tag. All packages move
together.
