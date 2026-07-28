# 3tears-iam

`threetears.iam` -- identity and access primitives: passwords, OAuth2/OIDC,
SAML, GitHub sign-in, session tokens, DPoP, TOTP, WebAuthn, and the
anti-automation controls that guard them.

## Problem

Authentication is the part of a service everyone writes from scratch and nobody
gets to write twice. Two services in this platform independently grew argon2id
password hashing with anti-enumeration timing, a GitHub OAuth2 authorization-code
flow, a NATS-KV login throttle, a single-use SHA-256 ticket store, and a JWT
mint/verify pair that pins its claim set. Neither could adopt the other's,
because each was welded to its own database schema, its own transport, and its
own config prefix.

The cost is not the duplicated lines. It is that a subtle protocol mistake --
accepting PKCE's `plain` transform, trusting an unverified GitHub email, reading
`alg` from the token you are about to verify -- has to be found and fixed in
every copy, on separate schedules, and the second copy is the one you discover
during an incident.

## What it does

- **Passwords** -- argon2id hashing, bcrypt verify-then-upgrade for migrated
  credentials, mandatory NFKC normalization, NIST-aligned length policy, and
  k-anonymity breach screening.
- **OAuth2 and OIDC** -- RFC 7636 PKCE (S256 only, never `plain`), OIDC
  discovery and `id_token` verification, and a GitHub client for the one major
  provider that is not an OIDC IdP.
- **SAML** -- service-provider metadata resolution and assertion identity
  extraction, behind an extra so nobody inherits `xmlsec1` for free.
- **Session tokens** -- one claim vocabulary over EdDSA or HS256, with the
  claim set pinned on mint *and* verify so a smuggled `role` claim is a
  verification failure rather than a field nobody reads yet.
- **Proof of possession** -- RFC 9449 DPoP validation, so a stolen bearer token
  is not enough on its own.
- **Anti-automation** -- attempt limiting, lockout, and trusted-proxy client-IP
  resolution, because a per-IP limiter keyed on an ingress pod locks out
  everyone at once.

## Design philosophy

**The package owns protocol, crypto, and policy. It owns nobody's database
schema and nobody's wire DTOs.** The services this was factored out of disagree
about persistence in every way that matters -- one is NATS-RPC-native over a
multi-tenant Postgres schema, the other a FastAPI app with its own control
plane -- and unifying that would have produced an abstraction neither could
use. State therefore sits behind narrow Protocols, with a NATS-KV
implementation supplied for the common case and an in-memory one for tests.

**Algorithms are pinned from literals, never read from the input.** A DPoP
proof does not choose which algorithm verifies it; an `id_token` does not get
to select `none`. The pins are written so a static reader can audit them, and
they are applied twice -- once before key selection, once in the decode call.

**Fail closed, and without a side channel.** A malformed stored hash is an
authentication failure, not a 500. A rejected password does not say which rule
it broke when saying so would build an oracle. Errors carry structural reasons
only -- never token strings, key material, or credentials -- so they are safe to
log at a verification boundary.

**Builds on core rather than beside it.** `jwk_thumbprint`, `build_jwks`,
`generate_signing_keypair`, `ReplayGuard`, and `seal`/`open_secret` already
exist in `threetears.core.security`; the sensitive-action taxonomy already
exists in `threetears.agent.acl`. This package imports all of them. Two of
those had been copied by hand into a downstream repo precisely because no
import path connected them -- this package is that path.

## When to adopt

Any service that authenticates a human or issues a session token. If you are
about to write `jwt.encode`, hash a password, or build an authorization URL,
the primitive is here. If you need a full identity provider -- tenants,
principals, connection records, an admin surface -- that is an application built
*on* this package, not this package.

## Composes with

- [`agent-acl`](agent-acl.md) -- authorization. This package answers "who are
  you"; that one answers "may you". Session tokens deliberately carry no grants
  so authorization stays live.
- [`agent-audit`](agent-audit.md) -- the audit envelope every authentication
  event should land in.
- [`nats`](nats.md) -- the JetStream KV buckets backing the shipped store
  implementations.
- [`observe`](observe.md) -- structured logging at the verification boundaries.

## Install

```bash
pip install 3tears-iam
pip install '3tears-iam[saml]'      # adds pysaml2; needs the xmlsec1 system binary
pip install '3tears-iam[webauthn]'  # adds passkey support
```
