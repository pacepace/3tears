"""Every security rejection reaches an operator.

These modules refuse things for a living, and until now they refused silently: each raised
a typed error carrying a structural reason and wrote nothing anywhere. That is fine when
the caller logs the exception and invisible when it does not -- and the cases that matter
most are exactly the ones a caller is likeliest to swallow, because a denied request looks
to it like an ordinary error path.

So each module routes every refusal through one ``_reject`` / ``_deny`` / ``_key_load_failed``
helper that logs and then raises. This file is what stops that from rotting: a rejection
added later that skips the helper fails here rather than going quiet in production.

**Levels are not uniform, deliberately.** A refused proof or a denied path is a
``warning`` -- one caller was turned away, which is the module working. A signing key that
will not load is an ``error`` -- the service holds nothing to mint with and no caller can
succeed until an operator fixes the deployment.

**Nothing secret is recorded.** The last test asserts that, because a log line is exactly
where a token or a key would end up if someone reached for a richer message.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import SecretStr

from threetears.core.security.encryption import open_secret
from threetears.core.security.identity_minter import IdentityMinter
from threetears.core.security.identity_token import verify_identity_token
from threetears.core.security.pop import verify_pop_proof
from threetears.core.security.proxy_assertion import verify_proxy_assertion
from threetears.core.security.proxy_signer import ProxyAssertionSigner
from threetears.core.security.sandbox import PathSandbox
from threetears.core.security.secret_refs import resolve_secret

_KEY = b"0" * 32


def _sandbox() -> PathSandbox:
    return PathSandbox(fs_roots={"r": Path("/tmp")}, allow_read=["**"], allow_write=[])


#: One refusal per module, each the cheapest way to reach that module's reject helper.
_REFUSALS: list[tuple[str, Callable[[], object], str]] = [
    ("encryption", lambda: open_secret("!!!not-base64!!!", key=_KEY), "decryption rejected"),
    (
        "pop",
        lambda: verify_pop_proof("nope", expected_jkt="x", access_token_hash="a", body_hash="b"),
        "proof-of-possession rejected",
    ),
    ("identity_token", lambda: verify_identity_token("nope", jwks={"keys": []}, issuer="i"), "identity token rejected"),
    (
        "proxy_assertion",
        lambda: verify_proxy_assertion("nope", jwks={"keys": []}, expected_pod_id="p", body_hash="b"),
        "proxy assertion rejected",
    ),
    ("sandbox", lambda: _sandbox().enforce("write", "anything.txt"), "sandbox denied"),
    ("secret_refs", lambda: resolve_secret("env://__NOT_SET_ANYWHERE__"), "secret resolution failed"),
]

#: Key-load failures, which are louder for a different reason -- see the module docstring.
_KEY_FAILURES: list[tuple[str, Callable[[], object], str]] = [
    ("proxy_signer", lambda: ProxyAssertionSigner.from_secret(SecretStr("aaaa")), "proxy assertion signing key"),
    ("identity_minter", lambda: IdentityMinter.from_pem(b"not a pem", kid="k", issuer="i"), "identity signing key"),
]


@pytest.mark.parametrize(("module", "refuse", "message"), _REFUSALS, ids=[case[0] for case in _REFUSALS])
def test_a_refusal_is_logged_at_warning(
    caplog: pytest.LogCaptureFixture,
    module: str,
    refuse: Callable[[], object],
    message: str,
) -> None:
    """A refused caller leaves a record, whatever the caller does with the exception."""
    with caplog.at_level(logging.WARNING), pytest.raises(Exception):  # noqa: B017, PT011
        refuse()
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings, f"{module} refused silently"
    assert any(message in r.getMessage() for r in warnings)


@pytest.mark.parametrize(("module", "refuse", "message"), _KEY_FAILURES, ids=[case[0] for case in _KEY_FAILURES])
def test_a_key_load_failure_is_logged_at_error(
    caplog: pytest.LogCaptureFixture,
    module: str,
    refuse: Callable[[], object],
    message: str,
) -> None:
    """Louder than a refusal: nothing this service signs can succeed until it is fixed."""
    with caplog.at_level(logging.DEBUG), pytest.raises(Exception):  # noqa: B017, PT011
        refuse()
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert errors, f"{module} failed to load a signing key silently"
    assert any(message in r.getMessage() for r in errors)


@pytest.mark.parametrize(
    ("refuse", "secret"),
    [
        # The ciphertext, the reference's resolved value, and the seed are each the one thing
        # that must not be echoed back out through the log line describing their rejection.
        (lambda: open_secret("!!!SUPERSECRETCIPHERTEXT!!!", key=_KEY), "SUPERSECRETCIPHERTEXT"),
        (lambda: ProxyAssertionSigner.from_secret(SecretStr("SUPERSECRETSEED")), "SUPERSECRETSEED"),
        (
            lambda: verify_pop_proof("SUPERSECRETPROOF", expected_jkt="x", access_token_hash="a", body_hash="b"),
            "SUPERSECRETPROOF",
        ),
    ],
    ids=["ciphertext", "signing-seed", "proof"],
)
def test_the_rejected_input_never_reaches_the_log(
    caplog: pytest.LogCaptureFixture,
    refuse: Callable[[], object],
    secret: str,
) -> None:
    """The reason is structural; the material that caused it is not repeated back.

    A log line is where a token or a key ends up the moment someone reaches for a more
    helpful message, and an operator's terminal, log aggregator and retention policy are
    all the wrong place for one.
    """
    with caplog.at_level(logging.DEBUG), pytest.raises(Exception):  # noqa: B017, PT011
        refuse()
    assert caplog.records, "expected the refusal to be logged at all"
    for record in caplog.records:
        assert secret not in record.getMessage()
        assert secret not in str(getattr(record, "extra_data", ""))
