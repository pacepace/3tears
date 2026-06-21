"""Secret-reference registry (core home) + the ``register_scheme`` extension hook.

The comprehensive ``env://`` / ``k8s://`` / ``parse_ref`` coverage lives with the
back-compat surface in ``datasources/tests/unit/test_secrets.py``; here we pin the
core contract and the NEW capability: an app (e.g. scriob) registers its own scheme
resolver and ``resolve_secret`` / ``validate_ref`` dispatch to it.

Tests stay independent via a snapshot/restore of the process-global registry, so a
test's registration never leaks into another test (or a re-run in the same process).
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from pydantic import SecretStr

from threetears.core.security import secret_refs
from threetears.core.security.secret_refs import (
    SecretResolutionError,
    parse_ref,
    register_scheme,
    resolve_secret,
    validate_ref,
)


@pytest.fixture(autouse=True)
def _isolate_registry() -> Iterator[None]:
    """Snapshot/restore the scheme registry so a test's registration never leaks."""
    saved = dict(secret_refs._BACKENDS)  # noqa: SLF001 -- test isolation for the module's registry
    try:
        yield
    finally:
        secret_refs._BACKENDS.clear()  # noqa: SLF001
        secret_refs._BACKENDS.update(saved)  # noqa: SLF001


def test_parse_ref_smoke() -> None:
    assert parse_ref("env://MY_VAR") == ("env", "MY_VAR")


def test_env_scheme_still_resolves(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CORE_SECRET_REFS_PROBE", "the-value")
    assert resolve_secret("env://CORE_SECRET_REFS_PROBE").get_secret_value() == "the-value"


def test_unknown_scheme_is_rejected() -> None:
    with pytest.raises(SecretResolutionError):
        validate_ref("neverregistered://repo-1")
    with pytest.raises(SecretResolutionError):
        resolve_secret("neverregistered://repo-1")


def test_register_scheme_makes_a_custom_scheme_resolvable() -> None:
    seen: list[str] = []

    def _resolver(locator: str) -> SecretStr:
        seen.append(locator)
        return SecretStr(f"resolved:{locator}")

    register_scheme("schemeregtest", _resolver)

    assert validate_ref("schemeregtest://repo-42") == "schemeregtest://repo-42"
    assert resolve_secret("schemeregtest://repo-42").get_secret_value() == "resolved:repo-42"
    assert seen == ["repo-42"]  # dispatched to the registered resolver, locator passed through


def test_register_scheme_refuses_to_clobber_an_existing_scheme() -> None:
    # built-in env must not be silently overridden (would be a security footgun).
    with pytest.raises(ValueError, match="env"):
        register_scheme("env", lambda loc: SecretStr("hijacked"))


def test_register_scheme_rejects_an_invalid_scheme_name() -> None:
    for bad in ("Scriob", "scr iob", "1scheme", ""):
        with pytest.raises(ValueError):
            register_scheme(bad, lambda loc: SecretStr("x"))


def test_resolution_failure_names_the_ref_not_the_value() -> None:
    def _boom(locator: str) -> SecretStr:
        raise SecretResolutionError("schemefailtest://secret: backend unavailable")

    register_scheme("schemefailtest", _boom)
    with pytest.raises(SecretResolutionError) as exc:
        resolve_secret("schemefailtest://secret")
    assert "schemefailtest://secret" in str(exc.value)
