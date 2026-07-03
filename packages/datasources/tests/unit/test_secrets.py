"""tests for threetears.datasources.secrets — pluggable secret resolution.

covers:

- ``parse_ref`` shape validation (scheme://locator split, rejection)
- ``validate_ref`` config-load-time checks WITHOUT touching env / fs
- ``resolve_secret`` use-time dispatch for the ``env://`` backend
- ``resolve_secret`` use-time dispatch for the ``k8s://`` backend
  (projected-Secret file under an overridable dir), incl. path traversal
- registered-but-unimplemented backends raise a clear error
- every failure raises ``SecretResolutionError`` and the message never
  carries the resolved value (only the safe reference)
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import SecretStr

from threetears.datasources.secrets import (
    SecretResolutionError,
    parse_ref,
    resolve_secret,
    validate_ref,
)

_SECRET_VALUE = "horse-battery-staple-NEVER-LOG-ME"


# ---------------------------------------------------------------------------
# parse_ref
# ---------------------------------------------------------------------------


class TestParseRef:
    def test_splits_env(self) -> None:
        assert parse_ref("env://MY_VAR") == ("env", "MY_VAR")

    def test_splits_k8s_with_path(self) -> None:
        assert parse_ref("k8s://central-reporting/password") == (
            "k8s",
            "central-reporting/password",
        )

    def test_scheme_lowercased_alnum_separators(self) -> None:
        assert parse_ref("aws-secretsmanager://arn:foo") == (
            "aws-secretsmanager",
            "arn:foo",
        )

    @pytest.mark.parametrize(
        "bad",
        ["MY_VAR", "env:/MY_VAR", "://locator", "env://", "", "envMY_VAR"],
    )
    def test_rejects_malformed(self, bad: str) -> None:
        with pytest.raises(SecretResolutionError):
            parse_ref(bad)


# ---------------------------------------------------------------------------
# validate_ref (config-load time, no env / fs access)
# ---------------------------------------------------------------------------


class TestValidateRef:
    def test_accepts_env(self) -> None:
        assert validate_ref("env://MY_VAR") == "env://MY_VAR"

    def test_accepts_k8s(self) -> None:
        assert validate_ref("k8s://ds/password") == "k8s://ds/password"

    def test_accepts_registered_unimplemented_scheme(self) -> None:
        # validate only checks the scheme is KNOWN; resolution is what
        # raises for unimplemented backends.
        assert validate_ref("vault://secret/data/db") == "vault://secret/data/db"

    def test_rejects_unknown_scheme(self) -> None:
        with pytest.raises(SecretResolutionError, match="unknown secret scheme"):
            validate_ref("duckdb://nope")

    def test_rejects_bad_env_name(self) -> None:
        with pytest.raises(SecretResolutionError):
            validate_ref("env://has space")

    def test_does_not_touch_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # validating a ref for an UNSET env var must still succeed --
        # resolution is a use-time concern.
        monkeypatch.delenv("DEFINITELY_UNSET_VAR", raising=False)
        assert validate_ref("env://DEFINITELY_UNSET_VAR") == "env://DEFINITELY_UNSET_VAR"


# ---------------------------------------------------------------------------
# resolve_secret — env:// backend
# ---------------------------------------------------------------------------


class TestResolveEnv:
    def test_returns_secret_str(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RESOLVE_ME", _SECRET_VALUE)
        secret = resolve_secret("env://RESOLVE_ME")
        assert isinstance(secret, SecretStr)
        assert secret.get_secret_value() == _SECRET_VALUE

    def test_raises_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("UNSET_VAR", raising=False)
        with pytest.raises(SecretResolutionError, match="UNSET_VAR"):
            resolve_secret("env://UNSET_VAR")

    def test_message_never_carries_value(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # even when set, an unrelated failure path must not embed the
        # value; here we just confirm the success path keeps it in
        # SecretStr (redacted repr).
        monkeypatch.setenv("RESOLVE_ME", _SECRET_VALUE)
        secret = resolve_secret("env://RESOLVE_ME")
        assert _SECRET_VALUE not in repr(secret)
        assert _SECRET_VALUE not in str(secret)


# ---------------------------------------------------------------------------
# resolve_secret — k8s:// backend
# ---------------------------------------------------------------------------


class TestResolveK8s:
    def test_reads_projected_file(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        secret_dir = tmp_path / "secrets"
        (secret_dir / "central-reporting").mkdir(parents=True)
        (secret_dir / "central-reporting" / "password").write_text(
            _SECRET_VALUE,
            encoding="utf-8",
        )
        monkeypatch.setenv("THREETEARS_DATASOURCE_SECRETS_DIR", str(secret_dir))
        secret = resolve_secret("k8s://central-reporting/password")
        assert isinstance(secret, SecretStr)
        assert secret.get_secret_value() == _SECRET_VALUE

    def test_missing_file_raises(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("THREETEARS_DATASOURCE_SECRETS_DIR", str(tmp_path))
        with pytest.raises(SecretResolutionError, match="not found"):
            resolve_secret("k8s://nope/password")

    def test_path_traversal_rejected(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("THREETEARS_DATASOURCE_SECRETS_DIR", str(tmp_path))
        with pytest.raises(SecretResolutionError, match="traversal"):
            resolve_secret("k8s://../../etc/passwd")


# ---------------------------------------------------------------------------
# resolve_secret — unimplemented / unknown
# ---------------------------------------------------------------------------


class TestResolveUnimplementedAndUnknown:
    @pytest.mark.parametrize("scheme", ["vault", "aws-secretsmanager", "gcp-sm"])
    def test_registered_unimplemented_raises_clear_error(self, scheme: str) -> None:
        with pytest.raises(SecretResolutionError, match="not implemented"):
            resolve_secret(f"{scheme}://some/locator")

    def test_unknown_scheme_raises(self) -> None:
        with pytest.raises(SecretResolutionError, match="unknown secret scheme"):
            resolve_secret("duckdb://nope")

    def test_malformed_ref_raises(self) -> None:
        with pytest.raises(SecretResolutionError):
            resolve_secret("not-a-ref")
