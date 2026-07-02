"""Tests for build_s3_object_store -- secret-ref resolution into a store.

A capturing fake session records the kwargs the store hands to ``session.client``
when it opens a connection, so we can assert the RESOLVED credentials + the
endpoint/region config reach the S3 client without touching the store's private
attributes (the store deliberately exposes none).
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from threetears.core.security.secret_refs import SecretResolutionError
from threetears.object_store.wiring import build_s3_object_store


class _NoopClient:
    """An S3 client stub that satisfies the ensure_bucket() call path."""

    async def __aenter__(self) -> _NoopClient:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    async def head_bucket(self, **kwargs: object) -> dict[str, Any]:
        return {}


class _CapturingSession:
    """Fake aioboto3 session recording the kwargs of the last client() call."""

    def __init__(self) -> None:
        self.client_kwargs: dict[str, Any] | None = None

    def client(self, *args: object, **kwargs: Any) -> _NoopClient:
        self.client_kwargs = kwargs
        return _NoopClient()


async def test_resolved_creds_and_config_reach_the_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """env:// refs resolve and flow (with endpoint/region) into session.client()."""
    monkeypatch.setenv("TEST_S3_ACCESS_KEY", "AKIA-RESOLVED")
    monkeypatch.setenv("TEST_S3_SECRET_KEY", "SECRET-RESOLVED")
    session = _CapturingSession()
    store = build_s3_object_store(
        endpoint_url="http://minio:9000",
        bucket="aibots-objects",
        access_key_ref="env://TEST_S3_ACCESS_KEY",
        secret_key_ref="env://TEST_S3_SECRET_KEY",
        region="eu-west-1",
        session=session,
    )
    # opening a client (ensure_bucket) is what hands the resolved creds to aioboto3.
    await store.ensure_bucket()
    assert session.client_kwargs is not None
    assert session.client_kwargs["aws_access_key_id"] == "AKIA-RESOLVED"
    assert session.client_kwargs["aws_secret_access_key"] == "SECRET-RESOLVED"
    assert session.client_kwargs["endpoint_url"] == "http://minio:9000"
    assert session.client_kwargs["region_name"] == "eu-west-1"


def test_no_credential_value_is_logged(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """The build log emits config shape only -- never the resolved secret values."""
    monkeypatch.setenv("TEST_AK", "AKIA-SUPERSECRET-VALUE")
    monkeypatch.setenv("TEST_SK", "SK-SUPERSECRET-VALUE")
    with caplog.at_level(logging.DEBUG, logger="threetears.object_store.wiring"):
        build_s3_object_store(
            endpoint_url="http://minio:9000",
            bucket="b",
            access_key_ref="env://TEST_AK",
            secret_key_ref="env://TEST_SK",
        )
    blob = " ".join(
        [r.getMessage() for r in caplog.records] + [repr(getattr(r, "extra_data", None)) for r in caplog.records]
    )
    assert "AKIA-SUPERSECRET-VALUE" not in blob
    assert "SK-SUPERSECRET-VALUE" not in blob
    # sanity: the build log actually fired, so the assertion above is not vacuous.
    assert any("built S3 object store" in r.getMessage() for r in caplog.records)


def test_unknown_scheme_ref_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A credential ref naming an unknown scheme fails closed at build time."""
    monkeypatch.setenv("TEST_S3_SECRET_KEY", "ok")
    with pytest.raises(SecretResolutionError):
        build_s3_object_store(
            endpoint_url=None,
            bucket="b",
            access_key_ref="bogus://nope",
            secret_key_ref="env://TEST_S3_SECRET_KEY",
        )


def test_missing_env_ref_raises() -> None:
    """An env:// ref pointing at an unset variable fails closed at build time."""
    with pytest.raises(SecretResolutionError):
        build_s3_object_store(
            endpoint_url=None,
            bucket="b",
            access_key_ref="env://DEFINITELY_UNSET_S3_KEY_XYZ",
            secret_key_ref="env://ALSO_UNSET_S3_SECRET_XYZ",
        )
