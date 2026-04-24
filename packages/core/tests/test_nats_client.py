"""Unit tests for NatsClient — all NATS interactions are mocked."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from nats.js.errors import KeyNotFoundError, KeyWrongLastSequenceError

from threetears.core.cache.nats import BucketConfig, NatsClient


@pytest.fixture
def client() -> NatsClient:
    """NatsClient with default prefix."""
    return NatsClient()


@pytest.fixture
def client_with_bucket() -> tuple[NatsClient, AsyncMock, str]:
    """NatsClient with a mock bucket pre-loaded (bypasses connect)."""
    c = NatsClient()
    mock_kv = AsyncMock()
    bucket = c.bucket_name("collections")
    c.buckets[bucket] = mock_kv
    return c, mock_kv, bucket


# ------------------------------------------------------------------
# bucket_name
# ------------------------------------------------------------------


class TestBucketName:
    def test_bucket_name(self) -> None:
        c = NatsClient(bucket_prefix="myapp")
        assert c.bucket_name("collections") == "myapp-collections"

    def test_bucket_name_default_prefix(self, client: NatsClient) -> None:
        assert client.bucket_name("collections") == "threetears-collections"


# ------------------------------------------------------------------
# connect
# ------------------------------------------------------------------


class TestConnect:
    @pytest.mark.asyncio
    async def test_connect_creates_default_bucket(self) -> None:
        mock_nc = MagicMock()
        mock_js = AsyncMock()
        mock_nc.jetstream.return_value = mock_js

        with patch("threetears.core.cache.nats.nats.connect", AsyncMock(return_value=mock_nc)):
            c = NatsClient()
            await c.connect("nats://localhost:4222")

        mock_js.create_key_value.assert_called_once()
        call_args = mock_js.create_key_value.call_args[0][0]
        assert call_args.bucket == "threetears-collections"

    @pytest.mark.asyncio
    async def test_connect_with_extra_buckets(self) -> None:
        mock_nc = MagicMock()
        mock_js = AsyncMock()
        mock_nc.jetstream.return_value = mock_js

        with patch("threetears.core.cache.nats.nats.connect", AsyncMock(return_value=mock_nc)):
            c = NatsClient()
            await c.connect(
                "nats://localhost:4222",
                extra_buckets=[BucketConfig("ratelimits", 3600)],
            )

        assert mock_js.create_key_value.call_count == 2
        bucket_names = [call[0][0].bucket for call in mock_js.create_key_value.call_args_list]
        assert "threetears-collections" in bucket_names
        assert "threetears-ratelimits" in bucket_names


# ------------------------------------------------------------------
# get
# ------------------------------------------------------------------


class TestGet:
    @pytest.mark.asyncio
    async def test_get_returns_value(self, client_with_bucket: tuple[NatsClient, AsyncMock, str]) -> None:
        c, mock_kv, bucket = client_with_bucket
        entry = MagicMock()
        entry.value = b"hello"
        mock_kv.get.return_value = entry

        result = await c.get(bucket, "key1")
        assert result == b"hello"
        mock_kv.get.assert_awaited_once_with("key1")

    @pytest.mark.asyncio
    async def test_get_returns_none_on_miss(self, client_with_bucket: tuple[NatsClient, AsyncMock, str]) -> None:
        c, mock_kv, bucket = client_with_bucket
        mock_kv.get.side_effect = KeyNotFoundError

        result = await c.get(bucket, "missing")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_returns_none_on_error(self, client_with_bucket: tuple[NatsClient, AsyncMock, str]) -> None:
        c, mock_kv, bucket = client_with_bucket
        mock_kv.get.side_effect = RuntimeError("connection lost")

        result = await c.get(bucket, "key1")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_unknown_bucket_returns_none(self, client: NatsClient) -> None:
        result = await client.get("nonexistent-bucket", "key1")
        assert result is None


# ------------------------------------------------------------------
# put
# ------------------------------------------------------------------


class TestPut:
    @pytest.mark.asyncio
    async def test_put_success(self, client_with_bucket: tuple[NatsClient, AsyncMock, str]) -> None:
        c, mock_kv, bucket = client_with_bucket

        result = await c.put(bucket, "key1", b"value")
        assert result is True
        mock_kv.put.assert_awaited_once_with("key1", b"value")

    @pytest.mark.asyncio
    async def test_put_returns_false_on_error(self, client_with_bucket: tuple[NatsClient, AsyncMock, str]) -> None:
        c, mock_kv, bucket = client_with_bucket
        mock_kv.put.side_effect = RuntimeError("write failed")

        result = await c.put(bucket, "key1", b"value")
        assert result is False


# ------------------------------------------------------------------
# delete
# ------------------------------------------------------------------


class TestDelete:
    @pytest.mark.asyncio
    async def test_delete_success(self, client_with_bucket: tuple[NatsClient, AsyncMock, str]) -> None:
        c, mock_kv, bucket = client_with_bucket

        result = await c.delete(bucket, "key1")
        assert result is True

    @pytest.mark.asyncio
    async def test_delete_returns_true_on_not_found(
        self, client_with_bucket: tuple[NatsClient, AsyncMock, str]
    ) -> None:
        c, mock_kv, bucket = client_with_bucket
        mock_kv.delete.side_effect = KeyNotFoundError

        result = await c.delete(bucket, "missing")
        assert result is True

    @pytest.mark.asyncio
    async def test_delete_returns_false_on_error(self, client_with_bucket: tuple[NatsClient, AsyncMock, str]) -> None:
        c, mock_kv, bucket = client_with_bucket
        mock_kv.delete.side_effect = RuntimeError("delete failed")

        result = await c.delete(bucket, "key1")
        assert result is False


# ------------------------------------------------------------------
# create
# ------------------------------------------------------------------


class TestCreate:
    @pytest.mark.asyncio
    async def test_create_success(self, client_with_bucket: tuple[NatsClient, AsyncMock, str]) -> None:
        c, mock_kv, bucket = client_with_bucket

        result = await c.create(bucket, "key1", b"value")
        assert result is True
        mock_kv.create.assert_awaited_once_with("key1", b"value")

    @pytest.mark.asyncio
    async def test_create_returns_false_if_exists(self, client_with_bucket: tuple[NatsClient, AsyncMock, str]) -> None:
        c, mock_kv, bucket = client_with_bucket
        mock_kv.create.side_effect = KeyWrongLastSequenceError

        result = await c.create(bucket, "key1", b"value")
        assert result is False

    @pytest.mark.asyncio
    async def test_create_returns_false_on_error(self, client_with_bucket: tuple[NatsClient, AsyncMock, str]) -> None:
        c, mock_kv, bucket = client_with_bucket
        mock_kv.create.side_effect = RuntimeError("create failed")

        result = await c.create(bucket, "key1", b"value")
        assert result is False


# ------------------------------------------------------------------
# update
# ------------------------------------------------------------------


class TestUpdate:
    @pytest.mark.asyncio
    async def test_update_success(self, client_with_bucket: tuple[NatsClient, AsyncMock, str]) -> None:
        c, mock_kv, bucket = client_with_bucket
        mock_kv.update.return_value = 42

        result = await c.update(bucket, "key1", b"new_value", 41)
        assert result == 42
        mock_kv.update.assert_awaited_once_with("key1", b"new_value", 41)

    @pytest.mark.asyncio
    async def test_update_returns_none_on_revision_mismatch(
        self, client_with_bucket: tuple[NatsClient, AsyncMock, str]
    ) -> None:
        c, mock_kv, bucket = client_with_bucket
        mock_kv.update.side_effect = KeyWrongLastSequenceError

        result = await c.update(bucket, "key1", b"new_value", 99)
        assert result is None


# ------------------------------------------------------------------
# get_entry
# ------------------------------------------------------------------


class TestGetEntry:
    @pytest.mark.asyncio
    async def test_get_entry_success(self, client_with_bucket: tuple[NatsClient, AsyncMock, str]) -> None:
        c, mock_kv, bucket = client_with_bucket
        entry = MagicMock()
        entry.value = b"data"
        entry.revision = 5
        mock_kv.get.return_value = entry

        result = await c.get_entry(bucket, "key1")
        assert result == (b"data", 5)

    @pytest.mark.asyncio
    async def test_get_entry_returns_none_on_miss(self, client_with_bucket: tuple[NatsClient, AsyncMock, str]) -> None:
        c, mock_kv, bucket = client_with_bucket
        mock_kv.get.side_effect = KeyNotFoundError

        result = await c.get_entry(bucket, "missing")
        assert result is None


# ------------------------------------------------------------------
# ping
# ------------------------------------------------------------------


class TestPing:
    @pytest.mark.asyncio
    async def test_ping_success(self, client: NatsClient) -> None:
        client.js = AsyncMock()

        result = await client.ping()
        assert result is True
        client.js.account_info.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_ping_failure(self, client: NatsClient) -> None:
        client.js = AsyncMock()
        client.js.account_info.side_effect = RuntimeError("unreachable")

        result = await client.ping()
        assert result is False


# ------------------------------------------------------------------
# close
# ------------------------------------------------------------------


class TestClose:
    @pytest.mark.asyncio
    async def test_close_drains(self, client: NatsClient) -> None:
        mock_nc = AsyncMock()
        client.nc = mock_nc

        await client.close()

        mock_nc.drain.assert_awaited_once()
        mock_nc.close.assert_awaited_once()
