"""unit tests for :class:`threetears.core.cache.kv.NatsKvClient`.

every NATS interaction is faked through the wrapper transport;
:class:`NatsKvClient` is exercised purely through its public api so
the tests stay independent of nats-py.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from threetears.core.cache.kv import BucketConfig, NatsKvClient


@pytest.fixture
def client() -> NatsKvClient:
    """:class:`NatsKvClient` with default prefix; no transport open."""
    return NatsKvClient()


@pytest.fixture
def client_with_bucket() -> tuple[NatsKvClient, AsyncMock, str]:
    """:class:`NatsKvClient` with a fake bucket pre-loaded (skips :meth:`connect`)."""
    c = NatsKvClient()
    mock_bucket = AsyncMock()
    full_name = c.bucket_name("collections")
    c.buckets[full_name] = mock_bucket
    return c, mock_bucket, full_name


# ------------------------------------------------------------------
# bucket_name
# ------------------------------------------------------------------


class TestBucketName:
    def test_bucket_name_custom_prefix(self) -> None:
        c = NatsKvClient(bucket_prefix="myapp")
        assert c.bucket_name("collections") == "myapp-collections"

    def test_bucket_name_default_prefix(self, client: NatsKvClient) -> None:
        assert client.bucket_name("collections") == "threetears-collections"


# ------------------------------------------------------------------
# connect
# ------------------------------------------------------------------


class TestConnect:
    @pytest.mark.asyncio
    async def test_connect_creates_default_bucket(self) -> None:
        """connect always opens the ``collections`` bucket."""
        mock_transport = AsyncMock()
        mock_bucket = AsyncMock()
        mock_transport.kv_bucket = AsyncMock(return_value=mock_bucket)

        with patch(
            "threetears.core.cache.kv._NatsTransport.connect",
            AsyncMock(return_value=mock_transport),
        ):
            c = NatsKvClient()
            await c.connect("nats://localhost:4222")

        # one call for the default ``collections`` bucket
        assert mock_transport.kv_bucket.await_count == 1
        kwargs = mock_transport.kv_bucket.call_args.kwargs
        assert kwargs["name"] == "collections"
        assert kwargs["storage"] == "file"

    @pytest.mark.asyncio
    async def test_connect_with_extra_buckets(self) -> None:
        """extra buckets register alongside the default."""
        mock_transport = AsyncMock()
        mock_bucket = AsyncMock()
        mock_transport.kv_bucket = AsyncMock(return_value=mock_bucket)

        with patch(
            "threetears.core.cache.kv._NatsTransport.connect",
            AsyncMock(return_value=mock_transport),
        ):
            c = NatsKvClient()
            await c.connect(
                "nats://localhost:4222",
                extra_buckets=[BucketConfig("ratelimits", 3600)],
            )

        assert mock_transport.kv_bucket.await_count == 2
        names = [call.kwargs["name"] for call in mock_transport.kv_bucket.call_args_list]
        assert "collections" in names
        assert "ratelimits" in names


# ------------------------------------------------------------------
# get
# ------------------------------------------------------------------


class TestGet:
    @pytest.mark.asyncio
    async def test_get_returns_value(
        self, client_with_bucket: tuple[NatsKvClient, AsyncMock, str]
    ) -> None:
        c, mock_bucket, bucket = client_with_bucket
        mock_bucket.get = AsyncMock(return_value=b"hello")

        result = await c.get(bucket, "key1")
        assert result == b"hello"
        mock_bucket.get.assert_awaited_once_with(key="key1")

    @pytest.mark.asyncio
    async def test_get_returns_none_on_miss(
        self, client_with_bucket: tuple[NatsKvClient, AsyncMock, str]
    ) -> None:
        c, mock_bucket, bucket = client_with_bucket
        mock_bucket.get = AsyncMock(return_value=None)

        result = await c.get(bucket, "missing")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_returns_none_on_error(
        self, client_with_bucket: tuple[NatsKvClient, AsyncMock, str]
    ) -> None:
        c, mock_bucket, bucket = client_with_bucket
        mock_bucket.get = AsyncMock(side_effect=RuntimeError("connection lost"))

        result = await c.get(bucket, "key1")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_unknown_bucket_returns_none(self, client: NatsKvClient) -> None:
        result = await client.get("nonexistent-bucket", "key1")
        assert result is None


# ------------------------------------------------------------------
# put
# ------------------------------------------------------------------


class TestPut:
    @pytest.mark.asyncio
    async def test_put_success(
        self, client_with_bucket: tuple[NatsKvClient, AsyncMock, str]
    ) -> None:
        c, mock_bucket, bucket = client_with_bucket
        mock_bucket.put = AsyncMock(return_value=42)

        result = await c.put(bucket, "key1", b"value")
        assert result is True
        mock_bucket.put.assert_awaited_once_with(key="key1", value=b"value")

    @pytest.mark.asyncio
    async def test_put_returns_false_on_error(
        self, client_with_bucket: tuple[NatsKvClient, AsyncMock, str]
    ) -> None:
        c, mock_bucket, bucket = client_with_bucket
        mock_bucket.put = AsyncMock(side_effect=RuntimeError("write failed"))

        result = await c.put(bucket, "key1", b"value")
        assert result is False


# ------------------------------------------------------------------
# delete
# ------------------------------------------------------------------


class TestDelete:
    @pytest.mark.asyncio
    async def test_delete_success(
        self, client_with_bucket: tuple[NatsKvClient, AsyncMock, str]
    ) -> None:
        c, mock_bucket, bucket = client_with_bucket
        mock_bucket.delete = AsyncMock(return_value=True)

        result = await c.delete(bucket, "key1")
        assert result is True

    @pytest.mark.asyncio
    async def test_delete_returns_false_on_error(
        self, client_with_bucket: tuple[NatsKvClient, AsyncMock, str]
    ) -> None:
        c, mock_bucket, bucket = client_with_bucket
        mock_bucket.delete = AsyncMock(side_effect=RuntimeError("delete failed"))

        result = await c.delete(bucket, "key1")
        assert result is False


# ------------------------------------------------------------------
# create
# ------------------------------------------------------------------


class TestCreate:
    @pytest.mark.asyncio
    async def test_create_success(
        self, client_with_bucket: tuple[NatsKvClient, AsyncMock, str]
    ) -> None:
        c, mock_bucket, bucket = client_with_bucket
        mock_bucket.create = AsyncMock(return_value=1)

        result = await c.create(bucket, "key1", b"value")
        assert result is True
        mock_bucket.create.assert_awaited_once_with(key="key1", value=b"value")

    @pytest.mark.asyncio
    async def test_create_returns_false_if_exists(
        self, client_with_bucket: tuple[NatsKvClient, AsyncMock, str]
    ) -> None:
        c, mock_bucket, bucket = client_with_bucket
        # NatsKvBucket returns None on CAS conflict
        mock_bucket.create = AsyncMock(return_value=None)

        result = await c.create(bucket, "key1", b"value")
        assert result is False

    @pytest.mark.asyncio
    async def test_create_returns_false_on_error(
        self, client_with_bucket: tuple[NatsKvClient, AsyncMock, str]
    ) -> None:
        c, mock_bucket, bucket = client_with_bucket
        mock_bucket.create = AsyncMock(side_effect=RuntimeError("create failed"))

        result = await c.create(bucket, "key1", b"value")
        assert result is False


# ------------------------------------------------------------------
# update
# ------------------------------------------------------------------


class TestUpdate:
    @pytest.mark.asyncio
    async def test_update_success(
        self, client_with_bucket: tuple[NatsKvClient, AsyncMock, str]
    ) -> None:
        c, mock_bucket, bucket = client_with_bucket
        mock_bucket.update = AsyncMock(return_value=42)

        result = await c.update(bucket, "key1", b"new_value", 41)
        assert result == 42
        mock_bucket.update.assert_awaited_once_with(
            key="key1", value=b"new_value", revision=41
        )

    @pytest.mark.asyncio
    async def test_update_returns_none_on_revision_mismatch(
        self, client_with_bucket: tuple[NatsKvClient, AsyncMock, str]
    ) -> None:
        c, mock_bucket, bucket = client_with_bucket
        mock_bucket.update = AsyncMock(return_value=None)

        result = await c.update(bucket, "key1", b"new_value", 99)
        assert result is None


# ------------------------------------------------------------------
# get_entry
# ------------------------------------------------------------------


class TestGetEntry:
    @pytest.mark.asyncio
    async def test_get_entry_success(
        self, client_with_bucket: tuple[NatsKvClient, AsyncMock, str]
    ) -> None:
        c, mock_bucket, bucket = client_with_bucket
        mock_bucket.get_entry = AsyncMock(return_value=(b"data", 5))

        result = await c.get_entry(bucket, "key1")
        assert result == (b"data", 5)

    @pytest.mark.asyncio
    async def test_get_entry_returns_none_on_miss(
        self, client_with_bucket: tuple[NatsKvClient, AsyncMock, str]
    ) -> None:
        c, mock_bucket, bucket = client_with_bucket
        mock_bucket.get_entry = AsyncMock(return_value=None)

        result = await c.get_entry(bucket, "missing")
        assert result is None


# ------------------------------------------------------------------
# ping
# ------------------------------------------------------------------


class TestPing:
    @pytest.mark.asyncio
    async def test_ping_success(self, client: NatsKvClient) -> None:
        mock_transport = MagicMock()
        mock_js = AsyncMock()
        mock_transport.jetstream_context = MagicMock(return_value=mock_js)
        # setattr bypasses ruff SLF001 (private member access) since
        # the unit test legitimately needs to inject a fake transport
        # without going through :meth:`connect`. the attribute is
        # private to express the stability contract; this test is
        # in-package so the access is local to the implementation.
        setattr(client, "_transport", mock_transport)

        result = await client.ping()
        assert result is True
        mock_js.account_info.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_ping_no_transport(self, client: NatsKvClient) -> None:
        result = await client.ping()
        assert result is False

    @pytest.mark.asyncio
    async def test_ping_failure(self, client: NatsKvClient) -> None:
        mock_transport = MagicMock()
        mock_js = AsyncMock()
        mock_js.account_info.side_effect = RuntimeError("unreachable")
        mock_transport.jetstream_context = MagicMock(return_value=mock_js)
        setattr(client, "_transport", mock_transport)

        result = await client.ping()
        assert result is False


# ------------------------------------------------------------------
# close
# ------------------------------------------------------------------


class TestClose:
    @pytest.mark.asyncio
    async def test_close_drains_transport(self, client: NatsKvClient) -> None:
        mock_transport = AsyncMock()
        setattr(client, "_transport", mock_transport)

        await client.close()

        mock_transport.shutdown.assert_awaited_once()
        assert client.transport is None

    @pytest.mark.asyncio
    async def test_close_idempotent(self, client: NatsKvClient) -> None:
        # second call (no transport) is a no-op
        await client.close()
        await client.close()
