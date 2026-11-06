"""Streaming S3-compatible object store (aiobotocore).

Implements :class:`threetears.media.contracts.ObjectStore` over any
S3-compatible backend (MinIO in dev, S3 in prod). Never buffers a whole
object: uploads stream through one part-size buffer at a time via S3
multipart (or a single PUT when the whole object fits one part); downloads
yield the response body in chunks. Lifted from metallm's ``S3Service`` and
made streaming.

A presign is signing alone, no request, but opening an aiobotocore client to sign builds a whole botocore client
class first: on a shared CPU that held an event loop ~100 ms a URL, and a load that presigns a file per partition
stalled its loop again and again (cobalt dev, 2026-10-27). So a store that opened its own session presigns on a
worker thread, with one synchronous botocore client built once and kept: botocore signs from it safely across
threads, and its credentials refresh themselves (an IRSA role's included).

**One aiobotocore client a loop, kept.** Every request once opened a client of its own and closed it after. Each
open built a botocore client on the loop (the same ~100 ms as a presign's), and each left about a thousand objects in
reference cycles: a process that holds its cycle collector off while it works (a refresh holding rows, sweeping only
the young generation) kept every request's, ~0.2 MiB each, until the hold ended (2026-10-27: a full LSF pull's tens of
thousands of export reads). So a store that opened its own session opens one client per event loop on first use and
keeps it, its connection pool sized for the reads a load runs at once, until :meth:`S3ObjectStore.aclose`. A store
given a session (a test's fake) opens one per request, as before.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager
from typing import Any

import botocore.session  # type: ignore[import-untyped]
from aiobotocore.session import get_session  # type: ignore[import-untyped]
from botocore.config import Config as BotoConfig  # type: ignore[import-untyped]
from botocore.exceptions import ClientError  # type: ignore[import-untyped]
from threetears.media.contracts import ObjectListing, ObjectStore
from threetears.observe import get_logger

__all__ = ["S3ObjectStore"]

log = get_logger(__name__)

#: the service models an S3 client of this store is built from, and the STS one its credentials refresh through
_WARMED_SERVICES = ("s3", "sts")


def _warm_loader(session: Any) -> None:
    """read the service models and endpoint data a client is built from into ``session``'s loader cache, on a worker
    thread, so opening a client on the loop finds them read. A session without a loader (a test's fake) is left alone.

    :param session: the aiobotocore session
    :ptype session: Any
    :return: nothing
    :rtype: None
    """
    get_component = getattr(session, "get_component", None)
    if get_component is None:
        return
    loader = get_component("data_loader")
    for service in _WARMED_SERVICES:
        # cached by the arguments exactly as given: asked every way a client asks (no version, None, the latest)
        version = loader.determine_latest_version(service, "service-2")
        for type_name in ("service-2", "endpoint-rule-set-1"):
            loader.load_service_model(service, type_name)
            for api_version in (None, version):
                loader.load_service_model(service, type_name, api_version=api_version)
    for data in ("endpoints", "partitions", "sdk-default-configuration", "_retry"):
        loader.load_data(data)


#: S3 multipart parts must be >= 5 MiB (except the final part). The default
#: part size doubles as the upload buffer ceiling -- one part-size buffer is
#: the most memory a single ``put`` holds, regardless of total object size.
_MIN_PART_SIZE = 5 * 1024 * 1024
_DEFAULT_PART_SIZE = 8 * 1024 * 1024

#: streamed-download chunk size.
_DOWNLOAD_CHUNK_SIZE = 1024 * 1024

#: S3 ``DeleteObjects`` accepts at most 1000 keys per request; the reconciler
#: sweep batches to this ceiling.
_DELETE_BATCH_SIZE = 1000

#: Socket read ceiling for one response read. Botocore defaults this to 60 seconds, which is a
#: sane ceiling for a request/response call and a WRONG one for a streaming read: the clock runs
#: between the consumer's reads, not during the transfer. A restore feeds its bytes to a database
#: that stops to commit, so a pause longer than the ceiling truncates the body and surfaces as
#: `ContentLengthError: received N of M bytes` -- a corrupt-looking failure with a healthy object
#: behind it. Observed on a 469 MB dump: 36 MB read, then a commit outlasted the default and the
#: stream died.
_DEFAULT_READ_TIMEOUT_SECONDS = 900.0

#: connections the kept client holds at most: every request of a store shares it, and a load reads a few states'
#: files at once, several files a state (botocore's default is 10, which would queue them)
_MAX_POOL_CONNECTIONS = 64


class S3ObjectStore:
    """Streaming ObjectStore over an S3-compatible backend.

    :param endpoint_url: S3 endpoint (e.g. ``http://minio:9000``); ``None``
        uses the AWS default endpoint
    :ptype endpoint_url: str | None
    :param access_key: access key id; ``None`` defers to the ambient AWS credential
        chain (environment, IRSA web-identity token, instance metadata) — the shape a
        pod running under an IAM role for its service account uses, where no static
        key exists to pass
    :ptype access_key: str | None
    :param secret_key: secret access key; ``None`` as for ``access_key``
    :ptype secret_key: str | None
    :param bucket: target bucket name
    :ptype bucket: str
    :param region: AWS region (MinIO ignores it; AWS S3 requires it)
    :ptype region: str
    :param part_size_bytes: multipart part size / upload buffer ceiling;
        must be >= 5 MiB
    :ptype part_size_bytes: int
    :param session: the session clients are opened from; defaults to a fresh aiobotocore session.
        Either an aiobotocore session (``create_client``) or an aioboto3 one (``client``), which
        this store used until 0.75. Injectable so tests can supply a fake client.
    :ptype session: Any
    :param read_timeout_seconds: socket read ceiling for a single response read. Botocore's
        default is 60, which truncates any streaming read whose CONSUMER pauses longer than
        that -- and a restore's consumer is a database committing transactions, which routinely
        does. See the constant below.
    :ptype read_timeout_seconds: float
    """

    def __init__(
        self,
        *,
        endpoint_url: str | None,
        access_key: str | None = None,
        secret_key: str | None = None,
        bucket: str = "",
        region: str = "us-east-1",
        part_size_bytes: int = _DEFAULT_PART_SIZE,
        session: Any = None,
        read_timeout_seconds: float = _DEFAULT_READ_TIMEOUT_SECONDS,
    ) -> None:
        if part_size_bytes < _MIN_PART_SIZE:
            raise ValueError("part_size_bytes must be >= 5 MiB (S3 multipart minimum)")
        self._read_timeout_seconds = read_timeout_seconds
        self._endpoint_url = endpoint_url
        self._access_key = access_key
        self._secret_key = secret_key
        self._bucket = bucket
        self._region = region
        self._part_size = part_size_bytes
        # an injected session (a test's fake, a caller's aioboto3) presigns through it, on the loop, as before
        self._presigns_off_loop = session is None
        self._session = session if session is not None else get_session()
        self._presigner: Any = None
        self._presigner_lock = threading.Lock()
        # the kept client, the loop it was opened on, and what closes it (a store that opened its own session)
        self._kept: Any = None
        self._kept_loop: asyncio.AbstractEventLoop | None = None
        self._kept_exit: AsyncExitStack | None = None
        self._kept_lock: asyncio.Lock | None = None
        # whether this store's session has read the service models a client is built from (:func:`_warm_loader`)
        self._loader_warm = False

    def _config(self) -> BotoConfig:
        """the client configuration every client of this store is opened with.

        :return: the configuration
        :rtype: BotoConfig
        """
        return BotoConfig(
            signature_version="s3v4",
            read_timeout=self._read_timeout_seconds,
            max_pool_connections=_MAX_POOL_CONNECTIONS,
        )

    def _client(self) -> AbstractAsyncContextManager[Any]:
        """an S3 client for one request: the store's kept client (its own session), or one opened for it (an
        injected session).

        :return: an async context manager yielding the client
        :rtype: AbstractAsyncContextManager[Any]
        """
        return self._kept_client() if self._presigns_off_loop else self._opened_client()

    @asynccontextmanager
    async def _kept_client(self) -> AsyncIterator[Any]:
        """the client this store keeps on the running loop, opened on first use (see the module's docstring).

        :return: the client, left open after the request
        :rtype: AsyncIterator[Any]
        """
        loop = asyncio.get_running_loop()
        if self._kept_loop is not loop:
            # a client is bound to the loop it was opened on: another loop (a test's, a new run) opens its own
            self._kept, self._kept_exit, self._kept_loop, self._kept_lock = None, None, loop, asyncio.Lock()
        assert self._kept_lock is not None  # noqa: S101 -- set with the loop above
        async with self._kept_lock:
            if self._kept is None:
                if not self._loader_warm:
                    # the session's service models read off the loop first: read on it, opening the client held the
                    # loop ~50 ms (json of the s3 model and its endpoint rules), at a load's first export read on
                    # cobalt prod (2026-10-31); read here, the open takes ~2 ms
                    await asyncio.to_thread(_warm_loader, self._session)
                    self._loader_warm = True
                exit_stack = AsyncExitStack()
                self._kept = await exit_stack.enter_async_context(self._opened_client())
                self._kept_exit = exit_stack
        yield self._kept

    async def aclose(self) -> None:
        """close the kept client, if one is open on the running loop; the next request opens another.

        :return: nothing
        :rtype: None
        """
        exit_stack, loop = self._kept_exit, self._kept_loop
        self._kept, self._kept_exit = None, None
        if exit_stack is not None and loop is asyncio.get_running_loop():
            await exit_stack.aclose()

    def _opened_client(self) -> Any:
        """a client opened for the caller's block and closed after it.

        :return: an S3 client context manager
        :rtype: Any
        """
        # aiobotocore names it create_client; an aioboto3 session (what a caller may still pass)
        # names the same call client
        open_client = getattr(self._session, "create_client", None) or self._session.client
        return open_client(
            "s3",
            endpoint_url=self._endpoint_url,
            aws_access_key_id=self._access_key,
            aws_secret_access_key=self._secret_key,
            region_name=self._region,
            config=self._config(),
        )

    def _presign(self, key: str, expires_in: int) -> str:
        """presign a GET of one key with the store's synchronous client, built on first use and kept; run on a
        worker thread.

        :param key: object key
        :ptype key: str
        :param expires_in: URL validity in seconds
        :ptype expires_in: int
        :return: presigned URL
        :rtype: str
        """
        with self._presigner_lock:
            if self._presigner is None:
                self._presigner = botocore.session.get_session().create_client(
                    "s3",
                    endpoint_url=self._endpoint_url,
                    aws_access_key_id=self._access_key,
                    aws_secret_access_key=self._secret_key,
                    region_name=self._region,
                    config=self._config(),
                )
            presigner = self._presigner
        url: str = presigner.generate_presigned_url(
            "get_object", Params={"Bucket": self._bucket, "Key": key}, ExpiresIn=expires_in
        )
        return url

    async def ensure_bucket(self) -> None:
        """Create the configured bucket if it does not already exist.

        :return: nothing
        :rtype: None
        """
        async with self._client() as client:
            try:
                await client.head_bucket(Bucket=self._bucket)
            except ClientError as err:
                code = str(err.response.get("Error", {}).get("Code", ""))
                if code not in ("404", "NoSuchBucket", "NotFound"):
                    raise
                await client.create_bucket(Bucket=self._bucket)
                log.info(
                    "object store bucket created",
                    extra={"extra_data": {"bucket": self._bucket}},
                )

    async def put(
        self,
        key: str,
        body: AsyncIterator[bytes],
        *,
        content_type: str,
        size: int | None = None,
    ) -> None:
        """Stream ``body`` to ``key``.

        Peak memory is one part plus the latest incoming chunk -- bounded
        independent of total object size (a multi-GB object never sits whole
        in memory). A single PUT is used when the whole object fits one
        part, otherwise S3 multipart. On any failure the partial multipart
        upload is aborted so no orphaned parts linger.

        :param key: tenant-scoped object key
        :ptype key: str
        :param body: async iterator yielding the object's bytes in chunks
        :ptype body: AsyncIterator[bytes]
        :param content_type: MIME type stored on the object
        :ptype content_type: str
        :param size: total byte length when known (advisory; the impl
            streams regardless)
        :ptype size: int | None
        :return: nothing
        :rtype: None
        """
        async with self._client() as client:
            buffer = bytearray()
            upload_id: str | None = None
            parts: list[dict[str, Any]] = []
            part_number = 1
            completed = False
            try:
                async for chunk in body:
                    buffer.extend(chunk)
                    while len(buffer) >= self._part_size:
                        if upload_id is None:
                            created = await client.create_multipart_upload(
                                Bucket=self._bucket,
                                Key=key,
                                ContentType=content_type,
                            )
                            upload_id = created["UploadId"]
                        part = bytes(buffer[: self._part_size])
                        del buffer[: self._part_size]
                        resp = await client.upload_part(
                            Bucket=self._bucket,
                            Key=key,
                            PartNumber=part_number,
                            UploadId=upload_id,
                            Body=part,
                        )
                        parts.append({"ETag": resp["ETag"], "PartNumber": part_number})
                        part_number += 1
                if upload_id is None:
                    await client.put_object(
                        Bucket=self._bucket,
                        Key=key,
                        Body=bytes(buffer),
                        ContentType=content_type,
                    )
                else:
                    if buffer:
                        resp = await client.upload_part(
                            Bucket=self._bucket,
                            Key=key,
                            PartNumber=part_number,
                            UploadId=upload_id,
                            Body=bytes(buffer),
                        )
                        parts.append({"ETag": resp["ETag"], "PartNumber": part_number})
                    await client.complete_multipart_upload(
                        Bucket=self._bucket,
                        Key=key,
                        UploadId=upload_id,
                        MultipartUpload={"Parts": parts},
                    )
                completed = True
            finally:
                if upload_id is not None and not completed:
                    try:
                        await client.abort_multipart_upload(Bucket=self._bucket, Key=key, UploadId=upload_id)
                        log.info(
                            "aborted partial multipart upload after error",
                            extra={"extra_data": {"key": key, "upload_id": upload_id}},
                        )
                    except ClientError as abort_err:
                        log.warning(
                            "failed to abort multipart upload after error",
                            extra={
                                "extra_data": {
                                    "key": key,
                                    "upload_id": upload_id,
                                    "error": str(abort_err),
                                }
                            },
                        )
        log.debug(
            "object stored",
            extra={
                "extra_data": {
                    "key": key,
                    "multipart": upload_id is not None,
                    "parts": len(parts),
                }
            },
        )

    async def open_read(self, key: str) -> AsyncIterator[bytes]:
        """Open ``key`` for streaming read, yielding bytes in chunks.

        :param key: object key
        :ptype key: str
        :return: async iterator over the object's bytes
        :rtype: AsyncIterator[bytes]
        """
        async with self._client() as client:
            resp = await client.get_object(Bucket=self._bucket, Key=key)
            async for chunk in resp["Body"].iter_chunks(_DOWNLOAD_CHUNK_SIZE):
                yield chunk

    async def delete(self, key: str) -> None:
        """Delete a single object.

        :param key: object key
        :ptype key: str
        :return: nothing
        :rtype: None
        """
        async with self._client() as client:
            await client.delete_object(Bucket=self._bucket, Key=key)

    async def delete_many(self, keys: list[str]) -> None:
        """Delete many objects, batched to S3's 1000-key request limit.

        The reconciler sweep can exceed 1000 keys, so deletes are chunked
        into ``_DELETE_BATCH_SIZE`` requests rather than one oversized call
        S3/MinIO would reject.

        :param keys: object keys to delete
        :ptype keys: list[str]
        :return: nothing
        :rtype: None
        """
        if keys:
            async with self._client() as client:
                for start in range(0, len(keys), _DELETE_BATCH_SIZE):
                    batch = keys[start : start + _DELETE_BATCH_SIZE]
                    await client.delete_objects(
                        Bucket=self._bucket,
                        Delete={
                            "Objects": [{"Key": k} for k in batch],
                            "Quiet": True,
                        },
                    )

    async def delete_versions(self, prefix: str) -> int:
        """Delete every version and delete marker under ``prefix``, for good.

        In a versioned bucket a plain delete only stacks a delete marker on the object and keeps
        its data as a noncurrent version. This removes the data itself: every version and every
        marker under the prefix, by version id, in batches of S3's 1000-key limit. A key S3
        reports it could not delete is raised, never skipped, so a caller that must not leave
        data behind learns that it did.

        :param prefix: the key prefix, non-empty (a whole bucket is never cleared by accident)
        :ptype prefix: str
        :return: how many versions and delete markers were deleted
        :rtype: int
        :raises ValueError: when ``prefix`` is empty
        :raises RuntimeError: when S3 refused to delete any of them
        """
        if not prefix:
            raise ValueError("delete_versions needs a non-empty prefix")
        deleted = 0
        async with self._client() as client:
            markers: dict[str, str] = {}
            while True:
                resp = await client.list_object_versions(Bucket=self._bucket, Prefix=prefix, **markers)
                entries = [
                    {"Key": v["Key"], "VersionId": v["VersionId"]}
                    for v in [*resp.get("Versions", []), *resp.get("DeleteMarkers", [])]
                ]
                for start in range(0, len(entries), _DELETE_BATCH_SIZE):
                    answer = await client.delete_objects(
                        Bucket=self._bucket,
                        Delete={"Objects": entries[start : start + _DELETE_BATCH_SIZE], "Quiet": True},
                    )
                    errors = answer.get("Errors") or []
                    if errors:
                        first = errors[0]
                        raise RuntimeError(
                            f"S3 did not delete {len(errors)} version(s) under {prefix!r}: "
                            f"{first.get('Key')} {first.get('Code')}"
                        )
                deleted += len(entries)
                if not resp.get("IsTruncated"):
                    break
                markers = {"KeyMarker": resp["NextKeyMarker"], "VersionIdMarker": resp["NextVersionIdMarker"]}
        log.info("object versions deleted", extra={"extra_data": {"prefix": prefix, "deleted": deleted}})
        return deleted

    async def _iter_contents(self, prefix: str | None) -> AsyncIterator[dict[str, Any]]:
        """Yield each ``Contents`` entry across every listing page.

        Shared pagination for :meth:`list_keys` and :meth:`list_entries` so the
        continuation-token walk lives in one place.

        :param prefix: key-prefix filter, or ``None`` for the whole bucket
        :ptype prefix: str | None
        :return: async iterator over raw ``list_objects_v2`` ``Contents`` dicts
        :rtype: AsyncIterator[dict[str, Any]]
        """
        async with self._client() as client:
            token: str | None = None
            while True:
                kwargs: dict[str, Any] = {"Bucket": self._bucket}
                if prefix is not None:
                    kwargs["Prefix"] = prefix
                if token is not None:
                    kwargs["ContinuationToken"] = token
                resp = await client.list_objects_v2(**kwargs)
                for obj in resp.get("Contents", []):
                    yield obj
                if not resp.get("IsTruncated"):
                    break
                token = resp.get("NextContinuationToken")

    async def list_keys(self, prefix: str | None = None) -> AsyncIterator[str]:
        """Yield object keys (paginated), optionally restricted to ``prefix``.

        :param prefix: key-prefix filter (e.g. a tenant's ``<customer_id>/``);
            ``None`` lists the whole bucket
        :ptype prefix: str | None
        :return: async iterator over object keys
        :rtype: AsyncIterator[str]
        """
        async for obj in self._iter_contents(prefix):
            yield obj["Key"]

    async def list_entries(self, prefix: str | None = None) -> AsyncIterator[ObjectListing]:
        """Yield object listings (key + last-modified + size), optionally by ``prefix``.

        Carries the ``LastModified`` + ``Size`` metadata S3 already returns on a
        list so the reconciler can judge orphan age without a per-key HEAD.

        :param prefix: key-prefix filter (e.g. a tenant's ``<customer_id>/``);
            ``None`` lists the whole bucket
        :ptype prefix: str | None
        :return: async iterator over object listings
        :rtype: AsyncIterator[ObjectListing]
        """
        async for obj in self._iter_contents(prefix):
            yield ObjectListing(
                key=obj["Key"],
                last_modified=obj["LastModified"],
                size_bytes=int(obj["Size"]),
            )

    async def presigned_get_url(self, key: str, *, expires_in: int = 300) -> str:
        """Presigned GET URL for delivery -- bytes never cross the agent.

        Signed on a worker thread with the store's kept synchronous client when the store opened its own session
        (see the module's docstring); through the injected session otherwise.

        :param key: object key
        :ptype key: str
        :param expires_in: URL validity in seconds
        :ptype expires_in: int
        :return: presigned URL
        :rtype: str
        """
        if self._presigns_off_loop:
            return await asyncio.to_thread(self._presign, key, expires_in)
        async with self._client() as client:
            url: str = await client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self._bucket, "Key": key},
                ExpiresIn=expires_in,
            )
        return url


#: static conformance guarantee -- S3ObjectStore must satisfy the ObjectStore
#: contract this package exists to implement. mypy verifies the structural
#: match here; a missing or mismatched method fails type-checking.
_OBJECTSTORE_IMPL: type[ObjectStore] = S3ObjectStore
