"""Streaming S3-compatible object store (aioboto3).

Implements :class:`threetears.media.contracts.ObjectStore` over any
S3-compatible backend (MinIO in dev, S3 in prod). Never buffers a whole
object: uploads stream through one part-size buffer at a time via S3
multipart (or a single PUT when the whole object fits one part); downloads
yield the response body in chunks. Lifted from metallm's ``S3Service`` and
made streaming.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import aioboto3  # type: ignore[import-untyped]
from botocore.config import Config as BotoConfig  # type: ignore[import-untyped]
from botocore.exceptions import ClientError  # type: ignore[import-untyped]
from threetears.media.contracts import ObjectStore
from threetears.observe import get_logger

__all__ = ["S3ObjectStore"]

log = get_logger(__name__)

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


class S3ObjectStore:
    """Streaming ObjectStore over an S3-compatible backend.

    :param endpoint_url: S3 endpoint (e.g. ``http://minio:9000``); ``None``
        uses the AWS default endpoint
    :ptype endpoint_url: str | None
    :param access_key: access key id
    :ptype access_key: str
    :param secret_key: secret access key
    :ptype secret_key: str
    :param bucket: target bucket name
    :ptype bucket: str
    :param region: AWS region (MinIO ignores it; AWS S3 requires it)
    :ptype region: str
    :param part_size_bytes: multipart part size / upload buffer ceiling;
        must be >= 5 MiB
    :ptype part_size_bytes: int
    :param session: aioboto3 session to use; defaults to a fresh
        ``aioboto3.Session()``. Injectable so tests can supply a fake client.
    :ptype session: Any
    """

    def __init__(
        self,
        *,
        endpoint_url: str | None,
        access_key: str,
        secret_key: str,
        bucket: str,
        region: str = "us-east-1",
        part_size_bytes: int = _DEFAULT_PART_SIZE,
        session: Any = None,
    ) -> None:
        if part_size_bytes < _MIN_PART_SIZE:
            raise ValueError("part_size_bytes must be >= 5 MiB (S3 multipart minimum)")
        self._endpoint_url = endpoint_url
        self._access_key = access_key
        self._secret_key = secret_key
        self._bucket = bucket
        self._region = region
        self._part_size = part_size_bytes
        self._session = session if session is not None else aioboto3.Session()

    def _client(self) -> Any:
        """Return an async-context-manager S3 client.

        :return: aioboto3 client context manager
        :rtype: Any
        """
        return self._session.client(
            "s3",
            endpoint_url=self._endpoint_url,
            aws_access_key_id=self._access_key,
            aws_secret_access_key=self._secret_key,
            region_name=self._region,
            config=BotoConfig(signature_version="s3v4"),
        )

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

    async def list_keys(self, prefix: str | None = None) -> AsyncIterator[str]:
        """Yield object keys (paginated), optionally restricted to ``prefix``.

        :param prefix: key-prefix filter (e.g. a tenant's ``<customer_id>/``);
            ``None`` lists the whole bucket
        :ptype prefix: str | None
        :return: async iterator over object keys
        :rtype: AsyncIterator[str]
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
                    yield obj["Key"]
                if not resp.get("IsTruncated"):
                    break
                token = resp.get("NextContinuationToken")

    async def presigned_get_url(self, key: str, *, expires_in: int = 300) -> str:
        """Presigned GET URL for delivery -- bytes never cross the agent.

        :param key: object key
        :ptype key: str
        :param expires_in: URL validity in seconds
        :ptype expires_in: int
        :return: presigned URL
        :rtype: str
        """
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
