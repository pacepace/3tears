"""Streaming S3-compatible object store for large binary artifacts (Path-2)."""

# the key builder is a CONTRACT (the locked scope-first layout), so it lives in
# the dependency-free media-contracts package -- a producing tool can build a key
# without inheriting this package's aioboto3 client tree. re-exported here for
# back-compat with callers importing it off the impl package.
from threetears.media.contracts.keys import build_object_key, sanitize_segment
from threetears.object_store.encrypted import EncryptedObjectStore
from threetears.object_store.s3 import S3ObjectStore
from threetears.object_store.wiring import build_s3_object_store

__all__ = [
    "EncryptedObjectStore",
    "S3ObjectStore",
    "build_object_key",
    "build_s3_object_store",
    "sanitize_segment",
]
