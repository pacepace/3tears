"""Build a configured :class:`S3ObjectStore` from config + secret references.

A producing or consuming pod wires its object store from deployment config:
``endpoint_url`` + ``bucket`` + ``region`` as plain values, and the S3
credentials as platform *secret references* (``env://`` in dev, ``k8s://`` in
prod) that this helper resolves at construction via
:func:`threetears.core.security.secret_refs.resolve_secret`. The raw
credentials are unwrapped at the last moment and live only inside the returned
store -- never logged, never returned, never held in a plain string here.

This lives beside the impl (not in a pod) so every pod -- the pure-``threetears``
tool pod, an SDK-spawned pod, the reconciler -- wires its store the same tested
way rather than re-resolving refs by hand.
"""

from __future__ import annotations

from typing import Any

from threetears.core.security.secret_refs import resolve_secret
from threetears.observe import get_logger
from threetears.object_store.s3 import S3ObjectStore

__all__ = ["build_s3_object_store"]

_log = get_logger(__name__)


def build_s3_object_store(
    *,
    endpoint_url: str | None,
    bucket: str,
    access_key_ref: str,
    secret_key_ref: str,
    region: str = "us-east-1",
    session: Any = None,
) -> S3ObjectStore:
    """Resolve the credential references and construct a streaming store.

    :param endpoint_url: S3 endpoint (e.g. ``http://minio:9000``); ``None`` uses
        the AWS default endpoint
    :ptype endpoint_url: str | None
    :param bucket: target bucket name
    :ptype bucket: str
    :param access_key_ref: secret reference for the access key id
        (``env://VAR`` / ``k8s://path``); resolved here
    :ptype access_key_ref: str
    :param secret_key_ref: secret reference for the secret access key; resolved here
    :ptype secret_key_ref: str
    :param region: AWS region (MinIO ignores it; AWS S3 requires it)
    :ptype region: str
    :param session: aioboto3 session passthrough for tests; ``None`` lets the
        store create its own
    :ptype session: Any
    :return: a streaming object store ready to put/get/delete
    :rtype: S3ObjectStore
    :raises SecretResolutionError: when either credential reference is malformed,
        names an unknown/unimplemented scheme, or cannot be resolved
    """
    access_key = resolve_secret(access_key_ref).get_secret_value()
    secret_key = resolve_secret(secret_key_ref).get_secret_value()
    store = S3ObjectStore(
        endpoint_url=endpoint_url,
        access_key=access_key,
        secret_key=secret_key,
        bucket=bucket,
        region=region,
        session=session,
    )
    _log.info(
        "built S3 object store",
        extra={
            "extra_data": {
                # config shape only -- never the resolved credential values.
                "bucket": bucket,
                "region": region,
                "endpoint_configured": endpoint_url is not None,
            }
        },
    )
    return store
