"""the registry's catalog bucket, declared through the shared persisted-copy owner.

the in-memory :class:`~threetears.registry.catalog.ToolCatalog` is what the registry routes from.
the catalog bucket (``tool_catalog`` by default) is its persisted copy, read only to warm-load a
starting registry -- but every registration also writes to it, and answers ``CATALOG_UNAVAILABLE``
when that write fails.

the bucket is owned by :class:`threetears.nats.PersistedCopyBucket`, built here by
:func:`catalog_bucket` so its name and shape are stated once:

- **exact name.** the bucket has always carried the bare name every deployment's NATS permissions
  grant (``$KV.tool_catalog.>``, ``KV_tool_catalog``), so it is declared without the namespace
  prefix; a prefixed name would orphan the persisted catalog and fall outside the grant.
- **the shape it has always had**, now stated: file storage, history 1, and ``allow_direct`` set
  (nats-py's own create left it unset; the first declaration of a live bucket sets it in place).
- **a handle that follows the client.** the owner hands the catalog the client's own bucket handle,
  which rebinds across a credential renewal or a move off a lame-duck server. the raw nats-py handle
  this replaced stayed bound to the retired connection, and every write after a NATS rolling
  restart failed until the pod was deleted by hand.
- **one recovery policy.** a start whose broker does not answer still serves from memory; the
  owner declares in the background until it lands, loads what an earlier registry persisted once
  (never replacing an entry registered since), and writes the catalog back. whenever the client
  creates the bucket again -- a NATS restart that lost it, or a handle's self-heal after a wipe --
  the catalog is written back into it; never loaded again, since a load then could re-add a tool
  whose deregistration failed to reach the bucket.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

from threetears.nats import PersistedCopyBucket

from threetears.registry.catalog import ToolCatalog

if TYPE_CHECKING:
    from threetears.nats.kv import KvBucketLike, KvDeclaring

__all__ = ["CatalogRestoreError", "catalog_bucket"]


class CatalogRestoreError(RuntimeError):
    """some catalog entries could not be written back into the bucket."""


async def _write_catalog_back(catalog: ToolCatalog, kv: KvBucketLike) -> None:
    """write every entry the catalog holds into its bucket, raising when any did not land.

    raising is what keeps the write-back owed: the owner retries a declaration whose write-back
    raised, and the client keeps a refill owed until it returns.

    :param catalog: the in-memory catalog
    :ptype catalog: ToolCatalog
    :param kv: the catalog's bucket
    :ptype kv: KvBucketLike
    :return: nothing
    :rtype: None
    :raises CatalogRestoreError: when an entry could not be written
    """
    failed = await catalog.restore_to_kv(kv)
    if failed:
        raise CatalogRestoreError(f"{len(failed)} catalog entries were not written back: {failed}")


def catalog_bucket(*, catalog: ToolCatalog, client: KvDeclaring, bucket: str) -> PersistedCopyBucket:
    """the owner of the catalog's bucket, not yet started.

    ``load`` is :meth:`ToolCatalog.load_from_kv`, which never replaces an entry the catalog holds;
    ``write_back`` is :meth:`ToolCatalog.restore_to_kv`, which reads each entry at the moment it
    writes it, so a tool deregistered meanwhile is not written back.

    :param catalog: the in-memory catalog the bucket is a copy of
    :ptype catalog: ToolCatalog
    :param client: the registry's NATS client
    :ptype client: KvDeclaring
    :param bucket: the bucket's exact name
    :ptype bucket: str
    :return: the owner; :meth:`PersistedCopyBucket.start` begins declaring
    :rtype: PersistedCopyBucket
    """
    return PersistedCopyBucket(
        client=client,
        bucket=bucket,
        load=catalog.load_from_kv,
        write_back=functools.partial(_write_catalog_back, catalog),
        storage="file",
        history=1,
        direct=True,
    )
