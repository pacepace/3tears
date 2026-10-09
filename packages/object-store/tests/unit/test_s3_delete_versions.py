"""S3ObjectStore.delete_versions: every version and delete marker under a prefix goes, or it says why."""

from __future__ import annotations

from typing import Any

import pytest

from threetears.object_store.s3 import S3ObjectStore


# parity-exempt: aioboto3 S3 client stub -- only list_object_versions and delete_objects, the calls delete_versions makes
class _VersionedClient:
    def __init__(self, versions: list[tuple[str, str]], markers: list[tuple[str, str]], *, refuse: set[str], page: int):
        self.versions = versions
        self.markers = markers
        self.refuse = refuse
        self.page = page
        self.deleted: list[dict[str, str]] = []

    async def __aenter__(self) -> _VersionedClient:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    async def list_object_versions(self, *, Bucket: str, Prefix: str, **markers: str) -> dict[str, Any]:
        rows = [(k, v, False) for k, v in self.versions if k.startswith(Prefix)]
        rows += [(k, v, True) for k, v in self.markers if k.startswith(Prefix)]
        start = int(markers.get("KeyMarker", "0"))
        page = rows[start : start + self.page]
        resp: dict[str, Any] = {
            "Versions": [{"Key": k, "VersionId": v} for k, v, m in page if not m],
            "DeleteMarkers": [{"Key": k, "VersionId": v} for k, v, m in page if m],
        }
        if start + self.page < len(rows):
            resp.update(IsTruncated=True, NextKeyMarker=str(start + self.page), NextVersionIdMarker="x")
        return resp

    async def delete_objects(self, *, Bucket: str, Delete: dict[str, Any]) -> dict[str, Any]:
        assert len(Delete["Objects"]) <= 1000
        errors = [{"Key": o["Key"], "Code": "AccessDenied"} for o in Delete["Objects"] if o["Key"] in self.refuse]
        self.deleted += [o for o in Delete["Objects"] if o["Key"] not in self.refuse]
        return {"Errors": errors} if errors else {}


# parity-exempt: aioboto3 Session.client() factory stub
class _Session:
    def __init__(self, client: _VersionedClient) -> None:
        self._client = client

    def client(self, *args: object, **kwargs: object) -> _VersionedClient:
        return self._client


def _store(client: _VersionedClient) -> S3ObjectStore:
    return S3ObjectStore(endpoint_url=None, bucket="b", session=_Session(client))


@pytest.mark.asyncio
async def test_every_version_and_marker_under_the_prefix_is_deleted_by_version_id() -> None:
    client = _VersionedClient(
        [("exports/a/0000.parquet", "v1"), ("exports/a/0000.parquet", "v2"), ("exports/b/x", "v1")],
        [("exports/a/manifest", "m1")],
        refuse=set(),
        page=2,
    )
    assert await _store(client).delete_versions("exports/a/") == 3
    assert client.deleted == [
        {"Key": "exports/a/0000.parquet", "VersionId": "v1"},
        {"Key": "exports/a/0000.parquet", "VersionId": "v2"},
        {"Key": "exports/a/manifest", "VersionId": "m1"},
    ]


@pytest.mark.asyncio
async def test_a_version_s3_would_not_delete_is_raised() -> None:
    client = _VersionedClient([("exports/a/x", "v1")], [], refuse={"exports/a/x"}, page=10)
    with pytest.raises(RuntimeError, match="did not delete 1"):
        await _store(client).delete_versions("exports/a/")


@pytest.mark.asyncio
async def test_an_empty_prefix_is_refused() -> None:
    with pytest.raises(ValueError):
        await _store(_VersionedClient([], [], refuse=set(), page=1)).delete_versions("")
