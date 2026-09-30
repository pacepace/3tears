"""the object-catalog MediaStorage bridges object ids to media items."""

from __future__ import annotations

from collections.abc import AsyncIterator
from uuid import uuid4

import pytest
from threetears.media.contracts import MediaInfo, MediaSizeLimitExceeded, MediaStorage, ObjectHandle

from threetears.agent.tools import object_media_storage as oms
from threetears.agent.tools.consume import ConsumeObjectError
from threetears.agent.tools.object_media_storage import (
    ObjectCatalogMediaStorage,
    media_category_for_mime,
)


def _handle(object_id, mime_type: str = "image/png") -> ObjectHandle:
    """build a resolved handle for the given id and mime."""
    return ObjectHandle(
        object_id=object_id,
        s3_key=f"cust/conv/image/{object_id}/x",
        mime_type=mime_type,
        size_bytes=3,
    )


class TestCategoryMapping:
    """MIME maps to the category analyze_media routes on."""

    def test_image_audio_video_document(self) -> None:
        """each MIME family maps to its category; unknown falls to document."""
        assert media_category_for_mime("image/png") == "image"
        assert media_category_for_mime("audio/mpeg") == "audio"
        assert media_category_for_mime("video/mp4") == "video"
        assert media_category_for_mime("application/pdf") == "document"
        assert media_category_for_mime("text/plain") == "document"


class TestGetMedia:
    """get_media resolves the object and infers its category."""

    async def test_returns_media_info_for_an_owned_image(self, monkeypatch) -> None:
        """a resolvable image object yields MediaInfo with category image."""
        mid = uuid4()

        async def _resolve(object_id):
            return _handle(object_id, "image/png")

        monkeypatch.setattr(oms, "resolve_object", _resolve)
        info = await ObjectCatalogMediaStorage().get_media(mid)
        assert info == MediaInfo(
            media_id=mid,
            media_category="image",
            mime_type="image/png",
            extraction_status=None,
            has_downloadable_data=True,
            size_bytes=3,
        )

    async def test_unresolvable_object_is_none(self, monkeypatch) -> None:
        """an object the caller cannot resolve returns None, not an error."""

        async def _resolve(object_id):
            raise ConsumeObjectError("no verified customer")

        monkeypatch.setattr(oms, "resolve_object", _resolve)
        assert await ObjectCatalogMediaStorage().get_media(uuid4()) is None


class TestDownloadMedia:
    """download_media streams the bytes for the byte-taking fallback."""

    async def test_streams_bytes_and_mime(self, monkeypatch) -> None:
        """a resolvable object streams its bytes joined, with its mime."""
        mid = uuid4()

        async def _resolve(object_id):
            return _handle(object_id, "image/jpeg")

        async def _stream(s3_key: str) -> AsyncIterator[bytes]:
            yield b"AB"
            yield b"C"

        monkeypatch.setattr(oms, "resolve_object", _resolve)
        monkeypatch.setattr(oms, "open_object_stream", _stream)
        result = await ObjectCatalogMediaStorage().download_media(mid)
        assert result == (b"ABC", "image/jpeg")

    async def test_unresolvable_is_none(self, monkeypatch) -> None:
        """an unresolvable object downloads to None."""

        async def _resolve(object_id):
            raise ConsumeObjectError("cross-tenant")

        monkeypatch.setattr(oms, "resolve_object", _resolve)
        assert await ObjectCatalogMediaStorage().download_media(uuid4()) is None


class TestBoundedDownload:
    """download_media(max_bytes=) never holds more than the limit plus one chunk."""

    async def test_a_catalogued_size_over_the_limit_refuses_before_opening(self, monkeypatch) -> None:
        """the recorded size answers first; the store is never opened."""
        mid = uuid4()
        opened: list[str] = []

        async def _resolve(object_id):
            return ObjectHandle(object_id=object_id, s3_key="k", mime_type="text/plain", size_bytes=11)

        async def _stream(s3_key: str) -> AsyncIterator[bytes]:
            opened.append(s3_key)
            yield b"x"

        monkeypatch.setattr(oms, "resolve_object", _resolve)
        monkeypatch.setattr(oms, "open_object_stream", _stream)
        with pytest.raises(MediaSizeLimitExceeded) as caught:
            await ObjectCatalogMediaStorage().download_media(mid, max_bytes=10)
        assert opened == []
        assert (caught.value.media_id, caught.value.limit_bytes, caught.value.size_bytes) == (mid, 10, 11)

    async def test_the_read_stops_past_the_limit_and_releases_the_stream(self, monkeypatch) -> None:
        """a size the catalog understates is caught on the read, which stops and closes the stream."""
        mid = uuid4()
        pulled: list[int] = []
        closed: list[bool] = []

        async def _resolve(object_id):
            return ObjectHandle(object_id=object_id, s3_key="k", mime_type="text/plain", size_bytes=0)

        async def _stream(s3_key: str) -> AsyncIterator[bytes]:
            try:
                for _ in range(100):
                    pulled.append(4)
                    yield b"abcd"
            finally:
                closed.append(True)

        monkeypatch.setattr(oms, "resolve_object", _resolve)
        monkeypatch.setattr(oms, "open_object_stream", _stream)
        with pytest.raises(MediaSizeLimitExceeded) as caught:
            await ObjectCatalogMediaStorage().download_media(mid, max_bytes=10)
        assert sum(pulled) == 12, "the read went on past the chunk that crossed the limit"
        assert closed == [True]
        assert caught.value.size_bytes is None

    async def test_an_object_at_the_limit_is_read_whole(self, monkeypatch) -> None:
        """the limit is inclusive: exactly max_bytes is not over it."""

        async def _resolve(object_id):
            return ObjectHandle(object_id=object_id, s3_key="k", mime_type="text/plain", size_bytes=4)

        async def _stream(s3_key: str) -> AsyncIterator[bytes]:
            yield b"ab"
            yield b"cd"

        monkeypatch.setattr(oms, "resolve_object", _resolve)
        monkeypatch.setattr(oms, "open_object_stream", _stream)
        assert await ObjectCatalogMediaStorage().download_media(uuid4(), max_bytes=4) == (b"abcd", "text/plain")


class TestContentIsUncached:
    """the object catalog stores no extracted content: honest no-ops."""

    async def test_get_content_is_always_none(self) -> None:
        """get_content is always a cache miss."""
        assert await ObjectCatalogMediaStorage().get_content(uuid4(), "description") is None

    async def test_store_content_returns_an_uncached_id_and_persists_nothing(self) -> None:
        """store_content returns a marked, non-persisted id."""
        mid = uuid4()
        cid = await ObjectCatalogMediaStorage().store_content(mid, uuid4(), "description", "a cat")
        assert cid == f"uncached:{mid}:description"


def test_bridge_satisfies_the_media_storage_protocol() -> None:
    """the bridge is structurally a MediaStorage (mypy proves the surface; this pins it)."""
    storage: MediaStorage = ObjectCatalogMediaStorage()
    assert storage is not None
